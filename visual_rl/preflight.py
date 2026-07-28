"""Single, CPU-only validation and launch-environment preflight.

The public API calls :func:`run_preflight` for both explicit validation and
the volatile check immediately before a run.  This module is the sole parser
for torchrun launch variables and the sole owner of the mode/device to backend
mapping.  It never creates a training component, imports torch, or writes a
run directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal

from visual_rl.core.components import CAPABILITY_OWNER
from visual_rl.core.types import (
    FrozenMapping,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
)

__all__ = ["run_preflight"]

_Phase = Literal["validate", "run"]
_LAUNCH_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_REQUIRED_DDP_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_FACTORY_METHODS = (
    "resolve_params",
    "check_environment",
    "from_config",
)


def run_preflight(
    config: Any,
    *,
    config_dir: str | Path,
    phase: _Phase,
    cached_report: Any | None = None,
    cached_env: ValidatedRuntimeEnv | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, ValidatedRuntimeEnv | None]:
    """Validate one canonical config without constructing runtime objects.

    Returns the production ``ValidationReport`` plus the launch snapshot from
    the same environment read.  Invalid topology always yields ``None`` for
    the snapshot and therefore ``None/None`` for the report rank projection.
    ``environ`` exists for deterministic tests; production callers omit it.
    """

    if phase not in {"validate", "run"}:
        raise ValueError("phase must be 'validate' or 'run'")
    resolved_config_dir = Path(config_dir).expanduser().resolve()
    environment = os.environ if environ is None else environ

    fresh_checks: list[ValidationCheck] = []
    selected = _selected_components(config)
    fresh_checks.extend(_validate_component_contracts(selected))
    fresh_checks.extend(_validate_capabilities(selected))
    fresh_checks.extend(_validate_group_size(selected, config))
    fresh_checks.extend(_validate_global_paths(config))

    runtime_env, topology_checks = _validate_runtime_environment(
        config.runtime,
        environment,
    )
    fresh_checks.extend(topology_checks)

    distributed = config.runtime.distributed
    validation_context = ValidationContext(
        phase=phase,
        config_dir=resolved_config_dir,
        distributed_mode=distributed.mode,
        world_size=1 if distributed.mode == "single" else 2,
        backend=backend_for(distributed.mode, distributed.device),
        device=distributed.device,
        timeout_s=distributed.timeout_s,
    )
    fresh_checks.extend(
        _validate_component_environment(selected, validation_context)
    )

    if phase == "run" and cached_report is not None:
        checks = [
            *(item for item in cached_report.checks if not item.volatile),
            *(item for item in fresh_checks if item.volatile),
        ]
    else:
        checks = fresh_checks

    if (
        phase == "run"
        and cached_env is not None
        and runtime_env is not None
        and cached_env != runtime_env
    ):
        checks.append(
            _error(
                "runtime.launch_environment_drift",
                "runtime.distributed",
                "validated launch environment changed before runtime construction",
                volatile=True,
            )
        )
        runtime_env = None

    report = _validation_report(tuple(checks), runtime_env)
    return report, runtime_env


def _selected_components(
    config: Any,
) -> tuple[tuple[str, str, Mapping[str, object], Any], ...]:
    """Resolve the canonical config selections through the only manifest."""

    # Kept inside the validation boundary so importing visual_rl and resolving
    # YAML remain free of runtime construction imports.
    from visual_rl.builtins import get_builtin_component

    values: list[tuple[str, str, Mapping[str, object], Any]] = []
    for kind, selection in (
        ("model", config.model),
        ("rollout", config.rollout),
        ("algorithm", config.algorithm),
    ):
        spec = get_builtin_component(kind, selection.name)
        values.append((kind, selection.name, selection.params, spec))
    for component in config.reward.components:
        spec = get_builtin_component("reward", component.name)
        values.append(("reward", component.name, component.params, spec))
    return tuple(values)


def _validate_component_contracts(
    selected: Sequence[tuple[str, str, Mapping[str, object], Any]],
) -> tuple[ValidationCheck, ...]:
    from visual_rl.feedback.base import RewardClient
    from visual_rl.model_adapters.base import ModelAdapter
    from visual_rl.optimizers.base import PolicyAlgorithm
    from visual_rl.rollout.base import RolloutEngine

    base_by_kind = {
        "model": ModelAdapter,
        "rollout": RolloutEngine,
        "reward": RewardClient,
        "algorithm": PolicyAlgorithm,
    }
    checks: list[ValidationCheck] = []
    checked: set[tuple[str, str]] = set()
    for kind, name, _params, spec in selected:
        key = (kind, name)
        if key in checked:
            continue
        checked.add(key)
        factory = spec.factory
        path = f"{kind}.{name}"
        if not inspect.isclass(factory) or not issubclass(factory, base_by_kind[kind]):
            checks.append(
                _error(
                    "component.invalid_factory",
                    path,
                    f"{kind}/{name} factory does not implement the {kind} base contract",
                )
            )
            continue
        if inspect.isabstract(factory):
            checks.append(
                _error(
                    "component.abstract_factory",
                    path,
                    f"{kind}/{name} factory is abstract",
                )
            )
        missing_methods = tuple(
            method for method in _FACTORY_METHODS if not callable(getattr(factory, method, None))
        )
        if missing_methods:
            checks.append(
                _error(
                    "component.missing_factory_method",
                    path,
                    "factory is missing required methods: "
                    + ", ".join(missing_methods),
                )
            )
        for dependency in spec.dependencies:
            try:
                available = importlib.util.find_spec(dependency) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            if not available:
                checks.append(
                    _error(
                        "component.missing_dependency",
                        path,
                        f"missing Python dependency {dependency!r}",
                    )
                )
    return tuple(checks)


def _validate_capabilities(
    selected: Sequence[tuple[str, str, Mapping[str, object], Any]],
) -> tuple[ValidationCheck, ...]:
    provided_by_kind: dict[str, set[str]] = {}
    for kind, _name, _params, spec in selected:
        provided_by_kind.setdefault(kind, set()).update(spec.provides)

    checks: list[ValidationCheck] = []
    for kind, name, params, spec in selected:
        required = set(spec.requires)
        hook = getattr(spec.factory, "required_capabilities", None)
        if callable(hook):
            try:
                conditional = hook(params)
            except Exception as exc:  # parameter hooks are bounded validation
                checks.append(
                    _error(
                        "component.capability_hook_failed",
                        f"{kind}.{name}",
                        f"required_capabilities() failed: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if not isinstance(conditional, frozenset):
                checks.append(
                    _error(
                        "component.invalid_capability_hook",
                        f"{kind}.{name}",
                        "required_capabilities() must return frozenset",
                    )
                )
                continue
            required.update(conditional)

        for capability in sorted(required):
            owner_kind = CAPABILITY_OWNER.get(capability)
            if owner_kind is None:
                checks.append(
                    _error(
                        "component.unknown_capability",
                        f"{kind}.{name}",
                        f"unknown capability {capability!r}",
                    )
                )
                continue
            if capability not in provided_by_kind.get(owner_kind, set()):
                checks.append(
                    _error(
                        "component.missing_capability",
                        f"{kind}.{name}",
                        f"requires {capability!r} from selected {owner_kind}",
                    )
                )
    return tuple(checks)


def _validate_group_size(
    selected: Sequence[tuple[str, str, Mapping[str, object], Any]],
    config: Any,
) -> tuple[ValidationCheck, ...]:
    algorithm_factory = next(
        spec.factory for kind, _name, _params, spec in selected if kind == "algorithm"
    )
    minimum = algorithm_factory.MIN_GROUP_SIZE
    rollout_params = config.rollout.params
    group_size = (
        rollout_params["branch_count"]
        if "branch_count" in rollout_params
        else rollout_params["samples_per_prompt"]
    )
    if group_size >= minimum:
        return ()
    return (
        _error(
            "algorithm.group_too_small",
            "rollout.params",
            f"resolved group size {group_size} is below algorithm minimum {minimum}",
        ),
    )


def _validate_component_environment(
    selected: Sequence[tuple[str, str, Mapping[str, object], Any]],
    context: ValidationContext,
) -> tuple[ValidationCheck, ...]:
    checks: list[ValidationCheck] = []
    for kind, name, params, spec in selected:
        try:
            result = spec.factory.check_environment(params, context)
        except Exception as exc:
            checks.append(
                _error(
                    "component.environment_check_failed",
                    f"{kind}.{name}",
                    f"check_environment() failed: {type(exc).__name__}: {exc}",
                    volatile=True,
                )
            )
            continue
        if not isinstance(result, tuple) or any(
            not isinstance(item, ValidationCheck) for item in result
        ):
            checks.append(
                _error(
                    "component.invalid_environment_checks",
                    f"{kind}.{name}",
                    "check_environment() must return tuple[ValidationCheck, ...]",
                )
            )
            continue
        checks.extend(result)
    return tuple(checks)


def _validate_global_paths(config: Any) -> tuple[ValidationCheck, ...]:
    checks: list[ValidationCheck] = []
    dataset_path = config.dataset.path
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.exists():
            checks.append(
                _error(
                    "dataset.path_missing",
                    "dataset.path",
                    f"prompt file does not exist: {path}",
                    volatile=True,
                )
            )
        elif not path.is_file():
            checks.append(
                _error(
                    "dataset.path_not_file",
                    "dataset.path",
                    f"prompt path is not a regular file: {path}",
                    volatile=True,
                )
            )
        elif not os.access(path, os.R_OK):
            checks.append(
                _error(
                    "dataset.path_not_readable",
                    "dataset.path",
                    f"prompt file is not readable: {path}",
                    volatile=True,
                )
            )

    output_dir = Path(config.artifacts.output_dir)
    reward_cache_dir = config.reward.cache_dir
    if reward_cache_dir is not None and _paths_overlap(
        output_dir, Path(reward_cache_dir)
    ):
        checks.append(
            _error(
                "paths.reward_cache_overlaps_output",
                "reward.cache_dir",
                "reward.cache_dir and artifacts.output_dir must not overlap",
            )
        )

    resume_from = config.resume.from_
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            checks.append(
                _error(
                    "resume.path_missing",
                    "resume.from",
                    f"resume run directory does not exist: {resume_path}",
                    volatile=True,
                )
            )
        elif not resume_path.is_dir():
            checks.append(
                _error(
                    "resume.path_not_directory",
                    "resume.from",
                    f"resume locator is not a directory: {resume_path}",
                    volatile=True,
                )
            )

    parent = _first_existing_parent(output_dir)
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        checks.append(
            _error(
                "artifacts.output_parent_not_writable",
                "artifacts.output_dir",
                f"nearest existing output parent is not writable: {parent}",
                volatile=True,
            )
        )
    return tuple(checks)


def _validate_runtime_environment(
    runtime: Any,
    environ: Mapping[str, str],
) -> tuple[ValidatedRuntimeEnv | None, tuple[ValidationCheck, ...]]:
    distributed = runtime.distributed
    mode = distributed.mode
    device = distributed.device
    raw_launch_env = {key: environ.get(key) for key in _LAUNCH_ENV_KEYS}
    visible_gpu_count = _visible_gpu_count(environ)
    checks: list[ValidationCheck] = []

    if device == "cpu" and runtime.precision != "fp32":
        checks.append(
            _error(
                "runtime.invalid_precision_device",
                "runtime.precision",
                "CPU runtime only supports fp32",
            )
        )
    if device == "cuda":
        required_gpus = 1 if mode == "single" else 2
        if visible_gpu_count < required_gpus:
            checks.append(
                _error(
                    "runtime.insufficient_visible_gpus",
                    "runtime.distributed.device",
                    f"CUDA {mode} requires {required_gpus} visible GPU(s), "
                    f"found {visible_gpu_count}",
                    volatile=True,
                )
            )

    if mode == "single":
        present = tuple(key for key, value in raw_launch_env.items() if value is not None)
        if present:
            checks.append(
                _error(
                    "runtime.unexpected_launch_env",
                    "runtime.distributed.mode",
                    "single mode rejects torchrun launch variables: "
                    + ", ".join(present),
                    volatile=True,
                )
            )
        if checks:
            return None, tuple(checks)
        return (
            ValidatedRuntimeEnv(
                mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
                local_world_size=1,
                group_rank=None,
                group_world_size=None,
                master_addr=None,
                master_port=None,
                visible_gpu_count=visible_gpu_count,
                raw_launch_env=FrozenMapping(raw_launch_env),
            ),
            (),
        )

    missing = tuple(
        key for key in _REQUIRED_DDP_ENV_KEYS if raw_launch_env[key] is None
    )
    if missing:
        checks.append(
            _error(
                "runtime.incomplete_launch_env",
                "runtime.distributed.mode",
                "DDP launch environment is missing: " + ", ".join(missing),
                volatile=True,
            )
        )
        return None, tuple(checks)

    group_values = (
        raw_launch_env["GROUP_RANK"],
        raw_launch_env["GROUP_WORLD_SIZE"],
    )
    if (group_values[0] is None) != (group_values[1] is None):
        checks.append(
            _error(
                "runtime.incomplete_group_env",
                "runtime.distributed.mode",
                "GROUP_RANK and GROUP_WORLD_SIZE must be provided together",
                volatile=True,
            )
        )

    parsed: dict[str, int] = {}
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_PORT",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
    ):
        value = raw_launch_env[key]
        if value is None:
            continue
        parsed_value = _parse_decimal(value)
        if parsed_value is None:
            checks.append(
                _error(
                    "runtime.invalid_launch_env",
                    f"environment.{key}",
                    f"{key} must be a canonical non-negative decimal integer",
                    volatile=True,
                )
            )
        else:
            parsed[key] = parsed_value

    if checks:
        return None, tuple(checks)

    rank = parsed["RANK"]
    local_rank = parsed["LOCAL_RANK"]
    world_size = parsed["WORLD_SIZE"]
    local_world_size = parsed["LOCAL_WORLD_SIZE"]
    master_port = parsed["MASTER_PORT"]
    master_addr = raw_launch_env["MASTER_ADDR"]
    if world_size != 2 or local_world_size != 2:
        checks.append(
            _error(
                "runtime.unsupported_world_size",
                "runtime.distributed.mode",
                "v0.7 DDP requires WORLD_SIZE == LOCAL_WORLD_SIZE == 2",
                volatile=True,
            )
        )
    if rank not in {0, 1} or local_rank not in {0, 1} or rank != local_rank:
        checks.append(
            _error(
                "runtime.invalid_rank_topology",
                "runtime.distributed.mode",
                "single-node DDP requires RANK == LOCAL_RANK in {0, 1}",
                volatile=True,
            )
        )
    if master_addr is None or not master_addr.strip():
        checks.append(
            _error(
                "runtime.invalid_master_addr",
                "environment.MASTER_ADDR",
                "MASTER_ADDR must be non-empty",
                volatile=True,
            )
        )
    if not 1 <= master_port <= 65535:
        checks.append(
            _error(
                "runtime.invalid_master_port",
                "environment.MASTER_PORT",
                "MASTER_PORT must be in 1..65535",
                volatile=True,
            )
        )

    group_rank = parsed.get("GROUP_RANK")
    group_world_size = parsed.get("GROUP_WORLD_SIZE")
    if group_rank is not None and (group_rank != 0 or group_world_size != 1):
        checks.append(
            _error(
                "runtime.unsupported_group_topology",
                "runtime.distributed.mode",
                "single-node DDP requires GROUP_RANK=0 and GROUP_WORLD_SIZE=1",
                volatile=True,
            )
        )
    if device == "cuda" and not (
        0 <= local_rank < local_world_size <= visible_gpu_count
    ):
        checks.append(
            _error(
                "runtime.invalid_cuda_rank",
                "runtime.distributed.device",
                "LOCAL_RANK/LOCAL_WORLD_SIZE exceed the visible CUDA device set",
                volatile=True,
            )
        )
    if checks:
        return None, tuple(checks)

    return (
        ValidatedRuntimeEnv(
            mode="ddp",
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            group_rank=group_rank,
            group_world_size=group_world_size,
            master_addr=master_addr,
            master_port=master_port,
            visible_gpu_count=visible_gpu_count,
            raw_launch_env=FrozenMapping(raw_launch_env),
        ),
        (),
    )


def backend_for(mode: str, device: str) -> str | None:
    """The single v0.7 mode/device to process-group backend mapping."""

    if mode == "single":
        return None
    if mode != "ddp":
        raise ValueError(f"unsupported distributed mode: {mode!r}")
    if device == "cpu":
        return "gloo"
    if device == "cuda":
        return "nccl"
    raise ValueError(f"unsupported runtime device: {device!r}")


def _visible_gpu_count(environ: Mapping[str, str]) -> int:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        value = visible.strip()
        if not value or value == "-1":
            return 0
        return len(tuple(item for item in value.split(",") if item.strip()))

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:
        return 0
    return len(tuple(line for line in result.stdout.splitlines() if line.strip()))


def _parse_decimal(value: str) -> int | None:
    if not value or (len(value) > 1 and value.startswith("0")) or not value.isdecimal():
        return None
    return int(value)


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _first_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _validation_report(
    checks: tuple[ValidationCheck, ...],
    runtime_env: ValidatedRuntimeEnv | None,
) -> Any:
    from visual_rl.api_types import ValidationReport

    return ValidationReport(
        checks=checks,
        runtime_rank=None if runtime_env is None else runtime_env.rank,
        runtime_world_size=None if runtime_env is None else runtime_env.world_size,
    )


def _error(
    code: str,
    path: str,
    message: str,
    *,
    volatile: bool = False,
) -> ValidationCheck:
    return ValidationCheck(
        level="error",
        code=code,
        path=path,
        message=message,
        volatile=volatile,
    )
