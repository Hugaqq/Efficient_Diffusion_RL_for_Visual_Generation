"""Environment/artifact preflight without runtime component construction."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.composition.registry import ResolvedComponentDeclaration
from visual_rl.composition.config.specs import ArtifactLocations
from visual_rl.core.contracts import (
    ComponentArtifactBinding,
    ComponentArtifactBindingSet,
    ComponentLoadPlan,
)
from visual_rl.core.types import (
    FrozenMapping,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
)
from visual_rl.errors import ValidationError
from visual_rl.composition.preflight.types import (
    ArtifactIdentityRequest,
    ArtifactIdentityResolution,
    ArtifactIdentityResolver,
    EnvironmentPreflightResult,
    StaticPreflightResult,
)

__all__ = ("backend_for", "run_environment_preflight")

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


def run_environment_preflight(
    static: StaticPreflightResult,
    artifact_resolver: ArtifactIdentityResolver,
    *,
    artifact_locations: ArtifactLocations,
) -> EnvironmentPreflightResult:
    """Lock artifact identities through an injected read-only resolver."""

    if not isinstance(static, StaticPreflightResult):
        raise TypeError("static must be a StaticPreflightResult")
    if not isinstance(artifact_locations, ArtifactLocations):
        raise TypeError("artifact_locations must be ArtifactLocations")
    if not static.can_materialize:
        raise ValidationError(
            "cannot materialize a statically incompatible recipe graph"
        )
    resolve = getattr(artifact_resolver, "resolve_artifact_identities", None)
    if not callable(resolve):
        raise TypeError(
            "artifact_resolver must implement resolve_artifact_identities()"
        )
    identities = resolve(
        ArtifactIdentityRequest(
            resolved=static.resolved,
            locations=artifact_locations,
        )
    )
    if not isinstance(identities, ArtifactIdentityResolution):
        raise TypeError("artifact resolver must return an ArtifactIdentityResolution")
    source_content = identities.source_locations.to_content_binding(
        static.resolved.source_plan
    )
    reward_plan = static.resolved.reward_plan.bind_artifacts(
        identities.reward_artifact_identities
    )
    materialized = MaterializedRecipe(
        resolved=static.resolved,
        model_artifact_identity=identities.model_artifact_identity,
        source_content_binding=source_content,
        reward_plan=reward_plan,
        code_artifact_identity=identities.code_artifact_identity,
    )
    binding_set, load_plan = _derive_component_load_contracts(materialized)
    return EnvironmentPreflightResult(
        static=static,
        materialized=materialized,
        artifact_locations=artifact_locations,
        source_locations=identities.source_locations,
        component_artifact_bindings=binding_set,
        component_load_plan=load_plan,
    )


def _derive_component_load_contracts(
    materialized: MaterializedRecipe,
) -> tuple[ComponentArtifactBindingSet, ComponentLoadPlan]:
    """Derive the sole all-slot G1 graph before any implementation import."""

    if not isinstance(materialized, MaterializedRecipe):
        raise TypeError("materialized must be a MaterializedRecipe")
    resolved = materialized.resolved
    declarations: tuple[tuple[str, ResolvedComponentDeclaration], ...] = (
        ("algorithm", resolved.algorithm.component),
        (resolved.model.slot, resolved.model.declaration),
        *(
            (component.slot, component.declaration)
            for component in resolved.internal_components
        ),
        *(
            (component.slot, component.declaration)
            for component in resolved.reward_components
        ),
    )
    declarations = tuple(sorted(declarations, key=lambda item: item[0]))
    slots = tuple(slot for slot, _declaration in declarations)
    if len(slots) != len(set(slots)):
        raise ValueError("resolved component graph contains duplicate slots")

    code_identity = _artifact_digest(
        materialized.code_artifact_identity,
        field_name="code_artifact_identity",
    )
    model_identity = _artifact_digest(
        materialized.model_artifact_identity,
        field_name="model_artifact_identity",
    )
    logical_rewards = {
        item.logical_reward_id: item
        for item in materialized.reward_plan.logical_rewards
    }
    resources = {
        item.resource_identity: item for item in materialized.reward_plan.resources
    }

    bindings: list[ComponentArtifactBinding] = []
    requirements: dict[str, tuple[str, ...]] = {}
    for slot, declaration in declarations:
        artifact_identities: dict[str, str] = {"code": code_identity}
        if slot == "model":
            artifact_identities["model"] = model_identity
        elif slot.startswith("rewards."):
            logical_id = slot.removeprefix("rewards.")
            logical = logical_rewards.get(logical_id)
            if logical is None:
                raise ValueError(f"reward slot {slot!r} has no materialized plan entry")
            resource = resources.get(logical.resource_identity)
            if resource is None or resource.artifact_identity is None:
                raise ValueError(
                    f"reward slot {slot!r} has no materialized resource identity"
                )
            artifact_identities["reward-resource"] = _artifact_digest(
                resource.artifact_identity,
                field_name=f"reward resource {logical_id!r}",
            )
        binding = ComponentArtifactBinding.create(
            recipe_id=materialized.recipe_id,
            slot=slot,
            component_declaration_id=declaration.declaration_id,
            declared=declaration.declared_contract,
            artifact_content_identities=tuple(sorted(artifact_identities.items())),
            code_identity=code_identity,
            implementation_identity=declaration.implementation_class_path,
            interface_version=declaration.descriptor.interface_version,
        )
        bindings.append(binding)
        requirements[slot] = tuple(sorted(artifact_identities))

    binding_set = ComponentArtifactBindingSet(
        recipe_id=materialized.recipe_id,
        bindings=tuple(bindings),
    )
    return binding_set, ComponentLoadPlan.create(
        binding_set,
        required_artifact_names_by_slot=requirements,
    )


def _artifact_digest(identity: FrozenMapping, *, field_name: str) -> str:
    if not isinstance(identity, FrozenMapping):
        raise TypeError(f"{field_name} must be a FrozenMapping")
    digest = identity.get("content_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field_name}.content_sha256 must be a SHA-256 digest")
    return digest


# Legacy v0.7 volatile/environment checks retained behind the package facade.
def _validate_component_environment(
    selected: Sequence[tuple[str, str, Mapping[str, object], Any]],
    context: ValidationContext,
) -> tuple[ValidationCheck, ...]:
    checks: list[ValidationCheck] = []
    for kind, name, params, spec in selected:
        try:
            result = spec.factory.check_environment(params, context)
        except Exception as exc:  # noqa: BLE001 - plugin environment boundary
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
        present = tuple(
            key for key, value in raw_launch_env.items() if value is not None
        )
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

    group_values = (raw_launch_env["GROUP_RANK"], raw_launch_env["GROUP_WORLD_SIZE"])
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
    """The single legacy mode/device to process-group backend mapping."""

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
