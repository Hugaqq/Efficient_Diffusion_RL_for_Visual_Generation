"""The one frozen canonical configuration schema for VisualRL v0.7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Literal

from visual_rl.core.types import (
    FrozenMapping,
    UINT32_MAX,
    validate_step_seed_budget,
)
from visual_rl.errors import ConfigError

__all__ = (
    "AdvantageConfig",
    "AlgorithmConfig",
    "ArtifactsConfig",
    "ComponentSelectionConfig",
    "DatasetConfig",
    "DistributedConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ResumeConfig",
    "RewardComponentConfig",
    "RewardConfig",
    "RewardExecutionConfig",
    "RunConfig",
    "RuntimeConfig",
    "VisualRLConfig",
    "config_from_mapping",
)

SamplingStrategy = Literal["sequential", "deterministic_shuffle"]
EmptyPromptPolicy = Literal["error", "skip"]
Precision = Literal["fp32", "fp16", "bf16"]
DistributedMode = Literal["single", "ddp"]
DistributedDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class RunConfig:
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    adapter_checkpoint: Path | None
    params: FrozenMapping


@dataclass(frozen=True)
class DatasetConfig:
    path: Path | None
    prompts: tuple[str, ...] | None
    split: str
    repeat_per_prompt: int
    require_unique: bool
    sampling_strategy: SamplingStrategy
    sampling_seed: int
    empty_prompt_policy: EmptyPromptPolicy


@dataclass(frozen=True)
class ComponentSelectionConfig:
    name: str
    params: FrozenMapping


@dataclass(frozen=True)
class RewardComponentConfig:
    name: str
    weight: float
    params: FrozenMapping


@dataclass(frozen=True)
class RewardExecutionConfig:
    microbatch_size: int | None
    max_retries: int


@dataclass(frozen=True)
class RewardConfig:
    components: tuple[RewardComponentConfig, ...]
    execution: RewardExecutionConfig
    cache_dir: Path | None


@dataclass(frozen=True)
class AdvantageConfig:
    epsilon: float


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    params: FrozenMapping
    advantage: AdvantageConfig


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float | None
    max_initial_logprob_delta: float | None
    require_initial_clipfrac_zero: bool
    require_finite_gradients: bool
    require_nonzero_gradients: bool


@dataclass(frozen=True)
class DistributedConfig:
    mode: DistributedMode
    device: DistributedDevice
    timeout_s: float
    max_snapshot_tensor_bytes: int | None


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int
    batch_size: int
    precision: Precision
    update_microbatch_size: int
    deterministic: bool
    progress: bool
    distributed: DistributedConfig


@dataclass(frozen=True)
class ArtifactsConfig:
    output_dir: Path
    checkpoint_every: int
    checkpoint_keep_last: int | None
    preview_samples_per_event: int


@dataclass(frozen=True)
class ResumeConfig:
    from_: Path | None = field(metadata={"plain_name": "from"})


@dataclass(frozen=True)
class VisualRLConfig:
    schema_version: int
    run: RunConfig
    model: ModelConfig
    dataset: DatasetConfig
    rollout: ComponentSelectionConfig
    reward: RewardConfig
    algorithm: AlgorithmConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    artifacts: ArtifactsConfig
    resume: ResumeConfig


_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "run",
        "model",
        "dataset",
        "rollout",
        "reward",
        "algorithm",
        "optimizer",
        "runtime",
        "artifacts",
        "resume",
    }
)


def config_from_mapping(
    values: Mapping[str, Any],
    *,
    config_dir: Path,
) -> VisualRLConfig:
    """Build and validate the canonical config from already-resolved values.

    Component-owned ``params`` must already have passed the selected
    component's ``resolve_params()``. This function owns only the global
    schema and cross-section invariants; it performs no filesystem I/O.
    """

    root = _mapping(values, path="")
    _exact_keys(root, _ROOT_FIELDS, path="")
    if _integer(root["schema_version"], path="schema_version") != 1:
        _fail("schema_version must equal 1", path="schema_version")
    if not isinstance(config_dir, Path) or not config_dir.is_absolute():
        raise TypeError("config_dir must be an absolute Path")

    run_raw = _section(root, "run", {"seed"})
    run = RunConfig(seed=_uint32(run_raw["seed"], path="run.seed"))

    model_raw = _section(root, "model", {"name", "adapter_checkpoint", "params"})
    model = ModelConfig(
        name=_name(model_raw["name"], path="model.name"),
        adapter_checkpoint=_optional_path(
            model_raw["adapter_checkpoint"],
            base=config_dir,
            path="model.adapter_checkpoint",
        ),
        params=_params(model_raw["params"], path="model.params"),
    )

    dataset_raw = _section(
        root,
        "dataset",
        {
            "path",
            "prompts",
            "split",
            "repeat_per_prompt",
            "require_unique",
            "sampling_strategy",
            "sampling_seed",
            "empty_prompt_policy",
        },
    )
    dataset_path = _optional_path(
        dataset_raw["path"], base=config_dir, path="dataset.path"
    )
    empty_policy = _literal(
        dataset_raw["empty_prompt_policy"],
        ("error", "skip"),
        path="dataset.empty_prompt_policy",
    )
    prompts = _inline_prompts(
        dataset_raw["prompts"],
        empty_policy=empty_policy,
        require_unique=_boolean(
            dataset_raw["require_unique"], path="dataset.require_unique"
        ),
    )
    if (dataset_path is None) == (prompts is None):
        _fail(
            "dataset.path and dataset.prompts must contain exactly one non-null value",
            path="dataset",
        )
    dataset = DatasetConfig(
        path=dataset_path,
        prompts=prompts,
        split=_nonempty_text(dataset_raw["split"], path="dataset.split"),
        repeat_per_prompt=_positive_int(
            dataset_raw["repeat_per_prompt"], path="dataset.repeat_per_prompt"
        ),
        require_unique=_boolean(
            dataset_raw["require_unique"], path="dataset.require_unique"
        ),
        sampling_strategy=_literal(
            dataset_raw["sampling_strategy"],
            ("sequential", "deterministic_shuffle"),
            path="dataset.sampling_strategy",
        ),
        sampling_seed=_uint32(
            dataset_raw["sampling_seed"], path="dataset.sampling_seed"
        ),
        empty_prompt_policy=empty_policy,
    )

    rollout_raw = _section(root, "rollout", {"name", "params"})
    rollout = ComponentSelectionConfig(
        name=_name(rollout_raw["name"], path="rollout.name"),
        params=_params(rollout_raw["params"], path="rollout.params"),
    )

    reward_raw = _section(
        root, "reward", {"components", "execution", "cache_dir"}
    )
    raw_components = reward_raw["components"]
    if not isinstance(raw_components, (list, tuple)) or not raw_components:
        _fail("reward.components must be a non-empty sequence", path="reward.components")
    components: list[RewardComponentConfig] = []
    names: set[str] = set()
    for index, raw_component in enumerate(raw_components):
        item_path = f"reward.components[{index}]"
        item = _mapping(raw_component, path=item_path)
        _exact_keys(item, {"name", "weight", "params"}, path=item_path)
        name = _name(item["name"], path=f"{item_path}.name")
        if name in names:
            _fail(
                f"reward component name {name!r} is duplicated",
                path=f"{item_path}.name",
            )
        names.add(name)
        components.append(
            RewardComponentConfig(
                name=name,
                weight=_finite_float(item["weight"], path=f"{item_path}.weight"),
                params=_params(item["params"], path=f"{item_path}.params"),
            )
        )
    if not any(component.weight != 0.0 for component in components):
        _fail(
            "reward.components must contain at least one non-zero weight",
            path="reward.components",
        )
    execution_raw = _mapping(reward_raw["execution"], path="reward.execution")
    _exact_keys(
        execution_raw, {"microbatch_size", "max_retries"}, path="reward.execution"
    )
    max_retries = _integer(
        execution_raw["max_retries"], path="reward.execution.max_retries"
    )
    if not 0 <= max_retries <= 10:
        _fail(
            "reward.execution.max_retries must be between 0 and 10",
            path="reward.execution.max_retries",
        )
    reward = RewardConfig(
        components=tuple(components),
        execution=RewardExecutionConfig(
            microbatch_size=_optional_positive_int(
                execution_raw["microbatch_size"],
                path="reward.execution.microbatch_size",
            ),
            max_retries=max_retries,
        ),
        cache_dir=_optional_path(
            reward_raw["cache_dir"], base=config_dir, path="reward.cache_dir"
        ),
    )

    algorithm_raw = _section(root, "algorithm", {"name", "params", "advantage"})
    advantage_raw = _mapping(
        algorithm_raw["advantage"], path="algorithm.advantage"
    )
    _exact_keys(advantage_raw, {"epsilon"}, path="algorithm.advantage")
    advantage_epsilon = _positive_float(
        advantage_raw["epsilon"], path="algorithm.advantage.epsilon"
    )
    algorithm = AlgorithmConfig(
        name=_name(algorithm_raw["name"], path="algorithm.name"),
        params=_params(algorithm_raw["params"], path="algorithm.params"),
        advantage=AdvantageConfig(epsilon=advantage_epsilon),
    )

    optimizer_raw = _section(
        root,
        "optimizer",
        {
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_weight_decay",
            "adam_epsilon",
            "max_grad_norm",
            "max_initial_logprob_delta",
            "require_initial_clipfrac_zero",
            "require_finite_gradients",
            "require_nonzero_gradients",
        },
    )
    adam_beta1 = _finite_float(
        optimizer_raw["adam_beta1"], path="optimizer.adam_beta1"
    )
    adam_beta2 = _finite_float(
        optimizer_raw["adam_beta2"], path="optimizer.adam_beta2"
    )
    for path, value in (
        ("optimizer.adam_beta1", adam_beta1),
        ("optimizer.adam_beta2", adam_beta2),
    ):
        if not 0.0 <= value < 1.0:
            _fail(f"{path} must be in [0, 1)", path=path)
    adam_weight_decay = _finite_float(
        optimizer_raw["adam_weight_decay"],
        path="optimizer.adam_weight_decay",
    )
    if adam_weight_decay < 0.0:
        _fail(
            "optimizer.adam_weight_decay must be non-negative",
            path="optimizer.adam_weight_decay",
        )
    optimizer = OptimizerConfig(
        learning_rate=_positive_float(
            optimizer_raw["learning_rate"], path="optimizer.learning_rate"
        ),
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_weight_decay=adam_weight_decay,
        adam_epsilon=_positive_float(
            optimizer_raw["adam_epsilon"], path="optimizer.adam_epsilon"
        ),
        max_grad_norm=_optional_positive_float(
            optimizer_raw["max_grad_norm"], path="optimizer.max_grad_norm"
        ),
        max_initial_logprob_delta=_optional_positive_float(
            optimizer_raw["max_initial_logprob_delta"],
            path="optimizer.max_initial_logprob_delta",
        ),
        require_initial_clipfrac_zero=_boolean(
            optimizer_raw["require_initial_clipfrac_zero"],
            path="optimizer.require_initial_clipfrac_zero",
        ),
        require_finite_gradients=_boolean(
            optimizer_raw["require_finite_gradients"],
            path="optimizer.require_finite_gradients",
        ),
        require_nonzero_gradients=_boolean(
            optimizer_raw["require_nonzero_gradients"],
            path="optimizer.require_nonzero_gradients",
        ),
    )

    runtime_raw = _section(
        root,
        "runtime",
        {
            "max_steps",
            "batch_size",
            "precision",
            "update_microbatch_size",
            "deterministic",
            "progress",
            "distributed",
        },
    )
    distributed_raw = _mapping(
        runtime_raw["distributed"], path="runtime.distributed"
    )
    _exact_keys(
        distributed_raw,
        {"mode", "device", "timeout_s", "max_snapshot_tensor_bytes"},
        path="runtime.distributed",
    )
    mode = _literal(
        distributed_raw["mode"], ("single", "ddp"), path="runtime.distributed.mode"
    )
    device = _literal(
        distributed_raw["device"],
        ("cpu", "cuda"),
        path="runtime.distributed.device",
    )
    precision = _literal(
        runtime_raw["precision"], ("fp32", "fp16", "bf16"), path="runtime.precision"
    )
    if device == "cpu" and precision != "fp32":
        _fail(
            "CPU runtime only supports fp32 precision",
            path="runtime.precision",
        )
    max_snapshot_tensor_bytes = _optional_positive_int(
        distributed_raw["max_snapshot_tensor_bytes"],
        path="runtime.distributed.max_snapshot_tensor_bytes",
    )
    if mode == "single" and max_snapshot_tensor_bytes is not None:
        _fail(
            "single mode requires max_snapshot_tensor_bytes to be null",
            path="runtime.distributed.max_snapshot_tensor_bytes",
        )
    max_steps = _positive_int(runtime_raw["max_steps"], path="runtime.max_steps")
    runtime = RuntimeConfig(
        max_steps=max_steps,
        batch_size=_positive_int(
            runtime_raw["batch_size"], path="runtime.batch_size"
        ),
        precision=precision,
        update_microbatch_size=_positive_int(
            runtime_raw["update_microbatch_size"],
            path="runtime.update_microbatch_size",
        ),
        deterministic=_boolean(
            runtime_raw["deterministic"], path="runtime.deterministic"
        ),
        progress=_boolean(runtime_raw["progress"], path="runtime.progress"),
        distributed=DistributedConfig(
            mode=mode,
            device=device,
            timeout_s=_positive_float(
                distributed_raw["timeout_s"],
                path="runtime.distributed.timeout_s",
            ),
            max_snapshot_tensor_bytes=max_snapshot_tensor_bytes,
        ),
    )

    artifacts_raw = _section(
        root,
        "artifacts",
        {
            "output_dir",
            "checkpoint_every",
            "checkpoint_keep_last",
            "preview_samples_per_event",
        },
    )
    output_dir = _required_path(
        artifacts_raw["output_dir"], base=config_dir, path="artifacts.output_dir"
    )
    preview_samples_per_event = _integer(
        artifacts_raw["preview_samples_per_event"],
        path="artifacts.preview_samples_per_event",
    )
    if not 0 <= preview_samples_per_event <= 2:
        _fail(
            "expected an integer between 0 and 2",
            path="artifacts.preview_samples_per_event",
        )
    artifacts = ArtifactsConfig(
        output_dir=output_dir,
        checkpoint_every=_positive_int(
            artifacts_raw["checkpoint_every"],
            path="artifacts.checkpoint_every",
        ),
        checkpoint_keep_last=_optional_positive_int(
            artifacts_raw["checkpoint_keep_last"],
            path="artifacts.checkpoint_keep_last",
        ),
        preview_samples_per_event=preview_samples_per_event,
    )

    resume_raw = _section(root, "resume", {"from"})
    resume_from = _optional_path(
        resume_raw["from"], base=config_dir, path="resume.from"
    )
    if resume_from is not None and resume_from != output_dir:
        _fail(
            "resume.from must resolve to the same path as artifacts.output_dir",
            path="resume.from",
        )
    if resume_from is not None and model.adapter_checkpoint is not None:
        _fail(
            "resume.from and model.adapter_checkpoint are mutually exclusive",
            path="resume",
        )
    resume = ResumeConfig(from_=resume_from)

    if reward.cache_dir is not None and _paths_overlap(
        reward.cache_dir, artifacts.output_dir
    ):
        _fail(
            "reward.cache_dir and artifacts.output_dir must not overlap",
            path="reward.cache_dir",
        )

    world_size = 1 if runtime.distributed.mode == "single" else 2
    try:
        validate_step_seed_budget(run.seed, runtime.max_steps, world_size)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc), key="run.seed") from exc

    return VisualRLConfig(
        schema_version=1,
        run=run,
        model=model,
        dataset=dataset,
        rollout=rollout,
        reward=reward,
        algorithm=algorithm,
        optimizer=optimizer,
        runtime=runtime,
        artifacts=artifacts,
        resume=resume,
    )


def _section(
    root: Mapping[str, Any],
    key: str,
    expected: set[str],
) -> Mapping[str, Any]:
    section = _mapping(root[key], path=key)
    _exact_keys(section, expected, path=key)
    return section


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("expected a mapping", path=path or "<root>")
    if any(not isinstance(key, str) for key in value):
        _fail("mapping keys must be strings", path=path or "<root>")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        pieces = []
        if missing:
            pieces.append(f"missing keys {missing}")
        if unknown:
            pieces.append(f"unknown keys {unknown}")
        _fail("; ".join(pieces), path=path or "<root>")


def _params(value: Any, *, path: str) -> FrozenMapping:
    mapping = _mapping(value, path=path)
    try:
        return value if isinstance(value, FrozenMapping) else FrozenMapping(mapping)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc), key=path) from exc


def _boolean(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        _fail("expected a bool", path=path)
    return value


def _integer(value: Any, *, path: str) -> int:
    if type(value) is not int:
        _fail("expected an integer, not bool", path=path)
    return value


def _positive_int(value: Any, *, path: str) -> int:
    result = _integer(value, path=path)
    if result <= 0:
        _fail("expected a positive integer", path=path)
    return result


def _optional_positive_int(value: Any, *, path: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, path=path)


def _uint32(value: Any, *, path: str) -> int:
    result = _integer(value, path=path)
    if not 0 <= result <= UINT32_MAX:
        _fail("expected a canonical uint32 integer", path=path)
    return result


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("expected a finite number, not bool", path=path)
    result = float(value)
    if not math.isfinite(result):
        _fail("expected a finite number", path=path)
    return result


def _positive_float(value: Any, *, path: str) -> float:
    result = _finite_float(value, path=path)
    if result <= 0.0:
        _fail("expected a positive finite number", path=path)
    return result


def _optional_positive_float(value: Any, *, path: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, path=path)


def _nonempty_text(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        _fail("expected a string", path=path)
    result = value.strip()
    if not result or "\r" in result or "\n" in result:
        _fail("expected a non-empty single-line string", path=path)
    return result


def _name(value: Any, *, path: str) -> str:
    return _nonempty_text(value, path=path)


def _literal(value: Any, allowed: tuple[str, ...], *, path: str):
    if not isinstance(value, str) or value not in allowed:
        _fail(f"expected one of {list(allowed)}", path=path)
    return value


def _optional_path(value: Any, *, base: Path, path: str) -> Path | None:
    if value is None:
        return None
    return _required_path(value, base=base, path=path)


def _required_path(value: Any, *, base: Path, path: str) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        _fail("expected a filesystem path", path=path)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    # Lexically normalize without stat/symlink resolution. Filesystem and
    # path-policy checks belong exclusively to Preflight.
    return Path(os.path.abspath(candidate))


def _inline_prompts(
    value: Any,
    *,
    empty_policy: EmptyPromptPolicy,
    require_unique: bool,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        _fail("dataset.prompts must be a sequence or null", path="dataset.prompts")
    canonical: list[str] = []
    for index, item in enumerate(value):
        item_path = f"dataset.prompts[{index}]"
        if not isinstance(item, str):
            _fail("prompt must be a string", path=item_path)
        if "\r" in item or "\n" in item:
            _fail("prompt must not contain CR/LF", path=item_path)
        prompt = item.strip()
        if not prompt:
            if empty_policy == "skip":
                continue
            _fail("empty prompt is forbidden by empty_prompt_policy", path=item_path)
        canonical.append(prompt)
    if not canonical:
        _fail("dataset.prompts must remain non-empty", path="dataset.prompts")
    if require_unique and len(set(canonical)) != len(canonical):
        _fail(
            "dataset.prompts must be unique before repeat expansion",
            path="dataset.prompts",
        )
    return tuple(canonical)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _fail(message: str, *, path: str) -> None:
    raise ConfigError(f"{path}: {message}", key=path)
