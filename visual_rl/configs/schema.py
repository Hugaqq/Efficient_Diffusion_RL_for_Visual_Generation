"""Strict typed configuration for the simplified VisualRL mainline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
import json
import math
from pathlib import Path
import re
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


@dataclass
class ModelConfig:
    name: str = "mock_wan"
    model_path: str = ""
    model_family: str = "wan"
    latent_shape: list[int] = field(default_factory=lambda: [4, 2, 2, 2])
    media_shape: list[int] = field(default_factory=lambda: [4, 3, 16, 16])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    path: str | None = None
    prompts: list[str] = field(default_factory=list)
    repeat_per_prompt: int = 1
    split_name: str = "train"
    content_sha256: str | None = None
    require_unique: bool = False
    sampling_strategy: str = "sequential"
    sampling_seed: int = 0
    empty_prompt_policy: str = "error"


@dataclass
class EvaluationConfig:
    path: str | None = None
    prompts: list[str] = field(default_factory=list)
    content_sha256: str | None = None
    split_name: str = "heldout"
    seeds: list[int] = field(default_factory=lambda: [1701, 1702, 1703])
    max_prompts: int | None = None


@dataclass
class SampleConfig:
    name: str = "full_trajectory"
    batch_size: int = 1
    num_steps: int = 2
    guidance_scale: float = 4.5
    samples_per_prompt: int = 2
    kl_reward: float = 0.0
    global_std: bool = False
    max_group_std: bool = False
    noise_level: float | None = 0.7
    sde_window_size: int | None = None
    sde_window_range: list[int] | None = None
    sde_type: str | None = "flow_sde"
    diffusion_clip: bool = False
    diffusion_clip_value: float = 0.45


@dataclass
class AlgorithmConfig:
    name: str = "grpo"
    objective_version: str = "legacy"
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0
    advantage_mode: str = "grpo"
    advantage_epsilon: float = 1e-6
    advantage_dtype: str = "float32"
    weight_advantages: bool = False
    credit_assignment: str = "all"
    noise_weighting: dict[str, Any] = field(default_factory=dict)
    branch: dict[str, Any] = field(default_factory=dict)
    rectification: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    lora_path: str | None = None
    max_steps: int = 1
    save_every: int = 1
    precision: str = "fp32"
    update_microbatch_size: int | None = None


@dataclass
class RewardConfig:
    provider: str = "reward_router"
    provider_params: dict[str, Any] = field(default_factory=dict)
    replace_defaults: bool = False
    weights: dict[str, float] = field(default_factory=lambda: {"mock": 1.0})
    clients: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"mock": {"name": "mock"}}
    )
    schedule: list[dict[str, Any]] = field(default_factory=list)
    cache_dir: str | None = None
    fail_policy: str = "invalid"


@dataclass
class RewardExecutorConfig:
    """Execution policy for synchronous or bounded asynchronous scoring."""

    mode: str = "sync"
    max_workers: int = 4
    microbatch_size: int | None = None
    timeout_s: float = 30.0
    max_retries: int = 0
    submit_timeout_s: float = 30.0
    max_in_flight: int | None = None
    require_hard_timeout: bool = False


@dataclass
class ConditionalScalingConfig:
    """Evidence-gated stages that are intentionally unavailable by default."""

    split_roles: bool = False
    fsdp2: bool = False


@dataclass
class DistributedConfig:
    """Native torchrun/DDP runtime options; rank identity comes from the environment."""

    backend: str | None = None
    device: str | None = None
    timeout_s: float = 30.0
    max_snapshot_tensor_bytes: int | None = 1 << 30


@dataclass
class OptimizerConfig:
    name: str = "algorithm"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectPaths:
    output_dir: str = "runs/default"
    pretrained_model: str | None = None
    resume_from: str | None = None


@dataclass
class RunnerConfig:
    auto_load_model: bool = True
    strict_rollout_validation: bool = False
    disable_rollout_cache: bool = False
    rollout_cache_dir: str | None = None
    show_progress: bool = True
    progress_interval: int = 1
    progress_leave: bool = False
    deterministic_run_dir: bool = True
    deterministic_runtime: bool = False
    checkpoint_keep_last: int | None = None
    rollout_cache_keep_last: int | None = None
    rollout_cache_max_bytes: int | None = None
    artifact_max_bytes: int | None = None
    reward_executor: RewardExecutorConfig = field(default_factory=RewardExecutorConfig)
    conditional_scaling: ConditionalScalingConfig = field(
        default_factory=ConditionalScalingConfig
    )
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


@dataclass
class VisualRLConfig:
    run_name: str
    seed: int = 42
    use_lora: bool = True
    per_prompt_stat_tracking: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    rollout: dict[str, Any] = field(default_factory=dict)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    paths: ProjectPaths = field(default_factory=ProjectPaths)


_EXTERNAL_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_EXTERNAL_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_EXTERNAL_PROVIDER_FIELDS = frozenset(
    {
        "controls",
        "dependencies",
        "params",
        "reward_name",
        "source_sha256",
        "target",
        "version",
        "weight",
    }
)


@dataclass(frozen=True)
class ExternalProviderMetadata:
    target: str
    version: str
    source_sha256: str
    dependencies: tuple[str, ...]
    params: dict[str, Any]
    reward_name: str
    weight: float
    controls: Any = None


def _json_canonical(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-safe: {exc}") from exc


def external_provider_metadata(
    provider_name: str,
    provider_params: Mapping[str, Any],
    weights: Mapping[str, float],
) -> ExternalProviderMetadata:
    """Validate and canonicalize the shared external-provider declaration."""

    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("External provider name must be a non-empty string")
    if not isinstance(provider_params, Mapping):
        raise TypeError("External provider_params must be a mapping")
    unknown = sorted(set(provider_params).difference(_EXTERNAL_PROVIDER_FIELDS))
    if unknown:
        raise ValueError(f"Unknown external provider_params fields: {unknown}")

    target = provider_params.get("target")
    if not isinstance(target, str) or not _EXTERNAL_TARGET_PATTERN.fullmatch(target):
        raise ValueError(
            f"External provider {provider_name!r} has invalid target {target!r}; "
            "expected module:attribute"
        )
    version = provider_params.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"External provider {provider_name!r} requires a non-empty version"
        )
    source_sha256 = provider_params.get("source_sha256")
    if not isinstance(source_sha256, str) or not _EXTERNAL_SHA256_PATTERN.fullmatch(
        source_sha256
    ):
        raise ValueError(
            f"External provider {provider_name!r} requires a 64-character source_sha256"
        )

    raw_dependencies = provider_params.get("dependencies", ())
    if not isinstance(raw_dependencies, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in raw_dependencies
    ):
        raise ValueError(
            f"External provider {provider_name!r} has an invalid dependency declaration"
        )
    dependencies = tuple(raw_dependencies)

    raw_params = provider_params.get("params", {})
    if not isinstance(raw_params, Mapping):
        raise TypeError(
            f"External provider {provider_name!r} params must be a JSON-safe mapping"
        )
    params = _json_canonical(
        dict(raw_params), label=f"External provider {provider_name!r} params"
    )
    controls = (
        None
        if "controls" not in provider_params
        else _json_canonical(
            provider_params["controls"],
            label=f"External provider {provider_name!r} controls",
        )
    )

    reward_name = provider_params.get("reward_name", provider_name)
    if not isinstance(reward_name, str) or not reward_name.strip():
        raise ValueError(
            f"External provider {provider_name!r} requires a non-empty reward_name"
        )
    reward_name = reward_name.strip()
    if not isinstance(weights, Mapping) or set(weights) != {reward_name}:
        raise ValueError(
            "rewards.weights must contain exactly one external reward entry for "
            f"{reward_name!r}"
        )
    canonical_weight = weights[reward_name]
    if (
        isinstance(canonical_weight, bool)
        or not isinstance(canonical_weight, (int, float))
        or not math.isfinite(float(canonical_weight))
    ):
        raise ValueError(f"rewards.weights[{reward_name!r}] must be a finite number")
    weight = float(canonical_weight)

    legacy_weight = provider_params.get("weight", weight)
    if (
        isinstance(legacy_weight, bool)
        or not isinstance(legacy_weight, (int, float))
        or not math.isfinite(float(legacy_weight))
    ):
        raise ValueError("External provider legacy weight must be a finite number")
    if float(legacy_weight) != weight:
        raise ValueError(
            "External provider legacy weight does not match canonical "
            f"rewards.weights[{reward_name!r}]"
        )

    return ExternalProviderMetadata(
        target=target,
        version=version.strip(),
        source_sha256=source_sha256.lower(),
        dependencies=dependencies,
        params=params,
        reward_name=reward_name,
        weight=weight,
        controls=controls,
    )


_ALGORITHM_SAMPLE_PAIRS = {
    "grpo": {"full_trajectory"},
    "flash_grpo": {"single_step"},
    "tempflow_grpo": {"branching"},
}


def _validate_value_type(value: Any, expected: Any, *, section: str) -> None:
    if expected is Any:
        return

    origin = get_origin(expected)
    if origin in (Union, types.UnionType):
        for option in get_args(expected):
            try:
                _validate_value_type(value, option, section=section)
            except TypeError:
                continue
            return
        raise TypeError(f"{section} has an invalid type: {type(value).__name__}")
    if expected is type(None):
        if value is None:
            return
        raise TypeError(f"{section} must be None")
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{section} must be a list")
        (item_type,) = get_args(expected) or (Any,)
        for index, item in enumerate(value):
            _validate_value_type(item, item_type, section=f"{section}[{index}]")
        return
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{section} must be a mapping")
        key_type, value_type = get_args(expected) or (Any, Any)
        for key, item in value.items():
            _validate_value_type(key, key_type, section=f"{section} key")
            _validate_value_type(item, value_type, section=f"{section}.{key}")
        return
    if expected is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return
        raise TypeError(f"{section} must be a float")
    if expected is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return
        raise TypeError(f"{section} must be an int")
    if expected is bool:
        if isinstance(value, bool):
            return
        raise TypeError(f"{section} must be a bool")
    if isinstance(expected, type) and isinstance(value, expected):
        return
    expected_name = getattr(expected, "__name__", str(expected))
    raise TypeError(f"{section} must be {expected_name}")


def _build_dataclass(cls, src: Mapping[str, Any], *, section: str = "config"):
    if not isinstance(src, Mapping):
        raise TypeError(f"{section} must be a mapping")
    non_string_keys = [key for key in src if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"{section} field names must be strings")

    kwargs = {}
    field_by_name = {item.name: item for item in fields(cls)}
    unknown = sorted(set(src).difference(field_by_name))
    if unknown:
        raise ValueError(f"Unknown fields in {section}: {unknown}")
    type_hints = get_type_hints(cls)
    for name, item in field_by_name.items():
        if name not in src:
            if item.default is MISSING and item.default_factory is MISSING:
                raise ValueError(f"Missing required field {section}.{name}")
            continue
        value = src[name]
        expected = type_hints[name]
        if isinstance(expected, type) and is_dataclass(expected):
            kwargs[name] = _build_dataclass(
                expected, value, section=f"{section}.{name}"
            )
        else:
            _validate_value_type(value, expected, section=f"{section}.{name}")
            kwargs[name] = value
    return cls(**kwargs)


def _validate_algorithm_sample_pair(cfg: VisualRLConfig) -> None:
    algorithm_name = cfg.algorithm.name
    sample_name = cfg.sample.name
    allowed_samples = _ALGORITHM_SAMPLE_PAIRS.get(algorithm_name)
    if allowed_samples is None or sample_name in allowed_samples:
        return

    expected = ", ".join(f"{name!r}" for name in sorted(allowed_samples))
    raise ValueError(
        f"Incompatible config: algorithm.name={algorithm_name!r} requires "
        f"sample.name in {{{expected}}}, got sample.name={sample_name!r}."
    )


def _validate_evaluation_config(cfg: VisualRLConfig) -> None:
    evaluation = cfg.evaluation
    if evaluation.path and evaluation.prompts:
        raise ValueError("evaluation config cannot provide both prompts and path")
    if not isinstance(evaluation.split_name, str) or not evaluation.split_name.strip():
        raise ValueError("evaluation.split_name must be non-empty")
    if not evaluation.seeds:
        raise ValueError("evaluation.seeds must contain at least one seed")
    if evaluation.max_prompts is not None and evaluation.max_prompts < 1:
        raise ValueError("evaluation.max_prompts must be positive when provided")


def _validate_positive_integer(path: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value <= 0:
        raise ValueError(f"{path} must be positive")


def _validate_finite_number(
    path: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{path} must be finite")
    if minimum is not None and (
        resolved < minimum if minimum_inclusive else resolved <= minimum
    ):
        relation = ">=" if minimum_inclusive else ">"
        raise ValueError(f"{path} must be {relation} {minimum}")
    if maximum is not None and (
        resolved > maximum if maximum_inclusive else resolved >= maximum
    ):
        relation = "<=" if maximum_inclusive else "<"
        raise ValueError(f"{path} must be {relation} {maximum}")


def _validate_core_numeric_config(cfg: VisualRLConfig) -> None:
    """Fail before runtime construction on invalid training semantics."""

    _validate_positive_integer("sample.batch_size", cfg.sample.batch_size)
    _validate_positive_integer("sample.num_steps", cfg.sample.num_steps)
    # SD3 and Wan intentionally support no-CFG values at and below 1.0.
    _validate_finite_number("sample.guidance_scale", cfg.sample.guidance_scale)

    _validate_finite_number(
        "algorithm.clip_range",
        cfg.algorithm.clip_range,
        minimum=0.0,
        maximum=1.0,
    )
    _validate_finite_number(
        "algorithm.adv_clip_max",
        cfg.algorithm.adv_clip_max,
        minimum=0.0,
    )
    _validate_finite_number(
        "algorithm.beta",
        cfg.algorithm.beta,
        minimum=0.0,
    )
    _validate_finite_number(
        "algorithm.advantage_epsilon",
        cfg.algorithm.advantage_epsilon,
        minimum=0.0,
        minimum_inclusive=False,
    )

    _validate_finite_number(
        "train.learning_rate",
        cfg.train.learning_rate,
        minimum=0.0,
    )
    _validate_positive_integer("train.max_steps", cfg.train.max_steps)
    for name in ("adam_beta1", "adam_beta2"):
        _validate_finite_number(
            f"train.{name}",
            getattr(cfg.train, name),
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
    _validate_finite_number(
        "train.adam_weight_decay",
        cfg.train.adam_weight_decay,
        minimum=0.0,
    )
    _validate_finite_number(
        "train.adam_epsilon",
        cfg.train.adam_epsilon,
        minimum=0.0,
        minimum_inclusive=False,
    )
    if cfg.train.max_grad_norm is not None:
        _validate_finite_number(
            "train.max_grad_norm",
            cfg.train.max_grad_norm,
            minimum=0.0,
            minimum_inclusive=False,
        )


_REWARD_SCHEDULE_PHASE_FIELDS = frozenset({"name", "start_step", "end_step", "weights"})


def normalize_reward_schedule(
    schedule: Any,
    *,
    weights: Mapping[str, Any],
    clients: Mapping[str, Any],
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Validate and detach a step-aware reward schedule.

    Phases use zero-based, half-open step intervals.  Omitting a reward from a
    phase is the only supported way to disable it, which lets the router avoid
    calling an inactive scorer altogether.
    """

    if not isinstance(schedule, list):
        raise TypeError("rewards.schedule must be a list")
    if not schedule:
        return []
    if not isinstance(weights, Mapping):
        raise TypeError("rewards.weights must be a mapping")
    if not isinstance(clients, Mapping):
        raise TypeError("rewards.clients must be a mapping")

    declared_weights = set(weights)
    declared_clients = set(clients)
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    previous_end: int | None = None

    for index, phase in enumerate(schedule):
        path = f"rewards.schedule[{index}]"
        if not isinstance(phase, Mapping):
            raise TypeError(f"{path} must be a mapping")
        unknown = sorted(set(phase).difference(_REWARD_SCHEDULE_PHASE_FIELDS))
        missing = sorted(_REWARD_SCHEDULE_PHASE_FIELDS.difference(phase))
        if unknown:
            raise ValueError(f"{path} has unknown fields: {unknown}")
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")

        name = phase["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        name = name.strip()
        if name in names:
            raise ValueError(f"rewards.schedule phase name {name!r} is duplicated")
        names.add(name)

        start_step = phase["start_step"]
        end_step = phase["end_step"]
        for field_name, value in (
            ("start_step", start_step),
            ("end_step", end_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{path}.{field_name} must be an integer")
        if start_step < 0:
            raise ValueError(f"{path}.start_step must be non-negative")
        if end_step <= start_step:
            raise ValueError(f"{path}.end_step must be greater than start_step")
        expected_start = 0 if previous_end is None else previous_end
        if start_step != expected_start:
            raise ValueError(
                "rewards.schedule phases must be contiguous and start at step 0: "
                f"{path}.start_step is {start_step}, expected {expected_start}"
            )

        phase_weights = phase["weights"]
        if not isinstance(phase_weights, Mapping):
            raise TypeError(f"{path}.weights must be a mapping")
        if not phase_weights:
            raise ValueError(f"{path}.weights must not be empty")
        unknown_weights = sorted(set(phase_weights).difference(declared_weights))
        if unknown_weights:
            raise ValueError(
                f"{path}.weights reference undeclared rewards.weights entries: "
                f"{unknown_weights}"
            )
        unknown_clients = sorted(set(phase_weights).difference(declared_clients))
        if unknown_clients:
            raise ValueError(
                f"{path}.weights reference undeclared rewards.clients entries: "
                f"{unknown_clients}"
            )

        normalized_weights: dict[str, float] = {}
        for reward_name, weight in phase_weights.items():
            weight_path = f"{path}.weights[{reward_name!r}]"
            if not isinstance(reward_name, str) or not reward_name:
                raise ValueError(f"{path}.weights keys must be non-empty strings")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
            ):
                raise ValueError(f"{weight_path} must be a finite number")
            if float(weight) == 0.0:
                raise ValueError(
                    f"{weight_path} must be non-zero; omit inactive rewards instead"
                )
            normalized_weights[reward_name] = float(weight)

        normalized.append(
            {
                "name": name,
                "start_step": start_step,
                "end_step": end_step,
                "weights": normalized_weights,
            }
        )
        previous_end = end_step

    if max_steps is not None and previous_end is not None and previous_end < max_steps:
        raise ValueError(
            "rewards.schedule must cover train.max_steps: "
            f"final end_step is {previous_end}, train.max_steps is {max_steps}"
        )
    return normalized


def _validate_reward_schedule_config(cfg: VisualRLConfig) -> None:
    schedule = normalize_reward_schedule(
        cfg.rewards.schedule,
        weights=cfg.rewards.weights,
        clients=cfg.rewards.clients,
        max_steps=cfg.train.max_steps,
    )
    if not schedule:
        return
    if cfg.rewards.provider != "reward_router":
        raise ValueError(
            "rewards.schedule is only supported by rewards.provider='reward_router'"
        )
    if cfg.algorithm.weight_advantages:
        raise ValueError(
            "rewards.schedule is incompatible with algorithm.weight_advantages=true; "
            "scheduled raw reward keys are phase-dependent"
        )


def _validate_runtime_config(cfg: VisualRLConfig) -> None:
    if cfg.train.precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("train.precision must be one of: fp32, bf16, fp16")
    if (
        cfg.train.update_microbatch_size is not None
        and cfg.train.update_microbatch_size < 1
    ):
        raise ValueError("train.update_microbatch_size must be positive")
    for name in (
        "checkpoint_keep_last",
        "rollout_cache_keep_last",
        "rollout_cache_max_bytes",
        "artifact_max_bytes",
    ):
        value = getattr(cfg.runner, name)
        if value is not None and value < 0:
            raise ValueError(f"runner.{name} must be non-negative")

    executor = cfg.runner.reward_executor
    if executor.mode not in {"sync", "async"}:
        raise ValueError("runner.reward_executor.mode must be one of: sync, async")
    if isinstance(executor.max_workers, bool) or not isinstance(
        executor.max_workers, int
    ):
        raise TypeError("runner.reward_executor.max_workers must be an integer")
    if executor.max_workers < 1:
        raise ValueError("runner.reward_executor.max_workers must be positive")
    if executor.microbatch_size is not None:
        if isinstance(executor.microbatch_size, bool) or not isinstance(
            executor.microbatch_size, int
        ):
            raise TypeError(
                "runner.reward_executor.microbatch_size must be an integer or null"
            )
        if executor.microbatch_size < 1:
            raise ValueError(
                "runner.reward_executor.microbatch_size must be positive when provided"
            )
    if executor.max_in_flight is not None:
        if isinstance(executor.max_in_flight, bool) or not isinstance(
            executor.max_in_flight, int
        ):
            raise TypeError(
                "runner.reward_executor.max_in_flight must be an integer or null"
            )
        if executor.max_in_flight < 1:
            raise ValueError(
                "runner.reward_executor.max_in_flight must be positive when provided"
            )
    if isinstance(executor.max_retries, bool) or not isinstance(
        executor.max_retries, int
    ):
        raise TypeError("runner.reward_executor.max_retries must be an integer")
    if executor.max_retries < 0:
        raise ValueError("runner.reward_executor.max_retries must be non-negative")
    if isinstance(executor.timeout_s, bool) or not isinstance(
        executor.timeout_s, (int, float)
    ):
        raise TypeError("runner.reward_executor.timeout_s must be numeric")
    if not math.isfinite(executor.timeout_s) or executor.timeout_s <= 0:
        raise ValueError("runner.reward_executor.timeout_s must be finite and positive")
    if isinstance(executor.submit_timeout_s, bool) or not isinstance(
        executor.submit_timeout_s, (int, float)
    ):
        raise TypeError("runner.reward_executor.submit_timeout_s must be numeric")
    if not math.isfinite(executor.submit_timeout_s) or executor.submit_timeout_s < 0:
        raise ValueError(
            "runner.reward_executor.submit_timeout_s must be finite and non-negative"
        )
    if not isinstance(executor.require_hard_timeout, bool):
        raise TypeError("runner.reward_executor.require_hard_timeout must be a bool")
    scaling = cfg.runner.conditional_scaling
    if scaling.split_roles:
        raise ValueError(
            "runner.conditional_scaling.split_roles is evidence-gated and is not "
            "enabled in the simplified core"
        )
    if scaling.fsdp2:
        raise ValueError(
            "runner.conditional_scaling.fsdp2 is evidence-gated and is not enabled "
            "in the simplified core"
        )
    distributed = cfg.runner.distributed
    if distributed.backend not in {None, "gloo", "nccl"}:
        raise ValueError("runner.distributed.backend must be one of: gloo, nccl, null")
    if distributed.device is not None and (
        not isinstance(distributed.device, str) or not distributed.device.strip()
    ):
        raise ValueError(
            "runner.distributed.device must be a non-empty device string or null"
        )
    if isinstance(distributed.timeout_s, bool) or not isinstance(
        distributed.timeout_s, (int, float)
    ):
        raise TypeError("runner.distributed.timeout_s must be numeric")
    if not math.isfinite(distributed.timeout_s) or distributed.timeout_s <= 0:
        raise ValueError("runner.distributed.timeout_s must be finite and positive")
    snapshot_limit = distributed.max_snapshot_tensor_bytes
    if snapshot_limit is not None and (
        isinstance(snapshot_limit, bool) or not isinstance(snapshot_limit, int)
    ):
        raise TypeError(
            "runner.distributed.max_snapshot_tensor_bytes must be a positive "
            "integer or null"
        )
    if snapshot_limit is not None and snapshot_limit <= 0:
        raise ValueError(
            "runner.distributed.max_snapshot_tensor_bytes must be positive"
        )


def _validate_typed_config(cfg: VisualRLConfig) -> None:
    """Apply every semantic validator to an already type-checked config."""

    _validate_algorithm_sample_pair(cfg)
    _validate_evaluation_config(cfg)
    _validate_core_numeric_config(cfg)
    _validate_reward_schedule_config(cfg)
    _validate_runtime_config(cfg)


def validate_config(config: VisualRLConfig) -> None:
    """Side-effect-free validation for direct or subsequently mutated configs.

    ``VisualRLConfig`` remains a convenient mutable Python API.  Rebuilding a
    detached copy through the schema closes the gap between configs produced by
    :func:`config_from_dict` and configs constructed (or edited) directly.
    """

    if not isinstance(config, VisualRLConfig):
        raise TypeError("config must be a VisualRLConfig")
    detached = _build_dataclass(
        VisualRLConfig,
        deepcopy(config_to_dict(config)),
    )
    _validate_typed_config(detached)


def config_from_dict(values: Mapping[str, Any]) -> VisualRLConfig:
    """Validate detached values and construct the typed runtime config."""

    if not isinstance(values, Mapping):
        raise TypeError("config must be a mapping")
    cfg = _build_dataclass(VisualRLConfig, deepcopy(dict(values)))
    _validate_typed_config(cfg)
    return cfg


def load_config(path: str | Path) -> VisualRLConfig:
    """Load either a resolver envelope or a legacy full-config YAML file."""

    from visual_rl.configs.resolver import resolve_experiment
    from visual_rl.configs.sources import read_experiment_spec

    return resolve_experiment(read_experiment_spec(path)).config


def config_to_dict(config: VisualRLConfig) -> dict[str, Any]:
    return asdict(config)


def section_to_dict(section: Any) -> dict[str, Any]:
    if is_dataclass(section):
        return asdict(section)
    return dict(section)
