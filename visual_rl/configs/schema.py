"""Strict typed configuration for the simplified VisualRL mainline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


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


@dataclass
class EvaluationConfig:
    path: str | None = None
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
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0
    advantage_mode: str = "grpo"
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
    lora_path: str | None = None
    max_steps: int = 1
    save_every: int = 1


@dataclass
class RewardConfig:
    provider: str = "reward_router"
    provider_params: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: {"mock": 1.0})
    clients: dict[str, dict[str, Any]] = field(default_factory=lambda: {"mock": {"name": "mock"}})
    cache_dir: str | None = None
    fail_policy: str = "invalid"


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


_ALGORITHM_SAMPLE_PAIRS = {
    "grpo": {"full_trajectory"},
    "flash_grpo": {"single_step"},
    "tempflow_grpo": {"branching"},
}


def _build_dataclass(cls, src: dict[str, Any], *, section: str = "config"):
    if src is None:
        return cls()
    if not isinstance(src, dict):
        raise TypeError(f"{section} must be a mapping")
    kwargs = {}
    field_by_name = {item.name: item for item in fields(cls)}
    unknown = sorted(set(src).difference(field_by_name))
    if unknown:
        raise ValueError(f"Unknown fields in {section}: {unknown}")
    for name, item in field_by_name.items():
        if name not in src:
            continue
        value = src[name]
        default = item.default
        default_factory = getattr(item, "default_factory", None)
        if is_dataclass(default):
            kwargs[name] = _build_dataclass(
                type(default), value, section=f"{section}.{name}"
            )
        elif default_factory is not None:
            try:
                default_value = default_factory()
            except TypeError:
                default_value = None
            if is_dataclass(default_value):
                kwargs[name] = _build_dataclass(
                    type(default_value), value, section=f"{section}.{name}"
                )
            else:
                kwargs[name] = value
        else:
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


def load_config(path: str | Path) -> VisualRLConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "run_name" not in data:
        data["run_name"] = path.stem
    cfg = _build_dataclass(VisualRLConfig, data)
    _validate_algorithm_sample_pair(cfg)
    return cfg


def config_to_dict(config: VisualRLConfig) -> dict[str, Any]:
    return asdict(config)


def section_to_dict(section: Any) -> dict[str, Any]:
    if is_dataclass(section):
        return asdict(section)
    return dict(section)
