"""Typed config loading and validation for VisualRL.

v0.2 borrows GenRL's strongly typed config shape while keeping the v0.1
`visual_rl` preset format compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FSDPConfig:
    auto_wrap_policy: str = "transformer_based_wrap"
    backward_prefetch: str = "backward_pre"
    forward_prefetch: bool = True
    cpu_ram_efficient_loading: bool = False
    cpu_offload: bool = False
    sharding_strategy: str = "full_shard"
    state_dict_type: str = "sharded_state_dict"
    sync_module_states: bool = False
    use_orig_params: bool = True
    activation_checkpointing: bool = True


@dataclass
class AccelerateConfig:
    distributed_type: str = "FSDP"
    mixed_precision: str = "bf16"
    num_processes: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


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
    prompt_fn: str = "inline"


@dataclass
class SampleConfig:
    name: str = "full_trajectory"
    batch_size: int = 1
    eval_batch_size: int = 1
    num_batches_per_epoch: int = 1
    num_steps: int = 2
    eval_num_steps: int = 2
    guidance_scale: float = 4.5
    eval_guidance_scale: float | None = None
    num_video_per_prompt: int = 1
    samples_per_prompt: int = 1
    kl_reward: float = 0.0
    global_std: bool = False
    max_group_std: bool = False
    same_latent: bool = False
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


@dataclass
class TrainConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int | None = None
    num_inner_epochs: int = 1
    timestep_fraction: float = 1.0
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False
    ema: bool = False
    ema_decay: float = 0.9
    ema_update_interval: int = 8
    cfg: bool = True
    full_finetune: bool = False
    lora_path: str | None = None
    max_steps: int = 1
    save_every: int = 1


@dataclass
class RewardClientConfig:
    name: str = "mock"
    version: str = "v1"
    timeout: float = 1000.0
    retries: int = 2
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=lambda: {"mock": 1.0})
    clients: dict[str, dict[str, Any]] = field(default_factory=lambda: {"mock": {"name": "mock"}})
    normalize: str = "none"
    cache_dir: str | None = None
    fail_policy: str = "invalid"


@dataclass
class ProjectPaths:
    output_dir: str = "runs/default"
    save_dir: str | None = None
    dataset: str | None = None
    pretrained_model: str | None = None
    resume_from: str | None = None
    accelerate_config: str | None = None


@dataclass
class VisualRLConfig:
    run_name: str
    seed: int = 42
    output_dir: str = "runs/default"
    project_name: str = "VisualRL"
    num_epochs: int = 1
    save_freq: int = 1
    eval_freq: int = 1
    use_lora: bool = True
    allow_tf32: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    per_prompt_stat_tracking: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    rollout: dict[str, Any] = field(default_factory=dict)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: dict[str, Any] = field(default_factory=dict)
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)
    legacy: dict[str, Any] = field(default_factory=dict)


_ALGORITHM_SAMPLE_PAIRS = {
    "grpo": {"full_trajectory"},
    "flash_grpo": {"single_step"},
    "tempflow_grpo": {"branching"},
}


def _build_dataclass(cls, src: dict[str, Any]):
    if src is None:
        return cls()
    kwargs = {}
    field_by_name = {item.name: item for item in fields(cls)}
    for name, item in field_by_name.items():
        if name not in src:
            continue
        value = src[name]
        default = item.default
        default_factory = getattr(item, "default_factory", None)
        if is_dataclass(default):
            kwargs[name] = _build_dataclass(type(default), value)
        elif default_factory is not None:
            try:
                default_value = default_factory()
            except TypeError:
                default_value = None
            if is_dataclass(default_value):
                kwargs[name] = _build_dataclass(type(default_value), value)
            else:
                kwargs[name] = value
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _normalize_legacy_keys(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if "rollout" in normalized:
        sample = dict(normalized.get("sample", {}))
        sample.update(normalized["rollout"])
        normalized["sample"] = sample
    if "trainer" in normalized:
        train = dict(normalized.get("train", {}))
        train.update(normalized["trainer"])
        normalized["train"] = train
    if "output_dir" in normalized:
        paths = dict(normalized.get("paths", {}))
        paths.setdefault("output_dir", normalized["output_dir"])
        paths.setdefault("save_dir", normalized["output_dir"])
        normalized["paths"] = paths
    return normalized


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
    data = _normalize_legacy_keys(data)
    cfg = _build_dataclass(VisualRLConfig, data)
    if cfg.paths.output_dir == "runs/default" and cfg.output_dir != "runs/default":
        cfg.paths.output_dir = cfg.output_dir
    if cfg.output_dir == "runs/default" and cfg.paths.output_dir != "runs/default":
        cfg.output_dir = cfg.paths.output_dir
    if cfg.paths.save_dir is None:
        cfg.paths.save_dir = cfg.paths.output_dir
    _validate_algorithm_sample_pair(cfg)
    return cfg


def config_to_dict(config: VisualRLConfig) -> dict[str, Any]:
    return asdict(config)


def section_to_dict(section: Any) -> dict[str, Any]:
    if is_dataclass(section):
        return asdict(section)
    return dict(section)
