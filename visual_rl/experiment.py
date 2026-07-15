"""Small, composable Python API for the existing VisualRL execution path."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal

from visual_rl.configs.resolver import resolve_experiment
from visual_rl.configs.schema import VisualRLConfig, config_to_dict
from visual_rl.configs.sources import ConfigDocument, ExperimentSpec, SourceRef
from visual_rl.preflight import PreflightReport, static_preflight


def _path(value: str | os.PathLike[str]) -> str:
    return os.fspath(value)


_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_EXTERNAL_PROVIDER_NAME = "external"


def _json_canonical(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-safe: {exc}") from exc


def _resolve_object_target(component: object) -> tuple[str, str]:
    if not callable(component):
        raise TypeError(
            "component must be a callable, class, or module:attribute string"
        )
    name = getattr(component, "__name__", "")
    qualname = getattr(component, "__qualname__", "")
    module_name = getattr(component, "__module__", "")
    if name == "<lambda>" or not module_name or not qualname or "<locals>" in qualname:
        raise ValueError(
            "component must be an importable module-level callable or class; "
            "lambdas and local definitions are not auditable"
        )
    if inspect.isfunction(component) and "." in qualname:
        raise ValueError("feedback functions must be defined at module level")
    target = f"{module_name}:{qualname}"
    if not _TARGET_PATTERN.fullmatch(target):
        raise ValueError(f"component has no stable import target: {target!r}")

    resolved: object = importlib.import_module(module_name)
    for attribute_name in qualname.split("."):
        try:
            resolved = getattr(resolved, attribute_name)
        except AttributeError as exc:
            raise ValueError(f"component is not importable from {target!r}") from exc
    if resolved is not component:
        raise ValueError(f"component target {target!r} resolves to a different object")

    source_file = inspect.getsourcefile(component)
    if source_file is None:
        raise ValueError(f"component {target!r} has no auditable source file")
    source_path = Path(source_file).resolve()
    if not source_path.is_file():
        raise ValueError(f"component source is not a readable file: {source_path}")
    return target, hashlib.sha256(source_path.read_bytes()).hexdigest()


def _validate_external_target(target: str) -> str:
    if not _TARGET_PATTERN.fullmatch(target):
        raise ValueError(
            f"component target {target!r} is invalid; expected module:attribute"
        )
    return target


def _validate_source_sha256(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError("source_sha256 is required for a string component target")
        return None
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("source_sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


@dataclass(frozen=True)
class MockWan:
    latent_shape: tuple[int, ...] = (4, 2, 2, 2)
    media_shape: tuple[int, ...] = (4, 3, 16, 16)

    def to_config(self) -> dict[str, Any]:
        return {
            "model": {
                "name": "mock_wan",
                "model_family": "wan",
                "latent_shape": list(self.latent_shape),
                "media_shape": list(self.media_shape),
            }
        }


@dataclass(frozen=True)
class TinyDiffusion:
    checkpoint: str | os.PathLike[str] | None = None
    image_size: int = 16
    device: str = "cpu"

    def to_config(self) -> dict[str, Any]:
        model: dict[str, Any] = {
            "name": "tiny_diffusion",
            "model_family": "image",
            "extra": {"image_size": self.image_size, "device": self.device},
        }
        if self.checkpoint is not None:
            model["model_path"] = _path(self.checkpoint)
        return {"model": model}


@dataclass(frozen=True)
class Wan:
    checkpoint: str | os.PathLike[str]
    world_r1_root: str | os.PathLike[str] = "World-R1-main"
    device: str = "cuda"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    low_cpu_mem_usage: bool = True
    backend: Literal["world_r1", "flash"] = "world_r1"
    flash_grpo_root: str | os.PathLike[str] = "Flash-GRPO-main"
    gradient_checkpointing: bool | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"world_r1", "flash"}:
            raise ValueError("backend must be one of: flash, world_r1")
        if self.gradient_checkpointing is not None and not isinstance(
            self.gradient_checkpointing, bool
        ):
            raise TypeError("gradient_checkpointing must be a bool or None")

    def to_config(self) -> dict[str, Any]:
        reference_root = (
            {"world_r1_root": _path(self.world_r1_root)}
            if self.backend == "world_r1"
            else {"flash_grpo_root": _path(self.flash_grpo_root)}
        )
        backend_config = {} if self.backend == "world_r1" else {"wan_backend": "flash"}
        return {
            "model": {
                "name": "world_r1_wan_legacy",
                "model_path": _path(self.checkpoint),
                "model_family": "wan",
                "extra": {
                    **backend_config,
                    **reference_root,
                    "device": self.device,
                    "dtype": self.dtype,
                    "local_files_only": self.local_files_only,
                    "low_cpu_mem_usage": self.low_cpu_mem_usage,
                    **(
                        {"gradient_checkpointing": self.gradient_checkpointing}
                        if self.gradient_checkpointing is not None
                        else {}
                    ),
                },
            }
        }


@dataclass(frozen=True)
class SD3:
    checkpoint: str | os.PathLike[str]
    repo_root: str | os.PathLike[str] = "reference_code/TempFlow-GRPO-main"
    resolution: int = 512
    device: str = "cuda"
    dtype: str = "bfloat16"
    lora_rank: int = 32
    lora_alpha: int = 64
    max_sequence_length: int = 128
    reference_mode: bool = True
    gradient_checkpointing: bool | None = None

    def __post_init__(self) -> None:
        if self.gradient_checkpointing is not None and not isinstance(
            self.gradient_checkpointing, bool
        ):
            raise TypeError("gradient_checkpointing must be a bool or None")

    def to_config(self) -> dict[str, Any]:
        return {
            "model": {
                "name": "sd3_tempflow",
                "model_path": _path(self.checkpoint),
                "model_family": "sd3",
                "extra": {
                    "repo_root": _path(self.repo_root),
                    "resolution": self.resolution,
                    "device": self.device,
                    "dtype": self.dtype,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "max_sequence_length": self.max_sequence_length,
                    "tempflow_reference_mode": self.reference_mode,
                    **(
                        {"gradient_checkpointing": self.gradient_checkpointing}
                        if self.gradient_checkpointing is not None
                        else {}
                    ),
                },
            }
        }


@dataclass(frozen=True)
class FullTrajectory:
    num_steps: int = 2
    batch_size: int = 1
    samples_per_prompt: int = 2
    guidance_scale: float = 4.5
    noise_level: float | None = 0.7

    def to_config(self) -> dict[str, Any]:
        return {
            "sample": {
                "name": "full_trajectory",
                "num_steps": self.num_steps,
                "batch_size": self.batch_size,
                "samples_per_prompt": self.samples_per_prompt,
                "guidance_scale": self.guidance_scale,
                "noise_level": self.noise_level,
            }
        }


@dataclass(frozen=True)
class Flash:
    selected_steps: int = 4
    batch_size: int = 1
    samples_per_prompt: int = 4
    selected_step_strategy: str = "iso_temporal"
    rectification_mode: str = "scheduler_formula"

    def to_config(self) -> dict[str, Any]:
        return {
            "sample": {
                "name": "single_step",
                "num_steps": self.selected_steps,
                "batch_size": self.batch_size,
                "samples_per_prompt": self.samples_per_prompt,
            },
            "rollout": {
                "selected_step_strategy": self.selected_step_strategy,
                "timestep_range": [0, self.selected_steps - 1],
                "rectification_mode": self.rectification_mode,
            },
        }


@dataclass(frozen=True)
class Branching:
    num_steps: int = 4
    branch_count: int = 3
    exploration_k: int | None = None
    include_main: bool = False
    batch_size: int = 1
    samples_per_prompt: int = 4
    branch_timestep_strategy: str = "cycle"

    def to_config(self) -> dict[str, Any]:
        exploration_k = (
            self.branch_count if self.exploration_k is None else self.exploration_k
        )
        return {
            "sample": {
                "name": "branching",
                "num_steps": self.num_steps,
                "batch_size": self.batch_size,
                "samples_per_prompt": self.samples_per_prompt,
            },
            "rollout": {
                "branch_count": self.branch_count,
                "exploration_k": exploration_k,
                "include_main": self.include_main,
                "branch_timesteps": "auto",
                "branch_timestep_strategy": self.branch_timestep_strategy,
            },
        }


@dataclass(frozen=True)
class MockReward:
    weight: float = 1.0
    mode: str = "prompt_media"
    fail_policy: str = "invalid"

    def to_config(self) -> dict[str, Any]:
        return {
            "rewards": {
                "replace_defaults": True,
                "weights": {"mock": self.weight},
                "clients": {
                    "mock": {"name": "mock", "version": "v2", "mode": self.mode}
                },
                "fail_policy": self.fail_policy,
            }
        }


@dataclass(frozen=True)
class PromptColor:
    weight: float = 1.0
    default_color: str = "red"
    fail_policy: str = "raise"

    def to_config(self) -> dict[str, Any]:
        return {
            "rewards": {
                "replace_defaults": True,
                "weights": {"prompt_color": self.weight},
                "clients": {
                    "prompt_color": {
                        "name": "prompt_color",
                        "version": "v1",
                        "default_color": self.default_color,
                    }
                },
                "fail_policy": self.fail_policy,
            }
        }


@dataclass(frozen=True)
class WorldR1:
    general_url: str
    geometry_url: str | None = None
    general_weight: float = 1.0
    geometry_weight: float = 1.0
    general_timeout: float = 1000.0
    geometry_timeout: float = 2000.0
    retries: int = 2
    fail_policy: str = "invalid"
    wire_format: str = "json_v1"
    allow_unsafe_pickle: bool = False
    trusted_hosts: tuple[str, ...] = ()
    max_response_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        from visual_rl.feedback.clients import (
            validate_max_response_bytes,
            validate_wire_security_policy,
        )

        validate_max_response_bytes(self.max_response_bytes)
        urls = (
            (self.general_url,)
            if self.geometry_url is None
            else (self.general_url, self.geometry_url)
        )
        for url in urls:
            validate_wire_security_policy(
                url,
                wire_format=self.wire_format,
                allow_unsafe_pickle=self.allow_unsafe_pickle,
                trusted_hosts=self.trusted_hosts,
            )

    def to_config(self) -> dict[str, Any]:
        weights = {"reward_general": self.general_weight}
        clients = {
            "reward_general": {
                "name": "reward_general",
                "version": "v1",
                "url": self.general_url,
                "timeout": self.general_timeout,
                "retries": self.retries,
                "wire_format": self.wire_format,
                "allow_unsafe_pickle": self.allow_unsafe_pickle,
                "trusted_hosts": list(self.trusted_hosts),
                "max_response_bytes": self.max_response_bytes,
            }
        }
        if self.geometry_url is not None:
            weights["reward_3d"] = self.geometry_weight
            clients["reward_3d"] = {
                "name": "reward_3d",
                "version": "v1",
                "url": self.geometry_url,
                "timeout": self.geometry_timeout,
                "retries": self.retries,
                "wire_format": self.wire_format,
                "allow_unsafe_pickle": self.allow_unsafe_pickle,
                "trusted_hosts": list(self.trusted_hosts),
                "max_response_bytes": self.max_response_bytes,
            }
        return {
            "rewards": {
                "replace_defaults": True,
                "weights": weights,
                "clients": clients,
                "fail_policy": self.fail_policy,
            }
        }


@dataclass(frozen=True, init=False)
class External:
    target: str
    version: str
    source_sha256: str
    params: dict[str, Any]
    dependencies: tuple[str, ...]
    reward_name: str
    weight: float

    def __init__(
        self,
        component: object | str,
        *,
        version: str,
        name: str | None = None,
        params: Mapping[str, Any] | None = None,
        weight: float = 1.0,
        dependencies: Iterable[str] = (),
        source_sha256: str | None = None,
    ) -> None:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version must be a non-empty string")
        if isinstance(component, str):
            target = _validate_external_target(component)
            resolved_sha256 = _validate_source_sha256(source_sha256, required=True)
            default_name = component.rsplit(":", 1)[1].rsplit(".", 1)[-1]
        else:
            target, computed_sha256 = _resolve_object_target(component)
            declared_sha256 = _validate_source_sha256(source_sha256, required=False)
            if declared_sha256 is not None and declared_sha256 != computed_sha256:
                raise ValueError(
                    "source_sha256 does not match the component source file bytes"
                )
            resolved_sha256 = computed_sha256
            default_name = getattr(component, "__name__", target.rsplit(":", 1)[1])

        reward_name = default_name if name is None else name
        if not isinstance(reward_name, str) or not reward_name.strip():
            raise ValueError("name must be a non-empty string")
        if params is None:
            params_value: dict[str, Any] = {}
        elif isinstance(params, Mapping):
            params_value = _json_canonical(dict(params), label="external reward params")
        else:
            raise TypeError("params must be a mapping")
        if isinstance(dependencies, (str, bytes)):
            raise TypeError("dependencies must be an iterable of dependency names")
        dependencies_value = tuple(dependencies)
        if any(
            not isinstance(dependency, str) or not dependency.strip()
            for dependency in dependencies_value
        ):
            raise ValueError("dependencies must contain non-empty strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("weight must be a finite number")
        weight_value = float(weight)
        if not math.isfinite(weight_value):
            raise ValueError("weight must be a finite number")

        metadata = {
            "target": target,
            "version": version.strip(),
            "source_sha256": resolved_sha256,
            "params": params_value,
            "dependencies": list(dependencies_value),
            "reward_name": reward_name.strip(),
        }
        metadata = _json_canonical(metadata, label="external reward metadata")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "version", version.strip())
        object.__setattr__(self, "source_sha256", str(resolved_sha256))
        object.__setattr__(self, "params", metadata["params"])
        object.__setattr__(self, "dependencies", dependencies_value)
        object.__setattr__(self, "reward_name", reward_name.strip())
        object.__setattr__(self, "weight", weight_value)

    def to_config(self) -> dict[str, Any]:
        provider_params = {
            "target": self.target,
            "version": self.version,
            "source_sha256": self.source_sha256,
            "params": deepcopy(self.params),
            "dependencies": list(self.dependencies),
            "reward_name": self.reward_name,
        }
        return {
            "rewards": {
                "provider": _EXTERNAL_PROVIDER_NAME,
                "provider_params": provider_params,
                "replace_defaults": True,
                "weights": {self.reward_name: self.weight},
                "clients": {},
            }
        }


@dataclass(frozen=True)
class GroupNormalize:
    epsilon: float = 1e-6
    dtype: str = "float32"
    weight_advantages: bool = False

    def to_config(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "advantage_mode": "grpo",
                "advantage_epsilon": self.epsilon,
                "advantage_dtype": self.dtype,
                "weight_advantages": self.weight_advantages,
            }
        }


@dataclass(frozen=True)
class GRPO:
    clip_range: float = 0.001
    beta: float = 0.0
    adv_clip_max: float = 5.0

    def to_config(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "name": "grpo",
                "clip_range": self.clip_range,
                "beta": self.beta,
                "adv_clip_max": self.adv_clip_max,
            }
        }


@dataclass(frozen=True)
class FlashGRPO:
    clip_range: float = 0.01
    beta: float = 0.0
    adv_clip_max: float = 5.0
    rectification_mode: str = "scheduler_formula"
    normalize_rectification: bool = True
    objective_version: str = "legacy"

    def to_config(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "name": "flash_grpo",
                "objective_version": self.objective_version,
                "clip_range": self.clip_range,
                "beta": self.beta,
                "adv_clip_max": self.adv_clip_max,
                "rectification": {
                    "enabled": True,
                    "mode": self.rectification_mode,
                    "normalize": self.normalize_rectification,
                },
            }
        }


@dataclass(frozen=True)
class TempFlow:
    clip_range: float = 0.0001
    temporal_scale: float = 2.25
    beta: float = 0.0
    adv_clip_max: float = 5.0
    weighting_mode: str = "std_dev_t"
    objective_version: str = "legacy"

    def to_config(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "name": "tempflow_grpo",
                "objective_version": self.objective_version,
                "clip_range": self.clip_range,
                "beta": self.beta,
                "adv_clip_max": self.adv_clip_max,
                "credit_assignment": "branch_timestep",
                "noise_weighting": {
                    "enabled": True,
                    "mode": self.weighting_mode,
                    "scale": self.temporal_scale,
                },
            }
        }


@dataclass(frozen=True)
class Train:
    steps: int = 1
    lr: float = 1e-4
    save_every: int = 1
    max_grad_norm: float | None = None
    precision: str = "fp32"
    update_microbatch_size: int | None = None

    def to_config(self) -> dict[str, Any]:
        return {
            "train": {
                "max_steps": self.steps,
                "learning_rate": self.lr,
                "save_every": self.save_every,
                "max_grad_norm": self.max_grad_norm,
                "precision": self.precision,
                "update_microbatch_size": self.update_microbatch_size,
            }
        }


@dataclass(frozen=True)
class RewardExecution:
    """Runtime policy for inline or bounded concurrent reward scoring."""

    mode: str = "sync"
    max_workers: int = 4
    microbatch_size: int = 1
    timeout_s: float = 30.0
    max_retries: int = 0
    submit_timeout_s: float = 30.0
    max_in_flight: int | None = None
    require_hard_timeout: bool = False

    def to_config(self) -> dict[str, Any]:
        return {
            "runner": {
                "reward_executor": {
                    "mode": self.mode,
                    "max_workers": self.max_workers,
                    "microbatch_size": self.microbatch_size,
                    "timeout_s": self.timeout_s,
                    "max_retries": self.max_retries,
                    "submit_timeout_s": self.submit_timeout_s,
                    "max_in_flight": self.max_in_flight,
                    "require_hard_timeout": self.require_hard_timeout,
                }
            }
        }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    output_dir: Path
    completed_steps: int
    metrics_path: Path
    manifest_path: Path
    latest_checkpoint: Path | None
    evaluations: tuple[Any, ...] = ()
    evaluation_paths: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_paths", MappingProxyType(dict(self.evaluation_paths))
        )

    def iter_metrics(self) -> Iterable[dict[str, Any]]:
        """Yield metric rows one at a time without retaining the full log."""

        with self.metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def load_manifest(self):
        """Load manifest metadata; referenced media remains on disk."""

        from visual_rl.artifacts.manifest import SampleManifest

        return SampleManifest.load(self.manifest_path)

    def load_evaluation(self, name: str):
        """Load one persisted evaluation result without touching its media paths."""

        from visual_rl.evaluation import EvaluationResult

        try:
            path = self.evaluation_paths[name]
        except KeyError as exc:
            raise KeyError(f"No evaluation named {name!r}") from exc
        return EvaluationResult.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


@dataclass(frozen=True, init=False)
class Experiment:
    model: Any
    rollout: Any
    reward: Any
    advantage: Any
    objective: Any
    train: Train
    reward_execution: RewardExecution
    run_name: str
    seed: int
    output_dir: str | os.PathLike[str]
    use_lora: bool
    show_progress: bool | None
    strict_rollout_validation: bool | None
    callbacks: tuple[Any, ...]
    evaluator: Any | None
    evaluation_prompts: tuple[str, ...] | None
    _base_dir: Path = field(repr=False)

    def __init__(
        self,
        *,
        model: Any,
        rollout: Any,
        reward: Any,
        advantage: Any,
        objective: Any,
        train: Train,
        reward_execution: RewardExecution | None = None,
        run_name: str = "experiment",
        seed: int = 42,
        output_dir: str | os.PathLike[str] = "runs/default",
        use_lora: bool = True,
        show_progress: bool | None = None,
        strict_rollout_validation: bool | None = None,
        callbacks: Iterable[Any] = (),
        evaluator: Any | None = None,
        evaluation_prompts: Iterable[str] | None = None,
    ) -> None:
        resolved_reward_execution = (
            RewardExecution() if reward_execution is None else reward_execution
        )
        components = {
            "model": model,
            "rollout": rollout,
            "reward": reward,
            "advantage": advantage,
            "objective": objective,
            "train": train,
            "reward_execution": resolved_reward_execution,
        }
        for name, component in components.items():
            if not callable(getattr(component, "to_config", None)):
                raise TypeError(f"{name} must be a VisualRL config descriptor")
        if not run_name:
            raise ValueError("run_name must be non-empty")
        callback_tuple = tuple(callbacks)
        from visual_rl.callbacks import RunCallback
        from visual_rl.evaluation import Evaluator

        if any(not isinstance(callback, RunCallback) for callback in callback_tuple):
            raise TypeError("callbacks must contain RunCallback instances")
        if evaluator is not None and not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must be an Evaluator instance or None")
        if evaluation_prompts is not None:
            evaluation_prompts = tuple(evaluation_prompts)
            if not evaluation_prompts or any(
                not isinstance(prompt, str) for prompt in evaluation_prompts
            ):
                raise ValueError("evaluation_prompts must contain at least one string")

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "rollout", rollout)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "advantage", advantage)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "train", train)
        object.__setattr__(
            self,
            "reward_execution",
            resolved_reward_execution,
        )
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "use_lora", use_lora)
        object.__setattr__(self, "show_progress", show_progress)
        object.__setattr__(self, "strict_rollout_validation", strict_rollout_validation)
        object.__setattr__(self, "callbacks", callback_tuple)
        object.__setattr__(self, "evaluator", evaluator)
        object.__setattr__(self, "evaluation_prompts", evaluation_prompts)
        object.__setattr__(self, "_base_dir", Path(os.getcwd()))

    def resolve(self) -> VisualRLConfig:
        """Resolve descriptors through the same typed path used by YAML and CLI."""

        return self._resolve_config()

    def to_config(self) -> dict[str, Any]:
        return config_to_dict(self.resolve())

    def validate(self, trusted_components: bool = False) -> PreflightReport:
        """Validate the experiment without runtime side effects by default."""

        config = self.resolve()
        report = static_preflight(config)
        if trusted_components:
            from visual_rl.preflight import trusted_component_load

            report = trusted_component_load(config, report)
        return report

    def run(
        self,
        prompts: Iterable[str],
        *,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> RunResult:
        prompt_list = list(prompts) if not isinstance(prompts, str) else [prompts]
        if not prompt_list or any(
            not isinstance(prompt, str) for prompt in prompt_list
        ):
            raise ValueError("prompts must contain at least one string")

        config = self._resolve_config(prompt_list, resume_from=resume_from)
        report = static_preflight(config)

        from visual_rl.preflight import trusted_component_load

        trusted_component_load(config, report)

        from visual_rl.runner import ExperimentRunner, prepare_resume_source

        prepare_resume_source(config.paths.resume_from)

        if self.callbacks or self.evaluator is not None:
            runner = ExperimentRunner(
                config,
                callbacks=self.callbacks,
                evaluator=self.evaluator,
            )
        else:
            runner = ExperimentRunner(config)
        runner.run()
        output_dir = Path(runner.output_dir)
        latest_checkpoint = _latest_checkpoint(output_dir)
        artifacts = runner.artifacts
        return RunResult(
            run_id=artifacts.run_id,
            output_dir=output_dir,
            completed_steps=int(runner.global_step),
            metrics_path=Path(artifacts.metric_path),
            manifest_path=Path(artifacts.manifest_path),
            latest_checkpoint=latest_checkpoint,
            evaluations=runner.evaluation_results,
            evaluation_paths=runner.evaluation_paths,
        )

    def _resolve_config(
        self,
        prompts: list[str] | None = None,
        *,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> VisualRLConfig:
        return resolve_experiment(
            self._spec(prompts, resume_from=resume_from)
        ).config

    def _spec(
        self,
        prompts: list[str] | None = None,
        *,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> ExperimentSpec:
        base: dict[str, Any] = {
            "run_name": self.run_name,
            "seed": self.seed,
            "use_lora": self.use_lora,
            "paths": {"output_dir": _path(self.output_dir)},
        }
        if self.show_progress is not None:
            base.setdefault("runner", {})["show_progress"] = self.show_progress
        if self.strict_rollout_validation is not None:
            base.setdefault("runner", {})["strict_rollout_validation"] = (
                self.strict_rollout_validation
            )

        fragments: list[tuple[str, Mapping[str, Any]]] = [
            ("experiment", base),
            ("model", self.model.to_config()),
            ("rollout", self.rollout.to_config()),
            ("reward", self.reward.to_config()),
            ("advantage", self.advantage.to_config()),
            ("objective", self.objective.to_config()),
            ("train", self.train.to_config()),
            ("reward_execution", self.reward_execution.to_config()),
        ]
        if prompts is not None:
            fragments.append(("prompts", {"dataset": {"prompts": prompts}}))
        if resume_from is not None:
            fragments.append(
                ("resume", {"paths": {"resume_from": _path(resume_from)}})
            )
        if self.evaluation_prompts is not None:
            fragments.append(
                (
                    "evaluation_prompts",
                    {"evaluation": {"prompts": list(self.evaluation_prompts)}},
                )
            )

        documents = tuple(
            self._document(name, fragment) for name, fragment in fragments
        )
        return ExperimentSpec(
            explicit_documents=documents,
            context_dir=self._base_dir,
        )

    def _document(self, name: str, fragment: Mapping[str, Any]) -> ConfigDocument:
        if not isinstance(fragment, Mapping):
            raise TypeError(f"{name}.to_config() must return a mapping")
        return ConfigDocument(
            fragment,
            SourceRef(
                kind="explicit",
                name=f"python:{name}",
                base_dir=self._base_dir,
            ),
        )


def _latest_checkpoint(output_dir: Path) -> Path | None:
    latest_path = output_dir / "latest.json"
    if not latest_path.is_file():
        return None
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = payload.get("checkpoint")
    if checkpoint is None:
        checkpoint = f"checkpoint_{int(payload['step']):06d}"
    return output_dir / str(checkpoint)


@dataclass(frozen=True)
class _ModelsNamespace:
    MockWan: type[MockWan] = MockWan
    TinyDiffusion: type[TinyDiffusion] = TinyDiffusion
    Wan: type[Wan] = Wan
    SD3: type[SD3] = SD3


@dataclass(frozen=True)
class _RolloutsNamespace:
    FullTrajectory: type[FullTrajectory] = FullTrajectory
    Flash: type[Flash] = Flash
    Branching: type[Branching] = Branching


@dataclass(frozen=True)
class _RewardsNamespace:
    Mock: type[MockReward] = MockReward
    PromptColor: type[PromptColor] = PromptColor
    WorldR1: type[WorldR1] = WorldR1
    External: type[External] = External


@dataclass(frozen=True)
class _AdvantagesNamespace:
    GroupNormalize: type[GroupNormalize] = GroupNormalize


@dataclass(frozen=True)
class _ObjectivesNamespace:
    GRPO: type[GRPO] = GRPO
    FlashGRPO: type[FlashGRPO] = FlashGRPO
    TempFlow: type[TempFlow] = TempFlow


models = _ModelsNamespace()
rollouts = _RolloutsNamespace()
rewards = _RewardsNamespace()
advantages = _AdvantagesNamespace()
objectives = _ObjectivesNamespace()


__all__ = [
    "Experiment",
    "External",
    "RunResult",
    "Train",
    "advantages",
    "models",
    "objectives",
    "rewards",
    "rollouts",
]
