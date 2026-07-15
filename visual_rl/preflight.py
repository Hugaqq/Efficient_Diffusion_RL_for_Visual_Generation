"""Side-effect-free static checks and explicit trusted built-in verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256, load_json
from visual_rl.configs.schema import (
    VisualRLConfig,
    config_from_dict,
    config_to_dict,
    external_provider_metadata,
)


class PreflightError(ValueError):
    """Base class for expected preflight failures."""


class StaticPreflightError(PreflightError):
    """The resolved configuration failed static validation."""


class TrustedComponentError(PreflightError):
    """A trusted component failed import or contract verification."""


class ResumePreflightError(PreflightError):
    """A requested resume path is absent or structurally unsupported."""


@dataclass(frozen=True)
class ComponentDescriptor:
    kind: str
    name: str
    target: str
    source: str
    interface: str
    version: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentCheck:
    kind: str
    name: str
    target: str
    version: str
    source_sha256: str | None
    dependencies: tuple[str, ...]
    trust_boundary: str | None = None


@dataclass(frozen=True)
class PreflightReport:
    components: tuple[ComponentCheck, ...]
    trusted: bool = False
    resolved_config_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trusted": self.trusted,
            "resolved_config_sha256": self.resolved_config_sha256,
            "components": [asdict(component) for component in self.components],
        }


_CATALOG = (
    ComponentDescriptor(
        "model",
        "mock_wan",
        "visual_rl.model_adapters.mock:MockWanAdapter",
        "model_adapters/mock.py",
        "model",
        "v1",
        ("torch",),
    ),
    ComponentDescriptor(
        "model",
        "tiny_diffusion",
        "visual_rl.model_adapters.tiny_diffusion:TinyDiffusionAdapter",
        "model_adapters/tiny_diffusion.py",
        "model",
        "v1",
        ("torch",),
    ),
    ComponentDescriptor(
        "model",
        "sd3_tempflow",
        "visual_rl.model_adapters.sd3:SD3TempFlowAdapter",
        "model_adapters/sd3.py",
        "model",
        "v1",
        ("torch", "diffusers"),
    ),
    ComponentDescriptor(
        "model",
        "tempflow_sd3_legacy",
        "visual_rl.model_adapters.sd3:SD3TempFlowAdapter",
        "model_adapters/sd3.py",
        "model",
        "v1",
        ("torch", "diffusers"),
    ),
    ComponentDescriptor(
        "model",
        "world_r1_wan_legacy",
        "visual_rl.model_adapters.wan:WorldR1WanLegacyAdapter",
        "model_adapters/wan.py",
        "model",
        "v1",
        ("torch", "diffusers", "numpy", "PIL"),
    ),
    ComponentDescriptor(
        "algorithm",
        "grpo",
        "visual_rl.optimizers.grpo:GRPOAlgorithm",
        "optimizers/grpo.py",
        "algorithm",
        "v1",
        ("torch",),
    ),
    ComponentDescriptor(
        "algorithm",
        "flash_grpo",
        "visual_rl.optimizers.flash_grpo:FlashGRPOAlgorithm",
        "optimizers/flash_grpo.py",
        "algorithm",
        "v1",
        ("torch",),
    ),
    ComponentDescriptor(
        "algorithm",
        "tempflow_grpo",
        "visual_rl.optimizers.tempflow_grpo:TempFlowGRPOAlgorithm",
        "optimizers/tempflow_grpo.py",
        "algorithm",
        "v1",
        ("torch",),
    ),
    ComponentDescriptor(
        "reward",
        "mock",
        "visual_rl.feedback.clients:MockRewardClient",
        "feedback/clients.py",
        "reward",
        "v2",
        ("numpy",),
    ),
    ComponentDescriptor(
        "reward",
        "remote_pickle",
        "visual_rl.feedback.clients:RemotePickleRewardClient",
        "feedback/clients.py",
        "reward",
        "v1",
        ("numpy",),
    ),
    ComponentDescriptor(
        "reward",
        "prompt_color",
        "visual_rl.feedback.image_rewards:PromptColorRewardClient",
        "feedback/image_rewards.py",
        "reward",
        "v1",
        ("numpy",),
    ),
    ComponentDescriptor(
        "reward",
        "prompt_color_margin",
        "visual_rl.feedback.image_rewards:PromptColorMarginRewardClient",
        "feedback/image_rewards.py",
        "reward",
        "v1",
        ("numpy",),
    ),
    ComponentDescriptor(
        "reward",
        "prompt_color_guarded",
        "visual_rl.feedback.image_rewards:PromptColorGuardedRewardClient",
        "feedback/image_rewards.py",
        "reward",
        "v1",
        ("numpy",),
    ),
    ComponentDescriptor(
        "reward",
        "reward_3d",
        "visual_rl.feedback.world_r1_rewards:WorldR1Reward3DClient",
        "feedback/world_r1_rewards.py",
        "reward",
        "v1",
        ("numpy", "PIL", "requests"),
    ),
    ComponentDescriptor(
        "reward",
        "reward_general",
        "visual_rl.feedback.world_r1_rewards:WorldR1RewardGeneralClient",
        "feedback/world_r1_rewards.py",
        "reward",
        "v1",
        ("numpy", "PIL", "requests"),
    ),
    ComponentDescriptor(
        "provider",
        "reward_router",
        "visual_rl.feedback.provider:RewardRouterFeedbackProvider",
        "feedback/provider.py",
        "provider",
        "v1",
        ("numpy",),
    ),
    ComponentDescriptor(
        "optimizer",
        "algorithm",
        "visual_rl.optimizers.factory:_build_algorithm_optimizer",
        "optimizers/factory.py",
        "optimizer",
        "v1",
        ("torch",),
    ),
    ComponentDescriptor(
        "rollout",
        "full_trajectory",
        "visual_rl.rollout.full_trajectory:FullTrajectoryRollout",
        "rollout/full_trajectory.py",
        "rollout",
        "v1",
    ),
    ComponentDescriptor(
        "rollout",
        "branching",
        "visual_rl.rollout.branching:BranchingRollout",
        "rollout/branching.py",
        "rollout",
        "v1",
    ),
    ComponentDescriptor(
        "rollout",
        "single_step",
        "visual_rl.rollout.single_step:SingleStepRollout",
        "rollout/single_step.py",
        "rollout",
        "v1",
    ),
    ComponentDescriptor(
        "rollout",
        "flash_single_step",
        "visual_rl.rollout.single_step:SingleStepRollout",
        "rollout/single_step.py",
        "rollout",
        "v1",
    ),
)
_CATALOG_BY_KEY = {(item.kind, item.name): item for item in _CATALOG}
_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_CHECKPOINT_NAME_PATTERN = re.compile(r"^checkpoint_(\d{6})$")
_WAN_ADAPTER_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
    re.ASCII,
)
_TRUSTED_LOCAL_CODE = "trusted_local_code"
_CONTROL_FIELDS = frozenset({"target", "version", "dependencies"})
_PATH_FIELDS = (
    "model.model_path",
    "dataset.path",
    "evaluation.path",
    "train.lora_path",
    "rewards.cache_dir",
    "runner.rollout_cache_dir",
    "paths.output_dir",
    "paths.pretrained_model",
    "paths.resume_from",
    "model.extra.repo_root",
    "model.extra.world_r1_root",
    "model.extra.flash_grpo_root",
    "model.extra.model_path",
    "model.extra.lora_path",
)


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lookup(values: dict[str, Any], dotted: str) -> Any:
    current: Any = values
    for segment in dotted.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _requested_components(
    config: VisualRLConfig,
) -> list[tuple[str, str, dict[str, Any]]]:
    requested = [
        ("model", config.model.name, config.model.extra),
        ("rollout", config.sample.name, config.rollout),
        ("algorithm", config.algorithm.name, config.algorithm.params),
        ("provider", config.rewards.provider, config.rewards.provider_params),
        ("optimizer", config.optimizer.name, config.optimizer.params),
    ]
    for key, client in config.rewards.clients.items():
        requested.append(("reward", str(client.get("name", key)), client))
    return requested


def _external_dependencies(
    kind: str, name: str, declaration: dict[str, Any], errors: list[str]
) -> tuple[str, ...]:
    raw = declaration.get("dependencies", ())
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in raw
    ):
        errors.append(f"{kind} {name!r} has an invalid dependency declaration")
        return ()
    return tuple(raw)


def _external_provider_metadata(
    name: str,
    declaration: dict[str, Any],
    weights: dict[str, float],
    errors: list[str],
) -> tuple[str, str, str | None, tuple[str, ...]]:
    try:
        metadata = external_provider_metadata(
            name,
            declaration,
            weights,
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return "", "", None, ()
    return (
        metadata.target,
        metadata.version,
        metadata.source_sha256,
        metadata.dependencies,
    )


def _resolve_target(target: str) -> object:
    module_name, attribute_path = target.split(":", 1)
    component: object = importlib.import_module(module_name)
    for attribute_name in attribute_path.split("."):
        component = getattr(component, attribute_name)
    return component


def _component_target(component: object) -> str:
    return f"{component.__module__}:{component.__qualname__}"


def _resolved_config_sha256(values: dict[str, Any]) -> str:
    try:
        payload = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"resolved config must contain only finite JSON-compatible values: {exc}"
        ) from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_selected_wan_lora(config: VisualRLConfig, errors: list[str]) -> bool:
    """Validate Wan's optional PEFT declaration without importing PEFT."""

    if config.model.name != "world_r1_wan_legacy":
        return False
    lora_selected = bool(config.use_lora)
    extra = config.model.extra
    backend = extra.get("wan_backend", "world_r1")
    if backend not in {"world_r1", "flash"}:
        errors.append("Wan wan_backend must be one of: flash, world_r1")
    root_key = "flash_grpo_root" if backend == "flash" else "world_r1_root"
    reference_root = extra.get(root_key)
    if reference_root is not None:
        if not isinstance(reference_root, str) or not reference_root.strip():
            errors.append(f"model.extra.{root_key} must be a non-empty string path")
        elif _URL_PATTERN.match(reference_root) or not os.path.isabs(reference_root):
            errors.append(f"model.extra.{root_key} must be a local absolute path")
    guidance_scale = config.rollout.get(
        "guidance_scale",
        config.sample.guidance_scale,
    )
    if isinstance(guidance_scale, bool) or not isinstance(
        guidance_scale, (int, float)
    ):
        errors.append("Wan guidance_scale must resolve to a number")
        guidance_scale_valid = False
    else:
        guidance_scale_valid = True
    train_cfg = config.rollout.get("train_cfg", extra.get("train_cfg", True))
    if not isinstance(train_cfg, bool):
        errors.append("Wan train_cfg must resolve to a bool")
    elif guidance_scale_valid and train_cfg != (float(guidance_scale) > 1.0):
        errors.append("Wan train_cfg must equal (guidance_scale > 1.0)")
    if "num_videos_per_prompt" in config.rollout:
        num_videos = config.rollout["num_videos_per_prompt"]
    elif "num_videos_per_prompt" in extra:
        num_videos = extra["num_videos_per_prompt"]
    elif "num_video_per_prompt" in config.rollout:
        num_videos = config.rollout["num_video_per_prompt"]
    else:
        num_videos = extra.get("num_video_per_prompt", 1)
    if isinstance(num_videos, bool) or not isinstance(num_videos, int):
        errors.append("Wan num_videos_per_prompt must resolve to an integer")
    elif num_videos != 1:
        errors.append("Wan v1 requires num_videos_per_prompt=1")
    use_camera = config.rollout.get(
        "use_camera_trajectory",
        extra.get("use_camera_trajectory", False),
    )
    if not isinstance(use_camera, bool):
        errors.append("Wan use_camera_trajectory must resolve to a bool")
    elif use_camera:
        if backend != "world_r1":
            errors.append("Wan camera trajectories require wan_backend=world_r1")
    objective_version = config.algorithm.objective_version
    if config.algorithm.name == "flash_grpo" and objective_version == "reference_v1":
        if backend != "flash":
            errors.append("Flash-GRPO reference_v1 requires wan_backend=flash")
        if config.algorithm.beta != 0:
            errors.append("Flash-GRPO reference_v1 requires beta=0")
    for name, default in (("lora_rank", 32), ("lora_alpha", 64)):
        value = extra.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"Wan {name} must be a positive integer when use_lora=True")
    targets = extra.get("lora_target_modules")
    if targets is not None:
        if not isinstance(targets, (list, tuple)) or not targets:
            errors.append("Wan lora_target_modules must be a non-empty list when use_lora=True")
        elif any(not isinstance(item, str) or not item.strip() for item in targets):
            errors.append("Wan lora_target_modules must contain non-empty strings")
        elif len({item.strip() for item in targets}) != len(targets):
            errors.append("Wan lora_target_modules must contain unique strings")
    adapter_name = extra.get("adapter_name", "default")
    if (
        not isinstance(adapter_name, str)
        or adapter_name in {".", ".."}
        or not _WAN_ADAPTER_NAME_PATTERN.fullmatch(adapter_name)
    ):
        errors.append(
            "Wan adapter_name must be a safe ASCII identifier of 1-64 characters "
            "when use_lora=True"
        )
    lora_path = (
        config.train.lora_path
        if config.train.lora_path is not None
        else extra.get("lora_path")
    )
    if lora_path is not None:
        if not isinstance(lora_path, str) or not lora_path.strip():
            errors.append("Wan effective lora_path must be a non-empty string or None")
        elif _URL_PATTERN.match(lora_path) or not os.path.isabs(lora_path):
            errors.append("Wan effective lora_path must be a local absolute path")
    return lora_selected


def _validate_selected_tempflow_contract(
    config: VisualRLConfig,
    errors: list[str],
) -> None:
    """Mirror strict TempFlow runtime contracts without importing runtime code."""

    if config.algorithm.name != "tempflow_grpo":
        return

    objective_version = config.algorithm.objective_version
    supported_versions = {"legacy", "policy_identity_v1", "reference_v1"}
    if objective_version not in supported_versions:
        errors.append(
            "TempFlow objective_version must be one of: legacy, "
            "policy_identity_v1, reference_v1"
        )
        return

    extra = config.model.extra
    reference_declared = "tempflow_reference_mode" in extra
    reference_mode = extra.get("tempflow_reference_mode", False)
    reference_mode_valid = isinstance(reference_mode, bool)
    if not reference_mode_valid:
        errors.append("model.extra.tempflow_reference_mode must be a bool")

    recompute_declared = "recompute_transformer_training" in extra
    recompute_training = extra.get("recompute_transformer_training")
    recompute_valid = isinstance(recompute_training, bool)
    if recompute_declared and not recompute_valid:
        errors.append(
            "model.extra.recompute_transformer_training must be a bool when provided"
        )

    if objective_version == "legacy":
        if reference_mode_valid and reference_mode:
            errors.append(
                "TempFlow reference mode requires explicit "
                "algorithm.objective_version='reference_v1'"
            )
        if (
            recompute_declared
            and recompute_valid
            and reference_mode_valid
            and recompute_training is not reference_mode
        ):
            errors.append(
                "model.extra.recompute_transformer_training must match "
                "model.extra.tempflow_reference_mode"
            )
        return

    expected_reference_mode = objective_version == "reference_v1"
    if expected_reference_mode:
        if not reference_declared or reference_mode is not True:
            errors.append(
                "TempFlow reference_v1 requires explicit "
                "model.extra.tempflow_reference_mode=true"
            )
    elif reference_mode_valid and reference_mode:
        errors.append(
            "TempFlow policy_identity_v1 requires "
            "model.extra.tempflow_reference_mode=false"
        )

    if (
        recompute_declared
        and recompute_valid
        and recompute_training is not expected_reference_mode
    ):
        errors.append(
            f"TempFlow {objective_version} requires derived recompute contract "
            "model.extra.recompute_transformer_training="
            f"{str(expected_reference_mode).lower()}"
        )

    if config.algorithm.beta != 0:
        errors.append(
            f"TempFlow {objective_version} requires beta=0 until differentiable "
            "current/reference mean KL is implemented"
        )

    if config.algorithm.advantage_dtype != "float64":
        errors.append(
            f"TempFlow {objective_version} requires advantage_dtype='float64'"
        )
    if config.algorithm.params.get("preserve_advantage_dtype", False) is not True:
        errors.append(
            f"TempFlow {objective_version} requires "
            "algorithm.params.preserve_advantage_dtype=true"
        )

    weighting = config.algorithm.noise_weighting
    if (
        weighting.get("enabled", True) is False
        or weighting.get("mode") != "reference_std_dev_t"
    ):
        errors.append(
            f"TempFlow {objective_version} requires enabled "
            "reference_std_dev_t noise weighting"
        )
    scale = weighting.get("scale", 2.25)
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        errors.append(
            "TempFlow reference_std_dev_t scale must be finite and positive"
        )

    noise_level = config.rollout.get("noise_level", config.sample.noise_level)
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, (int, float))
        or not math.isfinite(noise_level)
        or noise_level <= 0
    ):
        errors.append("TempFlow noise_level must be a finite positive number")
    elif expected_reference_mode and float(noise_level) != 0.7:
        errors.append(
            "TempFlow reference_v1 is pinned to the frozen SD3 kernel "
            "noise_level=0.7"
        )


def static_preflight(config: VisualRLConfig) -> PreflightReport:
    """Validate resolved structure and component metadata without imports."""

    if not isinstance(config, VisualRLConfig):
        raise StaticPreflightError("static_preflight requires VisualRLConfig")
    values = config_to_dict(config)
    try:
        config_from_dict(values)
        resolved_config_sha256 = _resolved_config_sha256(values)
    except (TypeError, ValueError) as exc:
        raise StaticPreflightError(str(exc)) from exc
    errors: list[str] = []
    wan_lora_selected = _validate_selected_wan_lora(config, errors)
    _validate_selected_tempflow_contract(config, errors)
    if config.evaluation.path and config.evaluation.prompts:
        errors.append("evaluation config cannot provide both prompts and path")
    if config.evaluation.max_prompts is not None and config.evaluation.max_prompts < 1:
        errors.append("evaluation.max_prompts must be positive when provided")
    for dotted in _PATH_FIELDS:
        if dotted == "model.extra.lora_path" and config.train.lora_path is not None:
            continue
        if config.model.name == "world_r1_wan_legacy":
            backend = config.model.extra.get("wan_backend", "world_r1")
            if dotted == "model.extra.world_r1_root" and backend == "flash":
                continue
            if dotted == "model.extra.flash_grpo_root" and backend != "flash":
                continue
        value = _lookup(values, dotted)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            errors.append(f"{dotted} must resolve to a string path")
        elif not os.path.isabs(value) and not _URL_PATTERN.match(value):
            errors.append(f"{dotted} is not an absolute resolved path: {value!r}")

    checks: list[ComponentCheck] = []
    seen: set[tuple[str, str]] = set()
    for kind, name, declaration in _requested_components(config):
        try:
            json.dumps(declaration, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            errors.append(f"{kind} {name!r} metadata must be JSON serializable: {exc}")
        declared_target = declaration.get("target")
        declared_version = declaration.get("version")
        key = (kind, name)
        descriptor = _CATALOG_BY_KEY.get(key)
        if descriptor is None:
            if kind == "provider":
                target, version, source_sha256, dependencies = (
                    _external_provider_metadata(
                        name,
                        declaration,
                        config.rewards.weights,
                        errors,
                    )
                )
                if key in seen:
                    continue
                seen.add(key)
                checks.append(
                    ComponentCheck(
                        kind=kind,
                        name=name,
                        target=target,
                        version=version,
                        source_sha256=source_sha256,
                        dependencies=dependencies,
                        trust_boundary=_TRUSTED_LOCAL_CODE,
                    )
                )
                continue
            if declared_target in (None, ""):
                errors.append(
                    f"External {kind} component {name!r} requires an explicit target"
                )
                continue
            if not isinstance(declared_target, str) or not _TARGET_PATTERN.fullmatch(
                declared_target
            ):
                errors.append(
                    f"External {kind} component {name!r} has invalid target "
                    f"{declared_target!r}; expected module:attribute"
                )
            if not isinstance(declared_version, str) or not declared_version.strip():
                errors.append(
                    f"External {kind} component {name!r} requires a non-empty version"
                )
            dependencies = _external_dependencies(kind, name, declaration, errors)
            if key in seen:
                continue
            seen.add(key)
            checks.append(
                ComponentCheck(
                    kind=kind,
                    name=name,
                    target=str(declared_target),
                    version=(
                        declared_version.strip()
                        if isinstance(declared_version, str)
                        else ""
                    ),
                    source_sha256=None,
                    dependencies=dependencies,
                )
            )
            continue
        if declared_target is not None and declared_target != descriptor.target:
            errors.append(
                f"{kind} {name!r} target {declared_target!r} does not match "
                f"trusted catalog target {descriptor.target!r}"
            )
        if declared_version is not None and declared_version != descriptor.version:
            errors.append(
                f"{kind} {name!r} version {declared_version!r} does not match "
                f"trusted catalog version {descriptor.version!r}"
            )
        if key in seen:
            continue
        seen.add(key)
        if any(
            not dependency or not isinstance(dependency, str)
            for dependency in descriptor.dependencies
        ):
            errors.append(f"{kind} {name!r} has an invalid dependency declaration")
        source = _package_root() / descriptor.source
        if not source.is_file():
            errors.append(f"Catalog source is missing for {kind} {name!r}: {source}")
            continue
        dependencies = descriptor.dependencies
        if kind == "model" and name == "world_r1_wan_legacy" and wan_lora_selected:
            dependencies = (*dependencies, "peft")
        checks.append(
            ComponentCheck(
                kind=kind,
                name=name,
                target=descriptor.target,
                version=descriptor.version,
                source_sha256=_sha256(source),
                dependencies=dependencies,
            )
        )
    if errors:
        raise StaticPreflightError("; ".join(errors))
    return PreflightReport(
        tuple(checks),
        resolved_config_sha256=resolved_config_sha256,
    )


def _validated_checkpoint_dir(
    run_root: Path,
    checkpoint_path: Path,
    *,
    expected_step: int,
) -> Path:
    expected_name = f"checkpoint_{expected_step:06d}"
    if checkpoint_path.name != expected_name:
        raise ResumePreflightError(
            f"Checkpoint name must match manifest step {expected_step}: {expected_name}"
        )
    try:
        resolved_root = run_root.resolve(strict=True)
        resolved_checkpoint = checkpoint_path.resolve(strict=True)
    except OSError as exc:
        raise ResumePreflightError(
            f"Resume checkpoint does not exist: {checkpoint_path}"
        ) from exc
    if (
        not resolved_checkpoint.is_dir()
        or resolved_checkpoint.parent != resolved_root
        or resolved_checkpoint.name != expected_name
    ):
        raise ResumePreflightError(
            f"Resume checkpoint escapes run root {resolved_root}: {checkpoint_path}"
        )
    state_path = resolved_checkpoint / "training_state.pt"
    try:
        resolved_state = state_path.resolve(strict=True)
    except OSError as exc:
        raise ResumePreflightError(
            f"Checkpoint directory has no training_state.pt: {resolved_checkpoint}"
        ) from exc
    if not resolved_state.is_file() or resolved_state.parent != resolved_checkpoint:
        raise ResumePreflightError(
            f"Checkpoint training_state.pt escapes checkpoint directory: {state_path}"
        )
    return resolved_checkpoint


def _commit_marker_payload(marker_path: Path) -> dict[str, Any]:
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ResumePreflightError(
            f"Authoritative commit marker is not a regular file: {marker_path}"
        )
    match = re.fullmatch(r"commit_(\d{6})\.json", marker_path.name)
    if match is None:
        raise ResumePreflightError(
            f"Authoritative commit marker has an invalid name: {marker_path}"
    )
    try:
        payload = load_json(marker_path)
        step = payload["completed_steps"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("completed_steps must be a positive integer")
        if (
            payload.get("schema_version") != "1"
            or payload.get("kind") != "artifact_commit"
            or payload.get("commit_id") != step
            or int(match.group(1)) != step
        ):
            raise ValueError("commit marker identity mismatch")
        checkpoint = payload.get("checkpoint")
        expected_name = f"checkpoint_{step:06d}"
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("completed_steps") != step
            or checkpoint.get("path") != expected_name
            or checkpoint.get("final_path") != expected_name
            or not isinstance(checkpoint.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
        ):
            raise ValueError("commit marker checkpoint metadata mismatch")
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        raise ResumePreflightError(
            f"Authoritative commit marker is invalid: {marker_path}: {exc}"
        ) from exc
    return payload


def _commit_marker_step(marker_path: Path) -> int:
    return int(_commit_marker_payload(marker_path)["completed_steps"])


def _validate_committed_checkpoint_tree(
    run_root: Path,
    checkpoint: Path,
    marker: dict[str, Any],
) -> None:
    expected_digest = marker["checkpoint"]["sha256"]
    try:
        actual_digest = checkpoint_tree_sha256(
            checkpoint,
            trusted_root=run_root,
        )
    except RuntimeError as exc:
        raise ResumePreflightError(
            f"Committed checkpoint tree is unsafe: {checkpoint}: {exc}"
        ) from exc
    if actual_digest != expected_digest:
        raise ResumePreflightError(
            f"Committed checkpoint tree SHA256 mismatch: {checkpoint}"
        )


def resume_run_root(path_value: str | os.PathLike[str]) -> Path:
    """Return the physical run root for a supported resume-path shape."""

    path = Path(path_value)
    direct_match = _CHECKPOINT_NAME_PATTERN.fullmatch(path.name)
    if direct_match is not None:
        run_root = path.parent
    elif path.name == "latest.json":
        run_root = path.parent
    elif path.is_dir():
        run_root = path
    else:
        raise ResumePreflightError(f"Unsupported resume path shape: {path}")
    if run_root.is_symlink():
        raise ResumePreflightError(
            f"Resume run root cannot be a symlink: {run_root}"
        )
    try:
        resolved_root = run_root.resolve(strict=True)
    except OSError as exc:
        raise ResumePreflightError(
            f"Resume run root does not exist: {run_root}"
        ) from exc
    if not resolved_root.is_dir():
        raise ResumePreflightError(
            f"Resume run root is not a directory: {run_root}"
        )
    return resolved_root


def has_transactional_artifact_layout(run_root: str | os.PathLike[str]) -> bool:
    """Detect layouts where commit markers, rather than latest.json, are authoritative."""

    root = Path(run_root)
    found = False
    for candidate, label in (
        (root / "commits", "commit"),
        (root / ".staging", "staging"),
    ):
        if candidate.is_symlink():
            raise ResumePreflightError(
                f"Resume {label} directory is not a safe directory: {candidate}"
            )
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ResumePreflightError(
                f"Cannot inspect resume {label} directory {candidate}: {exc}"
            ) from exc
        if not candidate.is_dir():
            raise ResumePreflightError(
                f"Resume {label} directory is not a safe directory: {candidate}"
            )
        found = True
    return found


def _missing_transaction_marker(checkpoint: Path) -> ResumePreflightError:
    return ResumePreflightError(
        "Transactionized artifact layout has no authoritative commit marker for "
        f"checkpoint {checkpoint}; recover the ready transaction before resuming"
    )


def _validate_direct_checkpoint_marker(
    run_root: Path,
    checkpoint: Path,
    *,
    step: int,
) -> None:
    """Bind a direct checkpoint to its marker when the run has a commit log."""

    commits_dir = run_root / "commits"
    if commits_dir.is_symlink():
        raise ResumePreflightError(
            f"Resume commit directory is not a safe directory: {commits_dir}"
        )
    transactionized = has_transactional_artifact_layout(run_root)
    if not commits_dir.exists():
        if transactionized:
            raise _missing_transaction_marker(checkpoint)
        return
    if not commits_dir.is_dir():
        raise ResumePreflightError(
            f"Resume commit directory is not a safe directory: {commits_dir}"
        )
    markers = [
        _commit_marker_payload(marker_path)
        for marker_path in sorted(commits_dir.glob("commit_*.json"))
    ]
    if not markers:
        raise _missing_transaction_marker(checkpoint)
    matching = [
        marker for marker in markers if int(marker["completed_steps"]) == step
    ]
    if len(matching) != 1:
        raise ResumePreflightError(
            f"Direct checkpoint has no unique authoritative commit marker: {checkpoint}"
        )
    _validate_committed_checkpoint_tree(run_root, checkpoint, matching[0])


def latest_committed_step(run_root: str | os.PathLike[str]) -> int | None:
    """Return the newest structurally valid marker, even if its checkpoint is gone."""

    commits_dir = Path(run_root) / "commits"
    if commits_dir.is_symlink():
        raise ResumePreflightError(
            f"Resume commit directory is not a safe directory: {commits_dir}"
        )
    if not commits_dir.exists():
        return None
    if not commits_dir.is_dir():
        raise ResumePreflightError(
            f"Resume commit directory is not a safe directory: {commits_dir}"
        )
    steps = [
        _commit_marker_step(marker_path)
        for marker_path in sorted(commits_dir.glob("commit_*.json"))
    ]
    return max(steps, default=None)


def _resolve_committed_checkpoint(run_root: Path) -> tuple[Path, int] | None:
    commits_dir = run_root / "commits"
    if commits_dir.is_symlink():
        raise ResumePreflightError(
            f"Resume commit directory is not a safe directory: {commits_dir}"
        )
    if not commits_dir.exists():
        return None
    if not commits_dir.is_dir():
        raise ResumePreflightError(
            f"Resume commit directory is not a safe directory: {commits_dir}"
        )
    marker_paths = sorted(
        commits_dir.glob("commit_*.json"),
        reverse=True,
    )
    if not marker_paths:
        raise ResumePreflightError(
            "Transactionized artifact layout has no authoritative commit markers: "
            f"{commits_dir}; recover any ready transaction before resuming"
        )
    markers = [
        (marker_path, _commit_marker_payload(marker_path))
        for marker_path in marker_paths
    ]
    for marker_path, marker in markers:
        step = int(marker["completed_steps"])
        expected_name = f"checkpoint_{step:06d}"
        checkpoint_path = run_root / expected_name
        try:
            checkpoint_path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ResumePreflightError(
                f"Cannot inspect committed checkpoint {checkpoint_path}: {exc}"
            ) from exc
        checkpoint = _validated_checkpoint_dir(
            run_root,
            checkpoint_path,
            expected_step=step,
        )
        _validate_committed_checkpoint_tree(run_root, checkpoint, marker)
        return checkpoint, step
    raise ResumePreflightError(
        f"No valid committed checkpoint exists under {commits_dir}"
    )


def resolve_resume_checkpoint(
    path_value: str | os.PathLike[str],
) -> tuple[Path, int]:
    """Resolve a checkpoint while confining manifests to their run root."""

    path = Path(path_value)
    missing_latest_cache = (
        path.name == "latest.json" and not path.exists() and path.parent.is_dir()
    )
    if not path.exists() and not missing_latest_cache:
        raise ResumePreflightError(f"Resume path does not exist: {path}")
    direct_match = _CHECKPOINT_NAME_PATTERN.fullmatch(path.name)
    if path.is_dir() and direct_match:
        step = int(direct_match.group(1))
        checkpoint = _validated_checkpoint_dir(
            path.parent,
            path,
            expected_step=step,
        )
        _validate_direct_checkpoint_marker(
            path.parent,
            checkpoint,
            step=step,
        )
        return checkpoint, step

    if path.name == "latest.json":
        run_root = path.parent
        latest = path
    elif path.is_dir():
        run_root = path
        latest = path / "latest.json"
    else:
        raise ResumePreflightError(f"Unsupported resume path shape: {path}")
    committed = _resolve_committed_checkpoint(run_root)
    if committed is not None:
        return committed
    if has_transactional_artifact_layout(run_root):
        raise ResumePreflightError(
            "Transactionized artifact layout has no authoritative commit markers: "
            f"{run_root}; recover any ready transaction before resuming"
        )
    if latest.name != "latest.json" or not latest.is_file():
        raise ResumePreflightError(f"Unsupported resume path shape: {path}")
    try:
        resolved_root = run_root.resolve(strict=True)
        resolved_latest = latest.resolve(strict=True)
    except OSError as exc:
        raise ResumePreflightError(f"Invalid resume manifest path {latest}: {exc}") from exc
    if resolved_latest.parent != resolved_root:
        raise ResumePreflightError(
            f"Resume manifest escapes run root {resolved_root}: {latest}"
        )
    try:
        payload = load_json(resolved_latest)
        step = payload["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        expected_name = f"checkpoint_{step:06d}"
        checkpoint = payload.get("checkpoint", expected_name)
        if not isinstance(checkpoint, str) or checkpoint != expected_name:
            raise ValueError(
                f"checkpoint must be the relative name {expected_name!r}"
            )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ResumePreflightError(f"Invalid resume manifest {latest}: {exc}") from exc
    checkpoint_path = _validated_checkpoint_dir(
        resolved_root,
        resolved_root / checkpoint,
        expected_step=step,
    )
    return checkpoint_path, step


def validate_resume_path(path_value: str | os.PathLike[str] | None) -> None:
    """Perform the same read-only safe resolution used by the runner."""

    if path_value:
        resolve_resume_checkpoint(path_value)


def _registry_for_kind(kind: str):
    from visual_rl.core.registry import (
        ALGORITHMS,
        FEEDBACK_PROVIDERS,
        MODEL_ADAPTERS,
        OPTIMIZER_PLUGINS,
        REWARD_CLIENTS,
        ROLLOUT_ENGINES,
    )

    return {
        "model": MODEL_ADAPTERS,
        "algorithm": ALGORITHMS,
        "reward": REWARD_CLIENTS,
        "provider": FEEDBACK_PROVIDERS,
        "optimizer": OPTIMIZER_PLUGINS,
        "rollout": ROLLOUT_ENGINES,
    }[kind]


def _verify_interface(descriptor: ComponentDescriptor, component: object) -> None:
    from visual_rl.feedback.base import FeedbackProvider
    from visual_rl.model_adapters.base import ModelAdapter
    from visual_rl.rollout.base import RolloutEngine

    if descriptor.interface == "model":
        if not inspect.isclass(component) or not issubclass(component, ModelAdapter):
            raise TypeError("must subclass ModelAdapter")
        inspect.signature(component).bind({})
        for method_name in (
            "parameters",
            "sample",
            "recompute_log_probs",
            "save_pretrained",
            "load_checkpoint",
        ):
            if not callable(getattr(component, method_name, None)):
                raise TypeError(f"must define callable {method_name}")
    elif descriptor.interface == "rollout":
        if not inspect.isclass(component) or not issubclass(component, RolloutEngine):
            raise TypeError("must subclass RolloutEngine")
        inspect.signature(component).bind({})
        if not callable(getattr(component, "sample", None)):
            raise TypeError("must define callable sample")
    elif descriptor.interface == "provider":
        if not inspect.isclass(component) or not issubclass(
            component, FeedbackProvider
        ):
            raise TypeError("must subclass FeedbackProvider")
        inspect.signature(component).bind(object(), cache_dir=None)
        if not callable(getattr(component, "score", None)):
            raise TypeError("must define score")
        inspect.signature(getattr(component, "score")).bind(object(), object())
    elif descriptor.interface == "algorithm":
        if not callable(getattr(component, "from_config", None)) or not callable(
            getattr(component, "compute_loss", None)
        ):
            raise TypeError("must define from_config and compute_loss")
        inspect.signature(getattr(component, "from_config")).bind({})
        inspect.signature(getattr(component, "compute_loss"))
    elif descriptor.interface == "reward":
        if not inspect.isclass(component) or not callable(
            getattr(component, "score", None)
        ):
            raise TypeError("must be a class defining score")
        inspect.signature(getattr(component, "score")).bind(object(), object(), [], [])
    elif descriptor.interface == "optimizer":
        if not callable(component):
            raise TypeError("must be callable")
        inspect.signature(component).bind(object())


def _verify_selected_signature(
    config: VisualRLConfig,
    check: ComponentCheck,
    component: object,
) -> None:
    if check.kind == "model":
        _verify_selected_model_contract(check, component)
        inspect.signature(component).bind(config_to_dict(config)["model"])
    elif check.kind == "rollout":
        rollout_config = config_to_dict(config)["sample"]
        rollout_config.update(config.rollout)
        inspect.signature(component).bind(rollout_config)
        try:
            inspect.signature(getattr(component, "sample")).bind(
                object(),
                object(),
                [],
                [],
                object(),
            )
        except TypeError as exc:
            raise TypeError(
                f"Selected rollout engine {check.name!r} sample must accept "
                "(self, adapter, prompts, metadata, context); add the C4 "
                "StepContext parameter to the legacy four-argument method"
            ) from exc
    elif check.kind == "provider":
        params = {
            key: value
            for key, value in config.rewards.provider_params.items()
            if key not in _CONTROL_FIELDS
        }
        inspect.signature(component).bind(config.rewards, cache_dir=None, **params)
    elif check.kind == "algorithm":
        inspect.signature(getattr(component, "from_config")).bind(
            config_to_dict(config)["algorithm"]
        )
    elif check.kind == "optimizer":
        inspect.signature(component).bind(config)
    elif check.kind == "reward":
        for key, client in config.rewards.clients.items():
            if str(client.get("name", key)) != check.name:
                continue
            params = dict(client.get("params", {}))
            params.update(
                {
                    name: value
                    for name, value in client.items()
                    if name not in {"name", "params", *_CONTROL_FIELDS}
                }
            )
            inspect.signature(component).bind(**params)


def _verify_selected_model_contract(
    check: ComponentCheck,
    component: object,
) -> None:
    abstract_methods = sorted(getattr(component, "__abstractmethods__", ()))
    if inspect.isabstract(component):
        detail = ", ".join(abstract_methods) or "unknown abstract methods"
        raise TypeError(
            f"Selected model adapter {check.name!r} is abstract ({detail}); "
            "migrate the adapter by implementing a concrete @property train_module"
        )
    train_module = inspect.getattr_static(component, "train_module", None)
    if not isinstance(train_module, property) or getattr(
        train_module.fget,
        "__isabstractmethod__",
        False,
    ):
        raise TypeError(
            f"Selected model adapter {check.name!r} must implement a concrete "
            "@property train_module; expose the nn.Module that owns trainable state"
        )


def _external_descriptor(check: ComponentCheck, source: str) -> ComponentDescriptor:
    return ComponentDescriptor(
        kind=check.kind,
        name=check.name,
        target=check.target,
        source=source,
        interface=check.kind,
        version=check.version,
        dependencies=check.dependencies,
    )


def _source_path(component: object) -> Path:
    source_file = inspect.getsourcefile(component)
    if source_file is None:
        raise TypeError(f"Cannot identify source for {_component_target(component)}")
    return Path(source_file).resolve()


def _verify_external_feedback_component(
    config: VisualRLConfig,
    check: ComponentCheck,
    component: object,
) -> None:
    from visual_rl.feedback.base import FeedbackProvider

    metadata = external_provider_metadata(
        check.name,
        config.rewards.provider_params,
        config.rewards.weights,
    )

    if inspect.isclass(component):
        if not issubclass(component, FeedbackProvider):
            raise TypeError(
                f"External provider target {check.target!r} must be a function "
                "or FeedbackProvider subclass"
            )
        if inspect.isabstract(component):
            raise TypeError(
                f"External FeedbackProvider {check.target!r} must be concrete"
            )
        inspect.signature(component).bind(
            config.rewards,
            cache_dir=None,
            **metadata.params,
        )
        score = getattr(component, "score", None)
        if not callable(score):
            raise TypeError(
                f"External FeedbackProvider {check.target!r} must define score"
            )
        inspect.signature(score).bind(object(), object())
        return
    if not callable(component):
        raise TypeError(f"External provider target {check.target!r} must be callable")
    try:
        inspect.signature(component).bind(object(), **metadata.params)
    except TypeError as exc:
        raise TypeError(
            f"External reward function {check.target!r} must accept (batch, **params)"
        ) from exc


def trusted_component_load(
    config: VisualRLConfig,
    static_report: PreflightReport | None = None,
) -> PreflightReport:
    """Import local built-ins and verify registry, interface, and identity drift."""

    report = static_report if static_report is not None else static_preflight(config)
    try:
        current_config_sha256 = _resolved_config_sha256(config_to_dict(config))
    except (TypeError, ValueError) as exc:
        raise TrustedComponentError(str(exc)) from exc
    if not report.resolved_config_sha256:
        raise TrustedComponentError(
            "Static preflight report is missing resolved_config_sha256"
        )
    if report.resolved_config_sha256 != current_config_sha256:
        raise TrustedComponentError(
            "Static preflight report resolved_config_sha256 does not match "
            "the current resolved config"
        )
    try:
        from visual_rl import __version__
        from visual_rl.builtins import register_builtin_plugins

        if __version__ != "0.6.0":
            raise RuntimeError(f"Unsupported VisualRL built-in version {__version__!r}")
        register_builtin_plugins()

        for descriptor in _CATALOG:
            registry = _registry_for_kind(descriptor.kind)
            if descriptor.name not in registry.keys():
                raise RuntimeError(
                    f"{descriptor.kind} catalog/registry drift: missing "
                    f"{descriptor.name!r}"
                )
            expected_component = _resolve_target(descriptor.target)
            if registry.get(descriptor.name) is not expected_component:
                raise RuntimeError(
                    f"{descriptor.kind} catalog/registry target drift for "
                    f"{descriptor.name!r}"
                )
            actual_target = _component_target(expected_component)
            if actual_target != descriptor.target:
                raise RuntimeError(
                    f"Target drift for {descriptor.kind} {descriptor.name!r}: "
                    f"{actual_target}"
                )
            _verify_interface(descriptor, expected_component)
            expected_source = (_package_root() / descriptor.source).resolve()
            actual_source = _source_path(expected_component)
            if actual_source != expected_source:
                raise RuntimeError(
                    f"Source path drift for {descriptor.kind} "
                    f"{descriptor.name!r}: {actual_source}"
                )
            if _sha256(actual_source) != _sha256(expected_source):
                raise RuntimeError(
                    f"Source identity drift for {descriptor.kind} {descriptor.name!r}"
                )

        trusted_checks: list[ComponentCheck] = []
        for check in report.components:
            for dependency in check.dependencies:
                if importlib.util.find_spec(dependency) is None:
                    raise RuntimeError(
                        f"Missing dependency {dependency!r} for "
                        f"{check.kind} {check.name!r}"
                    )
            descriptor = _CATALOG_BY_KEY.get((check.kind, check.name))
            component = _resolve_target(check.target)
            if check.kind == "provider" and check.trust_boundary == _TRUSTED_LOCAL_CODE:
                actual_target = _component_target(component)
                if actual_target != check.target:
                    raise RuntimeError(
                        f"Target drift for provider {check.name!r}: {actual_target}"
                    )
                actual_source = _source_path(component)
                source_sha256 = _sha256(actual_source)
                if source_sha256 != check.source_sha256:
                    raise RuntimeError(
                        f"Source SHA256 mismatch for trusted local code provider "
                        f"{check.name!r}: actual {source_sha256}, declared "
                        f"{check.source_sha256}"
                    )
                _verify_external_feedback_component(config, check, component)
                trusted_checks.append(replace(check, source_sha256=source_sha256))
                continue
            registry = _registry_for_kind(check.kind)
            try:
                registered = registry.get(check.name)
            except KeyError as exc:
                if check.kind in {"reward", "provider"}:
                    register_name = (
                        "register_reward_client"
                        if check.kind == "reward"
                        else "register_feedback_provider"
                    )
                    raise RuntimeError(
                        f"External {check.kind} {check.name!r} imported but did not "
                        f"register via visual_rl.plugins.{register_name}(). "
                        "Non-Registry C5 reward/provider adapters are not implemented "
                        "in this C3 gate."
                    ) from exc
                raise RuntimeError(
                    f"External {check.kind} {check.name!r} imported but did not "
                    "register through the matching visual_rl.plugins.register_* API"
                ) from exc
            if registered is not component:
                raise RuntimeError(
                    f"Registry target drift for {check.kind} {check.name!r}"
                )
            actual_target = _component_target(component)
            if actual_target != check.target:
                raise RuntimeError(
                    f"Target drift for {check.kind} {check.name!r}: {actual_target}"
                )
            actual_source = _source_path(component)
            selected_descriptor = descriptor or _external_descriptor(
                check, str(actual_source)
            )
            _verify_interface(selected_descriptor, component)
            _verify_selected_signature(config, check, component)
            source_sha256 = _sha256(actual_source)
            if check.source_sha256 is not None and source_sha256 != check.source_sha256:
                raise RuntimeError(
                    f"Source SHA256 drift for {check.kind} {check.name!r}"
                )
            trusted_checks.append(replace(check, source_sha256=source_sha256))
    except TrustedComponentError:
        raise
    except Exception as exc:
        raise TrustedComponentError(str(exc)) from exc
    return PreflightReport(
        tuple(trusted_checks),
        trusted=True,
        resolved_config_sha256=report.resolved_config_sha256,
    )


__all__ = [
    "ComponentCheck",
    "ComponentDescriptor",
    "PreflightError",
    "PreflightReport",
    "ResumePreflightError",
    "StaticPreflightError",
    "TrustedComponentError",
    "static_preflight",
    "trusted_component_load",
    "validate_resume_path",
]
