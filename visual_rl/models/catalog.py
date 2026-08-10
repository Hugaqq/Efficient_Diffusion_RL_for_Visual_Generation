"""Import-safe model declarations and the model-owned catalog fragment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    ComputePrecision,
    DeclaredContract,
    LatentLayout,
    MediaKind,
    ModelContract,
    PredictionType,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
)
from visual_rl.models.scheduler import SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA

__all__ = (
    "MODEL_CATALOG_FRAGMENT",
    "WAN_TEMPORAL_STRIDE",
    "SD3Config",
    "SD3DeclarationProvider",
    "WanConfig",
    "WanDeclarationProvider",
    "finite_positive",
    "model_catalog_fragment",
    "positive_int",
    "strict_values",
    "target_modules",
)

WAN_TEMPORAL_STRIDE = 4

_SD3_DEFAULT_TARGETS = (
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
)
_SD3_CONFIG_KEYS = frozenset(
    {
        "artifact_ref",
        "gradient_checkpointing",
        "guidance_scale",
        "lora_alpha",
        "lora_rank",
        "lora_target_modules",
        "max_sequence_length",
        "resolution",
    }
)
_WAN_DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
_WAN_CONFIG_KEYS = frozenset(
    {
        "artifact_ref",
        "frames",
        "frame_rate_denominator",
        "frame_rate_numerator",
        "gradient_checkpointing",
        "guidance_scale",
        "height",
        "lora_alpha",
        "lora_rank",
        "lora_target_modules",
        "max_sequence_length",
        "vae_tiling",
        "width",
    }
)


@dataclass(frozen=True, slots=True)
class SD3Config:
    """Model behavior and trainable topology; artifacts stay logical here."""

    artifact_ref: str
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: tuple[str, ...] = _SD3_DEFAULT_TARGETS
    gradient_checkpointing: bool = True
    guidance_scale: float = 4.5
    resolution: int = 512
    max_sequence_length: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, str) or not self.artifact_ref:
            raise ValueError("artifact_ref must be a non-empty string")
        object.__setattr__(self, "lora_rank", positive_int("lora_rank", self.lora_rank))
        object.__setattr__(
            self,
            "lora_alpha",
            positive_int("lora_alpha", self.lora_alpha),
        )
        object.__setattr__(
            self,
            "lora_target_modules",
            target_modules(self.lora_target_modules),
        )
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be bool")
        object.__setattr__(
            self,
            "guidance_scale",
            finite_positive("guidance_scale", self.guidance_scale),
        )
        resolution = positive_int("resolution", self.resolution)
        if resolution % 8:
            raise ValueError("resolution must be divisible by the SD3 VAE stride 8")
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(
            self,
            "max_sequence_length",
            positive_int("max_sequence_length", self.max_sequence_length),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> SD3Config:
        del context
        resolved = strict_values(
            values,
            allowed=_SD3_CONFIG_KEYS,
            required=frozenset({"artifact_ref"}),
            label="sd3",
        )
        if "lora_target_modules" in resolved:
            resolved["lora_target_modules"] = target_modules(
                resolved["lora_target_modules"]
            )
        return cls(**resolved)

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="model",
            component_id="sd3",
            model=ModelContract(
                tasks=(TaskKind.T2I,),
                output_media=(MediaKind.IMAGE,),
                latent_layouts=(LatentLayout.BCHW,),
                latent_ranks=(4,),
                axis_semantics=(("batch", "channel", "height", "width"),),
                prediction_types=(PredictionType.FLOW,),
                time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
                training_modes=(TrainingMode.LORA,),
                supported_precisions=(
                    ComputePrecision.FP32,
                    ComputePrecision.FP16,
                    ComputePrecision.BF16,
                ),
                provides_reference_policy=True,
                condition_payload_types=("sd3_prompt_embeddings.v1",),
                spatial_stride=(8, 8),
                scheduler_blueprint_schema=SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA,
                dynamics_binding_family="sd3.flow-sde.v1",
                schedule_coordinate=TimeCoordinate.FRACTIONAL_TIMESTEP,
                accepted_replay_state_schema_ids=("sd3.schedule-replay.v1",),
            ),
        )


@dataclass(frozen=True, slots=True)
class WanConfig:
    """Wan topology and generation geometry, independent of rollout kind."""

    artifact_ref: str
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: tuple[str, ...] = _WAN_DEFAULT_TARGETS
    gradient_checkpointing: bool = True
    guidance_scale: float = 5.0
    height: int = 480
    width: int = 832
    frames: int = 81
    frame_rate_numerator: int = 16
    frame_rate_denominator: int = 1
    max_sequence_length: int = 226
    vae_tiling: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, str) or not self.artifact_ref:
            raise ValueError("artifact_ref must be a non-empty string")
        object.__setattr__(self, "lora_rank", positive_int("lora_rank", self.lora_rank))
        object.__setattr__(
            self,
            "lora_alpha",
            positive_int("lora_alpha", self.lora_alpha),
        )
        object.__setattr__(
            self,
            "lora_target_modules",
            target_modules(self.lora_target_modules),
        )
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be bool")
        object.__setattr__(
            self,
            "guidance_scale",
            finite_positive("guidance_scale", self.guidance_scale),
        )
        for name in ("height", "width"):
            value = positive_int(name, getattr(self, name))
            if value % 8:
                raise ValueError(f"{name} must be divisible by Wan spatial stride 8")
            object.__setattr__(self, name, value)
        frames = positive_int("frames", self.frames)
        if (frames - 1) % WAN_TEMPORAL_STRIDE:
            raise ValueError("frames must satisfy (frames - 1) % 4 == 0")
        object.__setattr__(self, "frames", frames)
        numerator = positive_int(
            "frame_rate_numerator",
            self.frame_rate_numerator,
        )
        denominator = positive_int(
            "frame_rate_denominator",
            self.frame_rate_denominator,
        )
        divisor = math.gcd(numerator, denominator)
        object.__setattr__(self, "frame_rate_numerator", numerator // divisor)
        object.__setattr__(self, "frame_rate_denominator", denominator // divisor)
        object.__setattr__(
            self,
            "max_sequence_length",
            positive_int("max_sequence_length", self.max_sequence_length),
        )
        if type(self.vae_tiling) is not bool:
            raise TypeError("vae_tiling must be bool")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> WanConfig:
        del context
        resolved = strict_values(
            values,
            allowed=_WAN_CONFIG_KEYS,
            required=frozenset({"artifact_ref"}),
            label="wan-t2v",
        )
        if "lora_target_modules" in resolved:
            resolved["lora_target_modules"] = target_modules(
                resolved["lora_target_modules"]
            )
        return cls(**resolved)

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="model",
            component_id="wan-t2v",
            model=ModelContract(
                tasks=(TaskKind.T2V,),
                output_media=(MediaKind.VIDEO,),
                latent_layouts=(LatentLayout.BCTHW,),
                latent_ranks=(5,),
                axis_semantics=(
                    ("batch", "channel", "time", "height", "width"),
                ),
                prediction_types=(PredictionType.FLOW,),
                time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
                training_modes=(TrainingMode.LORA,),
                supported_precisions=(
                    ComputePrecision.FP32,
                    ComputePrecision.FP16,
                    ComputePrecision.BF16,
                ),
                provides_reference_policy=False,
                condition_payload_types=("wan_prompt_embeddings.v1",),
                spatial_stride=(8, 8),
                temporal_stride=WAN_TEMPORAL_STRIDE,
                scheduler_blueprint_schema=SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA,
                dynamics_binding_family="wan.flow-sde.v1",
                schedule_coordinate=TimeCoordinate.FRACTIONAL_TIMESTEP,
                accepted_replay_state_schema_ids=("wan.schedule-replay.v1",),
            ),
        )


class SD3DeclarationProvider:
    """Parse one SD3 declaration without importing the SD3 implementation."""

    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.models.catalog:SD3Config"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = SD3Config.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config,
            declared_contract=config.describe_contract(),
        )


class WanDeclarationProvider:
    """Parse one Wan declaration without importing the Wan implementation."""

    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.models.catalog:WanConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = WanConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config,
            declared_contract=config.describe_contract(),
        )


MODEL_CATALOG_FRAGMENT = CatalogFragment(
    owner="models",
    kind="model",
    descriptors=(
        ComponentDescriptor(
            alias="sd3",
            implementation_class_path=(
                "visual_rl.models.implementations.sd3:SD3Adapter"
            ),
            declaration_provider_path=(
                "visual_rl.models.catalog:SD3DeclarationProvider"
            ),
            optional_dependencies=("diffusers", "peft", "torch", "transformers"),
        ),
        ComponentDescriptor(
            alias="wan-t2v",
            implementation_class_path=(
                "visual_rl.models.implementations.wan:WanT2VAdapter"
            ),
            declaration_provider_path=(
                "visual_rl.models.catalog:WanDeclarationProvider"
            ),
            optional_dependencies=(
                "diffusers",
                "imageio_ffmpeg",
                "peft",
                "torch",
                "transformers",
            ),
        ),
    ),
)


def model_catalog_fragment() -> CatalogFragment:
    """Return the immutable model descriptor contribution."""

    return MODEL_CATALOG_FRAGMENT


def strict_values(
    values: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} params must be a mapping")
    unknown = tuple(sorted(set(values) - allowed))
    missing = tuple(sorted(required - set(values)))
    if unknown:
        raise ValueError(f"unknown {label} params: {list(unknown)}")
    if missing:
        raise ValueError(f"missing {label} params: {list(missing)}")
    return dict(values)


def positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_positive(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def target_modules(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise TypeError("lora_target_modules must be a non-empty sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("lora_target_modules must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError("lora_target_modules must not contain duplicates")
    return result
