"""Import-safe declarations for algorithm-owned latent conditioning."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    ConditionerContract,
    DeclaredContract,
    LatentLayout,
    TaskKind,
)

__all__ = (
    "CONDITIONING_CATALOG_FRAGMENT",
    "WorldR1CameraConfig",
    "WorldR1CameraDeclarationProvider",
    "conditioning_catalog_fragment",
)


def _canonical(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class WorldR1CameraConfig:
    frames_per_trajectory: int = 81
    wrap_strength: float = 0.8
    guidance_steps: int = 4
    injection_mode: Literal["blend", "lowpass_delta"] = "lowpass_delta"
    delta_lowpass_kernel: int = 9
    noise_downtemp_interp: Literal["nearest", "blend"] = "nearest"
    noise_downspatial_mode: Literal["area", "resize_noise"] = "area"
    noise_degradation: float = 0.35
    flow_scale: int = 16
    force_camera_movement: str | None = None

    def __post_init__(self) -> None:
        for name in ("frames_per_trajectory", "flow_scale"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.guidance_steps) is not int or self.guidance_steps < 0:
            raise ValueError("guidance_steps must be a non-negative integer")
        if (
            type(self.delta_lowpass_kernel) is not int
            or self.delta_lowpass_kernel < 1
            or self.delta_lowpass_kernel % 2 == 0
        ):
            raise ValueError("delta_lowpass_kernel must be a positive odd integer")
        for name in ("wrap_strength", "noise_degradation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.injection_mode not in {"blend", "lowpass_delta"}:
            raise ValueError("injection_mode is invalid")
        if self.noise_downtemp_interp not in {"nearest", "blend"}:
            raise ValueError("noise_downtemp_interp is invalid")
        if self.noise_downspatial_mode not in {"area", "resize_noise"}:
            raise ValueError("noise_downspatial_mode is invalid")
        if self.force_camera_movement is not None and not self.force_camera_movement:
            raise ValueError("force_camera_movement must be non-empty or None")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> WorldR1CameraConfig:
        del context
        if not isinstance(values, Mapping):
            raise TypeError("camera conditioner params must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown camera conditioner params: {unknown}")
        return cls(**dict(values))

    @property
    def config_identity(self) -> str:
        return hashlib.sha256(_canonical(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="conditioner",
            component_id="world-r1-camera",
            conditioner=ConditionerContract(
                accepted_tasks=(TaskKind.T2V,),
                accepted_latent_layouts=(LatentLayout.BCTHW,),
                payload_type="camera_trajectory_v1",
                has_initialize_hook=True,
                has_after_step_hook=self.guidance_steps > 0,
                deterministic_given_state=True,
                replay_state_serializable=True,
                independent_of_policy_parameters=True,
                required_modalities=("prompt_text",),
                provided_output_fields=("camera_trajectory",),
            ),
        )


class WorldR1CameraDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.conditioning.config:WorldR1CameraConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = WorldR1CameraConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


CONDITIONING_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.conditioning",
    kind="conditioner",
    descriptors=(
        ComponentDescriptor(
            alias="world-r1-camera",
            implementation_class_path=(
                "visual_rl.algorithms.conditioning.world_r1_camera:"
                "WorldR1CameraConditioner"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.conditioning.config:"
                "WorldR1CameraDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
    ),
)


def conditioning_catalog_fragment() -> CatalogFragment:
    return CONDITIONING_CATALOG_FRAGMENT
