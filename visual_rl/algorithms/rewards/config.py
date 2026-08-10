"""Import-safe declarations for the four built-in reward adapters.

This module owns logical reward configuration and capability declaration only.
Runtime scoring adapters remain behind implementation class paths and are not
imported while a catalog is assembled or a declaration is resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.algorithms.rewards.resource_descriptor import (
    RewardResourceDescriptor,
)
from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    DeclaredContract,
    MediaKind,
    RewardContract,
    RewardGranularity,
)

__all__ = (
    "REWARD_CATALOG_FRAGMENT",
    "ImageQualityRewardDeclarationProvider",
    "PointwiseRewardAdapterConfig",
    "RegisteredRewardProfile",
    "VideoGeneralRewardDeclarationProvider",
    "WorldR1GeneralRewardDeclarationProvider",
    "WorldR13DRewardDeclarationProvider",
    "describe_registered_reward",
    "registered_reward_profile",
    "reward_catalog_fragment",
    "validate_registered_reward_resource",
)


@dataclass(frozen=True, slots=True)
class RegisteredRewardProfile:
    """Static semantics shared by declaration providers and runtime adapters."""

    component_id: str
    accepted_media: MediaKind
    required_payload_type: str | None = None
    frame_aggregation: str | None = None
    required_factory_class: str | None = None


_REGISTERED_REWARD_PROFILES = (
    RegisteredRewardProfile(
        component_id="image-quality",
        accepted_media=MediaKind.IMAGE,
    ),
    RegisteredRewardProfile(
        component_id="video-general",
        accepted_media=MediaKind.VIDEO,
        frame_aggregation="resource_defined",
    ),
    RegisteredRewardProfile(
        component_id="world-r1-general",
        accepted_media=MediaKind.VIDEO,
        frame_aggregation="keyed_uniform_frame_batch_shared",
        required_factory_class="reward_general",
    ),
    RegisteredRewardProfile(
        component_id="world-r1-3d",
        accepted_media=MediaKind.VIDEO,
        required_payload_type="camera_trajectory_v1",
        frame_aggregation="all_frames",
        required_factory_class="reward_3d",
    ),
)


@dataclass(frozen=True, slots=True)
class PointwiseRewardAdapterConfig:
    """Logical adapter config with one explicit physical descriptor."""

    resource: RewardResourceDescriptor

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> PointwiseRewardAdapterConfig:
        del context
        if not isinstance(values, Mapping):
            raise TypeError("pointwise reward params must be a mapping")
        if set(values) != {"resource"}:
            raise ValueError("pointwise reward params must contain exactly resource")
        return cls(resource=RewardResourceDescriptor.from_mapping(values["resource"]))


def registered_reward_profile(component_id: str) -> RegisteredRewardProfile:
    """Return the frozen profile for one built-in logical reward alias."""

    if not isinstance(component_id, str) or not component_id:
        raise ValueError("component_id must be a non-empty string")
    profile = next(
        (
            item
            for item in _REGISTERED_REWARD_PROFILES
            if item.component_id == component_id
        ),
        None,
    )
    if profile is None:
        raise ValueError(f"unknown registered reward component {component_id!r}")
    return profile


def validate_registered_reward_resource(
    component_id: str,
    descriptor: RewardResourceDescriptor,
) -> None:
    """Validate logical-adapter to physical-factory role compatibility."""

    if not isinstance(descriptor, RewardResourceDescriptor):
        raise TypeError("resource must be RewardResourceDescriptor")
    profile = registered_reward_profile(component_id)
    required_factory = profile.required_factory_class
    if required_factory is not None and descriptor.factory_class != required_factory:
        raise ValueError(
            f"{component_id} requires factory_class {required_factory!r}, "
            f"got {descriptor.factory_class!r}"
        )


def describe_registered_reward(
    component_id: str,
    config: PointwiseRewardAdapterConfig,
) -> DeclaredContract:
    """Build the exact static reward contract without importing runtime code."""

    if not isinstance(config, PointwiseRewardAdapterConfig):
        raise TypeError("config must be PointwiseRewardAdapterConfig")
    validate_registered_reward_resource(component_id, config.resource)
    profile = registered_reward_profile(component_id)
    return DeclaredContract(
        component_kind="reward",
        component_id=profile.component_id,
        reward=RewardContract(
            accepted_media=(profile.accepted_media,),
            required_payload_type=profile.required_payload_type,
            granularity=RewardGranularity.POINTWISE,
            output_rank=1,
            frame_aggregation=profile.frame_aggregation,
        ),
    )


class _PointwiseRewardDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = (
        "visual_rl.algorithms.rewards.config:PointwiseRewardAdapterConfig"
    )
    COMPONENT_ID: str

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        config = PointwiseRewardAdapterConfig.from_mapping(
            raw_params,
            context=context,
        )
        return ComponentDeclaration(
            config=config,
            declared_contract=describe_registered_reward(cls.COMPONENT_ID, config),
        )


class ImageQualityRewardDeclarationProvider(_PointwiseRewardDeclarationProvider):
    """Declare the image-quality adapter without importing its runtime."""

    COMPONENT_ID = "image-quality"


class VideoGeneralRewardDeclarationProvider(_PointwiseRewardDeclarationProvider):
    """Declare the general-video adapter without importing its runtime."""

    COMPONENT_ID = "video-general"


class WorldR1GeneralRewardDeclarationProvider(_PointwiseRewardDeclarationProvider):
    """Declare World-R1's general-video adapter and factory role."""

    COMPONENT_ID = "world-r1-general"


class WorldR13DRewardDeclarationProvider(_PointwiseRewardDeclarationProvider):
    """Declare World-R1's camera-aware 3D adapter and payload contract."""

    COMPONENT_ID = "world-r1-3d"


REWARD_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.rewards",
    kind="reward",
    descriptors=(
        ComponentDescriptor(
            alias="image-quality",
            implementation_class_path=(
                "visual_rl.algorithms.rewards.components:RegisteredImageQualityReward"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rewards.config:"
                "ImageQualityRewardDeclarationProvider"
            ),
            optional_dependencies=("numpy",),
        ),
        ComponentDescriptor(
            alias="video-general",
            implementation_class_path=(
                "visual_rl.algorithms.rewards.components:RegisteredVideoGeneralReward"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rewards.config:"
                "VideoGeneralRewardDeclarationProvider"
            ),
            optional_dependencies=("numpy",),
        ),
        ComponentDescriptor(
            alias="world-r1-general",
            implementation_class_path=(
                "visual_rl.algorithms.rewards.components:RegisteredWorldR1GeneralReward"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rewards.config:"
                "WorldR1GeneralRewardDeclarationProvider"
            ),
            optional_dependencies=("numpy",),
        ),
        ComponentDescriptor(
            alias="world-r1-3d",
            implementation_class_path=(
                "visual_rl.algorithms.rewards.components:RegisteredWorldR13DReward"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rewards.config:WorldR13DRewardDeclarationProvider"
            ),
            optional_dependencies=("numpy",),
        ),
    ),
)


def reward_catalog_fragment() -> CatalogFragment:
    """Return the immutable built-in reward descriptor contribution."""

    return REWARD_CATALOG_FRAGMENT
