"""Import-safe reward declaration surface owned by the algorithm domain."""

from __future__ import annotations

from importlib import import_module

from visual_rl.algorithms.rewards.config import (
    REWARD_CATALOG_FRAGMENT,
    ImageQualityRewardDeclarationProvider,
    PointwiseRewardAdapterConfig,
    RegisteredRewardProfile,
    VideoGeneralRewardDeclarationProvider,
    WorldR1GeneralRewardDeclarationProvider,
    WorldR13DRewardDeclarationProvider,
    describe_registered_reward,
    registered_reward_profile,
    reward_catalog_fragment,
    validate_registered_reward_resource,
)
from visual_rl.algorithms.rewards.interface import RewardComponent
from visual_rl.algorithms.rewards.resource_descriptor import (
    RewardResourceDescriptor,
    RewardRuntimePolicy,
)

__all__ = (
    "GroupwiseReward",
    "GroupwiseRewardOutput",
    "REWARD_CATALOG_FRAGMENT",
    "ImageQualityRewardDeclarationProvider",
    "PointwiseRewardAdapterConfig",
    "PointwiseReward",
    "PointwiseRewardOutput",
    "RegisteredImageQualityReward",
    "RegisteredRewardProfile",
    "RegisteredVideoGeneralReward",
    "RegisteredWorldR13DReward",
    "RegisteredWorldR1GeneralReward",
    "RewardBatchIdentity",
    "RewardBatchView",
    "RewardComponent",
    "RewardInputSelection",
    "RewardInputSelectionPolicy",
    "RewardProcessor",
    "RewardResourceHandle",
    "RewardResourcePoolView",
    "RewardResourcePort",
    "RewardResourceState",
    "RewardResult",
    "RewardRuntimeContext",
    "RewardResourceDescriptor",
    "RewardRuntimePolicy",
    "RewardStage",
    "RewardStageExecutionError",
    "RewardStageInput",
    "RewardStageOutput",
    "VideoGeneralRewardDeclarationProvider",
    "WorldR1GeneralRewardDeclarationProvider",
    "WorldR13DRewardDeclarationProvider",
    "describe_registered_reward",
    "registered_reward_profile",
    "reward_catalog_fragment",
    "validate_registered_reward_resource",
)


_LAZY_EXPORT_MODULE = {
    **{
        name: "visual_rl.algorithms.rewards.components"
        for name in (
            "RegisteredImageQualityReward",
            "RegisteredVideoGeneralReward",
            "RegisteredWorldR13DReward",
            "RegisteredWorldR1GeneralReward",
        )
    },
    "RewardProcessor": "visual_rl.algorithms.rewards.execution",
    "RewardInputSelection": "visual_rl.algorithms.rewards.input_selection",
    "RewardInputSelectionPolicy": "visual_rl.algorithms.rewards.input_selection",
    **{
        name: "visual_rl.algorithms.rewards.resource_port"
        for name in (
            "RewardResourceHandle",
            "RewardResourcePoolView",
            "RewardResourcePort",
            "RewardResourceState",
        )
    },
    **{
        name: "visual_rl.algorithms.rewards.stage"
        for name in (
            "RewardStage",
            "RewardStageExecutionError",
            "RewardStageInput",
            "RewardStageOutput",
        )
    },
    **{
        name: "visual_rl.algorithms.rewards.types"
        for name in (
            "GroupwiseReward",
            "GroupwiseRewardOutput",
            "PointwiseReward",
            "PointwiseRewardOutput",
            "RewardBatchIdentity",
            "RewardBatchView",
            "RewardResult",
            "RewardRuntimeContext",
        )
    },
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)
