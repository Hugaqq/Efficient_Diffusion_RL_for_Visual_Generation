"""Import-safe logical adapters for native pointwise reward resources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from visual_rl.algorithms.rewards.config import (
    PointwiseRewardAdapterConfig,
    describe_registered_reward,
    registered_reward_profile,
    validate_registered_reward_resource,
)
from visual_rl.algorithms.rewards.interface import RewardComponent
from visual_rl.algorithms.rewards.resource_descriptor import (
    RewardResourceDescriptor,
    RewardRuntimePolicy,
)
from visual_rl.core.contracts import MediaKind
from visual_rl.algorithms.rewards.resource_port import (
    RewardResourceHandle,
    RewardResourcePort,
    RewardResourceState,
)
from visual_rl.algorithms.rewards.types import (
    PointwiseReward,
    PointwiseRewardOutput,
    RewardBatchView,
)
from visual_rl.data.samples import (
    CameraConditionBatchState,
    TrajectoryBatch,
)

__all__ = (
    "PointwiseRewardAdapterConfig",
    "RegisteredImageQualityReward",
    "RegisteredVideoGeneralReward",
    "RegisteredWorldR1GeneralReward",
    "RegisteredWorldR13DReward",
    "RewardResourceDescriptor",
    "RewardRuntimePolicy",
)

class _RegisteredPointwiseReward(PointwiseReward, RewardComponent):
    """Config-only logical bridge with one bind-once non-owning port."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.rewards.config:PointwiseRewardAdapterConfig"
    COMPONENT_ID: ClassVar[str]

    def __init__(
        self,
        config: PointwiseRewardAdapterConfig,
    ) -> None:
        if not isinstance(config, PointwiseRewardAdapterConfig):
            raise TypeError("config must be PointwiseRewardAdapterConfig")
        type(self).validate_resource_descriptor(config.resource)
        self.config = config
        self._resource_port = RewardResourcePort()

    @property
    def resource_state(self) -> RewardResourceState:
        return self._resource_port.state

    @property
    def bound_resource_identity(self) -> str | None:
        return self._resource_port.resource_identity

    def bind_resource(self, handle: RewardResourceHandle) -> None:
        self._resource_port.bind(handle, required_method="score")

    def is_bound_to(self, handle: RewardResourceHandle) -> bool:
        return self._resource_port.is_bound_to(handle)

    def close(self) -> None:
        """Close the logical port only; physical resource ownership is external."""

        self._resource_port.close()

    @classmethod
    def describe(cls, config: object) -> object:
        """Serve the legacy resolver without owning a second declaration."""

        if not isinstance(config, PointwiseRewardAdapterConfig):
            raise TypeError("config must be PointwiseRewardAdapterConfig")
        return describe_registered_reward(cls.COMPONENT_ID, config)

    @classmethod
    def validate_resource_descriptor(
        cls,
        descriptor: RewardResourceDescriptor,
    ) -> None:
        """Validate logical-adapter to physical-factory role compatibility."""

        validate_registered_reward_resource(cls.COMPONENT_ID, descriptor)

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> _RegisteredPointwiseReward:
        if not isinstance(config, PointwiseRewardAdapterConfig):
            raise TypeError("config must be PointwiseRewardAdapterConfig")
        if not isinstance(runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")
        # Heavy resources are acquired and activated by the session owner.
        # This construction boundary deliberately does not inspect context.
        return cls(config)

    def score(
        self,
        *,
        logical_reward_id: str,
        resource: object,
        batch: RewardBatchView,
    ) -> PointwiseRewardOutput:
        if not isinstance(logical_reward_id, str) or not logical_reward_id:
            raise ValueError("logical_reward_id must be a non-empty string")
        bound_resource = self._resource_port.resource_for_execution()
        if resource is not bound_resource:
            raise ValueError(
                "reward resource does not match the runtime-bound physical resource"
            )
        if not isinstance(batch, RewardBatchView):
            raise TypeError("batch must be a RewardBatchView")
        self._validate_media(batch)
        self._validate_required_payload(batch)
        output = bound_resource.score(batch=batch)  # type: ignore[attr-defined]
        if not isinstance(output, PointwiseRewardOutput):
            raise TypeError(
                "reward_resource.score() must return PointwiseRewardOutput"
            )
        if output.identity is not batch.identity:
            raise ValueError("reward resource changed batch identity")
        if output.score_axis_names != batch.score_axis_names:
            raise ValueError("reward resource changed score axes")
        if output.values.shape != batch.score_shape:
            raise ValueError("reward resource returned an invalid score shape")
        return output

    @classmethod
    def _validate_media(cls, batch: RewardBatchView) -> None:
        trajectory = batch.payload.get("trajectory")
        if not isinstance(trajectory, TrajectoryBatch):
            raise TypeError("reward payload requires a TrajectoryBatch")
        if batch.score_axis_names:
            layout = {
                "BTCHW": "BCHW",
                "BTFCHW": "BFCHW",
                "BTFHWC": "BFHWC",
            }.get(trajectory.transition_terminal_media_layout)
            if layout is None:
                raise ValueError("reward score axes require typed terminal media")
        else:
            layout = trajectory.media_layout
        profile = registered_reward_profile(cls.COMPONENT_ID)
        if profile.accepted_media is MediaKind.IMAGE:
            if layout != "BCHW":
                raise ValueError(f"{cls.COMPONENT_ID} requires BCHW image media")
            return
        if layout not in {"BFCHW", "BFHWC"}:
            raise ValueError(f"{cls.COMPONENT_ID} requires batched video media")

    @classmethod
    def _validate_required_payload(
        cls,
        batch: RewardBatchView,
    ) -> None:
        payload_type = registered_reward_profile(cls.COMPONENT_ID).required_payload_type
        if payload_type is None:
            return
        if payload_type not in batch.payload:
            raise ValueError(f"{cls.COMPONENT_ID} requires payload {payload_type!r}")
        trajectory = batch.payload.get("trajectory")
        if not isinstance(trajectory, TrajectoryBatch):
            raise TypeError("reward payload requires a TrajectoryBatch")
        state = trajectory.condition_state
        if not isinstance(state, CameraConditionBatchState):
            raise TypeError(f"{payload_type!r} requires camera condition state")
        if batch.payload[payload_type] is not state.camera_trajectory:
            raise ValueError(
                f"{payload_type!r} must be the trajectory camera payload object"
            )


class RegisteredImageQualityReward(_RegisteredPointwiseReward):
    """IMAGE pointwise adapter for the existing image-quality kernel."""

    COMPONENT_ID = "image-quality"


class RegisteredVideoGeneralReward(_RegisteredPointwiseReward):
    """VIDEO pointwise adapter for a runtime-bound general video kernel."""

    COMPONENT_ID = "video-general"


class RegisteredWorldR1GeneralReward(_RegisteredPointwiseReward):
    """VIDEO pointwise adapter for World-R1's batch-shared frame client."""

    COMPONENT_ID = "world-r1-general"


class RegisteredWorldR13DReward(_RegisteredPointwiseReward):
    """VIDEO pointwise adapter requiring the typed World-R1 camera payload."""

    COMPONENT_ID = "world-r1-3d"
