"""BaseTrainer-compatible reward stage over the resolved routing plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from visual_rl.algorithms.rollout.interface import RolloutExecution
from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue
from visual_rl.core.contracts import RewardGranularity, RewardPlanSpec
from visual_rl.algorithms.rewards.execution import RewardProcessor
from visual_rl.algorithms.rewards.resource_port import RewardResourcePoolView
from visual_rl.algorithms.rewards.types import (
    GroupwiseReward,
    PointwiseReward,
    RewardBatchIdentity,
    RewardBatchView,
    RewardResult,
    RewardRuntimeContext,
)
from visual_rl.data.samples import (
    CameraConditionBatchState,
    NoConditionBatchState,
    StackedSampleBatch,
    TrajectoryBatch,
)

__all__ = (
    "RewardStage",
    "RewardStageExecutionError",
    "RewardStageInput",
    "RewardStageOutput",
)


class RewardStageExecutionError(RuntimeError):
    """An applicable reward failed to produce a valid row."""


@dataclass(frozen=True, slots=True)
class RewardStageInput:
    """Complete typed input for one reward route and its replay provenance."""

    execution: RolloutExecution
    samples: StackedSampleBatch
    runtime_context: RewardRuntimeContext

    @property
    def trajectory(self) -> TrajectoryBatch:
        return self.execution.trajectory

    def __post_init__(self) -> None:
        if not isinstance(self.execution, RolloutExecution):
            raise TypeError("execution must be a RolloutExecution")
        if not isinstance(self.samples, StackedSampleBatch):
            raise TypeError("samples must be a StackedSampleBatch")
        if not isinstance(self.runtime_context, RewardRuntimeContext):
            raise TypeError("runtime_context must be a RewardRuntimeContext")
        if self.samples.batch_size != self.trajectory.batch_size:
            raise ValueError("samples and trajectory batch sizes differ")
        contexts = self.trajectory.contexts
        if tuple(row.identity for row in self.samples.rows) != tuple(
            item.batch_row_identity for item in contexts
        ):
            raise ValueError("samples and trajectory batch rows differ")
        if tuple(source.source_item_id for source in self.samples.sources) != tuple(
            item.batch_row.source_item_id for item in contexts
        ):
            raise ValueError("trajectory changed sample source identity")
        expected_layouts = {
            "t2i": {"BCHW"},
            "t2v": {"BFCHW", "BFHWC"},
            "i2v": {"BFCHW", "BFHWC"},
        }[self.samples.task_type]
        if self.trajectory.media_layout not in expected_layouts:
            raise ValueError("trajectory media layout does not match sample task")


@dataclass(frozen=True, slots=True)
class RewardStageOutput:
    """Trajectory retained alongside the reward result for advantage/credit."""

    trajectory: TrajectoryBatch
    reward_result: RewardResult

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TrajectoryBatch):
            raise TypeError("trajectory must be a TrajectoryBatch")
        if not isinstance(self.reward_result, RewardResult):
            raise TypeError("reward_result must be a RewardResult")
        sample_ids = tuple(item.sample_id for item in self.trajectory.contexts)
        if self.reward_result.identity.sample_ids != sample_ids:
            raise ValueError("reward result does not belong to the trajectory")


class RewardStage:
    """Execute one precompiled route without reward-local source/phase logic."""

    def __init__(
        self,
        *,
        plan: RewardPlanSpec,
        pool: RewardResourcePoolView,
        logical_rewards: Mapping[str, PointwiseReward | GroupwiseReward],
    ) -> None:
        if not isinstance(plan, RewardPlanSpec):
            raise TypeError("plan must be a RewardPlanSpec")
        if not plan.materialized:
            raise ValueError("RewardStage requires a materialized RewardPlanSpec")
        unsupported = tuple(
            logical.logical_reward_id
            for logical in plan.logical_rewards
            if logical.contract.granularity is not RewardGranularity.POINTWISE
        )
        if unsupported:
            raise NotImplementedError(
                "RewardStage currently supports only pointwise rewards; "
                f"groupwise logical ids={sorted(set(unsupported))}"
            )
        self.plan = plan
        self.pool = pool
        self.processor = RewardProcessor(
            plan=plan,
            pool=pool,
            logical_rewards=logical_rewards,
        )

    def __call__(self, value: StageValue[object]) -> StageValue[RewardStageOutput]:
        if not isinstance(value, StageValue):
            raise TypeError("RewardStage requires a StageValue")
        stage_input = value.payload
        if not isinstance(stage_input, RewardStageInput):
            raise TypeError("RewardStage payload must be a RewardStageInput")
        trajectory = stage_input.trajectory
        identity = value.identity
        self._validate_iteration(identity, stage_input)
        route = self.plan.route_for(
            source_id=identity.source_id,
            phase_id=identity.phase_id,
        )
        batch = self._batch_view(identity, stage_input, route.logical_reward_ids)
        for binding in route.rewards:
            payload_type = self.plan.logical_reward(
                binding.logical_reward_id
            ).contract.required_payload_type
            if payload_type is None:
                continue
            if payload_type not in batch.payload:
                raise ValueError(
                    f"logical reward {binding.logical_reward_id!r} requires "
                    f"payload {payload_type!r}"
                )
            if payload_type == "camera_trajectory_v1":
                state = trajectory.condition_state
                if not isinstance(state, CameraConditionBatchState):
                    raise ValueError(
                        "camera_trajectory_v1 must come from trajectory condition state"
                    )
                if batch.payload[payload_type] is not state.camera_trajectory:
                    raise ValueError(
                        "camera reward payload changed trajectory condition identity"
                    )
        result = self.processor.process(batch=batch, route=route)
        for logical_id in route.logical_reward_ids:
            applicable = result.component_applicable_masks[logical_id]
            valid = result.component_valid_masks[logical_id]
            if bool(np.any(applicable & ~valid)):
                raise RewardStageExecutionError(
                    f"applicable reward {logical_id!r} produced invalid rows"
                )
        return StageValue(
            identity=identity,
            payload=RewardStageOutput(
                trajectory=trajectory,
                reward_result=result,
            ),
        )

    @staticmethod
    def _validate_iteration(
        identity: IterationIdentity,
        stage_input: RewardStageInput,
    ) -> None:
        trajectory = stage_input.trajectory
        if trajectory.batch_size != identity.batch_size:
            raise ValueError("trajectory and iteration batch sizes differ")
        contexts = trajectory.contexts
        if (
            tuple(item.batch_row_identity for item in contexts)
            != identity.row_identities
        ):
            raise ValueError("trajectory batch rows do not match iteration identity")
        if tuple(item.batch_row.group_id for item in contexts) != identity.group_ids:
            raise ValueError("trajectory groups do not match iteration identity")
        if tuple(item.batch_row.member_id for item in contexts) != identity.member_ids:
            raise ValueError("trajectory members do not match iteration identity")
        if any(
            item.batch_row.phase != identity.phase_id
            or item.batch_row.optimizer_step != identity.optimizer_step
            for item in contexts
        ):
            raise ValueError("trajectory phase/step does not match iteration identity")
        if stage_input.runtime_context.step_context.step != identity.optimizer_step:
            raise ValueError("reward StepContext.step does not match optimizer_step")

    @classmethod
    def _batch_view(
        cls,
        identity: IterationIdentity,
        stage_input: RewardStageInput,
        logical_reward_ids: tuple[str, ...],
    ) -> RewardBatchView:
        trajectory = stage_input.trajectory
        score_axis_names: tuple[str, ...] = ()
        score_axis_sizes: tuple[int, ...] = ()
        reward_media = trajectory.media
        topology = trajectory.branch_topology
        if (
            trajectory.kind == "branching"
            and topology is not None
            and topology.kind == "every_policy_timestep"
        ):
            if trajectory.transition_terminal_media is None:
                raise ValueError(
                    "TempFlow paper topology requires transition terminal media"
                )
            score_axis_names = ("branch_timestep",)
            score_axis_sizes = (trajectory.transition_count,)
            reward_media = trajectory.transition_terminal_media
        payloads = {
            "trajectory": trajectory,
            "media": reward_media,
            "condition_state": trajectory.condition_state,
            "samples": stage_input.samples,
            "reward_runtime_context": stage_input.runtime_context,
        }
        payloads["rollout_execution"] = stage_input.execution
        condition_ids = cls._condition_payloads(trajectory, payloads)
        batch_identity = RewardBatchIdentity(
            source_id=identity.source_id,
            phase_id=identity.phase_id,
            batch_row_ids=identity.row_identities,
            sample_ids=tuple(item.sample_id for item in trajectory.contexts),
            trajectory_ids=tuple(item.trajectory_id for item in trajectory.contexts),
            condition_payload_ids=condition_ids,
            group_ids=identity.group_ids,
        )
        view = RewardBatchView(
            identity=batch_identity,
            active_reward_ids=logical_reward_ids,
            payload=payloads,
            score_axis_names=score_axis_names,
            score_axis_sizes=score_axis_sizes,
        )
        return view

    @staticmethod
    def _condition_payloads(
        trajectory: TrajectoryBatch,
        payloads: dict[str, object],
    ) -> tuple[str, ...]:
        state = trajectory.condition_state
        if isinstance(state, NoConditionBatchState):
            condition_ids = ("none",) * trajectory.batch_size
        elif isinstance(state, CameraConditionBatchState):
            # Bind rewards to the content-derived row identities, not merely
            # to the conditioner configuration that produced the payload.
            condition_ids = state.row_condition_identities
            payloads["camera_trajectory_v1"] = state.camera_trajectory
        else:
            raise TypeError("unsupported trajectory condition state")
        for row, expected in zip(
            trajectory.condition_identity,
            condition_ids,
            strict=True,
        ):
            if any(item != expected for item in row):
                raise ValueError(
                    "trajectory condition identity does not match condition payload"
                )
        return condition_ids
