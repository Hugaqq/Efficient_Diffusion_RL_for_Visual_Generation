"""Typed adapters closing the six-stage GRPO-family training hot path."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any

from visual_rl.algorithms.optimization.advantage import (
    AdvantageGrouping,
    GroupZScoreAdvantageProcessor,
    NormalizedAdvantage,
)
from visual_rl.algorithms.optimization.credit import CoefficientMeanReducer
from visual_rl.algorithms.optimization.execution import UpdateExecutionPlan
from visual_rl.algorithms.optimization.interface import CreditPlanningPort
from visual_rl.algorithms.optimization.kernel import (
    PolicyUpdateKernel,
    PolicyUpdateResult,
)
from visual_rl.algorithms.optimization.objective import PolicyLossInputs
from visual_rl.algorithms.optimization.recompute import (
    PolicyRecomputer,
    PolicyRecomputeRequest,
)
from visual_rl.algorithms.rewards import (
    RewardResult,
    RewardRuntimeContext,
    RewardStage,
    RewardStageInput,
    RewardStageOutput,
)
from visual_rl.algorithms.rollout.interface import (
    RolloutComponent,
    RolloutExecution,
    RolloutRequest,
)
from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue
from visual_rl.data import BatchPhaseBinding, PhaseRoute
from visual_rl.data.prelude import PreludeBatchPayload
from visual_rl.data.samples import StackedSampleBatch, TrajectoryBatch

__all__ = (
    "AdvantageStage",
    "AdvantagedRollout",
    "CreditAssignedRollout",
    "CreditStage",
    "OptimizeStage",
    "OptimizedIteration",
    "RewardPipelineStage",
    "RewardedRollout",
    "RolloutStage",
    "RolloutStagePayload",
)


ContextFactory = Callable[[], AbstractContextManager[Any]]
RolloutRequestFactory = Callable[
    [StackedSampleBatch, IterationIdentity],
    RolloutRequest,
]
RewardContextFactory = Callable[
    ["RolloutStagePayload", IterationIdentity],
    RewardRuntimeContext,
]


def _context(factory: ContextFactory | None) -> AbstractContextManager[Any]:
    return nullcontext() if factory is None else factory()


def _validate_iteration_trajectory(
    identity: IterationIdentity,
    trajectory: TrajectoryBatch,
) -> None:
    if not isinstance(identity, IterationIdentity):
        raise TypeError("identity must be an IterationIdentity")
    if not isinstance(trajectory, TrajectoryBatch):
        raise TypeError("trajectory must be a TrajectoryBatch")
    contexts = trajectory.contexts
    observed = (
        tuple(item.batch_row_identity for item in contexts),
        tuple(item.batch_row.group_id for item in contexts),
        tuple(item.batch_row.member_id for item in contexts),
    )
    expected = (
        identity.row_identities,
        identity.group_ids,
        identity.member_ids,
    )
    if observed != expected:
        raise ValueError("trajectory rows do not match the iteration identity")
    if any(
        item.batch_row.phase != identity.phase_id
        or item.batch_row.optimizer_step != identity.optimizer_step
        for item in contexts
    ):
        raise ValueError("trajectory phase/step does not match iteration identity")


@dataclass(frozen=True, slots=True)
class RolloutStagePayload:
    samples: StackedSampleBatch
    request: RolloutRequest
    execution: RolloutExecution
    data_plane: PreludeBatchPayload | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.samples, StackedSampleBatch):
            raise TypeError("samples must be a StackedSampleBatch")
        if not isinstance(self.request, RolloutRequest):
            raise TypeError("request must be a RolloutRequest")
        if not isinstance(self.execution, RolloutExecution):
            raise TypeError("execution must be a RolloutExecution")
        if self.request.samples is not self.samples:
            raise ValueError("rollout request must retain the exact sample batch")
        if self.data_plane is not None:
            if not isinstance(self.data_plane, PreludeBatchPayload):
                raise TypeError("data_plane must be PreludeBatchPayload or None")
            if self.data_plane.samples is not self.samples:
                raise ValueError("data_plane must retain the exact sample batch")

    @property
    def trajectory(self) -> TrajectoryBatch:
        return self.execution.trajectory

    @property
    def active_rewards(self) -> tuple[str, ...] | None:
        return None if self.data_plane is None else self.data_plane.active_rewards

    @property
    def phase_binding(self) -> BatchPhaseBinding | None:
        return None if self.data_plane is None else self.data_plane.phase_binding

    @property
    def phase_route(self) -> PhaseRoute | None:
        return None if self.data_plane is None else self.data_plane.route


@dataclass(frozen=True, slots=True)
class RewardedRollout:
    rollout: RolloutStagePayload
    reward_result: RewardResult

    def __post_init__(self) -> None:
        if not isinstance(self.rollout, RolloutStagePayload):
            raise TypeError("rollout must be a RolloutStagePayload")
        if not isinstance(self.reward_result, RewardResult):
            raise TypeError("reward_result must be a RewardResult")


@dataclass(frozen=True, slots=True)
class AdvantagedRollout:
    rewarded: RewardedRollout
    advantage: NormalizedAdvantage

    def __post_init__(self) -> None:
        if not isinstance(self.rewarded, RewardedRollout):
            raise TypeError("rewarded must be a RewardedRollout")
        if not isinstance(self.advantage, NormalizedAdvantage):
            raise TypeError("advantage must be a NormalizedAdvantage")
        self.advantage.validate_against_trajectory(self.rewarded.rollout.trajectory)


@dataclass(frozen=True, slots=True)
class CreditAssignedRollout:
    advantaged: AdvantagedRollout
    loss_inputs: PolicyLossInputs

    def __post_init__(self) -> None:
        if not isinstance(self.advantaged, AdvantagedRollout):
            raise TypeError("advantaged must be an AdvantagedRollout")
        if not isinstance(self.loss_inputs, PolicyLossInputs):
            raise TypeError("loss_inputs must be PolicyLossInputs")
        trajectory = self.advantaged.rewarded.rollout.trajectory
        if tuple(self.loss_inputs.base_advantage.shape) != tuple(
            trajectory.old_log_probs.shape
        ):
            raise ValueError("loss_inputs must match the full trajectory [B,T]")


@dataclass(frozen=True, slots=True)
class OptimizedIteration:
    credit: CreditAssignedRollout
    update: PolicyUpdateResult

    def __post_init__(self) -> None:
        if not isinstance(self.credit, CreditAssignedRollout):
            raise TypeError("credit must be a CreditAssignedRollout")
        if not isinstance(self.update, PolicyUpdateResult):
            raise TypeError("update must be a PolicyUpdateResult")


class RolloutStage:
    """Pre-encode under PREPROCESS, then sample under ROLLOUT residency."""

    def __init__(
        self,
        *,
        rollout: RolloutComponent,
        request_factory: RolloutRequestFactory,
        preprocess_context: ContextFactory | None = None,
        rollout_context: ContextFactory | None = None,
    ) -> None:
        if not isinstance(rollout, RolloutComponent):
            raise TypeError("rollout must be a RolloutComponent")
        if not callable(request_factory):
            raise TypeError("request_factory must be callable")
        for name, factory in (
            ("preprocess_context", preprocess_context),
            ("rollout_context", rollout_context),
        ):
            if factory is not None and not callable(factory):
                raise TypeError(f"{name} must be callable or None")
        self.rollout = rollout
        self.request_factory = request_factory
        self.preprocess_context = preprocess_context
        self.rollout_context = rollout_context

    def __call__(self, value: StageValue[object]) -> StageValue[RolloutStagePayload]:
        if not isinstance(value, StageValue):
            raise TypeError("RolloutStage requires a StageValue")
        incoming = value.payload
        data_plane = incoming if isinstance(incoming, PreludeBatchPayload) else None
        samples = data_plane.samples if data_plane is not None else incoming
        if not isinstance(samples, StackedSampleBatch):
            raise TypeError(
                "RolloutStage payload must be StackedSampleBatch or PreludeBatchPayload"
            )
        request = self.request_factory(samples, value.identity)
        if not isinstance(request, RolloutRequest):
            raise TypeError("request_factory must return a RolloutRequest")
        if request.samples is not samples:
            raise ValueError("request_factory replaced the canonical sample batch")
        if request.encoded_conditioning is None:
            with _context(self.preprocess_context):
                conditioning = request.adapter.encode(samples)
            identities = getattr(
                conditioning,
                "condition_identity",
                tuple(row.identity for row in samples.rows),
            )
            request = replace(
                request,
                encoded_conditioning=conditioning,
                model_condition_identity=identities,
            )
        with _context(self.rollout_context):
            execution = self.rollout.run_with_snapshot(request)
        _validate_iteration_trajectory(value.identity, execution.trajectory)
        return StageValue(
            identity=value.identity,
            payload=RolloutStagePayload(
                samples,
                request,
                execution,
                data_plane,
            ),
        )


class RewardPipelineStage:
    """Preserve rollout replay state while delegating reward execution."""

    def __init__(
        self,
        reward_stage: RewardStage,
        *,
        runtime_context_factory: RewardContextFactory,
    ) -> None:
        if not isinstance(reward_stage, RewardStage):
            raise TypeError("reward_stage must be a RewardStage")
        if not callable(runtime_context_factory):
            raise TypeError("runtime_context_factory must be callable")
        self.reward_stage = reward_stage
        self.runtime_context_factory = runtime_context_factory

    def __call__(self, value: StageValue[object]) -> StageValue[RewardedRollout]:
        if not isinstance(value, StageValue):
            raise TypeError("RewardPipelineStage requires a StageValue")
        payload = value.payload
        if not isinstance(payload, RolloutStagePayload):
            raise TypeError("reward payload must be a RolloutStagePayload")
        runtime_context = self.runtime_context_factory(payload, value.identity)
        if not isinstance(runtime_context, RewardRuntimeContext):
            raise TypeError("runtime_context_factory must return RewardRuntimeContext")
        rewarded = self.reward_stage(
            StageValue(
                identity=value.identity,
                payload=RewardStageInput(
                    execution=payload.execution,
                    samples=payload.samples,
                    runtime_context=runtime_context,
                ),
            )
        )
        if rewarded.identity is not value.identity:
            raise ValueError("reward stage replaced the iteration identity")
        result = rewarded.payload
        if not isinstance(result, RewardStageOutput):
            raise TypeError("reward stage returned an invalid payload")
        return StageValue(
            identity=value.identity,
            payload=RewardedRollout(payload, result.reward_result),
        )


class AdvantageStage:
    """Apply the one group normalizer and retain all replay dependencies."""

    def __init__(self, processor: GroupZScoreAdvantageProcessor) -> None:
        if not isinstance(processor, GroupZScoreAdvantageProcessor):
            raise TypeError("processor must be GroupZScoreAdvantageProcessor")
        self.processor = processor

    def __call__(self, value: StageValue[object]) -> StageValue[AdvantagedRollout]:
        if not isinstance(value, StageValue):
            raise TypeError("AdvantageStage requires a StageValue")
        payload = value.payload
        if not isinstance(payload, RewardedRollout):
            raise TypeError("advantage payload must be a RewardedRollout")
        trajectory = payload.rollout.trajectory
        advantage = self.processor.normalize(
            payload.reward_result,
            AdvantageGrouping.from_trajectory(trajectory),
            device=trajectory.old_log_probs.device,
        )
        return StageValue(
            identity=value.identity,
            payload=AdvantagedRollout(payload, advantage),
        )


class CreditStage:
    """Build a detached full-group credit plan with no model recomputation."""

    def __init__(
        self,
        *,
        strategy: CreditPlanningPort,
        coefficient_mean_reducer: CoefficientMeanReducer | None = None,
    ) -> None:
        if not isinstance(strategy, CreditPlanningPort):
            raise TypeError("strategy must implement CreditPlanningPort")
        self.strategy = strategy
        self.coefficient_mean_reducer = coefficient_mean_reducer

    def __call__(
        self,
        value: StageValue[object],
    ) -> StageValue[CreditAssignedRollout]:
        if not isinstance(value, StageValue):
            raise TypeError("CreditStage requires a StageValue")
        payload = value.payload
        if not isinstance(payload, AdvantagedRollout):
            raise TypeError("credit payload must be an AdvantagedRollout")
        rollout = payload.rewarded.rollout
        loss_inputs = self.strategy.plan(
            trajectory=rollout.trajectory,
            advantage=payload.advantage,
            coefficient_mean_reducer=self.coefficient_mean_reducer,
        )
        return StageValue(
            identity=value.identity,
            payload=CreditAssignedRollout(payload, loss_inputs),
        )


class OptimizeStage:
    """Commit one logical update through the sole PolicyUpdateKernel."""

    def __init__(
        self,
        *,
        optimizer: object,
        scaler: object | None,
        kernel: PolicyUpdateKernel | None = None,
        accelerator: object | None = None,
        prepared_root: object | None = None,
        lr_scheduler: object | None = None,
        ema_update: Callable[[], None] | None = None,
        reference_update: Callable[[], None] | None = None,
        logical_commit: Callable[[int], None] | None = None,
        execution_plan: UpdateExecutionPlan | None = None,
        require_reference_statistics: bool = False,
        recomputer: PolicyRecomputer | None = None,
        current_context: ContextFactory | None = None,
        reference_context: ContextFactory | None = None,
    ) -> None:
        if optimizer is None:
            raise TypeError("optimizer must not be None")
        if kernel is not None and not isinstance(kernel, PolicyUpdateKernel):
            raise TypeError("kernel must be PolicyUpdateKernel or None")
        if execution_plan is not None and not isinstance(
            execution_plan,
            UpdateExecutionPlan,
        ):
            raise TypeError("execution_plan must be UpdateExecutionPlan or None")
        if type(require_reference_statistics) is not bool:
            raise TypeError("require_reference_statistics must be bool")
        if recomputer is not None and not isinstance(recomputer, PolicyRecomputer):
            raise TypeError("recomputer must be PolicyRecomputer or None")
        for name, factory in (
            ("current_context", current_context),
            ("reference_context", reference_context),
        ):
            if factory is not None and not callable(factory):
                raise TypeError(f"{name} must be callable or None")
        if current_context is None and reference_context is not None:
            raise ValueError("reference_context requires current_context")
        if require_reference_statistics and (current_context is None) != (
            reference_context is None
        ):
            raise ValueError(
                "reference statistics require current/reference contexts together"
            )
        if not require_reference_statistics and reference_context is not None:
            raise ValueError(
                "reference_context is invalid when reference statistics are disabled"
            )
        self.optimizer = optimizer
        self.scaler = scaler
        self.kernel = kernel or PolicyUpdateKernel()
        self.accelerator = accelerator
        self.prepared_root = prepared_root
        self.lr_scheduler = lr_scheduler
        self.ema_update = ema_update
        self.reference_update = reference_update
        self.logical_commit = logical_commit
        self.execution_plan = execution_plan
        self.require_reference_statistics = require_reference_statistics
        self.recomputer = recomputer or PolicyRecomputer()
        self.current_context = current_context
        self.reference_context = reference_context

    def __call__(self, value: StageValue[object]) -> StageValue[OptimizedIteration]:
        if not isinstance(value, StageValue):
            raise TypeError("OptimizeStage requires a StageValue")
        payload = value.payload
        if not isinstance(payload, CreditAssignedRollout):
            raise TypeError("optimize payload must be a CreditAssignedRollout")
        trajectory = payload.advantaged.rewarded.rollout.trajectory
        rollout = payload.advantaged.rewarded.rollout
        request = rollout.request
        recompute_request = PolicyRecomputeRequest(
            adapter=request.adapter,
            dynamics=request.dynamics,
            rollout=rollout.execution,
            latent_spec=request.latent_spec,
            guidance=request.guidance,
            require_reference_statistics=self.require_reference_statistics,
            current_context=self.current_context,
            reference_context=self.reference_context,
        )
        try:
            update = self.kernel.step_slots(
                trajectory=trajectory,
                loss_inputs=payload.loss_inputs,
                recompute_request=recompute_request,
                optimizer=self.optimizer,
                scaler=self.scaler,
                optimizer_step=value.identity.optimizer_step,
                recomputer=self.recomputer,
                accelerator=self.accelerator,
                prepared_root=self.prepared_root,
                lr_scheduler=self.lr_scheduler,
                ema_update=self.ema_update,
                reference_update=self.reference_update,
                logical_commit=self.logical_commit,
                execution_plan=self.execution_plan,
            )
        except BaseException as error:
            # ``step_slots`` has already removed its heavy implementation frames.
            # Clear this forwarding frame too before the same typed failure is
            # allowed to propagate into the trainer/controller traceback.
            if "gradient gate failed: all gradients are zero" in str(error):
                reward = payload.advantaged.rewarded.reward_result
                reward_values = reward.weighted_total[reward.valid_mask]
                advantage = payload.advantaged.advantage
                advantage_values = advantage.values.masked_select(
                    advantage.valid_mask
                )
                active_advantage = payload.loss_inputs.base_advantage.masked_select(
                    payload.loss_inputs.active_mask
                )
                error = RuntimeError(
                    f"{error}; "
                    f"reward_min={float(reward_values.min()):.9g}; "
                    f"reward_max={float(reward_values.max()):.9g}; "
                    f"reward_unique={len({float(item) for item in reward_values})}; "
                    f"advantage_min={float(advantage_values.min().item()):.9g}; "
                    f"advantage_max={float(advantage_values.max().item()):.9g}; "
                    "active_advantage_nonzero="
                    f"{int(active_advantage.count_nonzero().item())}/"
                    f"{active_advantage.numel()}"
                )
            del value, payload, trajectory, rollout, request, recompute_request
            raise error from None
        return StageValue(
            identity=value.identity,
            payload=OptimizedIteration(payload, update),
        )
