"""P5 canonical advantage, credit, objective, and kernel foundations."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from visual_rl.algorithms.optimization.advantage import (
    AdvantageGrouping,
    GroupZScoreAdvantageProcessor,
    NormalizedAdvantage,
)
from visual_rl.algorithms.optimization.credit import (
    FlashGRPOCreditStrategy,
    GRPOCreditStrategy,
    TempFlowGRPOCreditStrategy,
)
from visual_rl.algorithms.optimization.execution import (
    UpdateDisposition,
    UpdateTransactionResult,
)
from visual_rl.algorithms.optimization.kernel import (
    PolicyUpdateKernel,
    PolicyUpdateResult,
)
from visual_rl.algorithms.optimization.objective import (
    ClippedSurrogateObjective,
    clipped_surrogate,
)
from visual_rl.algorithms.optimization.recompute import PolicyStats
from visual_rl.artifacts.checkpoint import CheckpointSafePoint
from visual_rl.algorithms.rewards import RewardBatchIdentity, RewardResult
from visual_rl.data.samples import (
    BatchRowContext,
    BranchTopology,
    NoConditionBatchState,
    TrajectoryBatch,
    TrajectoryContext,
)


def _contexts(batch_size: int) -> tuple[TrajectoryContext, ...]:
    return tuple(
        TrajectoryContext(
            sample_id=f"sample-{row}",
            trajectory_id=f"trajectory-{row}",
            batch_row=BatchRowContext(
                occurrence_id=f"occurrence-{row // 2}",
                group_id=f"group-{row // 2}",
                member_id=row % 2,
                phase="main",
                optimizer_step=0,
                source_item_id=f"source-{row // 2}",
            ),
        )
        for row in range(batch_size)
    )


def _trajectory(
    kind: str = "full_trajectory",
    *,
    transitions: int = 3,
    dtype: torch.dtype = torch.float32,
    branch_steps: tuple[int, ...] = (1, 1, 0, 0),
    tempflow_paper: bool = False,
) -> TrajectoryBatch:
    batch_size = 4
    contexts = _contexts(batch_size)
    latents = torch.zeros(batch_size, transitions, 1, dtype=dtype)
    transition_index = torch.arange(transitions, dtype=torch.int64).expand(
        batch_size,
        -1,
    )
    kwargs: dict[str, object] = {
        "branch_step_index": None,
        "selected_timestep_index": None,
    }
    if kind == "branching":
        kwargs["exploration_member_index"] = torch.tensor(
            [0, 1, 0, 1], dtype=torch.int64
        )
        if tempflow_paper:
            kwargs["branch_topology"] = BranchTopology.every_policy_timestep(2)
            kwargs["branch_timestep_index"] = transition_index.clone()
            kwargs["transition_terminal_media"] = torch.zeros(
                batch_size, transitions, 1, 1, 1, dtype=dtype
            )
            kwargs["transition_terminal_media_layout"] = "BTCHW"
        else:
            kwargs["branch_topology"] = BranchTopology.single_point_branch_ablation(2)
            kwargs["branch_step_index"] = torch.tensor(
                branch_steps,
                dtype=torch.int64,
            )
            kwargs["shared_prefix_id"] = tuple(
                f"shared-prefix-{row // 2}" for row in range(batch_size)
            )
            kwargs["branch_step_identity"] = tuple(
                f"branch-step-{branch_steps[row]}" for row in range(batch_size)
            )
    elif kind == "single_step":
        if transitions != 1:
            raise ValueError("single-step fixture requires transitions=1")
        kwargs["selected_timestep_index"] = torch.tensor(
            [8, 4, 2, 1],
            dtype=torch.int64,
        )
        kwargs["selection_policy_identity"] = "test.single-step-selection.v1"
        kwargs["selection_mapping_identity"] = "test.single-step-mapping.v1"
    row_string = tuple(str(dtype) for _ in range(transitions))
    return TrajectoryBatch(
        kind=kind,
        contexts=contexts,
        x_t=latents,
        sampled_action=torch.ones_like(latents),
        conditioned_next=torch.ones_like(latents),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        next_timesteps=torch.arange(1, transitions + 1).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions, dtype=dtype),
        transition_mask=torch.ones(batch_size, transitions, dtype=torch.bool),
        transition_index=transition_index,
        likelihood_semantics="exact_env_action",
        condition_identity=tuple(("none",) * transitions for _ in contexts),
        guidance_identity=tuple(("cfg-1",) * transitions for _ in contexts),
        storage_dtype_identity=tuple(row_string for _ in contexts),
        quantization_identity=tuple(("none",) * transitions for _ in contexts),
        media=torch.zeros(batch_size, 1, 1, 1, dtype=dtype),
        media_layout="BCHW",
        condition_state=NoConditionBatchState(batch_size),
        **kwargs,
    )


def _reward(
    trajectory: TrajectoryBatch,
    *,
    values: tuple[float, ...] = (1.0, 3.0, 2.0, 6.0),
    valid: tuple[bool, ...] = (True, True, True, True),
) -> RewardResult:
    contexts = trajectory.contexts
    identity = RewardBatchIdentity(
        source_id="main",
        phase_id="main",
        batch_row_ids=tuple(item.batch_row_identity for item in contexts),
        sample_ids=tuple(item.sample_id for item in contexts),
        trajectory_ids=tuple(item.trajectory_id for item in contexts),
        condition_payload_ids=("none",) * len(contexts),
        group_ids=tuple(item.batch_row.group_id for item in contexts),
    )
    score = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    return RewardResult(
        identity=identity,
        component_scores={"quality": score},
        weighted_scores={"quality": score},
        component_valid_masks={"quality": mask},
        weighted_total=score,
        valid_mask=mask,
        resource_identities={"quality": "quality-model@1"},
    )


def _advantage(
    trajectory: TrajectoryBatch,
    values: tuple[float, ...] = (1.0, -1.0, 2.0, -2.0),
) -> NormalizedAdvantage:
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    return NormalizedAdvantage(
        grouping=grouping,
        values=torch.tensor(
            values,
            dtype=trajectory.old_log_probs.dtype,
            device=trajectory.old_log_probs.device,
        ),
        valid_mask=torch.ones(trajectory.batch_size, dtype=torch.bool),
    )


def _stats(
    trajectory: TrajectoryBatch,
    current_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> PolicyStats:
    if current_log_probs is None:
        current_log_probs = torch.zeros_like(
            trajectory.old_log_probs,
            requires_grad=True,
        )
    return PolicyStats(
        grouping=AdvantageGrouping.from_trajectory(trajectory),
        current_log_probs=current_log_probs,
        **kwargs,
    )


class _ResolvedMeanReducer:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def reduce_mean(
        self,
        values: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert tuple(values.shape) == tuple(active_mask.shape)
        return self.value


def test_group_zscore_is_the_only_normalizer_and_preserves_typed_identity() -> None:
    trajectory = _trajectory()
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    result = GroupZScoreAdvantageProcessor(
        epsilon=1e-12,
        output_dtype="float64",
    ).normalize(_reward(trajectory), grouping)

    assert result.grouping is grouping
    assert result.values.dtype == torch.float64
    assert torch.equal(result.valid_mask, torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(
        result.values,
        torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    assert result.values.grad_fn is None
    result.validate_against_trajectory(trajectory)


def test_group_mean_can_use_upstream_global_batch_standard_deviation() -> None:
    trajectory = _trajectory()
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    result = GroupZScoreAdvantageProcessor(
        epsilon=1.0e-4,
        std_domain="batch",
        output_dtype="float64",
    ).normalize(_reward(trajectory), grouping)

    global_std = np.asarray((1.0, 3.0, 2.0, 6.0), dtype=np.float64).std(ddof=0)
    expected = torch.tensor(
        (-1.0, 1.0, -2.0, 2.0),
        dtype=torch.float64,
    ) / (global_std + 1.0e-4)
    torch.testing.assert_close(result.values, expected, rtol=1.0e-12, atol=1.0e-12)
    assert result.grouping is grouping


def test_group_zscore_rejects_identity_drift_and_too_few_valid_group_rows() -> None:
    trajectory = _trajectory()
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    processor = GroupZScoreAdvantageProcessor()
    drifted = replace(
        grouping,
        trajectory_ids=("other", *grouping.trajectory_ids[1:]),
    )
    with pytest.raises(ValueError, match="identity"):
        processor.normalize(_reward(trajectory), drifted)
    with pytest.raises(ValueError, match="at least two"):
        processor.normalize(
            _reward(
                trajectory,
                valid=(True, False, True, True),
            ),
            grouping,
        )


def test_grpo_credit_only_builds_complete_policy_loss_inputs() -> None:
    trajectory = _trajectory()
    strategy = GRPOCreditStrategy(
        clip_range=0.1,
        advantage_clip=1.5,
        reference_kl_weight=0.2,
    )
    inputs = strategy.plan(
        trajectory=trajectory,
        advantage=_advantage(trajectory, (4.0, -1.0, 0.5, -8.0)),
    )

    torch.testing.assert_close(
        inputs.base_advantage,
        torch.tensor([[1.5] * 3, [-1.0] * 3, [0.5] * 3, [-1.5] * 3]),
    )
    assert torch.equal(inputs.algorithm_weight, torch.ones(4, 3))
    assert torch.equal(inputs.active_mask, trajectory.transition_mask)
    assert inputs.reference_kl_weight == 0.2
    assert not hasattr(strategy, "optimizer")
    assert not hasattr(strategy, "advantage_processor")
    assert not hasattr(strategy, "step")


@pytest.mark.parametrize(
    ("trajectory", "strategy"),
    (
        (_trajectory(), GRPOCreditStrategy()),
        (_trajectory("single_step", transitions=1), FlashGRPOCreditStrategy()),
    ),
)
def test_non_tempflow_credit_rejects_branch_timestep_advantage(
    trajectory: TrajectoryBatch,
    strategy,
) -> None:
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    advantage = NormalizedAdvantage(
        grouping=grouping,
        values=torch.ones(
            trajectory.batch_size,
            trajectory.transition_count,
            dtype=trajectory.old_log_probs.dtype,
        ),
        valid_mask=torch.ones(
            trajectory.batch_size,
            trajectory.transition_count,
            dtype=torch.bool,
        ),
        score_axis_names=("branch_timestep",),
    )
    with pytest.raises(ValueError, match="requires TempFlow paper topology"):
        strategy.plan(
            trajectory=trajectory,
            advantage=advantage,
        )


def test_tempflow_credit_selects_one_branch_step_and_uses_typed_std() -> None:
    trajectory = _trajectory("branching", dtype=torch.float64)
    std = torch.tensor(
        [
            [0.2, 0.7, 1.1],
            [0.3, 1.2, 0.5],
            [0.4, 0.8, 1.4],
            [0.9, 0.6, 0.2],
        ],
        dtype=torch.float64,
    )
    trajectory = replace(trajectory, transition_std_dev=std)
    inputs = TempFlowGRPOCreditStrategy(clip_range=0.2).plan(
        trajectory=trajectory,
        advantage=_advantage(trajectory),
    )

    assert torch.equal(
        inputs.active_mask,
        torch.tensor(
            [
                [False, True, False],
                [False, True, False],
                [True, False, False],
                [True, False, False],
            ]
        ),
    )
    torch.testing.assert_close(inputs.algorithm_weight, std * 2.25)
    assert int(inputs.active_mask.sum()) == trajectory.batch_size


def test_tempflow_paper_credit_consumes_every_timestep_reward_cell() -> None:
    trajectory = _trajectory(
        "branching",
        dtype=torch.float64,
        tempflow_paper=True,
    )
    std = torch.tensor(
        [
            [0.2, 0.7, 1.1],
            [0.3, 1.2, 0.5],
            [0.4, 0.8, 1.4],
            [0.9, 0.6, 0.2],
        ],
        dtype=torch.float64,
    )
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    advantage = NormalizedAdvantage(
        grouping=grouping,
        values=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [4.0, 5.0, 6.0],
                [-4.0, -5.0, -6.0],
            ],
            dtype=torch.float64,
        ),
        valid_mask=torch.ones(4, 3, dtype=torch.bool),
        score_axis_names=("branch_timestep",),
    )

    trajectory = replace(trajectory, transition_std_dev=std)
    inputs = TempFlowGRPOCreditStrategy(clip_range=0.2).plan(
        trajectory=trajectory,
        advantage=advantage,
    )

    assert torch.equal(inputs.active_mask, torch.ones(4, 3, dtype=torch.bool))
    torch.testing.assert_close(
        inputs.base_advantage,
        advantage.values.clamp(-5.0, 5.0),
    )
    torch.testing.assert_close(inputs.algorithm_weight, std * 2.25)
    assert int(inputs.active_mask.sum()) == trajectory.batch_size * 3


def test_tempflow_paper_advantage_normalizes_k_members_per_timestep() -> None:
    trajectory = _trajectory(
        "branching",
        dtype=torch.float64,
        tempflow_paper=True,
    )
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    values = np.array(
        [
            [1.0, 8.0, 3.0],
            [3.0, 4.0, 7.0],
            [10.0, 5.0, -2.0],
            [14.0, 9.0, 2.0],
        ],
        dtype=np.float64,
    )
    valid = np.ones((4, 3), dtype=np.bool_)
    identity = RewardBatchIdentity(
        source_id="main",
        phase_id="main",
        batch_row_ids=grouping.batch_row_ids,
        sample_ids=grouping.sample_ids,
        trajectory_ids=grouping.trajectory_ids,
        condition_payload_ids=("none",) * 4,
        group_ids=grouping.group_ids,
    )
    reward = RewardResult(
        identity=identity,
        component_scores={"quality": values},
        weighted_scores={"quality": values},
        component_valid_masks={"quality": valid},
        weighted_total=values,
        valid_mask=valid,
        resource_identities={"quality": "quality-model@1"},
        score_axis_names=("branch_timestep",),
    )

    advantage = GroupZScoreAdvantageProcessor(
        epsilon=1e-8,
        output_dtype="float64",
    ).normalize(reward, grouping)

    assert advantage.score_axis_names == ("branch_timestep",)
    torch.testing.assert_close(
        advantage.values,
        torch.tensor(
            [
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-7,
    )


def test_tempflow_paper_credit_rejects_incomplete_k_member_slice() -> None:
    full_trajectory = _trajectory(
        "branching",
        dtype=torch.float64,
        tempflow_paper=True,
    )
    full_trajectory = replace(
        full_trajectory,
        transition_std_dev=torch.ones(4, 3, dtype=torch.float64),
    )
    trajectory = full_trajectory.slice((0, 2))
    assert trajectory.branch_group_completeness == "sliced_subset"
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    advantage = NormalizedAdvantage(
        grouping=grouping,
        values=torch.ones(2, 3, dtype=torch.float64),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
        score_axis_names=("branch_timestep",),
    )
    with pytest.raises(ValueError, match="complete K-member groups"):
        TempFlowGRPOCreditStrategy().plan(
            trajectory=trajectory,
            advantage=advantage,
        )


def test_tempflow_paper_trajectory_rejects_false_complete_and_drifted_time_axis() -> (
    None
):
    trajectory = _trajectory("branching", tempflow_paper=True)
    one_member = trajectory.slice((0,))
    with pytest.raises(ValueError, match="exactly K exploration members"):
        replace(one_member, branch_group_completeness="complete")

    drifted = trajectory.transition_index.clone()
    drifted[1, 1] = 2
    with pytest.raises(ValueError, match="transition-index axis"):
        replace(
            trajectory,
            transition_index=drifted,
            branch_timestep_index=drifted.clone(),
        )


def test_flash_credit_uses_resolved_global_mean_without_owning_reduction() -> None:
    trajectory = _trajectory("single_step", transitions=1)
    coefficient = torch.tensor([[1.0], [3.0], [2.0], [2.0]])
    mean = torch.tensor(2.0)
    trajectory = replace(trajectory, rectification_coefficient=coefficient)
    inputs = FlashGRPOCreditStrategy(clip_range=0.2).plan(
        trajectory=trajectory,
        advantage=_advantage(trajectory),
        coefficient_mean_reducer=_ResolvedMeanReducer(mean),
    )

    torch.testing.assert_close(
        inputs.algorithm_weight,
        torch.tensor([[0.5], [1.5], [1.0], [1.0]]),
    )
    assert torch.equal(inputs.active_mask, torch.ones(4, 1, dtype=torch.bool))


def test_new_objective_is_numerically_identical_to_canonical_clipped_math() -> None:
    trajectory = _trajectory()
    current = torch.tensor(
        [
            [0.0, 0.1, -0.2],
            [0.2, 0.0, -0.1],
            [0.4, 0.0, 0.2],
            [-0.3, 0.1, 0.0],
        ],
        requires_grad=True,
    )
    stats = _stats(trajectory, current)
    inputs = GRPOCreditStrategy(clip_range=0.1).plan(
        trajectory=trajectory,
        advantage=_advantage(trajectory),
    )

    output = ClippedSurrogateObjective().compute(
        old_log_probs=trajectory.old_log_probs,
        policy_stats=stats,
        loss_inputs=inputs,
    )
    canonical = clipped_surrogate(
        old_log_probs=trajectory.old_log_probs,
        new_log_probs=current,
        inputs=inputs,
    )
    torch.testing.assert_close(output.policy_loss, canonical.policy_loss)
    torch.testing.assert_close(output.approx_kl, canonical.approx_kl)
    torch.testing.assert_close(output.clipfrac, canonical.clipfrac)
    torch.testing.assert_close(output.loss, canonical.policy_loss)
    assert output.active_transition_count == canonical.active_transition_count


def test_flow_grpo_reference_stats_are_consumed_only_by_shared_objective() -> None:
    trajectory = _trajectory()
    current_log_probs = torch.zeros_like(
        trajectory.old_log_probs,
        requires_grad=True,
    )
    current_mean = torch.zeros(4, 3, 1, requires_grad=True)
    stats = _stats(
        trajectory,
        current_log_probs,
        current_transition_mean=current_mean,
        reference_transition_mean=torch.ones_like(current_mean).detach(),
        transition_std=torch.ones_like(current_mean).detach(),
    )
    inputs = GRPOCreditStrategy(reference_kl_weight=0.2).plan(
        trajectory=trajectory,
        advantage=_advantage(trajectory),
    )

    output = ClippedSurrogateObjective().compute(
        old_log_probs=trajectory.old_log_probs,
        policy_stats=stats,
        loss_inputs=inputs,
    )
    torch.testing.assert_close(output.reference_kl, torch.tensor(0.5))
    torch.testing.assert_close(
        output.loss,
        output.policy_loss + 0.2 * output.reference_kl,
    )


def test_committed_result_remains_the_checkpoint_safe_point_evidence() -> None:
    transaction = UpdateTransactionResult(
        optimizer_step=0,
        disposition=UpdateDisposition.COMMITTED,
        payload={"slot_id": "slot"},
        gradient_norm_pre_clip=0.5,
        gradient_norm_post_clip=0.25,
        trace=("accumulate", "objective", "backward", "optimizer"),
    )
    result = PolicyUpdateResult(
        optimizer_step=0,
        loss=1.0,
        policy_loss=1.0,
        reference_kl=0.0,
        approx_kl=0.0,
        clipfrac=0.0,
        active_transition_count=12,
        logprob_delta_max=0.0,
        gradient_norm_pre_clip=0.5,
        gradient_norm_post_clip=0.25,
        transaction=transaction,
    )
    safe_point = CheckpointSafePoint.from_policy_update_result(
        rank=0,
        world_size=1,
        policy_update_result=result,
        group_geometry_id="7" * 64,
    )
    assert safe_point.update_disposition == "committed"
    assert safe_point.committed_optimizer_step == 1
    with pytest.raises(ValueError, match="optimizer_step does not match"):
        replace(
            result,
            transaction=replace(transaction, optimizer_step=1),
        )
    with pytest.raises(TypeError, match="PolicyUpdateResult"):
        CheckpointSafePoint.from_policy_update_result(
            rank=0,
            world_size=1,
            policy_update_result={"transaction": transaction},
            group_geometry_id="7" * 64,
        )


def test_kernel_exposes_only_streaming_and_requires_runtime_injection() -> None:
    kernel = PolicyUpdateKernel()
    assert set(vars(kernel)) == {
        "max_initial_logprob_delta",
        "require_initial_clipfrac_zero",
        "require_finite_gradients",
        "require_nonzero_gradients",
        "max_grad_norm",
    }
    assert not hasattr(PolicyUpdateKernel, "step")
    parameters = inspect.signature(PolicyUpdateKernel.step_slots).parameters
    assert "loss_inputs" in parameters
    assert parameters["accelerator"].default is inspect.Parameter.empty
    assert parameters["prepared_root"].default is inspect.Parameter.empty
    assert "algorithm" not in parameters
    assert "credit" not in parameters
    assert "advantage" not in parameters
    assert "reward" not in parameters


def test_update_kernel_owner_is_canonical_and_training_namespace_is_absent() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "visual_rl" / "training").exists()
    assert PolicyUpdateKernel.__module__ == "visual_rl.algorithms.optimization.kernel"
    assert PolicyUpdateResult.__module__ == "visual_rl.algorithms.optimization.kernel"
