"""Non-TempFlow credit must reject temporal reward axes explicitly."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from visual_rl.algorithms import (
    CreditStrategy,
    FlashGRPOCreditStrategy,
    GRPOCreditStrategy,
)
from visual_rl.algorithms.optimization.advantage import (
    AdvantageGrouping,
    NormalizedAdvantage,
)
from visual_rl.data.samples import (
    BatchRowContext,
    BranchTopology,
    NoConditionBatchState,
    TrajectoryBatch,
    TrajectoryContext,
)


def _tempflow_inputs() -> tuple[
    TrajectoryBatch,
    NormalizedAdvantage,
]:
    batch_size, transition_count = 2, 2
    rows = tuple(
        BatchRowContext(
            occurrence_id="occurrence-0",
            group_id="group-0",
            member_id=row_index,
            phase="main",
            optimizer_step=0,
            source_item_id="source-item-0",
        )
        for row_index in range(batch_size)
    )
    contexts = tuple(
        TrajectoryContext(
            sample_id=f"sample-{row_index}",
            trajectory_id=f"trajectory-{row_index}",
            batch_row=rows[row_index],
        )
        for row_index in range(batch_size)
    )
    transition_index = torch.arange(
        transition_count,
        dtype=torch.int64,
    ).expand(batch_size, -1)
    latents = torch.zeros(batch_size, transition_count, 1)
    trajectory = TrajectoryBatch(
        kind="branching",
        contexts=contexts,
        x_t=latents,
        sampled_action=torch.ones_like(latents),
        conditioned_next=torch.ones_like(latents),
        timesteps=transition_index.float(),
        next_timesteps=transition_index.float() + 1.0,
        old_log_probs=torch.zeros(batch_size, transition_count),
        transition_mask=torch.ones(
            batch_size,
            transition_count,
            dtype=torch.bool,
        ),
        transition_index=transition_index,
        likelihood_semantics="exact_env_action",
        condition_identity=(("none",) * transition_count,) * batch_size,
        guidance_identity=(("cfg",) * transition_count,) * batch_size,
        storage_dtype_identity=(("torch.float32",) * transition_count,) * batch_size,
        quantization_identity=(("none",) * transition_count,) * batch_size,
        media=torch.zeros(batch_size, 1, 1, 1),
        media_layout="BCHW",
        condition_state=NoConditionBatchState(batch_size),
        branch_topology=BranchTopology.every_policy_timestep(batch_size),
        exploration_member_index=torch.arange(batch_size, dtype=torch.int64),
        branch_timestep_index=transition_index.clone(),
        transition_terminal_media=torch.zeros(
            batch_size,
            transition_count,
            1,
            1,
            1,
        ),
        transition_terminal_media_layout="BTCHW",
    )
    grouping = AdvantageGrouping.from_trajectory(trajectory)
    advantage = NormalizedAdvantage(
        grouping=grouping,
        values=torch.ones(batch_size, transition_count),
        valid_mask=torch.ones(
            batch_size,
            transition_count,
            dtype=torch.bool,
        ),
        score_axis_names=("branch_timestep",),
    )
    return trajectory, advantage


@pytest.mark.parametrize(
    ("strategy_factory", "message"),
    (
        (
            lambda: GRPOCreditStrategy(reference_kl_weight=0.1),
            "GRPO credit requires row-only advantage",
        ),
        (
            FlashGRPOCreditStrategy,
            "Flash credit requires row-only advantage",
        ),
    ),
    ids=("flow_grpo", "flash_grpo"),
)
def test_non_tempflow_credit_rejects_branch_timestep_advantage(
    strategy_factory: Callable[[], CreditStrategy],
    message: str,
) -> None:
    trajectory, advantage = _tempflow_inputs()

    strategy = strategy_factory()
    with pytest.raises(ValueError, match=message):
        strategy.plan(
            trajectory=trajectory,
            advantage=advantage,
        )
