"""Typed advantage and loss-input contracts."""

from __future__ import annotations

import pytest
import torch

from visual_rl.core.types import (
    RewardBatch,
    RolloutBatch,
    StepContext,
)
from visual_rl.optimizers.advantages import (
    AdvantageComputer,
    AdvantageResult,
)
from visual_rl.optimizers.objective import PolicyLossInputs


def _batch(*, transition_count: int = 2) -> RolloutBatch:
    batch_size = 4
    return RolloutBatch(
        prompts=tuple(f"prompt-{index // 2}" for index in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transition_count, 1),
        next_latents=torch.ones(batch_size, transition_count, 1),
        timesteps=torch.arange(transition_count).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transition_count),
        transition_mask=torch.ones(
            batch_size,
            transition_count,
            dtype=torch.bool,
        ),
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=("prompt-a", "prompt-a", "prompt-b", "prompt-b"),
        group_id=("group-a", "group-a", "group-b", "group-b"),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=1, seed=7),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={},
        artifact_metadata={},
    )


def _rewards(batch: RolloutBatch, values: torch.Tensor) -> RewardBatch:
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": tuple({} for _ in batch.sample_id)},
    )


def test_advantage_computer_emits_only_detached_base_and_contributions() -> None:
    batch = _batch()
    values = torch.tensor([1.0, 3.0, 5.0, 5.0], dtype=torch.float32)
    computer = AdvantageComputer(
        epsilon=1e-8,
        output_dtype="float32",
    )
    result = computer(batch, _rewards(batch, values))

    torch.testing.assert_close(
        result.base_advantage,
        torch.tensor([-1.0, 1.0, 0.0, 0.0]),
    )
    assert result.base_advantage.dtype == torch.float32
    assert not result.base_advantage.requires_grad
    assert not hasattr(result, "advantages")
    assert not hasattr(computer, "state_dict")
    assert not hasattr(computer, "load_state_dict")
    assert tuple(result.metrics) == (
        "advantage/zero_std_ratio",
        "advantage/group_size_mean",
        "advantage/trained_prompt_num",
    )
    zero_std = result.metrics["advantage/zero_std_ratio"]
    group_size = result.metrics["advantage/group_size_mean"]
    trained = result.metrics["advantage/trained_prompt_num"]
    assert zero_std.numerator.item() == 1.0
    assert zero_std.denominator == 2
    assert group_size.numerator.item() == 4.0
    assert group_size.denominator == 2
    assert trained.numerator.item() == 2.0
    assert trained.denominator is None


def test_advantage_computer_requires_complete_non_singleton_groups() -> None:
    batch = _batch().replace(
        group_id=("group-a", "group-a", "group-b", "group-c")
    )
    with pytest.raises(ValueError, match="at least two rows"):
        AdvantageComputer(
            epsilon=1e-8,
            output_dtype="float64",
        )(batch, _rewards(batch, torch.ones(4)))


def test_policy_loss_inputs_slice_keeps_one_detached_tensor_contract() -> None:
    batch = _batch()
    inputs = PolicyLossInputs(
        base_advantage=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        algorithm_weight=torch.ones(4, 2),
        active_mask=torch.tensor(
            [[True, False], [True, True], [False, True], [True, True]]
        ),
        clip_range=0.2,
        reference_kl_weight=0.3,
    )
    inputs.validate_against(batch)
    sliced = inputs.slice([3, 1])

    torch.testing.assert_close(
        sliced.base_advantage,
        inputs.base_advantage[[3, 1]],
    )
    torch.testing.assert_close(
        sliced.algorithm_weight,
        inputs.algorithm_weight[[3, 1]],
    )
    assert torch.equal(sliced.active_mask, inputs.active_mask[[3, 1]])
    assert sliced.clip_range == 0.2
    assert sliced.reference_kl_weight == 0.3
    with pytest.raises(ValueError, match="duplicates"):
        inputs.slice([1, 1])


def test_policy_loss_inputs_transition_slice_keeps_the_same_objective_contract() -> None:
    inputs = PolicyLossInputs(
        base_advantage=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        algorithm_weight=torch.ones(4, 3),
        active_mask=torch.tensor(
            [
                [True, False, True],
                [True, True, False],
                [False, True, True],
                [True, True, True],
            ]
        ),
        clip_range=0.2,
        reference_kl_weight=0.3,
    )

    sliced = inputs.slice_transitions(1, 3)

    torch.testing.assert_close(
        sliced.base_advantage,
        inputs.base_advantage[:, 1:3],
    )
    torch.testing.assert_close(
        sliced.algorithm_weight,
        inputs.algorithm_weight[:, 1:3],
    )
    assert torch.equal(sliced.active_mask, inputs.active_mask[:, 1:3])
    assert sliced.clip_range == inputs.clip_range
    assert sliced.reference_kl_weight == inputs.reference_kl_weight
    assert inputs.slice_transitions(0, 3) is inputs
    with pytest.raises(IndexError, match="transition interval"):
        inputs.slice_transitions(1, 1)
    with pytest.raises(TypeError, match="must be an integer"):
        inputs.slice_transitions(True, 2)


def test_policy_loss_inputs_rejects_grad_and_nonpositive_active_weight() -> None:
    with pytest.raises(ValueError, match="detached"):
        PolicyLossInputs(
            base_advantage=torch.ones(2, 1, requires_grad=True),
            algorithm_weight=torch.ones(2, 1),
            active_mask=torch.ones(2, 1, dtype=torch.bool),
            clip_range=0.2,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        PolicyLossInputs(
            base_advantage=torch.ones(2, 1),
            algorithm_weight=torch.tensor([[1.0], [0.0]]),
            active_mask=torch.ones(2, 1, dtype=torch.bool),
            clip_range=0.2,
        )


def test_advantage_result_rejects_unowned_metric_namespace() -> None:
    with pytest.raises(ValueError, match="advantage/ namespace"):
        AdvantageResult(
            base_advantage=torch.ones(2),
            metrics={"zero_std_ratio": object()},
        )
