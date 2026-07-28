"""TempFlow branch credit and noise weighting preparation."""

from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from visual_rl.core.types import PolicyRecomputeStats, RolloutBatch, StepContext
from visual_rl.optimizers.advantages import AdvantageResult
from visual_rl.optimizers.objective import PolicyObjective
from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm


def _batch(
    *,
    branch_steps: tuple[int, ...] = (1, 2),
    transition_mask: torch.Tensor | None = None,
) -> RolloutBatch:
    batch_size, transitions = 2, 4
    if transition_mask is None:
        transition_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, False]]
        )
    return RolloutBatch(
        prompts=("a", "a"),
        metadata=({}, {}),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=transition_mask,
        sample_id=("s0", "s1"),
        prompt_id=("p0", "p0"),
        group_id=("g0", "g0"),
        branch_id=branch_steps,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=0, seed=3),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=torch.tensor(branch_steps, dtype=torch.int64),
        trajectory_step_index=torch.arange(transitions, dtype=torch.int64),
        transition_std_dev=torch.tensor(
            [[0.2, 0.7, 1.1, 0.4], [0.3, 1.2, 0.5, 0.8]],
            dtype=torch.float64,
        ),
        recompute_payload={},
        artifact_metadata={},
    )


def test_tempflow_prepares_one_branch_credit_and_fixed_noise_scale() -> None:
    batch = _batch()
    advantages = AdvantageResult(
        base_advantage=torch.tensor([8.0, -0.5], dtype=torch.float64),
        metrics={},
    )
    algorithm = TempFlowGRPOAlgorithm(clip_range=0.2, adv_clip_max=5.0)

    assert algorithm.weight_normalization_request(batch, advantages) is None
    inputs = algorithm.prepare_loss_inputs(
        batch,
        advantages,
        normalization_mean=None,
    )

    torch.testing.assert_close(
        inputs.base_advantage,
        torch.tensor(
            [[5.0] * 4, [-0.5] * 4],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        inputs.algorithm_weight,
        batch.transition_std_dev * 2.25,
    )
    assert torch.equal(
        inputs.active_mask,
        torch.tensor(
            [[False, True, False, False], [False, False, True, False]]
        ),
    )
    assert inputs.reference_kl_weight == 0.0
    assert TempFlowGRPOAlgorithm.NOISE_SCALE == 2.25
    assert TempFlowGRPOAlgorithm.TRAINING_CONTRACT_VERSION == 2
    assert {item.name for item in fields(TempFlowGRPOAlgorithm)} == {
        "clip_range",
        "adv_clip_max",
    }
    assert not hasattr(algorithm, "compute_loss")


def test_tempflow_diagnostics_use_active_and_grid_denominators() -> None:
    batch = _batch()
    algorithm = TempFlowGRPOAlgorithm()
    inputs = algorithm.prepare_loss_inputs(
        batch,
        AdvantageResult(
            base_advantage=torch.tensor([1.0, -1.0], dtype=torch.float64),
            metrics={},
        ),
        normalization_mean=None,
    )
    diagnostics = algorithm.diagnostics(batch, inputs)

    assert tuple(diagnostics) == (
        "algorithm/tempflow_noise_weight_mean",
        "algorithm/tempflow_active_timestep_frac",
    )
    noise = diagnostics["algorithm/tempflow_noise_weight_mean"]
    active = diagnostics["algorithm/tempflow_active_timestep_frac"]
    expected_noise_sum = (
        batch.transition_std_dev[0, 1]
        + batch.transition_std_dev[1, 2]
    ) * 2.25
    assert noise.numerator.item() == pytest.approx(
        expected_noise_sum.item()
    )
    assert noise.denominator == 2
    assert active.numerator.item() == 2.0
    assert active.denominator == 8


def test_tempflow_preparation_objective_and_gradient_match_hand_oracle() -> None:
    batch = _batch()
    algorithm = TempFlowGRPOAlgorithm(clip_range=0.2, adv_clip_max=5.0)
    inputs = algorithm.prepare_loss_inputs(
        batch,
        AdvantageResult(
            base_advantage=torch.tensor([8.0, -0.5], dtype=torch.float64),
            metrics={},
        ),
        normalization_mean=None,
    )
    new_log_probs = torch.zeros((2, 4), dtype=torch.float64)
    new_log_probs[0, 1] = math.log(1.1)
    new_log_probs[1, 2] = math.log(0.7)
    new_log_probs.requires_grad_(True)
    objective_batch = batch.replace(
        old_log_probs=batch.old_log_probs.to(dtype=torch.float64)
    )

    output = PolicyObjective()(
        objective_batch,
        inputs,
        PolicyRecomputeStats(new_log_probs=new_log_probs),
    )

    expected_policy = torch.tensor(-4.10625, dtype=torch.float64)
    expected_kl = torch.tensor(
        0.25 * (math.log(1.1) ** 2 + math.log(0.7) ** 2),
        dtype=torch.float64,
    )
    torch.testing.assert_close(output.policy_loss, expected_policy)
    torch.testing.assert_close(output.loss, expected_policy)
    torch.testing.assert_close(
        output.reference_kl,
        torch.tensor(0.0, dtype=torch.float64),
    )
    torch.testing.assert_close(output.approx_kl, expected_kl)
    torch.testing.assert_close(
        output.clipfrac,
        torch.tensor(0.5, dtype=torch.float64),
    )
    assert output.active_transition_count == 2

    output.loss.backward()
    assert new_log_probs.grad is not None
    expected_gradient = torch.zeros_like(new_log_probs)
    expected_gradient[0, 1] = -4.33125
    torch.testing.assert_close(new_log_probs.grad, expected_gradient)
    diagnostics = algorithm.diagnostics(batch, inputs)
    assert diagnostics[
        "algorithm/tempflow_noise_weight_mean"
    ].denominator == 2
    assert diagnostics[
        "algorithm/tempflow_active_timestep_frac"
    ].numerator.item() == 2.0


def test_tempflow_requires_exact_branch_mapping_and_float64_advantage() -> None:
    algorithm = TempFlowGRPOAlgorithm()
    with pytest.raises(ValueError, match="exactly one"):
        algorithm.prepare_loss_inputs(
            _batch(branch_steps=(9, 2)),
            AdvantageResult(
                base_advantage=torch.ones(2, dtype=torch.float64),
                metrics={},
            ),
            normalization_mean=None,
        )
    with pytest.raises(TypeError, match="torch.float64"):
        algorithm.prepare_loss_inputs(
            _batch(),
            AdvantageResult(
                base_advantage=torch.ones(2, dtype=torch.float32),
                metrics={},
            ),
            normalization_mean=None,
        )
