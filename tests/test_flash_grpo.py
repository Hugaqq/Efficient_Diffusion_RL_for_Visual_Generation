"""Flash-GRPO global coefficient preparation and diagnostics."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from visual_rl.core.types import PolicyRecomputeStats, RolloutBatch, StepContext
from visual_rl.optimizers.advantages import AdvantageResult
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.optimizers.objective import PolicyObjective


def _batch(
    *,
    transitions: int = 1,
    coefficient: torch.Tensor | None = None,
) -> RolloutBatch:
    batch_size = 3
    return RolloutBatch(
        prompts=("a", "a", "a"),
        metadata=({}, {}, {}),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(
            batch_size,
            transitions,
            dtype=torch.bool,
        ),
        sample_id=("s0", "s1", "s2"),
        prompt_id=("p0", "p0", "p0"),
        group_id=("g0", "g0", "g0"),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=0, seed=3),
        selected_timestep_index=torch.tensor([8, 4, 2]),
        flash_coefficient=(
            torch.tensor([[1.0], [3.0], [8.0]])
            if coefficient is None
            else coefficient
        ),
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={},
        artifact_metadata={},
    )


def test_flash_request_and_inputs_use_external_global_mean_once() -> None:
    batch = _batch()
    advantages = AdvantageResult(
        base_advantage=torch.tensor([2.0, 0.0, -9.0]),
        metrics={},
    )
    algorithm = FlashGRPOAlgorithm(clip_range=0.1, adv_clip_max=5.0)

    local_mean, sample_count = algorithm.weight_normalization_request(
        batch,
        advantages,
    )
    torch.testing.assert_close(local_mean, torch.tensor(4.0))
    assert sample_count == 3
    inputs = algorithm.prepare_loss_inputs(
        batch,
        advantages,
        normalization_mean=torch.tensor(2.0),
    )

    torch.testing.assert_close(
        inputs.base_advantage,
        torch.tensor([[2.0], [0.0], [-5.0]]),
    )
    torch.testing.assert_close(
        inputs.algorithm_weight,
        torch.tensor([[0.5], [1.5], [4.0]]),
    )
    assert torch.equal(inputs.active_mask, batch.transition_mask)
    assert inputs.reference_kl_weight == 0.0
    assert algorithm.TRAINING_CONTRACT_VERSION == 2
    assert {item.name for item in fields(FlashGRPOAlgorithm)} == {
        "clip_range",
        "adv_clip_max",
    }
    assert not hasattr(algorithm, "compute_loss")


def test_flash_diagnostics_have_explicit_numerators_and_denominators() -> None:
    batch = _batch()
    algorithm = FlashGRPOAlgorithm()
    inputs = algorithm.prepare_loss_inputs(
        batch,
        AdvantageResult(
            base_advantage=torch.tensor([2.0, 0.0, -1.0]),
            metrics={},
        ),
        normalization_mean=torch.tensor(4.0),
    )
    diagnostics = algorithm.diagnostics(batch, inputs)

    assert tuple(diagnostics) == (
        "algorithm/flash_rectification_weight_mean",
        "algorithm/flash_selected_timestep_mean",
        "algorithm/flash_active_timestep_frac",
    )
    weight = diagnostics["algorithm/flash_rectification_weight_mean"]
    selected = diagnostics["algorithm/flash_selected_timestep_mean"]
    active = diagnostics["algorithm/flash_active_timestep_frac"]
    assert weight.numerator.item() == 3.0
    assert weight.denominator == 3
    assert selected.numerator.item() == 14.0
    assert selected.denominator == 3
    assert active.numerator.item() == 2.0
    assert active.denominator == 3


def test_flash_preparation_objective_and_gradient_match_hand_oracle() -> None:
    batch = _batch()
    algorithm = FlashGRPOAlgorithm(clip_range=0.1, adv_clip_max=5.0)
    inputs = algorithm.prepare_loss_inputs(
        batch,
        AdvantageResult(
            base_advantage=torch.tensor([2.0, 0.0, -9.0]),
            metrics={},
        ),
        normalization_mean=torch.tensor(2.0),
    )
    ratios = torch.tensor([[1.05], [0.8], [1.4]], dtype=torch.float32)
    new_log_probs = ratios.log().detach().requires_grad_(True)

    output = PolicyObjective()(
        batch,
        inputs,
        PolicyRecomputeStats(new_log_probs=new_log_probs),
    )

    expected_policy = torch.tensor((-1.05 + 0.0 + 28.0) / 3.0)
    expected_kl = 0.5 * ratios.log().square().mean()
    torch.testing.assert_close(output.policy_loss, expected_policy)
    torch.testing.assert_close(output.loss, expected_policy)
    torch.testing.assert_close(output.reference_kl, torch.tensor(0.0))
    torch.testing.assert_close(output.approx_kl, expected_kl)
    torch.testing.assert_close(output.clipfrac, torch.tensor(2.0 / 3.0))
    assert output.active_transition_count == 3

    output.loss.backward()
    assert new_log_probs.grad is not None
    torch.testing.assert_close(
        new_log_probs.grad,
        torch.tensor([[-1.05 / 3.0], [0.0], [28.0 / 3.0]]),
    )
    diagnostics = algorithm.diagnostics(batch, inputs)
    assert diagnostics[
        "algorithm/flash_rectification_weight_mean"
    ].numerator.item() == pytest.approx(6.0)
    assert diagnostics[
        "algorithm/flash_rectification_weight_mean"
    ].denominator == 3


def test_flash_rejects_non_single_step_or_missing_global_mean() -> None:
    algorithm = FlashGRPOAlgorithm()
    advantages = AdvantageResult(
        base_advantage=torch.ones(3),
        metrics={},
    )
    with pytest.raises(ValueError, match="physical T=1"):
        algorithm.weight_normalization_request(
            _batch(
                transitions=2,
                coefficient=torch.ones(3, 1),
            ),
            advantages,
        )
    with pytest.raises(ValueError, match="global coefficient mean"):
        algorithm.prepare_loss_inputs(
            _batch(),
            advantages,
            normalization_mean=None,
        )
