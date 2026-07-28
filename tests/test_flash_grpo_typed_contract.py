"""Focused characterization of Flash-GRPO's final RolloutBatch inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm


def _batch(
    *,
    old_log_probs: torch.Tensor,
    transition_mask: torch.Tensor,
    selected_timestep_index: torch.Tensor | None,
    flash_coefficient: torch.Tensor | None,
) -> RolloutBatch:
    batch_size, transitions = old_log_probs.shape
    return RolloutBatch(
        prompts=tuple(f"prompt-{index}" for index in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=old_log_probs.detach(),
        transition_mask=transition_mask,
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=tuple(f"prompt-{index}" for index in range(batch_size)),
        group_id=tuple(f"group-{index}" for index in range(batch_size)),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=2, seed=17),
        selected_timestep_index=selected_timestep_index,
        flash_coefficient=flash_coefficient,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={},
        artifact_metadata={},
    )


def _manual_loss(
    *,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    transition_mask: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    safe_delta = torch.where(
        transition_mask,
        new_log_probs - old_log_probs,
        torch.zeros_like(new_log_probs),
    )
    ratio = torch.exp(safe_delta)
    effective_advantages = advantages * weights
    unclipped = -effective_advantages * ratio
    clipped = -effective_advantages * ratio.clamp(
        1.0 - clip_range,
        1.0 + clip_range,
    )
    return torch.maximum(unclipped, clipped).masked_select(
        transition_mask
    ).mean()


def test_reference_formula_consumes_typed_flash_fields_without_mutating_raw_coefficient():
    old = torch.tensor([[-0.2], [0.1], [0.0]], dtype=torch.float32)
    mask = torch.tensor([[True], [False], [True]])
    coefficient = torch.tensor([[1.0], [2.0], [4.0]])
    selected = torch.tensor([0, 2, 4], dtype=torch.int64)
    batch = _batch(
        old_log_probs=old,
        transition_mask=mask,
        selected_timestep_index=selected,
        flash_coefficient=coefficient,
    )
    algorithm = FlashGRPOAlgorithm(
        objective_version="reference_v1",
        clip_range=0.1,
        adv_clip_max=5.0,
        beta=0.0,
    )
    advantages = torch.tensor([1.0, -2.0, 3.0])
    prepared = algorithm.prepare_batch(batch, advantages)

    expected_weights = coefficient / coefficient.mean()
    torch.testing.assert_close(prepared.flash_coefficient, coefficient)
    torch.testing.assert_close(
        prepared.recompute_payload[
            FlashGRPOAlgorithm._PREPARED_RECTIFICATION_KEY
        ],
        expected_weights,
    )

    actual_new = torch.tensor(
        [[-0.1], [0.5], [-0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    actual_loss, metrics = algorithm.compute_loss(
        prepared,
        advantages,
        actual_new,
    )
    expected_new = actual_new.detach().clone().requires_grad_(True)
    expected_loss = _manual_loss(
        old_log_probs=old,
        new_log_probs=expected_new,
        advantages=advantages[:, None],
        weights=expected_weights,
        transition_mask=mask,
        clip_range=algorithm.clip_range,
    )
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(
        torch.autograd.grad(actual_loss, actual_new)[0],
        torch.autograd.grad(expected_loss, expected_new)[0],
    )
    torch.testing.assert_close(
        metrics["flash_rectification_weight_mean"],
        expected_weights.masked_select(mask).mean(),
    )
    torch.testing.assert_close(
        metrics["flash_selected_timestep_mean"],
        selected.float().mean(),
    )


def test_global_normalization_uses_raw_typed_coefficient_after_local_preparation():
    coefficient = torch.tensor([[1.0], [3.0]])
    batch = _batch(
        old_log_probs=torch.zeros(2, 1),
        transition_mask=torch.ones(2, 1, dtype=torch.bool),
        selected_timestep_index=torch.tensor([0, 1], dtype=torch.int64),
        flash_coefficient=coefficient,
    )
    algorithm = FlashGRPOAlgorithm(objective_version="reference_v1")
    advantages = torch.ones(2)
    locally_prepared = algorithm.prepare_batch(batch, advantages)

    local_mean, count = algorithm.global_batch_reduction(
        locally_prepared,
        advantages,
    )
    torch.testing.assert_close(local_mean, torch.tensor(2.0))
    assert count == 2

    globally_prepared = algorithm.apply_global_batch_reduction(
        locally_prepared,
        advantages,
        torch.tensor(2.5),
    )
    torch.testing.assert_close(globally_prepared.flash_coefficient, coefficient)
    torch.testing.assert_close(
        globally_prepared.recompute_payload[
            FlashGRPOAlgorithm._PREPARED_RECTIFICATION_KEY
        ],
        coefficient / 2.5,
    )


def test_legacy_characterization_keeps_the_existing_clipped_surrogate_formula():
    old = torch.tensor(
        [[-0.2, 0.1], [0.0, -0.3]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False], [True, True]])
    batch = _batch(
        old_log_probs=old,
        transition_mask=mask,
        selected_timestep_index=None,
        flash_coefficient=None,
    )
    algorithm = FlashGRPOAlgorithm(
        objective_version="legacy_v0",
        clip_range=0.2,
        adv_clip_max=5.0,
        beta=0.0,
        rectification=None,
    )
    advantages = torch.tensor([1.5, -0.5])
    prepared = algorithm.prepare_batch(batch, advantages)
    new = torch.tensor(
        [[0.0, 0.2], [-0.1, -0.4]],
        requires_grad=True,
    )
    actual, _metrics = algorithm.compute_loss(prepared, advantages, new)
    expected = _manual_loss(
        old_log_probs=old,
        new_log_probs=new,
        advantages=advantages[:, None].expand_as(new),
        weights=torch.ones_like(new),
        transition_mask=mask,
        clip_range=algorithm.clip_range,
    )
    torch.testing.assert_close(actual, expected)


def test_reference_formula_requires_typed_flash_coefficient():
    batch = _batch(
        old_log_probs=torch.zeros(2, 1),
        transition_mask=torch.ones(2, 1, dtype=torch.bool),
        selected_timestep_index=torch.tensor([0, 1], dtype=torch.int64),
        flash_coefficient=None,
    )
    algorithm = FlashGRPOAlgorithm(objective_version="reference_v1")
    with pytest.raises(ValueError, match="batch.flash_coefficient"):
        algorithm.prepare_batch(batch, torch.ones(2))


def test_flash_algorithm_source_contains_no_retired_batch_aliases():
    source = (
        Path(__file__).parents[1]
        / "visual_rl"
        / "optimizers"
        / "flash_grpo.py"
    ).read_text(encoding="utf-8")
    for retired in (
        "model_tensors",
        "model_metadata",
        "batch.kl",
        "selected_timestep_indices",
        "flash_rectification_weights",
    ):
        assert retired not in source
