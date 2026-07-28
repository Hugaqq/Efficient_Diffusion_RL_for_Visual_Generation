"""GRPO typed preparation for the shared objective."""

from __future__ import annotations

from dataclasses import fields

import torch

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.optimizers.advantages import AdvantageResult
from visual_rl.optimizers.grpo import GRPOAlgorithm


def _batch() -> RolloutBatch:
    return RolloutBatch(
        prompts=("a", "a", "b", "b"),
        metadata=({}, {}, {}, {}),
        media=torch.zeros(4, 1, 1, 1),
        latents=torch.zeros(4, 3, 1),
        next_latents=torch.ones(4, 3, 1),
        timesteps=torch.arange(3).expand(4, -1),
        old_log_probs=torch.zeros(4, 3),
        transition_mask=torch.tensor(
            [
                [True, True, False],
                [True, False, False],
                [False, True, True],
                [True, True, True],
            ]
        ),
        sample_id=("s0", "s1", "s2", "s3"),
        prompt_id=("p0", "p0", "p1", "p1"),
        group_id=("g0", "g0", "g1", "g1"),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=0, seed=3),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={},
        artifact_metadata={},
    )


def test_grpo_prepares_clamped_base_unit_weight_and_reference_beta() -> None:
    batch = _batch()
    advantages = AdvantageResult(
        base_advantage=torch.tensor([-8.0, -1.0, 0.0, 9.0]),
        metrics={},
    )
    algorithm = GRPOAlgorithm(
        clip_range=0.2,
        adv_clip_max=5.0,
        beta=0.3,
    )

    assert algorithm.weight_normalization_request(batch, advantages) is None
    inputs = algorithm.prepare_loss_inputs(
        batch,
        advantages,
        normalization_mean=None,
    )

    torch.testing.assert_close(
        inputs.base_advantage,
        torch.tensor(
            [
                [-5.0, -5.0, -5.0],
                [-1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs.algorithm_weight,
        torch.ones_like(inputs.algorithm_weight),
    )
    assert torch.equal(inputs.active_mask, batch.transition_mask)
    assert inputs.reference_kl_weight == 0.3
    assert algorithm.diagnostics(batch, inputs) == {}
    assert algorithm.TRAINING_CONTRACT_VERSION == 2
    assert {item.name for item in fields(GRPOAlgorithm)} == {
        "clip_range",
        "adv_clip_max",
        "beta",
    }
    assert not hasattr(algorithm, "compute_loss")


def test_grpo_reference_capability_depends_only_on_beta() -> None:
    assert GRPOAlgorithm.required_capabilities({"beta": 0.0}) == frozenset()
    assert GRPOAlgorithm.required_capabilities(
        {"beta": 0.1}
    ) == frozenset({"policy.reference_stats"})
