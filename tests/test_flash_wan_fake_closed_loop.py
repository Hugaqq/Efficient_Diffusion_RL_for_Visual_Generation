"""Flash preparation consumes Adapter-owned typed coefficients exactly once."""

from __future__ import annotations

import torch

from visual_rl.core.types import (
    RewardBatch,
    RolloutRequest,
    StepContext,
)
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
from visual_rl.optimizers.advantages import AdvantageComputer
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm


def test_fake_single_step_closes_into_the_shared_loss_inputs() -> None:
    adapter = TinyDiffusionAdapter(image_size=4, device="cpu")
    request = RolloutRequest(
        prompts=("red", "red"),
        metadata=({}, {}),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-0"),
        group_id=("group-0", "group-0"),
        branch_id=None,
        context=StepContext(step=0, seed=19),
        kind="single_step",
        num_steps=4,
        group_size=2,
        selected_timestep_index=(1, 3),
    )
    batch = adapter.sample(request).replace(
        flash_coefficient=torch.tensor([[2.0], [4.0]])
    )
    values = torch.tensor([1.0, 3.0], dtype=torch.float32)
    rewards = RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(2, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": ({}, {})},
    )
    advantages = AdvantageComputer(
        epsilon=1e-8,
        output_dtype="float32",
    )(batch, rewards)
    algorithm = FlashGRPOAlgorithm(clip_range=0.2, adv_clip_max=5.0)

    local_mean, count = algorithm.weight_normalization_request(
        batch,
        advantages,
    )
    inputs = algorithm.prepare_loss_inputs(
        batch,
        advantages,
        normalization_mean=local_mean,
    )

    assert count == 2
    torch.testing.assert_close(local_mean, torch.tensor(3.0))
    torch.testing.assert_close(
        inputs.algorithm_weight,
        torch.tensor([[2.0 / 3.0], [4.0 / 3.0]]),
    )
    assert inputs.active_mask.equal(batch.transition_mask)
    assert batch.flash_coefficient.equal(torch.tensor([[2.0], [4.0]]))
    assert not hasattr(algorithm, "compute_loss")
