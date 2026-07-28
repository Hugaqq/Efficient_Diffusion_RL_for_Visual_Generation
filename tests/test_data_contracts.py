"""Selected-timestep data stays on the sole RolloutBatch contract."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from visual_rl.core.types import RolloutRequest, StepContext
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter


def _request() -> RolloutRequest:
    return RolloutRequest(
        prompts=("red", "blue"),
        metadata=({}, {}),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-1"),
        group_id=("group-0", "group-1"),
        branch_id=None,
        context=StepContext(step=0, seed=9),
        kind="single_step",
        num_steps=4,
        group_size=1,
        selected_timestep_index=(1, 3),
    )


def test_single_step_batch_carries_only_typed_flash_fields() -> None:
    adapter = TinyDiffusionAdapter(image_size=4, device="cpu")
    request = _request()
    batch = adapter.sample(request)
    batch.validate_against(request)

    assert batch.transition_count == 1
    assert torch.equal(
        batch.selected_timestep_index,
        torch.tensor([1, 3], dtype=torch.int64),
    )
    torch.testing.assert_close(
        batch.flash_coefficient,
        torch.ones(2, 1),
    )
    assert "flash_coefficient" not in batch.recompute_payload
    assert not hasattr(batch, "rectification_mode")
    assert not hasattr(batch, "flash_rectification_weights")


def test_rollout_side_rectification_module_is_physically_absent() -> None:
    assert importlib.util.find_spec("visual_rl.rollout.rectification") is None


def test_rollout_batch_rejects_untyped_flash_shapes() -> None:
    adapter = TinyDiffusionAdapter(image_size=4, device="cpu")
    batch = adapter.sample(_request())
    with pytest.raises(ValueError, match="flash_coefficient must have shape"):
        batch.replace(flash_coefficient=torch.ones(2))
