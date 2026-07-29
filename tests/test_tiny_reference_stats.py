"""Reference-statistics contract for the single Tiny model component."""

from __future__ import annotations

import torch

from visual_rl.core.types import RolloutRequest, StepContext
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter


def _request() -> RolloutRequest:
    return RolloutRequest(
        prompts=("red cube", "red cube"),
        metadata=({"source": "unit"}, {"source": "unit"}),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-0"),
        group_id=("group-0", "group-0"),
        branch_id=None,
        context=StepContext(step=0, seed=23),
        kind="full_trajectory",
        num_steps=2,
        group_size=2,
    )


def test_tiny_beta_zero_path_never_runs_reference_forward(monkeypatch):
    adapter = TinyDiffusionAdapter(image_size=4, device="cpu")
    batch = adapter.sample(_request())
    calls = 0
    original = adapter._reference_transition_mean

    def record_reference(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        adapter,
        "_reference_transition_mean",
        record_reference,
    )
    stats = adapter.recompute_policy_stats(
        batch,
        require_reference=False,
    )
    stats.validate_against(batch, require_reference=False)
    assert calls == 0
    assert stats.current_transition_mean is None
    assert stats.transition_std is None
    assert stats.reference_transition_mean is None


def test_tiny_reference_is_frozen_base_and_delta_changes_kl_gradient():
    adapter = TinyDiffusionAdapter(image_size=4, device="cpu")
    with torch.no_grad():
        adapter.base_color_bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
        adapter.color_bias.copy_(torch.tensor([0.2, -0.1, 0.3]))
    batch = adapter.sample(_request())

    stats = adapter.recompute_policy_stats(
        batch,
        require_reference=True,
    )
    stats.validate_against(batch, require_reference=True)
    assert tuple(stats.current_transition_mean.shape) == tuple(
        batch.next_latents.shape
    )
    assert tuple(stats.reference_transition_mean.shape) == tuple(
        batch.next_latents.shape
    )
    assert tuple(stats.transition_std.shape) == tuple(
        batch.old_log_probs.shape
    )
    assert stats.current_transition_mean.requires_grad
    assert not stats.reference_transition_mean.requires_grad
    assert stats.reference_transition_mean.grad_fn is None
    assert not stats.transition_std.requires_grad
    assert stats.transition_std.grad_fn is None

    expected_reference = (
        batch.latents
        + adapter.base_color_bias.view(1, 1, adapter.CHANNELS, 1, 1)
    )
    torch.testing.assert_close(
        stats.reference_transition_mean,
        expected_reference,
    )
    delta = (
        stats.current_transition_mean
        - stats.reference_transition_mean
    )
    std = stats.transition_std.reshape(
        *stats.transition_std.shape,
        1,
        1,
        1,
    )
    reference_kl = (delta.square() / (2.0 * std.square())).mean()
    assert torch.isfinite(reference_kl)
    assert float(reference_kl.detach()) > 0.0

    adapter.train_module.zero_grad(set_to_none=True)
    reference_kl.backward()
    assert adapter.color_bias.grad is not None
    assert bool(torch.isfinite(adapter.color_bias.grad).all())
    assert bool((adapter.color_bias.grad != 0).any())
    assert adapter.base_color_bias.grad is None
    assert tuple(name for name, _ in adapter.named_parameters()) == (
        "color_bias",
    )


def test_tiny_reference_recompute_restores_original_training_mode():
    adapter = TinyDiffusionAdapter(image_size=4, device="cpu")
    batch = adapter.sample(_request())
    adapter.train_module.eval()
    adapter.recompute_policy_stats(batch, require_reference=True)
    assert adapter.train_module.training is False
