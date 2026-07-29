"""The single clipped-surrogate implementation used by policy objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from visual_rl.optimizers.objective import PolicyLossInputs


@dataclass(frozen=True)
class ClippedSurrogateOutput:
    """Scalar outputs reduced over one shared set of active transitions."""

    policy_loss: Tensor
    approx_kl: Tensor
    clipfrac: Tensor
    active_transition_count: int


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Return a NaN-safe mean over ``mask=True`` entries.

    Inactive values are replaced before reduction rather than multiplied by
    zero.  This matters because ``NaN * 0`` remains ``NaN``.
    """

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have bool dtype")
    if tuple(value.shape) != tuple(mask.shape):
        raise ValueError("value and mask must have the same shape")
    if value.device != mask.device:
        raise ValueError("value and mask must be on the same device")
    active_count = int(mask.sum().item())
    if active_count <= 0:
        raise ValueError("masked mean requires at least one active value")
    safe_value = torch.where(mask, value, 0.0)
    return safe_value.sum() / mask.sum()


def clipped_surrogate(
    *,
    old_log_probs: Tensor,
    new_log_probs: Tensor,
    inputs: PolicyLossInputs,
) -> ClippedSurrogateOutput:
    """Compute the unique PPO-style clipped policy surrogate.

    ``inputs`` is type-only imported from :mod:`objective` so this numerical
    primitive stays independent at runtime.
    """

    import torch

    if not isinstance(old_log_probs, torch.Tensor):
        raise TypeError("old_log_probs must be a torch.Tensor")
    if not isinstance(new_log_probs, torch.Tensor):
        raise TypeError("new_log_probs must be a torch.Tensor")
    if not old_log_probs.is_floating_point():
        raise TypeError("old_log_probs must be floating point")
    if not new_log_probs.is_floating_point():
        raise TypeError("new_log_probs must be floating point")
    if old_log_probs.requires_grad or old_log_probs.grad_fn is not None:
        raise ValueError("old_log_probs must be detached without grad_fn")
    if not new_log_probs.requires_grad:
        raise ValueError("new_log_probs must require gradients")

    shape = tuple(old_log_probs.shape)
    if len(shape) != 2:
        raise ValueError("old_log_probs must have shape [B, T]")
    if tuple(new_log_probs.shape) != shape:
        raise ValueError("new_log_probs must have the same shape as old_log_probs")
    for name, value in (
        ("base_advantage", inputs.base_advantage),
        ("algorithm_weight", inputs.algorithm_weight),
        ("active_mask", inputs.active_mask),
    ):
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have the same shape as old_log_probs")

    tensors = (
        inputs.base_advantage,
        inputs.algorithm_weight,
        inputs.active_mask,
    )
    if any(value.device != old_log_probs.device for value in tensors):
        raise ValueError("surrogate inputs must be on the old_log_probs device")
    if new_log_probs.device != old_log_probs.device:
        raise ValueError("new_log_probs must be on the old_log_probs device")
    if (
        old_log_probs.dtype != new_log_probs.dtype
        or old_log_probs.dtype != inputs.base_advantage.dtype
        or old_log_probs.dtype != inputs.algorithm_weight.dtype
    ):
        raise TypeError(
            "old/new log-probs, base_advantage and algorithm_weight "
            "must have the same dtype"
        )

    active_mask = inputs.active_mask
    active_transition_count = int(active_mask.sum().item())
    if active_transition_count <= 0:
        raise ValueError("clipped surrogate requires at least one active transition")

    active_old = torch.where(active_mask, old_log_probs, 0.0)
    active_new = torch.where(active_mask, new_log_probs, 0.0)
    if not bool(torch.isfinite(active_old).all()):
        raise ValueError("old_log_probs must be finite at active transitions")
    if not bool(torch.isfinite(active_new).all()):
        raise ValueError("new_log_probs must be finite at active transitions")

    delta = active_new - active_old
    ratio = torch.exp(delta)
    safe_base = torch.where(active_mask, inputs.base_advantage, 0.0)
    safe_weight = torch.where(active_mask, inputs.algorithm_weight, 0.0)
    effective_advantage = safe_base * safe_weight
    unclipped = -effective_advantage * ratio
    clipped = -effective_advantage * ratio.clamp(
        1.0 - inputs.clip_range,
        1.0 + inputs.clip_range,
    )

    policy_loss = masked_mean(torch.maximum(unclipped, clipped), active_mask)
    approx_kl = 0.5 * masked_mean(delta.square(), active_mask)
    clipfrac = masked_mean(
        ((ratio - 1.0).abs() > inputs.clip_range).to(ratio.dtype),
        active_mask,
    )
    for name, value in (
        ("policy_loss", policy_loss),
        ("approx_kl", approx_kl),
        ("clipfrac", clipfrac),
    ):
        if not bool(torch.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    return ClippedSurrogateOutput(
        policy_loss=policy_loss,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        active_transition_count=active_transition_count,
    )
