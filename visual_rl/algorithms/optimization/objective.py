"""Canonical numerical objective for GRPO-family policy updates.

This module is the sole owner of policy-loss inputs, masked reductions, the
clipped surrogate, and the optional same-variance reference regularizer.  It
intentionally depends on neither the retired ``visual_rl.optimizers`` package
nor the legacy algorithm objective/credit modules.  Recompute owns policy
statistics; the objective consumes only the minimal structural view it needs.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from visual_rl.errors import RunError

if TYPE_CHECKING:
    from torch import Tensor


class _PolicyStatsView(Protocol):
    """The unconditional recompute-owned tensor view used by the objective."""

    @property
    def current_log_probs(self) -> Tensor: ...


class _ReferencePolicyStatsView(_PolicyStatsView, Protocol):
    """The conditional recompute-owned view required by reference KL."""

    @property
    def current_transition_mean(self) -> Tensor | None: ...

    @property
    def transition_std(self) -> Tensor | None: ...

    @property
    def reference_transition_mean(self) -> Tensor | None: ...


def _require_detached(name: str, value: Tensor) -> None:
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached without grad_fn")


def _finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class PolicyLossInputs:
    """Detached algorithm output consumed by the shared numerical objective."""

    base_advantage: Tensor
    algorithm_weight: Tensor
    active_mask: Tensor
    clip_range: float
    reference_kl_weight: float = 0.0

    def __post_init__(self) -> None:
        import torch

        for name in ("base_advantage", "algorithm_weight", "active_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            _require_detached(name, value)

        shape = tuple(self.base_advantage.shape)
        if len(shape) != 2:
            raise ValueError("base_advantage must have shape [B, T]")
        if tuple(self.algorithm_weight.shape) != shape:
            raise ValueError(
                "algorithm_weight must have the same shape as base_advantage"
            )
        if tuple(self.active_mask.shape) != shape:
            raise ValueError("active_mask must have the same shape as base_advantage")
        if not self.base_advantage.is_floating_point():
            raise TypeError("base_advantage must be floating point")
        if not self.algorithm_weight.is_floating_point():
            raise TypeError("algorithm_weight must be floating point")
        if self.active_mask.dtype != torch.bool:
            raise TypeError("active_mask must have bool dtype")
        if self.base_advantage.device != self.algorithm_weight.device:
            raise ValueError(
                "base_advantage and algorithm_weight must be on the same device"
            )
        if self.base_advantage.device != self.active_mask.device:
            raise ValueError("active_mask must be on the base_advantage device")
        if self.base_advantage.dtype != self.algorithm_weight.dtype:
            raise TypeError(
                "base_advantage and algorithm_weight must have the same dtype"
            )

        safe_base = torch.where(self.active_mask, self.base_advantage, 0.0)
        safe_weight = torch.where(self.active_mask, self.algorithm_weight, 1.0)
        if not bool(torch.isfinite(safe_base).all()):
            raise ValueError("base_advantage must be finite at active transitions")
        if not bool(torch.isfinite(safe_weight).all()):
            raise ValueError("algorithm_weight must be finite at active transitions")
        if not bool((safe_weight > 0).all()):
            raise ValueError(
                "algorithm_weight must be strictly positive at active transitions"
            )

        clip_range = _finite_scalar("clip_range", self.clip_range)
        if not 0.0 < clip_range < 1.0:
            raise ValueError("clip_range must satisfy 0 < clip_range < 1")
        reference_kl_weight = _finite_scalar(
            "reference_kl_weight",
            self.reference_kl_weight,
        )
        if reference_kl_weight < 0.0:
            raise ValueError("reference_kl_weight must be non-negative")
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "reference_kl_weight", reference_kl_weight)

    def slice(self, indices: Any) -> PolicyLossInputs:
        """Select a non-empty, duplicate-free set of samples along ``B``."""

        import torch

        if isinstance(indices, (str, bytes)):
            raise TypeError("indices must be a non-empty sequence of integers")
        try:
            resolved = list(indices)
        except TypeError as exc:
            raise TypeError("indices must be a non-empty sequence of integers") from exc
        if not resolved:
            raise ValueError("indices must not be empty")
        batch_size = int(self.base_advantage.shape[0])
        normalized: list[int] = []
        for position, index in enumerate(resolved):
            if isinstance(index, bool):
                raise TypeError(
                    f"sample index at position {position} must be an integer"
                )
            try:
                item = operator.index(index)
            except TypeError as exc:
                raise TypeError(
                    f"sample index at position {position} must be an integer"
                ) from exc
            if item < 0 or item >= batch_size:
                raise IndexError(
                    f"sample index {item} is out of bounds for batch size {batch_size}"
                )
            normalized.append(item)
        if len(set(normalized)) != len(normalized):
            raise ValueError("indices must not contain duplicates")
        tensor_index = torch.tensor(
            normalized,
            dtype=torch.long,
            device=self.base_advantage.device,
        )
        return replace(
            self,
            base_advantage=self.base_advantage.index_select(0, tensor_index),
            algorithm_weight=self.algorithm_weight.index_select(0, tensor_index),
            active_mask=self.active_mask.index_select(0, tensor_index),
        )

    def slice_transitions(self, start: int, stop: int) -> PolicyLossInputs:
        """Select one non-empty contiguous transition interval."""

        transition_count = int(self.base_advantage.shape[1])
        for name, value in (("start", start), ("stop", stop)):
            if type(value) is not int:
                raise TypeError(f"transition {name} must be an integer")
        if not 0 <= start < stop <= transition_count:
            raise IndexError(
                "transition interval must satisfy "
                f"0 <= start < stop <= {transition_count}"
            )
        if start == 0 and stop == transition_count:
            return self
        return replace(
            self,
            base_advantage=self.base_advantage[:, start:stop],
            algorithm_weight=self.algorithm_weight[:, start:stop],
            active_mask=self.active_mask[:, start:stop],
        )

    def validate_shape(self, old_log_probs: Any) -> None:
        """Require the loss plan to match a canonical ``[B,T]`` grid."""

        import torch

        if not isinstance(old_log_probs, torch.Tensor):
            raise TypeError("old_log_probs must be a torch.Tensor")
        if tuple(old_log_probs.shape) != tuple(self.base_advantage.shape):
            raise ValueError("PolicyLossInputs tensors must match old_log_probs shape")


@dataclass(frozen=True, slots=True)
class ClippedSurrogateOutput:
    """Scalar outputs reduced over one shared active-transition mask."""

    policy_loss: Tensor
    approx_kl: Tensor
    clipfrac: Tensor
    active_transition_count: int


@dataclass(frozen=True, slots=True)
class LossOutput:
    """The final differentiable loss and detached-reportable scalar metrics."""

    loss: Tensor
    policy_loss: Tensor
    reference_kl: Tensor
    approx_kl: Tensor
    clipfrac: Tensor
    active_transition_count: int

    def __post_init__(self) -> None:
        import torch

        for name in (
            "loss",
            "policy_loss",
            "reference_kl",
            "approx_kl",
            "clipfrac",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim != 0
            ):
                raise TypeError(f"{name} must be a floating scalar tensor")
            if not bool(torch.isfinite(value.detach())):
                raise ValueError(f"{name} must be finite")
        if not self.loss.requires_grad:
            raise ValueError("loss must require gradients")
        if (
            type(self.active_transition_count) is not int
            or self.active_transition_count < 1
        ):
            raise ValueError("active_transition_count must be positive")


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Return a NaN-safe mean over exactly the ``mask=True`` entries."""

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
    """Compute the canonical PPO-style clipped policy surrogate."""

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


def reference_regularizer(
    *,
    current_mean: Tensor,
    reference_mean: Tensor,
    transition_std: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Compute same-variance Gaussian transition KL over active entries."""

    import torch

    for name, value in (
        ("current_mean", current_mean),
        ("reference_mean", reference_mean),
        ("transition_std", transition_std),
        ("active_mask", active_mask),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if not current_mean.is_floating_point():
        raise TypeError("current_mean must be floating point")
    if not reference_mean.is_floating_point():
        raise TypeError("reference_mean must be floating point")
    if not transition_std.is_floating_point():
        raise TypeError("transition_std must be floating point")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must have bool dtype")
    if not current_mean.requires_grad:
        raise RunError("current_mean must require gradients")
    if reference_mean.requires_grad or reference_mean.grad_fn is not None:
        raise RunError("reference_mean must be detached without grad_fn")
    if transition_std.requires_grad or transition_std.grad_fn is not None:
        raise RunError("transition_std must be detached without grad_fn")
    if any(
        value.device != current_mean.device
        for value in (reference_mean, transition_std, active_mask)
    ):
        raise RunError("reference regularizer tensors must share one device")
    if (
        current_mean.dtype != reference_mean.dtype
        or current_mean.dtype != transition_std.dtype
    ):
        raise RunError("reference regularizer floating tensors must share one dtype")

    try:
        current, reference = torch.broadcast_tensors(current_mean, reference_mean)
    except RuntimeError as exc:
        raise RunError(
            "current and reference transition means cannot broadcast"
        ) from exc
    if current.ndim < 3:
        raise RunError("transition means must have shape [B, T, ...X]")
    if tuple(current.shape[:2]) != tuple(active_mask.shape):
        raise RunError("active_mask must match the [B, T] transition dimensions")

    expanded_mask = active_mask.reshape(
        *active_mask.shape,
        *((1,) * (current.ndim - 2)),
    )
    expanded_mask = torch.broadcast_to(expanded_mask, current.shape)
    if tuple(transition_std.shape) == tuple(active_mask.shape):
        std_base = transition_std.reshape(
            *active_mask.shape,
            *((1,) * (current.ndim - 2)),
        )
    elif tuple(transition_std.shape) == tuple(current.shape) or (
        transition_std.ndim == current.ndim
        and tuple(transition_std.shape[:2]) == tuple(active_mask.shape)
        and all(size == 1 for size in transition_std.shape[2:])
    ):
        std_base = transition_std
    else:
        raise RunError("unsupported transition_std shape")
    try:
        std = torch.broadcast_to(std_base, current.shape)
    except RuntimeError as exc:
        raise RunError("transition_std cannot broadcast to transition means") from exc

    active_current = torch.where(expanded_mask, current, 0.0)
    active_reference = torch.where(expanded_mask, reference, 0.0)
    active_std = torch.where(expanded_mask, std, 1.0)
    if not bool(torch.isfinite(active_current).all()):
        raise RunError("current_mean must be finite at active transitions")
    if not bool(torch.isfinite(active_reference).all()):
        raise RunError("reference_mean must be finite at active transitions")
    if not bool(torch.isfinite(active_std).all()):
        raise RunError("transition_std must be finite at active transitions")
    if not bool((active_std > 0).all()):
        raise RunError("transition_std must be positive at active transitions")

    safe_delta = active_current - active_reference
    feature_axes = tuple(range(2, current.ndim))
    per_transition_reference_kl = (
        safe_delta.square() / (2.0 * active_std.square())
    ).mean(dim=feature_axes)
    reference_kl = masked_mean(per_transition_reference_kl, active_mask)
    if not bool(torch.isfinite(reference_kl)):
        raise RunError("reference_kl must be finite")
    return reference_kl


class ClippedSurrogateObjective:
    """Combine the shared surrogate with an optional reference regularizer."""

    def compute(
        self,
        *,
        old_log_probs: Tensor,
        policy_stats: _PolicyStatsView,
        loss_inputs: PolicyLossInputs,
    ) -> LossOutput:
        import torch

        if not isinstance(old_log_probs, torch.Tensor):
            raise TypeError("old_log_probs must be a torch.Tensor")
        if not isinstance(loss_inputs, PolicyLossInputs):
            raise TypeError("loss_inputs must be PolicyLossInputs")
        try:
            current = policy_stats.current_log_probs
        except AttributeError as exc:
            raise TypeError("policy_stats must expose current_log_probs") from exc
        if not isinstance(current, torch.Tensor):
            raise TypeError("policy_stats.current_log_probs must be a torch.Tensor")
        shape = tuple(old_log_probs.shape)
        if len(shape) != 2 or tuple(current.shape) != shape:
            raise ValueError("old/current log-probs must share shape [B,T]")
        if old_log_probs.requires_grad or old_log_probs.grad_fn is not None:
            raise ValueError("old_log_probs must be detached")
        if old_log_probs.device != current.device:
            raise ValueError("old/current log-probs must share one device")
        if old_log_probs.dtype != current.dtype:
            raise TypeError("old/current log-probs must share one dtype")
        for name in ("base_advantage", "algorithm_weight", "active_mask"):
            value = getattr(loss_inputs, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must match old_log_probs shape")
        if loss_inputs.base_advantage.device != current.device:
            raise ValueError("PolicyLossInputs must be on the policy stats device")
        if loss_inputs.base_advantage.dtype != current.dtype:
            raise TypeError("PolicyLossInputs must use the policy log-prob dtype")

        surrogate = clipped_surrogate(
            old_log_probs=old_log_probs,
            new_log_probs=current,
            inputs=loss_inputs,
        )
        if loss_inputs.reference_kl_weight > 0.0:
            reference_stats = cast(_ReferencePolicyStatsView, policy_stats)
            try:
                current_mean = reference_stats.current_transition_mean
                transition_std = reference_stats.transition_std
                reference_mean = reference_stats.reference_transition_mean
            except AttributeError as exc:
                raise TypeError(
                    "policy_stats must expose reference transition statistics"
                ) from exc
            if current_mean is None or transition_std is None or reference_mean is None:
                raise ValueError(
                    "reference KL requires current/reference means and transition std"
                )
            reference_kl = reference_regularizer(
                current_mean=current_mean,
                reference_mean=reference_mean,
                transition_std=transition_std,
                active_mask=loss_inputs.active_mask,
            )
        else:
            reference_kl = surrogate.policy_loss.new_zeros(())
        loss = surrogate.policy_loss + (loss_inputs.reference_kl_weight * reference_kl)
        return LossOutput(
            loss=loss,
            policy_loss=surrogate.policy_loss,
            reference_kl=reference_kl,
            approx_kl=surrogate.approx_kl,
            clipfrac=surrogate.clipfrac,
            active_transition_count=surrogate.active_transition_count,
        )

    __call__ = compute


__all__ = (
    "ClippedSurrogateObjective",
    "ClippedSurrogateOutput",
    "LossOutput",
    "PolicyLossInputs",
    "clipped_surrogate",
    "masked_mean",
    "reference_regularizer",
)
