"""The single policy objective for every built-in policy algorithm."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import TYPE_CHECKING, Any

from visual_rl.core.types import PolicyRecomputeStats, RolloutBatch
from visual_rl.errors import RunError
from visual_rl.optimizers.clipped_surrogate import clipped_surrogate, masked_mean

if TYPE_CHECKING:
    from torch import Tensor


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


@dataclass(frozen=True)
class PolicyLossInputs:
    """Algorithm-prepared tensors consumed by the shared objective."""

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
            raise TypeError(
                "indices must be a non-empty sequence of integers"
            ) from exc
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

    def validate_against(self, batch: RolloutBatch) -> None:
        """Require the algorithm tensors to match the canonical transition grid."""

        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        old_log_probs = batch.old_log_probs
        if tuple(old_log_probs.shape) != tuple(self.base_advantage.shape):
            raise ValueError(
                "PolicyLossInputs tensors must match batch.old_log_probs shape"
            )


@dataclass(frozen=True)
class ObjectiveOutput:
    """The five core scalars and their one shared reduction weight."""

    loss: Tensor
    policy_loss: Tensor
    reference_kl: Tensor
    approx_kl: Tensor
    clipfrac: Tensor
    active_transition_count: int


def reference_regularizer(
    *,
    current_mean: Tensor,
    reference_mean: Tensor,
    transition_std: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Compute the same-variance Gaussian transition KL over active entries."""

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
        current, reference = torch.broadcast_tensors(
            current_mean,
            reference_mean,
        )
    except RuntimeError as exc:
        raise RunError("current and reference transition means cannot broadcast") from exc
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
    elif tuple(transition_std.shape) == tuple(current.shape):
        std_base = transition_std
    elif (
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


class PolicyObjective:
    """Combine the shared surrogate with the optional reference regularizer."""

    def __call__(
        self,
        batch: RolloutBatch,
        loss_inputs: PolicyLossInputs,
        stats: PolicyRecomputeStats,
    ) -> ObjectiveOutput:
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(loss_inputs, PolicyLossInputs):
            raise TypeError("loss_inputs must be a PolicyLossInputs")
        if not isinstance(stats, PolicyRecomputeStats):
            raise TypeError("stats must be a PolicyRecomputeStats")
        loss_inputs.validate_against(batch)

        surrogate = clipped_surrogate(
            old_log_probs=batch.old_log_probs,
            new_log_probs=stats.new_log_probs,
            inputs=loss_inputs,
        )
        if loss_inputs.reference_kl_weight > 0.0:
            if (
                stats.current_transition_mean is None
                or stats.reference_transition_mean is None
                or stats.transition_std is None
            ):
                raise RunError(
                    "reference KL requires current/reference means and transition std"
                )
            reference_kl = reference_regularizer(
                current_mean=stats.current_transition_mean,
                reference_mean=stats.reference_transition_mean,
                transition_std=stats.transition_std,
                active_mask=loss_inputs.active_mask,
            )
        else:
            reference_kl = surrogate.policy_loss.new_zeros(())
        loss = (
            surrogate.policy_loss
            + loss_inputs.reference_kl_weight * reference_kl
        )
        return ObjectiveOutput(
            loss=loss,
            policy_loss=surrogate.policy_loss,
            reference_kl=reference_kl,
            approx_kl=surrogate.approx_kl,
            clipfrac=surrogate.clipfrac,
            active_transition_count=surrogate.active_transition_count,
        )
