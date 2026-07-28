"""TempFlow-GRPO loss with typed branch credit and noise weighting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RolloutBatch,
    RuntimeBuildContext,
)
from visual_rl.optimizers.base import (
    PolicyAlgorithm,
    _resolve_algorithm_params,
)


@dataclass
class TempFlowGRPOAlgorithm(PolicyAlgorithm):
    """Current policy-identity TempFlow objective.

    The rollout contract owns branch/timestep alignment and transition standard
    deviations. This class applies the existing TempFlow formula without a
    metadata fallback, rollout-time KL, or configurable credit/noise variant.
    """

    TRAINING_CONTRACT_VERSION = 1
    ADVANTAGE_DTYPE = "float64"
    MIN_GROUP_SIZE = 2

    _PREPARED_NOISE_WEIGHT_KEY = "_visual_rl_tempflow_noise_weights"
    _NOISE_SCALE = 2.25

    clip_range: float = 0.001
    adv_clip_max: float = 5.0

    def __post_init__(self) -> None:
        for name, value in (
            ("clip_range", self.clip_range),
            ("adv_clip_max", self.adv_clip_max),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> FrozenMapping:
        return _resolve_algorithm_params(raw, context, allow_beta=False)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> TempFlowGRPOAlgorithm:
        del context
        return cls(
            clip_range=float(resolved["clip_range"]),
            adv_clip_max=float(resolved["adv_clip_max"]),
        )

    def compute_loss(self, batch: RolloutBatch, advantages, new_log_probs):
        import torch

        self._validate_batch(batch, new_log_probs)
        self._validate_advantages(advantages, batch.batch_size)
        expanded_advantages, branch_credit_mask = (
            self._expand_advantages_and_mask(
                batch,
                advantages,
                new_log_probs,
            )
        )
        active_mask = branch_credit_mask & self._transition_mask(
            batch,
            new_log_probs,
        )
        expanded_advantages = expanded_advantages.clamp(
            -self.adv_clip_max,
            self.adv_clip_max,
        )
        weights = self._prepared_noise_weights(batch, new_log_probs)
        if weights is None:
            weights = self._noise_weights(batch, new_log_probs)
        effective_advantages = expanded_advantages * weights

        old_log_probs = batch.old_log_probs.to(
            new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        logprob_delta = new_log_probs - old_log_probs
        safe_logprob_delta = torch.where(
            active_mask,
            logprob_delta,
            torch.zeros_like(logprob_delta),
        )
        ratio = torch.exp(safe_logprob_delta)
        unclipped = -effective_advantages * ratio
        clipped = -effective_advantages * ratio.clamp(
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        policy_loss = self._masked_mean(
            torch.maximum(unclipped, clipped),
            active_mask,
            name="PPO policy loss",
        )
        approx_kl = 0.5 * self._masked_mean(
            safe_logprob_delta.square(),
            active_mask,
            name="approx_kl",
        )
        clipfrac = self._masked_mean(
            ((ratio - 1.0).abs() > self.clip_range).to(
                new_log_probs.dtype
            ),
            active_mask,
            name="clipfrac",
        )
        return policy_loss, {
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "policy_loss": policy_loss.detach(),
            "tempflow_noise_weight_mean": self._masked_mean(
                weights,
                active_mask,
                name="TempFlow noise weight",
            ).detach(),
            "tempflow_active_timestep_frac": active_mask.to(
                new_log_probs.dtype
            )
            .mean()
            .detach(),
        }

    def prepare_batch(self, batch: RolloutBatch, advantages) -> RolloutBatch:
        import torch

        self._validate_advantages(advantages, batch.batch_size)
        old_log_probs = torch.as_tensor(batch.old_log_probs)
        objective_dtype = (
            torch.float64
            if advantages.dtype == torch.float64
            else torch.float32
        )
        weights = self._noise_weights(
            batch,
            old_log_probs.to(dtype=objective_dtype),
        )
        payload = dict(batch.recompute_payload)
        payload[self._PREPARED_NOISE_WEIGHT_KEY] = weights.detach()
        return batch.replace(recompute_payload=payload)

    def reduction_weight(self, batch: RolloutBatch, advantages) -> int:
        import torch

        old_log_probs = torch.as_tensor(batch.old_log_probs)
        self._validate_batch(batch, old_log_probs)
        self._validate_advantages(advantages, batch.batch_size)
        _, branch_credit_mask = self._expand_advantages_and_mask(
            batch,
            advantages,
            old_log_probs,
        )
        active_mask = branch_credit_mask & self._transition_mask(
            batch,
            old_log_probs,
        )
        return int(active_mask.sum().item())

    def metric_reduction_weight(
        self,
        batch: RolloutBatch,
        advantages,
        metric_name: str,
    ) -> int:
        if metric_name == "tempflow_active_timestep_frac":
            return int(batch.old_log_probs.numel())
        return self.reduction_weight(batch, advantages)

    def _prepared_noise_weights(self, batch, new_log_probs):
        import torch

        weights = batch.recompute_payload.get(
            self._PREPARED_NOISE_WEIGHT_KEY
        )
        if weights is None:
            return None
        weights = torch.as_tensor(
            weights,
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        if tuple(weights.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "Prepared TempFlow noise weights must match log probabilities: "
                f"{tuple(weights.shape)} != {tuple(new_log_probs.shape)}"
            )
        return weights

    @staticmethod
    def _validate_advantages(advantages, batch_size: int) -> None:
        import torch

        if not isinstance(advantages, torch.Tensor):
            raise TypeError(
                "TempFlow advantages must be a torch.Tensor with "
                "dtype=torch.float64"
            )
        if advantages.dtype != torch.float64:
            raise TypeError(
                "TempFlow advantages must have dtype=torch.float64; "
                f"got {advantages.dtype}"
            )
        if tuple(advantages.shape) != (batch_size,):
            raise ValueError(
                "TempFlow advantages must have shape [B], got "
                f"{tuple(advantages.shape)}"
            )
        if not bool(torch.isfinite(advantages).all()):
            raise ValueError("TempFlow advantages must be finite")

    @staticmethod
    def _validate_batch(batch: RolloutBatch, new_log_probs) -> None:
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if new_log_probs.ndim != 2:
            raise ValueError(
                "TempFlow log probabilities must have shape [B, T]"
            )
        if tuple(new_log_probs.shape) != tuple(batch.old_log_probs.shape):
            raise ValueError(
                "TempFlow old/new log probabilities must have identical shapes: "
                f"{tuple(batch.old_log_probs.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            )
        if batch.branch_step_index is None:
            raise ValueError("TempFlow requires branch_step_index")
        if batch.trajectory_step_index is None:
            raise ValueError("TempFlow requires trajectory_step_index")
        if batch.transition_std_dev is None:
            raise ValueError("TempFlow requires transition_std_dev")

    @staticmethod
    def _expand_advantages_and_mask(
        batch: RolloutBatch,
        advantages,
        new_log_probs,
    ):
        import torch

        advantages = advantages.to(new_log_probs.device)
        branch_steps = batch.branch_step_index.to(new_log_probs.device)
        trajectory_steps = batch.trajectory_step_index.to(
            new_log_probs.device
        )
        branch_credit_mask = (
            trajectory_steps.unsqueeze(0) == branch_steps.unsqueeze(1)
        )
        if tuple(branch_credit_mask.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "TempFlow branch credit mask must match log probabilities"
            )
        if not bool((branch_credit_mask.sum(dim=1) == 1).all()):
            raise ValueError(
                "each TempFlow branch_step_index must map to exactly one "
                "trajectory_step_index"
            )
        assigned = torch.zeros(
            new_log_probs.shape,
            dtype=advantages.dtype,
            device=new_log_probs.device,
        )
        assigned = torch.where(
            branch_credit_mask,
            advantages[:, None],
            assigned,
        )
        return assigned, branch_credit_mask

    @staticmethod
    def _transition_mask(batch: RolloutBatch, new_log_probs):
        mask = batch.transition_mask.to(
            device=new_log_probs.device,
        )
        if tuple(mask.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "TempFlow transition_mask must have the same shape as log "
                f"probabilities: {tuple(mask.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            )
        return mask

    def _noise_weights(self, batch: RolloutBatch, new_log_probs):
        import torch

        weights = batch.transition_std_dev.to(
            device=new_log_probs.device,
        )
        if tuple(weights.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "TempFlow transition_std_dev must have shape [B, T]: "
                f"{tuple(weights.shape)} != {tuple(new_log_probs.shape)}"
            )
        if (
            weights.dtype == torch.bool
            or weights.is_complex()
            or not bool(torch.isfinite(weights).all())
            or not bool((weights > 0).all())
        ):
            raise ValueError(
                "TempFlow transition_std_dev must contain finite positive values"
            )
        scaled = weights.to(dtype=new_log_probs.dtype) * self._NOISE_SCALE
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError(
                "TempFlow scaled transition_std_dev must remain finite"
            )
        return scaled

    @staticmethod
    def _masked_mean(values, active_mask, *, name: str):
        if values.shape != active_mask.shape:
            raise ValueError(
                f"{name} values and active mask must have identical shapes: "
                f"{tuple(values.shape)} != {tuple(active_mask.shape)}"
            )
        active_values = values.masked_select(active_mask)
        if active_values.numel() == 0:
            raise ValueError(f"{name} requires at least one active transition")
        return active_values.mean()
