"""Flash-GRPO loss for selected single-step diffusion RL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
class FlashGRPOAlgorithm(PolicyAlgorithm):
    TRAINING_CONTRACT_VERSION = 1
    ADVANTAGE_DTYPE = "float32"
    MIN_GROUP_SIZE = 2

    _PREPARED_RECTIFICATION_KEY = "_visual_rl_flash_prepared_weight"

    objective_version: str = "legacy_v0"
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0
    rectification: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.beta != 0.0:
            raise ValueError(
                "Flash-GRPO requires beta=0 until differentiable "
                "current/reference KL is recomputed during the update"
            )

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
    ) -> "FlashGRPOAlgorithm":
        del context
        return cls(
            objective_version="reference_v1",
            clip_range=float(resolved["clip_range"]),
            adv_clip_max=float(resolved["adv_clip_max"]),
            beta=0.0,
            rectification={},
        )

    def compute_loss(self, batch: RolloutBatch, rewards, new_log_probs):
        import torch

        if self.beta != 0.0:
            raise ValueError(
                "Flash-GRPO requires beta=0 until differentiable "
                "current/reference KL is recomputed during the update"
            )
        if new_log_probs.ndim != 2:
            raise ValueError(
                "Flash-GRPO log probabilities must have shape [batch, transitions]"
            )
        advantages = self._expand_advantages(rewards, new_log_probs)
        advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
        rectification_weights = self._prepared_rectification_weights(
            batch,
            new_log_probs,
        )
        if rectification_weights is None:
            if self.objective_version == "reference_v1":
                rectification_weights = self._reference_coefficients(
                    batch, new_log_probs
                )
            else:
                rectification_weights = self._rectification_weights(
                    batch, new_log_probs
                ).to(new_log_probs.device)
        advantages = advantages * rectification_weights
        old_log_probs = torch.as_tensor(
            batch.old_log_probs,
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        if tuple(old_log_probs.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "Flash-GRPO old/new log probabilities must have identical "
                f"shapes: {tuple(old_log_probs.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            )
        transition_mask = self._transition_mask(batch, new_log_probs)

        logprob_delta = new_log_probs - old_log_probs
        safe_logprob_delta = torch.where(
            transition_mask,
            logprob_delta,
            torch.zeros_like(logprob_delta),
        )
        ratio = torch.exp(safe_logprob_delta)
        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(
            1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = (
            torch.maximum(unclipped, clipped).masked_select(transition_mask).mean()
        )
        approx_kl = (
            0.5 * safe_logprob_delta.square().masked_select(transition_mask).mean()
        )
        clipfrac = (
            ((ratio - 1.0).abs() > self.clip_range)
            .to(new_log_probs.dtype)
            .masked_select(transition_mask)
            .mean()
        )
        active_weights = rectification_weights.masked_select(transition_mask)
        active_advantages = advantages.masked_select(transition_mask)

        return policy_loss, {
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "policy_loss": policy_loss.detach(),
            "flash_rectification_weight_mean": active_weights.mean().detach(),
            "flash_selected_timestep_mean": self._selected_timestep_mean(
                batch, new_log_probs
            ),
            "flash_active_timestep_frac": (
                (active_advantages != 0).to(new_log_probs.dtype).mean().detach()
            ),
        }

    def prepare_batch(self, batch: RolloutBatch, advantages) -> RolloutBatch:
        import torch

        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim == 0 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError(
                "Flash-GRPO old_log_probs must have a batch dimension"
            )
        advantage_values = torch.as_tensor(advantages)
        objective_dtype = (
            torch.float64
            if advantage_values.dtype == torch.float64
            else torch.float32
        )
        old_log_probs = old_log_probs.to(dtype=objective_dtype)
        if self.objective_version == "reference_v1":
            weights = self._reference_coefficients(batch, old_log_probs)
        else:
            weights = self._rectification_weights(batch, old_log_probs)
        return self._with_prepared_rectification_weights(batch, weights)

    def requires_global_batch_reduction(self) -> bool:
        """Match the published reference's cross-rank coefficient mean."""

        return self.objective_version == "reference_v1"

    def global_batch_reduction(
        self,
        batch: RolloutBatch,
        advantages,
    ) -> tuple[Any, int]:
        """Return this rank's coefficient mean and sample count once per batch."""

        import torch

        advantage_values = torch.as_tensor(advantages)
        objective_dtype = (
            torch.float64
            if advantage_values.dtype == torch.float64
            else torch.float32
        )
        old_log_probs = torch.as_tensor(
            batch.old_log_probs,
            dtype=objective_dtype,
        )
        coefficient = self._reference_coefficient(batch, old_log_probs)
        return coefficient.mean().detach(), coefficient.numel()

    def apply_global_batch_reduction(
        self,
        batch: RolloutBatch,
        advantages,
        global_mean: Any,
    ) -> RolloutBatch:
        """Store globally normalized full-batch weights for every microbatch."""

        import torch

        advantage_values = torch.as_tensor(advantages)
        objective_dtype = (
            torch.float64
            if advantage_values.dtype == torch.float64
            else torch.float32
        )
        old_log_probs = torch.as_tensor(
            batch.old_log_probs,
            dtype=objective_dtype,
        )
        weights = self._reference_coefficients(
            batch,
            old_log_probs,
            normalization_mean=global_mean,
        )
        return self._with_prepared_rectification_weights(batch, weights)

    @staticmethod
    def reduction_weight(batch: RolloutBatch, advantages) -> int:
        import torch

        del advantages
        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim == 0 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError("Flash-GRPO old_log_probs must have a batch dimension")
        mask = torch.as_tensor(batch.transition_mask, dtype=torch.bool)
        if tuple(mask.shape) != tuple(old_log_probs.shape):
            raise ValueError(
                "transition_mask must have the same shape as old_log_probs"
            )
        return int(mask.sum().item())

    def metric_reduction_weight(
        self,
        batch: RolloutBatch,
        advantages,
        metric_name: str,
    ) -> int:
        if metric_name == "flash_selected_timestep_mean":
            return batch.batch_size
        return self.reduction_weight(batch, advantages)

    @staticmethod
    def _transition_mask(batch: RolloutBatch, new_log_probs):
        import torch

        mask = torch.as_tensor(
            batch.transition_mask,
            device=new_log_probs.device,
            dtype=torch.bool,
        )
        if tuple(mask.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "Flash-GRPO transition_mask must have the same shape as log "
                f"probabilities: {tuple(mask.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            )
        if not bool(mask.any()):
            raise ValueError("Flash-GRPO requires at least one active transition")
        return mask

    def _prepared_rectification_weights(self, batch, new_log_probs):
        import torch

        weights = batch.recompute_payload.get(self._PREPARED_RECTIFICATION_KEY)
        if weights is None:
            return None
        weights = torch.as_tensor(
            weights,
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        try:
            return torch.broadcast_to(weights, new_log_probs.shape)
        except RuntimeError as exc:
            raise ValueError(
                "Prepared Flash rectification weights cannot broadcast to "
                f"log probabilities: {tuple(weights.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            ) from exc

    @classmethod
    def _with_prepared_rectification_weights(
        cls,
        batch: RolloutBatch,
        weights,
    ) -> RolloutBatch:
        payload = dict(batch.recompute_payload)
        payload[cls._PREPARED_RECTIFICATION_KEY] = weights.detach()
        return batch.replace(recompute_payload=payload)

    @staticmethod
    def _expand_advantages(rewards, new_log_probs):
        import torch

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
        rewards = rewards.to(new_log_probs.device, dtype=new_log_probs.dtype)
        if rewards.shape == new_log_probs.shape:
            return rewards
        if rewards.ndim != 1:
            raise ValueError("Flash-GRPO advantages must be either [batch] or [batch, 1].")
        return rewards[:, None].expand_as(new_log_probs)

    def _rectification_weights(self, batch: RolloutBatch, new_log_probs):
        import torch

        config = self.rectification or {}
        if config.get("enabled", True) is False or config.get("mode", "scheduler_formula") in {"none", None}:
            return torch.ones_like(new_log_probs)

        if batch.flash_coefficient is not None:
            weights = self._reference_coefficient(batch, new_log_probs)
            weights = weights.expand_as(new_log_probs)
            if config.get("normalize", True):
                weights = weights / weights.mean().clamp_min(1e-6)
            return weights

        selected = batch.selected_timestep_index
        positions = (
            torch.zeros(
                (new_log_probs.shape[0], 1),
                dtype=new_log_probs.dtype,
                device=new_log_probs.device,
            )
            if selected is None
            else torch.as_tensor(
                selected,
                dtype=new_log_probs.dtype,
                device=new_log_probs.device,
            )[:, None]
        )
        num_steps = max(1, batch.transition_count)
        if config.get("mode", "scheduler_formula") == "scheduler_formula":
            weights = torch.sqrt((num_steps - positions).clamp_min(1.0) / float(num_steps))
        else:
            raise ValueError(f"Unknown Flash rectification mode: {config.get('mode')}")
        if config.get("normalize", True):
            weights = weights / weights.mean().clamp_min(1e-6)
        return weights.expand_as(new_log_probs)

    @staticmethod
    def _reference_coefficient(batch: RolloutBatch, new_log_probs):
        import torch

        coefficient = batch.flash_coefficient
        if not isinstance(coefficient, torch.Tensor):
            raise ValueError(
                "Flash-GRPO reference_v1 requires batch.flash_coefficient"
            )
        expected_shape = (new_log_probs.shape[0], 1)
        if tuple(coefficient.shape) != expected_shape:
            raise ValueError(
                "Flash-GRPO reference_v1 coefficient must have shape "
                f"[B,1], got {tuple(coefficient.shape)}"
            )
        coefficient = coefficient.to(
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        if not torch.isfinite(coefficient).all() or not (coefficient > 0).all():
            raise ValueError(
                "Flash-GRPO reference_v1 coefficient must be finite and positive"
            )
        return coefficient

    @classmethod
    def _reference_coefficients(
        cls,
        batch: RolloutBatch,
        new_log_probs,
        *,
        normalization_mean: Any | None = None,
    ):
        import torch

        coefficient = cls._reference_coefficient(batch, new_log_probs)
        mean = (
            coefficient.mean()
            if normalization_mean is None
            else torch.as_tensor(
                normalization_mean,
                device=coefficient.device,
                dtype=coefficient.dtype,
            )
        )
        if mean.ndim != 0 or not torch.isfinite(mean) or not (mean > 0):
            raise ValueError(
                "Flash-GRPO reference_v1 coefficient normalization mean must "
                "be finite and positive"
            )
        return coefficient / mean

    @staticmethod
    def _selected_timestep_mean(batch: RolloutBatch, new_log_probs):
        import torch

        selected = batch.selected_timestep_index
        if selected is None:
            return torch.zeros((), dtype=new_log_probs.dtype, device=new_log_probs.device)
        return torch.as_tensor(selected, dtype=new_log_probs.dtype, device=new_log_probs.device).mean().detach()
