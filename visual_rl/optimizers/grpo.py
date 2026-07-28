"""GRPO loss used by the v0.1 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visual_rl.core.types import RolloutBatch
from visual_rl.core.registry import ALGORITHMS


@dataclass
class GRPOAlgorithm:
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.beta != 0.0:
            raise ValueError(
                "GRPO requires beta=0 until differentiable current/reference "
                "KL is recomputed during the update"
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GRPOAlgorithm":
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        return cls(
            clip_range=float(config.get("clip_range", 0.001)),
            adv_clip_max=float(config.get("adv_clip_max", 5.0)),
            beta=float(config.get("beta", 0.0)),
        )

    def compute_loss(self, batch: RolloutBatch, rewards, new_log_probs):
        import torch

        if self.beta != 0.0:
            raise ValueError(
                "GRPO requires beta=0 until differentiable current/reference "
                "KL is recomputed during the update"
            )
        if new_log_probs.ndim != 2:
            raise ValueError(
                "GRPO log probabilities must have shape [batch, transitions]"
            )
        if rewards.ndim == 1:
            advantages = rewards[:, None].expand_as(new_log_probs)
        elif tuple(rewards.shape) == tuple(new_log_probs.shape):
            advantages = rewards
        else:
            raise ValueError(
                "GRPO advantages must have shape [batch] or match log "
                "probabilities [batch, transitions]"
            )
        advantages = advantages.to(new_log_probs.device, dtype=new_log_probs.dtype)
        old_log_probs = torch.as_tensor(
            batch.old_log_probs,
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        if tuple(old_log_probs.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "GRPO old/new log probabilities must have identical shapes: "
                f"{tuple(old_log_probs.shape)} != {tuple(new_log_probs.shape)}"
            )
        transition_mask = self._transition_mask(batch, new_log_probs)
        advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
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
        return policy_loss, {
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "policy_loss": policy_loss.detach(),
        }

    @staticmethod
    def reduction_weight(batch: RolloutBatch, advantages) -> int:
        import torch

        del advantages
        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim == 0 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError("GRPO old_log_probs must have a batch dimension")
        if batch.transition_mask is None:
            return old_log_probs.numel()
        mask = torch.as_tensor(batch.transition_mask, dtype=torch.bool)
        if tuple(mask.shape) != tuple(old_log_probs.shape):
            raise ValueError(
                "transition_mask must have the same shape as old_log_probs"
            )
        return int(mask.sum().item())

    @staticmethod
    def _transition_mask(batch: RolloutBatch, new_log_probs):
        import torch

        if batch.transition_mask is None:
            return torch.ones_like(new_log_probs, dtype=torch.bool)
        mask = torch.as_tensor(
            batch.transition_mask,
            device=new_log_probs.device,
            dtype=torch.bool,
        )
        if tuple(mask.shape) != tuple(new_log_probs.shape):
            raise ValueError(
                "GRPO transition_mask must have the same shape as log "
                f"probabilities: {tuple(mask.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            )
        if not bool(mask.any()):
            raise ValueError("GRPO requires at least one active transition")
        return mask


ALGORITHMS.register("grpo", GRPOAlgorithm)
