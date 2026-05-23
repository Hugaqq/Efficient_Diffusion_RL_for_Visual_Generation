"""GRPO loss used by the v0.1 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visual_rl.core.types import RolloutBatch


@dataclass
class GRPOAlgorithm:
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0

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

        if rewards.ndim == 1:
            advantages = rewards[:, None].expand_as(new_log_probs)
        else:
            advantages = rewards
        advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
        ratio = torch.exp(new_log_probs - batch.old_log_probs)
        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = torch.maximum(unclipped, clipped).mean()
        approx_kl = 0.5 * ((new_log_probs - batch.old_log_probs) ** 2).mean()
        clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
        if batch.kl is not None and self.beta > 0:
            policy_loss = policy_loss + self.beta * batch.kl.mean()
        return policy_loss, {"approx_kl": approx_kl, "clipfrac": clipfrac, "policy_loss": policy_loss.detach()}
