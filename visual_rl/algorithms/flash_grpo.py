"""Flash-GRPO loss for selected single-step diffusion RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visual_rl.core.registry import ALGORITHMS
from visual_rl.core.types import RolloutBatch


@dataclass
class FlashGRPOAlgorithm:
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0
    rectification: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FlashGRPOAlgorithm":
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        return cls(
            clip_range=float(config.get("clip_range", 0.001)),
            adv_clip_max=float(config.get("adv_clip_max", 5.0)),
            beta=float(config.get("beta", 0.0)),
            rectification=dict(config.get("rectification") or {}),
        )

    def compute_loss(self, batch: RolloutBatch, rewards, new_log_probs):
        import torch

        advantages = self._expand_advantages(rewards, new_log_probs)
        advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
        rectification_weights = self._rectification_weights(batch, new_log_probs).to(new_log_probs.device)
        advantages = advantages * rectification_weights
        old_log_probs = batch.old_log_probs.to(new_log_probs.device, dtype=new_log_probs.dtype)

        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = torch.maximum(unclipped, clipped).mean()
        approx_kl = 0.5 * ((new_log_probs - old_log_probs) ** 2).mean()
        clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
        if batch.kl is not None and self.beta > 0:
            policy_loss = policy_loss + self.beta * batch.kl.to(new_log_probs.device, dtype=new_log_probs.dtype).mean()

        return policy_loss, {
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "policy_loss": policy_loss.detach(),
            "flash_rectification_weight_mean": rectification_weights.mean().detach(),
            "flash_selected_timestep_mean": self._selected_timestep_mean(batch, new_log_probs),
            "flash_active_timestep_frac": (advantages != 0).float().mean().detach(),
        }

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

        custom = batch.model_metadata.get("flash_rectification_weights")
        if custom is not None:
            weights = torch.as_tensor(custom, dtype=new_log_probs.dtype)
            if weights.ndim == 1:
                weights = weights[:, None]
            if weights.shape != new_log_probs.shape:
                weights = weights.expand_as(new_log_probs)
            if config.get("normalize", True):
                weights = weights / weights.mean().clamp_min(1e-6)
            return weights

        selected = batch.model_metadata.get("selected_timestep_indices", [0] * new_log_probs.shape[0])
        num_steps = int(batch.model_metadata.get("num_steps", max(1, new_log_probs.shape[1])))
        positions = torch.as_tensor(selected, dtype=new_log_probs.dtype)[:, None]
        if config.get("mode", "scheduler_formula") == "scheduler_formula":
            weights = torch.sqrt((num_steps - positions).clamp_min(1.0) / float(num_steps))
        else:
            raise ValueError(f"Unknown Flash rectification mode: {config.get('mode')}")
        if config.get("normalize", True):
            weights = weights / weights.mean().clamp_min(1e-6)
        return weights.expand_as(new_log_probs)

    @staticmethod
    def _selected_timestep_mean(batch: RolloutBatch, new_log_probs):
        import torch

        selected = batch.model_metadata.get("selected_timestep_indices")
        if selected is None:
            return torch.zeros((), dtype=new_log_probs.dtype, device=new_log_probs.device)
        return torch.as_tensor(selected, dtype=new_log_probs.dtype, device=new_log_probs.device).mean().detach()


ALGORITHMS.register("flash_grpo", FlashGRPOAlgorithm)
