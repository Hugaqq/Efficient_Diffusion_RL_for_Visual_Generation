"""TempFlow-GRPO loss with branch/timestep credit assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visual_rl.core.registry import ALGORITHMS
from visual_rl.core.types import RolloutBatch


@dataclass
class TempFlowGRPOAlgorithm:
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0
    credit_assignment: str = "branch_timestep"
    noise_weighting: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TempFlowGRPOAlgorithm":
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        return cls(
            clip_range=float(config.get("clip_range", 0.001)),
            adv_clip_max=float(config.get("adv_clip_max", 5.0)),
            beta=float(config.get("beta", 0.0)),
            credit_assignment=str(config.get("credit_assignment", "branch_timestep")),
            noise_weighting=dict(config.get("noise_weighting") or {}),
        )

    def compute_loss(self, batch: RolloutBatch, rewards, new_log_probs):
        import torch

        advantages = self._expand_advantages(batch, rewards, new_log_probs)
        advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
        weights = self._noise_weights(batch, new_log_probs).to(new_log_probs.device)
        advantages = advantages * weights
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
            "tempflow_noise_weight_mean": weights.mean().detach(),
            "tempflow_active_timestep_frac": (advantages != 0).float().mean().detach(),
        }

    def _expand_advantages(self, batch: RolloutBatch, rewards, new_log_probs):
        import torch

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
        rewards = rewards.to(new_log_probs.device, dtype=new_log_probs.dtype)
        if rewards.shape == new_log_probs.shape:
            return rewards
        if rewards.ndim != 1:
            raise ValueError(
                "TempFlow advantages must be either per-sample [batch] or per-timestep [batch, steps]."
            )

        if self.credit_assignment == "all":
            return rewards[:, None].expand_as(new_log_probs)

        assigned = torch.zeros_like(new_log_probs)
        branch_indices = self._branch_timestep_indices(batch, new_log_probs)
        for row, index in enumerate(branch_indices):
            if self.credit_assignment == "branch_timestep":
                assigned[row, index] = rewards[row]
            elif self.credit_assignment == "all_after_branch":
                assigned[row, index:] = rewards[row]
            else:
                raise ValueError(f"Unknown TempFlow credit_assignment: {self.credit_assignment}")
        return assigned

    @staticmethod
    def _branch_timestep_indices(batch: RolloutBatch, new_log_probs) -> list[int]:
        indices: list[int] = []
        trajectory_step_indices = batch.model_metadata.get("trajectory_step_indices")
        for row in range(new_log_probs.shape[0]):
            global_index = int(
                batch.metadata[row].get(
                    "branch_step_index",
                    batch.model_metadata.get("branch_step_index", 0),
                )
            )
            if trajectory_step_indices is None:
                index = global_index
            else:
                try:
                    index = list(trajectory_step_indices).index(global_index)
                except ValueError as exc:
                    raise ValueError(
                        f"branch_step_index {global_index} is absent from trajectory_step_indices"
                    ) from exc
            if index < 0 or index >= new_log_probs.shape[1]:
                raise ValueError(
                    f"branch_step_index {index} is outside trajectory length {new_log_probs.shape[1]}"
                )
            indices.append(index)
        return indices

    def _noise_weights(self, batch: RolloutBatch, new_log_probs):
        import torch

        config = self.noise_weighting or {}
        if config.get("enabled", True) is False or config.get("mode", "std_dev_t") in {"none", None}:
            return torch.ones_like(new_log_probs)

        custom = batch.model_metadata.get("noise_weights")
        if custom is not None:
            weights = torch.as_tensor(
                custom,
                dtype=new_log_probs.dtype,
                device=new_log_probs.device,
            )
            if weights.ndim == 1:
                weights = weights[None, :].expand_as(new_log_probs)
            elif weights.shape != new_log_probs.shape:
                weights = weights.expand_as(new_log_probs)
            if config.get("normalize_custom", False):
                weights = weights / weights.mean().clamp_min(1e-6)
            return weights

        steps = new_log_probs.shape[1]
        positions = torch.arange(steps, dtype=new_log_probs.dtype, device=new_log_probs.device)
        if config.get("mode", "std_dev_t") == "std_dev_t":
            weights = torch.sqrt((steps - positions).clamp_min(1.0) / float(steps))
        else:
            raise ValueError(f"Unknown TempFlow noise_weighting mode: {config.get('mode')}")
        weights = weights / weights.mean().clamp_min(1e-6)
        return weights[None, :].expand_as(new_log_probs)


ALGORITHMS.register("tempflow_grpo", TempFlowGRPOAlgorithm)
