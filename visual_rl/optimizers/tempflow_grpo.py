"""TempFlow-GRPO loss with branch/timestep credit assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from visual_rl.core.registry import ALGORITHMS
from visual_rl.core.types import RolloutBatch


TEMPFLOW_OBJECTIVE_VERSIONS = {
    "legacy",
    "policy_identity_v1",
    "reference_v1",
}
TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT = "sd3_tempflow_v3"


@dataclass
class TempFlowGRPOAlgorithm:
    _PREPARED_NOISE_WEIGHT_KEY = "_visual_rl_tempflow_noise_weights"

    objective_version: str = "legacy"
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0
    credit_assignment: str = "branch_timestep"
    noise_weighting: dict[str, Any] | None = None
    preserve_advantage_dtype: bool = False
    advantage_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.objective_version not in TEMPFLOW_OBJECTIVE_VERSIONS:
            raise ValueError(
                f"Unknown TempFlow objective_version: {self.objective_version!r}"
            )
        if self.objective_version == "legacy":
            return
        if self.beta > 0:
            raise ValueError(
                f"TempFlow {self.objective_version} does not support beta > 0 until the "
                "differentiable current/reference mean KL is implemented"
            )
        weighting = self.noise_weighting or {}
        if (
            weighting.get("enabled", True) is False
            or weighting.get("mode") != "reference_std_dev_t"
        ):
            raise ValueError(
                f"TempFlow {self.objective_version} requires enabled "
                "reference_std_dev_t weighting"
            )
        if self.advantage_dtype != "float64":
            raise ValueError(
                f"TempFlow {self.objective_version} requires advantage_dtype='float64'"
            )
        if self.preserve_advantage_dtype is not True:
            raise ValueError(
                f"TempFlow {self.objective_version} requires "
                "preserve_advantage_dtype=True"
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TempFlowGRPOAlgorithm":
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        else:
            config = dict(config)
        params = config.pop("params", {}) or {}
        if not isinstance(params, dict):
            raise TypeError("TempFlow algorithm params must be a mapping")
        config.update(params)
        return cls(
            objective_version=str(config.get("objective_version", "legacy")),
            clip_range=float(config.get("clip_range", 0.001)),
            adv_clip_max=float(config.get("adv_clip_max", 5.0)),
            beta=float(config.get("beta", 0.0)),
            credit_assignment=str(config.get("credit_assignment", "branch_timestep")),
            noise_weighting=dict(config.get("noise_weighting") or {}),
            advantage_dtype=config.get("advantage_dtype", "float32"),
            preserve_advantage_dtype=config.get("preserve_advantage_dtype", False),
        )

    def compute_loss(self, batch: RolloutBatch, rewards, new_log_probs):
        import torch

        if self.objective_version != "legacy":
            self._validate_strict_batch_provenance(batch)
            self._validate_strict_advantage_dtype(
                rewards,
                name="advantages",
            )
        if new_log_probs.ndim != 2:
            raise ValueError(
                "TempFlow log probabilities must have shape [batch, transitions]"
            )
        advantages, active_mask = self._expand_advantages_and_mask(
            batch,
            rewards,
            new_log_probs,
        )
        if self.objective_version != "legacy":
            active_mask = active_mask & self._transition_mask(
                batch,
                new_log_probs,
            )
            self._validate_strict_advantage_dtype(
                advantages,
                name="expanded advantages",
            )
        advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
        weights = self._prepared_noise_weights(batch, new_log_probs)
        if weights is None:
            weights = self._noise_weights(batch, new_log_probs).to(
                new_log_probs.device
            )
        advantages = advantages * weights
        old_log_probs = batch.old_log_probs.to(
            new_log_probs.device, dtype=new_log_probs.dtype
        )
        if old_log_probs.shape != new_log_probs.shape:
            raise ValueError(
                "TempFlow old/new log probabilities must have identical shapes: "
                f"{tuple(old_log_probs.shape)} != {tuple(new_log_probs.shape)}"
            )

        logprob_delta = new_log_probs - old_log_probs
        safe_logprob_delta = (
            logprob_delta
            if self.objective_version == "legacy"
            else torch.where(
                active_mask,
                logprob_delta,
                torch.zeros_like(logprob_delta),
            )
        )
        ratio = torch.exp(safe_logprob_delta)
        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(
            1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = self._objective_mean(
            torch.maximum(unclipped, clipped),
            active_mask,
            name="PPO policy loss",
        )
        approx_kl = 0.5 * self._objective_mean(
            safe_logprob_delta.square(),
            active_mask,
            name="approx_kl",
        )
        clipped_ratio = (ratio - 1.0).abs() > self.clip_range
        if self.objective_version == "legacy":
            clipfrac = clipped_ratio.float().mean()
        else:
            clipfrac = self._masked_mean(
                clipped_ratio.to(new_log_probs.dtype),
                active_mask,
                name="clipfrac",
            )
        if batch.kl is not None and self.beta > 0:
            kl = torch.as_tensor(
                batch.kl,
                device=new_log_probs.device,
                dtype=new_log_probs.dtype,
            )
            if self.objective_version != "legacy":
                kl = self._broadcast_transition_values(
                    kl,
                    new_log_probs.shape,
                    name="TempFlow KL",
                )
            policy_loss = policy_loss + self.beta * self._objective_mean(
                kl,
                active_mask,
                name="TempFlow KL",
            )
        active_timestep_values = (
            active_mask if self.objective_version != "legacy" else advantages != 0
        )
        return policy_loss, {
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "policy_loss": policy_loss.detach(),
            "tempflow_noise_weight_mean": self._objective_mean(
                weights,
                active_mask,
                name="TempFlow noise weight",
            ).detach(),
            "tempflow_active_timestep_frac": active_timestep_values.float()
            .mean()
            .detach(),
        }

    def prepare_batch(self, batch: RolloutBatch, advantages) -> RolloutBatch:
        import torch

        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim != 2 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError(
                "TempFlow old_log_probs must have shape [batch, transitions]"
            )
        advantage_values = torch.as_tensor(advantages)
        objective_dtype = (
            torch.float64
            if advantage_values.dtype == torch.float64
            else torch.float32
        )
        old_log_probs = old_log_probs.to(dtype=objective_dtype)
        weights = self._noise_weights(batch, old_log_probs)
        model_tensors = dict(batch.model_tensors)
        model_tensors[self._PREPARED_NOISE_WEIGHT_KEY] = weights.detach()
        return batch.replace(model_tensors=model_tensors)

    def reduction_weight(self, batch: RolloutBatch, advantages) -> int:
        import torch

        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim != 2 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError(
                "TempFlow old_log_probs must have shape [batch, transitions]"
            )
        if self.objective_version == "legacy":
            return old_log_probs.numel()
        _, active_mask = self._expand_advantages_and_mask(
            batch,
            advantages,
            old_log_probs,
        )
        active_mask = active_mask & self._transition_mask(batch, old_log_probs)
        return int(active_mask.sum().item())

    def metric_reduction_weight(
        self,
        batch: RolloutBatch,
        advantages,
        metric_name: str,
    ) -> int:
        import torch

        if metric_name == "tempflow_active_timestep_frac":
            return torch.as_tensor(batch.old_log_probs).numel()
        return self.reduction_weight(batch, advantages)

    def _prepared_noise_weights(self, batch, new_log_probs):
        import torch

        weights = batch.model_tensors.get(self._PREPARED_NOISE_WEIGHT_KEY)
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

    def _expand_advantages(self, batch: RolloutBatch, rewards, new_log_probs):
        advantages, _active_mask = self._expand_advantages_and_mask(
            batch,
            rewards,
            new_log_probs,
        )
        return advantages

    def _expand_advantages_and_mask(
        self,
        batch: RolloutBatch,
        rewards,
        new_log_probs,
    ):
        import torch

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards)
        if not rewards.is_floating_point():
            rewards = rewards.to(dtype=new_log_probs.dtype)
        if self.preserve_advantage_dtype:
            rewards = rewards.to(new_log_probs.device)
        else:
            rewards = rewards.to(new_log_probs.device, dtype=new_log_probs.dtype)
        if rewards.shape == new_log_probs.shape:
            return rewards, torch.ones_like(new_log_probs, dtype=torch.bool)
        if rewards.ndim != 1:
            raise ValueError(
                "TempFlow advantages must be either per-sample [batch] or per-timestep [batch, steps]."
            )
        if rewards.shape[0] != new_log_probs.shape[0]:
            raise ValueError(
                "TempFlow per-sample advantage count must match the log-probability "
                f"batch: {rewards.shape[0]} != {new_log_probs.shape[0]}"
            )

        if self.credit_assignment == "all":
            return (
                rewards[:, None].expand_as(new_log_probs),
                torch.ones_like(new_log_probs, dtype=torch.bool),
            )
        if self.credit_assignment not in {"branch_timestep", "all_after_branch"}:
            raise ValueError(
                f"Unknown TempFlow credit_assignment: {self.credit_assignment}"
            )

        assigned = rewards.new_zeros(new_log_probs.shape)
        active_mask = torch.zeros_like(new_log_probs, dtype=torch.bool)
        branch_indices = self._branch_timestep_indices(batch, new_log_probs)
        for row, index in enumerate(branch_indices):
            if self.credit_assignment == "branch_timestep":
                assigned[row, index] = rewards[row]
                active_mask[row, index] = True
            elif self.credit_assignment == "all_after_branch":
                assigned[row, index:] = rewards[row]
                active_mask[row, index:] = True
        return assigned, active_mask

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
                "TempFlow transition_mask must have the same shape as log "
                f"probabilities: {tuple(mask.shape)} != "
                f"{tuple(new_log_probs.shape)}"
            )
        return mask

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
        if config.get("enabled", True) is False or config.get("mode", "std_dev_t") in {
            "none",
            None,
        }:
            return torch.ones_like(new_log_probs)

        if config.get("mode") == "reference_std_dev_t":
            std_dev_t = batch.model_metadata.get("transition_std_dev_t")
            if std_dev_t is None:
                raise ValueError(
                    "TempFlow reference_std_dev_t requires transition_std_dev_t "
                    "from the rollout adapter"
                )
            try:
                weights = torch.as_tensor(
                    std_dev_t,
                    device=new_log_probs.device,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "TempFlow transition_std_dev_t must be numeric"
                ) from exc
            batch_size, steps = new_log_probs.shape
            expected_shape = (batch_size, steps)
            if weights.ndim == 1 and weights.shape[0] == steps:
                weights = weights[None, :].expand(expected_shape)
            elif weights.ndim == 2 and tuple(weights.shape) == (1, steps):
                weights = weights.expand(expected_shape)
            elif weights.ndim != 2 or tuple(weights.shape) != expected_shape:
                raise ValueError(
                    "TempFlow transition_std_dev_t must have shape [steps], "
                    f"[1, steps], or [batch, steps]; got {tuple(weights.shape)} "
                    f"for log probabilities {expected_shape}"
                )
            if weights.dtype == torch.bool or weights.is_complex():
                raise ValueError(
                    "TempFlow transition_std_dev_t must contain finite positive values"
                )
            if not bool(torch.isfinite(weights).all()):
                raise ValueError(
                    "TempFlow transition_std_dev_t must contain only finite values"
                )
            if not bool((weights > 0).all()):
                raise ValueError(
                    "TempFlow transition_std_dev_t must contain only positive values"
                )
            try:
                scale = float(config.get("scale", 2.25))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "TempFlow reference_std_dev_t scale must be finite and positive"
                ) from exc
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(
                    "TempFlow reference_std_dev_t scale must be finite and positive"
                )
            weights = weights.to(dtype=new_log_probs.dtype) * scale
            if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
                raise ValueError(
                    "TempFlow scaled transition_std_dev_t must remain finite and positive"
                )
            return weights

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
        positions = torch.arange(
            steps, dtype=new_log_probs.dtype, device=new_log_probs.device
        )
        if config.get("mode", "std_dev_t") == "std_dev_t":
            weights = torch.sqrt((steps - positions).clamp_min(1.0) / float(steps))
        else:
            raise ValueError(
                f"Unknown TempFlow noise_weighting mode: {config.get('mode')}"
            )
        weights = weights / weights.mean().clamp_min(1e-6)
        return weights[None, :].expand_as(new_log_probs)

    def _validate_strict_batch_provenance(self, batch: RolloutBatch) -> None:
        if self.objective_version == "reference_v1":
            required = {
                "tempflow_reference_mode": True,
                "trajectory_contract_version": TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT,
                "recompute_transformer_training": True,
            }
        elif self.objective_version == "policy_identity_v1":
            required = {
                "tempflow_reference_mode": False,
                "trajectory_contract_version": TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT,
                "recompute_transformer_training": False,
                "branching_mode": "shared_prefix",
            }
        else:  # pragma: no cover - __post_init__ rejects unknown versions
            raise AssertionError(f"unsupported objective {self.objective_version}")
        metadata = batch.model_metadata
        for key, expected in required.items():
            if key not in metadata:
                raise ValueError(
                    f"TempFlow {self.objective_version} batch is missing required provenance "
                    f"{key!r}"
            )
            actual = metadata[key]
            matches = actual is expected if isinstance(expected, bool) else actual == expected
            if not matches:
                raise ValueError(
                    f"TempFlow {self.objective_version} requires batch provenance "
                    f"{key}={expected!r}, got {actual!r}"
                )

    def _validate_strict_advantage_dtype(self, advantages, *, name: str) -> None:
        import torch

        if not isinstance(advantages, torch.Tensor):
            raise TypeError(
                f"TempFlow {self.objective_version} {name} must be a torch.Tensor with "
                "dtype=torch.float64"
            )
        if advantages.dtype != torch.float64:
            raise TypeError(
                f"TempFlow {self.objective_version} {name} must have dtype=torch.float64; "
                f"got {advantages.dtype}"
            )

    def _objective_mean(self, values, active_mask, *, name: str):
        if self.objective_version == "legacy":
            return values.mean()
        return self._masked_mean(values, active_mask, name=name)

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

    @staticmethod
    def _broadcast_transition_values(values, shape, *, name: str):
        import torch

        if tuple(values.shape) == tuple(shape):
            return values
        if values.ndim == 0:
            return values.expand(shape)
        if values.ndim == 1 and values.shape[0] == shape[0]:
            return values[:, None].expand(shape)
        try:
            return torch.broadcast_to(values, shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} cannot broadcast from {tuple(values.shape)} to {tuple(shape)}"
            ) from exc


ALGORITHMS.register("tempflow_grpo", TempFlowGRPOAlgorithm)
