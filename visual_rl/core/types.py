"""Shared data contracts used by rollout, rewards, algorithms, and trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RolloutBatch:
    prompts: list[str]
    metadata: list[dict[str, Any]]
    media: Any
    latents: Any
    next_latents: Any
    timesteps: Any
    old_log_probs: Any
    kl: Any | None = None
    branch_ids: Any | None = None
    epoch_tag: int | None = None
    seed: int | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    model_tensors: dict[str, Any] = field(default_factory=dict)

    def validate_lightweight(self, strict: bool = False) -> None:
        if len(self.prompts) != len(self.metadata):
            raise ValueError("prompts and metadata must have the same length")
        if strict:
            self.validate_strict()

    def validate_strict(self) -> None:
        """Validate batch dimensions before expensive model/reward work."""

        batch_size = len(self.prompts)
        self._check_batch_axis("media", self.media, batch_size, allow_scalar=False)
        self._check_batch_axis("latents", self.latents, batch_size, allow_scalar=False)
        self._check_batch_axis("next_latents", self.next_latents, batch_size, allow_scalar=False)
        self._check_batch_axis("timesteps", self.timesteps, batch_size, allow_scalar=False)
        self._check_batch_axis("old_log_probs", self.old_log_probs, batch_size, allow_scalar=False)
        self._check_batch_axis("kl", self.kl, batch_size, allow_scalar=True)
        self._check_batch_axis("branch_ids", self.branch_ids, batch_size, allow_scalar=True)

        old_shape = getattr(self.old_log_probs, "shape", None)
        timestep_shape = getattr(self.timesteps, "shape", None)
        if old_shape is not None and timestep_shape is not None and len(old_shape) >= 2 and len(timestep_shape) >= 2:
            if old_shape[:2] != timestep_shape[:2]:
                raise ValueError(
                    "old_log_probs and timesteps must share [batch, steps] dimensions: "
                    f"{old_shape[:2]} != {timestep_shape[:2]}"
                )

        latent_shape = getattr(self.latents, "shape", None)
        next_shape = getattr(self.next_latents, "shape", None)
        if latent_shape is not None and next_shape is not None and latent_shape != next_shape:
            raise ValueError(f"latents and next_latents must have the same shape: {latent_shape} != {next_shape}")

    @staticmethod
    def _check_batch_axis(name: str, value: Any, batch_size: int, allow_scalar: bool) -> None:
        if value is None:
            return
        shape = getattr(value, "shape", None)
        if shape is None:
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                if len(value) != batch_size:
                    raise ValueError(f"{name} length must match batch size {batch_size}, got {len(value)}")
            return
        if len(shape) == 0:
            if allow_scalar:
                return
            raise ValueError(f"{name} must have a batch dimension")
        if int(shape[0]) != batch_size:
            raise ValueError(f"{name} batch dimension must be {batch_size}, got {shape[0]}")


@dataclass
class RewardBatch:
    raw: dict[str, Any]
    weighted: dict[str, Any]
    weighted_total: Any
    normalized_total: Any
    valid_mask: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainStepMetrics:
    loss: float
    reward_mean: float
    reward_std: float
    approx_kl: float
    clipfrac: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class CheckpointRef:
    path: Path
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)
