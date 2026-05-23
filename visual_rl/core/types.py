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

    def validate_lightweight(self) -> None:
        if len(self.prompts) != len(self.metadata):
            raise ValueError("prompts and metadata must have the same length")


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
