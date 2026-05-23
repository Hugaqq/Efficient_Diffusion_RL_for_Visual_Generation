"""Rollout cache for recovering reward/training work without rerolling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RolloutCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, batch, rewards: Any | None = None) -> None:
        if self.root is None:
            return
        import torch

        base = self.root / f"batch_{step:06d}"
        tensor_payload = {
            "latents": batch.latents,
            "next_latents": batch.next_latents,
            "timesteps": batch.timesteps,
            "old_log_probs": batch.old_log_probs,
            "kl": batch.kl,
            "branch_ids": batch.branch_ids,
            "model_tensors": batch.model_tensors,
        }
        torch.save(tensor_payload, base.with_suffix(".pt"))
        media_path = base.with_suffix(".media.pt")
        torch.save(batch.media, media_path)

        metadata = {
            "prompts": batch.prompts,
            "metadata": batch.metadata,
            "model_metadata": batch.model_metadata,
            "media_path": str(media_path),
        }
        if rewards is not None:
            metadata["reward_metadata"] = rewards.metadata
            metadata["weighted_total"] = rewards.weighted_total.detach().cpu().tolist()
        with base.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
