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

    def save(self, step: int, batch, rewards: Any | None = None) -> dict[str, str]:
        if self.root is None:
            return {}
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
        tensor_path = base.with_suffix(".pt")
        tensor_tmp = tensor_path.with_suffix(".pt.tmp")
        torch.save(tensor_payload, tensor_tmp)
        tensor_tmp.replace(tensor_path)
        media_path = base.with_suffix(".media.pt")
        media_tmp = media_path.with_suffix(".pt.tmp")
        torch.save(batch.media, media_tmp)
        media_tmp.replace(media_path)

        metadata = {
            "prompts": batch.prompts,
            "metadata": batch.metadata,
            "model_metadata": batch.model_metadata,
            "media_path": str(media_path),
        }
        if rewards is not None:
            metadata["reward_metadata"] = rewards.metadata
            metadata["weighted_total"] = rewards.weighted_total.detach().cpu().tolist()
        metadata_path = base.with_suffix(".json")
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        with metadata_tmp.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
        metadata_tmp.replace(metadata_path)
        return {
            "rollout_cache_path": str(tensor_path),
            "media_path": str(media_path),
            "metadata_path": str(metadata_path),
        }

    def truncate_from_step(self, start_step: int) -> None:
        if self.root is None:
            return
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        for path in self.root.glob("batch_*"):
            prefix = path.name.split(".", maxsplit=1)[0]
            try:
                step = int(prefix.removeprefix("batch_"))
            except ValueError:
                continue
            if step >= start_step and path.is_file():
                path.unlink()
