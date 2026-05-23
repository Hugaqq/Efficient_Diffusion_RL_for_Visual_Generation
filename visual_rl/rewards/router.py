"""Unified reward routing, weighting, validity, normalization, and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.core.types import RewardBatch
from visual_rl.rewards.cache import RewardCache, stable_hash_json, stable_hash_media
from visual_rl.rewards.normalize import normalize_tensor


class RewardRouter:
    def __init__(self, config: dict[str, Any], cache_dir: str | Path | None = None):
        self.config = config
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        self.weights = dict(config.get("weights", {}))
        self.normalize_mode = config.get("normalize", "none")
        self.fail_policy = config.get("fail_policy", "invalid")
        self.cache = RewardCache(cache_dir)
        self.clients = {}
        self.client_versions = {}
        for key, client_config in config.get("clients", {}).items():
            name = client_config.get("name", key)
            cls = REWARD_CLIENTS.get(name)
            params = dict(client_config.get("params", {}))
            kwargs = {k: v for k, v in client_config.items() if k not in {"name", "version", "params"}}
            kwargs.update(params)
            self.clients[key] = cls(**kwargs)
            self.client_versions[key] = client_config.get("version", "v1")

        if not self.clients and self.weights:
            for key in self.weights:
                cls = REWARD_CLIENTS.get(key)
                self.clients[key] = cls()

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> RewardBatch:
        import torch

        raw_numpy: dict[str, np.ndarray] = {}
        weighted_numpy: dict[str, np.ndarray] = {}
        merged_metadata: dict[str, Any] = {}
        valid = np.ones(len(prompts), dtype=bool)

        for reward_name, weight in self.weights.items():
            if reward_name not in self.clients:
                raise KeyError(f"No reward client configured for {reward_name!r}")
            cache_key = self._cache_key(
                reward_name,
                self.client_versions.get(reward_name, "v1"),
                prompts,
                metadata,
                media,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                values = np.asarray(cached["values"], dtype=np.float32)
                reward_meta = cached.get("metadata", {})
            else:
                try:
                    values, reward_meta = self.clients[reward_name].score(media, prompts, metadata)
                    self.cache.set(cache_key, {"values": values.tolist(), "metadata": reward_meta})
                except Exception as exc:  # noqa: BLE001 - failure is represented in valid_mask
                    values = np.zeros(len(prompts), dtype=np.float32)
                    valid[:] = False
                    reward_meta = {"error": str(exc)}
                    if self.fail_policy == "raise":
                        raise
            raw_numpy[reward_name] = values.astype(np.float32)
            weighted_numpy[reward_name] = raw_numpy[reward_name] * float(weight)
            merged_metadata[reward_name] = reward_meta

        if weighted_numpy:
            total_np = sum(weighted_numpy.values())
        else:
            total_np = np.zeros(len(prompts), dtype=np.float32)

        weighted_total = torch.as_tensor(total_np, dtype=torch.float32)
        normalized_total = normalize_tensor(weighted_total, self.normalize_mode)
        return RewardBatch(
            raw={key: torch.as_tensor(value, dtype=torch.float32) for key, value in raw_numpy.items()},
            weighted={key: torch.as_tensor(value, dtype=torch.float32) for key, value in weighted_numpy.items()},
            weighted_total=weighted_total,
            normalized_total=normalized_total,
            valid_mask=torch.as_tensor(valid, dtype=torch.bool),
            metadata=merged_metadata,
        )

    @staticmethod
    def _cache_key(
        reward_name: str,
        reward_version: str,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        media: Any,
    ) -> str:
        payload_hash = stable_hash_json({"prompts": prompts, "metadata": metadata})
        media_hash = stable_hash_media(media)
        return f"{reward_name}-{reward_version}-{payload_hash}-{media_hash}"
