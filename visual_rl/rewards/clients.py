"""Reward client implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.rewards.cache import stable_hash_text


class RewardClient(Protocol):
    name: str

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        ...


@dataclass
class MockRewardClient:
    name: str = "mock"
    mode: str = "prompt_hash"

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        del metadata
        if self.mode == "constant":
            return np.ones(len(prompts), dtype=np.float32), {"mode": self.mode}
        if self.mode == "prompt_media":
            prompt_values = []
            for prompt in prompts:
                digest = stable_hash_text(prompt)
                prompt_values.append((int(digest[:8], 16) % 1000) / 1000.0)
            try:
                import torch

                if isinstance(media, torch.Tensor):
                    media_values = media.float().flatten(1).mean(dim=1).detach().cpu().numpy()
                else:
                    media_values = np.zeros(len(prompts), dtype=np.float32)
            except Exception:  # noqa: BLE001 - mock fallback should be resilient
                media_values = np.zeros(len(prompts), dtype=np.float32)
            values = 0.7 * np.asarray(prompt_values, dtype=np.float32) + 0.3 * media_values.astype(np.float32)
            return values.astype(np.float32), {"mode": self.mode}
        values = []
        for prompt in prompts:
            digest = stable_hash_text(prompt)
            values.append((int(digest[:8], 16) % 1000) / 1000.0)
        return np.asarray(values, dtype=np.float32), {"mode": self.mode}


@dataclass
class RemotePickleRewardClient:
    """Generic pickle-over-HTTP reward client used by the legacy projects."""

    name: str
    url: str
    payload_kind: str = "images"
    timeout: float = 1000.0
    retries: int = 2

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        del metadata
        import pickle
        import requests

        payload = {self.payload_kind: media, "prompts": prompts}
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                response = requests.post(self.url, data=pickle.dumps(payload), timeout=self.timeout)
                response.raise_for_status()
                data = pickle.loads(response.content)
                return np.asarray(data["outputs"], dtype=np.float32), data.get("metadata", {})
            except Exception as exc:  # noqa: BLE001 - attach final error to metadata
                last_error = exc
        raise RuntimeError(f"Reward client {self.name} failed: {last_error}") from last_error


REWARD_CLIENTS.register("mock", MockRewardClient)
REWARD_CLIENTS.register("remote_pickle", RemotePickleRewardClient)
