"""Model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visual_rl.core.types import RolloutBatch


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        pass

    @abstractmethod
    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        pass

    def save_pretrained(self, output_dir: str) -> None:
        del output_dir

