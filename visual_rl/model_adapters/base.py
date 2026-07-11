"""Model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visual_rl.core.types import RolloutBatch


class ModelAdapter(ABC):
    name: str
    media_type: str

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def named_parameters(self):
        return [
            (f"parameter_{index:06d}", parameter)
            for index, parameter in enumerate(self.parameters())
        ]

    @abstractmethod
    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        raise NotImplementedError

    @abstractmethod
    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        raise NotImplementedError

    def branch_transition_count(self, rollout_config: dict[str, Any]) -> int:
        """Return the number of valid transition indices for branching."""

        return int(rollout_config.get("num_steps", 1))

    @abstractmethod
    def save_pretrained(self, output_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        raise NotImplementedError
