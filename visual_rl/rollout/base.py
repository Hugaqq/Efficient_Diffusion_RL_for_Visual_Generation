"""Rollout engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter


class RolloutEngine(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def sample(self, adapter: ModelAdapter, prompts: list[str], metadata: list[dict[str, Any]]) -> RolloutBatch:
        pass

