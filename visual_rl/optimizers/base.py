"""Optimizer plugin interface."""

from abc import ABC, abstractmethod
from typing import Any

from visual_rl.core.types import RolloutBatch, RewardBatch
from visual_rl.model_adapters.base import ModelAdapter


class OptimizerPlugin(ABC):
    """Own one complete policy update while the runner owns orchestration."""

    @abstractmethod
    def build_optimizer(self, parameters: Any, train_config: Any) -> Any:
        """Return an optimizer with update and checkpoint state methods."""

        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        adapter: ModelAdapter,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        context: dict[str, Any],
    ) -> dict[str, float]:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(
                f"{type(self).__name__} does not define persistent plugin state."
            )
