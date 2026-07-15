"""Optimizer plugin interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
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
        context: StepContext,
        *,
        recompute_log_probs: Callable[[RolloutBatch], Any] | None = None,
        gradient_sync_context: Callable[[bool], Any] | None = None,
        reduce_tensor_weighted_mean: Callable[[Any, int], Any] | None = None,
        synchronize_failure: Callable[[bool | BaseException | None], bool]
        | None = None,
        before_optimizer_step: Callable[[], Any] | None = None,
        optimizer_step: Callable[..., Any] | None = None,
    ) -> dict[str, float]:
        """Run one update with optional forward, accumulation, and step routing."""

        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(
                f"{type(self).__name__} does not define persistent plugin state."
            )
