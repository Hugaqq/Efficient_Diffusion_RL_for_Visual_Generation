"""Optimizer plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from visual_rl.core.types import (
    FrozenMapping,
    RewardBatch,
    RolloutBatch,
    StepContext,
)
from visual_rl.model_adapters.base import ModelAdapter


class PolicyAlgorithm(ABC):
    """Explicit ABC for every builtin policy algorithm (plan stage 2.2).

    Only construction and the three class constants are frozen at this
    stage; the runtime loss interface (``weight_normalization_request()`` /
    ``prepare_loss_inputs()`` / ``diagnostics()``) arrives together with
    ``PolicyLossInputs`` in the stage-4 atomic migration, which also removes
    the legacy ``compute_loss()`` path. Checkpoint contract version,
    advantage dtype and the group-size precondition are read only from these
    class constants — never from a second config field.

    Declared without abstract methods so the existing algorithm dataclasses
    can mix this base in during the cutover without breaking instantiation.
    """

    TRAINING_CONTRACT_VERSION: ClassVar[int]
    ADVANTAGE_DTYPE: ClassVar[Literal["float32", "float64"]]
    MIN_GROUP_SIZE: ClassVar[int] = 2

    # ------------------------------------------------------------------
    # Unified component factory protocol (plan stage 2.2); see
    # ``ModelAdapter`` for the shared contract.
    # ------------------------------------------------------------------

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        """Whitelist/default/validate/canonicalize component params."""

        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__}.resolve_params() requires a mapping, "
                f"got {type(raw).__name__}"
            )
        return FrozenMapping(raw)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, Any],
        context: Any,
    ) -> tuple:
        """Bounded, read-only environment checks; default is no checks."""

        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, Any],
        context: Any,
    ):
        """Construct the runtime component from resolved params."""

        raise NotImplementedError(
            f"{cls.__name__} does not implement from_config() yet"
        )

    @classmethod
    def required_capabilities(cls, resolved_params: Mapping[str, Any]) -> frozenset:
        """Conditional capabilities implied by the component's own params."""

        return frozenset()


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
