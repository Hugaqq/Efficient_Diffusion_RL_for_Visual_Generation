"""Final algorithm-factory and optimizer-plugin contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import math
from typing import ClassVar, Literal, TYPE_CHECKING

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RewardBatch,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
    ValidationCheck,
    ValidationContext,
)

if TYPE_CHECKING:
    import torch

    from visual_rl.configs.schema import OptimizerConfig
    from visual_rl.distributed import DDPStrategy, SingleProcessStrategy
    from visual_rl.optimizers.update_engine import UpdateResult


class PolicyAlgorithm(ABC):
    """Construction contract for one selected policy algorithm."""

    TRAINING_CONTRACT_VERSION: ClassVar[int]
    ADVANTAGE_DTYPE: ClassVar[Literal["float32", "float64"]]
    MIN_GROUP_SIZE: ClassVar[int] = 2

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__}.resolve_params() requires a mapping, "
                f"got {type(raw).__name__}"
            )
        del context
        return FrozenMapping(raw)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        del resolved, context
        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> PolicyAlgorithm:
        del resolved, context
        raise NotImplementedError(f"{cls.__name__} must implement from_config()")

    @classmethod
    def required_capabilities(
        cls,
        resolved_params: Mapping[str, object],
    ) -> frozenset[str]:
        del resolved_params
        return frozenset()

    def close(self) -> None:
        """Release owned resources; pure algorithms are no-op."""


def _resolve_algorithm_params(
    raw: Mapping[str, object],
    context: ResolutionContext,
    *,
    allow_beta: bool,
) -> FrozenMapping:
    """Resolve the one shared clipped-surrogate parameter surface."""

    if not isinstance(raw, Mapping):
        raise TypeError("algorithm params must be a mapping")
    if not isinstance(context, ResolutionContext):
        raise TypeError("context must be a ResolutionContext")
    allowed = {"clip_range", "adv_clip_max"}
    if allow_beta:
        allowed.add("beta")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown algorithm params: {sorted(unknown)}")

    values: dict[str, float] = {
        "clip_range": _finite_number(
            "clip_range", raw.get("clip_range", 0.001)
        ),
        "adv_clip_max": _finite_number(
            "adv_clip_max", raw.get("adv_clip_max", 5.0)
        ),
    }
    if not 0.0 < values["clip_range"] < 1.0:
        raise ValueError("clip_range must satisfy 0 < clip_range < 1")
    if values["adv_clip_max"] <= 0.0:
        raise ValueError("adv_clip_max must be positive")
    if allow_beta:
        values["beta"] = _finite_number("beta", raw.get("beta", 0.0))
        if values["beta"] < 0.0:
            raise ValueError("beta must be non-negative")
    return FrozenMapping(values)


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, not bool")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


class OptimizerPlugin(ABC):
    """Fixed internal update facade; it is not a selectable component."""

    @abstractmethod
    def build_optimizer(
        self,
        trainable_named_parameters: tuple[
            tuple[str, "torch.nn.Parameter"],
            ...,
        ],
        config: OptimizerConfig,
    ) -> "torch.optim.AdamW":
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        *,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: "torch.optim.AdamW",
        scaler: "torch.amp.GradScaler | None",
        context: StepContext,
        strategy: "SingleProcessStrategy | DDPStrategy",
    ) -> UpdateResult:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources owned by the plugin; the final plugin is pure."""
