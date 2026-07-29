"""Final builtin reward-client contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RewardVector,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
    ValidationCheck,
    ValidationContext,
)


class RewardClient(ABC):
    """One synchronous builtin reward component."""

    name: str

    @abstractmethod
    def score(
        self,
        batch: RolloutBatch,
        context: StepContext,
    ) -> RewardVector:
        raise NotImplementedError

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
    ) -> RewardClient:
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
        """Release resources owned by this client; pure clients are no-op."""
