"""Import-safe contracts shared by the canonical trainer control flow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

__all__ = (
    "IterationIdentity",
    "IterationPrelude",
    "IterationResult",
    "PrepareRunContext",
    "StageValue",
    "TrainerComponent",
    "TrainerState",
    "UnaryStage",
)

PayloadT = TypeVar("PayloadT")


class TrainerState(str, Enum):
    NEW = "new"
    PREPARED = "prepared"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class PrepareRunContext:
    run_id: str
    recipe_id: str
    start_optimizer_step: int
    runtime_facts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("recipe_id must be non-empty")
        if type(self.start_optimizer_step) is not int or self.start_optimizer_step < 0:
            raise ValueError("start_optimizer_step must be a non-negative integer")
        if type(self.runtime_facts) is not tuple:
            raise TypeError("runtime_facts must be a tuple")
        keys = tuple(key for key, _ in self.runtime_facts)
        if len(keys) != len(set(keys)):
            raise ValueError("runtime fact keys must be unique")


@dataclass(frozen=True, slots=True)
class IterationIdentity:
    optimizer_step: int
    source_id: str
    phase_id: str
    row_identities: tuple[str, ...]
    group_ids: tuple[str, ...]
    member_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        for name in ("source_id", "phase_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        lengths = {
            len(self.row_identities),
            len(self.group_ids),
            len(self.member_ids),
        }
        if len(lengths) != 1 or not self.row_identities:
            raise ValueError("iteration row/group/member identities must align")
        if len(self.row_identities) != len(set(self.row_identities)):
            raise ValueError("iteration row identities must be unique")
        if any(type(item) is not int or item < 0 for item in self.member_ids):
            raise ValueError("member ids must be non-negative integers")

    @property
    def batch_size(self) -> int:
        return len(self.row_identities)


@dataclass(frozen=True, slots=True)
class StageValue(Generic[PayloadT]):
    identity: IterationIdentity
    payload: PayloadT

    def __post_init__(self) -> None:
        if not isinstance(self.identity, IterationIdentity):
            raise TypeError("identity must be an IterationIdentity")
        if self.payload is None:
            raise ValueError("stage payload must not be None")


@dataclass(frozen=True, slots=True)
class IterationResult(Generic[PayloadT]):
    optimizer_step: int
    value: StageValue[PayloadT]
    stage_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.value.identity.optimizer_step != self.optimizer_step:
            raise ValueError("result optimizer step does not match stage identity")
        if self.stage_order != (
            "prelude",
            "rollout",
            "reward",
            "advantage",
            "credit",
            "optimize",
        ):
            raise ValueError(
                "iteration stage order is not the canonical six-stage order"
            )


class IterationPrelude(Protocol):
    def build(self, optimizer_step: int) -> StageValue[object]: ...


class UnaryStage(Protocol):
    def __call__(self, value: StageValue[object]) -> StageValue[object]: ...


class TrainerComponent(ABC):
    """Registry-loadable six-stage trainer with one canonical runtime ABI."""

    INTERFACE_VERSION = "1.0"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> object:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> TrainerComponent:
        raise NotImplementedError
