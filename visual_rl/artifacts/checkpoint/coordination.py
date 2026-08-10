"""Safe-point proof and injected collective protocols for checkpointing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Protocol

from visual_rl.artifacts.checkpoint.protocol import CheckpointProgress

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_VERSION = 1


class CheckpointSafetyError(RuntimeError):
    """A rank has not reached an iteration checkpoint safe point."""

    def __init__(self, rank: int, violations: tuple[str, ...]) -> None:
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if type(violations) is not tuple or not violations:
            raise ValueError("violations must be a non-empty tuple")
        self.rank = rank
        self.violations = violations
        super().__init__(
            f"rank {rank} is not at a checkpoint safe point: " + "; ".join(violations)
        )


class CheckpointConsensusError(RuntimeError):
    """The gathered rank contracts, progress, or shard set disagree."""


class CheckpointCollectiveBackend(Protocol):
    """The only collective surface used by :class:`CheckpointCoordinator`."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def is_main_process(self) -> bool: ...

    def failure_gate(
        self,
        phase: str,
        failure: BaseException | None,
    ) -> None: ...

    def gather_object(self, value: object, *, dst: int = 0) -> list[object] | None: ...

    def broadcast_object(self, value: object, *, src: int = 0) -> object: ...

    def barrier(self, phase: str) -> None: ...


class SingleProcessCheckpointBackend:
    """Collective identity implementation for deterministic unit/smoke tests."""

    rank = 0
    world_size = 1
    is_main_process = True

    def failure_gate(
        self,
        phase: str,
        failure: BaseException | None,
    ) -> None:
        _phase(phase)
        if failure is not None:
            if not isinstance(failure, BaseException):
                raise TypeError("failure must be an exception or None")
            raise failure

    def gather_object(self, value: object, *, dst: int = 0) -> list[object]:
        if dst != 0:
            raise ValueError("single-process checkpoint gather destination must be 0")
        return [value]

    def broadcast_object(self, value: object, *, src: int = 0) -> object:
        if src != 0:
            raise ValueError("single-process checkpoint broadcast source must be 0")
        return value

    def barrier(self, phase: str) -> None:
        _phase(phase)


class StrategyCheckpointBackend:
    """Adapt the existing VisualRL Strategy without importing torch.distributed."""

    def __init__(self, strategy: object) -> None:
        for name in ("failure_gate", "gather_object", "broadcast_object"):
            if not callable(getattr(strategy, name, None)):
                raise TypeError(f"checkpoint strategy must define {name}()")
        rank = getattr(strategy, "rank", None)
        world_size = getattr(strategy, "world_size", None)
        _rank_and_world_size(rank, world_size)
        self._strategy = strategy

    @property
    def rank(self) -> int:
        return int(self._strategy.rank)  # type: ignore[attr-defined]

    @property
    def world_size(self) -> int:
        return int(self._strategy.world_size)  # type: ignore[attr-defined]

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def failure_gate(
        self,
        phase: str,
        failure: BaseException | None,
    ) -> None:
        self._strategy.failure_gate(phase, failure)  # type: ignore[attr-defined]

    def gather_object(
        self,
        value: object,
        *,
        dst: int = 0,
    ) -> list[object] | None:
        return self._strategy.gather_object(value, dst=dst)  # type: ignore[attr-defined]

    def broadcast_object(self, value: object, *, src: int = 0) -> object:
        return self._strategy.broadcast_object(value, src=src)  # type: ignore[attr-defined]

    def barrier(self, phase: str) -> None:
        """Use gather + failure consensus as an injected object barrier."""

        label = _phase(phase)
        arrivals: list[object] | None = None
        failure: BaseException | None = None
        try:
            arrivals = self.gather_object((label, self.rank), dst=0)
            if self.is_main_process:
                expected = [(label, rank) for rank in range(self.world_size)]
                if arrivals != expected:
                    raise CheckpointConsensusError(
                        f"checkpoint barrier {label!r} has invalid arrivals"
                    )
            elif arrivals is not None:
                raise CheckpointConsensusError(
                    "non-main checkpoint barrier unexpectedly received arrivals"
                )
        except BaseException as exc:
            failure = exc
        self.failure_gate(f"{label}.barrier", failure)
        if failure is not None:
            raise AssertionError(
                "failure_gate returned after barrier failure"
            ) from failure


@dataclass(frozen=True, slots=True)
class CheckpointSafePoint:
    """Rank-local proof that all transient iteration state has been drained."""

    rank: int
    world_size: int
    update_disposition: str
    committed_optimizer_step: int
    open_data_reservations: int
    active_reward_futures: int
    active_dynamics_sessions: int
    gradients_synchronized: bool
    gradient_accumulation_position: int
    poisoned: bool
    group_geometry_id: str

    def __post_init__(self) -> None:
        _rank_and_world_size(self.rank, self.world_size)
        if self.update_disposition not in {
            "committed",
            "accumulating",
            "scaler_skipped",
        }:
            raise ValueError("update_disposition is invalid")
        for name in (
            "committed_optimizer_step",
            "open_data_reservations",
            "active_reward_futures",
            "active_dynamics_sessions",
            "gradient_accumulation_position",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("gradients_synchronized", "poisoned"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        _digest("group_geometry_id", self.group_geometry_id)

    @classmethod
    def from_update_result(
        cls,
        *,
        rank: int,
        world_size: int,
        update_result: object,
        group_geometry_id: str,
        open_data_reservations: int = 0,
        active_reward_futures: int = 0,
        active_dynamics_sessions: int = 0,
        gradients_synchronized: bool = True,
        gradient_accumulation_position: int = 0,
        poisoned: bool = False,
    ) -> "CheckpointSafePoint":
        """Derive commit evidence from the immutable update transaction result."""

        from visual_rl.algorithms.optimization.execution import (
            UpdateTransactionResult,
        )

        if not isinstance(update_result, UpdateTransactionResult):
            raise TypeError("update_result must be an UpdateTransactionResult")
        return cls(
            rank=rank,
            world_size=world_size,
            update_disposition=update_result.disposition.value,
            committed_optimizer_step=update_result.next_optimizer_step,
            open_data_reservations=open_data_reservations,
            active_reward_futures=active_reward_futures,
            active_dynamics_sessions=active_dynamics_sessions,
            gradients_synchronized=gradients_synchronized,
            gradient_accumulation_position=gradient_accumulation_position,
            poisoned=poisoned,
            group_geometry_id=group_geometry_id,
        )

    @classmethod
    def from_policy_update_result(
        cls,
        *,
        rank: int,
        world_size: int,
        policy_update_result: object,
        group_geometry_id: str,
        open_data_reservations: int = 0,
        active_reward_futures: int = 0,
        active_dynamics_sessions: int = 0,
        gradients_synchronized: bool = True,
        gradient_accumulation_position: int = 0,
        poisoned: bool = False,
    ) -> "CheckpointSafePoint":
        """Derive safe-point evidence from the policy kernel's typed result."""

        from visual_rl.algorithms.optimization.kernel import PolicyUpdateResult

        if not isinstance(policy_update_result, PolicyUpdateResult):
            raise TypeError("policy_update_result must be a PolicyUpdateResult")
        return cls.from_update_result(
            rank=rank,
            world_size=world_size,
            update_result=policy_update_result.transaction,
            group_geometry_id=group_geometry_id,
            open_data_reservations=open_data_reservations,
            active_reward_futures=active_reward_futures,
            active_dynamics_sessions=active_dynamics_sessions,
            gradients_synchronized=gradients_synchronized,
            gradient_accumulation_position=gradient_accumulation_position,
            poisoned=poisoned,
        )

    @property
    def safe_point_id(self) -> str:
        return _payload_digest(self.to_payload())

    def violations(self, progress: CheckpointProgress) -> tuple[str, ...]:
        if not isinstance(progress, CheckpointProgress):
            raise TypeError("progress must be CheckpointProgress")
        violations: list[str] = []
        if self.update_disposition != "committed":
            violations.append("the preceding optimizer update did not commit")
        if self.committed_optimizer_step != progress.global_step:
            violations.append(
                "committed optimizer step does not equal checkpoint progress"
            )
        if self.open_data_reservations:
            violations.append("a data reservation is still open")
        if self.active_reward_futures:
            violations.append("a reward future is still active")
        if self.active_dynamics_sessions:
            violations.append("a Dynamics session is still active")
        if not self.gradients_synchronized:
            violations.append("gradients are not synchronized")
        if self.gradient_accumulation_position != 0:
            violations.append("gradient accumulation is between safe points")
        if self.gradient_accumulation_position != (
            progress.gradient_accumulation_position
        ):
            violations.append("accumulation position disagrees with progress")
        if self.poisoned:
            violations.append("the optimizer transaction is poisoned")
        return tuple(violations)

    def assert_ready(self, progress: CheckpointProgress) -> None:
        violations = self.violations(progress)
        if violations:
            raise CheckpointSafetyError(self.rank, violations)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
            "update_disposition": self.update_disposition,
            "committed_optimizer_step": self.committed_optimizer_step,
            "open_data_reservations": self.open_data_reservations,
            "active_reward_futures": self.active_reward_futures,
            "active_dynamics_sessions": self.active_dynamics_sessions,
            "gradients_synchronized": self.gradients_synchronized,
            "gradient_accumulation_position": (self.gradient_accumulation_position),
            "poisoned": self.poisoned,
            "group_geometry_id": self.group_geometry_id,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CheckpointSafePoint":
        if not isinstance(payload, Mapping):
            raise TypeError("safe-point payload must be a mapping")
        expected = set(cls.__dataclass_fields__) | {"schema_version"}
        if set(payload) != expected or payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("safe-point payload has invalid fields or version")
        values = dict(payload)
        values.pop("schema_version")
        return cls(**values)


def _validate_backend(backend: object) -> None:
    _rank_and_world_size(
        getattr(backend, "rank", None),
        getattr(backend, "world_size", None),
    )
    if type(getattr(backend, "is_main_process", None)) is not bool:
        raise TypeError("checkpoint backend is_main_process must be bool")
    if backend.is_main_process != (backend.rank == 0):
        raise ValueError("checkpoint backend main-process identity is invalid")
    for name in ("failure_gate", "gather_object", "broadcast_object", "barrier"):
        if not callable(getattr(backend, name, None)):
            raise TypeError(f"checkpoint backend must define {name}()")


def _rank_and_world_size(rank: object, world_size: object) -> None:
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{name} must not contain path separators")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _phase(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint collective phase must be non-empty")
    return value


def _payload_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "CheckpointCollectiveBackend",
    "CheckpointConsensusError",
    "CheckpointSafePoint",
    "CheckpointSafetyError",
    "SingleProcessCheckpointBackend",
    "StrategyCheckpointBackend",
)
