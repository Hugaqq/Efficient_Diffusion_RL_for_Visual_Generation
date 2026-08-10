"""Immutable optimizer-step phase routing for data and reward selection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from visual_rl.core.contracts import RewardRouteSpec

PHASE_SCHEDULE_SCHEMA_VERSION = 1
PHASE_SCHEDULE_STATE_SCHEMA_VERSION = 1
_SCHEDULE_KIND = "periodic_phase_schedule"
_STATE_KIND = "phase_schedule_state"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value.strip() != value or "\r" in value or "\n" in value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _non_negative_step(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_count(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _identifier_set(value: Any, *, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of identifiers")
    try:
        items = frozenset(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of identifiers") from exc
    if not items:
        raise ValueError(f"{field_name} must not be empty")
    for item in items:
        _identifier(item, field_name=f"{field_name} item")
    return items


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class PhaseDefinition:
    """One explicit, half-open interval inside a periodic optimizer cycle."""

    phase_id: str
    start_offset: int
    end_offset: int
    source_id: str
    active_rewards: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.phase_id, field_name="phase_id")
        _identifier(self.source_id, field_name="source_id")
        if type(self.start_offset) is not int or self.start_offset < 0:
            raise ValueError("start_offset must be a non-negative integer")
        if type(self.end_offset) is not int or self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if type(self.active_rewards) is not tuple or not self.active_rewards:
            raise ValueError("active_rewards must be a non-empty tuple")
        for reward_id in self.active_rewards:
            _identifier(reward_id, field_name="active_rewards item")
        if len(set(self.active_rewards)) != len(self.active_rewards):
            raise ValueError("active_rewards must not contain duplicates")

    @property
    def duration_steps(self) -> int:
        return self.end_offset - self.start_offset

    def to_payload(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_id": self.source_id,
            "active_rewards": list(self.active_rewards),
        }


@dataclass(frozen=True, slots=True)
class PhaseRoute:
    """The single source/reward route selected before K-repeat expansion."""

    optimizer_step: int
    cycle_index: int
    cycle_offset: int
    phase_id: str
    source_id: str
    active_rewards: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_negative_step(self.optimizer_step, field_name="optimizer_step")
        _non_negative_step(self.cycle_index, field_name="cycle_index")
        _non_negative_step(self.cycle_offset, field_name="cycle_offset")
        _identifier(self.phase_id, field_name="phase_id")
        _identifier(self.source_id, field_name="source_id")
        if type(self.active_rewards) is not tuple or not self.active_rewards:
            raise ValueError("active_rewards must be a non-empty tuple")
        for reward_id in self.active_rewards:
            _identifier(reward_id, field_name="active_rewards item")
        if len(set(self.active_rewards)) != len(self.active_rewards):
            raise ValueError("active_rewards must not contain duplicates")

    def bind_batch(
        self,
        batch_size: int,
        *,
        observed_phase_ids: tuple[str, ...] | None = None,
    ) -> BatchPhaseBinding:
        """Bind a batch to this one route and reject mixed-phase rows."""

        size = _positive_count(batch_size, field_name="batch_size")
        if observed_phase_ids is not None:
            if type(observed_phase_ids) is not tuple:
                raise TypeError("observed_phase_ids must be a tuple or None")
            if len(observed_phase_ids) != size:
                raise ValueError("observed_phase_ids must contain one id per row")
            if any(item != self.phase_id for item in observed_phase_ids):
                raise ValueError("one batch cannot contain rows from multiple phases")
        return BatchPhaseBinding(route=self, batch_size=size)

    def to_payload(self) -> dict[str, Any]:
        return {
            "optimizer_step": self.optimizer_step,
            "cycle_index": self.cycle_index,
            "cycle_offset": self.cycle_offset,
            "phase_id": self.phase_id,
            "source_id": self.source_id,
            "active_rewards": list(self.active_rewards),
        }


@dataclass(frozen=True, slots=True)
class BatchPhaseBinding:
    """A structurally homogeneous batch binding with exactly one phase route."""

    route: PhaseRoute
    batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.route, PhaseRoute):
            raise TypeError("route must be a PhaseRoute")
        _positive_count(self.batch_size, field_name="batch_size")

    @property
    def phase_id(self) -> str:
        return self.route.phase_id

    @property
    def source_id(self) -> str:
        return self.route.source_id

    @property
    def active_rewards(self) -> tuple[str, ...]:
        return self.route.active_rewards


@dataclass(frozen=True, slots=True)
class PhaseScheduleState:
    """Checkpoint state for the next optimizer step, independent of epochs."""

    schedule_id: str
    next_optimizer_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, str) or not _SHA256_RE.fullmatch(
            self.schedule_id
        ):
            raise ValueError("schedule_id must be a lowercase SHA-256 digest")
        _non_negative_step(
            self.next_optimizer_step,
            field_name="next_optimizer_step",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_SCHEDULE_STATE_SCHEMA_VERSION,
            "kind": _STATE_KIND,
            "schedule_id": self.schedule_id,
            "next_optimizer_step": self.next_optimizer_step,
        }


@runtime_checkable
class PhaseRouter(Protocol):
    """Checkpointable optimizer-step router consumed by the data plane."""

    @property
    def known_source_ids(self) -> frozenset[str]: ...

    @property
    def known_reward_ids(self) -> frozenset[str]: ...

    @property
    def schedule_id(self) -> str: ...

    def route_before_k_repeat(self, optimizer_step: int) -> PhaseRoute: ...

    def capture_state(self, next_optimizer_step: int) -> PhaseScheduleState: ...

    def restore_state(self, payload: Mapping[str, Any]) -> PhaseScheduleState: ...

    def route_from_state(self, state: PhaseScheduleState) -> PhaseRoute: ...


@dataclass(frozen=True, slots=True)
class ImplicitPhaseRouter:
    """Route every step through the sole typed RewardPlan route.

    This is deliberately not a one-step ``PeriodicPhaseSchedule``. Ordinary
    recipes retain ``phase_schedule=None`` in their semantic identity; runtime
    adapts their sole ``RewardRouteSpec`` directly to the data-plane port.
    """

    route: RewardRouteSpec
    _schedule_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.route, RewardRouteSpec):
            raise TypeError("route must be a RewardRouteSpec")
        payload = {
            "schema_version": 1,
            "kind": "implicit_reward_route",
            "route": self.route.to_payload(),
        }
        object.__setattr__(
            self,
            "_schedule_id",
            hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        )

    @property
    def known_source_ids(self) -> frozenset[str]:
        return frozenset({self.route.source_id})

    @property
    def known_reward_ids(self) -> frozenset[str]:
        return frozenset(self.route.logical_reward_ids)

    @property
    def schedule_id(self) -> str:
        return self._schedule_id

    def route_before_k_repeat(self, optimizer_step: int) -> PhaseRoute:
        step = _non_negative_step(optimizer_step, field_name="optimizer_step")
        return PhaseRoute(
            optimizer_step=step,
            cycle_index=0,
            cycle_offset=0,
            phase_id=self.route.phase_id,
            source_id=self.route.source_id,
            active_rewards=self.route.logical_reward_ids,
        )

    def capture_state(self, next_optimizer_step: int) -> PhaseScheduleState:
        return PhaseScheduleState(
            schedule_id=self.schedule_id,
            next_optimizer_step=_non_negative_step(
                next_optimizer_step,
                field_name="next_optimizer_step",
            ),
        )

    def restore_state(self, payload: Mapping[str, Any]) -> PhaseScheduleState:
        state = _restore_phase_schedule_state(payload)
        if state.schedule_id != self.schedule_id:
            raise ValueError("phase schedule state does not match this router")
        return state

    def route_from_state(self, state: PhaseScheduleState) -> PhaseRoute:
        if not isinstance(state, PhaseScheduleState):
            raise TypeError("state must be a PhaseScheduleState")
        if state.schedule_id != self.schedule_id:
            raise ValueError("phase schedule state does not match this router")
        return self.route_before_k_repeat(state.next_optimizer_step)


@dataclass(frozen=True, slots=True)
class PeriodicPhaseSchedule:
    """A gap-free periodic schedule selected solely by optimizer step."""

    phases: tuple[PhaseDefinition, ...]
    known_source_ids: frozenset[str]
    known_reward_ids: frozenset[str]
    _cycle_length: int = field(init=False, repr=False)
    _schedule_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.phases) is not tuple or not self.phases:
            raise ValueError("phases must be a non-empty tuple")
        if any(not isinstance(phase, PhaseDefinition) for phase in self.phases):
            raise TypeError("phases must contain only PhaseDefinition values")
        sources = _identifier_set(
            self.known_source_ids,
            field_name="known_source_ids",
        )
        rewards = _identifier_set(
            self.known_reward_ids,
            field_name="known_reward_ids",
        )
        ordered = tuple(
            sorted(
                self.phases,
                key=lambda item: (item.start_offset, item.end_offset, item.phase_id),
            )
        )
        phase_ids = tuple(phase.phase_id for phase in ordered)
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("phase_id values must be unique")
        if ordered[0].start_offset != 0:
            raise ValueError("the first phase must start at offset 0")
        for previous, current in pairwise(ordered):
            if current.start_offset < previous.end_offset:
                raise ValueError("phase intervals must not overlap")
            if current.start_offset > previous.end_offset:
                raise ValueError("phase intervals must not contain gaps")
        for phase in ordered:
            if phase.source_id not in sources:
                raise ValueError(
                    f"phase {phase.phase_id!r} uses unknown source {phase.source_id!r}"
                )
            unknown_rewards = set(phase.active_rewards).difference(rewards)
            if unknown_rewards:
                raise ValueError(
                    f"phase {phase.phase_id!r} uses unknown rewards "
                    f"{sorted(unknown_rewards)}"
                )

        object.__setattr__(self, "phases", ordered)
        object.__setattr__(self, "known_source_ids", sources)
        object.__setattr__(self, "known_reward_ids", rewards)
        object.__setattr__(self, "_cycle_length", ordered[-1].end_offset)
        digest = hashlib.sha256(_canonical_json_bytes(self.to_payload())).hexdigest()
        object.__setattr__(self, "_schedule_id", digest)

    @property
    def cycle_length(self) -> int:
        return self._cycle_length

    @property
    def schedule_id(self) -> str:
        return self._schedule_id

    def route_before_k_repeat(self, optimizer_step: int) -> PhaseRoute:
        """Resolve one iteration route before any prompt is repeated."""

        return self.resolve(optimizer_step)

    def resolve(self, optimizer_step: int) -> PhaseRoute:
        step = _non_negative_step(optimizer_step, field_name="optimizer_step")
        cycle_index, cycle_offset = divmod(step, self.cycle_length)
        phase = next(
            item
            for item in self.phases
            if item.start_offset <= cycle_offset < item.end_offset
        )
        return PhaseRoute(
            optimizer_step=step,
            cycle_index=cycle_index,
            cycle_offset=cycle_offset,
            phase_id=phase.phase_id,
            source_id=phase.source_id,
            active_rewards=phase.active_rewards,
        )

    def capture_state(self, next_optimizer_step: int) -> PhaseScheduleState:
        return PhaseScheduleState(
            schedule_id=self.schedule_id,
            next_optimizer_step=_non_negative_step(
                next_optimizer_step,
                field_name="next_optimizer_step",
            ),
        )

    def restore_state(self, payload: Mapping[str, Any]) -> PhaseScheduleState:
        state = _restore_phase_schedule_state(payload)
        if state.schedule_id != self.schedule_id:
            raise ValueError("phase schedule state does not match this schedule")
        return state

    def route_from_state(self, state: PhaseScheduleState) -> PhaseRoute:
        if not isinstance(state, PhaseScheduleState):
            raise TypeError("state must be a PhaseScheduleState")
        if state.schedule_id != self.schedule_id:
            raise ValueError("phase schedule state does not match this schedule")
        return self.route_before_k_repeat(state.next_optimizer_step)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_SCHEDULE_SCHEMA_VERSION,
            "kind": _SCHEDULE_KIND,
            "cycle_length": self.phases[-1].end_offset,
            "phases": [phase.to_payload() for phase in self.phases],
        }


def _restore_phase_schedule_state(
    payload: Mapping[str, Any],
) -> PhaseScheduleState:
    if not isinstance(payload, Mapping):
        raise TypeError("phase schedule state payload must be a mapping")
    expected = {
        "schema_version",
        "kind",
        "schedule_id",
        "next_optimizer_step",
    }
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        unknown = sorted(set(payload).difference(expected))
        raise ValueError(
            "phase schedule state has an invalid exact key set: "
            f"missing={missing}, unknown={unknown}"
        )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != PHASE_SCHEDULE_STATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported phase schedule state schema_version")
    if payload["kind"] != _STATE_KIND:
        raise ValueError("phase schedule state kind is invalid")
    return PhaseScheduleState(
        schedule_id=payload["schedule_id"],
        next_optimizer_step=payload["next_optimizer_step"],
    )


def world_r1_release_phase_schedule(
    *,
    main_source_id: str = "main",
    dynamic_source_id: str = "dynamic",
    general_reward_id: str = "reward_general",
    reward_3d_id: str = "reward_3d",
    main_steps: int = 100,
    dynamic_steps: int = 50,
) -> PeriodicPhaseSchedule:
    """Build the released World-R1 main/dynamic 100/50 phase cycle."""

    main_source = _identifier(main_source_id, field_name="main_source_id")
    dynamic_source = _identifier(
        dynamic_source_id,
        field_name="dynamic_source_id",
    )
    general_reward = _identifier(
        general_reward_id,
        field_name="general_reward_id",
    )
    reward_3d = _identifier(reward_3d_id, field_name="reward_3d_id")
    if main_source == dynamic_source:
        raise ValueError("World-R1 main and dynamic source ids must differ")
    if general_reward == reward_3d:
        raise ValueError("World-R1 general and 3D reward ids must differ")
    main_duration = _positive_count(main_steps, field_name="main_steps")
    dynamic_duration = _positive_count(dynamic_steps, field_name="dynamic_steps")
    cycle_end = main_duration + dynamic_duration
    return PeriodicPhaseSchedule(
        phases=(
            PhaseDefinition(
                phase_id="main",
                start_offset=0,
                end_offset=main_duration,
                source_id=main_source,
                active_rewards=(general_reward, reward_3d),
            ),
            PhaseDefinition(
                phase_id="dynamic",
                start_offset=main_duration,
                end_offset=cycle_end,
                source_id=dynamic_source,
                active_rewards=(general_reward,),
            ),
        ),
        known_source_ids=frozenset({main_source, dynamic_source}),
        known_reward_ids=frozenset({general_reward, reward_3d}),
    )


__all__ = (
    "PHASE_SCHEDULE_SCHEMA_VERSION",
    "PHASE_SCHEDULE_STATE_SCHEMA_VERSION",
    "BatchPhaseBinding",
    "ImplicitPhaseRouter",
    "PeriodicPhaseSchedule",
    "PhaseDefinition",
    "PhaseRoute",
    "PhaseRouter",
    "PhaseScheduleState",
    "world_r1_release_phase_schedule",
)
