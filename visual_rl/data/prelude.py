"""Transactional typed data plane feeding the canonical trainer interface."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue
from visual_rl.data.group_placement import (
    BoundGroupPlacementLayout,
    GroupPlacementContract,
)
from visual_rl.data.phase_schedule import (
    BatchPhaseBinding,
    PhaseRoute,
    PhaseRouter,
    PhaseScheduleState,
)
from visual_rl.data.source_sampler import (
    MultiSourceSampler,
    SamplerReservation,
    SamplerState,
)
from visual_rl.data.samples import (
    BatchRowContext,
    ExplicitCollator,
    SampleItem,
    StackedSampleBatch,
)

__all__ = (
    "DataPlaneCheckpointPort",
    "DataPlaneCheckpointView",
    "DataPlanePrelude",
    "DataPlanePreludeState",
    "PreludeBatchPayload",
)


_SCHEMA_VERSION = 1
_STATE_KIND = "data_plane_prelude_state"


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(namespace: str, *parts: object) -> str:
    return f"{namespace}-{hashlib.sha256(_canonical({'parts': list(parts)})).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class PreludeBatchPayload:
    """One homogeneous routed batch with its complete placement provenance."""

    samples: StackedSampleBatch
    phase_binding: BatchPhaseBinding
    placement: BoundGroupPlacementLayout

    def __post_init__(self) -> None:
        if not isinstance(self.samples, StackedSampleBatch):
            raise TypeError("samples must be a StackedSampleBatch")
        if not isinstance(self.phase_binding, BatchPhaseBinding):
            raise TypeError("phase_binding must be a BatchPhaseBinding")
        if not isinstance(self.placement, BoundGroupPlacementLayout):
            raise TypeError("placement must be a BoundGroupPlacementLayout")
        rows = tuple(item.batch_row for item in self.placement.rows)
        if self.samples.rows != rows:
            raise ValueError("sample rows must preserve canonical placement order")
        if self.phase_binding.batch_size != self.samples.batch_size:
            raise ValueError("phase binding and sample batch sizes differ")
        if self.placement.layout.optimizer_step != self.route.optimizer_step:
            raise ValueError("placement and phase route optimizer steps differ")
        if self.placement.phase != self.route.phase_id:
            raise ValueError("placement and phase route phase ids differ")
        if any(
            source.dataset_source_id != self.route.source_id
            for source in self.samples.sources
        ):
            raise ValueError("sample source does not match the routed source")
        if any(row.phase != self.route.phase_id for row in self.samples.rows):
            raise ValueError("sample rows are not phase homogeneous")

    @property
    def route(self) -> PhaseRoute:
        return self.phase_binding.route

    @property
    def source_id(self) -> str:
        return self.route.source_id

    @property
    def phase_id(self) -> str:
        return self.route.phase_id

    @property
    def active_rewards(self) -> tuple[str, ...]:
        return self.route.active_rewards


@dataclass(frozen=True, slots=True)
class DataPlanePreludeState:
    """Safe-point state for the next batch produced by this data plane."""

    prelude_id: str
    placement_contract_id: str
    phase_schedule_state: PhaseScheduleState
    sampler_state: SamplerState

    def __post_init__(self) -> None:
        for name in ("prelude_id", "placement_contract_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.phase_schedule_state, PhaseScheduleState):
            raise TypeError("phase_schedule_state must be PhaseScheduleState")
        if not isinstance(self.sampler_state, SamplerState):
            raise TypeError("sampler_state must be SamplerState")

    @property
    def next_optimizer_step(self) -> int:
        return self.phase_schedule_state.next_optimizer_step

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": _STATE_KIND,
            "prelude_id": self.prelude_id,
            "placement_contract_id": self.placement_contract_id,
            "phase_schedule_state": self.phase_schedule_state.to_payload(),
            "sampler_state": self.sampler_state.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class DataPlaneCheckpointView:
    """One internally consistent, side-effect-free next-batch checkpoint view."""

    state: DataPlanePreludeState
    next_route: PhaseRoute
    source_cursors: tuple[tuple[str, int], ...]
    next_prompt_batch_identity: str
    has_open_reservation: bool
    group_geometry_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, DataPlanePreludeState):
            raise TypeError("state must be a DataPlanePreludeState")
        if not isinstance(self.next_route, PhaseRoute):
            raise TypeError("next_route must be a PhaseRoute")
        if self.next_route.optimizer_step != self.state.next_optimizer_step:
            raise ValueError("next_route and state optimizer steps differ")
        if type(self.source_cursors) is not tuple or not self.source_cursors:
            raise ValueError("source_cursors must be a non-empty tuple")
        if self.source_cursors != self.state.sampler_state.cursors:
            raise ValueError("source_cursors must equal the captured sampler state")
        if self.next_route.source_id not in dict(self.source_cursors):
            raise ValueError("next route source is absent from source_cursors")
        for name in ("next_prompt_batch_identity", "group_geometry_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.has_open_reservation) is not bool:
            raise TypeError("has_open_reservation must be bool")
        if self.group_geometry_id != self.state.placement_contract_id:
            raise ValueError("group geometry and placement contract identities differ")

    @property
    def next_optimizer_step(self) -> int:
        return self.state.next_optimizer_step

    @property
    def next_source_id(self) -> str:
        return self.next_route.source_id

    @property
    def next_phase_id(self) -> str:
        return self.next_route.phase_id

    @property
    def active_rewards(self) -> tuple[str, ...]:
        return self.next_route.active_rewards


@runtime_checkable
class DataPlaneCheckpointPort(Protocol):
    """Minimal two-phase-restore port owned by the assembled data plane."""

    @property
    def has_open_reservation(self) -> bool: ...

    @property
    def group_geometry_id(self) -> str: ...

    def capture_checkpoint_view(
        self,
        next_optimizer_step: int,
    ) -> DataPlaneCheckpointView: ...

    def restore_checkpoint_state(
        self,
        payload: Mapping[str, object],
    ) -> DataPlanePreludeState: ...


class DataPlanePrelude:
    """Route, reserve, place, repeat, and collate one optimizer batch.

    This object owns the sampler reservation across the entire Trainer
    iteration.  Only ``commit_iteration`` advances a source cursor.
    """

    def __init__(
        self,
        *,
        phase_schedule: PhaseRouter,
        source_sampler: MultiSourceSampler,
        placement_contract: GroupPlacementContract,
        collator: ExplicitCollator,
    ) -> None:
        if not isinstance(phase_schedule, PhaseRouter):
            raise TypeError("phase_schedule must implement PhaseRouter")
        if not isinstance(source_sampler, MultiSourceSampler):
            raise TypeError("source_sampler must be a MultiSourceSampler")
        if not isinstance(placement_contract, GroupPlacementContract):
            raise TypeError("placement_contract must be a GroupPlacementContract")
        if not isinstance(collator, ExplicitCollator):
            raise TypeError("collator must be an ExplicitCollator")
        if placement_contract.world_size != 1:
            raise NotImplementedError(
                "DataPlanePrelude is single-rank only until rank-local and "
                "accumulation slicing is implemented"
            )
        if set(source_sampler.source_ids) != set(phase_schedule.known_source_ids):
            raise ValueError(
                "source sampler ids must exactly match phase schedule source ids"
            )
        self.phase_schedule = phase_schedule
        self.source_sampler = source_sampler
        self.placement_contract = placement_contract
        self.collator = collator
        identity_payload = {
            "schema_version": _SCHEMA_VERSION,
            "phase_schedule_id": phase_schedule.schedule_id,
            "sampler_id": source_sampler.sampler_id,
            "placement_contract_id": placement_contract.contract_id,
        }
        self._prelude_id = hashlib.sha256(_canonical(identity_payload)).hexdigest()
        self._active_reservation: SamplerReservation | None = None
        self._active_identity: IterationIdentity | None = None

    @property
    def prelude_id(self) -> str:
        return self._prelude_id

    @property
    def has_open_reservation(self) -> bool:
        return (
            self._active_reservation is not None
            or self.source_sampler.has_open_reservation
        )

    @property
    def group_geometry_id(self) -> str:
        return self.placement_contract.contract_id

    def build(self, optimizer_step: int) -> StageValue[PreludeBatchPayload]:
        if self.has_open_reservation:
            raise RuntimeError("the previous data-plane reservation is still open")
        route = self.phase_schedule.route_before_k_repeat(optimizer_step)
        reservation = self.source_sampler.reserve(
            route.source_id,
            self.placement_contract.global_prompt_batch_size,
        )
        try:
            result = self._build_reserved(route=route, reservation=reservation)
        except BaseException:
            if reservation.state == "open":
                reservation.abort()
            raise
        self._active_reservation = reservation
        self._active_identity = result.identity
        return result

    def _build_reserved(
        self,
        *,
        route: PhaseRoute,
        reservation: SamplerReservation,
    ) -> StageValue[PreludeBatchPayload]:
        items = reservation.items
        if len(items) != self.placement_contract.global_prompt_batch_size:
            raise ValueError("sampler reservation returned the wrong prompt count")
        if any(not isinstance(item, SampleItem) for item in items):
            raise TypeError("sampler reservation contains a non-SampleItem value")
        if any(item.source.dataset_source_id != route.source_id for item in items):
            raise ValueError("sampler reservation changed the routed source")

        group_ids = tuple(
            _digest(
                "group",
                self.prelude_id,
                route.optimizer_step,
                route.source_id,
                route.phase_id,
                reservation.start_cursor + position,
                item.source.source_item_id,
            )
            for position, item in enumerate(items)
        )
        layout = self.placement_contract.place(
            group_ids,
            optimizer_step=route.optimizer_step,
        )
        item_by_group = dict(zip(group_ids, items, strict=True))
        absolute_cursor_by_group = {
            group_id: reservation.start_cursor + position
            for position, group_id in enumerate(group_ids)
        }
        rows: list[BatchRowContext] = []
        repeated: list[SampleItem] = []
        for placement in layout.rows:
            item = item_by_group[placement.group_id]
            rows.append(
                BatchRowContext(
                    occurrence_id=_digest(
                        "occurrence",
                        self.prelude_id,
                        route.optimizer_step,
                        placement.group_id,
                        placement.member_id,
                        placement.rank,
                        placement.accumulation_index,
                        absolute_cursor_by_group[placement.group_id],
                    ),
                    group_id=placement.group_id,
                    member_id=placement.member_id,
                    phase=route.phase_id,
                    optimizer_step=route.optimizer_step,
                    source_item_id=item.source.source_item_id,
                )
            )
            repeated.append(item)
        row_values = tuple(rows)
        bound = layout.bind_batch_rows(row_values)
        phase_binding = route.bind_batch(
            len(row_values),
            observed_phase_ids=tuple(row.phase for row in row_values),
        )
        samples = self.collator.collate_samples(tuple(repeated), row_values)
        identity = IterationIdentity(
            optimizer_step=route.optimizer_step,
            source_id=route.source_id,
            phase_id=route.phase_id,
            row_identities=tuple(row.identity for row in row_values),
            group_ids=tuple(row.group_id for row in row_values),
            member_ids=tuple(row.member_id for row in row_values),
        )
        payload = PreludeBatchPayload(
            samples=samples,
            phase_binding=phase_binding,
            placement=bound,
        )
        return StageValue(identity=identity, payload=payload)

    def commit_iteration(self, identity: IterationIdentity) -> None:
        self._finish(identity, commit=True)

    def abort_iteration(self, identity: IterationIdentity) -> None:
        self._finish(identity, commit=False)

    def _finish(self, identity: IterationIdentity, *, commit: bool) -> None:
        if not isinstance(identity, IterationIdentity):
            raise TypeError("identity must be an IterationIdentity")
        reservation = self._active_reservation
        expected = self._active_identity
        if reservation is None or expected is None:
            raise RuntimeError("there is no open data-plane reservation")
        if identity is not expected:
            raise ValueError("iteration identity does not own the reservation")
        if commit:
            reservation.commit()
        else:
            reservation.abort()
        self._active_reservation = None
        self._active_identity = None

    def capture_state(self, next_optimizer_step: int) -> DataPlanePreludeState:
        return self.capture_checkpoint_view(next_optimizer_step).state

    def capture_checkpoint_view(
        self,
        next_optimizer_step: int,
    ) -> DataPlaneCheckpointView:
        """Capture state and the exact next route/window without consuming data."""

        if self.has_open_reservation:
            raise RuntimeError("cannot checkpoint with an open data-plane reservation")
        route = self.phase_schedule.route_before_k_repeat(next_optimizer_step)
        preview = self.source_sampler.preview(
            route.source_id,
            self.placement_contract.global_prompt_batch_size,
        )
        state = DataPlanePreludeState(
            prelude_id=self.prelude_id,
            placement_contract_id=self.placement_contract.contract_id,
            phase_schedule_state=self.phase_schedule.capture_state(next_optimizer_step),
            sampler_state=preview.sampler_state,
        )
        return DataPlaneCheckpointView(
            state=state,
            next_route=route,
            source_cursors=preview.sampler_state.cursors,
            next_prompt_batch_identity=preview.prompt_batch_identity,
            has_open_reservation=False,
            group_geometry_id=self.group_geometry_id,
        )

    def restore_state(
        self,
        payload: Mapping[str, object],
    ) -> DataPlanePreludeState:
        if self._active_reservation is not None:
            raise RuntimeError("cannot restore with an open data-plane reservation")
        if not isinstance(payload, Mapping):
            raise TypeError("data-plane state payload must be a mapping")
        expected = {
            "schema_version",
            "kind",
            "prelude_id",
            "placement_contract_id",
            "phase_schedule_state",
            "sampler_state",
        }
        if set(payload) != expected:
            raise ValueError("data-plane state has an invalid exact key set")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported data-plane state schema_version")
        if payload["kind"] != _STATE_KIND:
            raise ValueError("data-plane state kind is invalid")
        if payload["prelude_id"] != self.prelude_id:
            raise ValueError("data-plane state does not match this prelude")
        if payload["placement_contract_id"] != self.placement_contract.contract_id:
            raise ValueError("data-plane state placement contract changed")
        raw_phase = payload["phase_schedule_state"]
        raw_sampler = payload["sampler_state"]
        if not isinstance(raw_phase, Mapping) or not isinstance(raw_sampler, dict):
            raise TypeError("nested data-plane states have invalid types")
        phase_state = self.phase_schedule.restore_state(raw_phase)
        sampler_state = self.source_sampler.restore_state(raw_sampler)
        return DataPlanePreludeState(
            prelude_id=self.prelude_id,
            placement_contract_id=self.placement_contract.contract_id,
            phase_schedule_state=phase_state,
            sampler_state=sampler_state,
        )

    def restore_checkpoint_state(
        self,
        payload: Mapping[str, object],
    ) -> DataPlanePreludeState:
        return self.restore_state(payload)
