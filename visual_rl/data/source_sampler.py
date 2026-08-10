"""Deterministic per-source cursors with transactional iteration reservations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import random
import re
from typing import Literal

from visual_rl.data.samples import SampleItem

__all__ = (
    "MultiSourceSampler",
    "SamplerReservation",
    "SamplerPreview",
    "SamplerState",
    "SourceSequence",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SamplingStrategy = Literal["sequential", "deterministic_shuffle"]


@dataclass(frozen=True, slots=True)
class SourceSequence:
    source_id: str
    revision: str
    items: tuple[SampleItem, ...] = field(compare=False, repr=False)
    strategy: SamplingStrategy = "sequential"
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("source_id", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if type(self.items) is not tuple or not self.items:
            raise ValueError("source items must be a non-empty tuple")
        if any(not isinstance(item, SampleItem) for item in self.items):
            raise TypeError("source items must contain only SampleItem values")
        identities = tuple(item.source.source_item_id for item in self.items)
        if len(identities) != len(set(identities)):
            raise ValueError("source item identities must be unique")
        for item in self.items:
            item.validate()
            if item.source.dataset_source_id != self.source_id:
                raise ValueError("sample dataset_source_id must match SourceSequence")
            if item.source.dataset_revision != self.revision:
                raise ValueError("sample dataset_revision must match SourceSequence")
        if self.strategy not in {"sequential", "deterministic_shuffle"}:
            raise ValueError("unknown source sampling strategy")
        if (
            type(self.shuffle_seed) is not int
            or not 0 <= self.shuffle_seed <= 0xFFFF_FFFF
        ):
            raise ValueError("shuffle_seed must be a uint32 integer")

    def item_at_absolute_cursor(self, cursor: int) -> SampleItem:
        if type(cursor) is not int or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        cycle, position = divmod(cursor, len(self.items))
        if self.strategy == "sequential":
            index = position
        else:
            order = list(range(len(self.items)))
            seed_payload = f"{self.shuffle_seed}:{self.source_id}:{cycle}".encode()
            cycle_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:8], "big"
            )
            random.Random(cycle_seed).shuffle(order)
            index = order[position]
        return self.items[index]

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "revision": self.revision,
            "item_ids": [item.source.source_item_id for item in self.items],
            "strategy": self.strategy,
            "shuffle_seed": self.shuffle_seed,
        }


@dataclass(frozen=True, slots=True)
class SamplerState:
    sampler_id: str
    cursors: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sampler_id, str) or not _SHA256.fullmatch(
            self.sampler_id
        ):
            raise ValueError("sampler_id must be a lowercase SHA-256 digest")
        if type(self.cursors) is not tuple or not self.cursors:
            raise ValueError("sampler cursors must be a non-empty tuple")
        source_ids = tuple(source_id for source_id, _ in self.cursors)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("sampler source cursor ids must be unique")
        if any(
            not isinstance(source_id, str)
            or not source_id
            or type(cursor) is not int
            or cursor < 0
            for source_id, cursor in self.cursors
        ):
            raise ValueError("sampler cursors must be non-negative integers")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "sampler_id": self.sampler_id,
            "cursors": [
                {"source_id": source_id, "cursor": cursor}
                for source_id, cursor in self.cursors
            ],
        }


@dataclass(frozen=True, slots=True)
class SamplerPreview:
    """Read-only next-item window tied to one immutable sampler snapshot."""

    sampler_state: SamplerState
    source_id: str
    start_cursor: int
    items: tuple[SampleItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sampler_state, SamplerState):
            raise TypeError("sampler_state must be a SamplerState")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be non-empty")
        if type(self.start_cursor) is not int or self.start_cursor < 0:
            raise ValueError("start_cursor must be a non-negative integer")
        if type(self.items) is not tuple or not self.items:
            raise ValueError("items must be a non-empty tuple")
        if any(not isinstance(item, SampleItem) for item in self.items):
            raise TypeError("items must contain only SampleItem values")
        cursor_by_source = dict(self.sampler_state.cursors)
        if self.source_id not in cursor_by_source:
            raise ValueError("source_id is absent from sampler_state")
        if cursor_by_source[self.source_id] != self.start_cursor:
            raise ValueError("start_cursor does not match sampler_state")
        if any(item.source.dataset_source_id != self.source_id for item in self.items):
            raise ValueError("preview items do not match source_id")

    @property
    def prompt_batch_identity(self) -> str:
        """Order-sensitive identity for the exact next prompt window."""

        payload = {
            "schema_version": 1,
            "kind": "sampler_prompt_batch_preview",
            "sampler_id": self.sampler_state.sampler_id,
            "source_id": self.source_id,
            "start_cursor": self.start_cursor,
            "items": [
                {
                    "task_type": item.TASK_TYPE,
                    "prompt": item.prompt,
                    "source": item.source.serialize(),
                }
                for item in self.items
            ],
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


class SamplerReservation:
    """One provisional source batch committed only after a successful update."""

    __slots__ = (
        "_owner",
        "_state",
        "end_cursor",
        "items",
        "source_id",
        "start_cursor",
    )

    def __init__(
        self,
        owner: "MultiSourceSampler",
        source_id: str,
        start_cursor: int,
        items: tuple[SampleItem, ...],
    ) -> None:
        self._owner = owner
        self._state = "open"
        self.source_id = source_id
        self.start_cursor = start_cursor
        self.end_cursor = start_cursor + len(items)
        self.items = items

    @property
    def state(self) -> str:
        return self._state

    def commit(self) -> None:
        self._owner._finish(self, commit=True)

    def abort(self) -> None:
        self._owner._finish(self, commit=False)

    def __enter__(self) -> "SamplerReservation":
        if self._state != "open":
            raise RuntimeError("reservation is no longer open")
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if self._state == "open":
            self._owner._finish(self, commit=exc_type is None)


class MultiSourceSampler:
    """Own one independent absolute cursor per resolved dataset source."""

    def __init__(self, sources: tuple[SourceSequence, ...]) -> None:
        if type(sources) is not tuple or not sources:
            raise ValueError("sources must be a non-empty tuple")
        if any(not isinstance(item, SourceSequence) for item in sources):
            raise TypeError("sources must contain SourceSequence values")
        ordered = tuple(sorted(sources, key=lambda item: item.source_id))
        source_ids = tuple(item.source_id for item in ordered)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source ids must be unique")
        self._sources = {item.source_id: item for item in ordered}
        self._cursors = {item.source_id: 0 for item in ordered}
        self._open: SamplerReservation | None = None
        payload = {
            "schema_version": 1,
            "sources": [item.identity_payload() for item in ordered],
        }
        self._sampler_id = hashlib.sha256(_canonical(payload)).hexdigest()

    @property
    def sampler_id(self) -> str:
        return self._sampler_id

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(self._sources)

    @property
    def has_open_reservation(self) -> bool:
        return self._open is not None

    def preview(self, source_id: str, count: int) -> SamplerPreview:
        """Peek deterministically without advancing a cursor or reserving items."""

        if self._open is not None:
            raise RuntimeError("cannot preview with an open sampler reservation")
        source = self._sources.get(source_id)
        if source is None:
            raise KeyError(f"unknown source id: {source_id!r}")
        if type(count) is not int or count < 1:
            raise ValueError("preview count must be a positive integer")
        state = self.capture_state()
        start = dict(state.cursors)[source_id]
        items = tuple(
            source.item_at_absolute_cursor(start + index) for index in range(count)
        )
        return SamplerPreview(
            sampler_state=state,
            source_id=source_id,
            start_cursor=start,
            items=items,
        )

    def reserve(self, source_id: str, count: int) -> SamplerReservation:
        if self._open is not None:
            raise RuntimeError("a sampler reservation is already open")
        source = self._sources.get(source_id)
        if source is None:
            raise KeyError(f"unknown source id: {source_id!r}")
        if type(count) is not int or count < 1:
            raise ValueError("reservation count must be a positive integer")
        start = self._cursors[source_id]
        items = tuple(
            source.item_at_absolute_cursor(start + index) for index in range(count)
        )
        reservation = SamplerReservation(self, source_id, start, items)
        self._open = reservation
        return reservation

    def capture_state(self) -> SamplerState:
        if self._open is not None:
            raise RuntimeError("cannot checkpoint with an open sampler reservation")
        return SamplerState(
            sampler_id=self.sampler_id,
            cursors=tuple(self._cursors.items()),
        )

    def restore_state(self, payload: dict[str, object]) -> SamplerState:
        if self._open is not None:
            raise RuntimeError("cannot restore with an open sampler reservation")
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "sampler_id",
            "cursors",
        }:
            raise ValueError("sampler state has an invalid exact key set")
        if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
            raise ValueError("sampler state schema_version is unsupported")
        raw_cursors = payload["cursors"]
        if not isinstance(raw_cursors, list) or any(
            not isinstance(item, dict) or set(item) != {"source_id", "cursor"}
            for item in raw_cursors
        ):
            raise ValueError("sampler state cursors are invalid")
        state = SamplerState(
            sampler_id=payload["sampler_id"],
            cursors=tuple((item["source_id"], item["cursor"]) for item in raw_cursors),
        )
        if state.sampler_id != self.sampler_id:
            raise ValueError("sampler state identity does not match configured sources")
        if tuple(source_id for source_id, _ in state.cursors) != self.source_ids:
            raise ValueError("sampler state source order/set does not match")
        self._cursors = dict(state.cursors)
        return state

    def _finish(self, reservation: SamplerReservation, *, commit: bool) -> None:
        if reservation is not self._open or reservation._owner is not self:
            raise ValueError("reservation is not owned by this sampler")
        if reservation._state != "open":
            raise RuntimeError("reservation is no longer open")
        if self._cursors[reservation.source_id] != reservation.start_cursor:
            raise RuntimeError("source cursor changed during an open reservation")
        if commit:
            self._cursors[reservation.source_id] = reservation.end_cursor
            reservation._state = "committed"
        else:
            reservation._state = "aborted"
        self._open = None


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
