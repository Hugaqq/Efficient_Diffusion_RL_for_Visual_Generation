"""Immutable index plans for memory-bounded policy updates.

An update slot describes one rectangular replay window: a group of rollout
rows and a half-open transition range.  It deliberately owns no trajectory,
advantage, model, optimizer, or tensor view.  Consumers apply the indices to
their existing typed values, compute the slot loss, and backpropagate before
moving to the next slot.

``UpdateSlotPlan`` is constructed from the complete update active mask.  Empty
rectangles are omitted, while the remaining slots are verified to partition
every active ``(row, transition)`` cell exactly once.  The complete active-cell
count is repeated on every slot so a consumer can scale a slot mean by
``slot.active_count / slot.global_active_count`` without consulting mutable
runtime state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = (
    "UpdateSlot",
    "UpdateSlotPlan",
    "UpdateSlotPlanError",
)


class UpdateSlotPlanError(ValueError):
    """An active mask or slot configuration cannot define a safe update."""


def _positive_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _active_mask_cells(
    active_mask: Any,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    import torch

    if not isinstance(active_mask, torch.Tensor):
        raise TypeError("active_mask must be a torch.Tensor")
    if active_mask.layout is not torch.strided:
        raise TypeError("active_mask must use strided tensor layout")
    if active_mask.ndim != 2:
        raise ValueError("active_mask must have shape [B,T]")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must use bool dtype")
    if active_mask.requires_grad or active_mask.grad_fn is not None:
        raise ValueError("active_mask must be detached")
    batch_size, transition_count = (int(item) for item in active_mask.shape)
    if batch_size < 1 or transition_count < 1:
        raise ValueError("active_mask dimensions B and T must be positive")
    cells = tuple(
        sorted(
            (int(row), int(transition))
            for row, transition in torch.nonzero(
                active_mask.detach(),
                as_tuple=False,
            )
            .to(device="cpu")
            .tolist()
        )
    )
    if not cells:
        raise UpdateSlotPlanError("active_mask must contain an active transition")
    return batch_size, transition_count, cells


@dataclass(frozen=True, slots=True)
class UpdateSlot:
    """One non-empty row-by-transition replay rectangle."""

    slot_index: int
    row_indices: tuple[int, ...]
    transition_start: int
    transition_stop: int
    active_count: int
    global_active_count: int
    _slot_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("slot_index must be a non-negative integer")
        if type(self.row_indices) is not tuple or not self.row_indices:
            raise ValueError("row_indices must be a non-empty tuple")
        if any(type(row) is not int or row < 0 for row in self.row_indices):
            raise ValueError("row_indices must contain non-negative integers")
        if len(set(self.row_indices)) != len(self.row_indices):
            raise ValueError("row_indices must not contain duplicates")
        if type(self.transition_start) is not int or self.transition_start < 0:
            raise ValueError("transition_start must be a non-negative integer")
        if (
            type(self.transition_stop) is not int
            or self.transition_stop <= self.transition_start
        ):
            raise ValueError("transition_stop must be greater than transition_start")
        _positive_integer(self.active_count, field_name="active_count")
        _positive_integer(
            self.global_active_count,
            field_name="global_active_count",
        )
        rectangle_size = len(self.row_indices) * self.transition_count
        if self.active_count > rectangle_size:
            raise ValueError("active_count cannot exceed the slot rectangle size")
        if self.active_count > self.global_active_count:
            raise ValueError("active_count cannot exceed global_active_count")
        object.__setattr__(self, "_slot_id", _digest(self.to_payload()))

    @property
    def transition_count(self) -> int:
        """Number of physical transition columns in the slot window."""

        return self.transition_stop - self.transition_start

    @property
    def active_fraction(self) -> float:
        """Exact objective multiplier for a mean over this slot's active cells."""

        return self.active_count / self.global_active_count

    @property
    def slot_id(self) -> str:
        """Stable content identity for this index descriptor."""

        return self._slot_id

    def contains(self, row: int, transition: int) -> bool:
        """Return whether a cell falls in the slot's rectangular index window."""

        return (
            row in self.row_indices
            and self.transition_start <= transition < self.transition_stop
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "slot_index": self.slot_index,
            "row_indices": list(self.row_indices),
            "transition_start": self.transition_start,
            "transition_stop": self.transition_stop,
            "active_count": self.active_count,
            "global_active_count": self.global_active_count,
        }


@dataclass(frozen=True, slots=True)
class UpdateSlotPlan:
    """A deterministic, content-addressed partition of one update active mask.

    Row blocks are the outer iteration and transition windows are the inner
    iteration.  This matches the native parity harness's replay order while
    allowing either full-row slots (``row_microbatch_size=None``) or bounded
    row microbatches.
    """

    batch_size: int
    transition_count: int
    row_order: tuple[int, ...]
    active_cells: tuple[tuple[int, int], ...]
    row_microbatch_size: int | None = None
    transition_window_size: int = 1
    slots: tuple[UpdateSlot, ...] = field(init=False)
    global_active_count: int = field(init=False)
    _configuration_id: str = field(init=False, repr=False, compare=False)
    _active_mask_id: str = field(init=False, repr=False, compare=False)
    _plan_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_integer(self.batch_size, field_name="batch_size")
        _positive_integer(self.transition_count, field_name="transition_count")
        if self.row_microbatch_size is not None:
            row_microbatch_size = _positive_integer(
                self.row_microbatch_size,
                field_name="row_microbatch_size",
            )
            if row_microbatch_size > self.batch_size:
                raise ValueError("row_microbatch_size cannot exceed batch_size")
        _positive_integer(
            self.transition_window_size,
            field_name="transition_window_size",
        )
        if self.transition_window_size > self.transition_count:
            raise ValueError("transition_window_size cannot exceed transition_count")
        if type(self.row_order) is not tuple:
            raise TypeError("row_order must be a tuple")
        if any(type(row) is not int for row in self.row_order):
            raise TypeError("row_order must contain integers, not bool")
        if tuple(sorted(self.row_order)) != tuple(range(self.batch_size)):
            raise ValueError("row_order must be a permutation of range(batch_size)")
        self._validate_active_cells()

        global_active_count = len(self.active_cells)
        slots = self._build_slots(global_active_count=global_active_count)
        self._verify_exact_partition(slots)
        object.__setattr__(self, "global_active_count", global_active_count)
        object.__setattr__(self, "slots", slots)

        object.__setattr__(
            self,
            "_configuration_id",
            _digest(self.configuration_payload()),
        )
        object.__setattr__(
            self,
            "_active_mask_id",
            _digest(self.active_mask_payload()),
        )
        object.__setattr__(self, "_plan_id", _digest(self.to_payload()))

    @classmethod
    def from_active_mask(
        cls,
        active_mask: Any,
        *,
        row_microbatch_size: int | None = None,
        transition_window_size: int = 1,
        row_order: tuple[int, ...] | None = None,
    ) -> UpdateSlotPlan:
        """Freeze a complete ``[B,T]`` bool mask into an exact slot plan."""

        batch_size, transition_count, active_cells = _active_mask_cells(active_mask)
        resolved_order = tuple(range(batch_size)) if row_order is None else row_order
        return cls(
            batch_size=batch_size,
            transition_count=transition_count,
            row_order=resolved_order,
            active_cells=active_cells,
            row_microbatch_size=row_microbatch_size,
            transition_window_size=transition_window_size,
        )

    @property
    def resolved_row_microbatch_size(self) -> int:
        """Physical row-block width after resolving the full-row default."""

        return (
            self.batch_size
            if self.row_microbatch_size is None
            else self.row_microbatch_size
        )

    @property
    def configuration_id(self) -> str:
        """Identity of shape, packing sizes, and row order, excluding the mask."""

        return self._configuration_id

    @property
    def active_mask_id(self) -> str:
        """Device-independent identity of the active-cell set and its shape."""

        return self._active_mask_id

    @property
    def plan_id(self) -> str:
        """Stable identity of configuration, active cells, and derived slots."""

        return self._plan_id

    @property
    def fingerprint(self) -> str:
        """Alias exposing the complete plan identity as a fingerprint."""

        return self._plan_id

    def validate_against(self, active_mask: Any) -> None:
        """Reject a runtime mask that differs from the mask used for planning."""

        batch_size, transition_count, active_cells = _active_mask_cells(active_mask)
        if (batch_size, transition_count) != (
            self.batch_size,
            self.transition_count,
        ):
            raise UpdateSlotPlanError(
                "active_mask shape differs from the planned update shape"
            )
        if active_cells != self.active_cells:
            raise UpdateSlotPlanError(
                "active_mask cells differ from the planned update mask"
            )

    def configuration_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "order": "row_blocks_then_transition_windows",
            "batch_size": self.batch_size,
            "transition_count": self.transition_count,
            "row_order": list(self.row_order),
            "row_microbatch_size": self.row_microbatch_size,
            "transition_window_size": self.transition_window_size,
        }

    def active_mask_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "shape": [self.batch_size, self.transition_count],
            "active_cells": [list(cell) for cell in self.active_cells],
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "configuration": self.configuration_payload(),
            "active_mask": self.active_mask_payload(),
            "global_active_count": self.global_active_count,
            "slots": [slot.to_payload() for slot in self.slots],
        }

    def _validate_active_cells(self) -> None:
        if type(self.active_cells) is not tuple or not self.active_cells:
            raise UpdateSlotPlanError(
                "active_cells must be a non-empty canonical tuple"
            )
        for cell in self.active_cells:
            if (
                type(cell) is not tuple
                or len(cell) != 2
                or any(type(index) is not int for index in cell)
            ):
                raise TypeError(
                    "active_cells must contain (row, transition) integer tuples"
                )
            row, transition = cell
            if not 0 <= row < self.batch_size:
                raise ValueError("active cell row is out of range")
            if not 0 <= transition < self.transition_count:
                raise ValueError("active cell transition is out of range")
        if self.active_cells != tuple(sorted(set(self.active_cells))):
            raise UpdateSlotPlanError(
                "active_cells must be sorted and contain no duplicates"
            )

    def _build_slots(self, *, global_active_count: int) -> tuple[UpdateSlot, ...]:
        active = set(self.active_cells)
        slots: list[UpdateSlot] = []
        row_width = self.resolved_row_microbatch_size
        for row_start in range(0, self.batch_size, row_width):
            rows = self.row_order[row_start : row_start + row_width]
            for transition_start in range(
                0,
                self.transition_count,
                self.transition_window_size,
            ):
                transition_stop = min(
                    transition_start + self.transition_window_size,
                    self.transition_count,
                )
                active_count = sum(
                    (row, transition) in active
                    for row in rows
                    for transition in range(transition_start, transition_stop)
                )
                if active_count == 0:
                    continue
                slots.append(
                    UpdateSlot(
                        slot_index=len(slots),
                        row_indices=rows,
                        transition_start=transition_start,
                        transition_stop=transition_stop,
                        active_count=active_count,
                        global_active_count=global_active_count,
                    )
                )
        return tuple(slots)

    def _verify_exact_partition(self, slots: tuple[UpdateSlot, ...]) -> None:
        if not slots:
            raise UpdateSlotPlanError("slot plan must contain a non-empty slot")
        active = set(self.active_cells)
        covered: list[tuple[int, int]] = []
        for expected_index, slot in enumerate(slots):
            if slot.slot_index != expected_index:
                raise UpdateSlotPlanError("slot indices must be contiguous")
            slot_cells = [
                (row, transition)
                for row in slot.row_indices
                for transition in range(
                    slot.transition_start,
                    slot.transition_stop,
                )
                if (row, transition) in active
            ]
            if len(slot_cells) != slot.active_count:
                raise UpdateSlotPlanError(
                    "slot active_count differs from its active-mask intersection"
                )
            covered.extend(slot_cells)
        if len(covered) != len(set(covered)):
            raise UpdateSlotPlanError(
                "update slots cover an active cell more than once"
            )
        if tuple(sorted(covered)) != self.active_cells:
            raise UpdateSlotPlanError("update slots omit an active cell")
        if sum(slot.active_count for slot in slots) != len(self.active_cells):
            raise UpdateSlotPlanError(
                "slot active counts do not sum to global_active_count"
            )
