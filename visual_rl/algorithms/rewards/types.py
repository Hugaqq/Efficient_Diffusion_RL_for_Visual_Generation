"""Immutable resolved reward plans and NumPy-only runtime contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from visual_rl.core.types import FrozenMapping, StepContext


@dataclass(frozen=True, slots=True)
class RewardRuntimeContext:
    """Explicit runtime facts supplied by the composition root.

    Reward execution must never reconstruct a seed, rank, or world size from
    an optimizer step or a torch generator.  Keeping the exact ``StepContext``
    as a first-class value preserves keyed selection and remote request
    identity without coupling clients to the trainer or optimizer.
    """

    step_context: StepContext

    def __post_init__(self) -> None:
        if not isinstance(self.step_context, StepContext):
            raise TypeError("step_context must be a StepContext")


def _identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value.strip() != value or "\r" in value or "\n" in value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _identifier_tuple(
    value: Any,
    *,
    field_name: str,
    non_empty: bool = True,
    unique: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise ValueError(f"{field_name} must be a {qualifier}tuple")
    for item in value:
        _identifier(item, field_name=f"{field_name} item")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"{field_name} entries must be unique")
    return value


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("reward weight must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("reward weight must be finite")
    return resolved


def _score_axis_names(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("score_axis_names must be a tuple")
    _identifier_tuple(
        value,
        field_name="score_axis_names",
        non_empty=False,
        unique=True,
    )
    if any(item != "branch_timestep" for item in value):
        raise ValueError("unknown reward score axis")
    return value


def _readonly_values(
    value: Any,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a numpy.ndarray")
    if value.shape != shape:
        raise ValueError(f"{field_name} must have shape {list(shape)}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{field_name} must use a floating dtype")
    normalized = np.array(value, dtype=np.float64, copy=True, order="C")
    if not bool(np.isfinite(normalized).all()):
        raise ValueError(f"{field_name} must contain only finite values")
    normalized.setflags(write=False)
    return normalized


def _readonly_mask(
    value: Any,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a numpy.ndarray")
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{field_name} must be bool with shape {list(shape)}")
    normalized = np.array(value, dtype=np.bool_, copy=True, order="C")
    normalized.setflags(write=False)
    return normalized


@dataclass(frozen=True, slots=True)
class RewardBatchIdentity:
    """Exact row and payload identities every reward must echo unchanged."""

    source_id: str
    phase_id: str
    batch_row_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    condition_payload_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.source_id, field_name="source_id")
        _identifier(self.phase_id, field_name="phase_id")
        rows = _identifier_tuple(
            self.batch_row_ids,
            field_name="batch_row_ids",
            unique=True,
        )
        batch_size = len(rows)
        for field_name in (
            "sample_ids",
            "trajectory_ids",
            "condition_payload_ids",
            "group_ids",
        ):
            values = _identifier_tuple(
                getattr(self, field_name),
                field_name=field_name,
                unique=field_name in {"sample_ids", "trajectory_ids"},
            )
            if len(values) != batch_size:
                raise ValueError(f"{field_name} must contain one identity per row")

    @property
    def batch_size(self) -> int:
        return len(self.batch_row_ids)


@dataclass(frozen=True, slots=True)
class RewardBatchView:
    """A homogeneous reward input with routing fixed before execution."""

    identity: RewardBatchIdentity
    active_reward_ids: tuple[str, ...]
    payload: Mapping[str, object]
    score_axis_names: tuple[str, ...] = ()
    score_axis_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RewardBatchIdentity):
            raise TypeError("identity must be a RewardBatchIdentity")
        _identifier_tuple(
            self.active_reward_ids,
            field_name="active_reward_ids",
            unique=True,
        )
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        axes = _score_axis_names(self.score_axis_names)
        if type(self.score_axis_sizes) is not tuple or len(
            self.score_axis_sizes
        ) != len(axes):
            raise ValueError("score_axis_sizes must contain one size per score axis")
        if any(type(item) is not int or item < 1 for item in self.score_axis_sizes):
            raise ValueError("score_axis_sizes must contain positive integers")
        copied: dict[str, object] = {}
        for key, value in self.payload.items():
            _identifier(key, field_name="payload key")
            copied[key] = value
        object.__setattr__(self, "payload", MappingProxyType(copied))

    @property
    def batch_size(self) -> int:
        return self.identity.batch_size

    @property
    def score_shape(self) -> tuple[int, ...]:
        return (self.batch_size, *self.score_axis_sizes)

    @property
    def source_id(self) -> str:
        return self.identity.source_id

    @property
    def phase_id(self) -> str:
        return self.identity.phase_id


@dataclass(frozen=True, slots=True)
class PointwiseRewardOutput:
    """One score per row and declared score cell: ``[B, *score_axes]``."""

    identity: RewardBatchIdentity
    values: np.ndarray
    valid_mask: np.ndarray
    score_axis_names: tuple[str, ...] = ()
    execution_provenance: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RewardBatchIdentity):
            raise TypeError("identity must be a RewardBatchIdentity")
        axes = _score_axis_names(self.score_axis_names)
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if self.values.ndim != 1 + len(axes):
            if not axes:
                raise ValueError(f"values must have shape [{self.identity.batch_size}]")
            raise ValueError("values rank must match score_axis_names")
        shape = tuple(self.values.shape)
        if shape[0] != self.identity.batch_size:
            raise ValueError("values first dimension must equal B")
        object.__setattr__(
            self,
            "values",
            _readonly_values(self.values, shape=shape, field_name="values"),
        )
        object.__setattr__(
            self,
            "valid_mask",
            _readonly_mask(self.valid_mask, shape=shape, field_name="valid_mask"),
        )
        if not isinstance(self.execution_provenance, FrozenMapping):
            raise TypeError("execution_provenance must be a FrozenMapping")


@dataclass(frozen=True, slots=True)
class GroupwiseRewardOutput:
    """One score per explicit group: shape ``[G]``, aligned later to ``[B]``."""

    identity: RewardBatchIdentity
    group_ids: tuple[str, ...]
    values: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RewardBatchIdentity):
            raise TypeError("identity must be a RewardBatchIdentity")
        groups = _identifier_tuple(
            self.group_ids,
            field_name="group_ids",
            unique=True,
        )
        size = len(groups)
        object.__setattr__(
            self,
            "values",
            _readonly_values(self.values, shape=(size,), field_name="values"),
        )
        object.__setattr__(
            self,
            "valid_mask",
            _readonly_mask(self.valid_mask, shape=(size,), field_name="valid_mask"),
        )


class PointwiseReward(ABC):
    """Logical reward protocol returning one value for every batch row."""

    @abstractmethod
    def score(
        self,
        *,
        logical_reward_id: str,
        resource: object,
        batch: RewardBatchView,
    ) -> PointwiseRewardOutput:
        raise NotImplementedError


class GroupwiseReward(ABC):
    """Logical reward protocol returning one value for every explicit group."""

    @abstractmethod
    def score_groups(
        self,
        *,
        logical_reward_id: str,
        resource: object,
        batch: RewardBatchView,
        group_ids: tuple[str, ...],
    ) -> GroupwiseRewardOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Named ``[B, *score_axes]`` scores and their resolved weighted sum."""

    identity: RewardBatchIdentity
    component_scores: Mapping[str, np.ndarray]
    weighted_scores: Mapping[str, np.ndarray]
    component_valid_masks: Mapping[str, np.ndarray]
    weighted_total: np.ndarray
    valid_mask: np.ndarray
    resource_identities: Mapping[str, str]
    score_axis_names: tuple[str, ...] = ()
    component_applicable_masks: Mapping[str, np.ndarray] | None = None
    logical_weights: Mapping[str, float] | None = None
    logical_provenance: Mapping[str, FrozenMapping] | None = None
    logical_execution_provenance: Mapping[str, FrozenMapping] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RewardBatchIdentity):
            raise TypeError("identity must be a RewardBatchIdentity")
        axes = _score_axis_names(self.score_axis_names)
        if not isinstance(self.weighted_total, np.ndarray):
            raise TypeError("weighted_total must be a numpy.ndarray")
        if self.weighted_total.ndim != 1 + len(axes):
            raise ValueError("weighted_total rank must match score_axis_names")
        score_shape = tuple(self.weighted_total.shape)
        if score_shape[0] != self.identity.batch_size:
            raise ValueError("weighted_total first dimension must equal B")
        required_mappings = (
            self.component_scores,
            self.weighted_scores,
            self.component_valid_masks,
            self.resource_identities,
        )
        if any(not isinstance(item, Mapping) for item in required_mappings):
            raise TypeError("RewardResult named fields must be mappings")
        names = tuple(self.component_scores)
        if not names:
            raise ValueError("RewardResult requires named component scores")
        if (
            tuple(self.weighted_scores) != names
            or tuple(self.component_valid_masks) != names
            or tuple(self.resource_identities) != names
        ):
            raise ValueError("RewardResult named mappings must have identical order")
        optional_mappings = (
            self.component_applicable_masks,
            self.logical_weights,
            self.logical_provenance,
            self.logical_execution_provenance,
        )
        if any(
            item is not None and not isinstance(item, Mapping)
            for item in optional_mappings
        ):
            raise TypeError(
                "RewardResult optional named fields must be mappings or None"
            )
        for field_name, values in (
            ("component_applicable_masks", self.component_applicable_masks),
            ("logical_weights", self.logical_weights),
            ("logical_provenance", self.logical_provenance),
            (
                "logical_execution_provenance",
                self.logical_execution_provenance,
            ),
        ):
            if values is not None and tuple(values) != names:
                raise ValueError(
                    f"RewardResult {field_name} must have identical logical order"
                )
        component_scores: dict[str, np.ndarray] = {}
        weighted_scores: dict[str, np.ndarray] = {}
        component_masks: dict[str, np.ndarray] = {}
        applicable_masks: dict[str, np.ndarray] = {}
        resources: dict[str, str] = {}
        weights: dict[str, float] = {}
        provenance: dict[str, FrozenMapping] = {}
        execution_provenance: dict[str, FrozenMapping] = {}
        for name in names:
            _identifier(name, field_name="logical reward id")
            component_scores[name] = _readonly_values(
                self.component_scores[name],
                shape=score_shape,
                field_name=f"component_scores[{name!r}]",
            )
            weighted_scores[name] = _readonly_values(
                self.weighted_scores[name],
                shape=score_shape,
                field_name=f"weighted_scores[{name!r}]",
            )
            component_masks[name] = _readonly_mask(
                self.component_valid_masks[name],
                shape=score_shape,
                field_name=f"component_valid_masks[{name!r}]",
            )
            raw_applicable = (
                np.ones(score_shape, dtype=np.bool_)
                if self.component_applicable_masks is None
                else self.component_applicable_masks[name]
            )
            applicable_masks[name] = _readonly_mask(
                raw_applicable,
                shape=score_shape,
                field_name=f"component_applicable_masks[{name!r}]",
            )
            resources[name] = _identifier(
                self.resource_identities[name],
                field_name=f"resource_identities[{name!r}]",
            )
            weights[name] = _weight(
                1.0 if self.logical_weights is None else self.logical_weights[name]
            )
            raw_provenance = (
                FrozenMapping()
                if self.logical_provenance is None
                else self.logical_provenance[name]
            )
            if not isinstance(raw_provenance, FrozenMapping):
                raise TypeError(f"logical_provenance[{name!r}] must be a FrozenMapping")
            provenance[name] = raw_provenance
            raw_execution_provenance = (
                FrozenMapping()
                if self.logical_execution_provenance is None
                else self.logical_execution_provenance[name]
            )
            if not isinstance(raw_execution_provenance, FrozenMapping):
                raise TypeError(
                    f"logical_execution_provenance[{name!r}] must be a FrozenMapping"
                )
            execution_provenance[name] = raw_execution_provenance

            applicable = applicable_masks[name]
            valid = component_masks[name]
            expected_weighted = component_scores[name] * weights[name]
            if not bool(
                np.allclose(
                    expected_weighted,
                    weighted_scores[name],
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    f"weighted_scores[{name!r}] must equal component score "
                    "times logical weight"
                )
            if bool(np.any(valid & ~applicable)):
                raise ValueError(
                    f"component_valid_masks[{name!r}] cannot mark "
                    "non-applicable rows valid"
                )
            if bool(np.any(component_scores[name][~applicable] != 0.0)):
                raise ValueError(
                    f"component_scores[{name!r}] must be zero where not applicable"
                )
            if bool(np.any(weighted_scores[name][~applicable] != 0.0)):
                raise ValueError(
                    f"weighted_scores[{name!r}] must be zero where not applicable"
                )
            evaluated = applicable & valid
            if not bool(np.isfinite(component_scores[name][evaluated]).all()):
                raise ValueError(
                    f"component_scores[{name!r}] must be finite where applicable "
                    "and valid"
                )
        weighted_total = _readonly_values(
            self.weighted_total,
            shape=score_shape,
            field_name="weighted_total",
        )
        valid_mask = _readonly_mask(
            self.valid_mask,
            shape=score_shape,
            field_name="valid_mask",
        )
        expected_total = np.zeros(score_shape, dtype=np.float64)
        expected_mask = np.ones(score_shape, dtype=np.bool_)
        coverage_mask = np.zeros(score_shape, dtype=np.bool_)
        for name in names:
            expected_total += weighted_scores[name]
            applicable = applicable_masks[name]
            expected_mask &= ~applicable | component_masks[name]
            coverage_mask |= applicable
        if not bool(coverage_mask.all()):
            raise ValueError("every reward result row must have an applicable reward")
        if not bool(
            np.allclose(
                expected_total,
                weighted_total,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError("weighted_total must equal the named weighted scores")
        if not bool(np.array_equal(expected_mask, valid_mask)):
            raise ValueError("valid_mask must equal all named component masks")

        object.__setattr__(
            self,
            "component_scores",
            MappingProxyType(component_scores),
        )
        object.__setattr__(
            self,
            "weighted_scores",
            MappingProxyType(weighted_scores),
        )
        object.__setattr__(
            self,
            "component_valid_masks",
            MappingProxyType(component_masks),
        )
        object.__setattr__(
            self,
            "component_applicable_masks",
            MappingProxyType(applicable_masks),
        )
        object.__setattr__(
            self,
            "resource_identities",
            MappingProxyType(resources),
        )
        object.__setattr__(self, "logical_weights", MappingProxyType(weights))
        object.__setattr__(
            self,
            "logical_provenance",
            MappingProxyType(provenance),
        )
        object.__setattr__(
            self,
            "logical_execution_provenance",
            MappingProxyType(execution_provenance),
        )
        object.__setattr__(self, "weighted_total", weighted_total)
        object.__setattr__(self, "valid_mask", valid_mask)


__all__ = (
    "GroupwiseReward",
    "GroupwiseRewardOutput",
    "PointwiseReward",
    "PointwiseRewardOutput",
    "RewardBatchIdentity",
    "RewardBatchView",
    "RewardResult",
)
