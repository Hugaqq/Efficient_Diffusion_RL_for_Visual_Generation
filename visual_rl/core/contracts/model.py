"""Import-safe model capability contracts.

This module owns model/task/numerics vocabulary shared across domains.  It is
standard-library-only and must never import a concrete model, algorithm,
runtime, or composition implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = (
    "ComputePrecision",
    "LatentLayout",
    "MediaKind",
    "ModelDescriptorContract",
    "PredictionType",
    "TaskKind",
    "TimeCoordinate",
    "TrainingMode",
)


class _ValueEnum(str, Enum):
    pass


class TaskKind(_ValueEnum):
    T2I = "t2i"
    T2V = "t2v"
    I2V = "i2v"


class MediaKind(_ValueEnum):
    IMAGE = "image"
    VIDEO = "video"


class LatentLayout(_ValueEnum):
    BCHW = "bchw"
    BCTHW = "bcthw"
    PACKED_SEQUENCE = "packed_sequence"


class PredictionType(_ValueEnum):
    FLOW = "flow"
    EPSILON = "epsilon"
    VELOCITY = "velocity"


class TimeCoordinate(_ValueEnum):
    DISCRETE_TIMESTEP = "discrete_timestep"
    FRACTIONAL_TIMESTEP = "fractional_timestep"
    SIGMA = "sigma"


class TrainingMode(_ValueEnum):
    LORA = "lora"
    FULL = "full"


class ComputePrecision(_ValueEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class ModelDescriptorContract:
    tasks: tuple[TaskKind, ...]
    output_media: tuple[MediaKind, ...]
    latent_layouts: tuple[LatentLayout, ...]
    latent_ranks: tuple[int, ...]
    axis_semantics: tuple[tuple[str, ...], ...]
    prediction_types: tuple[PredictionType, ...]
    time_coordinates: tuple[TimeCoordinate, ...]
    training_modes: tuple[TrainingMode, ...]
    supported_precisions: tuple[ComputePrecision, ...]
    provides_reference_policy: bool | None
    condition_payload_types: tuple[str, ...] = ()
    spatial_stride: tuple[int, int] | None = None
    temporal_stride: int | None = None
    packer_identity: str | None = None
    scheduler_blueprint_schema: str | None = None
    dynamics_binding_family: str | None = None
    schedule_coordinate: TimeCoordinate | None = None
    accepted_replay_state_schema_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _unique("tasks", self.tasks)
        _unique("output_media", self.output_media)
        _unique("latent_layouts", self.latent_layouts)
        _unique("latent_ranks", self.latent_ranks)
        _unique("prediction_types", self.prediction_types)
        _unique("time_coordinates", self.time_coordinates)
        _unique("training_modes", self.training_modes)
        _unique("supported_precisions", self.supported_precisions)
        if any(
            not isinstance(item, ComputePrecision) for item in self.supported_precisions
        ):
            raise TypeError("supported_precisions must contain ComputePrecision values")
        if any(rank <= 0 for rank in self.latent_ranks):
            raise ValueError("latent ranks must be positive")
        if self.spatial_stride is not None and any(
            item <= 0 for item in self.spatial_stride
        ):
            raise ValueError("spatial stride must be positive")
        if self.temporal_stride is not None and self.temporal_stride <= 0:
            raise ValueError("temporal stride must be positive")
        _scheduler_binding_contract(
            scheduler_blueprint_schema=self.scheduler_blueprint_schema,
            dynamics_binding_family=self.dynamics_binding_family,
            schedule_coordinate=self.schedule_coordinate,
            accepted_replay_state_schema_ids=(
                self.accepted_replay_state_schema_ids
            ),
            declared_time_coordinates=self.time_coordinates,
        )

    @property
    def declares_scheduler_binding(self) -> bool:
        """Whether this descriptor carries the complete load-before ABI."""

        return self.scheduler_blueprint_schema is not None


def _unique(name: str, values: tuple[object, ...]) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


_SCHEMA_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


def _schema_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _SCHEMA_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical schema identifier")
    return value


def _scheduler_binding_contract(
    *,
    scheduler_blueprint_schema: str | None,
    dynamics_binding_family: str | None,
    schedule_coordinate: TimeCoordinate | None,
    accepted_replay_state_schema_ids: tuple[str, ...],
    declared_time_coordinates: tuple[TimeCoordinate, ...],
) -> None:
    _unique(
        "accepted_replay_state_schema_ids",
        accepted_replay_state_schema_ids,
    )
    values_present = (
        scheduler_blueprint_schema is not None,
        dynamics_binding_family is not None,
        schedule_coordinate is not None,
        bool(accepted_replay_state_schema_ids),
    )
    if not any(values_present):
        return
    if not all(values_present):
        raise ValueError(
            "scheduler binding fields must be declared together"
        )
    _schema_id("scheduler_blueprint_schema", scheduler_blueprint_schema)
    _schema_id("dynamics_binding_family", dynamics_binding_family)
    for value in accepted_replay_state_schema_ids:
        _schema_id("accepted_replay_state_schema_ids", value)
    if not isinstance(schedule_coordinate, TimeCoordinate):
        raise TypeError("schedule_coordinate must be a TimeCoordinate")
    if schedule_coordinate not in declared_time_coordinates:
        raise ValueError(
            "schedule_coordinate must also appear in time_coordinates"
        )
