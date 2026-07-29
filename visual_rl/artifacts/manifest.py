"""Strict v3 sample-manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
from collections.abc import Mapping
from typing import Any

from visual_rl.artifacts.serialization import strict_json_load
from visual_rl.core.types import FrozenMapping, to_plain_dict


SAMPLE_MANIFEST_SCHEMA_VERSION = "3"

_RECORD_FIELDS = (
    "run_id",
    "sample_id",
    "sample_index",
    "step",
    "rank",
    "prompt",
    "media_type",
    "prompt_metadata",
    "seed",
    "rollout_type",
    "timestep_summary",
    "reward_values",
    "media_path",
    "rollout_cache_path",
    "checkpoint_path",
    "model_metadata",
    "prompt_id",
    "group_id",
    "branch_id",
)


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _relative_posix_path(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative POSIX path or None")
    if "\\" in value:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return value


@dataclass(frozen=True)
class SampleRecord:
    """One immutable, JSON-safe row in the v3 sample manifest."""

    run_id: str
    sample_id: str
    sample_index: int
    step: int
    rank: int
    prompt: str
    media_type: str
    prompt_metadata: FrozenMapping
    seed: int
    rollout_type: str
    timestep_summary: FrozenMapping
    reward_values: FrozenMapping
    media_path: str | None
    rollout_cache_path: str | None
    checkpoint_path: str | None
    model_metadata: FrozenMapping
    prompt_id: str
    group_id: str
    branch_id: str | int | None

    def __post_init__(self) -> None:
        _non_empty_string("run_id", self.run_id)
        _non_empty_string("sample_id", self.sample_id)
        _non_negative_int("sample_index", self.sample_index)
        _non_negative_int("step", self.step)
        _non_negative_int("rank", self.rank)
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if self.media_type not in {"image", "video"}:
            raise ValueError("media_type must be 'image' or 'video'")
        _non_negative_int("seed", self.seed)
        if self.rollout_type not in {
            "full_trajectory",
            "single_step",
            "branching",
        }:
            raise ValueError(
                "rollout_type must be full_trajectory, single_step, or branching"
            )
        for name in (
            "prompt_metadata",
            "timestep_summary",
            "reward_values",
            "model_metadata",
        ):
            value = getattr(self, name)
            if not isinstance(value, FrozenMapping):
                object.__setattr__(self, name, FrozenMapping(value))
        _validate_timestep_summary(
            self.rollout_type,
            self.timestep_summary,
        )
        _validate_reward_values(self.reward_values)
        object.__setattr__(
            self,
            "media_path",
            _relative_posix_path("media_path", self.media_path),
        )
        object.__setattr__(
            self,
            "rollout_cache_path",
            _relative_posix_path(
                "rollout_cache_path",
                self.rollout_cache_path,
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_path",
            _relative_posix_path("checkpoint_path", self.checkpoint_path),
        )
        _non_empty_string("prompt_id", self.prompt_id)
        _non_empty_string("group_id", self.group_id)
        if isinstance(self.branch_id, bool) or not isinstance(
            self.branch_id,
            (str, int, type(None)),
        ):
            raise TypeError("branch_id must be str, int, or None")

    def to_plain_dict(self) -> dict[str, Any]:
        projected = to_plain_dict(self)
        if tuple(projected) != _RECORD_FIELDS:
            raise RuntimeError("SampleRecord projection field order drifted")
        return projected

    @classmethod
    def from_dict(cls, data: Any) -> SampleRecord:
        if not isinstance(data, dict):
            raise ValueError("SampleRecord must be a JSON object")
        if set(data) != set(_RECORD_FIELDS):
            missing = [name for name in _RECORD_FIELDS if name not in data]
            unknown = [name for name in data if name not in _RECORD_FIELDS]
            raise ValueError(
                "SampleRecord fields do not match v3 schema: "
                f"missing={missing}, unknown={unknown}"
            )
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid SampleRecord: {error}") from error


@dataclass(frozen=True)
class SampleManifest:
    """Authoritative-chain projection written as ``sample_manifest.json``."""

    run_id: str
    records: tuple[SampleRecord, ...]
    schema_version: str = SAMPLE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_string("run_id", self.run_id)
        if self.schema_version != SAMPLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported SampleManifest schema_version: "
                f"{self.schema_version!r}"
            )
        if type(self.records) is not tuple:
            raise TypeError("SampleManifest records must be a tuple")
        seen: set[str] = set()
        for record in self.records:
            if not isinstance(record, SampleRecord):
                raise TypeError("SampleManifest records must be SampleRecord values")
            if record.run_id != self.run_id:
                raise ValueError("record run_id does not match manifest run_id")
            if record.sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {record.sample_id}")
            seen.add(record.sample_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "records": [record.to_plain_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Any) -> SampleManifest:
        if not isinstance(data, dict):
            raise ValueError("SampleManifest must be a JSON object")
        if set(data) != {"schema_version", "run_id", "records"}:
            raise ValueError(
                "SampleManifest top-level fields must be exactly "
                "schema_version/run_id/records"
            )
        if data["schema_version"] != SAMPLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported SampleManifest schema_version: "
                f"{data['schema_version']!r}"
            )
        if not isinstance(data["records"], list):
            raise ValueError("SampleManifest records must be a list")
        return cls(
            run_id=data["run_id"],
            records=tuple(
                SampleRecord.from_dict(record) for record in data["records"]
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> SampleManifest:
        return cls.from_dict(strict_json_load(path))


def _validate_timestep_summary(
    rollout_type: str,
    summary: Mapping[str, Any],
) -> None:
    expected = {"values", "count"}
    if rollout_type == "single_step":
        expected.add("selected_timestep_index")
    elif rollout_type == "branching":
        expected.update({"branch_step_index", "trajectory_step_index"})
    if set(summary) != expected:
        raise ValueError(
            f"timestep_summary fields do not match {rollout_type} schema"
        )
    values = summary["values"]
    if not isinstance(values, tuple) or not values:
        raise ValueError("timestep_summary.values must be a non-empty sequence")
    _finite_numbers("timestep_summary.values", values)
    if type(summary["count"]) is not int or summary["count"] != len(values):
        raise ValueError("timestep_summary.count must equal len(values)")
    if rollout_type == "single_step":
        _non_negative_int(
            "selected_timestep_index",
            summary["selected_timestep_index"],
        )
    if rollout_type == "branching":
        _non_negative_int("branch_step_index", summary["branch_step_index"])
        trajectory = summary["trajectory_step_index"]
        if not isinstance(trajectory, tuple) or not trajectory:
            raise ValueError(
                "trajectory_step_index must be a non-empty sequence"
            )
        _finite_numbers("trajectory_step_index", trajectory)


def _validate_reward_values(value: Mapping[str, Any]) -> None:
    expected = {
        "raw",
        "weighted",
        "weighted_total",
        "valid",
        "shared_metadata",
        "sample_metadata",
    }
    if set(value) != expected:
        raise ValueError("reward_values fields do not match v3 schema")
    if value["valid"] is not True:
        raise ValueError("reward_values.valid must be true")
    for name in ("raw", "weighted", "shared_metadata", "sample_metadata"):
        if not isinstance(value[name], Mapping):
            raise TypeError(f"reward_values.{name} must be a mapping")
    component_names = tuple(value["raw"])
    if not component_names or any(
        not isinstance(name, str) or not name for name in component_names
    ):
        raise ValueError("reward component names must be non-empty strings")
    for name in ("weighted", "shared_metadata", "sample_metadata"):
        if tuple(value[name]) != component_names:
            raise ValueError(
                "reward_values component mappings must have identical order"
            )
    for name in component_names:
        _finite_number(f"reward_values.raw.{name}", value["raw"][name])
        _finite_number(
            f"reward_values.weighted.{name}",
            value["weighted"][name],
        )
    _finite_number("reward_values.weighted_total", value["weighted_total"])


def _finite_numbers(name: str, values: tuple[Any, ...]) -> None:
    for value in values:
        _finite_number(name, value)


def _finite_number(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must contain finite JSON numbers")
