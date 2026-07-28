"""Torch-free public result types for VisualRL's Python API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

from visual_rl.core.types import FrozenMapping, ValidationCheck

__all__ = (
    "AuditReport",
    "RunResult",
    "RunStatus",
    "ValidationReport",
)


def _partition_checks(
    checks: tuple[ValidationCheck, ...],
    level: str,
) -> tuple[ValidationCheck, ...]:
    return tuple(check for check in checks if check.level == level)


def _validate_checks(checks: tuple[ValidationCheck, ...]) -> None:
    if not isinstance(checks, tuple):
        raise TypeError("checks must be a tuple")
    if any(not isinstance(check, ValidationCheck) for check in checks):
        raise TypeError("checks must contain only ValidationCheck values")


class _CheckProperties:
    checks: tuple[ValidationCheck, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[ValidationCheck, ...]:
        return _partition_checks(self.checks, "error")

    @property
    def warnings(self) -> tuple[ValidationCheck, ...]:
        return _partition_checks(self.checks, "warning")


@dataclass(frozen=True)
class ValidationReport(_CheckProperties):
    """Structured output of the one production Preflight implementation."""

    checks: tuple[ValidationCheck, ...]
    runtime_rank: int | None = None
    runtime_world_size: int | None = None

    def __post_init__(self) -> None:
        _validate_checks(self.checks)
        rank_is_none = self.runtime_rank is None
        world_is_none = self.runtime_world_size is None
        if rank_is_none != world_is_none:
            raise ValueError(
                "runtime_rank and runtime_world_size must both be set or both be None"
            )
        if rank_is_none:
            return
        if type(self.runtime_rank) is not int:
            raise TypeError("runtime_rank must be an integer, not bool")
        if type(self.runtime_world_size) is not int:
            raise TypeError("runtime_world_size must be an integer, not bool")
        assert self.runtime_rank is not None
        assert self.runtime_world_size is not None
        if self.runtime_world_size not in (1, 2):
            raise ValueError("runtime_world_size must be 1 or 2")
        if not 0 <= self.runtime_rank < self.runtime_world_size:
            raise ValueError("runtime_rank must satisfy 0 <= rank < world size")


@dataclass(frozen=True)
class RunResult:
    """Authoritative result returned directly by ``ExperimentRunner.run()``."""

    run_id: str
    output_dir: Path
    committed_steps: int
    authoritative_checkpoint: Path
    resolved_config_path: Path
    manifest_path: Path
    metrics_path: Path
    marker_path: Path
    last_metrics: Mapping[str, float | int]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if type(self.committed_steps) is not int:
            raise TypeError("committed_steps must be an integer, not bool")
        if self.committed_steps <= 0:
            raise ValueError("committed_steps must be positive")

        output_dir = _absolute_path(self.output_dir, field_name="output_dir")
        if not output_dir.is_dir():
            raise ValueError("output_dir must be an existing directory")
        object.__setattr__(self, "output_dir", output_dir)
        for name in (
            "authoritative_checkpoint",
            "resolved_config_path",
            "manifest_path",
            "metrics_path",
            "marker_path",
        ):
            path = _absolute_path(getattr(self, name), field_name=name)
            if path != output_dir and output_dir not in path.parents:
                raise ValueError(f"{name} must be located inside output_dir")
            if not path.exists():
                raise ValueError(f"{name} must exist")
            object.__setattr__(self, name, path)

        metrics = FrozenMapping(self.last_metrics)
        required_integer_metrics = (
            "step",
            "sample_count",
            "active_transition_count",
        )
        missing = tuple(key for key in required_integer_metrics if key not in metrics)
        if missing:
            raise ValueError(f"last_metrics is missing required keys: {missing}")
        for key, value in metrics.items():
            if key == "schema_version":
                raise ValueError("last_metrics must not contain schema_version")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"last_metrics[{key!r}] must be a Python int or float"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"last_metrics[{key!r}] must be finite")
            if key not in required_integer_metrics and type(value) is not float:
                raise TypeError(
                    f"last_metrics[{key!r}] must be a finite Python float"
                )
        for integer_key in required_integer_metrics:
            if type(metrics[integer_key]) is not int:
                raise TypeError(f"last_metrics[{integer_key!r}] must be an integer")
        if metrics["step"] != self.committed_steps - 1:
            raise ValueError("last_metrics.step must equal committed_steps - 1")
        if metrics["sample_count"] <= 0:
            raise ValueError("last_metrics.sample_count must be positive")
        if metrics["active_transition_count"] <= 0:
            raise ValueError(
                "last_metrics.active_transition_count must be positive"
            )
        object.__setattr__(self, "last_metrics", metrics)


@dataclass(frozen=True)
class RunStatus(_CheckProperties):
    """Fast, read-only projection of one run's authoritative state."""

    output_dir: Path
    run_id: str | None
    committed_steps: int
    authoritative_checkpoint: Path | None
    resumable: bool
    pending_transaction_count: int
    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        _validate_checks(self.checks)
        object.__setattr__(
            self,
            "output_dir",
            _absolute_path(self.output_dir, field_name="output_dir"),
        )
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id
        ):
            raise ValueError("run_id must be None or a non-empty string")
        for name in ("committed_steps", "pending_transaction_count"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, not bool")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.resumable) is not bool:
            raise TypeError("resumable must be a bool")
        if self.authoritative_checkpoint is not None:
            object.__setattr__(
                self,
                "authoritative_checkpoint",
                _absolute_path(
                    self.authoritative_checkpoint,
                    field_name="authoritative_checkpoint",
                ),
            )


@dataclass(frozen=True)
class AuditReport(_CheckProperties):
    """Deep, read-only audit of the authoritative commit chain."""

    output_dir: Path
    run_id: str | None
    committed_steps: int
    checked_commit_count: int
    checked_artifact_paths: tuple[Path, ...]
    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        _validate_checks(self.checks)
        output_dir = _absolute_path(self.output_dir, field_name="output_dir")
        object.__setattr__(self, "output_dir", output_dir)
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id
        ):
            raise ValueError("run_id must be None or a non-empty string")
        for name in ("committed_steps", "checked_commit_count"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, not bool")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.checked_artifact_paths, tuple):
            raise TypeError("checked_artifact_paths must be a tuple")
        paths = tuple(
            _absolute_path(path, field_name="checked_artifact_paths")
            for path in self.checked_artifact_paths
        )
        object.__setattr__(self, "checked_artifact_paths", paths)


def _absolute_path(value: Path, *, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    return value
