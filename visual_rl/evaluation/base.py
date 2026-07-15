"""Minimal held-out evaluation contract for trusted local extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_MAX_ITEMS = 128


def _validate_name(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_NAME_PATTERN.pattern!r}, got {value!r}"
        )
    return value


def _readonly_scalars(values: Mapping[str, Any]) -> Mapping[str, float | int]:
    if not isinstance(values, Mapping) or len(values) > _MAX_ITEMS:
        raise ValueError(f"metrics must be a mapping with at most {_MAX_ITEMS} items")
    canonical: dict[str, float | int] = {}
    for name, value in values.items():
        _validate_name(name, label="metric name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {name!r} must be a finite numeric scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {name!r} must be finite")
        canonical[name] = value
    return MappingProxyType(canonical)


def _readonly_paths(values: Mapping[str, str | Path]) -> Mapping[str, str]:
    if not isinstance(values, Mapping) or len(values) > _MAX_ITEMS:
        raise ValueError(f"artifacts must be a mapping with at most {_MAX_ITEMS} items")
    canonical: dict[str, str] = {}
    for name, value in values.items():
        _validate_name(name, label="artifact name")
        if not isinstance(value, (str, Path)):
            raise TypeError(f"artifact {name!r} must be a path string")
        canonical[name] = str(value)
    return MappingProxyType(canonical)


@dataclass(frozen=True)
class EvaluationContext:
    """Read-only metadata for a held-out evaluation invocation."""

    run_id: str
    output_dir: Path
    evaluation_dir: Path
    name: str
    split_name: str
    content_sha256: str
    seeds: tuple[int, ...]
    prompt_count: int
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        _validate_name(self.name, label="evaluation name")
        if not isinstance(self.split_name, str) or not self.split_name:
            raise ValueError("evaluation split_name must be non-empty")
        if self.prompt_count < 1:
            raise ValueError("evaluation prompt_count must be positive")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))


@dataclass(frozen=True)
class EvaluationResult:
    """Small, JSON-persistable result returned by an :class:`Evaluator`."""

    metrics: Mapping[str, float | int] = field(default_factory=dict)
    artifacts: Mapping[str, str | Path] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            _validate_name(self.name, label="evaluation result name")
        object.__setattr__(self, "metrics", _readonly_scalars(self.metrics))
        object.__setattr__(self, "artifacts", _readonly_paths(self.artifacts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EvaluationResult":
        if not isinstance(values, Mapping):
            raise TypeError("evaluation result JSON must be a mapping")
        return cls(
            name=values.get("name"),
            metrics=values.get("metrics", {}),
            artifacts=values.get("artifacts", {}),
        )


class Evaluator:
    """Trusted local held-out evaluator interface."""

    name = "default"

    def evaluate(
        self,
        *,
        adapter: Any,
        prompts: tuple[str, ...],
        context: EvaluationContext,
    ) -> EvaluationResult:
        raise NotImplementedError


__all__ = ["EvaluationContext", "EvaluationResult", "Evaluator"]
