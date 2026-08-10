"""Composition-owned outcomes; compatible is not validated evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = (
    "CompatibilityIssue",
    "CompatibilityReport",
    "CompatibilityStatus",
)


class CompatibilityStatus(str, Enum):
    INVALID = "invalid"
    PENDING_ARTIFACT_BIND = "pending_artifact_bind"
    COMPATIBLE = "compatible"


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    producer_path: str
    consumer_path: str
    expected: str
    provided: str
    hint: str

    def __post_init__(self) -> None:
        for name in (
            "code",
            "producer_path",
            "consumer_path",
            "expected",
            "provided",
            "hint",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class CompatibilityReport:
    status: CompatibilityStatus
    issues: tuple[CompatibilityIssue, ...]
    pending_fields: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.status is CompatibilityStatus.INVALID and not self.issues:
            raise ValueError("invalid report must contain issues")
        if (
            self.status is CompatibilityStatus.PENDING_ARTIFACT_BIND
            and not self.pending_fields
        ):
            raise ValueError("pending report must name unresolved fields")
        if self.status is CompatibilityStatus.COMPATIBLE and (
            self.issues or self.pending_fields
        ):
            raise ValueError(
                "compatible report cannot contain issues or pending fields"
            )

    @property
    def is_compatible(self) -> bool:
        return self.status is CompatibilityStatus.COMPATIBLE
