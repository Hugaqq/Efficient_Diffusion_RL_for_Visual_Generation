"""Normalized compatibility identity and human inspection projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visual_rl.composition.compatibility.report import CompatibilityReport
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict

__all__ = (
    "COMPATIBILITY_RULE_SET_VERSION",
    "CompatibilitySnapshot",
)


COMPATIBILITY_RULE_SET_VERSION = "visualrl.compatibility.rules.v1"
_ISSUE_FIELDS = frozenset(
    {
        "code",
        "producer_path",
        "consumer_path",
        "expected",
        "provided",
        "hint",
    }
)


@dataclass(frozen=True, slots=True)
class CompatibilitySnapshot:
    """Immutable report facts with separate identity and diagnostic views."""

    status: str
    issues: tuple[FrozenMapping, ...]
    pending_fields: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.status not in {"invalid", "pending_artifact_bind", "compatible"}:
            raise ValueError(f"unsupported compatibility status: {self.status!r}")
        if type(self.issues) is not tuple or any(
            not isinstance(item, FrozenMapping) for item in self.issues
        ):
            raise TypeError("compatibility issues must be FrozenMapping values")
        for issue in self.issues:
            if set(issue) != _ISSUE_FIELDS:
                raise ValueError(
                    "compatibility issue must contain the canonical diagnostic fields"
                )
            if any(
                not isinstance(issue[field], str) or not issue[field]
                for field in _ISSUE_FIELDS
            ):
                raise ValueError(
                    "compatibility issue diagnostic fields must be non-empty strings"
                )
        if self.issues != tuple(sorted(self.issues, key=_diagnostic_sort_key)):
            raise ValueError("compatibility issues must use canonical diagnostic order")
        if type(self.pending_fields) is not tuple or any(
            not isinstance(item, str) or not item for item in self.pending_fields
        ):
            raise ValueError("pending_fields must contain non-empty strings")
        if self.pending_fields != tuple(sorted(set(self.pending_fields))):
            raise ValueError("pending_fields must be sorted and unique")
        if type(self.bindings) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(not isinstance(part, str) or not part for part in item)
            for item in self.bindings
        ):
            raise ValueError("bindings must contain string pairs")
        if self.bindings != tuple(sorted(set(self.bindings))):
            raise ValueError("bindings must be sorted and unique")

    @classmethod
    def from_report(cls, report: CompatibilityReport) -> CompatibilitySnapshot:
        if not isinstance(report, CompatibilityReport):
            raise TypeError("report must be a CompatibilityReport")
        issues = tuple(
            sorted(
                (
                    FrozenMapping(
                        {
                            "code": issue.code,
                            "producer_path": issue.producer_path,
                            "consumer_path": issue.consumer_path,
                            "expected": issue.expected,
                            "provided": issue.provided,
                            "hint": issue.hint,
                        }
                    )
                    for issue in report.issues
                ),
                key=_diagnostic_sort_key,
            )
        )
        return cls(
            status=report.status.value,
            issues=issues,
            pending_fields=tuple(sorted(set(report.pending_fields))),
            bindings=tuple(sorted(set(report.bindings))),
        )

    def identity_payload(self) -> dict[str, Any]:
        """Return only normalized rule facts used by semantic fingerprints."""

        facts = {
            (
                issue["code"],
                issue["producer_path"],
                issue["consumer_path"],
                issue["expected"],
                issue["provided"],
            )
            for issue in self.issues
        }
        return {
            "rule_set_version": COMPATIBILITY_RULE_SET_VERSION,
            "status": self.status,
            "issues": [
                {
                    "code": code,
                    "producer": producer,
                    "consumer": consumer,
                    "required": required,
                    "provided": provided,
                }
                for code, producer, consumer, required, provided in sorted(facts)
            ],
            "pending_fields": list(self.pending_fields),
            "bindings": [list(item) for item in self.bindings],
        }

    def inspection_payload(self) -> dict[str, Any]:
        """Return rule facts plus human diagnostics for manifests and tools."""

        return {
            "rule_set_version": COMPATIBILITY_RULE_SET_VERSION,
            "status": self.status,
            "issues": [to_plain_dict(item) for item in self.issues],
            "pending_fields": list(self.pending_fields),
            "bindings": [list(item) for item in self.bindings],
        }


def _diagnostic_sort_key(issue: FrozenMapping) -> tuple[str, ...]:
    return (
        issue["code"],
        issue["producer_path"],
        issue["consumer_path"],
        issue["expected"],
        issue["provided"],
        issue["hint"],
    )
