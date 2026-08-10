"""Immutable data-owned contracts for configured dataset sources.

``SourcePlanSpec`` is the path-free compiler projection.
``SourceLocationBinding`` owns launch-only locations and meets the plan only in
``SourceLoadRequest``.  ``SourceContentBinding`` is their validated path-free
materialization projection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict

__all__ = (
    "DatasetArtifactBinding",
    "DatasetSourceSpec",
    "SourceContentBinding",
    "SourceLoadError",
    "SourceLoadRequest",
    "SourceLocationBinding",
    "SourcePlanSpec",
)


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PLAN_ID = re.compile(r"^source-plan-spec\.v1:[0-9a-f]{64}$")
_CONTENT_IDENTITY_KEYS = frozenset(
    {
        "identity_schema",
        "content_policy",
        "node_type",
        "content_sha256",
        "file_count",
        "byte_count",
    }
)

DatasetArtifactKind = Literal["file"]
DatasetFormat = Literal["text", "jsonl"]


class SourceLoadError(ValueError):
    """One fail-closed source-plan projection or dataset loading error."""


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SourceLoadError(f"{field_name} must be a canonical lowercase identifier")
    return value


def _expected_file_content_identity(value: object) -> FrozenMapping:
    if not isinstance(value, FrozenMapping):
        raise TypeError("expected_content_identity must be a FrozenMapping")
    if set(value) != _CONTENT_IDENTITY_KEYS:
        raise SourceLoadError("expected_content_identity has an invalid exact key set")
    if value["identity_schema"] != "filesystem-artifact.v1":
        raise SourceLoadError(
            "expected_content_identity must use filesystem-artifact.v1"
        )
    if value["content_policy"] != "all-files.v1":
        raise SourceLoadError(
            "dataset content identity must use the all-files.v1 policy"
        )
    if value["node_type"] != "file":
        raise SourceLoadError("dataset content identity must describe a file")
    digest = value["content_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise SourceLoadError(
            "expected content_sha256 must be a lowercase SHA-256 digest"
        )
    if type(value["file_count"]) is not int or value["file_count"] != 1:
        raise SourceLoadError("dataset file content identity must have file_count=1")
    byte_count = value["byte_count"]
    if type(byte_count) is not int or byte_count < 0:
        raise SourceLoadError(
            "dataset content identity byte_count must be a non-negative integer"
        )
    return value


@dataclass(frozen=True, slots=True)
class DatasetSourceSpec:
    """One path-free source selection whose parser semantics are explicit."""

    source_id: str
    selector: str
    artifact_ref: str
    artifact_kind: DatasetArtifactKind
    format: DatasetFormat

    def __post_init__(self) -> None:
        _identifier(self.source_id, field_name="source_id")
        _identifier(self.selector, field_name="selector")
        _identifier(self.artifact_ref, field_name="artifact_ref")
        if self.artifact_kind != "file":
            raise SourceLoadError("dataset artifact_kind must be file")
        if self.format not in {"text", "jsonl"}:
            raise SourceLoadError("dataset format must be text or jsonl")

    def to_payload(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "selector": self.selector,
            "artifact_ref": self.artifact_ref,
            "artifact_kind": self.artifact_kind,
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class SourcePlanSpec:
    """Canonical path-free source plan emitted by the composition compiler."""

    sources: tuple[DatasetSourceSpec, ...]

    def __post_init__(self) -> None:
        if type(self.sources) is not tuple or not self.sources:
            raise SourceLoadError("source plan spec must contain at least one source")
        if any(not isinstance(item, DatasetSourceSpec) for item in self.sources):
            raise TypeError("sources must contain DatasetSourceSpec values")
        source_ids = tuple(item.source_id for item in self.sources)
        if source_ids != tuple(sorted(source_ids)):
            raise SourceLoadError("source plan spec sources must be sorted")
        if len(source_ids) != len(set(source_ids)):
            raise SourceLoadError("source plan spec source ids must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "source_plan_spec",
            "sources": tuple(item.to_payload() for item in self.sources),
        }

    @property
    def plan_id(self) -> str:
        return canonical_identity("source-plan-spec.v1", self.to_payload())


@dataclass(frozen=True, slots=True)
class DatasetArtifactBinding:
    """Launch-only location plus the content identity locked by preflight."""

    artifact_ref: str
    artifact_location: Path
    expected_content_identity: FrozenMapping

    def __post_init__(self) -> None:
        _identifier(self.artifact_ref, field_name="artifact_ref")
        if (
            not isinstance(self.artifact_location, Path)
            or not self.artifact_location.is_absolute()
        ):
            raise SourceLoadError("artifact_location must be an absolute Path")
        _expected_file_content_identity(self.expected_content_identity)

    def to_content_payload(self) -> dict[str, object]:
        """Return this artifact's path-free expected-content projection.

        Aggregate materialized/resume identity belongs to
        :class:`SourceContentBinding`, not this path-bearing launch object.
        """

        return {
            "artifact_ref": self.artifact_ref,
            "expected_content_identity": to_plain_dict(self.expected_content_identity),
        }

    def to_launch_audit_payload(self) -> dict[str, object]:
        """Return launch provenance; absolute location is intentionally visible."""

        return {
            **self.to_content_payload(),
            "artifact_location": str(self.artifact_location),
        }


@dataclass(frozen=True, slots=True)
class SourceContentBinding:
    """Path-free dataset content bound exactly to one semantic source plan.

    The tuple representation deliberately reuses the validated content-identity
    member of :class:`DatasetArtifactBinding` instead of introducing a second
    per-artifact DTO.  A location object is never retained by this contract.
    """

    source_plan_id: str
    artifact_content_identities: tuple[tuple[str, FrozenMapping], ...]

    def __post_init__(self) -> None:
        _source_plan_identity(self.source_plan_id)
        bindings = self.artifact_content_identities
        if type(bindings) is not tuple or not bindings:
            raise SourceLoadError(
                "source content binding must contain at least one artifact identity"
            )
        refs: list[str] = []
        for binding in bindings:
            if type(binding) is not tuple or len(binding) != 2:
                raise TypeError(
                    "artifact_content_identities must contain "
                    "(artifact_ref, FrozenMapping) pairs"
                )
            artifact_ref, content_identity = binding
            refs.append(_identifier(artifact_ref, field_name="artifact_ref"))
            _expected_file_content_identity(content_identity)
        artifact_refs = tuple(refs)
        if artifact_refs != tuple(sorted(artifact_refs)):
            raise SourceLoadError("source artifact content identities must be sorted")
        if len(artifact_refs) != len(set(artifact_refs)):
            raise SourceLoadError(
                "source artifact content identity refs must be unique"
            )

    @classmethod
    def from_location_binding(
        cls,
        *,
        plan: SourcePlanSpec,
        locations: SourceLocationBinding,
    ) -> SourceContentBinding:
        """Project exact content from launch locations after plan validation."""

        if not isinstance(locations, SourceLocationBinding):
            raise TypeError("locations must be a SourceLocationBinding")
        result = cls(
            source_plan_id=locations.source_plan_id,
            artifact_content_identities=tuple(
                (binding.artifact_ref, binding.expected_content_identity)
                for binding in locations.artifacts
            ),
        )
        result.validate_against(plan)
        return result

    def validate_against(self, plan: SourcePlanSpec) -> None:
        """Fail closed unless this binding exactly covers ``plan``."""

        _validate_exact_plan_coverage(
            plan=plan,
            source_plan_id=self.source_plan_id,
            observed_refs=tuple(
                artifact_ref
                for artifact_ref, _identity in self.artifact_content_identities
            ),
            binding_name="source content binding",
        )

    def artifact_identity(self, artifact_ref: str) -> FrozenMapping:
        """Return one content identity without exposing a launch location."""

        _identifier(artifact_ref, field_name="artifact_ref")
        for bound_ref, content_identity in self.artifact_content_identities:
            if bound_ref == artifact_ref:
                return content_identity
        raise KeyError(f"unknown dataset artifact content identity {artifact_ref!r}")

    def canonical_payload(self) -> dict[str, object]:
        """Return the canonical, path-free materialized/resume payload."""

        return {
            "schema_version": 1,
            "kind": "source_content_binding",
            "source_plan_id": self.source_plan_id,
            "artifacts": tuple(
                {
                    "artifact_ref": artifact_ref,
                    "content_identity": to_plain_dict(content_identity),
                }
                for artifact_ref, content_identity in (self.artifact_content_identities)
            ),
        }

    def to_payload(self) -> dict[str, object]:
        """Alias the canonical artifact projection used by sibling contracts."""

        return self.canonical_payload()

    @property
    def content_binding_id(self) -> str:
        return canonical_identity(
            "source-content-binding.v1",
            self.canonical_payload(),
        )


@dataclass(frozen=True, slots=True)
class SourceLocationBinding:
    """Launch-only paths and expected content for one source plan identity."""

    source_plan_id: str
    artifacts: tuple[DatasetArtifactBinding, ...]

    def __post_init__(self) -> None:
        _source_plan_identity(self.source_plan_id)
        if type(self.artifacts) is not tuple or not self.artifacts:
            raise SourceLoadError(
                "source location binding must contain at least one artifact"
            )
        if any(not isinstance(item, DatasetArtifactBinding) for item in self.artifacts):
            raise TypeError("artifacts must contain DatasetArtifactBinding values")
        artifact_refs = tuple(item.artifact_ref for item in self.artifacts)
        if artifact_refs != tuple(sorted(artifact_refs)):
            raise SourceLoadError("source artifact bindings must be sorted")
        if len(artifact_refs) != len(set(artifact_refs)):
            raise SourceLoadError("source artifact binding refs must be unique")

    def artifact(self, artifact_ref: str) -> DatasetArtifactBinding:
        _identifier(artifact_ref, field_name="artifact_ref")
        for binding in self.artifacts:
            if binding.artifact_ref == artifact_ref:
                return binding
        raise KeyError(f"unknown dataset artifact binding {artifact_ref!r}")

    def to_content_binding(self, plan: SourcePlanSpec) -> SourceContentBinding:
        """Project path-free content after exact semantic-plan coverage checks."""

        return SourceContentBinding.from_location_binding(
            plan=plan,
            locations=self,
        )

    def to_launch_audit_payload(self) -> dict[str, object]:
        """Return path-bearing provenance, never materialized/resume identity."""

        return {
            "schema_version": 1,
            "kind": "source_location_binding",
            "source_plan_id": self.source_plan_id,
            "artifacts": tuple(
                item.to_launch_audit_payload() for item in self.artifacts
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceLoadRequest:
    """Runtime handoff joining one semantic plan with exact artifact bindings."""

    plan: SourcePlanSpec
    locations: SourceLocationBinding

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SourcePlanSpec):
            raise TypeError("plan must be a SourcePlanSpec")
        if not isinstance(self.locations, SourceLocationBinding):
            raise TypeError("locations must be a SourceLocationBinding")
        _validate_exact_plan_coverage(
            plan=self.plan,
            source_plan_id=self.locations.source_plan_id,
            observed_refs=tuple(
                binding.artifact_ref for binding in self.locations.artifacts
            ),
            binding_name="source location binding",
        )

    def to_content_binding(self) -> SourceContentBinding:
        """Return the path-free identity projection of this validated request."""

        return self.locations.to_content_binding(self.plan)


def _source_plan_identity(value: object) -> str:
    if not isinstance(value, str) or _SOURCE_PLAN_ID.fullmatch(value) is None:
        raise SourceLoadError("source_plan_id must be a source-plan-spec.v1 identity")
    return value


def _validate_exact_plan_coverage(
    *,
    plan: SourcePlanSpec,
    source_plan_id: str,
    observed_refs: tuple[str, ...],
    binding_name: str,
) -> None:
    if not isinstance(plan, SourcePlanSpec):
        raise TypeError("plan must be a SourcePlanSpec")
    _source_plan_identity(source_plan_id)
    if source_plan_id != plan.plan_id:
        raise SourceLoadError(f"{binding_name} does not match plan_id")
    expected_refs = tuple(sorted({source.artifact_ref for source in plan.sources}))
    if observed_refs != expected_refs:
        raise SourceLoadError(
            f"{binding_name} does not exactly cover the source plan; "
            f"expected={list(expected_refs)}, observed={list(observed_refs)}"
        )
