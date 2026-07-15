"""Versioned sample-manifest contracts and explicit legacy migration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from visual_rl.artifacts.checkpoint import load_json, save_json


SAMPLE_MANIFEST_SCHEMA_VERSION = "2"
_LEGACY_MANIFEST_SCHEMA_VERSIONS = {None, "0", "1"}


@dataclass
class SampleRecord:
    run_id: str
    sample_id: str
    sample_index: int
    step: int
    prompt: str
    media_type: str
    prompt_metadata: dict[str, Any]
    seed: int | None = None
    rollout_type: str | None = None
    timestep_summary: dict[str, Any] = field(default_factory=dict)
    reward_values: dict[str, Any] = field(default_factory=dict)
    media_path: str | None = None
    rollout_cache_path: str | None = None
    checkpoint_path: str | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    prompt_id: str | None = None
    group_id: str | None = None
    branch_id: Any | None = None


@dataclass
class SampleManifest:
    run_id: str
    schema_version: str = SAMPLE_MANIFEST_SCHEMA_VERSION
    records: list[SampleRecord] = field(default_factory=list)

    def add(self, record: SampleRecord) -> None:
        if record.run_id != self.run_id:
            raise ValueError("record run_id does not match manifest run_id")
        if any(item.sample_id == record.sample_id for item in self.records):
            raise ValueError(f"Duplicate sample_id: {record.sample_id}")
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleManifest:
        if not isinstance(data, dict):
            raise ValueError("SampleManifest must contain a JSON object")
        if "schema_version" not in data:
            raise ValueError(
                "SampleManifest is missing schema_version; use "
                "migrate_legacy_to_v2() explicitly"
            )
        schema_version = str(data["schema_version"])
        if schema_version != SAMPLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SampleManifest schema_version: {schema_version!r}"
            )
        if set(data).difference({"schema_version", "run_id", "records"}):
            raise ValueError("SampleManifest contains unexpected top-level fields")
        run_id = data.get("run_id")
        raw_records = data.get("records")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("SampleManifest run_id must be a non-empty string")
        if not isinstance(raw_records, list):
            raise ValueError("SampleManifest records must be a list")
        records: list[SampleRecord] = []
        for index, record_data in enumerate(raw_records):
            if not isinstance(record_data, dict):
                raise ValueError(f"SampleManifest record {index} must be an object")
            try:
                records.append(SampleRecord(**record_data))
            except TypeError as exc:
                raise ValueError(
                    f"SampleManifest record {index} has invalid fields: {exc}"
                ) from exc
        manifest = cls(
            run_id=run_id,
            schema_version=schema_version,
            records=records,
        )
        manifest.validate()
        return manifest

    @classmethod
    def migrate_legacy_to_v2(cls, data: dict[str, Any]) -> SampleManifest:
        """Non-destructively migrate an explicit unversioned/v0/v1 payload."""

        if not isinstance(data, dict):
            raise ValueError("Legacy SampleManifest must contain a JSON object")
        source_version = data.get("schema_version")
        normalized_version = None if source_version is None else str(source_version)
        if normalized_version not in _LEGACY_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                "Legacy SampleManifest migration accepts only unversioned, v0, "
                f"or v1 payloads, got {source_version!r}"
            )
        migrated = deepcopy(data)
        migrated["schema_version"] = SAMPLE_MANIFEST_SCHEMA_VERSION
        records = migrated.get("records")
        if not isinstance(records, list):
            raise ValueError("Legacy SampleManifest records must be a list")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Legacy SampleManifest record {index} must be an object"
                )
            record.setdefault("prompt_id", None)
            record.setdefault("group_id", None)
            record.setdefault("branch_id", None)
        return cls.from_dict(migrated)

    def validate(self) -> None:
        if self.schema_version != SAMPLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SampleManifest schema_version: {self.schema_version!r}"
            )
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("SampleManifest run_id must be a non-empty string")
        if not isinstance(self.records, list):
            raise ValueError("SampleManifest records must be a list")

        seen_ids: set[str] = set()
        for record in self.records:
            if not isinstance(record, SampleRecord):
                raise ValueError("SampleManifest records must contain SampleRecord")
            if record.media_type not in {"image", "video"}:
                raise ValueError(f"Unsupported media type: {record.media_type!r}")
            if isinstance(record.step, bool) or not isinstance(record.step, int):
                raise ValueError("Sample record step must be an integer")
            if record.step < 0:
                raise ValueError("Sample record step must be non-negative")
            if isinstance(record.sample_index, bool) or not isinstance(
                record.sample_index, int
            ):
                raise ValueError("Sample index must be an integer")
            if record.sample_index < 0:
                raise ValueError("Sample index must be non-negative")
            if not isinstance(record.sample_id, str) or not record.sample_id.strip():
                raise ValueError("Sample ID must be a non-empty string")
            if record.run_id != self.run_id:
                raise ValueError("record run_id does not match manifest run_id")
            if record.sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)

    def save(self, path: str | Path) -> None:
        self.validate()
        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> SampleManifest:
        return cls.from_dict(load_json(path))
