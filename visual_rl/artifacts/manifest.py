from dataclasses import dataclass, asdict, field
from typing import Any
import json
from pathlib import Path


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


@dataclass
class SampleManifest:
    run_id: str
    schema_version: str = "1"
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
    def from_dict(cls, data: dict[str, Any]) -> "SampleManifest":
        records = [
            SampleRecord(**record_data) for record_data in data.get("records", [])
        ]

        manifest = cls(
            run_id=data["run_id"],
            schema_version=data.get("schema_version", "1"),
            records=records,
        )

        manifest.validate()

        return manifest

    def validate(self) -> None:
        seen_ids: set[str] = set()

        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Invalid run_id!")

        for record in self.records:
            if record.media_type not in {"image", "video"}:
                raise ValueError("Unsurportted media type")

            if record.step < 0:
                raise ValueError("Step must be non-negative")

            if record.sample_index < 0:
                raise ValueError("Sample index should be non-negative")

            if not isinstance(record.sample_id, str) or not record.sample_id.strip():
                raise ValueError("Invalid samplie id!")

            if record.run_id != self.run_id:
                raise ValueError("record run_id does not match manifest run_id")

            if record.sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id: {record.sample_id}")

            seen_ids.add(record.sample_id)

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "SampleManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("Manifest file must contain a JSON object")
        return cls.from_dict(data)
