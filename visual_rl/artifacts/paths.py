"""Read-only artifact paths for ranks that never own the writer manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """The artifact surface needed by ``Experiment.run`` on non-writer ranks."""

    output_dir: Path
    run_id: str
    metric_path: Path
    manifest_path: Path

    @classmethod
    def for_run(cls, output_dir: str | Path, run_id: str) -> "ArtifactPaths":
        root = Path(output_dir)
        return cls(
            output_dir=root,
            run_id=run_id,
            metric_path=root / "metrics.jsonl",
            manifest_path=root / "sample_manifest.json",
        )
