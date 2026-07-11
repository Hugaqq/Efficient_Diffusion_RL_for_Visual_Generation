"""Run-scoped artifact persistence built on SampleManifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.artifacts.manifest import SampleManifest, SampleRecord
from visual_rl.artifacts.serialization import to_jsonable
from visual_rl.core.types import RewardBatch, RolloutBatch


class ArtifactManager:
    """Persist reproducibility artifacts without participating in optimization."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        *,
        config: Any | None = None,
        resume: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.builder = ManifestBuilder(run_id)
        self.manifest_path = self.output_dir / "sample_manifest.json"
        self.metric_path = self.output_dir / "metrics.jsonl"
        if not resume and self.manifest_path.exists():
            raise FileExistsError(
                f"Artifact directory already contains a manifest: {self.manifest_path}. "
                "Use resume=True to continue it."
            )
        self.manifest = self._load_manifest(resume)
        self._metrics_by_step = self._load_metrics(resume)
        if config is not None and not resume:
            self._write_json(
                self.output_dir / "config.resolved.json", to_jsonable(config)
            )

    def record(
        self,
        *,
        step: int,
        batch: RolloutBatch,
        rewards: RewardBatch,
        metrics: dict[str, Any],
        media_type: str,
        rollout_type: str | None = None,
        media_paths: list[str | Path | None] | str | Path | None = None,
        rollout_cache_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[SampleRecord]:
        records = self.builder.build_records(
            step=step,
            batch=batch,
            rewards=rewards,
            media_type=media_type,
            rollout_type=rollout_type,
            media_paths=media_paths,
            rollout_cache_path=rollout_cache_path,
            checkpoint_path=checkpoint_path,
        )
        self.manifest.records = [
            record for record in self.manifest.records if record.step != step
        ]
        for record in records:
            self.manifest.add(record)
        self._metrics_by_step[step] = to_jsonable(dict(metrics))
        self._flush()
        return records

    def truncate_from_step(self, start_step: int) -> None:
        """Discard artifacts that are newer than the checkpoint being resumed."""

        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        self.manifest.records = [
            record for record in self.manifest.records if record.step < start_step
        ]
        self._metrics_by_step = {
            step: metrics
            for step, metrics in self._metrics_by_step.items()
            if step < start_step
        }
        self._flush()

    def _load_manifest(self, resume: bool) -> SampleManifest:
        if resume and self.manifest_path.exists():
            manifest = SampleManifest.load(self.manifest_path)
            if manifest.run_id != self.run_id:
                raise ValueError(
                    "Existing manifest run_id does not match ArtifactManager run_id"
                )
            return manifest
        return SampleManifest(run_id=self.run_id)

    def _load_metrics(self, resume: bool) -> dict[int, dict[str, Any]]:
        if not resume or not self.metric_path.exists():
            return {}
        metrics: dict[int, dict[str, Any]] = {}
        for line in self.metric_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metrics[int(row["step"])] = row
        return metrics

    def _flush(self) -> None:
        self.manifest.save(self.manifest_path)
        self._write_json(
            self.output_dir / "prompt_set.json",
            {
                "run_id": self.run_id,
                "prompts": self._prompt_rows(),
            },
        )
        self._write_json(
            self.output_dir / "reward_table.json",
            {
                "run_id": self.run_id,
                "records": [
                    {
                        "sample_id": record.sample_id,
                        "step": record.step,
                        "reward_values": record.reward_values,
                    }
                    for record in self.manifest.records
                ],
            },
        )
        metric_lines = [
            json.dumps(self._metrics_by_step[step], sort_keys=True, ensure_ascii=False)
            for step in sorted(self._metrics_by_step)
        ]
        self._write_text(
            self.metric_path,
            "\n".join(metric_lines) + ("\n" if metric_lines else ""),
        )
        self._write_text(
            self.output_dir / "visual_report.md",
            self._visual_report(),
        )

    def _prompt_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in self.manifest.records:
            row = {"prompt": record.prompt, "metadata": record.prompt_metadata}
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                rows.append(row)
        return rows

    def _visual_report(self) -> str:
        image_count = sum(
            record.media_type == "image" for record in self.manifest.records
        )
        video_count = sum(
            record.media_type == "video" for record in self.manifest.records
        )
        lines = [
            f"# VisualRL Run Report: {self.run_id}",
            "",
            f"- Samples: {len(self.manifest.records)}",
            f"- Images: {image_count}",
            f"- Videos: {video_count}",
            "",
            "## Samples",
            "",
            "| sample_id | step | media_type | prompt | weighted_total |",
            "|---|---:|---|---|---:|",
        ]
        for record in self.manifest.records:
            reward = record.reward_values.get("weighted_total", "")
            prompt = record.prompt.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {record.sample_id} | {record.step} | {record.media_type} | {prompt} | {reward} |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(data), handle, indent=2, sort_keys=True, ensure_ascii=False
            )
        tmp_path.replace(path)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
