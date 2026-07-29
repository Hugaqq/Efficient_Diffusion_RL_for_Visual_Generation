"""API-level coverage for the one typed rollout/reward/update dataflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import visual_rl as vr


ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG = ROOT / "tests/fixtures/configs/tiny_grpo.yaml"


def _copy_config(tmp_path: Path, *, max_steps: int = 1) -> Path:
    payload = yaml.safe_load(TINY_CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["max_steps"] = max_steps
    payload["artifacts"]["output_dir"] = str(tmp_path / "run")
    payload["artifacts"]["checkpoint_every"] = 1
    destination = tmp_path / "tiny_grpo.yaml"
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def test_typed_dataflow_runs_only_through_public_api(tmp_path: Path) -> None:
    experiment = vr.load(_copy_config(tmp_path))

    report = experiment.validate()
    assert report.ok

    result = experiment.run()

    assert isinstance(result, vr.RunResult)
    assert result.committed_steps == 1
    assert result.last_metrics["step"] == 0
    assert result.last_metrics["sample_count"] == 4
    assert result.last_metrics["active_transition_count"] == 8

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert len(records) == 4
    assert all(record["step"] == 0 for record in records)
    assert len({record["sample_id"] for record in records}) == 4
    assert len({record["prompt_id"] for record in records}) == 2
    assert len({record["group_id"] for record in records}) == 2
    assert all(record["branch_id"] is None for record in records)

    status = vr.inspect_run(result.output_dir)
    assert status.ok
    assert status.run_id == result.run_id
    assert status.committed_steps == result.committed_steps
    assert status.authoritative_checkpoint == result.authoritative_checkpoint


def test_two_steps_preserve_one_authoritative_typed_identity_chain(
    tmp_path: Path,
) -> None:
    result = vr.load(_copy_config(tmp_path, max_steps=2)).run()

    assert result.committed_steps == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert {record["step"] for record in records} == {0, 1}
    assert len(records) == 8
    assert len({record["sample_id"] for record in records}) == 8

    audit = vr.audit_run(result.output_dir)
    assert audit.ok
    assert audit.run_id == result.run_id
    assert audit.committed_steps == 2
    assert audit.checked_commit_count == 2
