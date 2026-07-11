from __future__ import annotations

import json

import pytest
import torch

from visual_rl.artifacts import ArtifactManager, ManifestBuilder, SampleManifest
from visual_rl.core.types import RewardBatch, RolloutBatch


def _batch(batch_size: int = 2) -> RolloutBatch:
    return RolloutBatch(
        prompts=[f"prompt {index}" for index in range(batch_size)],
        metadata=[
            {"source": "test", "branch_id": index - 1} for index in range(batch_size)
        ],
        media=torch.ones(batch_size, 3, 4, 4),
        latents=torch.zeros(batch_size, 2, 3, 4, 4),
        next_latents=torch.ones(batch_size, 2, 3, 4, 4),
        timesteps=torch.tensor([[3, 1]] * batch_size),
        old_log_probs=torch.zeros(batch_size, 2),
        branch_ids=torch.arange(batch_size) - 1,
        seed=17,
        model_metadata={"adapter": "tiny", "rollout": "branching"},
    )


def _rewards(batch_size: int = 2) -> RewardBatch:
    raw = torch.linspace(0.2, 0.8, batch_size, requires_grad=True)
    weighted = raw * 2
    return RewardBatch(
        raw={"quality": raw},
        weighted={"quality": weighted},
        weighted_total=weighted,
        valid_mask=torch.ones(batch_size, dtype=torch.bool),
    )


def test_manifest_builder_aligns_each_sample_and_detaches_tensors():
    records = ManifestBuilder("run-001").build_records(
        step=4,
        batch=_batch(),
        rewards=_rewards(),
        media_type="image",
        media_paths=["image-0.png", "image-1.png"],
        rollout_cache_path="rollouts/batch_000004.pt",
    )

    assert [record.sample_id for record in records] == [
        "run-001-step-000004-sample-000000",
        "run-001-step-000004-sample-000001",
    ]
    assert [record.prompt for record in records] == ["prompt 0", "prompt 1"]
    assert [
        record.reward_values["raw"]["quality"] for record in records
    ] == pytest.approx([0.2, 0.8])
    assert [
        record.reward_values["weighted_total"] for record in records
    ] == pytest.approx([0.4, 1.6])
    assert [record.media_path for record in records] == ["image-0.png", "image-1.png"]
    assert records[0].timestep_summary == {
        "values": [3, 1],
        "count": 2,
        "branch_id": -1,
    }
    assert records[0].seed == 17


def test_manifest_builder_rejects_reward_batch_mismatch():
    with pytest.raises(ValueError, match="rewards.raw.quality batch dimension"):
        ManifestBuilder("run-001").build_records(
            step=0,
            batch=_batch(batch_size=2),
            rewards=_rewards(batch_size=1),
            media_type="video",
        )


def test_manifest_builder_rejects_media_path_mismatch():
    with pytest.raises(ValueError, match="media_paths length"):
        ManifestBuilder("run-001").build_records(
            step=0,
            batch=_batch(),
            rewards=_rewards(),
            media_type="video",
            media_paths=["only-one.mp4"],
        )


def test_artifact_manager_writes_and_resumes_complete_run_artifacts(tmp_path):
    output_dir = tmp_path / "run"
    first = ArtifactManager(output_dir, "run-001", config={"seed": 17})
    first.record(
        step=0,
        batch=_batch(),
        rewards=_rewards(),
        metrics={"step": 0, "loss": 0.5},
        media_type="video",
        media_paths="rollouts/batch_000000.media.pt",
    )

    resumed = ArtifactManager(output_dir, "run-001", config={"seed": 17}, resume=True)
    resumed.record(
        step=1,
        batch=_batch(),
        rewards=_rewards(),
        metrics={"step": 1, "loss": 0.25},
        media_type="video",
    )

    expected_files = {
        "config.resolved.json",
        "metrics.jsonl",
        "prompt_set.json",
        "reward_table.json",
        "sample_manifest.json",
        "visual_report.md",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    assert len(SampleManifest.load(output_dir / "sample_manifest.json").records) == 4
    assert (
        len((output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
        == 2
    )
    reward_table = json.loads(
        (output_dir / "reward_table.json").read_text(encoding="utf-8")
    )
    assert [row["step"] for row in reward_table["records"]] == [0, 0, 1, 1]
    assert "Videos: 4" in (output_dir / "visual_report.md").read_text(encoding="utf-8")


def test_artifact_manager_record_is_idempotent_per_step(tmp_path):
    output_dir = tmp_path / "run"
    manager = ArtifactManager(output_dir, "run-001", config={"seed": 17})
    manager.record(
        step=0,
        batch=_batch(),
        rewards=_rewards(),
        metrics={"step": 0, "loss": 1.0},
        media_type="image",
    )
    manager.record(
        step=0,
        batch=_batch(),
        rewards=_rewards(),
        metrics={"step": 0, "loss": 0.5},
        media_type="image",
    )

    assert len(manager.manifest.records) == 2
    row = json.loads(manager.metric_path.read_text(encoding="utf-8"))
    assert row["loss"] == 0.5


def test_artifact_manager_truncates_uncommitted_steps(tmp_path):
    output_dir = tmp_path / "run"
    manager = ArtifactManager(output_dir, "run-001")
    for step in range(3):
        manager.record(
            step=step,
            batch=_batch(),
            rewards=_rewards(),
            metrics={"step": step, "loss": float(step)},
            media_type="video",
        )

    manager.truncate_from_step(2)

    assert {record.step for record in manager.manifest.records} == {0, 1}
    metric_rows = [
        json.loads(line)
        for line in manager.metric_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in metric_rows] == [0, 1]
