"""Rank-local record production and rank-zero artifact publication."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.artifacts.manager import ArtifactManager
from visual_rl.artifacts.manifest import SampleManifest, SampleRecord
from visual_rl.core.types import (
    FrozenMapping,
    RewardBatch,
    RolloutBatch,
    StepContext,
)


def _batch(*, rank: int, step: int = 0) -> RolloutBatch:
    batch_size, transitions = 2, 2
    sample_id = tuple(
        f"step-{step:06d}-rank-{rank:04d}-sample-{index:04d}"
        for index in range(batch_size)
    )
    return RolloutBatch(
        prompts=tuple(
            f"rank {rank} prompt {index}" for index in range(batch_size)
        ),
        metadata=tuple(
            {
                "source": "distributed-fixture",
                "rank": rank,
                "row": index,
            }
            for index in range(batch_size)
        ),
        media=torch.zeros(batch_size, 3, 2, 2),
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(
            batch_size,
            transitions,
            dtype=torch.bool,
        ),
        sample_id=sample_id,
        prompt_id=tuple(f"prompt-r{rank}-{index}" for index in range(batch_size)),
        group_id=tuple(f"group-r{rank}-{index}" for index in range(batch_size)),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(
            step=step,
            seed=1000 + step * 2 + rank,
            rank=rank,
            world_size=2,
        ),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={"features": torch.ones(batch_size, transitions)},
        artifact_metadata={"adapter": "fixture", "rank": rank},
    )


def _rewards(batch: RolloutBatch) -> RewardBatch:
    score = torch.arange(batch.batch_size, dtype=torch.float32)
    score = score + float(batch.context.rank)
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": score},
        weighted={"score": score},
        weighted_total=score,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        shared_metadata={"score": {"revision": "fixture-v1"}},
        sample_metadata={
            "score": tuple(
                {"sample_id": sample_id} for sample_id in batch.sample_id
            )
        },
    )


def _rank_records(*, rank: int, step: int = 0) -> tuple[SampleRecord, ...]:
    batch = _batch(rank=rank, step=step)
    return ManifestBuilder(
        run_id="distributed-run",
        media_type="image",
        rollout_type="full_trajectory",
    ).build_records(
        batch,
        _rewards(batch),
        media_paths=(None,) * batch.batch_size,
    )


def _metrics() -> SimpleNamespace:
    return SimpleNamespace(
        values=FrozenMapping(
            {
                "loss": 0.25,
                "reward_mean": 1.0,
            }
        ),
        sample_count=4,
        active_transition_count=8,
    )


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter").mkdir()
    (path / "adapter" / "adapter.json").write_text(
        '{"format":"fixture"}\n',
        encoding="utf-8",
    )
    (path / "adapter" / "adapter_state.pt").write_text(
        "weights",
        encoding="utf-8",
    )
    (path / "training_state.pt").write_text("state", encoding="utf-8")
    (path / "checkpoint.json").write_text(
        '{"format_version":5}\n',
        encoding="utf-8",
    )
    return path


def test_rank_local_builder_is_pure_and_records_keep_rank_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    rank_zero = _rank_records(rank=0)
    rank_one = _rank_records(rank=1)

    assert not run_dir.exists()
    assert [record.rank for record in (*rank_zero, *rank_one)] == [0, 0, 1, 1]
    assert all(record.checkpoint_path is None for record in (*rank_zero, *rank_one))
    assert len(
        {record.sample_id for record in (*rank_zero, *rank_one)}
    ) == 4
    assert all(record.group_id.startswith("group-r0") for record in rank_zero)
    assert all(record.group_id.startswith("group-r1") for record in rank_one)


def test_rank_zero_merges_in_rank_order_and_publishes_only_final_surfaces(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    rank_zero = _rank_records(rank=0)
    rank_one = _rank_records(rank=1)
    expected = (*rank_zero, *rank_one)
    boundary_records = tuple(
        replace(record, checkpoint_path="checkpoint_000001")
        for record in expected
    )

    with ArtifactManager(
        run_dir,
        "distributed-run",
        config={"runtime": {"world_size": 2}},
    ) as manager:
        transaction = manager.begin_transaction()
        manager.stage_records(
            transaction,
            step=0,
            records=boundary_records,
            metrics=_metrics(),
        )
        marker = manager.commit(
            transaction,
            checkpoint_path=_checkpoint(
                transaction.staging_dir / "checkpoint_000001"
            ),
        )
        manager.rebuild_projections()
        manager.cleanup_published_staging(transaction)

    expected_ids = [record.sample_id for record in expected]
    marker_ids = [
        row["sample_id"]
        for row in marker["steps"][0]["manifest_records"]
    ]
    assert marker_ids == expected_ids
    manifest = SampleManifest.load(run_dir / "sample_manifest.json")
    assert [record.sample_id for record in manifest.records] == expected_ids
    assert [record.rank for record in manifest.records] == [0, 0, 1, 1]
    metric = json.loads(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert metric == {
        "active_transition_count": 8,
        "loss": 0.25,
        "reward_mean": 1.0,
        "sample_count": 4,
        "schema_version": "3",
        "step": 0,
    }
    for retired in (
        "reward_rows.json",
        "reward_table.json",
        "runtime.json",
        "visual_report.md",
        "prompt_set.json",
        "run_status.json",
    ):
        assert not (run_dir / retired).exists()


def test_duplicate_sample_across_rank_payloads_fails_before_step_write(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    rank_zero = _rank_records(rank=0)
    rank_one = list(_rank_records(rank=1))
    rank_one[0] = replace(rank_one[0], sample_id=rank_zero[0].sample_id)

    with ArtifactManager(
        run_dir,
        "distributed-run",
        config={},
    ) as manager:
        transaction = manager.begin_transaction()
        with pytest.raises(ValueError, match="duplicate sample_id"):
            manager.stage_records(
                transaction,
                step=0,
                records=(*rank_zero, *rank_one),
                metrics=_metrics(),
            )
        assert not (
            transaction.staging_dir / "step_000000"
        ).exists()
        manager.abort(transaction)


def test_non_root_record_build_does_not_create_global_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    records = _rank_records(rank=1)

    assert records
    assert not run_dir.exists()
    assert not (tmp_path / "sample_manifest.json").exists()
    assert not (tmp_path / "metrics.jsonl").exists()


def test_retired_artifact_surfaces_are_physically_absent() -> None:
    repository_root = Path(__file__).parents[1]
    assert not (
        repository_root / "visual_rl" / "artifacts" / "paths.py"
    ).exists()
    for retired in (
        "stage_step",
        "record",
        "truncate_from_step",
        "_flush",
        "apply_retention",
    ):
        assert not hasattr(ArtifactManager, retired)
