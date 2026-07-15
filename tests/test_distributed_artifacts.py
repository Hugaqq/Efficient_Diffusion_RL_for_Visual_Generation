"""CPU-only contracts for rank-merged transactional artifacts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from visual_rl.artifacts import ArtifactManager, ManifestBuilder
from visual_rl.artifacts.manifest import SampleRecord
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext


def _rank_records(
    run_id: str,
    run_dir: Path,
    *,
    rank: int,
    step: int,
    batch_size: int = 2,
) -> list[SampleRecord]:
    sample_ids = [
        f"step-{step:06d}-rank-{rank:04d}-sample-{index:04d}"
        for index in range(batch_size)
    ]
    prompts = [f"rank {rank} prompt {index}" for index in range(batch_size)]
    metadata = [
        {
            "prompt_id": f"prompt-{rank}-{index}",
            "group_id": f"group-{rank}-{index}",
            "branch_id": index,
        }
        for index in range(batch_size)
    ]
    batch = RolloutBatch(
        prompts=prompts,
        metadata=metadata,
        media=torch.zeros(batch_size, 3, 2, 2),
        latents=torch.zeros(batch_size, 1, 1, 2, 2),
        next_latents=torch.ones(batch_size, 1, 1, 2, 2),
        timesteps=torch.full((batch_size, 1), step),
        old_log_probs=torch.zeros(batch_size, 1),
        kl=torch.zeros(batch_size, 1),
        sample_id=sample_ids,
        context=StepContext(
            step=step,
            seed=1000 + step,
            epoch_tag=step,
            rank=rank,
            world_size=2,
            policy_version=step,
        ),
    )
    rewards = RewardBatch(
        raw={"score": torch.arange(batch_size, dtype=torch.float32) + rank},
        weighted={
            "score": torch.arange(batch_size, dtype=torch.float32) + rank
        },
        weighted_total=torch.arange(batch_size, dtype=torch.float32) + rank,
        valid_mask=torch.ones(batch_size, dtype=torch.bool),
        sample_id=sample_ids,
    )
    rank_root = run_dir / "rollouts" / f"rank_{rank:04d}"
    return ManifestBuilder(run_id).build_records(
        step=step,
        batch=batch,
        rewards=rewards,
        media_type="image",
        rollout_type="distributed-test",
        media_paths=[
            rank_root / f"batch_{step:06d}.media.pt"
        ]
        * batch_size,
        rollout_cache_path=rank_root / f"batch_{step:06d}.pt",
        checkpoint_path=run_dir / f"checkpoint_{step + 1:06d}",
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_triplet(directory: Path, step: int) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"batch_{step:06d}"
    paths = (
        base.with_suffix(".pt"),
        base.with_suffix(".media.pt"),
        base.with_suffix(".json"),
    )
    paths[0].write_bytes(b"tensor")
    paths[1].write_bytes(b"media")
    paths[2].write_text("{}", encoding="utf-8")
    return paths


def test_two_rank_records_merge_in_stable_order_and_project_every_sample(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_id = "distributed-run"
    rank_zero = _rank_records(run_id, run_dir, rank=0, step=0)
    rank_one = _rank_records(run_id, run_dir, rank=1, step=0)
    merged = [*rank_zero, *rank_one]

    with ArtifactManager(run_dir, run_id) as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        staged = manager.stage_records(
            transaction,
            step=0,
            records=merged,
            metrics={"loss": 0.25},
        )
        marker = manager.commit(transaction)

    expected_ids = [record.sample_id for record in merged]
    assert [record.sample_id for record in staged] == expected_ids
    assert [
        row["sample_id"] for row in marker["steps"][0]["manifest_records"]
    ] == expected_ids
    manifest = _json(run_dir / "sample_manifest.json")
    rewards = _json(run_dir / "reward_table.json")
    assert [row["sample_id"] for row in manifest["records"]] == expected_ids
    assert [row["sample_id"] for row in rewards["records"]] == expected_ids
    assert len(_json(run_dir / "prompt_set.json")["prompts"]) == len(merged)
    report = (run_dir / "visual_report.md").read_text(encoding="utf-8")
    assert all(sample_id in report for sample_id in expected_ids)


def test_stage_records_rejects_invalid_rank_merges_before_writing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_id = "validation-run"
    records = _rank_records(run_id, run_dir, rank=0, step=0, batch_size=1)

    with ArtifactManager(run_dir, run_id) as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        with pytest.raises(ValueError, match="must not be empty"):
            manager.stage_records(
                transaction,
                step=0,
                records=[],
                metrics={},
            )
        with pytest.raises(ValueError, match="duplicate sample_id"):
            manager.stage_records(
                transaction,
                step=0,
                records=[records[0], records[0]],
                metrics={},
            )
        with pytest.raises(ValueError, match="run_id"):
            manager.stage_records(
                transaction,
                step=0,
                records=[replace(records[0], run_id="another-run")],
                metrics={},
            )
        with pytest.raises(ValueError, match="record step"):
            manager.stage_records(
                transaction,
                step=0,
                records=[replace(records[0], step=1)],
                metrics={},
            )
        with pytest.raises(ValueError, match="escapes"):
            manager.stage_records(
                transaction,
                step=0,
                records=[
                    replace(
                        records[0],
                        media_path=str(tmp_path / "outside.media.pt"),
                    )
                ],
                metrics={},
            )
        with pytest.raises(ValueError, match="strict JSON"):
            manager.stage_records(
                transaction,
                step=0,
                records=[
                    replace(
                        records[0],
                        reward_values={"weighted_total": float("inf")},
                    )
                ],
                metrics={},
            )
        with pytest.raises(ValueError, match="strict JSON"):
            manager.stage_records(
                transaction,
                step=0,
                records=records,
                metrics={"loss": float("nan")},
            )
        assert not (transaction.staging_dir / "step_000000").exists()
        manager.abort(transaction)


def test_sample_id_is_unique_across_committed_logical_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_id = "global-id-run"
    first = _rank_records(run_id, run_dir, rank=0, step=0, batch_size=1)
    second = _rank_records(run_id, run_dir, rank=0, step=1, batch_size=1)

    with ArtifactManager(run_dir, run_id) as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        manager.stage_records(
            transaction,
            step=0,
            records=first,
            metrics={},
        )
        manager.commit(transaction)

        duplicate = manager.begin_transaction(completed_steps=2)
        with pytest.raises(ValueError, match="already staged or committed"):
            manager.stage_records(
                duplicate,
                step=1,
                records=[replace(second[0], sample_id=first[0].sample_id)],
                metrics={},
            )
        manager.abort(duplicate)

        pending = manager.begin_transaction(completed_steps=2)
        manager.stage_records(
            pending,
            step=1,
            records=second,
            metrics={},
        )
        parallel = manager.begin_transaction(completed_steps=3)
        third = _rank_records(run_id, run_dir, rank=0, step=2, batch_size=1)
        with pytest.raises(ValueError, match="already staged or committed"):
            manager.stage_records(
                parallel,
                step=2,
                records=[replace(third[0], sample_id=second[0].sample_id)],
                metrics={},
            )
        manager.abort(parallel)
        manager.abort(pending)


def test_rank_shard_retention_groups_logical_steps_and_preserves_unknowns(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    rollout_root = run_dir / "rollouts"
    run_id = "rank-retention-run"

    with ArtifactManager(run_dir, run_id) as manager:
        for step in (0, 1):
            records = [
                *_rank_records(run_id, run_dir, rank=0, step=step, batch_size=1),
                *_rank_records(run_id, run_dir, rank=1, step=step, batch_size=1),
            ]
            transaction = manager.begin_transaction(completed_steps=step + 1)
            manager.stage_records(
                transaction,
                step=step,
                records=records,
                metrics={"loss": float(step)},
            )
            manager.commit(transaction)
            for rank in (0, 1):
                _write_triplet(rollout_root / f"rank_{rank:04d}", step)

        unknown = rollout_root / "rank_0000" / "operator-notes.txt"
        unknown.write_text("keep", encoding="utf-8")
        unknown_rank = _write_triplet(rollout_root / "rank_custom", 0)

        external = tmp_path / "external.pt"
        external.write_bytes(b"outside")
        unsafe_dir = rollout_root / "rank_0002"
        unsafe_dir.mkdir()
        unsafe_base = unsafe_dir / "batch_000000"
        unsafe_tensor = unsafe_base.with_suffix(".pt")
        unsafe_tensor.symlink_to(external)
        unsafe_media = unsafe_base.with_suffix(".media.pt")
        unsafe_metadata = unsafe_base.with_suffix(".json")
        unsafe_media.write_bytes(b"keep-media")
        unsafe_metadata.write_text("{}", encoding="utf-8")

        audit = manager.apply_retention(
            rollout_cache_keep_last=1,
            rollout_root=rollout_root,
        )
        deleted = next(row for row in audit if row["category"] == "rollout_cache")
        assert deleted["artifact_step"] == 0
        assert len(deleted["paths"]) == 6
        for rank in (0, 1):
            directory = rollout_root / f"rank_{rank:04d}"
            assert not (directory / "batch_000000.pt").exists()
            assert (directory / "batch_000001.pt").is_file()

        assert unknown.read_text(encoding="utf-8") == "keep"
        assert all(path.is_file() for path in unknown_rank)
        assert unsafe_tensor.is_symlink()
        assert unsafe_media.read_bytes() == b"keep-media"
        assert unsafe_metadata.is_file()
        assert external.read_bytes() == b"outside"

        budget_audit = manager.apply_retention(
            rollout_cache_max_bytes=0,
            rollout_root=rollout_root,
        )
        assert budget_audit[0]["artifact_step"] == 1
        assert len(budget_audit[0]["paths"]) == 6
        assert unknown.is_file()
        assert all(path.is_file() for path in unknown_rank)
        assert unsafe_tensor.is_symlink()
        assert external.read_bytes() == b"outside"
