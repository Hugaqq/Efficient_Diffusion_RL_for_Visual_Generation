"""Final v3 manifest and v2 authoritative artifact transaction contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import visual_rl.artifacts.manager as manager_module
from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.artifacts.manager import (
    COMMIT_SCHEMA_VERSION,
    CORE_METRIC_SCHEMA_VERSION,
    ArtifactManager,
    read_authoritative_commit_chain,
)
from visual_rl.artifacts.manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleRecord,
)
from visual_rl.core.types import (
    FrozenMapping,
    RewardBatch,
    RolloutBatch,
    StepContext,
)
from visual_rl.errors import ArtifactError, ResumeError


def _batch(
    step: int,
    *,
    rollout_type: str = "full_trajectory",
) -> RolloutBatch:
    selected = (
        torch.tensor([1], dtype=torch.int64)
        if rollout_type == "single_step"
        else None
    )
    branch_step = (
        torch.tensor([0], dtype=torch.int64)
        if rollout_type == "branching"
        else None
    )
    trajectory_step = (
        torch.tensor([0, 1], dtype=torch.int64)
        if rollout_type == "branching"
        else None
    )
    return RolloutBatch(
        prompts=(f"prompt {step}",),
        metadata=({"source": "fixture"},),
        media=torch.zeros(1, 3, 2, 2),
        latents=torch.zeros(1, 2, 1),
        next_latents=torch.ones(1, 2, 1),
        timesteps=torch.tensor([[9, 4]], dtype=torch.int64),
        old_log_probs=torch.zeros(1, 2),
        transition_mask=torch.ones(1, 2, dtype=torch.bool),
        sample_id=(f"sample-{step}",),
        prompt_id=(f"prompt-id-{step}",),
        group_id=(f"group-{step}",),
        branch_id=(0,) if rollout_type == "branching" else None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=step, seed=100 + step),
        selected_timestep_index=selected,
        flash_coefficient=(
            torch.ones(1, 1) if rollout_type == "single_step" else None
        ),
        branch_step_index=branch_step,
        trajectory_step_index=trajectory_step,
        transition_std_dev=(
            torch.ones(1, 2) if rollout_type == "branching" else None
        ),
        recompute_payload={"features": torch.ones(1, 2)},
        artifact_metadata={"adapter": "fixture", "nested": {"value": 1}},
    )


def _rewards(batch: RolloutBatch) -> RewardBatch:
    value = torch.tensor([float(batch.context.step + 1)], dtype=torch.float32)
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": value},
        weighted={"score": value},
        weighted_total=value,
        valid_mask=torch.ones(1, dtype=torch.bool),
        shared_metadata={"score": {"revision": "r1"}},
        sample_metadata={"score": ({"echo": batch.sample_id[0]},)},
    )


def _record(
    step: int,
    *,
    checkpoint_path: str | None = None,
    media_path: str | None = None,
) -> SampleRecord:
    batch = _batch(step)
    record = ManifestBuilder(
        run_id="run-test",
        media_type="image",
        rollout_type="full_trajectory",
    ).build_records(
        batch,
        _rewards(batch),
        media_paths=(None,),
    )[0]
    return replace(
        record,
        checkpoint_path=checkpoint_path,
        media_path=media_path,
    )


def _metrics(step: int) -> Any:
    return SimpleNamespace(
        values=FrozenMapping(
            {
                "loss": float(step) / 10.0,
                "reward_mean": float(step + 1),
            }
        ),
        sample_count=1,
        active_transition_count=2,
    )


def _checkpoint(path: Path, value: str) -> Path:
    path.mkdir(parents=True)
    (path / "adapter").mkdir()
    (path / "adapter" / "adapter.json").write_text(
        '{"format":"fixture"}\n',
        encoding="utf-8",
    )
    (path / "adapter" / "adapter_state.pt").write_text(
        value,
        encoding="utf-8",
    )
    (path / "training_state.pt").write_text(value, encoding="utf-8")
    (path / "checkpoint.json").write_text(
        '{"format_version":5}\n',
        encoding="utf-8",
    )
    return path


def _stage(
    manager: ArtifactManager,
    transaction,
    step: int,
    *,
    checkpoint_path: str | None = None,
    media_path: str | None = None,
) -> None:
    manager.stage_records(
        transaction,
        step=step,
        records=(
            _record(
                step,
                checkpoint_path=checkpoint_path,
                media_path=media_path,
            ),
        ),
        metrics=_metrics(step),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_sample_record_and_manifest_are_exact_frozen_v3_contracts(tmp_path) -> None:
    record = _record(0)
    row = record.to_plain_dict()

    assert SAMPLE_MANIFEST_SCHEMA_VERSION == "3"
    assert set(row) == {
        "run_id",
        "sample_id",
        "sample_index",
        "step",
        "rank",
        "prompt",
        "media_type",
        "prompt_metadata",
        "seed",
        "rollout_type",
        "timestep_summary",
        "reward_values",
        "media_path",
        "rollout_cache_path",
        "checkpoint_path",
        "model_metadata",
        "prompt_id",
        "group_id",
        "branch_id",
    }
    assert SampleRecord.from_dict(row) == record
    with pytest.raises(ValueError, match="fields do not match"):
        SampleRecord.from_dict({**row, "legacy": True})
    with pytest.raises(ValueError, match="fields do not match"):
        SampleRecord.from_dict(
            {name: value for name, value in row.items() if name != "rank"}
        )

    manifest = SampleManifest(run_id="run-test", records=(record,))
    path = tmp_path / "sample_manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), allow_nan=False),
        encoding="utf-8",
    )
    assert SampleManifest.load(path) == manifest

    old = manifest.to_dict()
    old["schema_version"] = "2"
    with pytest.raises(ValueError, match="Unsupported"):
        SampleManifest.from_dict(old)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"3","run_id":"a","run_id":"b","records":[]}',
        '{"schema_version":"3","run_id":"a","records":[],"value":NaN}',
    ],
)
def test_manifest_load_rejects_duplicate_and_nonfinite_json(
    tmp_path,
    payload: str,
) -> None:
    path = tmp_path / "sample_manifest.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate|Non-finite"):
        SampleManifest.load(path)


@pytest.mark.parametrize(
    ("rollout_type", "expected"),
    [
        (
            "full_trajectory",
            {"values": [9, 4], "count": 2},
        ),
        (
            "single_step",
            {
                "values": [9, 4],
                "count": 2,
                "selected_timestep_index": 1,
            },
        ),
        (
            "branching",
            {
                "values": [9, 4],
                "count": 2,
                "branch_step_index": 0,
                "trajectory_step_index": [0, 1],
            },
        ),
    ],
)
def test_manifest_builder_uses_only_typed_batch_fields(
    rollout_type: str,
    expected: dict[str, Any],
) -> None:
    batch = _batch(2, rollout_type=rollout_type)
    records = ManifestBuilder(
        run_id="run-test",
        media_type="image",
        rollout_type=rollout_type,
    ).build_records(
        batch,
        _rewards(batch),
        media_paths=("previews/step_000002/rank_0/sample_000000.jpg",),
    )

    assert len(records) == 1
    record = records[0]
    assert record.to_plain_dict()["timestep_summary"] == expected
    assert record.step == batch.context.step
    assert record.seed == batch.context.seed
    assert record.rank == batch.context.rank
    assert record.media_path == (
        "previews/step_000002/rank_0/sample_000000.jpg"
    )
    assert record.rollout_cache_path is None
    assert record.checkpoint_path is None
    assert dict(record.model_metadata) == {
        "adapter": "fixture",
        "nested": FrozenMapping({"value": 1}),
    }
    reward = dict(record.reward_values)
    assert reward == {
        "raw": FrozenMapping({"score": 3.0}),
        "weighted": FrozenMapping({"score": 3.0}),
        "weighted_total": 3.0,
        "valid": True,
        "shared_metadata": FrozenMapping(
            {"score": FrozenMapping({"revision": "r1"})}
        ),
        "sample_metadata": FrozenMapping(
            {
                "score": FrozenMapping(
                    {"echo": "sample-2"}
                )
            }
        ),
    }


def test_fresh_manager_owns_lock_and_only_resolved_config_writer(tmp_path) -> None:
    run_dir = tmp_path / "run"
    config = {
        "endpoint": "https://user:password@example.test/path?token=secret",
        "api_token": "secret",
        "nested": {"value": 2},
    }
    with ArtifactManager(run_dir, "run-test", config=config) as manager:
        persisted = _read_json(manager.config_path)
        assert persisted == {
            "api_token": "[REDACTED]",
            "endpoint": "https://example.test",
            "nested": {"value": 2},
        }
        with pytest.raises(RuntimeError, match="active writer"):
            ArtifactManager.open_resume(run_dir)
        manager.write_resolved_config({"value": 3})
        assert _read_json(manager.config_path) == {"value": 3}


def test_commit_v2_has_exact_marker_journal_and_staged_rows(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        _stage(manager, transaction, 0, checkpoint_path="checkpoint_000001")
        staged_wrapper = _read_json(
            transaction.staging_dir
            / "step_000000"
            / "manifest_records.json"
        )
        staged_metric = _read_json(
            transaction.staging_dir / "step_000000" / "metric.json"
        )
        checkpoint = _checkpoint(
            transaction.staging_dir / "checkpoint_000001",
            "one",
        )
        marker = manager.commit(transaction, checkpoint_path=checkpoint)

        assert COMMIT_SCHEMA_VERSION == "2"
        assert CORE_METRIC_SCHEMA_VERSION == "3"
        assert set(marker) == {
            "schema_version",
            "kind",
            "run_id",
            "transaction_id",
            "completed_steps",
            "staged_steps",
            "checkpoint",
            "steps",
        }
        assert set(marker["checkpoint"]) == {
            "completed_steps",
            "path",
            "tree_sha256",
        }
        assert set(marker["steps"][0]) == {
            "artifact_step",
            "manifest_records",
            "core_metric_row",
        }
        assert marker["steps"][0]["manifest_records"] == staged_wrapper["records"]
        assert marker["steps"][0]["core_metric_row"] == staged_metric
        assert set(staged_wrapper) == {
            "schema_version",
            "run_id",
            "artifact_step",
            "records",
        }
        assert staged_wrapper["schema_version"] == "3"
        assert staged_metric["schema_version"] == "3"
        journal = _read_json(transaction.staging_dir / "pending.json")
        assert set(journal) == {
            "schema_version",
            "kind",
            "state",
            "run_id",
            "transaction_id",
            "completed_steps",
            "staged_steps",
            "checkpoint",
        }
        assert journal["state"] == "ready"
        assert "reward_rows" not in journal
        assert not manager.manifest_path.exists()
        assert not manager.metrics_path.exists()

        manager.rebuild_projections()
        assert SampleManifest.load(manager.manifest_path).records[0] == _record(
            0,
            checkpoint_path="checkpoint_000001",
        )
        assert _read_json(manager.metrics_path) == staged_metric
        manager.cleanup_published_staging(transaction)
        assert not transaction.staging_dir.exists()

    names = {path.name for path in run_dir.iterdir()}
    assert names == {
        ".artifact.lock",
        ".staging",
        "commits",
        "config.resolved.json",
        "checkpoint_000001",
        "sample_manifest.json",
        "metrics.jsonl",
    }


def test_chain_reader_rebuilds_without_staging_and_retention_keeps_head(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        first = manager.begin_transaction()
        _stage(manager, first, 0, checkpoint_path="checkpoint_000001")
        manager.commit(
            first,
            checkpoint_path=_checkpoint(
                first.staging_dir / "checkpoint_000001",
                "one",
            ),
        )
        manager.rebuild_projections()
        manager.cleanup_published_staging(first)

        second = manager.begin_transaction()
        _stage(manager, second, 1, checkpoint_path="checkpoint_000002")
        manager.commit(
            second,
            checkpoint_path=_checkpoint(
                second.staging_dir / "checkpoint_000002",
                "two",
            ),
        )
        manager.rebuild_projections()
        expected_manifest = manager.manifest_path.read_bytes()
        expected_metrics = manager.metrics_path.read_bytes()
        manager.cleanup_published_staging(second)

        manager.manifest_path.unlink()
        manager.metrics_path.unlink()
        manager.rebuild_projections()
        assert manager.manifest_path.read_bytes() == expected_manifest
        assert manager.metrics_path.read_bytes() == expected_metrics

        chain = read_authoritative_commit_chain(
            run_dir,
            verify_checkpoint_trees=True,
        )
        assert [item["completed_steps"] for item in chain] == [1, 2]
        manager.apply_checkpoint_retention(keep_last=1)
        assert not (run_dir / "checkpoint_000001").exists()
        assert (run_dir / "checkpoint_000002").is_dir()
        assert len(
            read_authoritative_commit_chain(
                run_dir,
                verify_checkpoint_trees=True,
            )
        ) == 2


def test_ready_journal_recovers_after_checkpoint_publish_before_marker(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "run-test", config={})
    transaction = manager.begin_transaction()
    _stage(manager, transaction, 0, checkpoint_path="checkpoint_000001")
    checkpoint = _checkpoint(
        transaction.staging_dir / "checkpoint_000001",
        "one",
    )
    original_write = manager_module._atomic_write_json

    def fail_marker(path, value, *, root):
        if path.parent.name == "commits":
            raise OSError("injected marker failure")
        return original_write(path, value, root=root)

    monkeypatch.setattr(manager_module, "_atomic_write_json", fail_marker)
    with pytest.raises(OSError, match="injected marker failure"):
        manager.commit(transaction, checkpoint_path=checkpoint)
    assert (run_dir / "checkpoint_000001").is_dir()
    assert not (run_dir / "commits" / "commit_000001.json").exists()
    manager.close()

    monkeypatch.setattr(manager_module, "_atomic_write_json", original_write)
    with ArtifactManager.open_resume(run_dir) as recovered:
        recovered.recover()
        assert recovered.start_step == 1
        assert recovered.checkpoint_path == run_dir / "checkpoint_000001"
        assert recovered.manifest_path.is_file()
        assert recovered.metrics_path.is_file()
    assert not transaction.staging_dir.exists()


def test_preview_and_checkpoint_publish_in_one_authoritative_commit(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    relative = "previews/step_000000/rank_0/sample_000000.jpg"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        staged_preview = transaction.staging_dir / relative
        staged_preview.parent.mkdir(parents=True)
        staged_preview.write_bytes(b"jpeg-preview")
        _stage(
            manager,
            transaction,
            0,
            checkpoint_path="checkpoint_000001",
            media_path=relative,
        )
        marker = manager.commit(
            transaction,
            checkpoint_path=_checkpoint(
                transaction.staging_dir / "checkpoint_000001",
                "one",
            ),
        )

        assert not staged_preview.exists()
        assert (run_dir / relative).read_bytes() == b"jpeg-preview"
        assert marker["steps"][0]["manifest_records"][0]["media_path"] == relative
        manager.rebuild_projections()
        assert (
            SampleManifest.load(manager.manifest_path).records[0].media_path
            == relative
        )
        manager.cleanup_published_staging(transaction)


def test_ready_recovery_accepts_already_published_preview_and_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    relative = "previews/step_000000/rank_0/sample_000000.jpg"
    manager = ArtifactManager(run_dir, "run-test", config={})
    transaction = manager.begin_transaction()
    staged_preview = transaction.staging_dir / relative
    staged_preview.parent.mkdir(parents=True)
    staged_preview.write_bytes(b"jpeg-preview")
    _stage(
        manager,
        transaction,
        0,
        checkpoint_path="checkpoint_000001",
        media_path=relative,
    )
    checkpoint = _checkpoint(
        transaction.staging_dir / "checkpoint_000001",
        "one",
    )
    original_write = manager_module._atomic_write_json

    def fail_marker(path, value, *, root):
        if path.parent.name == "commits":
            raise OSError("injected marker failure after media publish")
        return original_write(path, value, root=root)

    monkeypatch.setattr(manager_module, "_atomic_write_json", fail_marker)
    with pytest.raises(OSError, match="after media publish"):
        manager.commit(transaction, checkpoint_path=checkpoint)
    assert (run_dir / relative).read_bytes() == b"jpeg-preview"
    assert (run_dir / "checkpoint_000001").is_dir()
    assert not staged_preview.exists()
    manager.close()

    monkeypatch.setattr(manager_module, "_atomic_write_json", original_write)
    with ArtifactManager.open_resume(run_dir) as recovered:
        recovered.recover()
        assert recovered.start_step == 1
        assert (run_dir / relative).read_bytes() == b"jpeg-preview"
        assert (
            SampleManifest.load(recovered.manifest_path).records[0].media_path
            == relative
        )
    assert not transaction.staging_dir.exists()


def test_ready_recovery_completes_partially_published_preview_sequence(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "run-test", config={})
    transaction = manager.begin_transaction()
    first = "previews/step_000000/rank_0/sample_000000.jpg"
    second = "previews/step_000001/rank_0/sample_000000.jpg"
    for relative, payload in ((first, b"first"), (second, b"second")):
        staged = transaction.staging_dir / relative
        staged.parent.mkdir(parents=True)
        staged.write_bytes(payload)
    _stage(manager, transaction, 0, media_path=first)
    _stage(
        manager,
        transaction,
        1,
        checkpoint_path="checkpoint_000002",
        media_path=second,
    )
    checkpoint = _checkpoint(
        transaction.staging_dir / "checkpoint_000002",
        "two",
    )
    original_publish = manager._publish_preview
    calls = 0

    def fail_second(active_transaction, relative):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second preview publish failure")
        return original_publish(active_transaction, relative)

    monkeypatch.setattr(manager, "_publish_preview", fail_second)
    with pytest.raises(OSError, match="second preview"):
        manager.commit(transaction, checkpoint_path=checkpoint)
    assert (run_dir / first).read_bytes() == b"first"
    assert not (transaction.staging_dir / first).exists()
    assert (transaction.staging_dir / second).read_bytes() == b"second"
    assert not (run_dir / second).exists()
    manager.close()

    with ArtifactManager.open_resume(run_dir) as recovered:
        recovered.recover()
        assert recovered.start_step == 2
        assert (run_dir / first).read_bytes() == b"first"
        assert (run_dir / second).read_bytes() == b"second"
        assert recovered.checkpoint_path == run_dir / "checkpoint_000002"


def test_record_cannot_reference_missing_or_noncanonical_preview(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        with pytest.raises(ArtifactError, match="no staged preview"):
            _stage(
                manager,
                transaction,
                0,
                media_path=(
                    "previews/step_000000/rank_0/sample_000000.jpg"
                ),
            )
        with pytest.raises(ValueError, match="canonical preview path"):
            _stage(
                manager,
                transaction,
                0,
                media_path="previews/wrong.jpg",
            )
        manager.abort(transaction)


def test_missing_preview_after_record_staging_is_fatal_before_marker(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    relative = "previews/step_000000/rank_0/sample_000000.jpg"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        staged_preview = transaction.staging_dir / relative
        staged_preview.parent.mkdir(parents=True)
        staged_preview.write_bytes(b"jpeg-preview")
        _stage(
            manager,
            transaction,
            0,
            checkpoint_path="checkpoint_000001",
            media_path=relative,
        )
        staged_preview.unlink()

        with pytest.raises(ArtifactError, match="preview is missing"):
            manager.commit(
                transaction,
                checkpoint_path=_checkpoint(
                    transaction.staging_dir / "checkpoint_000001",
                    "one",
                ),
            )

        assert not tuple((run_dir / "commits").glob("commit_*.json"))
        assert not (run_dir / relative).exists()
        assert transaction.state == "ready"


def test_abort_removes_unrecorded_staged_preview(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        result = manager.stage_previews(
            transaction,
            _batch(0),
            max_samples=1,
        )
        assert result.media_paths == (
            "previews/step_000000/rank_0/sample_000000.jpg",
        )
        staged = transaction.staging_dir / result.media_paths[0]
        assert staged.is_file()

        manager.abort(transaction)

        assert not transaction.staging_dir.exists()
        assert not (run_dir / result.media_paths[0]).exists()


def test_open_only_before_first_marker_is_not_a_resume_locator(tmp_path) -> None:
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "run-test", config={})
    transaction = manager.begin_transaction()
    _stage(manager, transaction, 0)
    manager.close()

    with pytest.raises(ResumeError, match="open-only"):
        ArtifactManager.open_resume(run_dir)


def test_chain_reader_rejects_old_commit_version_and_unknown_fields(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        _stage(manager, transaction, 0, checkpoint_path="checkpoint_000001")
        manager.commit(
            transaction,
            checkpoint_path=_checkpoint(
                transaction.staging_dir / "checkpoint_000001",
                "one",
            ),
        )
    marker_path = run_dir / "commits" / "commit_000001.json"
    marker = _read_json(marker_path)
    marker["schema_version"] = "1"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ArtifactError, match="unsupported"):
        read_authoritative_commit_chain(run_dir)

    marker["schema_version"] = "2"
    marker["latest"] = True
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ArtifactError, match="fields"):
        read_authoritative_commit_chain(run_dir)


def test_chain_reader_rejects_non_float_dynamic_core_metric(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        _stage(manager, transaction, 0, checkpoint_path="checkpoint_000001")
        manager.commit(
            transaction,
            checkpoint_path=_checkpoint(
                transaction.staging_dir / "checkpoint_000001",
                "one",
            ),
        )
    marker_path = run_dir / "commits" / "commit_000001.json"
    marker = _read_json(marker_path)
    marker["steps"][0]["core_metric_row"]["loss"] = 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ArtifactError, match="finite Python floats"):
        read_authoritative_commit_chain(run_dir)


def test_stage_rejects_noncontiguous_steps_and_unsafe_paths(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        with pytest.raises(ValueError, match="contiguous"):
            _stage(manager, transaction, 1)
        with pytest.raises(ValueError, match="normalized relative"):
            replace(_record(0), media_path="../escape")


@pytest.mark.parametrize("invalid_value", [True, 1])
def test_core_metric_v3_accepts_only_finite_python_floats(
    tmp_path,
    invalid_value,
) -> None:
    run_dir = tmp_path / "run"
    metrics = SimpleNamespace(
        values=FrozenMapping({"loss": invalid_value}),
        sample_count=1,
        active_transition_count=2,
    )
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        with pytest.raises(ValueError, match="finite Python floats"):
            manager.stage_records(
                transaction,
                step=0,
                records=(_record(0),),
                metrics=metrics,
            )
        assert not (
            transaction.staging_dir / "step_000000"
        ).exists()
        manager.abort(transaction)


def test_projection_rebuild_never_reads_journal_or_staging(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "run-test", config={}) as manager:
        transaction = manager.begin_transaction()
        _stage(manager, transaction, 0, checkpoint_path="checkpoint_000001")
        manager.commit(
            transaction,
            checkpoint_path=_checkpoint(
                transaction.staging_dir / "checkpoint_000001",
                "one",
            ),
        )

        original_load = manager_module._load_artifact_json
        observed: list[Path] = []

        def observe(path: Path):
            observed.append(path)
            if ".staging" in path.parts:
                raise AssertionError("projection read staging")
            return original_load(path)

        monkeypatch.setattr(manager_module, "_load_artifact_json", observe)
        manager.rebuild_projections()
        assert observed
        assert all(".staging" not in path.parts for path in observed)
