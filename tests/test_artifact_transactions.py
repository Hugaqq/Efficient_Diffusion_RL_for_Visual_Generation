"""Checkpoint-cycle transaction and retention tests for run artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import visual_rl.artifacts.manager as artifact_manager_module
from visual_rl.artifacts import ArtifactManager, SampleManifest
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext


def _step_payload(step: int) -> tuple[RolloutBatch, RewardBatch, dict[str, float]]:
    sample_id = [f"sample-{step}"]
    batch = RolloutBatch(
        prompts=[f"prompt {step}"],
        metadata=[{"prompt_id": f"prompt-{step}"}],
        media=torch.zeros(1, 3, 2, 2),
        latents=torch.zeros(1, 1, 1, 2, 2),
        next_latents=torch.ones(1, 1, 1, 2, 2),
        timesteps=torch.tensor([[step]]),
        old_log_probs=torch.zeros(1, 1),
        kl=torch.zeros(1, 1),
        sample_id=sample_id,
        context=StepContext(step=step, seed=100 + step, epoch_tag=step),
    )
    rewards = RewardBatch(
        raw={"score": torch.tensor([float(step + 1)])},
        weighted={"score": torch.tensor([float(step + 1)])},
        weighted_total=torch.tensor([float(step + 1)]),
        valid_mask=torch.tensor([True]),
        sample_id=sample_id,
    )
    return batch, rewards, {"loss": float(step) / 10}


def _stage(manager: ArtifactManager, transaction, step: int) -> None:
    batch, rewards, metrics = _step_payload(step)
    manager.stage_step(
        transaction,
        step=step,
        batch=batch,
        rewards=rewards,
        metrics=metrics,
        media_type="image",
    )


def _checkpoint(path: Path, value: str) -> Path:
    path.mkdir(parents=True)
    (path / "training_state.pt").write_text(value, encoding="utf-8")
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_manifest_payload() -> dict:
    return {
        "run_id": "legacy-run",
        "records": [
            {
                "run_id": "legacy-run",
                "sample_id": "legacy-sample",
                "sample_index": 0,
                "step": 0,
                "prompt": "legacy prompt",
                "media_type": "image",
                "prompt_metadata": {},
            }
        ],
    }


def test_manifest_requires_v2_and_legacy_migration_is_explicit(tmp_path):
    legacy = _legacy_manifest_payload()
    with pytest.raises(ValueError, match="missing schema_version"):
        SampleManifest.from_dict(legacy)
    with pytest.raises(ValueError, match="Unsupported SampleManifest"):
        SampleManifest.from_dict(
            {"schema_version": "999", "run_id": "future", "records": []}
        )

    migrated = SampleManifest.migrate_legacy_to_v2(legacy)
    assert "schema_version" not in legacy
    assert "prompt_id" not in legacy["records"][0]
    assert migrated.schema_version == "2"
    assert migrated.records[0].prompt_id is None
    assert migrated.records[0].group_id is None
    assert migrated.records[0].branch_id is None

    path = tmp_path / "sample_manifest.json"
    migrated.save(path)
    assert SampleManifest.load(path).to_dict() == migrated.to_dict()


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"2","run_id":"a","run_id":"b","records":[]}',
        '{"schema_version":"2","run_id":"a","records":[],"value":NaN}',
    ],
)
def test_manifest_load_rejects_duplicate_and_non_finite_json(tmp_path, payload):
    path = tmp_path / "sample_manifest.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate|Non-finite"):
        SampleManifest.load(path)


def test_writer_lock_is_nonblocking_and_released_by_context_manager(tmp_path):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "lock-run"):
        with pytest.raises(RuntimeError, match="active writer"):
            ArtifactManager(run_dir, "lock-run", resume=True)

    with ArtifactManager(run_dir, "lock-run", resume=True):
        pass


def test_writer_lock_can_be_repeatedly_acquired_and_released_without_closing(tmp_path):
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "lock-run", config={})

    manager.acquire_writer_lock()
    manager.acquire_writer_lock()
    batch, rewards, metrics = _step_payload(0)
    manager.record(
        step=0,
        batch=batch,
        rewards=rewards,
        metrics=metrics,
        media_type="image",
    )
    with pytest.raises(RuntimeError, match="active writer"):
        ArtifactManager(run_dir, "lock-run", resume=True)

    manager.release_writer_lock()
    manager.release_writer_lock()
    with ArtifactManager(run_dir, "lock-run", resume=True):
        pass

    manager.acquire_writer_lock()
    manager.release_writer_lock()
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.acquire_writer_lock()


@pytest.mark.parametrize("operation_name", ["ftruncate", "write", "fsync"])
def test_writer_lock_initialization_failure_unlocks_and_closes_fd(
    tmp_path,
    monkeypatch,
    operation_name,
):
    run_dir = tmp_path / "run"
    real_operation = getattr(artifact_manager_module.os, operation_name)
    real_flock = artifact_manager_module.fcntl.flock
    unlocked_fds = []

    def tracking_flock(fd, operation):
        if operation == artifact_manager_module.fcntl.LOCK_UN:
            unlocked_fds.append(fd)
        return real_flock(fd, operation)

    def fail_lock_initialization(*_args, **_kwargs):
        raise OSError(f"injected {operation_name} failure")

    monkeypatch.setattr(artifact_manager_module.fcntl, "flock", tracking_flock)
    monkeypatch.setattr(
        artifact_manager_module.os,
        operation_name,
        fail_lock_initialization,
    )

    with pytest.raises(OSError, match=f"injected {operation_name} failure"):
        ArtifactManager(run_dir, "lock-run")

    assert len(unlocked_fds) == 1
    with pytest.raises(OSError):
        os.fstat(unlocked_fds[0])

    monkeypatch.setattr(
        artifact_manager_module.os,
        operation_name,
        real_operation,
    )
    with ArtifactManager(run_dir, "lock-run"):
        pass


@pytest.mark.parametrize(
    "failure_point",
    ["_load_commit_markers", "_load_manifest", "_load_metrics", "_write_json"],
)
def test_constructor_failure_after_lock_releases_writer_for_retry(
    tmp_path,
    monkeypatch,
    failure_point,
):
    run_dir = tmp_path / "run"
    original = getattr(ArtifactManager, failure_point)
    failed = False

    def fail_once(manager, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            assert manager._lock_fd is not None
            raise OSError(f"injected {failure_point} failure")
        return original(manager, *args, **kwargs)

    monkeypatch.setattr(ArtifactManager, failure_point, fail_once)
    constructor_kwargs = {"config": {}} if failure_point == "_write_json" else {}

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        ArtifactManager(run_dir, "retry-run", **constructor_kwargs)

    with ArtifactManager(run_dir, "retry-run", resume=True):
        pass


def test_projection_rebuild_failure_after_lock_releases_writer_for_retry(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "rebuild-run") as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        manager.commit(transaction)

    original = ArtifactManager.rebuild_projections
    failed = False

    def fail_once(manager):
        nonlocal failed
        if not failed:
            failed = True
            assert manager._lock_fd is not None
            raise OSError("injected projection rebuild failure")
        return original(manager)

    monkeypatch.setattr(ArtifactManager, "rebuild_projections", fail_once)
    with pytest.raises(OSError, match="injected projection rebuild failure"):
        ArtifactManager(run_dir, "rebuild-run", resume=True)

    with ArtifactManager(run_dir, "rebuild-run", resume=True):
        pass


def test_output_dir_resolves_symlink_ancestors_but_rejects_own_symlink(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    requested = alias_parent / "run"
    with ArtifactManager(requested, "canonical-run") as manager:
        assert manager.output_dir == (real_parent / "run").resolve()
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        manager.commit(transaction)
    assert (real_parent / "run" / "commits" / "commit_000001.json").is_file()

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_symlink = tmp_path / "output-symlink"
    output_symlink.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        ArtifactManager(output_symlink, "rejected-run")


def test_persisted_config_is_recursively_redacted_without_mutating_runtime(tmp_path):
    run_dir = tmp_path / "run"
    runtime_config = {
        "api_key": "top-secret",
        "api_key_env": "VISUAL_RL_API_KEY",
        "tokenizer": "keep-this-name",
        "nested": {
            "Authorization": "Bearer secret-token",
            "endpoint": (
                "https://user:password@reward.example:8443/private/path"
                "?token=query-secret#fragment"
            ),
            "encoded_url": (
                "https%3A%2F%2Fuser%3Apassword%40encoded.example%2Fprivate"
                "%3Fapi_key%3Dsecret"
            ),
        },
    }

    with ArtifactManager(
        run_dir,
        "redacted-run",
        config=runtime_config,
    ):
        pass

    persisted = _json(run_dir / "config.resolved.json")
    assert persisted["api_key"] == "[REDACTED]"
    assert persisted["api_key_env"] == "VISUAL_RL_API_KEY"
    assert persisted["tokenizer"] == "keep-this-name"
    assert persisted["nested"]["Authorization"] == "[REDACTED]"
    assert persisted["nested"]["endpoint"] == "https://reward.example:8443"
    assert persisted["nested"]["encoded_url"] == "https://encoded.example"
    assert runtime_config["api_key"] == "top-secret"
    assert "private/path" in runtime_config["nested"]["endpoint"]


def test_multi_step_commit_is_atomic_versioned_and_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "txn-run") as manager:
        transaction = manager.begin_transaction(completed_steps=2)
        _stage(manager, transaction, 0)
        _stage(manager, transaction, 1)
        checkpoint = _checkpoint(
            transaction.staging_dir / "checkpoint_000002",
            "checkpoint two",
        )

        assert not (run_dir / "sample_manifest.json").exists()
        assert not (run_dir / "commits" / "commit_000002.json").exists()

        marker = manager.commit(transaction, checkpoint_path=checkpoint)
        assert manager.commit(transaction) == marker
        assert marker["schema_version"] == "1"
        assert marker["artifact_step_semantics"] == "zero_based"
        assert marker["completed_steps_semantics"] == "one_based_count"
        assert marker["staged_steps"] == [0, 1]
        assert marker["completed_steps"] == 2
        assert (run_dir / "checkpoint_000002" / "training_state.pt").is_file()

        manifest = _json(run_dir / "sample_manifest.json")
        rewards = _json(run_dir / "reward_table.json")
        metrics = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert manifest["schema_version"] == "2"
        assert [record["step"] for record in manifest["records"]] == [0, 1]
        assert rewards["schema_version"] == "2"
        assert {row["schema_version"] for row in rewards["records"]} == {"2"}
        assert [row["step"] for row in metrics] == [0, 1]
        assert {row["schema_version"] for row in metrics} == {"2"}

        duplicate = manager.begin_transaction(completed_steps=2)
        with pytest.raises(FileExistsError, match="already committed"):
            _stage(manager, duplicate, 1)
        manager.abort(duplicate)


def test_commit_fsyncs_all_staged_files_before_publishing_marker(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "fsync-run") as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        checkpoint = _checkpoint(
            transaction.staging_dir / "checkpoint_000001",
            "checkpoint one",
        )
        extra_payload = transaction.staging_dir / "directly-staged.bin"
        extra_payload.write_bytes(b"direct payload")
        marker_path = manager._commit_path(1)
        fsync_audit = []
        original_fsync_tree = manager._fsync_tree

        def audit_fsync_tree(path):
            fsync_audit.append(
                {
                    "path": path,
                    "marker_exists": marker_path.exists(),
                    "regular_files": {
                        child.relative_to(path).as_posix()
                        for child in path.rglob("*")
                        if child.is_file() and not child.is_symlink()
                    },
                }
            )
            original_fsync_tree(path)

        monkeypatch.setattr(manager, "_fsync_tree", audit_fsync_tree)
        manager.commit(transaction, checkpoint_path=checkpoint)

    staging_audit = next(
        row for row in fsync_audit if row["path"] == transaction.staging_dir
    )
    assert staging_audit["marker_exists"] is False
    assert "directly-staged.bin" in staging_audit["regular_files"]
    assert "checkpoint_000001/training_state.pt" in staging_audit["regular_files"]
    assert {
        "step_000000/manifest_records.json",
        "step_000000/reward_rows.json",
        "step_000000/metric.json",
    } <= staging_audit["regular_files"]


def test_commit_does_not_publish_marker_when_staging_fsync_fails(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "fsync-failure-run") as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        checkpoint = _checkpoint(
            transaction.staging_dir / "checkpoint_000001",
            "checkpoint one",
        )

        def fail_fsync_tree(_path):
            raise OSError("injected staging fsync failure")

        monkeypatch.setattr(manager, "_fsync_tree", fail_fsync_tree)
        with pytest.raises(OSError, match="injected staging fsync failure"):
            manager.commit(transaction, checkpoint_path=checkpoint)

        assert not manager._commit_path(1).exists()
        assert checkpoint.is_dir()
        assert not (run_dir / "checkpoint_000001").exists()


def test_recover_finishes_checkpoint_moved_before_commit_marker(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "recovery-run")
    transaction = manager.begin_transaction(completed_steps=1)
    _stage(manager, transaction, 0)
    checkpoint = _checkpoint(
        transaction.staging_dir / "checkpoint_000001",
        "recover me",
    )
    original_write_json = manager._write_json

    def crash_before_marker(path, data):
        if path.parent == manager.commits_dir:
            raise OSError("simulated marker crash")
        return original_write_json(path, data)

    monkeypatch.setattr(manager, "_write_json", crash_before_marker)
    with pytest.raises(OSError, match="simulated marker crash"):
        manager.commit(transaction, checkpoint_path=checkpoint)
    assert (run_dir / "checkpoint_000001" / "training_state.pt").is_file()
    assert not (run_dir / "commits" / "commit_000001.json").exists()
    assert not (run_dir / "sample_manifest.json").exists()
    assert _json(transaction.staging_dir / "pending.json")["checkpoint"]["staged_path"]
    with pytest.raises(RuntimeError, match="already published"):
        manager.abort(transaction)
    assert transaction.staging_dir.is_dir()
    manager.close()

    with ArtifactManager(run_dir, "recovery-run", resume=True) as recovered:
        audit = recovered.recover()
        assert audit == [
            {
                "action": "recovered",
                "transaction_id": transaction.transaction_id,
                "completed_steps": 1,
            }
        ]
        assert (run_dir / "commits" / "commit_000001.json").is_file()
        assert [
            record["step"]
            for record in _json(run_dir / "sample_manifest.json")["records"]
        ] == [0]


def test_recover_rejects_checkpoint_changed_after_ready_journal(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "tampered-recovery-run")
    transaction = manager.begin_transaction(completed_steps=1)
    _stage(manager, transaction, 0)
    checkpoint = _checkpoint(
        transaction.staging_dir / "checkpoint_000001",
        "original checkpoint",
    )
    original_write_json = manager._write_json

    def crash_before_marker(path, data):
        if path == manager._commit_path(1):
            raise OSError("simulated marker crash")
        return original_write_json(path, data)

    monkeypatch.setattr(manager, "_write_json", crash_before_marker)
    with pytest.raises(OSError, match="simulated marker crash"):
        manager.commit(transaction, checkpoint_path=checkpoint)
    manager.close()

    (run_dir / "checkpoint_000001" / "training_state.pt").write_text(
        "tampered checkpoint",
        encoding="utf-8",
    )
    with ArtifactManager(run_dir, "tampered-recovery-run", resume=True) as recovered:
        audit = recovered.recover()

    assert audit[0]["action"] == "quarantine"
    assert "does not match ready journal" in audit[0]["reason"]
    assert not (run_dir / "commits" / "commit_000001.json").exists()


def test_resume_rejects_checkpoint_changed_after_commit_marker(tmp_path):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "post-commit-tamper-run") as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        checkpoint = _checkpoint(
            transaction.staging_dir / "checkpoint_000001",
            "original checkpoint",
        )
        manager.commit(transaction, checkpoint_path=checkpoint)

    (run_dir / "checkpoint_000001" / "training_state.pt").write_text(
        "tampered after commit",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint tree SHA256 mismatch"):
        ArtifactManager(run_dir, "post-commit-tamper-run", resume=True)


def test_recover_rejects_staging_root_symlink_before_traversal(tmp_path):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "staging-link-run"):
        pass
    outside = tmp_path / "outside-staging"
    outside.mkdir()
    (run_dir / ".staging").symlink_to(outside, target_is_directory=True)

    with ArtifactManager(run_dir, "staging-link-run", resume=True) as manager:
        with pytest.raises(ValueError, match="staging root cannot be a symlink"):
            manager.recover()


def test_corrupt_authoritative_marker_fails_closed_and_releases_lock(tmp_path):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "corrupt-marker-run") as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        manager.commit(transaction)

    marker_path = run_dir / "commits" / "commit_000001.json"
    original_marker = marker_path.read_text(encoding="utf-8")
    marker_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative commit marker"):
        ArtifactManager(run_dir, "corrupt-marker-run", resume=True)

    marker_path.write_text(original_marker, encoding="utf-8")
    with ArtifactManager(run_dir, "corrupt-marker-run", resume=True):
        pass


def test_overlapping_authoritative_markers_fail_closed(tmp_path):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "overlap-run") as manager:
        first = manager.begin_transaction(completed_steps=1)
        _stage(manager, first, 0)
        manager.commit(first)
        second = manager.begin_transaction(completed_steps=2)
        _stage(manager, second, 1)
        manager.commit(second)

    first_marker = _json(run_dir / "commits" / "commit_000001.json")
    second_path = run_dir / "commits" / "commit_000002.json"
    second_marker = _json(second_path)
    second_marker["staged_steps"] = [0, 1]
    second_marker["steps"] = [first_marker["steps"][0], second_marker["steps"][0]]
    second_path.write_text(json.dumps(second_marker), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap artifact steps"):
        ArtifactManager(run_dir, "overlap-run", resume=True)


def test_keyboard_interrupt_after_marker_propagates_without_losing_commit(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "interrupt-run")
    transaction = manager.begin_transaction(completed_steps=1)
    _stage(manager, transaction, 0)

    def interrupt_projection():
        raise KeyboardInterrupt("injected interrupt")

    monkeypatch.setattr(manager, "rebuild_projections", interrupt_projection)
    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        manager.commit(transaction)

    assert transaction.state == "committed"
    assert (run_dir / "commits" / "commit_000001.json").is_file()
    manager.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_artifact_json_writer_rejects_nonfinite_numbers(tmp_path, value):
    run_dir = tmp_path / "run"
    target = run_dir / "nonfinite.json"
    with ArtifactManager(run_dir, "finite-json-run") as manager:
        with pytest.raises(ValueError, match="Out of range float values"):
            manager._write_json(target, {"value": value})
    assert not target.exists()


def test_recover_ignores_open_transaction_and_rebuild_excludes_it(tmp_path):
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "open-run")
    transaction = manager.begin_transaction(completed_steps=1)
    _stage(manager, transaction, 0)
    manager.close()

    with ArtifactManager(run_dir, "open-run", resume=True) as recovered:
        audit = recovered.recover()
        assert audit[0]["action"] == "ignored"
        assert not (run_dir / "sample_manifest.json").exists()
        assert not (run_dir / "commits" / "commit_000001.json").exists()


def test_retention_deletes_only_committed_known_objects_by_whole_group(tmp_path):
    run_dir = tmp_path / "run"
    rollout_dir = run_dir / "rollouts"
    with ArtifactManager(run_dir, "retention-run") as manager:
        for step in (0, 1):
            transaction = manager.begin_transaction(completed_steps=step + 1)
            _stage(manager, transaction, step)
            checkpoint = _checkpoint(
                transaction.staging_dir / f"checkpoint_{step + 1:06d}",
                f"checkpoint {step}",
            )
            manager.commit(transaction, checkpoint_path=checkpoint)
            rollout_dir.mkdir(exist_ok=True)
            base = rollout_dir / f"batch_{step:06d}"
            base.with_suffix(".pt").write_bytes(b"tensor" * (step + 1))
            base.with_suffix(".media.pt").write_bytes(b"media" * (step + 1))
            base.with_suffix(".json").write_text("{}", encoding="utf-8")

        unknown_run_file = run_dir / "user-notes.txt"
        unknown_rollout_file = rollout_dir / "batch_custom.pt"
        unknown_run_file.write_text("keep", encoding="utf-8")
        unknown_rollout_file.write_text("keep", encoding="utf-8")

        audit = manager.apply_retention(
            checkpoint_keep_last=1,
            rollout_cache_keep_last=1,
            rollout_root=rollout_dir,
        )
        assert not (run_dir / "checkpoint_000001").exists()
        assert (run_dir / "checkpoint_000002").is_dir()
        assert not (rollout_dir / "batch_000000.pt").exists()
        assert not (rollout_dir / "batch_000000.media.pt").exists()
        assert not (rollout_dir / "batch_000000.json").exists()
        assert (rollout_dir / "batch_000001.pt").is_file()
        assert unknown_run_file.read_text(encoding="utf-8") == "keep"
        assert unknown_rollout_file.read_text(encoding="utf-8") == "keep"
        assert {row["category"] for row in audit} == {
            "checkpoint",
            "rollout_cache",
        }

        budget_audit = manager.apply_retention(
            rollout_cache_max_bytes=0,
            rollout_root=rollout_dir,
        )
        assert not (rollout_dir / "batch_000001.pt").exists()
        assert budget_audit[0]["artifact_step"] == 1
        assert unknown_rollout_file.is_file()


def test_retention_preserves_latest_existing_checkpoint_even_with_zero_budgets(
    tmp_path,
):
    run_dir = tmp_path / "run"
    with ArtifactManager(run_dir, "recovery-point-run") as manager:
        for step in (0, 1):
            transaction = manager.begin_transaction(completed_steps=step + 1)
            _stage(manager, transaction, step)
            checkpoint = _checkpoint(
                transaction.staging_dir / f"checkpoint_{step + 1:06d}",
                f"checkpoint {step}",
            )
            manager.commit(transaction, checkpoint_path=checkpoint)

        unknown = run_dir / "user-owned.bin"
        unknown.write_bytes(b"do not delete")
        count_audit = manager.apply_retention(checkpoint_keep_last=0)
        assert not (run_dir / "checkpoint_000001").exists()
        assert (run_dir / "checkpoint_000002" / "training_state.pt").is_file()
        assert [row["completed_steps"] for row in count_audit] == [1]

        budget_audit = manager.apply_retention(artifact_max_bytes=0)
        assert (run_dir / "checkpoint_000002" / "training_state.pt").is_file()
        assert unknown.read_bytes() == b"do not delete"
        assert budget_audit[-1]["action"] == "budget_unsatisfied"
        assert budget_audit[-1]["max_bytes"] == 0
        assert "latest_checkpoint" in budget_audit[-1]["reason"]


def test_projection_failure_after_marker_is_recoverable(tmp_path):
    run_dir = tmp_path / "run"
    external = tmp_path / "external.json"
    external.write_text("unchanged", encoding="utf-8")
    with ArtifactManager(run_dir, "safe-run") as manager:
        transaction = manager.begin_transaction(completed_steps=1)
        _stage(manager, transaction, 0)
        (run_dir / "sample_manifest.json").symlink_to(external)
        marker = manager.commit(transaction)

        assert marker["completed_steps"] == 1
        assert transaction.state == "committed"
        assert (run_dir / "commits" / "commit_000001.json").is_file()
        assert len(manager.post_commit_errors) == 1
        assert manager.post_commit_errors[0]["operation"] == "projection_refresh"
        assert "symlink" in manager.post_commit_errors[0]["error"]
    assert external.read_text(encoding="utf-8") == "unchanged"

    (run_dir / "sample_manifest.json").unlink()
    with ArtifactManager(run_dir, "safe-run", resume=True) as recovered:
        assert recovered.recover() == []
        assert recovered.rebuild_projections() == {
            "commits": 1,
            "steps": 1,
            "records": 1,
        }
    assert _json(run_dir / "sample_manifest.json")["records"][0]["step"] == 0


def test_staging_cleanup_failure_after_marker_is_recovered(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, "cleanup-run")
    transaction = manager.begin_transaction(completed_steps=1)
    _stage(manager, transaction, 0)
    original_rmtree = manager._safe_rmtree

    def fail_staging_cleanup(path):
        if path == transaction.staging_dir:
            raise OSError("injected cleanup failure")
        return original_rmtree(path)

    monkeypatch.setattr(manager, "_safe_rmtree", fail_staging_cleanup)
    marker = manager.commit(transaction)

    assert marker["completed_steps"] == 1
    assert transaction.staging_dir.is_dir()
    assert manager.post_commit_errors == [
        {
            "operation": "staging_cleanup",
            "error": "OSError: injected cleanup failure",
        }
    ]
    manager.close()

    with ArtifactManager(run_dir, "cleanup-run", resume=True) as recovered:
        assert recovered.recover() == [
            {
                "action": "cleanup",
                "transaction_id": transaction.transaction_id,
                "reason": "commit_marker_exists",
            }
        ]
    assert not transaction.staging_dir.exists()
