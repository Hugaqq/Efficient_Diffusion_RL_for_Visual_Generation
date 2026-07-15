"""Runner integration coverage for C10 transactions and observability."""

from __future__ import annotations

import json
import shutil

import pytest

from visual_rl.artifacts import ArtifactManager
from visual_rl.artifacts.audit import audit_run_artifacts
from visual_rl.artifacts.status import inspect_run_status, write_run_status
from visual_rl.configs.schema import VisualRLConfig
from visual_rl.model_adapters.mock import MockWanAdapter
import visual_rl.runner as runner_module
from visual_rl.runner import ExperimentRunner, ResumeError


def _config(output_dir, *, steps: int = 3, save_every: int = 2):
    config = VisualRLConfig(run_name="c10-runner")
    config.paths.output_dir = str(output_dir)
    config.dataset.prompts = ["first prompt", "second prompt"]
    config.train.max_steps = steps
    config.train.save_every = save_every
    config.runner.show_progress = False
    config.runner.strict_rollout_validation = True
    return config


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_runner_revalidates_direct_config_before_output_side_effects(tmp_path):
    output_dir = tmp_path / "invalid-direct"
    config = _config(output_dir)
    config.sample.batch_size = 0

    with pytest.raises(ValueError, match="sample.batch_size"):
        ExperimentRunner(config)

    assert not output_dir.exists()


def test_runner_commits_checkpoint_cycles_and_persists_runtime_metrics(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "run"))
    metrics = runner.run()

    commit_two = _json(runner.output_dir / "commits" / "commit_000002.json")
    commit_three = _json(runner.output_dir / "commits" / "commit_000003.json")
    assert commit_two["staged_steps"] == [0, 1]
    assert commit_three["staged_steps"] == [2]
    assert commit_two["checkpoint"]["path"] == "checkpoint_000002"
    assert commit_three["checkpoint"]["path"] == "checkpoint_000003"
    assert _json(runner.output_dir / "latest.json")["step"] == 3
    assert [row["step"] for row in metrics] == [0, 1, 2]
    required = {
        "rollout_time_s",
        "reward_time_s",
        "rollout_cache_time_s",
        "update_time_s",
        "checkpoint_write_time_s",
        "artifact_stage_time_s",
        "artifact_commit_time_s",
        "post_commit_bookkeeping_time_s",
        "checkpoint_time_s",
        "step_time_s",
        "samples_per_second",
        "peak_gpu_memory_bytes",
        "reward_cache_hit_rate",
        "reward_latency_p50_s",
        "reward_latency_p95_s",
    }
    assert all(required <= row.keys() for row in metrics)
    persisted = [
        json.loads(line)
        for line in (runner.output_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(required <= row.keys() for row in persisted)
    assert not list((runner.output_dir / ".staging").glob("txn_*"))


def test_runtime_boundaries_assign_commit_and_post_commit_delays(
    tmp_path,
    monkeypatch,
):
    clock = 0.0
    events = []

    def perf_counter():
        return clock

    def advance(name, seconds):
        nonlocal clock
        clock += seconds
        events.append(name)

    monkeypatch.setattr(runner_module.time, "perf_counter", perf_counter)
    runner = ExperimentRunner(_config(tmp_path / "run", steps=1, save_every=1))
    manager = runner.artifacts

    original_save = runner._save_checkpoint

    def delayed_save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        advance("checkpoint_write", 13.0)
        return result

    original_stage = manager.stage_step

    def delayed_stage(*args, **kwargs):
        result = original_stage(*args, **kwargs)
        advance("artifact_stage", 11.0)
        return result

    original_digest = manager._tree_digest

    def delayed_digest(path):
        result = original_digest(path)
        advance("tree_digest", 2.0)
        return result

    original_fsync_tree = manager._fsync_tree

    def delayed_fsync_tree(path):
        result = original_fsync_tree(path)
        advance("fsync_tree", 3.0)
        return result

    original_publish = manager._publish_checkpoint

    def delayed_publish(checkpoint):
        result = original_publish(checkpoint)
        advance("checkpoint_publish", 5.0)
        return result

    original_write_json = manager._write_json

    def delayed_write_json(path, data):
        if path == manager._commit_path(1):
            advance("commit_marker", 7.0)
        return original_write_json(path, data)

    original_latest = runner._commit_checkpoint

    def delayed_latest(*args, **kwargs):
        result = original_latest(*args, **kwargs)
        advance("latest", 17.0)
        return result

    original_retention = runner._apply_retention

    def delayed_retention():
        result = original_retention()
        advance("retention", 19.0)
        return result

    original_runtime = manager.record_commit_runtime

    def observed_runtime(*args, **kwargs):
        events.append("runtime_sidecar")
        return original_runtime(*args, **kwargs)

    monkeypatch.setattr(runner, "_save_checkpoint", delayed_save)
    monkeypatch.setattr(manager, "stage_step", delayed_stage)
    monkeypatch.setattr(manager, "_tree_digest", delayed_digest)
    monkeypatch.setattr(manager, "_fsync_tree", delayed_fsync_tree)
    monkeypatch.setattr(manager, "_publish_checkpoint", delayed_publish)
    monkeypatch.setattr(manager, "_write_json", delayed_write_json)
    monkeypatch.setattr(runner, "_commit_checkpoint", delayed_latest)
    monkeypatch.setattr(runner, "_apply_retention", delayed_retention)
    monkeypatch.setattr(manager, "record_commit_runtime", observed_runtime)

    metrics = runner.run()[0]

    assert events == [
        "checkpoint_write",
        "artifact_stage",
        "tree_digest",
        "fsync_tree",
        "checkpoint_publish",
        "commit_marker",
        "latest",
        "retention",
        "runtime_sidecar",
    ]
    assert metrics["checkpoint_write_time_s"] == pytest.approx(13.0)
    assert metrics["artifact_stage_time_s"] == pytest.approx(11.0)
    assert metrics["artifact_commit_time_s"] == pytest.approx(17.0)
    assert metrics["post_commit_bookkeeping_time_s"] == pytest.approx(36.0)
    assert metrics["checkpoint_time_s"] == pytest.approx(77.0)
    assert metrics["step_time_s"] == pytest.approx(77.0)
    assert metrics["samples_per_second"] == pytest.approx(2.0 / 77.0)
    assert metrics["artifact_cycle_time_s"] == pytest.approx(77.0)
    assert metrics["artifact_cycle_steps"] == 1
    assert metrics["artifact_cycle_samples_per_second"] == pytest.approx(2.0 / 77.0)

    marker = _json(runner.output_dir / "commits" / "commit_000001.json")
    runtime = _json(runner.output_dir / "commits" / "runtime_000001.json")
    assert "artifact_commit_time_s" not in marker["steps"][0]["metric_row"]
    assert runtime["measurement_boundary"] == (
        "after_authoritative_commit_latest_and_retention_before_"
        "runtime_sidecar_persist_and_projection_refresh"
    )
    runtime_metrics = runtime["steps"][0]["metrics"]
    assert runtime_metrics["artifact_commit_time_s"] == pytest.approx(17.0)
    assert runtime_metrics["post_commit_bookkeeping_time_s"] == pytest.approx(36.0)

    with ArtifactManager(
        runner.output_dir,
        runner.config.run_name,
        resume=True,
    ) as recovered:
        recovered.rebuild_projections()
    persisted = json.loads(
        (runner.output_dir / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert persisted["artifact_commit_time_s"] == pytest.approx(17.0)
    assert persisted["step_time_s"] == pytest.approx(77.0)


def test_runner_holds_single_writer_lock_during_training(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "run", steps=1, save_every=1))
    original_score = runner.feedback_provider.score

    def checked_score(batch):
        with pytest.raises(RuntimeError, match="active writer"):
            ArtifactManager(runner.output_dir, runner.config.run_name, resume=True)
        return original_score(batch)

    runner.feedback_provider.score = checked_score
    runner.run()

    with ArtifactManager(runner.output_dir, runner.config.run_name, resume=True):
        pass


def test_failed_checkpoint_cycle_does_not_publish_partial_artifacts(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "run", steps=2, save_every=2))
    original_score = runner.feedback_provider.score
    calls = 0

    def fail_second_score(batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop before commit")
        return original_score(batch)

    runner.feedback_provider.score = fail_second_score
    with pytest.raises(RuntimeError, match="stop before commit"):
        runner.run()

    assert not (runner.output_dir / "sample_manifest.json").exists()
    assert not (runner.output_dir / "commits" / "commit_000002.json").exists()
    assert not (runner.output_dir / "checkpoint_000002").exists()
    assert not list((runner.output_dir / ".staging").glob("txn_*"))


def test_commit_log_wins_over_corrupted_latest_cache_on_resume(tmp_path):
    run_dir = tmp_path / "run"
    ExperimentRunner(_config(run_dir, steps=1, save_every=1)).run()
    (run_dir / "latest.json").write_text("{broken", encoding="utf-8")
    resumed_config = _config(run_dir, steps=2, save_every=1)
    resumed_config.paths.resume_from = str(run_dir / "latest.json")

    resumed = ExperimentRunner(resumed_config)

    assert resumed.start_step == 1
    assert [row["step"] for row in resumed.run()] == [1]


def _leave_checkpoint_published_before_marker(tmp_path, monkeypatch):
    run_dir = tmp_path / "publish-window"
    runner = ExperimentRunner(_config(run_dir, steps=1, save_every=1))
    manager = runner.artifacts
    original_write_json = manager._write_json

    def crash_before_marker(path, data):
        if path == manager._commit_path(1):
            path.parent.mkdir(exist_ok=True)
            raise OSError("simulated crash before commit marker")
        return original_write_json(path, data)

    monkeypatch.setattr(manager, "_write_json", crash_before_marker)
    with pytest.raises(OSError, match="crash before commit marker"):
        runner.run()

    assert (run_dir / "checkpoint_000001").is_dir()
    assert (run_dir / "commits").is_dir()
    assert not list((run_dir / "commits").glob("commit_*.json"))
    ready = list((run_dir / ".staging").glob("txn_*/pending.json"))
    assert len(ready) == 1
    assert _json(ready[0])["state"] == "ready"
    return run_dir


def test_runner_recovers_publish_window_before_loading_adapter(
    tmp_path,
    monkeypatch,
):
    run_dir = _leave_checkpoint_published_before_marker(tmp_path, monkeypatch)
    original_load = MockWanAdapter.load_checkpoint
    load_calls = 0

    def marker_checked_load(adapter, checkpoint_dir):
        nonlocal load_calls
        load_calls += 1
        assert (run_dir / "commits" / "commit_000001.json").is_file()
        assert not list((run_dir / ".staging").glob("txn_*/pending.json"))
        return original_load(adapter, checkpoint_dir)

    monkeypatch.setattr(MockWanAdapter, "load_checkpoint", marker_checked_load)
    resumed_config = _config(run_dir, steps=2, save_every=1)
    resumed_config.paths.resume_from = str(run_dir)
    resumed = ExperimentRunner(resumed_config)

    assert resumed.start_step == 1
    assert load_calls == 1
    assert [event["action"] for event in resumed.artifact_recovery_audit] == [
        "recovered"
    ]


def test_runner_never_loads_tampered_publish_window_checkpoint(
    tmp_path,
    monkeypatch,
):
    run_dir = _leave_checkpoint_published_before_marker(tmp_path, monkeypatch)
    (run_dir / "checkpoint_000001" / "mock_adapter.pt").write_bytes(
        b"tampered before recovery"
    )
    calls = {"training_state": 0, "adapter": 0}

    def reject_training_state(*_args, **_kwargs):
        calls["training_state"] += 1
        raise AssertionError("training state must not be read before recovery validation")

    def reject_adapter(*_args, **_kwargs):
        calls["adapter"] += 1
        raise AssertionError("adapter must not load before recovery validation")

    monkeypatch.setattr(
        runner_module,
        "read_and_validate_training_state",
        reject_training_state,
    )
    monkeypatch.setattr(MockWanAdapter, "load_checkpoint", reject_adapter)
    resumed_config = _config(run_dir, steps=2, save_every=1)
    resumed_config.paths.resume_from = str(run_dir)

    with pytest.raises(ResumeError, match="authoritative commit marker"):
        ExperimentRunner(resumed_config)

    assert calls == {"training_state": 0, "adapter": 0}
    assert not (run_dir / "commits" / "commit_000001.json").exists()
    assert list((run_dir / ".staging" / "quarantine").glob("txn_*"))


def test_missing_newest_checkpoint_requires_branch_for_older_fallback(tmp_path):
    run_dir = tmp_path / "source"
    ExperimentRunner(_config(run_dir, steps=2, save_every=1)).run()
    shutil.rmtree(run_dir / "checkpoint_000002")

    in_place = _config(run_dir, steps=2, save_every=1)
    in_place.paths.resume_from = str(run_dir)
    with pytest.raises(ResumeError, match="new output_dir"):
        ExperimentRunner(in_place)

    branch_dir = tmp_path / "branch"
    branch = _config(branch_dir, steps=2, save_every=1)
    branch.paths.resume_from = str(run_dir)
    resumed = ExperimentRunner(branch)

    assert resumed.start_step == 1
    assert [row["step"] for row in resumed.run()] == [1]
    assert (branch_dir / "commits" / "commit_000002.json").is_file()


@pytest.mark.parametrize(
    ("operation", "install_failure"),
    [
        (
            "latest_projection",
            lambda runner, fail: setattr(runner, "_commit_checkpoint", fail),
        ),
        (
            "retention",
            lambda runner, fail: setattr(runner, "_apply_retention", fail),
        ),
        (
            "runtime_sidecar",
            lambda runner, fail: setattr(
                runner.artifacts,
                "record_commit_runtime",
                fail,
            ),
        ),
    ],
)
def test_post_commit_bookkeeping_failure_cannot_invalidate_durable_step(
    tmp_path,
    operation,
    install_failure,
):
    run_dir = tmp_path / operation
    runner = ExperimentRunner(_config(run_dir, steps=1, save_every=1))

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {operation} failure")

    install_failure(runner, fail)
    metrics = runner.run()

    assert [row["step"] for row in metrics] == [0]
    assert (run_dir / "commits" / "commit_000001.json").is_file()
    assert runner.post_commit_bookkeeping_errors == [
        {
            "operation": operation,
            "error": f"OSError: injected {operation} failure",
        }
    ]

    (run_dir / "sample_manifest.json").write_text("{broken", encoding="utf-8")
    if (run_dir / "latest.json").exists():
        (run_dir / "latest.json").write_text("{broken", encoding="utf-8")
    resumed_config = _config(run_dir, steps=2, save_every=1)
    resumed_config.paths.resume_from = str(run_dir)
    resumed = ExperimentRunner(resumed_config)

    assert resumed.start_step == 1
    assert [row["step"] for row in resumed.run()] == [1]
    assert _json(run_dir / "latest.json")["step"] == 2
    assert [
        row["step"] for row in _json(run_dir / "sample_manifest.json")["records"]
    ] == [0, 0, 1, 1]


def test_post_commit_keyboard_interrupt_propagates_after_durable_marker(tmp_path):
    run_dir = tmp_path / "interrupt"
    runner = ExperimentRunner(_config(run_dir, steps=1, save_every=1))

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("injected latest interrupt")

    runner._commit_checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt, match="injected latest interrupt"):
        runner.run()

    assert (run_dir / "commits" / "commit_000001.json").is_file()


def test_runner_publishes_marker_aware_status_and_safe_audit(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "run", steps=1, save_every=1))
    runner.run()

    status = inspect_run_status(runner.output_dir / "run_status.json")
    audit = audit_run_artifacts(runner.output_dir)

    assert status["state"] == "completed"
    assert status["authoritative_completed_steps"] == 1
    assert status["marker_valid"] is True
    assert status["ready_for_aggregation"] is True
    assert audit["valid"] is True
    assert audit["steps"] == [0]


def test_audit_uses_configured_adapter_key_not_implementation_label(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "run", steps=1, save_every=1))
    runner.adapter.name = "mock_wan_implementation_label"
    runner.run()

    manifest = _json(runner.output_dir / "sample_manifest.json")
    model_metadata = manifest["records"][0]["model_metadata"]
    assert model_metadata["adapter"] == "mock_wan_implementation_label"
    assert model_metadata["adapter_key"] == "mock_wan"
    assert audit_run_artifacts(runner.output_dir)["valid"] is True


def test_status_is_rank_zero_only_and_redacts_failure_details(tmp_path):
    non_writer_path = tmp_path / "non-writer" / "run_status.json"
    assert write_run_status(
        non_writer_path,
        {"state": "running"},
        rank=1,
    ) is False
    assert not non_writer_path.exists()

    run_dir = tmp_path / "failed"
    runner = ExperimentRunner(_config(run_dir, steps=1, save_every=1))

    def fail_with_secret(_batch):
        raise RuntimeError("secret-token https://user:pass@example.test/private")

    runner.feedback_provider.score = fail_with_secret
    with pytest.raises(RuntimeError, match="secret-token"):
        runner.run()

    persisted = (run_dir / "run_status.json").read_text(encoding="utf-8")
    status = inspect_run_status(run_dir / "run_status.json")
    assert "secret-token" not in persisted
    assert "user:pass" not in persisted
    assert status["state"] == "failed"
    assert status["error"]["type"] == "RuntimeError"
    assert status["ready_for_aggregation"] is False


def test_audit_and_completed_status_reject_post_commit_tamper(tmp_path):
    run_dir = tmp_path / "tampered"
    runner = ExperimentRunner(_config(run_dir, steps=1, save_every=1))
    runner.run()
    (run_dir / "checkpoint_000001" / "mock_adapter.pt").write_bytes(
        b"tampered after commit"
    )

    with pytest.raises(RuntimeError, match="tree SHA256 mismatch"):
        audit_run_artifacts(run_dir)
    with pytest.raises(RuntimeError, match="tree SHA256 mismatch"):
        inspect_run_status(run_dir / "run_status.json")
