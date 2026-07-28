"""CPU-only contracts for VisualRL's sole high-level Python API."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import pytest

import visual_rl as vr
from visual_rl.api import audit_run, inspect_run, load
from visual_rl.api_types import AuditReport, RunResult, RunStatus, ValidationReport
from visual_rl.core.types import FrozenMapping, ValidatedRuntimeEnv, ValidationCheck
from visual_rl.errors import RunError, ValidationError


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"
PUBLIC = (
    "__version__",
    "load",
    "inspect_run",
    "audit_run",
    "ValidationReport",
    "RunResult",
    "RunStatus",
    "AuditReport",
    "ConfigError",
    "ComponentError",
    "ValidationError",
    "RunError",
    "ResumeError",
    "ArtifactError",
)


def _runtime_env() -> ValidatedRuntimeEnv:
    return ValidatedRuntimeEnv(
        mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        group_rank=None,
        group_world_size=None,
        master_addr=None,
        master_port=None,
        visible_gpu_count=0,
        raw_launch_env=FrozenMapping({}),
    )


def _materialize_result_paths(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint_000001"
    checkpoint.mkdir(exist_ok=True)
    commits = root / "commits"
    commits.mkdir(exist_ok=True)
    paths = {
        "authoritative_checkpoint": checkpoint,
        "resolved_config_path": root / "config.resolved.json",
        "manifest_path": root / "sample_manifest.jsonl",
        "metrics_path": root / "metrics.jsonl",
        "marker_path": commits / "commit_000001.json",
    }
    for name, path in paths.items():
        if name != "authoritative_checkpoint":
            path.touch()
    return paths


def test_top_level_public_allowlist_is_exact():
    assert tuple(vr.__all__) == PUBLIC
    assert vr.load is load
    assert vr.inspect_run is inspect_run
    assert vr.audit_run is audit_run
    for retired in (
        "Experiment",
        "ExperimentRunner",
        "VisualRLConfig",
        "Train",
        "load_config",
        "validate_config",
        "register_builtin_plugins",
    ):
        assert not hasattr(vr, retired)


def test_handle_has_zero_override_signatures_and_cannot_be_publicly_constructed():
    experiment = load(TINY)

    assert str(inspect.signature(experiment.resolve)) == "() -> 'VisualRLConfig'"
    assert str(inspect.signature(experiment.validate)) == "() -> 'ValidationReport'"
    assert str(inspect.signature(experiment.run)) == "() -> 'RunResult'"
    assert type(experiment).__name__ == "_Experiment"
    assert not hasattr(vr, "Experiment")


def test_load_freezes_yaml_bytes_and_resolve_reuses_same_object(tmp_path):
    path = tmp_path / "config.yaml"
    original = TINY.read_text(encoding="utf-8")
    path.write_text(original, encoding="utf-8")
    experiment = load(path)
    path.write_text(original.replace("seed: 42", "seed: 99"), encoding="utf-8")

    first = experiment.resolve()
    second = experiment.resolve()

    assert first is second
    assert first.run.seed == 42


def test_import_load_and_resolve_do_not_import_training_frameworks():
    script = f"""
import sys
import visual_rl as vr
assert 'torch' not in sys.modules
experiment = vr.load({str(TINY)!r})
assert 'torch' not in sys.modules
experiment.resolve()
assert 'torch' not in sys.modules
assert 'diffusers' not in sys.modules
assert 'transformers' not in sys.modules
assert 'peft' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_validate_caches_report_and_runtime_snapshot(monkeypatch):
    calls = []
    check = ValidationCheck("warning", "test.warning", "runtime", "warning")
    report = ValidationReport((check,), 0, 1)
    env = _runtime_env()

    def fake_preflight(config, **kwargs):
        calls.append((config, kwargs))
        return report, env

    import visual_rl.preflight as preflight

    monkeypatch.setattr(preflight, "run_preflight", fake_preflight)
    experiment = load(TINY)

    assert experiment.validate() is report
    assert experiment.validate() is report
    assert len(calls) == 1
    assert calls[0][1]["phase"] == "validate"


def test_run_claim_is_consumed_by_validation_failure(monkeypatch):
    error = ValidationCheck("error", "test.error", "runtime", "broken", True)

    def fake_preflight(config, **kwargs):
        del config, kwargs
        return ValidationReport((error,), None, None), None

    import visual_rl.preflight as preflight

    monkeypatch.setattr(preflight, "run_preflight", fake_preflight)
    experiment = load(TINY)

    with pytest.raises(ValidationError, match="run-phase validation"):
        experiment.run()
    with pytest.raises(RunError, match="already attempted"):
        experiment.run()


def test_run_passes_same_config_and_fresh_env_to_only_runner(
    monkeypatch, tmp_path
):
    env = _runtime_env()
    reports = []

    def fake_preflight(config, **kwargs):
        reports.append((config, kwargs))
        return ValidationReport((), 0, 1), env

    output_dir = tmp_path.resolve()
    paths = _materialize_result_paths(output_dir)
    result = RunResult(
        run_id="run-1",
        output_dir=output_dir,
        committed_steps=1,
        **paths,
        last_metrics={
            "step": 0,
            "sample_count": 4,
            "active_transition_count": 8,
            "reward_mean": 1.0,
        },
    )
    runner_calls = []

    class FakeRunner:
        def __init__(self, config, validated_env):
            runner_calls.append((config, validated_env))

        def run(self):
            return result

    import visual_rl.preflight as preflight
    import visual_rl.runner as runner

    monkeypatch.setattr(preflight, "run_preflight", fake_preflight)
    monkeypatch.setattr(runner, "ExperimentRunner", FakeRunner)
    experiment = load(TINY)

    assert experiment.run() is result
    assert len(reports) == 2
    assert reports[0][1]["phase"] == "validate"
    assert reports[1][1]["phase"] == "run"
    assert isinstance(reports[1][1]["cached_report"], ValidationReport)
    assert reports[1][1]["cached_report"].ok
    assert reports[1][1]["cached_env"] is env
    assert runner_calls == [(experiment.resolve(), env)]


def test_report_properties_are_derived_and_result_metrics_are_frozen(tmp_path):
    error = ValidationCheck("error", "e", "x", "bad")
    warning = ValidationCheck("warning", "w", "x", "careful")
    report = ValidationReport((warning, error), None, None)
    assert not report.ok
    assert report.errors == (error,)
    assert report.warnings == (warning,)

    metrics = {
        "step": 0,
        "sample_count": 2,
        "active_transition_count": 2,
        "reward_mean": 1.0,
    }
    root = tmp_path.resolve()
    paths = _materialize_result_paths(root)
    result = RunResult(
        run_id="r",
        output_dir=root,
        committed_steps=1,
        **paths,
        last_metrics=metrics,
    )
    metrics["reward_mean"] = 9.0
    assert result.last_metrics["reward_mean"] == 1.0
    with pytest.raises(TypeError):
        result.last_metrics["reward_mean"] = 2.0


def test_inspect_and_audit_build_public_types_from_internal_projections(
    monkeypatch, tmp_path
):
    root = tmp_path.resolve()

    import visual_rl.artifacts.audit as audit_module
    import visual_rl.artifacts.status as status_module

    monkeypatch.setattr(
        status_module,
        "inspect_run_status",
        lambda path: {
            "run_id": "r",
            "committed_steps": 2,
            "authoritative_checkpoint": "checkpoint_000002",
            "resumable": True,
            "pending_transaction_count": 0,
            "checks": (),
        },
    )
    monkeypatch.setattr(
        audit_module,
        "audit_run_artifacts",
        lambda path: {
            "run_id": "r",
            "committed_steps": 2,
            "checked_commit_count": 2,
            "checked_artifact_paths": [
                "commits/commit_000001.json",
                "checkpoint_000002/checkpoint.json",
            ],
            "checks": (),
        },
    )

    status = inspect_run(root)
    audit = audit_run(root)

    assert isinstance(status, RunStatus) and status.ok and status.resumable
    assert status.authoritative_checkpoint == root / "checkpoint_000002"
    assert isinstance(audit, AuditReport) and audit.ok
    assert audit.checked_commit_count == 2
    assert audit.checked_artifact_paths[0] == root / "commits/commit_000001.json"
