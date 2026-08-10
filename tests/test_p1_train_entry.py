"""P1 contracts for the thin module training entry."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from visual_rl.core.types import ValidationCheck
from visual_rl.errors import ConfigError, RunError, ValidationError
from visual_rl.runtime.types import RunResult

ROOT = Path(__file__).resolve().parents[1]


def _result(root: Path) -> RunResult:
    root = root.resolve()
    root.mkdir(parents=True)
    checkpoint = root / "checkpoint_000001"
    checkpoint.mkdir()
    commits = root / "commits"
    commits.mkdir()
    resolved = root / "config.resolved.json"
    manifest = root / "sample_manifest.json"
    metrics = root / "metrics.jsonl"
    marker = commits / "commit_000001.json"
    for path in (resolved, manifest, metrics, marker):
        path.touch()
    return RunResult(
        run_id="run-p1",
        output_dir=root,
        committed_steps=1,
        authoritative_checkpoint=checkpoint,
        resolved_config_path=resolved,
        manifest_path=manifest,
        metrics_path=metrics,
        marker_path=marker,
        last_metrics={
            "step": 0,
            "sample_count": 2,
            "active_transition_count": 4,
            "reward_mean": 1.0,
        },
    )


def test_main_uses_only_the_v08_controller_and_prints_canonical_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from visual_rl import train

    result = _result(tmp_path / "run")
    calls: list[object] = []

    class Controller:
        def run(self, path):
            calls.append(path)
            calls.append("run")
            return result

    monkeypatch.setattr(train, "_create_controller", lambda: Controller())
    stdout = StringIO()
    stderr = StringIO()

    code = train.main(["config.yaml"], stdout=stdout, stderr=stderr)

    assert code == 0
    assert calls == ["config.yaml", "run"]
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue()) == {
        "authoritative_checkpoint": str(result.authoritative_checkpoint),
        "committed_steps": 1,
        "output_dir": str(result.output_dir),
        "run_id": "run-p1",
        "status": "ok",
    }
    source = (ROOT / "visual_rl" / "train.py").read_text(encoding="utf-8")
    assert "visual_rl.api" not in source
    assert "from visual_rl.api import load" not in source
    assert "visual_rl.runner" not in source
    assert "ExperimentRunner" not in source


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ConfigError("bad config", key="model.name"), 2),
        (RunError("runtime failed", step=0), 1),
        (
            ValidationError(
                "preflight failed",
                checks=(
                    ValidationCheck(
                        "error",
                        "runtime.invalid",
                        "runtime",
                        "invalid",
                    ),
                ),
            ),
            2,
        ),
    ],
)
def test_main_has_stable_structured_error_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_code: int,
):
    from visual_rl import train

    class Controller:
        def run(self, _path):
            raise error

    monkeypatch.setattr(train, "_create_controller", lambda: Controller())
    stdout = StringIO()
    stderr = StringIO()

    code = train.main(["config.yaml"], stdout=stdout, stderr=stderr)

    assert code == expected_code
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "error"
    assert payload["error"] == type(error).__name__


def test_main_usage_and_help_are_side_effect_free():
    from visual_rl import train

    for arguments, expected_code, stream_name in (
        ([], 2, "stderr"),
        (["one", "two"], 2, "stderr"),
        (["--help"], 0, "stdout"),
    ):
        stdout = StringIO()
        stderr = StringIO()
        code = train.main(arguments, stdout=stdout, stderr=stderr)
        assert code == expected_code
        selected = stdout if stream_name == "stdout" else stderr
        assert selected.getvalue() == "usage: python -m visual_rl.train CONFIG\n"


def test_python_m_entry_reports_source_error_without_importing_runner(tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    completed = subprocess.run(
        [sys.executable, "-m", "visual_rl.train", str(missing)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["status"] == "error"
    assert payload["error"] == "ConfigSourceError"
    assert payload["code"] == "config.source_read"


def test_schema_v1_entry_reports_offline_migration_before_runtime_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visual_rl import train
    from visual_rl.runtime.controller import ControllerStage

    controller = train._create_controller()
    monkeypatch.setattr(train, "_create_controller", lambda: controller)
    stdout = StringIO()
    stderr = StringIO()

    code = train.main(
        [str(ROOT / "configs" / "flow_grpo_sd3.yaml")],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 2
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload == {
        "code": "config.schema_v1_migration_required",
        "error": "ConfigMigrationError",
        "key": "schema_version",
        "message": (
            "schema_version 1 is retired and cannot be executed by the v0.8 "
            "production entry; start from a schema-v2 example in configs/v2/ "
            "or migrate this file offline. Runtime legacy parsing is "
            "intentionally unavailable."
        ),
        "migration_examples": "configs/v2/",
        "migration_mode": "offline_only",
        "path": str((ROOT / "configs" / "flow_grpo_sd3.yaml").resolve()),
        "required_schema_version": 2,
        "source_schema_version": 1,
        "status": "error",
    }
    assert controller.attempted_stages == (ControllerStage.COMPILE,)
    assert controller.completed_stages == ()
