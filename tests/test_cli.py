"""C3 CLI behavior and stable exit-code contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from visual_rl import cli
from visual_rl.artifacts.status import write_run_status
from visual_rl.model_adapters.mock import MockWanAdapter
import visual_rl.runner as runner_module
from visual_rl.runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, **overrides) -> Path:
    values = {
        "run_name": "cli-test",
        "dataset": {"prompts": ["offline prompt"]},
        "paths": {"output_dir": "runs"},
        "runner": {"show_progress": False},
    }
    values.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _subprocess(tmp_path: Path, *arguments: str):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "visual_rl.cli", *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def completed_cli_run(tmp_path: Path) -> Path:
    output_dir = tmp_path / "operational-run"
    config = _write_config(
        tmp_path,
        paths={"output_dir": str(output_dir)},
        train={"max_steps": 1, "save_every": 1},
    )
    completed = _subprocess(tmp_path, "run", str(config), "--json")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output_dir


def _leave_cli_checkpoint_published_before_marker(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        train={"max_steps": 1, "save_every": 1},
    )
    args = cli._build_parser().parse_args(["run", str(config)])
    resolved = cli._resolve_from_args(args, tmp_path)
    runner = ExperimentRunner(resolved.config)
    manager = runner.artifacts
    original_write_json = manager._write_json

    def crash_before_marker(path, data):
        if path == manager._commit_path(1):
            path.parent.mkdir(exist_ok=True)
            raise OSError("simulated CLI crash before commit marker")
        return original_write_json(path, data)

    monkeypatch.setattr(manager, "_write_json", crash_before_marker)
    with pytest.raises(OSError, match="CLI crash before commit marker"):
        runner.run()

    run_dir = Path(resolved.config.paths.output_dir)
    assert (run_dir / "checkpoint_000001").is_dir()
    assert not list((run_dir / "commits").glob("commit_*.json"))
    assert len(list((run_dir / ".staging").glob("txn_*/pending.json"))) == 1
    return config, run_dir


def test_validate_json_writes_exactly_one_envelope(tmp_path):
    config = _write_config(tmp_path)
    completed = _subprocess(tmp_path, "validate", str(config), "--json")

    assert completed.returncode == 0
    assert len(completed.stdout.splitlines()) == 1
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["data"]["preflight"]["trusted"] is False

    help_result = _subprocess(tmp_path, "validate", "--help", "--json")
    assert help_result.returncode == 0
    assert len(help_result.stdout.splitlines()) == 1
    assert json.loads(help_result.stdout)["schema_version"] == 1


def test_packaged_presets_are_listed_from_any_cwd(tmp_path):
    completed = _subprocess(tmp_path, "presets", "--json")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    names = payload["data"]["presets"]
    assert names == sorted(names)
    assert {
        "flash_tiny_single_step",
        "tempflow_tiny_branching",
        "world_r1_wan_v02_mock",
    }.issubset(names)

    text_result = _subprocess(tmp_path, "presets")
    assert text_result.returncode == 0, text_result.stderr
    assert text_result.stdout.splitlines() == names


def test_status_and_audit_cli_report_healthy_run_with_json_envelopes(
    tmp_path,
    completed_cli_run,
):
    status = _subprocess(tmp_path, "status", str(completed_cli_run), "--json")
    audit = _subprocess(tmp_path, "audit", str(completed_cli_run), "--json")

    assert status.returncode == cli.EXIT_SUCCESS, status.stderr
    status_payload = json.loads(status.stdout)
    assert status_payload["command"] == "status"
    assert status_payload["ok"] is True
    assert status_payload["exit_code"] == cli.EXIT_SUCCESS
    assert status_payload["data"]["status"]["observed_state"] == "completed"
    assert status_payload["data"]["status"]["ready_for_aggregation"] is True

    assert audit.returncode == cli.EXIT_SUCCESS, audit.stderr
    audit_payload = json.loads(audit.stdout)
    assert audit_payload["command"] == "audit"
    assert audit_payload["ok"] is True
    assert audit_payload["exit_code"] == cli.EXIT_SUCCESS
    assert audit_payload["data"]["audit"]["valid"] is True

    status_text = _subprocess(tmp_path, "status", str(completed_cli_run))
    audit_text = _subprocess(tmp_path, "audit", str(completed_cli_run))
    assert status_text.stdout == "run status: completed (committed steps: 1)\n"
    assert audit_text.stdout == "artifact audit: valid (markers: 1, steps: 1)\n"


def test_audit_cli_returns_nonzero_json_envelope_for_invalid_audit(
    tmp_path,
    completed_cli_run,
):
    marker_path = completed_cli_run / "commits" / "commit_000001.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["steps"][0]["reward_rows"][0]["sample_id"] = "wrong-sample"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    completed = _subprocess(tmp_path, "audit", str(completed_cli_run), "--json")

    assert completed.returncode == cli.EXIT_ARTIFACT
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "error"
    assert payload["exit_code"] == cli.EXIT_ARTIFACT
    assert payload["data"]["audit"]["valid"] is False
    assert payload["data"]["audit"]["errors"]


def test_operational_cli_rejects_tamper_without_exposing_exception(
    tmp_path,
    completed_cli_run,
):
    secret = "unredacted-secret-in-tamper"
    (completed_cli_run / "checkpoint_000001" / "mock_adapter.pt").write_bytes(
        secret.encode("utf-8")
    )

    for command in ("status", "audit"):
        completed = _subprocess(
            tmp_path,
            command,
            str(completed_cli_run),
            "--json",
        )
        assert completed.returncode == cli.EXIT_ARTIFACT
        payload = json.loads(completed.stdout)
        assert payload["ok"] is False
        assert payload["exit_code"] == cli.EXIT_ARTIFACT
        assert payload["error"]["type"] == "ArtifactCheckError"
        assert "trusted process logs" in payload["error"]["message"]
        assert secret not in completed.stdout
        assert secret not in completed.stderr


def test_status_cli_returns_nonzero_for_missing_failed_and_stale(tmp_path):
    failed_dir = tmp_path / "failed-run"
    failed_dir.mkdir()
    write_run_status(
        failed_dir / "run_status.json",
        {"state": "failed", "completed_steps": 0},
        exception=RuntimeError("private-token-must-not-leak"),
    )
    stale_dir = tmp_path / "stale-run"
    stale_dir.mkdir()
    write_run_status(
        stale_dir / "run_status.json",
        {"state": "running", "completed_steps": 0, "pid": 2**30},
    )

    failed = _subprocess(tmp_path, "status", str(failed_dir), "--json")
    stale = _subprocess(tmp_path, "status", str(stale_dir), "--json")
    missing = _subprocess(tmp_path, "status", str(tmp_path / "missing"), "--json")

    assert failed.returncode == cli.EXIT_ARTIFACT
    failed_payload = json.loads(failed.stdout)
    assert failed_payload["data"]["status"]["observed_state"] == "failed"
    assert failed_payload["data"]["status"]["error"]["message"] == (
        "Run failed; inspect trusted process logs for details."
    )
    assert "private-token" not in failed.stdout + failed.stderr

    assert stale.returncode == cli.EXIT_ARTIFACT
    assert json.loads(stale.stdout)["data"]["status"]["observed_state"] == (
        "stale_running"
    )

    assert missing.returncode == cli.EXIT_ARTIFACT
    missing_payload = json.loads(missing.stdout)
    assert missing_payload["error"]["type"] == "ArtifactCheckError"
    assert missing_payload["error"]["message"] == (
        "Run status is missing or invalid; inspect trusted process logs."
    )
    assert str(tmp_path) not in missing.stdout + missing.stderr


@pytest.mark.parametrize(
    "preset_name",
    ["flash_tiny_single_step", "world_r1_wan_bounded"],
)
def test_packaged_preset_reference_matches_file_and_validates_from_any_cwd(
    tmp_path,
    preset_name,
):
    preset_path = ROOT / f"visual_rl/configs/presets/{preset_name}.yaml"
    output_dir = tmp_path / "equivalent-run"
    common = (
        "--set",
        f"paths.output_dir={output_dir}",
        "--set",
        "runner.show_progress=false",
        "--json",
    )

    packaged = _subprocess(
        tmp_path,
        "inspect",
        f"preset:{preset_name}",
        *common,
    )
    file_backed = _subprocess(tmp_path, "inspect", str(preset_path), *common)

    assert packaged.returncode == 0, packaged.stderr
    assert file_backed.returncode == 0, file_backed.stderr
    packaged_data = json.loads(packaged.stdout)["data"]
    file_data = json.loads(file_backed.stdout)["data"]
    assert packaged_data["config"] == file_data["config"]
    assert packaged_data["preflight"] == file_data["preflight"]
    assert packaged_data["provenance"]["sample.name"]["kind"] == "preset"
    assert file_data["provenance"]["sample.name"]["kind"] == "user"

    validated = _subprocess(
        tmp_path,
        "validate",
        f"preset:{preset_name}",
        "--json",
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["data"]["preflight"]["trusted"] is False


def test_packaged_tiny_preset_runs_and_unknown_name_lists_choices(tmp_path):
    output_dir = tmp_path / "packaged-preset-run"
    completed = _subprocess(
        tmp_path,
        "run",
        "preset:flash_tiny_single_step",
        "--set",
        f"paths.output_dir={output_dir}",
        "--set",
        "runner.show_progress=false",
        "--json",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)["data"]
    assert payload["steps"] == 2
    assert Path(payload["output_dir"]) == output_dir
    assert (output_dir / "config.resolved.json").is_file()
    assert (output_dir / "commits" / "commit_000002.json").is_file()

    unknown = _subprocess(
        tmp_path,
        "validate",
        "preset:not-a-real-preset",
        "--json",
    )
    assert unknown.returncode == cli.EXIT_USAGE
    message = json.loads(unknown.stdout)["error"]["message"]
    assert "Unknown packaged preset 'not-a-real-preset'" in message
    assert "Available:" in message
    assert "flash_tiny_single_step" in message


def test_inspect_set_is_ordered_yaml_parsed_and_cwd_anchored(tmp_path):
    config = _write_config(tmp_path)
    completed = _subprocess(
        tmp_path,
        "inspect",
        str(config),
        "--set",
        "seed=7",
        "--set",
        "seed=8",
        "--set",
        "dataset.path=data/prompts.txt",
        "--json",
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)["data"]
    assert payload["config"]["seed"] == 8
    assert payload["config"]["dataset"]["path"] == str(
        tmp_path / "data/prompts.txt"
    )
    assert payload["provenance"]["dataset.path"]["name"] == "CLI --set[2]"

    text_result = _subprocess(tmp_path, "inspect", str(config))
    assert text_result.returncode == 0
    rendered = yaml.safe_load(text_result.stdout)
    assert set(rendered) == {"config", "preflight", "provenance"}
    assert rendered["preflight"]["trusted"] is False
    assert rendered["preflight"]["components"]


def test_resume_is_last_explicit_document_and_uses_invocation_cwd(tmp_path):
    config = _write_config(
        tmp_path,
        explicit={"paths": {"resume_from": "from-config"}},
    )
    args = cli._build_parser().parse_args(
        [
            "run",
            str(config),
            "--set",
            "paths.resume_from=from-set",
            "--resume",
            "from-cli",
        ]
    )

    resolved = cli._resolve_from_args(args, tmp_path)
    assert resolved.config.paths.resume_from == str(tmp_path / "from-cli")
    assert resolved.provenance["paths.resume_from"].name == "CLI --resume"


def test_stable_static_and_resume_exit_codes(tmp_path):
    config = _write_config(tmp_path)
    usage = _subprocess(tmp_path, "validate", str(config), "--set", "bad", "--json")
    resume = _subprocess(
        tmp_path,
        "run",
        str(config),
        "--resume",
        "missing-checkpoint",
        "--json",
    )

    assert usage.returncode == 2
    usage_payload = json.loads(usage.stdout)
    assert usage_payload["ok"] is False
    assert usage_payload["exit_code"] == 2
    assert resume.returncode == 4
    assert json.loads(resume.stdout)["exit_code"] == 4


def test_stable_trusted_and_execution_exit_codes(tmp_path):
    trusted_config = _write_config(
        tmp_path,
        rewards={
            "provider_params": {"unexpected": True},
            "weights": {"mock": 1.0},
            "clients": {"mock": {"name": "mock"}},
        },
    )
    trusted = _subprocess(
        tmp_path,
        "validate",
        str(trusted_config),
        "--trusted-components",
        "--json",
    )
    assert trusted.returncode == 3
    assert json.loads(trusted.stdout)["exit_code"] == 3

    execution_config = _write_config(
        tmp_path,
        dataset={"path": str(tmp_path / "missing-prompts.txt")},
        paths={"output_dir": "execution-runs"},
    )
    execution = _subprocess(tmp_path, "run", str(execution_config), "--json")
    assert execution.returncode == 5
    assert json.loads(execution.stdout)["exit_code"] == 5


def test_cpu_offline_run_and_legacy_forwarding(tmp_path):
    config = _write_config(tmp_path)
    run = _subprocess(tmp_path, "run", str(config), "--json")

    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["data"]["steps"] == 1
    assert Path(payload["data"]["output_dir"]).is_dir()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    legacy = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--config",
            str(config),
            "--set",
            "paths.output_dir=legacy-runs",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert legacy.returncode == 0
    assert "deprecated" in legacy.stderr


def test_cli_run_recovers_ready_transaction_before_final_resume_validation(
    tmp_path,
    monkeypatch,
):
    config, run_dir = _leave_cli_checkpoint_published_before_marker(
        tmp_path,
        monkeypatch,
    )

    resumed = _subprocess(
        tmp_path,
        "run",
        str(config),
        "--resume",
        str(run_dir),
        "--set",
        "train.max_steps=2",
        "--json",
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (run_dir / "commits" / "commit_000001.json").is_file()
    assert (run_dir / "commits" / "commit_000002.json").is_file()


def test_cli_tampered_ready_transaction_fails_before_checkpoint_load(
    tmp_path,
    monkeypatch,
    capsys,
):
    config, run_dir = _leave_cli_checkpoint_published_before_marker(
        tmp_path,
        monkeypatch,
    )
    (run_dir / "checkpoint_000001" / "mock_adapter.pt").write_bytes(
        b"tampered before CLI recovery"
    )
    calls = {"training_state": 0, "adapter": 0}

    def reject_training_state(*_args, **_kwargs):
        calls["training_state"] += 1
        raise AssertionError("CLI must not read an unverified training state")

    def reject_adapter(*_args, **_kwargs):
        calls["adapter"] += 1
        raise AssertionError("CLI must not load an unverified adapter")

    monkeypatch.setattr(
        runner_module,
        "read_and_validate_training_state",
        reject_training_state,
    )
    monkeypatch.setattr(MockWanAdapter, "load_checkpoint", reject_adapter)
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        [
            "run",
            str(config),
            "--resume",
            str(run_dir),
            "--set",
            "train.max_steps=2",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_RESUME, payload
    assert payload["exit_code"] == cli.EXIT_RESUME
    assert calls == {"training_state": 0, "adapter": 0}
    assert not (run_dir / "commits" / "commit_000001.json").exists()
    assert list((run_dir / ".staging" / "quarantine").glob("txn_*"))
