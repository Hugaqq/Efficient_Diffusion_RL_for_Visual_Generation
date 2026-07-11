from __future__ import annotations

import json
import tarfile


def test_remote_smoke_archive_includes_package_and_excludes_heavy_dirs(tmp_path):
    from scripts.remote_smoke import create_source_archive

    (tmp_path / "visual_rl").mkdir()
    (tmp_path / "visual_rl" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts" / "legacy_cli.py").write_text("print('cli')\n", encoding="utf-8")
    (tmp_path / "scripts" / "remote_smoke.py").write_text("", encoding="utf-8")
    (tmp_path / "visual_rl" / "__pycache__").mkdir()
    (tmp_path / "visual_rl" / "__pycache__" / "cli.cpython-311.pyc").write_bytes(b"cache")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "output.txt").write_text("skip\n", encoding="utf-8")
    (tmp_path / "reference_code").mkdir()
    (tmp_path / "reference_code" / "legacy.py").write_text("skip\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'visual-rl'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# VisualRL\n", encoding="utf-8")

    archive_path = tmp_path / "source.tar.gz"
    members = create_source_archive(tmp_path, archive_path)

    assert "scripts/legacy_cli.py" in members
    assert "pyproject.toml" in members
    assert "README.md" in members
    assert all("__pycache__" not in member for member in members)
    assert all(not member.endswith(".pyc") for member in members)
    assert all(not member.startswith("runs/") for member in members)
    assert all(not member.startswith("reference_code/") for member in members)

    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    assert "scripts/legacy_cli.py" in names
    assert "visual_rl/__pycache__/cli.cpython-311.pyc" not in names
    assert "reference_code/legacy.py" not in names
    assert "runs/output.txt" not in names


def test_remote_smoke_dry_run_payload_uses_stage_dir_not_shared_framecode(tmp_path):
    from scripts.remote_smoke import RemoteSd3CliSmokeConfig, build_dry_run_payload

    (tmp_path / "visual_rl").mkdir()
    (tmp_path / "visual_rl" / "cli.py").write_text("print('cli')\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'visual-rl'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# VisualRL\n", encoding="utf-8")

    config = RemoteSd3CliSmokeConfig(
        remote_root="/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke",
        stage_name="unit_stage",
        model_path="/models/sd35",
    )
    payload = build_dry_run_payload(config, source_root=tmp_path)

    assert payload["remote_stage_dir"] == (
        "/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/unit_stage"
    )
    assert payload["remote_archive_path"].startswith(payload["remote_stage_dir"])
    assert payload["remote_script_path"].startswith(payload["remote_stage_dir"])
    assert "/home/v-qiaoqifan/visual_rl_experiments/framecode" not in json.dumps(
        {
            "remote_stage_dir": payload["remote_stage_dir"],
            "remote_archive_path": payload["remote_archive_path"],
            "remote_script_path": payload["remote_script_path"],
            "scp_archive_command": payload["scp_archive_command"],
            "scp_script_command": payload["scp_script_command"],
        }
    )


def test_remote_script_has_idle_guard_and_sd3_tempflow_command():
    from scripts.remote_smoke import RemoteSd3CliSmokeConfig, build_remote_script

    script = build_remote_script(
        RemoteSd3CliSmokeConfig(
            gpu=7,
            model_path="/models/sd35",
            repo_root="/remote/ref/TempFlow-GRPO-main",
            stage_name="unit_stage",
            prompt="a blue cube",
        )
    )

    assert "set -euo pipefail" in script
    assert "nvidia-smi -i \"$GPU\" --query-gpu=memory.used,utilization.gpu" in script
    assert "nvidia-smi pmon -i \"$GPU\" -c 1" in script
    assert "exit 77" in script
    assert "CUDA_VISIBLE_DEVICES=\"$GPU\"" in script
    assert "image-preview --adapter sd3_tempflow" in script
    assert "sd3-numeric-smoke --model-path" in script
    assert "sd3-branching-numeric-smoke --model-path" in script
    assert "--branch-step-index auto" in script
    assert "sd3-bounded-trainer-smoke --adapter sd3_tempflow" in script
    assert "--resume-from" in script
    assert "resume_from_1step_1step" in script
    assert "GPU before resume" in script
    assert "resume_idle_guard" in script
    assert "gpu_pmon_before_resume.log" in script
    assert "--repo-root /remote/ref/TempFlow-GRPO-main" in script
    assert "--disable-rollout-cache" in script
    assert "previews/before/preview_000.png" in script
    assert "previews/after/preview_000.png" in script
    assert '"resume_loaded": true' in script
    assert "remote-sd3-cli-smoke" in script
    assert "scripts/legacy_cli.py" in script
    assert "visual_rl/model_adapters/sd3.py" in script


def test_remote_sd3_cli_smoke_dry_run_exits_zero_and_prints_json(capsys):
    from scripts import legacy_cli as cli

    exit_code = cli.main(
        [
            "remote-sd3-cli-smoke",
            "--stage-name",
            "cli_unit_stage",
            "--remote-root",
            "/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke",
            "--model-path",
            "/models/sd35",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["remote_stage_dir"].endswith("/cli_unit_stage")
    assert payload["config"]["server"] == "v-qiaoqifan@10.130.140.73"
    assert payload["config"]["conda_env"] == "visual-rl-sd35"
    assert payload["remote_preview_dir"].endswith("/cli_unit_stage/preview")
    assert payload["remote_bounded_dir"].endswith("/cli_unit_stage/bounded_1step")
    assert payload["remote_bounded_checkpoint_dir"].endswith("/cli_unit_stage/bounded_1step/checkpoint_000001")
    assert payload["remote_resume_dir"].endswith("/cli_unit_stage/resume_from_1step_1step")
    assert payload["run_bounded_trainer"] is True
    assert payload["run_resume_validation"] is True
    assert "scripts/legacy_cli.py" in payload["archive_members"]
    assert "image-preview --adapter sd3_tempflow" in payload["remote_script"]
    assert "sd3-numeric-smoke --model-path" in payload["remote_script"]
    assert "sd3-branching-numeric-smoke --model-path" in payload["remote_script"]
    assert "sd3-bounded-trainer-smoke --adapter sd3_tempflow" in payload["remote_script"]
    assert "--resume-from" in payload["remote_script"]


def test_remote_sd3_cli_smoke_dry_run_can_mark_long_bounded_run(capsys):
    from scripts import legacy_cli as cli

    exit_code = cli.main(
        [
            "remote-sd3-cli-smoke",
            "--stage-name",
            "cli_long_stage",
            "--model-path",
            "/models/sd35",
            "--bounded-steps",
            "20",
            "--skip-resume-validation",
            "--allow-long-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["bounded_steps"] == 20
    assert payload["config"]["allow_long_run"] is True
    assert "--steps 20" in payload["remote_script"]
    assert "--allow-long-run" in payload["remote_script"]
