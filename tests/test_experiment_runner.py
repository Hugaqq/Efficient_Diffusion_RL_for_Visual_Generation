def test_direct_python_api_runs_without_cli_registration(tmp_path):
    from visual_rl import ExperimentRunner, load_config

    config = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    config.paths.output_dir = str(tmp_path / "run")
    config.runner.show_progress = False

    metrics = ExperimentRunner(config).run(max_steps=1)

    run_dir = tmp_path / "run"
    assert [row["step"] for row in metrics] == [0]
    assert (run_dir / "sample_manifest.json").exists()
    assert (run_dir / "reward_table.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert not (run_dir / "metric_table.jsonl").exists()
    assert (run_dir / "visual_report.md").exists()


def test_minimal_entrypoint_parses_config(monkeypatch, tmp_path):
    import visual_rl.entrypoint as entrypoint

    calls = []

    class FakeRunner:
        def __init__(self, config):
            calls.append(config.run_name)

        def run(self):
            calls.append("run")

    monkeypatch.setattr(entrypoint, "ExperimentRunner", FakeRunner)

    result = entrypoint.main(
        ["--config", "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"]
    )

    assert result == 0
    assert calls == ["world_r1_wan_v02_mock", "run"]


def test_clean_process_python_api_registers_builtins(tmp_path):
    import subprocess
    import sys

    script = """
import sys
from visual_rl import ExperimentRunner, load_config
config = load_config('visual_rl/configs/presets/flash_tiny_single_step.yaml')
config.paths.output_dir = sys.argv[1]
config.runner.show_progress = False
runner = ExperimentRunner(config)
runner.run(max_steps=1)
print(type(runner).__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "subprocess_run")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ExperimentRunner"


def test_runner_rejects_multi_process_mode(monkeypatch, tmp_path):
    import pytest

    from visual_rl.configs.schema import VisualRLConfig
    from visual_rl.runner import ExperimentRunner

    monkeypatch.setenv("WORLD_SIZE", "2")
    config = VisualRLConfig(run_name="multi-process")
    config.paths.output_dir = str(tmp_path / "must-not-exist")

    with pytest.raises(RuntimeError, match="one process only"):
        ExperimentRunner(config)

    assert not (tmp_path / "must-not-exist").exists()
