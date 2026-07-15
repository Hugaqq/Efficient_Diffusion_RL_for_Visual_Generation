"""Contracts for evidence-gated C13/C14 decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_rl.artifacts.checkpoint import config_fingerprint
from visual_rl.configs.schema import config_from_dict, config_to_dict, load_config
import visual_rl.runner as runner_module
from visual_rl.runner import ExperimentRunner, ResumeError
from visual_rl.scaling import (
    build_scaling_trigger_decision,
    validate_conditional_scaling,
)


def test_default_decision_is_deterministic_disabled_and_auditable() -> None:
    decision = build_scaling_trigger_decision(
        {"split_roles": False, "fsdp2": False}
    )

    assert decision == build_scaling_trigger_decision(
        {"split_roles": False, "fsdp2": False}
    )
    assert decision["schema_version"] == "1"
    assert decision["policy"] == "evidence_gated"
    for stage in ("C13", "C14"):
        item = decision["stages"][stage]
        assert item["decision"] == "not_triggered"
        assert item["triggered"] is False
        assert item["enabled"] is False
        assert item["runtime_validation"] == "not_run"
        assert item["observed_evidence"] == "not_provided"
        assert item["required_evidence"]
    json.dumps(decision, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "stage"),
    [("split_roles", "C13"), ("fsdp2", "C14")],
)
def test_unavailable_scaling_requests_fail_during_config_validation(
    field: str,
    stage: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        config_from_dict(
            {
                "run_name": "unsupported-scaling",
                "runner": {"conditional_scaling": {field: True}},
            }
        )

    with pytest.raises(ValueError, match=stage):
        validate_conditional_scaling(
            {
                "split_roles": field == "split_roles",
                "fsdp2": field == "fsdp2",
            }
        )


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_conditional_scaling_types_fail_closed(value) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        validate_conditional_scaling({"split_roles": value, "fsdp2": False})


def test_unknown_conditional_scaling_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown conditional scaling"):
        build_scaling_trigger_decision(
            {"split_roles": False, "fsdp2": False, "elastic": True}
        )


def test_runner_persists_default_trigger_decision(tmp_path: Path) -> None:
    preset = (
        Path(__file__).resolve().parents[1]
        / "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"
    )
    config = load_config(preset)
    config.paths.output_dir = str(tmp_path / "run")
    config.runner.show_progress = False

    runner = ExperimentRunner(config)
    runner.run(max_steps=0)

    decision_path = Path(runner.output_dir) / "trigger_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision == runner.scaling_trigger_decision
    assert set(decision["stages"]) == {"C13", "C14"}
    assert all(
        stage["decision"] == "not_triggered"
        and stage["enabled"] is False
        and stage["runtime_validation"] == "not_run"
        for stage in decision["stages"].values()
    )


def test_resume_requires_the_same_persisted_trigger_decision(tmp_path: Path) -> None:
    preset = (
        Path(__file__).resolve().parents[1]
        / "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"
    )
    config = load_config(preset)
    config.paths.output_dir = str(tmp_path / "run")
    config.runner.show_progress = False
    runner = ExperimentRunner(config)
    runner.run(max_steps=1)

    decision_path = runner.output_dir / "trigger_decision.json"
    tampered = json.loads(decision_path.read_text(encoding="utf-8"))
    tampered["stages"]["C14"]["runtime_validation"] = "passed"
    decision_path.write_text(json.dumps(tampered), encoding="utf-8")

    resumed = load_config(preset)
    resumed.paths.output_dir = str(runner.output_dir)
    resumed.paths.resume_from = str(runner.output_dir / "latest.json")
    resumed.runner.show_progress = False
    with pytest.raises(ResumeError, match="scaling trigger decision"):
        ExperimentRunner(resumed)


def test_branch_resume_validates_source_decision_and_copies_it(tmp_path: Path) -> None:
    preset = (
        Path(__file__).resolve().parents[1]
        / "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"
    )
    source_config = load_config(preset)
    source_config.paths.output_dir = str(tmp_path / "source")
    source_config.runner.show_progress = False
    source = ExperimentRunner(source_config)
    source.run(max_steps=1)

    branch_dir = tmp_path / "branch"
    branch_config = load_config(preset)
    branch_config.paths.output_dir = str(branch_dir)
    branch_config.paths.resume_from = str(source.output_dir / "latest.json")
    branch_config.train.max_steps = 2
    branch_config.runner.show_progress = False
    branch = ExperimentRunner(branch_config)

    assert branch.start_step == 1
    assert json.loads(
        (branch_dir / "trigger_decision.json").read_text(encoding="utf-8")
    ) == source.scaling_trigger_decision
    assert [row["step"] for row in branch.run()] == [1]


def test_corrupt_source_decision_fails_before_runtime_model_or_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preset = (
        Path(__file__).resolve().parents[1]
        / "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"
    )
    source_config = load_config(preset)
    source_config.paths.output_dir = str(tmp_path / "source")
    source_config.runner.show_progress = False
    source = ExperimentRunner(source_config)
    source.run(max_steps=1)
    (source.output_dir / "trigger_decision.json").write_text(
        "{",
        encoding="utf-8",
    )

    branch_dir = tmp_path / "must-not-exist"
    branch_config = load_config(preset)
    branch_config.paths.output_dir = str(branch_dir)
    branch_config.paths.resume_from = str(source.output_dir / "latest.json")
    branch_config.runner.show_progress = False
    calls = {"runtime": 0, "model": 0}

    def reject_runtime(*_args, **_kwargs):
        calls["runtime"] += 1
        raise AssertionError("runtime configuration must not run")

    def reject_model(*_args, **_kwargs):
        calls["model"] += 1
        raise AssertionError("model lookup must not run")

    monkeypatch.setattr(runner_module, "configure_runtime", reject_runtime)
    monkeypatch.setattr(runner_module.MODEL_ADAPTERS, "get", reject_model)

    with pytest.raises(ResumeError, match="scaling trigger decision"):
        ExperimentRunner(branch_config)

    assert calls == {"runtime": 0, "model": 0}
    assert not branch_dir.exists()


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "malformed", "nonfinite", "symlink"],
)
def test_resume_scaling_decision_format_fails_closed(
    tmp_path: Path,
    corruption: str,
) -> None:
    preset = (
        Path(__file__).resolve().parents[1]
        / "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"
    )
    config = load_config(preset)
    config.paths.output_dir = str(tmp_path / corruption)
    config.runner.show_progress = False
    runner = ExperimentRunner(config)
    runner.run(max_steps=1)

    decision_path = runner.output_dir / "trigger_decision.json"
    if corruption == "missing":
        decision_path.unlink()
    elif corruption == "duplicate":
        source = decision_path.read_text(encoding="utf-8")
        decision_path.write_text(
            source.replace(
                '"policy": "evidence_gated"',
                '"policy": "ignored", "policy": "evidence_gated"',
                1,
            ),
            encoding="utf-8",
        )
    elif corruption == "malformed":
        decision_path.write_text("{", encoding="utf-8")
    elif corruption == "nonfinite":
        decision_path.write_text('{"probe": NaN}', encoding="utf-8")
    else:
        original = runner.output_dir / "decision.original.json"
        decision_path.replace(original)
        decision_path.symlink_to(original)

    resumed = load_config(preset)
    resumed.paths.output_dir = str(runner.output_dir)
    resumed.paths.resume_from = str(runner.output_dir / "latest.json")
    resumed.runner.show_progress = False
    with pytest.raises(ResumeError, match="scaling trigger decision"):
        ExperimentRunner(resumed)


def test_trigger_decision_is_not_training_semantics(tmp_path: Path) -> None:
    preset = (
        Path(__file__).resolve().parents[1]
        / "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"
    )
    config = load_config(preset)
    config.paths.output_dir = str(tmp_path / "run")
    values = config_to_dict(config)
    baseline = config_fingerprint(values)

    values["trigger_decision"] = build_scaling_trigger_decision(
        config.runner.conditional_scaling
    )
    assert config_fingerprint(values) == baseline
