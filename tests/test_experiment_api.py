"""CPU/offline contracts for the small composable Experiment API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import visual_rl as vr
from visual_rl.artifacts.checkpoint import config_fingerprint
from visual_rl.configs.schema import config_to_dict, load_config


ROOT = Path(__file__).resolve().parents[1]


def _mock_experiment(output_dir: Path, **kwargs) -> vr.Experiment:
    return vr.Experiment(
        model=vr.models.MockWan(),
        rollout=vr.rollouts.FullTrajectory(batch_size=1),
        reward=vr.rewards.Mock(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.GRPO(),
        train=vr.Train(steps=kwargs.pop("steps", 1), lr=1e-3),
        output_dir=output_dir,
        show_progress=False,
        strict_rollout_validation=True,
        **kwargs,
    )


def _deep_merge(target: dict, incoming: dict) -> None:
    for key, value in incoming.items():
        if isinstance(target.get(key), dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def test_lazy_exports_keep_config_access_runtime_pure():
    script = """
import sys
import visual_rl as vr
assert 'visual_rl.experiment' not in sys.modules
assert 'visual_rl.runner' not in sys.modules
_ = vr.VisualRLConfig
assert 'visual_rl.experiment' not in sys.modules
assert 'visual_rl.runner' not in sys.modules
_ = vr.Experiment
assert 'visual_rl.experiment' in sys.modules
assert 'visual_rl.runner' not in sys.modules
assert 'torch' not in sys.modules
assert 'diffusers' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_descriptors_are_frozen_and_construction_validate_have_no_output_side_effect(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "not-created"
    descriptor = vr.models.TinyDiffusion(image_size=8)
    with pytest.raises(FrozenInstanceError):
        descriptor.image_size = 16

    experiment = _mock_experiment(output_dir)
    assert not output_dir.exists()
    import visual_rl.builtins as builtins_module

    monkeypatch.setattr(
        builtins_module,
        "register_builtin_plugins",
        lambda: (_ for _ in ()).throw(
            AssertionError("static validation loaded runtime plugins")
        ),
    )
    report = experiment.validate()
    assert not report.trusted
    assert not output_dir.exists()


def test_validate_can_explicitly_run_the_formal_trusted_preflight(tmp_path):
    output_dir = tmp_path / "trusted-not-created"

    report = _mock_experiment(output_dir).validate(trusted_components=True)

    assert report.trusted is True
    assert report.components
    assert not output_dir.exists()


def test_relative_paths_use_experiment_construction_cwd(tmp_path, monkeypatch):
    construction_dir = tmp_path / "construction"
    later_dir = tmp_path / "later"
    construction_dir.mkdir()
    later_dir.mkdir()
    monkeypatch.chdir(construction_dir)
    experiment = vr.Experiment(
        model=vr.models.Wan(
            checkpoint="models/wan",
            world_r1_root="repos/world-r1",
        ),
        rollout=vr.rollouts.FullTrajectory(),
        reward=vr.rewards.Mock(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.GRPO(),
        train=vr.Train(),
        output_dir="runs/api",
    )

    monkeypatch.chdir(later_dir)
    config = experiment.resolve()

    assert config.model.model_path == str(construction_dir / "models/wan")
    assert config.model.extra["world_r1_root"] == str(
        construction_dir / "repos/world-r1"
    )
    assert config.paths.output_dir == str(construction_dir / "runs/api")


def test_python_and_yaml_resolve_to_identical_config_and_fingerprint(tmp_path):
    experiment = vr.Experiment(
        model=vr.models.TinyDiffusion(image_size=12),
        rollout=vr.rollouts.Flash(selected_steps=3),
        reward=vr.rewards.PromptColor(default_color="blue"),
        advantage=vr.advantages.GroupNormalize(epsilon=1e-4),
        objective=vr.objectives.FlashGRPO(clip_range=0.02),
        train=vr.Train(steps=2, lr=0.05),
        run_name="equivalent",
        seed=9,
        output_dir=tmp_path / "run",
    )

    values = {
        "run_name": "equivalent",
        "seed": 9,
        "use_lora": True,
        "paths": {"output_dir": str(tmp_path / "run")},
    }
    for descriptor in (
        experiment.model,
        experiment.rollout,
        experiment.reward,
        experiment.advantage,
        experiment.objective,
        experiment.train,
    ):
        _deep_merge(values, descriptor.to_config())
    yaml_path = tmp_path / "equivalent.yaml"
    yaml_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    python_values = experiment.to_config()
    yaml_values = config_to_dict(load_config(yaml_path))
    assert python_values == yaml_values
    assert config_fingerprint(python_values, {}, version=1) == config_fingerprint(
        yaml_values, {}, version=1
    )


def test_reward_and_objective_are_single_fragment_replacements(tmp_path):
    base = _mock_experiment(tmp_path / "base").to_config()
    changed_reward = vr.Experiment(
        model=vr.models.MockWan(),
        rollout=vr.rollouts.FullTrajectory(batch_size=1),
        reward=vr.rewards.PromptColor(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.GRPO(),
        train=vr.Train(steps=1, lr=1e-3),
        output_dir=tmp_path / "base",
        show_progress=False,
        strict_rollout_validation=True,
    ).to_config()
    changed_objective = vr.Experiment(
        model=vr.models.MockWan(),
        rollout=vr.rollouts.Flash(),
        reward=vr.rewards.Mock(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.FlashGRPO(),
        train=vr.Train(steps=1, lr=1e-3),
        output_dir=tmp_path / "base",
        show_progress=False,
        strict_rollout_validation=True,
    ).to_config()

    assert {key for key in base if base[key] != changed_reward[key]} == {"rewards"}
    assert changed_objective["algorithm"]["name"] == "flash_grpo"
    assert changed_objective["sample"]["name"] == "single_step"


def test_existing_compatibility_validation_rejects_flash_tempflow(tmp_path):
    experiment = vr.Experiment(
        model=vr.models.TinyDiffusion(),
        rollout=vr.rollouts.Flash(),
        reward=vr.rewards.PromptColor(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.TempFlow(),
        train=vr.Train(),
        output_dir=tmp_path / "invalid",
    )
    with pytest.raises(ValueError, match="Incompatible config"):
        experiment.validate()
    assert not (tmp_path / "invalid").exists()


def test_python_api_exposes_precision_microbatch_and_reward_execution(tmp_path):
    experiment = vr.Experiment(
        model=vr.models.MockWan(),
        rollout=vr.rollouts.FullTrajectory(batch_size=1),
        reward=vr.rewards.Mock(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.GRPO(),
        train=vr.Train(
            steps=2,
            precision="bf16",
            update_microbatch_size=1,
        ),
        reward_execution=vr.RewardExecution(
            mode="async",
            max_workers=2,
            microbatch_size=1,
            timeout_s=2.0,
            max_in_flight=2,
            require_hard_timeout=True,
        ),
        output_dir=tmp_path / "runtime-api",
    )

    config = experiment.resolve()

    assert config.train.precision == "bf16"
    assert config.train.update_microbatch_size == 1
    assert config.runner.reward_executor.mode == "async"
    assert config.runner.reward_executor.max_workers == 2
    assert config.runner.reward_executor.microbatch_size == 1
    assert config.runner.reward_executor.max_in_flight == 2
    assert config.runner.reward_executor.require_hard_timeout is True
    assert not (tmp_path / "runtime-api").exists()


def test_mock_run_uses_one_runner_and_returns_lightweight_result(tmp_path, monkeypatch):
    import visual_rl.runner as runner_module

    real_runner = runner_module.ExperimentRunner
    constructions = 0

    class CountingRunner(real_runner):
        def __init__(self, config):
            nonlocal constructions
            constructions += 1
            super().__init__(config)

    monkeypatch.setattr(runner_module, "ExperimentRunner", CountingRunner)
    result = _mock_experiment(tmp_path / "run", run_name="api-mock").run(
        ["orbit around a small vase"]
    )

    assert constructions == 1
    assert isinstance(result, vr.RunResult)
    assert result.run_id == "api-mock"
    assert result.completed_steps == 1
    assert result.latest_checkpoint == result.output_dir / "checkpoint_000001"
    assert result.latest_checkpoint.is_dir()
    assert [row["step"] for row in result.iter_metrics()] == [0]
    assert result.load_manifest().records
    resolved = json.loads(
        (result.output_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["dataset"]["prompts"] == ["orbit around a small vase"]
    with pytest.raises(FrozenInstanceError):
        result.completed_steps = 2


def test_python_api_resume_matches_continuous_run_and_keeps_experiment_immutable(
    tmp_path,
):
    import torch

    prompts = ["orbit around a small vase"]
    continuous = _mock_experiment(tmp_path / "continuous", steps=2).run(prompts)

    split_dir = tmp_path / "split"
    first = _mock_experiment(split_dir, steps=1).run(prompts)
    resume_experiment = _mock_experiment(split_dir, steps=2)
    before = config_to_dict(resume_experiment.resolve())
    resumed = resume_experiment.run(
        prompts,
        resume_from=first.output_dir / "latest.json",
    )

    assert config_to_dict(resume_experiment.resolve()) == before
    assert resume_experiment.resolve().paths.resume_from is None
    continuous_rows = list(continuous.iter_metrics())
    resumed_rows = list(resumed.iter_metrics())
    assert [row["step"] for row in continuous_rows] == [0, 1]
    assert [row["step"] for row in resumed_rows] == [0, 1]
    for name in ("loss", "reward_mean", "approx_kl", "clipfrac"):
        assert resumed_rows[1][name] == continuous_rows[1][name]
    continuous_state = torch.load(
        continuous.latest_checkpoint / "mock_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        resumed.latest_checkpoint / "mock_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(resumed_state["policy_bias"], continuous_state["policy_bias"])


def test_python_api_rejects_tampered_resume_before_model_or_output_side_effects(
    tmp_path, monkeypatch
):
    prompts = ["orbit around a small vase"]
    source = _mock_experiment(tmp_path / "source").run(prompts)
    adapter_state = source.latest_checkpoint / "mock_adapter.pt"
    adapter_state.write_bytes(adapter_state.read_bytes() + b"tampered")

    from visual_rl.model_adapters.mock import MockWanAdapter

    model_calls = 0

    def fail_model_init(self, config):
        del self, config
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("resume validation instantiated a model")

    monkeypatch.setattr(MockWanAdapter, "__init__", fail_model_init)
    output_dir = tmp_path / "must-not-exist"
    experiment = _mock_experiment(output_dir, steps=2)
    from visual_rl.preflight import ResumePreflightError
    from visual_rl.runner import ResumeError

    with pytest.raises(ResumeError, match="recover resume-source"):
        experiment.run(prompts, resume_from=tmp_path / "missing" / "latest.json")
    with pytest.raises(
        (ResumeError, ResumePreflightError),
        match="SHA256 mismatch|recover resume-source",
    ):
        experiment.run(prompts, resume_from=source.output_dir / "latest.json")

    assert model_calls == 0
    assert not output_dir.exists()
