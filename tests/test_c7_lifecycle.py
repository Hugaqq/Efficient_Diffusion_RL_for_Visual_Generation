"""CPU/offline contracts for C7 callbacks and held-out evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import visual_rl as vr
from visual_rl.callbacks import CallbackError, RunCallback
from visual_rl.configs.schema import config_from_dict, config_to_dict
from visual_rl.datasets.prompt_dataset import prompt_content_sha256
from visual_rl.runner import ExperimentRunner


def _experiment(tmp_path: Path, **kwargs) -> vr.Experiment:
    return vr.Experiment(
        model=vr.models.MockWan(),
        rollout=vr.rollouts.FullTrajectory(batch_size=1),
        reward=vr.rewards.Mock(),
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.GRPO(),
        train=vr.Train(steps=kwargs.pop("steps", 1), lr=1e-3),
        output_dir=tmp_path / "run",
        show_progress=False,
        strict_rollout_validation=True,
        **kwargs,
    )


def _runner(tmp_path: Path, *, steps: int = 1, **kwargs) -> ExperimentRunner:
    experiment = _experiment(tmp_path, steps=steps)
    config = experiment._resolve_config(["train one", "train two"])
    config.train.max_steps = steps
    config.train.save_every = 1
    return ExperimentRunner(config, **kwargs)


class _Events(RunCallback):
    def __init__(self, label: str, events: list[tuple]):
        self.label = label
        self.events = events

    def _record(self, event: str, context) -> None:
        self.events.append((self.label, event, context.global_step))

    def on_run_start(self, context) -> None:
        self._record("start", context)

    def on_step_end(self, context) -> None:
        with pytest.raises(TypeError):
            context.metrics["new"] = 1
        with pytest.raises(TypeError):
            context.artifacts["new"] = "path"
        self._record("step", context)

    def on_checkpoint(self, context) -> None:
        self._record("checkpoint", context)

    def on_run_end(self, context) -> None:
        self._record("end", context)


def test_callbacks_are_ordered_read_only_and_resume_from_restored_step(tmp_path):
    first = _runner(tmp_path, steps=1)
    first.run()

    events: list[tuple] = []
    resumed = _runner(tmp_path / "resume", steps=2)
    resumed.config.paths.output_dir = str(first.output_dir)
    resumed.config.paths.resume_from = str(first.output_dir / "latest.json")
    resumed = ExperimentRunner(
        resumed.config,
        callbacks=(_Events("a", events), _Events("b", events)),
    )
    resumed.run()

    assert events == [
        ("a", "start", 1),
        ("b", "start", 1),
        ("a", "step", 2),
        ("b", "step", 2),
        ("a", "checkpoint", 2),
        ("b", "checkpoint", 2),
        ("a", "end", 2),
        ("b", "end", 2),
    ]


def test_callback_errors_fail_closed_and_end_error_does_not_mask_training_error(
    tmp_path, monkeypatch
):
    events: list[str] = []

    class StepFailure(RunCallback):
        def on_step_end(self, context) -> None:
            raise RuntimeError("stop")

        def on_run_end(self, context) -> None:
            events.append("end")

    with pytest.raises(CallbackError, match="StepFailure.on_step_end"):
        _runner(tmp_path, callbacks=(StepFailure(),)).run()
    assert events == ["end"]

    class EndFailure(RunCallback):
        def on_run_end(self, context) -> None:
            raise RuntimeError("end failure")

    runner = _runner(tmp_path / "primary", callbacks=(EndFailure(),))
    monkeypatch.setattr(
        runner.feedback_provider,
        "score",
        lambda batch: (_ for _ in ()).throw(RuntimeError("training failure")),
    )
    with pytest.raises(RuntimeError, match="training failure") as raised:
        runner.run()
    notes = getattr(raised.value, "__notes__", ())
    note = getattr(raised.value, "visual_rl_callback_note", "")
    assert any("on_run_end callback failed" in item for item in notes) or (
        "on_run_end callback failed" in note
    )


def test_checkpoint_callback_follows_marker_when_latest_projection_fails(
    tmp_path,
    monkeypatch,
):
    events: list[tuple] = []
    runner = _runner(tmp_path, callbacks=(_Events("only", events),))
    monkeypatch.setattr(
        runner,
        "_commit_checkpoint",
        lambda *args: (_ for _ in ()).throw(RuntimeError("commit failed")),
    )
    metrics = runner.run()

    assert [row["step"] for row in metrics] == [0]
    assert (runner.output_dir / "commits" / "commit_000001.json").is_file()
    assert runner.post_commit_bookkeeping_errors == [
        {
            "operation": "latest_projection",
            "error": "RuntimeError: commit failed",
        }
    ]
    assert events == [
        ("only", "start", 0),
        ("only", "step", 1),
        ("only", "checkpoint", 1),
        ("only", "end", 1),
    ]


class _ProbeEvaluator(vr.Evaluator):
    name = "probe"

    def __init__(self):
        self.prompts = ()
        self.context = None
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None
        self.mode = None

    def evaluate(self, *, adapter, prompts, context):
        self.prompts = prompts
        self.context = context
        self.mode = adapter.train_module.training
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state().clone()
        random.random()
        np.random.rand()
        torch.rand(1)
        adapter.train(True)
        return vr.EvaluationResult(metrics={"heldout_score": 0.75})


def test_inline_evaluation_is_isolated_persisted_and_lazily_loaded(tmp_path):
    evaluator = _ProbeEvaluator()
    result = _experiment(
        tmp_path,
        evaluator=evaluator,
        evaluation_prompts=["heldout one", "heldout two", "heldout three"],
    ).run(["train one", "train two"])

    result_path = result.evaluation_paths["probe"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert evaluator.prompts == ("heldout one", "heldout two", "heldout three")
    assert evaluator.mode is False
    assert evaluator.context.rank == 0
    assert evaluator.context.world_size == 1
    assert payload["metrics"] == {"heldout_score": 0.75}
    assert payload["context"]["prompt_count"] == 3
    assert result.load_evaluation("probe").metrics["heldout_score"] == 0.75
    assert not (result.output_dir / "reward_cache" / "evaluation").exists()
    assert [row["step"] for row in result.iter_metrics()] == [0]
    assert not (result.output_dir / "evaluation" / "probe" / "media").exists()
    assert random.getstate() == evaluator.python_state
    assert np.array_equal(np.random.get_state()[1], evaluator.numpy_state[1])
    assert torch.equal(torch.get_rng_state(), evaluator.torch_state)


def test_path_evaluation_hash_limit_and_split_rejection(tmp_path):
    heldout = tmp_path / "heldout.txt"
    heldout.write_text("heldout one\nheldout two\nheldout three\n", encoding="utf-8")
    evaluator = _ProbeEvaluator()
    runner = _runner(tmp_path / "path", evaluator=evaluator)
    runner.config.evaluation.path = str(heldout)
    runner.config.evaluation.max_prompts = 2
    runner = ExperimentRunner(runner.config, evaluator=evaluator)
    runner.run()
    assert evaluator.prompts == ("heldout one", "heldout two")
    assert runner.config.evaluation.content_sha256 == prompt_content_sha256(
        ["heldout one", "heldout two", "heldout three"]
    )

    invalid_values = config_to_dict(runner.config)
    invalid_values["evaluation"]["path"] = str(heldout)
    invalid_values["evaluation"]["prompts"] = ["also heldout"]
    with pytest.raises(ValueError, match="both prompts and path"):
        config_from_dict(invalid_values)

    overlap = _runner(tmp_path / "overlap")
    overlap.config.evaluation.prompts = ["train one"]
    with pytest.raises(RuntimeError, match="overlap"):
        ExperimentRunner(overlap.config)

    duplicate = _runner(tmp_path / "duplicate")
    duplicate.config.evaluation.prompts = ["heldout", "heldout"]
    with pytest.raises(RuntimeError, match="duplicates"):
        ExperimentRunner(duplicate.config)


def test_disabled_c7_keeps_legacy_artifact_surface(tmp_path):
    result = _experiment(tmp_path).run(["train only"])
    assert result.evaluations == ()
    assert dict(result.evaluation_paths) == {}
    assert not (result.output_dir / "evaluation").exists()


def test_experiment_run_resolves_preflights_and_constructs_runner_once(
    tmp_path, monkeypatch
):
    import visual_rl.experiment as experiment_module
    import visual_rl.preflight as preflight_module
    import visual_rl.runner as runner_module

    calls = {"resolve": 0, "static": 0, "trusted": 0, "runner": 0}
    real_resolve = experiment_module.resolve_experiment
    real_static = experiment_module.static_preflight
    real_trusted = preflight_module.trusted_component_load
    real_runner = runner_module.ExperimentRunner

    def count_resolve(*args, **kwargs):
        calls["resolve"] += 1
        return real_resolve(*args, **kwargs)

    def count_static(*args, **kwargs):
        calls["static"] += 1
        return real_static(*args, **kwargs)

    def count_trusted(*args, **kwargs):
        calls["trusted"] += 1
        return real_trusted(*args, **kwargs)

    class CountingRunner(real_runner):
        def __init__(self, *args, **kwargs):
            calls["runner"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "resolve_experiment", count_resolve)
    monkeypatch.setattr(experiment_module, "static_preflight", count_static)
    monkeypatch.setattr(preflight_module, "static_preflight", count_static)
    monkeypatch.setattr(preflight_module, "trusted_component_load", count_trusted)
    monkeypatch.setattr(runner_module, "ExperimentRunner", CountingRunner)

    _experiment(tmp_path).run(["train one"])
    assert calls == {"resolve": 1, "static": 1, "trusted": 1, "runner": 1}


def test_failing_evaluator_restores_mode_rng_optimizer_and_training_artifacts(
    tmp_path, monkeypatch
):
    class FailingEvaluator(vr.Evaluator):
        name = "failing"

        def __init__(self):
            self.python_state = None
            self.numpy_state = None
            self.torch_state = None

        def evaluate(self, *, adapter, prompts, context):
            assert not hasattr(context, "optimizer")
            assert not hasattr(context, "feedback_provider")
            self.python_state = random.getstate()
            self.numpy_state = np.random.get_state()
            self.torch_state = torch.get_rng_state().clone()
            random.random()
            np.random.rand()
            torch.rand(1)
            adapter.train(True)
            raise RuntimeError("evaluation failed")

    evaluator = FailingEvaluator()
    configured = _runner(tmp_path / "failing")
    configured.config.evaluation.prompts = ["heldout one"]
    runner = ExperimentRunner(configured.config, evaluator=evaluator)
    observed = {}
    original = runner._evaluate_with_preserved_state

    def observe(context):
        observed["mode"] = runner.adapter.train_module.training
        observed["optimizer"] = runner.optimizer.state_dict()
        observed["metrics"] = runner.artifacts.metric_path.read_bytes()
        observed["reward_table"] = (runner.output_dir / "reward_table.json").read_bytes()
        return original(context)

    monkeypatch.setattr(runner, "_evaluate_with_preserved_state", observe)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        runner.run()

    assert runner.adapter.train_module.training is observed["mode"]
    assert random.getstate() == evaluator.python_state
    assert np.array_equal(np.random.get_state()[1], evaluator.numpy_state[1])
    assert torch.equal(torch.get_rng_state(), evaluator.torch_state)
    assert runner.optimizer.state_dict()["param_groups"] == observed["optimizer"]["param_groups"]
    for key, state in observed["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(runner.optimizer.state_dict()["state"][key][name], value)
    assert runner.artifacts.metric_path.read_bytes() == observed["metrics"]
    assert (runner.output_dir / "reward_table.json").read_bytes() == observed["reward_table"]


class _CudaProbeEvaluator(vr.Evaluator):
    name = "cuda_probe"

    def evaluate(self, *, adapter, prompts, context):
        return vr.EvaluationResult(metrics={"score": 1.0})


def test_evaluation_does_not_capture_uninitialized_cuda_rng(tmp_path, monkeypatch):
    configured = _runner(tmp_path / "cuda-uninitialized")
    configured.config.evaluation.prompts = ["heldout one"]
    runner = ExperimentRunner(configured.config, evaluator=_CudaProbeEvaluator())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA RNG was captured")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: (_ for _ in ()).throw(AssertionError("CUDA RNG was restored")),
    )

    runner.run(max_steps=0)


def test_evaluation_restores_initialized_cuda_rng(tmp_path, monkeypatch):
    configured = _runner(tmp_path / "cuda-initialized")
    configured.config.evaluation.prompts = ["heldout one"]
    runner = ExperimentRunner(configured.config, evaluator=_CudaProbeEvaluator())
    captured_states = [torch.tensor([17], dtype=torch.uint8)]
    calls = {"get": 0, "set": []}

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def get_rng_state_all():
        calls["get"] += 1
        return captured_states

    def set_rng_state_all(states):
        calls["set"].append(states)

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", get_rng_state_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_rng_state_all)

    runner.run(max_steps=0)

    assert calls == {"get": 1, "set": [captured_states]}


def test_evaluation_star_import_keeps_new_and_legacy_exports():
    namespace = {}
    exec("from visual_rl.evaluation import *", namespace)

    assert {
        "Evaluator",
        "EvaluationContext",
        "EvaluationResult",
        "aggregate_sd3_run_summaries",
    }.issubset(namespace)
