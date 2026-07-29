"""Contracts for the minimal read-only Callback lifecycle."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import visual_rl as vr
from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256
from visual_rl.core.types import FrozenMapping

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"
ARTIFACT_KEYS = {
    "authoritative_checkpoint",
    "resolved_config_path",
    "manifest_path",
    "metrics_path",
    "marker_path",
}


class RecordingCallback(vr.Callback):
    def __init__(self) -> None:
        self.events: list[vr.CallbackEvent] = []
        self.artifacts_existed_at_dispatch: list[bool] = []

    def _record(self, event: vr.CallbackEvent) -> None:
        self.events.append(event)
        if event.artifacts:
            self.artifacts_existed_at_dispatch.append(
                all(path.exists() for path in event.artifacts.values())
            )

    on_run_start = _record
    on_step_end = _record
    on_commit = _record
    on_run_end = _record


class RandomCallback(vr.Callback):
    def _consume_rng(self, _event: vr.CallbackEvent) -> None:
        random.random()
        np.random.random()
        torch.rand(())
        if torch.cuda.is_initialized():
            torch.rand((), device="cuda")

    on_run_start = _consume_rng
    on_step_end = _consume_rng
    on_commit = _consume_rng
    on_run_end = _consume_rng


class FailingCallback(vr.Callback):
    def on_step_end(self, event: vr.CallbackEvent) -> None:
        del event
        raise RuntimeError("injected callback failure")


class ReturningCallback(vr.Callback):
    def on_step_end(self, event: vr.CallbackEvent):
        del event
        return False


class InspectingCallback(vr.Callback):
    def __init__(self) -> None:
        self.committed_steps: list[int] = []

    def on_commit(self, event: vr.CallbackEvent) -> None:
        status = vr.inspect_run(event.output_dir)
        self.committed_steps.append(status.committed_steps)


class OrderedCallback(vr.Callback):
    def __init__(self, name: str, sink: list[tuple[str, str]]) -> None:
        self.name = name
        self.sink = sink

    def _record(self, event: vr.CallbackEvent) -> None:
        self.sink.append((event.kind, self.name))

    on_run_start = _record
    on_step_end = _record
    on_commit = _record
    on_run_end = _record


class FatalCallback(vr.Callback):
    def __init__(self, error_type: type[BaseException]) -> None:
        self.error_type = error_type

    def on_step_end(self, event: vr.CallbackEvent) -> None:
        del event
        raise self.error_type("injected callback fatal error")


def _write_config(
    tmp_path: Path,
    *,
    output_name: str,
    max_steps: int,
    checkpoint_every: int,
    resume: bool = False,
) -> Path:
    payload = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    output_dir = (tmp_path / output_name).resolve()
    payload["runtime"]["max_steps"] = max_steps
    payload["artifacts"]["output_dir"] = str(output_dir)
    payload["artifacts"]["checkpoint_every"] = checkpoint_every
    payload["artifacts"]["checkpoint_keep_last"] = 1
    payload["artifacts"]["preview_samples_per_event"] = 0
    payload["resume"]["from"] = str(output_dir) if resume else None
    path = tmp_path / f"{output_name}-{max_steps}-{int(resume)}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _contains_forbidden_value(value: Any) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)) or callable(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_forbidden_value(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_value(item) for item in value)
    return False


def test_fresh_lifecycle_has_exact_event_schemas_and_commit_visibility(
    tmp_path: Path,
) -> None:
    recorder = RecordingCallback()
    inspector = InspectingCallback()
    config = _write_config(
        tmp_path,
        output_name="lifecycle",
        max_steps=3,
        checkpoint_every=2,
    )

    result = vr.load(config).run(callbacks=[recorder, inspector])

    assert [(event.kind, event.step) for event in recorder.events] == [
        ("run_start", None),
        ("step_end", 0),
        ("step_end", 1),
        ("commit", 1),
        ("step_end", 2),
        ("commit", 2),
        ("run_end", 2),
    ]
    assert [event.committed_steps for event in recorder.events] == [
        0,
        0,
        2,
        2,
        3,
        3,
        3,
    ]
    assert inspector.committed_steps == [2, 3]

    start = recorder.events[0]
    assert start.metrics == {}
    assert start.artifacts == {}
    for event in recorder.events:
        assert event.run_id == result.run_id
        assert event.output_dir == result.output_dir
        assert event.target_steps == 3
        assert isinstance(event.metrics, FrozenMapping)
        assert isinstance(event.artifacts, FrozenMapping)
        assert not _contains_forbidden_value(event.metrics)
        assert not _contains_forbidden_value(event.artifacts)

    for event in recorder.events:
        if event.kind == "step_end":
            assert event.artifacts == {}
            assert event.metrics["step"] == event.step
            assert type(event.metrics["sample_count"]) is int
            assert type(event.metrics["active_transition_count"]) is int
        elif event.kind in {"commit", "run_end"}:
            assert set(event.artifacts) == ARTIFACT_KEYS
            assert event.metrics["step"] == event.step

    assert recorder.artifacts_existed_at_dispatch == [True, True, True]
    assert all(path.exists() for path in recorder.events[-1].artifacts.values())
    step_one = recorder.events[2]
    commit_one = recorder.events[3]
    assert commit_one.metrics is step_one.metrics
    assert recorder.events[-1].metrics == result.last_metrics
    with pytest.raises(FrozenInstanceError):
        recorder.events[-1].step = 99
    with pytest.raises(TypeError):
        recorder.events[-1].metrics["step"] = 99


def test_multiple_callbacks_run_in_user_order(tmp_path: Path) -> None:
    sink: list[tuple[str, str]] = []
    config = _write_config(
        tmp_path,
        output_name="ordered",
        max_steps=1,
        checkpoint_every=1,
    )

    vr.load(config).run(
        callbacks=[
            OrderedCallback("first", sink),
            OrderedCallback("second", sink),
        ]
    )

    assert sink == [
        (kind, name)
        for kind in ("run_start", "step_end", "commit", "run_end")
        for name in ("first", "second")
    ]


def test_observers_fail_open_and_preserve_complete_training_state(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="visual_rl.callbacks")
    trailing_recorder = RecordingCallback()
    variants: dict[str, list[vr.Callback] | None] = {
        "none": None,
        "empty": [],
        "reader": [RecordingCallback()],
        "random": [RandomCallback()],
        "failure": [FailingCallback(), trailing_recorder],
        "return": [ReturningCallback()],
    }
    results = {}
    for name, callbacks in variants.items():
        config = _write_config(
            tmp_path,
            output_name=name,
            max_steps=3,
            checkpoint_every=2,
        )
        experiment = vr.load(config)
        results[name] = (
            experiment.run()
            if callbacks is None
            else experiment.run(callbacks=callbacks)
        )

    assert {result.committed_steps for result in results.values()} == {3}
    assert len({result.last_metrics for result in results.values()}) == 1
    assert (
        len(
            {
                checkpoint_tree_sha256(result.authoritative_checkpoint)
                for result in results.values()
            }
        )
        == 1
    )
    assert len({result.metrics_path.read_bytes() for result in results.values()}) == 1
    assert [event.kind for event in trailing_recorder.events] == [
        "run_start",
        "step_end",
        "step_end",
        "commit",
        "step_end",
        "commit",
        "run_end",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "callback failure index=0 class=FailingCallback "
        "hook=on_step_end exception=RuntimeError" in message
        for message in messages
    )
    assert any(
        "callback misuse index=0 class=ReturningCallback "
        "hook=on_step_end return_type=bool" in message
        for message in messages
    )


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
def test_base_exceptions_are_not_swallowed_and_run_end_is_not_emitted(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    recorder = RecordingCallback()
    config = _write_config(
        tmp_path,
        output_name=f"fatal-{error_type.__name__}",
        max_steps=2,
        checkpoint_every=2,
    )

    with pytest.raises(error_type, match="injected callback fatal error"):
        vr.load(config).run(callbacks=[recorder, FatalCallback(error_type)])

    assert [event.kind for event in recorder.events] == [
        "run_start",
        "step_end",
    ]
    assert not any(event.kind == "run_end" for event in recorder.events)


def test_dispatch_restores_already_initialized_cuda_rng_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visual_rl import callbacks as callbacks_module

    output_dir = tmp_path.resolve()
    output_dir.mkdir(exist_ok=True)
    fake_cuda_state = {"value": torch.tensor([1, 2, 3], dtype=torch.uint8)}

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [fake_cuda_state["value"].clone()],
    )

    def restore(states) -> None:
        fake_cuda_state["value"] = states[0].clone()

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore)

    class ConsumeFakeCuda(vr.Callback):
        def on_run_start(self, event: vr.CallbackEvent) -> None:
            del event
            fake_cuda_state["value"] = torch.tensor(
                [9, 9, 9],
                dtype=torch.uint8,
            )

    event = vr.CallbackEvent(
        kind="run_start",
        run_id="run-cuda-guard",
        output_dir=output_dir,
        step=None,
        target_steps=1,
        committed_steps=0,
        metrics=FrozenMapping(),
        artifacts=FrozenMapping(),
    )
    callbacks_module._dispatch_callbacks(
        (ConsumeFakeCuda(),),
        "on_run_start",
        event,
    )

    assert torch.equal(
        fake_cuda_state["value"],
        torch.tensor([1, 2, 3], dtype=torch.uint8),
    )


def test_cleanup_failure_does_not_emit_run_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visual_rl.runner as runner_module

    recorder = RecordingCallback()
    original = runner_module.ExperimentRunner._close_local_run_resources

    def fail_first_cleanup(self):
        errors = original(self)
        if not getattr(self, "_callback_cleanup_failure_injected", False):
            self._callback_cleanup_failure_injected = True
            return (*errors, RuntimeError("injected cleanup failure"))
        return errors

    monkeypatch.setattr(
        runner_module.ExperimentRunner,
        "_close_local_run_resources",
        fail_first_cleanup,
    )
    config = _write_config(
        tmp_path,
        output_name="cleanup-failure",
        max_steps=1,
        checkpoint_every=1,
    )

    with pytest.raises(vr.RunError, match="cleanup"):
        vr.load(config).run(callbacks=[recorder])

    assert [event.kind for event in recorder.events] == [
        "run_start",
        "step_end",
        "commit",
    ]


def test_resume_observes_only_new_steps_and_noop_is_start_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _write_config(
        tmp_path,
        output_name="resume",
        max_steps=2,
        checkpoint_every=2,
    )
    vr.load(first).run()

    active_recorder = RecordingCallback()
    active = _write_config(
        tmp_path,
        output_name="resume",
        max_steps=3,
        checkpoint_every=2,
        resume=True,
    )
    active_result = vr.load(active).run(callbacks=[active_recorder])
    assert [(event.kind, event.step) for event in active_recorder.events] == [
        ("run_start", None),
        ("step_end", 2),
        ("commit", 2),
        ("run_end", 2),
    ]
    assert active_recorder.events[0].committed_steps == 2
    assert set(active_recorder.events[0].artifacts) == ARTIFACT_KEYS

    from visual_rl import runtime_factory

    def fail_runtime_build(*_args, **_kwargs):
        raise AssertionError("no-op resume must not build runtime components")

    monkeypatch.setattr(
        runtime_factory,
        "build_runtime_components",
        fail_runtime_build,
    )
    noop_recorder = RecordingCallback()
    noop = _write_config(
        tmp_path,
        output_name="resume",
        max_steps=3,
        checkpoint_every=2,
        resume=True,
    )
    noop_result = vr.load(noop).run(callbacks=[noop_recorder])
    assert [(event.kind, event.step) for event in noop_recorder.events] == [
        ("run_start", None),
        ("run_end", 2),
    ]
    assert noop_recorder.events[0].committed_steps == 3
    assert noop_recorder.events[1].metrics == active_result.last_metrics
    assert noop_result.last_metrics == active_result.last_metrics


def test_tiny_20_step_callback_continuous_resume_parity(
    tmp_path: Path,
) -> None:
    continuous = vr.load(
        _write_config(
            tmp_path,
            output_name="continuous",
            max_steps=20,
            checkpoint_every=5,
        )
    ).run(callbacks=[RandomCallback()])

    vr.load(
        _write_config(
            tmp_path,
            output_name="resumed",
            max_steps=10,
            checkpoint_every=5,
        )
    ).run(callbacks=[RandomCallback()])
    resumed = vr.load(
        _write_config(
            tmp_path,
            output_name="resumed",
            max_steps=20,
            checkpoint_every=5,
            resume=True,
        )
    ).run(callbacks=[RandomCallback()])

    assert continuous.committed_steps == resumed.committed_steps == 20
    assert continuous.last_metrics == resumed.last_metrics
    assert checkpoint_tree_sha256(
        continuous.authoritative_checkpoint
    ) == checkpoint_tree_sha256(resumed.authoritative_checkpoint)
    for result in (continuous, resumed):
        status = vr.inspect_run(result.output_dir)
        audit = vr.audit_run(result.output_dir)
        assert status.ok and status.committed_steps == 20
        assert audit.ok and audit.checked_commit_count == 4


def test_callback_event_rejects_unclosed_public_contracts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path.resolve()
    output_dir.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="artifacts"):
        vr.CallbackEvent(
            kind="run_start",
            run_id="run-1",
            output_dir=output_dir,
            step=None,
            target_steps=2,
            committed_steps=1,
            metrics=FrozenMapping(),
            artifacts=FrozenMapping(),
        )
    with pytest.raises(ValueError, match="metrics"):
        vr.CallbackEvent(
            kind="step_end",
            run_id="run-1",
            output_dir=output_dir,
            step=0,
            target_steps=2,
            committed_steps=0,
            metrics=FrozenMapping({"step": 0}),
            artifacts=FrozenMapping(),
        )


def test_callback_event_defensively_freezes_input_mappings(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path.resolve()
    output_dir.mkdir(exist_ok=True)
    checkpoint = output_dir / "checkpoint_000001"
    checkpoint.mkdir()
    commits = output_dir / "commits"
    commits.mkdir()
    raw_metrics = {
        "step": 0,
        "sample_count": 1,
        "active_transition_count": 1,
        "reward_mean": 0.5,
    }
    raw_artifacts = {
        "authoritative_checkpoint": checkpoint,
        "resolved_config_path": output_dir / "config.resolved.json",
        "manifest_path": output_dir / "sample_manifest.json",
        "metrics_path": output_dir / "metrics.jsonl",
        "marker_path": commits / "commit_000001.json",
    }
    for key, path in raw_artifacts.items():
        if key != "authoritative_checkpoint":
            path.touch()

    event = vr.CallbackEvent(
        kind="commit",
        run_id="run-freeze",
        output_dir=output_dir,
        step=0,
        target_steps=1,
        committed_steps=1,
        metrics=raw_metrics,
        artifacts=raw_artifacts,
    )
    raw_metrics["reward_mean"] = 9.0
    raw_artifacts["metrics_path"] = output_dir / "replacement.jsonl"

    assert event.metrics["reward_mean"] == 0.5
    assert event.artifacts["metrics_path"] == output_dir / "metrics.jsonl"
    assert isinstance(event.metrics, FrozenMapping)
    assert isinstance(event.artifacts, FrozenMapping)
