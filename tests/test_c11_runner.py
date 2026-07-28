"""Final W05 Runner contracts: one step result and one commit schedule."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import visual_rl as vr
from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.artifacts.manifest import SampleRecord
from visual_rl.core.types import (
    FrozenMapping,
    RewardBatch,
    RolloutBatch,
    StepContext,
    ValidatedRuntimeEnv,
)
from visual_rl.optimizers.update_engine import UpdateResult
from visual_rl.runner import (
    ExperimentRunner,
    StepArtifacts,
    StepMetrics,
    StepResult,
)


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"


def _single_env() -> ValidatedRuntimeEnv:
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


def _record(context: StepContext, sample_id: str = "sample-0") -> SampleRecord:
    return SampleRecord(
        run_id="run-contract",
        sample_id=sample_id,
        sample_index=0,
        step=context.step,
        rank=context.rank,
        prompt="a red cube",
        media_type="image",
        prompt_metadata=FrozenMapping({}),
        seed=context.seed,
        rollout_type="full_trajectory",
        timestep_summary=FrozenMapping({"values": (9,), "count": 1}),
        reward_values=FrozenMapping(
            {
                "raw": {"mock": 1.0},
                "weighted": {"mock": 1.0},
                "weighted_total": 1.0,
                "valid": True,
                "shared_metadata": {"mock": {}},
                "sample_metadata": {"mock": {}},
            }
        ),
        media_path=None,
        rollout_cache_path=None,
        checkpoint_path=None,
        model_metadata=FrozenMapping({"adapter": "fixture"}),
        prompt_id="prompt-0",
        group_id="group-0",
        branch_id=None,
    )


def _write_config(
    tmp_path: Path,
    *,
    max_steps: int,
    checkpoint_every: int,
) -> Path:
    payload = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    payload["runtime"]["max_steps"] = max_steps
    payload["artifacts"]["output_dir"] = str((tmp_path / "run").resolve())
    payload["artifacts"]["checkpoint_every"] = checkpoint_every
    destination = tmp_path / "config.yaml"
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def test_step_result_contracts_are_frozen_tensor_free_and_pickle_safe() -> None:
    context = StepContext(step=2, seed=44, rank=0, world_size=1)
    metrics = StepMetrics(
        values=FrozenMapping(
            {
                "loss": -0.25,
                "reward_mean": 0.75,
            }
        ),
        sample_count=1,
        active_transition_count=1,
    )
    artifacts = StepArtifacts(local_records=(_record(context),))
    result = StepResult(
        context=context,
        metrics=metrics,
        artifacts=artifacts,
    )

    restored = pickle.loads(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored == result
    assert isinstance(restored.metrics.values["loss"], float)
    assert all(
        not isinstance(value, torch.Tensor)
        for value in restored.metrics.values.values()
    )
    with pytest.raises(FrozenInstanceError):
        result.context = StepContext(step=3, seed=45)
    with pytest.raises(FrozenInstanceError):
        metrics.sample_count = 2
    with pytest.raises(TypeError):
        metrics.values["loss"] = 0.0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"step": 0.0}, "reserved"),
        ({"loss": 1}, "finite Python floats"),
        ({"loss": float("nan")}, "non-finite floats"),
    ],
)
def test_step_metrics_fail_closed_on_noncanonical_values(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        StepMetrics(
            values=FrozenMapping(values),
            sample_count=1,
            active_transition_count=1,
        )


def test_execute_step_preserves_one_step_context_object_end_to_end() -> None:
    config = vr.load(TINY).resolve()
    observed: dict[str, StepContext] = {}

    class Strategy:
        rank = 0
        world_size = 1

        @staticmethod
        def dataset_start(step: int, batch_size: int) -> int:
            assert (step, batch_size) == (0, 2)
            return 0

        @staticmethod
        def run_phase(_name, operation):
            return operation()

        @staticmethod
        def reduce_reward_metrics(_rewards):
            return {"reward_mean": 0.5, "reward_std": 0.0}

    class Dataset:
        @staticmethod
        def batch(start: int, size: int):
            assert (start, size) == (0, 2)
            return ("red", "blue"), ({}, {})

    class Rollout:
        @staticmethod
        def sample(*, adapter, prompts, metadata, context):
            del adapter
            observed["rollout"] = context
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=torch.zeros(2, 3, 2, 2),
                latents=torch.zeros(2, 1, 1),
                next_latents=torch.ones(2, 1, 1),
                timesteps=torch.ones(2, 1, dtype=torch.int64),
                old_log_probs=torch.zeros(2, 1),
                transition_mask=torch.ones(2, 1, dtype=torch.bool),
                sample_id=("sample-0", "sample-1"),
                prompt_id=("prompt-0", "prompt-1"),
                group_id=("group-0", "group-1"),
                branch_id=None,
                media_layout="BCHW",
                camera_trajectory=None,
                context=context,
                selected_timestep_index=None,
                flash_coefficient=None,
                branch_step_index=None,
                trajectory_step_index=None,
                transition_std_dev=None,
                recompute_payload={},
                artifact_metadata={"adapter": "fixture"},
            )

    class RewardExecutor:
        @staticmethod
        def score(batch, context):
            assert context is batch.context
            observed["reward"] = context
            values = torch.tensor([0.25, 0.75], dtype=torch.float32)
            return RewardBatch(
                sample_id=batch.sample_id,
                raw={"mock": values},
                weighted={"mock": values.clone()},
                weighted_total=values.clone(),
                valid_mask=torch.ones(2, dtype=torch.bool),
                shared_metadata={"mock": {}},
                sample_metadata={"mock": ({}, {})},
            )

    class OptimizerPlugin:
        @staticmethod
        def step(*, batch, rewards, optimizer, scaler, context, strategy):
            del rewards, optimizer, scaler, strategy
            assert context is batch.context
            observed["update"] = context
            return UpdateResult(
                loss=-0.5,
                policy_loss=-0.5,
                reference_kl=0.0,
                approx_kl=0.0,
                clipfrac=0.0,
                active_transition_count=2,
                diagnostics={},
            )

    class Cache:
        @staticmethod
        def save(batch):
            observed["cache"] = batch.context
            return None, None

    runner = ExperimentRunner(config, _single_env())
    runner.strategy = Strategy()
    runner.components = SimpleNamespace(
        dataset=Dataset(),
        rollout=Rollout(),
        model=object(),
        reward_executor=RewardExecutor(),
        optimizer_plugin=OptimizerPlugin(),
    )
    runner.optimizer = object()
    runner.scaler = None
    runner.rollout_cache = Cache()
    runner.manifest_builder = ManifestBuilder(
        run_id="run-contract",
        media_type="image",
        rollout_type="full_trajectory",
    )

    result = runner._execute_step(0)

    assert result.context is observed["rollout"]
    assert all(value is result.context for value in observed.values())
    assert result.metrics.sample_count == 2
    assert result.metrics.active_transition_count == 2
    assert len(result.artifacts.local_records) == 2


def test_runner_uses_one_commit_coordinator_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visual_rl.runner as runner_module

    schedule: list[tuple[int, bool]] = []
    original = runner_module.CommitCoordinator.accept

    def record(self, step_result, *, should_checkpoint):
        schedule.append((step_result.context.step, should_checkpoint))
        return original(
            self,
            step_result,
            should_checkpoint=should_checkpoint,
        )

    monkeypatch.setattr(runner_module.CommitCoordinator, "accept", record)
    result = ExperimentRunner(
        vr.load(
            _write_config(
                tmp_path,
                max_steps=3,
                checkpoint_every=2,
            )
        ).resolve(),
        _single_env(),
    ).run()

    assert schedule == [(0, False), (1, True), (2, True)]
    assert result.committed_steps == 3
    assert result.authoritative_checkpoint.name == "checkpoint_000003"


def test_cleanup_failure_does_not_replace_primary_run_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ExperimentRunner(vr.load(TINY).resolve(), _single_env())

    def fail_preparation():
        raise ValueError("primary training failure")

    def cleanup_failure() -> tuple[BaseException, ...]:
        failure = RuntimeError("sensitive cleanup detail")
        failure._visual_rl_cleanup_owner = "coordinator"
        return (failure,)

    monkeypatch.setattr(runner, "_prepare_run", fail_preparation)
    monkeypatch.setattr(
        runner,
        "_close_local_run_resources",
        cleanup_failure,
    )

    with pytest.raises(ValueError, match="primary training failure") as caught:
        runner.run()

    notes = tuple(getattr(caught.value, "__notes__", ()))
    fallback = tuple(
        getattr(caught.value, "_visual_rl_cleanup_notes", ())
    )
    assert notes == (
        "VisualRL cleanup failures: RuntimeError@coordinator",
    ) or fallback == (("RuntimeError", "coordinator"),)
    assert "sensitive cleanup detail" not in "\n".join(notes)
    assert "sensitive cleanup detail" not in repr(fallback)


def test_broken_add_note_preserves_primary_and_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ExperimentRunner(vr.load(TINY).resolve(), _single_env())

    class BrokenAddNoteError(Exception):
        def add_note(self, _note: str) -> None:
            raise RuntimeError("injected add_note failure")

    primary = BrokenAddNoteError("primary training failure")

    def fail_preparation():
        raise primary

    def cleanup_failure() -> tuple[BaseException, ...]:
        failure = RuntimeError("sensitive cleanup detail")
        failure._visual_rl_cleanup_owner = "artifact_manager"
        return (failure,)

    monkeypatch.setattr(runner, "_prepare_run", fail_preparation)
    monkeypatch.setattr(
        runner,
        "_close_local_run_resources",
        cleanup_failure,
    )

    with pytest.raises(BrokenAddNoteError) as caught:
        runner.run()

    assert caught.value is primary
    assert str(caught.value) == "primary training failure"
    assert caught.value._visual_rl_cleanup_notes == (
        ("RuntimeError", "artifact_manager"),
    )
    assert "injected add_note failure" not in repr(
        caught.value._visual_rl_cleanup_notes
    )
    assert "sensitive cleanup detail" not in repr(
        caught.value._visual_rl_cleanup_notes
    )
