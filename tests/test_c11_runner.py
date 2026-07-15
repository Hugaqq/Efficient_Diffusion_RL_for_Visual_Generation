"""CPU-only Runner integration coverage for the C11 reward executor."""

from __future__ import annotations

from copy import deepcopy
import json
import threading
import time

import pytest
import torch

from visual_rl.artifacts import ArtifactManager
from visual_rl.artifacts.checkpoint import config_fingerprint
from visual_rl.callbacks import CallbackError, RunCallback
from visual_rl.configs.schema import VisualRLConfig, config_to_dict
from visual_rl.core.types import RewardBatch
from visual_rl.feedback import RewardExecutionError
import visual_rl.runner as runner_module
from visual_rl.runner import ExperimentRunner


def _config(output_dir, *, mode: str = "sync", batch_size: int = 2):
    config = VisualRLConfig(run_name=f"c11-{mode}")
    config.paths.output_dir = str(output_dir)
    config.dataset.prompts = [f"prompt-{index}" for index in range(batch_size)]
    config.sample.batch_size = batch_size
    config.train.max_steps = 1
    config.train.save_every = 1
    config.runner.show_progress = False
    config.runner.strict_rollout_validation = True
    config.runner.reward_executor.mode = mode
    if mode == "async":
        config.runner.reward_executor.max_workers = 2
        config.runner.reward_executor.max_in_flight = 2
        config.runner.reward_executor.microbatch_size = 2
        config.runner.reward_executor.timeout_s = 1.0
        config.runner.reward_executor.submit_timeout_s = 1.0
    return config


def _reward(batch, positions):
    values = torch.tensor(
        [float(positions[sample_id] + 1) for sample_id in batch.sample_id]
    )
    return RewardBatch(
        raw={"mock": values},
        weighted={"mock": values},
        weighted_total=values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        metadata={},
        sample_id=list(batch.sample_id),
    )


def test_async_runner_shards_restores_order_and_persists_executor_metrics(
    tmp_path, monkeypatch
):
    runner = ExperimentRunner(_config(tmp_path / "async", mode="async"))
    sample_order: list[str] = []
    shard_calls: list[list[str]] = []
    completed: list[list[str]] = []
    observed_update_order: list[str] = []
    lock = threading.Lock()
    original_sample = runner.rollout.sample
    original_update = runner.optimizer_plugin.step

    def record_sample(*args, **kwargs):
        batch = original_sample(*args, **kwargs)
        sample_order.extend(batch.sample_id)
        return batch

    def score_shard(batch):
        with lock:
            shard_calls.append(list(batch.sample_id))
            call_index = len(shard_calls) - 1
        if call_index == 0:
            time.sleep(0.02)
        result = _reward(batch, {item: i for i, item in enumerate(sample_order)})
        with lock:
            completed.append(list(batch.sample_id))
        return result

    def record_update(*args, **kwargs):
        observed_update_order.extend(kwargs["rewards"].sample_id)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(runner.rollout, "sample", record_sample)
    monkeypatch.setattr(runner.feedback_provider, "score", score_shard)
    monkeypatch.setattr(
        runner.feedback_provider,
        "supports_concurrent_score",
        True,
        raising=False,
    )
    monkeypatch.setattr(runner.optimizer_plugin, "step", record_update)

    metrics = runner.run()[0]

    assert len(shard_calls) == 2
    assert all(len(shard) == 2 for shard in shard_calls)
    assert completed[0] == shard_calls[1]
    assert observed_update_order == sample_order
    assert metrics["reward_executor_shards"] == 2.0
    assert metrics["reward_executor_attempts"] == 2.0
    assert metrics["reward_executor_retries"] == 0.0
    assert metrics["reward_executor_cancelled"] == 0.0
    assert metrics["reward_executor_timeouts"] == 0.0
    assert metrics["reward_executor_wall_time_s"] >= 0.0
    assert metrics["reward_executor_queue_wait_count"] == 2.0
    assert metrics["reward_executor_service_latency_p95_s"] >= 0.0
    assert metrics["reward_executor_first_failure_rate"] == 0.0
    persisted = json.loads(
        (runner.output_dir / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert persisted["reward_executor_shards"] == 2.0
    assert persisted["reward_executor_attempts"] == 2.0
    assert persisted["reward_executor_queue_wait_count"] == 2.0
    assert runner.reward_executor._closed is True


def test_default_sync_runner_keeps_provider_identity_and_training_fingerprint(
    tmp_path, monkeypatch
):
    runner = ExperimentRunner(_config(tmp_path / "sync"))
    identity = deepcopy(runner.checkpoint_identity)
    sync_config = config_to_dict(runner.config)
    async_config = deepcopy(sync_config)
    async_config["runner"]["reward_executor"]["mode"] = "async"
    async_config["runner"]["reward_executor"]["require_hard_timeout"] = True
    async_config["runner"]["distributed"]["max_snapshot_tensor_bytes"] = 2048
    calls = 0
    original_score = runner.feedback_provider.score

    def score(batch):
        nonlocal calls
        calls += 1
        return original_score(batch)

    monkeypatch.setattr(runner.feedback_provider, "score", score)

    assert runner.reward_executor is None
    assert config_fingerprint(sync_config, identity) == config_fingerprint(
        async_config, identity
    )
    metrics = runner.run()[0]

    assert calls == 1
    assert runner.reward_executor.mode == "sync"
    assert metrics["reward_executor_shards"] == 1.0
    assert runner.checkpoint_identity == identity
    assert "reward_executor" not in json.dumps(identity, sort_keys=True)
    runner.reward_executor.close()
    runner.reward_executor.close()


def test_reward_failure_skips_update_and_commit_and_close_cannot_mask_it(
    tmp_path, monkeypatch
):
    runner = ExperimentRunner(_config(tmp_path / "failure", mode="async"))
    calls = {"update": 0, "commit": 0}
    real_factory = runner_module.build_reward_executor

    def build_with_failing_close(*args, **kwargs):
        executor = real_factory(*args, **kwargs)
        real_close = executor.close

        def close():
            real_close()
            raise RuntimeError("close exploded")

        executor.close = close
        return executor

    def fail_reward(_batch):
        raise RuntimeError("reward exploded")

    def count_update(*args, **kwargs):
        calls["update"] += 1
        return {}

    def count_commit(*args, **kwargs):
        calls["commit"] += 1
        return {}

    monkeypatch.setattr(
        runner_module, "build_reward_executor", build_with_failing_close
    )
    monkeypatch.setattr(runner.feedback_provider, "score", fail_reward)
    monkeypatch.setattr(runner.optimizer_plugin, "step", count_update)
    monkeypatch.setattr(runner.artifacts, "commit", count_commit)

    with pytest.raises(RewardExecutionError, match="Reward shard") as raised:
        runner.run()

    assert calls == {"update": 0, "commit": 0}
    assert runner.reward_executor._closed is True
    notes = getattr(raised.value, "__notes__", ())
    fallback_note = getattr(raised.value, "visual_rl_executor_close_note", "")
    assert any("close exploded" in note for note in notes) or (
        "close exploded" in fallback_note
    )
    with ArtifactManager(runner.output_dir, runner.config.run_name, resume=True):
        pass


class _FailOnEnd(RunCallback):
    def on_run_end(self, context) -> None:
        raise RuntimeError("callback exploded")


@pytest.mark.parametrize("stage", ["rollout", "update", "commit", "callback"])
def test_executor_closes_for_every_runner_failure_stage(tmp_path, monkeypatch, stage):
    runner = ExperimentRunner(_config(tmp_path / stage))

    def fail(*args, **kwargs):
        raise RuntimeError(f"{stage} exploded")

    if stage == "rollout":
        monkeypatch.setattr(runner.rollout, "sample", fail)
    elif stage == "update":
        monkeypatch.setattr(runner.optimizer_plugin, "step", fail)
    elif stage == "commit":
        monkeypatch.setattr(runner.artifacts, "commit", fail)
    else:
        runner.callbacks = (_FailOnEnd(),)

    expected = CallbackError if stage == "callback" else RuntimeError
    message = "on_run_end" if stage == "callback" else stage
    with pytest.raises(expected, match=message):
        runner.run()

    assert runner.reward_executor._closed is True
    runner.reward_executor.close()
    with ArtifactManager(runner.output_dir, runner.config.run_name, resume=True):
        pass


def test_executor_build_failure_releases_writer_lock(tmp_path, monkeypatch):
    runner = ExperimentRunner(_config(tmp_path / "build-failure", mode="async"))

    def fail_build(*args, **kwargs):
        raise RuntimeError("executor build exploded")

    monkeypatch.setattr(runner_module, "build_reward_executor", fail_build)

    with pytest.raises(RuntimeError, match="executor build exploded"):
        runner.run()

    assert runner.reward_executor is None
    with ArtifactManager(runner.output_dir, runner.config.run_name, resume=True):
        pass


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("1", TypeError),
    ],
)
def test_executor_metrics_reject_non_finite_and_wrong_types(value, error):
    rewards = RewardBatch(
        raw={},
        weighted={},
        weighted_total=torch.tensor([]),
        valid_mask=torch.tensor([], dtype=torch.bool),
        metadata={"_executor": {"attempts": value}},
        sample_id=[],
    )

    with pytest.raises(error, match="attempts"):
        ExperimentRunner._reward_runtime_metrics(rewards)
