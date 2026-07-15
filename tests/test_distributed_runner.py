"""CPU/gloo integration coverage for the native distributed ExperimentRunner."""

from __future__ import annotations

import copy
import json
import os
import queue
import socket
import time
import traceback
from multiprocessing.context import SpawnContext
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.artifacts import ArtifactManager
from visual_rl.configs.schema import VisualRLConfig
from visual_rl.distributed import DistributedFailureError
import visual_rl.runner as runner_module
from visual_rl.runner import ExperimentRunner, ResumeError


def _config(
    output_dir: str | Path,
    *,
    steps: int,
    resume_from: str | Path | None = None,
    deterministic_run_dir: bool = True,
) -> VisualRLConfig:
    config = VisualRLConfig(run_name="distributed-runner")
    config.paths.output_dir = str(output_dir)
    config.paths.resume_from = None if resume_from is None else str(resume_from)
    config.dataset.prompts = [f"prompt-{index}" for index in range(8)]
    config.model.latent_shape = [1, 1, 1, 1]
    config.model.media_shape = [1, 3, 2, 2]
    config.sample.batch_size = 1
    config.sample.samples_per_prompt = 2
    config.sample.num_steps = 1
    config.train.max_steps = steps
    config.train.save_every = 1
    config.train.learning_rate = 1e-2
    config.runner.show_progress = False
    config.runner.strict_rollout_validation = True
    config.runner.deterministic_run_dir = deterministic_run_dir
    config.runner.distributed.backend = "gloo"
    config.runner.distributed.device = "cpu"
    config.runner.distributed.timeout_s = 5.0
    return config


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _runner_worker(
    rank: int,
    ports: tuple[int, int, int],
    output_base: str,
    failure_output: str,
    results: Any,
) -> None:
    commit_calls = 0
    trigger_decision_writes = 0
    original_commit = ArtifactManager.commit
    original_save_json = runner_module.save_json

    def counted_commit(self, *args, **kwargs):
        nonlocal commit_calls
        commit_calls += 1
        return original_commit(self, *args, **kwargs)

    ArtifactManager.commit = counted_commit

    def counted_save_json(path, data):
        nonlocal trigger_decision_writes
        if Path(path).name == "trigger_decision.json":
            trigger_decision_writes += 1
        return original_save_json(path, data)

    runner_module.save_json = counted_save_json
    try:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(ports[0]),
        )

        first = ExperimentRunner(
            _config(output_base, steps=1, deterministic_run_dir=False)
        )
        first_groups: list[str] = []
        original_sample = first.rollout.sample

        def record_first(*args, **kwargs):
            batch = original_sample(*args, **kwargs)
            first_groups.extend(batch.group_id)
            return batch

        first.rollout.sample = record_first
        first_metrics = first.run()
        first_parameter = float(first.adapter.policy_bias.detach().cpu())
        actual_output = str(first.output_dir)

        first_manifest_count = None
        first_manifest_ranks = None
        checkpoint_ranks = None
        if rank == 0:
            first_manifest = json.loads(
                (first.output_dir / "sample_manifest.json").read_text(encoding="utf-8")
            )["records"]
            first_manifest_count = len(first_manifest)
            first_manifest_ranks = [
                int(record["sample_id"].split("-rank-")[1][:4])
                for record in first_manifest
            ]
            checkpoint = torch.load(
                first.output_dir / "checkpoint_000001" / "training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            checkpoint_ranks = [
                entry["rank"] for entry in checkpoint["distributed_state"]["entries"]
            ]

        os.environ["MASTER_PORT"] = str(ports[1])
        resumed = ExperimentRunner(
            _config(
                actual_output,
                steps=2,
                resume_from=Path(actual_output) / "latest.json",
            )
        )
        resumed_groups: list[str] = []
        resumed_sample = resumed.rollout.sample

        def record_resumed(*args, **kwargs):
            batch = resumed_sample(*args, **kwargs)
            resumed_groups.extend(batch.group_id)
            return batch

        resumed.rollout.sample = record_resumed
        resumed_metrics = resumed.run()
        resumed_parameter = float(resumed.adapter.policy_bias.detach().cpu())

        final_manifest_count = None
        latest_step = None
        if rank == 0:
            final_manifest_count = len(
                json.loads(
                    (resumed.output_dir / "sample_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["records"]
            )
            latest_step = json.loads(
                (resumed.output_dir / "latest.json").read_text(encoding="utf-8")
            )["step"]

        os.environ["MASTER_PORT"] = str(ports[2])
        failing = ExperimentRunner(_config(failure_output, steps=1))
        parameters = [
            parameter.detach().clone() for parameter in failing.adapter.parameters()
        ]
        optimizer_state = copy.deepcopy(failing.optimizer.state_dict())
        rollback_probe = {"value": 0}
        original_plugin_state_dict = failing.optimizer_plugin.state_dict
        original_plugin_load_state_dict = failing.optimizer_plugin.load_state_dict

        def plugin_state_dict():
            return {
                **original_plugin_state_dict(),
                "rollback_probe": rollback_probe["value"],
            }

        def plugin_load_state_dict(state):
            restored = dict(state)
            rollback_probe["value"] = int(restored.pop("rollback_probe"))
            original_plugin_load_state_dict(restored)

        def bypass_atomic_callback(
            *,
            adapter,
            optimizer,
            **_kwargs,
        ):
            rollback_probe["value"] += 1
            for parameter in adapter.parameters():
                parameter.grad = torch.ones_like(parameter)
            result = optimizer.step()
            return {
                f"rank_{rank}_only_metric": float(result or 0.0),
                "loss": 0.0,
            }

        failing.optimizer_plugin.state_dict = plugin_state_dict
        failing.optimizer_plugin.load_state_dict = plugin_load_state_dict
        failing.optimizer_plugin.step = bypass_atomic_callback
        try:
            failing.run()
        except DistributedFailureError as exc:
            failure_error = str(exc)
        else:
            raise AssertionError("rank-divergent result contract was not rejected")
        parameters_restored = all(
            torch.equal(parameter.detach(), before)
            for parameter, before in zip(
                failing.adapter.parameters(),
                parameters,
                strict=True,
            )
        )
        optimizer_restored = _state_equal(
            failing.optimizer.state_dict(),
            optimizer_state,
        )
        plugin_state_restored = rollback_probe["value"] == 0
        failure_commit_exists = rank == 0 and any(
            (failing.output_dir / "commits").glob("commit_*.json")
        )

        results.put(
            (
                "ok",
                {
                    "rank": rank,
                    "actual_output": actual_output,
                    "first_groups": sorted(set(first_groups)),
                    "resumed_groups": sorted(set(resumed_groups)),
                    "first_metrics": first_metrics,
                    "resumed_metrics": resumed_metrics,
                    "first_parameter": first_parameter,
                    "resumed_parameter": resumed_parameter,
                    "first_manifest_count": first_manifest_count,
                    "first_manifest_ranks": first_manifest_ranks,
                    "final_manifest_count": final_manifest_count,
                    "checkpoint_ranks": checkpoint_ranks,
                    "latest_step": latest_step,
                    "global_step": resumed.global_step,
                    "commit_calls": commit_calls,
                    "trigger_decision_writes": trigger_decision_writes,
                    "failure_error": failure_error,
                    "failure_commit_exists": failure_commit_exists,
                    "failure_parameters_restored": parameters_restored,
                    "failure_optimizer_restored": optimizer_restored,
                    "failure_plugin_state_restored": plugin_state_restored,
                    "reward_cache": str(
                        Path(actual_output) / "reward_cache" / f"rank_{rank:04d}"
                    ),
                    "rollout_cache": str(
                        Path(actual_output) / "rollouts" / f"rank_{rank:04d}"
                    ),
                },
            )
        )
    except BaseException:
        results.put(("error", traceback.format_exc()))
    finally:
        ArtifactManager.commit = original_commit
        runner_module.save_json = original_save_json


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo distributed backend is unavailable",
)
@pytest.mark.distributed
def test_two_rank_runner_resume_artifacts_metrics_and_failure(tmp_path) -> None:
    started = time.monotonic()
    deadline = started + 12.0
    context: SpawnContext = mp.get_context("spawn")
    results = context.Queue()
    ports: list[int] = []
    while len(ports) < 3:
        candidate = _free_loopback_port()
        if candidate not in ports:
            ports.append(candidate)
    output_base = tmp_path / "nondeterministic"
    failure_output = tmp_path / "failure"
    processes = [
        context.Process(
            target=_runner_worker,
            args=(
                rank,
                tuple(ports),
                str(output_base),
                str(failure_output),
                results,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            pytest.fail("2-rank distributed runner integration timed out")
        assert process.exitcode == 0

    received = []
    for _ in range(2):
        try:
            status, payload = results.get(timeout=1)
        except queue.Empty:
            pytest.fail("distributed runner worker returned no result")
        assert status == "ok", payload
        received.append(payload)
    by_rank = {payload["rank"]: payload for payload in received}

    assert set(by_rank) == {0, 1}
    assert by_rank[0]["actual_output"] == by_rank[1]["actual_output"]
    assert by_rank[0]["first_groups"] != by_rank[1]["first_groups"]
    assert all(len(payload["first_groups"]) == 1 for payload in by_rank.values())
    assert by_rank[0]["first_parameter"] == pytest.approx(by_rank[1]["first_parameter"])
    assert by_rank[0]["resumed_parameter"] == pytest.approx(
        by_rank[1]["resumed_parameter"]
    )
    assert by_rank[0]["first_metrics"] == by_rank[1]["first_metrics"]
    assert by_rank[0]["resumed_metrics"] == by_rank[1]["resumed_metrics"]
    assert by_rank[0]["first_metrics"][0]["sample_count"] == 4
    assert by_rank[0]["first_metrics"][0]["reward_executor_attempts"] == 2
    assert by_rank[0]["first_metrics"][0]["reward_executor_shards"] == 2
    assert by_rank[0]["first_metrics"][0]["peak_rollback_snapshot_tensor_bytes"] > 0
    assert by_rank[0]["first_metrics"][0]["rollback_snapshot_capture_time_s"] >= 0
    assert by_rank[0]["first_manifest_count"] == 4
    assert by_rank[0]["first_manifest_ranks"] == [0, 0, 1, 1]
    assert by_rank[0]["final_manifest_count"] == 8
    assert by_rank[0]["checkpoint_ranks"] == [0, 1]
    assert by_rank[0]["latest_step"] == 2
    assert all(payload["global_step"] == 2 for payload in by_rank.values())
    assert by_rank[0]["commit_calls"] == 2
    assert by_rank[1]["commit_calls"] == 0
    assert by_rank[0]["trigger_decision_writes"] == 3
    assert by_rank[1]["trigger_decision_writes"] == 0
    assert all(Path(payload["reward_cache"]).is_dir() for payload in by_rank.values())
    assert all(Path(payload["rollout_cache"]).is_dir() for payload in by_rank.values())
    assert all(
        "result contracts must match on every rank" in payload["failure_error"]
        for payload in by_rank.values()
    )
    assert all(payload["failure_parameters_restored"] for payload in by_rank.values())
    assert all(payload["failure_optimizer_restored"] for payload in by_rank.values())
    assert all(payload["failure_plugin_state_restored"] for payload in by_rank.values())
    assert by_rank[0]["failure_commit_exists"] is False

    single_resume = _config(
        by_rank[0]["actual_output"],
        steps=3,
        resume_from=Path(by_rank[0]["actual_output"]) / "latest.json",
    )
    with pytest.raises(ResumeError, match="world size changed"):
        ExperimentRunner(single_resume)
    assert time.monotonic() - started < 12.0


@pytest.mark.parametrize("field", ["split_roles", "fsdp2"])
def test_invalid_conditional_scaling_closes_before_model_or_output(
    tmp_path,
    monkeypatch,
    field,
) -> None:
    config = _config(tmp_path / "must-not-exist", steps=1)
    setattr(config.runner.conditional_scaling, field, True)
    calls = {"distributed_context": 0, "model": 0}

    def record_distributed_context(*_args, **_kwargs):
        calls["distributed_context"] += 1
        raise AssertionError("distributed setup must not run")

    def record_model(*_args, **_kwargs):
        calls["model"] += 1
        raise AssertionError("model lookup must not run")

    monkeypatch.setattr(
        runner_module.DistributedContext, "from_env", record_distributed_context
    )
    monkeypatch.setattr(runner_module.MODEL_ADAPTERS, "get", record_model)

    with pytest.raises(ValueError, match=field):
        ExperimentRunner(config)

    assert calls == {"distributed_context": 0, "model": 0}
    assert not Path(config.paths.output_dir).exists()
