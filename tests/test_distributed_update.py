"""CPU contracts for routing distributed forwards through UpdateEngine."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
import queue
import socket
import time
import traceback
from multiprocessing.context import SpawnContext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.core.types import (
    FrozenMapping,
    RewardBatch,
    RolloutBatch,
    StepContext,
    ValidatedRuntimeEnv,
)
from visual_rl.distributed import (
    DDPStrategy,
    DistributedFailureError,
    build_strategy,
)
from visual_rl.optimizers import AdvantageResult, ObjectiveOutput, UpdateEngine
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm


def _ddp_strategy(rank: int, port: int) -> DDPStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="ddp",
            device="cpu",
            timeout_s=8.0,
            max_snapshot_tensor_bytes=1 << 30,
        ),
        ValidatedRuntimeEnv(
            mode="ddp",
            rank=rank,
            local_rank=rank,
            world_size=2,
            local_world_size=2,
            group_rank=0,
            group_world_size=1,
            master_addr="127.0.0.1",
            master_port=port,
            visible_gpu_count=0,
            raw_launch_env=FrozenMapping(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": "2",
                    "LOCAL_WORLD_SIZE": "2",
                    "GROUP_RANK": "0",
                    "GROUP_WORLD_SIZE": "1",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(port),
                }
            ),
        ),
    )
    assert isinstance(strategy, DDPStrategy)
    return strategy


def _batch() -> RolloutBatch:
    batch_size, transitions = 4, 2
    return RolloutBatch(
        prompts=tuple(f"prompt-{index}" for index in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(batch_size, transitions, dtype=torch.bool),
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=tuple(f"prompt-id-{index}" for index in range(batch_size)),
        group_id=("group-a", "group-a", "group-b", "group-b"),
        branch_id=tuple(range(batch_size)),
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=3, seed=19),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={
            "features": torch.arange(
                1,
                batch_size * transitions + 1,
                dtype=torch.float32,
            ).reshape(batch_size, transitions)
        },
        artifact_metadata={},
    )


def _rewards(batch: RolloutBatch) -> RewardBatch:
    values = torch.arange(1, batch.batch_size + 1, dtype=torch.float32)
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={
            "score": tuple({} for _ in range(batch.batch_size))
        },
    )


class _Advantage:
    def __call__(self, batch: RolloutBatch, rewards: RewardBatch) -> AdvantageResult:
        del rewards
        return AdvantageResult(torch.ones(batch.batch_size), {})


class _Objective:
    def __call__(self, batch, advantages, new_log_probs):
        loss = (new_log_probs * advantages[:, None]).mean()
        detached = loss.detach()
        zero = detached * 0.0
        return ObjectiveOutput(
            loss=loss,
            policy_loss=detached,
            approx_kl=zero,
            clipfrac=zero,
            metrics={},
        )


class _Adapter:
    def __init__(self, events: list[str] | None = None) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.25))
        self.events = events
        self.default_calls = 0

    def prepare_for_training(self) -> None:
        return None

    def recompute_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        self.default_calls += 1
        if self.events is not None:
            self.events.append("forward")
        return self.parameter * batch.recompute_payload["features"]

    def parameters(self):
        return [self.parameter]


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, events: list[str] | None = None) -> None:
        super().__init__(parameters, lr=0.01)
        self.events = events
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        if self.events is not None:
            self.events.append("step")
        return super().step(closure=closure)


class _FlashDistributedAdapter:
    def __init__(self) -> None:
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.transformer.weight.zero_()

    @property
    def train_module(self) -> torch.nn.Module:
        return self.transformer

    def prepare_for_training(self) -> None:
        return None

    def recompute_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        return self.transformer(batch.recompute_payload["features"])

    def parameters(self):
        return self.transformer.parameters()


class _RecordingFlashAlgorithm(FlashGRPOAlgorithm):
    observed_global_mean: torch.Tensor | None = None
    observed_prepared_weights: torch.Tensor | None = None

    def apply_global_batch_reduction(
        self,
        batch: RolloutBatch,
        advantages,
        global_mean: Any,
    ) -> RolloutBatch:
        self.observed_global_mean = torch.as_tensor(global_mean).detach().clone()
        prepared = super().apply_global_batch_reduction(
            batch,
            advantages,
            global_mean,
        )
        self.observed_prepared_weights = (
            prepared.recompute_payload[self._PREPARED_RECTIFICATION_KEY]
            .detach()
            .clone()
        )
        return prepared


class _MicrobatchDistributedAdapter:
    def __init__(self) -> None:
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.transformer.weight.fill_(0.25)

    @property
    def train_module(self) -> torch.nn.Module:
        return self.transformer

    def prepare_for_training(self) -> None:
        return None

    def recompute_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        features = batch.recompute_payload["features"].unsqueeze(-1)
        return self.transformer(features).squeeze(-1)

    def parameters(self):
        return self.transformer.parameters()


def _counting_allreduce_hook(
    state,
    bucket,
):
    state["calls"] += 1
    buffer = bucket.buffer()
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    buffer.div_(state["world_size"])
    future = torch.futures.Future()
    future.set_result(buffer)
    return future


def _rank_microbatch_batch(rank: int) -> RolloutBatch:
    batch = _batch()
    start = rank * 8 + 1
    features = torch.arange(start, start + 8, dtype=torch.float32).reshape(4, 2)
    return batch.replace(
        prompts=tuple(f"rank-{rank}-prompt-{index}" for index in range(4)),
        sample_id=tuple(f"rank-{rank}-sample-{index}" for index in range(4)),
        prompt_id=tuple(
            f"rank-{rank}-prompt-id-{index}" for index in range(4)
        ),
        group_id=tuple(
            f"rank-{rank}-group-{index // 2}" for index in range(4)
        ),
        recompute_payload={"features": features},
    )


def _microbatch_no_sync_worker(rank: int, port: int, results: Any) -> None:
    strategy: DDPStrategy | None = None
    try:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
        strategy = _ddp_strategy(rank, port)
        context = strategy.context
        adapter = _MicrobatchDistributedAdapter()
        strategy.prepare(adapter)
        hook_state = {"calls": 0, "world_size": context.world_size}
        strategy.module.register_comm_hook(hook_state, _counting_allreduce_hook)
        batch = _rank_microbatch_batch(rank)
        rewards = _rewards(batch)

        observations: dict[str, dict[str, float | int]] = {}
        cases = (
            ("full", None, strategy.gradient_sync_context),
            ("legacy_microbatch", 2, None),
            ("no_sync_microbatch", 2, strategy.gradient_sync_context),
        )
        for name, microbatch_size, sync_context in cases:
            with torch.no_grad():
                adapter.transformer.weight.fill_(0.25)
            adapter.transformer.weight.grad = None
            optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
            calls_before = int(hook_state["calls"])
            UpdateEngine(
                _Advantage(),
                _Objective(),
                update_microbatch_size=microbatch_size,
                require_nonzero_gradients=True,
            ).step(
                adapter,
                batch,
                rewards,
                optimizer,
                batch.context,
                recompute_log_probs=strategy.recompute_log_probs,
                gradient_sync_context=sync_context,
            )
            gradient = adapter.transformer.weight.grad
            if gradient is None:
                raise AssertionError("distributed update produced no gradient")
            observations[name] = {
                "gradient": float(gradient.item()),
                "parameter": float(adapter.transformer.weight.item()),
                "communication_calls": int(hook_state["calls"]) - calls_before,
            }

        strategy.barrier()
        strategy.close()
        results.put(("ok", {"rank": rank, "observations": observations}))
    except BaseException:
        if strategy is not None:
            strategy.close()
        results.put(("error", traceback.format_exc()))


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rank_flash_batch(rank: int, coefficient: Any, step: int) -> RolloutBatch:
    context = StepContext(step=step, seed=101)
    coefficients = torch.as_tensor(coefficient, dtype=torch.float32).reshape(-1)
    batch_size = coefficients.numel()
    return RolloutBatch(
        prompts=tuple(
            f"rank-{rank}-{index}" for index in range(batch_size)
        ),
        metadata=tuple({"rank": rank} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, 1, 1),
        next_latents=torch.ones(batch_size, 1, 1),
        timesteps=torch.zeros(batch_size, 1),
        old_log_probs=torch.zeros(batch_size, 1),
        transition_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        sample_id=tuple(
            f"sample-{rank}-{index}" for index in range(batch_size)
        ),
        prompt_id=tuple(
            f"prompt-{rank}-{index}" for index in range(batch_size)
        ),
        group_id=("shared",) * batch_size,
        branch_id=tuple(range(batch_size)),
        media_layout="BCHW",
        camera_trajectory=None,
        context=context,
        selected_timestep_index=torch.full(
            (batch_size,), rank, dtype=torch.int64
        ),
        flash_coefficient=coefficients[:, None],
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={
            "features": torch.ones(batch_size, 1, dtype=torch.float32),
        },
        artifact_metadata={"num_steps": 2},
    )


def _flash_global_mean_worker(rank: int, port: int, results: Any) -> None:
    strategy: DDPStrategy | None = None
    try:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
        strategy = _ddp_strategy(rank, port)
        adapter = _FlashDistributedAdapter()
        strategy.prepare(adapter)
        observations: list[dict[str, Any]] = []

        coefficient_cases = (
            ((1.0,), (3.0,)),
            ((0.1,), (0.7,)),
            ((1.0,), (3.0, 5.0)),
        )
        for step, coefficients_by_rank in enumerate(coefficient_cases):
            with torch.no_grad():
                adapter.transformer.weight.zero_()
            coefficients = coefficients_by_rank[rank]
            batch = _rank_flash_batch(rank, coefficients, step)
            rewards = RewardBatch(
                sample_id=batch.sample_id,
                raw={"score": torch.ones(batch.batch_size)},
                weighted={"score": torch.ones(batch.batch_size)},
                weighted_total=torch.ones(batch.batch_size),
                valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
                shared_metadata={"score": {}},
                sample_metadata={
                    "score": tuple({} for _ in range(batch.batch_size))
                },
            )
            algorithm = _RecordingFlashAlgorithm(
                objective_version="reference_v1",
                clip_range=0.1,
                beta=0.0,
            )
            plugin = AlgorithmOptimizerPlugin(algorithm, _Advantage())
            optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
            metrics = plugin.step(
                adapter,
                batch,
                rewards,
                optimizer,
                batch.context,
                recompute_log_probs=strategy.recompute_log_probs,
                reduce_tensor_weighted_mean=(
                    strategy.reduce_tensor_weighted_mean
                ),
                synchronize_failure=strategy.synchronize_failure,
            )
            if (
                algorithm.observed_global_mean is None
                or algorithm.observed_prepared_weights is None
            ):
                raise AssertionError("Flash global preparation was not observed")
            observations.append(
                {
                    "coefficients": list(coefficients),
                    "global_mean": algorithm.observed_global_mean.item(),
                    "global_mean_dtype": str(
                        algorithm.observed_global_mean.dtype
                    ),
                    "prepared_weights": (
                        algorithm.observed_prepared_weights.reshape(-1).tolist()
                    ),
                    "metric_weight": metrics[
                        "flash_rectification_weight_mean"
                    ],
                }
            )

        invalid_batch = _rank_flash_batch(rank, float(rank + 1), 2)
        if rank == 1:
            invalid_batch = invalid_batch.replace(
                flash_coefficient=torch.ones(1, 2)
            )
        invalid_rewards = RewardBatch(
            sample_id=invalid_batch.sample_id,
            raw={"score": torch.ones(1)},
            weighted={"score": torch.ones(1)},
            weighted_total=torch.ones(1),
            valid_mask=torch.ones(1, dtype=torch.bool),
            shared_metadata={"score": {}},
            sample_metadata={"score": ({},)},
        )
        invalid_plugin = AlgorithmOptimizerPlugin(
            _RecordingFlashAlgorithm(objective_version="reference_v1"),
            _Advantage(),
        )
        invalid_optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
        try:
            invalid_plugin.step(
                adapter,
                invalid_batch,
                invalid_rewards,
                invalid_optimizer,
                invalid_batch.context,
                recompute_log_probs=strategy.recompute_log_probs,
                reduce_tensor_weighted_mean=(
                    strategy.reduce_tensor_weighted_mean
                ),
                synchronize_failure=strategy.synchronize_failure,
            )
        except DistributedFailureError as error:
            synchronized_prepare_error = str(error)
        else:
            raise AssertionError(
                "rank-local Flash coefficient failure was not synchronized"
            )
        strategy.barrier()

        strategy.close()
        results.put(
            (
                "ok",
                {
                    "rank": rank,
                    "observations": observations,
                    "synchronized_prepare_error": synchronized_prepare_error,
                },
            )
        )
    except BaseException:
        if strategy is not None:
            strategy.close()
        results.put(("error", traceback.format_exc()))


def _run(
    adapter: _Adapter,
    *,
    recompute_log_probs=None,
    gradient_sync_context=None,
    before_optimizer_step=None,
    optimizer_step=None,
    microbatch_size: int | None = None,
):
    batch = _batch()
    optimizer = _CountingSGD(adapter.parameters())
    metrics = UpdateEngine(
        _Advantage(),
        _Objective(),
        update_microbatch_size=microbatch_size,
        require_nonzero_gradients=True,
    ).step(
        adapter,
        batch,
        _rewards(batch),
        optimizer,
        batch.context,
        recompute_log_probs=recompute_log_probs,
        gradient_sync_context=gradient_sync_context,
        before_optimizer_step=before_optimizer_step,
        optimizer_step=optimizer_step,
    )
    return adapter, optimizer, metrics


def test_default_single_process_recompute_matches_injected_callable() -> None:
    default_adapter, _, default_metrics = _run(_Adapter())
    injected_adapter = _Adapter()
    injected_adapter, _, injected_metrics = _run(
        injected_adapter,
        recompute_log_probs=injected_adapter.recompute_log_probs,
    )

    assert default_adapter.default_calls == injected_adapter.default_calls == 1
    torch.testing.assert_close(default_adapter.parameter, injected_adapter.parameter)
    for key in default_metrics.keys() - {
        "recompute_time_s",
        "backward_time_s",
        "optimizer_time_s",
    }:
        assert injected_metrics[key] == pytest.approx(default_metrics[key])


def test_every_microbatch_uses_injected_recompute_callable() -> None:
    adapter = _Adapter()
    routed_slices: list[list[str]] = []

    def ddp_recompute(micro_batch: RolloutBatch) -> torch.Tensor:
        routed_slices.append(list(micro_batch.sample_id))
        return adapter.parameter * micro_batch.recompute_payload["features"]

    _, optimizer, metrics = _run(
        adapter,
        recompute_log_probs=ddp_recompute,
        microbatch_size=2,
    )

    assert routed_slices == [["sample-0", "sample-1"], ["sample-2", "sample-3"]]
    assert adapter.default_calls == 0
    assert optimizer.step_calls == 1
    assert metrics["update_microbatches"] == 2


def test_only_nonfinal_contributing_microbatches_use_no_sync() -> None:
    adapter = _Adapter()

    class FakeDDP:
        def __init__(self) -> None:
            self.no_sync_calls = 0
            self.entries = 0
            self.exits = 0

        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            self.entries += 1
            try:
                yield
            finally:
                self.exits += 1

    fake_ddp = FakeDDP()
    synchronize_flags: list[bool] = []

    def gradient_sync_context(synchronize_gradients: bool):
        synchronize_flags.append(synchronize_gradients)
        return nullcontext() if synchronize_gradients else fake_ddp.no_sync()

    _, optimizer, metrics = _run(
        adapter,
        recompute_log_probs=adapter.recompute_log_probs,
        gradient_sync_context=gradient_sync_context,
        microbatch_size=2,
    )

    assert synchronize_flags == [False, True]
    assert fake_ddp.no_sync_calls == fake_ddp.entries == fake_ddp.exits == 1
    assert optimizer.step_calls == 1
    assert metrics["update_microbatches"] == 2


def test_no_sync_context_exits_when_microbatch_forward_fails() -> None:
    adapter = _Adapter()
    entered = 0
    exited = 0

    @contextmanager
    def no_sync():
        nonlocal entered, exited
        entered += 1
        try:
            yield
        finally:
            exited += 1

    def gradient_sync_context(synchronize_gradients: bool):
        return nullcontext() if synchronize_gradients else no_sync()

    def fail_recompute(_batch: RolloutBatch):
        raise RuntimeError("microbatch forward failed")

    with pytest.raises(RuntimeError, match="microbatch forward failed"):
        _run(
            adapter,
            recompute_log_probs=fail_recompute,
            gradient_sync_context=gradient_sync_context,
            microbatch_size=2,
        )

    assert entered == exited == 1


def test_guard_runs_once_after_backward_and_before_optimizer_step() -> None:
    events: list[str] = []
    adapter = _Adapter(events)
    adapter.parameter.register_hook(
        lambda gradient: events.append("backward") or gradient
    )
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        assert adapter.parameter.grad is not None
        events.append("guard")

    batch = _batch()
    optimizer = _CountingSGD(adapter.parameters(), events)
    UpdateEngine(
        _Advantage(),
        _Objective(),
        update_microbatch_size=2,
        require_nonzero_gradients=True,
    ).step(
        adapter,
        batch,
        _rewards(batch),
        optimizer,
        batch.context,
        before_optimizer_step=guard,
    )

    assert guard_calls == 1
    assert events == [
        "forward",
        "backward",
        "forward",
        "backward",
        "guard",
        "step",
    ]


def test_guard_failure_does_not_step_or_update_parameters() -> None:
    adapter = _Adapter()
    batch = _batch()
    optimizer = _CountingSGD(adapter.parameters())
    before = adapter.parameter.detach().clone()
    calls = 0

    def reject_commit() -> None:
        nonlocal calls
        calls += 1
        assert adapter.parameter.grad is not None
        raise RuntimeError("rank failed before commit")

    with pytest.raises(RuntimeError, match="rank failed before commit"):
        UpdateEngine(
            _Advantage(),
            _Objective(),
            require_nonzero_gradients=True,
        ).step(
            adapter,
            batch,
            _rewards(batch),
            optimizer,
            batch.context,
            before_optimizer_step=reject_commit,
        )

    assert calls == 1
    assert optimizer.step_calls == 0
    torch.testing.assert_close(adapter.parameter, before)


def test_optimizer_step_callback_is_an_explicit_boundary() -> None:
    adapter = _Adapter()
    routed: list[dict[str, Any]] = []

    def route(operation, **state):
        routed.append(state)
        return operation()

    _, optimizer, _ = _run(adapter, optimizer_step=route)

    assert optimizer.step_calls == 1
    assert len(routed) == 1
    assert routed[0]["optimizer"] is optimizer
    assert routed[0]["parameters"] == [adapter.parameter]
    assert routed[0]["scaler"] is None


def test_injected_recompute_return_type_is_validated_before_step() -> None:
    adapter = _Adapter()
    batch = _batch()
    optimizer = _CountingSGD(adapter.parameters())

    with pytest.raises(TypeError, match="recompute_log_probs.*torch.Tensor"):
        UpdateEngine(_Advantage(), _Objective()).step(
            adapter,
            batch,
            _rewards(batch),
            optimizer,
            batch.context,
            recompute_log_probs=lambda _batch: [0.0],
        )

    assert adapter.default_calls == 0
    assert optimizer.step_calls == 0


def test_algorithm_plugin_forwards_update_routing_hooks() -> None:
    captured: dict[str, Any] = {}

    class RecordingEngine:
        def step(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"loss": 0.0}

    def recompute(_batch):
        return torch.tensor(0.0)

    def guard():
        return None

    def optimizer_step(operation, **state):
        del state
        return operation()

    plugin = AlgorithmOptimizerPlugin.__new__(AlgorithmOptimizerPlugin)
    plugin.update_engine = RecordingEngine()
    result = plugin.step(
        "adapter",
        "batch",
        "rewards",
        "optimizer",
        "context",
        recompute_log_probs=recompute,
        gradient_sync_context=nullcontext,
        before_optimizer_step=guard,
        optimizer_step=optimizer_step,
    )

    assert result == {"loss": 0.0}
    assert captured["kwargs"] == {
        "recompute_log_probs": recompute,
        "gradient_sync_context": nullcontext,
        "before_optimizer_step": guard,
        "optimizer_step": optimizer_step,
    }


def test_legacy_external_plugin_default_step_remains_compatible() -> None:
    class LegacyPlugin(OptimizerPlugin):
        def build_optimizer(self, parameters, train_config):
            del parameters, train_config
            return object()

        def step(self, adapter, batch, rewards, optimizer, context):
            del adapter, batch, rewards, optimizer, context
            return {"legacy": 1.0}

    assert LegacyPlugin().step(None, None, None, None, None) == {"legacy": 1.0}


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo distributed backend is unavailable",
)
@pytest.mark.distributed
def test_flash_reference_uses_global_float32_coefficient_mean_across_ranks() -> None:
    started = time.monotonic()
    deadline = started + 12.0
    process_context: SpawnContext = mp.get_context("spawn")
    results = process_context.Queue()
    port = _free_loopback_port()
    processes = [
        process_context.Process(
            target=_flash_global_mean_worker,
            args=(rank, port, results),
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
            process.join(timeout=2)
            pytest.fail("2-rank Flash coefficient oracle timed out")
        assert process.exitcode == 0

    received = []
    for _ in range(2):
        try:
            status, payload = results.get(timeout=1)
        except queue.Empty:
            pytest.fail("Flash DDP worker exited without returning a result")
        assert status == "ok", payload
        received.append(payload)

    by_rank = {item["rank"]: item for item in received}
    assert set(by_rank) == {0, 1}
    coefficient_cases = (
        ((1.0,), (3.0,)),
        ((0.1,), (0.7,)),
        ((1.0,), (3.0, 5.0)),
    )
    for case_index, coefficients_by_rank in enumerate(coefficient_cases):
        local_tensors = [
            torch.tensor(values, dtype=torch.float32)
            for values in coefficients_by_rank
        ]
        if local_tensors[0].numel() == local_tensors[1].numel():
            expected_global_mean = torch.stack(
                [values.mean() for values in local_tensors]
            ).sum(dtype=torch.float32) / 2
        else:
            expected_global_mean = torch.cat(local_tensors).mean()
        for rank in range(2):
            observation = by_rank[rank]["observations"][case_index]
            expected_weights = local_tensors[rank] / expected_global_mean
            assert observation["global_mean_dtype"] == "torch.float32"
            assert torch.tensor(
                observation["global_mean"], dtype=torch.float32
            ).item() == expected_global_mean.item()
            torch.testing.assert_close(
                torch.tensor(
                    observation["prepared_weights"],
                    dtype=torch.float32,
                ),
                expected_weights,
                rtol=0,
                atol=0,
            )
            assert observation["metric_weight"] == pytest.approx(
                expected_weights.mean().item()
            )

    assert by_rank[0]["observations"][0]["prepared_weights"] == [0.5]
    assert by_rank[1]["observations"][0]["prepared_weights"] == [1.5]
    for rank in range(2):
        synchronized_error = by_rank[rank]["synchronized_prepare_error"]
        assert "rank 1" in synchronized_error
        assert "coefficient must have shape" in synchronized_error
    assert time.monotonic() - started < 12.0


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo distributed backend is unavailable",
)
@pytest.mark.distributed
def test_ddp_microbatch_no_sync_matches_full_and_uses_one_reduction() -> None:
    started = time.monotonic()
    deadline = started + 12.0
    process_context: SpawnContext = mp.get_context("spawn")
    results = process_context.Queue()
    port = _free_loopback_port()
    processes = [
        process_context.Process(
            target=_microbatch_no_sync_worker,
            args=(rank, port, results),
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
            process.join(timeout=2)
            pytest.fail("2-rank microbatch no_sync oracle timed out")
        assert process.exitcode == 0

    received = []
    for _ in range(2):
        try:
            status, payload = results.get(timeout=1)
        except queue.Empty:
            pytest.fail("microbatch no_sync worker returned no result")
        assert status == "ok", payload
        received.append(payload)

    assert {item["rank"] for item in received} == {0, 1}
    expected_gradient = 8.5
    expected_parameter = 0.25 - 0.01 * expected_gradient
    for payload in received:
        observations = payload["observations"]
        full = observations["full"]
        legacy = observations["legacy_microbatch"]
        optimized = observations["no_sync_microbatch"]
        for result in (full, legacy, optimized):
            assert result["gradient"] == pytest.approx(expected_gradient)
            assert result["parameter"] == pytest.approx(expected_parameter)
        assert legacy["gradient"] == pytest.approx(full["gradient"])
        assert optimized["gradient"] == pytest.approx(full["gradient"])
        assert legacy["parameter"] == pytest.approx(full["parameter"])
        assert optimized["parameter"] == pytest.approx(full["parameter"])
        assert full["communication_calls"] == 1
        assert legacy["communication_calls"] == 2
        assert optimized["communication_calls"] == 1
    assert time.monotonic() - started < 12.0
