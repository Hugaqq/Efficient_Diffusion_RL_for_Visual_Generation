"""The final single-process/DDP Strategy surface."""

from __future__ import annotations

import inspect
import multiprocessing
import queue
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import visual_rl.distributed as distributed_module
from visual_rl.core.types import (
    FrozenMapping,
    MetricContribution,
    RewardBatch,
    ValidatedRuntimeEnv,
)
from visual_rl.distributed import (
    DDPStrategy,
    DistributedContext,
    DistributedFailureError,
    SingleProcessStrategy,
    build_strategy,
)


def _strategy() -> SingleProcessStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="single",
            device="cpu",
            timeout_s=5.0,
            max_snapshot_tensor_bytes=None,
        ),
        ValidatedRuntimeEnv(
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
            raw_launch_env=FrozenMapping(),
        ),
    )
    assert isinstance(strategy, SingleProcessStrategy)
    return strategy


def _rewards() -> RewardBatch:
    values = torch.tensor([1.0, 3.0], dtype=torch.float32)
    return RewardBatch(
        sample_id=("sample-0", "sample-1"),
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(2, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": ({}, {})},
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_strategy(rank: int, port: int) -> DDPStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="ddp",
            device="cpu",
            timeout_s=15.0,
            max_snapshot_tensor_bytes=1 << 20,
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
            raw_launch_env=FrozenMapping(),
        ),
    )
    assert isinstance(strategy, DDPStrategy)
    return strategy


def _reducer_failure_worker(
    rank: int,
    port: int,
    mode: str,
    output,
) -> None:
    strategy = None
    try:
        strategy = _ddp_strategy(rank, port)
        if mode == "active_count":
            strategy.sum_active_transition_count(0 if rank == 1 else 1)
        elif mode == "reward_pack":
            if rank == 1:
                strategy._reward_values = lambda _rewards: (1e308, 1e308)
            strategy.reduce_reward_metrics(_rewards())
        else:
            raise AssertionError(f"unknown reducer failure mode: {mode}")
        output.put((rank, "unexpected success"))
    except DistributedFailureError as exc:
        output.put((rank, str(exc)))
    except BaseException as exc:
        output.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if strategy is not None:
            strategy.close()


def test_strategy_exposes_only_the_frozen_training_and_phase_surface() -> None:
    strategy = _strategy()
    try:
        expected = {
            "rank",
            "local_rank",
            "world_size",
            "device",
            "backend",
            "is_main_process",
            "run_phase",
            "dataset_start",
            "prepare",
            "recompute_policy_stats",
            "gradient_sync_context",
            "sum_active_transition_count",
            "reduce_tensor_weighted_mean",
            "reduce_metric_contributions",
            "reduce_reward_metrics",
            "atomic_optimizer_step",
            "gather_object",
            "broadcast_object",
            "failure_gate",
            "close",
        }
        public = {
            name
            for name in dir(strategy)
            if not name.startswith("_")
        }
        assert public == expected
        assert not hasattr(strategy, "synchronize_failure")
        assert not hasattr(strategy, "reduce_weighted_mean")
        assert not hasattr(strategy, "last_atomic_snapshot_metrics")
    finally:
        strategy.close()


def test_single_strategy_reduces_the_three_explicit_contracts() -> None:
    strategy = _strategy()
    try:
        assert strategy.dataset_start(3, 4) == 12
        assert strategy.sum_active_transition_count(7) == 7
        mean = strategy.reduce_tensor_weighted_mean(
            torch.tensor(2.5),
            3,
        )
        torch.testing.assert_close(mean, torch.tensor(2.5))
        metrics = strategy.reduce_metric_contributions(
            {
                "mean": MetricContribution(torch.tensor(9.0), 3),
                "sum": MetricContribution(torch.tensor(4.0), None),
            }
        )
        assert metrics == {"mean": 3.0, "sum": 4.0}
        assert strategy.reduce_reward_metrics(_rewards()) == {
            "reward_mean": 2.0,
            "reward_std": 1.0,
        }
    finally:
        strategy.close()


def test_single_atomic_boundary_validates_identity_then_steps_once() -> None:
    strategy = _strategy()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        parameter.grad = torch.tensor(1.0)
        optimizer.step()

    try:
        strategy.atomic_optimizer_step(
            operation,
            parameters=(parameter,),
            optimizer=optimizer,
            scaler=None,
        )
        assert calls == 1
        assert float(parameter.detach()) < 1.0
        with pytest.raises(ValueError, match="identity/order"):
            strategy.atomic_optimizer_step(
                operation,
                parameters=(torch.nn.Parameter(torch.tensor(2.0)),),
                optimizer=optimizer,
                scaler=None,
            )
        assert calls == 1
    finally:
        strategy.close()


def test_strategy_methods_keep_explicit_keyword_roots() -> None:
    assert tuple(
        inspect.signature(SingleProcessStrategy.gather_object).parameters
    ) == ("self", "value", "dst")
    assert tuple(
        inspect.signature(SingleProcessStrategy.broadcast_object).parameters
    ) == ("self", "value", "src")
    assert tuple(
        inspect.signature(
            SingleProcessStrategy.atomic_optimizer_step
        ).parameters
    ) == ("self", "operation", "parameters", "optimizer", "scaler")


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("active_count", "local active transition count must be positive"),
        ("reward_pack", "intermediate overflow"),
    ),
)
@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo is unavailable",
)
def test_gloo_rank_one_reducer_prepare_failure_is_bounded_and_synchronized(
    mode: str,
    message: str,
) -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_reducer_failure_worker,
            args=(rank, port, mode, output),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    rows = {}
    try:
        for _ in processes:
            rank, error = output.get(timeout=30)
            rows[rank] = error
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for reducer failure workers") from exc
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert set(rows) == {0, 1}
    assert all(message in rows[rank] for rank in (0, 1)), rows


def test_process_group_fatal_rejects_collective_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = DDPStrategy(
        DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            backend="gloo",
        ),
        timeout_s=5.0,
        max_snapshot_tensor_bytes=1 << 20,
        master_addr="127.0.0.1",
        master_port=29500,
    )
    backend_calls = 0

    def unexpected_backend_call(*_args, **_kwargs) -> None:
        nonlocal backend_calls
        backend_calls += 1

    monkeypatch.setattr(dist, "all_reduce", unexpected_backend_call)
    strategy._setup_complete = True
    strategy._mark_process_group_fatal()
    try:
        with pytest.raises(
            DistributedFailureError,
            match="unusable after a fatal failure",
        ):
            strategy._all_reduce(
                torch.tensor(1),
                operation="fatal poison regression",
            )
        assert backend_calls == 0
    finally:
        strategy.close()


def test_init_failure_poison_guards_every_ddp_training_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = DDPStrategy(
        DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            backend="gloo",
        ),
        timeout_s=5.0,
        max_snapshot_tensor_bytes=1 << 20,
        master_addr="127.0.0.1",
        master_port=29500,
    )
    init_calls = 0

    def fail_init(*_args, **_kwargs) -> None:
        nonlocal init_calls
        init_calls += 1
        raise RuntimeError("injected process-group initialization failure")

    monkeypatch.setattr(dist, "init_process_group", fail_init)
    try:
        with pytest.raises(
            DistributedFailureError,
            match="process-group initialization failed",
        ):
            strategy._setup()
        assert init_calls == 1

        backend_calls: list[str] = []

        def unexpected_backend_call(*_args, **_kwargs):
            backend_calls.append("called")
            raise AssertionError("fatal Strategy touched the backend")

        for name in (
            "all_gather_object",
            "all_reduce",
            "gather_object",
            "broadcast_object_list",
        ):
            monkeypatch.setattr(dist, name, unexpected_backend_call)
        monkeypatch.setattr(
            distributed_module,
            "DistributedDataParallel",
            unexpected_backend_call,
        )

        parameter = torch.nn.Parameter(torch.tensor(0.05))
        optimizer = torch.optim.AdamW([parameter], lr=1e-2)
        entries = (
            ("prepare", lambda: strategy.prepare(object())),
            (
                "recompute",
                lambda: strategy.recompute_policy_stats(
                    None,
                    require_reference=False,
                ),
            ),
            (
                "gradient context",
                lambda: strategy.gradient_sync_context(True),
            ),
            (
                "atomic optimizer",
                lambda: strategy.atomic_optimizer_step(
                    lambda: optimizer.step(),
                    parameters=(parameter,),
                    optimizer=optimizer,
                    scaler=None,
                ),
            ),
        )
        for _name, operation in entries:
            with pytest.raises(
                DistributedFailureError,
                match="unusable after a fatal failure",
            ):
                operation()

        assert backend_calls == []
        assert optimizer.state_dict()["state"] == {}
        assert float(parameter.detach()) == pytest.approx(0.05)
    finally:
        strategy.close()
