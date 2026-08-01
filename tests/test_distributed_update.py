"""Representative two-rank parity for the one UpdateEngine/Strategy path."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import socket
import time
from types import SimpleNamespace

import pytest
import torch

from visual_rl.core.types import (
    FrozenMapping,
    PolicyRecomputeStats,
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
from visual_rl.optimizers.advantages import AdvantageComputer
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.grpo import GRPOAlgorithm


class _TrainModule(torch.nn.Module):
    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.05, device=device))


class _Adapter:
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.train_module = _TrainModule(device)

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        if require_reference:
            raise AssertionError("this parity case has beta=0")
        return PolicyRecomputeStats(
            new_log_probs=(
                self.train_module.delta
                * batch.recompute_payload["features"]
            )
        )

    def named_parameters(self):
        return tuple(self.train_module.named_parameters())


class _FakeScaler:
    def __init__(self) -> None:
        self.value = 7

    def scale(self, value):
        return value

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        self.value += 1

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


def _plugin() -> AlgorithmOptimizerPlugin:
    return AlgorithmOptimizerPlugin(
        algorithm=GRPOAlgorithm(
            clip_range=0.2,
            adv_clip_max=5.0,
            beta=0.0,
        ),
        advantage_computer=AdvantageComputer(
            epsilon=1e-8,
            output_dtype="float32",
        ),
        update_microbatch_size=2,
        precision="fp32",
        max_grad_norm=None,
        max_initial_logprob_delta=None,
        require_initial_clipfrac_zero=False,
        require_finite_gradients=True,
        require_nonzero_gradients=True,
    )


def _optimizer(
    plugin: AlgorithmOptimizerPlugin,
    adapter: _Adapter,
) -> torch.optim.AdamW:
    return plugin.build_optimizer(
        adapter.named_parameters(),
        SimpleNamespace(
            learning_rate=1e-2,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_weight_decay=0.0,
            adam_epsilon=1e-8,
        ),
    )


def _batch(
    *,
    rank: int,
    world_size: int,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> RolloutBatch:
    batch_size, transition_count = features.shape
    device = features.device
    mask = mask.to(device=device)
    context = StepContext(
        step=0,
        seed=31 + rank,
        rank=rank,
        world_size=world_size,
    )
    return RolloutBatch(
        prompts=tuple(f"rank-{rank}" for _ in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1, device=device),
        latents=torch.zeros(batch_size, transition_count, 1, device=device),
        next_latents=torch.ones(
            batch_size,
            transition_count,
            1,
            device=device,
        ),
        timesteps=torch.arange(
            transition_count,
            device=device,
        ).expand(batch_size, -1),
        old_log_probs=torch.zeros(
            batch_size,
            transition_count,
            device=device,
        ),
        transition_mask=mask,
        sample_id=tuple(
            f"rank-{rank}-sample-{index}" for index in range(batch_size)
        ),
        prompt_id=tuple(f"rank-{rank}-prompt" for _ in range(batch_size)),
        group_id=tuple(f"rank-{rank}-group" for _ in range(batch_size)),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=context,
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={"features": features},
        artifact_metadata={},
    )


def _rewards(batch: RolloutBatch) -> RewardBatch:
    values = torch.tensor(
        [1.0, 3.0],
        dtype=torch.float32,
    )
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": ({}, {})},
    )


def _ddp_strategy(
    rank: int,
    port: int,
    *,
    max_snapshot_tensor_bytes: int = 1 << 20,
) -> DDPStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="ddp",
            device="cpu",
            timeout_s=15.0,
            max_snapshot_tensor_bytes=max_snapshot_tensor_bytes,
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


def _nccl_strategy_from_torchrun() -> DDPStrategy:
    required = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    present = tuple(name for name in required if name in os.environ)
    if present != required:
        pytest.skip(
            "requires a complete two-rank torchrun environment; "
            f"present launch keys: {present}"
        )
    if not torch.distributed.is_available():
        pytest.skip("requires torch.distributed")
    if not torch.distributed.is_nccl_available():
        pytest.skip("requires a PyTorch build with NCCL")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        master_port = int(os.environ["MASTER_PORT"])
    except ValueError as exc:
        pytest.skip(f"requires integer torchrun topology values: {exc}")
    if world_size != 2 or local_world_size != 2:
        pytest.skip(
            "requires WORLD_SIZE == LOCAL_WORLD_SIZE == 2, got "
            f"{world_size}/{local_world_size}"
        )
    if rank not in (0, 1) or local_rank not in (0, 1):
        pytest.skip(
            "requires RANK and LOCAL_RANK in {0, 1}, got "
            f"{rank}/{local_rank}"
        )
    if local_rank >= torch.cuda.device_count():
        pytest.skip(
            f"LOCAL_RANK={local_rank} has no visible CUDA device"
        )

    group_keys = ("GROUP_RANK", "GROUP_WORLD_SIZE")
    group_present = tuple(name for name in group_keys if name in os.environ)
    if group_present not in ((), group_keys):
        pytest.skip(
            "requires both GROUP_RANK and GROUP_WORLD_SIZE when either is set"
        )
    group_rank = None
    group_world_size = None
    if group_present:
        try:
            group_rank = int(os.environ["GROUP_RANK"])
            group_world_size = int(os.environ["GROUP_WORLD_SIZE"])
        except ValueError as exc:
            pytest.skip(f"requires integer torchrun group values: {exc}")
        if (group_rank, group_world_size) != (0, 1):
            pytest.skip(
                "requires a single-node torchrun group topology 0/1, got "
                f"{group_rank}/{group_world_size}"
            )

    launch_env = {
        name: os.environ[name]
        for name in (*required, *group_present)
    }
    strategy = build_strategy(
        SimpleNamespace(
            mode="ddp",
            device="cuda",
            timeout_s=30.0,
            max_snapshot_tensor_bytes=1 << 20,
        ),
        ValidatedRuntimeEnv(
            mode="ddp",
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            group_rank=group_rank,
            group_world_size=group_world_size,
            master_addr=os.environ["MASTER_ADDR"],
            master_port=master_port,
            visible_gpu_count=torch.cuda.device_count(),
            raw_launch_env=FrozenMapping(launch_env),
        ),
    )
    assert isinstance(strategy, DDPStrategy)
    assert strategy.backend == "nccl"
    assert strategy.device == torch.device("cuda", local_rank)
    return strategy


def _rank_case(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rank == 0:
        return (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[True, False], [True, False]]),
        )
    return (
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[True, True], [True, False]]),
    )


def _worker(rank: int, port: int, output) -> None:
    strategy = None
    try:
        strategy = _ddp_strategy(rank, port)
        adapter = _Adapter()
        strategy.prepare(adapter)
        plugin = _plugin()
        optimizer = _optimizer(plugin, adapter)
        features, mask = _rank_case(rank)
        batch = _batch(
            rank=rank,
            world_size=2,
            features=features,
            mask=mask,
        )
        result = plugin.step(
            batch=batch,
            rewards=_rewards(batch),
            optimizer=optimizer,
            scaler=None,
            context=batch.context,
            strategy=strategy,
        )
        output.put(
            (
                rank,
                {
                    "loss": result.loss,
                    "policy_loss": result.policy_loss,
                    "reference_kl": result.reference_kl,
                    "approx_kl": result.approx_kl,
                    "clipfrac": result.clipfrac,
                    "active_transition_count": (
                        result.active_transition_count
                    ),
                    "diagnostics": dict(result.diagnostics),
                    "gradient": float(
                        adapter.train_module.delta.grad.detach()
                    ),
                    "parameter": float(
                        adapter.train_module.delta.detach()
                    ),
                },
            )
        )
    except BaseException as exc:
        output.put((rank, (type(exc).__name__, str(exc))))
    finally:
        if strategy is not None:
            strategy.close()


def _atomic_failure_worker(
    rank: int,
    port: int,
    mode: str,
    output,
) -> None:
    strategy = None
    try:
        snapshot_limit = 1 if mode == "snapshot" and rank == 1 else 1 << 20
        strategy = _ddp_strategy(
            rank,
            port,
            max_snapshot_tensor_bytes=snapshot_limit,
        )
        parameter = torch.nn.Parameter(torch.tensor(0.05))
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=1e-2,
            weight_decay=0.0,
        )
        scaler = _FakeScaler()
        operation_calls = 0

        def operation() -> None:
            nonlocal operation_calls
            operation_calls += 1
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scaler.value += 1
            if mode == "operation" and rank == 1:
                raise RuntimeError("rank-one operation failure")

        error = None
        try:
            strategy.atomic_optimizer_step(
                operation,
                parameters=(parameter,),
                optimizer=optimizer,
                scaler=scaler,
            )
        except DistributedFailureError as exc:
            error = str(exc)
        if error is None:
            raise AssertionError("atomic failure case unexpectedly succeeded")
        output.put(
            (
                rank,
                {
                    "error": error,
                    "operation_calls": operation_calls,
                    "parameter": float(parameter.detach()),
                    "optimizer_state": optimizer.state_dict()["state"],
                    "scaler_value": scaler.value,
                },
            )
        )
    except BaseException as exc:
        output.put((rank, (type(exc).__name__, str(exc))))
    finally:
        if strategy is not None:
            strategy.close()


def _final_backward_prepare_failure_worker(
    rank: int,
    port: int,
    output,
) -> None:
    strategy = None
    try:
        strategy = _ddp_strategy(rank, port)
        adapter = _Adapter()
        strategy.prepare(adapter)
        plugin = _plugin()
        optimizer = _optimizer(plugin, adapter)
        features, mask = _rank_case(rank)
        batch = _batch(
            rank=rank,
            world_size=2,
            features=features,
            mask=mask,
        )
        if rank == 1:

            def fail_backward_preparation(*_args) -> float:
                raise RuntimeError(
                    "injected rank-one final-slot backward preparation failure"
                )

            plugin.update_engine._active_logprob_delta_max = (
                fail_backward_preparation
            )

        started = time.monotonic()
        caught: DistributedFailureError | None = None
        try:
            plugin.step(
                batch=batch,
                rewards=_rewards(batch),
                optimizer=optimizer,
                scaler=None,
                context=batch.context,
                strategy=strategy,
            )
        except DistributedFailureError as exc:
            caught = exc
        elapsed = time.monotonic() - started
        if caught is None:
            raise AssertionError(
                "final-slot backward preparation failure unexpectedly succeeded"
            )
        output.put(
            (
                rank,
                {
                    "error": str(caught),
                    "elapsed": elapsed,
                    "parameter": float(adapter.train_module.delta.detach()),
                    "optimizer_state": optimizer.state_dict()["state"],
                },
            )
        )
    except BaseException as exc:
        output.put((rank, (type(exc).__name__, str(exc))))
    finally:
        if strategy is not None:
            strategy.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _single_global_case() -> dict:
    adapter = _Adapter()
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
    strategy.prepare(adapter)
    plugin = _plugin()
    optimizer = _optimizer(plugin, adapter)
    features = torch.cat([_rank_case(0)[0], _rank_case(1)[0]])
    mask = torch.cat([_rank_case(0)[1], _rank_case(1)[1]])
    batch = _batch(
        rank=0,
        world_size=1,
        features=features,
        mask=mask,
    ).replace(
        prompts=("rank-0", "rank-0", "rank-1", "rank-1"),
        metadata=({}, {}, {}, {}),
        sample_id=(
            "rank-0-sample-0",
            "rank-0-sample-1",
            "rank-1-sample-0",
            "rank-1-sample-1",
        ),
        prompt_id=(
            "rank-0-prompt",
            "rank-0-prompt",
            "rank-1-prompt",
            "rank-1-prompt",
        ),
        group_id=(
            "rank-0-group",
            "rank-0-group",
            "rank-1-group",
            "rank-1-group",
        ),
    )
    values = torch.tensor([1.0, 3.0, 1.0, 3.0])
    rewards = RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(4, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": ({}, {}, {}, {})},
    )
    try:
        result = plugin.step(
            batch=batch,
            rewards=rewards,
            optimizer=optimizer,
            scaler=None,
            context=batch.context,
            strategy=strategy,
        )
        return {
            "loss": result.loss,
            "policy_loss": result.policy_loss,
            "reference_kl": result.reference_kl,
            "approx_kl": result.approx_kl,
            "clipfrac": result.clipfrac,
            "active_transition_count": result.active_transition_count,
            "diagnostics": dict(result.diagnostics),
            "gradient": float(adapter.train_module.delta.grad.detach()),
            "parameter": float(adapter.train_module.delta.detach()),
        }
    finally:
        strategy.close()


@pytest.mark.distributed
def test_gloo_fixed_batch_matches_single_process() -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_worker, args=(rank, port, output))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    rows = {}
    try:
        for _ in processes:
            rank, payload = output.get(timeout=30)
            rows[rank] = payload
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for Gloo workers") from exc
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
    assert all(isinstance(rows[rank], dict) for rank in range(2)), rows

    expected = _single_global_case()
    assert {
        "update/gradient_norm_pre_clip",
        "update/gradient_norm_post_clip",
    }.issubset(expected["diagnostics"])
    for payload in rows.values():
        assert payload["active_transition_count"] == 5
        for name in (
            "loss",
            "policy_loss",
            "reference_kl",
            "approx_kl",
            "clipfrac",
            "gradient",
            "parameter",
        ):
            assert payload[name] == pytest.approx(expected[name], abs=1e-6)
        assert payload["diagnostics"] == pytest.approx(
            expected["diagnostics"],
            abs=1e-6,
        )


@pytest.mark.parametrize(
    ("mode", "expected_calls", "message"),
    (
        ("snapshot", 0, "optimizer_snapshot"),
        ("operation", 1, "rank-one operation failure"),
    ),
)
@pytest.mark.distributed
def test_gloo_atomic_failure_prevents_or_rolls_back_all_mutation(
    mode: str,
    expected_calls: int,
    message: str,
) -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_atomic_failure_worker,
            args=(rank, port, mode, output),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    rows = {}
    try:
        for _ in processes:
            rank, payload = output.get(timeout=30)
            rows[rank] = payload
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for atomic workers") from exc
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
    assert all(isinstance(rows[rank], dict) for rank in range(2)), rows
    for payload in rows.values():
        assert message in payload["error"]
        assert payload["operation_calls"] == expected_calls
        assert payload["parameter"] == pytest.approx(0.05)
        assert payload["optimizer_state"] == {}
        assert payload["scaler_value"] == 7


@pytest.mark.distributed
def test_gloo_final_slot_backward_prepare_failure_is_synchronized() -> None:
    """Gate rank-local preparation before either rank enters DDP backward."""

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_final_backward_prepare_failure_worker,
            args=(rank, port, output),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    rows = {}
    try:
        for _ in processes:
            rank, payload = output.get(timeout=30)
            rows[rank] = payload
    except queue.Empty as exc:
        raise AssertionError(
            "timed out waiting for backward preparation workers"
        ) from exc
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert all(isinstance(rows[rank], dict) for rank in range(2)), rows
    for payload in rows.values():
        assert (
            "injected rank-one final-slot backward preparation failure"
            in payload["error"]
        )
        assert payload["elapsed"] < 10.0
        assert payload["parameter"] == pytest.approx(0.05)
        assert payload["optimizer_state"] == {}


@pytest.mark.distributed
def test_nccl_fixed_batch_matches_single_process() -> None:
    """Run the fixed global batch through the real two-rank NCCL update path."""

    expected = _single_global_case()
    strategy = _nccl_strategy_from_torchrun()
    rank = strategy.rank
    sentinel = None
    try:
        entered = strategy.gather_object(rank, dst=0)
        adapter = _Adapter(strategy.device)
        strategy.prepare(adapter)
        plugin = _plugin()
        optimizer = _optimizer(plugin, adapter)
        features, mask = _rank_case(rank)
        batch = _batch(
            rank=rank,
            world_size=2,
            features=features.to(strategy.device),
            mask=mask.to(strategy.device),
        )
        result = plugin.step(
            batch=batch,
            rewards=_rewards(batch),
            optimizer=optimizer,
            scaler=None,
            context=batch.context,
            strategy=strategy,
        )

        comparison_error = None
        try:
            assert result.active_transition_count == 5
            actual = {
                "loss": result.loss,
                "policy_loss": result.policy_loss,
                "reference_kl": result.reference_kl,
                "approx_kl": result.approx_kl,
                "clipfrac": result.clipfrac,
                "gradient": float(
                    adapter.train_module.delta.grad.detach()
                ),
                "parameter": float(adapter.train_module.delta.detach()),
            }
            for name, value in actual.items():
                assert value == pytest.approx(expected[name], abs=1e-6)
            assert dict(result.diagnostics) == pytest.approx(
                expected["diagnostics"],
                abs=1e-6,
            )
        except BaseException as exc:
            comparison_error = exc
        strategy.failure_gate(
            "test.nccl_fixed_batch.compare",
            comparison_error,
        )

        synchronize_error = None
        try:
            torch.cuda.synchronize(strategy.device)
        except BaseException as exc:
            synchronize_error = exc
        strategy.failure_gate(
            "test.nccl_fixed_batch.synchronize",
            synchronize_error,
        )
        exited = strategy.gather_object(rank, dst=0)
        if strategy.is_main_process:
            assert entered == [0, 1]
            assert exited == [0, 1]
            sentinel = {
                "nodeid": (
                    "tests/test_distributed_update.py::"
                    "test_nccl_fixed_batch_matches_single_process"
                ),
                "world_size": 2,
                "ranks_entered": entered,
                "ranks_exited": exited,
                "marker_advanced": False,
                "passed": True,
            }
    finally:
        strategy.close()

    if sentinel is not None:
        print(
            "VISUALRL_NCCL_RESULT="
            + json.dumps(
                sentinel,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )


@pytest.mark.distributed
def test_nccl_one_rank_update_failure_synchronizes() -> None:
    """Synchronize and roll back a rank-one failure after optimizer mutation."""

    strategy = _nccl_strategy_from_torchrun()
    rank = strategy.rank
    sentinel = None
    try:
        entered = strategy.gather_object(rank, dst=0)
        parameter = torch.nn.Parameter(
            torch.tensor(0.05, device=strategy.device)
        )
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=1e-2,
            weight_decay=0.0,
        )
        scaler = _FakeScaler()
        operation_calls = 0

        def operation() -> None:
            nonlocal operation_calls
            operation_calls += 1
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scaler.value += 1
            if rank == 1:
                raise RuntimeError("injected rank-one NCCL update failure")

        caught: DistributedFailureError | None = None
        try:
            strategy.atomic_optimizer_step(
                operation,
                parameters=(parameter,),
                optimizer=optimizer,
                scaler=scaler,
            )
        except DistributedFailureError as exc:
            caught = exc

        validation_error = None
        try:
            assert caught is not None
            assert "injected rank-one NCCL update failure" in str(caught)
            assert operation_calls == 1
            assert float(parameter.detach().cpu()) == pytest.approx(0.05)
            assert optimizer.state_dict()["state"] == {}
            assert scaler.value == 7
        except BaseException as exc:
            validation_error = exc
        strategy.failure_gate(
            "test.nccl_update_failure.validate",
            validation_error,
        )

        exited = strategy.gather_object(rank, dst=0)
        if strategy.is_main_process:
            assert entered == [0, 1]
            assert exited == [0, 1]
            sentinel = {
                "nodeid": (
                    "tests/test_distributed_update.py::"
                    "test_nccl_one_rank_update_failure_synchronizes"
                ),
                "world_size": 2,
                "ranks_entered": entered,
                "ranks_exited": exited,
                "marker_advanced": False,
                "rollback_verified": True,
                "passed": True,
            }
    finally:
        strategy.close()

    if sentinel is not None:
        print(
            "VISUALRL_NCCL_RESULT="
            + json.dumps(
                sentinel,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
