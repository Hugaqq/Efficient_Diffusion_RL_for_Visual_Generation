"""CPU-only contract tests for the C12 distributed strategy core."""

from __future__ import annotations

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
    PolicyRecomputeStats,
    RolloutBatch,
    StepContext,
    ValidatedRuntimeEnv,
)
from visual_rl.distributed import (
    DDPStrategy,
    DistributedContext,
    DistributedFailureError,
    SingleProcessStrategy,
    build_strategy,
)
import visual_rl.distributed as distributed_module


class _LinearAdapter:
    def __init__(self) -> None:
        self.transformer = torch.nn.Linear(1, 1, bias=False)

    @property
    def train_module(self) -> torch.nn.Module:
        return self.transformer

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        if require_reference:
            raise ValueError("test adapter does not support reference statistics")
        return PolicyRecomputeStats(
            new_log_probs=self.transformer(
                batch.recompute_payload["features"]
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return self.train_module.state_dict()


def _policy_batch(
    value: float,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> RolloutBatch:
    features = torch.tensor([[value]], dtype=torch.float32)
    return RolloutBatch(
        prompts=(f"prompt-{rank}",),
        metadata=({},),
        media=torch.zeros(1, 1, 1, 1),
        latents=torch.zeros(1, 1, 1),
        next_latents=torch.ones(1, 1, 1),
        timesteps=torch.zeros(1, 1),
        old_log_probs=torch.zeros(1, 1),
        transition_mask=torch.ones(1, 1, dtype=torch.bool),
        sample_id=(f"sample-{rank}",),
        prompt_id=(f"prompt-id-{rank}",),
        group_id=(f"group-{rank}",),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(
            step=0,
            seed=17,
            rank=rank,
            world_size=world_size,
        ),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={"features": features},
        artifact_metadata={},
    )


def _single_context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        backend=None,
    )


def _single_strategy() -> SingleProcessStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="single",
            device="cpu",
            timeout_s=30.0,
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
            raw_launch_env=FrozenMapping({}),
        ),
    )
    assert isinstance(strategy, SingleProcessStrategy)
    return strategy


def _ddp_strategy(rank: int, port: int, *, timeout_s: float) -> DDPStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="ddp",
            device="cpu",
            timeout_s=timeout_s,
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


def test_build_strategy_is_the_only_setup_and_env_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_calls = 0
    original_setup = SingleProcessStrategy.setup

    def counted_setup(self):
        nonlocal setup_calls
        setup_calls += 1
        return original_setup(self)

    monkeypatch.setattr(SingleProcessStrategy, "setup", counted_setup)
    strategy = _single_strategy()
    assert setup_calls == 1
    assert strategy.context.backend is None
    assert strategy.context.device == torch.device("cpu")

    monkeypatch.setattr(
        strategy,
        "setup",
        lambda: (_ for _ in ()).throw(
            AssertionError("prepare() called setup() a second time")
        ),
    )
    strategy.prepare(_LinearAdapter())
    assert setup_calls == 1


def test_build_strategy_rejects_mode_drift_and_external_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ValidatedRuntimeEnv(
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
    ddp_config = SimpleNamespace(
        mode="ddp",
        device="cpu",
        timeout_s=30.0,
        max_snapshot_tensor_bytes=1024,
    )
    with pytest.raises(ValueError, match="mode does not match"):
        build_strategy(ddp_config, env)

    single_config = SimpleNamespace(
        mode="single",
        device="cpu",
        timeout_s=30.0,
        max_snapshot_tensor_bytes=None,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="ownership"):
        build_strategy(single_config, env)


def test_single_process_context_and_strategy_have_no_distributed_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _single_context()
    adapter = _LinearAdapter()
    original_module = adapter.transformer
    original_keys = set(adapter.state_dict())

    def unexpected_init(*_args, **_kwargs):
        raise AssertionError("single-process setup initialized a process group")

    monkeypatch.setattr(dist, "init_process_group", unexpected_init)
    strategy = _single_strategy()
    assert strategy.prepare(adapter) is adapter
    batch = _policy_batch(2.0)
    actual = strategy.recompute_policy_stats(batch)

    assert context.rank == context.local_rank == 0
    assert context.world_size == 1
    assert context.device == torch.device("cpu")
    assert context.backend is None
    assert not context.is_distributed
    assert context.is_main_process
    assert context.step_seed(100, 3) == 103
    assert adapter.transformer is original_module
    assert set(adapter.state_dict()) == original_keys == {"weight"}
    torch.testing.assert_close(
        actual.new_log_probs,
        adapter.recompute_policy_stats(batch).new_log_probs,
    )
    assert strategy.reduce_weighted_mean(torch.tensor(3.0), 7) == 3.0
    assert strategy.reduce_weighted_scalars({"loss": 2, "reward": 4.5}, 7) == {
        "loss": 2.0,
        "reward": 4.5,
    }
    assert strategy.reduce_metrics(
        {
            "loss": 2.0,
            "sample_count": 2,
            "step_time_s": 0.25,
            "grad_abs_max": 3.0,
            "all_finite": True,
        },
        2,
        reward_values=[1.0, 3.0],
    ) == pytest.approx(
        {
            "loss": 2.0,
            "sample_count": 2.0,
            "step_time_s": 0.25,
            "grad_abs_max": 3.0,
            "all_finite": True,
            "reward_mean": 2.0,
            "reward_std": 1.0,
        }
    )
    broadcast_value = {"config": "single"}
    assert strategy.broadcast_object(broadcast_value) is broadcast_value
    assert strategy.gather_object({"rank": 0}) == [{"rank": 0}]
    assert strategy.synchronize_failure(False) is False
    with pytest.raises(DistributedFailureError, match="rank 0.*local failure"):
        strategy.synchronize_failure(RuntimeError("local failure"))

    strategy.barrier()
    strategy.close()
    strategy.close()
    assert strategy.closed
    with pytest.raises(RuntimeError, match="closed"):
        strategy.recompute_policy_stats(_policy_batch(1.0))
    with pytest.raises(RuntimeError, match="closed"):
        strategy.reduce_weighted_mean(1.0, 1)


@pytest.mark.parametrize("name,value", [("base_seed", -1), ("step", -1)])
def test_step_seed_rejects_negative_values(name: str, value: int) -> None:
    context = _single_context()
    arguments = {"base_seed": 1, "step": 1, name: value}
    with pytest.raises(ValueError, match="non-negative"):
        context.step_seed(**arguments)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_step_seed_rejects_non_integer_values(value: Any) -> None:
    context = _single_context()
    with pytest.raises(TypeError, match="integer"):
        context.step_seed(value, 0)


def test_single_metric_reduction_rejects_non_finite_values() -> None:
    strategy = _single_strategy()
    with pytest.raises(ValueError, match="finite"):
        strategy.reduce_metrics({"loss": float("nan")}, 1)
    with pytest.raises(ValueError, match="reward_values"):
        strategy.reduce_metrics({"reward_std": 0.0}, 1)


def test_single_process_optimizer_step_has_no_snapshot_overhead(monkeypatch) -> None:
    strategy = _single_strategy()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def unexpected_snapshot(*_args, **_kwargs):
        raise AssertionError("single-process optimizer step captured a snapshot")

    monkeypatch.setattr(
        distributed_module._OptimizerStepSnapshot,
        "capture",
        unexpected_snapshot,
    )
    validated: list[str] = []
    result = strategy.atomic_optimizer_step(
        lambda: "updated",
        parameters=[parameter],
        optimizer=optimizer,
        validate_result=lambda value: validated.append(value),
    )

    assert result == "updated"
    assert validated == ["updated"]

    operation_ran = False

    def record_operation():
        nonlocal operation_ran
        operation_ran = True

    with pytest.raises(TypeError, match="validator"):
        strategy.atomic_optimizer_step(
            record_operation,
            parameters=[parameter],
            optimizer=optimizer,
            validate_result=True,
        )
    assert operation_ran is False


def test_local_metric_contract_is_canonical_and_collective_free() -> None:
    strategy = _single_strategy()

    contract = strategy.metric_contract(
        {
            "step_time_s": 0.1,
            "loss": 2.0,
            "sample_count": 2,
            "all_finite": True,
        },
        2,
        reward_values=torch.tensor([1.0, 3.0]),
    )

    assert contract == (
        ("all_finite", "loss", "sample_count", "step_time_s"),
        (
            ("all_finite", "bool_and"),
            ("loss", "mean"),
            ("sample_count", "sum"),
            ("step_time_s", "max"),
        ),
        True,
    )


def test_optimizer_snapshot_limit_fails_before_allocating_parameter_copies() -> None:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    with pytest.raises(RuntimeError, match="exceeding the per-rank limit") as raised:
        distributed_module._OptimizerStepSnapshot.capture(
            [parameter],
            optimizer,
            None,
            None,
            15,
        )

    assert raised.value.metrics == {
        "parameter_count": 1,
        "parameter_tensor_bytes": 16,
        "optimizer_state_tensor_bytes": 0,
        "scaler_state_tensor_bytes": 0,
        "stateful_state_tensor_bytes": 0,
        "snapshot_limit_tensor_bytes": 15,
        "snapshot_limit_enabled": 1,
        "total_tensor_bytes": 16,
    }


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
def test_ddp_snapshot_limit_configuration_fails_closed(value: Any) -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        backend="gloo",
    )
    expected = TypeError if isinstance(value, (bool, float, str)) else ValueError
    with pytest.raises(expected, match="max_snapshot_tensor_bytes"):
        DDPStrategy(context, max_snapshot_tensor_bytes=value)

    unbounded = DDPStrategy(context, max_snapshot_tensor_bytes=None)
    assert unbounded.max_snapshot_tensor_bytes is None
    assert unbounded.last_atomic_snapshot_metrics is None


def test_optimizer_snapshot_restore_attempts_plugin_after_optimizer_failure() -> None:
    class Stateful:
        def __init__(self) -> None:
            self.value = 7
            self.load_calls = 0

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            self.load_calls += 1
            self.value = int(state["value"])

    class Scaler:
        def __init__(self) -> None:
            self.value = 16
            self.load_calls = 0
            self._per_optimizer_states = {"dirty": True}

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            self.load_calls += 1
            self.value = int(state["value"])

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.tensor(1.0)
    optimizer.step()
    stateful = Stateful()
    scaler = Scaler()
    snapshot = distributed_module._OptimizerStepSnapshot.capture(
        [parameter],
        optimizer,
        scaler,
        stateful,
        1024,
    )
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(10)
    stateful.value = 99
    scaler.value = 128
    optimizer.load_state_dict = lambda _state: (_ for _ in ()).throw(
        RuntimeError("optimizer restore injected")
    )

    with pytest.raises(RuntimeError, match="optimizer restore injected"):
        snapshot.restore(optimizer, scaler)

    torch.testing.assert_close(parameter, before)
    assert stateful.value == 7
    assert stateful.load_calls == 1
    assert scaler.value == 16
    assert scaler.load_calls == 1
    assert scaler._per_optimizer_states == {}
    assert snapshot.metrics["optimizer_state_tensor_bytes"] > 0


@pytest.mark.parametrize("operation", ["broadcast", "gather"])
def test_single_process_object_collectives_reject_invalid_roots(
    operation: str,
) -> None:
    strategy = _single_strategy()

    with pytest.raises(ValueError, match="rank in the process group"):
        if operation == "broadcast":
            strategy.broadcast_object("value", src=1)
        else:
            strategy.gather_object("value", dst=1)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _ddp_worker(rank: int, port: int, results: Any) -> None:
    strategy: DDPStrategy | None = None
    try:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
        strategy = _ddp_strategy(rank, port, timeout_s=6)
        context = strategy.context
        adapter = _LinearAdapter()
        with torch.no_grad():
            adapter.transformer.weight.fill_(rank + 1.0)
        original_module = adapter.transformer
        checkpoint_keys = set(adapter.state_dict())
        strategy.prepare(adapter)

        optimizer = torch.optim.SGD(adapter.train_module.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        strategy.recompute_policy_stats(
            _policy_batch(float(rank + 1), rank=rank, world_size=2)
        ).new_log_probs.sum().backward()
        optimizer.step()

        local_mean = 2.0 if rank == 0 else 10.0
        local_count = 1 if rank == 0 else 3
        weighted_mean = strategy.reduce_weighted_mean(local_mean, local_count)
        weighted_metrics = strategy.reduce_weighted_scalars(
            {"loss": local_mean, "reward": local_mean + 1.0},
            local_count,
        )
        reduced_metrics = strategy.reduce_metrics(
            {
                "loss": local_mean,
                "sample_count": local_count,
                "step_time_s": 0.25 if rank == 0 else 0.75,
                "grad_abs_max": 2.0 if rank == 0 else 5.0,
                "all_finite": rank == 0,
                "reward_mean": 1.0 if rank == 0 else 5.0,
                "reward_std": 0.0 if rank == 0 else 2.0**0.5,
            },
            local_count,
            reward_values=[1.0] if rank == 0 else [3.0, 5.0, 7.0],
        )
        broadcast = strategy.broadcast_object(
            {"run_id": "shared", "source_rank": rank} if rank == 0 else None
        )
        step_seed = context.step_seed(100, 3)
        try:
            strategy.reduce_metrics(
                {"loss": float("nan") if rank == 1 else 1.0},
                local_count,
            )
        except ValueError as error:
            non_finite_error = str(error)
        else:
            raise AssertionError("non-finite metrics were not rejected on every rank")

        class ExplodingMetrics(dict):
            def items(self):
                raise RuntimeError("injected metric mapping failure")

        try:
            strategy.reduce_metrics(
                ExplodingMetrics(loss=1.0) if rank == 1 else {"loss": 1.0},
                local_count,
            )
        except ValueError as error:
            metric_mapping_error = str(error)
        else:
            raise AssertionError("metric mapping failure was not synchronized")
        try:
            strategy.reduce_metrics(
                {"rank_zero_only": 1.0} if rank == 0 else {"loss": 1.0},
                local_count,
            )
        except ValueError as error:
            key_mismatch_error = str(error)
        else:
            raise AssertionError("metric key mismatch was not rejected")
        gathered = strategy.gather_object({"rank": context.rank})
        any_failed = strategy.synchronize_failure(rank == 1)
        try:
            failure = RuntimeError("rank-one failure") if rank == 1 else None
            strategy.synchronize_failure(failure)
        except DistributedFailureError as error:
            synchronized_error = str(error)
        else:
            raise AssertionError("failure was not synchronized to every rank")

        callable_parameter = torch.nn.Parameter(torch.tensor(1.0))
        callable_optimizer = torch.optim.SGD([callable_parameter], lr=0.1)
        try:
            strategy.atomic_optimizer_step(
                (lambda: None) if rank == 0 else None,
                parameters=[callable_parameter],
                optimizer=callable_optimizer,
            )
        except DistributedFailureError as error:
            callable_error = str(error)
        else:
            raise AssertionError("non-callable operation was not synchronized")

        validator_presence_operation_ran = False

        def record_validator_presence_operation() -> None:
            nonlocal validator_presence_operation_ran
            validator_presence_operation_ran = True

        try:
            strategy.atomic_optimizer_step(
                record_validator_presence_operation,
                parameters=[callable_parameter],
                optimizer=callable_optimizer,
                validate_result=(lambda _result: ("ok",)) if rank == 0 else None,
            )
        except DistributedFailureError as error:
            validator_presence_error = str(error)
        else:
            raise AssertionError("validator presence mismatch was not synchronized")

        validation_parameter = torch.nn.Parameter(torch.tensor(float(rank + 2)))
        validation_optimizer = torch.optim.SGD([validation_parameter], lr=0.1)
        validation_before = validation_parameter.detach().clone()

        def mutate_for_validation() -> dict[str, float]:
            with torch.no_grad():
                validation_parameter.add_(10.0)
            return {"loss": float(rank)}

        def reject_on_rank_one(_result: Any) -> tuple[str]:
            if rank == 1:
                raise ValueError("injected result validation failure")
            return ("loss",)

        try:
            strategy.atomic_optimizer_step(
                mutate_for_validation,
                parameters=[validation_parameter],
                optimizer=validation_optimizer,
                validate_result=reject_on_rank_one,
            )
        except DistributedFailureError as error:
            result_validation_error = str(error)
        else:
            raise AssertionError("result validation failure was not synchronized")
        validation_failure_restored = torch.equal(
            validation_parameter.detach(), validation_before
        )

        def unpickleable_on_rank_one(_result: Any) -> Any:
            if rank == 1:
                return lambda: None
            return ("pickleable-contract",)

        try:
            strategy.atomic_optimizer_step(
                mutate_for_validation,
                parameters=[validation_parameter],
                optimizer=validation_optimizer,
                validate_result=unpickleable_on_rank_one,
            )
        except DistributedFailureError as error:
            result_serialization_error = str(error)
        else:
            raise AssertionError("unpickleable result contract was not synchronized")
        result_serialization_restored = torch.equal(
            validation_parameter.detach(), validation_before
        )

        try:
            strategy.atomic_optimizer_step(
                mutate_for_validation,
                parameters=[validation_parameter],
                optimizer=validation_optimizer,
                validate_result=lambda _result: ("rank-contract", rank),
            )
        except DistributedFailureError as error:
            result_contract_error = str(error)
        else:
            raise AssertionError("result contract mismatch was not synchronized")
        result_contract_restored = torch.equal(
            validation_parameter.detach(), validation_before
        )

        try:
            strategy.gather_object((lambda: None) if rank == 1 else {"rank": rank})
        except DistributedFailureError as error:
            gather_serialization_error = str(error)
        else:
            raise AssertionError("gather serialization failure was not synchronized")

        try:
            strategy.broadcast_object((lambda: None) if rank == 0 else None)
        except DistributedFailureError as error:
            broadcast_serialization_error = str(error)
        else:
            raise AssertionError("broadcast serialization failure was not synchronized")

        snapshot_parameter = torch.nn.Parameter(torch.tensor(float(rank + 3)))
        snapshot_optimizer = torch.optim.SGD([snapshot_parameter], lr=0.1)
        original_state_dict = snapshot_optimizer.state_dict
        operation_ran = False
        if rank == 1:
            snapshot_optimizer.state_dict = lambda: (_ for _ in ()).throw(
                RuntimeError("injected snapshot failure")
            )

        def record_operation() -> None:
            nonlocal operation_ran
            operation_ran = True

        try:
            strategy.atomic_optimizer_step(
                record_operation,
                parameters=[snapshot_parameter],
                optimizer=snapshot_optimizer,
            )
        except DistributedFailureError as error:
            snapshot_error = str(error)
        else:
            raise AssertionError("snapshot failure was not synchronized")
        snapshot_optimizer.state_dict = original_state_dict

        restore_parameter = torch.nn.Parameter(torch.tensor(float(rank + 5)))
        restore_optimizer = torch.optim.SGD(
            [restore_parameter],
            lr=0.1,
            momentum=0.9,
        )
        restore_parameter.grad = torch.tensor(1.0)
        restore_optimizer.step()
        restore_before = restore_parameter.detach().clone()
        original_load_state_dict = restore_optimizer.load_state_dict
        if rank == 1:
            restore_optimizer.load_state_dict = lambda _state: (_ for _ in ()).throw(
                RuntimeError("injected restore failure")
            )

        def mutate_then_fail() -> None:
            with torch.no_grad():
                restore_parameter.add_(10.0)
            if rank == 1:
                raise RuntimeError("trigger rollback")

        try:
            strategy.atomic_optimizer_step(
                mutate_then_fail,
                parameters=[restore_parameter],
                optimizer=restore_optimizer,
            )
        except DistributedFailureError as error:
            restore_error = str(error)
        else:
            raise AssertionError("restore failure was not synchronized")
        snapshot_metrics = strategy.last_atomic_snapshot_metrics
        if snapshot_metrics is None:
            raise AssertionError("atomic optimizer snapshot metrics were not recorded")
        restore_optimizer.load_state_dict = original_load_state_dict
        strategy.barrier()

        result = {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
            "parameter": float(adapter.transformer.weight.detach().item()),
            "weighted_mean": weighted_mean,
            "weighted_metrics": weighted_metrics,
            "reduced_metrics": reduced_metrics,
            "broadcast": broadcast,
            "step_seed": step_seed,
            "non_finite_error": non_finite_error,
            "metric_mapping_error": metric_mapping_error,
            "key_mismatch_error": key_mismatch_error,
            "gathered": gathered,
            "any_failed": any_failed,
            "synchronized_error": synchronized_error,
            "callable_error": callable_error,
            "validator_presence_error": validator_presence_error,
            "validator_presence_operation_ran": validator_presence_operation_ran,
            "result_validation_error": result_validation_error,
            "validation_failure_restored": validation_failure_restored,
            "result_serialization_error": result_serialization_error,
            "result_serialization_restored": result_serialization_restored,
            "result_contract_error": result_contract_error,
            "result_contract_restored": result_contract_restored,
            "gather_serialization_error": gather_serialization_error,
            "broadcast_serialization_error": broadcast_serialization_error,
            "snapshot_error": snapshot_error,
            "snapshot_operation_ran": operation_ran,
            "restore_error": restore_error,
            "snapshot_metrics": snapshot_metrics,
            "restore_parameter_restored": torch.equal(
                restore_parameter.detach(), restore_before
            ),
            "module_stable": adapter.transformer is original_module,
            "checkpoint_keys": checkpoint_keys,
        }
        strategy.close()
        strategy.close()
        result["closed"] = strategy.closed
        result["process_group_closed"] = not dist.is_initialized()
        results.put(("ok", result))
    except BaseException:
        if strategy is not None:
            strategy.close()
        results.put(("error", traceback.format_exc()))


def _object_collective_consensus_worker(rank: int, port: int, results: Any) -> None:
    strategy: DDPStrategy | None = None
    try:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
        strategy = _ddp_strategy(rank, port, timeout_s=6)

        errors: dict[str, str] = {}

        def expect_consensus_failure(name: str, operation: Any) -> None:
            try:
                operation()
            except DistributedFailureError as error:
                errors[name] = str(error)
            else:
                raise AssertionError(f"{name} did not fail on rank {rank}")

        expect_consensus_failure(
            "broadcast_root_mismatch",
            lambda: strategy.broadcast_object({"rank": rank}, src=rank),
        )
        expect_consensus_failure(
            "broadcast_invalid_root",
            lambda: strategy.broadcast_object(
                {"rank": rank},
                src=2 if rank == 1 else 0,
            ),
        )
        expect_consensus_failure(
            "gather_root_mismatch",
            lambda: strategy.gather_object({"rank": rank}, dst=rank),
        )
        expect_consensus_failure(
            "gather_invalid_root",
            lambda: strategy.gather_object(
                {"rank": rank},
                dst=-1 if rank == 1 else 0,
            ),
        )

        def mismatched_operation() -> Any:
            if rank == 0:
                return strategy.broadcast_object({"rank": rank}, src=0)
            return strategy.gather_object({"rank": rank}, dst=0)

        expect_consensus_failure("operation_mismatch", mismatched_operation)
        strategy.barrier()
        strategy.close()
        results.put(("ok", {"rank": rank, "errors": errors}))
    except BaseException:
        if strategy is not None:
            strategy.close()
        results.put(("error", traceback.format_exc()))


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo distributed backend is unavailable",
)
@pytest.mark.distributed
def test_ddp_two_rank_cpu_smoke() -> None:
    started = time.monotonic()
    deadline = started + 6.0
    process_context: SpawnContext = mp.get_context("spawn")
    results = process_context.Queue()
    port = _free_loopback_port()
    processes = [
        process_context.Process(target=_ddp_worker, args=(rank, port, results))
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
            pytest.fail("2-rank gloo smoke timed out")
        assert process.exitcode == 0

    received = []
    for _ in range(2):
        try:
            status, payload = results.get(timeout=1)
        except queue.Empty:
            pytest.fail("DDP worker exited without returning a result")
        assert status == "ok", payload
        received.append(payload)
    by_rank = {item["rank"]: item for item in received}

    assert set(by_rank) == {0, 1}
    for rank, result in by_rank.items():
        assert result["local_rank"] == rank
        assert result["world_size"] == 2
        assert result["parameter"] == pytest.approx(0.85)
        assert result["weighted_mean"] == pytest.approx(8.0)
        assert result["weighted_metrics"] == pytest.approx({"loss": 8.0, "reward": 9.0})
        assert result["reduced_metrics"] == pytest.approx(
            {
                "loss": 8.0,
                "sample_count": 4.0,
                "step_time_s": 0.75,
                "grad_abs_max": 5.0,
                "all_finite": False,
                "reward_mean": 4.0,
                "reward_std": 5.0**0.5,
            }
        )
        assert result["broadcast"] == {"run_id": "shared", "source_rank": 0}
        assert result["step_seed"] == 106 + rank
        assert "rank 1" in result["non_finite_error"]
        assert "finite" in result["non_finite_error"]
        assert "rank 1" in result["metric_mapping_error"]
        assert "metric mapping failure" in result["metric_mapping_error"]
        assert "match on every rank" in result["key_mismatch_error"]
        assert result["any_failed"] is True
        assert "rank 1" in result["synchronized_error"]
        assert "rank-one failure" in result["synchronized_error"]
        assert "rank 1" in result["callable_error"]
        assert "operation must be callable" in result["callable_error"]
        assert "validator presence" in result["validator_presence_error"]
        assert "match on every rank" in result["validator_presence_error"]
        assert result["validator_presence_operation_ran"] is False
        assert "rank 1" in result["result_validation_error"]
        assert "injected result validation failure" in result["result_validation_error"]
        assert result["validation_failure_restored"] is True
        assert "rank 1" in result["result_serialization_error"]
        assert "pickle" in result["result_serialization_error"].lower()
        assert result["result_serialization_restored"] is True
        assert "result contracts must match" in result["result_contract_error"]
        assert result["result_contract_restored"] is True
        assert "rank 1" in result["gather_serialization_error"]
        assert "not serializable" in result["gather_serialization_error"]
        assert "rank 0" in result["broadcast_serialization_error"]
        assert "not serializable" in result["broadcast_serialization_error"]
        assert "rank 1" in result["snapshot_error"]
        assert "snapshot failure" in result["snapshot_error"]
        assert result["snapshot_operation_ran"] is False
        assert "rank 1" in result["restore_error"]
        assert "restore failure" in result["restore_error"]
        assert result["snapshot_metrics"]["parameter_count"] == 1
        assert result["snapshot_metrics"]["total_tensor_bytes"] > 0
        assert result["snapshot_metrics"]["capture_time_s"] >= 0.0
        assert result["snapshot_metrics"]["restore_time_s"] >= 0.0
        assert result["restore_parameter_restored"] is True
        assert result["module_stable"] is True
        assert result["checkpoint_keys"] == {"weight"}
        assert result["closed"] is True
        assert result["process_group_closed"] is True
    assert by_rank[0]["gathered"] == [{"rank": 0}, {"rank": 1}]
    assert by_rank[1]["gathered"] is None
    assert time.monotonic() - started < 6.0


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo distributed backend is unavailable",
)
@pytest.mark.distributed
def test_ddp_object_collective_contract_consensus() -> None:
    started = time.monotonic()
    deadline = started + 6.0
    process_context: SpawnContext = mp.get_context("spawn")
    results = process_context.Queue()
    port = _free_loopback_port()
    processes = [
        process_context.Process(
            target=_object_collective_consensus_worker,
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
            pytest.fail("2-rank object collective consensus test timed out")
        assert process.exitcode == 0

    received = []
    for _ in range(2):
        try:
            status, payload = results.get(timeout=1)
        except queue.Empty:
            pytest.fail("DDP worker exited without returning a result")
        assert status == "ok", payload
        received.append(payload)

    assert {item["rank"] for item in received} == {0, 1}
    for result in received:
        errors = result["errors"]
        assert set(errors) == {
            "broadcast_root_mismatch",
            "broadcast_invalid_root",
            "gather_root_mismatch",
            "gather_invalid_root",
            "operation_mismatch",
        }
        assert "broadcast(src=0)" in errors["broadcast_root_mismatch"]
        assert "broadcast(src=1)" in errors["broadcast_root_mismatch"]
        assert "rank 1" in errors["broadcast_invalid_root"]
        assert "src must identify a rank" in errors["broadcast_invalid_root"]
        assert "gather(dst=0)" in errors["gather_root_mismatch"]
        assert "gather(dst=1)" in errors["gather_root_mismatch"]
        assert "rank 1" in errors["gather_invalid_root"]
        assert "dst must identify a rank" in errors["gather_invalid_root"]
        assert "broadcast(src=0)" in errors["operation_mismatch"]
        assert "gather(dst=0)" in errors["operation_mismatch"]
    assert time.monotonic() - started < 6.0
