"""CPU-only coverage for bounded reward execution."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import torch

from visual_rl.configs.schema import RewardExecutorConfig, config_from_dict
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.feedback import (
    AsyncRewardExecutor,
    FeedbackProvider,
    RewardExecutionError,
    SyncRewardExecutor,
    build_reward_executor,
)
from visual_rl.feedback.clients import RewardProtocolError


def _context(policy_version: int = 4) -> StepContext:
    return StepContext(
        step=3,
        seed=17,
        epoch_tag=2,
        policy_version=policy_version,
    )


def _batch(
    sample_id: list[str] | None = None,
    *,
    context: StepContext | None = None,
) -> RolloutBatch:
    identities = sample_id or [f"sample-{index}" for index in range(4)]
    return RolloutBatch(
        prompts=[f"prompt-{index}" for index in range(len(identities))],
        metadata=[{"row": index} for index in range(len(identities))],
        media=torch.arange(len(identities), dtype=torch.float32)[:, None],
        sample_id=identities,
        prompt_id=[f"prompt-id-{index}" for index in range(len(identities))],
        group_id=[f"group-{index // 2}" for index in range(len(identities))],
        branch_id=list(range(len(identities))),
        context=context,
    )


def _reward(batch: RolloutBatch, *, metadata: dict[str, Any] | None = None):
    values = torch.tensor(
        [float(sample.rsplit("-", 1)[-1]) + 1.0 for sample in batch.sample_id]
    )
    weighted = {
        "quality": values * 0.5,
        "safety": values * 0.25,
    }
    return RewardBatch(
        raw={"quality": values, "safety": values + 10.0},
        weighted=weighted,
        weighted_total=weighted["quality"] + weighted["safety"],
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        metadata=dict(metadata or {}),
        sample_id=list(batch.sample_id),
    )


class _FunctionProvider(FeedbackProvider):
    supports_concurrent_score = True
    executor_retry_safe = True
    max_attempts_per_score = 1

    def __init__(self, function):
        self.function = function
        self.calls = 0
        self._lock = threading.Lock()

    def score(self, batch: RolloutBatch) -> RewardBatch:
        with self._lock:
            self.calls += 1
        return self.function(batch)


def test_async_out_of_order_shards_restore_original_order_and_metadata() -> None:
    context = _context()
    batch = _batch(context=context)
    completed: list[str] = []
    lock = threading.Lock()

    def score(shard: RolloutBatch) -> RewardBatch:
        index = int(shard.sample_id[0].rsplit("-", 1)[-1])
        time.sleep(0.008 * (3 - index))
        with lock:
            completed.extend(shard.sample_id)
        return _reward(
            shard,
            metadata={"provider": "delayed", "source_rows": [index]},
        )

    with AsyncRewardExecutor(
        _FunctionProvider(score),
        max_workers=4,
        max_in_flight=4,
        microbatch_size=1,
        timeout_s=0.5,
        submit_timeout_s=0.1,
    ) as executor:
        rewards = executor.score(batch, context)

    assert completed != batch.sample_id
    assert rewards.sample_id == batch.sample_id
    assert rewards.raw["quality"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert rewards.metadata["provider"] == "delayed"
    assert rewards.metadata["source_rows"] == [0, 1, 2, 3]
    assert rewards.metadata["_executor"]["mode"] == "async"
    assert rewards.metadata["_executor"]["shards"] == 4
    assert rewards.metadata["_executor"]["attempts"] == 4
    metrics = rewards.metadata["_executor"]
    assert metrics["pending"] == metrics["running"] == 0
    assert metrics["still_running"] == metrics["cancelled"] == 0
    assert metrics["timeout_scope"] == "soft_collection"
    assert metrics["hard_cancel_supported"] == 0
    assert metrics["provider_concurrency_limit"] == 4
    assert metrics["provider_attempt_budget_per_shard"] == 1
    assert metrics["provider_attempt_budget_known"] == 1
    assert metrics["provider_queue_wait_count"] == 4
    assert metrics["queue_wait_count"] == 4
    assert metrics["service_latency_count"] == 4
    for prefix in ("queue_wait", "service_latency"):
        assert metrics[f"{prefix}_sum_s"] >= 0.0
        assert metrics[f"{prefix}_max_s"] >= metrics[f"{prefix}_p95_s"]
        assert metrics[f"{prefix}_p95_s"] >= metrics[f"{prefix}_p50_s"]
    assert metrics["first_failure_count"] == 0
    assert metrics["first_failure_rate"] == 0.0
    assert metrics["retry_failure_count"] == 0
    assert metrics["retry_failure_rate"] == 0.0
    assert metrics["final_failure_count"] == 0
    assert metrics["final_failure_rate"] == 0.0


@pytest.mark.parametrize("failure", ["keys", "total", "mask", "finite"])
def test_executor_applies_complete_reward_validation(failure: str) -> None:
    context = _context()
    batch = _batch(context=context)

    def score(shard: RolloutBatch) -> RewardBatch:
        rewards = _reward(shard)
        if failure == "keys":
            rewards.raw = {"different": torch.ones(shard.batch_size)}
        elif failure == "total":
            rewards.weighted_total = torch.zeros(shard.batch_size)
        elif failure == "mask":
            rewards.valid_mask = torch.ones(shard.batch_size)
        else:
            rewards.raw["quality"][0] = float("nan")
        return rewards

    provider = _FunctionProvider(score)
    with SyncRewardExecutor(provider) as executor:
        with pytest.raises(RewardExecutionError) as exc_info:
            executor.score(batch, context)

    assert provider.calls == 1
    assert isinstance(exc_info.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize(
    "error",
    [TypeError("bad provider call"), RewardProtocolError("bad wire payload")],
)
def test_contract_failures_are_not_retried(error: Exception) -> None:
    context = _context()
    batch = _batch(["sample-0"], context=context)

    def fail(_batch: RolloutBatch) -> RewardBatch:
        raise error

    provider = _FunctionProvider(fail)
    with AsyncRewardExecutor(
        provider,
        max_workers=1,
        microbatch_size=1,
        max_retries=3,
        timeout_s=0.2,
    ) as executor:
        with pytest.raises(RewardExecutionError) as exc_info:
            executor.score(batch, context)

    assert provider.calls == 1
    assert exc_info.value.__cause__ is error
    assert exc_info.value.metrics["attempts"] == 1
    assert exc_info.value.metrics["retries"] == 0
    assert exc_info.value.metrics["first_failure_count"] == 1
    assert exc_info.value.metrics["first_failure_rate"] == 1.0
    assert exc_info.value.metrics["final_failure_count"] == 1
    assert exc_info.value.metrics["final_failure_rate"] == 1.0
    assert exc_info.value.metrics["queue_wait_count"] == 1
    assert exc_info.value.metrics["service_latency_count"] == 1


def test_invalid_collect_does_not_consume_handle_and_success_is_idempotent() -> None:
    context = _context()
    batch = _batch(context=context)
    provider = _FunctionProvider(_reward)

    with SyncRewardExecutor(provider) as executor:
        handle = executor.submit(batch, context)
        stale = _context(policy_version=context.policy_version + 1)
        with pytest.raises(RewardExecutionError, match="stale.*policy_version"):
            executor.collect(handle, batch, stale)

        equivalent_batch = batch.replace()
        with pytest.raises(RewardExecutionError, match="original RolloutBatch"):
            executor.collect(handle, equivalent_batch, context)

        first = executor.collect(handle, batch, context)
        second = executor.collect(handle, batch, context)

    assert provider.calls == 1
    assert first is second


def test_terminal_failure_collect_is_idempotent_without_provider_replay() -> None:
    context = _context()
    batch = _batch(["sample-0"], context=context)
    failure = TypeError("invalid reward contract")
    provider = _FunctionProvider(lambda _batch: (_ for _ in ()).throw(failure))

    with AsyncRewardExecutor(
        provider,
        max_workers=1,
        microbatch_size=1,
        timeout_s=0.2,
    ) as executor:
        handle = executor.submit(batch, context)
        errors = []
        for _ in range(2):
            with pytest.raises(RewardExecutionError) as exc_info:
                executor.collect(handle, batch, context)
            errors.append(exc_info.value)

    assert provider.calls == 1
    assert handle._cancel_event.is_set()
    assert errors[0] is errors[1]
    assert errors[0].__cause__ is failure


def test_retry_metrics_include_runtime_failure_attempts() -> None:
    context = _context()
    batch = _batch(["sample-0"], context=context)

    def flaky(shard: RolloutBatch) -> RewardBatch:
        if provider.calls < 3:
            raise RuntimeError("temporary reward service failure")
        return _reward(shard, metadata={"provider": "flaky"})

    provider = _FunctionProvider(flaky)
    with AsyncRewardExecutor(
        provider,
        max_workers=1,
        microbatch_size=1,
        max_retries=2,
        timeout_s=0.2,
    ) as executor:
        rewards = executor.score(batch, context)

    metrics = rewards.metadata["_executor"]
    assert provider.calls == 3
    assert metrics["attempts"] == 3
    assert metrics["retries"] == 2
    assert metrics["provider_attempt_budget_per_shard"] == 3
    assert metrics["timeouts"] == 0
    assert metrics["first_failure_count"] == 1
    assert metrics["first_failure_rate"] == 1.0
    assert metrics["retry_failure_count"] == 1
    assert metrics["retry_failure_rate"] == 0.5
    assert metrics["final_failure_count"] == 0
    assert rewards.metadata["provider"] == "flaky"


def test_provider_timeout_retries_serially_without_duplicate_running_call() -> None:
    context = _context()
    batch = _batch(["sample-0"], context=context)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def score(shard: RolloutBatch) -> RewardBatch:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if provider.calls == 1:
                raise TimeoutError("provider deadline")
            return _reward(shard)
        finally:
            with lock:
                active -= 1

    provider = _FunctionProvider(score)
    with AsyncRewardExecutor(
        provider,
        max_workers=2,
        microbatch_size=1,
        max_retries=1,
        timeout_s=0.2,
    ) as executor:
        rewards = executor.score(batch, context)

    metrics = rewards.metadata["_executor"]
    assert provider.calls == 2
    assert max_active == 1
    assert metrics["attempts"] == 2
    assert metrics["retries"] == 1
    assert metrics["timeouts"] == 1
    assert metrics["first_failure_count"] == 1
    assert metrics["retry_failure_count"] == 0


def test_collection_timeout_keeps_running_permit_and_backpressure_bound() -> None:
    context = _context()
    blocked = _batch(["sample-0"], context=context)
    fast = _batch(["sample-1"], context=context)
    gate = threading.Event()
    started = threading.Event()
    finished = threading.Event()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def score(shard: RolloutBatch) -> RewardBatch:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if shard.sample_id == ["sample-0"]:
                started.set()
                gate.wait(timeout=0.5)
            return _reward(shard)
        finally:
            with lock:
                active -= 1
            finished.set()

    provider = _FunctionProvider(score)
    executor = AsyncRewardExecutor(
        provider,
        max_workers=2,
        max_in_flight=1,
        microbatch_size=1,
        timeout_s=0.03,
        submit_timeout_s=0.03,
    )
    try:
        handle = executor.submit(blocked, context)
        assert started.wait(timeout=0.2)
        with pytest.raises(RewardExecutionError, match="Timed out") as exc_info:
            executor.collect(handle, blocked, context)
        assert exc_info.value.metrics["cancelled"] == 0
        assert exc_info.value.metrics["timeouts"] == 1
        assert exc_info.value.metrics["pending"] == 0
        assert exc_info.value.metrics["running"] == 1
        assert exc_info.value.metrics["still_running"] == 1
        assert exc_info.value.metrics["timeout_scope"] == "soft_collection"
        assert exc_info.value.metrics["hard_cancel_supported"] == 0
        assert exc_info.value.metrics["timed_out_work_may_continue"] == 1

        with pytest.raises(
            RewardExecutionError, match="waiting for reward executor capacity"
        ) as capacity_error:
            executor.submit(fast, context)
        assert capacity_error.value.metrics["cancelled"] == 0
        assert provider.calls == 1
        assert max_active == 1

        gate.set()
        assert finished.wait(timeout=0.2)
        second = executor.submit(fast, context)
        rewards = executor.collect(second, fast, context)
        assert rewards.sample_id == ["sample-1"]
        assert max_active == 1
    finally:
        gate.set()
        executor.close()
        executor.close()


def test_collection_deadline_starts_at_submit_and_cancels_only_pending() -> None:
    context = _context()
    batch = _batch(["sample-0", "sample-1"], context=context)
    gate = threading.Event()
    started = threading.Event()

    def score(shard: RolloutBatch) -> RewardBatch:
        if shard.sample_id == ["sample-0"]:
            started.set()
            gate.wait(timeout=0.5)
        return _reward(shard)

    executor = AsyncRewardExecutor(
        _FunctionProvider(score),
        max_workers=1,
        max_in_flight=2,
        microbatch_size=1,
        timeout_s=0.04,
        submit_timeout_s=0.05,
    )
    try:
        handle = executor.submit(batch, context)
        assert started.wait(timeout=0.2)
        time.sleep(0.05)
        with pytest.raises(RewardExecutionError, match="Timed out") as exc_info:
            executor.collect(handle, batch, context)

        metrics = exc_info.value.metrics
        assert metrics["cancelled"] == 1
        assert metrics["pending"] == 0
        assert metrics["running"] == metrics["still_running"] == 1
        assert metrics["attempts"] == 1
        assert metrics["queue_wait_count"] == 1
        assert metrics["service_latency_count"] == 0
    finally:
        gate.set()
        executor.close()


def test_serial_provider_timeout_cancels_shards_waiting_for_provider_lock() -> None:
    context = _context()
    batch = _batch(context=context)
    gate = threading.Event()
    first_call_started = threading.Event()

    def score(shard: RolloutBatch) -> RewardBatch:
        if not first_call_started.is_set():
            first_call_started.set()
            gate.wait(timeout=0.5)
        return _reward(shard)

    provider = _FunctionProvider(score)
    provider.supports_concurrent_score = False
    executor = AsyncRewardExecutor(
        provider,
        max_workers=4,
        max_in_flight=4,
        microbatch_size=1,
        timeout_s=0.04,
        submit_timeout_s=0.05,
    )
    try:
        handle = executor.submit(batch, context)
        assert first_call_started.wait(timeout=0.2)

        with pytest.raises(RewardExecutionError, match="Timed out") as exc_info:
            executor.collect(handle, batch, context)

        metrics = exc_info.value.metrics
        assert handle._cancel_event.is_set()
        assert metrics["attempts"] == 1
        assert metrics["hard_cancel_supported"] == 0
        assert metrics["timed_out_work_may_continue"] == 1

        gate.set()
        deadline = time.monotonic() + 0.5
        while not all(task.future.done() for task in handle._tasks):
            if time.monotonic() >= deadline:
                pytest.fail("cooperatively cancelled reward shards did not finish")
            time.sleep(0.005)

        assert provider.calls == 1
    finally:
        gate.set()
        executor.close()


def test_close_is_non_blocking_while_provider_thread_is_still_running() -> None:
    context = _context()
    batch = _batch(["sample-0"], context=context)
    gate = threading.Event()
    started = threading.Event()

    def score(shard: RolloutBatch) -> RewardBatch:
        started.set()
        gate.wait(timeout=0.5)
        return _reward(shard)

    executor = AsyncRewardExecutor(
        _FunctionProvider(score),
        max_workers=1,
        timeout_s=0.1,
    )
    handle = executor.submit(batch, context)
    assert started.wait(timeout=0.2)
    before = time.monotonic()
    executor.close()
    elapsed = time.monotonic() - before
    gate.set()
    executor.close()

    assert elapsed < 0.05
    assert handle._cancel_event.is_set()


def test_close_cancels_every_active_handle_and_releases_task_tracking() -> None:
    context = _context()
    gate = threading.Event()
    both_started = threading.Event()
    started = 0
    lock = threading.Lock()

    def score(shard: RolloutBatch) -> RewardBatch:
        nonlocal started
        with lock:
            started += 1
            if started == 2:
                both_started.set()
        gate.wait(timeout=0.5)
        return _reward(shard)

    executor = AsyncRewardExecutor(
        _FunctionProvider(score),
        max_workers=2,
        max_in_flight=2,
        microbatch_size=1,
        timeout_s=0.2,
    )
    try:
        first = executor.submit(_batch(["sample-0"], context=context), context)
        second = executor.submit(_batch(["sample-1"], context=context), context)
        assert both_started.wait(timeout=0.2)

        executor.close()
        assert first._cancel_event.is_set()
        assert second._cancel_event.is_set()

        gate.set()
        deadline = time.monotonic() + 0.5
        while executor._active_cancellations:
            if time.monotonic() >= deadline:
                pytest.fail("completed reward tasks retained cancellation tracking")
            time.sleep(0.005)
        assert all(
            task.future.done() for handle in (first, second) for task in handle._tasks
        )
    finally:
        gate.set()
        executor.close()


def test_provider_calls_are_serialized_without_explicit_concurrency_capability() -> (
    None
):
    class SerialProvider(FeedbackProvider):
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def score(self, batch: RolloutBatch) -> RewardBatch:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.01)
                return _reward(batch)
            finally:
                with self.lock:
                    self.active -= 1

    provider = SerialProvider()
    with AsyncRewardExecutor(
        provider,
        max_workers=4,
        max_in_flight=4,
        microbatch_size=1,
        timeout_s=0.5,
    ) as executor:
        rewards = executor.score(_batch(context=_context()), _context())

    assert provider.max_active == 1
    metrics = rewards.metadata["_executor"]
    assert metrics["provider_concurrency_limit"] == 1
    assert metrics["provider_attempt_budget_per_shard"] == 0
    assert metrics["provider_attempt_budget_known"] == 0
    assert metrics["provider_queue_wait_count"] == 4
    assert metrics["provider_queue_wait_sum_s"] > 0.0


def test_outer_retries_require_explicit_provider_budget_and_idempotency() -> None:
    class OpaqueRetryProvider(FeedbackProvider):
        def score(self, batch: RolloutBatch) -> RewardBatch:
            return _reward(batch)

    with pytest.raises(ValueError, match="executor_retry_safe"):
        AsyncRewardExecutor(OpaqueRetryProvider(), max_retries=1)

    OpaqueRetryProvider.executor_retry_safe = True
    with pytest.raises(ValueError, match="max_attempts_per_score"):
        AsyncRewardExecutor(OpaqueRetryProvider(), max_retries=1)


@pytest.mark.parametrize(
    "provider_requires_hard_timeout,constructor_requires_hard_timeout",
    [(True, False), (False, True)],
)
def test_hard_timeout_requests_fail_before_starting_a_thread_pool(
    provider_requires_hard_timeout: bool,
    constructor_requires_hard_timeout: bool,
) -> None:
    provider = _FunctionProvider(_reward)
    provider.requires_hard_timeout = provider_requires_hard_timeout

    with pytest.raises(ValueError, match="process-isolated"):
        AsyncRewardExecutor(
            provider,
            require_hard_timeout=constructor_requires_hard_timeout,
        )


@pytest.mark.parametrize(
    ("capability", "value", "kwargs"),
    [
        ("supports_concurrent_score", "yes", {}),
        ("requires_hard_timeout", 1, {}),
        (
            "executor_retry_safe",
            "yes",
            {"max_retries": 1},
        ),
        ("max_attempts_per_score", True, {}),
    ],
)
def test_provider_capability_types_fail_closed(capability, value, kwargs) -> None:
    provider = _FunctionProvider(_reward)
    setattr(provider, capability, value)

    with pytest.raises(TypeError, match=capability):
        AsyncRewardExecutor(provider, **kwargs)


def test_sync_path_preserves_provider_metadata_and_is_validated() -> None:
    context = _context()
    batch = _batch(context=context)
    provider_metadata = {
        "provider": "sync-test",
        "source_rows": [10, 11, 12, 13],
        "_executor": {"upstream": True},
    }
    provider = _FunctionProvider(
        lambda shard: _reward(shard, metadata=provider_metadata)
    )

    executor = SyncRewardExecutor(provider)
    rewards = executor.score(batch, context)
    executor.close()
    executor.close()

    rewards.validate_against(batch)
    assert rewards.metadata["provider"] == "sync-test"
    assert rewards.metadata["source_rows"] == [10, 11, 12, 13]
    metrics = rewards.metadata["_executor"]
    assert metrics["provider_metadata"] == {"upstream": True}
    assert metrics["mode"] == "sync"
    assert metrics["shards"] == 1
    assert metrics["attempts"] == 1
    assert metrics["retries"] == metrics["cancelled"] == metrics["timeouts"] == 0
    assert metrics["wall_time_s"] >= 0.0
    assert metrics["timeout_scope"] == "none"
    assert metrics["queue_wait_count"] == metrics["service_latency_count"] == 1
    assert metrics["pending"] == metrics["running"] == 0
    assert metrics["first_failure_count"] == metrics["final_failure_count"] == 0
    with pytest.raises(RewardExecutionError, match="closed"):
        executor.submit(batch, context)


def test_typed_config_builds_async_executor() -> None:
    config = config_from_dict(
        {
            "run_name": "async-reward",
            "runner": {
                "reward_executor": {
                    "mode": "async",
                    "max_workers": 2,
                    "microbatch_size": 2,
                    "timeout_s": 1.0,
                    "max_retries": 1,
                    "submit_timeout_s": 0.5,
                    "max_in_flight": 2,
                    "require_hard_timeout": False,
                }
            },
        }
    )
    provider = _FunctionProvider(_reward)

    with build_reward_executor(provider, config.runner.reward_executor) as executor:
        assert isinstance(executor, AsyncRewardExecutor)
        rewards = executor.score(_batch(context=_context()), _context())

    assert rewards.sample_id == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    assert provider.calls == 2


@pytest.mark.parametrize(
    ("executor_config", "message"),
    [
        ({"mode": "background"}, "mode"),
        ({"max_workers": 0}, "max_workers"),
        ({"microbatch_size": 0}, "microbatch_size"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"max_retries": -1}, "max_retries"),
        ({"submit_timeout_s": -1.0}, "submit_timeout_s"),
        ({"max_in_flight": 0}, "max_in_flight"),
    ],
)
def test_reward_executor_config_fails_closed(executor_config, message) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict(
            {
                "run_name": "invalid-executor",
                "runner": {"reward_executor": executor_config},
            }
        )


@pytest.mark.parametrize(
    "executor_config",
    [
        {"mode": "sync", "max_workers": 0},
        {"mode": "sync", "microbatch_size": True},
        {"mode": "sync", "timeout_s": float("inf")},
        {"mode": "sync", "max_retries": 1.5},
        {"mode": "sync", "submit_timeout_s": -1.0},
        {"mode": "sync", "max_in_flight": 0},
        {"mode": "sync", "unexpected": 1},
    ],
)
def test_factory_validates_every_mapping_field_even_for_sync(
    executor_config,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_reward_executor(_FunctionProvider(_reward), executor_config)


def test_factory_typed_config_is_complete_and_fail_closed() -> None:
    provider = _FunctionProvider(_reward)
    config = RewardExecutorConfig(mode="sync")
    assert config.require_hard_timeout is False
    config.max_workers = 0
    with pytest.raises(ValueError, match="max_workers"):
        build_reward_executor(provider, config)

    config.max_workers = 1
    config.unexpected = "field"
    with pytest.raises(ValueError, match="Unknown reward executor fields"):
        build_reward_executor(provider, config)

    with pytest.raises(TypeError, match="mapping or typed config"):
        build_reward_executor(provider, object())


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_factory_rejects_unavailable_hard_timeout(mode: str) -> None:
    provider = _FunctionProvider(_reward)

    with pytest.raises(ValueError, match="process-isolated"):
        build_reward_executor(
            provider,
            {"mode": mode, "require_hard_timeout": True},
        )


def test_sync_factory_honors_provider_hard_timeout_requirement() -> None:
    provider = _FunctionProvider(_reward)
    provider.requires_hard_timeout = True

    with pytest.raises(ValueError, match="process-isolated"):
        build_reward_executor(provider, {"mode": "sync"})


def test_reward_executor_hard_timeout_config_requires_bool() -> None:
    with pytest.raises(TypeError, match="require_hard_timeout"):
        config_from_dict(
            {
                "run_name": "invalid-hard-timeout",
                "runner": {
                    "reward_executor": {"require_hard_timeout": "yes"},
                },
            }
        )

    with pytest.raises(TypeError, match="require_hard_timeout"):
        build_reward_executor(
            _FunctionProvider(_reward),
            {"mode": "sync", "require_hard_timeout": 1},
        )
