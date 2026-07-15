"""Bounded synchronous and asynchronous reward execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.clients import RewardProtocolError


_SAMPLE_METADATA_FIELDS = frozenset(
    {
        "sample_id",
        "score_meta_view",
        "score_reconstruction",
        "score_trajectory_alignment",
        "source_rows",
        "trajectory_comparison_paths",
        "valid_mask",
    }
)
_RUNTIME_SEQUENCE_METADATA_FIELDS = frozenset(
    {"client_latencies_s", "reward_latencies_s"}
)
_REQUIRED_CONSISTENT_METADATA_FIELDS = frozenset(
    {
        *_SAMPLE_METADATA_FIELDS,
        "configured_batch_size",
        "encoding",
        "identity_mode",
        "payload_batch_sizes",
        "payload_kind",
        "protocol_mode",
        "protocol_version",
        "request_count",
        "sample_id_mode",
        "server_identity_echo",
        "server_revision",
    }
)


class RewardExecutionError(RuntimeError):
    """Reward execution failed without producing a complete valid batch."""

    def __init__(self, message: str, *, metrics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.metrics = dict(metrics or {})


@dataclass(eq=False)
class _HandleCancellation:
    """One cooperative cancellation signal shared by every shard in a handle."""

    event: threading.Event = field(default_factory=threading.Event)
    _pending_tasks: int = field(default=0, init=False, repr=False)
    _submission_open: bool = field(default=True, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def add_task(self) -> None:
        with self._lock:
            if not self._submission_open:
                raise RuntimeError("cannot add a task after reward submission closes")
            self._pending_tasks += 1

    def task_finished(self) -> bool:
        with self._lock:
            if self._pending_tasks < 1:
                raise RuntimeError("reward cancellation task count underflow")
            self._pending_tasks -= 1
            return not self._submission_open and self._pending_tasks == 0

    def finish_submission(self) -> bool:
        with self._lock:
            self._submission_open = False
            return self._pending_tasks == 0


@dataclass(eq=False)
class RewardHandle:
    """Opaque identity and work handle returned by ``RewardExecutor.submit``."""

    context: StepContext
    policy_version: int
    sample_id: tuple[str, ...]
    batch_identity: int
    submitted_at: float
    shards: int
    _owner: object = field(repr=False)
    _batch: RolloutBatch = field(repr=False)
    _tasks: tuple[_ShardTask, ...] = field(default=(), repr=False)
    _telemetry: tuple[_TaskTelemetry, ...] = field(default=(), repr=False)
    _sync_result: _ShardResult | None = field(default=None, repr=False)
    _sync_error: _ShardFailure | None = field(default=None, repr=False)
    _cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )
    _collection_state: str = field(default="new", repr=False)
    _cached_result: RewardBatch | None = field(default=None, repr=False)
    _cached_error: RewardExecutionError | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class _ShardResult:
    rewards: RewardBatch
    attempts: int
    retries: int
    timeouts: int


class _ShardFailure(Exception):
    def __init__(
        self,
        cause: Exception,
        *,
        attempts: int,
        retries: int,
        timeouts: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts
        self.retries = retries
        self.timeouts = timeouts


class _ShardCancelled(Exception):
    """A shard stopped cooperatively before entering a provider call."""


class _Permit:
    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self._semaphore = semaphore
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._semaphore.release()


class _TaskTelemetry:
    """Thread-safe lifecycle and attempt counters for one reward shard."""

    def __init__(self, submitted_at: float) -> None:
        self.submitted_at = submitted_at
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.attempts = 0
        self.retries = 0
        self.timeouts = 0
        self.first_failures = 0
        self.retry_failures = 0
        self.final_failures = 0
        self.provider_queue_wait_s: list[float] = []
        self._lock = threading.Lock()

    def mark_started(self) -> None:
        with self._lock:
            self.started_at = time.monotonic()

    def mark_attempt(self) -> int:
        with self._lock:
            self.attempts += 1
            return self.attempts

    def mark_failure(
        self,
        *,
        attempt: int,
        timeout: bool,
        retrying: bool,
    ) -> None:
        with self._lock:
            if timeout:
                self.timeouts += 1
            if attempt == 1:
                self.first_failures += 1
            else:
                self.retry_failures += 1
            if retrying:
                self.retries += 1
            else:
                self.final_failures += 1

    def mark_finished(self) -> None:
        with self._lock:
            self.finished_at = time.monotonic()

    def mark_provider_queue_wait(self, duration_s: float) -> None:
        with self._lock:
            self.provider_queue_wait_s.append(max(0.0, duration_s))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            started_at = self.started_at
            finished_at = self.finished_at
            return {
                "attempts": self.attempts,
                "retries": self.retries,
                "timeouts": self.timeouts,
                "first_failures": self.first_failures,
                "retry_failures": self.retry_failures,
                "final_failures": self.final_failures,
                "provider_queue_wait_s": tuple(self.provider_queue_wait_s),
                "queue_wait_s": (
                    None
                    if started_at is None
                    else max(0.0, started_at - self.submitted_at)
                ),
                "service_latency_s": (
                    None
                    if started_at is None or finished_at is None
                    else max(0.0, finished_at - started_at)
                ),
            }


@dataclass(frozen=True)
class _ShardTask:
    index: int
    batch: RolloutBatch
    future: Future[_ShardResult]
    permit: _Permit
    telemetry: _TaskTelemetry


class RewardExecutor(ABC):
    """Execute one provider while preserving rollout and policy identity."""

    mode = "base"

    def __init__(self, provider: FeedbackProvider) -> None:
        if not isinstance(provider, FeedbackProvider):
            raise TypeError("provider must be a FeedbackProvider")
        self.provider = provider
        self._owner = object()
        self._closed = False
        self._state_lock = threading.Lock()

    @abstractmethod
    def submit(self, batch: RolloutBatch, context: StepContext) -> RewardHandle:
        """Submit reward work bound to one rollout batch and step context."""

    @abstractmethod
    def collect(
        self,
        handle: RewardHandle,
        batch: RolloutBatch | StepContext,
        context: StepContext | None = None,
    ) -> RewardBatch:
        """Collect a complete reward batch after revalidating its identity."""

    def score(self, batch: RolloutBatch, context: StepContext) -> RewardBatch:
        """Submit and collect reward work for the same step."""

        handle = self.submit(batch, context)
        return self.collect(handle, batch, context)

    def close(self) -> None:
        """Close the executor. Repeated calls are harmless."""

        with self._state_lock:
            self._closed = True

    def __enter__(self) -> RewardExecutor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RewardExecutionError("RewardExecutor is closed")

    def _new_handle(
        self,
        batch: RolloutBatch,
        context: StepContext,
        *,
        submitted_at: float,
        shards: int,
        tasks: Sequence[_ShardTask] = (),
        telemetry: Sequence[_TaskTelemetry] = (),
        sync_result: _ShardResult | None = None,
        sync_error: _ShardFailure | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RewardHandle:
        return RewardHandle(
            context=context,
            policy_version=context.policy_version,
            sample_id=tuple(batch.sample_id),
            batch_identity=id(batch),
            submitted_at=submitted_at,
            shards=shards,
            _owner=self._owner,
            _batch=batch,
            _tasks=tuple(tasks),
            _telemetry=tuple(telemetry),
            _sync_result=sync_result,
            _sync_error=sync_error,
            _cancel_event=(threading.Event() if cancel_event is None else cancel_event),
        )

    def _prepare_submission(self, batch: RolloutBatch, context: StepContext) -> None:
        with self._state_lock:
            self._check_open()
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        batch.validate_lightweight()
        if batch.batch_size == 0:
            raise ValueError("reward execution requires a non-empty batch")
        if batch.context is not None and batch.context != context:
            raise RewardExecutionError(
                "RolloutBatch context does not match the submitted StepContext"
            )

    def _prepare_collection(
        self,
        handle: RewardHandle,
        batch: RolloutBatch | StepContext,
        context: StepContext | None,
    ) -> tuple[RolloutBatch, StepContext]:
        if not isinstance(handle, RewardHandle) or handle._owner is not self._owner:
            raise RewardExecutionError("RewardHandle belongs to another executor")
        if isinstance(batch, StepContext) and context is None:
            batch, context = handle._batch, batch
        if not isinstance(batch, RolloutBatch):
            raise TypeError("collect requires the original RolloutBatch")
        if not isinstance(context, StepContext):
            raise TypeError("collect requires the submitted StepContext")

        if context.policy_version != handle.policy_version:
            raise RewardExecutionError(
                "Refusing stale reward collection for a different policy_version"
            )
        if context != handle.context:
            raise RewardExecutionError(
                "Reward collection context does not match submission context"
            )
        if batch is not handle._batch or id(batch) != handle.batch_identity:
            raise RewardExecutionError(
                "Reward collection requires the original RolloutBatch identity"
            )
        try:
            batch.validate_lightweight()
            current_sample_id = tuple(batch.sample_id)
        except (TypeError, ValueError) as exc:
            raise RewardExecutionError(
                "Original RolloutBatch became invalid after reward submission"
            ) from exc
        if current_sample_id != handle.sample_id:
            raise RewardExecutionError(
                "RolloutBatch sample_id identity changed after reward submission"
            )
        if batch.context is not None and batch.context != context:
            raise RewardExecutionError(
                "RolloutBatch context changed after reward submission"
            )
        return batch, context

    @staticmethod
    def _claim_collection(handle: RewardHandle) -> RewardBatch | None:
        with handle._lock:
            if handle._collection_state == "success":
                if handle._cached_result is None:
                    raise RuntimeError("cached reward result is unavailable")
                return handle._cached_result
            if handle._collection_state == "failure":
                if handle._cached_error is None:
                    raise RuntimeError("cached reward error is unavailable")
                raise handle._cached_error
            if handle._collection_state == "collecting":
                raise RewardExecutionError(
                    "RewardHandle collection is already in progress"
                )
            handle._collection_state = "collecting"
        return None

    @staticmethod
    def _cache_success(handle: RewardHandle, rewards: RewardBatch) -> RewardBatch:
        with handle._lock:
            handle._cached_result = rewards
            handle._collection_state = "success"
        return rewards

    @staticmethod
    def _cache_failure(
        handle: RewardHandle, error: RewardExecutionError
    ) -> RewardExecutionError:
        with handle._lock:
            handle._cached_error = error
            handle._collection_state = "failure"
        return error


class SyncRewardExecutor(RewardExecutor):
    """Reference executor that evaluates one complete batch inline."""

    mode = "sync"

    def submit(self, batch: RolloutBatch, context: StepContext) -> RewardHandle:
        self._prepare_submission(batch, context)
        submitted_at = time.monotonic()
        telemetry = _TaskTelemetry(submitted_at)
        result = None
        error = None
        try:
            result = _execute_shard(
                self.provider,
                batch,
                max_retries=0,
                telemetry=telemetry,
            )
        except _ShardFailure as exc:
            error = exc
        return self._new_handle(
            batch,
            context,
            submitted_at=submitted_at,
            shards=1,
            telemetry=(telemetry,),
            sync_result=result,
            sync_error=error,
        )

    def collect(
        self,
        handle: RewardHandle,
        batch: RolloutBatch | StepContext,
        context: StepContext | None = None,
    ) -> RewardBatch:
        try:
            batch, _ = self._prepare_collection(handle, batch, context)
        except BaseException:
            if isinstance(handle, RewardHandle) and handle._owner is self._owner:
                _cancel_handle(handle)
            raise
        cached = self._claim_collection(handle)
        if cached is not None:
            return cached
        try:
            metrics = _metrics_from_handle(self.mode, handle)
            if handle._sync_error is not None:
                failure = handle._sync_error
                raise RewardExecutionError(
                    "Synchronous reward execution failed", metrics=metrics
                ) from failure.cause
            if handle._sync_result is None:
                raise RewardExecutionError(
                    "Synchronous reward result is unavailable", metrics=metrics
                )
            rewards = _with_executor_metadata(handle._sync_result.rewards, metrics)
            rewards.validate_against(batch)
        except RewardExecutionError as exc:
            raise self._cache_failure(handle, exc)
        except (TypeError, ValueError, RewardProtocolError) as exc:
            error = RewardExecutionError(
                "Synchronous reward result failed validation",
                metrics=_metrics_from_handle(self.mode, handle),
            )
            self._cache_failure(handle, error)
            raise error from exc
        return self._cache_success(handle, rewards)


class AsyncRewardExecutor(RewardExecutor):
    """Bounded thread-pool executor for trusted in-process providers.

    By default each submission stays intact so providers can preserve their own
    batch semantics. Set ``microbatch_size`` explicitly only when the provider
    or its resource envelope requires partitioning within one reward handle.

    Collection deadlines are soft: Python cannot terminate an already-running
    provider call. Providers requiring a hard deadline need process isolation.

    Provider calls are serialized unless the provider explicitly declares
    ``supports_concurrent_score = True``. Outer retries require explicit retry
    safety and attempt-budget declarations on the provider.
    """

    mode = "async"

    def __init__(
        self,
        provider: FeedbackProvider,
        *,
        max_workers: int = 4,
        microbatch_size: int | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 0,
        submit_timeout_s: float = 30.0,
        max_in_flight: int | None = None,
        require_hard_timeout: bool = False,
    ) -> None:
        super().__init__(provider)
        if not isinstance(require_hard_timeout, bool):
            raise TypeError("require_hard_timeout must be a bool")
        if require_hard_timeout or _provider_bool_capability(
            provider,
            "requires_hard_timeout",
            default=False,
        ):
            raise ValueError(
                "AsyncRewardExecutor cannot provide hard cancellation for "
                "in-process provider calls; use a process-isolated provider"
            )
        self.max_workers = _positive_int("max_workers", max_workers)
        self.microbatch_size = (
            None
            if microbatch_size is None
            else _positive_int("microbatch_size", microbatch_size)
        )
        self.timeout_s = _positive_float("timeout_s", timeout_s)
        self.max_retries = _non_negative_int("max_retries", max_retries)
        self.submit_timeout_s = _non_negative_float(
            "submit_timeout_s", submit_timeout_s
        )
        self.max_in_flight = _positive_int(
            "max_in_flight",
            self.max_workers if max_in_flight is None else max_in_flight,
        )
        concurrent_score = _provider_bool_capability(
            provider,
            "supports_concurrent_score",
            default=False,
        )
        self.provider_concurrency_limit = (
            min(self.max_workers, self.max_in_flight) if concurrent_score else 1
        )
        self.provider_attempt_budget_per_shard = _provider_attempt_budget(
            provider,
            max_retries=self.max_retries,
        )
        self._provider_lock = None if concurrent_score else threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="reward",
        )
        self._permits = threading.BoundedSemaphore(self.max_in_flight)
        self._active_cancellations: set[_HandleCancellation] = set()

    def submit(self, batch: RolloutBatch, context: StepContext) -> RewardHandle:
        self._prepare_submission(batch, context)
        submitted_at = time.monotonic()
        if self.microbatch_size is None:
            shards = [batch]
        else:
            shards = [
                batch.slice(
                    range(start, min(start + self.microbatch_size, batch.batch_size))
                )
                for start in range(0, batch.batch_size, self.microbatch_size)
            ]
        tasks: list[_ShardTask] = []
        cancellation = _HandleCancellation()
        with self._state_lock:
            self._check_open()
            self._active_cancellations.add(cancellation)
        try:
            for index, shard in enumerate(shards):
                if not self._permits.acquire(timeout=self.submit_timeout_s):
                    cancellation.event.set()
                    _cancel_tasks(tasks)
                    metrics = _metrics_from_tasks(
                        mode=self.mode,
                        shards=len(shards),
                        wall_time_s=time.monotonic() - submitted_at,
                        tasks=tasks,
                        extra_timeouts=1,
                        provider_concurrency_limit=self.provider_concurrency_limit,
                        provider_attempt_budget_per_shard=(
                            self.provider_attempt_budget_per_shard
                        ),
                    )
                    raise RewardExecutionError(
                        "Timed out waiting for reward executor capacity",
                        metrics=metrics,
                    )
                permit = _Permit(self._permits)
                telemetry = _TaskTelemetry(time.monotonic())
                cancellation.add_task()
                try:
                    future = self._pool.submit(
                        _execute_shard,
                        self.provider,
                        shard,
                        self.max_retries,
                        telemetry,
                        self._provider_lock,
                        cancellation.event,
                    )
                except Exception:
                    cancellation.task_finished()
                    permit.release()
                    raise
                tasks.append(_ShardTask(index, shard, future, permit, telemetry))

                def finish_task(
                    _future: Future[_ShardResult],
                    *,
                    item: _Permit = permit,
                    state: _HandleCancellation = cancellation,
                ) -> None:
                    item.release()
                    if state.task_finished():
                        self._forget_cancellation(state)

                future.add_done_callback(finish_task)
        except RewardExecutionError:
            cancellation.event.set()
            _cancel_tasks(tasks)
            raise
        except Exception as exc:
            cancellation.event.set()
            _cancel_tasks(tasks)
            metrics = _metrics_from_tasks(
                mode=self.mode,
                shards=len(shards),
                wall_time_s=time.monotonic() - submitted_at,
                tasks=tasks,
                provider_concurrency_limit=self.provider_concurrency_limit,
                provider_attempt_budget_per_shard=(
                    self.provider_attempt_budget_per_shard
                ),
            )
            raise RewardExecutionError(
                "Failed to submit reward work", metrics=metrics
            ) from exc
        finally:
            if cancellation.finish_submission():
                self._forget_cancellation(cancellation)

        return self._new_handle(
            batch,
            context,
            submitted_at=submitted_at,
            shards=len(shards),
            tasks=tasks,
            telemetry=[task.telemetry for task in tasks],
            cancel_event=cancellation.event,
        )

    def collect(
        self,
        handle: RewardHandle,
        batch: RolloutBatch | StepContext,
        context: StepContext | None = None,
    ) -> RewardBatch:
        batch, _ = self._prepare_collection(handle, batch, context)
        cached = self._claim_collection(handle)
        if cached is not None:
            return cached

        results: dict[int, _ShardResult] = {}
        future_to_task = {task.future: task for task in handle._tasks}
        deadline = handle.submitted_at + self.timeout_s
        try:
            remaining = max(0.0, deadline - time.monotonic())
            for future in as_completed(future_to_task, timeout=remaining):
                task = future_to_task[future]
                try:
                    results[task.index] = future.result()
                except _ShardFailure as failure:
                    _cancel_handle(handle)
                    metrics = _metrics_from_handle(
                        self.mode,
                        handle,
                        provider_concurrency_limit=self.provider_concurrency_limit,
                        provider_attempt_budget_per_shard=(
                            self.provider_attempt_budget_per_shard
                        ),
                    )
                    raise RewardExecutionError(
                        f"Reward shard {task.index} failed", metrics=metrics
                    ) from failure.cause
                except Exception as exc:
                    _cancel_handle(handle)
                    metrics = _metrics_from_handle(
                        self.mode,
                        handle,
                        provider_concurrency_limit=self.provider_concurrency_limit,
                        provider_attempt_budget_per_shard=(
                            self.provider_attempt_budget_per_shard
                        ),
                    )
                    raise RewardExecutionError(
                        f"Reward shard {task.index} was cancelled",
                        metrics=metrics,
                    ) from exc
        except FuturesTimeoutError as exc:
            _cancel_handle(handle)
            metrics = _metrics_from_handle(
                self.mode,
                handle,
                collect_timeout=True,
                provider_concurrency_limit=self.provider_concurrency_limit,
                provider_attempt_budget_per_shard=(
                    self.provider_attempt_budget_per_shard
                ),
            )
            error = RewardExecutionError(
                "Timed out collecting reward shards", metrics=metrics
            )
            self._cache_failure(handle, error)
            raise error from exc
        except RewardExecutionError as exc:
            raise self._cache_failure(handle, exc)

        ordered = [results[index] for index in range(handle.shards)]
        metrics = _metrics_from_handle(
            self.mode,
            handle,
            provider_concurrency_limit=self.provider_concurrency_limit,
            provider_attempt_budget_per_shard=(self.provider_attempt_budget_per_shard),
        )
        try:
            rewards = _merge_reward_batches(
                [item.rewards for item in ordered],
                [task.batch.batch_size for task in handle._tasks],
            )
            rewards = _with_executor_metadata(rewards, metrics)
            rewards.validate_against(batch)
        except (TypeError, ValueError, RewardProtocolError) as exc:
            _cancel_handle(handle)
            error = RewardExecutionError(
                "Merged reward batch failed validation", metrics=metrics
            )
            self._cache_failure(handle, error)
            raise error from exc
        return self._cache_success(handle, rewards)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            cancellations = tuple(self._active_cancellations)
        for cancellation in cancellations:
            cancellation.event.set()
        # Running Python calls cannot be force-terminated. They finish in the
        # background and release their permits through their done callbacks.
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _forget_cancellation(self, cancellation: _HandleCancellation) -> None:
        with self._state_lock:
            self._active_cancellations.discard(cancellation)


def _execute_shard(
    provider: FeedbackProvider,
    batch: RolloutBatch,
    max_retries: int,
    telemetry: _TaskTelemetry,
    provider_lock: threading.Lock | None = None,
    cancel_event: threading.Event | None = None,
) -> _ShardResult:
    telemetry.mark_started()
    try:
        return _score_shard(
            provider,
            batch,
            max_retries,
            telemetry,
            provider_lock,
            cancel_event,
        )
    finally:
        telemetry.mark_finished()


def _score_shard(
    provider: FeedbackProvider,
    batch: RolloutBatch,
    max_retries: int,
    telemetry: _TaskTelemetry,
    provider_lock: threading.Lock | None = None,
    cancel_event: threading.Event | None = None,
) -> _ShardResult:
    attempts = 0
    retries = 0
    timeouts = 0
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise _ShardCancelled("reward shard was cancelled before provider call")
        try:
            if provider_lock is None:
                telemetry.mark_provider_queue_wait(0.0)
                if cancel_event is not None and cancel_event.is_set():
                    raise _ShardCancelled(
                        "reward shard was cancelled before provider call"
                    )
                attempts += 1
                telemetry.mark_attempt()
                rewards = provider.score(batch)
            else:
                wait_started = time.monotonic()
                with provider_lock:
                    telemetry.mark_provider_queue_wait(time.monotonic() - wait_started)
                    if cancel_event is not None and cancel_event.is_set():
                        raise _ShardCancelled(
                            "reward shard was cancelled before provider call"
                        )
                    attempts += 1
                    telemetry.mark_attempt()
                    rewards = provider.score(batch)
            if not isinstance(rewards, RewardBatch):
                raise TypeError("FeedbackProvider.score must return RewardBatch")
            canonical = rewards.canonical()
            canonical.validate_against(batch)
            return _ShardResult(canonical, attempts, retries, timeouts)
        except _ShardCancelled:
            raise
        except Exception as exc:
            timeout = isinstance(exc, TimeoutError)
            if timeout:
                timeouts += 1
            retrying = retries < max_retries and _is_retryable(exc)
            telemetry.mark_failure(
                attempt=attempts,
                timeout=timeout,
                retrying=retrying,
            )
            if retrying:
                retries += 1
                continue
            raise _ShardFailure(
                exc,
                attempts=attempts,
                retries=retries,
                timeouts=timeouts,
            ) from exc


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RewardProtocolError, TypeError, ValueError)):
        return False
    return isinstance(exc, (RuntimeError, OSError, TimeoutError))


def _merge_reward_batches(
    rewards: Sequence[RewardBatch], shard_sizes: Sequence[int]
) -> RewardBatch:
    if not rewards:
        raise ValueError("Cannot merge an empty reward result")
    raw_keys = tuple(rewards[0].raw)
    weighted_keys = tuple(rewards[0].weighted)
    for shard in rewards[1:]:
        if set(shard.raw) != set(raw_keys):
            raise ValueError("RewardBatch raw keys must match across shards")
        if set(shard.weighted) != set(weighted_keys):
            raise ValueError("RewardBatch weighted keys must match across shards")

    import torch

    return RewardBatch(
        raw={
            key: torch.cat([shard.raw[key] for shard in rewards], dim=0)
            for key in raw_keys
        },
        weighted={
            key: torch.cat([shard.weighted[key] for shard in rewards], dim=0)
            for key in weighted_keys
        },
        weighted_total=torch.cat([shard.weighted_total for shard in rewards], dim=0),
        valid_mask=torch.cat([shard.valid_mask for shard in rewards], dim=0),
        metadata=_merge_metadata([shard.metadata for shard in rewards], shard_sizes),
        sample_id=[sample for shard in rewards for sample in shard.sample_id],
    )


def _merge_metadata(
    metadata: Sequence[Mapping[str, Any]],
    shard_sizes: Sequence[int],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    if len(metadata) == 1:
        return dict(metadata[0])
    keys = set().union(*(item.keys() for item in metadata))
    merged: dict[str, Any] = {}
    for key in keys:
        if not all(key in item for item in metadata):
            if str(key) in _REQUIRED_CONSISTENT_METADATA_FIELDS:
                child = ".".join((*path, str(key)))
                raise ValueError(
                    f"Shard metadata {child} must be present in every shard"
                )
            merged[key] = {
                str(index): item[key]
                for index, item in enumerate(metadata)
                if key in item
            }
            continue
        values = [item[key] for item in metadata]
        merged[key] = _merge_metadata_values(
            values,
            shard_sizes,
            path=(*path, str(key)),
        )
    if path and path[-1] == "_runtime":
        hits = merged.get("cache_hits")
        misses = merged.get("cache_misses")
        if _is_non_negative_int(hits) and _is_non_negative_int(misses):
            requests = hits + misses
            merged["cache_hit_rate"] = float(hits / requests) if requests else 0.0
    return merged


def _merge_metadata_values(
    values: Sequence[Any],
    shard_sizes: Sequence[int],
    *,
    path: tuple[str, ...],
) -> Any:
    field = path[-1]
    if _is_additive_count_path(path) and all(
        _is_non_negative_int(value) for value in values
    ):
        return sum(values)
    if all(isinstance(value, Mapping) for value in values):
        return _merge_metadata(values, shard_sizes, path=path)
    if field in _SAMPLE_METADATA_FIELDS and all(
        isinstance(value, list) and len(value) == size
        for value, size in zip(values, shard_sizes)
    ):
        return [entry for value in values for entry in value]
    if field in _SAMPLE_METADATA_FIELDS and all(
        isinstance(value, tuple) and len(value) == size
        for value, size in zip(values, shard_sizes)
    ):
        return tuple(entry for value in values for entry in value)
    if field in _SAMPLE_METADATA_FIELDS and _batch_arrays(values, shard_sizes):
        return _concatenate_arrays(values)
    if field in _SAMPLE_METADATA_FIELDS:
        raise ValueError(
            f"Shard metadata {'.'.join(path)} must contain one entry per sample"
        )
    if field == "payload_batch_sizes" and all(
        _valid_payload_batch_sizes(value, size)
        for value, size in zip(values, shard_sizes)
    ):
        return [entry for value in values for entry in value]
    if field == "payload_batch_sizes":
        raise ValueError(
            f"Shard metadata {'.'.join(path)} must contain positive batch sizes "
            "that sum to the shard size"
        )
    if _is_runtime_sequence_path(path) and all(
        isinstance(value, list) for value in values
    ):
        return [entry for value in values for entry in value]
    merged_input_shape = _merge_input_shapes(values, shard_sizes, path=path)
    if merged_input_shape is not None:
        return merged_input_shape
    if all(_values_equal(values[0], value) for value in values[1:]):
        return values[0]
    if all(isinstance(value, list) for value in values) and all(
        len(value) == size for value, size in zip(values, shard_sizes)
    ):
        return [entry for value in values for entry in value]
    if all(isinstance(value, tuple) for value in values) and all(
        len(value) == size for value, size in zip(values, shard_sizes)
    ):
        return tuple(entry for value in values for entry in value)
    if _batch_arrays(values, shard_sizes):
        return _concatenate_arrays(values)
    return list(values)


def _is_additive_count_path(path: tuple[str, ...]) -> bool:
    field = path[-1]
    return field == "request_count" or (
        len(path) >= 2
        and path[-2] == "_runtime"
        and field in {"cache_hits", "cache_misses"}
    )


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _valid_payload_batch_sizes(value: Any, shard_size: int) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_non_negative_int(item) and item > 0 for item in value)
        and sum(value) == shard_size
    )


def _is_runtime_sequence_path(path: tuple[str, ...]) -> bool:
    return (
        len(path) >= 2
        and path[-2] == "_runtime"
        and path[-1] in _RUNTIME_SEQUENCE_METADATA_FIELDS
    )


def _merge_input_shapes(
    values: Sequence[Any],
    shard_sizes: Sequence[int],
    *,
    path: tuple[str, ...],
) -> list[int] | None:
    if len(path) < 2 or path[-2:] != ("encoding", "input_shape"):
        return None
    if not all(
        isinstance(value, list)
        and len(value) >= 1
        and all(_is_non_negative_int(dimension) for dimension in value)
        and value[0] == size
        for value, size in zip(values, shard_sizes)
    ):
        raise ValueError(
            f"Shard metadata {'.'.join(path)} must start with the shard size"
        )
    tail = values[0][1:]
    if not all(value[1:] == tail for value in values[1:]):
        raise ValueError(
            f"Shard metadata {'.'.join(path)} has inconsistent media dimensions"
        )
    return [sum(shard_sizes), *tail]


def _values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        import torch

        if isinstance(left, torch.Tensor):
            return bool(torch.equal(left, right))
    except ImportError:
        pass
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    try:
        result = left == right
        return bool(result) if not hasattr(result, "all") else bool(result.all())
    except (TypeError, ValueError):
        return False


def _batch_arrays(values: Sequence[Any], shard_sizes: Sequence[int]) -> bool:
    return all(
        getattr(value, "shape", ()) and int(value.shape[0]) == size
        for value, size in zip(values, shard_sizes)
    )


def _concatenate_arrays(values: Sequence[Any]) -> Any:
    try:
        import torch

        if all(isinstance(value, torch.Tensor) for value in values):
            return torch.cat(list(values), dim=0)
    except ImportError:
        pass
    import numpy as np

    return np.concatenate(values, axis=0)


def _with_executor_metadata(
    rewards: RewardBatch, metrics: Mapping[str, Any]
) -> RewardBatch:
    metadata = dict(rewards.metadata)
    executor_metadata = dict(metrics)
    if "_executor" in metadata:
        executor_metadata["provider_metadata"] = metadata["_executor"]
    metadata["_executor"] = executor_metadata
    return RewardBatch(
        raw=dict(rewards.raw),
        weighted=dict(rewards.weighted),
        weighted_total=rewards.weighted_total,
        valid_mask=rewards.valid_mask,
        metadata=metadata,
        sample_id=list(rewards.sample_id),
    )


def _cancel_tasks(tasks: Sequence[_ShardTask]) -> int:
    cancelled = 0
    for task in tasks:
        if task.future.done():
            continue
        if task.future.cancel():
            task.permit.release()
            cancelled += 1
    return cancelled


def _cancel_handle(handle: RewardHandle) -> int:
    handle._cancel_event.set()
    return _cancel_tasks(handle._tasks)


def _metrics_from_handle(
    mode: str,
    handle: RewardHandle,
    *,
    collect_timeout: bool = False,
    provider_concurrency_limit: int = 1,
    provider_attempt_budget_per_shard: int = 0,
) -> dict[str, Any]:
    return _execution_metrics(
        mode=mode,
        shards=handle.shards,
        wall_time_s=time.monotonic() - handle.submitted_at,
        telemetry=handle._telemetry,
        tasks=handle._tasks,
        extra_timeouts=int(collect_timeout),
        provider_concurrency_limit=provider_concurrency_limit,
        provider_attempt_budget_per_shard=provider_attempt_budget_per_shard,
    )


def _metrics_from_tasks(
    *,
    mode: str,
    shards: int,
    wall_time_s: float,
    tasks: Sequence[_ShardTask],
    extra_timeouts: int = 0,
    provider_concurrency_limit: int = 1,
    provider_attempt_budget_per_shard: int = 0,
) -> dict[str, Any]:
    return _execution_metrics(
        mode=mode,
        shards=shards,
        wall_time_s=wall_time_s,
        telemetry=[task.telemetry for task in tasks],
        tasks=tasks,
        extra_timeouts=extra_timeouts,
        provider_concurrency_limit=provider_concurrency_limit,
        provider_attempt_budget_per_shard=provider_attempt_budget_per_shard,
    )


def _execution_metrics(
    *,
    mode: str,
    shards: int,
    wall_time_s: float,
    telemetry: Sequence[_TaskTelemetry],
    tasks: Sequence[_ShardTask],
    extra_timeouts: int = 0,
    provider_concurrency_limit: int = 1,
    provider_attempt_budget_per_shard: int = 0,
) -> dict[str, Any]:
    snapshots = [item.snapshot() for item in telemetry]
    attempts = sum(item["attempts"] for item in snapshots)
    retries = sum(item["retries"] for item in snapshots)
    first_failures = sum(item["first_failures"] for item in snapshots)
    retry_failures = sum(item["retry_failures"] for item in snapshots)
    final_failures = sum(item["final_failures"] for item in snapshots)
    queue_wait = [
        item["queue_wait_s"] for item in snapshots if item["queue_wait_s"] is not None
    ]
    service_latency = [
        item["service_latency_s"]
        for item in snapshots
        if item["service_latency_s"] is not None
    ]
    provider_queue_wait = [
        value for item in snapshots for value in item["provider_queue_wait_s"]
    ]
    pending, running, cancelled = _task_state_counts(tasks)
    metrics = {
        "mode": mode,
        "shards": shards,
        "attempts": attempts,
        "retries": retries,
        "wall_time_s": max(0.0, wall_time_s),
        "cancelled": cancelled,
        "timeouts": sum(item["timeouts"] for item in snapshots) + extra_timeouts,
        "pending": pending,
        "running": running,
        "still_running": running,
        "timeout_scope": "soft_collection" if mode == "async" else "none",
        "hard_cancel_supported": 0,
        "timed_out_work_may_continue": int(extra_timeouts > 0 and running > 0),
        "background_work_may_continue": int(running > 0),
        "provider_concurrency_limit": provider_concurrency_limit,
        "provider_attempt_budget_per_shard": provider_attempt_budget_per_shard,
        "provider_attempt_budget_known": int(provider_attempt_budget_per_shard > 0),
        "first_failure_count": first_failures,
        "first_failure_rate": _rate(first_failures, len(queue_wait)),
        "retry_failure_count": retry_failures,
        "retry_failure_rate": _rate(retry_failures, retries),
        "final_failure_count": final_failures,
        "final_failure_rate": _rate(final_failures, len(service_latency)),
    }
    metrics.update(_latency_metrics("queue_wait", queue_wait))
    metrics.update(_latency_metrics("service_latency", service_latency))
    metrics.update(_latency_metrics("provider_queue_wait", provider_queue_wait))
    return metrics


def _provider_bool_capability(
    provider: FeedbackProvider,
    name: str,
    *,
    default: bool,
) -> bool:
    value = getattr(provider, name, default)
    if not isinstance(value, bool):
        raise TypeError(f"FeedbackProvider.{name} must be a bool when defined")
    return value


def _provider_attempt_budget(
    provider: FeedbackProvider,
    *,
    max_retries: int,
) -> int:
    if max_retries and not _provider_bool_capability(
        provider,
        "executor_retry_safe",
        default=False,
    ):
        raise ValueError(
            "max_retries requires FeedbackProvider.executor_retry_safe=True; "
            "provider calls may otherwise hide nested retries or duplicate work"
        )
    declared = getattr(provider, "max_attempts_per_score", None)
    if declared is None and not max_retries:
        return 0
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        error_type = ValueError if max_retries else TypeError
        message = (
            "max_retries requires a positive "
            "FeedbackProvider.max_attempts_per_score declaration"
            if max_retries
            else "FeedbackProvider.max_attempts_per_score must be a positive "
            "integer when defined"
        )
        raise error_type(message)
    return (max_retries + 1) * declared


def _task_state_counts(tasks: Sequence[_ShardTask]) -> tuple[int, int, int]:
    pending = 0
    running = 0
    cancelled = 0
    for task in tasks:
        if task.future.cancelled():
            cancelled += 1
        elif task.future.running():
            running += 1
        elif not task.future.done():
            pending += 1
    return pending, running, cancelled


def _latency_metrics(prefix: str, values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        f"{prefix}_count": len(ordered),
        f"{prefix}_sum_s": sum(ordered),
        f"{prefix}_max_s": max(ordered, default=0.0),
        f"{prefix}_p50_s": _percentile(ordered, 0.50),
        f"{prefix}_p95_s": _percentile(ordered, 0.95),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _rate(count: int, total: int) -> float:
    return float(count) / total if total else 0.0


def _positive_int(name: str, value: Any) -> int:
    resolved = _non_negative_int(name, value)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_float(name: str, value: Any) -> float:
    resolved = _non_negative_float(name, value)
    if resolved == 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _non_negative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


__all__ = [
    "AsyncRewardExecutor",
    "RewardExecutionError",
    "RewardExecutor",
    "RewardHandle",
    "SyncRewardExecutor",
]
