"""Config-driven VisualRL experiment runner and single training loop."""

from __future__ import annotations

from collections.abc import Mapping
import datetime as _datetime
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import time
from typing import Any
import uuid

from visual_rl.artifacts import (
    ArtifactManager,
    ArtifactPaths,
    ManifestBuilder,
    StepArtifactTransaction,
)
from visual_rl.artifacts.checkpoint import (
    apply_training_state,
    build_implementation_identity,
    capture_rng_state,
    read_and_validate_training_state,
    save_json,
    save_training_state,
)
from visual_rl.artifacts.logging import TrainProgressPrinter
from visual_rl.artifacts.status import write_run_status
from visual_rl.builtins import register_builtin_plugins
from visual_rl.callbacks import CallbackContext, CallbackError, RunCallback
from visual_rl.configs.schema import (
    VisualRLConfig,
    config_to_dict,
    section_to_dict,
    validate_config,
)
from visual_rl.core.determinism import assert_runtime, configure_runtime
from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.seed import seed_everything
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.datasets.prompt_dataset import (
    PromptDataset,
    prompt_content_sha256,
    read_prompt_file,
    validate_prompt_splits,
)
from visual_rl.distributed import (
    DDPStrategy,
    DistributedContext,
    DistributedFailureError,
    SingleProcessStrategy,
)
from visual_rl.evaluation import EvaluationContext, EvaluationResult, Evaluator
from visual_rl.feedback import (
    RewardExecutionError,
    build_feedback_provider,
    build_reward_executor,
)
from visual_rl.optimizers import build_optimizer_plugin
from visual_rl.preflight import (
    has_transactional_artifact_layout,
    latest_committed_step,
    resolve_resume_checkpoint,
    resume_run_root,
)
from visual_rl.rollout.cache import RolloutCache
from visual_rl.rollout.full_trajectory import build_rollout_engine
from visual_rl.scaling import build_scaling_trigger_decision


class ResumeError(RuntimeError):
    """A requested checkpoint could not be validated or restored."""


def recover_resume_source_if_needed(
    resume_from: str | Path,
) -> list[dict[str, Any]]:
    """Finish ready source transactions before resolving or loading checkpoints."""

    try:
        run_root = resume_run_root(resume_from)
        if not has_transactional_artifact_layout(run_root):
            return []
        return ArtifactManager.recover_run(run_root)
    except ResumeError:
        raise
    except Exception as exc:
        raise ResumeError(
            f"Cannot recover resume-source artifact transactions: {exc}"
        ) from exc


def prepare_resume_source(
    resume_from: str | Path | None,
) -> list[dict[str, Any]]:
    """Recover, then fail-closed validate, a resume source before runtime setup.

    CLI and Python callers use this shared boundary after trusted component
    preflight and before constructing ``ExperimentRunner``.  The runner keeps
    its distributed recovery pass as the authoritative in-process safeguard.
    """

    from visual_rl.preflight import validate_resume_path

    audit = (
        recover_resume_source_if_needed(resume_from)
        if resume_from is not None
        else []
    )
    validate_resume_path(resume_from)
    return audit


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _load_scaling_trigger_decision(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Scaling decision validation requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Scaling trigger decision must be a regular file")
        if metadata.st_size > 1024 * 1024:
            raise RuntimeError("Scaling trigger decision exceeds the 1 MiB limit")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(payload, dict):
        raise ValueError("Scaling trigger decision must contain a JSON object")
    return payload


class ExperimentRunner:
    """Build components, execute policy updates, and persist a reproducible run."""

    def __init__(
        self,
        config: VisualRLConfig,
        *,
        callbacks: tuple[RunCallback, ...] | list[RunCallback] = (),
        evaluator: Evaluator | None = None,
    ):
        # Programmatic callers can construct or mutate the public dataclasses
        # without passing through config_from_dict().  Revalidate before process
        # groups, model loading, output directories, or artifacts have side effects.
        validate_config(config)
        self.config = config
        self.scaling_trigger_decision = build_scaling_trigger_decision(
            config.runner.conditional_scaling
        )
        distributed = config.runner.distributed
        self.distributed_context = DistributedContext.from_env(
            device=distributed.device,
            backend=distributed.backend,
        )
        self.rank = self.distributed_context.rank
        self.world_size = self.distributed_context.world_size
        self.strategy = (
            DDPStrategy(
                self.distributed_context,
                timeout_s=distributed.timeout_s,
                max_snapshot_tensor_bytes=distributed.max_snapshot_tensor_bytes,
            )
            if self.distributed_context.is_distributed
            else SingleProcessStrategy(self.distributed_context)
        )
        try:
            self.strategy.setup()
            self.callbacks = tuple(callbacks)
            if any(
                not isinstance(callback, RunCallback) for callback in self.callbacks
            ):
                raise TypeError("callbacks must contain RunCallback instances")
            if evaluator is not None and not isinstance(evaluator, Evaluator):
                raise TypeError("evaluator must be an Evaluator instance or None")
            self.evaluator = evaluator
            self.evaluation_results: tuple[EvaluationResult, ...] = ()
            self.evaluation_paths: dict[str, Path] = {}
            self._validate_config()
            self.artifact_recovery_audit: list[dict[str, Any]] = []
            self._resume_scaling_trigger_decision: dict[str, Any] | None = None
            if config.paths.resume_from:
                recovery_audit = self._run_distributed_phase(
                    "resume_recovery",
                    lambda: (
                        recover_resume_source_if_needed(config.paths.resume_from)
                        if self.rank == 0
                        else None
                    ),
                )
                recovery_audit = self.strategy.broadcast_object(
                    recovery_audit if self.rank == 0 else None
                )
                self.artifact_recovery_audit.extend(recovery_audit or ())
                source_decision = self._run_distributed_phase(
                    "resume_scaling_decision",
                    lambda: (
                        self._read_resume_scaling_trigger_decision(
                            config.paths.resume_from
                        )
                        if self.rank == 0
                        else None
                    ),
                )
                self._resume_scaling_trigger_decision = self.strategy.broadcast_object(
                    source_decision if self.rank == 0 else None
                )

            register_builtin_plugins()
            self.runtime_identity = configure_runtime(
                enabled=bool(config.runner.deterministic_runtime),
                seed=config.seed,
            )
            seed_everything(config.seed)
            self.output_dir = self._setup_shared_output_dir()
            self.global_step = 0

            model_config = self._resolved_model_config(config)
            adapter_cls = MODEL_ADAPTERS.get(model_config.get("name", "mock_wan"))
            self.adapter = adapter_cls(model_config)
            if (
                config.runner.auto_load_model
                and hasattr(self.adapter, "load")
                and getattr(self.adapter, "pipeline", None) is None
            ):
                self._run_distributed_phase("model_load", self.adapter.load)
                self._assert_runtime_integrity("after adapter load")

            try:
                self.dataset = PromptDataset.from_config(config.dataset)
            except OSError as exc:
                raise RuntimeError(
                    "Cannot validate dataset data source "
                    f"{config.dataset.path!r}: {exc}"
                ) from exc
            self.config.dataset.content_sha256 = self.dataset.content_sha256
            self._evaluation_prompts = self._validate_evaluation_data_identity()
            rollout_config = section_to_dict(config.sample)
            rollout_config.update(config.rollout)
            self.rollout = build_rollout_engine(rollout_config)

            reward_cache_root = Path(
                config.rewards.cache_dir or self.output_dir / "reward_cache"
            )
            reward_cache_dir = self._rank_cache_path(reward_cache_root)
            self.feedback_provider = build_feedback_provider(
                config.rewards,
                cache_dir=reward_cache_dir,
            )
            self.reward_executor = None
            self.rollout_cache_root: Path | None = None
            rollout_cache_dir = None
            if not config.runner.disable_rollout_cache:
                self.rollout_cache_root = Path(
                    config.runner.rollout_cache_dir or self.output_dir / "rollouts"
                )
                rollout_cache_dir = self._rank_cache_path(self.rollout_cache_root)
            self.rollout_cache = RolloutCache(rollout_cache_dir)
            self.optimizer_plugin = build_optimizer_plugin(config)
            self.checkpoint_identity = build_implementation_identity(
                self.adapter,
                self.optimizer_plugin,
                rollout=self.rollout,
                feedback=self.feedback_provider,
            )
            self.checkpoint_identity["runtime"] = self.runtime_identity
            self._resume_checkpoint_path: Path | None = None
            self._resume_metadata: dict[str, Any] | None = None
            validated_state = self._run_distributed_phase(
                "resume",
                self._read_resume_if_requested,
            )

            self._run_distributed_phase(
                "prepare",
                lambda: self.strategy.prepare(self.adapter),
            )

            def build_optimizer():
                optimizer = self.optimizer_plugin.build_optimizer(
                    self.adapter.parameters(),
                    config.train,
                )
                self._validate_optimizer_contract(optimizer)
                return optimizer

            self.optimizer = self._run_distributed_phase(
                "optimizer_setup",
                build_optimizer,
            )
            if validated_state is None:
                self.start_step = 0
            else:
                self.start_step = self._run_distributed_phase(
                    "resume_apply",
                    lambda: apply_training_state(
                        validated_state,
                        optimizer=self.optimizer,
                        plugin=self.optimizer_plugin,
                        rank=self.rank,
                    ),
                )
                self.global_step = self.start_step

            root_start_step = self.strategy.broadcast_object(
                self.start_step if self.rank == 0 else None
            )

            def validate_start_step() -> None:
                if self.start_step != root_start_step:
                    raise ResumeError(
                        "Distributed ranks resolved different checkpoint steps"
                    )

            self._run_distributed_phase("resume_consensus", validate_start_step)

            self.progress = TrainProgressPrinter(
                enabled=bool(config.runner.show_progress and self.rank == 0),
                interval=config.runner.progress_interval,
                leave=config.runner.progress_leave,
            )
            self._run_started = False
            self.retention_audit: list[dict[str, Any]] = []
            self.post_commit_bookkeeping_errors: list[dict[str, str]] = []
            self.artifacts: ArtifactManager | ArtifactPaths = ArtifactPaths.for_run(
                self.output_dir,
                config.run_name,
            )

            def initialize_artifacts() -> None:
                if self.rank != 0:
                    return
                continuing_artifacts = self._artifact_history_exists()
                manager = ArtifactManager(
                    self.output_dir,
                    config.run_name,
                    config=config_to_dict(config),
                    resume=bool(config.paths.resume_from and continuing_artifacts),
                    hold_writer_lock=False,
                )
                self.artifacts = manager
                try:
                    manager.acquire_writer_lock()
                    self.artifact_recovery_audit.extend(manager.recover())
                    self._validate_resume_head()
                    if self.start_step > 0 and not self._has_commit_markers():
                        manager.truncate_from_step(self.start_step)
                        self._rewrite_legacy_latest_if_local()
                    self._persist_scaling_trigger_decision()
                except BaseException:
                    manager.close()
                    raise
                finally:
                    if not manager._closed:
                        manager.release_writer_lock()

            self._run_distributed_phase("artifact_init", initialize_artifacts)
            if self.start_step > 0:
                self._run_distributed_phase(
                    "cache_resume",
                    lambda: self.rollout_cache.truncate_from_step(self.start_step),
                )
        except BaseException:
            artifacts = getattr(self, "artifacts", None)
            if isinstance(artifacts, ArtifactManager):
                artifacts.close()
            self.strategy.close()
            raise

    def run(self, max_steps: int | None = None) -> list[dict[str, Any]]:
        if self.world_size > 1:
            target_steps = int(
                self.config.train.max_steps if max_steps is None else max_steps
            )
            self._run_distributed_phase(
                "run_status_running",
                lambda: self._write_lifecycle_status(
                    "running",
                    target_steps=target_steps,
                ),
            )
            try:
                metrics = self._run_distributed(max_steps)
            except BaseException as exc:
                self._write_lifecycle_status(
                    "failed",
                    target_steps=target_steps,
                    error=exc,
                    owner_error=exc,
                )
                raise
            self._write_lifecycle_status("completed", target_steps=target_steps)
            return metrics
        target_steps = int(
            self.config.train.max_steps if max_steps is None else max_steps
        )
        if target_steps < self.start_step:
            self.strategy.close()
            raise ValueError(
                f"max_steps={target_steps} is before resumed step {self.start_step}"
            )
        batch_size = int(self.config.sample.batch_size)
        save_every = int(self.config.train.save_every or max(1, target_steps))
        all_metrics: list[dict[str, Any]] = []
        primary_error: BaseException | None = None
        transaction: StepArtifactTransaction | None = None
        pending_metrics: list[dict[str, Any]] = []
        cycle_started: float | None = None
        cycle_sample_count = 0

        if self._run_started:
            self.strategy.close()
            raise RuntimeError("ExperimentRunner.run() may only be called once")
        self._write_lifecycle_status("running", target_steps=target_steps)
        try:
            self.artifacts.acquire_writer_lock()
        except BaseException as exc:
            self._write_lifecycle_status(
                "failed",
                target_steps=target_steps,
                error=exc,
                owner_error=exc,
            )
            self.strategy.close()
            raise
        self._run_started = True
        try:
            self.reward_executor = build_reward_executor(
                self.feedback_provider,
                self.config.runner.reward_executor,
            )
            self._assert_run_head_unchanged()
            self.progress.start(target_steps, initial_step=self.start_step)
            self._dispatch_callbacks("on_run_start", self._callback_context())
            for step in range(self.start_step, target_steps):
                step_started = time.perf_counter()
                if transaction is None:
                    cycle_started = step_started
                    cycle_sample_count = 0
                    transaction = self.artifacts.begin_transaction()
                self._assert_runtime_integrity(f"before training step {step}")
                prompts, metadata, epoch_tag = self.dataset.batch(
                    step * batch_size,
                    batch_size,
                    epoch_tag=step,
                )
                context = StepContext(
                    step=step,
                    seed=self.config.seed + step,
                    epoch_tag=epoch_tag,
                    rank=0,
                    world_size=1,
                    policy_version=step,
                )
                self.adapter.prepare_for_sampling()
                rollout_started = time.perf_counter()
                batch = self.rollout.sample(
                    self.adapter,
                    prompts,
                    metadata,
                    context,
                )
                rollout_time = time.perf_counter() - rollout_started
                self._bind_rollout_adapter_identity(batch)
                if batch.context != context:
                    raise ValueError(
                        "Rollout batch context must match the current StepContext"
                    )
                batch.validate_lightweight(
                    strict=self.config.runner.strict_rollout_validation
                )

                reward_started = time.perf_counter()
                try:
                    rewards = self.reward_executor.score(batch, context)
                except RewardExecutionError as exc:
                    if (
                        self.reward_executor.mode == "sync"
                        and exc.__cause__ is not None
                    ):
                        raise exc.__cause__ from exc
                    raise
                reward_time = time.perf_counter() - reward_started
                if not isinstance(rewards, RewardBatch):
                    raise TypeError(
                        "FeedbackProvider.score must return a RewardBatch, got "
                        f"{type(rewards).__name__}"
                    )
                rewards = rewards.canonical()
                rewards.validate_against(batch)
                if not bool(rewards.valid_mask.all()):
                    raise RuntimeError(
                        f"Reward failure at step {step}: {rewards.metadata}"
                    )
                cache_started = time.perf_counter()
                cache_paths = self.rollout_cache.save(step, batch, rewards)
                cache_time = time.perf_counter() - cache_started
                update_started = time.perf_counter()
                plugin_metrics = self.optimizer_plugin.step(
                    adapter=self.adapter,
                    batch=batch,
                    rewards=rewards,
                    optimizer=self.optimizer,
                    context=context,
                )
                update_time = time.perf_counter() - update_started
                self._assert_runtime_integrity(f"after training step {step}")
                metrics = {"step": step, **plugin_metrics}

                staged_checkpoint_path = None
                checkpoint_path = None
                checkpoint_metadata = None
                checkpoint_started = None
                checkpoint_write_time = 0.0
                should_checkpoint = (
                    step + 1
                ) % save_every == 0 or step + 1 == target_steps
                if should_checkpoint:
                    checkpoint_started = time.perf_counter()
                    staged_checkpoint_path, checkpoint_metadata = self._save_checkpoint(
                        step + 1, transaction
                    )
                    checkpoint_write_time = time.perf_counter() - checkpoint_started
                    checkpoint_path = self.output_dir / f"checkpoint_{step + 1:06d}"

                metrics.update(
                    {
                        "rollout_time_s": rollout_time,
                        "reward_time_s": reward_time,
                        "rollout_cache_time_s": cache_time,
                        "update_time_s": update_time,
                        "checkpoint_write_time_s": checkpoint_write_time,
                        "peak_gpu_memory_bytes": self._peak_gpu_memory_bytes(),
                        **self._reward_runtime_metrics(rewards),
                    }
                )

                artifact_stage_started = time.perf_counter()
                self.artifacts.stage_step(
                    transaction,
                    step=step,
                    batch=batch,
                    rewards=rewards,
                    metrics=metrics,
                    media_type=self._media_type(),
                    rollout_type=self.config.sample.name,
                    media_paths=cache_paths.get("media_path"),
                    rollout_cache_path=cache_paths.get("rollout_cache_path"),
                    checkpoint_path=checkpoint_path,
                )
                artifact_stage_completed = time.perf_counter()
                artifact_stage_time = artifact_stage_completed - artifact_stage_started
                step_time = max(artifact_stage_completed - step_started, 1e-12)
                metrics.update(
                    {
                        "artifact_stage_time_s": artifact_stage_time,
                        "step_time_s": step_time,
                        "samples_per_second": float(batch.batch_size / step_time),
                    }
                )
                if not should_checkpoint:
                    metrics.update(
                        {
                            "artifact_commit_time_s": 0.0,
                            "post_commit_bookkeeping_time_s": 0.0,
                            "checkpoint_time_s": 0.0,
                        }
                    )
                cycle_sample_count += int(batch.batch_size)
                pending_metrics.append(metrics)
                if should_checkpoint:
                    if checkpoint_metadata is None or staged_checkpoint_path is None:
                        raise RuntimeError(
                            "checkpoint cycle ended without staged state"
                        )
                    artifact_commit_started = time.perf_counter()
                    marker = self.artifacts.commit(
                        transaction,
                        completed_steps=step + 1,
                        checkpoint_path=staged_checkpoint_path,
                    )
                    artifact_commit_time = time.perf_counter() - artifact_commit_started
                    transaction = None
                    post_commit_started = time.perf_counter()
                    self._run_post_commit_bookkeeping(
                        "latest_projection",
                        lambda: self._commit_checkpoint(
                            checkpoint_path,
                            checkpoint_metadata,
                            marker,
                        ),
                    )
                    self._run_post_commit_bookkeeping(
                        "retention",
                        self._apply_retention,
                    )
                    measured_boundary = time.perf_counter()
                    post_commit_time = measured_boundary - post_commit_started
                    step_time = max(measured_boundary - step_started, 1e-12)
                    if checkpoint_started is None or cycle_started is None:
                        raise RuntimeError(
                            "checkpoint timing boundary was not initialized"
                        )
                    cycle_time = max(measured_boundary - cycle_started, 1e-12)
                    metrics.update(
                        {
                            "artifact_commit_time_s": artifact_commit_time,
                            "post_commit_bookkeeping_time_s": post_commit_time,
                            "checkpoint_time_s": (
                                measured_boundary - checkpoint_started
                            ),
                            "step_time_s": step_time,
                            "samples_per_second": float(batch.batch_size / step_time),
                            "artifact_cycle_time_s": cycle_time,
                            "artifact_cycle_steps": len(pending_metrics),
                            "artifact_cycle_samples_per_second": float(
                                cycle_sample_count / cycle_time
                            ),
                        }
                    )
                    self._run_post_commit_bookkeeping(
                        "runtime_sidecar",
                        lambda: self.artifacts.record_commit_runtime(
                            marker,
                            {
                                int(row["step"]): self._commit_runtime_metrics(row)
                                for row in pending_metrics
                            },
                        ),
                    )
                    cycle_started = None
                    cycle_sample_count = 0
                    self.global_step = step + 1
                    for committed_metrics in pending_metrics:
                        all_metrics.append(committed_metrics)
                        self.progress.log_step(
                            committed_metrics,
                            total_steps=target_steps,
                        )
                        self._dispatch_callbacks(
                            "on_step_end",
                            self._callback_context(
                                metrics=committed_metrics,
                                checkpoint_path=(
                                    checkpoint_path
                                    if committed_metrics["step"] == step
                                    else None
                                ),
                                global_step=int(committed_metrics["step"]) + 1,
                            ),
                        )
                    pending_metrics.clear()
                    self._dispatch_callbacks(
                        "on_checkpoint",
                        self._callback_context(
                            metrics=metrics,
                            checkpoint_path=checkpoint_path,
                        ),
                    )
            self._run_evaluation()
        except BaseException as exc:
            primary_error = exc
            self._write_lifecycle_status(
                "failed",
                target_steps=target_steps,
                error=exc,
                owner_error=exc,
            )
            raise
        finally:
            cleanup_error: BaseException | None = None
            try:
                if transaction is not None:
                    try:
                        self.artifacts.abort(transaction)
                    except BaseException as abort_error:
                        if primary_error is not None:
                            add_note = getattr(primary_error, "add_note", None)
                            note = (
                                "artifact transaction was left for recovery after "
                                f"abort failed: {abort_error}"
                            )
                            if callable(add_note):
                                add_note(note)
                self._dispatch_callbacks("on_run_end", self._callback_context())
            except CallbackError as callback_error:
                if primary_error is None:
                    cleanup_error = callback_error
                else:
                    self._add_cleanup_note(
                        primary_error,
                        "on_run_end callback failed after the primary error: "
                        f"{callback_error}",
                        fallback_attribute="visual_rl_callback_note",
                    )
            finally:
                executor = self.reward_executor
                if executor is not None:
                    try:
                        executor.close()
                    except BaseException as close_error:
                        owner_error = primary_error or cleanup_error
                        if owner_error is None:
                            cleanup_error = close_error
                        else:
                            self._add_cleanup_note(
                                owner_error,
                                "reward executor close failed after the primary "
                                f"error: {close_error}",
                                fallback_attribute="visual_rl_executor_close_note",
                            )
                self.progress.close()
                self.artifacts.close()
                self.strategy.close()
            if primary_error is None and cleanup_error is not None:
                self._write_lifecycle_status(
                    "failed",
                    target_steps=target_steps,
                    error=cleanup_error,
                    owner_error=cleanup_error,
                )
                raise cleanup_error
        self._write_lifecycle_status("completed", target_steps=target_steps)
        return all_metrics

    def _run_distributed(
        self,
        max_steps: int | None,
    ) -> list[dict[str, Any]]:
        """Execute the native DDP path while rank zero owns all transactions."""

        def prepare_run() -> tuple[int, int, int]:
            target_steps = int(
                self.config.train.max_steps if max_steps is None else max_steps
            )
            if target_steps < self.start_step:
                raise ValueError(
                    f"max_steps={target_steps} is before resumed step {self.start_step}"
                )
            if self._run_started:
                raise RuntimeError("ExperimentRunner.run() may only be called once")
            batch_size = int(self.config.sample.batch_size)
            save_every = int(self.config.train.save_every or max(1, target_steps))
            return target_steps, batch_size, save_every

        target_steps, batch_size, save_every = self._run_distributed_phase(
            "run_setup",
            prepare_run,
        )
        manager = self._run_distributed_phase(
            "artifact_setup",
            lambda: self._artifact_manager() if self.rank == 0 else None,
        )
        all_metrics: list[dict[str, Any]] = []
        pending_metrics: list[dict[str, Any]] = []
        transaction: StepArtifactTransaction | None = None
        cycle_started: float | None = None
        cycle_sample_count = 0
        primary_error: BaseException | None = None
        self._run_started = True

        try:

            def acquire_writer() -> None:
                if manager is not None:
                    manager.acquire_writer_lock()
                    self._assert_run_head_unchanged()

            self._run_distributed_phase("commit", acquire_writer)
            self.reward_executor = self._run_distributed_phase(
                "reward",
                lambda: build_reward_executor(
                    self.feedback_provider,
                    self.config.runner.reward_executor,
                ),
            )
            if self.rank == 0:
                self.progress.start(target_steps, initial_step=self.start_step)
            self._run_distributed_phase(
                "commit",
                lambda: self._dispatch_callbacks(
                    "on_run_start", self._callback_context()
                ),
            )

            for step in range(self.start_step, target_steps):
                step_started = time.perf_counter()
                if transaction is None:
                    cycle_started = step_started
                    cycle_sample_count = 0
                    transaction = self._run_distributed_phase(
                        "commit",
                        lambda: (
                            manager.begin_transaction() if manager is not None else None
                        ),
                    )

                def prepare_step():
                    self._reset_peak_gpu_memory()
                    self._assert_runtime_integrity(f"before training step {step}")
                    prompts, metadata, epoch_tag = self.dataset.batch(
                        (step * self.world_size + self.rank) * batch_size,
                        batch_size,
                        epoch_tag=step,
                    )
                    context = StepContext(
                        step=step,
                        seed=self.distributed_context.step_seed(
                            self.config.seed,
                            step,
                        ),
                        epoch_tag=epoch_tag,
                        rank=self.rank,
                        world_size=self.world_size,
                        policy_version=step,
                    )
                    return prompts, metadata, context

                prompts, metadata, context = self._run_distributed_phase(
                    "step_setup",
                    prepare_step,
                )

                rollout_started = time.perf_counter()

                def sample_rollout():
                    self.adapter.prepare_for_sampling()
                    value = self.rollout.sample(
                        self.adapter,
                        prompts,
                        metadata,
                        context,
                    )
                    self._bind_rollout_adapter_identity(value)
                    if value.context != context:
                        raise ValueError(
                            "Rollout batch context must match the current StepContext"
                        )
                    value.validate_lightweight(
                        strict=self.config.runner.strict_rollout_validation
                    )
                    return value

                batch = self._run_distributed_phase("rollout", sample_rollout)
                rollout_time = time.perf_counter() - rollout_started

                reward_started = time.perf_counter()

                def score_rewards():
                    try:
                        value = self.reward_executor.score(batch, context)
                    except RewardExecutionError as exc:
                        if (
                            self.reward_executor.mode == "sync"
                            and exc.__cause__ is not None
                        ):
                            raise exc.__cause__ from exc
                        raise
                    if not isinstance(value, RewardBatch):
                        raise TypeError(
                            "FeedbackProvider.score must return a RewardBatch, got "
                            f"{type(value).__name__}"
                        )
                    value = value.canonical()
                    value.validate_against(batch)
                    if not bool(value.valid_mask.all()):
                        raise RuntimeError(
                            f"Reward failure at step {step}: {value.metadata}"
                        )
                    return value

                rewards = self._run_distributed_phase("reward", score_rewards)
                reward_time = time.perf_counter() - reward_started

                cache_started = time.perf_counter()
                cache_paths = self._run_distributed_phase(
                    "cache",
                    lambda: self.rollout_cache.save(step, batch, rewards),
                )
                cache_time = time.perf_counter() - cache_started

                update_started = time.perf_counter()
                validated_plugin_metrics: dict[str, Any] | None = None
                validated_reward_runtime_metrics: dict[str, float] | None = None
                validated_snapshot_metrics: dict[str, float] | None = None
                validated_peak_gpu_memory_bytes: float | None = None

                def run_plugin_update():
                    return self.optimizer_plugin.step(
                        adapter=self.adapter,
                        batch=batch,
                        rewards=rewards,
                        optimizer=self.optimizer,
                        context=context,
                        recompute_log_probs=self.strategy.recompute_log_probs,
                        gradient_sync_context=(
                            self.strategy.gradient_sync_context
                        ),
                        reduce_tensor_weighted_mean=(
                            self.strategy.reduce_tensor_weighted_mean
                        ),
                        synchronize_failure=self.strategy.synchronize_failure,
                    )

                def validate_plugin_result(result: Any):
                    nonlocal validated_peak_gpu_memory_bytes
                    nonlocal validated_plugin_metrics
                    nonlocal validated_reward_runtime_metrics
                    nonlocal validated_snapshot_metrics

                    self._assert_runtime_integrity(f"after training step {step}")
                    if not isinstance(result, Mapping):
                        raise TypeError(
                            "OptimizerPlugin.step() must return a metric mapping"
                        )
                    normalized_plugin_metrics = dict(result.items())
                    reward_runtime_metrics = self._reward_runtime_metrics(rewards)
                    snapshot_metrics = self._rollback_snapshot_runtime_metrics()
                    peak_gpu_memory_bytes = self._peak_gpu_memory_bytes()
                    reserved_metric_names = {
                        "step",
                        "sample_count",
                        "rollout_time_s",
                        "reward_time_s",
                        "rollout_cache_time_s",
                        "update_time_s",
                        "peak_gpu_memory_bytes",
                        *reward_runtime_metrics,
                        *snapshot_metrics,
                    }
                    collisions = sorted(
                        key
                        for key in normalized_plugin_metrics
                        if key in reserved_metric_names
                    )
                    if collisions:
                        raise ValueError(
                            "OptimizerPlugin.step() metrics collide with runner-owned "
                            f"metrics: {collisions}"
                        )
                    candidate_metrics = {
                        "step": step,
                        **normalized_plugin_metrics,
                        "sample_count": batch.batch_size,
                        "rollout_time_s": rollout_time,
                        "reward_time_s": reward_time,
                        "rollout_cache_time_s": cache_time,
                        "update_time_s": 0.0,
                        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                        **reward_runtime_metrics,
                        **snapshot_metrics,
                    }
                    contract = self.strategy.metric_contract(
                        candidate_metrics,
                        batch.batch_size,
                        reward_values=rewards.weighted_total,
                    )
                    validated_plugin_metrics = normalized_plugin_metrics
                    validated_reward_runtime_metrics = reward_runtime_metrics
                    validated_snapshot_metrics = snapshot_metrics
                    validated_peak_gpu_memory_bytes = peak_gpu_memory_bytes
                    return contract

                self._run_distributed_phase(
                    "update",
                    lambda: self.strategy.atomic_optimizer_step(
                        run_plugin_update,
                        parameters=list(self.adapter.parameters()),
                        optimizer=self.optimizer,
                        stateful=self.optimizer_plugin,
                        validate_result=validate_plugin_result,
                    ),
                )
                update_time = time.perf_counter() - update_started
                if (
                    validated_plugin_metrics is None
                    or validated_reward_runtime_metrics is None
                    or validated_snapshot_metrics is None
                    or validated_peak_gpu_memory_bytes is None
                ):
                    raise RuntimeError(
                        "distributed optimizer result validation lost local metrics"
                    )

                local_metrics = self._run_distributed_phase(
                    "metric_setup",
                    lambda: {
                        "step": step,
                        **validated_plugin_metrics,
                        "sample_count": batch.batch_size,
                        "rollout_time_s": rollout_time,
                        "reward_time_s": reward_time,
                        "rollout_cache_time_s": cache_time,
                        "update_time_s": update_time,
                        "peak_gpu_memory_bytes": validated_peak_gpu_memory_bytes,
                        **validated_reward_runtime_metrics,
                        **validated_snapshot_metrics,
                    },
                )
                reduced = self._run_distributed_phase(
                    "update",
                    lambda: self.strategy.reduce_metrics(
                        local_metrics,
                        batch.batch_size,
                        reward_values=rewards.weighted_total,
                    ),
                )
                metrics = {"step": step, **reduced}

                should_checkpoint = (
                    step + 1
                ) % save_every == 0 or step + 1 == target_steps
                checkpoint_path = (
                    self.output_dir / f"checkpoint_{step + 1:06d}"
                    if should_checkpoint
                    else None
                )
                staged_checkpoint_path = None
                checkpoint_metadata = None
                checkpoint_started: float | None = None
                checkpoint_write_time = 0.0
                if should_checkpoint:
                    checkpoint_started = time.perf_counter()
                    rank_state = self._run_distributed_phase(
                        "commit",
                        lambda: self._rank_runtime_state(step + 1, batch_size),
                    )
                    rank_states = self.strategy.gather_object(rank_state)

                    def save_checkpoint_on_main():
                        if manager is None:
                            return None
                        if transaction is None or rank_states is None:
                            raise RuntimeError(
                                "checkpoint save is missing rank runtime state"
                            )
                        return self._save_checkpoint(
                            step + 1,
                            transaction,
                            distributed_state={
                                "world_size": self.world_size,
                                "backend": self.distributed_context.backend,
                                "entries": rank_states,
                            },
                        )

                    checkpoint_result = self._run_distributed_phase(
                        "commit", save_checkpoint_on_main
                    )
                    if self.rank == 0:
                        staged_checkpoint_path, checkpoint_metadata = checkpoint_result
                        checkpoint_write_time = time.perf_counter() - checkpoint_started
                metrics["checkpoint_write_time_s"] = checkpoint_write_time

                artifact_stage_started = time.perf_counter()
                local_records = self._run_distributed_phase(
                    "commit",
                    lambda: ManifestBuilder(self.config.run_name).build_records(
                        step=step,
                        batch=batch,
                        rewards=rewards,
                        media_type=self._media_type(),
                        rollout_type=self.config.sample.name,
                        media_paths=cache_paths.get("media_path"),
                        rollout_cache_path=cache_paths.get("rollout_cache_path"),
                        checkpoint_path=checkpoint_path,
                    ),
                )
                records_by_rank = self.strategy.gather_object(local_records)

                def stage_records_on_main() -> None:
                    if manager is None:
                        return
                    if transaction is None or records_by_rank is None:
                        raise RuntimeError("artifact staging is missing rank records")
                    records = [
                        record
                        for rank_records in records_by_rank
                        for record in rank_records
                    ]
                    manager.stage_records(
                        transaction,
                        step=step,
                        records=records,
                        metrics=metrics,
                    )

                self._run_distributed_phase("commit", stage_records_on_main)
                artifact_stage_completed = time.perf_counter()
                if self.rank == 0:
                    step_time = max(
                        artifact_stage_completed - step_started,
                        1e-12,
                    )
                    metrics.update(
                        {
                            "artifact_stage_time_s": (
                                artifact_stage_completed - artifact_stage_started
                            ),
                            "step_time_s": step_time,
                            "samples_per_second": float(
                                metrics["sample_count"] / step_time
                            ),
                        }
                    )
                if not should_checkpoint:
                    metrics.update(
                        {
                            "artifact_commit_time_s": 0.0,
                            "post_commit_bookkeeping_time_s": 0.0,
                            "checkpoint_time_s": 0.0,
                        }
                    )
                cycle_sample_count += int(metrics["sample_count"])
                pending_metrics.append(metrics)

                if should_checkpoint:

                    def commit_on_main() -> None:
                        nonlocal transaction
                        if manager is None:
                            return
                        if (
                            transaction is None
                            or staged_checkpoint_path is None
                            or checkpoint_metadata is None
                            or checkpoint_path is None
                            or checkpoint_started is None
                            or cycle_started is None
                        ):
                            raise RuntimeError(
                                "checkpoint cycle ended without staged state"
                            )
                        commit_started = time.perf_counter()
                        marker = manager.commit(
                            transaction,
                            completed_steps=step + 1,
                            checkpoint_path=staged_checkpoint_path,
                        )
                        commit_time = time.perf_counter() - commit_started
                        transaction = None
                        bookkeeping_started = time.perf_counter()
                        self._run_post_commit_bookkeeping(
                            "latest_projection",
                            lambda: self._commit_checkpoint(
                                checkpoint_path,
                                checkpoint_metadata,
                                marker,
                            ),
                        )
                        self._run_post_commit_bookkeeping(
                            "retention",
                            self._apply_retention,
                        )
                        boundary = time.perf_counter()
                        step_time = max(boundary - step_started, 1e-12)
                        cycle_time = max(boundary - cycle_started, 1e-12)
                        metrics.update(
                            {
                                "artifact_commit_time_s": commit_time,
                                "post_commit_bookkeeping_time_s": (
                                    boundary - bookkeeping_started
                                ),
                                "checkpoint_time_s": boundary - checkpoint_started,
                                "step_time_s": step_time,
                                "samples_per_second": float(
                                    metrics["sample_count"] / step_time
                                ),
                                "artifact_cycle_time_s": cycle_time,
                                "artifact_cycle_steps": len(pending_metrics),
                                "artifact_cycle_samples_per_second": float(
                                    cycle_sample_count / cycle_time
                                ),
                            }
                        )
                        self._run_post_commit_bookkeeping(
                            "runtime_sidecar",
                            lambda: manager.record_commit_runtime(
                                marker,
                                {
                                    int(row["step"]): self._commit_runtime_metrics(row)
                                    for row in pending_metrics
                                },
                            ),
                        )

                    self._run_distributed_phase("commit", commit_on_main)
                    committed_rows = self.strategy.broadcast_object(
                        [dict(row) for row in pending_metrics]
                        if self.rank == 0
                        else None
                    )
                    self.global_step = step + 1
                    all_metrics.extend(committed_rows)

                    def publish_on_main() -> None:
                        if self.rank != 0:
                            return
                        for row in committed_rows:
                            self.progress.log_step(row, total_steps=target_steps)
                            self._dispatch_callbacks(
                                "on_step_end",
                                self._callback_context(
                                    metrics=row,
                                    checkpoint_path=(
                                        checkpoint_path
                                        if int(row["step"]) == step
                                        else None
                                    ),
                                    global_step=int(row["step"]) + 1,
                                ),
                            )
                        self._dispatch_callbacks(
                            "on_checkpoint",
                            self._callback_context(
                                metrics=committed_rows[-1],
                                checkpoint_path=checkpoint_path,
                            ),
                        )

                    self._run_distributed_phase("commit", publish_on_main)
                    pending_metrics.clear()
                    cycle_started = None
                    cycle_sample_count = 0

            self._run_distributed_phase("commit", self._run_evaluation)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            if transaction is not None and manager is not None:
                try:
                    manager.abort(transaction)
                except BaseException as abort_error:
                    if primary_error is None:
                        cleanup_error = abort_error
                    else:
                        self._add_cleanup_note(
                            primary_error,
                            f"artifact abort failed after primary error: {abort_error}",
                            fallback_attribute="visual_rl_artifact_abort_note",
                        )
            try:
                self._dispatch_callbacks("on_run_end", self._callback_context())
            except CallbackError as callback_error:
                if primary_error is None:
                    cleanup_error = callback_error
                else:
                    self._add_cleanup_note(
                        primary_error,
                        "on_run_end callback failed after the primary error: "
                        f"{callback_error}",
                        fallback_attribute="visual_rl_callback_note",
                    )
            executor = self.reward_executor
            if executor is not None:
                try:
                    executor.close()
                except BaseException as close_error:
                    owner = primary_error or cleanup_error
                    if owner is None:
                        cleanup_error = close_error
                    else:
                        self._add_cleanup_note(
                            owner,
                            "reward executor close failed after the primary error: "
                            f"{close_error}",
                            fallback_attribute="visual_rl_executor_close_note",
                        )
            if self.rank == 0:
                self.progress.close()
                if manager is not None:
                    manager.close()
            self.strategy.close()
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error
        return all_metrics

    def _run_post_commit_bookkeeping(self, operation: str, callback):
        """Treat derived work as recoverable once the marker is durable."""

        try:
            return callback()
        except Exception as exc:
            self.post_commit_bookkeeping_errors.append(
                {
                    "operation": operation,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

    def _run_distributed_phase(self, phase: str, operation):
        if self.world_size == 1:
            return operation()
        try:
            result = operation()
        except DistributedFailureError as exc:
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"distributed phase: {phase}")
            raise
        except BaseException as exc:
            failure: BaseException | None = exc
            result = None
        else:
            failure = None
        try:
            self.strategy.synchronize_failure(failure)
        except DistributedFailureError as exc:
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"distributed phase: {phase}")
            raise
        return result

    def _rank_runtime_state(
        self,
        completed_steps: int,
        prompt_batch_size: int,
    ) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rng": capture_rng_state(),
            "sampler_cursor": {
                "completed_steps": completed_steps,
                "next_start": (completed_steps * self.world_size + self.rank)
                * prompt_batch_size,
            },
            "runtime_identity": {
                "runtime": dict(self.runtime_identity),
                "rank": self.rank,
                "world_size": self.world_size,
                "backend": self.distributed_context.backend,
                "device": str(self.distributed_context.device),
            },
        }

    def _artifact_manager(self) -> ArtifactManager:
        if not isinstance(self.artifacts, ArtifactManager):
            raise RuntimeError("Only rank zero owns the ArtifactManager")
        return self.artifacts

    def _rank_cache_path(self, root: Path) -> Path:
        if self.world_size == 1:
            return root
        return root / f"rank_{self.rank:04d}"

    def _setup_shared_output_dir(self) -> Path:
        if self.world_size == 1:
            return self._setup_output_dir()
        payload: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                payload = {"path": str(self._setup_output_dir()), "error": None}
            except BaseException as exc:
                payload = {
                    "path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        payload = self.strategy.broadcast_object(payload)
        if payload["error"] is not None:
            raise RuntimeError(
                "Rank zero could not create the distributed output directory: "
                f"{payload['error']}"
            )
        output_dir = Path(payload["path"])
        self.config.paths.output_dir = str(output_dir)
        return output_dir

    def _read_resume_if_requested(self):
        resume_from = self.config.paths.resume_from
        if not resume_from:
            return None
        try:
            checkpoint_dir, declared_step = self._resolve_resume_checkpoint(resume_from)
            self._assert_runtime_integrity("before checkpoint validation")
            validated = read_and_validate_training_state(
                checkpoint_dir,
                config=config_to_dict(self.config),
                implementation=self.checkpoint_identity,
                trusted_root=checkpoint_dir.parent,
                expected_world_size=self.world_size,
                expected_rank=self.rank,
            )
            if validated.step != declared_step:
                raise RuntimeError(
                    "Checkpoint step mismatch: "
                    f"path says {declared_step}, state says {validated.step}"
                )
            self.adapter.load_checkpoint(str(checkpoint_dir))
            self._assert_runtime_integrity("after adapter checkpoint load")
            self._resume_checkpoint_path = validated.checkpoint_dir
            self._resume_metadata = dict(validated.metadata)
            return validated
        except ResumeError:
            raise
        except Exception as exc:
            raise ResumeError(f"Cannot resume from {resume_from!r}: {exc}") from exc

    @staticmethod
    def _add_cleanup_note(
        error: BaseException,
        note: str,
        *,
        fallback_attribute: str,
    ) -> None:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:
            setattr(error, fallback_attribute, note)

    def _write_lifecycle_status(
        self,
        state: str,
        *,
        target_steps: int,
        error: BaseException | None = None,
        owner_error: BaseException | None = None,
    ) -> None:
        """Publish marker-aware lifecycle state without leaking exception text."""

        try:
            write_run_status(
                self.output_dir / "run_status.json",
                {
                    "state": state,
                    "run_id": self.config.run_name,
                    "pid": os.getpid(),
                    "target_steps": int(target_steps),
                    "start_step": int(self.start_step),
                    "started_from_step": int(self.start_step),
                    "world_size": int(self.world_size),
                },
                rank=self.rank,
                exception=error,
            )
        except Exception as status_error:
            if owner_error is None:
                raise
            self._add_cleanup_note(
                owner_error,
                "run_status.json update failed after the primary error: "
                f"{type(status_error).__name__}",
                fallback_attribute="visual_rl_status_note",
            )

    def _callback_context(
        self,
        *,
        metrics: dict[str, Any] | None = None,
        checkpoint_path: Path | None = None,
        global_step: int | None = None,
    ) -> CallbackContext:
        artifacts: dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "metrics_path": str(self.artifacts.metric_path),
            "manifest_path": str(self.artifacts.manifest_path),
        }
        if checkpoint_path is not None:
            artifacts["checkpoint_path"] = str(checkpoint_path)
        if self.evaluation_paths:
            artifacts["evaluation_paths"] = {
                name: str(path) for name, path in self.evaluation_paths.items()
            }
        return CallbackContext(
            run_id=self.artifacts.run_id,
            output_dir=self.output_dir,
            global_step=self.global_step if global_step is None else global_step,
            rank=self.rank,
            world_size=self.world_size,
            metrics=metrics or {},
            artifacts=artifacts,
        )

    def _dispatch_callbacks(self, method_name: str, context: CallbackContext) -> None:
        if self.rank != 0:
            return
        for callback in self.callbacks:
            try:
                getattr(callback, method_name)(context)
            except BaseException as exc:
                raise CallbackError(
                    f"{type(callback).__name__}.{method_name} failed"
                ) from exc

    def _run_evaluation(self) -> None:
        if self.rank != 0:
            return
        if self.evaluator is None:
            return
        if not self._evaluation_prompts:
            raise ValueError("evaluator requires non-empty held-out evaluation prompts")

        evaluation_name = str(getattr(self.evaluator, "name", "default"))
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", evaluation_name):
            raise ValueError(f"invalid evaluator name: {evaluation_name!r}")
        evaluation_dir = self.output_dir / "evaluation" / evaluation_name
        context = EvaluationContext(
            run_id=self.artifacts.run_id,
            output_dir=self.output_dir,
            evaluation_dir=evaluation_dir,
            name=evaluation_name,
            split_name=self.config.evaluation.split_name,
            content_sha256=str(self.config.evaluation.content_sha256),
            seeds=tuple(self.config.evaluation.seeds),
            prompt_count=len(self._evaluation_prompts),
            rank=self.rank,
            world_size=self.world_size,
        )
        result = self._evaluate_with_preserved_state(context)
        if not isinstance(result, EvaluationResult):
            raise TypeError("Evaluator.evaluate() must return an EvaluationResult")
        if result.name is not None and result.name != evaluation_name:
            raise ValueError(
                "EvaluationResult.name must match the evaluator name when provided"
            )
        result = replace(result, name=evaluation_name)
        result_path = self._write_evaluation_result(context, result)
        self.evaluation_results = (*self.evaluation_results, result)
        self.evaluation_paths[evaluation_name] = result_path

    def _evaluate_with_preserved_state(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        import numpy as np
        import torch

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        was_training = bool(getattr(self.adapter.train_module, "training", False))
        try:
            self.adapter.eval()
            with torch.inference_mode():
                return self.evaluator.evaluate(
                    adapter=self.adapter,
                    prompts=tuple(self._evaluation_prompts),
                    context=context,
                )
        finally:
            self.adapter.train(was_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _write_evaluation_result(
        context: EvaluationContext,
        result: EvaluationResult,
    ) -> Path:
        context.evaluation_dir.mkdir(parents=True, exist_ok=True)
        result_path = context.evaluation_dir / "result.json"
        payload = result.to_dict()
        payload["context"] = {
            "run_id": context.run_id,
            "split_name": context.split_name,
            "content_sha256": context.content_sha256,
            "seeds": list(context.seeds),
            "rank": context.rank,
            "world_size": context.world_size,
            "prompt_count": context.prompt_count,
        }
        temp_path = context.evaluation_dir / f".result.tmp-{uuid.uuid4().hex}.json"
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(result_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return result_path

    def _save_checkpoint(
        self,
        completed_steps: int,
        transaction: StepArtifactTransaction,
        *,
        distributed_state: dict[str, Any] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        self._assert_runtime_integrity("before checkpoint save")
        staging_path = transaction.staging_dir / (f"checkpoint_{completed_steps:06d}")
        if staging_path.exists():
            raise FileExistsError(f"staged checkpoint already exists: {staging_path}")
        self.adapter.save_pretrained(str(staging_path))
        if not staging_path.exists() or not any(
            path.is_file() for path in staging_path.rglob("*")
        ):
            raise RuntimeError(
                f"Adapter {self.adapter.name!r} did not write checkpoint files"
            )
        metadata = save_training_state(
            staging_path,
            optimizer=self.optimizer,
            plugin=self.optimizer_plugin,
            step=completed_steps,
            config=config_to_dict(self.config),
            implementation=self.checkpoint_identity,
            distributed_state=distributed_state,
        )
        return staging_path, metadata

    def _commit_checkpoint(
        self,
        checkpoint_path: Path,
        metadata: dict[str, Any],
        marker: dict[str, Any] | None = None,
    ) -> None:
        save_json(
            self.output_dir / "latest.json",
            {
                "step": int(metadata["step"]),
                "checkpoint": checkpoint_path.name,
                "config_fingerprint": metadata["config_fingerprint"],
                "config_fingerprint_version": metadata.get(
                    "config_fingerprint_version",
                    1,
                ),
                "commit": (
                    f"commits/commit_{int(metadata['step']):06d}.json"
                    if marker is not None
                    else None
                ),
            },
        )

    def _apply_retention(self) -> None:
        runner = self.config.runner
        audit = self.artifacts.apply_retention(
            checkpoint_keep_last=runner.checkpoint_keep_last,
            rollout_cache_keep_last=runner.rollout_cache_keep_last,
            rollout_cache_max_bytes=runner.rollout_cache_max_bytes,
            artifact_max_bytes=runner.artifact_max_bytes,
            rollout_root=self.rollout_cache_root,
        )
        if not audit:
            return
        self.retention_audit.extend(audit)
        save_json(
            self.output_dir / "retention_audit.json",
            {
                "schema_version": "1",
                "events": self.retention_audit,
            },
        )

    @staticmethod
    def _commit_runtime_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        names = (
            "checkpoint_write_time_s",
            "artifact_stage_time_s",
            "artifact_commit_time_s",
            "post_commit_bookkeeping_time_s",
            "checkpoint_time_s",
            "step_time_s",
            "samples_per_second",
            "artifact_cycle_time_s",
            "artifact_cycle_steps",
            "artifact_cycle_samples_per_second",
        )
        return {name: metrics[name] for name in names if name in metrics}

    @staticmethod
    def _reward_runtime_metrics(rewards: RewardBatch) -> dict[str, float]:
        metrics: dict[str, float] = {}
        runtime = rewards.metadata.get("_runtime")
        if runtime is not None:
            if not isinstance(runtime, Mapping):
                raise TypeError("RewardBatch metadata['_runtime'] must be a mapping")
            latencies = runtime.get("reward_latencies_s", [])
            if not isinstance(latencies, list):
                raise TypeError(
                    "RewardBatch metadata['_runtime']['reward_latencies_s'] "
                    "must be a list"
                )
            ordered = sorted(
                ExperimentRunner._finite_metric(
                    value,
                    "RewardBatch metadata['_runtime']['reward_latencies_s']",
                )
                for value in latencies
            )
            metrics.update(
                {
                    "reward_cache_hit_rate": ExperimentRunner._finite_metric(
                        runtime.get("cache_hit_rate", 0.0),
                        "RewardBatch metadata['_runtime']['cache_hit_rate']",
                    ),
                    "reward_cache_hits": ExperimentRunner._finite_metric(
                        runtime.get("cache_hits", 0),
                        "RewardBatch metadata['_runtime']['cache_hits']",
                    ),
                    "reward_cache_misses": ExperimentRunner._finite_metric(
                        runtime.get("cache_misses", 0),
                        "RewardBatch metadata['_runtime']['cache_misses']",
                    ),
                }
            )
            if ordered:
                metrics["reward_latency_p50_s"] = ordered[(len(ordered) - 1) // 2]
                metrics["reward_latency_p95_s"] = ordered[
                    min(len(ordered) - 1, int(0.95 * len(ordered)))
                ]

        executor = rewards.metadata.get("_executor")
        if executor is not None:
            if not isinstance(executor, Mapping):
                raise TypeError("RewardBatch metadata['_executor'] must be a mapping")
            for name, value in sorted(executor.items()):
                if name in {"mode", "timeout_scope"}:
                    if not isinstance(value, str):
                        raise TypeError(
                            f"RewardBatch metadata['_executor']['{name}'] "
                            "must be a string"
                        )
                    continue
                if name == "provider_metadata":
                    continue
                metrics[f"reward_executor_{name}"] = ExperimentRunner._finite_metric(
                    value,
                    f"RewardBatch metadata['_executor']['{name}']",
                )
        return metrics

    def _rollback_snapshot_runtime_metrics(self) -> dict[str, float]:
        snapshot = getattr(self.strategy, "last_atomic_snapshot_metrics", None)
        if snapshot is None:
            return {}
        if not isinstance(snapshot, Mapping):
            raise TypeError("atomic optimizer snapshot metrics must be a mapping")

        def non_negative(name: str) -> float:
            if name not in snapshot:
                raise ValueError(
                    f"atomic optimizer snapshot metrics are missing {name!r}"
                )
            value = self._finite_metric(
                snapshot[name],
                f"atomic optimizer snapshot metric {name!r}",
            )
            if value < 0:
                raise ValueError(
                    f"atomic optimizer snapshot metric {name!r} must be non-negative"
                )
            return value

        metrics = {
            "peak_rollback_snapshot_tensor_bytes": non_negative("total_tensor_bytes"),
            "rollback_snapshot_capture_time_s": non_negative("capture_time_s"),
        }
        if "restore_time_s" in snapshot:
            metrics["rollback_snapshot_restore_time_s"] = non_negative("restore_time_s")
        return metrics

    @staticmethod
    def _finite_metric(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be numeric")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{label} must be finite")
        return resolved

    def _cuda_device(self):
        import torch

        try:
            device = torch.device(getattr(self.adapter, "device", "cpu") or "cpu")
        except (TypeError, RuntimeError):
            return None
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        return device

    def _reset_peak_gpu_memory(self) -> None:
        import torch

        device = self._cuda_device()
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)

    def _peak_gpu_memory_bytes(self) -> float:
        import torch

        device = self._cuda_device()
        if device is None:
            return 0.0
        return float(torch.cuda.max_memory_allocated(device))

    def _bind_rollout_adapter_identity(self, batch: RolloutBatch) -> None:
        """Bind artifact identity to the configured registry key.

        Adapter implementations may expose a descriptive ``name`` that differs
        from the registry key selected by configuration.  Artifacts need both:
        ``adapter`` remains the implementation label, while ``adapter_key`` is
        the stable configuration identity used by resume and audit.
        """

        expected = self.config.model.name
        recorded = batch.model_metadata.get("adapter_key")
        if recorded is not None and recorded != expected:
            raise ValueError(
                "Rollout adapter_key must match the configured model adapter: "
                f"{recorded!r} != {expected!r}"
            )
        batch.model_metadata["adapter_key"] = expected

    def _validate_config(self) -> None:
        errors = []
        if not self.config.rewards.weights:
            errors.append("rewards.weights cannot be empty")
        if self.config.sample.sde_window_range and self.config.sample.sde_window_size:
            start, end = self.config.sample.sde_window_range
            if start < 0 or end < start:
                errors.append(
                    "sample.sde_window_range must be [start, end] with 0 <= start <= end"
                )
            if end > self.config.sample.num_steps:
                errors.append("sample.sde_window_range cannot exceed sample.num_steps")
            if end - start < self.config.sample.sde_window_size:
                errors.append(
                    "sample.sde_window_range span must be >= sample.sde_window_size"
                )
        if errors:
            raise ValueError("; ".join(errors))

    def _artifact_history_exists(self) -> bool:
        return (
            self.output_dir / "sample_manifest.json"
        ).is_file() or self._has_commit_markers()

    def _has_commit_markers(self) -> bool:
        commits_dir = self.output_dir / "commits"
        return (
            commits_dir.is_dir()
            and not commits_dir.is_symlink()
            and any(
                path.is_file() and not path.is_symlink()
                for path in commits_dir.glob("commit_*.json")
            )
        )

    def _persist_scaling_trigger_decision(self) -> None:
        decision_path = self.output_dir / "trigger_decision.json"
        if (
            self.config.paths.resume_from
            and self._resume_scaling_trigger_decision != self.scaling_trigger_decision
        ):
            raise ResumeError(
                "Validated resume scaling trigger decision was lost before "
                "artifact initialization"
            )
        save_json(decision_path, self.scaling_trigger_decision)

    def _read_resume_scaling_trigger_decision(
        self,
        resume_from: str | Path,
    ) -> dict[str, Any]:
        """Validate the source run decision before runtime or output mutation."""

        try:
            checkpoint_dir, recovery_step = resolve_resume_checkpoint(resume_from)
            committed_head = latest_committed_step(checkpoint_dir.parent)
            destination = (
                Path(self.config.paths.output_dir).absolute().resolve(strict=False)
            )
            source_run = checkpoint_dir.parent.resolve(strict=True)
            if (
                committed_head is not None
                and committed_head > recovery_step
                and destination == source_run
            ):
                raise ResumeError(
                    "The newest committed checkpoint is unavailable; in-place "
                    "fallback would overlap authoritative commits. Resume the "
                    "older checkpoint into a new output_dir instead."
                )
            persisted = _load_scaling_trigger_decision(
                checkpoint_dir.parent / "trigger_decision.json"
            )
        except FileNotFoundError as exc:
            raise ResumeError(
                "Resume requires the source scaling trigger decision"
            ) from exc
        except ResumeError:
            raise
        except Exception as exc:
            raise ResumeError(
                f"Cannot validate the source scaling trigger decision: {exc}"
            ) from exc
        if persisted != self.scaling_trigger_decision:
            raise ResumeError(
                "Resume scaling trigger decision does not match the current "
                "evidence-gated decision"
            )
        return persisted

    def _validate_resume_head(self) -> None:
        if not self._has_commit_markers():
            return
        if not self.config.paths.resume_from:
            raise ResumeError(
                "Output directory already contains committed artifacts; "
                "set paths.resume_from to continue the run"
            )
        authoritative_path, authoritative_step = resolve_resume_checkpoint(
            self.output_dir
        )
        if (
            self._resume_checkpoint_path is None
            or self.start_step != authoritative_step
            or self._resume_checkpoint_path.resolve() != authoritative_path.resolve()
        ):
            raise ResumeError(
                "In-place resume must use the latest committed checkpoint; "
                "use a new output_dir to branch from an older checkpoint"
            )

    def _rewrite_legacy_latest_if_local(self) -> None:
        checkpoint = self._resume_checkpoint_path
        metadata = self._resume_metadata
        if checkpoint is None or metadata is None:
            return
        if checkpoint.parent.resolve() == self.output_dir.resolve():
            self._commit_checkpoint(checkpoint, metadata)

    def _assert_run_head_unchanged(self) -> None:
        recovery_audit = self.artifacts.recover()
        if recovery_audit:
            self.artifact_recovery_audit.extend(recovery_audit)
        if self._has_commit_markers():
            if not self.config.paths.resume_from:
                raise ResumeError(
                    "Committed artifacts appeared after runner construction; "
                    "refusing a second writer"
                )
            current_path, current_step = resolve_resume_checkpoint(self.output_dir)
        elif self.config.paths.resume_from:
            current_path, current_step = resolve_resume_checkpoint(
                self.config.paths.resume_from
            )
        else:
            return
        if (
            self._resume_checkpoint_path is None
            or current_step != self.start_step
            or current_path.resolve() != self._resume_checkpoint_path.resolve()
        ):
            raise ResumeError(
                "Resume head changed after runner construction; construct a new "
                "ExperimentRunner from the current committed checkpoint"
            )

    def _setup_output_dir(self) -> Path:
        base = Path(self.config.paths.output_dir)
        if self.config.runner.deterministic_run_dir:
            output_dir = base
        else:
            stamp = _datetime.datetime.now(tz=_datetime.timezone.utc).strftime(
                "%Y%m%d_%H%M%S"
            )
            output_dir = base / f"{self.config.run_name}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.output_dir = str(output_dir)
        return output_dir

    def _media_type(self) -> str:
        media_type = getattr(self.adapter, "media_type", None)
        if media_type not in {"image", "video"}:
            raise ValueError(
                f"Adapter {self.adapter.name} must declare media_type as 'image' or 'video'."
            )
        return str(media_type)

    def _validate_evaluation_data_identity(self) -> tuple[str, ...]:
        path = self.config.evaluation.path
        inline_prompts = self.config.evaluation.prompts
        if path and inline_prompts:
            raise ValueError("evaluation config cannot provide both prompts and path")
        if path:
            try:
                prompts = read_prompt_file(path)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot validate evaluation data source {path!r}: {exc}"
                ) from exc
        elif inline_prompts:
            prompts = list(inline_prompts)
        else:
            return ()
        if not prompts:
            raise ValueError("evaluation prompt source is empty")
        actual = prompt_content_sha256(prompts)
        declared = self.config.evaluation.content_sha256
        if declared and str(declared) != actual:
            raise RuntimeError(
                "evaluation content SHA256 mismatch: "
                f"actual {actual} != declared {declared}"
            )
        self.config.evaluation.content_sha256 = actual
        try:
            validate_prompt_splits(
                self.dataset.source_prompts,
                prompts,
                train_path=self.config.dataset.path,
                heldout_path=path,
            )
        except ValueError as exc:
            raise RuntimeError(f"Invalid held-out evaluation prompts: {exc}") from exc
        max_prompts = self.config.evaluation.max_prompts
        if max_prompts is not None:
            prompts = prompts[: int(max_prompts)]
        return tuple(prompts)

    def _assert_runtime_integrity(self, context: str) -> None:
        assert_runtime(self.runtime_identity, context=context)

    @staticmethod
    def _resolve_resume_checkpoint(resume_from: str | Path) -> tuple[Path, int]:
        return resolve_resume_checkpoint(resume_from)

    @staticmethod
    def _resolved_model_config(config: VisualRLConfig) -> dict[str, Any]:
        model_config = section_to_dict(config.model)
        model_config.setdefault("use_lora", config.use_lora)
        if config.train.lora_path is not None:
            model_config["lora_path"] = config.train.lora_path
        if config.paths.pretrained_model and not model_config.get("model_path"):
            model_config["model_path"] = config.paths.pretrained_model
        return model_config

    @staticmethod
    def _validate_optimizer_contract(optimizer: Any) -> None:
        required = ("zero_grad", "step", "state_dict", "load_state_dict")
        missing = [
            name for name in required if not callable(getattr(optimizer, name, None))
        ]
        if missing:
            raise TypeError(
                "OptimizerPlugin.build_optimizer() returned an object missing "
                f"required methods: {missing}"
            )
