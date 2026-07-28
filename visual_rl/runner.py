"""The sole config-driven VisualRL training runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from pathlib import Path
import random
from typing import Any, TypeVar
import uuid

from visual_rl.api_types import RunResult
from visual_rl.configs.schema import VisualRLConfig
from visual_rl.core.types import (
    FrozenMapping,
    RuntimeBuildContext,
    StepContext,
    ValidatedRuntimeEnv,
)
from visual_rl.errors import ResumeError


_T = TypeVar("_T")
_CORE_UPDATE_METRICS = (
    "loss",
    "policy_loss",
    "reference_kl",
    "approx_kl",
    "clipfrac",
)


@dataclass(frozen=True)
class _ArtifactMetrics:
    """W03 bridge into the one artifact writer.

    W05 replaces this private staging value in-place with the frozen
    ``StepMetrics`` contract. It is deliberately not exported or accepted by
    any training component.
    """

    values: FrozenMapping
    sample_count: int
    active_transition_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, FrozenMapping):
            object.__setattr__(self, "values", FrozenMapping(self.values))
        for name in ("sample_count", "active_transition_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in self.values.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "artifact metric values must be named finite Python numbers"
                )


class ExperimentRunner:
    """Consume one validated config/environment snapshot and run it once.

    The public run-once guard belongs exclusively to ``visual_rl.api``.
    Constructing this internal runner directly is unsupported; it therefore
    performs no second config resolution, Preflight, or run-once bookkeeping.
    """

    def __init__(
        self,
        config: VisualRLConfig,
        validated_env: ValidatedRuntimeEnv,
    ) -> None:
        if not isinstance(config, VisualRLConfig):
            raise TypeError("config must be a VisualRLConfig")
        if not isinstance(validated_env, ValidatedRuntimeEnv):
            raise TypeError("validated_env must be a ValidatedRuntimeEnv")
        self.config = config
        self.validated_env = validated_env
        self.optimizer: Any | None = None
        self.scaler: Any | None = None

    def run(self) -> RunResult:
        """Execute the one shared single/DDP loop and return its committed head."""

        from visual_rl.artifacts.builder import ManifestBuilder
        from visual_rl.artifacts.checkpoint import (
            TrainingContract,
            apply_training_state,
            read_and_validate_training_state,
            save_training_state,
        )
        from visual_rl.artifacts.logging import TrainProgressPrinter
        from visual_rl.artifacts.manager import ArtifactManager
        from visual_rl.core.determinism import configure_deterministic_runtime
        from visual_rl.core.seed import seed_everything
        from visual_rl.distributed import build_strategy
        from visual_rl.runtime_factory import build_runtime_components

        strategy = None
        manager: ArtifactManager | None = None
        components = None
        optimizer = None
        progress = None
        transaction = None
        try:
            strategy = build_strategy(
                self.config.runtime.distributed,
                self.validated_env,
            )
            configure_deterministic_runtime(self.config.runtime.deterministic)
            seed_everything(self.config.run.seed + strategy.rank)

            def prepare_artifacts() -> tuple[str, int, str | None] | None:
                nonlocal manager
                if not strategy.is_main_process:
                    return None
                output_dir = self.config.artifacts.output_dir
                if self.config.resume.from_ is None:
                    run_id = f"run-{uuid.uuid4().hex}"
                    manager = ArtifactManager(
                        output_dir,
                        run_id,
                        config=self.config,
                    )
                else:
                    manager = ArtifactManager.open_resume(output_dir)
                    manager.recover()
                    run_id = manager.run_id
                checkpoint = manager.checkpoint_path
                return (
                    run_id,
                    manager.start_step,
                    None if checkpoint is None else str(checkpoint),
                )

            preparation = self._run_phase(
                strategy,
                "artifact_prepare",
                prepare_artifacts,
            )
            preparation = strategy.broadcast_object(preparation)
            if (
                not isinstance(preparation, tuple)
                or len(preparation) != 3
                or not isinstance(preparation[0], str)
                or type(preparation[1]) is not int
            ):
                raise RuntimeError("artifact preparation returned an invalid payload")
            run_id, start_step, checkpoint_text = preparation
            checkpoint_path = (
                None if checkpoint_text is None else Path(checkpoint_text)
            )
            target_steps = self.config.runtime.max_steps
            if target_steps < start_step:
                raise ResumeError(
                    "runtime.max_steps is behind the authoritative commit",
                    path=str(self.config.artifacts.output_dir),
                )
            if target_steps == start_step:
                local_result = self._run_phase(
                    strategy,
                    "noop_result",
                    lambda: (
                        self._build_run_result(manager)
                        if strategy.is_main_process
                        else None
                    ),
                )
                result = strategy.broadcast_object(local_result)
                if not isinstance(result, RunResult):
                    raise RuntimeError("no-op resume did not produce a RunResult")
                return result

            if self.config.resume.from_ is not None:
                self._run_phase(
                    strategy,
                    "config_projection",
                    lambda: (
                        manager.write_resolved_config(self.config)
                        if strategy.is_main_process and manager is not None
                        else None
                    ),
                )

            runtime_context = RuntimeBuildContext(
                rank=strategy.rank,
                local_rank=strategy.local_rank,
                world_size=strategy.world_size,
                backend=strategy.backend,
                device=strategy.device,
                precision=self.config.runtime.precision,
            )
            components = self._run_phase(
                strategy,
                "runtime_build",
                lambda: build_runtime_components(
                    self.config,
                    runtime_context,
                ),
            )
            self._run_phase(
                strategy,
                "model_prepare",
                lambda: strategy.prepare(components.model),
            )
            optimizer = self._run_phase(
                strategy,
                "optimizer_setup",
                lambda: components.optimizer_plugin.build_optimizer(
                    components.model.named_parameters(),
                    self.config.optimizer,
                ),
            )
            scaler = self._build_gradient_scaler(
                precision=self.config.runtime.precision,
                device=strategy.device,
            )
            self.optimizer = optimizer
            self.scaler = scaler
            training_contract = TrainingContract(
                algorithm=self.config.algorithm.name,
                version=(
                    components.optimizer_plugin.algorithm.TRAINING_CONTRACT_VERSION
                ),
            )

            def restore_training_state() -> None:
                if checkpoint_path is not None:
                    validated = read_and_validate_training_state(
                        checkpoint_path,
                        adapter=components.model,
                        optimizer=optimizer,
                        scaler=scaler,
                        expected_global_step=start_step,
                        expected_world_size=strategy.world_size,
                        expected_training_contract=training_contract,
                    )
                    apply_training_state(
                        validated,
                        adapter=components.model,
                        optimizer=optimizer,
                        scaler=scaler,
                        optimizer_config=self.config.optimizer,
                        rank=strategy.rank,
                    )
                elif self.config.model.adapter_checkpoint is not None:
                    path = self.config.model.adapter_checkpoint
                    components.model.validate_checkpoint(path)
                    components.model.load_checkpoint(path)

            self._run_phase(
                strategy,
                "training_state_restore",
                restore_training_state,
            )

            builder = ManifestBuilder(
                run_id=run_id,
                media_type=components.model.MEDIA_TYPE,
                rollout_type=self.config.rollout.name,
            )
            if strategy.is_main_process and self.config.runtime.progress:
                progress = TrainProgressPrinter(enabled=True)
                progress.start(target_steps, initial_step=start_step)

            checkpoint_every = self.config.artifacts.checkpoint_every
            for step in range(start_step, target_steps):
                def ensure_transaction() -> None:
                    nonlocal transaction
                    if strategy.is_main_process and transaction is None:
                        assert manager is not None
                        transaction = manager.begin_transaction()

                self._run_phase(
                    strategy,
                    "artifact_cycle",
                    ensure_transaction,
                )

                dataset_start = strategy.dataset_start(
                    step,
                    self.config.runtime.batch_size,
                )
                prompts, metadata = self._run_phase(
                    strategy,
                    "dataset",
                    lambda: components.dataset.batch(
                        dataset_start,
                        self.config.runtime.batch_size,
                    ),
                )
                step_context = StepContext(
                    step=step,
                    seed=(
                        self.config.run.seed
                        + step * strategy.world_size
                        + strategy.rank
                    ),
                    rank=strategy.rank,
                    world_size=strategy.world_size,
                )
                batch = self._run_phase(
                    strategy,
                    "rollout",
                    lambda: components.rollout.sample(
                        adapter=components.model,
                        prompts=prompts,
                        metadata=metadata,
                        context=step_context,
                    ),
                )
                rewards = self._run_phase(
                    strategy,
                    "reward",
                    lambda: components.reward_executor.score(
                        batch,
                        step_context,
                    ),
                )
                update_metrics = self._run_phase(
                    strategy,
                    "update",
                    lambda: components.optimizer_plugin.step(
                        batch=batch,
                        rewards=rewards,
                        optimizer=optimizer,
                        scaler=scaler,
                        context=step_context,
                        strategy=strategy,
                    ),
                )
                metrics = self._run_phase(
                    strategy,
                    "metrics",
                    lambda: self._reduce_artifact_metrics(
                        strategy,
                        update_result=update_metrics,
                        batch=batch,
                        rewards=rewards,
                    ),
                )
                records = self._run_phase(
                    strategy,
                    "record",
                    lambda: builder.build_records(
                        batch,
                        rewards,
                        media_path=None,
                        rollout_cache_path=None,
                    ),
                )
                gathered_records = strategy.gather_object(records, dst=0)
                should_checkpoint = (
                    (step + 1) % checkpoint_every == 0
                    or step + 1 == target_steps
                )
                cycle_end = self._cycle_end(
                    step,
                    target_steps=target_steps,
                    checkpoint_every=checkpoint_every,
                )

                def stage_records() -> None:
                    if not strategy.is_main_process:
                        return
                    assert manager is not None
                    assert transaction is not None
                    assert gathered_records is not None
                    merged = tuple(
                        record
                        for rank_records in gathered_records
                        for record in rank_records
                    )
                    checkpoint_relative = f"checkpoint_{cycle_end:06d}"
                    staged = tuple(
                        replace(
                            record,
                            checkpoint_path=checkpoint_relative,
                        )
                        for record in merged
                    )
                    manager.stage_records(
                        transaction,
                        step=step,
                        records=staged,
                        metrics=metrics,
                    )

                self._run_phase(strategy, "artifact_stage", stage_records)

                if should_checkpoint:
                    rank_state = self._capture_rank_state(
                        rank=strategy.rank,
                        device=strategy.device,
                    )
                    gathered_states = strategy.gather_object(rank_state, dst=0)

                    def commit_checkpoint() -> None:
                        nonlocal transaction
                        if not strategy.is_main_process:
                            return
                        assert manager is not None
                        assert transaction is not None
                        assert gathered_states is not None
                        rank_states = tuple(gathered_states)
                        checkpoint = (
                            transaction.staging_dir
                            / f"checkpoint_{step + 1:06d}"
                        )
                        save_training_state(
                            checkpoint,
                            adapter=components.model,
                            optimizer=optimizer,
                            scaler=scaler,
                            global_step=step + 1,
                            training_contract=training_contract,
                            rank_states=rank_states,
                            writer_rank=0,
                            writer_device=strategy.device,
                        )
                        committed = transaction
                        manager.commit(
                            committed,
                            checkpoint_path=checkpoint,
                        )
                        manager.rebuild_projections()
                        manager.cleanup_published_staging(committed)
                        manager.apply_checkpoint_retention(
                            keep_last=(
                                self.config.artifacts.checkpoint_keep_last
                            )
                        )
                        transaction = None

                    self._run_phase(
                        strategy,
                        "artifact_commit",
                        commit_checkpoint,
                    )
                if progress is not None:
                    progress.update(step + 1, metrics)

            local_result = self._run_phase(
                strategy,
                "run_result",
                lambda: (
                    self._build_run_result(manager)
                    if strategy.is_main_process
                    else None
                ),
            )
            result = strategy.broadcast_object(local_result)
            if not isinstance(result, RunResult):
                raise RuntimeError("training did not produce a RunResult")
            return result
        finally:
            if (
                manager is not None
                and transaction is not None
                and transaction.state == "open"
            ):
                try:
                    manager.abort(transaction)
                except Exception:
                    pass
            if components is not None:
                try:
                    components.close()
                except Exception:
                    pass
            if progress is not None:
                progress.close()
            if manager is not None:
                manager.close()
            if strategy is not None:
                strategy.close()

    @staticmethod
    def _run_phase(
        strategy: Any,
        name: str,
        operation: Callable[[], _T],
    ) -> _T:
        """Run one local operation and synchronize ordinary DDP failures."""

        if not isinstance(name, str) or not name:
            raise ValueError("phase name must be a non-empty string")
        return strategy.run_phase(name, operation)

    @staticmethod
    def _build_gradient_scaler(
        *,
        precision: str,
        device: Any,
    ) -> Any | None:
        """Build the Runner-owned scaler exactly once for CUDA fp16."""

        if precision != "fp16":
            return None
        if getattr(device, "type", None) != "cuda":
            raise ValueError("fp16 training requires a CUDA runtime device")
        import torch

        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    @staticmethod
    def _reduce_artifact_metrics(
        strategy: Any,
        *,
        update_result: Any,
        batch: Any,
        rewards: Any,
    ) -> _ArtifactMetrics:
        from visual_rl.optimizers.update_engine import UpdateResult

        if not isinstance(update_result, UpdateResult):
            raise TypeError("optimizer step must return UpdateResult")
        values: dict[str, float] = {}
        for name in _CORE_UPDATE_METRICS:
            value = float(getattr(update_result, name))
            if not math.isfinite(value):
                raise ValueError(f"optimizer metric {name!r} must be finite")
            values[name] = value
        for name, value in update_result.diagnostics.items():
            if name in values:
                raise ValueError(f"duplicate update metric {name!r}")
            values[name] = float(value)
        reward_metrics = strategy.reduce_reward_metrics(rewards)
        for name, value in reward_metrics.items():
            if name in values:
                raise ValueError(f"duplicate reward metric {name!r}")
            values[name] = float(value)
        sample_count = batch.batch_size * strategy.world_size
        return _ArtifactMetrics(
            values=FrozenMapping(values),
            sample_count=sample_count,
            active_transition_count=update_result.active_transition_count,
        )

    @staticmethod
    def _capture_rank_state(*, rank: int, device: Any) -> Any:
        import numpy as np
        import torch

        from visual_rl.artifacts.checkpoint import RankState

        torch_cuda = (
            torch.cuda.get_rng_state(device).cpu().contiguous()
            if getattr(device, "type", None) == "cuda"
            else None
        )
        return RankState.from_rng(
            rank=rank,
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            torch_cpu=torch.get_rng_state().cpu().contiguous(),
            torch_cuda=torch_cuda,
        )

    @staticmethod
    def _cycle_end(
        step: int,
        *,
        target_steps: int,
        checkpoint_every: int,
    ) -> int:
        next_boundary = (
            (step // checkpoint_every) + 1
        ) * checkpoint_every
        return min(next_boundary, target_steps)

    @staticmethod
    def _build_run_result(manager: Any) -> RunResult:
        if manager is None:
            raise RuntimeError("rank zero has no ArtifactManager")
        head = manager.head
        checkpoint = manager.checkpoint_path
        if head is None or checkpoint is None:
            raise ResumeError(
                "run has no authoritative committed checkpoint",
                path=str(manager.output_dir),
            )
        completed_steps = int(head["completed_steps"])
        last_row = head["steps"][-1]["core_metric_row"]
        last_metrics = {
            name: value
            for name, value in last_row.items()
            if name != "schema_version"
        }
        marker_path = (
            manager.output_dir
            / "commits"
            / f"commit_{completed_steps:06d}.json"
        )
        return RunResult(
            run_id=manager.run_id,
            output_dir=manager.output_dir,
            committed_steps=completed_steps,
            authoritative_checkpoint=checkpoint,
            resolved_config_path=manager.config_path,
            manifest_path=manager.manifest_path,
            metrics_path=manager.metrics_path,
            marker_path=marker_path,
            last_metrics=last_metrics,
        )
