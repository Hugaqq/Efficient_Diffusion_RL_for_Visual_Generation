"""The sole config-driven VisualRL training runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
            context = strategy.context
            configure_deterministic_runtime(self.config.runtime.deterministic)
            seed_everything(self.config.run.seed)

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
                rank=context.rank,
                local_rank=context.local_rank,
                world_size=context.world_size,
                backend=context.backend,
                device=context.device,
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
            scaler = self._current_gradient_scaler(
                components.optimizer_plugin,
                device=context.device,
            )
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
                        expected_world_size=context.world_size,
                        expected_training_contract=training_contract,
                    )
                    apply_training_state(
                        validated,
                        adapter=components.model,
                        optimizer=optimizer,
                        scaler=scaler,
                        optimizer_config=self.config.optimizer,
                        rank=context.rank,
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

                dataset_start = (
                    step * self.config.runtime.batch_size * context.world_size
                    + context.rank * self.config.runtime.batch_size
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
                        + step * context.world_size
                        + context.rank
                    ),
                    rank=context.rank,
                    world_size=context.world_size,
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
                        adapter=components.model,
                        batch=batch,
                        rewards=rewards,
                        optimizer=optimizer,
                        context=step_context,
                        recompute_policy_stats=(
                            strategy.recompute_policy_stats
                        ),
                        gradient_sync_context=(
                            strategy.gradient_sync_context
                        ),
                        reduce_tensor_weighted_mean=(
                            strategy.reduce_tensor_weighted_mean
                        ),
                        synchronize_failure=strategy.synchronize_failure,
                        optimizer_step=strategy.atomic_optimizer_step,
                    ),
                )
                metrics = self._reduce_artifact_metrics(
                    strategy,
                    update_metrics=update_metrics,
                    batch=batch,
                    rewards=rewards,
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
                        rank=context.rank,
                        device=context.device,
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
                            writer_device=context.device,
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
        error: BaseException | None = None
        result: Any = None
        try:
            result = operation()
        except BaseException as exc:
            error = exc
        if strategy.context.world_size > 1:
            strategy.synchronize_failure(error)
        if error is not None:
            raise error
        return result

    @staticmethod
    def _current_gradient_scaler(
        optimizer_plugin: Any,
        *,
        device: Any,
    ) -> Any | None:
        """Reuse the pre-W04 scaler owner until W04 moves it into Runner."""

        if optimizer_plugin.precision != "fp16":
            return None
        if getattr(device, "type", None) != "cuda":
            raise ValueError("fp16 training requires a CUDA runtime device")
        return optimizer_plugin.update_engine._get_grad_scaler()

    @staticmethod
    def _reduce_artifact_metrics(
        strategy: Any,
        *,
        update_metrics: Mapping[str, Any],
        batch: Any,
        rewards: Any,
    ) -> _ArtifactMetrics:
        if not isinstance(update_metrics, Mapping):
            raise TypeError("optimizer step metrics must be a mapping")
        local_values: dict[str, float] = {}
        for name in _CORE_UPDATE_METRICS:
            if name not in update_metrics:
                raise ValueError(f"optimizer metrics are missing {name!r}")
            value = float(update_metrics[name])
            if not math.isfinite(value):
                raise ValueError(f"optimizer metric {name!r} must be finite")
            local_values[name] = value
        local_values["reference_kl"] = float(
            update_metrics.get("reference_kl", 0.0)
        )
        active_count = int(batch.transition_mask.sum().detach().cpu().item())
        reward_values = tuple(
            float(item)
            for item in rewards.weighted_total.detach().cpu().tolist()
        )
        contribution = {
            "values": local_values,
            "sample_count": batch.batch_size,
            "active_transition_count": active_count,
            "reward_values": reward_values,
        }
        gathered = strategy.gather_object(contribution, dst=0)
        aggregate = None
        if strategy.is_main_process:
            assert gathered is not None
            total_samples = sum(item["sample_count"] for item in gathered)
            total_active = sum(
                item["active_transition_count"] for item in gathered
            )
            all_rewards = tuple(
                value
                for item in gathered
                for value in item["reward_values"]
            )
            if total_samples <= 0 or total_active <= 0 or not all_rewards:
                raise ValueError("global metric reduction requires positive counts")
            values = {
                name: sum(
                    item["values"][name]
                    * item["active_transition_count"]
                    for item in gathered
                )
                / total_active
                for name in (*_CORE_UPDATE_METRICS, "reference_kl")
            }
            reward_mean = sum(all_rewards) / len(all_rewards)
            values["reward_mean"] = reward_mean
            values["reward_std"] = math.sqrt(
                sum((value - reward_mean) ** 2 for value in all_rewards)
                / len(all_rewards)
            )
            aggregate = (
                values,
                total_samples,
                total_active,
            )
        aggregate = strategy.broadcast_object(aggregate)
        if (
            not isinstance(aggregate, tuple)
            or len(aggregate) != 3
        ):
            raise RuntimeError("metric reduction returned an invalid payload")
        return _ArtifactMetrics(
            values=FrozenMapping(aggregate[0]),
            sample_count=aggregate[1],
            active_transition_count=aggregate[2],
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
