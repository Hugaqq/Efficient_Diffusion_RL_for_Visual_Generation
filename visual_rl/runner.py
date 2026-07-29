"""The sole config-driven VisualRL training runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import pickle
import random
from typing import Any
import uuid
import warnings

from visual_rl.api_types import RunResult
from visual_rl.artifacts.manifest import SampleRecord
from visual_rl.configs.schema import VisualRLConfig
from visual_rl.core.types import (
    FrozenMapping,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
    ValidatedRuntimeEnv,
    to_plain_dict,
)
from visual_rl.errors import ResumeError, RunError, attach_cleanup_notes


_CORE_UPDATE_METRICS = (
    "loss",
    "policy_loss",
    "reference_kl",
    "approx_kl",
    "clipfrac",
)
_RESERVED_METRIC_KEYS = frozenset(
    {"schema_version", "step", "sample_count", "active_transition_count"}
)


@dataclass(frozen=True)
class StepMetrics:
    """The one scalar-only result contract for a completed policy update."""

    values: FrozenMapping
    sample_count: int
    active_transition_count: int

    def __post_init__(self) -> None:
        values = (
            self.values
            if isinstance(self.values, FrozenMapping)
            else FrozenMapping(self.values)
        )
        object.__setattr__(self, "values", values)
        for name in ("sample_count", "active_transition_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if _RESERVED_METRIC_KEYS.intersection(values):
            raise ValueError("StepMetrics.values contains a reserved metric key")
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(
                    "StepMetrics.values must contain finite Python floats"
                )


@dataclass(frozen=True)
class StepArtifacts:
    """Immutable, tensor-free rank-local records awaiting one commit path."""

    local_records: tuple[SampleRecord, ...]

    def __post_init__(self) -> None:
        if type(self.local_records) is not tuple or not self.local_records:
            raise ValueError("local_records must be a non-empty tuple")
        if any(
            not isinstance(record, SampleRecord)
            for record in self.local_records
        ):
            raise TypeError("local_records must contain only SampleRecord values")


@dataclass(frozen=True)
class StepResult:
    """One step's identity, global scalars, and rank-local commit material."""

    context: StepContext
    metrics: StepMetrics
    artifacts: StepArtifacts

    def __post_init__(self) -> None:
        if not isinstance(self.context, StepContext):
            raise TypeError("context must be a StepContext")
        if not isinstance(self.metrics, StepMetrics):
            raise TypeError("metrics must be StepMetrics")
        if not isinstance(self.artifacts, StepArtifacts):
            raise TypeError("artifacts must be StepArtifacts")


@dataclass(frozen=True)
class _RunPreparation:
    runtime_context: RuntimeBuildContext
    run_id: str
    start_step: int
    authoritative_checkpoint: Path | None
    noop_result: RunResult | None


class CommitCoordinator:
    """The sole owner of checkpoint and global artifact commit lifecycle."""

    def __init__(
        self,
        *,
        run_id: str,
        start_step: int,
        base_seed: int,
        output_dir: Path,
        checkpoint_keep_last: int | None,
        training_contract: Any,
        strategy: Any,
        artifact_manager: Any | None,
        adapter: Any,
        trainable_named_parameters: tuple[tuple[str, Any], ...],
        optimizer: Any,
        scaler: Any | None,
    ) -> None:
        import torch

        from visual_rl.artifacts.checkpoint import TrainingContract
        from visual_rl.artifacts.manager import ArtifactManager
        from visual_rl.distributed import DDPStrategy, SingleProcessStrategy
        from visual_rl.model_adapters.base import ModelAdapter

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if type(start_step) is not int or start_step < 0:
            raise ValueError("start_step must be a non-negative integer")
        if type(base_seed) is not int or base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        if not isinstance(output_dir, Path) or not output_dir.is_absolute():
            raise TypeError("output_dir must be an absolute pathlib.Path")
        if checkpoint_keep_last is not None and (
            type(checkpoint_keep_last) is not int
            or checkpoint_keep_last <= 0
        ):
            raise ValueError(
                "checkpoint_keep_last must be a positive integer or None"
            )
        if not isinstance(training_contract, TrainingContract):
            raise TypeError("training_contract must be a TrainingContract")
        if not isinstance(strategy, (SingleProcessStrategy, DDPStrategy)):
            raise TypeError("strategy must implement the final Strategy contract")
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter must be a ModelAdapter")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("optimizer must be torch.optim.AdamW")
        if scaler is not None:
            scaler_methods = (
                "scale",
                "unscale_",
                "step",
                "update",
                "state_dict",
                "load_state_dict",
            )
            if any(
                not callable(getattr(scaler, name, None))
                for name in scaler_methods
            ):
                raise TypeError("scaler must implement the GradScaler contract")
        if strategy.is_main_process:
            if not isinstance(artifact_manager, ArtifactManager):
                raise ValueError("rank zero must own one ArtifactManager")
            if artifact_manager.output_dir != output_dir:
                raise ValueError(
                    "ArtifactManager output_dir does not match coordinator"
                )
        elif artifact_manager is not None:
            raise ValueError("non-main ranks must not own an ArtifactManager")
        if type(trainable_named_parameters) is not tuple or not (
            trainable_named_parameters
        ):
            raise ValueError(
                "trainable_named_parameters must be a non-empty tuple"
            )
        names = tuple(name for name, _parameter in trainable_named_parameters)
        parameters = tuple(
            parameter for _name, parameter in trainable_named_parameters
        )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("trainable parameter names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("trainable parameter names must be unique")
        if any(
            not isinstance(parameter, torch.nn.Parameter)
            or not parameter.requires_grad
            for parameter in parameters
        ):
            raise TypeError(
                "trainable parameters must be trainable torch.nn.Parameter"
            )
        adapter_named = adapter.named_parameters()
        if tuple(name for name, _parameter in adapter_named) != names or tuple(
            id(parameter) for _name, parameter in adapter_named
        ) != tuple(map(id, parameters)):
            raise ValueError(
                "trainable parameter identity/order does not match Adapter"
            )
        optimizer_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if tuple(map(id, optimizer_parameters)) != tuple(map(id, parameters)):
            raise ValueError(
                "optimizer parameter identity/order does not match Adapter"
            )

        self.run_id = run_id
        self.expected_step = start_step
        self.base_seed = base_seed
        self.output_dir = output_dir
        self.checkpoint_keep_last = checkpoint_keep_last
        self.training_contract = training_contract
        self.strategy = strategy
        self.artifact_manager = artifact_manager
        self.adapter = adapter
        self.trainable_named_parameters = trainable_named_parameters
        self.optimizer = optimizer
        self.scaler = scaler
        self.transaction: Any | None = None
        self._aborted = False

    def ensure_cycle(self, step: int) -> None:
        """Validate the next logical step and create one rank-zero transaction."""

        def operation() -> None:
            self._validate_expected_step(step)
            if self.strategy.is_main_process and self.transaction is None:
                assert self.artifact_manager is not None
                self.transaction = self.artifact_manager.begin_transaction()

        self.strategy.run_phase("cycle_setup", operation)

    def accept(
        self,
        step_result: StepResult,
        *,
        should_checkpoint: bool,
    ) -> None:
        """Validate, stage, and conditionally commit one shared step result."""

        local_error: BaseException | None = None
        try:
            self._validate_local_result(step_result)
            if type(should_checkpoint) is not bool:
                raise TypeError("should_checkpoint must be a bool")
        except BaseException as exc:
            local_error = exc
        self.strategy.failure_gate("commit_contract.local", local_error)

        gathered = self.strategy.gather_object(
            (step_result, should_checkpoint),
            dst=0,
        )
        merged_records: tuple[Any, ...] | None = None

        def validate_contract() -> None:
            nonlocal merged_records
            if not self.strategy.is_main_process:
                return
            if gathered is None:
                raise RuntimeError("rank zero did not receive commit payloads")
            merged_records = self._validate_gathered_results(gathered)

        self.strategy.run_phase("commit_contract", validate_contract)

        checkpoint_relative: str | None = None
        if should_checkpoint:
            completed_steps = step_result.context.step + 1
            checkpoint_relative = f"checkpoint_{completed_steps:06d}"
            rank_state: Any | None = None
            capture_error: BaseException | None = None
            try:
                rank_state = self._capture_rank_state(
                    rank=self.strategy.rank,
                    device=self.strategy.device,
                )
            except BaseException as exc:
                capture_error = exc
            self.strategy.failure_gate("checkpoint_capture", capture_error)
            gathered_states = self.strategy.gather_object(rank_state, dst=0)

            def stage_checkpoint() -> None:
                if not self.strategy.is_main_process:
                    return
                rank_states = self._validated_rank_states(gathered_states)
                self._save_checkpoint(completed_steps, rank_states)

            self.strategy.run_phase("checkpoint_stage", stage_checkpoint)

        def stage_artifacts() -> None:
            if not self.strategy.is_main_process:
                return
            if merged_records is None:
                raise RuntimeError("rank zero lost validated records")
            if self.artifact_manager is None or self.transaction is None:
                raise RuntimeError("rank zero has no open artifact transaction")
            records = (
                tuple(
                    replace(record, checkpoint_path=checkpoint_relative)
                    for record in merged_records
                )
                if checkpoint_relative is not None
                else merged_records
            )
            self.artifact_manager.stage_records(
                self.transaction,
                step=step_result.context.step,
                records=records,
                metrics=step_result.metrics,
            )

        self.strategy.run_phase("artifact_stage", stage_artifacts)
        if not should_checkpoint:
            self.expected_step += 1
            return

        committed_transaction: Any | None = None

        def commit_artifacts() -> None:
            nonlocal committed_transaction
            if not self.strategy.is_main_process:
                return
            if self.artifact_manager is None or self.transaction is None:
                raise RuntimeError("rank zero has no transaction to commit")
            completed_steps = step_result.context.step + 1
            checkpoint_path = (
                self.transaction.staging_dir
                / f"checkpoint_{completed_steps:06d}"
            )
            committed_transaction = self.transaction
            self.artifact_manager.commit(
                committed_transaction,
                checkpoint_path=checkpoint_path,
            )
            self.transaction = None

        self.strategy.run_phase("artifact_commit", commit_artifacts)

        def rebuild_projections() -> None:
            if self.strategy.is_main_process:
                assert self.artifact_manager is not None
                self.artifact_manager.rebuild_projections()

        try:
            self.strategy.run_phase(
                "post_commit.projection",
                rebuild_projections,
            )

            def cleanup_staging() -> None:
                if self.strategy.is_main_process:
                    assert self.artifact_manager is not None
                    if committed_transaction is None:
                        raise RuntimeError(
                            "committed transaction handle was lost"
                        )
                    self.artifact_manager.cleanup_published_staging(
                        committed_transaction
                    )

            self.strategy.run_phase("post_commit.cleanup", cleanup_staging)

            def apply_retention() -> None:
                if self.strategy.is_main_process:
                    assert self.artifact_manager is not None
                    self.artifact_manager.apply_checkpoint_retention(
                        keep_last=self.checkpoint_keep_last
                    )

            self.strategy.run_phase("post_commit.retention", apply_retention)
        except Exception as exc:
            raise RunError(
                "post-commit artifact maintenance failed",
                step=step_result.context.step,
            ) from exc
        self.expected_step += 1

    def stage_previews(
        self,
        batch: RolloutBatch,
        *,
        max_samples: int,
    ) -> tuple[str | None, ...]:
        """Stage rank-zero previews without retaining the rollout batch."""

        if not self.strategy.is_main_process:
            raise RuntimeError("only rank zero can stage previews")
        if self.artifact_manager is None or self.transaction is None:
            raise RuntimeError("rank zero has no open artifact transaction")
        from visual_rl.artifacts.preview import PreviewWriteResult

        result = self.artifact_manager.stage_previews(
            self.transaction,
            batch,
            max_samples=max_samples,
        )
        if not isinstance(result, PreviewWriteResult):
            raise TypeError("preview writer must return PreviewWriteResult")
        for message in result.warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return result.media_paths

    def build_run_result(self) -> RunResult:
        if not self.strategy.is_main_process or self.artifact_manager is None:
            raise RuntimeError("only rank zero can build the authoritative result")
        return _build_run_result(self.artifact_manager)

    def abort(self) -> None:
        """Best-effort abort of the one still-open pre-marker transaction."""

        if self._aborted:
            return
        self._aborted = True
        transaction = self.transaction
        if (
            transaction is not None
            and transaction.state == "open"
            and self.artifact_manager is not None
        ):
            self.artifact_manager.abort(transaction)
        self.transaction = None

    def _validate_expected_step(self, step: int) -> None:
        if type(step) is not int or step != self.expected_step:
            raise ValueError(
                f"expected logical step {self.expected_step}, got {step!r}"
            )

    def _validate_local_result(self, result: StepResult) -> None:
        if not isinstance(result, StepResult):
            raise TypeError("step_result must be a StepResult")
        context = result.context
        if context.step != self.expected_step:
            raise ValueError("StepResult step does not match coordinator")
        if (
            context.rank != self.strategy.rank
            or context.world_size != self.strategy.world_size
        ):
            raise ValueError("StepResult topology does not match Strategy")
        expected_seed = (
            self.base_seed
            + context.step * context.world_size
            + context.rank
        )
        if context.seed != expected_seed:
            raise ValueError("StepResult seed does not match the canonical formula")
        sample_ids: set[str] = set()
        for record in result.artifacts.local_records:
            if (
                record.run_id != self.run_id
                or record.step != context.step
                or record.rank != context.rank
                or record.seed != context.seed
                or record.checkpoint_path is not None
            ):
                raise ValueError("SampleRecord identity does not match StepResult")
            if not record.sample_id or record.sample_id in sample_ids:
                raise ValueError("local SampleRecord sample_id values must be unique")
            sample_ids.add(record.sample_id)
        if len(sample_ids) > result.metrics.sample_count:
            raise ValueError("local records exceed the global sample_count")

    def _validate_gathered_results(
        self,
        gathered: list[Any],
    ) -> tuple[Any, ...]:
        if len(gathered) != self.strategy.world_size:
            raise ValueError("commit payload count does not match world_size")
        parsed: list[tuple[StepResult, bool]] = []
        for item in gathered:
            if (
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], StepResult)
                or type(item[1]) is not bool
            ):
                raise TypeError("gathered commit payload is invalid")
            parsed.append(item)
        parsed.sort(key=lambda item: item[0].context.rank)
        ranks = tuple(item[0].context.rank for item in parsed)
        if ranks != tuple(range(self.strategy.world_size)):
            raise ValueError("gathered ranks must cover the complete world")
        first_result, first_schedule = parsed[0]
        for result, schedule in parsed:
            self._validate_gathered_result(result)
            if (
                result.context.step != first_result.context.step
                or result.context.world_size != first_result.context.world_size
            ):
                raise ValueError("gathered StepContext values disagree")
            if schedule != first_schedule:
                raise ValueError("checkpoint schedule differs across ranks")
            if result.metrics != first_result.metrics:
                raise ValueError("global StepMetrics differ across ranks")
        records = tuple(
            record
            for result, _schedule in parsed
            for record in result.artifacts.local_records
        )
        if len(records) != first_result.metrics.sample_count:
            raise ValueError("merged record count does not match sample_count")
        sample_ids = tuple(record.sample_id for record in records)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("merged SampleRecord sample_id values are not unique")
        return records

    def _validate_gathered_result(self, result: StepResult) -> None:
        context = result.context
        if context.step != self.expected_step:
            raise ValueError("gathered result step does not match coordinator")
        if context.world_size != self.strategy.world_size:
            raise ValueError("gathered result world_size is invalid")
        expected_seed = (
            self.base_seed
            + context.step * context.world_size
            + context.rank
        )
        if context.seed != expected_seed:
            raise ValueError("gathered result seed is invalid")
        for record in result.artifacts.local_records:
            if (
                record.run_id != self.run_id
                or record.step != context.step
                or record.rank != context.rank
                or record.seed != context.seed
                or record.checkpoint_path is not None
            ):
                raise ValueError("gathered SampleRecord identity is invalid")

    def _validated_rank_states(
        self,
        gathered_states: list[Any] | None,
    ) -> tuple[Any, ...]:
        from visual_rl.artifacts.checkpoint import RankState

        if gathered_states is None or len(gathered_states) != (
            self.strategy.world_size
        ):
            raise ValueError("rank-state count does not match world_size")
        states = tuple(gathered_states)
        if any(not isinstance(state, RankState) for state in states):
            raise TypeError("gathered checkpoint state must contain RankState")
        if tuple(state.rank for state in states) != tuple(
            range(self.strategy.world_size)
        ):
            raise ValueError("rank states must be ordered by complete rank")
        return states

    def _save_checkpoint(
        self,
        completed_steps: int,
        rank_states: tuple[Any, ...],
    ) -> None:
        from visual_rl.artifacts.checkpoint import save_training_state

        if self.transaction is None:
            raise RuntimeError("checkpoint stage requires an open transaction")
        checkpoint_path = (
            self.transaction.staging_dir
            / f"checkpoint_{completed_steps:06d}"
        )
        save_training_state(
            checkpoint_path,
            adapter=self.adapter,
            optimizer=self.optimizer,
            scaler=self.scaler,
            global_step=completed_steps,
            training_contract=self.training_contract,
            rank_states=rank_states,
            writer_rank=self.strategy.rank,
            writer_device=self.strategy.device,
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


class ExperimentRunner:
    """Consume one validated config/environment snapshot and run it once."""

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
        self.strategy: Any | None = None
        self.artifact_manager: Any | None = None
        self.components: Any | None = None
        self.optimizer: Any | None = None
        self.scaler: Any | None = None
        self.validated_training_state: Any | None = None
        self.progress: Any | None = None
        self.manifest_builder: Any | None = None
        self.coordinator: CommitCoordinator | None = None
        self._resources_closed = False

    def run(self) -> RunResult:
        """Execute the one shared single/DDP loop and return its committed head."""

        from visual_rl.artifacts.builder import ManifestBuilder
        from visual_rl.artifacts.checkpoint import (
            TrainingContract,
            apply_training_state,
            read_and_validate_training_state,
        )
        from visual_rl.artifacts.logging import TrainProgressPrinter
        from visual_rl.runtime_factory import build_runtime_components

        self._reset_run_state()
        primary_error: BaseException | None = None
        try:
            preparation = self._prepare_run()
            if preparation.noop_result is not None:
                return preparation.noop_result
            assert self.strategy is not None

            target_steps = int(self.config.runtime.max_steps)

            def build_progress() -> None:
                if (
                    self.strategy.is_main_process
                    and self.config.runtime.progress
                ):
                    self.progress = TrainProgressPrinter()
                    self.progress.start(
                        target_steps,
                        initial_step=preparation.start_step,
                    )

            self.strategy.run_phase("progress_setup", build_progress)

            def build_runtime() -> None:
                self.components = build_runtime_components(
                    self.config,
                    preparation.runtime_context,
                )

            self.strategy.run_phase("runtime_build", build_runtime)
            assert self.components is not None
            self.strategy.run_phase(
                "model_prepare",
                lambda: self.strategy.prepare(self.components.model),
            )
            self.strategy.run_phase(
                "optimizer_setup",
                self._build_optimizer_and_scaler_into_self,
            )
            assert self.optimizer is not None

            training_contract = TrainingContract(
                algorithm=self.config.algorithm.name,
                version=(
                    self.components.optimizer_plugin.algorithm
                    .TRAINING_CONTRACT_VERSION
                ),
            )

            def preflight_training_state() -> None:
                if preparation.authoritative_checkpoint is not None:
                    self.validated_training_state = (
                        read_and_validate_training_state(
                            preparation.authoritative_checkpoint,
                            adapter=self.components.model,
                            optimizer=self.optimizer,
                            scaler=self.scaler,
                            expected_global_step=preparation.start_step,
                            expected_world_size=self.strategy.world_size,
                            expected_training_contract=training_contract,
                        )
                    )
                elif self.config.model.adapter_checkpoint is not None:
                    self.components.model.validate_checkpoint(
                        self.config.model.adapter_checkpoint
                    )

            self.strategy.run_phase(
                "training_state_preflight",
                preflight_training_state,
            )

            def restore_training_state() -> None:
                if self.validated_training_state is not None:
                    apply_training_state(
                        self.validated_training_state,
                        adapter=self.components.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        optimizer_config=self.config.optimizer,
                        rank=self.strategy.rank,
                    )
                    self.validated_training_state = None
                elif self.config.model.adapter_checkpoint is not None:
                    self.components.model.load_checkpoint(
                        self.config.model.adapter_checkpoint
                    )

            self.strategy.run_phase(
                "training_state_restore",
                restore_training_state,
            )

            def build_manifest_builder() -> None:
                self.manifest_builder = ManifestBuilder(
                    run_id=preparation.run_id,
                    media_type=self.components.model.MEDIA_TYPE,
                    rollout_type=self.config.rollout.name,
                )

            self.strategy.run_phase("record_setup", build_manifest_builder)

            def build_coordinator() -> None:
                self.coordinator = CommitCoordinator(
                    run_id=preparation.run_id,
                    start_step=preparation.start_step,
                    base_seed=self.config.run.seed,
                    output_dir=self.config.artifacts.output_dir,
                    checkpoint_keep_last=(
                        self.config.artifacts.checkpoint_keep_last
                    ),
                    training_contract=training_contract,
                    strategy=self.strategy,
                    artifact_manager=self.artifact_manager,
                    adapter=self.components.model,
                    trainable_named_parameters=(
                        self.components.model.named_parameters()
                    ),
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                )

            self.strategy.run_phase("commit_setup", build_coordinator)
            assert self.coordinator is not None
            checkpoint_every = int(
                self.config.artifacts.checkpoint_every
            )
            for step in range(preparation.start_step, target_steps):
                should_checkpoint = (
                    (step + 1) % checkpoint_every == 0
                    or step + 1 == target_steps
                )
                should_preview = (
                    self.config.artifacts.preview_samples_per_event > 0
                    and (step == 0 or should_checkpoint)
                )
                self.coordinator.ensure_cycle(step)
                step_result = self._execute_step(
                    step,
                    should_preview=should_preview,
                )
                self.coordinator.accept(
                    step_result,
                    should_checkpoint=should_checkpoint,
                )
                if self.progress is not None:
                    self.progress.update(step + 1, step_result.metrics)

            local_result: RunResult | None = None

            def build_local_result() -> None:
                nonlocal local_result
                if self.strategy.is_main_process:
                    local_result = self.coordinator.build_run_result()

            self.strategy.run_phase("run_result_build", build_local_result)
            result = self.strategy.broadcast_object(local_result)
            if not isinstance(result, RunResult):
                raise RuntimeError("training did not produce a RunResult")
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_errors = self._close_local_run_resources()
            try:
                if primary_error is None and self.strategy is not None:
                    try:
                        self.strategy.failure_gate(
                            "cleanup",
                            cleanup_errors[0] if cleanup_errors else None,
                        )
                    except BaseException as consensus_error:
                        if (
                            type(consensus_error).__name__
                            == "_ProcessGroupFatalError"
                        ):
                            raise
                        raise RunError("run cleanup failed") from consensus_error
                elif primary_error is not None and cleanup_errors:
                    attach_cleanup_notes(primary_error, cleanup_errors)
            finally:
                if self.strategy is not None:
                    self.strategy.close()

    def _prepare_run(self) -> _RunPreparation:
        from visual_rl.artifacts.manager import ArtifactManager
        from visual_rl.core.determinism import configure_deterministic_runtime
        from visual_rl.core.seed import seed_everything
        from visual_rl.distributed import build_strategy

        configure_deterministic_runtime(self.config.runtime.deterministic)
        self.strategy = build_strategy(
            self.config.runtime.distributed,
            self.validated_env,
        )
        strategy = self.strategy
        strategy.run_phase(
            "seed_setup",
            lambda: seed_everything(self.config.run.seed + strategy.rank),
        )

        local_config: Any | None = None

        def build_local_config() -> None:
            nonlocal local_config
            local_config = to_plain_dict(self.config)
            pickle.dumps(local_config, protocol=pickle.HIGHEST_PROTOCOL)

        strategy.run_phase("config_contract.local", build_local_config)
        gathered_configs = strategy.gather_object(local_config, dst=0)

        def validate_configs() -> None:
            if not strategy.is_main_process:
                return
            if (
                gathered_configs is None
                or len(gathered_configs) != strategy.world_size
                or any(
                    item != gathered_configs[0]
                    for item in gathered_configs[1:]
                )
            ):
                raise ValueError(
                    "canonical configuration differs across distributed ranks"
                )

        strategy.run_phase("config_contract", validate_configs)

        artifact_payload: tuple[str, int, str | None] | None = None

        def prepare_artifacts() -> None:
            nonlocal artifact_payload
            if not strategy.is_main_process:
                return
            output_dir = self.config.artifacts.output_dir
            if self.config.resume.from_ is None:
                run_id = f"run-{uuid.uuid4().hex}"
                self.artifact_manager = ArtifactManager(
                    output_dir,
                    run_id,
                    config=self.config,
                )
            else:
                self.artifact_manager = ArtifactManager.open_resume(output_dir)
                self.artifact_manager.recover()
                self.artifact_manager.apply_checkpoint_retention(
                    keep_last=self.config.artifacts.checkpoint_keep_last
                )
                run_id = self.artifact_manager.run_id
            checkpoint = self.artifact_manager.checkpoint_path
            artifact_payload = (
                run_id,
                self.artifact_manager.start_step,
                None if checkpoint is None else str(checkpoint),
            )

        strategy.run_phase("artifact_prepare", prepare_artifacts)
        artifact_payload = strategy.broadcast_object(artifact_payload)
        if (
            type(artifact_payload) is not tuple
            or len(artifact_payload) != 3
            or not isinstance(artifact_payload[0], str)
            or type(artifact_payload[1]) is not int
            or not (
                artifact_payload[2] is None
                or isinstance(artifact_payload[2], str)
            )
        ):
            raise RuntimeError("artifact preparation returned an invalid payload")
        run_id, start_step, checkpoint_text = artifact_payload

        range_error: BaseException | None = None
        if self.config.runtime.max_steps < start_step:
            range_error = ResumeError(
                "runtime.max_steps is behind the authoritative commit",
                path=str(self.config.artifacts.output_dir),
            )
        strategy.failure_gate("resume_range", range_error)

        runtime_context = RuntimeBuildContext(
            rank=strategy.rank,
            local_rank=strategy.local_rank,
            world_size=strategy.world_size,
            backend=strategy.backend,
            device=strategy.device,
            precision=self.config.runtime.precision,
        )
        checkpoint_path = (
            None if checkpoint_text is None else Path(checkpoint_text)
        )
        if self.config.runtime.max_steps == start_step:
            local_result: RunResult | None = None

            def build_noop_result() -> None:
                nonlocal local_result
                if strategy.is_main_process:
                    local_result = _build_run_result(self.artifact_manager)

            strategy.run_phase("noop_result", build_noop_result)
            result = strategy.broadcast_object(local_result)
            if not isinstance(result, RunResult):
                raise RuntimeError("no-op resume did not produce a RunResult")
            return _RunPreparation(
                runtime_context=runtime_context,
                run_id=run_id,
                start_step=start_step,
                authoritative_checkpoint=checkpoint_path,
                noop_result=result,
            )

        if self.config.resume.from_ is not None:

            def write_resolved_config() -> None:
                if strategy.is_main_process:
                    assert self.artifact_manager is not None
                    self.artifact_manager.write_resolved_config(self.config)

            strategy.run_phase(
                "config_projection",
                write_resolved_config,
            )
        return _RunPreparation(
            runtime_context=runtime_context,
            run_id=run_id,
            start_step=start_step,
            authoritative_checkpoint=checkpoint_path,
            noop_result=None,
        )

    def _execute_step(
        self,
        step: int,
        *,
        should_preview: bool,
    ) -> StepResult:
        """Run the sole dataset-to-record step for single-process and DDP."""

        if type(should_preview) is not bool:
            raise TypeError("should_preview must be a bool")
        if (
            self.strategy is None
            or self.components is None
            or self.optimizer is None
            or self.manifest_builder is None
            or self.coordinator is None
        ):
            raise RuntimeError("Runner step resources are not prepared")
        strategy = self.strategy
        components = self.components
        prompt_batch_size = int(self.config.runtime.batch_size)

        setup: tuple[
            tuple[str, ...],
            tuple[Any, ...],
            StepContext,
        ] | None = None

        def step_setup() -> None:
            nonlocal setup
            dataset_start = strategy.dataset_start(step, prompt_batch_size)
            prompts, metadata = components.dataset.batch(
                dataset_start,
                prompt_batch_size,
            )
            if (
                type(prompts) is not tuple
                or type(metadata) is not tuple
                or len(prompts) != prompt_batch_size
                or len(metadata) != prompt_batch_size
            ):
                raise ValueError(
                    "dataset.batch() must return exactly batch_size prompt rows"
                )
            context = StepContext(
                step=step,
                seed=(
                    self.config.run.seed
                    + step * strategy.world_size
                    + strategy.rank
                ),
                rank=strategy.rank,
                world_size=strategy.world_size,
            )
            setup = prompts, metadata, context

        strategy.run_phase("step_setup", step_setup)
        if setup is None:
            raise RuntimeError("step setup lost its local result")
        prompts, metadata, context = setup

        def sample_and_validate() -> Any:
            sampled = components.rollout.sample(
                adapter=components.model,
                prompts=prompts,
                metadata=metadata,
                context=context,
            )
            if sampled.context is not context:
                raise ValueError(
                    "rollout returned a different StepContext object"
                )
            return sampled

        batch = strategy.run_phase("rollout", sample_and_validate)

        def score_and_validate() -> Any:
            result = components.reward_executor.score(batch, context)
            result.validate_against(batch)
            return result

        rewards = strategy.run_phase("reward", score_and_validate)
        reward_metrics = strategy.run_phase(
            "reduce",
            lambda: strategy.reduce_reward_metrics(rewards),
        )
        update_result = strategy.run_phase(
            "update",
            lambda: components.optimizer_plugin.step(
                batch=batch,
                rewards=rewards,
                optimizer=self.optimizer,
                scaler=self.scaler,
                context=context,
                strategy=strategy,
            ),
        )
        media_paths: tuple[str | None, ...] = (None,) * batch.batch_size
        if should_preview:

            def stage_previews() -> tuple[str | None, ...]:
                if not strategy.is_main_process:
                    return (None,) * batch.batch_size
                assert self.coordinator is not None
                return self.coordinator.stage_previews(
                    batch,
                    max_samples=(
                        self.config.artifacts.preview_samples_per_event
                    ),
                )

            media_paths = strategy.run_phase("preview", stage_previews)
            if (
                type(media_paths) is not tuple
                or len(media_paths) != batch.batch_size
                or any(
                    item is not None and not isinstance(item, str)
                    for item in media_paths
                )
            ):
                raise TypeError(
                    "preview phase must return one optional path per sample"
                )

        def build_result() -> StepResult:
            from visual_rl.optimizers.update_engine import UpdateResult

            if not isinstance(update_result, UpdateResult):
                raise TypeError("optimizer step must return UpdateResult")
            values: dict[str, float] = {
                name: float(getattr(update_result, name))
                for name in _CORE_UPDATE_METRICS
            }
            for name, value in update_result.diagnostics.items():
                if name in values:
                    raise ValueError(f"duplicate update metric {name!r}")
                values[name] = float(value)
            if set(reward_metrics) != {"reward_mean", "reward_std"}:
                raise ValueError(
                    "reward reduction must return reward_mean and reward_std"
                )
            for name, value in reward_metrics.items():
                if name in values:
                    raise ValueError(f"duplicate reward metric {name!r}")
                values[name] = float(value)
            records = self.manifest_builder.build_records(
                batch,
                rewards,
                media_paths=media_paths,
            )
            metrics = StepMetrics(
                values=FrozenMapping(values),
                sample_count=batch.batch_size * strategy.world_size,
                active_transition_count=(
                    update_result.active_transition_count
                ),
            )
            step_result = StepResult(
                context=context,
                metrics=metrics,
                artifacts=StepArtifacts(local_records=records),
            )
            if (
                step_result.context is not batch.context
                or step_result.context.step != step
                or step_result.context.rank != strategy.rank
                or step_result.context.world_size != strategy.world_size
            ):
                raise ValueError("StepResult does not preserve rollout identity")
            return step_result

        result = strategy.run_phase("record", build_result)
        return result

    def _build_optimizer_and_scaler_into_self(self) -> None:
        if self.strategy is None or self.components is None:
            raise RuntimeError("optimizer setup requires prepared components")
        self.optimizer = self.components.optimizer_plugin.build_optimizer(
            self.components.model.named_parameters(),
            self.config.optimizer,
        )
        self.scaler = self._build_gradient_scaler(
            precision=self.config.runtime.precision,
            device=self.strategy.device,
        )

    def _close_local_run_resources(self) -> tuple[BaseException, ...]:
        if self._resources_closed:
            return ()
        self._resources_closed = True
        errors: list[BaseException] = []
        resources = (
            ("coordinator", self.coordinator, "abort"),
            ("components", self.components, "close"),
            ("progress", self.progress, "close"),
            ("artifact_manager", self.artifact_manager, "close"),
        )
        for owner, resource, method_name in resources:
            if resource is None:
                continue
            try:
                getattr(resource, method_name)()
            except BaseException as exc:
                try:
                    setattr(exc, "_visual_rl_cleanup_owner", owner)
                except (AttributeError, TypeError):
                    pass
                errors.append(exc)
        return tuple(errors)

    def _reset_run_state(self) -> None:
        self.strategy = None
        self.artifact_manager = None
        self.components = None
        self.optimizer = None
        self.scaler = None
        self.validated_training_state = None
        self.progress = None
        self.manifest_builder = None
        self.coordinator = None
        self._resources_closed = False

    @staticmethod
    def _build_gradient_scaler(
        *,
        precision: str,
        device: Any,
    ) -> Any | None:
        if precision != "fp16":
            return None
        if getattr(device, "type", None) != "cuda":
            raise ValueError("fp16 training requires a CUDA runtime device")
        import torch

        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()


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
