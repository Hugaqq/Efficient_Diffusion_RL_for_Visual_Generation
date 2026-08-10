"""Default V0 final-only checkpoint sink and two-gate restore service."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from visual_rl.algorithms.optimization.kernel import PolicyUpdateResult
from visual_rl.algorithms.trainer.execution_plan import AlgorithmExecutionPlan
from visual_rl.algorithms.trainer.stages import (
    OptimizedIteration,
    OptimizeStage,
)
from visual_rl.core.serialization import canonical_json_text
from visual_rl.artifacts.checkpoint import (
    DERIVED_REFERENCE_STATE_SCHEMA,
    NO_REFERENCE_STATE_SCHEMA,
    AtomicCheckpointManager,
    CheckpointBuildInput,
    CheckpointContract,
    CheckpointCoordinator,
    CheckpointInspection,
    CheckpointProgress,
    CheckpointSafePoint,
    CheckpointStateCollector,
    CommittedCheckpoint,
    PreparedCheckpointBuildInput,
    PreparedCheckpointContract,
    RankCheckpointReader,
    RankCheckpointSnapshot,
    RankRNGSnapshot,
    ReferencePolicyStateEvidence,
    SingleProcessCheckpointBackend,
    assert_compatible_contract,
    assert_compatible_prepared_contract,
    build_checkpoint_contract,
    build_prepared_checkpoint_contract,
    derive_reference_policy_state_evidence,
)
from visual_rl.artifacts.run_manifest import recipe_manifest_payload
from visual_rl.artifacts.terminal import (
    TerminalArtifactError,
    TerminalFinalizationRequest,
    artifact_paths as _artifact_paths,
    atomic_write as _atomic_write,
    canonical_line as _canonical_line,
    finalize_terminal_run,
    prepare_output_dir as _prepare_output_dir,
    regular_file as _regular_file,
    sha256_bytes as _sha256_bytes,
    sha256_file as _sha256_file,
)
from visual_rl.core.contracts import DeclaredContract, ModelContract
from visual_rl.core.contracts.runtime import ExecutionTransformPlan
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.models import ModelParameterState, ModelStateAdapter
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.algorithms.rewards import RewardResourceState
from visual_rl.runtime.types import (
    BoundRestoreRequest,
    BoundRestoreResult,
    CheckpointRequest,
    PreparedRestoreRequest,
    PreparedRestoreResult,
    ProductionBoundRun,
    ProductionPreparedRun,
    ProductionRuntimeError,
    SafePointCheckpointReceipt,
    StageCheckpointPorts,
)
from visual_rl.runtime.resources import DefaultRuntimeResourceContainer
from visual_rl.runtime.types import RunResult

__all__ = (
    "CoordinatorCheckpointSink",
    "CoordinatorRestoreService",
    "CoordinatorRunFinalizer",
    "RuntimeCheckpointError",
)


_REFERENCE_STATE_SCHEMA = DERIVED_REFERENCE_STATE_SCHEMA
_NO_REFERENCE_STATE_SCHEMA = NO_REFERENCE_STATE_SCHEMA
_EMA_STATE_SCHEMA = NO_REFERENCE_STATE_SCHEMA
_STATE_SCHEMA_VERSIONS = tuple(
    sorted(
        (
            ("data_plane", 1),
            ("dynamics_selection_policy", 2),
            ("run_checkpoint_summary", 1),
            ("lr_scheduler", 1),
            ("model", 2),
            ("optimizer", 1),
            ("phase_schedule", 1),
            ("progress", 2),
            ("rng", 2),
            ("sampler", 1),
        )
    )
)
_COMPONENT_NAMES = (
    "data_plane",
    "lr_scheduler",
    "model",
    "optimizer",
    "run_checkpoint_summary",
)
_RESOLVED_RECIPE_FILE = "resolved_recipe.json"
_RUN_MANIFEST_FILE = "run_manifest.json"
_METRICS_FILE = "metrics.jsonl"
_SUCCESS_FILE = "SUCCESS"


class RuntimeCheckpointError(ProductionRuntimeError):
    """The V0 checkpoint or restore transaction cannot remain equivalent."""


@dataclass(frozen=True, slots=True)
class _ComponentState:
    kind: str
    schema_version: int
    payload: object

    def __post_init__(self) -> None:
        if self.kind not in _COMPONENT_NAMES:
            raise ValueError("component checkpoint state kind is invalid")
        expected_schema_version = 2 if self.kind == "model" else 1
        if (
            type(self.schema_version) is not int
            or self.schema_version != expected_schema_version
        ):
            raise ValueError(
                f"component checkpoint state {self.kind!r} schema_version "
                f"must be {expected_schema_version}"
            )
        if self.payload is None:
            raise ValueError("component checkpoint state payload must not be None")


@dataclass(frozen=True, slots=True)
class _RunCheckpointSummaryState:
    run_id: str
    recipe_id: str
    bound_contract_id: str
    checkpoint_contract_id: str
    update_execution_plan_id: str
    start_optimizer_step: int
    committed_steps: int
    update_count: int
    terminal: bool
    terminal_artifacts: FrozenMapping
    bound_reward_resource_ids: FrozenMapping
    last_metrics: FrozenMapping

    def __post_init__(self) -> None:
        _materialized_recipe_id("recipe_id", self.recipe_id)
        for name in (
            "run_id",
            "bound_contract_id",
            "checkpoint_contract_id",
            "update_execution_plan_id",
        ):
            _digest(name, getattr(self, name))
        if type(self.start_optimizer_step) is not int or self.start_optimizer_step < 0:
            raise ValueError("final summary start_optimizer_step must be non-negative")
        if type(self.committed_steps) is not int or self.committed_steps < 1:
            raise ValueError("final summary committed_steps must be positive")
        if type(self.update_count) is not int or self.update_count < 1:
            raise ValueError("final summary update_count must be positive")
        if self.update_count != self.committed_steps - self.start_optimizer_step:
            raise ValueError("final summary update_count disagrees with step range")
        if type(self.terminal) is not bool:
            raise TypeError("run checkpoint terminal flag must be bool")
        if not isinstance(self.terminal_artifacts, FrozenMapping):
            raise TypeError("terminal_artifacts must be FrozenMapping")
        if self.terminal:
            expected = {
                "checkpoint_relative_path",
                "resolved_recipe_relative_path",
                "run_manifest_relative_path",
                "metrics_relative_path",
                "marker_relative_path",
                "resolved_recipe_sha256",
                "run_manifest_sha256",
                "metrics_sha256",
            }
            if set(self.terminal_artifacts) != expected:
                raise ValueError(
                    "terminal artifact metadata has an invalid exact key set"
                )
            for name in (
                "checkpoint_relative_path",
                "resolved_recipe_relative_path",
                "run_manifest_relative_path",
                "metrics_relative_path",
                "marker_relative_path",
            ):
                _relative_path(name, self.terminal_artifacts[name])
            expected_files = {
                self.terminal_artifacts["resolved_recipe_relative_path"],
                self.terminal_artifacts["run_manifest_relative_path"],
                self.terminal_artifacts["metrics_relative_path"],
                self.terminal_artifacts["marker_relative_path"],
            }
            if len(expected_files) != 4:
                raise ValueError("terminal artifact paths must be unique")
            for name in (
                "resolved_recipe_sha256",
                "run_manifest_sha256",
                "metrics_sha256",
            ):
                _digest(name, self.terminal_artifacts[name])
        elif self.terminal_artifacts:
            raise ValueError("non-terminal checkpoint cannot own final artifacts")
        if not isinstance(self.bound_reward_resource_ids, FrozenMapping) or not (
            self.bound_reward_resource_ids
        ):
            raise ValueError("final summary reward resource ids must be non-empty")
        for name, value in self.bound_reward_resource_ids.items():
            if not isinstance(name, str) or not name:
                raise ValueError("reward resource names must be non-empty")
            _digest(f"bound reward resource id for {name}", value)
        if not isinstance(self.last_metrics, FrozenMapping):
            raise TypeError("final summary last_metrics must be FrozenMapping")
        _validate_last_metrics(self.last_metrics, self.committed_steps)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "run_checkpoint_summary",
            "run_id": self.run_id,
            "recipe_id": self.recipe_id,
            "bound_contract_id": self.bound_contract_id,
            "checkpoint_contract_id": self.checkpoint_contract_id,
            "update_execution_plan_id": self.update_execution_plan_id,
            "start_optimizer_step": self.start_optimizer_step,
            "committed_steps": self.committed_steps,
            "update_count": self.update_count,
            "terminal": self.terminal,
            "terminal_artifacts": to_plain_dict(self.terminal_artifacts),
            "bound_reward_resource_ids": to_plain_dict(self.bound_reward_resource_ids),
            "last_metrics": to_plain_dict(self.last_metrics),
        }

    @classmethod
    def from_payload(cls, payload: object) -> _RunCheckpointSummaryState:
        if not isinstance(payload, Mapping):
            raise TypeError("final run summary payload must be a mapping")
        expected = {
            "schema_version",
            "kind",
            "run_id",
            "recipe_id",
            "bound_contract_id",
            "checkpoint_contract_id",
            "update_execution_plan_id",
            "start_optimizer_step",
            "committed_steps",
            "update_count",
            "terminal",
            "terminal_artifacts",
            "bound_reward_resource_ids",
            "last_metrics",
        }
        if set(payload) != expected:
            raise ValueError("final run summary has an invalid exact key set")
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "run_checkpoint_summary"
        ):
            raise ValueError("run checkpoint summary schema or kind is invalid")
        values = dict(payload)
        values.pop("schema_version")
        values.pop("kind")
        values["terminal_artifacts"] = FrozenMapping(values["terminal_artifacts"])
        values["bound_reward_resource_ids"] = FrozenMapping(
            values["bound_reward_resource_ids"]
        )
        values["last_metrics"] = FrozenMapping(values["last_metrics"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class _DecodedStates:
    data_plane_payload: Mapping[str, object]
    run_summary: _RunCheckpointSummaryState
    model_state: ModelParameterState
    optimizer_state: Mapping[str, object]
    lr_scheduler_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _PreparedContinuation:
    inspection: CheckpointInspection
    snapshot: RankCheckpointSnapshot
    prepared_projection: PreparedCheckpointContract
    prepared_owner: ProductionPreparedRun
    preflight_owner: object
    runtime_owner: object
    graph_owner: object


class CoordinatorCheckpointSink:
    """Commit non-terminal safe points and the terminal run checkpoint."""

    def __init__(self, *, finalizer: CoordinatorRunFinalizer | None = None) -> None:
        self._finalizer = CoordinatorRunFinalizer() if finalizer is None else finalizer
        if not isinstance(self._finalizer, CoordinatorRunFinalizer):
            raise TypeError("finalizer must be CoordinatorRunFinalizer")

    def checkpoint_safe_point(
        self,
        request: CheckpointRequest,
    ) -> SafePointCheckpointReceipt:
        bound, rewards, contract = _validate_checkpoint_request(request)
        summary = request.summary
        if summary.committed_steps >= bound.prepared.training.max_optimizer_steps:
            raise RuntimeCheckpointError(
                "checkpoint_safe_point requires a non-terminal optimizer step"
            )
        if summary.committed_steps % request.cadence != 0:
            raise RuntimeCheckpointError(
                "safe-point step does not satisfy LaunchSpec cadence"
            )
        committed, _run_summary = _capture_and_commit(
            bound=bound,
            summary=summary,
            rewards=rewards,
            contract=contract,
            terminal_artifacts=FrozenMapping({}),
        )
        return SafePointCheckpointReceipt(
            checkpoint_path=committed.path,
            committed_steps=summary.committed_steps,
            checkpoint_contract_id=committed.checkpoint_contract_id,
            progress_id=committed.progress_id,
            state_tree_id=committed.state_tree_id,
        )

    def checkpoint(self, request: CheckpointRequest) -> RunResult:
        bound, rewards, contract = _validate_checkpoint_request(request)
        summary = request.summary
        if summary.committed_steps != bound.prepared.training.max_optimizer_steps:
            raise RuntimeCheckpointError(
                "terminal checkpoint requires max_optimizer_steps"
            )
        output_dir = _prepare_output_dir(bound.preflight.compiled.launch.output_dir)
        checkpoint_relative = f"checkpoints/step-{summary.committed_steps}"
        paths = _artifact_paths(output_dir)
        if paths["marker"].exists() or paths["marker"].is_symlink():
            raise FileExistsError("SUCCESS already exists for this output directory")

        _optimized, update = _final_update(summary.last_iteration.value.payload)
        last_metrics = FrozenMapping(
            _last_metrics(summary.last_iteration.value.identity.batch_size, update)
        )
        resolved_bytes = _canonical_line(
            recipe_manifest_payload(
                bound.preflight.environment.materialized,
                bound.preflight.environment.component_artifact_bindings,
            )
        )
        metrics_bytes = _canonical_line(
            {"schema_version": 1, **to_plain_dict(last_metrics)}
        )
        manifest_bytes = _canonical_line(
            _run_manifest_payload(
                bound=bound,
                summary=summary,
                contract=contract,
                checkpoint_relative_path=checkpoint_relative,
                rewards=rewards,
                resolved_recipe_sha256=_sha256_bytes(resolved_bytes),
                metrics_sha256=_sha256_bytes(metrics_bytes),
            )
        )
        _atomic_write(paths["resolved"], resolved_bytes)
        _atomic_write(paths["manifest"], manifest_bytes)
        _atomic_write(paths["metrics"], metrics_bytes)
        terminal_artifacts = FrozenMapping(
            {
                "checkpoint_relative_path": checkpoint_relative,
                "resolved_recipe_relative_path": _RESOLVED_RECIPE_FILE,
                "run_manifest_relative_path": _RUN_MANIFEST_FILE,
                "metrics_relative_path": _METRICS_FILE,
                "marker_relative_path": _SUCCESS_FILE,
                "resolved_recipe_sha256": _sha256_bytes(resolved_bytes),
                "run_manifest_sha256": _sha256_bytes(manifest_bytes),
                "metrics_sha256": _sha256_bytes(metrics_bytes),
            }
        )
        committed, run_summary = _capture_and_commit(
            bound=bound,
            summary=summary,
            rewards=rewards,
            contract=contract,
            terminal_artifacts=terminal_artifacts,
        )
        inspection = AtomicCheckpointManager(
            output_dir / "checkpoints"
        ).inspect_complete(committed.path, expected_contract=contract)
        return self._finalizer.finalize(
            bound=bound,
            inspection=inspection,
            summary=run_summary,
            rewards=rewards,
        )


class CoordinatorRunFinalizer:
    """Idempotently converge one terminal checkpoint to a SUCCESS run."""

    def finalize(
        self,
        *,
        bound: ProductionBoundRun,
        inspection: CheckpointInspection,
        summary: _RunCheckpointSummaryState,
        rewards: FrozenMapping,
    ) -> RunResult:
        if not summary.terminal:
            raise RuntimeCheckpointError(
                "run finalization requires terminal checkpoint metadata"
            )
        _validate_run_summary(bound, inspection, summary, rewards)
        launch = bound.preflight.compiled.launch
        try:
            terminal = finalize_terminal_run(
                TerminalFinalizationRequest(
                    current_output_dir=launch.output_dir,
                    resume_from=launch.resume_from,
                    checkpoint_path=inspection.committed.path,
                    run_id=summary.run_id,
                    committed_steps=summary.committed_steps,
                    checkpoint_contract_id=(
                        inspection.contract.checkpoint_contract_id
                    ),
                    progress_id=inspection.progress.progress_id,
                    state_tree_id=inspection.committed.state_tree_id,
                    terminal_artifacts=summary.terminal_artifacts,
                    current_recipe_payload=recipe_manifest_payload(
                        bound.preflight.environment.materialized,
                        bound.preflight.environment.component_artifact_bindings,
                    ),
                    last_metrics=summary.last_metrics,
                )
            )
        except TerminalArtifactError as exc:
            raise RuntimeCheckpointError(str(exc)) from exc
        return RunResult(
            run_id=summary.run_id,
            output_dir=terminal.output_dir,
            committed_steps=summary.committed_steps,
            authoritative_checkpoint=terminal.checkpoint_path,
            resolved_config_path=terminal.resolved_path,
            manifest_path=terminal.manifest_path,
            metrics_path=terminal.metrics_path,
            marker_path=terminal.marker_path,
            last_metrics=summary.last_metrics,
        )


def _validate_checkpoint_request(
    request: CheckpointRequest,
) -> tuple[ProductionBoundRun, FrozenMapping, CheckpointContract]:
    if not isinstance(request, CheckpointRequest):
        raise TypeError("request must be CheckpointRequest")
    bound = request.bound
    if not isinstance(bound, ProductionBoundRun):
        raise TypeError("checkpoint request bound must be ProductionBoundRun")
    _validate_v0_prepared(bound.runtime, bound.prepared)
    _validate_v0_optimize(bound)
    _checkpoint_ports(bound)
    rewards = _validated_physical_rewards(bound, require_inactive=False)
    summary = request.summary
    if (
        summary.recipe_id != bound.preflight.environment.materialized.recipe_id
        or summary.launch_id != bound.runtime.launch_binding.launch_id
        or summary.bound_contract_id != bound.graph_binding.bound_contract_id
    ):
        raise RuntimeCheckpointError("checkpoint summary differs from bound run")
    if request.cadence != (
        bound.preflight.compiled.launch.checkpoint_every_optimizer_steps
    ):
        raise RuntimeCheckpointError("checkpoint cadence differs from LaunchSpec")
    if summary.committed_steps > bound.prepared.training.max_optimizer_steps:
        raise RuntimeCheckpointError("checkpoint step exceeds training stop")
    _optimized, update = _final_update(summary.last_iteration.value.payload)
    if update.optimizer_step + 1 != summary.committed_steps:
        raise RuntimeCheckpointError(
            "policy update step disagrees with checkpoint summary"
        )
    return bound, rewards, _build_full_contract(bound)


def _capture_and_commit(
    *,
    bound: ProductionBoundRun,
    summary: object,
    rewards: FrozenMapping,
    contract: CheckpointContract,
    terminal_artifacts: FrozenMapping,
) -> tuple[CommittedCheckpoint, _RunCheckpointSummaryState]:
    from visual_rl.runtime.types import PendingRunSummary

    if not isinstance(summary, PendingRunSummary):
        raise TypeError("summary must be PendingRunSummary")
    if not isinstance(terminal_artifacts, FrozenMapping):
        raise TypeError("terminal_artifacts must be FrozenMapping")
    ports = _checkpoint_ports(bound)
    reference_state = _bound_reference_policy_state(bound)
    _validate_contract_modes(contract, reference_state)
    _optimized, update = _final_update(summary.last_iteration.value.payload)
    data_view = ports.data_plane.capture_checkpoint_view(summary.committed_steps)
    if data_view.next_optimizer_step != summary.committed_steps:
        raise RuntimeCheckpointError("data plane next step disagrees with summary")
    rank = bound.runtime.session.runtime_facts.rank
    world_size = bound.runtime.session.runtime_facts.world_size
    rng_snapshot = RankRNGSnapshot.capture_current(rank)
    progress = CheckpointProgress(
        global_step=summary.committed_steps,
        iteration=summary.committed_steps,
        next_optimizer_step=data_view.next_optimizer_step,
        next_source_id=data_view.next_source_id,
        next_prompt_batch_id=data_view.next_prompt_batch_identity,
        next_phase_id=data_view.next_phase_id,
        active_reward_ids=data_view.active_rewards,
        source_cursors=data_view.source_cursors,
        dynamics_selection_policy=ports.dynamics_selection_policy,
        gradient_accumulation_position=0,
        ema_state_saved=False,
        reference_state_saved=reference_state.has_active_reference_owner,
        execution_transform_plan_id=bound.transforms.plan_id,
        rng_state_id=rng_snapshot.state_identity,
    )
    safe_point = CheckpointSafePoint.from_update_result(
        rank=rank,
        world_size=world_size,
        update_result=update.transaction,
        group_geometry_id=data_view.group_geometry_id,
    )
    safe_point.assert_ready(progress)
    run_summary = _RunCheckpointSummaryState(
        run_id=summary.launch_id,
        recipe_id=summary.recipe_id,
        bound_contract_id=summary.bound_contract_id,
        checkpoint_contract_id=contract.checkpoint_contract_id,
        update_execution_plan_id=ports.update_execution_plan.plan_id,
        start_optimizer_step=summary.start_optimizer_step,
        committed_steps=summary.committed_steps,
        update_count=summary.update_count,
        terminal=bool(terminal_artifacts),
        terminal_artifacts=terminal_artifacts,
        bound_reward_resource_ids=rewards,
        last_metrics=FrozenMapping(
            _last_metrics(summary.last_iteration.value.identity.batch_size, update)
        ),
    )
    component_states = {
        "data_plane": _ComponentState(
            "data_plane",
            1,
            data_view.state.to_payload(),
        ),
        "run_checkpoint_summary": _ComponentState(
            "run_checkpoint_summary",
            1,
            run_summary,
        ),
        "lr_scheduler": _ComponentState(
            "lr_scheduler",
            1,
            _capture_state_dict(bound.prepared.lr_scheduler, "lr_scheduler"),
        ),
        "model": _ComponentState(
            "model",
            2,
            ModelStateAdapter(bound.prepared.manager.parameter_state)
            .capture()
            .to_payload(),
        ),
        "optimizer": _ComponentState(
            "optimizer",
            1,
            _capture_state_dict(bound.prepared.optimizer, "optimizer"),
        ),
    }
    collector = CheckpointStateCollector(
        component_state_sources={
            name: (lambda state=state: state)
            for name, state in component_states.items()
        },
        dynamics_selection_policy_source=(lambda: ports.dynamics_selection_policy),
        rng_state_source=lambda observed_rank: _same_rng_snapshot(
            rng_snapshot,
            observed_rank,
        ),
    )
    output_dir = _prepare_output_dir(bound.preflight.compiled.launch.output_dir)
    committed = CheckpointCoordinator(
        manager=AtomicCheckpointManager(output_dir / "checkpoints"),
        backend=SingleProcessCheckpointBackend(),
        collector=collector,
    ).checkpoint(
        contract=contract,
        progress=progress,
        safe_point=safe_point,
    )
    expected = output_dir / "checkpoints" / f"step-{summary.committed_steps}"
    if committed.path != expected:
        raise RuntimeCheckpointError("committed checkpoint path is not canonical")
    return committed, run_summary


class CoordinatorRestoreService:
    """Restore prepared owners first, then logical state and RNG at G3."""

    def __init__(self, *, finalizer: CoordinatorRunFinalizer | None = None) -> None:
        self._finalizer = CoordinatorRunFinalizer() if finalizer is None else finalizer
        if not isinstance(self._finalizer, CoordinatorRunFinalizer):
            raise TypeError("finalizer must be CoordinatorRunFinalizer")

    def restore_prepared(
        self,
        request: PreparedRestoreRequest,
    ) -> PreparedRestoreResult:
        if not isinstance(request, PreparedRestoreRequest):
            raise TypeError("request must be PreparedRestoreRequest")
        _validate_v0_prepared(request.runtime, request.prepared)
        manager = _manager_for_checkpoint(request.checkpoint_path)
        inspection = manager.inspect_complete(request.checkpoint_path)
        reference_state = _prepared_reference_policy_state(request)
        _validate_contract_modes(inspection.contract, reference_state)
        _validate_reference_state_progress(inspection, reference_state)
        live_projection = _build_prepared_contract(
            request,
            reference_state=reference_state,
        )
        assert_compatible_prepared_contract(
            live_projection,
            inspection.contract.prepared_projection(),
        )
        reader = RankCheckpointReader(manager)
        facts = request.runtime.session.runtime_facts
        snapshot = reader.read_rank_snapshot(
            inspection,
            expected_world_size=facts.world_size,
            expected_rank=facts.rank,
        )
        _validate_rng_binding(inspection, snapshot)
        decoded = _decode_states(snapshot, request.prepared)
        _restore_prepared_owners(request.prepared, decoded)
        restored_ids = FrozenMapping(
            {
                name: _validated_state_id(inspection, snapshot, name)
                for name in ("lr_scheduler", "model", "optimizer")
            }
        )
        continuation = _PreparedContinuation(
            inspection=inspection,
            snapshot=snapshot,
            prepared_projection=live_projection,
            prepared_owner=request.prepared,
            preflight_owner=request.preflight,
            runtime_owner=request.runtime,
            graph_owner=request.graph,
        )
        return PreparedRestoreResult(
            checkpoint_path=inspection.committed.path,
            next_optimizer_step=inspection.progress.next_optimizer_step,
            restored_state_ids=restored_ids,
            continuation=continuation,
        )

    def restore_bound(self, request: BoundRestoreRequest) -> BoundRestoreResult:
        if not isinstance(request, BoundRestoreRequest):
            raise TypeError("request must be BoundRestoreRequest")
        _validate_v0_prepared(request.runtime, request.prepared)
        _validate_v0_optimize(request.bound)
        ports = _checkpoint_ports(request.bound)
        rewards = _validated_physical_rewards(
            request.bound,
            require_inactive=True,
        )
        continuation = request.prepared_restore.continuation
        if not isinstance(continuation, _PreparedContinuation):
            raise TypeError("prepared restore continuation has the wrong service type")
        if (
            continuation.prepared_owner is not request.prepared
            or continuation.preflight_owner is not request.preflight
            or continuation.runtime_owner is not request.runtime
            or continuation.graph_owner is not request.graph
        ):
            raise RuntimeCheckpointError("prepared restore continuation owner changed")
        if request.prepared_restore.checkpoint_path != request.checkpoint_path:
            raise RuntimeCheckpointError("prepared and bound checkpoint paths differ")

        manager = _manager_for_checkpoint(request.checkpoint_path)
        reference_state = _bound_reference_policy_state(request.bound)
        full_contract = _build_full_contract(
            request.bound,
            reference_state=reference_state,
        )
        inspection = manager.inspect_complete(
            request.checkpoint_path,
            expected_contract=full_contract,
        )
        assert_compatible_contract(full_contract, inspection.contract)
        _validate_contract_modes(inspection.contract, reference_state)
        _validate_reference_state_progress(inspection, reference_state)
        if continuation.prepared_projection != full_contract.prepared_projection():
            raise RuntimeCheckpointError(
                "prepared and full contract projections differ"
            )
        if continuation.inspection.committed.state_tree_id != (
            inspection.committed.state_tree_id
        ):
            raise RuntimeCheckpointError("checkpoint changed between restore gates")

        facts = request.runtime.session.runtime_facts
        snapshot = RankCheckpointReader(manager).read_rank_snapshot(
            inspection,
            expected_world_size=facts.world_size,
            expected_rank=facts.rank,
        )
        _validate_rng_binding(inspection, snapshot)
        decoded = _decode_states(snapshot, request.prepared)
        _validate_cross_gate_continuity(
            request.prepared_restore,
            continuation,
            inspection,
            snapshot,
        )
        if snapshot.component_names != continuation.snapshot.component_names:
            raise RuntimeCheckpointError(
                "checkpoint component set changed between gates"
            )
        if snapshot.safe_point.safe_point_id != (
            continuation.snapshot.safe_point.safe_point_id
        ):
            raise RuntimeCheckpointError("checkpoint safe point changed between gates")
        if ports.dynamics_selection_policy != (
            snapshot.dynamics_selection_policy
        ) or ports.dynamics_selection_policy != (
            inspection.progress.dynamics_selection_policy
        ):
            raise RuntimeCheckpointError("live Dynamics selection policy changed")
        if decoded.run_summary.update_execution_plan_id != (
            ports.update_execution_plan.plan_id
        ):
            raise RuntimeCheckpointError("live update execution plan changed")
        _validate_run_summary(
            request.bound,
            inspection,
            decoded.run_summary,
            rewards,
        )

        old_data_view = ports.data_plane.capture_checkpoint_view(0)
        old_rng = RankRNGSnapshot.capture_current(facts.rank)
        data_mutated = False
        try:
            restored_state = ports.data_plane.restore_checkpoint_state(
                decoded.data_plane_payload
            )
            data_mutated = True
            restored_view = ports.data_plane.capture_checkpoint_view(
                inspection.progress.next_optimizer_step
            )
            _validate_restored_data(
                restored_state,
                restored_view,
                inspection,
                snapshot,
                ports,
            )
            terminal_progress = inspection.progress.next_optimizer_step == (
                request.prepared.training.max_optimizer_steps
            )
            if decoded.run_summary.terminal != terminal_progress:
                raise RuntimeCheckpointError(
                    "checkpoint terminal metadata disagrees with training progress"
                )
            completed = None
            if terminal_progress:
                completed = self._finalizer.finalize(
                    bound=request.bound,
                    inspection=inspection,
                    summary=decoded.run_summary,
                    rewards=rewards,
                )
            restored_ids = FrozenMapping(
                {
                    "data_plane": _validated_state_id(
                        inspection,
                        snapshot,
                        "data_plane",
                    ),
                    "dynamics_selection_policy": _validated_state_id(
                        inspection,
                        snapshot,
                        "dynamics_selection_policy",
                        intrinsic=snapshot.dynamics_selection_policy.policy_identity,
                    ),
                    "progress": _validated_state_id(
                        inspection,
                        snapshot,
                        "progress",
                        intrinsic=inspection.progress.progress_id,
                    ),
                    "rng": _validated_state_id(
                        inspection,
                        snapshot,
                        "rng",
                        intrinsic=snapshot.rng_state.state_identity,
                    ),
                }
            )
            try:
                snapshot.rng_state.restore_current()
            except BaseException as exc:
                _rollback_rng_and_data(
                    old_rng,
                    ports,
                    old_data_view.state.to_payload(),
                    exc,
                )
                data_mutated = False
                raise RuntimeCheckpointError(
                    "RNG restore failed at the final gate"
                ) from exc
            return BoundRestoreResult(
                checkpoint_path=inspection.committed.path,
                next_optimizer_step=inspection.progress.next_optimizer_step,
                restored_state_ids=restored_ids,
                completed_result=completed,
            )
        except BaseException as failure:
            if data_mutated:
                try:
                    ports.data_plane.restore_checkpoint_state(
                        old_data_view.state.to_payload()
                    )
                except BaseException as rollback_error:  # noqa: BLE001
                    # Restore rollback is best-effort even under cancellation.
                    if hasattr(failure, "add_note"):
                        failure.add_note(
                            "data-plane rollback failed: "
                            f"{type(rollback_error).__name__}: {rollback_error}"
                        )
            raise


def _build_prepared_contract(
    request: PreparedRestoreRequest,
    *,
    reference_state: ReferencePolicyStateEvidence | None = None,
) -> PreparedCheckpointContract:
    materialized = request.preflight.environment.materialized
    effective_reference_state = (
        _prepared_reference_policy_state(request)
        if reference_state is None
        else reference_state
    )
    return build_prepared_checkpoint_contract(
        PreparedCheckpointBuildInput(
            recipe=materialized,
            artifact_binding_set=(
                request.preflight.environment.component_artifact_bindings
            ),
            runtime_facts=request.runtime.session.runtime_facts,
            parameter_state=request.prepared.manager.parameter_state,
            model_execution_numerics=(
                request.prepared.manager.model_execution_numerics
            ),
            optimizer=request.prepared.optimizer,
            scaler=None,
            lr_scheduler=request.prepared.lr_scheduler,
            execution_transform_plan=_execution_transform_plan(materialized),
            ema_state_schema=_EMA_STATE_SCHEMA,
            reference_state_schema=_reference_state_schema(effective_reference_state),
            state_schema_versions=_STATE_SCHEMA_VERSIONS,
        )
    )


def _build_full_contract(
    bound: ProductionBoundRun,
    *,
    reference_state: ReferencePolicyStateEvidence | None = None,
) -> CheckpointContract:
    materialized = bound.preflight.environment.materialized
    effective_reference_state = (
        _bound_reference_policy_state(bound)
        if reference_state is None
        else reference_state
    )
    return build_checkpoint_contract(
        CheckpointBuildInput(
            recipe=materialized,
            artifact_binding_set=(
                bound.preflight.environment.component_artifact_bindings
            ),
            runtime_facts=bound.runtime.session.runtime_facts,
            graph_binding=bound.graph_binding,
            runtime_bound_contracts=bound.evidence.runtime_bound_contracts,
            parameter_state=bound.prepared.manager.parameter_state,
            model_execution_numerics=(bound.prepared.manager.model_execution_numerics),
            optimizer=bound.prepared.optimizer,
            scaler=None,
            lr_scheduler=bound.prepared.lr_scheduler,
            execution_transform_plan=_execution_transform_plan(materialized),
            preprocess_identity=bound.evidence.preprocess_identity,
            preprocess_requirement_set_id=(
                bound.evidence.preprocess_requirement_set_id
            ),
            ema_state_schema=_EMA_STATE_SCHEMA,
            reference_state_schema=_reference_state_schema(effective_reference_state),
            state_schema_versions=_STATE_SCHEMA_VERSIONS,
        )
    )


def _prepared_reference_policy_state(
    request: PreparedRestoreRequest,
) -> ReferencePolicyStateEvidence:
    if not isinstance(request, PreparedRestoreRequest):
        raise TypeError("request must be PreparedRestoreRequest")
    model_binding = request.graph.components.binding("model")
    model_contract = _model_contract(model_binding.declared_contract)
    return derive_reference_policy_state_evidence(
        algorithm=_algorithm_execution_plan(request.preflight.environment.materialized),
        model=model_contract,
        model_execution_numerics=request.prepared.manager.model_execution_numerics,
    )


def _bound_reference_policy_state(
    bound: ProductionBoundRun,
) -> ReferencePolicyStateEvidence:
    if not isinstance(bound, ProductionBoundRun):
        raise TypeError("bound must be ProductionBoundRun")
    graph_declared = bound.graph.components.binding("model").declared_contract
    graph_model = _model_contract(graph_declared)
    runtime_model = _model_contract(
        bound.evidence.model_runtime_contract.artifact.declared
    )
    if runtime_model != graph_model:
        raise RuntimeCheckpointError(
            "runtime-bound model reference contract differs from the constructed "
            "component graph"
        )
    evidence = bound.evidence.reference_policy_state_evidence
    if not isinstance(evidence, ReferencePolicyStateEvidence):
        raise TypeError("bound G3 evidence must retain ReferencePolicyStateEvidence")
    evidence.assert_integrity()
    numerics = bound.prepared.manager.model_execution_numerics
    if evidence.model_execution_numerics_id != numerics.execution_numerics_id:
        raise RuntimeCheckpointError(
            "bound reference-policy evidence uses stale model execution numerics"
        )
    if evidence.source_projection_id != numerics.source_projection_id:
        raise RuntimeCheckpointError(
            "bound reference-policy evidence uses a stale state projection"
        )
    if evidence.model_provides_reference_policy is not (
        runtime_model.provides_reference_policy
    ):
        raise RuntimeCheckpointError(
            "bound reference-policy evidence differs from the model contract"
        )
    algorithm = _algorithm_execution_plan(bound.preflight.environment.materialized)
    if (
        evidence.algorithm_plan_id != algorithm.plan_id
        or evidence.algorithm_requires_reference_statistics
        is not algorithm.requires_reference_statistics
    ):
        raise RuntimeCheckpointError(
            "bound reference-policy evidence differs from the algorithm plan"
        )
    reference_view_ids = {
        item.evidence_id
        for item in numerics.parameter_view_evidence
        if item.parameter_view is ParameterView.REFERENCE
    }
    if evidence.parameter_view_evidence_id is not None and (
        evidence.parameter_view_evidence_id not in reference_view_ids
    ):
        raise RuntimeCheckpointError(
            "bound reference-policy evidence is absent from prepared numerics"
        )
    return evidence


def _model_contract(declared: object) -> ModelContract:
    if not isinstance(declared, DeclaredContract):
        raise TypeError("model binding must retain a typed DeclaredContract")
    if declared.component_kind != "model" or declared.model is None:
        raise RuntimeCheckpointError(
            "checkpoint reference ownership requires a model contract"
        )
    return declared.model


def _reference_state_schema(evidence: ReferencePolicyStateEvidence) -> str:
    if not isinstance(evidence, ReferencePolicyStateEvidence):
        raise TypeError("evidence must be ReferencePolicyStateEvidence")
    evidence.assert_integrity()
    if evidence.has_active_reference_owner and (
        evidence.state_schema != _REFERENCE_STATE_SCHEMA
    ):
        raise RuntimeCheckpointError(
            "V0 supports only reference state derived from the model artifact"
        )
    return evidence.checkpoint_state_schema


def _validate_reference_state_progress(
    inspection: CheckpointInspection,
    evidence: ReferencePolicyStateEvidence,
) -> None:
    if not isinstance(inspection, CheckpointInspection):
        raise TypeError("inspection must be CheckpointInspection")
    if inspection.progress.reference_state_saved is not (
        evidence.has_active_reference_owner
    ):
        raise RuntimeCheckpointError(
            "checkpoint progress reference-state claim disagrees with effective "
            "typed ownership"
        )


def _execution_transform_plan(materialized: object) -> ExecutionTransformPlan:
    from visual_rl.composition.recipes.schema import MaterializedRecipe

    if not isinstance(materialized, MaterializedRecipe):
        raise TypeError("materialized must be MaterializedRecipe")
    return materialized.resolved.execution_policy.transform_plan


def _algorithm_execution_plan(materialized: object) -> AlgorithmExecutionPlan:
    from visual_rl.composition.recipes.schema import MaterializedRecipe

    if not isinstance(materialized, MaterializedRecipe):
        raise TypeError("materialized must be MaterializedRecipe")
    resolved = materialized.resolved
    return AlgorithmExecutionPlan.from_spec(
        resolved.algorithm_spec,
        execution_policy=resolved.execution_policy.to_receipt(),
    )


def _validate_contract_modes(
    contract: CheckpointContract,
    reference_state: ReferencePolicyStateEvidence,
) -> None:
    reference_state.assert_integrity()
    if (
        contract.model_execution_numerics_id
        != reference_state.model_execution_numerics_id
        or contract.model_state_projection_id != reference_state.source_projection_id
    ):
        raise RuntimeCheckpointError(
            "checkpoint model numerics/projection differ from reference-state evidence"
        )
    if contract.scaler_schema != "none.v1":
        raise RuntimeCheckpointError("V0 does not restore mixed-precision scalers")
    if contract.ema_state_schema != _EMA_STATE_SCHEMA:
        raise RuntimeCheckpointError("V0 does not restore independent EMA state")
    expected_reference_schema = _reference_state_schema(reference_state)
    if contract.reference_state_schema != expected_reference_schema:
        raise RuntimeCheckpointError(
            "checkpoint reference-state schema disagrees with effective typed ownership"
        )
    if contract.state_schema_versions != _STATE_SCHEMA_VERSIONS:
        raise RuntimeCheckpointError("checkpoint state schema set is not V0 exact")


def _validate_v0_prepared(runtime: object, prepared: ProductionPreparedRun) -> None:
    from visual_rl.runtime.types import ProductionRuntime

    if not isinstance(runtime, ProductionRuntime):
        raise TypeError("runtime must be ProductionRuntime")
    if not isinstance(prepared, ProductionPreparedRun):
        raise TypeError("prepared must be ProductionPreparedRun")
    facts = runtime.session.runtime_facts
    if (
        facts.distribution_mode != "single"
        or facts.rank != 0
        or facts.local_rank != 0
        or facts.world_size != 1
    ):
        raise RuntimeCheckpointError("V0 checkpoint IO supports world_size=1 only")
    if prepared.training.gradient_accumulation_steps != 1:
        raise RuntimeCheckpointError("V0 checkpoint IO requires GA=1")
    if prepared.lr_scheduler is None:
        raise RuntimeCheckpointError("V0 requires one LR scheduler state owner")
    if getattr(runtime.session.accelerator, "scaler", None) is not None:
        raise RuntimeCheckpointError("V0 does not support accelerator scaler state")


def _validate_v0_optimize(bound: ProductionBoundRun) -> None:
    optimize = bound.assembly.optimize
    if not isinstance(optimize, OptimizeStage):
        raise RuntimeCheckpointError("V0 checkpoint IO requires OptimizeStage")
    if optimize.optimizer is not bound.prepared.optimizer:
        raise RuntimeCheckpointError("OptimizeStage optimizer owner changed")
    if optimize.lr_scheduler is not bound.prepared.lr_scheduler:
        raise RuntimeCheckpointError("OptimizeStage LR scheduler owner changed")
    if optimize.scaler is not None:
        raise RuntimeCheckpointError("V0 OptimizeStage scaler must be None")
    if optimize.ema_update is not None:
        raise RuntimeCheckpointError("V0 does not support independent EMA updates")
    if optimize.reference_update is not None:
        raise RuntimeCheckpointError(
            "V0 does not support independent reference state updates"
        )


def _checkpoint_ports(bound: ProductionBoundRun) -> StageCheckpointPorts:
    ports = bound.assembly.checkpoint_ports
    if not isinstance(ports, StageCheckpointPorts):
        raise RuntimeCheckpointError("default checkpoint IO requires checkpoint_ports")
    ports.validate_assembly(
        prelude=bound.assembly.prelude,
        rollout=bound.assembly.rollout,
        optimize=bound.assembly.optimize,
    )
    return ports


def _validated_physical_rewards(
    bound: ProductionBoundRun,
    *,
    require_inactive: bool,
) -> FrozenMapping:
    evidence = bound.evidence.bound_reward_resource_ids
    graph = bound.graph_binding.bound_reward_resource_ids
    container = bound.runtime.session.resource_container
    if not isinstance(container, DefaultRuntimeResourceContainer):
        raise RuntimeCheckpointError("V0 requires the default reward resource owner")
    live = container.bound_reward_resource_ids
    if not evidence or evidence != graph or evidence != live:
        raise RuntimeCheckpointError(
            "physical reward resource ids differ across G3 evidence and owner"
        )
    if require_inactive:
        if container.state is not RewardResourceState.ACQUIRED or container.is_active:
            raise RuntimeCheckpointError(
                "bound restore requires acquired but inactive reward resources"
            )
    elif container.state not in {
        RewardResourceState.ACQUIRED,
        RewardResourceState.ACTIVE,
    }:
        raise RuntimeCheckpointError("checkpoint reward resource owner is not live")
    return evidence


def _final_update(payload: object) -> tuple[OptimizedIteration, PolicyUpdateResult]:
    if not isinstance(payload, OptimizedIteration):
        raise TypeError("final iteration payload must be OptimizedIteration")
    if not isinstance(payload.update, PolicyUpdateResult):
        raise TypeError("final OptimizedIteration update must be PolicyUpdateResult")
    return payload, payload.update


def _capture_state_dict(owner: object, name: str) -> Mapping[str, object]:
    state_dict = getattr(owner, "state_dict", None)
    load_state_dict = getattr(owner, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        raise TypeError(f"{name} must implement state_dict/load_state_dict")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{name}.state_dict() must return a mapping")
    return copy.deepcopy(dict(state))


def _decode_states(
    snapshot: RankCheckpointSnapshot,
    prepared: ProductionPreparedRun,
) -> _DecodedStates:
    if snapshot.component_names != _COMPONENT_NAMES:
        raise RuntimeCheckpointError(
            "checkpoint rank shard must contain the exact V0 component set"
        )
    values = {name: _component_state(snapshot, name) for name in _COMPONENT_NAMES}
    data_payload = values["data_plane"].payload
    if not isinstance(data_payload, Mapping):
        raise TypeError("data_plane checkpoint payload must be a mapping")
    _validate_data_payload_schema(data_payload)

    summary_payload = values["run_checkpoint_summary"].payload
    if not isinstance(summary_payload, _RunCheckpointSummaryState):
        raise TypeError("run_checkpoint_summary payload must be typed")
    run_summary = _RunCheckpointSummaryState.from_payload(summary_payload.to_payload())

    model_payload = values["model"].payload
    if not isinstance(model_payload, Mapping):
        raise TypeError("model checkpoint payload must be a mapping")
    model_state = ModelParameterState.from_payload(
        prepared.manager.parameter_state.topology,
        model_payload,
    )
    optimizer_payload = values["optimizer"].payload
    scheduler_payload = values["lr_scheduler"].payload
    optimizer_state = _validate_optimizer_state(
        optimizer_payload,
        prepared.optimizer,
    )
    scheduler_state = _validate_scheduler_state(
        scheduler_payload,
        prepared.lr_scheduler,
    )
    return _DecodedStates(
        data_plane_payload=copy.deepcopy(dict(data_payload)),
        run_summary=run_summary,
        model_state=model_state,
        optimizer_state=optimizer_state,
        lr_scheduler_state=scheduler_state,
    )


def _component_state(
    snapshot: RankCheckpointSnapshot,
    name: str,
) -> _ComponentState:
    value = snapshot.component_state(name)
    if not isinstance(value, _ComponentState):
        raise TypeError(f"checkpoint component {name!r} has no typed envelope")
    reconstructed = _ComponentState(value.kind, value.schema_version, value.payload)
    if reconstructed.kind != name:
        raise ValueError(f"checkpoint component {name!r} kind disagrees")
    return reconstructed


def _validate_data_payload_schema(payload: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "kind",
        "prelude_id",
        "placement_contract_id",
        "phase_schedule_state",
        "sampler_state",
    }
    if set(payload) != expected:
        raise ValueError("data-plane state has an invalid exact key set")
    if payload["schema_version"] != 1 or payload["kind"] != "data_plane_prelude_state":
        raise ValueError("data-plane state schema or kind is invalid")
    if not isinstance(payload["phase_schedule_state"], Mapping) or not isinstance(
        payload["sampler_state"], Mapping
    ):
        raise TypeError("data-plane nested states must be mappings")


def _validate_optimizer_state(
    payload: object,
    optimizer: object,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {"state", "param_groups"}:
        raise ValueError("optimizer state has an invalid exact key set")
    live = _capture_state_dict(optimizer, "optimizer")
    incoming_groups = payload["param_groups"]
    live_groups = live["param_groups"]
    if not isinstance(incoming_groups, list) or not isinstance(live_groups, list):
        raise TypeError("optimizer param_groups must be lists")
    if len(incoming_groups) != len(live_groups):
        raise ValueError("optimizer state parameter-group count changed")
    for incoming, observed in zip(incoming_groups, live_groups, strict=True):
        if not isinstance(incoming, Mapping) or not isinstance(observed, Mapping):
            raise TypeError("optimizer parameter groups must be mappings")
        if set(incoming) != set(observed):
            raise ValueError("optimizer parameter-group schema changed")
        if len(incoming["params"]) != len(observed["params"]):
            raise ValueError("optimizer parameter-group topology changed")
    return copy.deepcopy(dict(payload))


def _validate_scheduler_state(
    payload: object,
    scheduler: object,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("lr_scheduler checkpoint payload must be a mapping")
    live = _capture_state_dict(scheduler, "lr_scheduler")
    if set(payload) != set(live):
        raise ValueError("lr_scheduler state schema changed")
    return copy.deepcopy(dict(payload))


def _restore_prepared_owners(
    prepared: ProductionPreparedRun,
    decoded: _DecodedStates,
) -> None:
    model = ModelStateAdapter(prepared.manager.parameter_state)
    model_backup = model.capture()
    optimizer_backup = _capture_state_dict(prepared.optimizer, "optimizer")
    scheduler_backup = _capture_state_dict(
        prepared.lr_scheduler,
        "lr_scheduler",
    )
    try:
        model.restore(decoded.model_state)
        prepared.optimizer.load_state_dict(copy.deepcopy(decoded.optimizer_state))
        prepared.lr_scheduler.load_state_dict(copy.deepcopy(decoded.lr_scheduler_state))
    except BaseException as exc:
        rollback_errors: list[BaseException] = []
        for restore in (
            lambda: prepared.lr_scheduler.load_state_dict(scheduler_backup),
            lambda: prepared.optimizer.load_state_dict(optimizer_backup),
            lambda: model.restore(model_backup),
        ):
            try:
                restore()
            except BaseException as rollback_error:  # noqa: BLE001
                # Preserve every rollback failure on the primary exception.
                rollback_errors.append(rollback_error)
        for rollback_error in rollback_errors:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "prepared restore rollback failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        raise RuntimeCheckpointError(
            "prepared model/optimizer/LR restore failed and rollback was attempted"
        ) from exc


def _validate_restored_data(
    restored_state: object,
    restored_view: object,
    inspection: CheckpointInspection,
    snapshot: RankCheckpointSnapshot,
    ports: StageCheckpointPorts,
) -> None:
    from visual_rl.data.prelude import (
        DataPlaneCheckpointView,
        DataPlanePreludeState,
    )

    if not isinstance(restored_state, DataPlanePreludeState):
        raise TypeError("data plane restore must return DataPlanePreludeState")
    if not isinstance(restored_view, DataPlaneCheckpointView):
        raise TypeError("data plane capture must return DataPlaneCheckpointView")
    progress = inspection.progress
    if restored_view.state != restored_state:
        raise RuntimeCheckpointError("restored data state and captured view differ")
    if (
        restored_view.next_optimizer_step != progress.next_optimizer_step
        or restored_view.next_source_id != progress.next_source_id
        or restored_view.next_phase_id != progress.next_phase_id
        or restored_view.active_rewards != progress.active_reward_ids
        or restored_view.source_cursors != progress.source_cursors
        or restored_view.next_prompt_batch_identity != progress.next_prompt_batch_id
    ):
        raise RuntimeCheckpointError("restored next data window differs from progress")
    if (
        restored_view.group_geometry_id != ports.data_plane.group_geometry_id
        or restored_view.group_geometry_id != snapshot.safe_point.group_geometry_id
    ):
        raise RuntimeCheckpointError("restored group geometry differs from safe point")
    if restored_view.has_open_reservation:
        raise RuntimeCheckpointError("restored data plane has an open reservation")


def _validate_run_summary(
    bound: ProductionBoundRun,
    inspection: CheckpointInspection,
    summary: _RunCheckpointSummaryState,
    rewards: FrozenMapping,
) -> None:
    if (
        summary.run_id != bound.runtime.launch_binding.launch_id
        or summary.recipe_id != bound.preflight.environment.materialized.recipe_id
        or summary.bound_contract_id != bound.graph_binding.bound_contract_id
        or summary.checkpoint_contract_id != inspection.contract.checkpoint_contract_id
        or summary.update_execution_plan_id
        != _checkpoint_ports(bound).update_execution_plan.plan_id
        or summary.bound_reward_resource_ids != rewards
    ):
        raise RuntimeCheckpointError("final run summary differs from current bound run")
    if summary.committed_steps != inspection.progress.next_optimizer_step:
        raise RuntimeCheckpointError(
            "run checkpoint summary step differs from progress"
        )
    if summary.committed_steps != inspection.committed.step:
        raise RuntimeCheckpointError(
            "run checkpoint summary step differs from checkpoint receipt"
        )


def _manager_for_checkpoint(path: Path) -> AtomicCheckpointManager:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("checkpoint_path must be an absolute Path")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint_path must be a real directory")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("checkpoint root must be a real directory")
    return AtomicCheckpointManager(path.parent)


def _same_rng_snapshot(snapshot: RankRNGSnapshot, rank: int) -> RankRNGSnapshot:
    if rank != snapshot.rank:
        raise ValueError("collector requested RNG state for the wrong rank")
    return snapshot


def _validate_rng_binding(
    inspection: CheckpointInspection,
    snapshot: RankCheckpointSnapshot,
) -> None:
    if inspection.progress.rng_state_id != snapshot.rng_state.state_identity:
        raise RuntimeCheckpointError(
            "checkpoint progress RNG identity differs from rank shard"
        )


def _validate_cross_gate_continuity(
    receipt: PreparedRestoreResult,
    continuation: _PreparedContinuation,
    inspection: CheckpointInspection,
    snapshot: RankCheckpointSnapshot,
) -> None:
    first = continuation.inspection.committed
    current = inspection.committed
    if (
        first.path != current.path
        or first.checkpoint_contract_id != current.checkpoint_contract_id
        or first.progress_id != current.progress_id
        or first.state_tree_id != current.state_tree_id
    ):
        raise RuntimeCheckpointError(
            "checkpoint receipt changed between prepared and bound gates"
        )
    if receipt.next_optimizer_step != inspection.progress.next_optimizer_step:
        raise RuntimeCheckpointError(
            "prepared restore next step changed before bound gate"
        )
    expected_ids = FrozenMapping(
        {
            name: _validated_state_id(inspection, snapshot, name)
            for name in ("lr_scheduler", "model", "optimizer")
        }
    )
    if receipt.restored_state_ids != expected_ids:
        raise RuntimeCheckpointError(
            "prepared restored-state receipt changed before bound gate"
        )
    if continuation.snapshot.rng_state != snapshot.rng_state:
        raise RuntimeCheckpointError("rank RNG shard changed between restore gates")


def _validated_state_id(
    inspection: CheckpointInspection,
    snapshot: RankCheckpointSnapshot,
    name: str,
    *,
    intrinsic: str | None = None,
) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("validated state name must be non-empty")
    shard_path = inspection.committed.path / "rank_shards" / f"rank-{snapshot.rank}.pt"
    _regular_file(shard_path)
    payload = {
        "schema_version": 1,
        "kind": "validated_checkpoint_state",
        "state_tree_id": inspection.committed.state_tree_id,
        "checkpoint_contract_id": inspection.contract.checkpoint_contract_id,
        "progress_id": inspection.progress.progress_id,
        "rank": snapshot.rank,
        "world_size": snapshot.world_size,
        "shard_sha256": _sha256_file(shard_path),
        "component_names": list(snapshot.component_names),
        "state_name": name,
        "intrinsic_identity": intrinsic,
    }
    return _sha256_bytes(canonical_json_text(payload).encode("utf-8"))


def _rollback_rng_and_data(
    old_rng: RankRNGSnapshot,
    ports: StageCheckpointPorts,
    old_data_payload: Mapping[str, object],
    primary: BaseException,
) -> None:
    for label, rollback in (
        ("RNG", old_rng.restore_current),
        (
            "data plane",
            lambda: ports.data_plane.restore_checkpoint_state(old_data_payload),
        ),
    ):
        try:
            rollback()
        except BaseException as error:  # noqa: BLE001
            # Safe-point rollback must annotate cancellation-path failures.
            if hasattr(primary, "add_note"):
                primary.add_note(
                    f"{label} rollback failed: {type(error).__name__}: {error}"
                )


def _last_metrics(
    sample_count: int, update: PolicyUpdateResult
) -> dict[str, float | int]:
    return {
        "step": update.optimizer_step,
        "sample_count": sample_count,
        "active_transition_count": update.active_transition_count,
        "loss": update.loss,
        "policy_loss": update.policy_loss,
        "reference_kl": update.reference_kl,
        "approx_kl": update.approx_kl,
        "clipfrac": update.clipfrac,
        "logprob_delta_max": update.logprob_delta_max,
        "gradient_norm_pre_clip": update.gradient_norm_pre_clip,
        "gradient_norm_post_clip": update.gradient_norm_post_clip,
    }


def _validate_last_metrics(values: Mapping[str, object], committed_steps: int) -> None:
    required = {"step", "sample_count", "active_transition_count"}
    if not required.issubset(values):
        raise ValueError("final summary metrics are missing required counters")
    if values["step"] != committed_steps - 1:
        raise ValueError("final summary metric step is not the final update")
    for name in required:
        if type(values[name]) is not int or values[name] < 1 - int(name == "step"):
            raise ValueError(f"final summary metric {name!r} is invalid")
    for name, value in values.items():
        if name in required:
            continue
        if type(value) is not float or not _finite(value):
            raise ValueError(f"final summary metric {name!r} must be finite float")


def _run_manifest_payload(
    *,
    bound: ProductionBoundRun,
    summary: object,
    contract: CheckpointContract,
    checkpoint_relative_path: str,
    rewards: FrozenMapping,
    resolved_recipe_sha256: str,
    metrics_sha256: str,
) -> dict[str, object]:
    from visual_rl.runtime.types import PendingRunSummary

    if not isinstance(summary, PendingRunSummary):
        raise TypeError("summary must be PendingRunSummary")
    return {
        "schema_version": 1,
        "kind": "visual_rl_final_run_manifest",
        "run_id": summary.launch_id,
        "recipe_id": summary.recipe_id,
        "bound_contract_id": summary.bound_contract_id,
        "checkpoint_contract_id": contract.checkpoint_contract_id,
        "update_execution_plan_id": (
            _checkpoint_ports(bound).update_execution_plan.plan_id
        ),
        "start_optimizer_step": summary.start_optimizer_step,
        "committed_steps": summary.committed_steps,
        "update_count": summary.update_count,
        "checkpoint_relative_path": checkpoint_relative_path,
        "bound_reward_resource_ids": to_plain_dict(rewards),
        "policy_tensor_runtime_spec": (
            bound.evidence.policy_tensor_runtime_spec.to_payload()
        ),
        "resolved_recipe_sha256": resolved_recipe_sha256,
        "metrics_sha256": metrics_sha256,
    }


def _relative_path(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{name} must be a canonical safe relative path")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _materialized_recipe_id(name: str, value: object) -> str:
    prefix = "materialized-recipe.v2:"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{name} must be a materialized-recipe.v2 identity")
    digest = value.removeprefix(prefix)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a materialized-recipe.v2 identity")
    return value


def _finite(value: float) -> bool:
    import math

    return math.isfinite(value)
