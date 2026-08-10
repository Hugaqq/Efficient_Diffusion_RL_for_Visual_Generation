"""Canonical runtime DTOs, ports, receipts, and the public run result."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from visual_rl.core.types import FrozenMapping

from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.algorithms.modules.interface import AlgorithmModule, BoundAlgorithm
from visual_rl.algorithms.optimization.execution import UpdateExecutionPlan
from visual_rl.algorithms.rollout.request import IterationRolloutRequestFactory
from visual_rl.algorithms.trainer.interface import IterationResult
from visual_rl.algorithms.trainer.stages import OptimizeStage, RolloutStage
from visual_rl.artifacts.checkpoint.reference import ReferencePolicyStateEvidence
from visual_rl.composition.config.source import SourceRecipe
from visual_rl.composition.config.specs import LaunchSpec, TrainingSpec
from visual_rl.composition.preflight import (
    EnvironmentPreflightResult,
    RuntimeBindResult,
    RuntimeFacts,
    RuntimeGraphBindResult,
    StaticPreflightResult,
)
from visual_rl.composition.registry import ResolvedComponentDeclaration
from visual_rl.core.contracts import (
    CANONICAL_FLOATING_DTYPE_NAMES,
    ComputePrecision,
    DeclaredContract,
    RuntimeBoundContract,
)
from visual_rl.core.contracts.runtime import (
    ExecutionTransformPlan,
    PolicyRuntimePort,
)
from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import to_plain_dict
from visual_rl.data.prelude import DataPlaneCheckpointPort
from visual_rl.models.lifecycle.components import ComponentManager
from visual_rl.models.lifecycle.prepared import PreparedComponentHandle
from visual_rl.models.numerics.policy import (
    ModelExecutionNumericsEvidence,
)
from visual_rl.runtime.algorithm_binding import (
    BindOncePrelude,
    BindOnceStage,
    DynamicsRuntimeBindEvidence,
    PerRolloutDynamicsFactory,
)
from visual_rl.runtime.component_graph import RuntimeComponentGraph
from visual_rl.runtime.resources import DefaultRuntimeResourceContainer

__all__ = (
    "BoundRestoreRequest",
    "BoundRestoreResult",
    "CheckpointRequest",
    "CheckpointSink",
    "CompiledProductionRun",
    "ComponentBindRequest",
    "ComponentRuntimeBinder",
    "ComponentRuntimeEvidence",
    "DynamicsRuntimeBindEvidence",
    "PerRolloutDynamicsFactory",
    "PolicyTensorRuntimeSpec",
    "PreparedRestoreRequest",
    "PreparedRestoreResult",
    "ProductionBoundRun",
    "ProductionGraph",
    "ProductionPreflight",
    "ProductionPreparedRun",
    "ProductionRuntime",
    "ProductionRuntimeContextProvider",
    "ProductionRuntimeError",
    "RestoreService",
    "RuntimeContextRequest",
    "RuntimeCreateRequest",
    "RuntimeSession",
    "RuntimeSessionFactory",
    "SafePointCheckpointReceipt",
    "StageAssembler",
    "StageAssemblyRequest",
    "StageCheckpointPortError",
    "StageCheckpointPorts",
    "TrainerStageAssembly",
    "TransformExecution",
    "TransformExecutor",
    "TransformRequest",
    "RunResult",
)


@dataclass(frozen=True)
class RunResult:
    """Authoritative result returned by the production runtime controller."""

    run_id: str
    output_dir: Path
    committed_steps: int
    authoritative_checkpoint: Path
    resolved_config_path: Path
    manifest_path: Path
    metrics_path: Path
    marker_path: Path
    last_metrics: Mapping[str, float | int]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if type(self.committed_steps) is not int:
            raise TypeError("committed_steps must be an integer, not bool")
        if self.committed_steps <= 0:
            raise ValueError("committed_steps must be positive")

        output_dir = _absolute_path(self.output_dir, field_name="output_dir")
        if not output_dir.is_dir():
            raise ValueError("output_dir must be an existing directory")
        object.__setattr__(self, "output_dir", output_dir)
        for name in (
            "authoritative_checkpoint",
            "resolved_config_path",
            "manifest_path",
            "metrics_path",
            "marker_path",
        ):
            path = _absolute_path(getattr(self, name), field_name=name)
            if path != output_dir and output_dir not in path.parents:
                raise ValueError(f"{name} must be located inside output_dir")
            if not path.exists():
                raise ValueError(f"{name} must exist")
            object.__setattr__(self, name, path)

        metrics = FrozenMapping(self.last_metrics)
        required_integer_metrics = (
            "step",
            "sample_count",
            "active_transition_count",
        )
        missing = tuple(key for key in required_integer_metrics if key not in metrics)
        if missing:
            raise ValueError(f"last_metrics is missing required keys: {missing}")
        for key, value in metrics.items():
            if key == "schema_version":
                raise ValueError("last_metrics must not contain schema_version")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"last_metrics[{key!r}] must be a Python int or float")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"last_metrics[{key!r}] must be finite")
            if key not in required_integer_metrics and type(value) is not float:
                raise TypeError(f"last_metrics[{key!r}] must be a finite Python float")
        for integer_key in required_integer_metrics:
            if type(metrics[integer_key]) is not int:
                raise TypeError(f"last_metrics[{integer_key!r}] must be an integer")
        if metrics["step"] != self.committed_steps - 1:
            raise ValueError("last_metrics.step must equal committed_steps - 1")
        if metrics["sample_count"] <= 0:
            raise ValueError("last_metrics.sample_count must be positive")
        if metrics["active_transition_count"] <= 0:
            raise ValueError("last_metrics.active_transition_count must be positive")
        object.__setattr__(self, "last_metrics", metrics)


def _absolute_path(value: Path, *, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
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
    _digest(name, value.removeprefix(prefix))
    return value


class ProductionRuntimeError(RuntimeError):
    """A required production lifecycle service or invariant is missing."""


@dataclass(frozen=True, slots=True)
class CompiledProductionRun:
    """Static recipe semantics plus a separate location-only launch spec."""

    source: SourceRecipe
    static: StaticPreflightResult
    training: TrainingSpec
    launch: LaunchSpec

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRecipe):
            raise TypeError("source must be SourceRecipe")
        if not isinstance(self.static, StaticPreflightResult):
            raise TypeError("static must be StaticPreflightResult")
        if not isinstance(self.training, TrainingSpec):
            raise TypeError("training must be TrainingSpec")
        if not isinstance(self.launch, LaunchSpec):
            raise TypeError("launch must be LaunchSpec")
        if self.static.resolved.training != self.training:
            raise ValueError("compiled TrainingSpec differs from recipe semantics")


@dataclass(frozen=True, slots=True)
class ProductionPreflight:
    compiled: CompiledProductionRun
    environment: EnvironmentPreflightResult

    def __post_init__(self) -> None:
        if not isinstance(self.compiled, CompiledProductionRun):
            raise TypeError("compiled must be CompiledProductionRun")
        if not isinstance(self.environment, EnvironmentPreflightResult):
            raise TypeError("environment must be EnvironmentPreflightResult")
        if self.environment.static is not self.compiled.static:
            raise ValueError("environment preflight replaced the static result")


@dataclass(frozen=True, slots=True)
class RuntimeCreateRequest:
    environment: EnvironmentPreflightResult
    training: TrainingSpec
    launch: LaunchSpec


@dataclass(slots=True)
class RuntimeSession:
    """Injected runtime objects and launch facts, before G2/G3 binding."""

    accelerator: object
    runtime_facts: RuntimeFacts
    peer_recipe_ids: tuple[str, ...]
    model_loader: object | None = None
    resource_container: DefaultRuntimeResourceContainer | None = None
    closer: object | None = None
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.accelerator is None:
            raise TypeError("accelerator must not be None")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if type(self.peer_recipe_ids) is not tuple or any(
            not isinstance(item, str) or not item for item in self.peer_recipe_ids
        ):
            raise ValueError("peer_recipe_ids must contain non-empty strings")
        if len(self.peer_recipe_ids) != self.runtime_facts.world_size:
            raise ValueError("peer_recipe_ids must contain one id per rank")
        if self.model_loader is not None and not callable(self.model_loader):
            raise TypeError("model_loader must be callable or None")
        if self.resource_container is not None and not isinstance(
            self.resource_container,
            DefaultRuntimeResourceContainer,
        ):
            raise TypeError(
                "resource_container must be DefaultRuntimeResourceContainer or None"
            )
        if self.closer is not None and not callable(self.closer):
            raise TypeError("closer must be callable or None")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        if self.resource_container is not None:
            try:
                self.resource_container.close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if self.closer is not None:
            try:
                self.closer()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            primary = errors[0]
            for cleanup_error in errors[1:]:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "additional RuntimeSession close failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise primary


class RuntimeSessionFactory(Protocol):
    def create(self, request: RuntimeCreateRequest) -> RuntimeSession: ...


@dataclass(slots=True)
class ProductionRuntime:
    preflight: ProductionPreflight
    session: RuntimeSession
    launch_binding: RuntimeBindResult

    def __post_init__(self) -> None:
        if not isinstance(self.preflight, ProductionPreflight):
            raise TypeError("preflight must be ProductionPreflight")
        if not isinstance(self.session, RuntimeSession):
            raise TypeError("session must be RuntimeSession")
        if not isinstance(self.launch_binding, RuntimeBindResult):
            raise TypeError("launch_binding must be RuntimeBindResult")
        if self.launch_binding.runtime_facts is not self.session.runtime_facts:
            raise ValueError("runtime bind replaced RuntimeFacts")

    def close(self) -> None:
        self.session.close()


@dataclass(frozen=True, slots=True)
class RuntimeContextRequest:
    slot: str
    declaration: ResolvedComponentDeclaration
    loaded_components: Mapping[str, object]
    preflight: ProductionPreflight
    runtime: ProductionRuntime

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or not self.slot:
            raise ValueError("runtime component slot must be non-empty")
        if not isinstance(self.declaration, ResolvedComponentDeclaration):
            raise TypeError("declaration must be ResolvedComponentDeclaration")
        if not isinstance(self.loaded_components, Mapping):
            raise TypeError("loaded_components must be a mapping")
        if not isinstance(self.preflight, ProductionPreflight):
            raise TypeError("preflight must be ProductionPreflight")
        if not isinstance(self.runtime, ProductionRuntime):
            raise TypeError("runtime must be ProductionRuntime")

    @property
    def kind(self) -> str:
        return self.declaration.kind


class ProductionRuntimeContextProvider(Protocol):
    def context_for(self, request: RuntimeContextRequest) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class _TrainerPorts:
    prelude: BindOncePrelude
    rollout: BindOnceStage
    reward: BindOnceStage
    advantage: BindOnceStage
    credit: BindOnceStage
    optimize: BindOnceStage

    @classmethod
    def create(cls) -> _TrainerPorts:
        return cls(
            prelude=BindOncePrelude(),
            rollout=BindOnceStage("rollout"),
            reward=BindOnceStage("reward"),
            advantage=BindOnceStage("advantage"),
            credit=BindOnceStage("credit"),
            optimize=BindOnceStage("optimize"),
        )


@dataclass(slots=True)
class ProductionGraph:
    components: RuntimeComponentGraph
    trainer_ports: _TrainerPorts

    def __post_init__(self) -> None:
        if not isinstance(self.components, RuntimeComponentGraph):
            raise TypeError("components must be RuntimeComponentGraph")
        if not isinstance(self.trainer_ports, _TrainerPorts):
            raise TypeError("trainer_ports must be _TrainerPorts")
        binding = self.components.binding("algorithm")
        if binding.kind != "algorithm":
            raise ValueError("algorithm slot must contain an algorithm declaration")
        if not isinstance(binding.instance, AlgorithmModule):
            raise TypeError("algorithm_module must be AlgorithmModule")

    @property
    def algorithm_module(self) -> AlgorithmModule:
        module = self.components.component("algorithm")
        if not isinstance(module, AlgorithmModule):  # pragma: no cover - guarded
            raise TypeError("algorithm slot does not contain AlgorithmModule")
        return module

    def close(self) -> None:
        # RuntimeComponentGraph is the unique owner of every loaded slot,
        # including the public AlgorithmModule.
        self.components.close()


@dataclass(frozen=True, slots=True)
class StageAssemblyRequest:
    preflight: ProductionPreflight
    runtime: ProductionRuntime
    graph: ProductionGraph
    prepared: ProductionPreparedRun
    transforms: TransformExecution
    evidence: ComponentRuntimeEvidence
    graph_binding: RuntimeGraphBindResult
    dynamics_factory: PerRolloutDynamicsFactory
    policy_runtime: PolicyRuntimePort | None = None


class StageCheckpointPortError(ProductionRuntimeError):
    """Checkpoint projection does not belong to the assembled stage graph."""


@dataclass(frozen=True, slots=True)
class StageCheckpointPorts:
    """Immutable checkpoint-facing view of one concrete stage assembly.

    ``rollout_request_factory`` is retained only as typed provenance for the
    selection-policy state.  It prevents a checkpoint sink from receiving an
    equal-looking state copied from another assembly.
    """

    data_plane: DataPlaneCheckpointPort
    dynamics_selection_policy: DynamicsSelectionPolicyState
    update_execution_plan: UpdateExecutionPlan
    rollout_request_factory: IterationRolloutRequestFactory = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.data_plane, DataPlaneCheckpointPort):
            raise TypeError(
                "data_plane must implement the DataPlaneCheckpointPort protocol"
            )
        if not isinstance(
            self.dynamics_selection_policy,
            DynamicsSelectionPolicyState,
        ):
            raise TypeError(
                "dynamics_selection_policy must be DynamicsSelectionPolicyState"
            )
        if not isinstance(self.update_execution_plan, UpdateExecutionPlan):
            raise TypeError("update_execution_plan must be UpdateExecutionPlan")
        if not isinstance(
            self.rollout_request_factory,
            IterationRolloutRequestFactory,
        ):
            raise TypeError(
                "rollout_request_factory must be IterationRolloutRequestFactory"
            )
        if (
            self.dynamics_selection_policy
            is not self.rollout_request_factory.dynamics_selection_policy
        ):
            raise StageCheckpointPortError(
                "checkpoint selection policy does not originate from its "
                "rollout request factory"
            )

    def validate_assembly(
        self,
        *,
        prelude: object,
        rollout: object,
        optimize: object,
    ) -> None:
        """Prove all projected ports are exact owners in one assembly."""

        if prelude is not self.data_plane:
            raise StageCheckpointPortError(
                "checkpoint data plane does not originate from this assembly"
            )
        if not isinstance(rollout, RolloutStage):
            raise StageCheckpointPortError(
                "checkpoint projection requires the concrete RolloutStage"
            )
        if rollout.request_factory is not self.rollout_request_factory:
            raise StageCheckpointPortError(
                "checkpoint selection policy does not originate from this "
                "assembly request factory"
            )
        if not isinstance(optimize, OptimizeStage):
            raise StageCheckpointPortError(
                "checkpoint projection requires the concrete OptimizeStage"
            )
        if optimize.execution_plan is not self.update_execution_plan:
            raise StageCheckpointPortError(
                "checkpoint update plan does not originate from this assembly"
            )


@dataclass(slots=True)
class TrainerStageAssembly:
    """Explicit data/stage handoff bound only after optimizer preparation.

    ``reward_runtime_context_factory`` is mandatory and must be the exact
    factory retained by the reward stage.  This prevents the composition root
    from guessing a ``StepContext`` or reconstructing reward routing state.
    """

    prelude: object
    rollout: object
    reward: object
    advantage: object
    credit: object
    optimize: object
    reward_runtime_context_factory: object
    close_resources: tuple[object, ...] = ()
    checkpoint_ports: StageCheckpointPorts | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.prelude, "build", None)):
            raise TypeError("stage assembly prelude must implement build(step)")
        for name in ("rollout", "reward", "advantage", "credit", "optimize"):
            if not callable(getattr(self, name)):
                raise TypeError(f"stage assembly {name} must be callable")
        if not callable(self.reward_runtime_context_factory):
            raise TypeError("reward_runtime_context_factory must be callable")
        retained = getattr(self.reward, "runtime_context_factory", None)
        if retained is not self.reward_runtime_context_factory:
            raise ProductionRuntimeError(
                "reward stage must retain the explicit RewardRuntimeContext factory"
            )
        if self.checkpoint_ports is not None:
            if not isinstance(self.checkpoint_ports, StageCheckpointPorts):
                raise TypeError("checkpoint_ports must be StageCheckpointPorts or None")
            self.checkpoint_ports.validate_assembly(
                prelude=self.prelude,
                rollout=self.rollout,
                optimize=self.optimize,
            )
        if type(self.close_resources) is not tuple:
            raise TypeError("close_resources must be a tuple")
        identities = tuple(id(resource) for resource in self.close_resources)
        if len(identities) != len(set(identities)):
            raise ValueError("close_resources must be unique by object identity")
        for resource in self.close_resources:
            if not callable(getattr(resource, "close", None)):
                raise TypeError("each close resource must implement close()")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for resource in reversed(self.close_resources):
            try:
                resource.close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "additional stage resource close failure: "
                        f"{type(error).__name__}: {error}"
                    )
            raise primary


class StageAssembler(Protocol):
    def assemble(self, request: StageAssemblyRequest) -> TrainerStageAssembly: ...


@dataclass(slots=True)
class ProductionPreparedRun:
    manager: ComponentManager
    handle: PreparedComponentHandle
    optimizer: object
    lr_scheduler: object
    training: TrainingSpec
    start_optimizer_step: int = 0
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manager, ComponentManager):
            raise TypeError("manager must be ComponentManager")
        if not isinstance(self.handle, PreparedComponentHandle):
            raise TypeError("handle must be PreparedComponentHandle")
        if self.handle.optimizer is not self.optimizer:
            raise ValueError("prepared optimizer is not the unique optimizer")
        if self.handle.scheduler is not self.lr_scheduler:
            raise ValueError("prepared scheduler is not the unique LR scheduler")
        if not isinstance(self.training, TrainingSpec):
            raise TypeError("training must be TrainingSpec")
        self.manager.parameter_dtype_owner.validate_applied()
        if (
            self.manager.model_execution_numerics.source_projection_id
            != self.manager.parameter_state.state_projection.projection_id
        ):
            raise ValueError(
                "prepared model execution numerics use a stale state projection"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.manager.close()


@dataclass(frozen=True, slots=True)
class PreparedRestoreRequest:
    checkpoint_path: Path
    preflight: ProductionPreflight
    runtime: ProductionRuntime
    graph: ProductionGraph
    prepared: ProductionPreparedRun


_REQUIRED_PREPARED_RESTORE_STATES = frozenset(
    {
        "lr_scheduler",
        "model",
        "optimizer",
    }
)


@dataclass(slots=True)
class PreparedRestoreResult:
    """First-gate receipt after prepared state has been restored.

    ``continuation`` is service-owned, already validated state carried to the
    full bound gate.  It is never interpreted by the controller or components.
    """

    checkpoint_path: Path
    next_optimizer_step: int
    restored_state_ids: FrozenMapping
    continuation: object = field(repr=False, compare=False)
    closer: object | None = field(default=None, repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_path, Path) or not (
            self.checkpoint_path.is_absolute()
        ):
            raise ValueError("prepared restore checkpoint_path must be absolute")
        if not self.checkpoint_path.exists():
            raise ValueError("prepared restore checkpoint_path must exist")
        if type(self.next_optimizer_step) is not int or self.next_optimizer_step < 1:
            raise ValueError("next_optimizer_step must be a positive integer")
        if not isinstance(self.restored_state_ids, FrozenMapping):
            raise TypeError("restored_state_ids must be a FrozenMapping")
        missing = sorted(
            _REQUIRED_PREPARED_RESTORE_STATES - set(self.restored_state_ids)
        )
        if missing:
            raise ValueError(f"restore evidence is missing states: {missing}")
        for name, state_id in self.restored_state_ids.items():
            if not isinstance(name, str) or not name:
                raise ValueError("restore state names must be non-empty")
            _digest(f"restore state id for {name}", state_id)
        if self.continuation is None:
            raise ValueError("prepared restore continuation must not be None")
        if self.closer is not None and not callable(self.closer):
            raise TypeError("prepared restore closer must be callable or None")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.closer is not None:
            self.closer()


@dataclass(frozen=True, slots=True)
class BoundRestoreRequest:
    checkpoint_path: Path
    preflight: ProductionPreflight
    runtime: ProductionRuntime
    graph: ProductionGraph
    prepared: ProductionPreparedRun
    bound: ProductionBoundRun
    prepared_restore: PreparedRestoreResult


_REQUIRED_BOUND_RESTORE_STATES = frozenset(
    {
        "data_plane",
        "dynamics_selection_policy",
        "progress",
        "rng",
    }
)


@dataclass(frozen=True, slots=True)
class BoundRestoreResult:
    """Full-gate receipt after logical state and finally RNG were restored."""

    checkpoint_path: Path
    next_optimizer_step: int
    restored_state_ids: FrozenMapping
    completed_result: RunResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_path, Path) or not (
            self.checkpoint_path.is_absolute()
        ):
            raise ValueError("bound restore checkpoint_path must be absolute")
        if not self.checkpoint_path.exists():
            raise ValueError("bound restore checkpoint_path must exist")
        if type(self.next_optimizer_step) is not int or self.next_optimizer_step < 1:
            raise ValueError("next_optimizer_step must be a positive integer")
        if not isinstance(self.restored_state_ids, FrozenMapping):
            raise TypeError("restored_state_ids must be a FrozenMapping")
        missing = sorted(_REQUIRED_BOUND_RESTORE_STATES - set(self.restored_state_ids))
        if missing:
            raise ValueError(f"bound restore evidence is missing states: {missing}")
        for name, state_id in self.restored_state_ids.items():
            if not isinstance(name, str) or not name:
                raise ValueError("restore state names must be non-empty")
            _digest(f"restore state id for {name}", state_id)
        if self.completed_result is not None:
            if not isinstance(self.completed_result, RunResult):
                raise TypeError("completed_result must be RunResult or None")
            if self.completed_result.committed_steps != self.next_optimizer_step:
                raise ValueError(
                    "completed RunResult disagrees with next_optimizer_step"
                )


class RestoreService(Protocol):
    def restore_prepared(
        self,
        request: PreparedRestoreRequest,
    ) -> PreparedRestoreResult: ...

    def restore_bound(self, request: BoundRestoreRequest) -> BoundRestoreResult: ...


@dataclass(frozen=True, slots=True)
class TransformRequest:
    plan: ExecutionTransformPlan
    preflight: ProductionPreflight
    runtime: ProductionRuntime
    graph: ProductionGraph
    prepared: ProductionPreparedRun


@dataclass(frozen=True, slots=True)
class TransformExecution:
    plan_id: str
    applied_transform_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest("transform plan_id", self.plan_id)
        if type(self.applied_transform_ids) is not tuple or any(
            not isinstance(item, str) or not item for item in self.applied_transform_ids
        ):
            raise ValueError("applied_transform_ids must contain strings")
        if len(self.applied_transform_ids) != len(set(self.applied_transform_ids)):
            raise ValueError("applied_transform_ids must be unique")


class TransformExecutor(Protocol):
    def execute(self, request: TransformRequest) -> TransformExecution: ...


@dataclass(frozen=True, slots=True)
class PolicyTensorRuntimeSpec:
    """Resolved device and latent storage dtype at the model/Dynamics seam."""

    device: str
    latent_storage_dtype: str
    model_compute_precision: str
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty canonical torch device")
        import torch

        try:
            resolved_device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("device must be a canonical torch device") from exc
        if str(resolved_device) != self.device:
            raise ValueError("device must use its canonical torch spelling")
        if not isinstance(self.latent_storage_dtype, str):
            raise TypeError("latent_storage_dtype must be a canonical dtype name")
        if self.latent_storage_dtype not in CANONICAL_FLOATING_DTYPE_NAMES:
            raise ValueError(
                "latent_storage_dtype must be a canonical floating dtype name"
            )
        if not isinstance(self.model_compute_precision, str):
            raise TypeError("model_compute_precision must be a canonical precision")
        try:
            ComputePrecision(self.model_compute_precision)
        except ValueError as exc:
            raise ValueError(
                "model_compute_precision must be one of fp32, fp16, or bf16"
            ) from exc
        object.__setattr__(
            self,
            "spec_id",
            hashlib.sha256(
                canonical_json_text(self.to_payload()).encode("utf-8")
            ).hexdigest(),
        )

    @property
    def torch_device(self) -> Any:
        import torch

        return torch.device(self.device)

    @property
    def latent_storage_torch_dtype(self) -> Any:
        """Resolve only the already validated canonical latent dtype name."""

        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return mapping[self.latent_storage_dtype]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "policy_tensor_runtime_spec",
            "device": self.device,
            "latent_storage_dtype": self.latent_storage_dtype,
            "model_compute_precision": self.model_compute_precision,
        }


@dataclass(frozen=True, slots=True)
class ComponentRuntimeEvidence:
    runtime_bound_contracts: tuple[tuple[str, RuntimeBoundContract], ...]
    verified_fields: FrozenMapping
    model_runtime_contract: RuntimeBoundContract
    policy_tensor_runtime_spec: PolicyTensorRuntimeSpec
    model_execution_numerics: ModelExecutionNumericsEvidence
    reference_policy_state_evidence: ReferencePolicyStateEvidence
    preprocess_identity: str
    preprocess_requirement_set_id: str
    bound_reward_resource_ids: FrozenMapping = field(default_factory=FrozenMapping)
    peer_bound_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        contracts = self.runtime_bound_contracts
        if type(contracts) is not tuple or not contracts:
            raise ValueError("runtime_bound_contracts must be a non-empty tuple")
        slots: list[str] = []
        model_contract: RuntimeBoundContract | None = None
        for item in contracts:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "runtime_bound_contracts must contain (slot, contract) pairs"
                )
            slot, contract = item
            if not isinstance(slot, str) or not slot:
                raise ValueError("component runtime evidence slots must be non-empty")
            if not isinstance(contract, RuntimeBoundContract):
                raise TypeError("component runtime evidence must contain contracts")
            if not contract.is_declaration_bound:
                raise ValueError(
                    "production G3 evidence must reference declaration-bound G1"
                )
            if contract.artifact.slot != slot:
                raise ValueError("runtime bound contract slot differs from its G1")
            slots.append(slot)
            if slot == "model":
                model_contract = contract
        if tuple(slots) != tuple(sorted(set(slots))):
            raise ValueError("runtime_bound_contracts must be sorted and unique")
        if not isinstance(self.verified_fields, FrozenMapping) or not (
            self.verified_fields
        ):
            raise ValueError("verified_fields must be non-empty")
        if not isinstance(self.model_runtime_contract, RuntimeBoundContract):
            raise TypeError("model_runtime_contract must be RuntimeBoundContract")
        if model_contract is not self.model_runtime_contract:
            raise ValueError(
                "model_runtime_contract must be the exact model G3 graph contract"
            )
        if not isinstance(
            self.policy_tensor_runtime_spec,
            PolicyTensorRuntimeSpec,
        ):
            raise TypeError(
                "policy_tensor_runtime_spec must be PolicyTensorRuntimeSpec"
            )
        expected_tensor_runtime = {
            "spec_id": self.policy_tensor_runtime_spec.spec_id,
            "spec": self.policy_tensor_runtime_spec.to_payload(),
        }
        tensor_runtime_evidence = self.verified_fields.get("policy_tensor_runtime")
        if not isinstance(tensor_runtime_evidence, Mapping) or (
            to_plain_dict(tensor_runtime_evidence) != expected_tensor_runtime
        ):
            raise ValueError(
                "verified_fields policy tensor runtime evidence must exactly "
                "match policy_tensor_runtime_spec"
            )
        if not isinstance(
            self.model_execution_numerics,
            ModelExecutionNumericsEvidence,
        ):
            raise TypeError(
                "model_execution_numerics must be ModelExecutionNumericsEvidence"
            )
        execution_numerics_evidence = self.verified_fields.get(
            "model_execution_numerics"
        )
        if not isinstance(execution_numerics_evidence, Mapping) or (
            to_plain_dict(execution_numerics_evidence)
            != self.model_execution_numerics.to_payload()
        ):
            raise ValueError(
                "verified_fields model execution numerics must exactly match "
                "model_execution_numerics"
            )
        if not isinstance(
            self.reference_policy_state_evidence,
            ReferencePolicyStateEvidence,
        ):
            raise TypeError(
                "reference_policy_state_evidence must be ReferencePolicyStateEvidence"
            )
        self.reference_policy_state_evidence.assert_integrity()
        reference_policy_state = self.verified_fields.get("reference_policy_state")
        if not isinstance(reference_policy_state, Mapping) or (
            to_plain_dict(reference_policy_state)
            != self.reference_policy_state_evidence.to_payload()
        ):
            raise ValueError(
                "verified_fields reference policy state must exactly match "
                "reference_policy_state_evidence"
            )
        if self.reference_policy_state_evidence.model_execution_numerics_id != (
            self.model_execution_numerics.execution_numerics_id
        ):
            raise ValueError(
                "reference policy state uses different model execution numerics"
            )
        if self.reference_policy_state_evidence.source_projection_id != (
            self.model_execution_numerics.source_projection_id
        ):
            raise ValueError(
                "reference policy state uses a different model state projection"
            )
        declared = self.model_runtime_contract.artifact.declared
        if (
            not isinstance(declared, DeclaredContract)
            or declared.component_kind != "model"
            or declared.model is None
        ):
            raise TypeError(
                "model_runtime_contract must retain a declared ModelContract"
            )
        if (
            self.reference_policy_state_evidence.model_provides_reference_policy
            is not declared.model.provides_reference_policy
        ):
            raise ValueError(
                "reference policy state differs from the declared model capability"
            )
        _digest("preprocess_identity", self.preprocess_identity)
        _digest(
            "preprocess_requirement_set_id",
            self.preprocess_requirement_set_id,
        )
        preprocess_evidence = self.verified_fields.get("preprocess")
        if not isinstance(preprocess_evidence, Mapping) or (
            preprocess_evidence.get("identity") != self.preprocess_identity
            or preprocess_evidence.get("requirement_set_id")
            != self.preprocess_requirement_set_id
        ):
            raise ValueError(
                "verified_fields preprocess evidence must exactly match the "
                "preprocess and requirement-set identities"
            )
        requirement_payload = preprocess_evidence.get("requirement_set")
        if not isinstance(requirement_payload, Mapping):
            raise TypeError(
                "verified_fields preprocess evidence must contain the canonical "
                "requirement set"
            )
        observed_requirement_id = hashlib.sha256(
            canonical_json_text(requirement_payload).encode("utf-8")
        ).hexdigest()
        if observed_requirement_id != self.preprocess_requirement_set_id:
            raise ValueError(
                "verified preprocess requirement payload has the wrong identity"
            )
        if not isinstance(self.bound_reward_resource_ids, FrozenMapping):
            raise TypeError("bound_reward_resource_ids must be a FrozenMapping")
        for resource_spec_id, bound_id in self.bound_reward_resource_ids.items():
            if not isinstance(resource_spec_id, str) or not resource_spec_id:
                raise ValueError("reward resource spec ids must be non-empty strings")
            _digest(
                f"bound reward resource id for {resource_spec_id}",
                bound_id,
            )
        if type(self.peer_bound_contract_ids) is not tuple:
            raise TypeError("peer_bound_contract_ids must be a tuple")
        for value in self.peer_bound_contract_ids:
            _digest("peer bound contract id", value)


@dataclass(frozen=True, slots=True)
class ComponentBindRequest:
    preflight: ProductionPreflight
    runtime: ProductionRuntime
    graph: ProductionGraph
    prepared: ProductionPreparedRun
    transforms: TransformExecution


class ComponentRuntimeBinder(Protocol):
    def bind(self, request: ComponentBindRequest) -> ComponentRuntimeEvidence: ...


@dataclass(slots=True)
class ProductionBoundRun:
    preflight: ProductionPreflight
    runtime: ProductionRuntime
    graph: ProductionGraph
    prepared: ProductionPreparedRun
    transforms: TransformExecution
    evidence: ComponentRuntimeEvidence
    graph_binding: RuntimeGraphBindResult
    assembly: TrainerStageAssembly
    algorithm: BoundAlgorithm
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary: BaseException | None = None
        try:
            self.algorithm.close()
        except BaseException as exc:  # noqa: BLE001
            primary = exc
        try:
            self.assembly.close()
        except BaseException as exc:
            if primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note(
                    f"stage assembly close also failed: {type(exc).__name__}: {exc}"
                )
        if primary is not None:
            raise primary


@dataclass(frozen=True, slots=True)
class PendingRunSummary:
    """Internal pre-checkpoint execution summary; never leaves the controller."""

    recipe_id: str
    launch_id: str
    bound_contract_id: str
    start_optimizer_step: int
    committed_steps: int
    update_count: int
    last_iteration: IterationResult[object]

    def __post_init__(self) -> None:
        _materialized_recipe_id("recipe_id", self.recipe_id)
        for name in ("launch_id", "bound_contract_id"):
            _digest(name, getattr(self, name))
        if type(self.start_optimizer_step) is not int or self.start_optimizer_step < 0:
            raise ValueError("start_optimizer_step must be non-negative")
        if type(self.committed_steps) is not int or self.committed_steps < 1:
            raise ValueError("committed_steps must be positive")
        if type(self.update_count) is not int or self.update_count < 1:
            raise ValueError("update_count must be positive")
        if not isinstance(self.last_iteration, IterationResult):
            raise TypeError("last_iteration must be IterationResult")
        if self.committed_steps != self.last_iteration.optimizer_step + 1:
            raise ValueError("committed_steps must follow the last optimizer step")
        if self.update_count != self.committed_steps - self.start_optimizer_step:
            raise ValueError("update_count disagrees with committed step range")


@dataclass(frozen=True, slots=True)
class CheckpointRequest:
    bound: ProductionBoundRun
    summary: PendingRunSummary
    cadence: int


@dataclass(frozen=True, slots=True)
class SafePointCheckpointReceipt:
    """Durable receipt returned for a non-terminal optimizer safe point."""

    checkpoint_path: Path
    committed_steps: int
    checkpoint_contract_id: str
    progress_id: str
    state_tree_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_path, Path) or not (
            self.checkpoint_path.is_absolute()
        ):
            raise ValueError("checkpoint_path must be an absolute Path")
        if type(self.committed_steps) is not int or self.committed_steps < 1:
            raise ValueError("committed_steps must be positive")
        for name in (
            "checkpoint_contract_id",
            "progress_id",
            "state_tree_id",
        ):
            _digest(name, getattr(self, name))


class CheckpointSink(Protocol):
    def checkpoint_safe_point(
        self,
        request: CheckpointRequest,
    ) -> SafePointCheckpointReceipt: ...

    def checkpoint(self, request: CheckpointRequest) -> RunResult: ...
