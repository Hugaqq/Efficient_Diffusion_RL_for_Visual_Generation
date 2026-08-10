"""Strict one-time bindings from the runtime graph to trainer stage ports."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from visual_rl.algorithms.conditioning.interface import LatentConditioner
from visual_rl.algorithms.dynamics.interface import Dynamics
from visual_rl.algorithms.dynamics.interface import DynamicsComponent
from visual_rl.algorithms.dynamics.replay import (
    DynamicsInstanceFactory,
    DynamicsReplayRequest,
    DynamicsReplayStateFactory,
    FlowMatchScheduleConditioning,
    SchedulerDynamicsBinder,
)
from visual_rl.algorithms.modules.interface import (
    AlgorithmExecutionPort,
    AlgorithmMaterializationPort,
    AlgorithmMaterializationRequest,
)
from visual_rl.algorithms.optimization.interface import CreditPlanningPort
from visual_rl.algorithms.optimization.advantage import GroupZScoreAdvantageProcessor
from visual_rl.algorithms.optimization.credit import LocalCoefficientMeanReducer
from visual_rl.algorithms.optimization.execution import UpdateExecutionPlan
from visual_rl.algorithms.optimization.kernel import PolicyUpdateKernel
from visual_rl.algorithms.rollout.interface import RolloutComponent
from visual_rl.algorithms.rollout.request import IterationRolloutRequestFactory
from visual_rl.algorithms.trainer.execution_plan import AlgorithmExecutionPlan
from visual_rl.algorithms.trainer.interface import (
    IterationIdentity,
    IterationResult,
    PrepareRunContext,
    StageValue,
    TrainerComponent,
    TrainerState,
)
from visual_rl.algorithms.trainer.stages import (
    AdvantageStage,
    CreditStage,
    OptimizeStage,
    RewardPipelineStage,
    RolloutStage,
    RolloutStagePayload,
)
from visual_rl.algorithms.rewards import (
    GroupwiseReward,
    PointwiseReward,
    RewardResourceState,
    RewardRuntimeContext,
    RewardStage,
)
from visual_rl.artifacts.checkpoint.reference import (
    ReferencePolicyStateError,
    ReferencePolicyStateEvidence,
    derive_reference_policy_state_evidence,
)
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.core.contracts import (
    AlgorithmMaterializationSpec,
    AlgorithmComponentResolution,
    AlgorithmComponentRole,
    AlgorithmComponentSelection,
    ExecutionPolicyReceipt,
    ModelContract,
    RuntimeBoundContract,
)
from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import FrozenMapping, StepContext
from visual_rl.data import (
    GroupPlacementContract,
    GroupPlacementKind,
    ImplicitPhaseRouter,
    MultiSourceSampler,
    SourceLoadRequest,
    load_stable_source_sequences,
)
from visual_rl.data.prelude import DataPlanePrelude
from visual_rl.data.samples import ExplicitCollator
from visual_rl.models import ModelAdapter
from visual_rl.models.lifecycle.components import ComponentManager
from visual_rl.models.lifecycle.components import ExecutionMode
from visual_rl.models.numerics.execution import ParameterView, StageExecutionPolicy
from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence
from visual_rl.models.scheduler import ModelScheduleContext, SchedulerArtifactBlueprint
from visual_rl.runtime.component_graph import (
    ComponentRuntimeBindingError,
    RuntimeComponentGraph,
)
from visual_rl.runtime.resources import DefaultRuntimeResourceContainer

if TYPE_CHECKING:
    from visual_rl.runtime.types import (
        ProductionRuntime,
        StageAssemblyRequest,
        TrainerStageAssembly,
    )

__all__ = (
    "AlgorithmExecutionBinding",
    "AlgorithmRuntimeComponent",
    "AlgorithmRuntimeComponents",
    "AlgorithmRuntimeBindingError",
    "BindOncePrelude",
    "BindOnceStage",
    "CanonicalAlgorithmMaterializationError",
    "CanonicalAlgorithmMaterializer",
    "DynamicsRuntimeBindEvidence",
    "DefaultStageAssembler",
    "DefaultStageAssemblyError",
    "PerRolloutDynamicsFactory",
    "StageBindingError",
    "TrainerAlgorithmExecution",
    "bind_algorithm_execution_plan",
    "bind_per_rollout_dynamics_factory",
    "materialize_algorithm_runtime_components",
    "validate_algorithm_runtime_contracts",
)


class AlgorithmRuntimeBindingError(ComponentRuntimeBindingError):
    """Algorithm runtime projection drifted from the loaded G1 component graph."""


@dataclass(frozen=True, slots=True)
class AlgorithmExecutionBinding:
    """Resolved algorithm plan plus its model-reference capability evidence."""

    execution_plan: AlgorithmExecutionPlan
    reference_policy_state_evidence: ReferencePolicyStateEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.execution_plan, AlgorithmExecutionPlan):
            raise TypeError("execution_plan must be AlgorithmExecutionPlan")
        if not isinstance(
            self.reference_policy_state_evidence,
            ReferencePolicyStateEvidence,
        ):
            raise TypeError(
                "reference_policy_state_evidence must be ReferencePolicyStateEvidence"
            )


def bind_algorithm_execution_plan(
    materialized: MaterializedRecipe,
    model: ModelContract,
    model_execution_numerics: ModelExecutionNumericsEvidence,
) -> AlgorithmExecutionBinding:
    """Bind algorithm semantics to model capabilities without a concrete adapter."""

    if not isinstance(materialized, MaterializedRecipe):
        raise TypeError("materialized must be MaterializedRecipe")
    if not isinstance(model, ModelContract):
        raise TypeError("model must be ModelContract")
    if not isinstance(model_execution_numerics, ModelExecutionNumericsEvidence):
        raise TypeError(
            "model_execution_numerics must be ModelExecutionNumericsEvidence"
        )
    execution_plan = AlgorithmExecutionPlan.from_spec(
        materialized.resolved.algorithm_spec,
        execution_policy=materialized.resolved.execution_policy.to_receipt(),
    )
    try:
        reference_evidence = derive_reference_policy_state_evidence(
            algorithm=execution_plan,
            model=model,
            model_execution_numerics=model_execution_numerics,
        )
    except ReferencePolicyStateError as exc:
        raise AlgorithmRuntimeBindingError(
            "G3 cannot bind the declared reference-policy capability"
        ) from exc
    return AlgorithmExecutionBinding(
        execution_plan=execution_plan,
        reference_policy_state_evidence=reference_evidence,
    )


@dataclass(frozen=True, slots=True)
class DynamicsRuntimeBindEvidence:
    """Runtime proof of one scheduler artifact/Dynamics equation bind."""

    dynamics_binding_family: str
    scheduler_blueprint_schema: str
    scheduler_blueprint_identity: str
    scheduler_artifact_identity: str
    replay_state_schema_id: str
    replay_state_factory_identity: str
    replay_state_type_path: str
    binding_identity: str = field(init=False)

    def __post_init__(self) -> None:
        payload: dict[str, str] = {}
        for name in (
            "dynamics_binding_family",
            "scheduler_blueprint_schema",
            "scheduler_blueprint_identity",
            "scheduler_artifact_identity",
            "replay_state_schema_id",
            "replay_state_factory_identity",
            "replay_state_type_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a canonical non-empty string")
            payload[name] = value
        object.__setattr__(
            self,
            "binding_identity",
            hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest(),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "dynamics_binding_family": self.dynamics_binding_family,
            "scheduler_blueprint_schema": self.scheduler_blueprint_schema,
            "scheduler_blueprint_identity": self.scheduler_blueprint_identity,
            "scheduler_artifact_identity": self.scheduler_artifact_identity,
            "replay_state_schema_id": self.replay_state_schema_id,
            "replay_state_factory_identity": self.replay_state_factory_identity,
            "replay_state_type_path": self.replay_state_type_path,
            "binding_identity": self.binding_identity,
        }


@dataclass(frozen=True, slots=True)
class PerRolloutDynamicsFactory:
    """Bind algorithm equations to one prepared model scheduler artifact."""

    component: DynamicsInstanceFactory[Any]
    scheduler_blueprint: SchedulerArtifactBlueprint
    replay_state_factory: DynamicsReplayStateFactory[Any]
    dynamics_binding_family: str
    replay_state_schema_id: str
    binding_evidence: DynamicsRuntimeBindEvidence = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.component, DynamicsInstanceFactory):
            raise TypeError("component must be a DynamicsInstanceFactory")
        if not isinstance(self.component, SchedulerDynamicsBinder):
            raise TypeError("component must implement SchedulerDynamicsBinder")
        if not isinstance(self.scheduler_blueprint, SchedulerArtifactBlueprint):
            raise TypeError("scheduler_blueprint must be a SchedulerArtifactBlueprint")
        if not isinstance(self.replay_state_factory, DynamicsReplayStateFactory):
            raise TypeError(
                "replay_state_factory must implement DynamicsReplayStateFactory"
            )
        if self.component.replay_state_type is not (
            self.replay_state_factory.replay_state_type
        ):
            raise AlgorithmRuntimeBindingError(
                "model replay state and dynamics equation factory are incompatible"
            )
        if self.dynamics_binding_family != self.component.dynamics_binding_family:
            raise AlgorithmRuntimeBindingError(
                "runtime Dynamics binding family differs from the resolved component"
            )
        if self.replay_state_schema_id != self.component.replay_state_schema_id:
            raise AlgorithmRuntimeBindingError(
                "runtime replay-state schema differs from the resolved component"
            )
        state_type = self.replay_state_factory.replay_state_type
        object.__setattr__(
            self,
            "binding_evidence",
            DynamicsRuntimeBindEvidence(
                dynamics_binding_family=self.dynamics_binding_family,
                scheduler_blueprint_schema=self.scheduler_blueprint.schema_id,
                scheduler_blueprint_identity=(
                    self.scheduler_blueprint.blueprint_identity
                ),
                scheduler_artifact_identity=(
                    self.scheduler_blueprint.artifact_identity
                ),
                replay_state_schema_id=self.replay_state_schema_id,
                replay_state_factory_identity=(
                    self.replay_state_factory.factory_identity
                ),
                replay_state_type_path=(
                    f"{state_type.__module__}:{state_type.__qualname__}"
                ),
            ),
        )

    def schedule_conditioning(
        self,
        context: ModelScheduleContext,
    ) -> FlowMatchScheduleConditioning | None:
        if not isinstance(context, ModelScheduleContext):
            raise TypeError("context must implement ModelScheduleContext")
        result = self.component.schedule_conditioning(
            self.scheduler_blueprint,
            context,
        )
        if result is not None and not isinstance(
            result,
            FlowMatchScheduleConditioning,
        ):
            raise TypeError(
                "Dynamics scheduler binder returned invalid schedule conditioning"
            )
        return result

    def create(self, request: DynamicsReplayRequest) -> Dynamics:
        """Create fresh replay state and a fresh transition kernel per rollout."""

        if not isinstance(request, DynamicsReplayRequest):
            raise TypeError("request must be a DynamicsReplayRequest")
        binding = self.replay_state_factory.create(request)
        if type(binding.replay_state) is not self.component.replay_state_type:
            raise AlgorithmRuntimeBindingError(
                "bound replay state has the wrong exact type for Dynamics"
            )
        if binding.factory_identity != self.replay_state_factory.factory_identity:
            raise AlgorithmRuntimeBindingError(
                "replay binding differs from the runtime-bound state factory"
            )
        dynamics = self.component.create(binding)
        if not isinstance(dynamics, Dynamics):
            raise TypeError("dynamics component factory must return Dynamics")
        if getattr(dynamics, "replay_binding", None) is not binding:
            raise AlgorithmRuntimeBindingError(
                "per-rollout Dynamics must retain its exact replay binding"
            )
        return dynamics


def bind_per_rollout_dynamics_factory(
    components: RuntimeComponentGraph,
    manager: ComponentManager,
) -> PerRolloutDynamicsFactory:
    """Bind the resolved Dynamics contract to one prepared model artifact."""

    if not isinstance(components, RuntimeComponentGraph):
        raise TypeError("components must be RuntimeComponentGraph")
    if not isinstance(manager, ComponentManager):
        raise TypeError("manager must be ComponentManager")
    component = components.component("dynamics")
    if not isinstance(component, DynamicsInstanceFactory):
        raise TypeError(
            "production dynamics component must be a DynamicsInstanceFactory"
        )
    if not isinstance(component, SchedulerDynamicsBinder):
        raise AlgorithmRuntimeBindingError(
            "resolved Dynamics does not implement SchedulerDynamicsBinder"
        )
    model = components.binding("model").declared_contract.model
    if model is None or not model.declares_scheduler_binding:
        raise AlgorithmRuntimeBindingError(
            "resolved model does not declare the scheduler/Dynamics binding ABI"
        )
    blueprint = manager.adapter.scheduler_artifact_blueprint
    if not isinstance(blueprint, SchedulerArtifactBlueprint):
        raise AlgorithmRuntimeBindingError(
            "prepared model scheduler blueprint violates its typed port"
        )
    if blueprint.schema_id != model.scheduler_blueprint_schema:
        raise AlgorithmRuntimeBindingError(
            "prepared scheduler blueprint schema differs from the model descriptor"
        )
    dynamics_declared = components.binding("dynamics").declared_contract.dynamics
    if dynamics_declared is None or not dynamics_declared.declares_scheduler_binding:
        raise AlgorithmRuntimeBindingError(
            "resolved Dynamics does not declare the scheduler binding ABI"
        )
    if blueprint.schema_id not in (
        dynamics_declared.accepted_scheduler_blueprint_schemas
    ):
        raise AlgorithmRuntimeBindingError(
            "resolved Dynamics does not accept the prepared scheduler schema"
        )
    if component.dynamics_binding_family != model.dynamics_binding_family:
        raise AlgorithmRuntimeBindingError(
            "resolved Dynamics binding family is incompatible with the model"
        )
    if component.dynamics_binding_family not in (
        dynamics_declared.accepted_model_binding_families
    ):
        raise AlgorithmRuntimeBindingError(
            "runtime Dynamics binder family differs from its static descriptor"
        )
    if component.replay_state_schema_id not in model.accepted_replay_state_schema_ids:
        raise AlgorithmRuntimeBindingError(
            "resolved Dynamics replay-state schema is incompatible with the model"
        )
    if component.replay_state_schema_id != (
        dynamics_declared.produced_replay_state_schema_id
    ):
        raise AlgorithmRuntimeBindingError(
            "runtime Dynamics replay-state schema differs from its descriptor"
        )
    replay_factory = component.bind_replay_state_factory(blueprint)
    if not isinstance(replay_factory, DynamicsReplayStateFactory):
        raise AlgorithmRuntimeBindingError(
            "Dynamics scheduler binder returned an invalid replay-state factory"
        )
    return PerRolloutDynamicsFactory(
        component=component,
        scheduler_blueprint=blueprint,
        replay_state_factory=replay_factory,
        dynamics_binding_family=component.dynamics_binding_family,
        replay_state_schema_id=component.replay_state_schema_id,
    )


def materialize_algorithm_runtime_components(
    spec: AlgorithmMaterializationSpec,
    components: RuntimeComponentGraph,
) -> AlgorithmRuntimeComponents:
    """Project compiler-owned selections onto the exact loaded graph."""

    if not isinstance(spec, AlgorithmMaterializationSpec):
        raise TypeError("spec must be AlgorithmMaterializationSpec")
    if not isinstance(components, RuntimeComponentGraph):
        raise TypeError("components must be RuntimeComponentGraph")
    return AlgorithmRuntimeComponents(
        tuple(
            AlgorithmRuntimeComponent(
                selection=selection,
                instance=components.component(selection.role.value),
            )
            for selection in spec.components
        )
    )


def validate_algorithm_runtime_contracts(
    components: RuntimeComponentGraph,
    contracts: tuple[tuple[str, RuntimeBoundContract], ...],
) -> None:
    """Prove every G3 contract continues the exact loaded G1 receipt."""

    if not isinstance(components, RuntimeComponentGraph):
        raise TypeError("components must be RuntimeComponentGraph")
    if type(contracts) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], RuntimeBoundContract)
        for item in contracts
    ):
        raise TypeError("contracts must be (slot, RuntimeBoundContract) pairs")
    observed_slots = tuple(slot for slot, _contract in contracts)
    expected_slots = tuple(sorted(components.slots))
    if observed_slots != expected_slots:
        raise AlgorithmRuntimeBindingError(
            "G3 contracts do not exactly cover the loaded component graph"
        )
    for slot, contract in contracts:
        binding = components.binding(slot)
        if contract.artifact is not binding.artifact_binding:
            raise AlgorithmRuntimeBindingError(
                f"G3 contract for slot {slot!r} does not retain its exact G1 binding"
            )
        if contract.component_load_attestation is not binding.load_attestation:
            raise AlgorithmRuntimeBindingError(
                f"G3 contract for slot {slot!r} does not retain its load receipt"
            )


class StageBindingError(RuntimeError):
    """A trainer stage port was called unbound or rebound after preparation."""


class BindOnceStage:
    """Callable stage proxy with one explicit, irreversible bind operation."""

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("stage port name must be non-empty")
        self.name = name
        self._stage: object | None = None

    @property
    def is_bound(self) -> bool:
        return self._stage is not None

    def bind(self, stage: object) -> None:
        if self._stage is not None:
            raise StageBindingError(f"stage port {self.name!r} is already bound")
        if not callable(stage):
            raise TypeError("bound stage must be callable")
        self._stage = stage

    def __call__(self, value: StageValue[object]) -> StageValue[object]:
        if not isinstance(value, StageValue):
            raise TypeError(f"stage port {self.name!r} requires a StageValue")
        stage = self._stage
        if stage is None:
            raise StageBindingError(f"stage port {self.name!r} is not bound")
        result = stage(value)  # type: ignore[operator]
        if not isinstance(result, StageValue):
            raise TypeError(f"stage port {self.name!r} returned a non-StageValue")
        return result


class BindOncePrelude:
    """Prelude proxy preserving reservation commit/abort after late binding."""

    def __init__(self) -> None:
        self._prelude: object | None = None

    @property
    def is_bound(self) -> bool:
        return self._prelude is not None

    def bind(self, prelude: object) -> None:
        if self._prelude is not None:
            raise StageBindingError("prelude port is already bound")
        if not callable(getattr(prelude, "build", None)):
            raise TypeError("bound prelude must implement build(optimizer_step)")
        self._prelude = prelude

    def build(self, optimizer_step: int) -> StageValue[object]:
        prelude = self._require_bound()
        result = prelude.build(optimizer_step)  # type: ignore[attr-defined]
        if not isinstance(result, StageValue):
            raise TypeError("prelude port returned a non-StageValue")
        return result

    def commit_iteration(self, identity: IterationIdentity) -> None:
        self._finish("commit_iteration", identity)

    def abort_iteration(self, identity: IterationIdentity) -> None:
        self._finish("abort_iteration", identity)

    def _finish(self, method_name: str, identity: IterationIdentity) -> None:
        if not isinstance(identity, IterationIdentity):
            raise TypeError("iteration identity must be IterationIdentity")
        prelude = self._require_bound()
        method = getattr(prelude, method_name, None)
        if callable(method):
            method(identity)

    def _require_bound(self) -> Any:
        if self._prelude is None:
            raise StageBindingError("prelude port is not bound")
        return self._prelude


class CanonicalAlgorithmMaterializationError(ValueError):
    """A canonical request and its injected runtime components disagree."""


_ROLE_INSTANCE_TYPES = {
    AlgorithmComponentRole.TRAINER: TrainerComponent,
    AlgorithmComponentRole.DYNAMICS: DynamicsComponent,
    AlgorithmComponentRole.ROLLOUT: RolloutComponent,
    AlgorithmComponentRole.CONDITIONER: LatentConditioner,
    AlgorithmComponentRole.CREDIT: CreditPlanningPort,
}
_ROLE_ORDER = {role: index for index, role in enumerate(AlgorithmComponentRole)}


@dataclass(frozen=True, slots=True)
class AlgorithmRuntimeComponent:
    """One compiler selection paired with its already typed runtime instance."""

    selection: AlgorithmComponentSelection
    instance: object = field(compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selection, AlgorithmComponentSelection):
            raise TypeError("selection must be an AlgorithmComponentSelection")
        expected_type = _ROLE_INSTANCE_TYPES[self.selection.role]
        if not isinstance(self.instance, expected_type):
            raise TypeError(
                f"{self.selection.role.value} runtime component must implement "
                f"{expected_type.__name__}"
            )
        if self.selection.role is AlgorithmComponentRole.TRAINER:
            for method_name in ("prepare_run", "run_iteration", "close"):
                if not callable(getattr(self.instance, method_name, None)):
                    raise TypeError(
                        "trainer runtime component must implement " + method_name
                    )


@dataclass(frozen=True, slots=True)
class AlgorithmRuntimeComponents:
    """Complete typed component set injected after compatibility resolution."""

    values: tuple[AlgorithmRuntimeComponent, ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or any(
            not isinstance(item, AlgorithmRuntimeComponent) for item in self.values
        ):
            raise TypeError("values must contain AlgorithmRuntimeComponent instances")
        roles = tuple(item.selection.role for item in self.values)
        if len(roles) != len(set(roles)):
            raise ValueError("runtime algorithm component roles must be unique")
        required = {
            AlgorithmComponentRole.TRAINER,
            AlgorithmComponentRole.DYNAMICS,
            AlgorithmComponentRole.ROLLOUT,
            AlgorithmComponentRole.CREDIT,
        }
        missing = tuple(sorted(role.value for role in required - set(roles)))
        if missing:
            raise ValueError(
                f"runtime algorithm components are missing roles {list(missing)}"
            )
        ordered = tuple(
            sorted(self.values, key=lambda item: _ROLE_ORDER[item.selection.role])
        )
        identities = tuple(id(item.instance) for item in ordered)
        if len(identities) != len(set(identities)):
            raise ValueError("runtime algorithm component instances must be unique")
        object.__setattr__(self, "values", ordered)

    def binding(self, role: AlgorithmComponentRole) -> AlgorithmRuntimeComponent:
        if not isinstance(role, AlgorithmComponentRole):
            raise TypeError("role must be an AlgorithmComponentRole")
        matches = tuple(item for item in self.values if item.selection.role is role)
        if len(matches) != 1:
            raise KeyError(role.value)
        return matches[0]


class TrainerAlgorithmExecution(AlgorithmExecutionPort):
    """Lifecycle and checkpoint adapter around the canonical Trainer owner."""

    _STATE_KEYS = frozenset(
        {
            "schema_version",
            "execution_plan_id",
            "execution_policy_id",
            "materialization_spec_id",
            "algorithm_binding_id",
            "next_optimizer_step",
        }
    )

    def __init__(
        self,
        *,
        request: AlgorithmMaterializationRequest,
        plan: AlgorithmExecutionPlan,
        trainer: TrainerComponent,
    ) -> None:
        if not isinstance(request, AlgorithmMaterializationRequest):
            raise TypeError("request must be an AlgorithmMaterializationRequest")
        if not isinstance(plan, AlgorithmExecutionPlan):
            raise TypeError("plan must be an AlgorithmExecutionPlan")
        if not isinstance(trainer, TrainerComponent):
            raise TypeError("trainer must implement TrainerComponent")
        if getattr(trainer, "state", None) is not TrainerState.NEW:
            raise ValueError("injected trainer must be in the NEW state")
        self._request = request
        self._plan = plan
        self._trainer = trainer
        self._restored_next_optimizer_step: int | None = None

    @property
    def plan(self) -> AlgorithmExecutionPlan:
        return self._plan

    @property
    def trainer(self) -> TrainerComponent:
        return self._trainer

    def prepare_run(self, context: object) -> None:
        if not isinstance(context, PrepareRunContext):
            raise TypeError("context must be a PrepareRunContext")
        restored = self._restored_next_optimizer_step
        if restored is not None and context.start_optimizer_step != restored:
            raise ValueError(
                "prepare context start_optimizer_step differs from restored state"
            )
        self._trainer.prepare_run(context)  # type: ignore[attr-defined]

    def run_iteration(self, optimizer_step: int) -> IterationResult[object]:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        result = self._trainer.run_iteration(optimizer_step)  # type: ignore[attr-defined]
        if not isinstance(result, IterationResult):
            raise TypeError("canonical trainer must return IterationResult")
        return result

    def state_dict(self) -> Mapping[str, object]:
        if getattr(self._trainer, "state", None) is TrainerState.CLOSED:
            raise RuntimeError("a closed trainer execution has no capturable state")
        next_optimizer_step = getattr(self._trainer, "next_optimizer_step", None)
        if next_optimizer_step is None:
            next_optimizer_step = self._restored_next_optimizer_step
        if next_optimizer_step is not None and (
            type(next_optimizer_step) is not int or next_optimizer_step < 0
        ):
            raise TypeError("trainer next_optimizer_step must be non-negative or None")
        return {
            "schema_version": 1,
            "execution_plan_id": self._plan.plan_id,
            "execution_policy_id": self._request.execution_policy.policy_id,
            "materialization_spec_id": self._request.spec.spec_id,
            "algorithm_binding_id": self._request.binding.binding_id,
            "next_optimizer_step": next_optimizer_step,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if getattr(self._trainer, "state", None) is not TrainerState.NEW:
            raise RuntimeError("trainer execution state must load before prepare_run()")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if set(state) != self._STATE_KEYS:
            raise ValueError("trainer execution state has unexpected fields")
        expected = {
            "schema_version": 1,
            "execution_plan_id": self._plan.plan_id,
            "execution_policy_id": self._request.execution_policy.policy_id,
            "materialization_spec_id": self._request.spec.spec_id,
            "algorithm_binding_id": self._request.binding.binding_id,
        }
        for name, value in expected.items():
            if state[name] != value:
                raise ValueError(f"trainer execution state {name} mismatch")
        next_optimizer_step = state["next_optimizer_step"]
        if next_optimizer_step is not None and (
            type(next_optimizer_step) is not int or next_optimizer_step < 0
        ):
            raise ValueError("next_optimizer_step must be non-negative or None")
        self._restored_next_optimizer_step = next_optimizer_step

    def close(self) -> None:
        # The canonical runtime component graph owns the injected Trainer.
        # BoundAlgorithm owns only this execution facade, so closing the facade
        # must not close a graph component ahead of model/prepared teardown.
        return None


class CanonicalAlgorithmMaterializer(AlgorithmMaterializationPort):
    """One-shot shadow materializer for a compiler-owned algorithm spec."""

    def __init__(self, components: AlgorithmRuntimeComponents) -> None:
        if not isinstance(components, AlgorithmRuntimeComponents):
            raise TypeError("components must be AlgorithmRuntimeComponents")
        self._components = components
        self._execution: TrainerAlgorithmExecution | None = None

    @property
    def execution(self) -> TrainerAlgorithmExecution | None:
        return self._execution

    def materialize(
        self,
        request: AlgorithmMaterializationRequest,
    ) -> AlgorithmExecutionPort:
        if self._execution is not None:
            raise RuntimeError("canonical algorithm materializer is one-shot")
        if not isinstance(request, AlgorithmMaterializationRequest):
            raise TypeError("request must be an AlgorithmMaterializationRequest")
        self._validate_components(request)
        plan = AlgorithmExecutionPlan.from_spec(
            request.spec,
            execution_policy=request.execution_policy,
        )
        trainer_binding = self._components.binding(AlgorithmComponentRole.TRAINER)
        if plan.trainer_family != trainer_binding.selection.selected_component_id:
            raise CanonicalAlgorithmMaterializationError(
                "execution plan trainer differs from the injected trainer"
            )
        execution = TrainerAlgorithmExecution(
            request=request,
            plan=plan,
            trainer=trainer_binding.instance,  # type: ignore[arg-type]
        )
        self._execution = execution
        return execution

    def _validate_components(self, request: AlgorithmMaterializationRequest) -> None:
        expected = {item.role: item for item in request.spec.components}
        observed = {item.selection.role: item for item in self._components.values}
        if set(observed) != set(expected):
            missing = tuple(
                sorted(role.value for role in set(expected) - set(observed))
            )
            extra = tuple(sorted(role.value for role in set(observed) - set(expected)))
            raise CanonicalAlgorithmMaterializationError(
                f"runtime/spec component roles differ; missing={list(missing)}, "
                f"extra={list(extra)}"
            )
        for role, selection in expected.items():
            if observed[role].selection != selection:
                raise CanonicalAlgorithmMaterializationError(
                    f"injected {role.value} selection differs from the spec"
                )

        rollout = observed[AlgorithmComponentRole.ROLLOUT].instance
        observed_execution_policy = rollout.execution_policy  # type: ignore[attr-defined]
        if type(observed_execution_policy) is not ExecutionPolicyReceipt:
            raise CanonicalAlgorithmMaterializationError(
                "injected rollout has no canonical execution-policy receipt"
            )
        try:
            observed_execution_policy = observed_execution_policy.validated_projection(
                request.spec.execution_policy_id
            )
            expected_execution_policy = request.execution_policy.validated_projection(
                request.spec.execution_policy_id
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalAlgorithmMaterializationError(
                "injected rollout execution policy differs from the materialization "
                "request"
            ) from exc
        if observed_execution_policy != expected_execution_policy:
            raise CanonicalAlgorithmMaterializationError(
                "injected rollout execution policy differs from the materialization "
                "request"
            )

        for slot in request.blueprint.slots:
            selection = expected[slot.role]
            if selection.implementation_family != slot.implementation_family:
                raise CanonicalAlgorithmMaterializationError(
                    f"{slot.role.value} implementation family differs from blueprint"
                )
            if selection.resolution is not slot.resolution:
                raise CanonicalAlgorithmMaterializationError(
                    f"{slot.role.value} resolution differs from blueprint"
                )
            if slot.resolution is AlgorithmComponentResolution.ALGORITHM_DEFAULT and (
                selection.selected_component_id != slot.component_id
            ):
                raise CanonicalAlgorithmMaterializationError(
                    f"{slot.role.value} component differs from blueprint default"
                )


class DefaultStageAssemblyError(RuntimeError):
    """A bound production graph cannot form the supported V0 hot path."""


class DefaultStageAssembler:
    """Assemble one real, single-rank, one-update-per-iteration stage graph."""

    def assemble(self, request: StageAssemblyRequest) -> TrainerStageAssembly:
        from visual_rl.runtime.types import StageCheckpointPorts, TrainerStageAssembly

        self._validate_request(request)

        materialized = request.preflight.environment.materialized
        resolved = materialized.resolved
        training = request.prepared.training
        runtime_facts = request.runtime.session.runtime_facts
        components = request.graph.components.as_mapping()
        algorithm = AlgorithmExecutionPlan.from_spec(
            resolved.algorithm_spec,
            execution_policy=resolved.execution_policy.to_receipt(),
        )

        adapter = self._component(components, "model", ModelAdapter)
        if adapter is not request.prepared.manager.adapter:
            raise DefaultStageAssemblyError(
                "prepared adapter differs from the runtime graph model"
            )
        policy_runtime = request.policy_runtime
        if policy_runtime is None:
            # Compatibility for focused unit assemblies. Production always
            # supplies the G3-bound facade, and its tests assert this exact
            # object reaches rollout and policy recompute.
            policy_runtime = adapter
        rollout = self._component(components, "rollout", RolloutComponent)
        credit = self._component(components, "credit", CreditPlanningPort)
        conditioner = components.get("conditioner")
        if conditioner is not None and not isinstance(conditioner, LatentConditioner):
            raise TypeError("conditioner component must be a LatentConditioner")

        live_num_steps = rollout.num_steps
        if type(live_num_steps) is not int or live_num_steps < 1:
            raise DefaultStageAssemblyError(
                "rollout.num_steps must be a positive integer"
            )
        schedule_step_count = algorithm.schedule_step_count
        if live_num_steps != schedule_step_count:
            raise DefaultStageAssemblyError(
                "live rollout.num_steps differs from "
                "AlgorithmExecutionPlan.schedule_step_count"
            )

        phase_schedule = resolved.phase_schedule
        if phase_schedule is None:
            routes = materialized.reward_plan.routes
            if len(routes) != 1:
                raise DefaultStageAssemblyError(
                    "phase_schedule=None requires exactly one typed reward route"
                )
            phase_schedule = ImplicitPhaseRouter(routes[0])
        source_sequences = load_stable_source_sequences(
            SourceLoadRequest(
                plan=resolved.source_plan,
                locations=request.preflight.environment.source_locations,
            )
        )
        source_sampler = MultiSourceSampler(source_sequences)
        group_size = algorithm.policy_microbatch_cardinality.row_count
        placement = GroupPlacementContract(
            placement=GroupPlacementKind.LOCAL_COMPLETE,
            global_prompt_batch_size=training.global_prompt_batch_size,
            group_size=group_size,
            world_size=runtime_facts.world_size,
            per_rank_microbatch_rows=(training.global_prompt_batch_size * group_size),
            gradient_accumulation_steps=training.gradient_accumulation_steps,
        )
        prelude = DataPlanePrelude(
            phase_schedule=phase_schedule,
            source_sampler=source_sampler,
            placement_contract=placement,
            collator=ExplicitCollator(),
        )

        tensor_runtime = request.evidence.policy_tensor_runtime_spec
        rollout_request_factory = IterationRolloutRequestFactory(
            adapter=policy_runtime,
            dynamics_factory=request.dynamics_factory,
            num_steps=schedule_step_count,
            likelihood_semantics=algorithm.likelihood_semantics,
            base_seed=training.seed,
            device=tensor_runtime.torch_device,
            dtype=tensor_runtime.latent_storage_torch_dtype,
            conditioner=conditioner,
            selection_contract_identity=rollout.selection_contract_identity,
        )
        transform_plan_id = request.transforms.plan_id
        manager = request.prepared.manager
        preprocess_policy = StageExecutionPolicy.canonical(
            ExecutionMode.PREPROCESS,
            transform_plan_id=transform_plan_id,
        )
        rollout_policy = StageExecutionPolicy.canonical(
            ExecutionMode.ROLLOUT,
            transform_plan_id=transform_plan_id,
        )
        current_policy = StageExecutionPolicy.canonical(
            ExecutionMode.TRAIN,
            transform_plan_id=transform_plan_id,
        )
        reference_policy = (
            StageExecutionPolicy.canonical(
                ExecutionMode.TRAIN,
                parameter_view=ParameterView.REFERENCE,
                transform_plan_id=transform_plan_id,
            )
            if algorithm.requires_reference_statistics
            else None
        )

        rollout_stage = RolloutStage(
            rollout=rollout,
            request_factory=rollout_request_factory,
            preprocess_context=lambda: manager.execution(preprocess_policy),
            rollout_context=lambda: manager.execution(rollout_policy),
        )

        reward_plan = materialized.reward_plan
        container = self._reward_container(request.runtime)
        if container.state is not RewardResourceState.ACQUIRED:
            raise DefaultStageAssemblyError(
                "stage assembly requires ACQUIRED, not ACTIVE, reward resources"
            )
        if container.plan != reward_plan:
            raise DefaultStageAssemblyError(
                "session reward plan differs from the materialized recipe"
            )
        logical_rewards = self._logical_rewards(
            components, reward_plan.logical_reward_ids
        )
        reward_stage = RewardStage(
            plan=reward_plan,
            pool=container.view(),
            logical_rewards=logical_rewards,
        )

        def reward_runtime_context_factory(
            payload: RolloutStagePayload,
            identity: IterationIdentity,
        ) -> RewardRuntimeContext:
            if not isinstance(payload, RolloutStagePayload):
                raise TypeError("reward context payload must be RolloutStagePayload")
            if not isinstance(identity, IterationIdentity):
                raise TypeError("reward context identity must be IterationIdentity")
            route = payload.phase_route
            if route is None:
                raise DefaultStageAssemblyError(
                    "default reward execution requires DataPlanePrelude routing"
                )
            if (
                route.optimizer_step != identity.optimizer_step
                or route.source_id != identity.source_id
                or route.phase_id != identity.phase_id
                or route.active_rewards != payload.active_rewards
            ):
                raise DefaultStageAssemblyError(
                    "reward context route differs from the iteration identity"
                )
            seed = (
                training.seed
                + identity.optimizer_step * runtime_facts.world_size
                + runtime_facts.rank
            )
            return RewardRuntimeContext(
                step_context=StepContext(
                    step=identity.optimizer_step,
                    seed=seed,
                    rank=runtime_facts.rank,
                    world_size=runtime_facts.world_size,
                )
            )

        reward_pipeline = RewardPipelineStage(
            reward_stage,
            runtime_context_factory=reward_runtime_context_factory,
        )
        advantage_stage = AdvantageStage(
            GroupZScoreAdvantageProcessor(
                epsilon=credit.advantage_epsilon,
                std_domain=credit.advantage_std_domain,
            )
        )
        credit_stage = CreditStage(
            strategy=credit,
            coefficient_mean_reducer=LocalCoefficientMeanReducer(),
        )

        safety = training.update_safety
        recompute = training.policy_recompute
        update_plan = UpdateExecutionPlan(
            require_finite_gradients=safety.require_finite_gradients,
            require_nonzero_gradients=safety.require_nonzero_gradients,
            max_grad_norm=safety.max_grad_norm,
            zero_grad_set_to_none=safety.zero_grad_set_to_none,
            # This only bounds autograd replay geometry. Reward grouping,
            # advantage normalization and exact active-count weighting still
            # operate over the complete logical K x T update.
            row_microbatch_size=recompute.row_microbatch_size,
            transition_window_size=recompute.transition_window_size,
        )
        optimize_stage = OptimizeStage(
            optimizer=request.prepared.optimizer,
            # Accelerate owns mixed-precision scaling for its prepared optimizer.
            # Supplying the same scaler again would create a second step owner.
            scaler=None,
            kernel=PolicyUpdateKernel(
                max_initial_logprob_delta=safety.max_initial_logprob_delta,
                require_initial_clipfrac_zero=(safety.require_initial_clipfrac_zero),
                require_finite_gradients=safety.require_finite_gradients,
                require_nonzero_gradients=safety.require_nonzero_gradients,
                max_grad_norm=safety.max_grad_norm,
            ),
            accelerator=request.runtime.session.accelerator,
            prepared_root=request.prepared.handle.accumulation_root,
            lr_scheduler=request.prepared.lr_scheduler,
            execution_plan=update_plan,
            require_reference_statistics=algorithm.requires_reference_statistics,
            current_context=lambda: manager.execution(current_policy),
            reference_context=(
                None
                if reference_policy is None
                else lambda: manager.execution(reference_policy)
            ),
        )
        return TrainerStageAssembly(
            prelude=prelude,
            rollout=rollout_stage,
            reward=reward_pipeline,
            advantage=advantage_stage,
            credit=credit_stage,
            optimize=optimize_stage,
            reward_runtime_context_factory=reward_runtime_context_factory,
            # RewardPool ownership stays with RuntimeSession.resource_container.
            close_resources=(),
            checkpoint_ports=StageCheckpointPorts(
                data_plane=prelude,
                dynamics_selection_policy=(
                    rollout_request_factory.dynamics_selection_policy
                ),
                update_execution_plan=update_plan,
                rollout_request_factory=rollout_request_factory,
            ),
        )

    @staticmethod
    def _validate_request(request: StageAssemblyRequest) -> None:
        from visual_rl.runtime.types import (
            ProductionGraph,
            ProductionPreflight,
            ProductionPreparedRun,
            ProductionRuntime,
            StageAssemblyRequest,
        )

        if not isinstance(request, StageAssemblyRequest):
            raise TypeError("request must be StageAssemblyRequest")
        if not isinstance(request.preflight, ProductionPreflight):
            raise TypeError("request.preflight must be ProductionPreflight")
        if not isinstance(request.runtime, ProductionRuntime):
            raise TypeError("request.runtime must be ProductionRuntime")
        if not isinstance(request.graph, ProductionGraph):
            raise TypeError("request.graph must be ProductionGraph")
        if not isinstance(request.prepared, ProductionPreparedRun):
            raise TypeError("request.prepared must be ProductionPreparedRun")
        if request.runtime.preflight is not request.preflight:
            raise DefaultStageAssemblyError(
                "runtime and stage assembly use different preflight objects"
            )
        if request.prepared.training != request.preflight.compiled.training:
            raise DefaultStageAssemblyError(
                "prepared and compiled training semantics differ"
            )
        facts = request.runtime.session.runtime_facts
        tensor_runtime = request.evidence.policy_tensor_runtime_spec
        if tensor_runtime.device != facts.device:
            raise DefaultStageAssemblyError(
                "policy tensor device differs from RuntimeFacts"
            )
        if tensor_runtime.model_compute_precision != facts.precision:
            raise DefaultStageAssemblyError(
                "policy tensor compute precision differs from RuntimeFacts"
            )
        if (
            facts.distribution_mode != "single"
            or facts.rank != 0
            or facts.local_rank != 0
            or facts.world_size != 1
        ):
            raise DefaultStageAssemblyError(
                "DefaultStageAssembler supports only a single-process runtime"
            )
        if request.prepared.training.gradient_accumulation_steps != 1:
            raise DefaultStageAssemblyError(
                "DefaultStageAssembler requires gradient_accumulation_steps=1"
            )
        if request.prepared.manager.runtime_bound is not (
            request.evidence.model_runtime_contract
        ):
            raise DefaultStageAssemblyError(
                "ComponentManager is not bound to the supplied model evidence"
            )
        evidence_contract_ids = FrozenMapping(
            (slot, contract.contract_id)
            for slot, contract in request.evidence.runtime_bound_contracts
        )
        if request.graph_binding.component_bound_contract_ids != evidence_contract_ids:
            raise DefaultStageAssemblyError(
                "graph binding and component runtime evidence differ"
            )
        if (
            request.graph_binding.bound_reward_resource_ids
            != request.evidence.bound_reward_resource_ids
        ):
            raise DefaultStageAssemblyError(
                "graph binding and physical reward evidence differ"
            )
        accelerator = request.runtime.session.accelerator
        for method in ("accumulate", "backward"):
            if not callable(getattr(accelerator, method, None)):
                raise TypeError(f"runtime accelerator must implement {method}()")

    @staticmethod
    def _component(
        components: Mapping[str, object],
        slot: str,
        expected_type: type,
    ) -> object:
        try:
            component = components[slot]
        except KeyError as exc:
            raise DefaultStageAssemblyError(
                f"runtime graph is missing required slot {slot!r}"
            ) from exc
        if not isinstance(component, expected_type):
            raise TypeError(
                f"runtime component {slot!r} must be {expected_type.__name__}"
            )
        return component

    @staticmethod
    def _reward_container(
        runtime: ProductionRuntime,
    ) -> DefaultRuntimeResourceContainer:
        container = runtime.session.resource_container
        if not isinstance(container, DefaultRuntimeResourceContainer):
            raise DefaultStageAssemblyError(
                "default stage assembly requires a session reward container"
            )
        try:
            bound_ids = container.bound_reward_resource_ids
        except RuntimeError as exc:
            raise DefaultStageAssemblyError(
                "reward resources must be acquired before stage assembly"
            ) from exc
        if not bound_ids:
            raise DefaultStageAssemblyError(
                "acquired reward resources must expose bound identities"
            )
        return container

    @staticmethod
    def _logical_rewards(
        components: Mapping[str, object],
        logical_ids: tuple[str, ...],
    ) -> Mapping[str, PointwiseReward | GroupwiseReward]:
        rewards: dict[str, PointwiseReward | GroupwiseReward] = {}
        for logical_id in logical_ids:
            slot = f"rewards.{logical_id}"
            try:
                reward = components[slot]
            except KeyError as exc:
                raise DefaultStageAssemblyError(
                    f"runtime graph is missing logical reward {logical_id!r}"
                ) from exc
            if not isinstance(reward, (PointwiseReward, GroupwiseReward)):
                raise TypeError(
                    f"runtime component {slot!r} must be a typed logical reward"
                )
            rewards[logical_id] = reward
        observed_reward_slots = {
            slot.removeprefix("rewards.")
            for slot in components
            if slot.startswith("rewards.")
        }
        if observed_reward_slots != set(logical_ids):
            raise DefaultStageAssemblyError(
                "runtime logical reward slots do not exactly cover the reward plan"
            )
        return rewards
