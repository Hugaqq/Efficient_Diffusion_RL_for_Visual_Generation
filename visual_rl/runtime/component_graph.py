"""Declaration-bound construction of the canonical runtime component graph.

The environment preflight result is the only graph authority.  This module
never resolves aliases, re-reads source configuration, consults a registry, or
re-derives G1.  It gives the exact preflight G1 set/load plan to
:class:`RuntimeComponentLoader`, which verifies every slot before importing the
first runtime implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from visual_rl.artifacts.checkpoint.reference import ReferencePolicyStateEvidence
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.composition.registry import ResolvedComponentDeclaration
from visual_rl.core.contracts import (
    ComponentArtifactBinding,
    ComponentArtifactBindingSet,
    ComponentLoadAttestation,
    ComponentLoadPlan,
    RuntimeBoundContract,
    ModelContract,
)
from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data import PreprocessRequirementSet
from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence
from visual_rl.composition.preflight.types import (
    EnvironmentPreflightResult,
    RuntimeBindResult,
)
from visual_rl.runtime.component_loader import (
    RuntimeComponentLoader,
    RuntimeComponentLoadGate,
    RuntimeComponentLoadRequest,
)

if TYPE_CHECKING:
    AlgorithmExecutionPlan = Any
    from visual_rl.runtime.model_binding import (
        ModelRuntimeProbe,
        ModelRuntimeProbeResult,
    )
    from visual_rl.runtime.preprocess_binding import PreprocessIdentityProvider
    from visual_rl.runtime.reward_resources import RewardLogicalBinding
    from visual_rl.runtime.types import (
        ComponentBindRequest,
        ComponentRuntimeEvidence,
        PolicyTensorRuntimeSpec,
        ProductionGraph,
    )

__all__ = (
    "ComponentRuntimeBindingError",
    "DefaultComponentRuntimeBinder",
    "RuntimeAssemblyError",
    "RuntimeComponentBinding",
    "RuntimeComponentGraph",
    "RuntimeContextFactory",
    "load_component_graph",
)


class ComponentRuntimeBindingError(RuntimeError):
    """Prepared G3 inputs drifted from the artifact-locked component graph."""


class RuntimeAssemblyError(RuntimeError):
    """The typed materialized graph cannot form one exact runtime graph."""


RuntimeContextFactory = Callable[
    [str, ResolvedComponentDeclaration, Mapping[str, object]],
    Mapping[str, Any],
]


_KIND_LOAD_ORDER = {
    "model": 0,
    "algorithm": 1,
    "dynamics": 2,
    "conditioner": 3,
    "rollout": 4,
    "reward": 5,
    "credit": 6,
    "trainer": 7,
}


@dataclass(frozen=True, slots=True)
class RuntimeComponentBinding:
    """One live component with the exact static declaration and G1 receipt."""

    slot: str
    declaration: ResolvedComponentDeclaration
    artifact_binding: ComponentArtifactBinding
    load_attestation: ComponentLoadAttestation
    instance: object = field(compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or not self.slot:
            raise ValueError("runtime component slot must be non-empty")
        if not isinstance(self.declaration, ResolvedComponentDeclaration):
            raise TypeError("declaration must be ResolvedComponentDeclaration")
        if not isinstance(self.artifact_binding, ComponentArtifactBinding):
            raise TypeError("artifact_binding must be ComponentArtifactBinding")
        if not isinstance(self.load_attestation, ComponentLoadAttestation):
            raise TypeError("load_attestation must be ComponentLoadAttestation")
        if self.artifact_binding.slot != self.slot:
            raise ValueError("runtime slot differs from its G1 artifact binding")
        if (
            self.artifact_binding.component_declaration_id
            != self.declaration.declaration_id
        ):
            raise ValueError("runtime declaration differs from its G1 binding")
        if self.load_attestation.binding_id != self.artifact_binding.binding_id:
            raise ValueError("runtime load attestation differs from its G1 binding")

    @property
    def kind(self) -> str:
        return self.declaration.kind

    @property
    def config(self) -> object:
        return self.declaration.config

    @property
    def declared_contract(self) -> object:
        return self.declaration.declared_contract

    def attest_prepare(
        self,
        *,
        runtime_identity: str,
        verified_fields: tuple[tuple[str, str], ...],
    ) -> RuntimeBoundContract:
        """Emit G3 evidence which references this exact in-memory G1 receipt."""

        from visual_rl.core.contracts import ComponentPrepareAttestation

        prepare_attestation = ComponentPrepareAttestation(
            binding_id=self.artifact_binding.binding_id,
            load_attestation_id=self.load_attestation.attestation_id,
            runtime_identity=runtime_identity,
            verified_fields=verified_fields,
        )
        return RuntimeBoundContract(
            artifact=self.artifact_binding,
            runtime_identity=runtime_identity,
            verified_fields=verified_fields,
            load_attestation=self.load_attestation,
            prepare_attestation=prepare_attestation,
        )


@dataclass(slots=True)
class RuntimeComponentGraph:
    """Owned all-slot component graph and the immutable G1 proof that loaded it."""

    _bindings: tuple[RuntimeComponentBinding, ...]
    artifact_binding_set: ComponentArtifactBindingSet
    load_plan: ComponentLoadPlan
    _closed: bool = False

    def __post_init__(self) -> None:
        if type(self._bindings) is not tuple or not self._bindings:
            raise ValueError("runtime component graph must not be empty")
        if any(
            not isinstance(binding, RuntimeComponentBinding)
            for binding in self._bindings
        ):
            raise TypeError(
                "runtime component graph must contain RuntimeComponentBinding values"
            )
        if not isinstance(self.artifact_binding_set, ComponentArtifactBindingSet):
            raise TypeError("artifact_binding_set must be ComponentArtifactBindingSet")
        if not isinstance(self.load_plan, ComponentLoadPlan):
            raise TypeError("load_plan must be ComponentLoadPlan")
        slots = tuple(binding.slot for binding in self._bindings)
        if len(slots) != len(set(slots)):
            raise ValueError("runtime component slots must be unique")
        if set(slots) != set(self.artifact_binding_set.slots):
            raise ValueError("runtime components do not exactly cover the G1 set")
        if self.load_plan.expected_binding_set_id != (
            self.artifact_binding_set.binding_set_id
        ):
            raise ValueError("runtime load plan differs from its G1 binding set")
        for binding in self._bindings:
            if (
                binding.artifact_binding.binding_id
                != self.artifact_binding_set.binding(binding.slot).binding_id
            ):
                raise ValueError(
                    f"runtime component {binding.slot!r} differs from the G1 set"
                )

    @property
    def recipe_id(self) -> str:
        return self.artifact_binding_set.recipe_id

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(binding.slot for binding in self._bindings)

    @property
    def bindings(self) -> tuple[RuntimeComponentBinding, ...]:
        if self._closed:
            raise RuntimeError("runtime component graph is closed")
        return self._bindings

    def binding(self, slot: str) -> RuntimeComponentBinding:
        if self._closed:
            raise RuntimeError("runtime component graph is closed")
        for binding in self._bindings:
            if binding.slot == slot:
                return binding
        raise KeyError(slot)

    def component(self, slot: str) -> object:
        return self.binding(slot).instance

    def as_mapping(self) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("runtime component graph is closed")
        return {binding.slot: binding.instance for binding in self._bindings}

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        seen: set[int] = set()
        for binding in reversed(self._bindings):
            component = binding.instance
            if id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
        self._closed = True
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        f"additional component close failure: {type(error).__name__}"
                    )
            raise primary

    def __enter__(self) -> RuntimeComponentGraph:  # noqa: PYI034
        if self._closed:
            raise RuntimeError("runtime component graph is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def load_component_graph(
    environment: EnvironmentPreflightResult,
    runtime_binding: RuntimeBindResult,
    *,
    runtime_context_factory: RuntimeContextFactory,
    loader: RuntimeComponentLoader | None = None,
) -> RuntimeComponentGraph:
    """Construct the exact typed recipe graph through the strict all-slot gate."""

    if not isinstance(environment, EnvironmentPreflightResult):
        raise TypeError("environment must be an EnvironmentPreflightResult")
    if not isinstance(runtime_binding, RuntimeBindResult):
        raise TypeError("runtime_binding must be a RuntimeBindResult")
    if not callable(runtime_context_factory):
        raise TypeError("runtime_context_factory must be callable")
    component_loader = RuntimeComponentLoader() if loader is None else loader
    if not isinstance(component_loader, RuntimeComponentLoader):
        raise TypeError("loader must be RuntimeComponentLoader or None")

    recipe = environment.materialized
    if runtime_binding.recipe_id != recipe.recipe_id:
        raise RuntimeAssemblyError(
            "validated runtime binding belongs to a stale materialized recipe"
        )
    binding_set = environment.component_artifact_bindings
    load_plan = environment.component_load_plan
    if binding_set.recipe_id != recipe.recipe_id:
        raise RuntimeAssemblyError("preflight G1 set belongs to a stale recipe")
    if load_plan.expected_recipe_id != recipe.recipe_id:
        raise RuntimeAssemblyError("preflight load plan belongs to a stale recipe")
    declarations = tuple(
        sorted(
            _recipe_declarations(recipe),
            key=lambda item: (_KIND_LOAD_ORDER[item[1].kind], item[0]),
        )
    )
    requests: list[RuntimeComponentLoadRequest] = []
    for slot, declaration in declarations:
        requests.append(
            RuntimeComponentLoadRequest(
                declaration=declaration,
                gate=RuntimeComponentLoadGate(
                    runtime_binding=runtime_binding,
                    artifact_binding=binding_set.binding(slot),
                ),
                # The strict loader resolves this only after every slot has
                # passed G1/load-plan/interface gates.
                runtime_context={},
            )
        )

    def resolve_context(
        request: RuntimeComponentLoadRequest,
        loaded: Mapping[str, object],
    ) -> Mapping[str, Any]:
        slot = request.gate.artifact_binding.slot
        context = runtime_context_factory(slot, request.declaration, loaded)
        if not isinstance(context, Mapping):
            raise TypeError(
                f"runtime_context_factory must return a Mapping for slot {slot!r}"
            )
        return context

    results = component_loader.load_all(
        tuple(requests),
        binding_set=binding_set,
        load_plan=load_plan,
        context_resolver=resolve_context,
    )
    bindings = tuple(
        RuntimeComponentBinding(
            slot=slot,
            declaration=request.declaration,
            artifact_binding=result.artifact_binding,
            load_attestation=result.load_attestation,
            instance=result.instance,
        )
        for (slot, _declaration), request, result in zip(
            declarations,
            requests,
            results,
            strict=True,
        )
    )
    return RuntimeComponentGraph(bindings, binding_set, load_plan)


def _recipe_declarations(
    recipe: MaterializedRecipe,
) -> tuple[tuple[str, ResolvedComponentDeclaration], ...]:
    resolved = recipe.resolved
    values = (
        ("algorithm", resolved.algorithm.component),
        (resolved.model.slot, resolved.model.declaration),
        *((item.slot, item.declaration) for item in resolved.internal_components),
        *((item.slot, item.declaration) for item in resolved.reward_components),
    )
    slots = tuple(slot for slot, _declaration in values)
    if len(slots) != len(set(slots)):
        raise RuntimeAssemblyError("materialized recipe component slots are not unique")
    return tuple(values)


_COMPONENT_BOUND_ID_DOMAIN = b"visual_rl.component-bound-contract.v1\0"


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _domain_hash(domain: bytes, payload: Mapping[str, object]) -> str:
    encoded = canonical_json_text(payload).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


class DefaultComponentRuntimeBinder:
    """Bind one already prepared single-process component graph at G3."""

    def __init__(
        self,
        *,
        model_probe: ModelRuntimeProbe,
        preprocess_identity_provider: PreprocessIdentityProvider,
    ) -> None:
        from visual_rl.runtime.model_binding import ModelRuntimeProbe
        from visual_rl.runtime.preprocess_binding import PreprocessIdentityProvider

        if not isinstance(model_probe, ModelRuntimeProbe):
            raise TypeError("model_probe must implement ModelRuntimeProbe")
        if not isinstance(preprocess_identity_provider, PreprocessIdentityProvider):
            raise TypeError(
                "preprocess_identity_provider must implement PreprocessIdentityProvider"
            )
        self._model_probe = model_probe
        self._preprocess_identity_provider = preprocess_identity_provider

    def bind(self, request: ComponentBindRequest) -> ComponentRuntimeEvidence:
        """Acquire rewards, bind logical ports, then collect exact G3 evidence."""

        from visual_rl.runtime.algorithm_binding import bind_algorithm_execution_plan
        from visual_rl.runtime.model_binding import (
            ModelRuntimeProbeRequest,
            ModelRuntimeProbeResult,
            resolve_policy_tensor_runtime_spec,
            validate_model_runtime_contract,
            validate_prepared_model,
        )
        from visual_rl.runtime.preprocess_binding import (
            resolve_preprocess_runtime_binding,
        )
        from visual_rl.runtime.resources import DefaultRuntimeResourceContainer
        from visual_rl.runtime.reward_resources import (
            acquire_reward_view,
            bind_logical_rewards,
            resolve_reward_bindings,
        )
        from visual_rl.runtime.types import ComponentRuntimeEvidence

        self._validate_request(request)
        materialized = request.preflight.environment.materialized
        runtime_facts = request.runtime.session.runtime_facts
        graph = request.graph.components
        bindings = self._validate_graph_coverage(request, graph)
        model_binding = bindings["model"]
        validate_prepared_model(request.prepared, model_binding)

        reward_plan = materialized.reward_plan
        logical_rewards, acquisition_requests = resolve_reward_bindings(
            materialized=materialized,
            launch=request.preflight.compiled.launch,
            runtime_facts=runtime_facts,
            plan=reward_plan,
            bindings=bindings,
        )
        container = request.runtime.session.resource_container
        if not isinstance(container, DefaultRuntimeResourceContainer):
            raise ComponentRuntimeBindingError(
                "runtime session has no DefaultRuntimeResourceContainer"
            )
        reward_view = acquire_reward_view(
            container,
            reward_plan,
            acquisition_requests,
        )
        bind_logical_rewards(logical_rewards, reward_view)

        probe_request = ModelRuntimeProbeRequest(
            materialized=materialized,
            model_binding=model_binding,
            manager=request.prepared.manager,
            handle=request.prepared.handle,
            runtime_facts=runtime_facts,
        )
        probe_result = self._model_probe.probe(probe_request)
        if not isinstance(probe_result, ModelRuntimeProbeResult):
            raise TypeError("model probe must return ModelRuntimeProbeResult")
        validate_model_runtime_contract(
            probe_result,
            model_binding=model_binding,
        )
        policy_tensor_runtime_spec = resolve_policy_tensor_runtime_spec(
            dynamics_binding=bindings["dynamics"],
            runtime_facts=runtime_facts,
            runtime_numerics=probe_result.runtime_numerics,
        )
        model_execution_numerics = request.prepared.manager.model_execution_numerics
        if (
            model_execution_numerics.source_projection_id
            != request.prepared.manager.parameter_state.state_projection.projection_id
        ):
            raise ComponentRuntimeBindingError(
                "model execution numerics use a stale state projection"
            )
        declared_model = model_binding.declared_contract.model
        if not isinstance(declared_model, ModelContract):
            raise ComponentRuntimeBindingError(
                "resolved model descriptor has no ModelContract"
            )
        algorithm_binding = bind_algorithm_execution_plan(
            materialized,
            declared_model,
            model_execution_numerics,
        )
        algorithm_execution_plan = algorithm_binding.execution_plan
        reference_policy_state_evidence = (
            algorithm_binding.reference_policy_state_evidence
        )

        preprocess_binding = resolve_preprocess_runtime_binding(
            materialized=materialized,
            bindings=dict(bindings),
            model_binding=model_binding,
            manager=request.prepared.manager,
            handle=request.prepared.handle,
            runtime_facts=runtime_facts,
            model_runtime_contract=probe_result.model_runtime_contract,
            algorithm=algorithm_execution_plan,
            identity_provider=self._preprocess_identity_provider,
        )
        requirement_set = preprocess_binding.requirement_set
        preprocess_result = preprocess_binding.identity_result
        preprocess_identity = preprocess_result.preprocess_identity
        _digest("preprocess identity", preprocess_identity)
        requirement_set_id = requirement_set.requirement_set_id

        bound_reward_ids = container.bound_reward_resource_ids
        runtime_bound_contracts = self._runtime_bound_contracts(
            request,
            bindings=bindings,
            logical_rewards=logical_rewards,
            bound_reward_ids=bound_reward_ids,
            probe_result=probe_result,
            policy_tensor_runtime_spec=policy_tensor_runtime_spec,
            model_execution_numerics=model_execution_numerics,
            preprocess_identity=preprocess_identity,
            preprocess_requirement_set_id=requirement_set_id,
            algorithm_execution_plan=algorithm_execution_plan,
        )
        verified_fields = self._verified_fields(
            logical_rewards=logical_rewards,
            bound_reward_ids=bound_reward_ids,
            probe_result=probe_result,
            policy_tensor_runtime_spec=policy_tensor_runtime_spec,
            model_execution_numerics=model_execution_numerics,
            reference_policy_state_evidence=(reference_policy_state_evidence),
            preprocess_identity=preprocess_identity,
            requirement_set=requirement_set,
            graph=request.graph,
            algorithm_execution_plan=algorithm_execution_plan,
        )
        return ComponentRuntimeEvidence(
            runtime_bound_contracts=runtime_bound_contracts,
            verified_fields=verified_fields,
            model_runtime_contract=probe_result.model_runtime_contract,
            policy_tensor_runtime_spec=policy_tensor_runtime_spec,
            model_execution_numerics=model_execution_numerics,
            reference_policy_state_evidence=(reference_policy_state_evidence),
            preprocess_identity=preprocess_identity,
            preprocess_requirement_set_id=requirement_set_id,
            bound_reward_resource_ids=bound_reward_ids,
            peer_bound_contract_ids=(),
        )

    @staticmethod
    def _validate_request(request: ComponentBindRequest) -> None:
        from visual_rl.runtime.types import (
            ComponentBindRequest,
            ProductionGraph,
            ProductionPreflight,
            ProductionPreparedRun,
            ProductionRuntime,
            TransformExecution,
        )

        if not isinstance(request, ComponentBindRequest):
            raise TypeError("request must be ComponentBindRequest")
        if not isinstance(request.preflight, ProductionPreflight):
            raise TypeError("request.preflight must be ProductionPreflight")
        if not isinstance(request.runtime, ProductionRuntime):
            raise TypeError("request.runtime must be ProductionRuntime")
        if not isinstance(request.graph, ProductionGraph):
            raise TypeError("request.graph must be ProductionGraph")
        if not isinstance(request.prepared, ProductionPreparedRun):
            raise TypeError("request.prepared must be ProductionPreparedRun")
        if not isinstance(request.transforms, TransformExecution):
            raise TypeError("request.transforms must be TransformExecution")
        if request.runtime.preflight is not request.preflight:
            raise ComponentRuntimeBindingError(
                "runtime and bind request use different preflight objects"
            )
        if request.prepared.training != request.preflight.compiled.training:
            raise ComponentRuntimeBindingError(
                "prepared training semantics differ from compiled training"
            )

        materialized = request.preflight.environment.materialized
        facts = request.runtime.session.runtime_facts
        if (
            facts.distribution_mode != "single"
            or facts.rank != 0
            or facts.local_rank != 0
            or facts.world_size != 1
        ):
            raise ComponentRuntimeBindingError(
                "DefaultComponentRuntimeBinder is single-process only"
            )
        if request.prepared.training.gradient_accumulation_steps != 1:
            raise ComponentRuntimeBindingError(
                "DefaultComponentRuntimeBinder currently requires "
                "gradient_accumulation_steps=1"
            )
        if request.runtime.launch_binding.recipe_id != materialized.recipe_id:
            raise ComponentRuntimeBindingError(
                "runtime launch and MaterializedRecipe identities differ"
            )
        if request.runtime.session.peer_recipe_ids != (materialized.recipe_id,):
            raise ComponentRuntimeBindingError(
                "single-process runtime must contain only its local recipe id"
            )

    @staticmethod
    def _validate_graph_coverage(
        request: ComponentBindRequest,
        graph: RuntimeComponentGraph,
    ) -> dict[str, RuntimeComponentBinding]:
        if not isinstance(graph, RuntimeComponentGraph):
            raise TypeError("graph components must be RuntimeComponentGraph")
        materialized = request.preflight.environment.materialized
        if graph.artifact_binding_set is not (
            request.preflight.environment.component_artifact_bindings
        ):
            raise ComponentRuntimeBindingError(
                "runtime graph did not retain the exact environment G1 binding set"
            )
        if graph.load_plan is not request.preflight.environment.component_load_plan:
            raise ComponentRuntimeBindingError(
                "runtime graph did not retain the exact environment load plan"
            )
        resolved = materialized.resolved
        expected = {
            "algorithm": resolved.algorithm.component,
            resolved.model.slot: resolved.model.declaration,
            **{
                item.slot: item.declaration
                for item in (
                    *resolved.internal_components,
                    *resolved.reward_components,
                )
            },
        }
        actual_bindings = graph.bindings
        actual_slots = tuple(binding.slot for binding in actual_bindings)
        if len(actual_slots) != len(set(actual_slots)):
            raise ComponentRuntimeBindingError("runtime graph contains duplicate slots")
        if set(actual_slots) != set(expected):
            raise ComponentRuntimeBindingError(
                "runtime graph slots do not exactly cover MaterializedRecipe: "
                f"missing={sorted(set(expected) - set(actual_slots))}, "
                f"unknown={sorted(set(actual_slots) - set(expected))}"
            )

        result: dict[str, RuntimeComponentBinding] = {}
        for binding in actual_bindings:
            declaration = expected[binding.slot]
            if binding.declaration != declaration:
                raise ComponentRuntimeBindingError(
                    f"runtime graph declaration drifted for slot {binding.slot!r}"
                )
            expected_artifact = graph.artifact_binding_set.binding(binding.slot)
            if binding.artifact_binding is not expected_artifact:
                raise ComponentRuntimeBindingError(
                    f"runtime graph G1 receipt drifted for slot {binding.slot!r}"
                )
            result[binding.slot] = binding

        model_slots = tuple(
            slot for slot, binding in result.items() if binding.kind == "model"
        )
        if model_slots != ("model",):
            raise ComponentRuntimeBindingError(
                "runtime graph must contain exactly the declared model slot"
            )
        return result

    @staticmethod
    def _runtime_bound_contracts(
        request: ComponentBindRequest,
        *,
        bindings: Mapping[str, RuntimeComponentBinding],
        logical_rewards: tuple[RewardLogicalBinding, ...],
        bound_reward_ids: FrozenMapping,
        probe_result: ModelRuntimeProbeResult,
        policy_tensor_runtime_spec: PolicyTensorRuntimeSpec,
        model_execution_numerics: ModelExecutionNumericsEvidence,
        preprocess_identity: str,
        preprocess_requirement_set_id: str,
        algorithm_execution_plan: AlgorithmExecutionPlan,
    ) -> tuple[tuple[str, RuntimeBoundContract], ...]:
        _digest(
            "preprocess_requirement_set_id",
            preprocess_requirement_set_id,
        )
        prepared = request.prepared
        common_model_evidence = {
            "model_g1_binding_id": (
                probe_result.model_runtime_contract.component_binding_id
            ),
            "model_runtime_identity": (
                probe_result.model_runtime_contract.runtime_identity
            ),
            "model_verified_fields": to_plain_dict(probe_result.verified_fields),
            "policy_tensor_runtime_spec_id": policy_tensor_runtime_spec.spec_id,
            "policy_tensor_runtime_spec": policy_tensor_runtime_spec.to_payload(),
            "model_execution_numerics_id": (
                model_execution_numerics.execution_numerics_id
            ),
            "model_execution_numerics": model_execution_numerics.to_payload(),
            "trainable_topology_id": (
                prepared.manager.parameter_state.topology.identity
            ),
            "prepared_component_names": sorted(prepared.handle.component_names),
            "execution_transform_plan_id": request.transforms.plan_id,
            "resource_plan_id": prepared.manager.resource_plan.plan_id,
            "preprocess_identity": preprocess_identity,
            "runtime_compatibility": (
                request.runtime.session.runtime_facts.resume_compatibility_payload()
            ),
        }
        reward_by_slot = {item.graph_binding.slot: item for item in logical_rewards}
        result: list[tuple[str, RuntimeBoundContract]] = []
        for slot in sorted(bindings):
            binding = bindings[slot]
            if binding.kind == "model":
                contract = probe_result.model_runtime_contract
                if contract.artifact is not binding.artifact_binding:
                    raise ComponentRuntimeBindingError(
                        "model probe contract differs from the exact model G1 binding"
                    )
                result.append((slot, contract))
                continue
            if binding.kind == "reward":
                logical = reward_by_slot[slot]
                spec_id = logical.plan_binding.resource_identity
                evidence: dict[str, object] = {
                    "slot": slot,
                    "logical_reward_id": logical.logical_id,
                    "reward_resource_spec_id": spec_id,
                    "bound_reward_resource_id": bound_reward_ids[spec_id],
                }
            elif binding.kind == "conditioner":
                evidence = {
                    "slot": slot,
                    "preprocess_identity": preprocess_identity,
                    "preprocess_requirement_set_id": (preprocess_requirement_set_id),
                    "model_runtime_identity": (
                        probe_result.model_runtime_contract.runtime_identity
                    ),
                    "policy_tensor_runtime_spec_id": (
                        policy_tensor_runtime_spec.spec_id
                    ),
                    "policy_tensor_runtime_spec": (
                        policy_tensor_runtime_spec.to_payload()
                    ),
                }
            elif binding.kind == "algorithm":
                evidence = {
                    "slot": slot,
                    **common_model_evidence,
                    "algorithm_execution_plan_id": algorithm_execution_plan.plan_id,
                    "algorithm_requirement_id": (
                        request.graph.algorithm_module.requirements.requirement_id
                    ),
                }
            else:
                evidence = {"slot": slot, **common_model_evidence}
                if binding.kind in {"credit", "rollout", "trainer"}:
                    evidence["preprocess_requirement_set_id"] = (
                        preprocess_requirement_set_id
                    )
            runtime_identity = _domain_hash(
                _COMPONENT_BOUND_ID_DOMAIN,
                {
                    "component_binding_id": binding.artifact_binding.binding_id,
                    "component_load_attestation_id": (
                        binding.load_attestation.attestation_id
                    ),
                    "relevant_runtime_evidence": evidence,
                },
            )
            verified_fields = tuple(
                sorted(
                    (
                        ("component.binding_id", binding.artifact_binding.binding_id),
                        (
                            "component.declaration_id",
                            binding.declaration.declaration_id,
                        ),
                        ("component.kind", binding.kind),
                        (
                            "component.load_attestation_id",
                            binding.load_attestation.attestation_id,
                        ),
                        ("component.runtime_evidence", canonical_json_text(evidence)),
                        ("component.slot", slot),
                    )
                )
            )
            contract = binding.attest_prepare(
                runtime_identity=runtime_identity,
                verified_fields=verified_fields,
            )
            result.append((slot, contract))
        return tuple(result)

    @staticmethod
    def _verified_fields(
        *,
        logical_rewards: tuple[RewardLogicalBinding, ...],
        bound_reward_ids: FrozenMapping,
        probe_result: ModelRuntimeProbeResult,
        policy_tensor_runtime_spec: PolicyTensorRuntimeSpec,
        model_execution_numerics: ModelExecutionNumericsEvidence,
        reference_policy_state_evidence: ReferencePolicyStateEvidence,
        preprocess_identity: str,
        requirement_set: PreprocessRequirementSet,
        graph: ProductionGraph,
        algorithm_execution_plan: AlgorithmExecutionPlan,
    ) -> FrozenMapping:
        from visual_rl.runtime.reward_resources import RewardResourceState

        if not isinstance(requirement_set, PreprocessRequirementSet):
            raise TypeError("requirement_set must be PreprocessRequirementSet")
        reward_evidence: dict[str, object] = {}
        for logical in logical_rewards:
            spec_id = logical.plan_binding.resource_identity
            bound_id = bound_reward_ids[spec_id]
            _digest(f"bound reward resource id for {logical.logical_id}", bound_id)
            reward_evidence[logical.logical_id] = {
                "reward_resource_spec_id": spec_id,
                "bound_reward_resource_id": bound_id,
                "state": RewardResourceState.ACQUIRED.value,
            }
        return FrozenMapping(
            {
                "model_probe": {
                    "runtime_identity": (
                        probe_result.model_runtime_contract.runtime_identity
                    ),
                    "component_binding_id": (
                        probe_result.model_runtime_contract.component_binding_id
                    ),
                    "verified_fields": to_plain_dict(probe_result.verified_fields),
                },
                "preprocess": {
                    "identity": preprocess_identity,
                    "requirement_set_id": requirement_set.requirement_set_id,
                    "requirement_set": requirement_set.to_payload(),
                },
                "policy_tensor_runtime": {
                    "spec_id": policy_tensor_runtime_spec.spec_id,
                    "spec": policy_tensor_runtime_spec.to_payload(),
                },
                "model_execution_numerics": (model_execution_numerics.to_payload()),
                "reference_policy_state": (
                    reference_policy_state_evidence.to_payload()
                ),
                "algorithm_module": {
                    "resolved_declaration": (
                        graph.components.binding(
                            "algorithm"
                        ).declaration.to_identity_payload()
                    ),
                    "requirement_id": (
                        graph.algorithm_module.requirements.requirement_id
                    ),
                    "execution_plan_id": algorithm_execution_plan.plan_id,
                    "execution_plan": algorithm_execution_plan.to_payload(),
                },
                "reward_bindings": reward_evidence,
            }
        )
