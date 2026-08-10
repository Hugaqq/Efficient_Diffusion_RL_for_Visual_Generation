"""Typed model-runtime G3 probe contracts and tensor compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from visual_rl.algorithms.dynamics.session import DynamicsSession
from visual_rl.composition.compatibility import bind_model_algorithm
from visual_rl.composition.preflight.types import RuntimeFacts
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.core.contracts.algorithm import AlgorithmRequirements
from visual_rl.core.contracts.composition import (
    BoundPolicyCapabilities,
    ModelAlgorithmBinding,
)
from visual_rl.core.contracts import (
    ComputePrecision,
    DeclaredContract,
    DynamicsContract,
    RuntimeBoundContract,
)
from visual_rl.core.contracts.runtime import (
    PolicyTransitionRequest,
    PolicyTransitionResult,
)
from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data.media import DecodedMediaBatch
from visual_rl.models import ModelAdapter
from visual_rl.models.lifecycle.components import ComponentManager, ExecutionMode
from visual_rl.models.lifecycle.prepared import PreparedComponentHandle
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.models.numerics.policy import (
    ForwardAutocastPolicy,
    ModelExecutionNumericsEvidence,
    ParameterViewMode,
)
from visual_rl.models.numerics.runtime import ModelRuntimeNumerics
from visual_rl.runtime.component_graph import (
    ComponentRuntimeBindingError,
    RuntimeComponentBinding,
)
from visual_rl.runtime.types import (
    PolicyTensorRuntimeSpec,
    ProductionPreparedRun,
)

__all__ = (
    "DefaultPolicyRuntimePort",
    "ModelRuntimeProbe",
    "ModelRuntimeProbeRequest",
    "ModelRuntimeProbeResult",
    "resolve_policy_tensor_runtime_spec",
    "resolve_model_execution_numerics",
    "resolve_transition_dtype",
    "validate_model_runtime_contract",
    "validate_prepared_model",
)


def _canonical_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ModelRuntimeProbeRequest:
    """Narrow leaf input for observing a prepared model runtime."""

    materialized: MaterializedRecipe
    model_binding: RuntimeComponentBinding
    manager: ComponentManager
    handle: PreparedComponentHandle
    runtime_facts: RuntimeFacts

    def __post_init__(self) -> None:
        if not isinstance(self.materialized, MaterializedRecipe):
            raise TypeError("materialized must be a MaterializedRecipe")
        if not isinstance(self.model_binding, RuntimeComponentBinding):
            raise TypeError("model_binding must be RuntimeComponentBinding")
        if self.model_binding.slot != "model" or self.model_binding.kind != "model":
            raise ValueError("model_binding must be the declared model slot")
        if not isinstance(self.manager, ComponentManager):
            raise TypeError("manager must be ComponentManager")
        if not isinstance(self.handle, PreparedComponentHandle):
            raise TypeError("handle must be PreparedComponentHandle")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if self.manager.adapter is not self.model_binding.instance:
            raise ComponentRuntimeBindingError(
                "prepared manager adapter differs from the graph model"
            )
        if self.manager.prepared_handle is not self.handle:
            raise ComponentRuntimeBindingError(
                "prepared handle differs from the manager's unique handle"
            )


@dataclass(frozen=True, slots=True)
class ModelRuntimeProbeResult:
    """Observed model contract plus the exact probe fields which justify it."""

    model_runtime_contract: RuntimeBoundContract
    runtime_numerics: ModelRuntimeNumerics
    verified_fields: FrozenMapping

    def __post_init__(self) -> None:
        if not isinstance(self.model_runtime_contract, RuntimeBoundContract):
            raise TypeError("model_runtime_contract must be RuntimeBoundContract")
        if not isinstance(self.runtime_numerics, ModelRuntimeNumerics):
            raise TypeError("runtime_numerics must be ModelRuntimeNumerics")
        if (
            not isinstance(self.verified_fields, FrozenMapping)
            or not self.verified_fields
        ):
            raise ValueError("model probe verified_fields must be non-empty")
        for name, value in self.verified_fields.items():
            _canonical_text("model probe field name", name)
            _canonical_text(f"model probe field {name!r}", value)

        contract_fields = self.model_runtime_contract.verified_fields
        if not contract_fields:
            raise ValueError("model runtime contract must contain verified fields")
        for item in contract_fields:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("runtime contract verified fields must be string pairs")
            _canonical_text("runtime contract verified field name", item[0])
            _canonical_text(f"runtime contract verified field {item[0]!r}", item[1])
        if dict(contract_fields) != to_plain_dict(self.verified_fields):
            raise ValueError(
                "model probe evidence differs from RuntimeBoundContract verified fields"
            )
        expected_numerics_payload = canonical_json_text(
            self.runtime_numerics.to_payload()
        )
        if (
            self.verified_fields.get("model.runtime_numerics")
            != expected_numerics_payload
        ):
            raise ValueError(
                "model.runtime_numerics evidence must exactly match the typed "
                "runtime numerics payload"
            )
        if (
            self.verified_fields.get("model.runtime_numerics_id")
            != self.runtime_numerics.runtime_numerics_id
        ):
            raise ValueError(
                "model.runtime_numerics_id evidence must exactly match the typed "
                "runtime numerics identity"
            )


def resolve_transition_dtype(
    runtime_numerics: ModelRuntimeNumerics,
    dynamics_contract: DynamicsContract,
) -> str:
    """Resolve the unique typed model/Dynamics dtype intersection."""

    if not isinstance(runtime_numerics, ModelRuntimeNumerics):
        raise TypeError("runtime_numerics must be ModelRuntimeNumerics")
    if not isinstance(dynamics_contract, DynamicsContract):
        raise TypeError("dynamics_contract must be DynamicsContract")
    intersection = tuple(
        dtype
        for dtype in dynamics_contract.accepted_transition_dtypes
        if dtype == runtime_numerics.transition_latent_dtype
    )
    if not intersection:
        raise ComponentRuntimeBindingError(
            "model transition latent dtype has no intersection with the "
            "Dynamics accepted transition dtypes"
        )
    if len(intersection) != 1:
        raise ComponentRuntimeBindingError(
            "model/Dynamics transition dtype intersection is not unique"
        )
    return intersection[0]


@runtime_checkable
class ModelRuntimeProbe(Protocol):
    """Observe actual prepared-model ports and return one runtime contract."""

    def probe(self, request: ModelRuntimeProbeRequest) -> ModelRuntimeProbeResult: ...


def validate_prepared_model(
    prepared: ProductionPreparedRun,
    model_binding: RuntimeComponentBinding,
) -> None:
    """Prove the prepared model root still owns the exact G1 model instance."""

    if not isinstance(prepared, ProductionPreparedRun):
        raise TypeError("prepared must be ProductionPreparedRun")
    if not isinstance(model_binding, RuntimeComponentBinding):
        raise TypeError("model_binding must be RuntimeComponentBinding")
    if prepared.manager.adapter is not model_binding.instance:
        raise ComponentRuntimeBindingError(
            "G3 prepared owner does not retain the exact graph model instance"
        )
    if prepared.manager.prepared_handle is not prepared.handle:
        raise ComponentRuntimeBindingError(
            "G3 prepared handle is not the manager's unique prepared root"
        )
    if prepared.handle.optimizer is not prepared.optimizer:
        raise ComponentRuntimeBindingError("prepared optimizer ownership drifted")
    if prepared.handle.scheduler is not prepared.lr_scheduler:
        raise ComponentRuntimeBindingError("prepared scheduler ownership drifted")
    _digest(
        "trainable topology identity",
        prepared.manager.parameter_state.topology.identity,
    )
    prepared.manager.parameter_dtype_owner.validate_applied()
    if (
        prepared.manager.model_execution_numerics.source_projection_id
        != prepared.manager.parameter_state.state_projection.projection_id
    ):
        raise ComponentRuntimeBindingError(
            "prepared execution numerics use a stale state projection"
        )
    names = prepared.handle.component_names
    if not names or len(names) != len(set(names)):
        raise ComponentRuntimeBindingError(
            "prepared component names must be non-empty and unique"
        )


def validate_model_runtime_contract(
    result: ModelRuntimeProbeResult,
    *,
    model_binding: RuntimeComponentBinding,
) -> None:
    """Ensure the model probe continues the exact model G1/load receipts."""

    if not isinstance(result, ModelRuntimeProbeResult):
        raise TypeError("result must be ModelRuntimeProbeResult")
    declared = model_binding.declared_contract
    if not isinstance(declared, DeclaredContract) or (
        declared.component_kind != "model"
    ):
        raise ComponentRuntimeBindingError(
            "resolved model descriptor is not a DeclaredContract"
        )
    contract = result.model_runtime_contract
    if contract.artifact is not model_binding.artifact_binding:
        raise ComponentRuntimeBindingError(
            "model probe contract does not retain the exact model G1 binding"
        )
    if contract.component_load_attestation is not model_binding.load_attestation:
        raise ComponentRuntimeBindingError(
            "model probe contract does not retain the exact model load receipt"
        )
    _digest("model runtime identity", contract.runtime_identity)


def resolve_policy_tensor_runtime_spec(
    *,
    dynamics_binding: RuntimeComponentBinding,
    runtime_facts: RuntimeFacts,
    runtime_numerics: ModelRuntimeNumerics,
) -> PolicyTensorRuntimeSpec:
    """Resolve the prepared model/Dynamics tensor ABI for rollout and update."""

    if not isinstance(dynamics_binding, RuntimeComponentBinding) or (
        dynamics_binding.kind != "dynamics"
    ):
        raise ComponentRuntimeBindingError(
            "runtime graph dynamics slot is not a dynamics binding"
        )
    declared = dynamics_binding.declared_contract
    if not isinstance(declared, DeclaredContract) or (
        declared.component_kind != "dynamics"
    ):
        raise ComponentRuntimeBindingError(
            "resolved dynamics descriptor is not a DeclaredContract"
        )
    dynamics_contract = declared.dynamics
    if not isinstance(dynamics_contract, DynamicsContract):
        raise ComponentRuntimeBindingError(
            "resolved dynamics descriptor has no DynamicsContract"
        )
    latent_storage_dtype = resolve_transition_dtype(
        runtime_numerics,
        dynamics_contract,
    )
    return PolicyTensorRuntimeSpec(
        device=runtime_facts.device,
        latent_storage_dtype=latent_storage_dtype,
        model_compute_precision=runtime_facts.precision,
    )


def resolve_model_execution_numerics(
    manager: ComponentManager,
    runtime_facts: RuntimeFacts,
) -> ModelExecutionNumericsEvidence:
    """Bind the supported single-process LoRA parameter views and autocast policy."""

    if not isinstance(manager, ComponentManager):
        raise TypeError("manager must be ComponentManager")
    if not isinstance(runtime_facts, RuntimeFacts):
        raise TypeError("runtime_facts must be RuntimeFacts")
    if runtime_facts.distribution_mode != "single":
        raise ComponentRuntimeBindingError(
            "model execution numerics supports only single-process LoRA"
        )
    import torch

    device_type = torch.device(runtime_facts.device).type
    precision = ComputePrecision(runtime_facts.precision)
    compute_dtype = {
        ComputePrecision.FP32: "float32",
        ComputePrecision.FP16: "float16",
        ComputePrecision.BF16: "bfloat16",
    }[precision]
    enabled = precision is not ComputePrecision.FP32
    views = manager.adapter.describe_parameter_view_evidence(
        manager.parameter_state,
        distribution_mode=runtime_facts.distribution_mode,
    )
    view_by_kind = {item.parameter_view: item for item in views}
    current = view_by_kind.get(ParameterView.CURRENT)
    if current is None or current.mode is not ParameterViewMode.CURRENT:
        raise ComponentRuntimeBindingError(
            "model requires non-mutating current LoRA parameter view evidence"
        )
    if current.mutates_parameters_in_place:
        raise ComponentRuntimeBindingError(
            "current parameter view cannot mutate in place"
        )
    reference = view_by_kind.get(ParameterView.REFERENCE)
    if reference is not None and (
        reference.mode is not ParameterViewMode.LORA_DISABLE
        or reference.mutates_parameters_in_place
    ):
        raise ComponentRuntimeBindingError(
            "reference parameters support only non-mutating LoRA disable"
        )
    if ParameterView.EMA in view_by_kind:
        raise ComponentRuntimeBindingError("EMA parameter view is not supported")
    if len(view_by_kind) != len(views):
        raise ComponentRuntimeBindingError("model parameter views must be unique")

    policies = [
        ForwardAutocastPolicy(
            stage=stage,
            parameter_view=ParameterView.CURRENT,
            device_type=device_type,
            compute_dtype=compute_dtype,
            enabled=enabled,
        )
        for stage in (
            ExecutionMode.ROLLOUT,
            ExecutionMode.TRAIN,
            ExecutionMode.EVAL,
        )
    ]
    if reference is not None:
        policies.append(
            ForwardAutocastPolicy(
                stage=ExecutionMode.TRAIN,
                parameter_view=ParameterView.REFERENCE,
                device_type=device_type,
                compute_dtype=compute_dtype,
                enabled=enabled,
            )
        )
    return ModelExecutionNumericsEvidence(
        parameter_dtype_policy=manager.parameter_dtype_owner.policy,
        forward_autocast_policies=tuple(policies),
        parameter_view_evidence=views,
    )


@dataclass(frozen=True, slots=True)
class DefaultPolicyRuntimePort:
    """One immutable, capability-checked facade over a prepared model root."""

    _adapter: ModelAdapter = field(repr=False, compare=False)
    _manager: ComponentManager = field(repr=False, compare=False)
    _prepared_handle: PreparedComponentHandle = field(repr=False, compare=False)
    capabilities: BoundPolicyCapabilities
    algorithm_requirements: AlgorithmRequirements
    runtime_capabilities: object
    _binding: ModelAlgorithmBinding = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._adapter, ModelAdapter):
            raise TypeError("_adapter must be ModelAdapter")
        if not isinstance(self._manager, ComponentManager):
            raise TypeError("_manager must be ComponentManager")
        if not isinstance(self._prepared_handle, PreparedComponentHandle):
            raise TypeError("_prepared_handle must be PreparedComponentHandle")
        if self._manager.adapter is not self._adapter:
            raise ValueError("policy port manager does not own the supplied adapter")
        if self._manager.prepared_handle is not self._prepared_handle:
            raise ValueError("policy port handle is not the manager's prepared root")
        if not isinstance(self.capabilities, BoundPolicyCapabilities):
            raise TypeError("capabilities must be BoundPolicyCapabilities")
        if not isinstance(self.algorithm_requirements, AlgorithmRequirements):
            raise TypeError("algorithm_requirements must be AlgorithmRequirements")
        object.__setattr__(
            self,
            "_binding",
            bind_model_algorithm(self.capabilities, self.algorithm_requirements),
        )

    @property
    def binding(self) -> ModelAlgorithmBinding:
        return self._binding

    @property
    def trainable_parameters(self) -> tuple[object, ...]:
        return tuple(self._manager.parameter_state.parameters())

    @property
    def prepared_forward_handle(self) -> object:
        return self._prepared_handle

    @property
    def state_contract(self) -> object:
        return self._manager.parameter_state.state_projection

    def preprocess(self, raw_batch: object) -> object:
        return self._adapter.encode(raw_batch)

    def encode(self, batch: object) -> object:
        return self.preprocess(batch)

    def initialize_latents(self, batch_geometry: object, rng: object) -> object:
        return self._adapter.prepare_latents(batch_geometry, generator=rng)

    def prepare_latents(
        self,
        latent_spec: object,
        *,
        generator: object,
    ) -> object:
        return self.initialize_latents(latent_spec, generator)

    def latent_spec_for_batch(
        self,
        batch: object,
        *,
        device: object,
        dtype: object,
    ) -> object:
        return self._adapter.latent_spec_for_batch(
            batch,
            device=device,
            dtype=dtype,
        )

    def model_schedule_context(self, latent_spec: object) -> object:
        return self._adapter.model_schedule_context(latent_spec)

    def predict(
        self,
        model_input: object,
        parameter_view: object | None = None,
    ) -> object:
        view = (
            ParameterView.CURRENT
            if parameter_view is None
            else ParameterView(parameter_view)
        )
        if view is ParameterView.CURRENT:
            return self._adapter.predict(model_input)
        if view is ParameterView.REFERENCE:
            return self._adapter.predict_reference(model_input)
        raise ValueError("the Phase-A policy port supports current/reference views")

    def predict_reference(self, model_input: object) -> object:
        return self.predict(model_input, ParameterView.REFERENCE)

    def transition(self, request: PolicyTransitionRequest) -> PolicyTransitionResult:
        if not isinstance(request, PolicyTransitionRequest):
            raise TypeError("request must be PolicyTransitionRequest")
        session = request.transition_session
        if not isinstance(session, DynamicsSession):
            raise TypeError("transition_session must be a DynamicsSession")
        if request.mode == "sample":
            result = session.sample_transition(
                request.transition_input,
                generator=request.generator,
            )
            return PolicyTransitionResult(
                next_latents=result.sampled_next,
                log_prob=result.log_prob,
                transition_output=result,
                replay_state=session.snapshot,
            )
        evaluation = session.evaluate_transition(
            request.transition_input,
            request.action_latent,
        )
        metadata = session.policy_metadata(
            request.transition_input,
            evaluation.stats,
        )
        return PolicyTransitionResult(
            next_latents=request.action_latent,
            log_prob=evaluation.log_prob,
            transition_output=evaluation,
            policy_metadata=metadata,
            replay_state=session.snapshot,
        )

    def decode(
        self,
        latents: object,
        latent_spec: object,
    ) -> DecodedMediaBatch:
        decoded = self._adapter.decode(latents, latent_spec)
        if not isinstance(decoded, DecodedMediaBatch):
            raise TypeError("model adapter decode() must return DecodedMediaBatch")
        return decoded
