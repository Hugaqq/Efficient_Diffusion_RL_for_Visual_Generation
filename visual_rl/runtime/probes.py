"""Default non-executing G3 probes for one prepared model adapter.

The probes in this module observe public, already-prepared runtime state.  They
never create model components, encode samples, materialize latents, or execute
a model forward.  A successful result therefore proves port availability and
prepared ownership/topology, not numerical correctness of an unseen forward.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.contracts import (
    ComputePrecision,
    DeclaredContract,
)
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data.preprocess import PreprocessProducerSpec
from visual_rl.data.preprocess_factory import (
    InlinePreprocessPlanFactory,
    InlinePreprocessPlanRequest,
)
from visual_rl.models.interface import ModelAdapter
from visual_rl.models.lifecycle.components import (
    ComponentRole,
    ExecutionMode,
    OwnershipState,
    Residency,
)
from visual_rl.models.numerics.runtime import ModelRuntimeNumerics
from visual_rl.models.scheduler import SchedulerArtifactBlueprint
from visual_rl.runtime.component_graph import ComponentRuntimeBindingError
from visual_rl.runtime.model_binding import (
    ModelRuntimeProbeRequest,
    ModelRuntimeProbeResult,
)
from visual_rl.runtime.preprocess_binding import (
    PreprocessIdentityRequest,
    PreprocessIdentityResult,
)

__all__ = (
    "DefaultModelRuntimeProbe",
    "DefaultPreprocessIdentityProvider",
)


_RUNTIME_ID_DOMAIN = b"visual_rl.default-model-runtime-probe.v1\0"
_SHARDABLE_ROLES = frozenset(
    {
        ComponentRole.INFERENCE,
        ComponentRole.TRAINABLE,
        ComponentRole.REFERENCE,
    }
)


def _digest(payload: Mapping[str, object]) -> str:
    encoded = canonical_json_text(payload).encode("utf-8")
    return hashlib.sha256(_RUNTIME_ID_DOMAIN + encoded).hexdigest()


def _payload_digest(payload: object) -> str:
    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()


def _canonical_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise ComponentRuntimeBindingError(
            f"{name} must be a canonical non-empty string"
        )
    return value


def _type_path(value: type[object]) -> str:
    return _canonical_text(
        "runtime type path",
        f"{value.__module__}:{value.__qualname__}",
    )


def _method_path(
    adapter: ModelAdapter,
    name: str,
    *,
    base_implementation: object,
    allow_base: bool = False,
) -> str:
    implementation = getattr(type(adapter), name, None)
    if not callable(implementation) or (
        implementation is base_implementation and not allow_base
    ):
        raise ComponentRuntimeBindingError(
            f"prepared model does not implement the typed {name} port"
        )
    return _canonical_text(
        f"model {name} implementation",
        f"{implementation.__module__}:{implementation.__qualname__}",
    )


def _validate_request_ownership(
    request: ModelRuntimeProbeRequest | PreprocessIdentityRequest,
) -> ModelAdapter:
    adapter = request.model_binding.instance
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("prepared model binding must contain a ModelAdapter")
    if request.manager.adapter is not adapter:
        raise ComponentRuntimeBindingError(
            "prepared manager adapter differs from the model binding"
        )
    if request.manager.state is not OwnershipState.PREPARED:
        raise ComponentRuntimeBindingError(
            "default runtime probing requires PREPARED component ownership"
        )
    if request.manager.mode is not ExecutionMode.IDLE:
        raise ComponentRuntimeBindingError(
            "default runtime probing requires an idle ComponentManager"
        )
    if request.manager.prepared_handle is not request.handle:
        raise ComponentRuntimeBindingError(
            "prepared handle differs from the manager's unique handle"
        )
    if adapter.prepared_components is not request.handle:
        raise ComponentRuntimeBindingError(
            "adapter prepared-component port differs from the manager handle"
        )
    if request.model_binding.config is not getattr(
        adapter,
        "config",
        None,
    ):
        raise ComponentRuntimeBindingError(
            "prepared adapter did not retain the exact resolved model config"
        )
    adapter_path = _type_path(type(adapter))
    if adapter_path != request.model_binding.declaration.implementation_class_path:
        raise ComponentRuntimeBindingError(
            "prepared adapter class differs from the resolved model manifest"
        )
    return adapter


def _validate_declared_contract(
    request: ModelRuntimeProbeRequest,
    adapter: ModelAdapter,
) -> DeclaredContract:
    declared = request.model_binding.declared_contract
    if not isinstance(declared, DeclaredContract) or declared.component_kind != "model":
        raise ComponentRuntimeBindingError(
            "resolved model descriptor is not a model DeclaredContract"
        )
    observed = type(adapter).describe(request.model_binding.config)
    if not isinstance(observed, DeclaredContract) or observed != declared:
        raise ComponentRuntimeBindingError(
            "prepared adapter declaration differs from the resolved model contract"
        )
    if declared.model is None:
        raise ComponentRuntimeBindingError("resolved model contract has no model body")
    return declared


def _component_inventory(
    request: ModelRuntimeProbeRequest,
) -> tuple[dict[str, object], ...]:
    manager = request.manager
    components = manager.components
    prepared_names = request.handle.component_names
    if not prepared_names or len(prepared_names) != len(set(prepared_names)):
        raise ComponentRuntimeBindingError(
            "prepared component names must be non-empty and unique"
        )
    expected_prepared = tuple(
        binding.name
        for binding in components.bindings
        if any(role in _SHARDABLE_ROLES for role in binding.roles)
    )
    if prepared_names != expected_prepared:
        raise ComponentRuntimeBindingError(
            "prepared component names differ from the manager-owned shardable roots"
        )
    if any(not request.handle.owns(name) for name in prepared_names):
        raise ComponentRuntimeBindingError(
            "prepared handle does not own every declared prepared component"
        )
    if request.handle.accumulation_root is not request.handle.prepared_root:
        raise ComponentRuntimeBindingError(
            "prepared accumulation root differs from the unique prepared root"
        )

    prepared_set = frozenset(prepared_names)
    inventory: list[dict[str, object]] = []
    for binding in components.bindings:
        residency = manager.residency(binding.name)
        owner = manager.resource_plan.owner_for(
            binding,
            prepared_component_names=prepared_set,
        )
        if binding.name in prepared_set and residency is not Residency.PREPARED:
            raise ComponentRuntimeBindingError(
                f"prepared component {binding.name!r} lost backend residency"
            )
        if binding.name not in prepared_set:
            expected_residency = (
                Residency.OFFLOADED if binding.managed_residency else Residency.STATIC
            )
            if residency is not expected_residency:
                raise ComponentRuntimeBindingError(
                    f"unprepared component {binding.name!r} has unexpected residency"
                )
        inventory.append(
            {
                "name": binding.name,
                "roles": [role.value for role in binding.roles],
                "managed_residency": binding.managed_residency,
                "owner": owner.value,
                "residency": residency.value,
            }
        )
    return tuple(inventory)


def _validate_parameter_topology(
    request: ModelRuntimeProbeRequest,
) -> tuple[str, tuple[str, ...], int]:
    import torch

    manager = request.manager
    named = manager.parameter_state.named_trainable_parameters()
    topology = manager.parameter_state.topology
    if not named or len(named) != len(topology.entries):
        raise ComponentRuntimeBindingError(
            "prepared trainable parameters differ from the frozen topology"
        )
    prepared_names = frozenset(request.handle.component_names)
    devices: list[str] = []
    for item, entry in zip(named, topology.entries):
        if item.name != entry.name or item.component_name != entry.component_name:
            raise ComponentRuntimeBindingError(
                "prepared trainable parameter paths differ from their topology"
            )
        if item.component_name not in prepared_names:
            raise ComponentRuntimeBindingError(
                "trainable parameter is not owned by the prepared root"
            )
        parameter_device = torch.device(item.parameter.device)
        execution_device = manager.execution_device
        device_matches = parameter_device.type == execution_device.type and (
            execution_device.index is None
            or parameter_device.index == execution_device.index
        )
        if not device_matches:
            raise ComponentRuntimeBindingError(
                "trainable parameter device differs from manager execution_device"
            )
        devices.append(str(parameter_device))
    return topology.identity, tuple(sorted(set(devices))), topology.total_numel


def _validate_runtime(
    request: ModelRuntimeProbeRequest,
    adapter: ModelAdapter,
    declared: DeclaredContract,
) -> ComputePrecision:
    import torch

    facts = request.runtime_facts
    try:
        facts_device = torch.device(facts.device)
    except (TypeError, RuntimeError):
        raise ComponentRuntimeBindingError(
            "RuntimeFacts device is not torch.device-compatible"
        ) from None
    if facts_device != request.manager.execution_device:
        raise ComponentRuntimeBindingError(
            "RuntimeFacts device differs from manager execution_device"
        )
    try:
        precision = ComputePrecision(facts.precision)
    except (TypeError, ValueError):
        raise ComponentRuntimeBindingError(
            "RuntimeFacts precision is not a supported ComputePrecision"
        ) from None
    model = declared.model
    assert model is not None
    if precision not in model.supported_precisions:
        raise ComponentRuntimeBindingError(
            "runtime precision is not supported by the resolved model contract"
        )
    adapter_precision = getattr(adapter, "precision", None)
    if not isinstance(adapter_precision, ComputePrecision):
        raise ComponentRuntimeBindingError(
            "prepared adapter exposes no typed runtime precision evidence"
        )
    if adapter_precision is not precision:
        raise ComponentRuntimeBindingError(
            "prepared adapter precision differs from RuntimeFacts"
        )
    return precision


def _validate_ports(
    adapter: ModelAdapter,
    declared: DeclaredContract,
) -> tuple[dict[str, str], ModelRuntimeNumerics]:
    latent_port = _method_path(
        adapter,
        "latent_spec_for_batch",
        base_implementation=ModelAdapter.latent_spec_for_batch,
    )
    numerics_port = _method_path(
        adapter,
        "describe_runtime_numerics",
        base_implementation=ModelAdapter.describe_runtime_numerics,
    )
    schedule_context_port = _method_path(
        adapter,
        "model_schedule_context",
        base_implementation=ModelAdapter.model_schedule_context,
        allow_base=True,
    )
    numerics = adapter.describe_runtime_numerics()
    if not isinstance(numerics, ModelRuntimeNumerics):
        raise TypeError("describe_runtime_numerics() must return ModelRuntimeNumerics")
    numerics_payload = numerics.to_payload()
    if numerics.runtime_numerics_id != _payload_digest(numerics_payload):
        raise ComponentRuntimeBindingError(
            "model runtime numerics id differs from its canonical payload"
        )
    import torch

    if not isinstance(numerics.rollout_torch_dtype, torch.dtype) or not isinstance(
        numerics.transition_torch_dtype,
        torch.dtype,
    ):
        raise ComponentRuntimeBindingError(
            "model runtime numerics did not resolve typed torch dtypes"
        )

    blueprint_descriptor = getattr(
        type(adapter),
        "scheduler_artifact_blueprint",
        None,
    )
    if blueprint_descriptor is ModelAdapter.scheduler_artifact_blueprint:
        raise ComponentRuntimeBindingError(
            "prepared model does not implement scheduler artifact metadata"
        )
    blueprint = adapter.scheduler_artifact_blueprint
    if not isinstance(blueprint, SchedulerArtifactBlueprint):
        raise ComponentRuntimeBindingError(
            "prepared model scheduler blueprint violates its typed port"
        )
    model = declared.model
    assert model is not None
    if not model.declares_scheduler_binding:
        raise ComponentRuntimeBindingError(
            "resolved model does not declare the scheduler/Dynamics binding ABI"
        )
    if blueprint.schema_id != model.scheduler_blueprint_schema:
        raise ComponentRuntimeBindingError(
            "scheduler blueprint schema differs from the resolved model descriptor"
        )

    reference = model.provides_reference_policy
    if type(reference) is not bool:
        raise ComponentRuntimeBindingError(
            "model reference-policy availability remains unresolved at G3"
        )
    result = {
        "model.latent_spec_port": latent_port,
        "model.schedule_context_port": schedule_context_port,
        "model.runtime_numerics_port": numerics_port,
        "model.runtime_numerics_id": numerics.runtime_numerics_id,
        "model.runtime_numerics": canonical_json_text(numerics_payload),
        "model.scheduler_blueprint_schema": _canonical_text(
            "scheduler blueprint schema",
            blueprint.schema_id,
        ),
        "model.scheduler_blueprint_identity": _canonical_text(
            "scheduler blueprint identity",
            blueprint.blueprint_identity,
        ),
        "model.scheduler_artifact_identity": _canonical_text(
            "scheduler artifact identity",
            blueprint.artifact_identity,
        ),
        "model.scheduler_class_path": _canonical_text(
            "scheduler class path",
            blueprint.scheduler_class_path,
        ),
    }
    if reference:
        result["model.reference_port"] = _method_path(
            adapter,
            "predict_reference",
            base_implementation=ModelAdapter.predict_reference,
        )
        # ComponentManager.bind_runtime consumes this exact structural probe.
        # It means the typed implementation exists; no model forward ran here.
        result["model.reference_forward"] = "verified"
    else:
        result["model.reference_port"] = "not_declared"
        result["model.reference_forward"] = "not_required"
    return result, numerics


class DefaultModelRuntimeProbe:
    """Build exact G3 model evidence without executing model computation."""

    def probe(self, request: ModelRuntimeProbeRequest) -> ModelRuntimeProbeResult:
        if not isinstance(request, ModelRuntimeProbeRequest):
            raise TypeError("request must be ModelRuntimeProbeRequest")
        adapter = _validate_request_ownership(request)
        declared = _validate_declared_contract(request, adapter)
        inventory = _component_inventory(request)
        topology_id, parameter_devices, total_numel = _validate_parameter_topology(
            request
        )
        precision = _validate_runtime(request, adapter, declared)
        port_fields, numerics = _validate_ports(adapter, declared)

        manifest_payload = request.model_binding.declaration.to_identity_payload()
        manifest_digest = _payload_digest(manifest_payload)
        artifact_binding = request.model_binding.artifact_binding
        resource_plan_payload = request.manager.resource_plan.to_payload()
        resource_plan_id = request.manager.resource_plan.plan_id
        verified_payload = {
            "model.adapter_class": _type_path(type(adapter)),
            "model.prepared_root_type": _type_path(type(request.handle.prepared_root)),
            "model.component_inventory": canonical_json_text(inventory),
            "model.prepared_component_names": canonical_json_text(
                list(request.handle.component_names)
            ),
            "model.trainable_topology_id": topology_id,
            "model.trainable_parameter_devices": canonical_json_text(
                list(parameter_devices)
            ),
            "model.trainable_parameter_total_numel": str(total_numel),
            "model.resource_plan_id": resource_plan_id,
            "model.resource_plan": canonical_json_text(resource_plan_payload),
            "model.execution_device": str(request.manager.execution_device),
            "model.offload_device": str(request.manager.offload_device),
            "model.runtime_precision": precision.value,
            **port_fields,
        }
        verified_fields = FrozenMapping(verified_payload)
        runtime_identity = _digest(
            {
                "schema_version": 1,
                "component_artifact_binding_id": artifact_binding.binding_id,
                "resolved_model_manifest_sha256": manifest_digest,
                "verified_fields": to_plain_dict(verified_fields),
            }
        )
        contract = request.model_binding.attest_prepare(
            runtime_identity=runtime_identity,
            verified_fields=tuple(sorted(verified_fields.items())),
        )
        return ModelRuntimeProbeResult(
            model_runtime_contract=contract,
            runtime_numerics=numerics,
            verified_fields=verified_fields,
        )


class DefaultPreprocessIdentityProvider:
    """Identify adapter-declared inline preprocessing without live model work."""

    def __init__(self, factory: InlinePreprocessPlanFactory | None = None) -> None:
        if factory is not None and not isinstance(factory, InlinePreprocessPlanFactory):
            raise TypeError("factory must be InlinePreprocessPlanFactory or None")
        self._factory = InlinePreprocessPlanFactory() if factory is None else factory

    def identity_for(self, request: PreprocessIdentityRequest) -> str:
        """Compatibility inspection helper; production binding uses resolve()."""

        return self._resolution_for(request).plan.plan_id

    def resolve(self, request: PreprocessIdentityRequest) -> PreprocessIdentityResult:
        if not isinstance(request, PreprocessIdentityRequest):
            raise TypeError("request must be PreprocessIdentityRequest")
        requirement_set = request.requirement_set
        if requirement_set is None:
            raise ComponentRuntimeBindingError(
                "production preprocess resolution requires effective requirements"
            )
        resolution = self._resolution_for(request)
        receipt = resolution.compatibility_receipt
        if receipt is None or receipt.requirement_set_id != (
            requirement_set.requirement_set_id
        ):
            raise ComponentRuntimeBindingError(
                "preprocess compatibility receipt lost its effective "
                "requirement-set identity"
            )
        return PreprocessIdentityResult(
            preprocess_identity=resolution.plan.plan_id,
            requirement_set_id=requirement_set.requirement_set_id,
        )

    def _resolution_for(self, request: PreprocessIdentityRequest):
        if not isinstance(request, PreprocessIdentityRequest):
            raise TypeError("request must be PreprocessIdentityRequest")
        adapter = _validate_request_ownership(request)
        contract = request.model_runtime_contract
        declared = request.model_binding.declared_contract
        if contract.artifact is not request.model_binding.artifact_binding:
            raise ComponentRuntimeBindingError(
                "preprocess request contract differs from the exact model G1"
            )
        if contract.artifact.declared != declared:
            raise ComponentRuntimeBindingError(
                "preprocess request declaration differs from the resolved model"
            )
        spec = adapter.describe_preprocess()
        if not isinstance(spec, PreprocessProducerSpec):
            raise TypeError("describe_preprocess() must return PreprocessProducerSpec")
        return self._factory.resolve(
            InlinePreprocessPlanRequest(
                spec=spec,
                model_artifact_identity=(request.materialized.model_artifact_identity),
                resolved_model_manifest=FrozenMapping(
                    request.model_binding.declaration.to_identity_payload()
                ),
                requirements=request.requirement_set,
            )
        )
