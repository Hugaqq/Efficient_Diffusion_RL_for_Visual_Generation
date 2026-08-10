"""The sole runtime importer and constructor for resolved components."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from visual_rl.composition.registry import ResolvedComponentDeclaration
from visual_rl.core.contracts.composition import (
    ComponentArtifactBinding,
    ComponentArtifactBindingSet,
    ComponentLoadAttestation,
    ComponentLoadPlan,
    ComponentPrepareAttestation,
    RuntimeBoundContract,
)
from visual_rl.errors import ComponentError
from visual_rl.composition.preflight.types import RuntimeBindResult

__all__ = (
    "RuntimeComponentLoadError",
    "RuntimeComponentLoadGate",
    "RuntimeComponentLoadRequest",
    "RuntimeComponentLoadResult",
    "RuntimeComponentLoader",
    "RuntimeContextResolver",
    "RuntimeLoadIssueCode",
    "build_component_artifact_binding",
)


class RuntimeLoadIssueCode(str, Enum):
    DECLARATION_MISMATCH = "declaration_mismatch"
    LOAD_PLAN_MISMATCH = "load_plan_mismatch"
    IMPLEMENTATION_IMPORT_FAILED = "implementation_import_failed"
    IMPLEMENTATION_MISMATCH = "implementation_mismatch"
    INVALID_IMPLEMENTATION = "invalid_implementation"
    INTERFACE_IMPORT_FAILED = "interface_import_failed"
    INTERFACE_MISMATCH = "interface_mismatch"
    CONSTRUCTION_FAILED = "construction_failed"
    INVALID_INSTANCE = "invalid_instance"


class RuntimeComponentLoadError(ComponentError):
    """Structured failure after static, environment, and artifact gates."""

    def __init__(
        self,
        code: RuntimeLoadIssueCode,
        message: str,
        *,
        declaration: ResolvedComponentDeclaration,
    ) -> None:
        if not isinstance(code, RuntimeLoadIssueCode):
            raise TypeError("code must be a RuntimeLoadIssueCode")
        super().__init__(
            message,
            kind=declaration.kind,
            name=declaration.alias,
        )
        self.code = code.value
        self.implementation_class_path = declaration.implementation_class_path


@dataclass(frozen=True, slots=True)
class RuntimeComponentLoadGate:
    """Proof that environment and artifact binding preceded construction."""

    runtime_binding: RuntimeBindResult
    artifact_binding: ComponentArtifactBinding

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_binding, RuntimeBindResult):
            raise TypeError("runtime_binding must be a RuntimeBindResult")
        if not isinstance(self.artifact_binding, ComponentArtifactBinding):
            raise TypeError(
                "artifact_binding must be a declaration-bound ComponentArtifactBinding"
            )
        if self.artifact_binding.recipe_id != self.runtime_binding.recipe_id:
            raise ValueError(
                "artifact binding recipe differs from the validated runtime binding"
            )

    def validate(self, declaration: ResolvedComponentDeclaration) -> None:
        if not isinstance(declaration, ResolvedComponentDeclaration):
            raise TypeError("declaration must be a ResolvedComponentDeclaration")
        binding = self.artifact_binding
        if binding.component_declaration_id != declaration.declaration_id:
            raise RuntimeComponentLoadError(
                RuntimeLoadIssueCode.DECLARATION_MISMATCH,
                "G1 binding declaration identity differs from the selected "
                "component declaration",
                declaration=declaration,
            )
        if binding.declared != declaration.declared_contract:
            raise RuntimeComponentLoadError(
                RuntimeLoadIssueCode.DECLARATION_MISMATCH,
                "G1 binding contract differs from the selected declaration",
                declaration=declaration,
            )
        if binding.implementation_identity != declaration.implementation_class_path:
            raise RuntimeComponentLoadError(
                RuntimeLoadIssueCode.IMPLEMENTATION_MISMATCH,
                "G1 implementation identity differs from the selected declaration",
                declaration=declaration,
            )
        if binding.interface_version != declaration.descriptor.interface_version:
            raise RuntimeComponentLoadError(
                RuntimeLoadIssueCode.INTERFACE_MISMATCH,
                "G1 canonical interface version differs from the descriptor",
                declaration=declaration,
            )


@dataclass(frozen=True, slots=True)
class RuntimeComponentLoadRequest:
    """Transient request; every request is gated before any implementation import."""

    declaration: ResolvedComponentDeclaration
    gate: RuntimeComponentLoadGate
    runtime_context: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, ResolvedComponentDeclaration):
            raise TypeError("declaration must be a ResolvedComponentDeclaration")
        if not isinstance(self.gate, RuntimeComponentLoadGate):
            raise TypeError("gate must be a RuntimeComponentLoadGate")
        if not isinstance(self.runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")


@dataclass(frozen=True, slots=True)
class RuntimeComponentLoadResult:
    """Live instance paired with the immutable G1 and post-load G3 receipts."""

    instance: object
    artifact_binding: ComponentArtifactBinding
    load_attestation: ComponentLoadAttestation

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_binding, ComponentArtifactBinding):
            raise TypeError("artifact_binding must be a ComponentArtifactBinding")
        if not isinstance(self.load_attestation, ComponentLoadAttestation):
            raise TypeError("load_attestation must be a ComponentLoadAttestation")
        if self.load_attestation.binding_id != self.artifact_binding.binding_id:
            raise ValueError("load attestation differs from the G1 binding")
        if (
            self.load_attestation.implementation_identity
            != self.artifact_binding.implementation_identity
            or self.load_attestation.canonical_interface
            != self.artifact_binding.canonical_interface
            or self.load_attestation.interface_version
            != self.artifact_binding.interface_version
        ):
            raise ValueError("load attestation facts differ from the G1 binding")

    def attest_prepared(
        self,
        *,
        runtime_identity: str,
        verified_fields: tuple[tuple[str, str], ...],
    ) -> RuntimeBoundContract:
        """Create G3 from this exact in-memory G1 receipt after preparation."""

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


RuntimeContextResolver = Callable[
    [RuntimeComponentLoadRequest, Mapping[str, object]],
    Mapping[str, Any],
]


class RuntimeComponentLoader:
    """Import and construct an implementation only through a validated gate."""

    def load(
        self,
        declaration: ResolvedComponentDeclaration,
        *,
        gate: RuntimeComponentLoadGate,
        binding_set: ComponentArtifactBindingSet,
        load_plan: ComponentLoadPlan,
        runtime_context: Mapping[str, Any],
    ) -> RuntimeComponentLoadResult:
        """Load an exactly one-slot final graph through the strict graph gate."""

        request = RuntimeComponentLoadRequest(
            declaration=declaration,
            gate=gate,
            runtime_context=runtime_context,
        )
        return self.load_all(
            (request,),
            binding_set=binding_set,
            load_plan=load_plan,
        )[0]

    def load_all(
        self,
        requests: tuple[RuntimeComponentLoadRequest, ...],
        *,
        binding_set: ComponentArtifactBindingSet,
        load_plan: ComponentLoadPlan,
        context_resolver: RuntimeContextResolver | None = None,
    ) -> tuple[RuntimeComponentLoadResult, ...]:
        """Prove exact final-graph coverage before importing any implementation."""

        if type(requests) is not tuple or not requests:
            raise ValueError("requests must be a non-empty tuple")
        if any(not isinstance(item, RuntimeComponentLoadRequest) for item in requests):
            raise TypeError("requests must contain RuntimeComponentLoadRequest values")
        if not isinstance(binding_set, ComponentArtifactBindingSet):
            raise TypeError("binding_set must be a ComponentArtifactBindingSet")
        if not isinstance(load_plan, ComponentLoadPlan):
            raise TypeError("load_plan must be a ComponentLoadPlan")
        if context_resolver is not None and not callable(context_resolver):
            raise TypeError("context_resolver must be callable or None")
        _validate_load_plan(requests, binding_set=binding_set, load_plan=load_plan)

        # Every declaration/G1 relationship is checked before the first import,
        # including the lightweight canonical-interface imports.
        for request in requests:
            request.gate.validate(request.declaration)

        expected_types: list[type] = []
        for request in requests:
            expected_types.append(
                _load_canonical_interface(request.declaration, request.gate)
            )

        results: list[RuntimeComponentLoadResult] = []
        loaded_by_slot: dict[str, object] = {}
        try:
            for request, expected_type in zip(requests, expected_types, strict=True):
                runtime_context = request.runtime_context
                if context_resolver is not None:
                    runtime_context = context_resolver(
                        request,
                        MappingProxyType(dict(loaded_by_slot)),
                    )
                    if not isinstance(runtime_context, Mapping):
                        raise TypeError(
                            "context_resolver must return a Mapping for slot "
                            f"{request.gate.artifact_binding.slot!r}"
                        )
                result = _construct_component(
                    request.declaration,
                    gate=request.gate,
                    expected_type=expected_type,
                    runtime_context=runtime_context,
                )
                results.append(result)
                loaded_by_slot[request.gate.artifact_binding.slot] = result.instance
        except BaseException as primary:
            for result in reversed(results):
                close = getattr(result.instance, "close", None)
                if not callable(close):
                    continue
                try:
                    close()
                except BaseException as cleanup_error:  # noqa: BLE001
                    if hasattr(primary, "add_note"):
                        primary.add_note(
                            "component rollback close failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            raise
        return tuple(results)


def _validate_load_plan(
    requests: tuple[RuntimeComponentLoadRequest, ...],
    *,
    binding_set: ComponentArtifactBindingSet,
    load_plan: ComponentLoadPlan,
) -> None:
    """Fail closed on stale, partial, or artifact-drifted all-slot graphs."""

    declaration = requests[0].declaration

    def fail(message: str) -> None:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.LOAD_PLAN_MISMATCH,
            message,
            declaration=declaration,
        )

    if binding_set.recipe_id != load_plan.expected_recipe_id:
        fail("component binding set recipe differs from the expected load-plan recipe")
    if binding_set.binding_set_id != load_plan.expected_binding_set_id:
        fail("component binding set identity differs from the expected load plan")
    if any(
        binding.recipe_id != load_plan.expected_recipe_id
        for binding in binding_set.bindings
    ):
        fail("component binding recipe differs from the expected load-plan recipe")

    request_slots = tuple(item.gate.artifact_binding.slot for item in requests)
    if len(request_slots) != len(set(request_slots)):
        fail("component load request slots must be unique")
    request_slot_set = set(request_slots)
    binding_slot_set = set(binding_set.slots)
    plan_slot_set = set(load_plan.slots)
    if request_slot_set != binding_slot_set or request_slot_set != plan_slot_set:
        fail(
            "component load requests, binding set, and load plan must exactly "
            "cover the same slots"
        )

    requests_by_slot = {
        request.gate.artifact_binding.slot: request for request in requests
    }
    requirements_by_slot = {
        requirement.slot: requirement for requirement in load_plan.requirements
    }
    for binding in binding_set.bindings:
        request_binding = requests_by_slot[binding.slot].gate.artifact_binding
        requirement = requirements_by_slot[binding.slot]
        if request_binding.recipe_id != load_plan.expected_recipe_id:
            fail(
                f"request binding recipe for slot {binding.slot!r} differs from "
                "the expected load-plan recipe"
            )
        if request_binding.binding_id != binding.binding_id:
            fail(
                f"request binding for slot {binding.slot!r} differs from the "
                "component binding set"
            )
        if requirement.binding_id != binding.binding_id:
            fail(
                f"load-plan binding for slot {binding.slot!r} differs from the "
                "component binding set"
            )
        artifact_names = tuple(
            name for name, _identity in binding.artifact_content_identities
        )
        if artifact_names != requirement.required_artifact_names:
            fail(
                f"artifacts for slot {binding.slot!r} must exactly cover the "
                "load-plan requirement"
            )


def build_component_artifact_binding(
    declaration: ResolvedComponentDeclaration,
    *,
    recipe_id: str,
    slot: str,
    artifact_content_identities: Mapping[str, str],
    code_identity: str,
) -> ComponentArtifactBinding:
    """Pure G1 factory deriving interface and implementation from declaration."""

    if not isinstance(declaration, ResolvedComponentDeclaration):
        raise TypeError("declaration must be a ResolvedComponentDeclaration")
    if not isinstance(artifact_content_identities, Mapping):
        raise TypeError("artifact_content_identities must be a mapping")
    contents = tuple(sorted(artifact_content_identities.items()))
    return ComponentArtifactBinding.create(
        recipe_id=recipe_id,
        slot=slot,
        component_declaration_id=declaration.declaration_id,
        declared=declaration.declared_contract,
        artifact_content_identities=contents,
        code_identity=code_identity,
        implementation_identity=declaration.implementation_class_path,
        interface_version=declaration.descriptor.interface_version,
    )


def _load_canonical_interface(
    declaration: ResolvedComponentDeclaration,
    gate: RuntimeComponentLoadGate,
) -> type:
    class_path = gate.artifact_binding.canonical_interface
    module_name, qualname = class_path.split(":", 1)
    try:
        value: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INTERFACE_IMPORT_FAILED,
            f"cannot import canonical runtime interface {class_path!r}",
            declaration=declaration,
        ) from exc
    if not isinstance(value, type) or value is object:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INTERFACE_MISMATCH,
            "canonical runtime interface must resolve to a non-object class",
            declaration=declaration,
        )
    observed_version = getattr(value, "INTERFACE_VERSION", None)
    if observed_version != gate.artifact_binding.interface_version:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INTERFACE_MISMATCH,
            f"canonical interface version {observed_version!r} does not match "
            f"G1 {gate.artifact_binding.interface_version!r}",
            declaration=declaration,
        )
    return value


def _construct_component(
    declaration: ResolvedComponentDeclaration,
    *,
    gate: RuntimeComponentLoadGate,
    expected_type: type,
    runtime_context: Mapping[str, Any],
) -> RuntimeComponentLoadResult:

    implementation = _load_implementation(declaration)
    if not issubclass(implementation, expected_type):
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INTERFACE_MISMATCH,
            f"{declaration.implementation_class_path} does not implement "
            f"{expected_type.__module__}:{expected_type.__qualname__}",
            declaration=declaration,
        )
    observed_version = getattr(implementation, "INTERFACE_VERSION", None)
    if observed_version != gate.artifact_binding.interface_version:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INTERFACE_MISMATCH,
            f"implementation interface {observed_version!r} does not match "
            f"G1 {gate.artifact_binding.interface_version!r}",
            declaration=declaration,
        )
    abstract_methods = tuple(sorted(getattr(implementation, "__abstractmethods__", ())))
    required_methods = getattr(expected_type, "REQUIRED_RUNTIME_METHODS", ())
    if type(required_methods) is not tuple or any(
        not isinstance(name, str) or not name for name in required_methods
    ):
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INVALID_IMPLEMENTATION,
            "canonical runtime interface REQUIRED_RUNTIME_METHODS must be a "
            "tuple of non-empty method names",
            declaration=declaration,
        )
    missing_methods = tuple(
        name
        for name in required_methods
        if not callable(getattr(implementation, name, None))
    )
    if abstract_methods or missing_methods:
        details: list[str] = []
        if abstract_methods:
            details.append(f"abstract methods {list(abstract_methods)}")
        if missing_methods:
            details.append(f"non-callable operations {list(missing_methods)}")
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INVALID_IMPLEMENTATION,
            "runtime implementation has an incomplete canonical interface: "
            + "; ".join(details),
            declaration=declaration,
        )
    constructor = getattr(implementation, "from_config", None)
    if not callable(constructor):
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INVALID_IMPLEMENTATION,
            "runtime implementation must expose from_config()",
            declaration=declaration,
        )
    try:
        instance = constructor(
            declaration.config,
            runtime_context=runtime_context,
        )
    except Exception as exc:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.CONSTRUCTION_FAILED,
            f"runtime implementation construction failed: {type(exc).__name__}",
            declaration=declaration,
        ) from exc
    if not isinstance(instance, expected_type):
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INVALID_INSTANCE,
            "from_config() returned an incompatible runtime instance",
            declaration=declaration,
        )
    binding = gate.artifact_binding
    load_attestation = ComponentLoadAttestation(
        binding_id=binding.binding_id,
        implementation_identity=declaration.implementation_class_path,
        canonical_interface=binding.canonical_interface,
        interface_version=binding.interface_version,
    )
    return RuntimeComponentLoadResult(
        instance=instance,
        artifact_binding=binding,
        load_attestation=load_attestation,
    )


def _load_implementation(declaration: ResolvedComponentDeclaration) -> type:
    class_path = declaration.implementation_class_path
    module_name, qualname = class_path.split(":", 1)
    try:
        value: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.IMPLEMENTATION_IMPORT_FAILED,
            f"cannot import runtime implementation {class_path!r}",
            declaration=declaration,
        ) from exc
    if not isinstance(value, type):
        raise RuntimeComponentLoadError(
            RuntimeLoadIssueCode.INVALID_IMPLEMENTATION,
            f"runtime implementation path {class_path!r} is not a class",
            declaration=declaration,
        )
    return value
