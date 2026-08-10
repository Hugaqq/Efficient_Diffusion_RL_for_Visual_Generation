"""Pure descriptors and staged contracts shared with composition.

Registry state and resolution logic do not live here.  Domain catalogs may
depend on these immutable values without depending on the composition layer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol

from visual_rl.core.contracts.algorithm import (
    AlgorithmRequirements,
    ConditionerContract,
    CreditContract,
    DistributionMode,
    DynamicsContract,
    LikelihoodSemantics,
    RolloutContract,
    TrainerContract,
    TransitionKind,
)
from visual_rl.core.contracts.model import (
    ComputePrecision,
    LatentLayout,
    MediaKind,
    ModelDescriptorContract,
    PredictionType,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
)
from visual_rl.core.contracts.reward import RewardContract
from visual_rl.core.identity import canonical_identity, to_identity_value

__all__ = (
    "COMPONENT_KINDS",
    "DECLARATION_PROVIDER_ABI",
    "ArtifactBoundContract",
    "BoundPolicyCapabilities",
    "CapabilityMismatch",
    "CatalogFragment",
    "ComponentArtifactBinding",
    "ComponentArtifactBindingSet",
    "ComponentDeclaration",
    "ComponentDescriptor",
    "ComponentKind",
    "ComponentLoadAttestation",
    "ComponentLoadPlan",
    "ComponentLoadSlotRequirement",
    "ComponentPrepareAttestation",
    "DeclarationProvider",
    "DeclaredContract",
    "ModelAlgorithmBinding",
    "RuntimeBoundContract",
    "canonical_component_interface",
    "contract_field_names",
)

ComponentKind = Literal[
    "model",
    "algorithm",
    "trainer",
    "dynamics",
    "rollout",
    "reward",
    "conditioner",
    "credit",
]

COMPONENT_KINDS: tuple[ComponentKind, ...] = (
    "model",
    "algorithm",
    "trainer",
    "dynamics",
    "rollout",
    "reward",
    "conditioner",
    "credit",
)
DECLARATION_PROVIDER_ABI = "visual-rl.declaration-provider.v1"

_ALIAS_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_CLASS_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_SLOT_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_DIGEST_ID_RE = re.compile(r"^(?:[a-z][a-z0-9_.-]*:)?[0-9a-f]{64}$")
_COMPONENT_DECLARATION_ID_RE = re.compile(r"^component-declaration\.v1:[0-9a-f]{64}$")

_CANONICAL_COMPONENT_INTERFACES: Mapping[ComponentKind, str] = MappingProxyType(
    {
        "model": "visual_rl.models.interface:ModelAdapter",
        "algorithm": "visual_rl.algorithms.modules.interface:AlgorithmModule",
        "trainer": "visual_rl.algorithms.trainer.interface:TrainerComponent",
        "dynamics": "visual_rl.algorithms.dynamics.interface:DynamicsComponent",
        "rollout": "visual_rl.algorithms.rollout.interface:RolloutComponent",
        "reward": "visual_rl.algorithms.rewards.interface:RewardComponent",
        "conditioner": (
            "visual_rl.algorithms.conditioning.interface:LatentConditioner"
        ),
        "credit": "visual_rl.algorithms.optimization.interface:CreditComponent",
    }
)


def canonical_component_interface(kind: ComponentKind) -> str:
    """Return the sole import-safe runtime interface path for one slot kind."""

    try:
        return _CANONICAL_COMPONENT_INTERFACES[kind]
    except (KeyError, TypeError):
        raise ValueError(f"unsupported component kind: {kind!r}") from None


@dataclass(frozen=True)
class ComponentDescriptor:
    """Serializable alias descriptor that never stores a live class."""

    alias: str
    implementation_class_path: str | None
    declaration_provider_path: str | None
    declaration_provider_abi: str = DECLARATION_PROVIDER_ABI
    interface_version: str = "1.0"
    optional_dependencies: tuple[str, ...] = ()
    removed_message: str | None = None

    def __post_init__(self) -> None:
        if not _ALIAS_RE.fullmatch(self.alias):
            raise ValueError(f"invalid component alias: {self.alias!r}")
        if not self.interface_version:
            raise ValueError("interface_version must be non-empty")
        if (
            not isinstance(self.declaration_provider_abi, str)
            or not self.declaration_provider_abi
            or self.declaration_provider_abi.strip() != self.declaration_provider_abi
        ):
            raise ValueError("declaration_provider_abi must be canonical text")
        if type(self.optional_dependencies) is not tuple:
            raise TypeError("optional_dependencies must be a tuple")
        if any(
            not isinstance(item, str) or not item for item in self.optional_dependencies
        ):
            raise ValueError("optional_dependencies must contain non-empty strings")
        if self.optional_dependencies != tuple(sorted(set(self.optional_dependencies))):
            raise ValueError("optional_dependencies must be sorted and unique")
        if self.removed_message is not None:
            if not self.removed_message.strip():
                raise ValueError("removed_message must be non-empty")
            if (
                self.implementation_class_path is not None
                or self.declaration_provider_path is not None
            ):
                raise ValueError("a tombstone cannot resolve a class or config type")
            return
        _validate_class_path(
            self.implementation_class_path,
            field="implementation_class_path",
        )
        _validate_class_path(
            self.declaration_provider_path,
            field="declaration_provider_path",
        )


@dataclass(frozen=True, slots=True)
class CatalogFragment:
    """One domain-owned, implementation-free descriptor contribution."""

    owner: str
    kind: ComponentKind
    descriptors: tuple[ComponentDescriptor, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.owner, str)
            or not self.owner
            or self.owner.strip() != self.owner
        ):
            raise ValueError("catalog fragment owner must be canonical text")
        if self.kind not in COMPONENT_KINDS:
            raise ValueError(f"unsupported catalog fragment kind: {self.kind!r}")
        if type(self.descriptors) is not tuple or any(
            not isinstance(item, ComponentDescriptor) for item in self.descriptors
        ):
            raise TypeError("fragment descriptors must be ComponentDescriptor values")
        aliases = tuple(item.alias for item in self.descriptors)
        if len(aliases) != len(set(aliases)):
            duplicate = next(alias for alias in aliases if aliases.count(alias) > 1)
            raise ValueError(
                f"catalog fragment {self.owner!r} contains duplicate alias "
                f"{duplicate!r}"
            )
        object.__setattr__(
            self,
            "descriptors",
            tuple(sorted(self.descriptors, key=lambda item: item.alias)),
        )


@dataclass(frozen=True, slots=True)
class ComponentDeclaration:
    """Import-safe provider output: typed config plus its static contract."""

    config: object
    declared_contract: DeclaredContract

    def __post_init__(self) -> None:
        if isinstance(self.config, type) or not is_dataclass(self.config):
            raise TypeError("declaration config must be a frozen dataclass instance")
        params = getattr(type(self.config), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError("declaration config must be a frozen dataclass instance")
        if not isinstance(self.declared_contract, DeclaredContract):
            raise TypeError("declared_contract must be a DeclaredContract")


class DeclarationProvider(Protocol):
    """ABI implemented by an import-safe domain declaration provider."""

    PROVIDER_ABI: ClassVar[str]
    CONFIG_TYPE_PATH: ClassVar[str]

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        """Parse params and return a static declaration without live resources."""


@dataclass(frozen=True)
class DeclaredContract:
    component_kind: ComponentKind
    component_id: str
    model: ModelDescriptorContract | None = None
    algorithm: AlgorithmRequirements | None = None
    trainer: TrainerContract | None = None
    dynamics: DynamicsContract | None = None
    rollout: RolloutContract | None = None
    reward: RewardContract | None = None
    conditioner: ConditionerContract | None = None
    credit: CreditContract | None = None
    pending_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        detail_names = (
            "model",
            "algorithm",
            "trainer",
            "dynamics",
            "rollout",
            "reward",
            "conditioner",
            "credit",
        )
        selected = tuple(
            name for name in detail_names if getattr(self, name) is not None
        )
        if selected != (self.component_kind,):
            raise ValueError(
                "declared contract must set exactly the detail matching component_kind"
            )
        _unique("pending_fields", self.pending_fields)

    @property
    def detail(self) -> object:
        return getattr(self, self.component_kind)


@dataclass(frozen=True)
class ArtifactBoundContract:
    """Compatibility-only artifact contract used by the uncut production path."""

    declared: DeclaredContract
    artifact_identity: str
    resolved_fields: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.artifact_identity:
            raise ValueError("artifact_identity must be non-empty")
        _unique("resolved field paths", tuple(key for key, _ in self.resolved_fields))


@dataclass(frozen=True, slots=True)
class ComponentArtifactBinding:
    """G1 receipt binding one full declaration to exact artifacts and code."""

    recipe_id: str
    slot: str
    component_declaration_id: str
    declared: DeclaredContract
    artifact_set_identity: str
    artifact_content_identities: tuple[tuple[str, str], ...]
    code_identity: str
    implementation_identity: str
    canonical_interface: str
    interface_version: str
    schema: str = "visual-rl.component-artifact-binding"
    schema_version: int = 1
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest_identity("recipe_id", self.recipe_id)
        if not isinstance(self.slot, str) or _SLOT_RE.fullmatch(self.slot) is None:
            raise ValueError("slot must be a canonical component slot")
        if (
            not isinstance(self.component_declaration_id, str)
            or _COMPONENT_DECLARATION_ID_RE.fullmatch(self.component_declaration_id)
            is None
        ):
            raise ValueError(
                "component_declaration_id must be a complete component declaration "
                "identity"
            )
        if not isinstance(self.declared, DeclaredContract):
            raise TypeError("declared must be a DeclaredContract")
        _string_pairs(
            "artifact_content_identities",
            self.artifact_content_identities,
        )
        if self.artifact_content_identities != tuple(
            sorted(self.artifact_content_identities)
        ):
            raise ValueError(
                "artifact_content_identities must use canonical sorted order"
            )
        for name, identity in self.artifact_content_identities:
            if not _SLOT_RE.fullmatch(name):
                raise ValueError("artifact content names must be canonical slots")
            _require_digest_identity(
                f"artifact content identity {name!r}",
                identity,
            )
        _require_digest_identity("code_identity", self.code_identity)
        content_by_name = dict(self.artifact_content_identities)
        if content_by_name.get("code") != self.code_identity:
            raise ValueError(
                "artifact content identities must contain the exact code identity"
            )
        expected_artifact_set = canonical_identity(
            "component-artifact-set.v1",
            self.artifact_content_identities,
        )
        if self.artifact_set_identity != expected_artifact_set:
            raise ValueError(
                "artifact_set_identity differs from the canonical content set"
            )
        _validate_class_path(
            self.implementation_identity,
            field="implementation_identity",
        )
        _validate_class_path(
            self.canonical_interface,
            field="canonical_interface",
        )
        expected_interface = canonical_component_interface(self.declared.component_kind)
        if self.canonical_interface != expected_interface:
            raise ValueError(
                "canonical_interface differs from the controlled component-kind map"
            )
        _canonical_text("interface_version", self.interface_version)
        if self.schema != "visual-rl.component-artifact-binding":
            raise ValueError("unsupported component artifact binding schema")
        if self.schema_version != 1:
            raise ValueError("unsupported component artifact binding schema version")
        object.__setattr__(
            self,
            "binding_id",
            canonical_identity("component-artifact-binding.v1", self),
        )

    @classmethod
    def create(
        cls,
        *,
        recipe_id: str,
        slot: str,
        component_declaration_id: str,
        declared: DeclaredContract,
        artifact_content_identities: tuple[tuple[str, str], ...],
        code_identity: str,
        implementation_identity: str,
        interface_version: str,
    ) -> ComponentArtifactBinding:
        """Purely derive the controlled interface and canonical artifact-set id."""

        if not isinstance(declared, DeclaredContract):
            raise TypeError("declared must be a DeclaredContract")
        contents = tuple(sorted(artifact_content_identities))
        return cls(
            recipe_id=recipe_id,
            slot=slot,
            component_declaration_id=component_declaration_id,
            declared=declared,
            artifact_set_identity=canonical_identity(
                "component-artifact-set.v1",
                contents,
            ),
            artifact_content_identities=contents,
            code_identity=code_identity,
            implementation_identity=implementation_identity,
            canonical_interface=canonical_component_interface(declared.component_kind),
            interface_version=interface_version,
        )

    @property
    def artifact_identity(self) -> str:
        """Compatibility-shaped access to the complete artifact-set identity."""

        return self.artifact_set_identity

    @property
    def resolved_fields(self) -> tuple[tuple[str, str], ...]:
        return self.artifact_content_identities

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "slot": self.slot,
            "component_declaration_id": self.component_declaration_id,
            "declared_contract": to_identity_value(self.declared),
            "artifact_set_identity": self.artifact_set_identity,
            "artifact_content_identities": [
                {"name": name, "content_identity": identity}
                for name, identity in self.artifact_content_identities
            ],
            "code_identity": self.code_identity,
            "implementation_identity": self.implementation_identity,
            "canonical_interface": self.canonical_interface,
            "interface_version": self.interface_version,
            "binding_id": self.binding_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentArtifactBindingSet:
    """Canonical all-slot G1 receipts derived after ``recipe_id`` exists."""

    recipe_id: str
    bindings: tuple[ComponentArtifactBinding, ...]
    binding_set_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest_identity("recipe_id", self.recipe_id)
        if type(self.bindings) is not tuple or not self.bindings:
            raise ValueError("component artifact binding set must not be empty")
        if any(
            not isinstance(item, ComponentArtifactBinding) for item in self.bindings
        ):
            raise TypeError("bindings must contain ComponentArtifactBinding values")
        bindings = tuple(sorted(self.bindings, key=lambda item: item.slot))
        slots = tuple(item.slot for item in bindings)
        if len(slots) != len(set(slots)):
            raise ValueError("component artifact binding slots must be unique")
        if any(item.recipe_id != self.recipe_id for item in bindings):
            raise ValueError(
                "every component artifact binding must reference the same recipe_id"
            )
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self,
            "binding_set_id",
            canonical_identity(
                "component-artifact-binding-set.v1",
                {
                    "recipe_id": self.recipe_id,
                    "binding_ids": tuple(item.binding_id for item in bindings),
                },
            ),
        )

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(item.slot for item in self.bindings)

    def binding(self, slot: str) -> ComponentArtifactBinding:
        if not isinstance(slot, str) or _SLOT_RE.fullmatch(slot) is None:
            raise ValueError("slot must be a canonical component slot")
        for binding in self.bindings:
            if binding.slot == slot:
                return binding
        raise KeyError(f"unknown component artifact binding slot {slot!r}")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "visual-rl.component-artifact-binding-set",
            "schema_version": 1,
            "recipe_id": self.recipe_id,
            "bindings": tuple(item.to_payload() for item in self.bindings),
            "binding_set_id": self.binding_set_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentLoadSlotRequirement:
    """Import-safe final-graph requirement for one canonical component slot."""

    slot: str
    binding_id: str
    required_artifact_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or _SLOT_RE.fullmatch(self.slot) is None:
            raise ValueError("slot must be a canonical component slot")
        _require_namespaced_identity(
            "binding_id",
            self.binding_id,
            namespace="component-artifact-binding.v1",
        )
        _strings("required_artifact_names", self.required_artifact_names)
        if self.required_artifact_names != tuple(sorted(self.required_artifact_names)):
            raise ValueError("required_artifact_names must use canonical sorted order")
        if any(
            _SLOT_RE.fullmatch(name) is None for name in self.required_artifact_names
        ):
            raise ValueError("required artifact names must be canonical slots")
        if "code" not in self.required_artifact_names:
            raise ValueError("every component load requirement must include code")

    def to_payload(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "binding_id": self.binding_id,
            "required_artifact_names": self.required_artifact_names,
        }


@dataclass(frozen=True, slots=True)
class ComponentLoadPlan:
    """Expected all-slot graph consumed before any runtime implementation import.

    This is an import-safe shadow ABI.  A future recipe compiler must derive it
    from the complete ``MaterializedRecipe`` graph; the runtime loader only
    verifies a supplied plan and does not claim that integration yet.
    """

    expected_recipe_id: str
    expected_binding_set_id: str
    requirements: tuple[ComponentLoadSlotRequirement, ...]
    schema: str = "visual-rl.component-load-plan"
    schema_version: int = 1
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest_identity("expected_recipe_id", self.expected_recipe_id)
        _require_namespaced_identity(
            "expected_binding_set_id",
            self.expected_binding_set_id,
            namespace="component-artifact-binding-set.v1",
        )
        if type(self.requirements) is not tuple or not self.requirements:
            raise ValueError("component load plan requirements must not be empty")
        if any(
            not isinstance(item, ComponentLoadSlotRequirement)
            for item in self.requirements
        ):
            raise TypeError(
                "requirements must contain ComponentLoadSlotRequirement values"
            )
        requirements = tuple(sorted(self.requirements, key=lambda item: item.slot))
        slots = tuple(item.slot for item in requirements)
        if len(slots) != len(set(slots)):
            raise ValueError("component load plan slots must be unique")
        binding_ids = tuple(item.binding_id for item in requirements)
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("component load plan binding ids must be unique")
        if self.schema != "visual-rl.component-load-plan":
            raise ValueError("unsupported component load plan schema")
        if self.schema_version != 1:
            raise ValueError("unsupported component load plan schema version")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(
            self,
            "plan_id",
            canonical_identity("component-load-plan.v1", self),
        )

    @classmethod
    def create(
        cls,
        binding_set: ComponentArtifactBindingSet,
        *,
        required_artifact_names_by_slot: Mapping[str, tuple[str, ...]],
    ) -> ComponentLoadPlan:
        """Build a plan from explicit final-graph artifact requirements."""

        if not isinstance(binding_set, ComponentArtifactBindingSet):
            raise TypeError("binding_set must be a ComponentArtifactBindingSet")
        if not isinstance(required_artifact_names_by_slot, Mapping):
            raise TypeError("required_artifact_names_by_slot must be a mapping")
        if set(required_artifact_names_by_slot) != set(binding_set.slots):
            raise ValueError(
                "artifact requirements must exactly cover component binding slots"
            )
        requirements = tuple(
            ComponentLoadSlotRequirement(
                slot=binding.slot,
                binding_id=binding.binding_id,
                required_artifact_names=required_artifact_names_by_slot[binding.slot],
            )
            for binding in binding_set.bindings
        )
        return cls(
            expected_recipe_id=binding_set.recipe_id,
            expected_binding_set_id=binding_set.binding_set_id,
            requirements=requirements,
        )

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(item.slot for item in self.requirements)

    def requirement(self, slot: str) -> ComponentLoadSlotRequirement:
        if not isinstance(slot, str) or _SLOT_RE.fullmatch(slot) is None:
            raise ValueError("slot must be a canonical component slot")
        for requirement in self.requirements:
            if requirement.slot == slot:
                return requirement
        raise KeyError(f"unknown component load plan slot {slot!r}")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "expected_recipe_id": self.expected_recipe_id,
            "expected_binding_set_id": self.expected_binding_set_id,
            "requirements": tuple(item.to_payload() for item in self.requirements),
            "plan_id": self.plan_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentLoadAttestation:
    """Immutable G3 evidence emitted only after construction succeeds."""

    binding_id: str
    implementation_identity: str
    canonical_interface: str
    interface_version: str
    schema: str = "visual-rl.component-load-attestation"
    schema_version: int = 1
    attestation_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_namespaced_identity(
            "binding_id",
            self.binding_id,
            namespace="component-artifact-binding.v1",
        )
        _validate_class_path(
            self.implementation_identity,
            field="implementation_identity",
        )
        _validate_class_path(
            self.canonical_interface,
            field="canonical_interface",
        )
        _canonical_text("interface_version", self.interface_version)
        if self.schema != "visual-rl.component-load-attestation":
            raise ValueError("unsupported component load attestation schema")
        if self.schema_version != 1:
            raise ValueError("unsupported component load attestation schema version")
        object.__setattr__(
            self,
            "attestation_id",
            canonical_identity("component-load-attestation.v1", self),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "binding_id": self.binding_id,
            "implementation_identity": self.implementation_identity,
            "canonical_interface": self.canonical_interface,
            "interface_version": self.interface_version,
            "attestation_id": self.attestation_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentPrepareAttestation:
    """Immutable G3 receipt for preparation performed after one exact load."""

    binding_id: str
    load_attestation_id: str
    runtime_identity: str
    verified_fields: tuple[tuple[str, str], ...]
    schema: str = "visual-rl.component-prepare-attestation"
    schema_version: int = 1
    attestation_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_namespaced_identity(
            "binding_id",
            self.binding_id,
            namespace="component-artifact-binding.v1",
        )
        _require_namespaced_identity(
            "load_attestation_id",
            self.load_attestation_id,
            namespace="component-load-attestation.v1",
        )
        _require_digest_identity("runtime_identity", self.runtime_identity)
        _string_pairs("verified_fields", self.verified_fields)
        if not self.verified_fields:
            raise ValueError("verified_fields must be non-empty")
        if self.verified_fields != tuple(sorted(self.verified_fields)):
            raise ValueError("verified_fields must use canonical sorted order")
        for field_path, value in self.verified_fields:
            _canonical_text("verified field path", field_path)
            _canonical_text(f"verified field {field_path!r}", value)
        if self.schema != "visual-rl.component-prepare-attestation":
            raise ValueError("unsupported component prepare attestation schema")
        if self.schema_version != 1:
            raise ValueError("unsupported prepare attestation schema version")
        object.__setattr__(
            self,
            "attestation_id",
            canonical_identity("component-prepare-attestation.v1", self),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "binding_id": self.binding_id,
            "load_attestation_id": self.load_attestation_id,
            "runtime_identity": self.runtime_identity,
            "verified_fields": [list(item) for item in self.verified_fields],
            "attestation_id": self.attestation_id,
        }


@dataclass(frozen=True)
class RuntimeBoundContract:
    """G3 evidence, with an explicit compatibility branch until production cut.

    Legacy production continues to pass ``ArtifactBoundContract`` without
    attestations.  A declaration-bound ``ComponentArtifactBinding`` always
    requires exact load and prepare receipts; the two modes cannot be mixed.
    """

    artifact: ArtifactBoundContract | ComponentArtifactBinding
    runtime_identity: str
    verified_fields: tuple[tuple[str, str], ...]
    load_attestation: InitVar[ComponentLoadAttestation | None] = None
    prepare_attestation: InitVar[ComponentPrepareAttestation | None] = None
    _load_attestation: ComponentLoadAttestation | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _prepare_attestation: ComponentPrepareAttestation | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
        load_attestation: ComponentLoadAttestation | None,
        prepare_attestation: ComponentPrepareAttestation | None,
    ) -> None:
        if not self.runtime_identity:
            raise ValueError("runtime_identity must be non-empty")
        _unique("verified field paths", tuple(key for key, _ in self.verified_fields))
        if isinstance(self.artifact, ComponentArtifactBinding):
            if not isinstance(load_attestation, ComponentLoadAttestation):
                raise TypeError(
                    "declaration-bound runtime contract requires load attestation"
                )
            if not isinstance(prepare_attestation, ComponentPrepareAttestation):
                raise TypeError(
                    "declaration-bound runtime contract requires prepare attestation"
                )
            if load_attestation.binding_id != self.artifact.binding_id:
                raise ValueError("load attestation differs from the G1 binding")
            if (
                load_attestation.implementation_identity
                != self.artifact.implementation_identity
                or load_attestation.canonical_interface
                != self.artifact.canonical_interface
                or load_attestation.interface_version != self.artifact.interface_version
            ):
                raise ValueError("load attestation facts differ from the G1 binding")
            if prepare_attestation.binding_id != self.artifact.binding_id:
                raise ValueError("prepare attestation differs from the G1 binding")
            if (
                prepare_attestation.load_attestation_id
                != load_attestation.attestation_id
            ):
                raise ValueError("prepare attestation differs from the load receipt")
            if prepare_attestation.runtime_identity != self.runtime_identity:
                raise ValueError("runtime identity differs from prepare attestation")
            if prepare_attestation.verified_fields != self.verified_fields:
                raise ValueError("verified fields differ from prepare attestation")
            object.__setattr__(self, "_load_attestation", load_attestation)
            object.__setattr__(self, "_prepare_attestation", prepare_attestation)
            return
        if not isinstance(self.artifact, ArtifactBoundContract):
            raise TypeError(
                "artifact must be ArtifactBoundContract or ComponentArtifactBinding"
            )
        if load_attestation is not None or prepare_attestation is not None:
            raise ValueError(
                "legacy artifact contract cannot carry declaration-bound attestations"
            )

    @property
    def is_declaration_bound(self) -> bool:
        return isinstance(self.artifact, ComponentArtifactBinding)

    @property
    def component_binding_id(self) -> str:
        if not isinstance(self.artifact, ComponentArtifactBinding):
            raise AttributeError(  # noqa: TRY004 - unavailable in legacy mode
                "legacy compatibility contract has no component binding id"
            )
        return self.artifact.binding_id

    @property
    def component_load_attestation(self) -> ComponentLoadAttestation:
        value = self._load_attestation
        if value is None:
            raise AttributeError(
                "legacy compatibility contract has no component load attestation"
            )
        return value

    @property
    def component_prepare_attestation(self) -> ComponentPrepareAttestation:
        value = self._prepare_attestation
        if value is None:
            raise AttributeError(
                "legacy compatibility contract has no prepare attestation"
            )
        return value

    @property
    def contract_id(self) -> str:
        """Return the canonical raw digest used by aggregate G3/checkpoint ABIs.

        The live contract remains the authority.  Callers may persist this
        projection, but they must never accept an independently supplied digest
        in place of the typed contract.
        """

        identity = canonical_identity("runtime-bound-contract.v1", self.to_payload())
        return identity.partition(":")[2]

    def to_payload(self) -> dict[str, object]:
        if isinstance(self.artifact, ComponentArtifactBinding):
            return {
                "schema": "visual-rl.runtime-bound-contract",
                "schema_version": 1,
                "mode": "declaration_bound",
                "binding_id": self.component_binding_id,
                "artifact_binding": self.artifact.to_payload(),
                "load_attestation": self.component_load_attestation.to_payload(),
                "prepare_attestation": (
                    self.component_prepare_attestation.to_payload()
                ),
                "runtime_identity": self.runtime_identity,
                "verified_fields": [list(item) for item in self.verified_fields],
            }
        return {
            "schema": "visual-rl.runtime-bound-contract",
            "schema_version": 1,
            "mode": "legacy_compatibility",
            "artifact_contract": {
                "declared_contract": to_identity_value(self.artifact.declared),
                "artifact_identity": self.artifact.artifact_identity,
                "resolved_fields": [
                    list(item) for item in self.artifact.resolved_fields
                ],
            },
            "runtime_identity": self.runtime_identity,
            "verified_fields": [list(item) for item in self.verified_fields],
        }


@dataclass(frozen=True, slots=True)
class BoundPolicyCapabilities:
    """Model semantics enriched by the selected Dynamics and Trainer ports."""

    tasks: tuple[TaskKind, ...]
    output_media: tuple[MediaKind, ...]
    latent_layouts: tuple[LatentLayout, ...]
    latent_ranks: tuple[int, ...]
    prediction_types: tuple[PredictionType, ...]
    time_coordinates: tuple[TimeCoordinate, ...]
    training_modes: tuple[TrainingMode, ...]
    supported_precisions: tuple[ComputePrecision, ...]
    provides_reference_policy: bool | None
    condition_payload_types: tuple[str, ...] = ()
    transition_kinds: tuple[TransitionKind, ...] = ()
    transition_dtypes: tuple[str, ...] = ()
    transition_features: tuple[str, ...] = ()
    policy_metadata_fields: tuple[str, ...] = ()
    likelihood_semantics: tuple[LikelihoodSemantics, ...] = ()
    distribution_modes: tuple[DistributionMode, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "tasks",
            "output_media",
            "latent_layouts",
            "latent_ranks",
            "prediction_types",
            "time_coordinates",
            "training_modes",
            "supported_precisions",
            "transition_kinds",
            "likelihood_semantics",
            "distribution_modes",
        ):
            _unique(name, getattr(self, name))
        for name in (
            "condition_payload_types",
            "transition_dtypes",
            "transition_features",
            "policy_metadata_fields",
        ):
            _strings(name, getattr(self, name))
        if not self.tasks or not self.output_media:
            raise ValueError(
                "model task and output-media capabilities must not be empty"
            )
        if not self.latent_layouts or not self.latent_ranks:
            raise ValueError("model latent capabilities must not be empty")
        if any(type(rank) is not int or rank < 1 for rank in self.latent_ranks):
            raise ValueError("latent_ranks must contain positive integers")
        if not self.prediction_types or not self.time_coordinates:
            raise ValueError("model prediction/time capabilities must not be empty")
        if not self.training_modes or not self.supported_precisions:
            raise ValueError("model training/numerics capabilities must not be empty")
        if (
            self.provides_reference_policy is not None
            and type(self.provides_reference_policy) is not bool
        ):
            raise TypeError("provides_reference_policy must be bool or None")
        for name in (
            "transition_dtypes",
            "transition_features",
            "policy_metadata_fields",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(values)):
                raise ValueError(f"{name} must be sorted")

    @classmethod
    def from_model_contract(
        cls,
        model: ModelDescriptorContract,
    ) -> BoundPolicyCapabilities:
        """Project model-owned semantics without claiming runtime capabilities."""

        return cls.from_contracts(model)

    @classmethod
    def from_contracts(
        cls,
        model: ModelDescriptorContract,
        *,
        dynamics: DynamicsContract | None = None,
        trainer: TrainerContract | None = None,
    ) -> BoundPolicyCapabilities:
        if not isinstance(model, ModelDescriptorContract):
            raise TypeError("model must be a ModelDescriptorContract")
        if dynamics is not None and not isinstance(dynamics, DynamicsContract):
            raise TypeError("dynamics must be a DynamicsContract or None")
        if trainer is not None and not isinstance(trainer, TrainerContract):
            raise TypeError("trainer must be a TrainerContract or None")
        return cls(
            tasks=model.tasks,
            output_media=model.output_media,
            latent_layouts=model.latent_layouts,
            latent_ranks=model.latent_ranks,
            prediction_types=model.prediction_types,
            time_coordinates=model.time_coordinates,
            training_modes=model.training_modes,
            supported_precisions=model.supported_precisions,
            provides_reference_policy=model.provides_reference_policy,
            condition_payload_types=model.condition_payload_types,
            transition_kinds=(dynamics.transition_kind,) if dynamics else (),
            transition_dtypes=(dynamics.accepted_transition_dtypes if dynamics else ()),
            transition_features=(
                _transition_features(dynamics) if dynamics is not None else ()
            ),
            policy_metadata_fields=(
                dynamics.produced_policy_metadata_fields if dynamics is not None else ()
            ),
            likelihood_semantics=(
                dynamics.supported_likelihoods if dynamics is not None else ()
            ),
            distribution_modes=(
                trainer.accepted_distribution_modes if trainer is not None else ()
            ),
        )

    @property
    def capability_id(self) -> str:
        return canonical_identity("model-capabilities.v1", self)


@dataclass(frozen=True, slots=True)
class CapabilityMismatch:
    """One structured producer/consumer incompatibility."""

    code: str
    producer_field: str
    consumer_field: str
    required: object
    provided: object
    hint: str

    def __post_init__(self) -> None:
        for name in ("code", "producer_field", "consumer_field", "hint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ModelAlgorithmBinding:
    """Validated capability selection emitted by the composition resolver."""

    model_capabilities: BoundPolicyCapabilities
    algorithm_requirements: AlgorithmRequirements
    selections: tuple[tuple[str, str], ...]
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_capabilities, BoundPolicyCapabilities):
            raise TypeError("model_capabilities must be BoundPolicyCapabilities")
        if not isinstance(self.algorithm_requirements, AlgorithmRequirements):
            raise TypeError("algorithm_requirements must be AlgorithmRequirements")
        _string_pairs("selections", self.selections)
        if self.selections != tuple(sorted(self.selections)):
            raise ValueError("selections must use canonical sorted order")
        object.__setattr__(
            self,
            "binding_id",
            canonical_identity(
                "model-algorithm-binding.v1",
                {
                    "model_capability_id": self.model_capabilities.capability_id,
                    "algorithm_requirement_id": (
                        self.algorithm_requirements.requirement_id
                    ),
                    "selections": self.selections,
                },
            ),
        )


def contract_field_names() -> tuple[str, ...]:
    """Stable schema hook used by manifest/checkpoint tests."""

    return tuple(field.name for field in fields(DeclaredContract))


def _validate_class_path(value: str | None, *, field: str) -> None:
    if not isinstance(value, str) or not _CLASS_PATH_RE.fullmatch(value):
        raise ValueError(f"{field} must use explicit module:Class syntax")


def _canonical_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{name} must be canonical non-empty text")
    return value


def _require_digest_identity(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 identity")
    return value


def _require_namespaced_identity(
    name: str,
    value: object,
    *,
    namespace: str,
) -> str:
    expected_prefix = f"{namespace}:"
    resolved = _require_digest_identity(name, value)
    if not resolved.startswith(expected_prefix):
        raise ValueError(f"{name} must use namespace {namespace!r}")
    return resolved


def _unique(name: str, values: tuple[object, ...]) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def _strings(name: str, values: tuple[str, ...]) -> None:
    _unique(name, values)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")


def _string_pairs(name: str, values: tuple[tuple[str, str], ...]) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(
        type(item) is not tuple
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        or not isinstance(item[1], str)
        or not item[1]
        for item in values
    ):
        raise ValueError(f"{name} must contain non-empty string pairs")
    _unique(f"{name} keys", tuple(key for key, _ in values))


def _transition_features(contract: DynamicsContract) -> tuple[str, ...]:
    declared = {
        "stochastic": contract.stochastic,
        "mean_std": contract.exposes_mean_std,
        "scores_arbitrary_action": contract.scores_arbitrary_action,
        "differentiable_log_prob": contract.differentiable_log_prob,
        "replayable": contract.replayable,
        "branchable": contract.branchable,
        "deterministic_ode": contract.supports_deterministic_ode,
    }
    features = {"transition", *(name for name, enabled in declared.items() if enabled)}
    return tuple(sorted(features))
