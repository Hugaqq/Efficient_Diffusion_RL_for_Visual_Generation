"""Composition-owned frozen values shared by the three preflight layers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from visual_rl.core.serialization import canonical_json_text
from visual_rl.composition.recipes.schema import MaterializedRecipe, ResolvedRecipe
from visual_rl.composition.config.specs import ArtifactLocations
from visual_rl.core.contracts import (
    ComponentArtifactBindingSet,
    ComponentLoadPlan,
    RuntimeBoundContract,
)
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data import SourceLocationBinding

_EMPTY_LAUNCH_AUDIT = FrozenMapping(
    {"schema_version": 1, "reward_runtime_bindings": ()}
)
_EMPTY_LAUNCH_AUDIT_ID = hashlib.sha256(
    canonical_json_text(to_plain_dict(_EMPTY_LAUNCH_AUDIT)).encode("utf-8")
).hexdigest()

__all__ = (
    "ArtifactIdentityRequest",
    "ArtifactIdentityResolution",
    "ArtifactIdentityResolver",
    "EnvironmentPreflightResult",
    "RuntimeBindInput",
    "RuntimeBindResult",
    "RuntimeFacts",
    "RuntimeGraphBindInput",
    "RuntimeGraphBindResult",
    "StaticPreflightResult",
    "runtime_launch_payload_id",
)

_MATERIALIZED_RECIPE_ID = re.compile(r"^materialized-recipe\.v2:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ArtifactIdentityRequest:
    """Resolved semantics plus launch-only locations for read-only probing."""

    resolved: ResolvedRecipe
    locations: ArtifactLocations

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedRecipe):
            raise TypeError("resolved must be a ResolvedRecipe")
        if not isinstance(self.locations, ArtifactLocations):
            raise TypeError("locations must be ArtifactLocations")


@dataclass(frozen=True, slots=True)
class ArtifactIdentityResolution:
    """Typed artifact-gate result before path-free recipe materialization.

    Dataset locations remain launch-only in ``source_locations``.  The other
    fields are immutable content records; ``run_environment_preflight`` is the
    sole owner that binds them into ``MaterializedRecipe`` and derives G1.
    """

    model_artifact_identity: FrozenMapping
    source_locations: SourceLocationBinding
    reward_artifact_identities: tuple[tuple[str, FrozenMapping], ...]
    code_artifact_identity: FrozenMapping

    def __post_init__(self) -> None:
        for field_name in (
            "model_artifact_identity",
            "code_artifact_identity",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, FrozenMapping) or not value:
                raise ValueError(f"{field_name} must be a non-empty FrozenMapping")
        if not isinstance(self.source_locations, SourceLocationBinding):
            raise TypeError("source_locations must be a SourceLocationBinding")
        values = self.reward_artifact_identities
        if type(values) is not tuple or not values:
            raise ValueError("reward_artifact_identities must not be empty")
        refs: list[str] = []
        for item in values:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "reward_artifact_identities must contain "
                    "(artifact_ref, FrozenMapping) pairs"
                )
            artifact_ref, identity = item
            if not isinstance(artifact_ref, str) or not artifact_ref:
                raise ValueError("reward artifact refs must be non-empty strings")
            if not isinstance(identity, FrozenMapping) or not identity:
                raise ValueError(
                    "reward artifact identities must be non-empty FrozenMappings"
                )
            refs.append(artifact_ref)
        if tuple(refs) != tuple(sorted(set(refs))):
            raise ValueError("reward artifact identities must be sorted and unique")

    def reward_identity(self, artifact_ref: str) -> FrozenMapping:
        if not isinstance(artifact_ref, str) or not artifact_ref:
            raise ValueError("artifact_ref must be non-empty")
        for bound_ref, identity in self.reward_artifact_identities:
            if bound_ref == artifact_ref:
                return identity
        raise KeyError(f"unknown reward artifact identity {artifact_ref!r}")


class ArtifactIdentityResolver(Protocol):
    """Read-only environment resolver; it must not construct components."""

    def resolve_artifact_identities(
        self,
        request: ArtifactIdentityRequest,
    ) -> ArtifactIdentityResolution:
        """Return exact typed artifact evidence for one resolved graph."""


@dataclass(frozen=True, slots=True)
class StaticPreflightResult:
    """Schema, descriptors, and typed ports resolved without artifact I/O."""

    resolved: ResolvedRecipe

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedRecipe):
            raise TypeError("resolved must be a ResolvedRecipe")

    @property
    def status(self) -> str:
        return self.resolved.compatibility.status

    @property
    def can_materialize(self) -> bool:
        return self.status != "invalid"


@dataclass(frozen=True, slots=True)
class EnvironmentPreflightResult:
    """Artifact-locked recipe produced without loading tensors/components."""

    static: StaticPreflightResult
    materialized: MaterializedRecipe
    artifact_locations: ArtifactLocations
    source_locations: SourceLocationBinding
    component_artifact_bindings: ComponentArtifactBindingSet
    component_load_plan: ComponentLoadPlan

    def __post_init__(self) -> None:
        if not isinstance(self.static, StaticPreflightResult):
            raise TypeError("static must be a StaticPreflightResult")
        if not isinstance(self.materialized, MaterializedRecipe):
            raise TypeError("materialized must be a MaterializedRecipe")
        if self.materialized.resolved is not self.static.resolved:
            raise ValueError("environment preflight must not replace ResolvedRecipe")
        if not isinstance(self.artifact_locations, ArtifactLocations):
            raise TypeError("artifact_locations must be ArtifactLocations")
        if not isinstance(self.source_locations, SourceLocationBinding):
            raise TypeError("source_locations must be SourceLocationBinding")
        observed_source_content = self.source_locations.to_content_binding(
            self.static.resolved.source_plan
        )
        if observed_source_content != self.materialized.source_content_binding:
            raise ValueError(
                "source launch locations differ from MaterializedRecipe content"
            )
        if not isinstance(
            self.component_artifact_bindings,
            ComponentArtifactBindingSet,
        ):
            raise TypeError(
                "component_artifact_bindings must be ComponentArtifactBindingSet"
            )
        if not isinstance(self.component_load_plan, ComponentLoadPlan):
            raise TypeError("component_load_plan must be ComponentLoadPlan")
        if self.component_artifact_bindings.recipe_id != self.materialized.recipe_id:
            raise ValueError("G1 bindings differ from MaterializedRecipe identity")
        if self.component_load_plan.expected_recipe_id != self.materialized.recipe_id:
            raise ValueError("component load plan differs from MaterializedRecipe")
        if (
            self.component_load_plan.expected_binding_set_id
            != self.component_artifact_bindings.binding_set_id
        ):
            raise ValueError("component load plan differs from the G1 binding set")
        if self.component_load_plan.slots != self.component_artifact_bindings.slots:
            raise ValueError("component load plan does not exactly cover G1 slots")


@dataclass(frozen=True, slots=True)
class RuntimeFacts:
    """Launch-only facts that are forbidden from MaterializedRecipe identity."""

    distribution_mode: str
    rank: int
    local_rank: int
    world_size: int
    device: str
    precision: str
    backend: str | None
    extra: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if self.distribution_mode not in {"single", "ddp", "fsdp", "deepspeed"}:
            raise ValueError("unsupported runtime distribution_mode")
        for field_name in ("rank", "local_rank", "world_size"):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if not 0 <= self.local_rank < self.world_size:
            raise ValueError("local_rank must satisfy 0 <= local_rank < world_size")
        if self.distribution_mode == "single" and (
            self.rank != 0 or self.local_rank != 0 or self.world_size != 1
        ):
            raise ValueError("single runtime requires rank=local_rank=0, world_size=1")
        for field_name in ("device", "precision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend
        ):
            raise ValueError("backend must be a non-empty string or None")
        if not isinstance(self.extra, FrozenMapping):
            raise TypeError("extra must be a FrozenMapping")

    def to_payload(self) -> dict[str, Any]:
        return {
            "distribution_mode": self.distribution_mode,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": self.device,
            "precision": self.precision,
            "backend": self.backend,
            "extra": to_plain_dict(self.extra),
        }

    def resume_compatibility_payload(self) -> dict[str, object]:
        """Return only topology facts that constrain checkpoint compatibility.

        Per-process rank/local-rank and ``extra`` remain launch audit facts.
        Device/backend, precision, distribution mode, and world size constrain
        the prepared runtime and therefore remain resume-compatible facts.
        """

        return {
            "distribution_mode": self.distribution_mode,
            "world_size": self.world_size,
            "device": self.device,
            "precision": self.precision,
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBindInput:
    """Runtime binding request with explicit cross-rank recipe consensus."""

    environment: EnvironmentPreflightResult
    runtime_facts: RuntimeFacts
    peer_recipe_ids: tuple[str, ...]
    launch_audit: FrozenMapping = _EMPTY_LAUNCH_AUDIT

    def __post_init__(self) -> None:
        if not isinstance(self.environment, EnvironmentPreflightResult):
            raise TypeError("environment must be an EnvironmentPreflightResult")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if type(self.peer_recipe_ids) is not tuple or not self.peer_recipe_ids:
            raise ValueError("peer_recipe_ids must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.peer_recipe_ids):
            raise ValueError("peer_recipe_ids must contain non-empty strings")
        if len(self.peer_recipe_ids) != self.runtime_facts.world_size:
            raise ValueError("peer_recipe_ids must contain one id per runtime rank")
        if not isinstance(self.launch_audit, FrozenMapping):
            raise TypeError("launch_audit must be a FrozenMapping")


@dataclass(frozen=True, slots=True)
class RuntimeBindResult:
    """Separate launch identity referencing, never mutating, one recipe id."""

    recipe_id: str
    launch_id: str
    runtime_facts: RuntimeFacts
    launch_audit_id: str = _EMPTY_LAUNCH_AUDIT_ID
    launch_audit: FrozenMapping = _EMPTY_LAUNCH_AUDIT

    def __post_init__(self) -> None:
        if (
            not isinstance(self.recipe_id, str)
            or _MATERIALIZED_RECIPE_ID.fullmatch(self.recipe_id) is None
        ):
            raise ValueError("recipe_id must be a materialized-recipe.v2 identity")
        for field_name in ("launch_id", "launch_audit_id"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if self.launch_id != runtime_launch_payload_id(
            self.recipe_id,
            self.runtime_facts,
        ):
            raise ValueError(
                "launch_id differs from recipe/runtime compatibility payload"
            )
        if not isinstance(self.launch_audit, FrozenMapping):
            raise TypeError("launch_audit must be a FrozenMapping")
        observed_audit_id = hashlib.sha256(
            canonical_json_text(to_plain_dict(self.launch_audit)).encode("utf-8")
        ).hexdigest()
        if observed_audit_id != self.launch_audit_id:
            raise ValueError("launch_audit_id differs from launch_audit payload")

    def launch_manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "runtime_bind",
            "recipe_id": self.recipe_id,
            "launch_id": self.launch_id,
            "runtime_compatibility": (
                self.runtime_facts.resume_compatibility_payload()
            ),
            "runtime_facts": self.runtime_facts.to_payload(),
            "launch_audit_id": self.launch_audit_id,
            "launch_audit": to_plain_dict(self.launch_audit),
        }


@dataclass(frozen=True, slots=True)
class RuntimeGraphBindInput:
    """G3 facts observed after component construction and model preparation."""

    environment: EnvironmentPreflightResult
    launch: RuntimeBindResult
    runtime_bound_contracts: tuple[tuple[str, RuntimeBoundContract], ...]
    trainable_topology_id: str
    prepared_component_names: tuple[str, ...]
    execution_transform_plan_id: str
    resource_plan_id: str
    verified_fields: FrozenMapping
    bound_reward_resource_ids: FrozenMapping = field(default_factory=FrozenMapping)
    peer_bound_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.environment, EnvironmentPreflightResult):
            raise TypeError("environment must be EnvironmentPreflightResult")
        if not isinstance(self.launch, RuntimeBindResult):
            raise TypeError("launch must be RuntimeBindResult")
        if self.launch.recipe_id != self.environment.materialized.recipe_id:
            raise ValueError("launch and environment recipe identities differ")
        contracts = self.runtime_bound_contracts
        if type(contracts) is not tuple or not contracts:
            raise ValueError("runtime_bound_contracts must be a non-empty tuple")
        slots: list[str] = []
        for item in contracts:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "runtime_bound_contracts must contain "
                    "(slot, RuntimeBoundContract) pairs"
                )
            slot, contract = item
            if not isinstance(slot, str) or not slot:
                raise ValueError("runtime bound contract slots must be non-empty")
            if not isinstance(contract, RuntimeBoundContract):
                raise TypeError(
                    "runtime_bound_contracts values must be RuntimeBoundContract"
                )
            if not contract.is_declaration_bound:
                raise ValueError(
                    "runtime graph binding rejects legacy compatibility contracts"
                )
            try:
                expected_binding = self.environment.component_artifact_bindings.binding(
                    slot
                )
            except KeyError as exc:
                raise ValueError(
                    f"runtime contract slot {slot!r} is absent from environment G1"
                ) from exc
            if contract.artifact != expected_binding:
                raise ValueError(
                    f"runtime contract for {slot!r} does not reference the exact "
                    "environment G1 binding"
                )
            slots.append(slot)
        if tuple(slots) != tuple(sorted(set(slots))):
            raise ValueError(
                "runtime_bound_contracts must be sorted by unique component slot"
            )
        for name in (
            "trainable_topology_id",
            "execution_transform_plan_id",
            "resource_plan_id",
        ):
            _runtime_digest(name, getattr(self, name))
        if type(self.prepared_component_names) is not tuple or not (
            self.prepared_component_names
        ):
            raise ValueError("prepared_component_names must be a non-empty tuple")
        if self.prepared_component_names != tuple(
            sorted(set(self.prepared_component_names))
        ):
            raise ValueError("prepared_component_names must be sorted and unique")
        if not isinstance(self.verified_fields, FrozenMapping) or not (
            self.verified_fields
        ):
            raise ValueError("verified_fields must be a non-empty FrozenMapping")
        if not isinstance(self.bound_reward_resource_ids, FrozenMapping):
            raise TypeError("bound_reward_resource_ids must be a FrozenMapping")
        for resource_spec_id, bound_id in self.bound_reward_resource_ids.items():
            if not isinstance(resource_spec_id, str) or not resource_spec_id:
                raise ValueError("reward resource spec ids must be non-empty strings")
            _runtime_digest(
                f"bound reward resource id for {resource_spec_id}",
                bound_id,
            )
        if type(self.peer_bound_contract_ids) is not tuple:
            raise TypeError("peer_bound_contract_ids must be a tuple")
        for value in self.peer_bound_contract_ids:
            _runtime_digest("peer bound contract id", value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "recipe_id": self.environment.materialized.recipe_id,
            "component_bound_contract_ids": to_plain_dict(
                self.component_bound_contract_ids
            ),
            "trainable_topology_id": self.trainable_topology_id,
            "prepared_component_names": list(self.prepared_component_names),
            "execution_transform_plan_id": self.execution_transform_plan_id,
            "resource_plan_id": self.resource_plan_id,
            "bound_reward_resource_ids": to_plain_dict(self.bound_reward_resource_ids),
            "verified_fields": to_plain_dict(self.verified_fields),
        }

    @property
    def component_bound_contract_ids(self) -> FrozenMapping:
        """Project canonical IDs from the already validated typed G3 contracts."""

        return FrozenMapping(
            (slot, contract.contract_id)
            for slot, contract in self.runtime_bound_contracts
        )


@dataclass(frozen=True, slots=True)
class RuntimeGraphBindResult:
    """Canonical aggregate G3 contract, separate from launch/runtime facts."""

    recipe_id: str
    launch_id: str
    bound_contract_id: str
    component_bound_contract_ids: FrozenMapping
    trainable_topology_id: str
    prepared_component_names: tuple[str, ...]
    execution_transform_plan_id: str
    resource_plan_id: str
    verified_fields: FrozenMapping
    bound_reward_resource_ids: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.recipe_id, str)
            or _MATERIALIZED_RECIPE_ID.fullmatch(self.recipe_id) is None
        ):
            raise ValueError("recipe_id must be a materialized-recipe.v2 identity")
        for name in (
            "launch_id",
            "bound_contract_id",
            "trainable_topology_id",
            "execution_transform_plan_id",
            "resource_plan_id",
        ):
            _runtime_digest(name, getattr(self, name))
        if not isinstance(self.component_bound_contract_ids, FrozenMapping):
            raise TypeError("component_bound_contract_ids must be FrozenMapping")
        if not self.component_bound_contract_ids:
            raise ValueError("component_bound_contract_ids must not be empty")
        for slot, contract_id in self.component_bound_contract_ids.items():
            if not isinstance(slot, str) or not slot:
                raise ValueError("component bound contract slots must be non-empty")
            _runtime_digest(f"component bound contract id for {slot}", contract_id)
        if type(self.prepared_component_names) is not tuple or not (
            self.prepared_component_names
        ):
            raise ValueError("prepared_component_names must be a non-empty tuple")
        if self.prepared_component_names != tuple(
            sorted(set(self.prepared_component_names))
        ):
            raise ValueError("prepared_component_names must be sorted and unique")
        if not isinstance(self.verified_fields, FrozenMapping) or not (
            self.verified_fields
        ):
            raise ValueError("verified_fields must be a non-empty FrozenMapping")
        if not isinstance(self.bound_reward_resource_ids, FrozenMapping):
            raise TypeError("bound_reward_resource_ids must be FrozenMapping")
        for resource_spec_id, bound_id in self.bound_reward_resource_ids.items():
            if not isinstance(resource_spec_id, str) or not resource_spec_id:
                raise ValueError("reward resource spec ids must be non-empty strings")
            _runtime_digest(
                f"bound reward resource id for {resource_spec_id}",
                bound_id,
            )
        if self.bound_contract_id != self.canonical_bound_contract_id:
            raise ValueError(
                "bound_contract_id differs from canonical runtime graph payload"
            )

    def canonical_payload(self) -> dict[str, Any]:
        """Reconstruct the exact launch-independent payload bound by G3."""

        return {
            "schema_version": 1,
            "recipe_id": self.recipe_id,
            "component_bound_contract_ids": to_plain_dict(
                self.component_bound_contract_ids
            ),
            "trainable_topology_id": self.trainable_topology_id,
            "prepared_component_names": list(self.prepared_component_names),
            "execution_transform_plan_id": self.execution_transform_plan_id,
            "resource_plan_id": self.resource_plan_id,
            "bound_reward_resource_ids": to_plain_dict(self.bound_reward_resource_ids),
            "verified_fields": to_plain_dict(self.verified_fields),
        }

    @property
    def canonical_bound_contract_id(self) -> str:
        """Return the digest independently reproducible from this result."""

        return runtime_graph_payload_id(self.canonical_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.canonical_payload(),
            "launch_id": self.launch_id,
            "bound_contract_id": self.bound_contract_id,
        }


def _runtime_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def runtime_graph_payload_id(payload: Mapping[str, Any]) -> str:
    """Internal canonical digest helper shared with the G3 binder."""

    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()


def runtime_launch_payload_id(
    recipe_id: str,
    runtime_facts: RuntimeFacts,
) -> str:
    """Return the stable launch id over resume-compatible runtime facts only."""

    if (
        not isinstance(recipe_id, str)
        or _MATERIALIZED_RECIPE_ID.fullmatch(recipe_id) is None
    ):
        raise ValueError("recipe_id must be a materialized-recipe.v2 identity")
    if not isinstance(runtime_facts, RuntimeFacts):
        raise TypeError("runtime_facts must be RuntimeFacts")
    payload = {
        "schema_version": 1,
        "recipe_id": recipe_id,
        "runtime_compatibility": runtime_facts.resume_compatibility_payload(),
    }
    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()
