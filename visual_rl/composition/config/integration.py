"""Pure model-bound Dynamics projection for the public composition axes.

This module translates model scheduler ABI plus algorithm-owned blueprint facts
into one internal Dynamics selection.  It never imports a concrete model,
algorithm runtime, recipe definition, or training backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache

from visual_rl.composition.compatibility.dynamics import (
    match_model_algorithm_dynamics,
)
from visual_rl.composition.registry import (
    ResolvedAlgorithmDeclaration,
    ResolvedComponentDeclaration,
)
from visual_rl.core.contracts import (
    AlgorithmComponentResolution,
    AlgorithmComponentRole,
    AlgorithmComponentSelection,
    AlgorithmRequirements,
    DynamicsContract,
    LikelihoodSemantics,
    ModelDescriptorContract,
    ReferenceRequirement,
    ReplayTarget,
    TrajectoryKind,
)
from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict

__all__ = (
    "DynamicsConditioningMode",
    "DynamicsIntegrationSpec",
    "DynamicsProjectionRegistry",
    "ModelBoundDynamicsProjection",
    "ModelBoundDynamicsProjectionError",
    "ModelBoundDynamicsProjector",
    "bind_model_bound_dynamics_declaration",
    "default_dynamics_projection_registry",
    "project_model_bound_dynamics",
)


class ModelBoundDynamicsProjectionError(ValueError):
    """A model ABI, algorithm blueprint, and integration policy cannot bind."""


DynamicsProjectorFunction = Callable[
    [AlgorithmRequirements, FrozenMapping, "DynamicsIntegrationSpec"],
    tuple[str, FrozenMapping],
]


class DynamicsConditioningMode(str, Enum):
    UNCONDITIONED = "unconditioned"
    CONDITIONED = "conditioned"


@dataclass(frozen=True, slots=True)
class DynamicsIntegrationSpec:
    """Recipe-owned transition facts without an internal component override."""

    conditioning: DynamicsConditioningMode
    likelihood_semantics: LikelihoodSemantics
    replay_target: ReplayTarget

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, DynamicsConditioningMode):
            raise TypeError("conditioning must be a DynamicsConditioningMode")
        if not isinstance(self.likelihood_semantics, LikelihoodSemantics):
            raise TypeError("likelihood_semantics must be a LikelihoodSemantics")
        if not isinstance(self.replay_target, ReplayTarget):
            raise TypeError("replay_target must be a ReplayTarget")
        pair = (self.likelihood_semantics, self.replay_target)
        exact_action = (
            LikelihoodSemantics.EXACT_ENV_ACTION,
            ReplayTarget.SAMPLED_ACTION,
        )
        conditioned_surrogate = (
            LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
            ReplayTarget.CONDITIONED_NEXT,
        )
        if self.conditioning is DynamicsConditioningMode.UNCONDITIONED:
            if pair != exact_action:
                raise ValueError(
                    "unconditioned Dynamics requires exact_env_action with "
                    "sampled_action replay"
                )
        elif pair not in {exact_action, conditioned_surrogate}:
            raise ValueError(
                "conditioned Dynamics requires exact sampled-action scoring or "
                "the explicit post-hook conditioned-next surrogate"
            )

    @classmethod
    def unconditioned(cls) -> DynamicsIntegrationSpec:
        return cls(
            conditioning=DynamicsConditioningMode.UNCONDITIONED,
            likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
            replay_target=ReplayTarget.SAMPLED_ACTION,
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "conditioning": self.conditioning.value,
            "likelihood_semantics": self.likelihood_semantics.value,
            "replay_target": self.replay_target.value,
        }

    @property
    def integration_id(self) -> str:
        return canonical_identity("dynamics-integration-spec.v1", self.to_payload())


@dataclass(frozen=True, slots=True)
class ModelBoundDynamicsProjection:
    """Canonical resolver input selected from typed cross-axis facts."""

    component_id: str
    implementation_family: str
    model_binding_family: str
    params: FrozenMapping
    integration_id: str

    def __post_init__(self) -> None:
        for name in (
            "component_id",
            "implementation_family",
            "model_binding_family",
            "integration_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be canonical text")
        if not isinstance(self.params, FrozenMapping):
            raise TypeError("params must be a FrozenMapping")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "component_id": self.component_id,
            "implementation_family": self.implementation_family,
            "model_binding_family": self.model_binding_family,
            "params": to_plain_dict(self.params),
            "integration_id": self.integration_id,
        }

    @property
    def projection_id(self) -> str:
        return canonical_identity(
            "model-bound-dynamics-projection.v1",
            self.to_payload(),
        )


@dataclass(frozen=True, slots=True)
class ModelBoundDynamicsProjector:
    """One pure plugin seam from typed cross-axis facts to Dynamics config."""

    model_binding_family: str
    implementation_family: str
    project: DynamicsProjectorFunction

    def __post_init__(self) -> None:
        for name in ("model_binding_family", "implementation_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be canonical text")
        if not callable(self.project):
            raise TypeError("project must be callable")


@dataclass(frozen=True, slots=True)
class DynamicsProjectionRegistry:
    """Immutable registry replacing model-family branches in the compiler."""

    projectors: tuple[ModelBoundDynamicsProjector, ...] = ()

    def __post_init__(self) -> None:
        if type(self.projectors) is not tuple or any(
            not isinstance(item, ModelBoundDynamicsProjector)
            for item in self.projectors
        ):
            raise TypeError(
                "projectors must be a tuple of ModelBoundDynamicsProjector values"
            )
        keys = tuple(
            (item.model_binding_family, item.implementation_family)
            for item in self.projectors
        )
        if len(keys) != len(set(keys)):
            raise ValueError("Dynamics projector registry keys must be unique")
        object.__setattr__(
            self,
            "projectors",
            tuple(
                sorted(
                    self.projectors,
                    key=lambda item: (
                        item.model_binding_family,
                        item.implementation_family,
                    ),
                )
            ),
        )

    def register(
        self,
        projector: ModelBoundDynamicsProjector,
    ) -> DynamicsProjectionRegistry:
        if not isinstance(projector, ModelBoundDynamicsProjector):
            raise TypeError("projector must be a ModelBoundDynamicsProjector")
        key = (projector.model_binding_family, projector.implementation_family)
        if any(
            (item.model_binding_family, item.implementation_family) == key
            for item in self.projectors
        ):
            raise ValueError(f"Dynamics projector key is already registered: {key!r}")
        return DynamicsProjectionRegistry((*self.projectors, projector))

    def resolve(
        self,
        *,
        model_binding_family: str,
        implementation_family: str,
    ) -> ModelBoundDynamicsProjector:
        if not isinstance(model_binding_family, str) or not model_binding_family:
            raise ValueError("model_binding_family must be non-empty text")
        if not isinstance(implementation_family, str) or not implementation_family:
            raise ValueError("implementation_family must be non-empty text")
        match = next(
            (
                item
                for item in self.projectors
                if item.model_binding_family == model_binding_family
                and item.implementation_family == implementation_family
            ),
            None,
        )
        if match is not None:
            return match
        known_implementations = {
            item.implementation_family for item in self.projectors
        }
        if implementation_family not in known_implementations:
            raise ModelBoundDynamicsProjectionError(
                "unsupported algorithm Dynamics implementation family"
            )
        known_models = {item.model_binding_family for item in self.projectors}
        if model_binding_family not in known_models:
            raise ModelBoundDynamicsProjectionError(
                "unsupported model Dynamics binding family "
                f"{model_binding_family!r}"
            )
        raise ModelBoundDynamicsProjectionError(
            "model Dynamics binding family does not support the algorithm "
            f"implementation family: model={model_binding_family!r}, "
            f"algorithm={implementation_family!r}"
        )


@cache
def default_dynamics_projection_registry() -> DynamicsProjectionRegistry:
    """Return the immutable built-in projector set without runtime imports."""

    return DynamicsProjectionRegistry(
        (
            ModelBoundDynamicsProjector(
                model_binding_family="sd3.flow-sde.v1",
                implementation_family="flow-sde",
                project=_project_sd3,
            ),
            ModelBoundDynamicsProjector(
                model_binding_family="wan.flow-sde.v1",
                implementation_family="flow-sde",
                project=_project_wan,
            ),
        )
    )


def bind_model_bound_dynamics_declaration(
    *,
    projection: ModelBoundDynamicsProjection,
    declaration: ResolvedComponentDeclaration,
    model: ResolvedComponentDeclaration,
    algorithm: ResolvedAlgorithmDeclaration,
    integration: DynamicsIntegrationSpec,
    projector_registry: DynamicsProjectionRegistry | None = None,
) -> AlgorithmComponentSelection:
    """Validate and retain one exact resolved model-bound declaration."""

    if not isinstance(projection, ModelBoundDynamicsProjection):
        raise TypeError("projection must be a ModelBoundDynamicsProjection")
    if not isinstance(declaration, ResolvedComponentDeclaration):
        raise TypeError("declaration must be a ResolvedComponentDeclaration")
    expected_projection = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=integration,
        projector_registry=projector_registry,
    )
    if projection != expected_projection:
        raise ModelBoundDynamicsProjectionError(
            "Dynamics projection differs from the supplied model/algorithm/integration"
        )
    if declaration.kind != "dynamics":
        raise ValueError("declaration must resolve the Dynamics role")
    if declaration.alias != projection.component_id:
        raise ValueError("resolved Dynamics alias differs from the projection")
    dynamics_contract = declaration.declared_contract.dynamics
    if not isinstance(dynamics_contract, DynamicsContract):
        raise TypeError("resolved declaration has no DynamicsContract")
    model_contract = model.declared_contract.model
    if not isinstance(model_contract, ModelDescriptorContract):
        raise TypeError("resolved model declaration has no ModelDescriptorContract")
    match = match_model_algorithm_dynamics(
        model=model_contract,
        dynamics=dynamics_contract,
        algorithm=algorithm.requirements,
        likelihood_semantics=integration.likelihood_semantics,
        beta=algorithm.blueprint.beta,
    )
    if not match.is_compatible:
        codes = ", ".join(item.code for item in match.mismatches)
        raise ModelBoundDynamicsProjectionError(
            f"resolved Dynamics contract is incompatible: {codes}"
        )
    return AlgorithmComponentSelection(
        role=AlgorithmComponentRole.DYNAMICS,
        selected_component_id=declaration.alias,
        component_declaration_id=declaration.declaration_id,
        implementation_family=projection.implementation_family,
        resolution=AlgorithmComponentResolution.MODEL_BOUND,
    )


def project_model_bound_dynamics(
    *,
    model: ResolvedComponentDeclaration,
    algorithm: ResolvedAlgorithmDeclaration,
    integration: DynamicsIntegrationSpec,
    projector_registry: DynamicsProjectionRegistry | None = None,
) -> ModelBoundDynamicsProjection:
    """Select one Dynamics resolver input without config or recipe duck typing."""

    if not isinstance(model, ResolvedComponentDeclaration) or model.kind != "model":
        raise TypeError("model must be a resolved model declaration")
    if not isinstance(algorithm, ResolvedAlgorithmDeclaration):
        raise TypeError("algorithm must be a ResolvedAlgorithmDeclaration")
    if not isinstance(integration, DynamicsIntegrationSpec):
        raise TypeError("integration must be a DynamicsIntegrationSpec")
    model_contract = model.declared_contract.model
    if not isinstance(model_contract, ModelDescriptorContract):
        raise TypeError("resolved model declaration has no ModelDescriptorContract")
    requirements = algorithm.requirements
    if not isinstance(requirements, AlgorithmRequirements):
        raise TypeError("resolved algorithm has no AlgorithmRequirements")
    slot = algorithm.blueprint.slot(AlgorithmComponentRole.DYNAMICS)
    registry = (
        default_dynamics_projection_registry()
        if projector_registry is None
        else projector_registry
    )
    if not isinstance(registry, DynamicsProjectionRegistry):
        raise TypeError(
            "projector_registry must be a DynamicsProjectionRegistry or None"
        )
    if integration.likelihood_semantics not in requirements.likelihood_semantics:
        raise ModelBoundDynamicsProjectionError(
            "integration likelihood is not accepted by the algorithm"
        )
    _validate_reference_semantics(
        beta=algorithm.blueprint.beta,
        requirements=requirements,
        model=model_contract,
    )

    binding_family = model_contract.dynamics_binding_family
    projector = registry.resolve(
        model_binding_family=binding_family,
        implementation_family=slot.implementation_family,
    )
    component_id, params = projector.project(requirements, slot.params, integration)
    if not isinstance(component_id, str) or not component_id:
        raise ModelBoundDynamicsProjectionError(
            "Dynamics projector returned an invalid component id"
        )
    if not isinstance(params, FrozenMapping):
        raise ModelBoundDynamicsProjectionError(
            "Dynamics projector returned non-frozen params"
        )
    return ModelBoundDynamicsProjection(
        component_id=component_id,
        implementation_family=slot.implementation_family,
        model_binding_family=binding_family,
        params=params,
        integration_id=integration.integration_id,
    )


def _validate_reference_semantics(
    *,
    beta: float,
    requirements: AlgorithmRequirements,
    model: ModelDescriptorContract,
) -> None:
    reference_requirement = requirements.reference_requirement
    if reference_requirement is ReferenceRequirement.NEVER and beta != 0.0:
        raise ModelBoundDynamicsProjectionError(
            "an algorithm that forbids reference statistics must use beta=0"
        )
    expected = reference_requirement is ReferenceRequirement.ALWAYS or (
        reference_requirement is ReferenceRequirement.WHEN_BETA_POSITIVE and beta > 0.0
    )
    if requirements.reference_required is not expected:
        raise ModelBoundDynamicsProjectionError(
            "algorithm reference requirement differs from blueprint beta"
        )
    if expected and not model.provides_reference_policy:
        raise ModelBoundDynamicsProjectionError(
            "algorithm beta/reference semantics require a model reference policy"
        )


def _project_sd3(
    requirements: AlgorithmRequirements,
    slot_params: FrozenMapping,
    integration: DynamicsIntegrationSpec,
) -> tuple[str, FrozenMapping]:
    trajectory = requirements.trajectory_kind
    if integration.conditioning is not DynamicsConditioningMode.UNCONDITIONED:
        raise ModelBoundDynamicsProjectionError(
            "the SD3 Dynamics binding does not accept conditioned transitions"
        )
    if trajectory not in {TrajectoryKind.FULL, TrajectoryKind.BRANCHING}:
        raise ModelBoundDynamicsProjectionError(
            "the SD3 Dynamics binding does not implement single-step rectification"
        )
    return "flow-sde", slot_params


def _project_wan(
    requirements: AlgorithmRequirements,
    slot_params: FrozenMapping,
    integration: DynamicsIntegrationSpec,
) -> tuple[str, FrozenMapping]:
    trajectory = requirements.trajectory_kind
    required_policy_metadata = requirements.required_policy_metadata_fields
    if trajectory is TrajectoryKind.BRANCHING:
        raise ModelBoundDynamicsProjectionError(
            "the Wan Dynamics binding is not branchable"
        )
    if trajectory is TrajectoryKind.SINGLE_STEP:
        if integration.conditioning is not DynamicsConditioningMode.UNCONDITIONED:
            raise ModelBoundDynamicsProjectionError(
                "single-step Wan Dynamics does not accept a conditioned hook"
            )
        if "rectification_coefficient" not in required_policy_metadata:
            raise ModelBoundDynamicsProjectionError(
                "single-step Wan Dynamics requires rectification metadata"
            )
        profile = "flash"
    elif trajectory is TrajectoryKind.FULL:
        profile = (
            "conditioned"
            if integration.conditioning is DynamicsConditioningMode.CONDITIONED
            else "standard"
        )
    else:  # pragma: no cover - closed enum, retained as a fail-closed boundary
        raise ModelBoundDynamicsProjectionError("unsupported trajectory kind")
    params = FrozenMapping(
        {
            "profile": profile,
            "likelihood_semantics": integration.likelihood_semantics.value,
            "replay_target": integration.replay_target.value,
            "stochastic_sampling": True,
        }
    )
    if slot_params and slot_params != params:
        raise ModelBoundDynamicsProjectionError(
            "algorithm Dynamics slot params differ from the model-bound projection"
        )
    return "wan-flow-sde", params
