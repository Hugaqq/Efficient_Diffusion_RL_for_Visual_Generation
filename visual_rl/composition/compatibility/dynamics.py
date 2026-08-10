"""Import-safe compatibility matching for one resolved Dynamics declaration.

The projection layer may select a Dynamics alias and configuration, but an
alias is not a capability proof.  This module validates the resolved contract
against the model scheduler ABI, the recipe likelihood semantic, and the
coarse algorithm requirements without importing a runtime implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from visual_rl.core.contracts import (
    AlgorithmRequirements,
    DynamicsContract,
    LikelihoodSemantics,
    ModelDescriptorContract,
    ReferenceRequirement,
)
from visual_rl.core.contracts.composition import CapabilityMismatch

__all__ = (
    "DynamicsCompatibilityMatch",
    "match_model_algorithm_dynamics",
)


@dataclass(frozen=True, slots=True)
class DynamicsCompatibilityMatch:
    """Structured result of static model/algorithm/Dynamics unification."""

    mismatches: tuple[CapabilityMismatch, ...]
    bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.mismatches) is not tuple or any(
            not isinstance(item, CapabilityMismatch) for item in self.mismatches
        ):
            raise TypeError("mismatches must contain CapabilityMismatch values")
        if type(self.bindings) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
            for item in self.bindings
        ):
            raise TypeError("bindings must contain non-empty string pairs")
        if self.bindings != tuple(sorted(set(self.bindings))):
            raise ValueError("bindings must be sorted and unique")

    @property
    def is_compatible(self) -> bool:
        return not self.mismatches


def match_model_algorithm_dynamics(
    *,
    model: ModelDescriptorContract,
    dynamics: DynamicsContract,
    algorithm: AlgorithmRequirements,
    likelihood_semantics: LikelihoodSemantics,
    beta: float,
) -> DynamicsCompatibilityMatch:
    """Match a resolved Dynamics contract to both mandatory public axes."""

    _validate_model_dynamics_inputs(model, dynamics, likelihood_semantics)
    if not isinstance(algorithm, AlgorithmRequirements):
        raise TypeError("algorithm must be an AlgorithmRequirements")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) < 0.0
    ):
        raise ValueError("beta must be finite and non-negative")
    resolved_beta = float(beta)

    mismatches: list[CapabilityMismatch] = []
    bindings: list[tuple[str, str]] = []

    _match_algorithm_model_ports(mismatches, bindings, model, algorithm)
    _match_algorithm_dynamics_ports(
        mismatches,
        bindings,
        model=model,
        dynamics=dynamics,
        algorithm=algorithm,
        likelihood_semantics=likelihood_semantics,
        beta=resolved_beta,
    )
    _match_model_dynamics_ports(
        mismatches,
        bindings,
        model=model,
        dynamics=dynamics,
        likelihood_semantics=likelihood_semantics,
    )

    return DynamicsCompatibilityMatch(
        mismatches=tuple(mismatches),
        bindings=tuple(sorted(set(bindings))),
    )


def _validate_model_dynamics_inputs(
    model: object,
    dynamics: object,
    likelihood_semantics: object,
) -> None:
    if not isinstance(model, ModelDescriptorContract):
        raise TypeError("model must be a ModelDescriptorContract")
    if not isinstance(dynamics, DynamicsContract):
        raise TypeError("dynamics must be a DynamicsContract")
    if not isinstance(likelihood_semantics, LikelihoodSemantics):
        raise TypeError("likelihood_semantics must be a LikelihoodSemantics")


def _match_model_dynamics_ports(
    mismatches: list[CapabilityMismatch],
    bindings: list[tuple[str, str]],
    *,
    model: ModelDescriptorContract,
    dynamics: DynamicsContract,
    likelihood_semantics: LikelihoodSemantics,
) -> None:

    for producer, consumer, provided, accepted in (
        (
            "model.latent_layouts",
            "dynamics.accepted_latent_layouts",
            model.latent_layouts,
            dynamics.accepted_latent_layouts,
        ),
        (
            "model.prediction_types",
            "dynamics.accepted_prediction_types",
            model.prediction_types,
            dynamics.accepted_prediction_types,
        ),
        (
            "model.time_coordinates",
            "dynamics.accepted_time_coordinates",
            model.time_coordinates,
            dynamics.accepted_time_coordinates,
        ),
    ):
        _intersect(mismatches, bindings, producer, consumer, provided, accepted)
    _match_scheduler_abi(mismatches, bindings, model=model, dynamics=dynamics)
    if likelihood_semantics not in dynamics.supported_likelihoods:
        _mismatch(
            mismatches,
            code="likelihood_semantics_mismatch",
            producer_field="dynamics.supported_likelihoods",
            consumer_field="recipe.likelihood_semantics",
            required=dynamics.supported_likelihoods,
            provided=likelihood_semantics,
            hint="choose a declared likelihood semantic or different Dynamics",
        )


def _match_algorithm_model_ports(
    mismatches: list[CapabilityMismatch],
    bindings: list[tuple[str, str]],
    model: ModelDescriptorContract,
    algorithm: AlgorithmRequirements,
) -> None:
    for producer, consumer, provided, accepted in (
        (
            "model.tasks",
            "algorithm.accepted_tasks",
            model.tasks,
            algorithm.accepted_tasks,
        ),
        (
            "model.output_media",
            "algorithm.accepted_media",
            model.output_media,
            algorithm.accepted_media,
        ),
        (
            "model.latent_layouts",
            "algorithm.accepted_latent_layouts",
            model.latent_layouts,
            algorithm.accepted_latent_layouts,
        ),
        (
            "model.prediction_types",
            "algorithm.accepted_prediction_types",
            model.prediction_types,
            algorithm.accepted_prediction_types,
        ),
        (
            "model.time_coordinates",
            "algorithm.accepted_time_coordinates",
            model.time_coordinates,
            algorithm.accepted_time_coordinates,
        ),
    ):
        _intersect(mismatches, bindings, producer, consumer, provided, accepted)


def _match_algorithm_dynamics_ports(
    mismatches: list[CapabilityMismatch],
    bindings: list[tuple[str, str]],
    *,
    model: ModelDescriptorContract,
    dynamics: DynamicsContract,
    algorithm: AlgorithmRequirements,
    likelihood_semantics: LikelihoodSemantics,
    beta: float,
) -> None:
    if algorithm.transition_kind is not dynamics.transition_kind:
        _mismatch(
            mismatches,
            code="algorithm_transition_kind_mismatch",
            producer_field="dynamics.transition_kind",
            consumer_field="algorithm.transition_kind",
            required=algorithm.transition_kind,
            provided=dynamics.transition_kind,
            hint="select internal Dynamics implementing the AlgorithmModule contract",
        )
    else:
        bindings.append(("algorithm.transition_kind", algorithm.transition_kind.value))

    if likelihood_semantics not in algorithm.likelihood_semantics:
        _mismatch(
            mismatches,
            code="algorithm_likelihood_semantics_mismatch",
            producer_field="recipe.likelihood_semantics",
            consumer_field="algorithm.likelihood_semantics",
            required=algorithm.likelihood_semantics,
            provided=likelihood_semantics,
            hint="select an AlgorithmModule accepting this recipe semantic",
        )

    required_reference = (
        algorithm.reference_requirement is ReferenceRequirement.ALWAYS
        or (
            algorithm.reference_requirement is ReferenceRequirement.WHEN_BETA_POSITIVE
            and beta > 0.0
        )
    )
    if algorithm.reference_requirement is ReferenceRequirement.NEVER and beta > 0.0:
        _mismatch(
            mismatches,
            code="algorithm_reference_activation_mismatch",
            producer_field="recipe.beta",
            consumer_field="algorithm.reference_required",
            required=0.0,
            provided=beta,
            hint="an algorithm forbidding reference statistics must use beta=0",
        )
    elif algorithm.reference_required is not required_reference:
        _mismatch(
            mismatches,
            code="algorithm_reference_activation_mismatch",
            producer_field="recipe.beta",
            consumer_field="algorithm.reference_required",
            required=required_reference,
            provided=algorithm.reference_required,
            hint="keep the module's typed reference activation aligned with beta",
        )
    if required_reference and model.provides_reference_policy is not True:
        _mismatch(
            mismatches,
            code="algorithm_reference_policy_missing",
            producer_field="model.provides_reference_policy",
            consumer_field="algorithm.reference_requirement",
            required=True,
            provided=model.provides_reference_policy,
            hint="provide a reference view or configure a zero-reference variant",
        )

    for payload_type in algorithm.required_condition_payload_types:
        if payload_type not in model.condition_payload_types:
            _mismatch(
                mismatches,
                code="algorithm_condition_payload_missing",
                producer_field="model.condition_payload_types",
                consumer_field="algorithm.required_condition_payload_types",
                required=payload_type,
                provided=model.condition_payload_types,
                hint="select a model port producing the required opaque condition",
            )

    transition_features = {
        "transition": True,
        "stochastic": dynamics.stochastic,
        "mean_std": dynamics.exposes_mean_std,
        "scores_arbitrary_action": dynamics.scores_arbitrary_action,
        "differentiable_log_prob": dynamics.differentiable_log_prob,
        "replayable": dynamics.replayable,
        "branchable": dynamics.branchable,
        "deterministic_ode": dynamics.supports_deterministic_ode,
    }
    for feature in algorithm.required_transition_features:
        if not transition_features.get(feature, False):
            _mismatch(
                mismatches,
                code="algorithm_transition_feature_missing",
                producer_field=f"dynamics.{feature}",
                consumer_field="algorithm.required_transition_features",
                required=True,
                provided=transition_features.get(feature, False),
                hint="select Dynamics providing the required transition port",
            )

    missing_metadata = tuple(
        field
        for field in algorithm.required_policy_metadata_fields
        if field not in dynamics.produced_policy_metadata_fields
    )
    if missing_metadata:
        _mismatch(
            mismatches,
            code="algorithm_policy_metadata_fields_missing",
            producer_field="dynamics.produced_policy_metadata_fields",
            consumer_field="algorithm.required_policy_metadata_fields",
            required=algorithm.required_policy_metadata_fields,
            provided=dynamics.produced_policy_metadata_fields,
            hint=(
                "select internal Dynamics producing every policy metadata field "
                "required by AlgorithmModule"
            ),
        )
    elif algorithm.required_policy_metadata_fields:
        bindings.append(
            (
                (
                    "dynamics.produced_policy_metadata_fields"
                    "->algorithm.required_policy_metadata_fields"
                ),
                _show(algorithm.required_policy_metadata_fields),
            )
        )


def _match_scheduler_abi(
    mismatches: list[CapabilityMismatch],
    bindings: list[tuple[str, str]],
    *,
    model: ModelDescriptorContract,
    dynamics: DynamicsContract,
) -> None:
    model_declares = model.declares_scheduler_binding
    dynamics_declares = dynamics.declares_scheduler_binding
    if not model_declares and not dynamics_declares:
        return
    if model_declares and not dynamics_declares:
        _mismatch(
            mismatches,
            code="dynamics_scheduler_binding_undeclared",
            producer_field="model.scheduler_blueprint_schema",
            consumer_field="dynamics.accepted_scheduler_blueprint_schemas",
            required="complete dynamics scheduler binding ABI",
            provided=(),
            hint="select Dynamics declaring the scheduler artifact and replay-state ABI",
        )
        return
    if dynamics_declares and not model_declares:
        _mismatch(
            mismatches,
            code="model_scheduler_binding_undeclared",
            producer_field="dynamics.accepted_scheduler_blueprint_schemas",
            consumer_field="model.scheduler_blueprint_schema",
            required="complete model scheduler binding ABI",
            provided=None,
            hint="select a model declaring its scheduler artifact and replay-state ABI",
        )
        return

    assert model.scheduler_blueprint_schema is not None
    assert model.dynamics_binding_family is not None
    assert model.schedule_coordinate is not None
    assert dynamics.produced_replay_state_schema_id is not None
    for code, producer, consumer, provided, accepted, hint in (
        (
            "scheduler_blueprint_schema_mismatch",
            "model.scheduler_blueprint_schema",
            "dynamics.accepted_scheduler_blueprint_schemas",
            model.scheduler_blueprint_schema,
            dynamics.accepted_scheduler_blueprint_schemas,
            "select Dynamics accepting the model scheduler artifact schema",
        ),
        (
            "dynamics_binding_family_mismatch",
            "model.dynamics_binding_family",
            "dynamics.accepted_model_binding_families",
            model.dynamics_binding_family,
            dynamics.accepted_model_binding_families,
            "select Dynamics implementing the model's declared binding family",
        ),
        (
            "schedule_coordinate_mismatch",
            "model.schedule_coordinate",
            "dynamics.accepted_time_coordinates",
            model.schedule_coordinate,
            dynamics.accepted_time_coordinates,
            "select Dynamics accepting the model scheduler coordinate",
        ),
        (
            "replay_state_schema_mismatch",
            "dynamics.produced_replay_state_schema_id",
            "model.accepted_replay_state_schema_ids",
            dynamics.produced_replay_state_schema_id,
            model.accepted_replay_state_schema_ids,
            "select Dynamics producing a replay state accepted by the model ABI",
        ),
    ):
        _membership(
            mismatches,
            bindings,
            code=code,
            producer_field=producer,
            consumer_field=consumer,
            provided=provided,
            accepted=accepted,
            hint=hint,
        )


def _intersect(
    mismatches: list[CapabilityMismatch],
    bindings: list[tuple[str, str]],
    producer_field: str,
    consumer_field: str,
    provided: tuple[object, ...],
    accepted: tuple[object, ...],
) -> None:
    common = tuple(item for item in provided if item in accepted)
    if not common:
        _mismatch(
            mismatches,
            code="typed_port_mismatch",
            producer_field=producer_field,
            consumer_field=consumer_field,
            required=accepted,
            provided=provided,
            hint="select components with a shared typed port",
        )
        return
    bindings.append((f"{producer_field}->{consumer_field}", _show(common)))


def _membership(
    mismatches: list[CapabilityMismatch],
    bindings: list[tuple[str, str]],
    *,
    code: str,
    producer_field: str,
    consumer_field: str,
    provided: object,
    accepted: tuple[object, ...],
    hint: str,
) -> None:
    if provided not in accepted:
        _mismatch(
            mismatches,
            code=code,
            producer_field=producer_field,
            consumer_field=consumer_field,
            required=accepted,
            provided=provided,
            hint=hint,
        )
        return
    bindings.append((f"{producer_field}->{consumer_field}", _show(provided)))


def _mismatch(
    mismatches: list[CapabilityMismatch],
    *,
    code: str,
    producer_field: str,
    consumer_field: str,
    required: object,
    provided: object,
    hint: str,
) -> None:
    mismatches.append(
        CapabilityMismatch(
            code=code,
            producer_field=producer_field,
            consumer_field=consumer_field,
            required=required,
            provided=provided,
            hint=hint,
        )
    )


def _show(value: object) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_show(item) for item in value) + "]"
    return str(value)
