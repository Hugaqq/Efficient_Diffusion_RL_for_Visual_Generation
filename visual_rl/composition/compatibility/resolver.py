"""Pure capability matching owned by the composition layer."""

from __future__ import annotations

from visual_rl.composition.compatibility.errors import ModelAlgorithmMismatch
from visual_rl.core.contracts.algorithm import AlgorithmRequirements
from visual_rl.core.contracts.composition import (
    BoundPolicyCapabilities,
    CapabilityMismatch,
    ModelAlgorithmBinding,
)
from visual_rl.core.identity import to_identity_value

__all__ = ("bind_model_algorithm",)


def bind_model_algorithm(
    model: BoundPolicyCapabilities,
    algorithm: AlgorithmRequirements,
) -> ModelAlgorithmBinding:
    """Unify static capabilities or fail before heavyweight construction."""

    if not isinstance(model, BoundPolicyCapabilities):
        raise TypeError("model must be BoundPolicyCapabilities")
    if not isinstance(algorithm, AlgorithmRequirements):
        raise TypeError("algorithm must be AlgorithmRequirements")
    selections, mismatches = _match(model, algorithm)
    if mismatches:
        raise ModelAlgorithmMismatch(mismatches)
    return ModelAlgorithmBinding(model, algorithm, selections)


def _match(
    model: BoundPolicyCapabilities,
    algorithm: AlgorithmRequirements,
) -> tuple[tuple[tuple[str, str], ...], tuple[CapabilityMismatch, ...]]:
    selections: list[tuple[str, str]] = []
    mismatches: list[CapabilityMismatch] = []

    def intersect(
        code: str,
        producer_field: str,
        consumer_field: str,
        provided: tuple[object, ...],
        accepted: tuple[object, ...],
    ) -> None:
        common = tuple(item for item in provided if item in accepted)
        if not common:
            mismatches.append(
                CapabilityMismatch(
                    code=code,
                    producer_field=producer_field,
                    consumer_field=consumer_field,
                    required=to_identity_value(accepted),
                    provided=to_identity_value(provided),
                    hint=f"select a policy runtime accepted by {consumer_field}",
                )
            )
            return
        selected = min(common, key=lambda item: str(to_identity_value(item)))
        selections.append((consumer_field, str(to_identity_value(selected))))

    intersect(
        "task_mismatch",
        "model.tasks",
        "algorithm.accepted_tasks",
        model.tasks,
        algorithm.accepted_tasks,
    )
    intersect(
        "media_mismatch",
        "model.output_media",
        "algorithm.accepted_media",
        model.output_media,
        algorithm.accepted_media,
    )
    intersect(
        "latent_layout_mismatch",
        "model.latent_layouts",
        "algorithm.accepted_latent_layouts",
        model.latent_layouts,
        algorithm.accepted_latent_layouts,
    )
    intersect(
        "prediction_type_mismatch",
        "model.prediction_types",
        "algorithm.accepted_prediction_types",
        model.prediction_types,
        algorithm.accepted_prediction_types,
    )
    intersect(
        "time_coordinate_mismatch",
        "model.time_coordinates",
        "algorithm.accepted_time_coordinates",
        model.time_coordinates,
        algorithm.accepted_time_coordinates,
    )
    intersect(
        "training_mode_mismatch",
        "model.training_modes",
        "algorithm.accepted_training_modes",
        model.training_modes,
        algorithm.accepted_training_modes,
    )
    intersect(
        "precision_mismatch",
        "model.supported_precisions",
        "algorithm.accepted_precisions",
        model.supported_precisions,
        algorithm.accepted_precisions,
    )
    intersect(
        "likelihood_mismatch",
        "policy_runtime.likelihood_semantics",
        "algorithm.likelihood_semantics",
        model.likelihood_semantics,
        algorithm.likelihood_semantics,
    )
    intersect(
        "distribution_mode_mismatch",
        "policy_runtime.distribution_modes",
        "algorithm.accepted_distribution_modes",
        model.distribution_modes,
        algorithm.accepted_distribution_modes,
    )

    if algorithm.transition_kind not in model.transition_kinds:
        mismatches.append(
            CapabilityMismatch(
                code="transition_kind_mismatch",
                producer_field="policy_runtime.transition_kinds",
                consumer_field="algorithm.transition_kind",
                required=algorithm.transition_kind.value,
                provided=to_identity_value(model.transition_kinds),
                hint="bind Dynamics with the transition kind required by the algorithm",
            )
        )
    else:
        selections.append(
            ("algorithm.transition_kind", algorithm.transition_kind.value)
        )

    missing_features = tuple(
        item
        for item in algorithm.required_transition_features
        if item not in model.transition_features
    )
    if missing_features:
        mismatches.append(
            CapabilityMismatch(
                code="transition_feature_missing",
                producer_field="policy_runtime.transition_features",
                consumer_field="algorithm.required_transition_features",
                required=missing_features,
                provided=model.transition_features,
                hint="bind a policy runtime implementing every required transition feature",
            )
        )

    missing_policy_metadata = tuple(
        item
        for item in algorithm.required_policy_metadata_fields
        if item not in model.policy_metadata_fields
    )
    if missing_policy_metadata:
        mismatches.append(
            CapabilityMismatch(
                code="policy_metadata_fields_missing",
                producer_field="policy_runtime.policy_metadata_fields",
                consumer_field="algorithm.required_policy_metadata_fields",
                required=missing_policy_metadata,
                provided=model.policy_metadata_fields,
                hint=(
                    "bind Dynamics producing every policy metadata field required "
                    "by the algorithm"
                ),
            )
        )

    missing_payloads = tuple(
        item
        for item in algorithm.required_condition_payload_types
        if item not in model.condition_payload_types
    )
    if missing_payloads:
        mismatches.append(
            CapabilityMismatch(
                code="condition_payload_missing",
                producer_field="model.condition_payload_types",
                consumer_field="algorithm.required_condition_payload_types",
                required=missing_payloads,
                provided=model.condition_payload_types,
                hint="select a model preprocess port providing the required payload",
            )
        )

    if algorithm.reference_required and model.provides_reference_policy is not True:
        mismatches.append(
            CapabilityMismatch(
                code="reference_policy_missing",
                producer_field="model.provides_reference_policy",
                consumer_field="algorithm.reference_required",
                required=True,
                provided=model.provides_reference_policy,
                hint="bind a typed frozen-reference policy or disable optional KL",
            )
        )

    selections.extend(
        (
            ("algorithm.trajectory_kind", algorithm.trajectory_kind.value),
            ("algorithm.grouping", algorithm.grouping.value),
        )
    )
    return tuple(sorted(selections)), tuple(mismatches)
