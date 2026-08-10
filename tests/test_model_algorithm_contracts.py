"""Canonical name-independent model/Algorithm capability binding contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from visual_rl.algorithms.catalog import algorithm_domain_catalog_fragments
from visual_rl.algorithms.modules.flow_grpo import FlowGRPOAlgorithmModule
from visual_rl.composition.compatibility import (
    ModelAlgorithmMismatch,
    bind_model_algorithm,
)
from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    build_catalog,
)
from visual_rl.core.contracts import (
    BoundPolicyCapabilities,
    ComputePrecision,
    DistributionMode,
    DynamicsContract,
    LatentLayout,
    LikelihoodSemantics,
    MediaKind,
    ModelDescriptorContract,
    PolicyTransitionRequest,
    PredictionType,
    TaskKind,
    TimeCoordinate,
    TrainerContract,
    TrainingMode,
    TransitionKind,
)


def _model(*, reference: bool = True) -> ModelDescriptorContract:
    return ModelDescriptorContract(
        tasks=(TaskKind.T2I,),
        output_media=(MediaKind.IMAGE,),
        latent_layouts=(LatentLayout.BCHW,),
        latent_ranks=(4,),
        axis_semantics=(("batch", "channel", "height", "width"),),
        prediction_types=(PredictionType.FLOW,),
        time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
        training_modes=(TrainingMode.LORA,),
        supported_precisions=(ComputePrecision.FP32,),
        provides_reference_policy=reference,
    )


def _dynamics(*, branchable: bool = True) -> DynamicsContract:
    return DynamicsContract(
        accepted_latent_layouts=(LatentLayout.BCHW,),
        accepted_prediction_types=(PredictionType.FLOW,),
        accepted_time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
        accepted_transition_dtypes=("float32",),
        transition_kind=TransitionKind.SDE,
        stochastic=True,
        exposes_mean_std=True,
        scores_arbitrary_action=True,
        differentiable_log_prob=True,
        replayable=True,
        branchable=branchable,
        supports_deterministic_ode=True,
        log_prob_reduction="latent_mean.v1",
        supported_likelihoods=(
            LikelihoodSemantics.EXACT_ENV_ACTION,
            LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
        ),
    )


def _trainer() -> TrainerContract:
    return TrainerContract(
        accepted_training_modes=(TrainingMode.LORA,),
        accepted_distribution_modes=(DistributionMode.SINGLE,),
        required_policy_fields=(),
        supports_reference_policy=True,
    )


def _requirements(*, beta: float = 0.0):
    catalog = build_catalog(algorithm_domain_catalog_fragments())
    declaration = AlgorithmDeclarationResolver().resolve(
        catalog.for_kind("algorithm"),
        "flow-grpo",
        {"beta": beta},
    )
    module = FlowGRPOAlgorithmModule(declaration.config)
    assert module.requirements == declaration.requirements
    return declaration.requirements


def test_bare_model_never_claims_transition_or_distribution_capabilities() -> None:
    capabilities = BoundPolicyCapabilities.from_model_contract(_model())

    assert capabilities.transition_kinds == ()
    assert capabilities.transition_features == ()
    assert capabilities.distribution_modes == ()
    with pytest.raises(ModelAlgorithmMismatch) as excinfo:
        bind_model_algorithm(capabilities, _requirements())
    assert {item.code for item in excinfo.value.mismatches} >= {
        "transition_kind_mismatch",
        "distribution_mode_mismatch",
    }


def test_composition_enriched_port_emits_one_typed_binding() -> None:
    capabilities = BoundPolicyCapabilities.from_contracts(
        _model(),
        dynamics=_dynamics(),
        trainer=_trainer(),
    )
    requirements = _requirements()

    binding = bind_model_algorithm(capabilities, requirements)

    assert binding.model_capabilities is capabilities
    assert binding.algorithm_requirements is requirements
    assert binding.binding_id.startswith("model-algorithm-binding.v1:")
    assert ("algorithm.transition_kind", "sde") in binding.selections
    assert tuple(binding.selections) == tuple(sorted(binding.selections))


def test_reference_requirement_fails_with_structured_reason_before_runtime() -> None:
    capabilities = BoundPolicyCapabilities.from_contracts(
        _model(reference=False),
        dynamics=_dynamics(),
        trainer=_trainer(),
    )

    with pytest.raises(ModelAlgorithmMismatch) as excinfo:
        bind_model_algorithm(capabilities, _requirements(beta=0.1))

    mismatch = next(
        item
        for item in excinfo.value.mismatches
        if item.code == "reference_policy_missing"
    )
    assert mismatch.producer_field == "model.provides_reference_policy"
    assert mismatch.required is True
    assert mismatch.provided is False


def test_transition_feature_mismatch_names_only_the_missing_capability() -> None:
    capabilities = BoundPolicyCapabilities.from_contracts(
        _model(),
        dynamics=_dynamics(branchable=False),
        trainer=_trainer(),
    )
    requirements = replace(
        _requirements(),
        required_transition_features=("branchable",),
    )

    with pytest.raises(ModelAlgorithmMismatch) as excinfo:
        bind_model_algorithm(capabilities, requirements)

    mismatch = next(
        item
        for item in excinfo.value.mismatches
        if item.code == "transition_feature_missing"
    )
    assert mismatch.required == ("branchable",)
    assert "branchable" not in mismatch.provided


def test_policy_metadata_mismatch_names_only_the_missing_field() -> None:
    capabilities = BoundPolicyCapabilities.from_contracts(
        _model(),
        dynamics=_dynamics(),
        trainer=_trainer(),
    )
    requirements = replace(
        _requirements(),
        required_policy_metadata_fields=("rectification_coefficient",),
    )

    with pytest.raises(ModelAlgorithmMismatch) as excinfo:
        bind_model_algorithm(capabilities, requirements)

    mismatch = next(
        item
        for item in excinfo.value.mismatches
        if item.code == "policy_metadata_fields_missing"
    )
    assert mismatch.producer_field == "policy_runtime.policy_metadata_fields"
    assert mismatch.required == ("rectification_coefficient",)
    assert mismatch.provided == ()


def test_policy_transition_request_carries_one_explicit_session() -> None:
    session = object()
    request = PolicyTransitionRequest(
        mode="sample",
        transition_session=session,
        transition_input=object(),
        generator=object(),
    )

    assert request.transition_session is session
    with pytest.raises(ValueError, match="transition_session"):
        PolicyTransitionRequest(
            mode="sample",
            transition_session=None,
            transition_input=object(),
        )
