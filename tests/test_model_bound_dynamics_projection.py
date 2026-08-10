"""Pure compiler projection for model-bound Dynamics declarations."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from visual_rl.algorithms.catalog import algorithm_domain_catalog_fragments
from visual_rl.composition.compatibility import match_model_algorithm_dynamics
from visual_rl.composition.config.integration import (
    DynamicsConditioningMode,
    DynamicsIntegrationSpec,
    DynamicsProjectionRegistry,
    ModelBoundDynamicsProjectionError,
    ModelBoundDynamicsProjector,
    bind_model_bound_dynamics_declaration,
    default_dynamics_projection_registry,
    project_model_bound_dynamics,
)
from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    DeclarationResolver,
    build_catalog,
)
from visual_rl.core.contracts import (
    AlgorithmComponentResolution,
    LikelihoodSemantics,
    ReplayTarget,
)
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
from visual_rl.models.catalog import model_catalog_fragment

ROOT = Path(__file__).resolve().parents[1]


def _catalog():
    return build_catalog(
        (model_catalog_fragment(), *algorithm_domain_catalog_fragments())
    )


def _model(alias: str):
    return DeclarationResolver().resolve(
        _catalog().for_kind("model"),
        alias,
        {"artifact_ref": "main"},
    )


def _algorithm(alias: str, params: dict[str, object] | None = None):
    return AlgorithmDeclarationResolver().resolve(
        _catalog().for_kind("algorithm"),
        alias,
        {} if params is None else params,
    )


def _conditioned(*, surrogate: bool) -> DynamicsIntegrationSpec:
    return DynamicsIntegrationSpec(
        conditioning=DynamicsConditioningMode.CONDITIONED,
        likelihood_semantics=(
            LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE
            if surrogate
            else LikelihoodSemantics.EXACT_ENV_ACTION
        ),
        replay_target=(
            ReplayTarget.CONDITIONED_NEXT if surrogate else ReplayTarget.SAMPLED_ACTION
        ),
    )


def _resolve_projected_dynamics(projection):
    return DeclarationResolver().resolve(
        _catalog().for_kind("dynamics"),
        projection.component_id,
        to_plain_dict(projection.params),
    )


def _match_projection(*, model, algorithm, integration, declaration):
    dynamics = declaration.declared_contract.dynamics
    assert dynamics is not None
    return match_model_algorithm_dynamics(
        model=model.declared_contract.model,
        dynamics=dynamics,
        algorithm=algorithm.requirements,
        likelihood_semantics=integration.likelihood_semantics,
        beta=algorithm.blueprint.beta,
    )


@pytest.mark.parametrize(
    ("algorithm_alias", "model_alias", "component_id", "params"),
    (
        ("flow-grpo", "sd3", "flow-sde", {}),
        ("tempflow-grpo", "sd3", "flow-sde", {}),
        (
            "flow-grpo",
            "wan-t2v",
            "wan-flow-sde",
            {
                "profile": "standard",
                "likelihood_semantics": "exact_env_action",
                "replay_target": "sampled_action",
                "stochastic_sampling": True,
            },
        ),
        (
            "flash-grpo",
            "wan-t2v",
            "wan-flow-sde",
            {
                "profile": "flash",
                "likelihood_semantics": "exact_env_action",
                "replay_target": "sampled_action",
                "stochastic_sampling": True,
            },
        ),
    ),
)
def test_default_model_bound_dynamics_projection(
    algorithm_alias: str,
    model_alias: str,
    component_id: str,
    params: dict[str, object],
) -> None:
    model = _model(model_alias)
    algorithm = _algorithm(algorithm_alias)
    integration = DynamicsIntegrationSpec.unconditioned()
    projection = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=integration,
    )

    assert projection.component_id == component_id
    assert to_plain_dict(projection.params) == params
    assert projection.implementation_family == "flow-sde"
    assert projection.projection_id.startswith("model-bound-dynamics-projection.v1:")
    declaration = _resolve_projected_dynamics(projection)
    match = _match_projection(
        model=model,
        algorithm=algorithm,
        integration=integration,
        declaration=declaration,
    )
    selection = bind_model_bound_dynamics_declaration(
        projection=projection,
        declaration=declaration,
        model=model,
        algorithm=algorithm,
        integration=integration,
    )
    assert match.is_compatible
    assert declaration.declared_contract.dynamics is not None
    assert declaration.declaration_id.startswith("component-declaration.v1:")
    assert selection.selected_component_id == declaration.alias
    assert selection.component_declaration_id == declaration.declaration_id
    assert selection.resolution is AlgorithmComponentResolution.MODEL_BOUND


@pytest.mark.parametrize("surrogate", (False, True))
def test_conditioned_wan_projection_is_integration_owned(surrogate: bool) -> None:
    integration = _conditioned(surrogate=surrogate)
    model = _model("wan-t2v")
    algorithm = _algorithm("flow-grpo")
    projection = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=integration,
    )

    assert projection.params["profile"] == "conditioned"
    assert projection.params["likelihood_semantics"] == (
        integration.likelihood_semantics.value
    )
    assert projection.params["replay_target"] == integration.replay_target.value
    assert projection.integration_id == integration.integration_id
    standard = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=DynamicsIntegrationSpec.unconditioned(),
    )
    assert projection.projection_id != standard.projection_id
    declaration = _resolve_projected_dynamics(projection)
    match = _match_projection(
        model=model,
        algorithm=algorithm,
        integration=integration,
        declaration=declaration,
    )
    selection = bind_model_bound_dynamics_declaration(
        projection=projection,
        declaration=declaration,
        model=model,
        algorithm=algorithm,
        integration=integration,
    )
    assert match.is_compatible
    assert selection.component_declaration_id == declaration.declaration_id
    assert selection.component_declaration_id != (
        _resolve_projected_dynamics(standard).declaration_id
    )


def test_positive_beta_flow_sd3_resolves_and_matches() -> None:
    model = _model("sd3")
    algorithm = _algorithm("flow-grpo", {"beta": 0.004})
    integration = DynamicsIntegrationSpec.unconditioned()
    projection = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=integration,
    )
    declaration = _resolve_projected_dynamics(projection)

    match = _match_projection(
        model=model,
        algorithm=algorithm,
        integration=integration,
        declaration=declaration,
    )
    selection = bind_model_bound_dynamics_declaration(
        projection=projection,
        declaration=declaration,
        model=model,
        algorithm=algorithm,
        integration=integration,
    )

    assert algorithm.requirements.reference_required is True
    assert match.is_compatible
    assert selection.component_declaration_id == declaration.declaration_id


def test_projection_rejects_blueprint_requirement_reference_disagreement() -> None:
    algorithm = _algorithm("flow-grpo", {"beta": 0.004})
    inconsistent_requirements = replace(
        algorithm.requirements,
        reference_required=False,
    )
    inconsistent_component = replace(
        algorithm.component,
        declared_contract=replace(
            algorithm.component.declared_contract,
            algorithm=inconsistent_requirements,
        ),
    )
    inconsistent_algorithm = replace(
        algorithm,
        component=inconsistent_component,
    )

    with pytest.raises(
        ModelBoundDynamicsProjectionError,
        match="reference requirement differs from blueprint beta",
    ):
        project_model_bound_dynamics(
            model=_model("sd3"),
            algorithm=inconsistent_algorithm,
            integration=DynamicsIntegrationSpec.unconditioned(),
        )


def test_same_alias_forged_dynamics_contract_is_rejected_after_resolution() -> None:
    model = _model("wan-t2v")
    algorithm = _algorithm("flow-grpo")
    integration = DynamicsIntegrationSpec.unconditioned()
    projection = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=integration,
    )
    declaration = _resolve_projected_dynamics(projection)
    dynamics = declaration.declared_contract.dynamics
    assert dynamics is not None
    forged = replace(
        declaration,
        declared_contract=replace(
            declaration.declared_contract,
            dynamics=replace(
                dynamics,
                accepted_model_binding_families=("sd3.flow-sde.v1",),
            ),
        ),
    )

    match = _match_projection(
        model=model,
        algorithm=algorithm,
        integration=integration,
        declaration=forged,
    )

    assert forged.alias == declaration.alias == projection.component_id
    assert not match.is_compatible
    assert {item.code for item in match.mismatches} >= {
        "dynamics_binding_family_mismatch"
    }
    with pytest.raises(
        ModelBoundDynamicsProjectionError,
        match="dynamics_binding_family_mismatch",
    ):
        bind_model_bound_dynamics_declaration(
            projection=projection,
            declaration=forged,
            model=model,
            algorithm=algorithm,
            integration=integration,
        )


@pytest.mark.parametrize(
    ("algorithm_alias", "model_alias", "params", "message"),
    (
        ("tempflow-grpo", "wan-t2v", {}, "not branchable"),
        ("flash-grpo", "sd3", {}, "single-step rectification"),
        (
            "flow-grpo",
            "wan-t2v",
            {"beta": 0.004},
            "require a model reference policy",
        ),
    ),
)
def test_invalid_cross_axis_projection_fails_before_runtime_import(
    algorithm_alias: str,
    model_alias: str,
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ModelBoundDynamicsProjectionError, match=message):
        project_model_bound_dynamics(
            model=_model(model_alias),
            algorithm=_algorithm(algorithm_alias, params),
            integration=DynamicsIntegrationSpec.unconditioned(),
        )


def test_conditioned_projection_rejects_sd3_and_flash() -> None:
    integration = _conditioned(surrogate=False)
    with pytest.raises(
        ModelBoundDynamicsProjectionError,
        match="SD3 Dynamics binding does not accept conditioned",
    ):
        project_model_bound_dynamics(
            model=_model("sd3"),
            algorithm=_algorithm("flow-grpo"),
            integration=integration,
        )
    with pytest.raises(
        ModelBoundDynamicsProjectionError,
        match="single-step Wan Dynamics does not accept",
    ):
        project_model_bound_dynamics(
            model=_model("wan-t2v"),
            algorithm=_algorithm("flash-grpo"),
            integration=integration,
        )


def test_integration_pairs_and_blueprint_beta_fail_closed() -> None:
    with pytest.raises(ValueError, match="unconditioned Dynamics requires"):
        DynamicsIntegrationSpec(
            conditioning=DynamicsConditioningMode.UNCONDITIONED,
            likelihood_semantics=(LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE),
            replay_target=ReplayTarget.CONDITIONED_NEXT,
        )
    blueprint = _algorithm("flow-grpo").blueprint
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(blueprint, beta=-0.1)


def test_projection_source_has_no_recipe_or_algorithm_alias_switch() -> None:
    source = (ROOT / "visual_rl/composition/config/integration.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        '"flow-grpo"',
        '"tempflow-grpo"',
        '"flash-grpo"',
        "world_r1",
        "visual_rl.models.implementations",
        "visual_rl.algorithms.modules.config",
    ):
        assert forbidden not in source
    assert "if binding_family ==" not in source
    assert "elif binding_family ==" not in source


def test_default_projection_registry_has_unique_builtin_family_keys() -> None:
    registry = default_dynamics_projection_registry()

    assert tuple(
        (item.model_binding_family, item.implementation_family)
        for item in registry.projectors
    ) == (
        ("sd3.flow-sde.v1", "flow-sde"),
        ("wan.flow-sde.v1", "flow-sde"),
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.projectors[0])


def test_custom_model_binding_projector_is_injected_without_core_switch() -> None:
    model = _model("sd3")
    model_contract = model.declared_contract.model
    assert model_contract is not None
    custom_model = replace(
        model,
        declared_contract=replace(
            model.declared_contract,
            model=replace(
                model_contract,
                dynamics_binding_family="custom.flow-sde.v1",
            ),
        ),
    )
    algorithm = _algorithm("flow-grpo")
    integration = DynamicsIntegrationSpec.unconditioned()
    calls = []

    def project(requirements, slot_params, supplied_integration):
        calls.append((requirements, slot_params, supplied_integration))
        return "flow-sde", FrozenMapping(slot_params)

    registry = DynamicsProjectionRegistry().register(
        ModelBoundDynamicsProjector(
            model_binding_family="custom.flow-sde.v1",
            implementation_family="flow-sde",
            project=project,
        )
    )
    projection = project_model_bound_dynamics(
        model=custom_model,
        algorithm=algorithm,
        integration=integration,
        projector_registry=registry,
    )
    declaration = _resolve_projected_dynamics(projection)
    dynamics_contract = declaration.declared_contract.dynamics
    assert dynamics_contract is not None
    custom_declaration = replace(
        declaration,
        declared_contract=replace(
            declaration.declared_contract,
            dynamics=replace(
                dynamics_contract,
                accepted_model_binding_families=("custom.flow-sde.v1",),
            ),
        ),
    )
    selection = bind_model_bound_dynamics_declaration(
        projection=projection,
        declaration=custom_declaration,
        model=custom_model,
        algorithm=algorithm,
        integration=integration,
        projector_registry=registry,
    )

    assert len(calls) == 2
    assert projection.model_binding_family == "custom.flow-sde.v1"
    assert projection.component_id == "flow-sde"
    assert selection.selected_component_id == "flow-sde"
    with pytest.raises(
        ModelBoundDynamicsProjectionError,
        match="unsupported model Dynamics binding family",
    ):
        project_model_bound_dynamics(
            model=custom_model,
            algorithm=algorithm,
            integration=integration,
        )


def test_projection_import_is_backend_free() -> None:
    script = r"""
import json
import sys
from visual_rl.composition.config.integration import DynamicsIntegrationSpec

heavy = {"accelerate", "diffusers", "peft", "requests", "torch", "transformers"}
print(json.dumps({
    "integration": DynamicsIntegrationSpec.unconditioned().to_payload(),
    "heavy": sorted(heavy.intersection(name.split(".", 1)[0] for name in sys.modules)),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["heavy"] == []
    assert payload["integration"] == {
        "conditioning": "unconditioned",
        "likelihood_semantics": "exact_env_action",
        "replay_target": "sampled_action",
    }
