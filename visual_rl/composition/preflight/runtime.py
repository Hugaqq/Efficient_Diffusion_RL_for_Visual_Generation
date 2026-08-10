"""Runtime-bind identity checks; no component loader or recipe mutation."""

from __future__ import annotations

import hashlib

from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import to_plain_dict
from visual_rl.errors import ValidationError
from visual_rl.composition.preflight.types import (
    RuntimeBindInput,
    RuntimeBindResult,
    RuntimeGraphBindInput,
    RuntimeGraphBindResult,
    runtime_graph_payload_id,
    runtime_launch_payload_id,
)

__all__ = ("bind_runtime", "bind_runtime_graph")


def bind_runtime(request: RuntimeBindInput) -> RuntimeBindResult:
    """Validate launch facts and return a separate launch identity."""

    if not isinstance(request, RuntimeBindInput):
        raise TypeError("request must be a RuntimeBindInput")
    materialized = request.environment.materialized
    recipe_id = materialized.recipe_id
    if any(peer != recipe_id for peer in request.peer_recipe_ids):
        raise ValidationError("runtime ranks resolved different recipe_id values")

    execution = materialized.resolved.execution_policy
    facts = request.runtime_facts
    if facts.distribution_mode != execution.distribution_mode.value:
        raise ValidationError(
            "runtime distribution_mode differs from MaterializedRecipe"
        )
    if facts.precision != execution.precision.value:
        raise ValidationError("runtime precision differs from MaterializedRecipe")

    launch_audit_id = hashlib.sha256(
        canonical_json_text(to_plain_dict(request.launch_audit)).encode("utf-8")
    ).hexdigest()
    launch_id = runtime_launch_payload_id(recipe_id, facts)
    result = RuntimeBindResult(
        recipe_id=recipe_id,
        launch_id=launch_id,
        runtime_facts=facts,
        launch_audit_id=launch_audit_id,
        launch_audit=request.launch_audit,
    )
    if materialized.recipe_id != recipe_id:
        raise RuntimeError("runtime bind mutated MaterializedRecipe identity")
    return result


def bind_runtime_graph(request: RuntimeGraphBindInput) -> RuntimeGraphBindResult:
    """Bind prepared graph facts and compare the resulting id across ranks."""

    if not isinstance(request, RuntimeGraphBindInput):
        raise TypeError("request must be RuntimeGraphBindInput")
    recipe = request.environment.materialized
    slots = (
        "algorithm",
        recipe.resolved.model.slot,
        *(item.slot for item in recipe.resolved.internal_components),
        *(item.slot for item in recipe.resolved.reward_components),
    )
    component_bound_contract_ids = request.component_bound_contract_ids
    if set(component_bound_contract_ids) != set(slots):
        raise ValidationError(
            "runtime component contracts do not exactly cover resolved slots"
        )
    configured_plan = recipe.resolved.execution_policy.transform_plan
    if request.execution_transform_plan_id != configured_plan.plan_id:
        raise ValidationError(
            "runtime execution transform order differs from MaterializedRecipe"
        )
    payload = request.canonical_payload()
    bound_id = runtime_graph_payload_id(payload)
    peer_ids = request.peer_bound_contract_ids
    if peer_ids:
        if len(peer_ids) != request.launch.runtime_facts.world_size:
            raise ValidationError("runtime bound id count does not match world_size")
        if any(item != bound_id for item in peer_ids):
            raise ValidationError(
                "runtime ranks produced different bound_contract_id values"
            )
    return RuntimeGraphBindResult(
        recipe_id=recipe.recipe_id,
        launch_id=request.launch.launch_id,
        bound_contract_id=bound_id,
        component_bound_contract_ids=component_bound_contract_ids,
        trainable_topology_id=request.trainable_topology_id,
        prepared_component_names=request.prepared_component_names,
        execution_transform_plan_id=request.execution_transform_plan_id,
        resource_plan_id=request.resource_plan_id,
        verified_fields=request.verified_fields,
        bound_reward_resource_ids=request.bound_reward_resource_ids,
    )
