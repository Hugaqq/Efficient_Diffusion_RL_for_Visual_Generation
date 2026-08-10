"""Small G1-gated runtime loader helper shared by focused component tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from visual_rl.composition.registry import ResolvedComponentDeclaration
from visual_rl.core.contracts import ComponentArtifactBindingSet, ComponentLoadPlan
from visual_rl.composition.preflight import (
    RuntimeBindResult,
    RuntimeFacts,
    runtime_launch_payload_id,
)
from visual_rl.runtime.component_loader import (
    RuntimeComponentLoader,
    RuntimeComponentLoadGate,
    RuntimeComponentLoadResult,
    build_component_artifact_binding,
)


def load_test_component(
    declaration: ResolvedComponentDeclaration,
    *,
    slot: str,
    runtime_context: Mapping[str, object],
) -> RuntimeComponentLoadResult:
    """Materialize one declaration through an exact, code-only test G1 graph."""

    recipe_digest = hashlib.sha256(
        f"test-recipe\0{slot}\0{declaration.declaration_id}".encode()
    ).hexdigest()
    recipe_id = f"materialized-recipe.v2:{recipe_digest}"
    code_identity = hashlib.sha256(
        declaration.implementation_class_path.encode("utf-8")
    ).hexdigest()
    runtime_facts = RuntimeFacts(
        distribution_mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        device="cpu",
        precision="fp32",
        backend=None,
    )
    runtime_binding = RuntimeBindResult(
        recipe_id=recipe_id,
        launch_id=runtime_launch_payload_id(recipe_id, runtime_facts),
        runtime_facts=runtime_facts,
    )
    artifact_binding = build_component_artifact_binding(
        declaration,
        recipe_id=recipe_id,
        slot=slot,
        artifact_content_identities={"code": code_identity},
        code_identity=code_identity,
    )
    binding_set = ComponentArtifactBindingSet(recipe_id, (artifact_binding,))
    load_plan = ComponentLoadPlan.create(
        binding_set,
        required_artifact_names_by_slot={slot: ("code",)},
    )
    return RuntimeComponentLoader().load(
        declaration,
        gate=RuntimeComponentLoadGate(runtime_binding, artifact_binding),
        binding_set=binding_set,
        load_plan=load_plan,
        runtime_context=runtime_context,
    )
