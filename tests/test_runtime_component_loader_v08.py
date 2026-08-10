"""Declaration-bound G1/G3 gates for the canonical runtime loader shadow path."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from visual_rl.algorithms.rewards.interface import RewardComponent
from visual_rl.composition.registry import ResolvedComponentDeclaration
from visual_rl.core.contracts import (
    ArtifactBoundContract,
    ComponentArtifactBinding,
    ComponentArtifactBindingSet,
    ComponentDescriptor,
    ComponentLoadPlan,
    ComponentLoadSlotRequirement,
    DeclaredContract,
    MediaKind,
    RewardContract,
    RewardGranularity,
    RuntimeBoundContract,
)
from visual_rl.core.identity import canonical_identity, to_identity_value
from visual_rl.composition.preflight import (
    RuntimeBindResult,
    RuntimeFacts,
    runtime_launch_payload_id,
)
from visual_rl.runtime.component_loader import (
    RuntimeComponentLoader,
    RuntimeComponentLoadError,
    RuntimeComponentLoadGate,
    RuntimeComponentLoadRequest,
    build_component_artifact_binding,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


RECIPE_ID = f"materialized-recipe.v2:{_digest('recipe')}"
CODE_ID = _digest("code")
REWARD_ARTIFACT_ID = _digest("reward-artifact")


@dataclass(frozen=True, slots=True)
class _Config:
    value: int


class _RuntimeImplementation(RewardComponent):
    INTERFACE_VERSION = "1.0"

    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def from_config(cls, config: object, *, runtime_context: object):
        assert runtime_context == {"rank": 0}
        assert isinstance(config, _Config)
        return cls(config.value)


class _NonCallableRequiredMethodRuntimeImplementation(_RuntimeImplementation):
    score = None


_CLOSE_EVENTS: list[str] = []


class _DependentRuntimeImplementation(RewardComponent):
    INTERFACE_VERSION = "1.0"

    def __init__(self, slot: str, seen: tuple[str, ...]) -> None:
        self.slot = slot
        self.seen = seen

    @classmethod
    def from_config(cls, config: object, *, runtime_context: object):
        assert isinstance(config, _Config)
        assert isinstance(runtime_context, dict)
        return cls(runtime_context["slot"], runtime_context["seen"])

    def close(self) -> None:
        _CLOSE_EVENTS.append(self.slot)


def _declared(alias: str = "quality") -> DeclaredContract:
    return DeclaredContract(
        component_kind="reward",
        component_id=alias,
        reward=RewardContract(
            accepted_media=(MediaKind.IMAGE,),
            required_payload_type=None,
            granularity=RewardGranularity.POINTWISE,
            output_rank=0,
            frame_aggregation=None,
        ),
    )


def _resolution(
    *,
    alias: str = "quality",
    value: int = 7,
    interface_version: str = "1.0",
) -> ResolvedComponentDeclaration:
    return ResolvedComponentDeclaration(
        kind="reward",
        alias=alias,
        descriptor=ComponentDescriptor(
            alias=alias,
            implementation_class_path=(
                "tests.test_runtime_component_loader_v08:_RuntimeImplementation"
            ),
            declaration_provider_path=(
                "tests.test_runtime_component_loader_v08:_UnusedProvider"
            ),
            interface_version=interface_version,
        ),
        config_type_path="tests.test_runtime_component_loader_v08:_Config",
        config=_Config(value),
        declared_contract=_declared(alias),
    )


def _dependent_resolution() -> ResolvedComponentDeclaration:
    declaration = _resolution()
    return replace(
        declaration,
        descriptor=replace(
            declaration.descriptor,
            implementation_class_path=(
                "tests.test_runtime_component_loader_v08:_DependentRuntimeImplementation"
            ),
        ),
    )


def _runtime_binding() -> RuntimeBindResult:
    runtime_facts = RuntimeFacts(
        distribution_mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        device="cpu",
        precision="fp32",
        backend=None,
    )
    return RuntimeBindResult(
        recipe_id=RECIPE_ID,
        launch_id=runtime_launch_payload_id(RECIPE_ID, runtime_facts),
        runtime_facts=runtime_facts,
    )


def _binding(
    declaration: ResolvedComponentDeclaration,
    *,
    slot: str = "rewards.quality",
    reward_artifact_identity: str = REWARD_ARTIFACT_ID,
) -> ComponentArtifactBinding:
    return build_component_artifact_binding(
        declaration,
        recipe_id=RECIPE_ID,
        slot=slot,
        artifact_content_identities={
            "code": CODE_ID,
            "reward-resource": reward_artifact_identity,
        },
        code_identity=CODE_ID,
    )


def _gate(
    declaration: ResolvedComponentDeclaration,
    *,
    binding: ComponentArtifactBinding | None = None,
) -> RuntimeComponentLoadGate:
    return RuntimeComponentLoadGate(
        runtime_binding=_runtime_binding(),
        artifact_binding=_binding(declaration) if binding is None else binding,
    )


def test_runtime_binding_recipe_must_match_the_g1_artifact_binding() -> None:
    declaration = _resolution()
    current = _runtime_binding()
    stale_recipe_id = f"materialized-recipe.v2:{_digest('stale-recipe')}"
    stale_runtime = RuntimeBindResult(
        recipe_id=stale_recipe_id,
        launch_id=runtime_launch_payload_id(
            stale_recipe_id,
            current.runtime_facts,
        ),
        runtime_facts=current.runtime_facts,
    )

    with pytest.raises(ValueError, match="validated runtime binding"):
        RuntimeComponentLoadGate(
            runtime_binding=stale_runtime,
            artifact_binding=_binding(declaration),
        )


def _strict_graph(
    *bindings: ComponentArtifactBinding,
    required_artifact_names_by_slot: dict[str, tuple[str, ...]] | None = None,
) -> tuple[ComponentArtifactBindingSet, ComponentLoadPlan]:
    binding_set = ComponentArtifactBindingSet(bindings[0].recipe_id, bindings)
    requirements = (
        {
            binding.slot: tuple(
                name for name, _identity in binding.artifact_content_identities
            )
            for binding in bindings
        }
        if required_artifact_names_by_slot is None
        else required_artifact_names_by_slot
    )
    return binding_set, ComponentLoadPlan.create(
        binding_set,
        required_artifact_names_by_slot=requirements,
    )


def _load_one(
    declaration: ResolvedComponentDeclaration,
    *,
    gate: RuntimeComponentLoadGate,
):
    binding_set, load_plan = _strict_graph(gate.artifact_binding)
    return RuntimeComponentLoader().load(
        declaration,
        gate=gate,
        binding_set=binding_set,
        load_plan=load_plan,
        runtime_context={"rank": 0},
    )


def test_g1_binding_locks_full_declaration_artifacts_code_and_interface() -> None:
    declaration = _resolution()
    binding = _binding(declaration)

    assert binding.component_declaration_id == declaration.declaration_id
    assert binding.declared is declaration.declared_contract
    assert binding.implementation_identity == declaration.implementation_class_path
    assert binding.canonical_interface == (
        "visual_rl.algorithms.rewards.interface:RewardComponent"
    )
    assert binding.interface_version == "1.0"
    assert binding.artifact_content_identities == (
        ("code", CODE_ID),
        ("reward-resource", REWARD_ARTIFACT_ID),
    )
    assert binding.artifact_set_identity.startswith("component-artifact-set.v1:")
    assert binding.binding_id.startswith("component-artifact-binding.v1:")
    assert binding.to_payload() == {
        "schema": "visual-rl.component-artifact-binding",
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "slot": "rewards.quality",
        "component_declaration_id": declaration.declaration_id,
        "declared_contract": binding.to_payload()["declared_contract"],
        "artifact_set_identity": binding.artifact_set_identity,
        "artifact_content_identities": [
            {"name": "code", "content_identity": CODE_ID},
            {
                "name": "reward-resource",
                "content_identity": REWARD_ARTIFACT_ID,
            },
        ],
        "code_identity": CODE_ID,
        "implementation_identity": declaration.implementation_class_path,
        "canonical_interface": (
            "visual_rl.algorithms.rewards.interface:RewardComponent"
        ),
        "interface_version": "1.0",
        "binding_id": binding.binding_id,
    }


def test_g1_binding_set_is_canonical_and_recipe_bound() -> None:
    declaration = _resolution()
    first = _binding(declaration, slot="rewards.first")
    second = _binding(declaration, slot="rewards.second")

    forward = ComponentArtifactBindingSet(RECIPE_ID, (first, second))
    reverse = ComponentArtifactBindingSet(RECIPE_ID, (second, first))

    assert forward == reverse
    assert forward.slots == ("rewards.first", "rewards.second")
    assert forward.binding("rewards.second") is second
    assert forward.binding_set_id.startswith("component-artifact-binding-set.v1:")
    assert forward.to_payload()["binding_set_id"] == forward.binding_set_id


def test_g1_binding_set_rejects_duplicate_slots_and_cross_recipe_receipts() -> None:
    declaration = _resolution()
    first = _binding(declaration, slot="rewards.first")
    with pytest.raises(ValueError, match="slots must be unique"):
        ComponentArtifactBindingSet(RECIPE_ID, (first, first))

    different_recipe = replace(
        first,
        recipe_id=_digest("different-recipe"),
        slot="rewards.second",
    )
    with pytest.raises(ValueError, match="same recipe_id"):
        ComponentArtifactBindingSet(RECIPE_ID, (first, different_recipe))


def test_component_load_plan_is_canonical_and_binds_expected_final_graph() -> None:
    declaration = _resolution()
    first = _binding(declaration, slot="rewards.first")
    second = _binding(declaration, slot="rewards.second")
    binding_set = ComponentArtifactBindingSet(RECIPE_ID, (second, first))
    plan = ComponentLoadPlan.create(
        binding_set,
        required_artifact_names_by_slot={
            "rewards.second": ("code", "reward-resource"),
            "rewards.first": ("code", "reward-resource"),
        },
    )

    assert plan.expected_recipe_id == RECIPE_ID
    assert plan.expected_binding_set_id == binding_set.binding_set_id
    assert plan.slots == ("rewards.first", "rewards.second")
    assert isinstance(
        plan.requirement("rewards.first"),
        ComponentLoadSlotRequirement,
    )
    assert plan.requirement("rewards.first").binding_id == first.binding_id
    assert plan.plan_id.startswith("component-load-plan.v1:")
    assert plan.to_payload()["plan_id"] == plan.plan_id

    with pytest.raises(ValueError, match="exactly cover"):
        ComponentLoadPlan.create(
            binding_set,
            required_artifact_names_by_slot={
                "rewards.first": ("code", "reward-resource"),
            },
        )


def test_context_resolver_observes_immutable_incremental_loaded_graph() -> None:
    declaration = _dependent_resolution()
    first_binding = _binding(declaration, slot="rewards.first")
    second_binding = _binding(declaration, slot="rewards.second")
    binding_set, load_plan = _strict_graph(first_binding, second_binding)
    requests = tuple(
        RuntimeComponentLoadRequest(
            declaration,
            _gate(declaration, binding=binding),
            {},
        )
        for binding in (first_binding, second_binding)
    )
    observed: list[tuple[str, ...]] = []

    def resolve(request, loaded):
        assert isinstance(loaded, MappingProxyType)
        seen = tuple(loaded)
        observed.append(seen)
        with pytest.raises(TypeError):
            loaded["forbidden"] = object()
        return {
            "slot": request.gate.artifact_binding.slot,
            "seen": seen,
        }

    results = RuntimeComponentLoader().load_all(
        requests,
        binding_set=binding_set,
        load_plan=load_plan,
        context_resolver=resolve,
    )

    assert observed == [(), ("rewards.first",)]
    assert tuple(result.instance.seen for result in results) == (
        (),
        ("rewards.first",),
    )


def test_context_failure_rolls_back_constructed_components_in_reverse() -> None:
    _CLOSE_EVENTS.clear()
    declaration = _dependent_resolution()
    bindings = tuple(
        _binding(declaration, slot=f"rewards.{name}")
        for name in ("first", "second", "third")
    )
    binding_set, load_plan = _strict_graph(*bindings)
    requests = tuple(
        RuntimeComponentLoadRequest(
            declaration,
            _gate(declaration, binding=binding),
            {},
        )
        for binding in bindings
    )

    def resolve(request, loaded):
        slot = request.gate.artifact_binding.slot
        if slot == "rewards.third":
            raise RuntimeError("third context failed")
        return {"slot": slot, "seen": tuple(loaded)}

    with pytest.raises(RuntimeError, match="third context failed"):
        RuntimeComponentLoader().load_all(
            requests,
            binding_set=binding_set,
            load_plan=load_plan,
            context_resolver=resolve,
        )

    assert _CLOSE_EVENTS == ["rewards.second", "rewards.first"]


def test_stale_recipe_subset_and_extra_slot_fail_before_any_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _resolution()
    first_binding = _binding(declaration, slot="rewards.first")
    second_binding = _binding(declaration, slot="rewards.second")
    first = RuntimeComponentLoadRequest(
        declaration,
        _gate(declaration, binding=first_binding),
        {"rank": 0},
    )
    second = RuntimeComponentLoadRequest(
        declaration,
        _gate(declaration, binding=second_binding),
        {"rank": 0},
    )
    binding_set, plan = _strict_graph(first_binding, second_binding)
    stale_plan = ComponentLoadPlan(
        expected_recipe_id=_digest("stale-recipe"),
        expected_binding_set_id=binding_set.binding_set_id,
        requirements=plan.requirements,
    )
    first_set, first_plan = _strict_graph(first_binding)
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("load-plan failures must precede every import"),
    )

    attempts = (
        ((first, second), binding_set, stale_plan),
        ((first,), binding_set, plan),
        ((first, second), first_set, first_plan),
    )
    for requests, attempted_set, attempted_plan in attempts:
        with pytest.raises(RuntimeComponentLoadError) as failure:
            RuntimeComponentLoader().load_all(
                requests,
                binding_set=attempted_set,
                load_plan=attempted_plan,
            )
        assert failure.value.code == "load_plan_mismatch"


def test_missing_and_extra_slot_artifacts_fail_before_any_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _resolution()
    code_only = build_component_artifact_binding(
        declaration,
        recipe_id=RECIPE_ID,
        slot="rewards.quality",
        artifact_content_identities={"code": CODE_ID},
        code_identity=CODE_ID,
    )
    missing_set, missing_plan = _strict_graph(
        code_only,
        required_artifact_names_by_slot={
            "rewards.quality": ("code", "reward-resource"),
        },
    )
    extra = build_component_artifact_binding(
        declaration,
        recipe_id=RECIPE_ID,
        slot="rewards.quality",
        artifact_content_identities={
            "auxiliary": _digest("unexpected-artifact"),
            "code": CODE_ID,
            "reward-resource": REWARD_ARTIFACT_ID,
        },
        code_identity=CODE_ID,
    )
    extra_set, extra_plan = _strict_graph(
        extra,
        required_artifact_names_by_slot={
            "rewards.quality": ("code", "reward-resource"),
        },
    )
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("artifact coverage failure must precede import"),
    )

    for binding, attempted_set, attempted_plan in (
        (code_only, missing_set, missing_plan),
        (extra, extra_set, extra_plan),
    ):
        request = RuntimeComponentLoadRequest(
            declaration,
            _gate(declaration, binding=binding),
            {"rank": 0},
        )
        with pytest.raises(RuntimeComponentLoadError) as failure:
            RuntimeComponentLoader().load_all(
                (request,),
                binding_set=attempted_set,
                load_plan=attempted_plan,
            )
        assert failure.value.code == "load_plan_mismatch"


def test_request_binding_id_must_exactly_match_binding_set_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _resolution()
    expected = _binding(declaration)
    drifted = _binding(
        declaration,
        reward_artifact_identity=_digest("drifted-reward-artifact"),
    )
    binding_set, plan = _strict_graph(expected)
    request = RuntimeComponentLoadRequest(
        declaration,
        _gate(declaration, binding=drifted),
        {"rank": 0},
    )
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("binding mismatch must precede every import"),
    )

    with pytest.raises(RuntimeComponentLoadError) as failure:
        RuntimeComponentLoader().load_all(
            (request,),
            binding_set=binding_set,
            load_plan=plan,
        )
    assert failure.value.code == "load_plan_mismatch"


def test_g1_binding_locks_runtime_component_declaration_namespace_only() -> None:
    binding = _binding(_resolution())
    with pytest.raises(ValueError, match="component_declaration_id"):
        replace(
            binding,
            component_declaration_id=(
                "algorithm-declaration.v1:" + _digest("algorithm")
            ),
        )


def test_runtime_loader_imports_implementation_only_after_exact_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _resolution()
    gate = _gate(declaration)
    observed: list[str] = []
    real_import = importlib.import_module

    def observed_import(module_name: str):
        observed.append(module_name)
        return real_import(module_name)

    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        observed_import,
    )
    loaded = _load_one(declaration, gate=gate)

    assert isinstance(loaded.instance, _RuntimeImplementation)
    assert loaded.instance.value == 7
    assert loaded.artifact_binding is gate.artifact_binding
    assert loaded.load_attestation.binding_id == gate.artifact_binding.binding_id
    assert observed == [
        "visual_rl.algorithms.rewards.interface",
        "tests.test_runtime_component_loader_v08",
    ]


def test_runtime_loader_rejects_explicit_non_callable_interface_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RewardComponent,
        "REQUIRED_RUNTIME_METHODS",
        ("score",),
        raising=False,
    )
    declaration = _resolution()
    declaration = replace(
        declaration,
        descriptor=replace(
            declaration.descriptor,
            implementation_class_path=(
                "tests.test_runtime_component_loader_v08:"
                "_NonCallableRequiredMethodRuntimeImplementation"
            ),
        ),
    )

    with pytest.raises(RuntimeComponentLoadError) as failure:
        _load_one(declaration, gate=_gate(declaration))

    assert failure.value.code == "invalid_implementation"
    assert "non-callable operations ['score']" in str(failure.value)


def test_all_slot_gates_fail_before_first_implementation_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _resolution()
    drifted = _resolution(interface_version="9.9")
    first = RuntimeComponentLoadRequest(valid, _gate(valid), {"rank": 0})
    drifted_binding = replace(
        _binding(valid, slot="rewards.second"),
        component_declaration_id=drifted.declaration_id,
    )
    second = RuntimeComponentLoadRequest(
        drifted,
        _gate(drifted, binding=drifted_binding),
        {"rank": 0},
    )
    real_import = importlib.import_module

    def guarded_import(module_name: str):
        if module_name == "test_runtime_component_loader_v08":
            pytest.fail("implementation imported before every slot gate passed")
        return real_import(module_name)

    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        guarded_import,
    )
    binding_set, load_plan = _strict_graph(
        first.gate.artifact_binding,
        second.gate.artifact_binding,
    )
    with pytest.raises(RuntimeComponentLoadError) as failure:
        RuntimeComponentLoader().load_all(
            (first, second),
            binding_set=binding_set,
            load_plan=load_plan,
        )

    assert failure.value.code == "interface_mismatch"


def test_same_capability_but_different_declaration_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _resolution(value=7)
    changed_config = _resolution(value=8)
    assert original.declared_contract == changed_config.declared_contract
    assert original.declaration_id != changed_config.declaration_id
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("implementation imported before declaration gate"),
    )

    with pytest.raises(RuntimeComponentLoadError) as failure:
        _load_one(changed_config, gate=_gate(original))
    assert failure.value.code == "declaration_mismatch"


def test_artifact_content_and_set_mismatch_fail_closed_without_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(_resolution())
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("artifact receipt validation must not import"),
    )

    with pytest.raises(ValueError, match="artifact_set_identity"):
        replace(binding, artifact_set_identity=_digest("wrong-artifact-set"))
    with pytest.raises(ValueError, match="exact code identity"):
        replace(binding, code_identity=_digest("different-code"))
    with pytest.raises(ValueError, match="canonical SHA-256 identity"):
        build_component_artifact_binding(
            _resolution(),
            recipe_id=RECIPE_ID,
            slot="rewards.quality",
            artifact_content_identities={"code": "artifact:test"},
            code_identity="artifact:test",
        )


def test_implementation_identity_mismatch_precedes_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _resolution()
    mismatched = replace(
        _binding(declaration),
        implementation_identity="forbidden_runtime_component:OtherReward",
    )
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("implementation imported before identity gate"),
    )

    with pytest.raises(RuntimeComponentLoadError) as failure:
        _load_one(declaration, gate=_gate(declaration, binding=mismatched))
    assert failure.value.code == "implementation_mismatch"


def test_interface_is_controlled_and_version_mismatch_precedes_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _resolution()
    binding = _binding(declaration)
    with pytest.raises(ValueError, match="controlled component-kind map"):
        replace(binding, canonical_interface="builtins:object")

    drifted = _resolution(interface_version="9.9")
    matching_declaration_id = replace(
        binding,
        component_declaration_id=drifted.declaration_id,
    )
    monkeypatch.setattr(
        "visual_rl.runtime.component_loader.importlib.import_module",
        lambda _name: pytest.fail("implementation imported before interface gate"),
    )
    with pytest.raises(RuntimeComponentLoadError) as failure:
        _load_one(
            drifted,
            gate=_gate(drifted, binding=matching_declaration_id),
        )
    assert failure.value.code == "interface_mismatch"

    signature = inspect.signature(RuntimeComponentLoader.load)
    assert "expected_type" not in signature.parameters


def test_legacy_artifact_contract_cannot_masquerade_as_new_loader_gate() -> None:
    declaration = _resolution()
    legacy = ArtifactBoundContract(
        declared=declaration.declared_contract,
        artifact_identity="artifact:test",
        resolved_fields=(),
    )
    with pytest.raises(TypeError, match="ComponentArtifactBinding"):
        RuntimeComponentLoadGate(
            runtime_binding=_runtime_binding(),
            artifact_binding=legacy,  # type: ignore[arg-type]
        )


def test_g3_reuses_g1_receipt_and_survives_identity_round_trip() -> None:
    declaration = _resolution()
    loaded = _load_one(declaration, gate=_gate(declaration))
    runtime_identity = canonical_identity(
        "component-runtime.v1",
        {"binding_id": loaded.artifact_binding.binding_id},
    )
    verified_fields = (
        ("component.load", "verified"),
        ("component.prepare", "verified"),
    )
    contract = loaded.attest_prepared(
        runtime_identity=runtime_identity,
        verified_fields=verified_fields,
    )

    assert contract.is_declaration_bound
    assert contract.artifact is loaded.artifact_binding
    assert contract.component_load_attestation is loaded.load_attestation
    assert contract.component_prepare_attestation.binding_id == (
        loaded.artifact_binding.binding_id
    )
    assert contract.component_prepare_attestation.load_attestation_id == (
        loaded.load_attestation.attestation_id
    )
    payload = contract.to_payload()
    assert payload["mode"] == "declaration_bound"
    assert payload["binding_id"] == loaded.artifact_binding.binding_id
    assert payload["artifact_binding"] == loaded.artifact_binding.to_payload()
    assert payload["load_attestation"] == loaded.load_attestation.to_payload()
    assert payload["prepare_attestation"] == (
        contract.component_prepare_attestation.to_payload()
    )

    # A checkpoint round trip reconstructs equal immutable receipts, not the
    # original Python object. Canonical ids still prove exact G1/G3 continuity.
    restored_binding = replace(loaded.artifact_binding)
    restored_load = replace(loaded.load_attestation)
    restored_prepare = replace(contract.component_prepare_attestation)
    assert restored_binding is not loaded.artifact_binding
    restored = RuntimeBoundContract(
        artifact=restored_binding,
        runtime_identity=runtime_identity,
        verified_fields=verified_fields,
        load_attestation=restored_load,
        prepare_attestation=restored_prepare,
    )
    assert restored.to_payload() == payload


def test_g3_new_branch_rejects_missing_or_cross_binding_attestations() -> None:
    declaration = _resolution()
    loaded = _load_one(declaration, gate=_gate(declaration))
    with pytest.raises(TypeError, match="requires load attestation"):
        RuntimeBoundContract(
            artifact=loaded.artifact_binding,
            runtime_identity=_digest("runtime"),
            verified_fields=(("component.prepare", "verified"),),
        )

    different_binding = _binding(
        declaration,
        reward_artifact_identity=_digest("different-reward-artifact"),
    )
    prepared = loaded.attest_prepared(
        runtime_identity=_digest("runtime"),
        verified_fields=(("component.prepare", "verified"),),
    )
    with pytest.raises(ValueError, match="load attestation differs"):
        RuntimeBoundContract(
            artifact=different_binding,
            runtime_identity=prepared.runtime_identity,
            verified_fields=prepared.verified_fields,
            load_attestation=prepared.component_load_attestation,
            prepare_attestation=prepared.component_prepare_attestation,
        )

    tampered_load = replace(
        prepared.component_load_attestation,
        implementation_identity="forbidden_runtime_component:OtherReward",
    )
    with pytest.raises(ValueError, match="facts differ"):
        RuntimeBoundContract(
            artifact=loaded.artifact_binding,
            runtime_identity=prepared.runtime_identity,
            verified_fields=prepared.verified_fields,
            load_attestation=tampered_load,
            prepare_attestation=prepared.component_prepare_attestation,
        )


def test_legacy_runtime_contract_remains_an_explicit_compatibility_branch() -> None:
    declaration = _resolution()
    legacy = RuntimeBoundContract(
        artifact=ArtifactBoundContract(
            declared=declaration.declared_contract,
            artifact_identity="legacy-production-artifact",
            resolved_fields=(),
        ),
        runtime_identity="legacy-production-runtime",
        verified_fields=(("legacy.probe", "verified"),),
    )

    assert not legacy.is_declaration_bound
    assert legacy.to_payload()["mode"] == "legacy_compatibility"
    assert to_identity_value(legacy) == {
        "artifact": {
            "declared": to_identity_value(declaration.declared_contract),
            "artifact_identity": "legacy-production-artifact",
            "resolved_fields": [],
        },
        "runtime_identity": "legacy-production-runtime",
        "verified_fields": [["legacy.probe", "verified"]],
    }
    with pytest.raises(AttributeError, match="legacy compatibility"):
        _ = legacy.component_load_attestation


def test_component_loader_fresh_import_is_lightweight() -> None:
    script = r"""
import importlib
import json
import sys
import visual_rl.runtime.component_loader
from visual_rl.core.contracts import COMPONENT_KINDS, canonical_component_interface

heavy = {"accelerate", "diffusers", "peft", "requests", "torch", "transformers"}
interfaces = {}
for kind in COMPONENT_KINDS:
    class_path = canonical_component_interface(kind)
    module_name, qualname = class_path.split(":", 1)
    value = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    interfaces[kind] = [class_path, value.INTERFACE_VERSION]
print(json.dumps({
    "heavy": sorted(heavy.intersection(name.split(".", 1)[0] for name in sys.modules)),
    "interfaces": interfaces,
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
    assert json.loads(completed.stdout) == {
        "heavy": [],
        "interfaces": {
            "model": ["visual_rl.models.interface:ModelAdapter", "1.0"],
            "algorithm": [
                "visual_rl.algorithms.modules.interface:AlgorithmModule",
                "1.0",
            ],
            "trainer": [
                "visual_rl.algorithms.trainer.interface:TrainerComponent",
                "1.0",
            ],
            "dynamics": [
                "visual_rl.algorithms.dynamics.interface:DynamicsComponent",
                "1.0",
            ],
            "rollout": [
                "visual_rl.algorithms.rollout.interface:RolloutComponent",
                "1.0",
            ],
            "reward": [
                "visual_rl.algorithms.rewards.interface:RewardComponent",
                "1.0",
            ],
            "conditioner": [
                "visual_rl.algorithms.conditioning.interface:LatentConditioner",
                "1.0",
            ],
            "credit": [
                "visual_rl.algorithms.optimization.interface:CreditComponent",
                "1.0",
            ],
        },
    }


def test_loader_does_not_rebuild_post_import_contracts() -> None:
    source = (ROOT / "visual_rl/runtime/component_loader.py").read_text(
        encoding="utf-8"
    )

    assert ".describe(" not in source
    assert "ArtifactBoundContract(" not in source
