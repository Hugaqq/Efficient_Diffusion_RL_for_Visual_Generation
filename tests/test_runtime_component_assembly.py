"""Canonical environment-G1 to runtime-component-graph closure tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.core.contracts import ComponentLoadAttestation
from visual_rl.core.types import FrozenMapping
from visual_rl.data import DatasetArtifactBinding, SourceLocationBinding
from visual_rl.errors import ValidationError
from visual_rl.composition.preflight import (
    ArtifactIdentityResolution,
    RuntimeBindInput,
    RuntimeFacts,
    RuntimeGraphBindInput,
    bind_runtime,
    bind_runtime_graph,
    run_environment_preflight,
    run_static_preflight,
    runtime_launch_payload_id,
)
from visual_rl.runtime.component_graph import load_component_graph
from visual_rl.runtime.component_loader import (
    RuntimeComponentLoader,
    RuntimeComponentLoadResult,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(
    label: str,
    *,
    node_type: str,
    content_policy: str = "all-files.v1",
) -> FrozenMapping:
    return FrozenMapping(
        {
            "identity_schema": "filesystem-artifact.v1",
            "content_policy": content_policy,
            "node_type": node_type,
            "content_sha256": _digest(label),
            "file_count": 1,
            "byte_count": len(label),
        }
    )


class _ArtifactResolver:
    def resolve_artifact_identities(self, request):
        source_refs = tuple(
            sorted({item.artifact_ref for item in request.resolved.source_plan.sources})
        )
        reward_refs = tuple(
            sorted(
                {item.artifact_ref for item in request.resolved.reward_plan.resources}
            )
        )
        return ArtifactIdentityResolution(
            model_artifact_identity=_identity("model", node_type="tree"),
            source_locations=SourceLocationBinding(
                request.resolved.source_plan.plan_id,
                tuple(
                    DatasetArtifactBinding(
                        artifact_ref=artifact_ref,
                        artifact_location=request.locations.dataset(artifact_ref),
                        expected_content_identity=_identity(
                            f"dataset:{artifact_ref}",
                            node_type="file",
                        ),
                    )
                    for artifact_ref in source_refs
                ),
            ),
            reward_artifact_identities=tuple(
                (
                    artifact_ref,
                    _identity(f"reward:{artifact_ref}", node_type="tree"),
                )
                for artifact_ref in reward_refs
            ),
            code_artifact_identity=_identity(
                "code",
                node_type="tree",
                content_policy="python-code.v1",
            ),
        )


def _environment():
    source = load_source_recipe(ROOT / "configs/v2/flow_grpo_sd3.yaml")
    static = run_static_preflight(source)
    launch = bootstrap_recipe_v2(source).require_launch()
    return run_environment_preflight(
        static,
        _ArtifactResolver(),
        artifact_locations=launch.artifacts,
    )


def _runtime_binding(environment):
    return bind_runtime(
        RuntimeBindInput(
            environment=environment,
            runtime_facts=RuntimeFacts(
                distribution_mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
                device="cpu",
                precision=(
                    environment.materialized.resolved.execution_policy.precision.value
                ),
                backend=None,
            ),
            peer_recipe_ids=(environment.materialized.recipe_id,),
        )
    )


class _Component:
    def __init__(self, slot: str, closed: list[str]) -> None:
        self.slot = slot
        self._closed = closed

    def close(self) -> None:
        self._closed.append(self.slot)


class _RecordingLoader(RuntimeComponentLoader):
    def __init__(self) -> None:
        self.calls = 0
        self.binding_set = None
        self.load_plan = None
        self.requests = ()
        self.closed: list[str] = []

    def load_all(
        self,
        requests,
        *,
        binding_set,
        load_plan,
        context_resolver=None,
    ):
        self.calls += 1
        self.binding_set = binding_set
        self.load_plan = load_plan
        self.requests = requests
        assert tuple(item.gate.artifact_binding.slot for item in requests) == (
            "model",
            "algorithm",
            "dynamics",
            "rollout",
            "rewards.reward_quality",
            "credit",
            "trainer",
        )
        assert set(binding_set.slots) == {
            item.gate.artifact_binding.slot for item in requests
        }
        assert set(load_plan.slots) == set(binding_set.slots)
        results = []
        loaded = {}
        for request in requests:
            if context_resolver is not None:
                context_resolver(request, loaded)
            result = RuntimeComponentLoadResult(
                instance=_Component(request.gate.artifact_binding.slot, self.closed),
                artifact_binding=request.gate.artifact_binding,
                load_attestation=ComponentLoadAttestation(
                    binding_id=request.gate.artifact_binding.binding_id,
                    implementation_identity=(
                        request.gate.artifact_binding.implementation_identity
                    ),
                    canonical_interface=(
                        request.gate.artifact_binding.canonical_interface
                    ),
                    interface_version=request.gate.artifact_binding.interface_version,
                ),
            )
            results.append(result)
            loaded[request.gate.artifact_binding.slot] = result.instance
        return tuple(results)


def test_component_graph_consumes_exact_environment_g1_once() -> None:
    environment = _environment()
    loader = _RecordingLoader()
    contexts: list[tuple[str, str, tuple[str, ...]]] = []

    graph = load_component_graph(
        environment,
        _runtime_binding(environment),
        runtime_context_factory=lambda slot, declaration, loaded: (
            contexts.append((slot, declaration.declaration_id, tuple(loaded))) or {}
        ),
        loader=loader,
    )

    assert loader.calls == 1
    assert loader.binding_set is environment.component_artifact_bindings
    assert loader.load_plan is environment.component_load_plan
    assert graph.artifact_binding_set is environment.component_artifact_bindings
    assert graph.load_plan is environment.component_load_plan
    assert graph.recipe_id == environment.materialized.recipe_id
    assert graph.slots == tuple(item[0] for item in contexts)
    assert tuple(loaded for _slot, _declaration_id, loaded in contexts) == tuple(
        tuple(slot for slot, _declaration_id, _loaded in contexts[:index])
        for index in range(len(contexts))
    )
    assert tuple(binding.artifact_binding for binding in graph.bindings) == tuple(
        request.gate.artifact_binding for request in loader.requests
    )


def test_launch_audit_and_host_metadata_do_not_perturb_compatible_id() -> None:
    environment = _environment()
    baseline = _runtime_binding(environment)
    relocated = bind_runtime(
        RuntimeBindInput(
            environment=environment,
            runtime_facts=RuntimeFacts(
                distribution_mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
                device="cpu",
                precision=environment.materialized.resolved.execution_policy.precision.value,
                backend=None,
                extra=FrozenMapping({"hostname": "relocated-host"}),
            ),
            peer_recipe_ids=(environment.materialized.recipe_id,),
            launch_audit=FrozenMapping(
                {
                    "schema_version": 1,
                    "reward_runtime_bindings": (
                        {
                            "logical_reward_id": "reward_quality",
                            "endpoint": "https://relocated.invalid/score",
                        },
                    ),
                }
            ),
        )
    )

    assert relocated.launch_id == baseline.launch_id
    assert relocated.launch_audit_id != baseline.launch_audit_id
    assert relocated.runtime_facts.to_payload() != baseline.runtime_facts.to_payload()
    assert relocated.runtime_facts.resume_compatibility_payload() == (
        baseline.runtime_facts.resume_compatibility_payload()
    )
    different_device = replace(relocated.runtime_facts, device="mps")
    assert (
        runtime_launch_payload_id(
            environment.materialized.recipe_id,
            different_device,
        )
        != baseline.launch_id
    )


def test_component_graph_binding_attests_g3_from_the_same_g1() -> None:
    environment = _environment()
    graph = load_component_graph(
        environment,
        _runtime_binding(environment),
        runtime_context_factory=lambda _slot, _declaration, _loaded: {},
        loader=_RecordingLoader(),
    )

    binding = graph.binding("model")
    contract = binding.attest_prepare(
        runtime_identity=_digest("runtime:model"),
        verified_fields=(("slot", "model"),),
    )

    assert contract.artifact is environment.component_artifact_bindings.binding("model")
    assert contract.component_load_attestation is binding.load_attestation
    assert contract.component_prepare_attestation.binding_id == (
        binding.artifact_binding.binding_id
    )


def test_runtime_graph_binder_consumes_typed_all_slot_g3_contracts() -> None:
    environment = _environment()
    graph = load_component_graph(
        environment,
        _runtime_binding(environment),
        runtime_context_factory=lambda _slot, _declaration, _loaded: {},
        loader=_RecordingLoader(),
    )
    launch = bind_runtime(
        RuntimeBindInput(
            environment=environment,
            runtime_facts=RuntimeFacts(
                distribution_mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
                device="cpu",
                precision=environment.materialized.resolved.execution_policy.precision.value,
                backend=None,
            ),
            peer_recipe_ids=(environment.materialized.recipe_id,),
        )
    )
    contracts = tuple(
        sorted(
            (
                binding.slot,
                binding.attest_prepare(
                    runtime_identity=_digest(f"runtime:{binding.slot}"),
                    verified_fields=(("slot", binding.slot),),
                ),
            )
            for binding in graph.bindings
        )
    )
    request = RuntimeGraphBindInput(
        environment=environment,
        launch=launch,
        runtime_bound_contracts=contracts,
        trainable_topology_id=_digest("trainable-topology"),
        prepared_component_names=tuple(sorted(graph.slots)),
        execution_transform_plan_id=(
            environment.materialized.resolved.execution_policy.transform_plan.plan_id
        ),
        resource_plan_id=_digest("resource-plan"),
        verified_fields=FrozenMapping({"graph": "verified"}),
    )

    result = bind_runtime_graph(request)

    assert result.component_bound_contract_ids == FrozenMapping(
        (slot, contract.contract_id) for slot, contract in contracts
    )
    assert set(result.component_bound_contract_ids) == set(graph.slots)
    with pytest.raises(ValidationError, match="exactly cover"):
        bind_runtime_graph(replace(request, runtime_bound_contracts=contracts[:-1]))


def test_runtime_graph_input_rejects_cross_slot_g3_contract() -> None:
    environment = _environment()
    graph = load_component_graph(
        environment,
        _runtime_binding(environment),
        runtime_context_factory=lambda _slot, _declaration, _loaded: {},
        loader=_RecordingLoader(),
    )
    model_contract = graph.binding("model").attest_prepare(
        runtime_identity=_digest("runtime:model"),
        verified_fields=(("slot", "model"),),
    )
    launch = bind_runtime(
        RuntimeBindInput(
            environment=environment,
            runtime_facts=RuntimeFacts(
                distribution_mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
                device="cpu",
                precision=environment.materialized.resolved.execution_policy.precision.value,
                backend=None,
            ),
            peer_recipe_ids=(environment.materialized.recipe_id,),
        )
    )

    with pytest.raises(ValueError, match="exact environment G1"):
        RuntimeGraphBindInput(
            environment=environment,
            launch=launch,
            runtime_bound_contracts=(("algorithm", model_contract),),
            trainable_topology_id=_digest("trainable-topology"),
            prepared_component_names=("algorithm",),
            execution_transform_plan_id=(
                environment.materialized.resolved.execution_policy.transform_plan.plan_id
            ),
            resource_plan_id=_digest("resource-plan"),
            verified_fields=FrozenMapping({"graph": "verified"}),
        )


def test_stale_environment_g1_fails_before_loader() -> None:
    environment = _environment()
    loader = _RecordingLoader()
    stale_set = environment.component_artifact_bindings
    object.__setattr__(
        stale_set,
        "recipe_id",
        f"materialized-recipe.v2:{_digest('stale')}",
    )
    # Bypass EnvironmentPreflightResult construction deliberately: this models
    # in-memory corruption after the environment gate and exercises the runtime
    # gate before the loader receives any request.
    with pytest.raises(RuntimeError, match="stale recipe"):
        load_component_graph(
            environment,
            _runtime_binding(environment),
            runtime_context_factory=lambda _slot, _declaration, _loaded: {},
            loader=loader,
        )

    assert loader.calls == 0


def test_graph_close_is_reverse_order_and_idempotent() -> None:
    loader = _RecordingLoader()
    environment = _environment()
    graph = load_component_graph(
        environment,
        _runtime_binding(environment),
        runtime_context_factory=lambda _slot, _declaration, _loaded: {},
        loader=loader,
    )

    graph.close()
    graph.close()

    assert loader.closed == list(reversed(graph.slots))
    with pytest.raises(RuntimeError, match="closed"):
        graph.binding("model")
