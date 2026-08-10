"""Runtime-only reward acquisition, identity, and ownership contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from visual_rl.composition.config.specs import RewardRuntimeBindingSpec
from visual_rl.core.contracts import (
    LogicalRewardSpec,
    MediaKind,
    RewardContract,
    RewardGranularity,
    RewardPlanSpec,
    RewardResourceSpec,
    RewardRouteBinding,
    RewardRouteSpec,
)
from visual_rl.core.types import FrozenMapping
from visual_rl.composition.preflight import RuntimeFacts
from visual_rl.algorithms.rewards import (
    RewardResourceDescriptor,
    RewardResourceState,
    RewardRuntimePolicy,
)
from visual_rl.runtime.reward_resources import (
    AcquiredRewardResource,
    RewardPoolActivationError,
    RewardPoolBuildError,
    RewardPoolCloseError,
    RewardResourceAcquireRequest,
    RewardResourceBindingFacts,
    RuntimeResourceAcquisitionError,
    bound_reward_resource_id,
    bound_reward_resource_payload,
)
from visual_rl.runtime.resources import DefaultRuntimeResourceContainer


def _component_declaration_id(logical_id: str) -> str:
    return (
        "component-declaration.v1:"
        f"{hashlib.sha256(logical_id.encode()).hexdigest()}"
    )


def _spec(artifact_ref: str) -> RewardResourceSpec:
    return RewardResourceSpec(
        descriptor=FrozenMapping(_descriptor(artifact_ref=artifact_ref).to_payload()),
        artifact_identity=FrozenMapping(
            {"content_sha256": hashlib.sha256(artifact_ref.encode()).hexdigest()}
        ),
    )


def _descriptor(
    *,
    artifact_ref: str,
    protocol: str = "visual_rl_reward_client",
    protocol_version: str = "v1",
) -> RewardResourceDescriptor:
    return RewardResourceDescriptor(
        schema_version=1,
        factory_class="mock",
        artifact_ref=artifact_ref,
        protocol=protocol,
        protocol_version=protocol_version,
        semantic_factory_config=FrozenMapping({"revision": "stable"}),
        allowed_runtime_policy=RewardRuntimePolicy(
            allowed_devices=("cpu", "cuda"),
            allowed_dtypes=("bf16", "fp32"),
            allowed_worker_domains=("in_process", "remote"),
        ),
    )


def _facts(**overrides: str) -> RewardResourceBindingFacts:
    values = {
        "endpoint_identity": "e" * 64,
        "protocol": "visual_rl_reward_client",
        "protocol_version": "v1",
        "device": "cpu",
        "dtype": "fp32",
        "worker_domain": "in_process",
    }
    values.update(overrides)
    return RewardResourceBindingFacts(**values)


def _runtime_facts() -> RuntimeFacts:
    return RuntimeFacts(
        distribution_mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        device="cpu",
        precision="fp32",
        backend=None,
    )


def _plan(
    bindings: tuple[tuple[str, RewardResourceSpec, float], ...],
) -> RewardPlanSpec:
    resources = tuple(dict.fromkeys(spec for _logical_id, spec, _weight in bindings))
    return RewardPlanSpec(
        resources=resources,
        logical_rewards=tuple(
            LogicalRewardSpec(
                logical_reward_id=logical_id,
                component_declaration_id=_component_declaration_id(logical_id),
                resource_identity=spec.resource_identity,
                contract=RewardContract(
                    accepted_media=(MediaKind.IMAGE,),
                    required_payload_type=None,
                    granularity=RewardGranularity.POINTWISE,
                    output_rank=1,
                    frame_aggregation=None,
                ),
            )
            for logical_id, spec, _weight in bindings
        ),
        routes=(
            RewardRouteSpec(
                source_id="main",
                phase_id="main",
                rewards=tuple(
                    RewardRouteBinding(logical_reward_id=logical_id, weight=weight)
                    for logical_id, _spec, weight in bindings
                ),
            ),
        )
    )


def _request(
    tmp_path,
    resource_spec: RewardResourceSpec,
    *,
    location_name: str | None = None,
) -> RewardResourceAcquireRequest:
    descriptor = RewardResourceDescriptor.from_mapping(resource_spec.descriptor)
    artifact_identity = resource_spec.artifact_identity
    assert artifact_identity is not None
    return RewardResourceAcquireRequest(
        reward_resource_spec_id=resource_spec.resource_identity,
        descriptor=descriptor,
        immutable_artifact_identity=artifact_identity,
        artifact_location=(
            tmp_path / (location_name or f"{resource_spec.artifact_ref}.artifact")
        ).resolve(),
        runtime_facts=_runtime_facts(),
    )


class _Resource:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        activation_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.activation_error = activation_error
        self.close_error = close_error
        self.activate_calls = 0
        self.close_calls = 0

    def activate(self) -> None:
        self.activate_calls += 1
        self.events.append(f"{self.name}.activate")
        if self.activation_error is not None:
            raise self.activation_error

    def close(self) -> None:
        self.close_calls += 1
        self.events.append(f"{self.name}.close")
        if self.close_error is not None:
            raise self.close_error


class _Factory:
    def __init__(
        self,
        facts_by_spec: dict[str, RewardResourceBindingFacts],
        *,
        events: list[str] | None = None,
        resources: dict[str, _Resource] | None = None,
    ) -> None:
        self.facts_by_spec = facts_by_spec
        self.events = [] if events is None else events
        self.resources = {} if resources is None else resources
        self.requests: list[RewardResourceAcquireRequest] = []

    def __call__(
        self,
        request: RewardResourceAcquireRequest,
    ) -> AcquiredRewardResource:
        self.requests.append(request)
        spec_id = request.reward_resource_spec_id
        resource = self.resources.setdefault(
            spec_id,
            _Resource(spec_id[-1], self.events),
        )
        return AcquiredRewardResource(resource, self.facts_by_spec[spec_id])


def test_container_deduplicates_specs_and_owns_activation_and_close(tmp_path) -> None:
    spec = _spec("quality")
    spec_id = spec.resource_identity
    plan = _plan(
        (
            ("quality", spec, 1.0),
            ("guard", spec, 0.25),
        )
    )
    events: list[str] = []
    factory = _Factory({spec_id: _facts()}, events=events)
    container = DefaultRuntimeResourceContainer(factory)

    view = container.acquire(
        plan,
        (_request(tmp_path, spec),),
    )

    assert container.state is RewardResourceState.ACQUIRED
    assert len(factory.requests) == 1
    assert view.resource_identities == (spec_id,)
    assert not hasattr(view, "close")
    handle = view.handle(spec_id)
    assert handle.resource_identity == spec_id
    assert handle.state is RewardResourceState.ACQUIRED
    assert container.view() is view
    assert set(container.bound_reward_resource_ids) == {spec_id}
    assert isinstance(container.bound_reward_resource_ids, FrozenMapping)

    container.activate()
    assert container.state is RewardResourceState.ACTIVE
    assert container.is_active
    assert view.get(spec_id) is factory.resources[spec_id]
    assert events == [f"{spec_id[-1]}.activate"]

    container.close()
    container.close()
    assert container.state is RewardResourceState.CLOSED
    assert handle.state is RewardResourceState.CLOSED
    assert events == [f"{spec_id[-1]}.activate", f"{spec_id[-1]}.close"]
    assert factory.resources[spec_id].close_calls == 1


@pytest.mark.parametrize(
    "drifted_field",
    (
        "descriptor",
        "immutable_artifact_identity",
        "artifact_location",
        "runtime_facts",
        "runtime_binding",
    ),
)
def test_preacquired_fingerprints_exact_match_every_factory_input(
    tmp_path,
    drifted_field: str,
) -> None:
    spec = _spec("quality")
    spec_id = spec.resource_identity
    plan = _plan((("quality", spec, 1.0),))
    baseline = _request(tmp_path, spec)
    factory = _Factory({spec_id: _facts()})
    container = DefaultRuntimeResourceContainer(factory)
    container.acquire(plan, (baseline,))

    fingerprints = container.acquisition_request_fingerprints
    assert tuple(fingerprints) == (spec_id,)
    assert isinstance(fingerprints, FrozenMapping)
    assert len(fingerprints[spec_id]) == 64
    assert all(character in "0123456789abcdef" for character in fingerprints[spec_id])
    container.assert_acquisition_requests_match(plan, (baseline,))

    changes = {
        "descriptor": replace(
            baseline.descriptor,
            semantic_factory_config=FrozenMapping({"revision": "drifted"}),
        ),
        "immutable_artifact_identity": FrozenMapping({"content_sha256": "f" * 64}),
        "artifact_location": (tmp_path / "different.artifact").resolve(),
        "runtime_facts": replace(
            baseline.runtime_facts,
            extra=FrozenMapping({"worker_revision": "drifted"}),
        ),
        "runtime_binding": RewardRuntimeBindingSpec(
            artifact_ref="quality",
            execution_domain="in_process",
            device="cpu",
            dtype="fp32",
        ),
    }
    drifted = replace(baseline, **{drifted_field: changes[drifted_field]})

    with pytest.raises(
        RuntimeResourceAcquisitionError,
        match="request fingerprints differ",
    ):
        container.assert_acquisition_requests_match(plan, (drifted,))

    assert container.acquisition_request_fingerprints == fingerprints
    assert len(factory.requests) == 1
    assert container.state is RewardResourceState.ACQUIRED
    assert not container.is_active


def test_bound_identity_is_runtime_only_and_excludes_location_and_logical_route(
    tmp_path,
) -> None:
    spec = _spec("quality")
    spec_id = spec.resource_identity
    facts = _facts()
    payload = bound_reward_resource_payload(spec_id, facts)

    assert payload == {
        "schema_version": 1,
        "reward_resource_spec_id": spec_id,
        "actual": facts.to_payload(),
    }
    encoded = repr(payload).lower()
    assert "path" not in encoded
    assert "logical" not in encoded
    assert "weight" not in encoded
    canonical_without_domain = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert (
        bound_reward_resource_id(spec_id, facts)
        != hashlib.sha256(canonical_without_domain).hexdigest()
    )

    first_plan = _plan((("quality", spec, 1.0),))
    second_plan = _plan((("renamed", spec, 9.0),))
    first = DefaultRuntimeResourceContainer(_Factory({spec_id: facts}))
    second = DefaultRuntimeResourceContainer(_Factory({spec_id: facts}))
    first.acquire(
        first_plan,
        (
            _request(
                tmp_path,
                spec,
                location_name="first/location",
            ),
        ),
    )
    second.acquire(
        second_plan,
        (
            _request(
                tmp_path,
                spec,
                location_name="second/location",
            ),
        ),
    )

    assert first.bound_reward_resource_ids == second.bound_reward_resource_ids
    first.close()
    second.close()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("endpoint_identity", "f" * 64),
        ("protocol", "other_protocol"),
        ("protocol_version", "v2"),
        ("device", "cuda:1"),
        ("dtype", "bf16"),
        ("worker_domain", "remote"),
    ),
)
def test_each_actual_runtime_fact_changes_bound_identity(field, value) -> None:
    spec_id = _spec("quality").resource_identity
    baseline = _facts()
    changed = replace(baseline, **{field: value})

    assert bound_reward_resource_id(spec_id, baseline) != (
        bound_reward_resource_id(spec_id, changed)
    )


def test_requests_must_exactly_cover_unique_plan_specs_before_factory_call(
    tmp_path,
) -> None:
    first_declared = _spec("first")
    second_declared = _spec("second")
    plan = _plan(
        (
            ("first", first_declared, 1.0),
            ("second", second_declared, 1.0),
        )
    )
    first_spec, second_spec = plan.resources
    first_id, second_id = plan.resource_identities
    factory = _Factory({first_id: _facts(), second_id: _facts()})
    container = DefaultRuntimeResourceContainer(factory)

    with pytest.raises(ValueError, match="exactly cover"):
        container.acquire(
            plan,
            (_request(tmp_path, first_spec),),
        )

    assert factory.requests == []
    assert container.state is RewardResourceState.CLOSED
    with pytest.raises(RuntimeError, match="acquire-once"):
        container.acquire(
            plan,
            (
                _request(tmp_path, first_spec),
                _request(tmp_path, second_spec),
            ),
        )


@pytest.mark.parametrize(
    ("facts", "message"),
    (
        (_facts(protocol="other_protocol"), "protocol differs"),
        (_facts(protocol_version="v2"), "protocol version differs"),
        (_facts(device="mps"), "device"),
        (_facts(dtype="fp16"), "dtype"),
        (_facts(worker_domain="isolated"), "worker domain"),
    ),
)
def test_invalid_observed_facts_close_candidate_and_prior_resources(
    tmp_path,
    facts,
    message,
) -> None:
    first_declared = _spec("first")
    second_declared = _spec("second")
    plan = _plan(
        (
            ("first", first_declared, 1.0),
            ("second", second_declared, 1.0),
        )
    )
    first_spec, second_spec = plan.resources
    first_id, second_id = plan.resource_identities
    events: list[str] = []
    factory = _Factory(
        {first_id: _facts(), second_id: facts},
        events=events,
    )
    container = DefaultRuntimeResourceContainer(factory)

    with pytest.raises(RewardPoolBuildError) as caught:
        container.acquire(
            plan,
            (
                _request(tmp_path, first_spec),
                _request(tmp_path, second_spec),
            ),
        )

    assert isinstance(caught.value.__cause__, RuntimeResourceAcquisitionError)
    assert message in str(caught.value.__cause__)
    assert events == [f"{second_id[-1]}.close", f"{first_id[-1]}.close"]
    assert all(resource.close_calls == 1 for resource in factory.resources.values())
    assert all(resource.activate_calls == 0 for resource in factory.resources.values())
    assert container.state is RewardResourceState.CLOSED
    container.close()
    assert events == [f"{second_id[-1]}.close", f"{first_id[-1]}.close"]


def test_reused_physical_object_fails_closed_without_double_close(tmp_path) -> None:
    first_declared = _spec("first")
    second_declared = _spec("second")
    plan = _plan(
        (
            ("first", first_declared, 1.0),
            ("second", second_declared, 1.0),
        )
    )
    first_spec, second_spec = plan.resources
    events: list[str] = []
    shared = _Resource("shared", events)

    def factory(request: RewardResourceAcquireRequest) -> AcquiredRewardResource:
        return AcquiredRewardResource(shared, _facts())

    container = DefaultRuntimeResourceContainer(factory)
    with pytest.raises(RewardPoolBuildError) as caught:
        container.acquire(
            plan,
            (
                _request(tmp_path, first_spec),
                _request(tmp_path, second_spec),
            ),
        )

    assert isinstance(caught.value.__cause__, RuntimeResourceAcquisitionError)
    assert shared.close_calls == 1
    assert events == ["shared.close"]
    assert container.state is RewardResourceState.CLOSED


def test_partial_factory_failure_closes_prior_resource_and_container(tmp_path) -> None:
    plan = _plan(
        (
            ("first", _spec("first"), 1.0),
            ("second", _spec("second"), 1.0),
        )
    )
    first_spec, second_spec = plan.resources
    second_id = second_spec.resource_identity
    events: list[str] = []
    first = _Resource("first", events)

    def factory(
        request: RewardResourceAcquireRequest,
    ) -> AcquiredRewardResource:
        if request.reward_resource_spec_id == second_id:
            raise LookupError("factory failed")
        return AcquiredRewardResource(first, _facts())

    container = DefaultRuntimeResourceContainer(factory)
    with pytest.raises(RewardPoolBuildError) as caught:
        container.acquire(
            plan,
            (
                _request(tmp_path, first_spec),
                _request(tmp_path, second_spec),
            ),
        )

    assert isinstance(caught.value.__cause__, LookupError)
    assert first.close_calls == 1
    assert first.activate_calls == 0
    assert events == ["first.close"]
    assert container.state is RewardResourceState.CLOSED
    container.close()
    assert events == ["first.close"]


def test_invalid_factory_output_cleanup_error_does_not_replace_type_error(
    tmp_path,
) -> None:
    spec = _spec("quality")
    plan = _plan((("quality", spec, 1.0),))
    events: list[str] = []
    invalid = _Resource(
        "invalid",
        events,
        close_error=RuntimeError("cleanup failed"),
    )

    def factory(_request: RewardResourceAcquireRequest) -> object:
        return invalid

    container = DefaultRuntimeResourceContainer(factory)  # type: ignore[arg-type]
    with pytest.raises(RewardPoolBuildError) as caught:
        container.acquire(
            plan,
            (_request(tmp_path, spec),),
        )

    primary = caught.value.__cause__
    assert isinstance(primary, TypeError)
    assert "AcquiredRewardResource" in str(primary)
    assert invalid.close_calls == 1
    notes = getattr(primary, "__notes__", ())
    if hasattr(primary, "add_note"):
        assert any("invalid factory output rollback failed" in note for note in notes)
    assert container.state is RewardResourceState.CLOSED


def test_activation_failure_closes_every_handle_in_reverse_and_is_idempotent(
    tmp_path,
) -> None:
    declared_specs = (_spec("first"), _spec("second"), _spec("third"))
    plan = _plan(
        tuple(
            (f"logical-{index}", spec, 1.0)
            for index, spec in enumerate(declared_specs)
        )
    )
    specs = plan.resources
    spec_ids = plan.resource_identities
    events: list[str] = []
    resources = {
        spec_ids[0]: _Resource("first", events),
        spec_ids[1]: _Resource(
            "second",
            events,
            activation_error=LookupError("activation failed"),
        ),
        spec_ids[2]: _Resource("third", events),
    }
    factory = _Factory(
        {spec_id: _facts() for spec_id in spec_ids},
        events=events,
        resources=resources,
    )
    container = DefaultRuntimeResourceContainer(factory)
    view = container.acquire(
        plan,
        tuple(
            _request(tmp_path, spec) for spec in specs
        ),
    )

    with pytest.raises(RewardPoolActivationError) as caught:
        container.activate()

    assert isinstance(caught.value.__cause__, LookupError)
    assert events == [
        "first.activate",
        "second.activate",
        "third.close",
        "second.close",
        "first.close",
    ]
    assert container.state is RewardResourceState.CLOSED
    assert all(
        view.handle(spec_id).state is RewardResourceState.CLOSED
        for spec_id in spec_ids
    )
    container.close()
    assert len(events) == 5
    with pytest.raises(RuntimeError, match="unavailable"):
        container.view()


def test_close_attempts_every_resource_in_reverse_even_when_one_close_fails(
    tmp_path,
) -> None:
    plan = _plan(
        (
            ("first", _spec("first"), 1.0),
            ("second", _spec("second"), 1.0),
        )
    )
    specs = plan.resources
    spec_ids = plan.resource_identities
    events: list[str] = []
    resources = {
        spec_ids[0]: _Resource("first", events),
        spec_ids[1]: _Resource(
            "second",
            events,
            close_error=RuntimeError("close failed"),
        ),
    }
    container = DefaultRuntimeResourceContainer(
        _Factory(
            {spec_id: _facts() for spec_id in spec_ids},
            events=events,
            resources=resources,
        )
    )
    container.acquire(
        plan,
        tuple(_request(tmp_path, spec) for spec in specs),
    )

    with pytest.raises(RewardPoolCloseError):
        container.close()

    assert events == ["second.close", "first.close"]
    assert container.state is RewardResourceState.CLOSED
    container.close()
    assert len(events) == 2
