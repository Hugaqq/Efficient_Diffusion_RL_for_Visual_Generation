"""Contracts for resolved reward routing, shared resources, and aggregation."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

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
from visual_rl.algorithms.rewards import (
    GroupwiseReward,
    GroupwiseRewardOutput,
    PointwiseReward,
    PointwiseRewardOutput,
    RewardBatchIdentity,
    RewardBatchView,
    RewardProcessor,
    RewardResourceState,
    RewardResult,
)
from visual_rl.runtime.reward_resources import (
    RewardPool,
    RewardPoolBuildError,
    RewardPoolView,
)
from visual_rl.core.types import FrozenMapping


def _component_declaration_id(logical_id: str) -> str:
    return (
        "component-declaration.v1:"
        f"{hashlib.sha256(logical_id.encode()).hexdigest()}"
    )


def _contract(protocol: str) -> RewardContract:
    return RewardContract(
        accepted_media=(MediaKind.IMAGE, MediaKind.VIDEO),
        required_payload_type=None,
        granularity=RewardGranularity(protocol),
        output_rank=1,
        frame_aggregation=None,
    )


def _plan(
    routes: tuple[
        tuple[str, str, tuple[tuple[str, str, str, float], ...]], ...
    ],
) -> RewardPlanSpec:
    resource_labels = tuple(
        sorted(
            {
                resource_label
                for _source, _phase, bindings in routes
                for _logical_id, resource_label, _protocol, _weight in bindings
            }
        )
    )
    resources_by_label = {
        label: RewardResourceSpec(
            descriptor=FrozenMapping({"artifact_ref": label}),
            artifact_identity=FrozenMapping(
                {"content_sha256": hashlib.sha256(label.encode()).hexdigest()}
            ),
        )
        for label in resource_labels
    }
    logical_inputs: dict[str, tuple[str, str]] = {}
    for _source, _phase, bindings in routes:
        for logical_id, resource_label, protocol, _weight in bindings:
            value = (resource_label, protocol)
            previous = logical_inputs.setdefault(logical_id, value)
            if previous != value:
                raise ValueError(f"logical reward {logical_id!r} changes its contract")
    return RewardPlanSpec(
        resources=tuple(resources_by_label.values()),
        logical_rewards=tuple(
            LogicalRewardSpec(
                logical_reward_id=logical_id,
                component_declaration_id=_component_declaration_id(logical_id),
                resource_identity=resources_by_label[resource_label].resource_identity,
                contract=_contract(protocol),
            )
            for logical_id, (resource_label, protocol) in logical_inputs.items()
        ),
        routes=tuple(
            RewardRouteSpec(
                source_id=source,
                phase_id=phase,
                rewards=tuple(
                    RewardRouteBinding(logical_reward_id=logical_id, weight=weight)
                    for logical_id, _resource_label, _protocol, weight in bindings
                ),
            )
            for source, phase, bindings in routes
        ),
    )


def _resource_id(plan: RewardPlanSpec, artifact_ref: str) -> str:
    return next(
        resource.resource_identity
        for resource in plan.resources
        if resource.artifact_ref == artifact_ref
    )


def _main_plan() -> RewardPlanSpec:
    return _plan(
        (
            (
                "main-source",
                "main",
                (
                    ("quality", "shared-model", "pointwise", 0.5),
                    ("coherence", "shared-model", "groupwise", 2.0),
                ),
            ),
            (
                "dynamic-source",
                "dynamic",
                (("quality", "shared-model", "pointwise", 1.0),),
            ),
        )
    )


def _batch(
    *,
    source_id: str = "main-source",
    phase_id: str = "main",
    active_reward_ids: tuple[str, ...] = ("coherence", "quality"),
) -> RewardBatchView:
    return RewardBatchView(
        identity=RewardBatchIdentity(
            source_id=source_id,
            phase_id=phase_id,
            batch_row_ids=("row-0", "row-1", "row-2", "row-3"),
            sample_ids=("sample-0", "sample-1", "sample-2", "sample-3"),
            trajectory_ids=("traj-0", "traj-1", "traj-2", "traj-3"),
            condition_payload_ids=("camera-a",) * 2 + ("camera-b",) * 2,
            group_ids=("group-a", "group-a", "group-b", "group-b"),
        ),
        active_reward_ids=active_reward_ids,
        payload={"media": object()},
    )


class _Resource:
    def __init__(self, identity: str, events: list[str], *, fail_close: bool = False):
        self.identity = identity
        self.events = events
        self.fail_close = fail_close
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.events.append(f"close:{self.identity}")
        if self.fail_close:
            raise RuntimeError(f"close failed: {self.identity}")


class _Pointwise(PointwiseReward):
    def __init__(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray,
        *,
        identity_override: RewardBatchIdentity | None = None,
        execution_provenance: FrozenMapping | None = None,
    ) -> None:
        self.values = values
        self.valid_mask = valid_mask
        self.identity_override = identity_override
        self.execution_provenance = execution_provenance or FrozenMapping()

    def score(self, *, logical_reward_id, resource, batch):
        assert logical_reward_id == "quality"
        assert resource.identity.startswith("reward-resource-spec.v1:")
        return PointwiseRewardOutput(
            identity=self.identity_override or batch.identity,
            values=self.values,
            valid_mask=self.valid_mask,
            execution_provenance=self.execution_provenance,
        )


class _Groupwise(GroupwiseReward):
    def __init__(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray,
        *,
        output_group_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.values = values
        self.valid_mask = valid_mask
        self.output_group_ids = output_group_ids

    def score_groups(
        self,
        *,
        logical_reward_id,
        resource,
        batch,
        group_ids,
    ):
        assert logical_reward_id == "coherence"
        assert resource.identity.startswith("reward-resource-spec.v1:")
        return GroupwiseRewardOutput(
            identity=batch.identity,
            group_ids=self.output_group_ids or group_ids,
            values=self.values,
            valid_mask=self.valid_mask,
        )


def _processor(
    *,
    pointwise: PointwiseReward | None = None,
    groupwise: GroupwiseReward | None = None,
) -> tuple[RewardProcessor, RewardPool, list[str]]:
    plan = _main_plan()
    events: list[str] = []

    def factory(identity: str) -> _Resource:
        events.append(f"build:{identity}")
        return _Resource(identity, events)

    pool = RewardPool(plan, factory)
    processor = RewardProcessor(
        plan=plan,
        pool=pool.view(),
        logical_rewards={
            "coherence": groupwise
            or _Groupwise(
                np.array([10.0, 20.0]),
                np.array([True, False]),
            ),
            "quality": pointwise
            or _Pointwise(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([True, True, False, True]),
                execution_provenance=FrozenMapping(
                    {"selection_key_id": "selection-key"}
                ),
            ),
        },
    )
    pool.activate()
    return processor, pool, events


def test_plan_separates_logical_and_resource_identity_and_is_immutable() -> None:
    plan = _main_plan()
    main = plan.route_for(source_id="main-source", phase_id="main")
    dynamic = plan.route_for(source_id="dynamic-source", phase_id="dynamic")

    assert main.logical_reward_ids == ("coherence", "quality")
    assert dynamic.logical_reward_ids == ("quality",)
    assert plan.logical_reward_ids == ("coherence", "quality")
    assert plan.resource_identities == (_resource_id(plan, "shared-model"),)
    assert main.rewards[0].logical_reward_id != main.rewards[1].logical_reward_id
    assert (
        plan.logical_reward(main.rewards[0].logical_reward_id).resource_identity
        == plan.logical_reward(main.rewards[1].logical_reward_id).resource_identity
    )
    assert plan.to_payload()["schema_version"] == 1
    assert main.binding("quality").weight == 0.5
    assert dynamic.binding("quality").weight == 1.0
    with pytest.raises(FrozenInstanceError):
        main.rewards[0].weight = 9.0
    with pytest.raises(KeyError, match="unknown reward route"):
        plan.route_for(source_id="missing", phase_id="main")


def test_plan_rejects_route_and_cross_route_identity_ambiguity() -> None:
    duplicate = RewardRouteSpec(
        source_id="source",
        phase_id="phase",
        rewards=(RewardRouteBinding("reward", 1.0),),
    )
    resource = RewardResourceSpec(
        descriptor=FrozenMapping({"artifact_ref": "resource"}),
        artifact_identity=FrozenMapping({"content_sha256": "a" * 64}),
    )
    logical = LogicalRewardSpec(
        logical_reward_id="reward",
        component_declaration_id=_component_declaration_id("reward"),
        resource_identity=resource.resource_identity,
        contract=_contract("pointwise"),
    )
    with pytest.raises(ValueError, match="routes must be unique"):
        RewardPlanSpec(
            resources=(resource,),
            logical_rewards=(logical,),
            routes=(duplicate, duplicate),
        )
    unknown = replace(resource, descriptor=FrozenMapping({"artifact_ref": "other"}))
    with pytest.raises(ValueError, match="logical rewards/resources differ"):
        RewardPlanSpec(
            resources=(resource,),
            logical_rewards=(replace(logical, resource_identity=unknown.resource_identity),),
            routes=(duplicate,),
        )


def test_pool_constructs_shared_identity_once_and_closes_reverse_once() -> None:
    plan = _plan(
        (
            (
                "source",
                "phase",
                (
                    ("a", "resource-a", "pointwise", 1.0),
                    ("b", "resource-b", "pointwise", 1.0),
                    ("c", "resource-a", "pointwise", 1.0),
                ),
            ),
        )
    )
    resource_a = _resource_id(plan, "resource-a")
    resource_b = _resource_id(plan, "resource-b")
    events: list[str] = []
    resources: dict[str, _Resource] = {}

    def factory(identity: str) -> _Resource:
        events.append(f"build:{identity}")
        resource = _Resource(identity, events)
        resources[identity] = resource
        return resource

    pool = RewardPool(plan, factory)
    view = pool.view()

    assert pool.resource_identities == plan.resource_identities
    assert isinstance(view, RewardPoolView)
    assert not hasattr(view, "close")
    assert view.handle(resource_a) is view.handle(resource_a)
    assert view.handle(resource_a).state is RewardResourceState.ACQUIRED
    with pytest.raises(RuntimeError, match="ACTIVE"):
        view.get(resource_a)
    assert events == [f"build:{identity}" for identity in plan.resource_identities]
    pool.activate()
    assert view.get(resource_a) is resources[resource_a]
    assert view.handle(resource_a).state is RewardResourceState.ACTIVE
    pool.close()
    pool.close()

    assert events[-2:] == [
        f"close:{identity}" for identity in reversed(plan.resource_identities)
    ]
    assert resources[resource_a].close_count == 1
    assert resources[resource_b].close_count == 1
    assert view.handle(resource_a).state is RewardResourceState.CLOSED


def test_explicit_activation_calls_optional_physical_hook_once() -> None:
    plan = _plan(
        (("source", "phase", (("a", "resource-a", "pointwise", 1.0),)),)
    )
    resource_a = _resource_id(plan, "resource-a")
    events: list[str] = []

    class _ActivatingResource(_Resource):
        def activate(self) -> None:
            events.append(f"activate:{resource_a}")

    pool = RewardPool(
        plan,
        lambda identity: _ActivatingResource(identity, events),
    )
    assert events == []
    pool.activate()
    assert events == [f"activate:{resource_a}"]
    with pytest.raises(RuntimeError, match="already ACTIVE"):
        pool.activate()
    pool.close()
    assert events == [f"activate:{resource_a}", f"close:{resource_a}"]


def test_processor_rejects_execution_before_owner_activation() -> None:
    plan = _main_plan()
    events: list[str] = []
    pool = RewardPool(plan, lambda identity: _Resource(identity, events))
    point = _Pointwise(np.ones(4), np.ones(4, dtype=np.bool_))
    group = _Groupwise(np.ones(2), np.ones(2, dtype=np.bool_))
    processor = RewardProcessor(
        plan=plan,
        pool=pool.view(),
        logical_rewards={"coherence": group, "quality": point},
    )

    with pytest.raises(RuntimeError, match="ACTIVE"):
        processor.process(
            batch=_batch(),
            route=plan.route_for(source_id="main-source", phase_id="main"),
        )
    pool.close()


def test_pool_partial_construction_failure_unwinds_prior_resources_reverse() -> None:
    plan = _plan(
        (
            (
                "source",
                "phase",
                (
                    ("a", "resource-a", "pointwise", 1.0),
                    ("b", "resource-b", "pointwise", 1.0),
                    ("c", "resource-c", "pointwise", 1.0),
                ),
            ),
        )
    )
    failure_identity = plan.resource_identities[-1]
    events: list[str] = []

    def factory(identity: str) -> _Resource:
        events.append(f"build:{identity}")
        if identity == failure_identity:
            raise RuntimeError("injected build failure")
        return _Resource(identity, events)

    with pytest.raises(RewardPoolBuildError) as caught:
        RewardPool(plan, factory)

    assert caught.value.resource_identity == failure_identity
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.cleanup_errors == ()
    assert events == [
        *(f"build:{identity}" for identity in plan.resource_identities),
        *(f"close:{identity}" for identity in reversed(plan.resource_identities[:-1])),
    ]


def test_pool_partial_failure_continues_cleanup_and_reports_close_errors() -> None:
    plan = _plan(
        (
            (
                "source",
                "phase",
                (
                    ("a", "resource-a", "pointwise", 1.0),
                    ("b", "resource-b", "pointwise", 1.0),
                    ("c", "resource-c", "pointwise", 1.0),
                ),
            ),
        )
    )
    close_failure_identity = plan.resource_identities[0]
    build_failure_identity = plan.resource_identities[-1]
    events: list[str] = []

    def factory(identity: str) -> _Resource:
        events.append(f"build:{identity}")
        if identity == build_failure_identity:
            raise RuntimeError("injected build failure")
        return _Resource(
            identity,
            events,
            fail_close=identity == close_failure_identity,
        )

    with pytest.raises(RewardPoolBuildError) as caught:
        RewardPool(plan, factory)

    assert len(caught.value.cleanup_errors) == 1
    assert close_failure_identity in str(caught.value.cleanup_errors[0])
    assert events[-2:] == [
        f"close:{identity}" for identity in reversed(plan.resource_identities[:-1])
    ]


def test_processor_aligns_group_scores_preserves_names_and_resolved_weights() -> None:
    processor, pool, events = _processor()
    batch = _batch()
    route = processor.plan.route_for(source_id="main-source", phase_id="main")

    result = processor.process(batch=batch, route=route)

    np.testing.assert_array_equal(
        result.component_scores["quality"],
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    np.testing.assert_array_equal(
        result.component_scores["coherence"],
        np.array([10.0, 10.0, 20.0, 20.0]),
    )
    np.testing.assert_array_equal(
        result.weighted_scores["quality"],
        np.array([0.5, 1.0, 1.5, 2.0]),
    )
    np.testing.assert_array_equal(
        result.weighted_scores["coherence"],
        np.array([20.0, 20.0, 40.0, 40.0]),
    )
    np.testing.assert_array_equal(
        result.weighted_total,
        np.array([20.5, 21.0, 41.5, 42.0]),
    )
    np.testing.assert_array_equal(
        result.valid_mask,
        np.array([True, True, False, False]),
    )
    assert tuple(result.component_scores) == ("coherence", "quality")
    shared_resource_id = _resource_id(processor.plan, "shared-model")
    assert dict(result.resource_identities) == {
        "coherence": shared_resource_id,
        "quality": shared_resource_id,
    }
    assert dict(result.logical_weights) == {"quality": 0.5, "coherence": 2.0}
    assert result.logical_execution_provenance["quality"] == {
        "selection_key_id": "selection-key"
    }
    assert result.logical_execution_provenance["coherence"] == {}
    assert all(bool(mask.all()) for mask in result.component_applicable_masks.values())
    assert set(result.component_applicable_masks) == {
        "quality",
        "coherence",
    }
    assert result.identity is batch.identity
    assert not result.weighted_total.flags.writeable
    assert events == [f"build:{shared_resource_id}"]
    pool.close()


def test_reward_result_rejects_inconsistent_total_or_valid_mask() -> None:
    processor, pool, _events = _processor()
    route = processor.plan.route_for(source_id="main-source", phase_id="main")
    result = processor.process(batch=_batch(), route=route)

    with pytest.raises(ValueError, match="weighted_total"):
        replace(result, weighted_total=np.zeros(4))
    with pytest.raises(ValueError, match="valid_mask"):
        replace(result, valid_mask=np.ones(4, dtype=np.bool_))
    pool.close()


def test_reward_result_separates_applicability_from_execution_validity() -> None:
    identity = _batch().identity
    result = RewardResult(
        identity=identity,
        component_scores={
            "image": np.array([1.0, 0.0, 3.0, 0.0]),
            "video": np.array([0.0, 2.0, 0.0, 4.0]),
        },
        weighted_scores={
            "image": np.array([0.5, 0.0, 1.5, 0.0]),
            "video": np.array([0.0, 4.0, 0.0, 8.0]),
        },
        component_valid_masks={
            "image": np.array([True, False, False, False]),
            "video": np.array([False, True, False, True]),
        },
        component_applicable_masks={
            "image": np.array([True, False, True, False]),
            "video": np.array([False, True, False, True]),
        },
        weighted_total=np.array([0.5, 4.0, 1.5, 8.0]),
        valid_mask=np.array([True, True, False, True]),
        resource_identities={"image": "shared", "video": "shared"},
        logical_weights={"image": 0.5, "video": 2.0},
        logical_provenance={
            "image": FrozenMapping({"paper": "image-r1"}),
            "video": FrozenMapping({"paper": "video-r1"}),
        },
    )

    np.testing.assert_array_equal(
        result.component_applicable_masks["image"],
        np.array([True, False, True, False]),
    )
    np.testing.assert_array_equal(
        result.component_valid_masks["image"],
        np.array([True, False, False, False]),
    )
    np.testing.assert_array_equal(
        result.valid_mask,
        np.array([True, True, False, True]),
    )

    with pytest.raises(ValueError, match="finite"):
        replace(
            result,
            component_scores={
                "image": np.array([1.0, np.nan, 3.0, 0.0]),
                "video": result.component_scores["video"],
            },
        )
    with pytest.raises(ValueError, match="cannot mark non-applicable"):
        replace(
            result,
            component_valid_masks={
                "image": np.array([True, True, False, False]),
                "video": result.component_valid_masks["video"],
            },
        )
    with pytest.raises(ValueError, match="times logical weight"):
        replace(result, logical_weights={"image": 1.0, "video": 2.0})


class _NamedPointwise(PointwiseReward):
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, *, logical_reward_id, resource, batch):
        assert logical_reward_id in {"semantic", "camera"}
        assert resource.identity.startswith("reward-resource-spec.v1:")
        return PointwiseRewardOutput(
            identity=batch.identity,
            values=np.full(batch.batch_size, self.value),
            valid_mask=np.ones(batch.batch_size, dtype=np.bool_),
        )


def test_resource_dedup_preserves_logical_weight_and_provenance() -> None:
    plan = _plan(
        (
            (
                "main-source",
                "main",
                (
                    ("semantic", "shared-model", "pointwise", 0.25),
                    ("camera", "shared-model", "pointwise", 3.0),
                ),
            ),
        )
    )
    events: list[str] = []

    def factory(identity: str) -> _Resource:
        events.append(f"build:{identity}")
        return _Resource(identity, events)

    pool = RewardPool(plan, factory)
    processor = RewardProcessor(
        plan=plan,
        pool=pool.view(),
        logical_rewards={
            "camera": _NamedPointwise(5.0),
            "semantic": _NamedPointwise(2.0),
        },
    )
    pool.activate()
    result = processor.process(
        batch=_batch(active_reward_ids=("camera", "semantic")),
        route=plan.route_for(source_id="main-source", phase_id="main"),
    )

    shared_resource_id = _resource_id(plan, "shared-model")
    assert events == [f"build:{shared_resource_id}"]
    assert tuple(result.component_scores) == ("camera", "semantic")
    assert dict(result.logical_weights) == {"camera": 3.0, "semantic": 0.25}
    assert result.logical_provenance["semantic"]["component_declaration_id"] == (
        _component_declaration_id("semantic")
    )
    assert result.logical_provenance["camera"]["component_declaration_id"] == (
        _component_declaration_id("camera")
    )
    assert result.resource_identities["semantic"] == shared_resource_id
    assert result.resource_identities["camera"] == shared_resource_id
    pool.close()


@pytest.mark.parametrize(
    "identity_update",
    (
        {"batch_row_ids": ("wrong", "row-1", "row-2", "row-3")},
        {"sample_ids": ("wrong", "sample-1", "sample-2", "sample-3")},
        {"trajectory_ids": ("wrong", "traj-1", "traj-2", "traj-3")},
        {
            "condition_payload_ids": (
                "wrong",
                "camera-a",
                "camera-b",
                "camera-b",
            )
        },
    ),
)
def test_processor_rejects_row_sample_trajectory_and_condition_identity_drift(
    identity_update: dict[str, tuple[str, ...]],
) -> None:
    batch = _batch()
    wrong_identity = replace(batch.identity, **identity_update)
    processor, pool, _events = _processor(
        pointwise=_Pointwise(
            np.ones(4),
            np.ones(4, dtype=np.bool_),
            identity_override=wrong_identity,
        )
    )
    route = processor.plan.route_for(source_id="main-source", phase_id="main")

    with pytest.raises(ValueError, match="identity mismatch"):
        processor.process(batch=batch, route=route)
    pool.close()


@pytest.mark.parametrize(
    "batch",
    (
        _batch(source_id="wrong-source"),
        _batch(phase_id="wrong-phase"),
    ),
)
def test_processor_rejects_batch_source_or_phase_mismatch(
    batch: RewardBatchView,
) -> None:
    processor, pool, _events = _processor()
    route = processor.plan.route_for(source_id="main-source", phase_id="main")
    with pytest.raises(ValueError, match="source/phase"):
        processor.process(batch=batch, route=route)
    pool.close()


def test_processor_rejects_active_reward_or_forged_route_drift() -> None:
    processor, pool, _events = _processor()
    route = processor.plan.route_for(source_id="main-source", phase_id="main")
    with pytest.raises(ValueError, match="active rewards"):
        processor.process(
            batch=_batch(active_reward_ids=("quality", "coherence")),
            route=route,
        )
    forged = replace(
        route,
        rewards=(
            replace(route.rewards[0], weight=99.0),
            route.rewards[1],
        ),
    )
    with pytest.raises(ValueError, match="canonical route"):
        processor.process(batch=_batch(), route=forged)
    pool.close()


def test_pointwise_and_groupwise_output_shapes_masks_and_finite_values_are_strict() -> (
    None
):
    identity = _batch().identity
    with pytest.raises(ValueError, match=r"shape \[4\]"):
        PointwiseRewardOutput(
            identity=identity,
            values=np.ones((4, 1)),
            valid_mask=np.ones(4, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="finite"):
        PointwiseRewardOutput(
            identity=identity,
            values=np.array([1.0, 2.0, np.nan, 4.0]),
            valid_mask=np.ones(4, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="bool"):
        PointwiseRewardOutput(
            identity=identity,
            values=np.ones(4),
            valid_mask=np.ones(4),
        )
    with pytest.raises(ValueError, match=r"shape \[2\]"):
        GroupwiseRewardOutput(
            identity=identity,
            group_ids=("group-a", "group-b"),
            values=np.ones(4),
            valid_mask=np.ones(2, dtype=np.bool_),
        )


def test_processor_rejects_group_identity_mismatch_before_alignment() -> None:
    processor, pool, _events = _processor(
        groupwise=_Groupwise(
            np.array([10.0, 20.0]),
            np.ones(2, dtype=np.bool_),
            output_group_ids=("group-b", "group-a"),
        )
    )
    route = processor.plan.route_for(source_id="main-source", phase_id="main")
    with pytest.raises(ValueError, match="group identity/order mismatch"):
        processor.process(batch=_batch(), route=route)
    pool.close()


def test_processor_constructor_enforces_resolved_protocol_and_logical_order() -> None:
    plan = _main_plan()
    events: list[str] = []
    pool = RewardPool(plan, lambda identity: _Resource(identity, events))
    point = _Pointwise(np.ones(4), np.ones(4, dtype=np.bool_))
    group = _Groupwise(np.ones(2), np.ones(2, dtype=np.bool_))

    with pytest.raises(ValueError, match="ids and order exactly"):
        RewardProcessor(
            plan=plan,
            pool=pool.view(),
            logical_rewards={"quality": point, "coherence": group},
        )
    with pytest.raises(TypeError, match="must be GroupwiseReward"):
        RewardProcessor(
            plan=plan,
            pool=pool.view(),
            logical_rewards={"coherence": point, "quality": point},
        )
    pool.close()
