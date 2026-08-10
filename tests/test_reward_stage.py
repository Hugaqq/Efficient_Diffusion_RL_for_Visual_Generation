"""Resolved recipe reward planning and BaseTrainer reward-stage contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from visual_rl.algorithms.dynamics.session import (
    DynamicsSelectionState,
    ScheduleSnapshot,
)
from visual_rl.algorithms.rollout.interface import RolloutExecution
from visual_rl.algorithms.trainer.grpo import BaseTrainer
from visual_rl.algorithms.trainer.interface import (
    IterationIdentity,
    PrepareRunContext,
    StageValue,
)
from visual_rl.composition.config.compiler import compile_recipe_v2
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.recipes.schema import (
    MaterializedRecipe,
    ResolvedRecipe,
)
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
from visual_rl.core.types import FrozenMapping, StepContext
from visual_rl.data import SourceContentBinding
from visual_rl.algorithms.rewards import (
    PointwiseReward,
    PointwiseRewardOutput,
    RewardRuntimeContext,
    RewardStage,
    RewardStageExecutionError,
    RewardStageInput,
    RewardStageOutput,
)
from visual_rl.runtime.reward_resources import RewardPool
from visual_rl.data.samples import (
    BatchRowContext,
    CameraConditionBatchState,
    NoConditionBatchState,
    SourceItemContext,
    StackedSampleBatch,
    TrajectoryBatch,
    TrajectoryContext,
    camera_condition_identity,
)


def _resolved_recipe(tmp_path) -> ResolvedRecipe:
    source_path = tmp_path / "world-r1.yaml"
    source_path.write_text(
        "schema_version: 2\nrecipe: world_r1_release_surrogate_v1\n",
        encoding="utf-8",
    )
    return compile_recipe_v2(load_source_recipe(source_path))


def _filesystem_identity(
    digest: str,
    *,
    policy: str = "all-files.v1",
    node_type: str = "tree",
) -> FrozenMapping:
    return FrozenMapping(
        {
            "identity_schema": "filesystem-artifact.v1",
            "content_policy": policy,
            "node_type": node_type,
            "content_sha256": digest * 64,
            "file_count": 1,
            "byte_count": 17,
        }
    )


def _materialized(resolved: ResolvedRecipe) -> MaterializedRecipe:
    source_refs = tuple(
        sorted({source.artifact_ref for source in resolved.source_plan.sources})
    )
    reward_refs = tuple(
        sorted({resource.artifact_ref for resource in resolved.reward_plan.resources})
    )
    return MaterializedRecipe(
        resolved=resolved,
        model_artifact_identity=_filesystem_identity("1"),
        source_content_binding=SourceContentBinding(
            source_plan_id=resolved.source_plan.plan_id,
            artifact_content_identities=tuple(
                (artifact_ref, _filesystem_identity(str(index + 2), node_type="file"))
                for index, artifact_ref in enumerate(source_refs)
            ),
        ),
        reward_plan=resolved.reward_plan.bind_artifacts(
            {
                artifact_ref: _filesystem_identity(str(index + 5))
                for index, artifact_ref in enumerate(reward_refs)
            }
        ),
        code_artifact_identity=_filesystem_identity(
            "9",
            policy="python-code.v1",
        ),
    )


def test_recipe_routes_compile_to_stable_logical_and_physical_plan(tmp_path) -> None:
    resolved = _resolved_recipe(tmp_path)
    from_resolved = resolved.reward_plan
    from_materialized = _materialized(resolved).reward_plan

    assert from_resolved.plan_id != from_materialized.plan_id
    assert from_resolved.provisional is True
    assert from_materialized.provisional is False
    with pytest.raises(ValueError, match="provisional reward descriptors"):
        RewardPool(from_resolved, lambda identity: _Resource(identity))
    assert from_resolved.plan_id.startswith("reward-plan-spec.v1:")
    assert from_resolved.logical_reward_ids == ("reward_3d", "reward_general")
    main = from_resolved.route_for(source_id="main", phase_id="main")
    dynamic = from_resolved.route_for(source_id="dynamic", phase_id="dynamic")
    assert set(main.logical_reward_ids) == {"reward_general", "reward_3d"}
    assert dynamic.logical_reward_ids == ("reward_general",)
    reward_3d = from_resolved.logical_reward("reward_3d")
    assert reward_3d.contract.required_payload_type == "camera_trajectory_v1"
    assert reward_3d.contract.granularity is RewardGranularity.POINTWISE
    assert reward_3d.component_declaration_id == (
        resolved.component("rewards.reward_3d").declaration.declaration_id
    )
    assert "reward_3d" not in reward_3d.resource_identity
    assert len(from_resolved.resource_identities) == 2
    assert from_resolved.resource(reward_3d.resource_identity).provisional
    materialized_3d = from_materialized.logical_reward("reward_3d")
    assert from_materialized.resource(materialized_3d.resource_identity).materialized
    assert materialized_3d.resource_identity.startswith("reward-resource-spec.v1:")


def test_two_logical_rewards_share_one_materialized_resource_spec(tmp_path) -> None:
    del tmp_path
    resource = RewardResourceSpec(
        descriptor=FrozenMapping({"artifact_ref": "reward_quality"}),
        artifact_identity=_filesystem_identity("7"),
    )
    contract = RewardContract(
        accepted_media=(MediaKind.IMAGE,),
        required_payload_type=None,
        granularity=RewardGranularity.POINTWISE,
        output_rank=1,
        frame_aggregation=None,
    )
    logical_ids = ("reward_quality", "reward_quality_copy")
    plan = RewardPlanSpec(
        resources=(resource,),
        logical_rewards=tuple(
            LogicalRewardSpec(
                logical_reward_id=logical_id,
                component_declaration_id=(
                    "component-declaration.v1:"
                    f"{hashlib.sha256(logical_id.encode()).hexdigest()}"
                ),
                resource_identity=resource.resource_identity,
                contract=contract,
            )
            for logical_id in logical_ids
        ),
        routes=(
            RewardRouteSpec(
                source_id="main",
                phase_id="main",
                rewards=(
                    RewardRouteBinding("reward_quality", 1.0),
                    RewardRouteBinding("reward_quality_copy", 0.5),
                ),
            ),
        ),
    )

    first = plan.logical_reward("reward_quality")
    second = plan.logical_reward("reward_quality_copy")
    route = plan.route_for(source_id="main", phase_id="main")
    assert first.logical_reward_id != second.logical_reward_id
    assert route.binding(first.logical_reward_id).weight != route.binding(
        second.logical_reward_id
    ).weight
    assert first.resource_identity == second.resource_identity
    assert plan.resource_identities == (first.resource_identity,)


def test_actual_runtime_device_is_not_part_of_resource_spec_identity(tmp_path) -> None:
    materialized = _materialized(_resolved_recipe(tmp_path))
    expected = materialized.reward_plan.logical_reward(
        "reward_general"
    ).resource_identity

    # Runtime facts are deliberately not accepted by the materialized compiler.
    for _actual_device in ("cpu", "cuda:7", "mps"):
        actual = materialized.reward_plan.logical_reward(
            "reward_general"
        ).resource_identity
        assert actual == expected


@dataclass
class _Resource:
    identity: str
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class _ControlledReward(PointwiseReward):
    def __init__(
        self,
        value: float,
        *,
        valid: bool = True,
        fail: bool = False,
        require_camera: bool = False,
    ) -> None:
        self.value = value
        self.valid = valid
        self.fail = fail
        self.require_camera = require_camera
        self.calls = 0

    def score(self, *, logical_reward_id, resource, batch):
        del logical_reward_id
        assert isinstance(resource, _Resource)
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected reward execution failure")
        if self.require_camera:
            trajectory = batch.payload["trajectory"]
            assert (
                batch.payload["camera_trajectory_v1"]
                is trajectory.condition_state.camera_trajectory
            )
            assert batch.identity.condition_payload_ids == (
                trajectory.condition_state.row_condition_identities
            )
        return PointwiseRewardOutput(
            identity=batch.identity,
            values=np.full(batch.batch_size, self.value, dtype=np.float64),
            valid_mask=np.full(batch.batch_size, self.valid, dtype=np.bool_),
        )


def _trajectory(
    *,
    source_id: str,
    phase_id: str,
    camera: bool,
) -> tuple[IterationIdentity, TrajectoryBatch, StackedSampleBatch]:
    batch_size = 2
    transitions = 1
    contexts = tuple(
        TrajectoryContext(
            sample_id=f"sample-{source_id}-{row}",
            trajectory_id=f"trajectory-{source_id}-{row}",
            batch_row=BatchRowContext(
                occurrence_id=f"occurrence-{row}",
                group_id=f"group-{row}",
                member_id=0,
                phase=phase_id,
                optimizer_step=7,
                source_item_id=f"source-item-{row}",
            ),
        )
        for row in range(batch_size)
    )
    if camera:
        identities = ("camera-config-a", "camera-config-b")
        camera_tensor = torch.eye(4, dtype=torch.float64).repeat(
            batch_size,
            3,
            1,
            1,
        )
        row_condition_identities = tuple(
            camera_condition_identity(camera_tensor[row], identities[row])
            for row in range(batch_size)
        )
        condition_state = CameraConditionBatchState(
            camera_trajectory=camera_tensor,
            conditioner_config_identity=identities,
            row_condition_identities=row_condition_identities,
        )
        media = torch.zeros(batch_size, 3, 3, 2, 2)
        condition_identity = tuple((item,) for item in row_condition_identities)
    else:
        condition_state = NoConditionBatchState(batch_size)
        media = torch.zeros(batch_size, 3, 3, 2, 2)
        condition_identity = (("none",),) * batch_size
    trajectory = TrajectoryBatch(
        kind="full_trajectory",
        contexts=contexts,
        x_t=torch.zeros(batch_size, transitions, 1),
        sampled_action=torch.ones(batch_size, transitions, 1),
        conditioned_next=torch.ones(batch_size, transitions, 1),
        timesteps=torch.zeros(batch_size, transitions),
        next_timesteps=torch.ones(batch_size, transitions),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(batch_size, transitions, dtype=torch.bool),
        transition_index=torch.zeros(
            batch_size,
            transitions,
            dtype=torch.int64,
        ),
        likelihood_semantics="exact_env_action",
        condition_identity=condition_identity,
        guidance_identity=(("cfg",),) * batch_size,
        storage_dtype_identity=(("torch.float32",),) * batch_size,
        quantization_identity=(("none",),) * batch_size,
        media=media,
        media_layout="BFCHW",
        condition_state=condition_state,
    )
    identity = IterationIdentity(
        optimizer_step=7,
        source_id=source_id,
        phase_id=phase_id,
        row_identities=tuple(item.batch_row_identity for item in contexts),
        group_ids=tuple(item.batch_row.group_id for item in contexts),
        member_ids=tuple(item.batch_row.member_id for item in contexts),
    )
    samples = StackedSampleBatch(
        task_type="t2v",
        prompts=tuple(f"prompt-{source_id}-{row}" for row in range(batch_size)),
        sources=tuple(
            SourceItemContext(
                source_item_id=f"source-item-{row}",
                dataset_source_id=source_id,
                dataset_index=row,
                dataset_revision="dataset-revision-v1",
            )
            for row in range(batch_size)
        ),
        rows=tuple(item.batch_row for item in contexts),
        metadata=tuple({"row": row} for row in range(batch_size)),
        condition_state=condition_state,
    )
    return identity, trajectory, samples


def _reward_input(
    trajectory: TrajectoryBatch,
    samples: StackedSampleBatch,
) -> RewardStageInput:
    selection_state = DynamicsSelectionState.from_generator(
        torch.Generator().manual_seed(17)
    )
    snapshot = ScheduleSnapshot(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        sigmas=None,
        dt=torch.tensor([1.0]),
        dynamics_config_identity="reward-stage-test-dynamics",
        scheduler_identity="reward-stage-test-scheduler",
        selection_policy="full_trajectory",
        selected_policy_step_indices=(0,),
        randomness_identity="reward-stage-test-selection",
        next_selection_state=selection_state,
    )
    return RewardStageInput(
        execution=RolloutExecution(
            trajectory=trajectory,
            schedule_snapshot=snapshot,
            encoded_conditioning=object(),
            model_condition_identity=tuple(row.identity for row in samples.rows),
        ),
        samples=samples,
        runtime_context=RewardRuntimeContext(
            StepContext(step=7, seed=71, rank=0, world_size=1)
        ),
    )


def _stage(plan, *, general=None, reward_3d=None):
    built: list[str] = []
    resources: dict[str, _Resource] = {}

    def factory(identity: str) -> _Resource:
        built.append(identity)
        resource = _Resource(identity)
        resources[identity] = resource
        return resource

    pool = RewardPool(plan, factory)
    logical = {
        "reward_3d": reward_3d or _ControlledReward(5.0, require_camera=True),
        "reward_general": general or _ControlledReward(2.0),
    }
    stage = RewardStage(
        plan=plan,
        pool=pool.view(),
        logical_rewards=logical,
    )
    pool.activate()
    return (
        stage,
        pool,
        built,
        resources,
        logical,
    )


def test_stage_preserves_identity_and_expands_main_and_dynamic_masks(tmp_path) -> None:
    plan = _materialized(_resolved_recipe(tmp_path)).reward_plan
    stage, pool, built, resources, logical = _stage(plan)
    main_identity, main_trajectory, main_samples = _trajectory(
        source_id="main",
        phase_id="main",
        camera=True,
    )
    main = stage(
        StageValue(main_identity, _reward_input(main_trajectory, main_samples))
    )

    assert main.identity is main_identity
    assert isinstance(main.payload, RewardStageOutput)
    assert main.payload.trajectory is main_trajectory
    main_result = main.payload.reward_result
    assert tuple(main_result.component_scores) == (
        "reward_3d",
        "reward_general",
    )
    assert all(mask.all() for mask in main_result.component_applicable_masks.values())
    np.testing.assert_array_equal(main_result.weighted_total, np.array([7.0, 7.0]))
    assert logical["reward_general"].calls == 1
    assert logical["reward_3d"].calls == 1

    dynamic_identity, dynamic_trajectory, dynamic_samples = _trajectory(
        source_id="dynamic",
        phase_id="dynamic",
        camera=False,
    )
    dynamic = stage(
        StageValue(
            dynamic_identity,
            _reward_input(dynamic_trajectory, dynamic_samples),
        )
    )
    assert dynamic.identity is dynamic_identity
    dynamic_result = dynamic.payload.reward_result
    np.testing.assert_array_equal(
        dynamic_result.component_scores["reward_3d"],
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        dynamic_result.component_applicable_masks["reward_3d"],
        np.zeros(2, dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        dynamic_result.component_valid_masks["reward_3d"],
        np.zeros(2, dtype=np.bool_),
    )
    np.testing.assert_array_equal(dynamic_result.valid_mask, np.ones(2, dtype=np.bool_))
    assert dynamic_result.logical_weights["reward_3d"] == 0.0
    assert logical["reward_general"].calls == 2
    assert logical["reward_3d"].calls == 1

    assert built == list(plan.resource_identities)
    assert len(resources) == 2
    assert (
        dynamic_result.resource_identities["reward_general"]
        != (dynamic_result.resource_identities["reward_3d"])
    )
    pool.close()
    assert all(item.close_calls == 1 for item in resources.values())


def test_reward_stage_binds_directly_to_base_trainer_after_trajectory_rollout(
    tmp_path,
) -> None:
    plan = _materialized(_resolved_recipe(tmp_path)).reward_plan
    stage, pool, _built, _resources, _logical = _stage(plan)
    identity, trajectory, samples = _trajectory(
        source_id="dynamic",
        phase_id="dynamic",
        camera=False,
    )

    class _Prelude:
        def build(self, optimizer_step):
            assert optimizer_step == identity.optimizer_step
            return StageValue(identity, object())

    class _Rollout:
        def __call__(self, value):
            assert value.identity is identity
            return StageValue(
                value.identity,
                _reward_input(trajectory, samples),
            )

    class _PassThrough:
        def __call__(self, value):
            return StageValue(value.identity, value.payload)

    passthrough = _PassThrough()
    trainer = BaseTrainer(
        prelude=_Prelude(),
        rollout=_Rollout(),
        reward=stage,
        advantage=passthrough,
        credit=passthrough,
        optimize=passthrough,
    )
    trainer.prepare_run(
        PrepareRunContext(
            run_id="reward-stage-run",
            recipe_id="reward-stage-recipe",
            start_optimizer_step=identity.optimizer_step,
        )
    )

    result = trainer.run_iteration(identity.optimizer_step)

    assert result.value.identity is identity
    assert isinstance(result.value.payload, RewardStageOutput)
    assert result.value.payload.trajectory is trajectory
    assert result.stage_order == (
        "prelude",
        "rollout",
        "reward",
        "advantage",
        "credit",
        "optimize",
    )
    pool.close()


@pytest.mark.parametrize(
    ("reward", "error"),
    (
        (_ControlledReward(1.0, valid=False), RewardStageExecutionError),
        (_ControlledReward(float("nan")), ValueError),
        (_ControlledReward(1.0, fail=True), RuntimeError),
    ),
)
def test_applicable_reward_failure_or_nan_hard_fails(
    tmp_path,
    reward,
    error,
) -> None:
    plan = _materialized(_resolved_recipe(tmp_path)).reward_plan
    stage, pool, _built, _resources, _logical = _stage(plan, general=reward)
    identity, trajectory, samples = _trajectory(
        source_id="dynamic",
        phase_id="dynamic",
        camera=False,
    )
    with pytest.raises(error):
        stage(StageValue(identity, _reward_input(trajectory, samples)))
    pool.close()


def test_camera_route_rejects_missing_condition_payload(tmp_path) -> None:
    plan = _materialized(_resolved_recipe(tmp_path)).reward_plan
    stage, pool, _built, _resources, logical = _stage(plan)
    identity, no_camera, samples = _trajectory(
        source_id="main",
        phase_id="main",
        camera=False,
    )
    with pytest.raises(ValueError, match="camera_trajectory_v1"):
        stage(StageValue(identity, _reward_input(no_camera, samples)))
    assert logical["reward_general"].calls == 0
    assert logical["reward_3d"].calls == 0
    pool.close()


def test_groupwise_plan_is_an_explicit_stage_placeholder() -> None:
    resource = RewardResourceSpec(
        descriptor=FrozenMapping({"artifact_ref": "group-resource"}),
        artifact_identity=_filesystem_identity("8"),
    )
    plan = RewardPlanSpec(
        resources=(resource,),
        logical_rewards=(
            LogicalRewardSpec(
                logical_reward_id="group",
                component_declaration_id=(
                    "component-declaration.v1:"
                    f"{hashlib.sha256(b'group').hexdigest()}"
                ),
                resource_identity=resource.resource_identity,
                contract=RewardContract(
                    accepted_media=(MediaKind.IMAGE,),
                    required_payload_type=None,
                    granularity=RewardGranularity.GROUPWISE,
                    output_rank=1,
                    frame_aggregation=None,
                ),
            ),
        ),
        routes=(
            RewardRouteSpec(
                source_id="main",
                phase_id="main",
                rewards=(RewardRouteBinding("group", 1.0),),
            ),
        )
    )
    pool = RewardPool(plan, lambda identity: _Resource(identity))
    with pytest.raises(NotImplementedError, match="only pointwise"):
        RewardStage(plan=plan, pool=pool.view(), logical_rewards={})
    pool.close()
