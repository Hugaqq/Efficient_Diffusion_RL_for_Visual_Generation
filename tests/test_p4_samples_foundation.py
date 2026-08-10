"""P4 contracts for unbatched items, explicit collation, and trajectories."""

from __future__ import annotations

import json

import pytest
import torch

from visual_rl.core.contracts import (
    LikelihoodSemantics as ContractLikelihoodSemantics,
)
from visual_rl.data.samples import (
    BatchRowContext,
    CameraConditionBatchState,
    CameraConditionPayload,
    ExplicitCollator,
    FullTrajectoryItem,
    I2VItem,
    LikelihoodSemantics,
    NoConditionBatchState,
    SingleStepTrajectoryItem,
    SourceItemContext,
    T2IItem,
    T2VItem,
    TrajectoryContext,
    TrajectoryStep,
)


def _source(index: int = 0) -> SourceItemContext:
    return SourceItemContext(
        source_item_id=f"source-item-{index}",
        dataset_source_id="dataset-main",
        dataset_index=index,
        dataset_revision="sha256:dataset-v1",
    )


def _row(
    member: int,
    *,
    source_item_id: str = "source-item-0",
) -> BatchRowContext:
    return BatchRowContext(
        occurrence_id="occurrence-0",
        group_id="group-0",
        member_id=member,
        phase="main",
        optimizer_step=7,
        source_item_id=source_item_id,
    )


def _camera(offset: float = 0.0) -> CameraConditionPayload:
    trajectory = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
    trajectory[:, 0, 3] += offset
    return CameraConditionPayload(
        camera_trajectory=trajectory,
        conditioner_config_identity="camera-config-v1",
    )


def _trajectory_context(member: int) -> TrajectoryContext:
    return TrajectoryContext(
        sample_id=f"sample-{member}",
        trajectory_id=f"trajectory-{member}",
        batch_row=_row(member),
    )


def _step(
    index: int,
    x_t: torch.Tensor,
    conditioned_next: torch.Tensor,
    *,
    semantics: LikelihoodSemantics = LikelihoodSemantics.EXACT_ENV_ACTION,
    condition_identity: str = "camera-config-v1",
) -> TrajectoryStep:
    sampled = x_t + 0.25
    return TrajectoryStep(
        x_t=x_t,
        sampled_action=sampled,
        conditioned_next=conditioned_next,
        t=torch.tensor(9.5 - index, dtype=torch.float32),
        t_next=torch.tensor(8.5 - index, dtype=torch.float32),
        old_log_prob=torch.tensor(-0.75 - index, dtype=torch.float32),
        likelihood_semantics=semantics,
        condition_identity=condition_identity,
        guidance_identity="cfg:4.5",
        transition_index=index,
        storage_dtype_identity=str(x_t.dtype),
    )


def _full_item(member: int) -> FullTrajectoryItem:
    x0 = torch.full((1, 2, 2), float(member), dtype=torch.float32)
    x1 = x0 + 0.5
    x2 = x1 + 0.5
    condition = _camera(float(member))
    return FullTrajectoryItem(
        context=_trajectory_context(member),
        steps=(
            _step(0, x0, x1, condition_identity=condition.condition_identity),
            _step(1, x1, x2, condition_identity=condition.condition_identity),
        ),
        media=torch.full((3, 3, 2, 2), float(member)),
        media_layout="FCHW",
        condition=condition,
    )


def test_item_contract_has_no_batch_dimension_and_serializes_json() -> None:
    item = T2VItem(
        prompt="camera pushes in",
        source=_source(),
        condition=_camera(),
        metadata={"license": "test", "tags": ["camera", "forest"]},
    )

    item.validate()
    payload = item.serialize()

    assert payload["task_type"] == "t2v"
    assert payload["condition"]["camera_trajectory"]["shape"] == [3, 4, 4]
    assert len(payload["condition"]["condition_identity"]) == 64
    assert "group_id" not in payload["source"]
    assert "phase" not in payload["condition"]
    json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError, match="without a batch dimension"):
        CameraConditionPayload(
            camera_trajectory=torch.eye(4).repeat(2, 3, 1, 1),
            conditioner_config_identity="camera-config-v1",
        )


def test_likelihood_semantics_has_one_shared_type_identity() -> None:
    assert LikelihoodSemantics is ContractLikelihoodSemantics


def test_camera_identity_changes_with_trajectory_under_the_same_config() -> None:
    first = _camera(0.0)
    same = _camera(0.0)
    moved = _camera(0.25)

    assert first.condition_identity == same.condition_identity
    assert first.condition_identity != moved.condition_identity


def test_i2v_item_rejects_batched_image_and_t2i_rejects_camera() -> None:
    with pytest.raises(ValueError, match="without a batch dimension"):
        I2VItem(
            prompt="animate",
            source=_source(),
            input_image=torch.zeros(1, 3, 8, 8),
        )
    with pytest.raises(ValueError, match="does not accept"):
        T2IItem(
            prompt="draw",
            source=_source(),
            condition=_camera(),
        )


def test_explicit_sample_collator_creates_and_transforms_one_batch() -> None:
    condition = _camera()
    items = (
        T2VItem(prompt="push in", source=_source(), condition=condition),
        T2VItem(prompt="push in", source=_source(), condition=condition),
    )
    batch = ExplicitCollator().collate_samples(items, (_row(0), _row(1)))

    assert batch.batch_size == 2
    assert isinstance(batch.condition_state, CameraConditionBatchState)
    assert tuple(batch.condition_state.camera_trajectory.shape) == (2, 3, 4, 4)
    assert batch.rows[0].group_id == batch.rows[1].group_id
    assert batch.rows[0].member_id != batch.rows[1].member_id

    sliced = batch.slice([1])
    assert sliced.batch_size == 1
    assert sliced.rows[0].member_id == 1
    moved = sliced.to("cpu")
    assert moved.condition_state.camera_trajectory.dtype == torch.float64
    assert moved.detach().condition_state.camera_trajectory.grad_fn is None


def test_collator_rejects_source_drift_duplicate_members_and_mixed_conditions() -> None:
    no_camera = T2VItem(prompt="still", source=_source())
    camera = T2VItem(prompt="move", source=_source(), condition=_camera())
    collator = ExplicitCollator()

    with pytest.raises(ValueError, match="mix condition"):
        collator.collate_samples((no_camera, camera), (_row(0), _row(1)))
    with pytest.raises(ValueError, match="source_item_id"):
        collator.collate_samples(
            (no_camera,),
            (_row(0, source_item_id="wrong-source"),),
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        collator.collate_samples((no_camera, no_camera), (_row(0), _row(0)))


def test_no_condition_and_i2v_batch_paths_are_explicit() -> None:
    i2v = I2VItem(
        prompt="animate a still",
        source=_source(),
        input_image=torch.ones(3, 8, 8),
    )
    batch = ExplicitCollator().collate_samples((i2v,), (_row(0),))

    assert isinstance(batch.condition_state, NoConditionBatchState)
    assert tuple(batch.input_images.shape) == (1, 3, 8, 8)
    converted = batch.to("cpu", dtype=torch.float64)
    assert converted.input_images.dtype == torch.float64


def test_trajectory_step_keeps_both_states_and_declares_scoring_target() -> None:
    x_t = torch.zeros(1, 2, 2)
    conditioned = torch.ones_like(x_t)
    exact = _step(0, x_t, conditioned)
    surrogate = _step(
        0,
        x_t,
        conditioned,
        semantics=LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
    )

    assert exact.scoring_target is exact.sampled_action
    assert surrogate.scoring_target is surrogate.conditioned_next
    payload = surrogate.serialize()
    assert payload["likelihood_semantics"] == "post_hook_base_density_surrogate"
    assert set(payload) >= {
        "x_t",
        "sampled_action",
        "conditioned_next",
        "t",
        "t_next",
        "old_log_prob",
    }


def test_trajectory_requires_post_hook_state_continuity() -> None:
    item = _full_item(0)
    item.validate()
    bad_second = _step(
        1,
        torch.full((1, 2, 2), 99.0),
        torch.ones(1, 2, 2),
        condition_identity=item.condition.condition_identity,
    )

    with pytest.raises(ValueError, match="conditioned_next"):
        FullTrajectoryItem(
            context=item.context,
            steps=(item.steps[0], bad_second),
            media=item.media,
            media_layout=item.media_layout,
            condition=item.condition,
        )


def test_trajectory_collation_slice_to_detach_preserves_semantics() -> None:
    batch = ExplicitCollator().collate_trajectories((_full_item(0), _full_item(1)))

    assert batch.kind == "full_trajectory"
    assert batch.batch_size == 2
    assert batch.transition_count == 2
    assert tuple(batch.x_t.shape) == (2, 2, 1, 2, 2)
    assert tuple(batch.sampled_action.shape) == tuple(batch.conditioned_next.shape)
    assert batch.likelihood_semantics is LikelihoodSemantics.EXACT_ENV_ACTION
    assert batch.scoring_target is batch.sampled_action
    assert isinstance(batch.condition_state, CameraConditionBatchState)

    sliced = batch.slice(torch.tensor([1], dtype=torch.int64))
    assert sliced.contexts[0].sample_id == "sample-1"
    converted = sliced.to("cpu", dtype=torch.float64)
    assert converted.x_t.dtype == torch.float64
    assert converted.timesteps.dtype == torch.float32
    assert converted.storage_dtype_identity == (("torch.float64", "torch.float64"),)
    assert converted.detach().x_t.grad_fn is None


def test_single_step_trajectory_index_is_explicit() -> None:
    x_t = torch.zeros(1, 2, 2)
    step = _step(3, x_t, x_t + 1.0, condition_identity="none")
    item = SingleStepTrajectoryItem(
        context=_trajectory_context(0),
        steps=(step,),
        media=torch.zeros(3, 2, 2),
        media_layout="CHW",
        selected_timestep_index=3,
        selection_policy_identity="test.single-step-selection.v1",
        selection_mapping_identity="test.single-step-mapping.v1",
    )
    batch = ExplicitCollator().collate_trajectories((item,))

    assert batch.kind == "single_step"
    assert torch.equal(batch.selected_timestep_index, torch.tensor([3]))
    assert batch.branch_step_index is None
    assert batch.selection_policy_identity == "test.single-step-selection.v1"
    assert batch.selection_mapping_identity == "test.single-step-mapping.v1"
