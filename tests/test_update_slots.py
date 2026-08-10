"""Focused contracts for the production update-slot index layer."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from visual_rl.algorithms.optimization.slots import (
    UpdateSlotPlan,
    UpdateSlotPlanError,
)

ROOT = Path(__file__).resolve().parents[1]


def test_update_slot_owner_is_canonical_and_training_namespace_is_absent() -> None:
    assert not (ROOT / "visual_rl" / "training").exists()
    assert (
        UpdateSlotPlan.__module__ == "visual_rl.algorithms.optimization.slots"
    )


def test_slot_plan_construction_is_global_torch_rng_neutral() -> None:
    state_before = torch.random.get_rng_state().clone()

    UpdateSlotPlan.from_active_mask(
        torch.tensor([[True, False], [True, True]], dtype=torch.bool),
        row_microbatch_size=1,
        transition_window_size=1,
    )

    assert torch.equal(torch.random.get_rng_state(), state_before)


def _covered_active_cells(
    plan: UpdateSlotPlan,
    active_mask: torch.Tensor,
) -> Counter[tuple[int, int]]:
    covered: Counter[tuple[int, int]] = Counter()
    for slot in plan.slots:
        for row in slot.row_indices:
            for transition in range(
                slot.transition_start,
                slot.transition_stop,
            ):
                if bool(active_mask[row, transition]):
                    covered[(row, transition)] += 1
    return covered


def _expected_active_cells(
    active_mask: torch.Tensor,
) -> Counter[tuple[int, int]]:
    return Counter(
        (int(row), int(transition))
        for row, transition in torch.nonzero(active_mask, as_tuple=False).tolist()
    )


def test_b8_t28_full_row_slots_partition_all_224_active_cells() -> None:
    active_mask = torch.ones((8, 28), dtype=torch.bool)

    plan = UpdateSlotPlan.from_active_mask(
        active_mask,
        transition_window_size=1,
    )

    assert plan.batch_size == 8
    assert plan.transition_count == 28
    assert plan.resolved_row_microbatch_size == 8
    assert plan.global_active_count == 224
    assert len(plan.slots) == 28
    assert tuple(slot.slot_index for slot in plan.slots) == tuple(range(28))
    for transition, slot in enumerate(plan.slots):
        assert slot.row_indices == tuple(range(8))
        assert (slot.transition_start, slot.transition_stop) == (
            transition,
            transition + 1,
        )
        assert slot.active_count == 8
        assert slot.global_active_count == 224
        assert slot.active_fraction == pytest.approx(1.0 / 28.0)
    assert sum(slot.active_fraction for slot in plan.slots) == pytest.approx(1.0)
    assert _covered_active_cells(plan, active_mask) == _expected_active_cells(
        active_mask
    )


def test_b8_t28_row_microbatch_one_uses_native_row_then_transition_order() -> None:
    active_mask = torch.ones((8, 28), dtype=torch.bool)

    plan = UpdateSlotPlan.from_active_mask(
        active_mask,
        row_microbatch_size=1,
        transition_window_size=1,
    )

    assert plan.global_active_count == 224
    assert len(plan.slots) == 224
    assert plan.slots[0].row_indices == (0,)
    assert (plan.slots[0].transition_start, plan.slots[0].transition_stop) == (
        0,
        1,
    )
    assert plan.slots[27].row_indices == (0,)
    assert plan.slots[27].transition_start == 27
    assert plan.slots[28].row_indices == (1,)
    assert plan.slots[28].transition_start == 0
    assert plan.slots[-1].row_indices == (7,)
    assert plan.slots[-1].transition_start == 27
    assert all(slot.active_count == 1 for slot in plan.slots)
    assert all(slot.global_active_count == 224 for slot in plan.slots)
    assert _covered_active_cells(plan, active_mask) == _expected_active_cells(
        active_mask
    )


def test_irregular_mask_skips_empty_rectangles_without_duplicates_or_omissions() -> (
    None
):
    active_mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, False, False, True, False],
            [True, False, False, False, False],
        ]
    )

    plan = UpdateSlotPlan.from_active_mask(
        active_mask,
        row_microbatch_size=2,
        transition_window_size=2,
    )

    assert plan.global_active_count == 4
    assert [
        (
            slot.row_indices,
            slot.transition_start,
            slot.transition_stop,
            slot.active_count,
        )
        for slot in plan.slots
    ] == [
        ((0, 1), 0, 2, 1),
        ((0, 1), 2, 4, 2),
        ((2,), 0, 2, 1),
    ]
    assert tuple(slot.slot_index for slot in plan.slots) == (0, 1, 2)
    assert all(slot.active_count > 0 for slot in plan.slots)
    assert sum(slot.active_count for slot in plan.slots) == 4
    assert sum(slot.active_fraction for slot in plan.slots) == pytest.approx(1.0)
    assert _covered_active_cells(plan, active_mask) == _expected_active_cells(
        active_mask
    )


def test_plan_identity_is_stable_and_sensitive_to_mask_and_configuration() -> None:
    active_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
        ]
    )
    same_noncontiguous = active_mask.t().contiguous().t()
    assert not same_noncontiguous.is_contiguous()
    configuration = {
        "row_microbatch_size": 2,
        "transition_window_size": 2,
        "row_order": (2, 0, 3, 1),
    }

    first = UpdateSlotPlan.from_active_mask(active_mask, **configuration)
    second = UpdateSlotPlan.from_active_mask(
        same_noncontiguous,
        **configuration,
    )

    assert first == second
    assert first.to_payload() == second.to_payload()
    assert first.configuration_id == second.configuration_id
    assert first.active_mask_id == second.active_mask_id
    assert first.plan_id == second.plan_id
    assert len(first.configuration_id) == 64
    assert len(first.active_mask_id) == 64
    assert len(first.plan_id) == 64
    assert first.slots[0].row_indices == (2, 0)

    changed_mask = active_mask.clone()
    changed_mask[0, 0] = False
    mask_variant = UpdateSlotPlan.from_active_mask(
        changed_mask,
        **configuration,
    )
    assert mask_variant.configuration_id == first.configuration_id
    assert mask_variant.active_mask_id != first.active_mask_id
    assert mask_variant.plan_id != first.plan_id

    order_variant = UpdateSlotPlan.from_active_mask(
        active_mask,
        row_microbatch_size=2,
        transition_window_size=2,
        row_order=(0, 2, 3, 1),
    )
    assert order_variant.configuration_id != first.configuration_id
    assert order_variant.active_mask_id == first.active_mask_id
    assert order_variant.plan_id != first.plan_id


def test_plan_is_immutable_and_rejects_invalid_or_changed_active_masks() -> None:
    active_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    plan = UpdateSlotPlan.from_active_mask(active_mask)
    plan.validate_against(active_mask.clone())

    with pytest.raises(FrozenInstanceError):
        plan.transition_window_size = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.slots[0].active_count = 99  # type: ignore[misc]
    with pytest.raises(UpdateSlotPlanError, match="cells differ"):
        changed = active_mask.clone()
        changed[0, 1] = True
        plan.validate_against(changed)
    with pytest.raises(UpdateSlotPlanError, match="shape differs"):
        plan.validate_against(torch.ones((2, 4), dtype=torch.bool))
    with pytest.raises(UpdateSlotPlanError, match="active transition"):
        UpdateSlotPlan.from_active_mask(torch.zeros((8, 28), dtype=torch.bool))
    with pytest.raises(TypeError, match="bool dtype"):
        UpdateSlotPlan.from_active_mask(torch.ones((8, 28)))
    with pytest.raises(ValueError, match=r"shape \[B,T\]"):
        UpdateSlotPlan.from_active_mask(torch.ones((8,), dtype=torch.bool))
    with pytest.raises(ValueError, match="permutation"):
        UpdateSlotPlan.from_active_mask(active_mask, row_order=(0, 0))
    with pytest.raises(ValueError, match="row_microbatch_size"):
        UpdateSlotPlan.from_active_mask(active_mask, row_microbatch_size=3)
    with pytest.raises(ValueError, match="transition_window_size"):
        UpdateSlotPlan.from_active_mask(active_mask, transition_window_size=4)
