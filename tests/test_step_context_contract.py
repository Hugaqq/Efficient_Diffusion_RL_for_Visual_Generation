"""StepContext construction contract tests (v0.7 W02).

Covers the frozen four-field identity contract from the master plan stage
2.1: non-bool ints, ``step >= 0``, canonical uint32 seed range,
``world_size >= 1``, ``0 <= rank < world_size``, and the max-seed formula
budget. All violations must fail before any dataset/rollout work — i.e. at
``StepContext`` construction time.
"""

from __future__ import annotations

import dataclasses
import pickle

import pytest

from visual_rl.core.types import (
    UINT32_MAX,
    StepContext,
    validate_step_seed_budget,
)


def test_minimal_and_full_construction():
    minimal = StepContext(step=0, seed=0, epoch_tag=0)
    assert minimal.rank == 0
    assert minimal.world_size == 1
    full = StepContext(step=3, seed=7, epoch_tag=3, rank=1, world_size=2)
    assert (full.step, full.seed, full.rank, full.world_size) == (3, 7, 1, 2)
    assert dataclasses.asdict(full)["seed"] == 7


def test_context_is_frozen_hashable_and_picklable():
    context = StepContext(step=1, seed=2, epoch_tag=1, rank=0, world_size=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.step = 5
    assert hash(context) == hash(StepContext(step=1, seed=2, epoch_tag=1))
    assert pickle.loads(pickle.dumps(context)) == context


@pytest.mark.parametrize("field_name", ("step", "seed", "rank", "world_size"))
def test_bool_is_rejected_for_every_identity_field(field_name):
    kwargs = {"step": 0, "seed": 0, "epoch_tag": 0, "rank": 0, "world_size": 1}
    kwargs[field_name] = True
    with pytest.raises(TypeError, match="not bool"):
        StepContext(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("step", 1.0),
        ("seed", "123"),
        ("rank", 0.5),
        ("world_size", None),
    ),
)
def test_non_integer_values_are_rejected(field_name, value):
    kwargs = {"step": 0, "seed": 0, "epoch_tag": 0, "rank": 0, "world_size": 1}
    kwargs[field_name] = value
    with pytest.raises(TypeError, match="must be an integer"):
        StepContext(**kwargs)


def test_negative_step_is_rejected():
    with pytest.raises(ValueError, match="step must be non-negative"):
        StepContext(step=-1, seed=0, epoch_tag=0)


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="uint32"):
        StepContext(step=0, seed=-1, epoch_tag=0)


def test_seed_above_uint32_is_rejected():
    with pytest.raises(ValueError, match="uint32"):
        StepContext(step=0, seed=UINT32_MAX + 1, epoch_tag=0)


def test_uint32_boundary_seed_is_accepted():
    context = StepContext(step=0, seed=UINT32_MAX, epoch_tag=0)
    assert context.seed == UINT32_MAX


def test_zero_world_size_is_rejected():
    with pytest.raises(ValueError, match="world_size must be positive"):
        StepContext(step=0, seed=0, epoch_tag=0, world_size=0)


def test_negative_world_size_is_rejected():
    with pytest.raises(ValueError, match="world_size must be positive"):
        StepContext(step=0, seed=0, epoch_tag=0, rank=0, world_size=-2)


def test_rank_must_stay_below_world_size():
    with pytest.raises(ValueError, match="0 <= rank < world_size"):
        StepContext(step=0, seed=0, epoch_tag=0, rank=2, world_size=2)
    with pytest.raises(ValueError, match="0 <= rank < world_size"):
        StepContext(step=0, seed=0, epoch_tag=0, rank=-1, world_size=2)


def test_rank_world_size_boundary_is_accepted():
    context = StepContext(step=0, seed=0, epoch_tag=0, rank=1, world_size=2)
    assert context.rank == 1


# ---------------------------------------------------------------------------
# Canonical per-step seed formula budget:
# seed + (max_steps - 1) * world_size + (world_size - 1) <= 0xFFFFFFFF
# ---------------------------------------------------------------------------


def test_seed_budget_accepts_exact_boundary():
    validate_step_seed_budget(UINT32_MAX, 1, 1)
    validate_step_seed_budget(0, 1, 1)
    validate_step_seed_budget(UINT32_MAX - 3, 2, 2)  # final: UINT32_MAX


def test_seed_budget_rejects_formula_overflow():
    with pytest.raises(ValueError, match="uint32"):
        validate_step_seed_budget(UINT32_MAX, 2, 1)  # final: UINT32_MAX + 1
    with pytest.raises(ValueError, match="uint32"):
        validate_step_seed_budget(UINT32_MAX, 1, 2)  # final: UINT32_MAX + 1
    with pytest.raises(ValueError, match="uint32"):
        validate_step_seed_budget(UINT32_MAX - 3, 3, 2)  # final: UINT32_MAX + 1


def test_seed_budget_rejects_invalid_inputs_before_rollout():
    with pytest.raises(TypeError, match="not bool"):
        validate_step_seed_budget(True, 1, 1)
    with pytest.raises(ValueError, match="uint32"):
        validate_step_seed_budget(-1, 1, 1)
    with pytest.raises(ValueError, match="uint32"):
        validate_step_seed_budget(UINT32_MAX + 1, 1, 1)
    with pytest.raises(ValueError, match="max_steps must be positive"):
        validate_step_seed_budget(0, 0, 1)
    with pytest.raises(ValueError, match="world_size must be positive"):
        validate_step_seed_budget(0, 1, 0)


def test_max_seed_formula_produces_valid_final_step_contexts():
    seed, max_steps, world_size = UINT32_MAX - 3, 2, 2
    validate_step_seed_budget(seed, max_steps, world_size)
    final_step = max_steps - 1
    final_rank = world_size - 1
    final_seed = seed + final_step * world_size + final_rank
    assert final_seed == UINT32_MAX
    context = StepContext(
        step=final_step,
        seed=final_seed,
        epoch_tag=final_step,
        rank=final_rank,
        world_size=world_size,
    )
    assert context.seed == UINT32_MAX
