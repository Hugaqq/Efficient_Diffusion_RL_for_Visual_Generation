"""Tests for the sole four-field per-step runtime identity."""

from __future__ import annotations

import dataclasses
import pickle

import pytest

from visual_rl.core.types import UINT32_MAX, StepContext, validate_step_seed_budget


def test_context_has_exactly_four_fields_and_round_trips():
    assert tuple(field.name for field in dataclasses.fields(StepContext)) == (
        "step",
        "seed",
        "rank",
        "world_size",
    )
    context = StepContext(step=3, seed=7, rank=1, world_size=2)
    assert pickle.loads(pickle.dumps(context)) == context
    assert hash(context) == hash(StepContext(step=3, seed=7, rank=1, world_size=2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.step = 4


@pytest.mark.parametrize("field_name", ("step", "seed", "rank", "world_size"))
def test_identity_fields_reject_bool(field_name):
    values = {"step": 0, "seed": 0, "rank": 0, "world_size": 1}
    values[field_name] = True
    with pytest.raises(TypeError, match="not bool"):
        StepContext(**values)


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ({"step": -1, "seed": 0}, "step must be non-negative"),
        ({"step": 0, "seed": -1}, "uint32"),
        ({"step": 0, "seed": UINT32_MAX + 1}, "uint32"),
        ({"step": 0, "seed": 0, "world_size": 0}, "world_size"),
        ({"step": 0, "seed": 0, "rank": -1, "world_size": 2}, "0 <= rank"),
        ({"step": 0, "seed": 0, "rank": 2, "world_size": 2}, "0 <= rank"),
    ),
)
def test_context_rejects_invalid_ranges(values, message):
    with pytest.raises(ValueError, match=message):
        StepContext(**values)


def test_seed_budget_uses_the_one_non_modulo_formula():
    validate_step_seed_budget(UINT32_MAX - 3, 2, 2)
    with pytest.raises(ValueError, match="uint32"):
        validate_step_seed_budget(UINT32_MAX - 3, 3, 2)
    with pytest.raises(TypeError, match="not bool"):
        validate_step_seed_budget(True, 1, 1)
    with pytest.raises(ValueError, match="max_steps"):
        validate_step_seed_budget(0, 0, 1)
    with pytest.raises(ValueError, match="world_size"):
        validate_step_seed_budget(0, 1, 0)
