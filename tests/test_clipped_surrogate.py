from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import json
import math
from pathlib import Path

import pytest
import torch

from visual_rl.optimizers.clipped_surrogate import (
    ClippedSurrogateOutput,
    clipped_surrogate,
)
from visual_rl.optimizers.objective import PolicyLossInputs


def _inputs(
    *,
    base_advantage: torch.Tensor | None = None,
    algorithm_weight: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    clip_range: float = 0.2,
    reference_kl_weight: float = 0.0,
) -> PolicyLossInputs:
    return PolicyLossInputs(
        base_advantage=(
            base_advantage
            if base_advantage is not None
            else torch.ones((2, 2), dtype=torch.float64)
        ),
        algorithm_weight=(
            algorithm_weight
            if algorithm_weight is not None
            else torch.ones((2, 2), dtype=torch.float64)
        ),
        active_mask=(
            active_mask
            if active_mask is not None
            else torch.ones((2, 2), dtype=torch.bool)
        ),
        clip_range=clip_range,
        reference_kl_weight=reference_kl_weight,
    )


def test_policy_loss_inputs_has_the_frozen_exact_contract_and_slices_b() -> None:
    inputs = _inputs(
        base_advantage=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        ),
        algorithm_weight=torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float64,
        ),
        active_mask=torch.tensor([[True, False], [False, True]]),
        reference_kl_weight=0.3,
    )

    assert [item.name for item in fields(PolicyLossInputs)] == [
        "base_advantage",
        "algorithm_weight",
        "active_mask",
        "clip_range",
        "reference_kl_weight",
    ]
    sliced = inputs.slice([1])
    assert torch.equal(sliced.base_advantage, inputs.base_advantage[1:])
    assert torch.equal(sliced.algorithm_weight, inputs.algorithm_weight[1:])
    assert torch.equal(sliced.active_mask, inputs.active_mask[1:])
    assert sliced.clip_range == inputs.clip_range
    assert sliced.reference_kl_weight == inputs.reference_kl_weight
    with pytest.raises(FrozenInstanceError):
        inputs.clip_range = 0.1


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"clip_range": 0.0}, "0 < clip_range < 1"),
        ({"clip_range": 1.0}, "0 < clip_range < 1"),
        ({"clip_range": math.nan}, "finite"),
        ({"reference_kl_weight": -1.0}, "non-negative"),
        ({"reference_kl_weight": math.inf}, "finite"),
        (
            {
                "algorithm_weight": torch.tensor(
                    [[0.0, 1.0], [1.0, 1.0]],
                    dtype=torch.float64,
                )
            },
            "strictly positive",
        ),
        (
            {
                "base_advantage": torch.tensor(
                    [[math.nan, 1.0], [1.0, 1.0]],
                    dtype=torch.float64,
                )
            },
            "finite",
        ),
    ],
)
def test_policy_loss_inputs_rejects_invalid_active_values(
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _inputs(**updates)


def test_policy_loss_inputs_allows_nan_only_where_inactive() -> None:
    inputs = _inputs(
        base_advantage=torch.tensor(
            [[1.0, math.nan], [2.0, 3.0]],
            dtype=torch.float64,
        ),
        algorithm_weight=torch.tensor(
            [[1.0, math.nan], [2.0, 3.0]],
            dtype=torch.float64,
        ),
        active_mask=torch.tensor([[True, False], [True, True]]),
    )
    with pytest.raises(TypeError, match="RolloutBatch"):
        inputs.validate_against(object())


def test_clipped_surrogate_matches_hand_formula_and_masks_nan_gradients() -> None:
    active_mask = torch.tensor([[True, False], [True, True]])
    inputs = _inputs(
        base_advantage=torch.tensor(
            [[1.0, math.nan], [-1.0, 0.5]],
            dtype=torch.float64,
        ),
        algorithm_weight=torch.tensor(
            [[2.0, math.nan], [1.0, 1.0]],
            dtype=torch.float64,
        ),
        active_mask=active_mask,
    )
    old_log_probs = torch.tensor(
        [[0.0, math.nan], [0.0, 0.0]],
        dtype=torch.float64,
    )
    new_log_probs = torch.tensor(
        [[math.log(1.3), math.nan], [math.log(0.7), math.log(1.1)]],
        dtype=torch.float64,
        requires_grad=True,
    )

    output = clipped_surrogate(
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        inputs=inputs,
    )

    assert isinstance(output, ClippedSurrogateOutput)
    assert output.active_transition_count == 3
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "characterization"
            / "v0_6_loss_probe.json"
        ).read_text(encoding="utf-8")
    )
    expected = fixture["v0_7_cutover"]["clipped_surrogate"]
    expected_policy = expected["policy_loss"]
    expected_kl = expected["approx_kl"]
    torch.testing.assert_close(
        output.policy_loss,
        torch.tensor(expected_policy, dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.approx_kl,
        torch.tensor(expected_kl, dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.clipfrac,
        torch.tensor(expected["clipfrac"], dtype=torch.float64),
    )
    assert output.active_transition_count == expected[
        "active_transition_count"
    ]

    output.policy_loss.backward()
    assert new_log_probs.grad is not None
    assert new_log_probs.grad[0, 1].item() == 0.0
    assert bool(torch.isfinite(new_log_probs.grad).all())


def test_clipped_surrogate_rejects_zero_active_and_shape_broadcast() -> None:
    with pytest.raises(ValueError, match="at least one active"):
        clipped_surrogate(
            old_log_probs=torch.zeros((2, 2), dtype=torch.float64),
            new_log_probs=torch.zeros(
                (2, 2),
                dtype=torch.float64,
                requires_grad=True,
            ),
            inputs=_inputs(active_mask=torch.zeros((2, 2), dtype=torch.bool)),
        )

    with pytest.raises(ValueError, match="same shape"):
        clipped_surrogate(
            old_log_probs=torch.zeros((2, 2), dtype=torch.float64),
            new_log_probs=torch.zeros(
                (2, 2),
                dtype=torch.float64,
                requires_grad=True,
            ),
            inputs=PolicyLossInputs(
                base_advantage=torch.ones((2, 1), dtype=torch.float64),
                algorithm_weight=torch.ones((2, 1), dtype=torch.float64),
                active_mask=torch.ones((2, 1), dtype=torch.bool),
                clip_range=0.2,
            ),
        )


def test_clipped_surrogate_only_new_log_probs_is_differentiable() -> None:
    detached = torch.ones((2, 2), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        _inputs(base_advantage=detached)
    with pytest.raises(ValueError, match="detached"):
        clipped_surrogate(
            old_log_probs=detached,
            new_log_probs=torch.ones(
                (2, 2),
                dtype=torch.float64,
                requires_grad=True,
            ),
            inputs=_inputs(),
        )


def test_production_has_one_clipped_surrogate_caller_and_one_exp_formula() -> None:
    package = Path(__file__).parents[1] / "visual_rl"
    call_sites: list[str] = []
    exp_sites: list[str] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(package).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "clipped_surrogate":
                call_sites.append(relative)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "exp"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                exp_sites.append(relative)

    assert call_sites == ["optimizers/objective.py"]
    assert exp_sites == ["optimizers/clipped_surrogate.py"]
