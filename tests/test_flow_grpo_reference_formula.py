from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from visual_rl.core.types import (
    PolicyRecomputeStats,
    RolloutBatch,
    StepContext,
)
from visual_rl.errors import RunError
from visual_rl.optimizers.objective import (
    ObjectiveOutput,
    PolicyLossInputs,
    PolicyObjective,
    reference_regularizer,
)


def _batch(
    old_log_probs: torch.Tensor,
    active_mask: torch.Tensor,
) -> RolloutBatch:
    batch_size, transition_count = old_log_probs.shape
    return RolloutBatch(
        prompts=tuple("prompt" for _ in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transition_count, 1),
        next_latents=torch.ones(batch_size, transition_count, 1),
        timesteps=torch.arange(transition_count).expand(batch_size, -1),
        old_log_probs=old_log_probs,
        transition_mask=active_mask,
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=tuple(f"prompt-{index}" for index in range(batch_size)),
        group_id=tuple(f"group-{index}" for index in range(batch_size)),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=0, seed=1),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={},
        artifact_metadata={},
    )


def _reference_tensors(
    std_shape: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_mask = torch.tensor([[True, False]])
    current = torch.tensor(
        [[[[[2.0]], [[4.0]]], [[[math.nan]], [[math.nan]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference = torch.tensor(
        [[[[[0.0]], [[0.0]]], [[[math.nan]], [[math.nan]]]]],
        dtype=torch.float64,
    )
    if std_shape == "bt":
        std = torch.tensor([[2.0, math.nan]], dtype=torch.float64)
    elif std_shape == "full":
        std = torch.tensor(
            [[[[[2.0]], [[2.0]]], [[[math.nan]], [[math.nan]]]]],
            dtype=torch.float64,
        )
    elif std_shape == "singleton":
        std = torch.tensor(
            [[[[[2.0]]], [[[math.nan]]]]],
            dtype=torch.float64,
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(std_shape)
    return current, reference, std, active_mask


@pytest.mark.parametrize("std_shape", ["bt", "full", "singleton"])
def test_reference_regularizer_matches_hand_formula_for_all_std_shapes(
    std_shape: str,
) -> None:
    current, reference, std, active_mask = _reference_tensors(std_shape)

    result = reference_regularizer(
        current_mean=current,
        reference_mean=reference,
        transition_std=std,
        active_mask=active_mask,
    )

    torch.testing.assert_close(
        result,
        torch.tensor(1.25, dtype=torch.float64),
    )
    result.backward()
    assert current.grad is not None
    assert torch.equal(
        current.grad[0, 1],
        torch.zeros_like(current.grad[0, 1]),
    )
    assert bool(torch.isfinite(current.grad).all())


@pytest.mark.parametrize("active_std", [0.0, -1.0, math.nan, math.inf])
def test_reference_regularizer_rejects_nonpositive_or_nonfinite_active_std(
    active_std: float,
) -> None:
    current, reference, _std, active_mask = _reference_tensors("bt")
    std = torch.tensor([[active_std, math.nan]], dtype=torch.float64)
    with pytest.raises(RunError, match="transition_std"):
        reference_regularizer(
            current_mean=current,
            reference_mean=reference,
            transition_std=std,
            active_mask=active_mask,
        )


def test_reference_regularizer_rejects_unsupported_std_shape() -> None:
    current, reference, _std, active_mask = _reference_tensors("bt")
    with pytest.raises(RunError, match="unsupported transition_std shape"):
        reference_regularizer(
            current_mean=current,
            reference_mean=reference,
            transition_std=torch.ones((1, 2, 2), dtype=torch.float64),
            active_mask=active_mask,
        )


def test_policy_objective_combines_surrogate_and_reference_kl() -> None:
    active_mask = torch.tensor([[True, False]])
    inputs = PolicyLossInputs(
        base_advantage=torch.tensor(
            [[1.0, math.nan]],
            dtype=torch.float64,
        ),
        algorithm_weight=torch.tensor(
            [[1.0, math.nan]],
            dtype=torch.float64,
        ),
        active_mask=active_mask,
        clip_range=0.2,
        reference_kl_weight=0.3,
    )
    old_log_probs = torch.tensor([[0.0, math.nan]], dtype=torch.float64)
    new_log_probs = torch.tensor(
        [[math.log(1.1), math.nan]],
        dtype=torch.float64,
        requires_grad=True,
    )
    current, reference, std, _ = _reference_tensors("bt")
    stats = PolicyRecomputeStats(
        new_log_probs=new_log_probs,
        current_transition_mean=current,
        transition_std=std,
        reference_transition_mean=reference,
    )

    output = PolicyObjective()(
        _batch(old_log_probs, active_mask),
        inputs,
        stats,
    )

    assert [item.name for item in fields(ObjectiveOutput)] == [
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
        "active_transition_count",
    ]
    assert output.active_transition_count == 1
    torch.testing.assert_close(
        output.policy_loss,
        torch.tensor(-1.1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.reference_kl,
        torch.tensor(1.25, dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.loss,
        torch.tensor(-1.1 + 0.3 * 1.25, dtype=torch.float64),
    )
    total_new_grad, total_current_grad = torch.autograd.grad(
        output.loss,
        (new_log_probs, current),
        retain_graph=True,
    )
    policy_new_grad = torch.autograd.grad(
        output.policy_loss,
        new_log_probs,
        retain_graph=True,
    )[0]
    reference_current_grad = torch.autograd.grad(
        output.reference_kl,
        current,
    )[0]
    torch.testing.assert_close(total_new_grad, policy_new_grad)
    torch.testing.assert_close(
        total_current_grad,
        inputs.reference_kl_weight * reference_current_grad,
    )
    assert total_new_grad[0, 1].item() == 0.0
    assert torch.equal(
        total_current_grad[0, 1],
        torch.zeros_like(total_current_grad[0, 1]),
    )
    assert bool(torch.isfinite(total_new_grad).all())
    assert bool(torch.isfinite(total_current_grad).all())


def test_policy_objective_beta_zero_returns_typed_zero_without_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visual_rl.optimizers.objective as objective_module

    def forbidden_reference(**_kwargs: object) -> torch.Tensor:
        raise AssertionError("beta=0 must not evaluate reference statistics")

    monkeypatch.setattr(
        objective_module,
        "reference_regularizer",
        forbidden_reference,
    )
    old_log_probs = torch.zeros((1, 1), dtype=torch.float64)
    new_log_probs = torch.zeros(
        (1, 1),
        dtype=torch.float64,
        requires_grad=True,
    )
    output = PolicyObjective()(
        _batch(
            old_log_probs,
            torch.ones((1, 1), dtype=torch.bool),
        ),
        PolicyLossInputs(
            base_advantage=torch.ones((1, 1), dtype=torch.float64),
            algorithm_weight=torch.ones((1, 1), dtype=torch.float64),
            active_mask=torch.ones((1, 1), dtype=torch.bool),
            clip_range=0.2,
        ),
        PolicyRecomputeStats(new_log_probs=new_log_probs),
    )
    assert output.reference_kl.shape == ()
    assert output.reference_kl.dtype == output.policy_loss.dtype
    assert output.reference_kl.device == output.policy_loss.device
    assert output.reference_kl.item() == 0.0
    assert output.loss.item() == output.policy_loss.item()


def test_policy_objective_beta_positive_requires_reference_statistics() -> None:
    with pytest.raises(RunError, match="reference KL requires"):
        PolicyObjective()(
            _batch(
                torch.zeros((1, 1), dtype=torch.float64),
                torch.ones((1, 1), dtype=torch.bool),
            ),
            PolicyLossInputs(
                base_advantage=torch.ones((1, 1), dtype=torch.float64),
                algorithm_weight=torch.ones((1, 1), dtype=torch.float64),
                active_mask=torch.ones((1, 1), dtype=torch.bool),
                clip_range=0.2,
                reference_kl_weight=0.1,
            ),
            PolicyRecomputeStats(
                new_log_probs=torch.zeros(
                    (1, 1),
                    dtype=torch.float64,
                    requires_grad=True,
                )
            ),
        )


def test_equal_current_and_reference_has_zero_kl_and_gradient() -> None:
    current = torch.tensor([[[1.0, -2.0]]], requires_grad=True)
    reference = current.detach().clone()
    result = reference_regularizer(
        current_mean=current,
        reference_mean=reference,
        transition_std=torch.ones((1, 1)),
        active_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    assert result.item() == 0.0
    result.backward()
    assert current.grad is not None
    assert torch.equal(current.grad, torch.zeros_like(current.grad))
