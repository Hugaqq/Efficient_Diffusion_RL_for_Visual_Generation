"""CPU characterization for the typed TempFlow policy-identity objective."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from visual_rl.core.types import (
    ResolutionContext,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm


def _batch(
    old_log_probs: torch.Tensor,
    *,
    branch_steps: tuple[int, ...],
    trajectory_steps: tuple[int, ...],
    transition_std_dev: torch.Tensor,
    transition_mask: torch.Tensor | None = None,
) -> RolloutBatch:
    batch_size, transition_count = old_log_probs.shape
    if transition_mask is None:
        transition_mask = torch.ones_like(old_log_probs, dtype=torch.bool)
    return RolloutBatch(
        prompts=tuple(f"prompt-{index}" for index in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transition_count, 1),
        next_latents=torch.ones(batch_size, transition_count, 1),
        timesteps=torch.arange(transition_count).expand(batch_size, -1),
        old_log_probs=old_log_probs,
        transition_mask=transition_mask,
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=tuple(
            f"prompt-id-{index}" for index in range(batch_size)
        ),
        group_id=tuple(f"group-{index // 2}" for index in range(batch_size)),
        branch_id=branch_steps,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=3, seed=19),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=torch.tensor(branch_steps, dtype=torch.int64),
        trajectory_step_index=torch.tensor(
            trajectory_steps,
            dtype=torch.int64,
        ),
        transition_std_dev=transition_std_dev,
        recompute_payload={
            "features": torch.ones(batch_size, transition_count)
        },
        artifact_metadata={},
    )


def _algorithm() -> TempFlowGRPOAlgorithm:
    return TempFlowGRPOAlgorithm(clip_range=0.2, adv_clip_max=5.0)


def test_canonical_params_have_no_legacy_tempflow_surface(tmp_path) -> None:
    context = ResolutionContext(
        config_path=(tmp_path / "config.yaml").absolute(),
        config_dir=tmp_path.absolute(),
    )
    resolved = TempFlowGRPOAlgorithm.resolve_params(
        {"clip_range": 0.2, "adv_clip_max": 4.0},
        context,
    )
    runtime = RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )
    algorithm = TempFlowGRPOAlgorithm.from_config(resolved, runtime)

    assert {item.name for item in fields(TempFlowGRPOAlgorithm)} == {
        "clip_range",
        "adv_clip_max",
    }
    assert algorithm.clip_range == 0.2
    assert algorithm.adv_clip_max == 4.0
    with pytest.raises(ValueError, match="unknown algorithm params"):
        TempFlowGRPOAlgorithm.resolve_params(
            {"clip_range": 0.2, "credit_assignment": "all"},
            context,
        )


def test_typed_branch_credit_and_noise_weight_match_literal_loss_gradient() -> None:
    algorithm = _algorithm()
    old_log_probs = torch.tensor(
        [[0.0, 0.02, -0.4, 0.1], [0.2, -0.3, -0.03, 0.5]],
        dtype=torch.float64,
    )
    std_dev = torch.tensor(
        [[0.2, 0.7, 1.1, 0.4], [0.3, 1.2, 0.5, 0.8]],
        dtype=torch.float64,
    )
    transition_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    batch = _batch(
        old_log_probs,
        branch_steps=(1, 2),
        trajectory_steps=(0, 1, 2, 3),
        transition_std_dev=std_dev,
        transition_mask=transition_mask,
    )
    advantages = torch.tensor([1.25, -0.5], dtype=torch.float64)
    actual_new = torch.tensor(
        [[0.1, 0.12, -0.2, 5000.0], [0.4, -0.1, -0.13, -5000.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    literal_new = actual_new.detach().clone().requires_grad_(True)

    actual_loss, actual_metrics = algorithm.compute_loss(
        batch,
        advantages,
        actual_new,
    )

    branch_mask = torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
    )
    active_mask = branch_mask & transition_mask
    expanded = torch.where(
        branch_mask,
        advantages[:, None],
        torch.zeros_like(literal_new),
    )
    weights = std_dev * 2.25
    effective = expanded.clamp(-5.0, 5.0) * weights
    delta = torch.where(
        active_mask,
        literal_new - old_log_probs,
        torch.zeros_like(literal_new),
    )
    ratio = torch.exp(delta)
    literal_loss = torch.maximum(
        -effective * ratio,
        -effective * ratio.clamp(0.8, 1.2),
    ).masked_select(active_mask).mean()
    literal_kl = (
        0.5 * delta.square().masked_select(active_mask).mean()
    )
    literal_clipfrac = (
        ((ratio - 1.0).abs() > 0.2)
        .to(torch.float64)
        .masked_select(active_mask)
        .mean()
    )

    actual_loss.backward()
    literal_loss.backward()

    torch.testing.assert_close(actual_loss, literal_loss)
    torch.testing.assert_close(actual_new.grad, literal_new.grad)
    torch.testing.assert_close(actual_metrics["approx_kl"], literal_kl)
    torch.testing.assert_close(
        actual_metrics["clipfrac"],
        literal_clipfrac,
    )
    torch.testing.assert_close(
        actual_metrics["tempflow_noise_weight_mean"],
        weights.masked_select(active_mask).mean(),
    )
    assert actual_metrics["tempflow_active_timestep_frac"].item() == 0.25
    assert torch.count_nonzero(
        actual_new.grad.masked_select(~active_mask)
    ).item() == 0


def test_compact_and_embedded_typed_trajectory_have_same_loss_and_gradient() -> None:
    algorithm = _algorithm()
    advantages = torch.tensor([1.25, 0.0], dtype=torch.float64)
    compact_old = torch.tensor([[0.02], [-0.03]], dtype=torch.float64)
    embedded_old = torch.tensor(
        [[0.4, -0.2, 0.02, 0.1], [-0.3, 0.5, -0.03, -0.4]],
        dtype=torch.float64,
    )
    compact = _batch(
        compact_old,
        branch_steps=(2, 2),
        trajectory_steps=(2,),
        transition_std_dev=torch.full((2, 1), 0.7, dtype=torch.float64),
    )
    embedded_std = torch.tensor(
        [[0.2, 1.3, 0.7, 1.8], [0.2, 1.3, 0.7, 1.8]],
        dtype=torch.float64,
    )
    embedded = _batch(
        embedded_old,
        branch_steps=(2, 2),
        trajectory_steps=(0, 1, 2, 3),
        transition_std_dev=embedded_std,
    )
    compact_parameter = torch.tensor(
        0.1,
        dtype=torch.float64,
        requires_grad=True,
    )
    embedded_parameter = torch.tensor(
        0.1,
        dtype=torch.float64,
        requires_grad=True,
    )
    compact_new = compact_parameter * torch.tensor(
        [[0.5], [-0.25]],
        dtype=torch.float64,
    )
    embedded_new = embedded_parameter * torch.tensor(
        [[1.5, -2.0, 0.5, 0.7], [-1.0, 1.25, -0.25, 2.0]],
        dtype=torch.float64,
    )

    compact_loss, compact_metrics = algorithm.compute_loss(
        compact,
        advantages,
        compact_new,
    )
    embedded_loss, embedded_metrics = algorithm.compute_loss(
        embedded,
        advantages,
        embedded_new,
    )
    compact_loss.backward()
    embedded_loss.backward()

    torch.testing.assert_close(compact_loss, embedded_loss)
    torch.testing.assert_close(
        compact_parameter.grad,
        embedded_parameter.grad,
    )
    for name in (
        "approx_kl",
        "clipfrac",
        "tempflow_noise_weight_mean",
    ):
        torch.testing.assert_close(
            compact_metrics[name],
            embedded_metrics[name],
        )
    assert compact_metrics["tempflow_active_timestep_frac"].item() == 1.0
    assert embedded_metrics["tempflow_active_timestep_frac"].item() == 0.25


def test_prepare_batch_only_adds_detached_typed_recompute_payload() -> None:
    algorithm = _algorithm()
    old = torch.zeros(2, 3, dtype=torch.float64)
    std_dev = torch.tensor(
        [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
        dtype=torch.float64,
    )
    batch = _batch(
        old,
        branch_steps=(1, 2),
        trajectory_steps=(0, 1, 2),
        transition_std_dev=std_dev,
    )
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    prepared = algorithm.prepare_batch(batch, advantages)

    assert algorithm._PREPARED_NOISE_WEIGHT_KEY not in batch.recompute_payload
    weights = prepared.recompute_payload[
        algorithm._PREPARED_NOISE_WEIGHT_KEY
    ]
    torch.testing.assert_close(weights, std_dev * 2.25)
    assert not weights.requires_grad

    parameter_a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    parameter_b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    features = torch.tensor(
        [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
        dtype=torch.float64,
    )
    direct_loss, _ = algorithm.compute_loss(
        batch,
        advantages,
        parameter_a * features,
    )
    prepared_loss, _ = algorithm.compute_loss(
        prepared,
        advantages,
        parameter_b * features,
    )
    direct_loss.backward()
    prepared_loss.backward()
    torch.testing.assert_close(direct_loss, prepared_loss)
    torch.testing.assert_close(parameter_a.grad, parameter_b.grad)


def test_branch_credit_requires_exact_typed_trajectory_match() -> None:
    algorithm = _algorithm()
    batch = _batch(
        torch.zeros(1, 2, dtype=torch.float64),
        branch_steps=(9,),
        trajectory_steps=(0, 1),
        transition_std_dev=torch.ones(1, 2, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="exactly one"):
        algorithm.compute_loss(
            batch,
            torch.ones(1, dtype=torch.float64),
            torch.zeros(1, 2, dtype=torch.float64),
        )


def test_masked_branch_transition_fails_without_active_transition() -> None:
    algorithm = _algorithm()
    batch = _batch(
        torch.zeros(1, 2, dtype=torch.float64),
        branch_steps=(1,),
        trajectory_steps=(0, 1),
        transition_std_dev=torch.ones(1, 2, dtype=torch.float64),
        transition_mask=torch.tensor([[True, False]]),
    )
    with pytest.raises(ValueError, match="at least one active transition"):
        algorithm.compute_loss(
            batch,
            torch.ones(1, dtype=torch.float64),
            torch.zeros(1, 2, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "advantages",
    [
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([1], dtype=torch.int64),
        [1.0],
    ],
)
def test_tempflow_requires_float64_group_advantages(advantages) -> None:
    algorithm = _algorithm()
    batch = _batch(
        torch.zeros(1, 1, dtype=torch.float64),
        branch_steps=(0,),
        trajectory_steps=(0,),
        transition_std_dev=torch.ones(1, 1, dtype=torch.float64),
    )
    with pytest.raises(TypeError, match="dtype=torch.float64"):
        algorithm.compute_loss(
            batch,
            advantages,
            torch.zeros(1, 1, dtype=torch.float64),
        )


def test_source_has_no_retired_rollout_side_channels() -> None:
    source = (
        Path(__file__).parents[1]
        / "visual_rl"
        / "optimizers"
        / "tempflow_grpo.py"
    ).read_text(encoding="utf-8")
    for retired in (
        "model_tensors",
        "model_metadata",
        "batch.kl",
        "credit_assignment",
        "noise_weighting",
        "reference_v1",
        "legacy",
    ):
        assert retired not in source
