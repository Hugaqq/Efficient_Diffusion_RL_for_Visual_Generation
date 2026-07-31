"""Numerical contracts for the one in-package diffusion transition module."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from visual_rl.model_adapters.diffusion_transition import (
    sd3_sde_step_with_logprob,
    wan_sde_step_with_logprob,
)


@dataclass
class _Config:
    stochastic_sampling: bool


class _Scheduler:
    def __init__(
        self,
        sigmas: tuple[float, ...],
        *,
        stochastic_sampling: bool,
    ) -> None:
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.timesteps = torch.tensor([900.0, 500.0], dtype=torch.float32)
        self.config = _Config(stochastic_sampling=stochastic_sampling)

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        matches = (self.timesteps == timestep).nonzero().reshape(-1)
        if matches.numel() != 1:
            raise ValueError("unknown timestep")
        return int(matches.item())


def _expected_log_prob(sample, mean, std, *, epsilon: float = 0.0):
    value = (
        -(sample - mean).square() / (2 * (std.square() + epsilon))
        - torch.log(std + epsilon)
        - 0.5 * math.log(2 * math.pi)
    )
    return value.mean(dim=tuple(range(1, value.ndim)))


def test_sd3_transition_matches_the_frozen_flow_equations() -> None:
    scheduler = _Scheduler(
        (1.0, 0.75, 0.25),
        stochastic_sampling=True,
    )
    sample = torch.tensor([[[[0.5]]], [[[1.0]]]])
    model_output = torch.tensor([[[[0.2]]], [[[0.4]]]], requires_grad=True)
    timestep = torch.tensor([900.0, 900.0])
    target = torch.tensor([[[[0.1]]], [[[0.3]]]])

    next_sample, log_prob, mean, std = sd3_sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        prev_sample=target,
    )

    sigma = torch.tensor(1.0).reshape(1, 1, 1, 1)
    sigma_prev = torch.tensor(0.75).reshape(1, 1, 1, 1)
    dt = sigma_prev - sigma
    diffusion = torch.sqrt(sigma / (1 - torch.tensor(0.75))) * 0.7
    expected_mean = (
        sample * (1 + diffusion.square() / (2 * sigma) * dt)
        + model_output
        * (1 + diffusion.square() * (1 - sigma) / (2 * sigma))
        * dt
    )
    expected_std = diffusion * torch.sqrt(-dt)
    assert torch.equal(next_sample, target)
    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(std, expected_std)
    assert torch.allclose(
        log_prob,
        _expected_log_prob(target, expected_mean, expected_std),
    )
    log_prob.sum().backward()
    assert model_output.grad is not None
    assert torch.isfinite(model_output.grad).all()


def test_world_transition_uses_scheduler_step_mean_when_deterministic() -> None:
    scheduler = _Scheduler(
        (1.0, 0.5, 0.0),
        stochastic_sampling=False,
    )
    sample = torch.ones(2, 1, 1, 1, 1)
    model_output = torch.full_like(sample, 0.25, requires_grad=True)
    timestep = torch.tensor([900.0, 900.0])
    expected = sample - 0.5 * model_output

    next_sample, log_prob, mean, std = wan_sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        variant="world_r1",
        prev_sample=expected.detach(),
    )

    assert torch.allclose(next_sample, expected)
    assert torch.allclose(mean, expected)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(std).all()
    log_prob.sum().backward()
    assert model_output.grad is not None


def test_flash_transition_exposes_the_frozen_rectification_coefficient() -> None:
    scheduler = _Scheduler(
        (1.0, 0.8, 0.2),
        stochastic_sampling=True,
    )
    sample = torch.ones(2, 1, 1, 1, 1)
    model_output = torch.full_like(sample, 0.25)
    timestep = torch.tensor([900.0, 900.0])
    target = torch.zeros_like(sample)

    result = wan_sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        variant="flash",
        prev_sample=target,
        return_flash_coefficient=True,
    )

    assert len(result) == 6
    _, log_prob, _, diffusion, sqrt_negative_dt, coefficient = result
    sigma = torch.tensor(1.0).reshape(1, 1, 1, 1, 1)
    expected = 1 / (
        sqrt_negative_dt / diffusion
        + diffusion * sqrt_negative_dt * (1 - sigma) / (2 * sigma)
    )
    assert torch.allclose(coefficient, expected)
    assert torch.isfinite(log_prob).all()


def test_transition_rejects_generator_and_observed_target_together() -> None:
    scheduler = _Scheduler(
        (1.0, 0.5, 0.0),
        stochastic_sampling=True,
    )
    sample = torch.ones(1, 1, 1, 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        sd3_sde_step_with_logprob(
            scheduler,
            sample,
            torch.tensor([900.0]),
            sample,
            prev_sample=sample,
            generator=torch.Generator(),
        )
