"""Numerical transition kernels shared by native SD3 and Wan rollouts."""

from __future__ import annotations

import math
from typing import Literal

from visual_rl.errors import RunError


def sd3_sde_step_with_logprob(
    scheduler,
    model_output,
    timestep,
    sample,
    *,
    prev_sample=None,
    generator=None,
    deterministic: bool = False,
    noise_level: float = 0.7,
):
    """Advance one SD3 flow-matching transition and evaluate its log-prob."""

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()
    if prev_sample is not None and generator is not None:
        raise ValueError("prev_sample and generator are mutually exclusive")

    sigma, sigma_prev = _sigma_pair(scheduler, timestep, sample)
    sigmas = torch.as_tensor(
        scheduler.sigmas,
        device=sample.device,
        dtype=sample.dtype,
    )
    if sigmas.numel() < 2:
        raise RunError("SD3 scheduler must expose at least two sigmas")
    sigma_max = sigmas[1]
    dt = sigma_prev - sigma
    safe_denominator = 1 - torch.where(sigma == 1, sigma_max, sigma)
    diffusion_scale = torch.sqrt(sigma / safe_denominator) * noise_level
    mean = (
        sample * (1 + diffusion_scale.square() / (2 * sigma) * dt)
        + model_output
        * (1 + diffusion_scale.square() * (1 - sigma) / (2 * sigma))
        * dt
    )
    transition_std = diffusion_scale * torch.sqrt(-dt)

    if deterministic:
        next_sample = sample + dt * model_output
    elif prev_sample is None:
        noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        next_sample = mean + transition_std * noise
    else:
        next_sample = prev_sample

    log_prob = _gaussian_log_prob(next_sample, mean, transition_std)
    return next_sample, log_prob, mean, transition_std


def wan_sde_step_with_logprob(
    scheduler,
    model_output,
    timestep,
    sample,
    *,
    variant: Literal["flash", "world_r1"],
    prev_sample=None,
    generator=None,
    deterministic: bool = False,
    return_flash_coefficient: bool = False,
):
    """Advance one Wan transition using the frozen Flash or World-R1 rule."""

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    if variant not in {"flash", "world_r1"}:
        raise ValueError(f"unsupported Wan transition variant: {variant}")
    if return_flash_coefficient and variant != "flash":
        raise ValueError("only Flash transitions expose a coefficient")
    if prev_sample is not None and generator is not None:
        raise ValueError("prev_sample and generator are mutually exclusive")

    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()
    sigma, sigma_prev = _sigma_pair(scheduler, timestep, sample)
    sigmas = torch.as_tensor(
        scheduler.sigmas,
        device=sample.device,
        dtype=sample.dtype,
    )
    sigma_max_index = 1 if variant == "flash" else 0
    if sigmas.numel() <= sigma_max_index:
        raise RunError("Wan scheduler does not expose enough sigmas")
    sigma_max = sigmas[sigma_max_index]
    sigma_min = sigmas[-1]
    dt = sigma_prev - sigma
    diffusion_scale = sigma_min + (sigma_max - sigma_min) * sigma

    scheduler_is_deterministic = (
        variant == "world_r1"
        and not bool(
            getattr(
                getattr(scheduler, "config", object()),
                "stochastic_sampling",
                False,
            )
        )
    )
    if scheduler_is_deterministic:
        mean = sample + dt * model_output
    else:
        mean = (
            sample * (1 + diffusion_scale.square() / (2 * sigma) * dt)
            + model_output
            * (1 + diffusion_scale.square() * (1 - sigma) / (2 * sigma))
            * dt
        )
    sqrt_negative_dt = torch.sqrt(-dt)
    transition_std = diffusion_scale * sqrt_negative_dt

    if prev_sample is not None:
        next_sample = (
            sample + dt * model_output
            if variant == "flash" and deterministic
            else prev_sample
        )
    elif deterministic or scheduler_is_deterministic:
        next_sample = mean if scheduler_is_deterministic else sample + dt * model_output
    else:
        noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        next_sample = mean + transition_std * noise

    epsilon = 1.0e-12 if variant == "world_r1" else 0.0
    log_prob = _gaussian_log_prob(
        next_sample,
        mean,
        transition_std,
        epsilon=epsilon,
    )
    if not return_flash_coefficient:
        return next_sample, log_prob, mean, transition_std

    coefficient = 1 / (
        sqrt_negative_dt / diffusion_scale
        + diffusion_scale * sqrt_negative_dt * (1 - sigma) / (2 * sigma)
    )
    return (
        next_sample,
        log_prob,
        mean,
        diffusion_scale,
        sqrt_negative_dt,
        coefficient,
    )


def _sigma_pair(scheduler, timestep, sample):
    import torch

    values = torch.as_tensor(timestep, device=sample.device).reshape(-1)
    indices = [int(scheduler.index_for_timestep(value)) for value in values]
    previous = [index + 1 for index in indices]
    sigmas = torch.as_tensor(
        scheduler.sigmas,
        device=sample.device,
        dtype=sample.dtype,
    )
    if not indices or max(previous) >= sigmas.numel():
        raise RunError("scheduler timestep does not have a following sigma")
    shape = (-1, *([1] * (sample.ndim - 1)))
    return sigmas[indices].view(shape), sigmas[previous].view(shape)


def _gaussian_log_prob(sample, mean, std, *, epsilon: float = 0.0):
    import torch

    variance = std.square() + epsilon
    normalization = std + epsilon
    log_prob = (
        -(sample.detach() - mean).square() / (2 * variance)
        - torch.log(normalization)
        - 0.5
        * torch.log(
            torch.as_tensor(
                2 * math.pi,
                device=sample.device,
                dtype=sample.dtype,
            )
        )
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
