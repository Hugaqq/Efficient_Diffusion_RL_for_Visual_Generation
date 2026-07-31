"""Closed-loop CPU coverage for the in-package Wan sampler."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from visual_rl.core.types import (
    RolloutRequest,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.model_adapters.wan import WanFlashAdapter, WanWorldR1Adapter


class _Scheduler:
    def __init__(self) -> None:
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.timesteps = torch.tensor([900.0, 500.0])
        self.config = SimpleNamespace(stochastic_sampling=False)
        self.order = 1

    def set_timesteps(self, steps: int, *, device) -> None:
        assert steps == 2
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)

    def index_for_timestep(self, timestep) -> int:
        matches = (self.timesteps == timestep).nonzero().reshape(-1)
        return int(matches.item())

    def step(self, model_output, timestep, sample, *, return_dict):
        assert return_dict is False
        index = self.index_for_timestep(timestep)
        delta = self.sigmas[index + 1] - self.sigmas[index]
        return (sample + delta * model_output,)


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.05))
        self.config = SimpleNamespace(in_channels=1)
        self.training_modes: list[bool] = []
        self.cache_contexts: list[str] = []

    @contextmanager
    def cache_context(self, name: str):
        self.cache_contexts.append(name)
        yield

    def forward(self, *, hidden_states, encoder_hidden_states, **_kwargs):
        self.training_modes.append(self.training)
        conditioning = encoder_hidden_states.reshape(
            encoder_hidden_states.shape[0],
            *([1] * (hidden_states.ndim - 1)),
        )
        return (hidden_states * 0.1 + conditioning + self.anchor,)


class _VAE(torch.nn.Module):
    def __init__(self, frames: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.frames = frames
        self.config = SimpleNamespace(
            latents_mean=[0.0],
            latents_std=[1.0],
            z_dim=1,
        )

    def decode(self, latents, *, return_dict):
        assert return_dict is False
        image = latents[:, :1, :1].repeat(1, 3, self.frames, 1, 1)
        return (image,)


class _VideoProcessor:
    @staticmethod
    def postprocess_video(video, *, output_type):
        assert output_type == "pt"
        return video.permute(0, 2, 1, 3, 4).contiguous()


class _Pipeline:
    def __init__(self, transformer, scheduler, *, frames: int) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = _VAE(frames)
        self.video_processor = _VideoProcessor()
        self.config = SimpleNamespace(expand_timesteps=False)

    def encode_prompt(self, *, prompt, negative_prompt, **_kwargs):
        positive = torch.arange(
            1,
            len(prompt) + 1,
            dtype=torch.float32,
        )[:, None]
        negative = torch.zeros_like(positive)
        assert len(negative_prompt) == len(prompt)
        return positive, negative

    @staticmethod
    def prepare_latents(
        batch_size,
        channels,
        _height,
        _width,
        _frames,
        dtype,
        device,
        _generator,
        latents,
    ):
        if latents is not None:
            return latents
        return torch.ones(
            batch_size,
            channels,
            2,
            2,
            2,
            dtype=dtype,
            device=device,
        )


def _context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _attach(adapter):
    transformer = _Transformer()
    scheduler = _Scheduler()
    pipeline = _Pipeline(transformer, scheduler, frames=adapter.frames)
    adapter.transformer = transformer
    adapter.scheduler = scheduler
    adapter.pipeline = pipeline
    return adapter


def test_flash_native_sampler_recomputes_the_same_selected_transition() -> None:
    adapter = _attach(
        WanFlashAdapter(
            checkpoint=None,
            lora_rank=4,
            lora_alpha=8,
            lora_target_modules=("to_q",),
            gradient_checkpointing=False,
            guidance_scale=1.0,
            height=16,
            width=16,
            frames=9,
            max_sequence_length=32,
            local_files_only=True,
            low_cpu_mem_usage=True,
            context=_context(),
        )
    )
    context = StepContext(step=0, seed=11)
    request = RolloutRequest(
        prompts=("first", "second"),
        metadata=({}, {}),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-1"),
        group_id=("group-0", "group-1"),
        branch_id=None,
        context=context,
        kind="single_step",
        num_steps=2,
        group_size=1,
        selected_timestep_index=(1, 1),
    )

    batch = adapter.sample(request)
    stats = adapter.recompute_policy_stats(batch)

    assert batch.media.shape == (2, 9, 3, 2, 2)
    assert batch.old_log_probs.shape == (2, 1)
    assert batch.flash_coefficient.shape == (2, 1)
    torch.testing.assert_close(stats.new_log_probs, batch.old_log_probs)
    assert adapter.transformer.training_modes
    assert not any(adapter.transformer.training_modes)
    assert adapter.transformer.cache_contexts == ["cond"] * 3


def test_world_native_sampler_keeps_all_transitions_and_camera_frames() -> None:
    adapter = _attach(
        WanWorldR1Adapter(
            checkpoint=None,
            lora_rank=4,
            lora_alpha=8,
            lora_target_modules=("to_q",),
            gradient_checkpointing=False,
            guidance_scale=1.0,
            height=16,
            width=16,
            frames=81,
            max_sequence_length=32,
            local_files_only=True,
            low_cpu_mem_usage=True,
            context=_context(),
        )
    )
    camera = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4).repeat(
        1,
        81,
        1,
        1,
    )
    adapter._prepare_world_camera = lambda _prompts, _generator: (
        torch.ones(1, 1, 2, 2, 2),
        None,
        camera,
    )
    context = StepContext(step=0, seed=13)
    request = RolloutRequest(
        prompts=("camera motion",),
        metadata=({},),
        sample_id=("sample-0",),
        prompt_id=("prompt-0",),
        group_id=("group-0",),
        branch_id=None,
        context=context,
        kind="full_trajectory",
        num_steps=2,
        group_size=1,
    )

    batch = adapter.sample(request)
    stats = adapter.recompute_policy_stats(batch)

    assert batch.media.shape == (1, 81, 3, 2, 2)
    assert batch.old_log_probs.shape == (1, 2)
    assert torch.equal(batch.camera_trajectory, camera)
    assert batch.trajectory_step_index is None
    torch.testing.assert_close(stats.new_log_probs, batch.old_log_probs)
    assert adapter.transformer.training_modes
    assert not any(adapter.transformer.training_modes)
    assert adapter.transformer.cache_contexts == ["cond"] * 4
