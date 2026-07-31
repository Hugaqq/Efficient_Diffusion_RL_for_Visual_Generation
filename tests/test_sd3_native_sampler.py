"""Closed-loop CPU coverage for the in-package SD3 samplers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from visual_rl.core.types import (
    RolloutRequest,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.model_adapters.diffusion_transition import (
    sd3_sde_step_with_logprob,
)
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter


class _Scheduler:
    def __init__(self) -> None:
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.timesteps = torch.tensor([900.0, 500.0])
        self.config = {
            "use_dynamic_shifting": False,
        }
        self.order = 1

    def set_timesteps(self, num_inference_steps, *, device, **_kwargs):
        assert num_inference_steps == 2
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)

    def index_for_timestep(self, timestep) -> int:
        matches = (self.timesteps == timestep).nonzero().reshape(-1)
        return int(matches.item())


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.05))
        self.config = SimpleNamespace(in_channels=1, patch_size=1)

    def forward(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        **_kwargs,
    ):
        prompt = encoder_hidden_states.mean(dim=(1, 2)).reshape(
            hidden_states.shape[0],
            1,
            1,
            1,
        )
        pooled = pooled_projections.mean(dim=1).reshape(
            hidden_states.shape[0],
            1,
            1,
            1,
        )
        return (hidden_states * 0.1 + prompt + pooled + self.anchor,)


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    @staticmethod
    def decode(latents, *, return_dict):
        assert return_dict is False
        return (latents.repeat(1, 3, 1, 1),)


class _ImageProcessor:
    @staticmethod
    def postprocess(image, *, output_type):
        assert output_type == "pt"
        return image


class _Pipeline:
    def __init__(self, transformer, scheduler) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = _VAE()
        self.image_processor = _ImageProcessor()

    @staticmethod
    def prepare_latents(
        batch_size,
        channels,
        _height,
        _width,
        dtype,
        device,
        _generator,
        _latents,
    ):
        return torch.ones(
            batch_size,
            channels,
            2,
            2,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def encode_prompt(**kwargs):
        batch_size = len(kwargs["prompt"])
        positive = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.float32,
        )[:, None, None].expand(batch_size, 2, 2)
        negative = torch.zeros_like(positive)
        pooled = positive[:, 0]
        negative_pooled = torch.zeros_like(pooled)
        return positive, negative, pooled, negative_pooled


def _adapter() -> SD3TempFlowAdapter:
    adapter = SD3TempFlowAdapter(
        checkpoint=None,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=1.0,
        resolution=16,
        max_sequence_length=32,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=RuntimeBuildContext(
            rank=0,
            local_rank=0,
            world_size=1,
            backend=None,
            device=torch.device("cpu"),
            precision="fp32",
        ),
    )
    transformer = _Transformer()
    pipeline = _Pipeline(transformer, _Scheduler())
    adapter.pipeline = pipeline
    adapter.transformer = transformer
    adapter._encode_prompt = pipeline.encode_prompt
    adapter._sde_step = sd3_sde_step_with_logprob
    return adapter


def test_sd3_full_native_sampler_recomputes_the_same_trajectory() -> None:
    adapter = _adapter()
    context = StepContext(step=0, seed=17)
    request = RolloutRequest(
        prompts=("first", "first"),
        metadata=({}, {}),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-0"),
        group_id=("group-0", "group-0"),
        branch_id=None,
        context=context,
        kind="full_trajectory",
        num_steps=2,
        group_size=2,
    )

    batch = adapter.sample(request)
    stats = adapter.recompute_policy_stats(batch)

    assert batch.media.shape == (2, 3, 2, 2)
    assert batch.old_log_probs.shape == (2, 2)
    torch.testing.assert_close(stats.new_log_probs, batch.old_log_probs)


def test_sd3_branching_native_sampler_recomputes_selected_branch() -> None:
    adapter = _adapter()
    context = StepContext(step=0, seed=19)
    branch_count = 6
    request = RolloutRequest(
        prompts=("branch",) * branch_count,
        metadata=({},) * branch_count,
        sample_id=tuple(f"sample-{index}" for index in range(branch_count)),
        prompt_id=("prompt-0",) * branch_count,
        group_id=("group-0",) * branch_count,
        branch_id=tuple(range(branch_count)),
        context=context,
        kind="branching",
        num_steps=2,
        group_size=branch_count,
        branch_step_index=(0,) * branch_count,
    )

    batch = adapter.sample(request)
    stats = adapter.recompute_policy_stats(batch)

    assert batch.media.shape == (branch_count, 3, 2, 2)
    assert batch.old_log_probs.shape == (branch_count, 1)
    assert batch.transition_std_dev.shape == (branch_count, 1)
    torch.testing.assert_close(stats.new_log_probs, batch.old_log_probs)
