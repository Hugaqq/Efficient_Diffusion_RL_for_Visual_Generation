"""Regression coverage for Wan recomputation latent precision."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from visual_rl.core.types import (
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.model_adapters.wan import WanFlashAdapter


def test_wan_recompute_casts_model_input_without_rounding_sde_latents() -> None:
    received: dict[str, torch.Tensor] = {}

    class RecordingTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # PEFT commonly enumerates its FP32 adapter before the frozen BF16
            # Wan patch embedding. The first parameter must not determine the
            # activation dtype.
            self.lora_anchor = torch.nn.Parameter(
                torch.zeros((), dtype=torch.float32)
            )
            self.patch_embedding = torch.nn.Conv3d(
                1,
                1,
                kernel_size=1,
                dtype=torch.bfloat16,
            )

        def forward(self, *, hidden_states, **_kwargs):
            received["transformer_hidden_states"] = hidden_states.detach().clone()
            return (hidden_states + self.lora_anchor.to(hidden_states.dtype),)

    def recording_sde_step(
        _scheduler,
        model_output,
        _timestep,
        sample,
        *,
        prev_sample,
        return_dt_and_std_dev_t=False,
    ):
        assert return_dt_and_std_dev_t is True
        received["sde_current_latent"] = sample.detach().clone()
        received["sde_next_latent"] = prev_sample.detach().clone()
        log_prob = model_output.reshape(model_output.shape[0], -1).mean(dim=1)
        coefficient = torch.ones(
            (sample.shape[0], 1, 1, 1, 1),
            dtype=sample.dtype,
            device=sample.device,
        )
        return prev_sample, log_prob, sample, coefficient, coefficient, coefficient

    transformer = RecordingTransformer()
    adapter = WanFlashAdapter(
        checkpoint=Path("/offline/fake-wan"),
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=1.0,
        height=64,
        width=64,
        frames=9,
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
    adapter.pipeline = SimpleNamespace(transformer=transformer)
    adapter.transformer = transformer
    adapter.scheduler = object()
    adapter._load_sde_function = lambda: recording_sde_step

    current = torch.tensor(
        [[[[[[1.0001, -2.0003]]]]]],
        dtype=torch.float32,
    )
    next_ = torch.tensor(
        [[[[[[3.0007, -4.0013]]]]]],
        dtype=torch.float32,
    )
    assert not torch.equal(current.to(torch.bfloat16).float(), current)
    assert not torch.equal(next_.to(torch.bfloat16).float(), next_)

    batch = RolloutBatch(
        prompts=("precision regression",),
        metadata=({},),
        media=torch.zeros(1, 1, 3, 1, 1),
        latents=current,
        next_latents=next_,
        timesteps=torch.tensor([[999]]),
        old_log_probs=torch.zeros(1, 1),
        transition_mask=torch.ones(1, 1, dtype=torch.bool),
        sample_id=("sample-0",),
        prompt_id=("prompt-0",),
        group_id=("group-0",),
        branch_id=None,
        media_layout="BFCHW",
        camera_trajectory=None,
        context=StepContext(step=0, seed=7),
        selected_timestep_index=torch.tensor([0], dtype=torch.int64),
        flash_coefficient=torch.ones(1, 1),
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={"prompt_embeds": torch.ones(1, 1)},
        artifact_metadata={"adapter": "wan_flash"},
    )

    stats = adapter.recompute_policy_stats(batch)
    stats.new_log_probs.sum().backward()

    hidden_states = received["transformer_hidden_states"]
    assert hidden_states.dtype is torch.bfloat16
    assert not torch.equal(hidden_states.float(), current[:, 0])
    assert transformer.lora_anchor.grad is not None

    sde_current = received["sde_current_latent"]
    sde_next = received["sde_next_latent"]
    assert sde_current.dtype is torch.float32
    assert sde_next.dtype is torch.float32
    torch.testing.assert_close(sde_current, current[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(sde_next, next_[:, 0], rtol=0, atol=0)
