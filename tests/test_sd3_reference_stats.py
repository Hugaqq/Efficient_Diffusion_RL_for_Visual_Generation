"""CPU fakes for SD3 current/reference transition statistics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.core.types import (
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.errors import RunError
from visual_rl.model_adapters import sd3 as sd3_module
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter


class _FakePeftTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.25))
        self.register_buffer("base", torch.tensor(0.5))
        self.adapter_disabled = False
        self.disable_calls = 0
        self.restore_calls = 0
        self.training_modes: list[bool] = []

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        return_dict,
    ):
        del timestep, encoder_hidden_states, pooled_projections, return_dict
        self.training_modes.append(self.training)
        value = self.base
        if not self.adapter_disabled:
            value = value + self.delta
        return (torch.zeros_like(hidden_states) + value,)

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_disabled
        self.disable_calls += 1
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = previous
            self.restore_calls += 1


class _NoDisableTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.25))
        self.register_buffer("base", torch.tensor(0.5))

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        return_dict,
    ):
        del timestep, encoder_hidden_states, pooled_projections, return_dict
        return (torch.zeros_like(hidden_states) + self.base + self.delta,)


def _runtime_context(precision: str = "fp32") -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision=precision,
    )


def _adapter() -> tuple[SD3TempFlowAdapter, _FakePeftTransformer]:
    adapter = SD3TempFlowAdapter(
        checkpoint=Path("/checkpoint"),
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=1.0,
        resolution=4,
        max_sequence_length=8,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=_runtime_context(),
    )
    transformer = _FakePeftTransformer()
    adapter.transformer = transformer
    adapter.pipeline = SimpleNamespace(scheduler=object())
    adapter._pipeline_full = lambda *_args, **_kwargs: None
    adapter._pipeline_branching = lambda *_args, **_kwargs: None

    def sde_step(
        _scheduler,
        model_output,
        _timestep,
        sample,
        *,
        prev_sample,
    ):
        mean = sample + 0.1 * model_output
        std = torch.full(
            (sample.shape[0], 1, 1, 1),
            0.2,
            dtype=sample.dtype,
            device=sample.device,
        )
        log_prob = -(
            (prev_sample.detach() - mean).square()
            / (2.0 * std.square())
        ).mean(dim=(1, 2, 3))
        return prev_sample, log_prob, mean, std

    adapter._sde_step = sde_step
    return adapter, transformer


def _batch() -> RolloutBatch:
    batch_size, transitions = 2, 2
    return RolloutBatch(
        prompts=("red", "blue"),
        metadata=({}, {}),
        media=torch.zeros(batch_size, 3, 4, 4),
        latents=torch.zeros(batch_size, transitions, 2, 2, 2),
        next_latents=torch.full(
            (batch_size, transitions, 2, 2, 2),
            0.15,
        ),
        timesteps=torch.tensor([[9, 4], [9, 4]], dtype=torch.int64),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.tensor([[True, True], [True, False]]),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-1"),
        group_id=("group-0", "group-1"),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=StepContext(step=0, seed=7),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={
            "prompt_embeds": torch.zeros(batch_size, 1, 1),
            "pooled_prompt_embeds": torch.zeros(batch_size, 1),
            "negative_prompt_embeds": torch.zeros(batch_size, 1, 1),
            "negative_pooled_prompt_embeds": torch.zeros(batch_size, 1),
        },
        artifact_metadata={},
    )


def test_sd3_beta_zero_path_performs_no_disabled_adapter_forward():
    adapter, transformer = _adapter()
    transformer.train(True)
    batch = _batch()
    stats = adapter.recompute_policy_stats(
        batch,
        require_reference=False,
    )
    stats.validate_against(batch, require_reference=False)
    assert transformer.disable_calls == 0
    assert transformer.restore_calls == 0
    assert transformer.training_modes
    assert not any(transformer.training_modes)
    assert transformer.training is True
    assert stats.current_transition_mean is None
    assert stats.transition_std is None
    assert stats.reference_transition_mean is None


def test_sd3_prompt_payload_matches_policy_precision_before_latent_creation():
    adapter = SD3TempFlowAdapter(
        checkpoint=Path("/checkpoint"),
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=1.0,
        resolution=4,
        max_sequence_length=8,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=_runtime_context("bf16"),
    )
    adapter.pipeline = SimpleNamespace(
        text_encoder=object(),
        text_encoder_2=object(),
        text_encoder_3=object(),
        tokenizer=object(),
        tokenizer_2=object(),
        tokenizer_3=object(),
    )

    def encode(**kwargs):
        batch_size = len(kwargs["prompt"])
        return (
            torch.zeros(batch_size, 2, 4, dtype=torch.float32),
            torch.zeros(batch_size, 2, 4, dtype=torch.float32),
            torch.zeros(batch_size, 4, dtype=torch.float32),
            torch.zeros(batch_size, 4, dtype=torch.float32),
        )

    adapter._encode_prompt = encode

    values = adapter._prompt_payload(("red", "blue"))

    assert len(values) == 4
    assert all(value.dtype == torch.bfloat16 for value in values)
    assert all(value.device.type == "cpu" for value in values)


def test_sd3_scheduler_timesteps_preserve_fractional_values_for_recompute():
    scheduler = SimpleNamespace(
        timesteps=torch.tensor([999.0, 833.3333, 0.25], dtype=torch.float32)
    )

    values = sd3_module._scheduler_timesteps(
        scheduler,
        batch_size=2,
        expected=3,
        device=torch.device("cpu"),
    )

    assert values.dtype == torch.float32
    assert tuple(values.shape) == (2, 3)
    assert torch.equal(values[0], scheduler.timesteps)
    assert torch.equal(values[1], scheduler.timesteps)


def test_sd3_policy_dtype_guard_normalizes_reference_pipeline_latents():
    adapter = SD3TempFlowAdapter(
        checkpoint=Path("/checkpoint"),
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=1.0,
        resolution=4,
        max_sequence_length=8,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=_runtime_context("bf16"),
    )

    class Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observed_dtype = None

        def forward(self, *, hidden_states):
            self.observed_dtype = hidden_states.dtype
            return hidden_states

    transformer = Transformer()
    adapter._install_policy_dtype_guard(transformer)

    result = transformer(hidden_states=torch.ones(1, dtype=torch.float32))

    assert transformer.observed_dtype == torch.bfloat16
    assert result.dtype == torch.bfloat16
    adapter.close()
    assert not transformer._forward_pre_hooks


def test_sd3_reference_stats_use_current_lora_and_frozen_disabled_adapter():
    adapter, transformer = _adapter()
    batch = _batch()
    stats = adapter.recompute_policy_stats(
        batch,
        require_reference=True,
    )
    stats.validate_against(batch, require_reference=True)
    assert transformer.disable_calls == 1
    assert transformer.restore_calls == 1
    assert transformer.adapter_disabled is False
    assert tuple(stats.current_transition_mean.shape) == tuple(
        batch.next_latents.shape
    )
    assert tuple(stats.reference_transition_mean.shape) == tuple(
        batch.next_latents.shape
    )
    assert tuple(stats.transition_std.shape) == (2, 2, 1, 1, 1)
    assert stats.current_transition_mean.requires_grad
    assert not stats.reference_transition_mean.requires_grad
    assert stats.reference_transition_mean.grad_fn is None
    assert not stats.transition_std.requires_grad
    assert stats.transition_std.grad_fn is None

    active = batch.transition_mask.reshape(2, 2, 1, 1, 1)
    delta = torch.where(
        active,
        stats.current_transition_mean - stats.reference_transition_mean,
        0.0,
    )
    std = torch.where(active, stats.transition_std, 1.0)
    reference_kl = (delta.square() / (2.0 * std.square())).mean()
    adapter.train_module.zero_grad(set_to_none=True)
    reference_kl.backward()
    assert transformer.delta.grad is not None
    assert bool(torch.isfinite(transformer.delta.grad))
    assert float(transformer.delta.grad) != 0.0
    assert transformer.base.grad is None
    assert tuple(name for name, _ in adapter.named_parameters()) == ("delta",)


def test_sd3_disabled_adapter_context_restores_after_reference_failure():
    adapter, transformer = _adapter()
    batch = _batch()
    calls = 0
    original = adapter._sde_step

    def fail_on_reference(*args, **kwargs):
        nonlocal calls
        calls += 1
        if transformer.adapter_disabled:
            raise RuntimeError("reference forward failed")
        return original(*args, **kwargs)

    adapter._sde_step = fail_on_reference
    with pytest.raises(RuntimeError, match="reference forward failed"):
        adapter.recompute_policy_stats(batch, require_reference=True)
    assert calls == batch.transition_count + 1
    assert transformer.adapter_disabled is False
    assert transformer.restore_calls == 1


def test_sd3_reference_requires_peft_disable_context():
    adapter, _transformer = _adapter()
    adapter.transformer = _NoDisableTransformer()
    with pytest.raises(RunError, match=r"disable_adapter\(\)"):
        adapter.recompute_policy_stats(_batch(), require_reference=True)
