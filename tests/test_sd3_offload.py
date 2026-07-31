"""Frozen-module lifecycle contracts for the SD3 recipes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.core.types import RuntimeBuildContext
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter


class _TrackedModule:
    def __init__(self) -> None:
        self.moves: list[tuple[object, torch.dtype | None]] = []

    def to(self, device, *, dtype=None):
        self.moves.append((device, dtype))
        return self


class _TrackedTrainModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.moves: list[object] = []

    def to(self, device, *args, **kwargs):
        del args, kwargs
        self.moves.append(device)
        return self


class _TrackedVAE(_TrackedTrainModule):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def to(self, device, *, dtype=None):
        self.moves.append((device, dtype))
        return self

    def decode(self, _value, *, return_dict):
        del return_dict
        raise RuntimeError("decode failed")


class _FailingMoveVAE(_TrackedVAE):
    def to(self, device, *, dtype=None):
        super().to(device, dtype=dtype)
        if torch.device(device).type == "cuda":
            raise RuntimeError("VAE move failed")
        return self


class _ReturningVAE(_TrackedVAE):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls = 0

    def decode(self, value, *, return_dict):
        del return_dict
        self.decode_calls += 1
        return (value,)


class _EchoTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

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
        self.batch_sizes.append(int(hidden_states.shape[0]))
        return (hidden_states,)


class _BatchShapeSensitiveTransformer(torch.nn.Module):
    """Stand in for low-precision kernels whose result depends on batch shape."""

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
        offset = hidden_states.new_tensor(hidden_states.shape[0] / 1000.0)
        return (hidden_states + offset,)


def _adapter(*, enabled: bool):
    context = RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cuda"),
        precision="bf16",
    )
    adapter = SD3TempFlowAdapter(
        checkpoint=Path("/checkpoint"),
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=1.0,
        resolution=64,
        max_sequence_length=8,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=context,
        offload_frozen_modules_during_update=enabled,
    )
    modules = {
        "text_encoder": _TrackedModule(),
        "text_encoder_2": _TrackedModule(),
        "text_encoder_3": _TrackedModule(),
        "vae": _TrackedVAE(),
    }
    adapter.pipeline = SimpleNamespace(**modules)
    adapter.transformer = _TrackedTrainModule()
    adapter._sde_step = lambda *_args, **_kwargs: None
    adapter._policy_active = True
    adapter._text_encoders_active = not enabled
    adapter._vae_active = not enabled
    adapter._frozen_modules_offloaded = enabled
    return adapter, modules


def test_sd3_staged_module_lifecycle_is_typed_and_idempotent(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    empty_cache_calls: list[None] = []
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(None),
    )

    adapter._activate_text_encoders_for_prompt()
    adapter._activate_text_encoders_for_prompt()
    adapter._offload_text_encoders()
    adapter._offload_text_encoders()
    adapter._activate_policy_module()
    adapter._activate_policy_module()
    adapter._activate_vae_for_decode()
    adapter._activate_vae_for_decode()
    adapter._offload_vae_after_decode()
    adapter._offload_vae_after_decode()

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        assert modules[name].moves == [
            (torch.device("cuda"), torch.bfloat16),
            ("cpu", torch.bfloat16),
        ]
    assert modules["vae"].moves == [
        (torch.device("cuda"), torch.float32),
        ("cpu", torch.float32),
    ]
    assert adapter.transformer.moves == [
        "cpu",
        torch.device("cuda"),
        "cpu",
        torch.device("cuda"),
    ]
    assert empty_cache_calls == [None, None, None, None]
    assert adapter._frozen_modules_offloaded is True
    assert adapter._policy_active is True
    assert adapter._text_encoders_active is False
    assert adapter._vae_active is False


def test_sd3_offload_disabled_preserves_existing_lifecycle(monkeypatch):
    adapter, modules = _adapter(enabled=False)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("disabled offload must not clear the CUDA cache"),
    )

    adapter._offload_frozen_modules_for_update()
    adapter._activate_text_encoders_for_prompt()
    adapter._activate_policy_module()
    adapter._activate_vae_for_decode()
    adapter._offload_vae_after_decode()

    assert all(not module.moves for module in modules.values())
    assert not adapter.transformer.moves
    assert adapter._frozen_modules_offloaded is False


def test_sd3_prompt_failure_restores_policy_and_offloads_encoders(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fail_prompt(**_kwargs):
        raise RuntimeError("prompt failed")

    adapter._encode_prompt = fail_prompt

    with pytest.raises(RuntimeError, match="prompt failed"):
        adapter._prompt_payload(("red cube",))

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        assert modules[name].moves == [
            (torch.device("cuda"), torch.bfloat16),
            ("cpu", torch.bfloat16),
        ]
    assert not modules["vae"].moves
    assert adapter.transformer.moves == ["cpu", torch.device("cuda")]
    assert adapter._frozen_modules_offloaded is True


def test_sd3_decode_failure_returns_vae_to_cpu(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="decode failed"):
        adapter._decode_sd3_latents(torch.zeros(1, 1, 2, 2))

    assert modules["vae"].moves == [
        (torch.device("cuda"), torch.float32),
        ("cpu", torch.float32),
    ]
    assert adapter.transformer.moves == ["cpu", torch.device("cuda")]
    assert adapter._policy_active is True
    assert adapter._vae_active is False
    assert adapter._frozen_modules_offloaded is True


def test_sd3_decode_sequence_uses_one_offload_phase(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    vae = _ReturningVAE()
    modules["vae"] = vae
    adapter.pipeline.vae = vae
    adapter.pipeline.image_processor = SimpleNamespace(
        postprocess=lambda value, *, output_type: (
            value if output_type == "pt" else None
        )
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    values = tuple(
        torch.full((1, 1, 2, 2), float(index))
        for index in range(3)
    )

    decoded = adapter._decode_sd3_latent_sequence(values)

    assert vae.decode_calls == len(values)
    assert vae.moves == [
        (torch.device("cuda"), torch.float32),
        ("cpu", torch.float32),
    ]
    assert adapter.transformer.moves == ["cpu", torch.device("cuda")]
    assert len(decoded) == len(values)
    for actual, expected in zip(decoded, values, strict=True):
        torch.testing.assert_close(actual, expected)
    assert adapter._policy_active is True
    assert adapter._vae_active is False
    assert adapter._frozen_modules_offloaded is True


def test_sd3_partial_vae_move_restores_policy(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    failing_vae = _FailingMoveVAE()
    modules["vae"] = failing_vae
    adapter.pipeline.vae = failing_vae
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="VAE move failed"):
        adapter._activate_vae_for_decode()

    assert failing_vae.moves == [
        (torch.device("cuda"), torch.float32),
        ("cpu", torch.float32),
    ]
    assert adapter.transformer.moves == ["cpu", torch.device("cuda")]
    assert adapter._policy_active is True
    assert adapter._vae_active is False
    assert adapter._frozen_modules_offloaded is True


def test_sd3_policy_forward_microbatch_preserves_order_and_gradients():
    adapter, _modules = _adapter(enabled=False)
    transformer = _EchoTransformer()
    adapter.transformer = transformer
    adapter.guidance_scale = 4.5
    adapter.policy_forward_microbatch_size = 2
    latent = torch.randn(5, 2, 3, 3, requires_grad=True)
    timestep = torch.arange(5)
    prompt = torch.zeros(5, 2, 4)
    pooled = torch.zeros(5, 4)

    output = adapter._predict_noise(
        latent,
        timestep,
        prompt,
        pooled,
        prompt,
        pooled,
    )
    output.sum().backward()

    assert transformer.batch_sizes == [4, 4, 2]
    torch.testing.assert_close(output, latent.detach())
    torch.testing.assert_close(latent.grad, torch.ones_like(latent))


def test_sd3_single_row_policy_microbatch_matches_shared_parent_forward_shape():
    adapter, _modules = _adapter(enabled=False)
    adapter.transformer = _BatchShapeSensitiveTransformer()
    adapter.guidance_scale = 4.5
    latent = torch.randn(1, 2, 3, 3)
    timestep = torch.tensor([500.0])
    prompt = torch.zeros(1, 2, 4)
    pooled = torch.zeros(1, 4)

    parent = adapter._predict_noise(
        latent,
        timestep,
        prompt,
        pooled,
        prompt,
        pooled,
    )
    branch_count = 6
    repeated = tuple(
        value.repeat_interleave(branch_count, dim=0)
        for value in (latent, timestep, prompt, pooled)
    )
    full_batch = adapter._predict_noise(
        repeated[0],
        repeated[1],
        repeated[2],
        repeated[3],
        repeated[2],
        repeated[3],
    )
    assert not torch.equal(full_batch, parent.repeat_interleave(branch_count, dim=0))

    adapter.policy_forward_microbatch_size = 1
    microbatched = adapter._predict_noise(
        repeated[0],
        repeated[1],
        repeated[2],
        repeated[3],
        repeated[2],
        repeated[3],
    )
    torch.testing.assert_close(
        microbatched,
        parent.repeat_interleave(branch_count, dim=0),
    )
