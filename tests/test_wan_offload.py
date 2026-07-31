"""Frozen-module lifecycle contracts for both Wan recipes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.core.types import RuntimeBuildContext
from visual_rl.model_adapters.wan import WanFlashAdapter


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
        self.config = SimpleNamespace(
            latents_mean=[0.0],
            latents_std=[1.0],
            z_dim=1,
        )
        self.tiling_calls = 0

    def to(self, device, *, dtype=None):
        self.moves.append((device, dtype))
        return self

    def decode(self, _value, *, return_dict):
        del return_dict
        raise RuntimeError("decode failed")

    def enable_tiling(self):
        self.tiling_calls += 1


class _FailingPipeline:
    def __init__(self, modules) -> None:
        self.text_encoder = modules["text_encoder"]
        self.text_encoder_2 = modules["text_encoder_2"]
        self.vae = modules["vae"]

    def encode_prompt(self, **_kwargs):
        raise RuntimeError("prompt failed")


def _adapter(*, enabled: bool):
    context = RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cuda"),
        precision="bf16",
    )
    adapter = WanFlashAdapter(
        checkpoint=Path("/checkpoint"),
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q",),
        gradient_checkpointing=False,
        guidance_scale=5.0,
        height=64,
        width=64,
        frames=9,
        max_sequence_length=8,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=context,
        offload_frozen_modules_during_update=enabled,
    )
    modules = {
        "text_encoder": _TrackedModule(),
        "text_encoder_2": _TrackedModule(),
        "vae": _TrackedVAE(),
    }
    adapter.pipeline = SimpleNamespace(**modules)
    adapter.transformer = _TrackedTrainModule()
    adapter.scheduler = object()
    adapter._policy_active = True
    adapter._text_encoders_active = not enabled
    adapter._vae_active = not enabled
    adapter._frozen_modules_offloaded = enabled
    return adapter, modules


def test_wan_staged_module_lifecycle_is_typed_and_idempotent(monkeypatch):
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

    for name in ("text_encoder", "text_encoder_2"):
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


def test_wan_offload_disabled_preserves_existing_lifecycle(monkeypatch):
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


def test_wan_vae_tiling_is_explicit_and_idempotent():
    adapter, modules = _adapter(enabled=True)
    adapter._configure_vae_decode(modules["vae"])
    assert modules["vae"].tiling_calls == 0

    adapter.vae_tiling = True
    adapter._configure_vae_decode(modules["vae"])
    assert modules["vae"].tiling_calls == 1


def test_wan_prompt_failure_restores_policy_and_offloads_encoders(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    adapter.pipeline = _FailingPipeline(modules)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="prompt failed"):
        adapter._encode_prompt(("red cube",))

    for name in ("text_encoder", "text_encoder_2"):
        assert modules[name].moves == [
            (torch.device("cuda"), torch.bfloat16),
            ("cpu", torch.bfloat16),
        ]
    assert not modules["vae"].moves
    assert adapter.transformer.moves == ["cpu", torch.device("cuda")]
    assert adapter._frozen_modules_offloaded is True


def test_wan_decode_failure_returns_vae_to_cpu(monkeypatch):
    adapter, modules = _adapter(enabled=True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="decode failed"):
        adapter._decode_wan_latents(torch.zeros(1, 1, 1, 1, 1))

    assert modules["vae"].moves == [
        (torch.device("cuda"), torch.float32),
        ("cpu", torch.float32),
    ]
    assert adapter.transformer.moves == ["cpu", torch.device("cuda")]
    assert adapter._policy_active is True
    assert adapter._vae_active is False
    assert adapter._frozen_modules_offloaded is True
