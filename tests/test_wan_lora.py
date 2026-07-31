"""Focused LoRA construction coverage for the final Wan adapters."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from visual_rl.core.types import RuntimeBuildContext
from visual_rl.model_adapters.wan import WanFlashAdapter


class _CheckpointingTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.ones(()))
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False


class _PeftTransformer(torch.nn.Module):
    def __init__(self, base_model: _CheckpointingTransformer) -> None:
        super().__init__()
        self.base_model = base_model
        self.lora_weight = torch.nn.Parameter(torch.zeros(()))

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.base_model.is_gradient_checkpointing


class _Pipeline(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vae = torch.nn.Linear(1, 1)
        self.text_encoder = torch.nn.Linear(1, 1)
        self.text_encoder_2 = torch.nn.Linear(1, 1)
        self.transformer = _CheckpointingTransformer()
        self.scheduler = object()


def _runtime_context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _adapter(tmp_path: Path, *, gradient_checkpointing: bool) -> WanFlashAdapter:
    return WanFlashAdapter(
        checkpoint=tmp_path / "checkpoint",
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q", "to_v"),
        gradient_checkpointing=gradient_checkpointing,
        guidance_scale=4.5,
        height=64,
        width=64,
        frames=9,
        max_sequence_length=32,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=_runtime_context(),
    )


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
def test_wan_load_freezes_base_and_exposes_only_attached_lora(
    tmp_path,
    monkeypatch,
    gradient_checkpointing,
):
    pipeline = _Pipeline()
    observed: dict[str, object] = {}

    class _WanPipeline:
        @staticmethod
        def from_pretrained(path, **kwargs):
            observed["path"] = path
            observed["pretrained_kwargs"] = kwargs
            return pipeline

    class _LoraConfig:
        def __init__(self, **kwargs):
            observed["lora_config"] = kwargs

    def get_peft_model(module, _config):
        observed["wrapped_base"] = module
        return _PeftTransformer(module)

    diffusers = ModuleType("diffusers")
    diffusers.WanPipeline = _WanPipeline
    peft = ModuleType("peft")
    peft.LoraConfig = _LoraConfig
    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    adapter = _adapter(
        tmp_path,
        gradient_checkpointing=gradient_checkpointing,
    )
    adapter._load_base_pipeline()

    assert observed["path"] == str(tmp_path / "checkpoint")
    assert observed["pretrained_kwargs"] == {
        "torch_dtype": torch.float32,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    assert observed["lora_config"] == {
        "r": 4,
        "lora_alpha": 8,
        "init_lora_weights": "gaussian",
        "target_modules": ["to_q", "to_v"],
    }
    assert observed["wrapped_base"] is pipeline.transformer.base_model
    assert adapter.pipeline is pipeline
    assert adapter.transformer is pipeline.transformer
    assert adapter.scheduler is pipeline.scheduler
    assert adapter.transformer.is_gradient_checkpointing is gradient_checkpointing
    assert all(not parameter.requires_grad for parameter in pipeline.vae.parameters())
    assert all(
        not parameter.requires_grad
        for encoder in (pipeline.text_encoder, pipeline.text_encoder_2)
        for parameter in encoder.parameters()
    )
    assert tuple(name for name, _parameter in adapter.named_parameters()) == (
        "lora_weight",
    )


def test_wan_load_rejects_peft_wrapper_without_trainable_lora(
    tmp_path,
    monkeypatch,
):
    pipeline = _Pipeline()

    diffusers = ModuleType("diffusers")
    diffusers.WanPipeline = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: pipeline
    )
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **_kwargs: object()
    peft.get_peft_model = lambda module, _config: module
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    with pytest.raises(RuntimeError, match="trainable LoRA"):
        _adapter(tmp_path, gradient_checkpointing=True)._load_base_pipeline()
