from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _adapter(tmp_path, transformer, *, use_lora: bool):
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    repo_root = tmp_path / "TempFlow-GRPO-main"
    repo_root.mkdir(exist_ok=True)
    adapter = SD3TempFlowAdapter(
        {
            "repo_root": str(repo_root),
            "device": "cpu",
            "dtype": "float32",
            "use_lora": use_lora,
            "extra": {"defer_load": True},
        }
    )
    adapter.pipeline = SimpleNamespace()
    adapter.transformer = transformer
    return adapter


def _install_fake_peft_loader(monkeypatch, *, set_state=None):
    import torch

    calls = []

    def load_peft_weights(model_id, device=None):
        calls.append(("load", model_id, device))
        return torch.load(
            str(Path(model_id) / "adapter_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

    def default_set_state(model, state, adapter_name="default"):
        calls.append(("set", model, adapter_name))
        with torch.no_grad():
            model.lora_weight.copy_(state["lora_weight"])
        return SimpleNamespace(missing_keys=["base_weight"], unexpected_keys=[])

    peft_module = ModuleType("peft")
    peft_module.__path__ = []
    utils_module = ModuleType("peft.utils")
    utils_module.__path__ = []
    save_load_module = ModuleType("peft.utils.save_and_load")
    save_load_module.load_peft_weights = load_peft_weights
    save_load_module.set_peft_model_state_dict = set_state or default_set_state
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "peft.utils", utils_module)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", save_load_module)
    return calls


def _fake_peft_transformer(torch):
    class FakePeftTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = torch.nn.Parameter(
                torch.tensor(7.0),
                requires_grad=False,
            )
            self.lora_weight = torch.nn.Parameter(torch.tensor(2.5))
            self.peft_config = {"default": {"r": 1}}
            self.safe_serialization = None

        def save_pretrained(self, path, *, safe_serialization=True):
            path.mkdir(parents=True, exist_ok=True)
            self.safe_serialization = safe_serialization
            (path / "adapter_config.json").write_text(
                '{"r": 1}',
                encoding="utf-8",
            )
            torch.save(
                {"lora_weight": self.lora_weight.detach().clone()},
                path / "adapter_model.bin",
            )

        def state_dict(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("LoRA-only save must not request the full state_dict")

    return FakePeftTransformer()


def test_sd3_lora_checkpoint_is_adapter_only_and_restores_in_place(
    tmp_path,
    monkeypatch,
):
    import torch

    transformer = _fake_peft_transformer(torch)
    adapter = _adapter(tmp_path, transformer, use_lora=True)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    parameter = transformer.lora_weight
    parameter_id = id(parameter)
    optimizer_parameter = optimizer.param_groups[0]["params"][0]
    checkpoint = tmp_path / "checkpoint_000001"

    adapter.save_pretrained(str(checkpoint))

    assert transformer.safe_serialization is True
    assert not (checkpoint / "transformer_state.pt").exists()
    assert (checkpoint / "transformer" / "adapter_config.json").is_file()
    assert (checkpoint / "transformer" / "adapter_model.bin").is_file()
    metadata = json.loads(
        (checkpoint / "adapter_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {
        "adapter": "tempflow_sd3_legacy",
        "adapter_config": "transformer/adapter_config.json",
        "adapter_name": "default",
        "format_version": 1,
        "save_kind": "peft_adapter",
        "use_lora": True,
        "weights": "transformer/adapter_model.bin",
    }

    with torch.no_grad():
        transformer.lora_weight.fill_(-4.0)
        transformer.base_weight.fill_(99.0)
    calls = _install_fake_peft_loader(monkeypatch)
    adapter.load_checkpoint(str(checkpoint))

    assert transformer.lora_weight.item() == pytest.approx(2.5)
    assert transformer.base_weight.item() == pytest.approx(99.0)
    assert id(transformer.lora_weight) == parameter_id
    assert optimizer.param_groups[0]["params"][0] is optimizer_parameter
    assert optimizer_parameter is transformer.lora_weight
    assert calls[0] == ("load", str(checkpoint / "transformer"), "cpu")
    assert calls[1][0] == "set"
    assert calls[1][1] is transformer
    assert calls[1][2] == "default"


def test_sd3_lora_checkpoint_loads_root_level_peft_adapter(tmp_path, monkeypatch):
    import torch

    transformer = _fake_peft_transformer(torch)
    adapter = _adapter(tmp_path, transformer, use_lora=True)
    checkpoint = tmp_path / "legacy_lora"
    transformer.save_pretrained(checkpoint, safe_serialization=True)
    with torch.no_grad():
        transformer.lora_weight.fill_(-8.0)
    calls = _install_fake_peft_loader(monkeypatch)

    adapter.load_checkpoint(str(checkpoint))

    assert transformer.lora_weight.item() == pytest.approx(2.5)
    assert calls[0] == ("load", str(checkpoint), "cpu")


def test_sd3_load_checkpoint_accepts_legacy_full_state(tmp_path):
    import torch

    transformer = torch.nn.Linear(2, 1, bias=False)
    adapter = _adapter(tmp_path, transformer, use_lora=True)
    checkpoint = tmp_path / "legacy_full"
    checkpoint.mkdir()
    adapter_dir = checkpoint / "transformer"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.bin").write_bytes(b"legacy adapter copy")
    expected = transformer.weight.detach().clone()
    torch.save(transformer.state_dict(), checkpoint / "transformer_state.pt")
    with torch.no_grad():
        transformer.weight.fill_(-12.0)

    adapter.load_checkpoint(str(checkpoint))

    assert torch.equal(transformer.weight, expected)


def test_sd3_non_lora_checkpoint_keeps_full_state_round_trip(tmp_path):
    import torch

    transformer = torch.nn.Linear(2, 1, bias=False)
    adapter = _adapter(tmp_path, transformer, use_lora=False)
    checkpoint = tmp_path / "checkpoint_000001"
    expected = transformer.weight.detach().clone()

    adapter.save_pretrained(str(checkpoint))
    with torch.no_grad():
        transformer.weight.fill_(-3.0)
    adapter.load_checkpoint(str(checkpoint))

    assert torch.equal(transformer.weight, expected)
    assert (checkpoint / "transformer_state.pt").is_file()
    metadata = json.loads(
        (checkpoint / "adapter_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["save_kind"] == "full_transformer_state"
    assert metadata["use_lora"] is False


def test_sd3_lora_save_rejects_non_peft_transformer(tmp_path):
    import torch

    adapter = _adapter(tmp_path, torch.nn.Linear(1, 1), use_lora=True)

    with pytest.raises(RuntimeError, match="requires a loaded PEFT transformer"):
        adapter.save_pretrained(str(tmp_path / "checkpoint_000001"))


def test_sd3_checkpoint_errors_are_explicit(tmp_path, monkeypatch):
    import torch

    transformer = _fake_peft_transformer(torch)
    adapter = _adapter(tmp_path, transformer, use_lora=True)
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(RuntimeError, match="Missing SD3 checkpoint weights"):
        adapter.load_checkpoint(str(missing))

    checkpoint = tmp_path / "bad_adapter"
    transformer.save_pretrained(checkpoint, safe_serialization=True)

    def fail_set_state(model, state, adapter_name="default"):
        del model, state, adapter_name
        raise ValueError("shape mismatch")

    _install_fake_peft_loader(monkeypatch, set_state=fail_set_state)
    with pytest.raises(
        RuntimeError,
        match="Failed to load SD3 PEFT adapter checkpoint.*shape mismatch",
    ):
        adapter.load_checkpoint(str(checkpoint))


def test_sd3_non_lora_load_requires_full_state(tmp_path):
    import torch

    adapter = _adapter(tmp_path, torch.nn.Linear(1, 1), use_lora=False)
    checkpoint = tmp_path / "missing_full"
    checkpoint.mkdir()

    with pytest.raises(RuntimeError, match="use_lora=False"):
        adapter.load_checkpoint(str(checkpoint))
