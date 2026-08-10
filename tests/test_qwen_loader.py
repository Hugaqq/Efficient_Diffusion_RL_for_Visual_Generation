from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.world_r1_strict.qwen_loader import load_qwen_model_on_cuda_device


@dataclass(frozen=True)
class _FakeDevice:
    type: str
    index: int | None


class _LoadedModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.requires_grad_calls: list[bool] = []
        self.eval_calls = 0

    def to(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("the direct-device loader must not call model.to()")

    def requires_grad_(self, enabled: bool):
        self.requires_grad_calls.append(enabled)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


def test_qwen_loader_streams_to_one_explicit_cuda_device_without_post_load_to(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    loaded = _LoadedModel()

    class FakeQwenModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["class"] = cls
            captured["path"] = path
            captured["kwargs"] = kwargs
            return loaded

    device = _FakeDevice(type="cuda", index=3)
    dtype = object()
    result = load_qwen_model_on_cuda_device(
        model_class=FakeQwenModelClass,
        model_path=tmp_path,
        device=device,
        dtype=dtype,
    )

    assert result is loaded
    assert captured == {
        "class": FakeQwenModelClass,
        "path": str(tmp_path),
        "kwargs": {
            "dtype": dtype,
            "device_map": {"": device},
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "use_safetensors": True,
        },
    }
    assert loaded.requires_grad_calls == [False]
    assert loaded.eval_calls == 1
    assert loaded.config.use_cache is False


@pytest.mark.parametrize(
    "device",
    [
        _FakeDevice(type="cpu", index=None),
        _FakeDevice(type="cuda", index=None),
    ],
)
def test_qwen_loader_rejects_non_explicit_cuda_devices(device: _FakeDevice) -> None:
    class ModelClassThatMustNotLoad:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            raise AssertionError("invalid placement must fail before model loading")

    with pytest.raises(ValueError, match="explicit CUDA|logical index"):
        load_qwen_model_on_cuda_device(
            model_class=ModelClassThatMustNotLoad,
            model_path="/local/model",
            device=device,
            dtype=object(),
        )
