from __future__ import annotations


def test_world_r1_wan_adapter_load_populates_runtime_handles(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import torch

    from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

    repo_root = tmp_path / "World-R1-main"
    model_path = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    repo_root.mkdir()
    model_path.mkdir()
    calls = []

    class FakeTransformer(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    class FakeScheduler:
        timesteps = torch.tensor([2, 1])

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()
            self.scheduler = FakeScheduler()
            self.tokenizer = object()
            self.text_encoder = object()
            self._execution_device = torch.device("cpu")
            self.sample_called = False

        def to(self, device):
            self._execution_device = torch.device(device)
            return self

    class WanPipeline:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append({"path": path, "kwargs": kwargs})
            return FakePipeline()

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(WanPipeline=WanPipeline))

    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": str(model_path),
            "repo_root": str(repo_root),
            "device": "cpu",
            "dtype": "float32",
            "low_cpu_mem_usage": True,
        }
    ).load()

    assert adapter.pipeline is not None
    assert adapter.transformer is adapter.pipeline.transformer
    assert adapter.scheduler is adapter.pipeline.scheduler
    assert adapter.tokenizer is adapter.pipeline.tokenizer
    assert adapter.text_encoder is adapter.pipeline.text_encoder
    assert adapter.device == torch.device("cpu")
    assert adapter.dtype == torch.float32
    assert calls == [
        {
            "path": str(model_path),
            "kwargs": {
                "local_files_only": True,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
            },
        }
    ]
    assert adapter.runtime_metadata()["pipeline_class"] == "FakePipeline"
    assert adapter.runtime_metadata()["transformer_class"] == "FakeTransformer"


def test_wan_checkpoint_round_trip_when_transformer_uses_save_pretrained(tmp_path):
    import torch

    from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

    repo_root = tmp_path / "World-R1-main"
    repo_root.mkdir()

    class SavePretrainedTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.5))

        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")

    adapter = WorldR1WanLegacyAdapter(
        {
            "repo_root": str(repo_root),
            "device": "cpu",
        }
    )
    adapter.transformer = SavePretrainedTransformer()
    adapter.device = torch.device("cpu")
    checkpoint = tmp_path / "checkpoint_000001"

    adapter.save_pretrained(str(checkpoint))
    with torch.no_grad():
        adapter.transformer.weight.fill_(-10.0)
    adapter.load_checkpoint(str(checkpoint))

    assert adapter.transformer.weight.item() == 2.5
    assert (checkpoint / "transformer" / "config.json").exists()
    assert (checkpoint / "transformer_state.pt").exists()
