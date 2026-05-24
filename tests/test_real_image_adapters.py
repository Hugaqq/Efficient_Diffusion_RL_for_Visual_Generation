def test_real_image_adapters_register_deferred():
    import pytest

    import visual_rl.model_adapters.flux  # noqa: F401
    import visual_rl.model_adapters.qwenimage  # noqa: F401
    import visual_rl.model_adapters.sd15  # noqa: F401
    import visual_rl.model_adapters.sd3  # noqa: F401
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.model_adapters.diffusers_common import AdapterNotLoadedError

    for name in ["sd15_lora", "sd3_tempflow", "flux_tempflow", "qwenimage_tempflow"]:
        adapter = MODEL_ADAPTERS.get(name)({"name": name, "model_path": "", "extra": {"defer_load": True}})
        assert adapter.name
        with pytest.raises(AdapterNotLoadedError):
            adapter.parameters()


def test_real_image_presets_load():
    from visual_rl.configs.schema import load_config

    for path in [
        "visual_rl/configs/presets/sd15_lora_rl.yaml",
        "visual_rl/configs/presets/sd3_tempflow_adapter.yaml",
        "visual_rl/configs/presets/flux_tempflow_adapter.yaml",
        "visual_rl/configs/presets/qwenimage_tempflow_adapter.yaml",
    ]:
        cfg = load_config(path)
        assert cfg.model.model_family in {"image", "sd3", "flux", "qwenimage"}
        assert cfg.trainer["strict_rollout_validation"] is True


def test_sd15_numeric_smoke_cli_uses_explicit_model_path(monkeypatch, capsys):
    import torch

    import visual_rl.cli as cli
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.core.types import RolloutBatch

    class FakeSD15Adapter:
        name = "sd15_lora"

        def __init__(self, config):
            assert config["model_path"] == "/models/sd15"
            assert config["extra"]["resolution"] == 64
            assert config["extra"]["dtype"] == "float32"
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.weight = torch.nn.Parameter(torch.ones(2))

        def sample(self, prompts, metadata, rollout_config):
            assert prompts == ["a red square"]
            assert rollout_config["num_steps"] == 1
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=torch.zeros(1, 3, 8, 8),
                latents=torch.zeros(1, 1, 4, 8, 8),
                next_latents=torch.zeros(1, 1, 4, 8, 8),
                timesteps=torch.tensor([[0]]),
                old_log_probs=torch.zeros(1, 1),
                kl=torch.zeros(1, 1),
                seed=rollout_config["seed"],
                model_metadata={"adapter": self.name, "logprob": "ddim_surrogate"},
            )

        def recompute_log_probs(self, batch):
            return torch.zeros_like(batch.old_log_probs)

        def parameters(self):
            return [self.weight]

    monkeypatch.setattr(cli, "_register_builtin_plugins", lambda: None)
    monkeypatch.setitem(MODEL_ADAPTERS._items, "sd15_lora", FakeSD15Adapter)  # noqa: SLF001

    exit_code = cli.main(
        [
            "sd15-numeric-smoke",
            "--model-path",
            "/models/sd15",
            "--resolution",
            "64",
            "--num-steps",
            "1",
            "--dtype",
            "float32",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"model_path": "/models/sd15"' in output
    assert '"media_finite": true' in output
    assert '"max_abs_logprob_delta": 0.0' in output
