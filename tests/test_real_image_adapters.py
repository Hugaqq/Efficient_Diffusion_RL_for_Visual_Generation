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
