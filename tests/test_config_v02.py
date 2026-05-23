def test_v02_config_loads_typed_sections():
    from visual_rl.configs.schema import AlgorithmConfig, ModelConfig, SampleConfig, load_config

    cfg = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.sample, SampleConfig)
    assert isinstance(cfg.algorithm, AlgorithmConfig)
    assert cfg.model.name == "mock_wan"
    assert cfg.sample.same_latent is True

