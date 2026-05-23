def test_mock_trainer(tmp_path):
    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.model_adapters.wan  # noqa: F401
    import visual_rl.rewards.clients  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.trainer import VisualRLTrainer

    config = load_config("visual_rl/configs/presets/world_r1_wan_v01_mock.yaml")
    config.output_dir = str(tmp_path / "run")
    metrics = VisualRLTrainer(config).train(max_steps=1)
    assert len(metrics) == 1
    assert (tmp_path / "run" / "metrics.jsonl").exists()

