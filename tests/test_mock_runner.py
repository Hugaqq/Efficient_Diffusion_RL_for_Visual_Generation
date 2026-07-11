def test_mock_runner(tmp_path, capsys):
    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.model_adapters.wan  # noqa: F401
    import visual_rl.feedback.clients  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.runner import ExperimentRunner

    config = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    config.paths.output_dir = str(tmp_path / "run")
    metrics = ExperimentRunner(config).run(max_steps=1)
    assert len(metrics) == 1
    progress_output = capsys.readouterr().err
    assert "train" in progress_output
    assert "loss=" in progress_output
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    for key in ["old_logprob_mean", "new_logprob_mean", "logprob_delta_abs_max", "rollout_kl_mean"]:
        assert key in metrics[0]


def test_runner_can_disable_rollout_cache(tmp_path):
    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.model_adapters.wan  # noqa: F401
    import visual_rl.feedback.clients  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.runner import ExperimentRunner

    config = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    config.paths.output_dir = str(tmp_path / "run_no_rollout_cache")
    config.runner.disable_rollout_cache = True

    metrics = ExperimentRunner(config).run(max_steps=1)

    assert len(metrics) == 1
    assert (tmp_path / "run_no_rollout_cache" / "metrics.jsonl").exists()
    assert not (tmp_path / "run_no_rollout_cache" / "rollouts").exists()
