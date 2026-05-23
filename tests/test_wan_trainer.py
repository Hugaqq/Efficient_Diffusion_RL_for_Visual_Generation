def test_wan_trainer_builds_local_runtime_plan(tmp_path):
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.wan_trainer import WanTrainer

    cfg = load_config("visual_rl/configs/presets/wan_runtime_v02_plan.yaml")
    cfg.output_dir = str(tmp_path / "wan_plan")
    cfg.paths.output_dir = cfg.output_dir
    genrl_root = tmp_path / "GenRL-main"
    world_r1_root = tmp_path / "World-R1-main"
    genrl_root.mkdir()
    world_r1_root.mkdir()
    cfg.legacy["genrl_root"] = str(genrl_root)
    cfg.legacy["world_r1_root"] = str(world_r1_root)

    plan = WanTrainer(cfg).build_runtime_plan()

    assert plan.trainer == "wan"
    assert plan.readiness["genrl_root_exists"] is True
    assert plan.readiness["world_r1_root_exists"] is True
    assert plan.readiness["model_path_set"] is False
    assert plan.train_timesteps == [0, 1]
    assert plan.gradient_accumulation_steps == 1
    assert plan.effective_gradient_accumulation_steps == 2
    assert "model.model_path is empty" in plan.warnings[0]
