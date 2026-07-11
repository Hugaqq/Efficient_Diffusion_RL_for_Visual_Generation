def test_wan_runtime_plan_builds_local_runtime_plan(tmp_path):
    from visual_rl.configs.schema import load_config
    from scripts.wan_runtime_plan import WanRuntimePlanner

    cfg = load_config("scripts/configs/wan_runtime_plan.yaml")
    cfg.paths.output_dir = str(tmp_path / "wan_plan")
    world_r1_root = tmp_path / "World-R1-main"
    world_r1_root.mkdir()
    cfg.model.extra["world_r1_root"] = str(world_r1_root)

    plan = WanRuntimePlanner(cfg).build_runtime_plan()

    assert plan.backend == "wan"
    assert plan.readiness["world_r1_root_exists"] is True
    assert plan.readiness["model_path_set"] is False
    assert "model.model_path is empty" in plan.warnings[0]


def test_wan_runtime_plan_reports_world_r1_reward_server_status(tmp_path):
    from visual_rl.configs.schema import load_config
    from scripts.wan_runtime_plan import WanRuntimePlanner

    cfg = load_config("scripts/configs/wan_runtime_plan.yaml")
    cfg.paths.output_dir = str(tmp_path / "wan_plan")
    world_r1_root = tmp_path / "World-R1-main"
    world_r1_root.mkdir()
    cfg.model.extra["world_r1_root"] = str(world_r1_root)
    cfg.model.model_path = "/models/Wan2.1-T2V-1.3B-Diffusers"
    cfg.rewards.weights = {"reward_general": 1.0}
    cfg.rewards.clients = {
        "reward_general": {
            "name": "remote_pickle",
            "url": " http://127.0.0.1:18080/general ",
            "timeout": 0.01,
        }
    }

    plan = WanRuntimePlanner(cfg).build_runtime_plan()

    assert plan.reward_servers["required_clients"] == ["reward_general"]
    assert plan.reward_servers["urls"] == {"reward_general": "http://127.0.0.1:18080/general"}
    assert plan.reward_servers["valid"] is True
    assert plan.readiness["reward_server_required"] is True
    assert plan.readiness["reward_server_urls_valid"] is True
    assert plan.warnings == []


def test_wan_runtime_plan_reports_invalid_world_r1_reward_server_url(tmp_path):
    from visual_rl.configs.schema import load_config
    from scripts.wan_runtime_plan import WanRuntimePlanner

    cfg = load_config("scripts/configs/wan_runtime_plan.yaml")
    cfg.paths.output_dir = str(tmp_path / "wan_plan")
    world_r1_root = tmp_path / "World-R1-main"
    world_r1_root.mkdir()
    cfg.model.extra["world_r1_root"] = str(world_r1_root)
    cfg.model.model_path = "/models/Wan2.1-T2V-1.3B-Diffusers"
    cfg.rewards.weights = {"reward_3d": 1.0}
    cfg.rewards.clients = {"reward_3d": {"name": "remote_pickle", "url": "file:///tmp/reward"}}

    plan = WanRuntimePlanner(cfg).build_runtime_plan()

    assert plan.reward_servers["required_clients"] == ["reward_3d"]
    assert "reward_3d" in plan.reward_servers["invalid_urls"]
    assert plan.reward_servers["valid"] is False
    assert plan.readiness["reward_server_urls_valid"] is False
    assert any("must use http or https" in warning for warning in plan.warnings)
