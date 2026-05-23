def test_world_r1_plan_smoke_defaults():
    from visual_rl.trainer.world_r1_launcher import build_world_r1_launch_plan

    plan = build_world_r1_launch_plan(
        model_path="/models/wan",
        train_visible_devices="6,7",
    )
    assert plan.num_processes == 2
    assert plan.train_num_steps == 2
    assert plan.as_env()["MODEL_PATH"] == "/models/wan"

