import pytest


def test_world_r1_plan_smoke_defaults():
    from scripts.world_r1_launcher import build_world_r1_launch_plan

    plan = build_world_r1_launch_plan(
        model_path="/models/wan",
        train_visible_devices="6,7",
    )
    assert plan.num_processes == 2
    assert plan.train_num_steps == 2
    assert plan.as_env()["MODEL_PATH"] == "/models/wan"
    assert plan.repo_dir.endswith("reference_code/World-R1-main")


def test_world_r1_plan_normalizes_devices_and_requires_model_path():
    from scripts.world_r1_launcher import build_world_r1_launch_plan

    plan = build_world_r1_launch_plan(
        model_path=" /models/wan ",
        train_visible_devices="6, 7, ",
    )

    assert plan.model_path == "/models/wan"
    assert plan.train_visible_devices == "6,7"
    assert plan.as_env()["NUM_PROCESSES"] == "2"

    with pytest.raises(ValueError, match="model_path must be set"):
        build_world_r1_launch_plan(model_path="")
