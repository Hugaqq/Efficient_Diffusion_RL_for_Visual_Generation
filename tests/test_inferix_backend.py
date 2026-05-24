from __future__ import annotations


def test_inferix_preview_plan_builds_reference_command():
    from visual_rl.eval.inferix_backend import InferixEvalBackend

    payload = InferixEvalBackend().generate_preview(
        checkpoint_path="/models/wan-inferix",
        output_dir="runs/inferix_preview",
        prompt="a red cube rolling across a table",
        num_output_frames=17,
        seed=7,
    )

    assert payload["task"] == "preview"
    assert payload["repo_dir"].endswith("reference_code/Inferix-main")
    assert payload["cwd"] == payload["repo_dir"]
    assert payload["online_rl_ready"] is False
    command = payload["command"]
    assert command[:2] == ["python", "example/self_forcing/run_self_forcing.py"]
    assert "--checkpoint_path" in command
    assert command[command.index("--checkpoint_path") + 1] == "/models/wan-inferix"
    assert "--output_folder" in command
    assert command[command.index("--output_folder") + 1] == "runs/inferix_preview"
    assert "--num_output_frames" in command
    assert command[command.index("--num_output_frames") + 1] == "17"
    assert "--memory_mode" in command
    assert "--enable_profiling" not in command


def test_inferix_profile_plan_uses_profiling_script():
    from visual_rl.integrations.inferix.profiling import build_inferix_profiling_plan

    plan = build_inferix_profiling_plan(
        checkpoint_path="/models/wan-inferix",
        output_dir="runs/inferix_profile",
        prompt="a blue cube",
        num_samples=2,
    )

    payload = plan.to_dict()
    command = payload["command"]
    assert payload["task"] == "profile"
    assert command[:2] == ["python", "example/profiling/self_forcing_profiling.py"]
    assert "--enable_profiling" in command
    assert "--profiling_config" in command
    assert "--profiling_output_dir" in command
    assert command[command.index("--num_samples") + 1] == "2"


def test_inferix_long_video_plan_validates_inputs():
    import pytest

    from visual_rl.eval.inferix_backend import InferixEvalBackend, build_inferix_eval_plan

    payload = InferixEvalBackend().run_long_video_eval(
        checkpoint_path="/models/wan-inferix",
        output_dir="runs/inferix_long",
        prompt="a long video prompt",
        num_output_frames=81,
        no_decode=True,
    )
    assert payload["task"] == "long_video_eval"
    assert payload["num_output_frames"] == 81
    assert payload["no_decode"] is True

    with pytest.raises(ValueError, match="Unknown Inferix eval task"):
        build_inferix_eval_plan(
            checkpoint_path="/models/wan-inferix",
            output_dir="runs/inferix",
            prompt="prompt",
            task="online_rl",
        )

    with pytest.raises(NotImplementedError, match="not wired"):
        InferixEvalBackend().generate_preview(
            execute=True,
            checkpoint_path="/models/wan-inferix",
            output_dir="runs/inferix",
            prompt="prompt",
        )
