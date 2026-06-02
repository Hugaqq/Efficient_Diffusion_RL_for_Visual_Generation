def test_v02_config_loads_typed_sections():
    from visual_rl.configs.schema import AlgorithmConfig, ModelConfig, SampleConfig, load_config

    cfg = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.sample, SampleConfig)
    assert isinstance(cfg.algorithm, AlgorithmConfig)
    assert cfg.model.name == "mock_wan"
    assert cfg.sample.same_latent is True


def test_all_presets_use_compatible_algorithm_sample_pairs():
    from pathlib import Path

    from visual_rl.configs.schema import load_config

    for path in sorted(Path("visual_rl/configs/presets").glob("*.yaml")):
        load_config(path)


def test_config_rejects_incompatible_algorithm_sample_pairs(tmp_path):
    import pytest

    from visual_rl.configs.schema import load_config

    incompatible_pairs = [
        ("grpo", "single_step", "sample.name in {'full_trajectory'}"),
        ("grpo", "branching", "sample.name in {'full_trajectory'}"),
        ("flash_grpo", "full_trajectory", "sample.name in {'single_step'}"),
        ("flash_grpo", "branching", "sample.name in {'single_step'}"),
        ("tempflow_grpo", "full_trajectory", "sample.name in {'branching'}"),
        ("tempflow_grpo", "single_step", "sample.name in {'branching'}"),
    ]

    for algorithm_name, sample_name, expected_message in incompatible_pairs:
        path = tmp_path / f"{algorithm_name}_{sample_name}.yaml"
        path.write_text(
            "\n".join(
                [
                    "run_name: invalid_pair",
                    "sample:",
                    f"  name: {sample_name}",
                    "algorithm:",
                    f"  name: {algorithm_name}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Incompatible config") as exc_info:
            load_config(path)
        assert expected_message in str(exc_info.value)


def test_validate_config_cli_accepts_shipped_preset(capsys):
    import json

    import visual_rl.cli as cli

    path = "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"

    exit_code = cli.main(["validate-config", path])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert len(payload["configs"]) == 1
    config = payload["configs"][0]
    assert config["path"] == path
    assert config["run_name"] == "world_r1_wan_v02_mock"
    assert config["model"]["name"] == "mock_wan"
    assert config["model"]["model_family"] == "wan"
    assert config["model_family"] == "wan"
    assert config["sample"]["name"] == "full_trajectory"
    assert config["algorithm"]["name"] == "grpo"
    assert config["output_dir"] == "runs/world_r1_wan_v02_mock"
    assert config["rewards"]["names"] == ["mock"]
    assert config["rewards"]["clients"]["mock"] == "mock"


def test_validate_config_cli_rejects_invalid_algorithm_sample_pair(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    path = tmp_path / "invalid_pair.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: invalid_pair",
                "model:",
                "  name: mock_wan",
                "sample:",
                "  name: full_trajectory",
                "algorithm:",
                "  name: flash_grpo",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-config", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert payload["configs"] == []
    assert payload["errors"][0]["path"] == str(path)
    assert "Incompatible config" in payload["errors"][0]["message"]


def test_validate_config_cli_rejects_missing_reward_client_alias(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    path = tmp_path / "missing_reward_client.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: missing_reward_client",
                "model:",
                "  name: mock_wan",
                "sample:",
                "  name: full_trajectory",
                "algorithm:",
                "  name: grpo",
                "rewards:",
                "  weights:",
                "    prompt_color: 1.0",
                "  clients:",
                "    mock:",
                "      name: mock",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-config", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert payload["configs"] == []
    assert payload["errors"][0]["path"] == str(path)
    assert "Missing reward client configuration" in payload["errors"][0]["message"]


def test_rollout_probe_mock_wan_returns_valid_shapes(capsys):
    import json

    import visual_rl.cli as cli

    path = "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"

    exit_code = cli.main(["rollout-probe", path, "--seed", "123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["config_path"] == path
    assert payload["run_name"] == "world_r1_wan_v02_mock"
    assert payload["adapter"] == "mock_wan"
    assert payload["adapter_key"] == "mock_wan"
    assert payload["sample"]["name"] == "full_trajectory"
    assert payload["rollout"]["name"] == "full_trajectory"
    assert payload["input_prompt_count"] == 2
    assert payload["prompt_count"] == 2
    assert payload["media_shape"] == [2, 4, 3, 16, 16]
    assert payload["latents_shape"] == [2, 2, 4, 2, 2, 2]
    assert payload["next_latents_shape"] == [2, 2, 4, 2, 2, 2]
    assert payload["timesteps_shape"] == [2, 2]
    assert payload["old_log_probs_shape"] == [2, 2]
    assert payload["model_metadata"]["adapter"] == "mock_wan"
    assert payload["seed"] == 123
    assert payload["strict"] is True


def test_rollout_probe_flash_tiny_validates_single_step_contract(capsys):
    import json

    import visual_rl.cli as cli

    path = "visual_rl/configs/presets/flash_tiny_single_step.yaml"

    exit_code = cli.main(["rollout-probe", path, "--batch-size", "1", "--num-steps", "4", "--seed", "77"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["adapter"] == "tiny_diffusion"
    assert payload["sample"]["name"] == "single_step"
    assert payload["rollout"]["name"] == "single_step"
    assert payload["input_prompt_count"] == 1
    assert payload["prompt_count"] == 4
    assert payload["media_shape"] == [4, 3, 16, 16]
    assert payload["latents_shape"] == [4, 1, 3, 16, 16]
    assert payload["next_latents_shape"] == [4, 1, 3, 16, 16]
    assert payload["timesteps_shape"] == [4, 1]
    assert payload["old_log_probs_shape"] == [4, 1]
    assert payload["model_metadata"]["rollout"] == "single_step"
    assert payload["model_metadata"]["samples_per_prompt"] == 4
    assert payload["model_metadata"]["parent_prompt_indices"] == [0, 0, 0, 0]
    assert payload["seed"] == 77


def test_rollout_probe_missing_config_path_returns_structured_json(capsys):
    import json

    import visual_rl.cli as cli

    path = "visual_rl/configs/presets/does_not_exist.yaml"

    exit_code = cli.main(["rollout-probe", path])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert payload["config_path"] == path
    assert payload["errors"]
    assert "Traceback" not in captured.out


def test_reward_probe_mock_wan_routes_mock_reward_without_cache(capsys):
    import json

    import visual_rl.cli as cli

    path = "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml"

    exit_code = cli.main(["reward-probe", path, "--seed", "123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["config_path"] == path
    assert payload["run_name"] == "world_r1_wan_v02_mock"
    assert payload["prompt_count"] == 2
    assert payload["media_shape"] == [2, 3, 16, 16]
    assert payload["media_height"] == 16
    assert payload["media_width"] == 16
    assert payload["reward_names"] == ["mock"]
    assert "mock" in payload["raw"]
    assert payload["raw"]["mock"]["shape"] == [2]
    assert payload["weighted"]["mock"]["shape"] == [2]
    assert payload["weighted_total"]["shape"] == [2]
    assert payload["normalized_total"]["shape"] == [2]
    assert payload["valid_mask"] == [True, True]
    assert payload["metadata"]["mock"]["mode"] == "prompt_media"
    assert payload["normalize"] == "none"
    assert payload["fail_policy"] == "invalid"
    assert payload["seed"] == 123


def test_reward_probe_flash_tiny_routes_prompt_color_reward(capsys):
    import json

    import visual_rl.cli as cli

    path = "visual_rl/configs/presets/flash_tiny_single_step.yaml"

    exit_code = cli.main(["reward-probe", path, "--batch-size", "2", "--seed", "77"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["config_path"] == path
    assert payload["run_name"] == "flash_tiny_single_step"
    assert payload["prompt_count"] == 2
    assert payload["media_shape"] == [2, 3, 16, 16]
    assert payload["media_height"] == 16
    assert payload["media_width"] == 16
    assert payload["reward_names"] == ["prompt_color"]
    assert payload["raw"]["prompt_color"]["shape"] == [2]
    assert payload["weighted"]["prompt_color"]["shape"] == [2]
    assert payload["raw"]["prompt_color"]["values"] == [1.0, 1.0]
    assert payload["weighted"]["prompt_color"]["values"] == [1.0, 1.0]
    assert payload["valid_mask"] == [True, True]
    assert payload["metadata"]["prompt_color"]["targets"] == ["red", "blue"]
    assert payload["fail_policy"] == "raise"
    assert payload["seed"] == 77


def test_reward_probe_rejects_mismatched_reward_client_alias_without_traceback(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    path = tmp_path / "missing_reward_client.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: missing_reward_client",
                "model:",
                "  name: mock_wan",
                "sample:",
                "  name: full_trajectory",
                "algorithm:",
                "  name: grpo",
                "dataset:",
                "  prompts:",
                "    - a red square",
                "rewards:",
                "  weights:",
                "    prompt_color: 1.0",
                "  clients:",
                "    mock:",
                "      name: mock",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["reward-probe", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert payload["config_path"] == str(path)
    assert "Missing reward client configuration" in payload["errors"][0]
    assert "Traceback" not in captured.out


def test_reward_probe_uses_real_image_preset_resolution(capsys):
    import json

    import visual_rl.cli as cli

    paths = [
        "visual_rl/configs/presets/sd3_tempflow_adapter.yaml",
    ]

    for path in paths:
        exit_code = cli.main(["reward-probe", path, "--seed", "123"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.err == ""
        payload = json.loads(captured.out)
        assert payload["valid"] is True
        assert payload["config_path"] == path
        assert payload["media_shape"] == [1, 3, 256, 256]
        assert payload["media_height"] == 256
        assert payload["media_width"] == 256


def test_reward_probe_uses_explicit_extra_height_width(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    path = tmp_path / "explicit_height_width.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: explicit_height_width",
                "model:",
                "  name: mock_wan",
                "  extra:",
                "    height: 20",
                "    width: 28",
                "sample:",
                "  name: full_trajectory",
                "  batch_size: 2",
                "algorithm:",
                "  name: grpo",
                "dataset:",
                "  prompts:",
                "    - a red square",
                "    - a blue square",
                "rewards:",
                "  weights:",
                "    prompt_color: 1.0",
                "  clients:",
                "    prompt_color:",
                "      name: prompt_color",
                "  fail_policy: raise",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["reward-probe", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["media_shape"] == [2, 3, 20, 28]
    assert payload["media_height"] == 20
    assert payload["media_width"] == 28


def test_infer_probe_image_size_accepts_top_level_model_fields():
    from types import SimpleNamespace

    from visual_rl.cli import _infer_probe_image_size

    dict_config = {"model": {"height": 12, "width": 14, "media_shape": [4, 3, 16, 16]}}
    assert _infer_probe_image_size(dict_config) == (12, 14)

    object_config = SimpleNamespace(
        model=SimpleNamespace(extra={}, resolution=30, media_shape=[4, 3, 16, 16])
    )
    assert _infer_probe_image_size(object_config) == (30, 30)


def test_reward_probe_invalid_media_size_returns_structured_json(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    path = tmp_path / "invalid_resolution.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: invalid_resolution",
                "model:",
                "  name: mock_wan",
                "  extra:",
                "    resolution: 0",
                "sample:",
                "  name: full_trajectory",
                "algorithm:",
                "  name: grpo",
                "dataset:",
                "  prompts:",
                "    - a red square",
                "rewards:",
                "  weights:",
                "    prompt_color: 1.0",
                "  clients:",
                "    prompt_color:",
                "      name: prompt_color",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["reward-probe", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert payload["config_path"] == str(path)
    assert "model.extra.resolution=0" in payload["errors"][0]
    assert "Traceback" not in captured.out
