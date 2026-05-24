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
