def _tiny_branching_setup(epoch_tag=1):
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    from visual_rl.configs.schema import load_config, section_to_dict
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.rollout.full_trajectory import build_rollout_engine

    config = load_config("visual_rl/configs/presets/tempflow_tiny_branching.yaml")
    adapter = MODEL_ADAPTERS.get("tiny_diffusion")(section_to_dict(config.model))
    rollout_config = section_to_dict(config.sample)
    rollout_config.update(config.rollout)
    rollout_config.update(
        {
            "seed": 123,
            "epoch_tag": epoch_tag,
            "timestep_values": [900, 600, 300, 100],
        }
    )
    return adapter, build_rollout_engine(rollout_config)


def test_branching_rollout_has_shared_prefix_and_divergent_suffix():
    import torch

    adapter, rollout = _tiny_branching_setup(epoch_tag=1)
    batch = rollout.sample(adapter, ["a red square"], [{}])

    assert batch.prompts == ["a red square"] * 3
    assert batch.branch_ids.tolist() == [0, 1, 2]
    assert [item["branch_step_index"] for item in batch.metadata] == [1, 1, 1]
    assert [item["branch_timestep_value"] for item in batch.metadata] == [600, 600, 600]
    assert batch.model_metadata["branching_mode"] == "shared_prefix"
    assert batch.model_metadata["branch_step_index"] == 1
    assert torch.equal(batch.latents[0, 0], batch.latents[1, 0])
    assert torch.equal(batch.next_latents[0, 0], batch.next_latents[2, 0])
    assert torch.equal(batch.latents[0, 1], batch.latents[2, 1])
    assert not torch.equal(batch.next_latents[0, 1], batch.next_latents[1, 1])
    assert tuple(batch.old_log_probs.shape) == (3, 4)


def test_branching_rollout_metadata_is_grouped_by_parent_prompt():
    adapter, rollout = _tiny_branching_setup(epoch_tag=3)
    batch = rollout.sample(
        adapter,
        ["a red square", "a blue square"],
        [{"tag": "red"}, {"tag": "blue"}],
    )

    assert batch.prompts == ["a red square"] * 3 + ["a blue square"] * 3
    assert batch.branch_ids.tolist() == [0, 1, 2, 0, 1, 2]
    assert [item["parent_prompt_index"] for item in batch.metadata] == [0, 0, 0, 1, 1, 1]
    assert [item["branch_step_index"] for item in batch.metadata] == [3] * 6
    assert [item["branch_timestep_value"] for item in batch.metadata] == [100] * 6
    assert [item["tag"] for item in batch.metadata] == [
        "red",
        "red",
        "red",
        "blue",
        "blue",
        "blue",
    ]


def test_tempflow_loss_uses_branch_step_index_not_timestep_value():
    import torch

    from visual_rl.core.types import RolloutBatch
    from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm

    batch = RolloutBatch(
        prompts=["p", "p"],
        metadata=[
            {"branch_step_index": 2, "branch_timestep_value": 300},
            {"branch_step_index": 2, "branch_timestep_value": 300},
        ],
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(2, 4, 3, 4, 4),
        next_latents=torch.zeros(2, 4, 3, 4, 4),
        timesteps=torch.tensor([[900, 600, 300, 100]]).repeat(2, 1),
        old_log_probs=torch.zeros(2, 4),
        branch_ids=torch.tensor([0, 1]),
        model_metadata={"branch_step_index": 2},
    )
    algorithm = TempFlowGRPOAlgorithm(
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": False},
    )
    expanded = algorithm._expand_advantages(
        batch,
        torch.tensor([1.0, -1.0]),
        torch.zeros(2, 4),
    )

    assert expanded.tolist() == [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ]


def test_branching_rollout_rejects_adapter_without_shared_prefix_support():
    import pytest

    class PlainAdapter:
        name = "plain"

    _adapter, rollout = _tiny_branching_setup()
    with pytest.raises(NotImplementedError, match="sample_branching"):
        rollout.sample(PlainAdapter(), ["prompt"], [{}])


def test_tempflow_tiny_training_smoke(tmp_path):
    from visual_rl.configs.schema import load_config
    from visual_rl.runner import ExperimentRunner

    config = load_config("visual_rl/configs/presets/tempflow_tiny_branching.yaml")
    config.paths.output_dir = str(tmp_path / "tempflow")
    config.runner.show_progress = False
    metrics = ExperimentRunner(config).run(max_steps=1)

    assert len(metrics) == 1
    assert metrics[0]["group_size"] == 3.0
    assert "tempflow_active_timestep_frac" in metrics[0]
    assert (tmp_path / "tempflow" / "rollouts" / "batch_000000.pt").exists()
