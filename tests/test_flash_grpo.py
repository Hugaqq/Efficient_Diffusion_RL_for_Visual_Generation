def test_single_step_rollout_expands_same_timestep_group():
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    import visual_rl.rewards.image_rewards  # noqa: F401
    from visual_rl.configs.schema import load_config, section_to_dict
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.rollout.full_trajectory import build_rollout_engine

    cfg = load_config("visual_rl/configs/presets/flash_tiny_single_step.yaml")
    adapter = MODEL_ADAPTERS.get("tiny_diffusion")(section_to_dict(cfg.model))
    rollout_config = section_to_dict(cfg.sample)
    rollout_config.update(cfg.rollout)
    rollout_config["seed"] = 123
    rollout_config["epoch_tag"] = 1

    batch = build_rollout_engine(rollout_config).sample(adapter, ["a red square"], [{}])

    assert len(batch.prompts) == 4
    assert tuple(batch.old_log_probs.shape) == (4, 1)
    assert tuple(batch.timesteps.shape) == (4, 1)
    assert batch.model_metadata["selected_timestep_indices"] == [1, 1, 1, 1]
    assert all(item["selected_timestep"] == 1 for item in batch.metadata)
    assert all(item["rollout_kind"] == "flash_single_step" for item in batch.metadata)


def test_flash_rectification_uses_rollout_weights():
    import torch

    from visual_rl.algorithms.flash_grpo import FlashGRPOAlgorithm
    from visual_rl.core.types import RolloutBatch

    batch = RolloutBatch(
        prompts=["p", "p"],
        metadata=[{"selected_timestep": 0}, {"selected_timestep": 3}],
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(2, 1, 3, 4, 4),
        next_latents=torch.zeros(2, 1, 3, 4, 4),
        timesteps=torch.tensor([[0], [3]]),
        old_log_probs=torch.zeros(2, 1),
        model_metadata={
            "selected_timestep_indices": [0, 3],
            "num_steps": 4,
            "flash_rectification_weights": [[1.6], [0.4]],
        },
    )
    algorithm = FlashGRPOAlgorithm(rectification={"enabled": True, "mode": "scheduler_formula"})
    weights = algorithm._rectification_weights(batch, torch.zeros(2, 1))

    assert torch.allclose(weights, torch.tensor([[1.6], [0.4]]))


def test_flash_tiny_training_smoke(tmp_path):
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    import visual_rl.rewards.image_rewards  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.trainer import VisualRLTrainer

    cfg = load_config("visual_rl/configs/presets/flash_tiny_single_step.yaml")
    cfg.output_dir = str(tmp_path / "flash")
    cfg.paths.output_dir = cfg.output_dir
    metrics = VisualRLTrainer(cfg).train(max_steps=1)

    assert len(metrics) == 1
    assert "flash_selected_timestep_mean" in metrics[0]
    assert "flash_rectification_weight_mean" in metrics[0]
    assert (tmp_path / "flash" / "rollouts" / "batch_000000.pt").exists()
