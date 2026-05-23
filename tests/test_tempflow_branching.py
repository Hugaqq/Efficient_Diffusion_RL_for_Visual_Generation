def test_branching_rollout_expands_main_and_branches():
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    import visual_rl.rewards.image_rewards  # noqa: F401
    from visual_rl.configs.schema import load_config, section_to_dict
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.rollout.full_trajectory import build_rollout_engine

    cfg = load_config("visual_rl/configs/presets/tempflow_tiny_branching.yaml")
    adapter = MODEL_ADAPTERS.get("tiny_diffusion")(section_to_dict(cfg.model))
    rollout_config = section_to_dict(cfg.sample)
    rollout_config.update(cfg.rollout)
    rollout_config["seed"] = 123
    rollout_config["epoch_tag"] = 1

    batch = build_rollout_engine(rollout_config).sample(adapter, ["a red square"], [{}])

    assert len(batch.prompts) == 4
    assert batch.branch_ids.tolist() == [-1, 0, 1, 2]
    assert batch.model_metadata["branch_timestep"] == 1
    assert all(item["branch_timestep"] == 1 for item in batch.metadata)
    assert tuple(batch.old_log_probs.shape) == (4, 4)


def test_tempflow_loss_uses_branch_timestep_only():
    import torch

    from visual_rl.algorithms.tempflow_grpo import TempFlowGRPOAlgorithm
    from visual_rl.core.types import RolloutBatch

    batch = RolloutBatch(
        prompts=["p", "p"],
        metadata=[{"branch_timestep": 2}, {"branch_timestep": 2}],
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(2, 4, 3, 4, 4),
        next_latents=torch.zeros(2, 4, 3, 4, 4),
        timesteps=torch.arange(4).repeat(2, 1),
        old_log_probs=torch.zeros(2, 4),
        branch_ids=torch.tensor([0, 1]),
        model_metadata={"branch_timestep": 2},
    )
    algorithm = TempFlowGRPOAlgorithm(credit_assignment="branch_timestep", noise_weighting={"enabled": False})
    expanded = algorithm._expand_advantages(batch, torch.tensor([1.0, -1.0]), torch.zeros(2, 4))

    assert expanded.tolist() == [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, -1.0, 0.0]]


def test_tempflow_tiny_training_smoke(tmp_path):
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    import visual_rl.rewards.image_rewards  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.trainer import VisualRLTrainer

    cfg = load_config("visual_rl/configs/presets/tempflow_tiny_branching.yaml")
    cfg.output_dir = str(tmp_path / "tempflow")
    cfg.paths.output_dir = cfg.output_dir
    metrics = VisualRLTrainer(cfg).train(max_steps=1)

    assert len(metrics) == 1
    assert "tempflow_active_timestep_frac" in metrics[0]
    assert (tmp_path / "tempflow" / "rollouts" / "batch_000000.pt").exists()
