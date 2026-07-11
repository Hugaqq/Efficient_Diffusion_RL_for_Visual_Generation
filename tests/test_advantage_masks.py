def _rollout_batch(old_log_probs, timesteps=None, metadata=None, model_metadata=None):
    import torch

    from visual_rl.core.types import RolloutBatch

    batch_size = old_log_probs.shape[0]
    if timesteps is None:
        timesteps = torch.arange(old_log_probs.shape[1]).repeat(batch_size, 1)
    if metadata is None:
        metadata = [{} for _ in range(batch_size)]
    return RolloutBatch(
        prompts=[f"prompt {index}" for index in range(batch_size)],
        metadata=metadata,
        media=torch.zeros(batch_size, 3, 4, 4),
        latents=torch.zeros(batch_size, old_log_probs.shape[1], 3, 4, 4),
        next_latents=torch.zeros(batch_size, old_log_probs.shape[1], 3, 4, 4),
        timesteps=timesteps,
        old_log_probs=old_log_probs,
        model_metadata=model_metadata or {},
    )


def test_full_trajectory_grpo_expands_per_sample_advantages_across_all_timesteps():
    import torch

    from visual_rl.optimizers.grpo import GRPOAlgorithm

    old_log_probs = torch.zeros(2, 3)
    batch = _rollout_batch(old_log_probs)
    algorithm = GRPOAlgorithm(clip_range=0.2)

    loss, metrics = algorithm.compute_loss(batch, torch.tensor([1.0, -2.0]), torch.zeros(2, 3))

    assert torch.allclose(loss, torch.tensor(0.5))
    assert torch.allclose(metrics["policy_loss"], torch.tensor(0.5))
    assert torch.allclose(metrics["approx_kl"], torch.tensor(0.0))
    assert torch.allclose(metrics["clipfrac"], torch.tensor(0.0))


def test_flash_single_step_advantages_expand_and_rectification_masks_rows():
    import torch

    from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm

    old_log_probs = torch.zeros(3, 1)
    batch = _rollout_batch(
        old_log_probs,
        timesteps=torch.tensor([[0], [2], [3]]),
        metadata=[
            {"selected_timestep": 0},
            {"selected_timestep": 2},
            {"selected_timestep": 3},
        ],
        model_metadata={
            "selected_timestep_indices": [0, 2, 3],
            "num_steps": 4,
            "flash_rectification_weights": [[1.0], [0.0], [1.0]],
        },
    )
    algorithm = FlashGRPOAlgorithm(
        clip_range=0.2,
        rectification={"enabled": True, "mode": "scheduler_formula", "normalize": False},
    )

    expanded = algorithm._expand_advantages(torch.tensor([1.5, -3.0, 0.5]), torch.zeros(3, 1))
    loss, metrics = algorithm.compute_loss(batch, torch.tensor([1.5, -3.0, 0.5]), torch.zeros(3, 1))

    assert expanded.tolist() == [[1.5], [-3.0], [0.5]]
    assert torch.allclose(loss, torch.tensor(-2.0 / 3.0))
    assert torch.allclose(metrics["flash_active_timestep_frac"], torch.tensor(2.0 / 3.0))
    assert torch.allclose(metrics["flash_rectification_weight_mean"], torch.tensor(2.0 / 3.0))
    assert torch.allclose(metrics["flash_selected_timestep_mean"], torch.tensor(5.0 / 3.0))


def test_tempflow_branch_credit_assignment_masks_expected_timesteps():
    import torch

    from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm

    old_log_probs = torch.zeros(2, 4)
    batch = _rollout_batch(
        old_log_probs,
        timesteps=torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]]),
        metadata=[{"branch_step_index": 2}, {"branch_step_index": 2}],
        model_metadata={"branch_step_index": 0},
    )
    rewards = torch.tensor([2.0, -1.0])

    branch_only = TempFlowGRPOAlgorithm(
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": False},
    )._expand_advantages(batch, rewards, old_log_probs)
    after_branch = TempFlowGRPOAlgorithm(
        credit_assignment="all_after_branch",
        noise_weighting={"enabled": False},
    )._expand_advantages(batch, rewards, old_log_probs)
    all_steps = TempFlowGRPOAlgorithm(
        credit_assignment="all",
        noise_weighting={"enabled": False},
    )._expand_advantages(batch, rewards, old_log_probs)

    assert branch_only.tolist() == [
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ]
    assert after_branch.tolist() == [
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 0.0, -1.0, -1.0],
    ]
    assert all_steps.tolist() == [
        [2.0, 2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]


def test_tempflow_active_timestep_fraction_matches_branch_mask():
    import torch

    from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm

    old_log_probs = torch.zeros(2, 4)
    batch = _rollout_batch(
        old_log_probs,
        timesteps=torch.arange(4).repeat(2, 1),
        metadata=[{"branch_step_index": 1}, {"branch_step_index": 3}],
    )
    algorithm = TempFlowGRPOAlgorithm(
        clip_range=0.2,
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": False},
    )

    loss, metrics = algorithm.compute_loss(batch, torch.tensor([2.0, -4.0]), torch.zeros(2, 4))

    assert torch.allclose(loss, torch.tensor(0.25))
    assert torch.allclose(metrics["tempflow_active_timestep_frac"], torch.tensor(0.25))
    assert torch.allclose(metrics["tempflow_noise_weight_mean"], torch.tensor(1.0))
