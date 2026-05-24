def _rollout_batch(old_log_probs, timesteps, metadata=None, model_metadata=None):
    import torch

    from visual_rl.core.types import RolloutBatch

    batch_size, num_steps = old_log_probs.shape
    return RolloutBatch(
        prompts=["a red cube", "a blue sphere"],
        metadata=metadata or [{} for _ in range(batch_size)],
        media=torch.zeros(batch_size, 3, 4, 4),
        latents=torch.zeros(batch_size, num_steps, 3, 4, 4),
        next_latents=torch.zeros(batch_size, num_steps, 3, 4, 4),
        timesteps=timesteps,
        old_log_probs=old_log_probs,
        model_metadata=model_metadata or {},
    )


def test_tiny_algorithms_differ_on_same_prompt_reward_and_logprob_values():
    import torch

    from visual_rl.algorithms.flash_grpo import FlashGRPOAlgorithm
    from visual_rl.algorithms.grpo import GRPOAlgorithm
    from visual_rl.algorithms.tempflow_grpo import TempFlowGRPOAlgorithm

    prompts_rewards = torch.tensor([2.0, -1.0])
    full_timesteps = torch.arange(4).repeat(2, 1)
    full_old_log_probs = torch.zeros(2, 4)
    full_new_log_probs = torch.zeros(2, 4)
    selected_indices = torch.tensor([1, 3])

    grpo_batch = _rollout_batch(full_old_log_probs, full_timesteps)
    grpo_loss, grpo_metrics = GRPOAlgorithm(clip_range=0.2).compute_loss(
        grpo_batch,
        prompts_rewards,
        full_new_log_probs,
    )

    flash_old_log_probs = full_old_log_probs.gather(1, selected_indices[:, None])
    flash_new_log_probs = full_new_log_probs.gather(1, selected_indices[:, None])
    flash_batch = _rollout_batch(
        flash_old_log_probs,
        full_timesteps.gather(1, selected_indices[:, None]),
        metadata=[
            {"parent_prompt_index": 0, "sample_index": 0, "selected_timestep": 1},
            {"parent_prompt_index": 1, "sample_index": 0, "selected_timestep": 3},
        ],
        model_metadata={
            "num_steps": 4,
            "selected_timestep_indices": selected_indices.tolist(),
            "selected_timesteps": selected_indices.tolist(),
        },
    )
    flash_loss, flash_metrics = FlashGRPOAlgorithm(
        clip_range=0.2,
        rectification={"enabled": False},
    ).compute_loss(flash_batch, prompts_rewards, flash_new_log_probs)

    tempflow_batch = _rollout_batch(
        full_old_log_probs,
        full_timesteps,
        metadata=[
            {"parent_prompt_index": 0, "sample_index": 0, "branch_timestep": 1},
            {"parent_prompt_index": 1, "sample_index": 0, "branch_timestep": 3},
        ],
        model_metadata={"branch_timestep": 1},
    )
    tempflow_loss, tempflow_metrics = TempFlowGRPOAlgorithm(
        clip_range=0.2,
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": False},
    ).compute_loss(tempflow_batch, prompts_rewards, full_new_log_probs)

    assert torch.allclose(grpo_loss, torch.tensor(-0.5))
    assert torch.allclose(grpo_metrics["approx_kl"], torch.tensor(0.0))
    assert torch.allclose(grpo_metrics["clipfrac"], torch.tensor(0.0))

    assert torch.allclose(flash_loss, torch.tensor(-0.5))
    assert torch.allclose(flash_metrics["flash_active_timestep_frac"], torch.tensor(1.0))
    assert torch.allclose(flash_metrics["flash_selected_timestep_mean"], torch.tensor(2.0))
    assert torch.allclose(flash_metrics["flash_rectification_weight_mean"], torch.tensor(1.0))

    assert torch.allclose(tempflow_loss, torch.tensor(-0.125))
    assert torch.allclose(tempflow_metrics["tempflow_active_timestep_frac"], torch.tensor(0.25))
    assert torch.allclose(tempflow_metrics["tempflow_noise_weight_mean"], torch.tensor(1.0))

    assert grpo_batch.prompts == flash_batch.prompts == tempflow_batch.prompts
    assert flash_batch.old_log_probs.tolist() == [[0.0], [0.0]]
    assert tempflow_batch.old_log_probs.tolist() == full_old_log_probs.tolist()
