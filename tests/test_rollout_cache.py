def test_rollout_cache_writes_stable_filenames_and_metadata(tmp_path):
    import json

    import torch

    from visual_rl.core.types import RewardBatch, RolloutBatch
    from visual_rl.rollout.cache import RolloutCache

    batch = RolloutBatch(
        prompts=["a red square", "a blue square"],
        metadata=[
            {"parent_prompt_index": 0, "sample_index": 0, "selected_timestep": 1},
            {"parent_prompt_index": 1, "sample_index": 0, "selected_timestep": 2},
        ],
        media=torch.ones(2, 3, 4, 4),
        latents=torch.zeros(2, 1, 3, 4, 4),
        next_latents=torch.ones(2, 1, 3, 4, 4),
        timesteps=torch.tensor([[1], [2]]),
        old_log_probs=torch.tensor([[-0.1], [-0.2]]),
        kl=torch.zeros(2, 1),
        branch_ids=torch.tensor([-1, 0]),
        model_metadata={
            "rollout": "single_step",
            "selected_timestep_indices": [1, 2],
            "parent_prompt_indices": [0, 1],
        },
    )
    rewards = RewardBatch(
        raw={"prompt_color": torch.tensor([0.25, 0.75])},
        weighted={"prompt_color": torch.tensor([0.25, 0.75])},
        weighted_total=torch.tensor([0.25, 0.75]),
        valid_mask=torch.tensor([True, True]),
        metadata={"clients": {"prompt_color": {"version": "v1"}}},
    )

    RolloutCache(tmp_path).save(7, batch, rewards)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "batch_000007.json",
        "batch_000007.media.pt",
        "batch_000007.pt",
    ]

    metadata = json.loads((tmp_path / "batch_000007.json").read_text(encoding="utf-8"))
    assert metadata["prompts"] == ["a red square", "a blue square"]
    assert metadata["metadata"] == batch.metadata
    assert metadata["model_metadata"] == batch.model_metadata
    assert metadata["media_path"].endswith("batch_000007.media.pt")
    assert metadata["reward_metadata"] == rewards.metadata
    assert metadata["weighted_total"] == [0.25, 0.75]

    tensor_payload = torch.load(tmp_path / "batch_000007.pt", weights_only=False)
    assert torch.equal(tensor_payload["timesteps"], torch.tensor([[1], [2]]))
    assert torch.equal(tensor_payload["branch_ids"], torch.tensor([-1, 0]))
