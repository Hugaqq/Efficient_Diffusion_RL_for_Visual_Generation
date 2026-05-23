def test_reward_router_raw_weighted_and_cache(tmp_path):
    import torch

    import visual_rl.rewards.clients  # noqa: F401
    from visual_rl.rewards.router import RewardRouter

    router = RewardRouter(
        {
            "weights": {"mock": 2.0},
            "clients": {"mock": {"name": "mock", "version": "test", "mode": "prompt_media"}},
            "normalize": "none",
        },
        cache_dir=tmp_path / "reward_cache",
    )
    rewards = router.score(
        media=torch.ones(2, 3, 4, 4),
        prompts=["a prompt", "a prompt"],
        metadata=[{}, {}],
    )
    assert "mock" in rewards.raw
    assert torch.allclose(rewards.weighted["mock"], rewards.raw["mock"] * 2.0)
    assert rewards.valid_mask.all()
    assert list((tmp_path / "reward_cache").glob("*.json"))

