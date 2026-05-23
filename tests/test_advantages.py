def test_weighted_advantages_shape_and_metrics():
    import torch

    from visual_rl.advantages import AdvantageComputer

    computer = AdvantageComputer(
        reward_weights={"a": 0.25, "b": 0.75},
        per_prompt=True,
        weight_advantages=True,
    )
    result = computer.compute(
        prompts=["p", "p", "q", "q"],
        raw_rewards={
            "a": torch.tensor([0.0, 1.0, 0.2, 0.3]),
            "b": torch.tensor([1.0, 0.0, 0.4, 0.6]),
        },
        weighted_total=torch.tensor([0.75, 0.25, 0.35, 0.525]),
    )
    assert tuple(result.advantages.shape) == (4,)
    assert "zero_std_ratio" in result.metrics
    assert result.metrics["group_size"] == 2.0
    assert result.metrics["trained_prompt_num"] == 2.0
