def test_weighted_advantages_shape_and_metrics():
    import torch

    from visual_rl.optimizers.advantages import AdvantageComputer

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


def test_grpo_normalizes_weighted_total_exactly_once():
    import torch

    from visual_rl.optimizers.advantages import AdvantageComputer

    result = AdvantageComputer(
        reward_weights={"quality": 1.0},
        per_prompt=True,
    ).compute(
        prompts=["same", "same"],
        raw_rewards={"quality": torch.tensor([1.0, 3.0])},
        weighted_total=torch.tensor([1.0, 3.0]),
    )

    assert torch.allclose(result.advantages, torch.tensor([-1.0, 1.0]), atol=2e-6)


def test_grpo_rejects_singleton_prompt_groups():
    import pytest
    import torch

    from visual_rl.optimizers.advantages import AdvantageComputer

    with pytest.raises(ValueError, match="at least two samples"):
        AdvantageComputer(reward_weights={"quality": 1.0}).compute(
            prompts=["a", "b"],
            raw_rewards={"quality": torch.tensor([1.0, 2.0])},
            weighted_total=torch.tensor([1.0, 2.0]),
        )


def test_explicit_group_ids_keep_duplicate_prompt_parents_separate():
    import torch

    from visual_rl.optimizers.advantages import AdvantageComputer

    result = AdvantageComputer(reward_weights={"quality": 1.0}).compute(
        prompts=["same", "same", "same", "same"],
        raw_rewards={"quality": torch.tensor([0.0, 2.0, 100.0, 102.0])},
        weighted_total=torch.tensor([0.0, 2.0, 100.0, 102.0]),
        group_ids=[0, 0, 1, 1],
    )

    assert torch.allclose(
        result.advantages,
        torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        atol=2e-6,
    )
    assert result.metrics["trained_prompt_num"] == 2.0
