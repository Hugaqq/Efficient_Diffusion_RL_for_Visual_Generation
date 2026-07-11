"""02: 把一个 rollout batch 转成 batch 对齐的 reward。"""

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    prompts: list[str]
    media: torch.Tensor


@dataclass
class RewardBatch:
    raw: dict[str, torch.Tensor]
    weighted_total: torch.Tensor
    valid_mask: torch.Tensor


class ConstantFeedback:
    def __init__(self, value: float, weight: float):
        self.value = value
        self.weight = weight

    def score(self, batch: RolloutBatch) -> RewardBatch:
        # FILL_ME 1: 创建长度等于 prompt 数量的 raw tensor。
        # FILL_ME 2: 计算 weighted_total。
        # FILL_ME 3: 返回全部为 True 的 valid_mask。
        raise NotImplementedError("FILL_ME")


def check() -> None:
    batch = RolloutBatch(["a", "b", "c"], torch.zeros(3, 3, 2, 2))
    rewards = ConstantFeedback(0.5, 2.0).score(batch)
    assert rewards.raw["constant"].tolist() == [0.5, 0.5, 0.5]
    assert rewards.weighted_total.tolist() == [1.0, 1.0, 1.0]
    assert rewards.valid_mask.tolist() == [True, True, True]
    print("02 PASS")


if __name__ == "__main__":
    check()
