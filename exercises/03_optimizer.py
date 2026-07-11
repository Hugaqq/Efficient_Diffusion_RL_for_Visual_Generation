"""03: 一个 OptimizerPlugin 完成 advantage、loss、backward 和参数更新。"""

import torch


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        return -((actions - self.bias) ** 2)


class TinyOptimizerPlugin:
    def build_optimizer(self, policy, learning_rate):
        # FILL_ME 1: 为 policy.parameters() 创建 SGD optimizer。
        raise NotImplementedError("FILL_ME")

    def step(self, policy, actions, rewards, optimizer) -> dict[str, float]:
        # FILL_ME 2: 将 rewards 标准化为 advantage。
        # FILL_ME 3: 用 policy.log_probs(actions) 构造 policy loss。
        # FILL_ME 4: 按 zero_grad -> backward -> step 更新。
        # FILL_ME 5: 返回 loss 和 reward_mean 两个 float metric。
        raise NotImplementedError("FILL_ME")


def check() -> None:
    policy = TinyPolicy()
    plugin = TinyOptimizerPlugin()
    optimizer = plugin.build_optimizer(policy, learning_rate=0.1)
    metrics = plugin.step(
        policy,
        actions=torch.tensor([-1.0, 1.0]),
        rewards=torch.tensor([0.0, 2.0]),
        optimizer=optimizer,
    )
    assert policy.bias.item() > 0
    assert set(metrics) == {"loss", "reward_mean"}
    assert metrics["reward_mean"] == 1.0
    print("03 PASS")


if __name__ == "__main__":
    check()
