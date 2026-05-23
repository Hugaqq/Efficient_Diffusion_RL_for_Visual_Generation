"""GenRL-style advantage computation for VisualRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPSILON = 1e-6


class PerPromptStatTracker:
    def __init__(self, use_global_std: bool = False, max_group_std: bool = False):
        self.use_global_std = use_global_std
        self.max_group_std = max_group_std
        self.stats: dict[str, np.ndarray] = {}
        self.history_prompts: set[int] = set()

    def update(self, prompts, rewards, mode: str = "grpo") -> np.ndarray:
        prompts = np.asarray(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards, dtype=np.float64)

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = prompt_rewards
            else:
                self.stats[prompt] = np.concatenate([np.asarray(self.stats[prompt]), prompt_rewards], axis=0)
            self.history_prompts.add(hash(prompt))

        max_std = None
        if self.max_group_std and unique.size:
            max_std_value = max(float(np.std(self.stats[prompt])) for prompt in unique) + EPSILON
            max_std = np.full((1,), max_std_value)

        for prompt in unique:
            prompt_mask = prompts == prompt
            prompt_rewards = rewards[prompt_mask]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.use_global_std:
                std = np.std(rewards, axis=0, keepdims=True) + EPSILON
            elif self.max_group_std:
                std = max_std
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + EPSILON

            if mode == "grpo":
                advantages[prompt_mask] = (prompt_rewards - mean) / std
            elif mode == "rwr":
                advantages[prompt_mask] = prompt_rewards
            elif mode == "sft":
                advantages[prompt_mask] = prompt_rewards == np.max(prompt_rewards)
            elif mode == "dpo":
                shaped = np.zeros_like(prompt_rewards, dtype=np.float64)
                if len(prompt_rewards) >= 2:
                    shaped[int(np.argmax(prompt_rewards))] = 1.0
                    shaped[int(np.argmin(prompt_rewards))] = -1.0
                advantages[prompt_mask] = shaped
            else:
                raise ValueError(f"Unknown advantage mode: {mode}")
        return advantages

    def get_stats(self) -> tuple[float, int]:
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0.0
        return avg_group_size, len(self.history_prompts)

    def clear(self) -> None:
        self.stats = {}


def normalize_rewards(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + EPSILON)


def calculate_zero_std_ratio(prompts: list[str], rewards: np.ndarray) -> float:
    prompt_array = np.asarray(prompts)
    unique, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    if unique.size == 0:
        return 0.0
    values = np.asarray(rewards)
    if values.ndim > 1:
        values = values.mean(axis=1)
    grouped = values[np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    groups = np.split(grouped, split_indices)
    stds = np.asarray([np.std(group) for group in groups])
    return float(np.count_nonzero(stds == 0) / len(stds))


@dataclass
class AdvantageResult:
    advantages: Any
    metrics: dict[str, float]


class AdvantageComputer:
    def __init__(
        self,
        reward_weights: dict[str, float],
        per_prompt: bool = True,
        weight_advantages: bool = False,
        use_global_std: bool = False,
        max_group_std: bool = False,
        mode: str = "grpo",
    ):
        self.reward_weights = reward_weights
        self.per_prompt = per_prompt
        self.weight_advantages = weight_advantages
        self.mode = mode
        self.total_tracker = PerPromptStatTracker(use_global_std, max_group_std)
        self.reward_trackers = {
            name: PerPromptStatTracker(use_global_std, max_group_std) for name in reward_weights
        }

    def compute(self, prompts: list[str], raw_rewards: dict[str, Any], weighted_total: Any) -> AdvantageResult:
        import torch

        total_np = weighted_total.detach().cpu().numpy() if isinstance(weighted_total, torch.Tensor) else weighted_total
        metrics = {
            "zero_std_ratio": calculate_zero_std_ratio(prompts, np.asarray(total_np)),
        }

        if self.weight_advantages:
            pieces = []
            for name, weight in self.reward_weights.items():
                values = raw_rewards[name]
                values_np = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
                if self.per_prompt:
                    adv_np = self.reward_trackers[name].update(prompts, values_np, mode=self.mode)
                else:
                    adv_np = normalize_rewards(np.asarray(values_np))
                pieces.append(torch.as_tensor(adv_np, dtype=torch.float32) * float(weight))
            advantages = sum(pieces)
        elif self.per_prompt:
            adv_np = self.total_tracker.update(prompts, total_np, mode=self.mode)
            advantages = torch.as_tensor(adv_np, dtype=torch.float32)
        else:
            advantages = torch.as_tensor(normalize_rewards(np.asarray(total_np)), dtype=torch.float32)

        if self.per_prompt:
            group_size, trained_prompt_num = self.total_tracker.get_stats()
            metrics.update({"group_size": float(group_size), "trained_prompt_num": float(trained_prompt_num)})
            self.total_tracker.clear()
            for tracker in self.reward_trackers.values():
                tracker.clear()
        return AdvantageResult(advantages=advantages, metrics=metrics)

