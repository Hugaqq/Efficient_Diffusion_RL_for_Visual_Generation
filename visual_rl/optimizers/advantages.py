"""Single-owner reward-to-advantage conversion for VisualRL optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

EPSILON = 1e-6


def normalize_rewards(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + float(epsilon))


def calculate_zero_std_ratio(
    prompts: list[str],
    rewards: np.ndarray,
    group_ids: Sequence[Hashable] | None = None,
    epsilon: float = EPSILON,
) -> float:
    groups = _group_indices(group_ids if group_ids is not None else prompts)
    if not groups:
        return 0.0
    values = np.asarray(rewards, dtype=np.float64)
    return float(
        sum(
            float(np.std(values[indices])) <= float(epsilon)
            for indices in groups.values()
        )
        / len(groups)
    )


@dataclass
class AdvantageResult:
    advantages: Any
    metrics: dict[str, float]


class AdvantageFunction:
    """Convert one validated reward batch into per-sample advantages."""

    def __call__(self, batch: Any, rewards: Any) -> AdvantageResult:
        raise NotImplementedError


class AdvantageComputer(AdvantageFunction):
    """Normalize rewards exactly once, immediately before policy optimization."""

    def __init__(
        self,
        reward_weights: dict[str, float],
        per_prompt: bool = True,
        weight_advantages: bool = False,
        use_global_std: bool = False,
        max_group_std: bool = False,
        mode: str = "grpo",
        epsilon: float = EPSILON,
        output_dtype: str = "float32",
    ):
        if use_global_std and max_group_std:
            raise ValueError("use_global_std and max_group_std are mutually exclusive")
        self.reward_weights = dict(reward_weights)
        self.per_prompt = bool(per_prompt)
        self.weight_advantages = bool(weight_advantages)
        self.use_global_std = bool(use_global_std)
        self.max_group_std = bool(max_group_std)
        self.mode = str(mode)
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("advantage epsilon must be positive")
        self.output_dtype = str(output_dtype)
        if self.output_dtype not in {"float32", "float64"}:
            raise ValueError("advantage output_dtype must be 'float32' or 'float64'")

    def __call__(self, batch: Any, rewards: Any) -> AdvantageResult:
        """Apply the canonical batch/reward contract while retaining ``compute``.

        Formal rollout batches carry an explicit ``StepContext`` and use their
        declared ``group_id``. Older batches preserve the historical grouping
        by ``parent_prompt_index`` (falling back to the prompt itself).
        """

        if getattr(batch, "context", None) is not None:
            group_ids = list(batch.group_id)
        else:
            group_ids = [
                item.get("parent_prompt_index", prompt)
                for prompt, item in zip(
                    batch.prompts,
                    batch.metadata,
                    strict=True,
                )
            ]
        return self.compute(
            batch.prompts,
            rewards.raw,
            rewards.weighted_total,
            group_ids=group_ids,
        )

    def compute(
        self,
        prompts: list[str],
        raw_rewards: dict[str, Any],
        weighted_total: Any,
        group_ids: Sequence[Hashable] | None = None,
    ) -> AdvantageResult:
        import torch

        total_np = _to_numpy_vector(weighted_total, len(prompts), "weighted_total")
        grouping_keys = group_ids if group_ids is not None else prompts
        if len(grouping_keys) != len(prompts):
            raise ValueError("group_ids must have one entry per sample")
        groups = _group_indices(grouping_keys)
        self._validate_groups(groups, len(prompts))
        metrics = {
            "zero_std_ratio": calculate_zero_std_ratio(
                prompts,
                total_np,
                group_ids=group_ids,
                epsilon=self.epsilon,
            ),
            "group_size": float(np.mean([len(indices) for indices in groups.values()])),
            "trained_prompt_num": float(len(groups)),
        }

        if self.weight_advantages:
            advantage_np = np.zeros(len(prompts), dtype=np.float64)
            for name, weight in self.reward_weights.items():
                if name not in raw_rewards:
                    raise KeyError(f"Missing raw reward {name!r} for weighted advantages")
                values = _to_numpy_vector(raw_rewards[name], len(prompts), name)
                advantage_np += self._shape(values, groups) * float(weight)
        else:
            advantage_np = self._shape(total_np, groups)

        return AdvantageResult(
            advantages=torch.as_tensor(
                advantage_np,
                dtype={"float32": torch.float32, "float64": torch.float64}[
                    self.output_dtype
                ],
            ),
            metrics=metrics,
        )

    def _shape(
        self,
        values: np.ndarray,
        groups: dict[Hashable, np.ndarray],
    ) -> np.ndarray:
        if not self.per_prompt:
            return _shape_one_group(
                values,
                self.mode,
                values.std() + self.epsilon,
            )

        result = np.zeros_like(values, dtype=np.float64)
        if self.use_global_std:
            shared_std = float(values.std()) + self.epsilon
        elif self.max_group_std:
            shared_std = (
                max(float(values[indices].std()) for indices in groups.values())
                + self.epsilon
            )
        else:
            shared_std = None

        for indices in groups.values():
            group_values = values[indices]
            denominator = shared_std or float(group_values.std()) + self.epsilon
            result[indices] = _shape_one_group(group_values, self.mode, denominator)
        return result

    def _validate_groups(
        self,
        groups: dict[Hashable, np.ndarray],
        batch_size: int,
    ) -> None:
        if self.mode not in {"grpo", "dpo"}:
            return
        if self.per_prompt:
            singleton_groups = [group for group, indices in groups.items() if len(indices) < 2]
            if singleton_groups:
                raise ValueError(
                    "GRPO/DPO requires at least two samples for every group; "
                    f"singleton groups: {singleton_groups}"
                )
        elif batch_size < 2:
            raise ValueError("GRPO/DPO requires at least two samples to normalize advantages")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("AdvantageComputer has no persistent state")


def _group_indices(keys: Sequence[Hashable]) -> dict[Hashable, np.ndarray]:
    groups: dict[Hashable, list[int]] = {}
    for index, key in enumerate(keys):
        if not isinstance(key, Hashable):
            raise ValueError(f"group_ids[{index}] must be hashable")
        groups.setdefault(key, []).append(index)
    return {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in groups.items()
    }


def _shape_one_group(values: np.ndarray, mode: str, denominator: float) -> np.ndarray:
    if mode == "grpo":
        return (values - values.mean()) / denominator
    if mode == "rwr":
        return values.astype(np.float64, copy=True)
    if mode == "sft":
        return (values == values.max()).astype(np.float64)
    if mode == "dpo":
        shaped = np.zeros_like(values, dtype=np.float64)
        shaped[int(np.argmax(values))] = 1.0
        shaped[int(np.argmin(values))] = -1.0
        return shaped
    raise ValueError(f"Unknown advantage mode: {mode}")


def _to_numpy_vector(value: Any, expected: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array
