"""Temporal gradient rectification utilities for Flash-GRPO."""

from __future__ import annotations

import math
from typing import Iterable


def scheduler_rectification_weights(
    selected_indices: Iterable[int],
    num_steps: int,
    mode: str = "scheduler_formula",
    normalize: bool = True,
) -> list[float]:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    mode = mode.lower()
    raw: list[float] = []
    for index in selected_indices:
        position = max(0, min(num_steps - 1, int(index)))
        if mode in {"none", "disabled"}:
            value = 1.0
        elif mode == "scheduler_formula":
            value = math.sqrt(max(1.0, float(num_steps - position)) / float(num_steps))
        elif mode == "inverse_scheduler_formula":
            value = math.sqrt(float(num_steps) / max(1.0, float(num_steps - position)))
        else:
            raise ValueError(f"Unknown Flash rectification mode: {mode}")
        raw.append(float(value))

    if not raw or not normalize:
        return raw
    mean = sum(raw) / len(raw)
    if mean <= 0:
        return [1.0 for _ in raw]
    return [value / mean for value in raw]
