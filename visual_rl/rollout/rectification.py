"""Temporal gradient rectification utilities for Flash-GRPO."""

from __future__ import annotations

import math
from typing import Iterable


FLASH_REFERENCE_RECTIFICATION = {
    999: 7.4770,
    982: 7.0414,
    963: 6.6112,
    944: 6.1867,
    922: 5.7682,
    899: 5.3559,
    874: 4.9502,
    847: 4.5513,
    817: 4.1596,
    785: 3.7754,
}


def scheduler_rectification_weights(
    selected_indices: Iterable[int],
    num_steps: int,
    mode: str = "scheduler_formula",
    normalize: bool = True,
    timestep_values: Iterable[int | float] | None = None,
) -> list[float]:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    mode = mode.lower()
    indices = [int(index) for index in selected_indices]
    timesteps = (
        [int(float(timestep)) for timestep in timestep_values]
        if timestep_values is not None
        else None
    )
    if timesteps is not None and len(timesteps) != len(indices):
        raise ValueError("timestep_values must have one entry per selected index")

    raw: list[float] = []
    for row, index in enumerate(indices):
        position = max(0, min(num_steps - 1, int(index)))
        if mode in {"none", "disabled"}:
            value = 1.0
        elif mode == "scheduler_formula":
            value = math.sqrt(max(1.0, float(num_steps - position)) / float(num_steps))
        elif mode == "inverse_scheduler_formula":
            value = math.sqrt(float(num_steps) / max(1.0, float(num_steps - position)))
        elif mode == "flash_reference_table":
            if timesteps is None:
                raise ValueError(
                    "flash_reference_table requires actual scheduler timestep values"
                )
            timestep = timesteps[row]
            try:
                value = FLASH_REFERENCE_RECTIFICATION[timestep]
            except KeyError as exc:
                supported = ", ".join(str(item) for item in FLASH_REFERENCE_RECTIFICATION)
                raise ValueError(
                    f"Unsupported Flash reference timestep {timestep}; expected one of {supported}"
                ) from exc
        else:
            raise ValueError(f"Unknown Flash rectification mode: {mode}")
        raw.append(float(value))

    if not raw or not normalize:
        return raw
    mean = sum(raw) / len(raw)
    if mean <= 0:
        return [1.0 for _ in raw]
    return [value / mean for value in raw]
