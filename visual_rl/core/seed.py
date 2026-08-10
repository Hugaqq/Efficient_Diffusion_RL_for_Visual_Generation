"""Canonical deterministic seed bounds shared by training domains."""

from __future__ import annotations

__all__ = ("UINT32_MAX", "seed_everything", "validate_step_seed_budget")

UINT32_MAX = 0xFFFF_FFFF


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and available CUDA generators once."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer, not bool")
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("seed must fit the canonical uint32 range")

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_step_seed_budget(seed: int, max_steps: int, world_size: int) -> None:
    """Reject the unique per-step seed schedule when it would overflow uint32."""

    for name, value in (
        ("seed", seed),
        ("max_steps", max_steps),
        ("world_size", world_size),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer, not bool")
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("seed must fit the canonical uint32 range")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    final_seed = seed + (max_steps - 1) * world_size + (world_size - 1)
    if final_seed > UINT32_MAX:
        raise ValueError(
            "seed + (max_steps - 1) * world_size + (world_size - 1) exceeds "
            f"the canonical uint32 range: {final_seed} > {UINT32_MAX}"
        )
