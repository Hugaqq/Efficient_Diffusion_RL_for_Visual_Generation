"""Reward normalization helpers."""

from __future__ import annotations


def normalize_tensor(values, mode: str):
    if mode in {"none", None}:
        return values
    if mode == "per_batch":
        std = values.std(unbiased=False).clamp_min(1e-6)
        return (values - values.mean()) / std
    raise ValueError(f"Unknown reward normalization mode: {mode}")

