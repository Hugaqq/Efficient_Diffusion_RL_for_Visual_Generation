"""Explicit, immutable approved experiment identities (not a second scheduler)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FlashPreset:
    name: str
    optimizer_steps: int
    collection_rows: int
    rows_per_commit: int
    prompt_count: int
    sampling_steps: int
    evaluation_steps: tuple[int, ...]
    save_videos: bool = False
    noise_seeds: tuple[int, ...] = ()


APPROVED140 = FlashPreset("approved140", 140, 384, 192, 300, 50, (0, 36, 76, 116))
SMALL40 = FlashPreset("small40", 40, 32, 16, 16, 20, (0, 20, 40), True,
                      tuple(range(42000, 42016)))


def flash_preset(name):
    if name == "approved140":
        return APPROVED140
    if name == "small40":
        return SMALL40
    raise ValueError(f"unknown Flash experiment preset: {name}")
