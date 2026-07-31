"""Deterministic local image rewards used by builtin SD3 experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import ClassVar

import numpy as np

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RewardVector,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.feedback.base import RewardClient

__all__ = [
    "PromptColorGuardedRewardClient",
    "PromptColorMarginRewardClient",
    "PromptColorRewardClient",
]

COLOR_TO_INDEX = {"red": 0, "green": 1, "blue": 2}
_COLOR_KEYS = frozenset(COLOR_TO_INDEX)


@dataclass(frozen=True)
class PromptColorRewardClient(RewardClient):
    """Bounded target-channel margin for image contract tests."""

    default_color: str
    name: ClassVar[str] = "prompt_color"

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        del context
        default_color = _resolve_default_color(raw, component=cls.name)
        return FrozenMapping({"default_color": default_color})

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> PromptColorRewardClient:
        del context
        return cls(
            default_color=_resolve_default_color(resolved, component=cls.name)
        )

    def score(
        self,
        batch: RolloutBatch,
        context: StepContext,
    ) -> RewardVector:
        images = _image_batch(batch, context)
        scores: list[float] = []
        records: list[dict[str, object]] = []
        for index, prompt in enumerate(batch.prompts):
            color = _target_color(
                prompt,
                batch.metadata[index],
                default=self.default_color,
            )
            target = COLOR_TO_INDEX[color]
            channel_means = images[index].reshape(3, -1).mean(axis=1)
            target_score = channel_means[target]
            distractor_score = np.delete(channel_means, target).mean()
            raw_margin = float(target_score - distractor_score)
            scores.append(float(np.clip(0.5 + raw_margin, 0.0, 1.0)))
            records.append({"target_color": color, "raw_margin": raw_margin})
        return _reward_vector(
            batch,
            scores,
            shared={"score_kind": "bounded_target_channel_margin"},
            records=records,
        )

    def close(self) -> None:
        """Pure client: no owned resource."""


@dataclass(frozen=True)
class PromptColorMarginRewardClient(RewardClient):
    """Unclipped target-channel margin for branch-relative optimization."""

    default_color: str
    name: ClassVar[str] = "prompt_color_margin"

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        del context
        default_color = _resolve_default_color(raw, component=cls.name)
        return FrozenMapping({"default_color": default_color})

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> PromptColorMarginRewardClient:
        del context
        return cls(
            default_color=_resolve_default_color(resolved, component=cls.name)
        )

    def score(
        self,
        batch: RolloutBatch,
        context: StepContext,
    ) -> RewardVector:
        images = _image_batch(batch, context)
        scores: list[float] = []
        records: list[dict[str, object]] = []
        for index, prompt in enumerate(batch.prompts):
            color = _target_color(
                prompt,
                batch.metadata[index],
                default=self.default_color,
            )
            target = COLOR_TO_INDEX[color]
            channel_means = images[index].reshape(3, -1).mean(axis=1)
            margin = float(
                channel_means[target] - np.delete(channel_means, target).mean()
            )
            scores.append(margin)
            records.append({"target_color": color, "raw_margin": margin})
        return _reward_vector(
            batch,
            scores,
            shared={"score_kind": "unclipped_target_channel_margin"},
            records=records,
        )

    def close(self) -> None:
        """Pure client: no owned resource."""


_GUARDED_KEYS = frozenset(
    {
        "default_color",
        "margin_clip",
        "saturation_max",
        "luminance_min",
        "luminance_max",
        "spatial_std_min",
        "spatial_std_max",
        "saturation_penalty_weight",
        "luminance_penalty_weight",
        "spatial_penalty_weight",
    }
)


@dataclass(frozen=True)
class PromptColorGuardedRewardClient(RewardClient):
    """Color margin with bounded pixel-level anti-collapse penalties."""

    default_color: str
    margin_clip: float
    saturation_max: float
    luminance_min: float
    luminance_max: float
    spatial_std_min: float
    spatial_std_max: float
    saturation_penalty_weight: float
    luminance_penalty_weight: float
    spatial_penalty_weight: float
    name: ClassVar[str] = "prompt_color_guarded"

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        del context
        return FrozenMapping(_resolve_guarded_params(raw))

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> PromptColorGuardedRewardClient:
        del context
        params = _resolve_guarded_params(resolved)
        return cls(**params)

    def score(
        self,
        batch: RolloutBatch,
        context: StepContext,
    ) -> RewardVector:
        images = np.clip(_image_batch(batch, context), 0.0, 1.0)
        scores: list[float] = []
        records: list[dict[str, object]] = []
        for index, prompt in enumerate(batch.prompts):
            color = _target_color(
                prompt,
                batch.metadata[index],
                default=self.default_color,
            )
            target = COLOR_TO_INDEX[color]
            image = images[index]
            channel_means = image.reshape(3, -1).mean(axis=1)
            raw_margin = float(
                channel_means[target] - np.delete(channel_means, target).mean()
            )
            bounded_margin = float(
                np.clip(raw_margin, -self.margin_clip, self.margin_clip)
            )
            maximum = image.max(axis=0)
            minimum = image.min(axis=0)
            saturation_ratio = np.zeros_like(maximum)
            np.divide(
                maximum - minimum,
                maximum,
                out=saturation_ratio,
                where=maximum > 1e-6,
            )
            saturation = float(np.mean(saturation_ratio))
            luminance = float(
                np.mean(
                    0.2126 * image[0]
                    + 0.7152 * image[1]
                    + 0.0722 * image[2]
                )
            )
            spatial_std = float(np.std(image, axis=(1, 2)).mean())
            saturation_penalty = max(0.0, saturation - self.saturation_max)
            luminance_penalty = max(
                0.0,
                self.luminance_min - luminance,
                luminance - self.luminance_max,
            )
            spatial_penalty = max(
                0.0,
                self.spatial_std_min - spatial_std,
                spatial_std - self.spatial_std_max,
            )
            score = (
                bounded_margin
                - self.saturation_penalty_weight * saturation_penalty
                - self.luminance_penalty_weight * luminance_penalty
                - self.spatial_penalty_weight * spatial_penalty
            )
            scores.append(float(score))
            records.append(
                {
                    "target_color": color,
                    "raw_margin": raw_margin,
                    "bounded_margin": bounded_margin,
                    "saturation_mean": saturation,
                    "luminance_mean": luminance,
                    "spatial_std": spatial_std,
                    "penalty_total": float(bounded_margin - score),
                }
            )
        return _reward_vector(
            batch,
            scores,
            shared={
                "score_kind": "bounded_color_margin_with_pixel_guardrails",
                "margin_clip": self.margin_clip,
                "saturation_max": self.saturation_max,
                "luminance_min": self.luminance_min,
                "luminance_max": self.luminance_max,
                "spatial_std_min": self.spatial_std_min,
                "spatial_std_max": self.spatial_std_max,
                "saturation_penalty_weight": self.saturation_penalty_weight,
                "luminance_penalty_weight": self.luminance_penalty_weight,
                "spatial_penalty_weight": self.spatial_penalty_weight,
            },
            records=records,
        )

    def close(self) -> None:
        """Pure client: no owned resource."""


def _resolve_default_color(
    raw: Mapping[str, object],
    *,
    component: str,
) -> str:
    _require_exact_keys(raw, {"default_color"}, component=component)
    color = raw["default_color"]
    if not isinstance(color, str) or color not in _COLOR_KEYS:
        raise ValueError(
            f"{component}.default_color must be one of {sorted(_COLOR_KEYS)}"
        )
    return color


def _resolve_guarded_params(raw: Mapping[str, object]) -> dict[str, object]:
    _require_exact_keys(raw, set(_GUARDED_KEYS), component="prompt_color_guarded")
    default_color = raw["default_color"]
    if not isinstance(default_color, str) or default_color not in _COLOR_KEYS:
        raise ValueError(
            "prompt_color_guarded.default_color must be one of "
            f"{sorted(_COLOR_KEYS)}"
        )
    values = {
        key: _finite_number(raw[key], field=f"prompt_color_guarded.{key}")
        for key in _GUARDED_KEYS
        if key != "default_color"
    }
    for key in (
        "margin_clip",
        "saturation_penalty_weight",
        "luminance_penalty_weight",
        "spatial_penalty_weight",
    ):
        if values[key] <= 0:
            raise ValueError(f"prompt_color_guarded.{key} must be positive")
    for key in (
        "saturation_max",
        "luminance_min",
        "luminance_max",
        "spatial_std_min",
        "spatial_std_max",
    ):
        if not 0.0 <= values[key] <= 1.0:
            raise ValueError(f"prompt_color_guarded.{key} must be in [0, 1]")
    if values["luminance_min"] >= values["luminance_max"]:
        raise ValueError(
            "prompt_color_guarded luminance_min must be less than luminance_max"
        )
    if values["spatial_std_min"] >= values["spatial_std_max"]:
        raise ValueError(
            "prompt_color_guarded spatial_std_min must be less than spatial_std_max"
        )
    return {"default_color": default_color, **values}


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _require_exact_keys(
    raw: Mapping[str, object],
    expected: set[str],
    *,
    component: str,
) -> None:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{component} params must be a mapping")
    actual = set(raw)
    if actual != expected:
        raise ValueError(
            f"{component} params must contain exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _image_batch(batch: RolloutBatch, context: StepContext) -> np.ndarray:
    if batch.context is not context:
        raise ValueError("batch.context must be the identical StepContext")
    if batch.media_layout != "BCHW":
        raise ValueError("prompt color rewards require media_layout='BCHW'")
    media = batch.media
    try:
        import torch

        if isinstance(media, torch.Tensor):
            media = media.detach().to(device="cpu", dtype=torch.float32).numpy()
    except ModuleNotFoundError:  # pragma: no cover - RolloutBatch requires torch
        pass
    images = np.asarray(media, dtype=np.float32)
    if (
        images.ndim != 4
        or images.shape[0] != batch.batch_size
        or images.shape[1] != 3
    ):
        raise ValueError("prompt color rewards require media shape [B, 3, H, W]")
    if not np.isfinite(images).all():
        raise ValueError("prompt color media must be finite")
    return images


def _target_color(
    prompt: str,
    metadata: Mapping[str, object],
    *,
    default: str,
) -> str:
    declared = metadata.get("target_color")
    if isinstance(declared, str) and declared in _COLOR_KEYS:
        return declared
    tokens = set(re.findall(r"[a-z]+", prompt.lower()))
    matches = [color for color in COLOR_TO_INDEX if color in tokens]
    return matches[0] if len(matches) == 1 else default


def _reward_vector(
    batch: RolloutBatch,
    scores: list[float],
    *,
    shared: Mapping[str, object],
    records: list[dict[str, object]],
) -> RewardVector:
    import torch

    values = torch.tensor(scores, dtype=torch.float32).detach().contiguous()
    return RewardVector(
        sample_id=batch.sample_id,
        values=values,
        shared_metadata=shared,
        sample_metadata=tuple(records),
    )
