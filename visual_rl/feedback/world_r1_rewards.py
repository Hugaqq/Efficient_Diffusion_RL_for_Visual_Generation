"""World-R1 reward client helpers.

Real World-R1 rewards are server-backed. Keep the local contract explicit so
dry-run plans can reject malformed endpoints before launching Wan jobs.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.feedback.clients import RemotePickleRewardClient

WORLD_R1_REWARD_CLIENT_NAMES = frozenset({"reward_3d", "reward_general"})


def validate_reward_server_url(url: str, *, reward_name: str = "reward server") -> str:
    """Return a normalized HTTP(S) reward-server URL or raise ValueError."""
    normalized = str(url).strip()
    if not normalized:
        raise ValueError(f"{reward_name} URL must not be empty.")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{reward_name} URL must use http or https, got {parsed.scheme!r}.")
    if not parsed.netloc:
        raise ValueError(f"{reward_name} URL must include a host.")
    if parsed.username or parsed.password:
        raise ValueError(f"{reward_name} URL must not embed credentials.")
    return normalized


def sample_video_frame_for_reward_general(media: Any, *, frame_index: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Convert Wan video media to image media for World-R1 reward_general."""

    shape = list(getattr(media, "shape", []))
    if len(shape) != 5:
        return media, {"input_shape": shape, "output_shape": shape, "selected_frame_index": None}

    selected = shape[1] // 2 if frame_index is None else int(frame_index)
    if selected < 0 or selected >= shape[1]:
        raise ValueError(f"reward_general frame_index {selected} is outside video frame range 0..{shape[1] - 1}.")

    if shape[2] in {1, 3, 4}:
        images = media[:, selected]
    elif shape[-1] in {1, 3, 4}:
        images = media[:, selected]
    else:
        raise ValueError(
            "reward_general expected video media shaped [batch, frames, channels, height, width] "
            "or [batch, frames, height, width, channels]."
        )
    return images, {
        "input_shape": shape,
        "output_shape": list(getattr(images, "shape", [])),
        "selected_frame_index": selected,
    }


class WorldR1Reward3DClient(RemotePickleRewardClient):
    def __init__(self, url: str, timeout: float = 2000.0, retries: int = 2, **kwargs: Any):
        del kwargs
        super().__init__(
            name="reward_3d",
            url=validate_reward_server_url(url, reward_name="reward_3d"),
            payload_kind="videos",
            timeout=timeout,
            retries=retries,
        )


class WorldR1RewardGeneralClient(RemotePickleRewardClient):
    def __init__(
        self,
        url: str,
        timeout: float = 1000.0,
        retries: int = 2,
        frame_index: int | None = None,
        **kwargs: Any,
    ):
        del kwargs
        super().__init__(
            name="reward_general",
            url=validate_reward_server_url(url, reward_name="reward_general"),
            payload_kind="images",
            timeout=timeout,
            retries=retries,
        )
        self.frame_index = frame_index

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]):
        images, frame_metadata = sample_video_frame_for_reward_general(media, frame_index=self.frame_index)
        values, reward_metadata = super().score(images, prompts, metadata)
        merged_metadata = dict(reward_metadata)
        if frame_metadata["selected_frame_index"] is not None:
            merged_metadata["payload_kind"] = "images"
            merged_metadata["frame_sampling"] = frame_metadata
        return values, merged_metadata


def reward_3d_client(url: str) -> RemotePickleRewardClient:
    return WorldR1Reward3DClient(url=url)


def reward_general_client(url: str) -> RemotePickleRewardClient:
    return WorldR1RewardGeneralClient(url=url)


REWARD_CLIENTS.register("reward_3d", WorldR1Reward3DClient)
REWARD_CLIENTS.register("reward_general", WorldR1RewardGeneralClient)
