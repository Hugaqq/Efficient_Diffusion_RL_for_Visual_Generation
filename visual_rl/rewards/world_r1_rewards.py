"""World-R1 reward client helpers.

Real World-R1 rewards are server-backed. Keep the local contract explicit so
dry-run plans can reject malformed endpoints before launching Wan jobs.
"""

from __future__ import annotations

from urllib.parse import urlparse

from visual_rl.rewards.clients import RemotePickleRewardClient

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


def reward_3d_client(url: str) -> RemotePickleRewardClient:
    url = validate_reward_server_url(url, reward_name="reward_3d")
    return RemotePickleRewardClient(name="reward_3d", url=url, payload_kind="videos", timeout=2000.0)


def reward_general_client(url: str) -> RemotePickleRewardClient:
    url = validate_reward_server_url(url, reward_name="reward_general")
    return RemotePickleRewardClient(name="reward_general", url=url, payload_kind="images", timeout=1000.0)
