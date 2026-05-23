"""World-R1 reward client helpers.

v0.1 keeps real reward servers outside the local smoke path. The generic
RemotePickleRewardClient is used for server-backed rewards once URLs are set.
"""

from __future__ import annotations

from visual_rl.rewards.clients import RemotePickleRewardClient


def reward_3d_client(url: str) -> RemotePickleRewardClient:
    return RemotePickleRewardClient(name="reward_3d", url=url, payload_kind="videos", timeout=2000.0)


def reward_general_client(url: str) -> RemotePickleRewardClient:
    return RemotePickleRewardClient(name="reward_general", url=url, payload_kind="images", timeout=1000.0)

