"""World-R1 reward-server probe helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visual_rl.rewards.world_r1_rewards import reward_3d_client, reward_general_client


@dataclass
class WorldR1RewardServerProbeConfig:
    reward: str
    url: str
    timeout: float = 5.0
    retries: int = 0
    batch_size: int = 1
    frames: int = 2
    height: int = 4
    width: int = 4
    seed: int = 123
    prompt: str = "a red cube"


def _synthetic_media(config: WorldR1RewardServerProbeConfig) -> np.ndarray:
    rng = np.random.default_rng(int(config.seed))
    if config.reward == "reward_3d":
        return rng.random(
            (int(config.batch_size), int(config.frames), 3, int(config.height), int(config.width)),
            dtype=np.float32,
        )
    return rng.random((int(config.batch_size), 3, int(config.height), int(config.width)), dtype=np.float32)


def _build_client(config: WorldR1RewardServerProbeConfig):
    if config.reward == "reward_general":
        client = reward_general_client(config.url)
    elif config.reward == "reward_3d":
        client = reward_3d_client(config.url)
    else:
        raise ValueError(f"Unknown World-R1 reward probe target {config.reward!r}.")
    client.timeout = float(config.timeout)
    client.retries = int(config.retries)
    return client


def run_world_r1_reward_server_probe(config: WorldR1RewardServerProbeConfig) -> dict[str, Any]:
    for key in ("batch_size", "frames", "height", "width"):
        value = int(getattr(config, key))
        if value <= 0:
            raise ValueError(f"{key} must be positive, got {value}.")
    if float(config.timeout) <= 0:
        raise ValueError(f"timeout must be positive, got {config.timeout}.")
    if int(config.retries) < 0:
        raise ValueError(f"retries must be non-negative, got {config.retries}.")

    prompts = [config.prompt for _ in range(int(config.batch_size))]
    metadata = [{"source": "world_r1_reward_server_probe", "index": index} for index in range(len(prompts))]
    media = _synthetic_media(config)
    client = _build_client(config)
    values, reward_metadata = client.score(media, prompts, metadata)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(prompts),):
        raise ValueError(f"Reward server returned shape {values.shape}; expected shape ({len(prompts)},).")
    if not np.isfinite(values).all():
        raise ValueError("Reward server returned non-finite values.")
    return {
        "valid": True,
        "reward": config.reward,
        "url": config.url,
        "payload_kind": client.payload_kind,
        "prompt_count": len(prompts),
        "media_shape": [int(item) for item in media.shape],
        "timeout": float(config.timeout),
        "retries": int(config.retries),
        "values": values.tolist(),
        "metadata": reward_metadata,
        "side_effects": {
            "trainer_constructed": False,
            "checkpoint_written": False,
            "output_dir_written": False,
        },
    }
