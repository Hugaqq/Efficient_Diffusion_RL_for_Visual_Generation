"""Offline-first World-R1 reward wire probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import struct
from typing import Any

import numpy as np

from visual_rl.feedback.clients import (
    DEFAULT_MAX_RESPONSE_BYTES,
    JSON_V1,
    LEGACY_PICKLE,
    redact_url,
)
from visual_rl.feedback.world_r1_rewards import (
    REFERENCE_V1,
    reward_3d_client,
    reward_general_client,
)


class _DisabledTransport:
    def post(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("HTTP is disabled for the default World-R1 dry-run probe.")


def _synthetic_dry_run_encoder(image: np.ndarray) -> bytes:
    height, width, channels = image.shape
    return b"SYNTHETIC_DRY_RUN\0" + struct.pack(">III", width, height, channels)


_synthetic_dry_run_encoder.encoding_metadata = {
    "name": "synthetic_dry_run",
    "format": "synthetic_bytes",
    "real_jpeg": False,
    "wire_compatible": False,
    "sendable": False,
    "purpose": "payload_shape_and_length_preview_only",
}
_synthetic_dry_run_encoder.cache_fingerprint = "synthetic_dry_run_v1"


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
    protocol_mode: str = REFERENCE_V1
    wire_format: str = JSON_V1
    allow_unsafe_pickle: bool = False
    trusted_hosts: tuple[str, ...] = ()
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    execute_http: bool = False
    transport: Any = None


def _synthetic_media(config: WorldR1RewardServerProbeConfig) -> np.ndarray:
    rng = np.random.default_rng(int(config.seed))
    if config.reward == "reward_3d":
        return rng.random(
            (
                int(config.batch_size),
                int(config.frames),
                3,
                int(config.height),
                int(config.width),
            ),
            dtype=np.float32,
        )
    return rng.random(
        (int(config.batch_size), 3, int(config.height), int(config.width)),
        dtype=np.float32,
    )


def _build_client(config: WorldR1RewardServerProbeConfig):
    transport = config.transport
    if transport is None and not config.execute_http:
        transport = _DisabledTransport()
    kwargs = {
        "timeout": float(config.timeout),
        "retries": int(config.retries),
        "protocol_mode": config.protocol_mode,
        "wire_format": config.wire_format,
        "allow_unsafe_pickle": config.allow_unsafe_pickle,
        "trusted_hosts": config.trusted_hosts,
        "max_response_bytes": config.max_response_bytes,
        "transport": transport,
        "jpeg_encoder": None if config.execute_http else _synthetic_dry_run_encoder,
    }
    if config.reward == "reward_general":
        return reward_general_client(config.url, media_layout="BCHW", **kwargs)
    if config.reward == "reward_3d":
        return reward_3d_client(
            config.url,
            media_layout="BFCHW",
            require_camera_trajectory=True,
            **kwargs,
        )
    raise ValueError(f"Unknown World-R1 reward probe target {config.reward!r}.")


def _probe_camera_trajectory(*, frames: int, sample_index: int) -> dict[str, str]:
    trajectory = {}
    for frame_index in range(frames):
        translation = sample_index * 0.01 + frame_index * 0.001
        trajectory[f"frame{frame_index}"] = (
            "[1 0 0 0] [0 1 0 0] [0 0 1 0] "
            f"[{translation:.6g} 0 0 1] "
        )
    _validate_probe_camera_trajectory(trajectory, expected_frames=frames)
    return trajectory


def _validate_probe_camera_trajectory(
    trajectory: dict[str, str], *, expected_frames: int
) -> None:
    expected_keys = [f"frame{index}" for index in range(expected_frames)]
    if list(trajectory) != expected_keys:
        raise ValueError("Probe camera trajectory frame keys are not contiguous.")
    for frame_key, matrix_string in trajectory.items():
        stripped = matrix_string.strip()
        if not stripped.startswith("[") or not stripped.endswith("]"):
            raise ValueError(f"Probe trajectory {frame_key} is not a matrix string.")
        columns = stripped[1:-1].split("] [")
        try:
            matrix = np.asarray(
                [[float(value) for value in column.split()] for column in columns],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Probe trajectory {frame_key} contains a non-numeric matrix."
            ) from exc
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                f"Probe trajectory {frame_key} must contain a finite 4x4 matrix."
            )


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "keys": sorted(payload),
        "prompt_count": len(payload["prompts"]),
    }
    if "images" in payload:
        result["image_byte_lengths"] = [len(item) for item in payload["images"]]
    if "videos" in payload:
        result["video_frame_byte_lengths"] = [
            [len(frame) for frame in video] for video in payload["videos"]
        ]
        result["camera_trajectories"] = payload["camera_trajectories"]
    if "protocol_version" in payload:
        result["protocol_version"] = payload["protocol_version"]
        result["sample_id"] = payload["sample_id"]
    return result


def run_world_r1_reward_server_probe(
    config: WorldR1RewardServerProbeConfig,
) -> dict[str, Any]:
    for key in ("batch_size", "frames", "height", "width"):
        value = int(getattr(config, key))
        if value <= 0:
            raise ValueError(f"{key} must be positive, got {value}.")
    if float(config.timeout) <= 0:
        raise ValueError(f"timeout must be positive, got {config.timeout}.")
    if int(config.retries) < 0:
        raise ValueError(f"retries must be non-negative, got {config.retries}.")

    prompts = [config.prompt for _ in range(int(config.batch_size))]
    metadata = []
    for index in range(len(prompts)):
        item = {
            "source": "world_r1_reward_server_probe",
            "index": index,
        }
        if config.reward == "reward_3d":
            item["camera_trajectory"] = _probe_camera_trajectory(
                frames=int(config.frames),
                sample_index=index,
            )
        metadata.append(item)
    sample_id = [f"world-r1-probe-{index:04d}" for index in range(len(prompts))]
    media = _synthetic_media(config)
    client = _build_client(config)
    payloads, encoding = client.prepare_payloads(
        media,
        prompts,
        metadata,
        sample_id=sample_id,
    )
    result = {
        "valid": True,
        "mode": "http" if config.execute_http else "dry_run",
        "http_executed": bool(config.execute_http),
        "reward": config.reward,
        "url": redact_url(config.url),
        "protocol_mode": config.protocol_mode,
        "wire_format": config.wire_format,
        "payload_kind": client.payload_kind,
        "prompt_count": len(prompts),
        "media_shape": [int(item) for item in media.shape],
        "encoding": encoding,
        "payload_bytes_sendable": bool(config.execute_http),
        "payloads": [_payload_summary(payload) for payload in payloads],
        "timeout": float(config.timeout),
        "retries": int(config.retries),
        "side_effects": {
            "trainer_constructed": False,
            "checkpoint_written": False,
            "output_dir_written": False,
        },
    }
    if config.execute_http:
        values, reward_metadata = client.score(
            media,
            prompts,
            metadata,
            sample_id=sample_id,
        )
        result["values"] = np.asarray(values, dtype=np.float32).tolist()
        result["metadata"] = reward_metadata
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", choices=["reward_general", "reward_3d"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--protocol-mode", choices=["reference_v1", "strict_v2"], default=REFERENCE_V1)
    parser.add_argument("--wire-format", choices=[JSON_V1, LEGACY_PICKLE], default=JSON_V1)
    parser.add_argument("--allow-unsafe-pickle", action="store_true")
    parser.add_argument("--trusted-host", action="append", default=[])
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    parser.add_argument("--http", action="store_true", help="Explicitly send the prepared HTTP request.")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", default="a red cube")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run_world_r1_reward_server_probe(
        WorldR1RewardServerProbeConfig(
            reward=args.reward,
            url=args.url,
            timeout=args.timeout,
            retries=args.retries,
            batch_size=args.batch_size,
            frames=args.frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
            prompt=args.prompt,
            protocol_mode=args.protocol_mode,
            wire_format=args.wire_format,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
            trusted_hosts=tuple(args.trusted_host),
            max_response_bytes=args.max_response_bytes,
            execute_http=args.http,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
