#!/usr/bin/env python3
"""Probe the real World-R1 HPSv2 reward through its HTTP client/server path."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
import traceback
from typing import Any


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_port(host: str, port: int, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"reward server exited before bind: {process.returncode}")
        with socket.socket() as sock:
            sock.settimeout(0.25)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise TimeoutError(f"reward server did not bind {host}:{port} within {timeout}s")


def build_images() -> tuple[Any, list[Any], list[str]]:
    import numpy as np
    from PIL import Image
    import torch

    size = 224
    red = np.zeros((size, size, 3), dtype=np.uint8)
    blue = np.zeros((size, size, 3), dtype=np.uint8)
    red[32:192, 32:192, 0] = 255
    blue[32:192, 32:192, 2] = 255
    arrays = [red, blue]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
    jpeg_roundtrips: list[Image.Image] = []
    for array in arrays:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG")
        buffer.seek(0)
        jpeg_roundtrips.append(Image.open(buffer).convert("RGB").copy())
    prompts = ["a vivid red square on a black background"] * 2
    return tensor, jpeg_roundtrips, prompts


def terminate_own_child(process: subprocess.Popen[bytes]) -> int | None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args()
    import requests
    import torch

    if args.host not in LOOPBACK_HOSTS:
        parser.error(
            "reward_general legacy-pickle validation is restricted to an exact "
            "loopback host (127.0.0.1 or localhost)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    result: dict[str, object] = {
        "schema_version": 1,
        "valid": False,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reference_root": str(args.reference_root.resolve()),
        "protocol": {
            "mode": "reference_v1",
            "wire_format": "legacy_pickle",
            "allow_unsafe_pickle": True,
            "trusted_hosts": [args.host],
            "network_scope": "loopback_only",
        },
    }
    log_path = args.output_dir / "server.log"
    server: subprocess.Popen[bytes] | None = None
    try:
        if socket.socket().connect_ex((args.host, args.port)) == 0:
            raise RuntimeError(f"refusing duplicate start: {args.host}:{args.port} is occupied")

        sys.path.insert(0, str(args.reference_root))
        from flow_grpo.rewards import remote_reward_general

        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.reference_root)
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["GENERAL_REWARD_PORT"] = str(args.port)
        with log_path.open("wb") as log_handle:
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(args.reference_root / "scripts" / "serve_general_reward.py"),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ],
                cwd=args.output_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            result["server_pid"] = server.pid
            wait_for_port(args.host, args.port, server, args.startup_timeout)

            images, direct_images, prompts = build_images()
            os.environ["GENERAL_REWARD_SERVER_URL"] = f"http://{args.host}:{args.port}"
            reward_fn = remote_reward_general("cuda")
            started = time.monotonic()
            http_scores, http_metadata = reward_fn(images, prompts, metadata=None)
            result["http_seconds"] = time.monotonic() - started
            result["http_scores"] = [float(value) for value in http_scores]
            result["http_metadata"] = http_metadata

            from visual_rl.feedback.world_r1_rewards import (
                WorldR1RewardGeneralClient,
            )

            infra_client = WorldR1RewardGeneralClient(
                url=f"http://{args.host}:{args.port}",
                timeout=60.0,
                retries=0,
                protocol_mode="reference_v1",
                wire_format="legacy_pickle",
                allow_unsafe_pickle=True,
                trusted_hosts=[args.host],
            )
            started = time.monotonic()
            infra_scores, infra_metadata = infra_client.score(
                images,
                prompts,
                [{}, {}],
            )
            result["infra_seconds"] = time.monotonic() - started
            result["infra_scores"] = [float(value) for value in infra_scores]
            result["infra_metadata"] = infra_metadata

            invalid = requests.post(
                f"http://{args.host}:{args.port}/",
                data=b"not-a-pickle-payload",
                timeout=30,
            )
            result["invalid_payload_status"] = invalid.status_code
            result["invalid_payload_has_traceback"] = b"Traceback" in invalid.content
            invalid_image = requests.post(
                f"http://{args.host}:{args.port}/",
                data=pickle.dumps(
                    {"images": [b"not-an-image"], "prompts": ["invalid"]}
                ),
                timeout=30,
            )
            result["internal_failure_status"] = invalid_image.status_code
            result["internal_failure_has_traceback"] = (
                b"Traceback" in invalid_image.content
            )

        result["server_returncode"] = terminate_own_child(server)
        server = None

        import hpsv2

        started = time.monotonic()
        direct_scores = hpsv2.score(
            direct_images,
            prompts[0],
            hps_version="v2.1",
        )
        result["direct_seconds"] = time.monotonic() - started
        result["direct_scores"] = [float(value) for value in direct_scores]
        max_abs = max(
            abs(float(left) - float(right))
            for left, right in zip(result["http_scores"], result["direct_scores"])
        )
        result["max_abs_http_vs_direct"] = max_abs
        max_abs_infra = max(
            abs(float(left) - float(right))
            for left, right in zip(result["infra_scores"], result["direct_scores"])
        )
        result["max_abs_infra_vs_direct"] = max_abs_infra
        result["direction_red_gt_blue"] = bool(http_scores[0] > http_scores[1])
        result["silent_fallback_detected"] = any(float(value) == 0.5 for value in http_scores)
        result["versions"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "hpsv2": getattr(hpsv2, "__version__", "unknown"),
        }
        checkpoint = Path(
            os.environ["HF_HOME"]
        ) / "hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
        result["checkpoint"] = {
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        }
        finite = all(
            math.isfinite(float(value))
            for value in list(http_scores) + list(infra_scores) + list(direct_scores)
        )
        result["valid"] = bool(
            finite
            and len(http_scores) == 2
            and len(infra_scores) == 2
            and len(direct_scores) == 2
            and max_abs <= 1e-6
            and max_abs_infra <= 1e-6
            and result["direction_red_gt_blue"]
            and not result["silent_fallback_detected"]
            and result["invalid_payload_status"] == 500
            and result["invalid_payload_has_traceback"]
            and result["internal_failure_status"] == 500
            and result["internal_failure_has_traceback"]
            and result["server_returncode"] in (-15, 0)
        )
    except Exception as error:  # preserve machine-readable failure evidence
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        if server is not None:
            result["server_returncode"] = terminate_own_child(server)
        if torch.cuda.is_available():
            result["peak_cuda_allocated"] = torch.cuda.max_memory_allocated()
            result["peak_cuda_reserved"] = torch.cuda.max_memory_reserved()
        result_path = args.output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
