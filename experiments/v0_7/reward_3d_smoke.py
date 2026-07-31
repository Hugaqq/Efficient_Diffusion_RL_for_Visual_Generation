"""Bounded real-model smoke for the bundled World-R1 3D reward origin.

This is an operator diagnostic for the companion service, not a training
entry point. Configuration is supplied through environment variables so the
VisualRL package remains Python-API only.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
import os

from PIL import Image, ImageDraw
import requests

from visual_rl.world_r1_protocol import SCORE_ROUTE, validate_score_response

MAX_INPUT_FRAMES = 81


def _frame(index: int) -> str:
    image = Image.new("RGB", (128, 128), color=(12, 18, 32))
    draw = ImageDraw.Draw(image)
    offset = 12 + index * 10
    draw.rectangle(
        (offset, 34, offset + 52, 86),
        fill=(210, 48 + index * 8, 36),
    )
    draw.ellipse((76 - index * 4, 52, 108 - index * 4, 84), fill=(36, 96, 220))
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _camera(index: int) -> list[list[float]]:
    angle = math.radians(index * 5.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [cosine, 0.0, sine, index * 0.02],
        [0.0, 1.0, 0.0, 0.0],
        [-sine, 0.0, cosine, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main() -> None:
    revision = os.environ["WORLD_R1_SERVER_REVISION"]
    origin = os.environ.get("WORLD_R1_REWARD_3D_ORIGIN", "http://127.0.0.1:8089")
    frame_count = int(os.environ.get("WORLD_R1_SMOKE_FRAMES", "4"))
    if not 1 <= frame_count <= MAX_INPUT_FRAMES:
        raise ValueError(
            f"WORLD_R1_SMOKE_FRAMES must be in 1..{MAX_INPUT_FRAMES}"
        )
    payload = {
        "protocol_version": "strict_v2",
        "server_revision": revision,
        "sample_id": ["reward-3d-smoke-0"],
        "prompts": ["a red cube with a blue sphere, orbit camera to the left"],
        "videos": [[_frame(index) for index in range(frame_count)]],
        "camera_trajectories": [
            [_camera(index) for index in range(frame_count)]
        ],
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{origin}{SCORE_ROUTE}",
        data=json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        ),
        headers={"Content-Type": "application/json"},
        timeout=1830,
        allow_redirects=False,
    )
    response.raise_for_status()
    body = response.json()
    scores = validate_score_response(
        body,
        expected_sample_ids=payload["sample_id"],
        server_revision=revision,
    )
    if len(scores) != 1 or not math.isfinite(scores[0]):
        raise RuntimeError("3D reward smoke received an invalid score")
    print(body, flush=True)


if __name__ == "__main__":
    main()
