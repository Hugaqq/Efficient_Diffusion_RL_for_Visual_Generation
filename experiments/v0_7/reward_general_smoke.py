"""Bounded real-model smoke for the bundled World-R1 general reward origin.

This operator diagnostic sends one deterministic JPEG through the strict_v2
service and prints a canonical request/response receipt.  It is deliberately
separate from the training entry point.
"""

from __future__ import annotations

import base64
import json
import math
import os
from io import BytesIO

import requests
from PIL import Image, ImageDraw

from visual_rl.core.protocols.world_r1 import SCORE_ROUTE, validate_score_response


def _image() -> str:
    image = Image.new("RGB", (192, 192), color=(16, 24, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 54, 112, 142), fill=(214, 50, 42))
    draw.ellipse((104, 72, 166, 134), fill=(42, 104, 224))
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return base64.b64encode(output.getvalue()).decode("ascii")


def main() -> None:
    revision = os.environ["WORLD_R1_SERVER_REVISION"]
    origin = os.environ.get(
        "WORLD_R1_REWARD_GENERAL_ORIGIN", "http://127.0.0.1:8090"
    )
    payload = {
        "protocol_version": "strict_v2",
        "server_revision": revision,
        "sample_id": ["reward-general-smoke-0"],
        "prompts": ["a red cube beside a blue sphere on a dark background"],
        "images": [_image()],
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
        raise RuntimeError("general reward smoke received an invalid score")
    print(
        json.dumps(
            {"request": payload, "response": body},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
