"""Exact strict_v2 + json_v1 decoder/encoder shared by both reward apps.

Pure Python and Flask-free: the decoder turns bounded request bytes into a
typed :class:`ScoreRequest` before any manager compute happens, and the
encoder builds the exact-key response mappings.  All grammar rules come from
:mod:`visual_rl.core.protocols.world_r1`; this module adds no second definitions.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from visual_rl.core.protocols.world_r1 import (
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    REWARD_GENERAL,
    SCORE_REQUEST_3D_KEYS,
    SCORE_REQUEST_COMMON_KEYS,
    SCORE_REQUEST_GENERAL_KEYS,
    WorldR1ProtocolError,
    WorldR1RevisionError,
    decode_json,
    encode_json,
    require_revision_echo,
    validate_camera_trajectory,
    validate_outputs,
    validate_reward_kind,
    validate_sample_ids,
    validate_server_revision,
)

_JSON_CONTENT_TYPES = {"application/json"}


@dataclass(frozen=True)
class ScoreRequest:
    """One fully validated strict_v2 score request."""

    reward: str
    server_revision: str
    sample_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    images: tuple[bytes, ...] | None
    videos: tuple[tuple[bytes, ...], ...] | None
    camera_trajectories: tuple[tuple[Any, ...], ...] | None


def _validate_content_type(content_type: str | None) -> None:
    if not isinstance(content_type, str):
        raise WorldR1ProtocolError("strict_v2 requests must use an application/json content type.")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in _JSON_CONTENT_TYPES:
        raise WorldR1ProtocolError(
            f"strict_v2 requests must use application/json, got {media_type!r}."
        )


def _decode_base64(value: Any, *, what: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise WorldR1ProtocolError(f"{what} must be a non-empty base64 string.")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise WorldR1ProtocolError(f"{what} is not valid base64.") from exc
    if not decoded:
        raise WorldR1ProtocolError(f"{what} must decode to non-empty bytes.")
    return bytes(decoded)


def _validate_prompts(value: Any, *, expected: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise WorldR1ProtocolError("strict_v2 prompts must be a list of strings.")
    prompts = list(value)
    if len(prompts) != expected:
        raise WorldR1ProtocolError(
            f"strict_v2 prompts length must match sample_id ({expected}), got {len(prompts)}."
        )
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise WorldR1ProtocolError("strict_v2 prompts must be non-empty strings.")
    return tuple(prompts)


def _decode_images(value: Any, *, expected: int) -> tuple[bytes, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise WorldR1ProtocolError(
            f"strict_v2 images must be a list of {expected} base64 JPEG entries."
        )
    return tuple(
        _decode_base64(item, what=f"images entry {index}")
        for index, item in enumerate(value)
    )


def _decode_videos(value: Any, *, expected: int) -> tuple[tuple[bytes, ...], ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise WorldR1ProtocolError(
            f"strict_v2 videos must be a list of {expected} base64 JPEG frame lists."
        )
    videos: list[tuple[bytes, ...]] = []
    frame_count: int | None = None
    for index, video in enumerate(value):
        if not isinstance(video, list) or not video:
            raise WorldR1ProtocolError(
                f"strict_v2 videos entry {index} must be a non-empty frame list."
            )
        frames = tuple(
            _decode_base64(frame, what=f"videos entry {index} frame {frame_index}")
            for frame_index, frame in enumerate(video)
        )
        if frame_count is None:
            frame_count = len(frames)
        elif len(frames) != frame_count:
            raise WorldR1ProtocolError(
                "strict_v2 videos must use a consistent frame count F across the batch."
            )
        videos.append(frames)
    return tuple(videos)


def _decode_camera_trajectories(
    value: Any, *, videos: tuple[tuple[bytes, ...], ...]
) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, list) or len(value) != len(videos):
        raise WorldR1ProtocolError(
            "strict_v2 camera_trajectories must align with videos one-to-one."
        )
    trajectories: list[tuple[Any, ...]] = []
    for entry, (trajectory, video) in enumerate(zip(value, videos, strict=True)):
        typed = validate_camera_trajectory(
            trajectory, entry=entry, expected_frames=len(video)
        )
        trajectories.append(tuple(tuple(tuple(row) for row in matrix) for matrix in typed))
    return tuple(trajectories)


def decode_score_request(
    body: bytes,
    *,
    content_type: str | None,
    reward: str,
    server_revision: str,
) -> ScoreRequest:
    """Decode and fully validate one strict_v2 score request.

    Raises WorldR1RevisionError (HTTP 409) before any other field check when
    the declared revision differs from the configured one, and
    WorldR1ProtocolError (HTTP 400) for every other schema violation.  No
    manager compute may happen before this function returns.
    """

    reward = validate_reward_kind(reward)
    server_revision = validate_server_revision(server_revision)
    _validate_content_type(content_type)
    payload = decode_json(body, max_bytes=MAX_REQUEST_BYTES, what="score request")

    allowed = set(SCORE_REQUEST_COMMON_KEYS) | (
        set(SCORE_REQUEST_GENERAL_KEYS) if reward == REWARD_GENERAL else set(SCORE_REQUEST_3D_KEYS)
    )
    keys = set(payload)
    if keys != allowed:
        raise WorldR1ProtocolError(
            f"strict_v2 {reward} request keys must be exactly {sorted(allowed)}, "
            f"got {sorted(keys)}."
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise WorldR1ProtocolError(
            f"strict_v2 request protocol_version must be {PROTOCOL_VERSION!r}."
        )
    require_revision_echo(server_revision, payload["server_revision"])
    sample_ids = tuple(validate_sample_ids(payload["sample_id"]))
    prompts = _validate_prompts(payload["prompts"], expected=len(sample_ids))

    images: tuple[bytes, ...] | None = None
    videos: tuple[tuple[bytes, ...], ...] | None = None
    trajectories: tuple[tuple[Any, ...], ...] | None = None
    if reward == REWARD_GENERAL:
        images = _decode_images(payload["images"], expected=len(sample_ids))
    else:
        videos = _decode_videos(payload["videos"], expected=len(sample_ids))
        trajectories = _decode_camera_trajectories(
            payload["camera_trajectories"], videos=videos
        )
    return ScoreRequest(
        reward=reward,
        server_revision=server_revision,
        sample_ids=sample_ids,
        prompts=prompts,
        images=images,
        videos=videos,
        camera_trajectories=trajectories,
    )


def encode_score_response(
    *, server_revision: str, sample_ids: tuple[str, ...], outputs: Any
) -> dict[str, Any]:
    """Build the exact-key HTTP 200 response body for one scored batch."""

    scores = validate_outputs(list(outputs), expected=len(sample_ids))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "server_revision": validate_server_revision(server_revision),
        "sample_id": list(sample_ids),
        "outputs": scores,
        "valid_mask": [True] * len(sample_ids),
    }


def encode_response_body(payload: dict[str, Any]) -> bytes:
    """Encode a response mapping as canonical json_v1 bytes."""

    return encode_json(payload)


def error_body(error: str) -> dict[str, Any]:
    """Build a stable single-key error body."""

    return {"error": str(error)}


__all__ = (
    "ScoreRequest",
    "WorldR1ProtocolError",
    "WorldR1RevisionError",
    "decode_score_request",
    "encode_response_body",
    "encode_score_response",
    "error_body",
)
