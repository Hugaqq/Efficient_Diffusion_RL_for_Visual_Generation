"""Single owner of the World-R1 ``strict_v2`` + ``json_v1`` wire protocol.

This module is intentionally free of Torch, Requests, Flask and NumPy so the
training client, the strict companion service and experiment evidence wires can
all import it in any environment.  The ``server_revision`` public-identifier
grammar is defined exactly once here; every other component calls
:func:`validate_server_revision` and never re-compiles the pattern.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any

# The one and only definition of the public server-revision grammar.
SERVER_REVISION_PATTERN = re.compile(r"^world-r1-[0-9a-f]{12,40}$")

PROTOCOL_VERSION = "strict_v2"
WIRE_FORMAT = "json_v1"
MANAGER_CONTRACT = "world_r1_fail_closed_v1"

REWARD_GENERAL = "reward_general"
REWARD_3D = "reward_3d"
REWARD_KINDS = frozenset({REWARD_GENERAL, REWARD_3D})

HEALTH_ROUTE = "/healthz"
SCORE_ROUTE = "/v2/reward"
HEALTH_TIMEOUT_S = 5.0

# Fail-closed manager deadline (patch constant, never YAML-overridable), the
# bounded cleanup budget, the client margin and the outer WSGI/proxy timeout.
# Invariant: STRICT_MANAGER_TIMEOUT_S < MIN_CLIENT_TIMEOUT_S < WSGI_TIMEOUT_S.
STRICT_MANAGER_TIMEOUT_S = 1800.0
STRICT_CLEANUP_TIMEOUT_S = 10.0
MIN_CLIENT_TIMEOUT_S = 1830.0
WSGI_TIMEOUT_S = 1860.0
STRICT_FATAL_EXIT_CODE = 70

# Transport size limits for the json_v1 wire.
MAX_REQUEST_BYTES = 256 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024

SCORE_REQUEST_COMMON_KEYS = frozenset(
    {"protocol_version", "server_revision", "sample_id", "prompts"}
)
SCORE_REQUEST_GENERAL_KEYS = frozenset({"images"})
SCORE_REQUEST_3D_KEYS = frozenset({"videos", "camera_trajectories"})
SCORE_RESPONSE_KEYS = frozenset(
    {"protocol_version", "server_revision", "sample_id", "outputs", "valid_mask"}
)
HEALTH_KEYS = frozenset(
    {
        "status",
        "protocol_version",
        "wire_format",
        "reward",
        "server_revision",
        "manager_contract",
    }
)

ERROR_INVALID_REQUEST = "invalid_request"
ERROR_REVISION_MISMATCH = "server_revision_mismatch"
ERROR_MANAGER_NOT_READY = "manager_not_ready"
ERROR_COMPUTE_FAILED = "reward_compute_failed"


class WorldR1ProtocolError(ValueError):
    """A World-R1 wire payload violates the strict_v2 + json_v1 contract."""


class WorldR1RevisionError(WorldR1ProtocolError):
    """The peer echoed a server_revision different from the configured one."""


def validate_server_revision(value: object) -> str:
    """Return a valid public ``server_revision`` identifier unchanged.

    Raises TypeError for non-strings and WorldR1ProtocolError for strings that
    do not fully match SERVER_REVISION_PATTERN.  No URL/path/token heuristics:
    the exact grammar already excludes them.
    """

    if not isinstance(value, str):
        raise TypeError(
            "World-R1 server_revision must be a string matching "
            f"{SERVER_REVISION_PATTERN.pattern}, got {type(value).__name__}."
        )
    if SERVER_REVISION_PATTERN.fullmatch(value) is None:
        raise WorldR1ProtocolError(
            "World-R1 server_revision must match "
            f"{SERVER_REVISION_PATTERN.pattern}, got {value!r}."
        )
    return value


def validate_reward_kind(value: object) -> str:
    if not isinstance(value, str) or value not in REWARD_KINDS:
        raise WorldR1ProtocolError(
            f"World-R1 reward kind must be one of {sorted(REWARD_KINDS)}, got {value!r}."
        )
    return value


def validate_sample_ids(value: Any, *, expected: int | None = None) -> list[str]:
    """Return the strict_v2 sample_id sequence or raise WorldR1ProtocolError."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorldR1ProtocolError("strict_v2 sample_id must be a sequence of strings.")
    result = list(value)
    if expected is not None and len(result) != expected:
        raise WorldR1ProtocolError(
            f"strict_v2 sample_id length must be {expected}, got {len(result)}."
        )
    if not result:
        raise WorldR1ProtocolError("strict_v2 sample_id must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise WorldR1ProtocolError(
            "strict_v2 sample_id values must be non-empty strings."
        )
    if len(set(result)) != len(result):
        raise WorldR1ProtocolError("strict_v2 sample_id values must be unique.")
    return result


def require_sample_id_echo(
    requested: Sequence[str], echoed: Any
) -> list[str]:
    """Validate that a response echoes the request sample_id order exactly."""

    echoed_ids = validate_sample_ids(echoed, expected=len(requested))
    if echoed_ids != list(requested):
        raise WorldR1ProtocolError(
            "strict_v2 response sample_id does not match the request order."
        )
    return echoed_ids


def require_revision_echo(expected: str, echoed: Any) -> str:
    """Validate that a response echoes the configured server_revision."""

    expected = validate_server_revision(expected)
    if echoed != expected:
        raise WorldR1RevisionError(
            "strict_v2 response server_revision does not match the configured scorer."
        )
    return echoed


def _reject_json_constant(value: str) -> None:
    del value
    raise WorldR1ProtocolError("json_v1 payloads must not contain non-finite constants.")


def encode_json(payload: Mapping[str, Any]) -> bytes:
    """Encode a wire payload as canonical finite json_v1 bytes."""

    try:
        return json.dumps(
            payload, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise WorldR1ProtocolError(
            "World-R1 payload cannot be encoded as json_v1."
        ) from exc


def decode_json(data: bytes, *, max_bytes: int, what: str = "payload") -> dict[str, Any]:
    """Decode bounded utf-8 json_v1 bytes into a mapping, failing closed."""

    if not isinstance(data, (bytes, bytearray)):
        raise WorldR1ProtocolError(f"World-R1 {what} must be raw bytes.")
    validate_json_size(len(data), max_bytes=max_bytes, what=what)
    try:
        decoded = json.loads(bytes(data).decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorldR1ProtocolError(
            f"World-R1 {what} is not valid finite json_v1."
        ) from exc
    if not isinstance(decoded, dict):
        raise WorldR1ProtocolError(f"World-R1 {what} must decode to a JSON object.")
    return decoded


def validate_json_size(nbytes: int, *, max_bytes: int, what: str = "payload") -> int:
    """Enforce the json_v1 size limit before any decode work happens."""

    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise WorldR1ProtocolError(f"World-R1 {what} size must be a non-negative integer.")
    if nbytes > max_bytes:
        raise WorldR1ProtocolError(
            f"World-R1 {what} exceeds the {max_bytes}-byte json_v1 limit."
        )
    return nbytes


def _json_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldR1ProtocolError(
            f"camera trajectory values must be JSON numbers, got {type(value).__name__}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise WorldR1ProtocolError("camera trajectory values must be finite.")
    return result


def validate_camera_matrix(value: Any, *, entry: int, frame_index: int) -> list[list[float]]:
    """Return a typed row-major 4x4 camera matrix of finite JSON numbers.

    The strict_v2 wire carries canonical row-major ``list[4][4]`` numeric
    matrices; the legacy ``frameN`` mapping and matrix-string forms are
    rejected here.
    """

    where = f"camera_trajectories entry {entry} frame {frame_index}"
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorldR1ProtocolError(
            f"{where} must be a row-major 4x4 list of JSON numbers."
        )
    rows = list(value)
    if len(rows) != 4:
        raise WorldR1ProtocolError(f"{where} must have exactly 4 rows, got {len(rows)}.")
    matrix: list[list[float]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise WorldR1ProtocolError(f"{where} rows must be lists of 4 JSON numbers.")
        cells = list(row)
        if len(cells) != 4:
            raise WorldR1ProtocolError(
                f"{where} rows must have exactly 4 columns, got {len(cells)}."
            )
        matrix.append([_json_number(cell) for cell in cells])
    return matrix


def validate_camera_trajectory(
    value: Any, *, entry: int, expected_frames: int | None = None
) -> list[list[list[float]]]:
    """Return a typed ``[F][4][4]`` camera trajectory for one sample."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise WorldR1ProtocolError(
            f"camera_trajectories entry {entry} must be a list of 4x4 matrices."
        )
    frames = list(value)
    if not frames:
        raise WorldR1ProtocolError(
            f"camera_trajectories entry {entry} must contain at least one frame."
        )
    if expected_frames is not None and len(frames) != expected_frames:
        raise WorldR1ProtocolError(
            f"camera_trajectories entry {entry} must contain {expected_frames} "
            f"frames to match the video, got {len(frames)}."
        )
    return [
        validate_camera_matrix(frame, entry=entry, frame_index=index)
        for index, frame in enumerate(frames)
    ]


def build_health_payload(*, reward: str, server_revision: str) -> dict[str, Any]:
    """Build the exact GET /healthz success body."""

    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "wire_format": WIRE_FORMAT,
        "reward": validate_reward_kind(reward),
        "server_revision": validate_server_revision(server_revision),
        "manager_contract": MANAGER_CONTRACT,
    }


def validate_health_payload(
    payload: Any, *, reward: str, server_revision: str
) -> dict[str, Any]:
    """Validate a health body against the exact strict_v2 schema."""

    if not isinstance(payload, Mapping):
        raise WorldR1ProtocolError("World-R1 health response must be a JSON object.")
    keys = set(payload)
    if keys != set(HEALTH_KEYS):
        raise WorldR1ProtocolError(
            f"World-R1 health response keys must be exactly {sorted(HEALTH_KEYS)}, "
            f"got {sorted(keys)}."
        )
    if payload["status"] != "ok":
        raise WorldR1ProtocolError("World-R1 health status must be 'ok'.")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise WorldR1ProtocolError(
            f"World-R1 health protocol_version must be {PROTOCOL_VERSION!r}."
        )
    if payload["wire_format"] != WIRE_FORMAT:
        raise WorldR1ProtocolError(
            f"World-R1 health wire_format must be {WIRE_FORMAT!r}."
        )
    if payload["reward"] != reward:
        raise WorldR1ProtocolError(
            f"World-R1 health reward must be {reward!r}, got {payload['reward']!r}."
        )
    if payload["server_revision"] != validate_server_revision(server_revision):
        raise WorldR1RevisionError(
            "World-R1 health server_revision does not match the configured scorer."
        )
    if payload["manager_contract"] != MANAGER_CONTRACT:
        raise WorldR1ProtocolError(
            f"World-R1 health manager_contract must be {MANAGER_CONTRACT!r}."
        )
    return dict(payload)


def validate_outputs(value: Any, *, expected: int) -> list[float]:
    """Return finite ``outputs`` scores of shape ``[expected]``."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorldR1ProtocolError("World-R1 response outputs must be a list of numbers.")
    items = list(value)
    if len(items) != expected:
        raise WorldR1ProtocolError(
            f"World-R1 response outputs length must be {expected}, got {len(items)}."
        )
    outputs: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise WorldR1ProtocolError("World-R1 response outputs must be JSON numbers.")
        score = float(item)
        if not math.isfinite(score):
            raise WorldR1ProtocolError("World-R1 response outputs must be finite.")
        outputs.append(score)
    return outputs


def validate_valid_mask(value: Any, *, expected: int) -> list[bool]:
    """Return an all-True ``valid_mask`` of shape ``[expected]``."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorldR1ProtocolError("World-R1 response valid_mask must be a list of booleans.")
    items = list(value)
    if len(items) != expected:
        raise WorldR1ProtocolError(
            f"World-R1 response valid_mask length must be {expected}, got {len(items)}."
        )
    if any(not isinstance(item, bool) for item in items):
        raise WorldR1ProtocolError("World-R1 response valid_mask entries must be booleans.")
    if not all(items):
        raise WorldR1ProtocolError("World-R1 response valid_mask must be all True.")
    return [bool(item) for item in items]


def validate_score_response(
    payload: Any, *, expected_sample_ids: Sequence[str], server_revision: str
) -> list[float]:
    """Validate a strict_v2 score response and return its finite outputs."""

    if not isinstance(payload, Mapping):
        raise WorldR1ProtocolError("World-R1 score response must be a JSON object.")
    keys = set(payload)
    if keys != set(SCORE_RESPONSE_KEYS):
        raise WorldR1ProtocolError(
            f"World-R1 score response keys must be exactly "
            f"{sorted(SCORE_RESPONSE_KEYS)}, got {sorted(keys)}."
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise WorldR1ProtocolError(
            "strict_v2 response did not echo protocol_version='strict_v2'."
        )
    require_revision_echo(server_revision, payload["server_revision"])
    require_sample_id_echo(expected_sample_ids, payload["sample_id"])
    outputs = validate_outputs(payload["outputs"], expected=len(expected_sample_ids))
    validate_valid_mask(payload["valid_mask"], expected=len(expected_sample_ids))
    return outputs


__all__ = (
    "ERROR_COMPUTE_FAILED",
    "ERROR_INVALID_REQUEST",
    "ERROR_MANAGER_NOT_READY",
    "ERROR_REVISION_MISMATCH",
    "HEALTH_KEYS",
    "HEALTH_ROUTE",
    "HEALTH_TIMEOUT_S",
    "MANAGER_CONTRACT",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MIN_CLIENT_TIMEOUT_S",
    "PROTOCOL_VERSION",
    "REWARD_3D",
    "REWARD_GENERAL",
    "REWARD_KINDS",
    "SCORE_REQUEST_3D_KEYS",
    "SCORE_REQUEST_COMMON_KEYS",
    "SCORE_REQUEST_GENERAL_KEYS",
    "SCORE_RESPONSE_KEYS",
    "SCORE_ROUTE",
    "SERVER_REVISION_PATTERN",
    "STRICT_CLEANUP_TIMEOUT_S",
    "STRICT_FATAL_EXIT_CODE",
    "STRICT_MANAGER_TIMEOUT_S",
    "WIRE_FORMAT",
    "WSGI_TIMEOUT_S",
    "WorldR1ProtocolError",
    "WorldR1RevisionError",
    "build_health_payload",
    "decode_json",
    "encode_json",
    "require_revision_echo",
    "require_sample_id_echo",
    "validate_camera_matrix",
    "validate_camera_trajectory",
    "validate_health_payload",
    "validate_json_size",
    "validate_outputs",
    "validate_reward_kind",
    "validate_sample_ids",
    "validate_score_response",
    "validate_server_revision",
    "validate_valid_mask",
)
