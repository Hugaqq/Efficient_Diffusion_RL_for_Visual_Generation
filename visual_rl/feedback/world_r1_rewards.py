"""Wire-compatible clients for the released World-R1 reward servers."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import errno
from io import BytesIO
import json
import math
import pickle
import re
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np

from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.feedback.clients import (
    DEFAULT_MAX_RESPONSE_BYTES,
    JSON_V1,
    RewardProtocolError,
    RewardTransportError,
    close_http_response,
    read_bounded_http_response,
    redact_error_text,
    requests_session,
    validate_max_response_bytes,
    validate_wire_security_policy,
)

REFERENCE_V1 = "reference_v1"
STRICT_V2 = "strict_v2"
WORLD_R1_PROTOCOL_MODES = frozenset({REFERENCE_V1, STRICT_V2})
WORLD_R1_REWARD_CLIENT_NAMES = frozenset({"reward_3d", "reward_general"})

SCORE_META_VIEW = "score_meta_view"
SCORE_RECONSTRUCTION = "score_reconstruction"
SCORE_TRAJECTORY_ALIGNMENT = "score_trajectory_alignment"
TRAJECTORY_COMPARISON_PATHS = "trajectory_comparison_paths"


def validate_server_revision(
    value: Any,
    *,
    field: str = "server_revision",
) -> str | None:
    """Normalize an optional, user-declared remote scorer revision."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"World-R1 {field} must be a non-empty string or None.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"World-R1 {field} must be a non-empty string or None.")
    return normalized


def validate_reward_server_url(url: str, *, reward_name: str = "reward server") -> str:
    """Return a normalized HTTP(S) reward-server URL or raise ValueError."""

    normalized = str(url).strip()
    if not normalized:
        raise ValueError(f"{reward_name} URL must not be empty.")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"{reward_name} URL must use http or https, got {parsed.scheme!r}."
        )
    if not parsed.netloc:
        raise ValueError(f"{reward_name} URL must include a host.")
    if parsed.username or parsed.password:
        raise ValueError(f"{reward_name} URL must not embed credentials.")
    return normalized


def _as_numpy(media: Any) -> np.ndarray:
    if isinstance(media, np.ndarray):
        return media
    try:
        import torch
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - torch is an optional runtime dependency
        torch = None
    if torch is not None and isinstance(media, torch.Tensor):
        return media.detach().cpu().numpy()
    raise TypeError(
        "World-R1 reward media must be a numpy.ndarray or torch.Tensor, "
        f"got {type(media).__name__}."
    )


def _to_uint8_rgb(media: Any, *, float_quantization: str) -> np.ndarray:
    array = _as_numpy(media)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            "World-R1 reward media supports only uint8 or floating-point pixels, "
            f"got {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise ValueError("World-R1 reward media contains NaN or infinity.")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError("Floating-point World-R1 reward media must be in [0, 1].")
    scaled = array * 255.0
    if float_quantization == "round_to_nearest_even":
        quantized = np.rint(scaled)
    elif float_quantization == "truncate":
        quantized = scaled
    else:  # pragma: no cover - fixed by concrete client classes
        raise ValueError(f"Unknown World-R1 float quantization {float_quantization!r}.")
    return np.ascontiguousarray(quantized.astype(np.uint8))


def _resolve_layout(
    shape: tuple[int, ...],
    *,
    allowed: tuple[str, ...],
    requested: str,
) -> str:
    expected_dims = {"BCHW": 4, "BHWC": 4, "BFCHW": 5, "BFHWC": 5}
    channel_axes = {"BCHW": 1, "BHWC": 3, "BFCHW": 2, "BFHWC": 4}
    if requested != "auto":
        if requested not in allowed:
            raise ValueError(
                f"media_layout must be one of {['auto', *allowed]}, got {requested!r}."
            )
        if (
            len(shape) != expected_dims[requested]
            or shape[channel_axes[requested]] != 3
        ):
            raise ValueError(
                f"media_layout {requested} requires {expected_dims[requested]} dimensions "
                "with exactly 3 RGB channels."
            )
        return requested

    matches = [
        layout
        for layout in allowed
        if len(shape) == expected_dims[layout] and shape[channel_axes[layout]] == 3
    ]
    if len(matches) != 1:
        raise ValueError(
            "World-R1 media layout must be unambiguous; set media_layout explicitly. "
            f"Shape {shape} matched {matches or 'no supported RGB layout'}."
        )
    return matches[0]


def _rgb_jpeg(image: np.ndarray) -> bytes:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"JPEG frames must have shape [height, width, 3], got {image.shape}."
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("JPEG frames must have positive height and width.")
    try:
        from PIL import Image
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError(
            "World-R1 JPEG encoding requires the 'pillow' package."
        ) from exc
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return buffer.getvalue()


_CAMERA_MATRIX_ROW = re.compile(r"\[([^\[\]]+)\]")


def _validate_camera_matrix(value: Any, *, entry: int, frame_key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"reward_3d camera_trajectory entry {entry} {frame_key} must be a matrix string."
        )
    rows = _CAMERA_MATRIX_ROW.findall(value)
    if len(rows) != 4 or _CAMERA_MATRIX_ROW.sub("", value).strip():
        raise ValueError(
            f"reward_3d camera_trajectory entry {entry} {frame_key} "
            "must use the World-R1 finite 4x4 matrix string format."
        )
    try:
        matrix = np.asarray(
            [[float(item) for item in row.split()] for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reward_3d camera_trajectory entry {entry} {frame_key} "
            "must contain numeric matrix values."
        ) from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(
            f"reward_3d camera_trajectory entry {entry} {frame_key} "
            "must contain a finite 4x4 matrix."
        )
    return value


def _validate_camera_trajectory(
    trajectory: Any,
    *,
    expected_frames: int | None,
    entry: int,
) -> dict[str, str]:
    if not isinstance(trajectory, Mapping) or not trajectory:
        raise ValueError(
            "reward_3d camera_trajectory must be a non-empty mapping; "
            f"entry {entry} is invalid."
        )
    keys = list(trajectory)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(
            f"reward_3d camera_trajectory entry {entry} keys must be frame strings."
        )
    duplicate = sorted({key for key in keys if keys.count(key) > 1})
    trajectory_frames = len(keys)
    expected_keys = [f"frame{index}" for index in range(trajectory_frames)]
    missing = sorted(set(expected_keys) - set(keys))
    extra = sorted(set(keys) - set(expected_keys))
    if duplicate or missing or extra or len(keys) != len(expected_keys):
        raise ValueError(
            f"reward_3d camera_trajectory entry {entry} must exactly cover "
            f"frame0..frame{trajectory_frames - 1}; missing={missing}, "
            f"extra={extra}, duplicate={duplicate}."
        )
    if expected_frames is not None and trajectory_frames != expected_frames:
        raise ValueError(
            f"strict_v2 reward_3d camera_trajectory entry {entry} must contain "
            f"{expected_frames} frames to match the video, got {trajectory_frames}."
        )
    return {
        key: _validate_camera_matrix(trajectory[key], entry=entry, frame_key=key)
        for key in expected_keys
    }


def _world_r1_camera_matrix_string(matrix: np.ndarray) -> str:
    return " ".join(
        "[" + " ".join(format(float(value), ".17g") for value in row) + "]"
        for row in matrix
    )


def _normalize_reward_camera_trajectory(
    trajectory: Any,
    *,
    expected_frames: int | None,
    entry: int,
) -> dict[str, str]:
    """Convert canonical MinWM w2c/OpenCV metadata to World-R1 wire form."""

    if not isinstance(trajectory, Mapping):
        return _validate_camera_trajectory(
            trajectory,
            expected_frames=expected_frames,
            entry=entry,
        )
    minwm_keys = {"viewmats", "Ks", "convention", "coordinate_system"}
    if not (set(trajectory) & minwm_keys):
        return _validate_camera_trajectory(
            trajectory,
            expected_frames=expected_frames,
            entry=entry,
        )
    if set(trajectory) != minwm_keys:
        missing = sorted(minwm_keys - set(trajectory))
        extra = sorted(set(trajectory) - minwm_keys)
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} must contain exactly "
            f"{sorted(minwm_keys)}; missing={missing}, extra={extra}."
        )
    convention = trajectory["convention"]
    coordinate_system = trajectory["coordinate_system"]
    if not isinstance(convention, str) or convention.casefold() not in {
        "w2c",
        "world_to_camera",
    }:
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} must use w2c convention."
        )
    if (
        not isinstance(coordinate_system, str)
        or coordinate_system.casefold() != "opencv"
    ):
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} must use OpenCV "
            "coordinates."
        )
    try:
        viewmats = np.asarray(trajectory["viewmats"], dtype=np.float64)
        intrinsics = np.asarray(trajectory["Ks"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} must contain numeric "
            "viewmats and Ks."
        ) from exc
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4) or viewmats.shape[0] < 1:
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} viewmats must have "
            "shape [frames, 4, 4]."
        )
    if intrinsics.shape != (viewmats.shape[0], 3, 3):
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} Ks must have shape "
            f"[{viewmats.shape[0]}, 3, 3]."
        )
    if not np.isfinite(viewmats).all() or not np.isfinite(intrinsics).all():
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} viewmats and Ks "
            "must be finite."
        )
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} Ks must have "
            "positive fx and fy."
        )
    if not np.allclose(
        intrinsics[:, 2, :],
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"reward_3d MinWM camera_trajectory entry {entry} Ks must use "
            "homogeneous OpenCV intrinsics with last row [0, 0, 1]."
        )
    converted = {
        f"frame{index}": _world_r1_camera_matrix_string(matrix)
        for index, matrix in enumerate(viewmats)
    }
    return _validate_camera_trajectory(
        converted,
        expected_frames=expected_frames,
        entry=entry,
    )


def _validate_inputs(
    media: Any,
    prompts: list[str],
    metadata: list[dict[str, Any]],
    *,
    float_quantization: str,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    array = _to_uint8_rgb(media, float_quantization=float_quantization)
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("World-R1 prompts must be a sequence of strings.")
    resolved_prompts = list(prompts)
    if any(not isinstance(prompt, str) for prompt in resolved_prompts):
        raise TypeError("World-R1 prompts must contain only strings.")
    if isinstance(metadata, (str, bytes)) or not isinstance(metadata, Sequence):
        raise TypeError("World-R1 metadata must be a sequence of mappings.")
    resolved_metadata = list(metadata)
    if any(not isinstance(item, Mapping) for item in resolved_metadata):
        raise TypeError("World-R1 metadata entries must be mappings.")
    if len(resolved_prompts) != len(resolved_metadata):
        raise ValueError(
            "World-R1 prompts and metadata must have the same batch length, got "
            f"{len(resolved_prompts)} and {len(resolved_metadata)}."
        )
    if array.ndim == 0 or int(array.shape[0]) != len(resolved_prompts):
        raise ValueError(
            "World-R1 media batch dimension must match prompts, got "
            f"{array.shape[0] if array.ndim else 'scalar'} and {len(resolved_prompts)}."
        )
    return array, resolved_prompts, [dict(item) for item in resolved_metadata]


def _validate_sample_id(sample_id: Any, *, expected: int) -> list[str]:
    if isinstance(sample_id, (str, bytes)) or not isinstance(sample_id, Sequence):
        raise RewardProtocolError(
            f"strict_v2 requires sample_id to be a sequence of length {expected}."
        )
    result = list(sample_id)
    if len(result) != expected:
        raise RewardProtocolError(
            f"strict_v2 sample_id length must be {expected}, got {len(result)}."
        )
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise RewardProtocolError(
            "strict_v2 sample_id values must be non-empty strings."
        )
    if len(set(result)) != len(result):
        raise RewardProtocolError("strict_v2 sample_id values must be unique.")
    return result


def _float_vector(value: Any, *, expected: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception:  # noqa: BLE001 - untrusted array conversion must not escape
        array = None
    if array is None:
        raise RewardProtocolError(
            f"World-R1 response {field!r} must be numeric."
        ) from None
    if array.shape != (expected,):
        raise RewardProtocolError(
            f"World-R1 response {field!r} shape must be ({expected},), got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise RewardProtocolError(
            f"World-R1 response {field!r} contains non-finite values."
        )
    return array


def _bool_vector(value: Any, *, expected: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception:  # noqa: BLE001 - untrusted array conversion must not escape
        array = None
    if array is None:
        raise RewardProtocolError(
            f"World-R1 response {field!r} must be a boolean vector of shape ({expected},)."
        ) from None
    if array.shape != (expected,) or array.dtype.kind != "b":
        raise RewardProtocolError(
            f"World-R1 response {field!r} must be a boolean vector of shape ({expected},)."
        )
    return array.astype(bool, copy=False)


def _json_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Encode only the JPEG byte fields in the existing World-R1 payload shape."""

    result = dict(payload)
    if "images" in result:
        result["images"] = [
            base64.b64encode(bytes(image)).decode("ascii") for image in result["images"]
        ]
    if "videos" in result:
        result["videos"] = [
            [base64.b64encode(bytes(frame)).decode("ascii") for frame in video]
            for video in result["videos"]
        ]
    return result


def _positive_alignment_int(
    alignment: Mapping[str, Any], field: str, *, entry: int
) -> int:
    value = alignment.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"metadata[{entry}].minwm_reward_frame_alignment.{field} "
            "must be a positive integer."
        )
    return value


def _reject_json_constant(value: str) -> None:
    del value
    raise ValueError("non-finite JSON constant")


class _WorldR1RewardClient:
    name = "world_r1"
    payload_kind = "media"
    default_batch_size = 1
    float_quantization = "round_to_nearest_even"

    def __init__(
        self,
        url: str,
        *,
        timeout: float,
        retries: int = 2,
        backoff_seconds: float = 0.25,
        protocol_mode: str = REFERENCE_V1,
        wire_format: str = JSON_V1,
        allow_unsafe_pickle: bool = False,
        trusted_hosts: Sequence[str] = (),
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        media_layout: str = "auto",
        batch_size: int | None = None,
        server_revision: str | None = None,
        transport: Any = None,
        sleep: Any = time.sleep,
        jpeg_encoder: Any = None,
    ) -> None:
        self.url = validate_reward_server_url(url, reward_name=self.name)
        try:
            self.timeout = float(timeout)
            self.backoff_seconds = float(backoff_seconds)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "World-R1 reward timeout and backoff_seconds must be finite and positive."
            ) from None
        self.retries = int(retries)
        self.protocol_mode = str(protocol_mode)
        self.wire_format, self.trusted_hosts = validate_wire_security_policy(
            self.url,
            wire_format=wire_format,
            allow_unsafe_pickle=allow_unsafe_pickle,
            trusted_hosts=trusted_hosts,
        )
        self.allow_unsafe_pickle = allow_unsafe_pickle
        self.max_response_bytes = validate_max_response_bytes(max_response_bytes)
        self.media_layout = str(media_layout)
        self.batch_size = (
            self.default_batch_size if batch_size is None else int(batch_size)
        )
        self.server_revision = validate_server_revision(server_revision)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("World-R1 reward timeout must be finite and positive.")
        if self.retries < 0 or self.retries > 10:
            raise ValueError("World-R1 reward retries must be between 0 and 10.")
        if (
            not math.isfinite(self.backoff_seconds)
            or self.backoff_seconds <= 0
            or not math.isfinite(self.backoff_seconds * (2**self.retries))
        ):
            raise ValueError(
                "World-R1 reward backoff_seconds must be finite and positive."
            )
        if self.protocol_mode not in WORLD_R1_PROTOCOL_MODES:
            raise ValueError(
                f"protocol_mode must be one of {sorted(WORLD_R1_PROTOCOL_MODES)}."
            )
        if self.batch_size <= 0:
            raise ValueError("World-R1 reward batch_size must be positive.")
        self.transport = requests_session() if transport is None else transport
        if not callable(getattr(self.transport, "post", None)):
            raise TypeError("World-R1 transport must provide a post(...) method.")
        if not callable(sleep):
            raise TypeError("World-R1 sleep must be callable.")
        self._sleep = sleep
        self._jpeg_encoder = _rgb_jpeg if jpeg_encoder is None else jpeg_encoder
        if not callable(self._jpeg_encoder):
            raise TypeError("World-R1 jpeg_encoder must be callable.")
        if jpeg_encoder is None:
            self._jpeg_encoding = {
                "name": "pillow_default_rgb",
                "format": "jpeg",
                "wire_compatible": True,
            }
        else:
            declared_encoding = getattr(jpeg_encoder, "encoding_metadata", None)
            self._jpeg_encoding = (
                dict(declared_encoding)
                if isinstance(declared_encoding, Mapping)
                else {
                    "name": "injected",
                    "format": "unknown",
                    "wire_compatible": False,
                }
            )

    def cache_fingerprint(self) -> dict[str, Any] | None:
        if self.server_revision is None:
            return None
        return {
            "client": f"{type(self).__module__}:{type(self).__qualname__}",
            "url": self.url,
            "server_revision": self.server_revision,
            "protocol_mode": self.protocol_mode,
            "wire_format": self.wire_format,
            "allow_unsafe_pickle": self.allow_unsafe_pickle,
            "trusted_hosts": list(self.trusted_hosts),
            "max_response_bytes": self.max_response_bytes,
            "payload_kind": self.payload_kind,
            "media_layout": self.media_layout,
            "batch_size": self.batch_size,
            "frame_policy": self._frame_policy(),
            "jpeg_encoding": self._jpeg_encoding,
            "float_quantization": self.float_quantization,
        }

    def _frame_policy(self) -> Any:
        return None

    def prepare_payloads(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        sample_id: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        array, prompts, metadata = _validate_inputs(
            media,
            prompts,
            metadata,
            float_quantization=self.float_quantization,
        )
        encoded, encoding_metadata = self._encode_media(array)
        identities = None
        if self.protocol_mode == STRICT_V2:
            identities = _validate_sample_id(sample_id, expected=len(prompts))

        payloads = []
        for start in range(0, len(prompts), self.batch_size):
            stop = min(start + self.batch_size, len(prompts))
            payload = self._build_payload(
                encoded[start:stop], prompts[start:stop], metadata[start:stop]
            )
            if identities is not None:
                payload["protocol_version"] = STRICT_V2
                payload["sample_id"] = identities[start:stop]
                if self.server_revision is not None:
                    payload["server_revision"] = self.server_revision
            payloads.append(payload)
        return payloads, encoding_metadata

    def score(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        sample_id: Any = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        payloads, encoding_metadata = self.prepare_payloads(
            media, prompts, metadata, sample_id=sample_id
        )
        all_values: list[float] = []
        all_valid: list[bool] = []
        all_details: dict[str, list[Any]] = {}
        echoed_ids: list[str] = []
        for payload in payloads:
            expected = len(payload["prompts"])
            response = self._request(payload)
            values, valid, details, response_ids = self._parse_response(
                response,
                expected=expected,
                requested_sample_id=payload.get("sample_id"),
            )
            all_values.extend(values.tolist())
            all_valid.extend(valid.tolist())
            if response_ids is not None:
                echoed_ids.extend(response_ids)
            for key, items in details.items():
                all_details.setdefault(key, []).extend(items)

        expected_total = len(prompts)
        for key, items in all_details.items():
            if len(items) != expected_total:
                raise RewardProtocolError(
                    f"World-R1 response metadata {key!r} length must be "
                    f"{expected_total}, got {len(items)}."
                )
        result_metadata: dict[str, Any] = {
            "server_revision": self.server_revision,
            "protocol_mode": self.protocol_mode,
            "configured_batch_size": self.batch_size,
            "request_count": len(payloads),
            "payload_batch_sizes": [len(payload["prompts"]) for payload in payloads],
            "valid_mask": all_valid,
            "payload_kind": self.payload_kind,
            "encoding": encoding_metadata,
        }
        if self.protocol_mode == REFERENCE_V1:
            result_metadata.update(
                {
                    "identity_mode": "trusted_input_order",
                    "sample_id_mode": "trusted_input_order_reference_v1",
                    "server_identity_echo": False,
                }
            )
        else:
            result_metadata.update(
                {
                    "protocol_version": STRICT_V2,
                    "sample_id": echoed_ids,
                    "identity_mode": "server_echo",
                    "sample_id_mode": "server_echo_strict_v2",
                    "server_identity_echo": True,
                    "server_revision_echo": self.server_revision is not None,
                }
            )
        result_metadata.update(all_details)
        if self.protocol_mode == STRICT_V2:
            result_metadata["sample_evidence"] = [
                {
                    "sample_id": echoed_ids[index],
                    **{key: values[index] for key, values in all_details.items()},
                }
                for index in range(expected_total)
            ]
        return np.asarray(all_values, dtype=np.float32), result_metadata

    def _request(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        if self.wire_format == JSON_V1:
            try:
                body = json.dumps(
                    _json_request_payload(payload),
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                body = None
            if body is None:
                raise RewardProtocolError(
                    "World-R1 request cannot be encoded as json_v1."
                ) from None
            content_type = "application/json"
        else:
            try:
                body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                body = None
            if body is None:
                raise RewardProtocolError(
                    "World-R1 request cannot be encoded as legacy_pickle."
                ) from None
            content_type = "application/octet-stream"
        for attempt in range(self.retries + 1):
            request_error: RewardTransportError | None = None
            try:
                response = self.transport.post(
                    self.url,
                    data=body,
                    timeout=self.timeout,
                    headers={"Content-Type": content_type},
                    allow_redirects=False,
                    stream=True,
                )
            except Exception as exc:
                safe_error = redact_error_text(exc)
                if not _retryable_transport_exception(exc):
                    request_error = RewardTransportError(
                        f"World-R1 reward request failed with a permanent transport "
                        f"error: {safe_error}"
                    )
                elif attempt == self.retries:
                    request_error = RewardTransportError(
                        f"World-R1 reward request failed after {attempt + 1} attempts: "
                        f"{safe_error}"
                    )
                else:
                    self._backoff(attempt)
                    continue
            if request_error is not None:
                raise request_error from None

            status = getattr(response, "status_code", None)
            if not isinstance(status, int):
                close_http_response(response)
                raise RewardTransportError(
                    "World-R1 transport response has no integer status_code."
                )
            if 500 <= status <= 599 and attempt < self.retries:
                close_http_response(response)
                self._backoff(attempt)
                continue
            if status < 200 or status >= 300:
                close_http_response(response)
                raise RewardTransportError(
                    f"World-R1 reward server returned HTTP {status}."
                )
            response_error: RewardProtocolError | RewardTransportError | None = None
            try:
                content_bytes = read_bounded_http_response(
                    response, max_response_bytes=self.max_response_bytes
                )
            except RewardProtocolError as exc:
                if "max_response_bytes" in str(exc):
                    response_error = RewardProtocolError(
                        "World-R1 response exceeds max_response_bytes."
                    )
                else:
                    response_error = exc
            except Exception as exc:
                safe_error = redact_error_text(exc)
                if not _retryable_transport_exception(exc):
                    response_error = RewardTransportError(
                        "World-R1 reward response streaming failed with a permanent "
                        f"transport error: {safe_error}"
                    )
                elif attempt == self.retries:
                    response_error = RewardTransportError(
                        "World-R1 reward response streaming failed after "
                        f"{attempt + 1} attempts: {safe_error}"
                    )
                else:
                    self._backoff(attempt)
                    continue
            if response_error is not None:
                raise response_error from None
            if self.wire_format == JSON_V1:
                decoded: Any = None
                try:
                    decoded = json.loads(
                        content_bytes.decode("utf-8"),
                        parse_constant=_reject_json_constant,
                    )
                except (UnicodeError, json.JSONDecodeError, ValueError):
                    decoded = None
                if decoded is None:
                    raise RewardProtocolError(
                        "World-R1 response is not valid finite json_v1."
                    ) from None
            else:
                decoded = None
                try:
                    decoded = pickle.loads(content_bytes)
                except Exception:
                    decoded = None
                if decoded is None:
                    raise RewardProtocolError(
                        "World-R1 response is not a valid legacy_pickle payload."
                    ) from None
            if not isinstance(decoded, Mapping):
                raise RewardProtocolError("World-R1 response must contain a mapping.")
            return decoded
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2**attempt)
        if delay:
            self._sleep(delay)

    def _parse_response(
        self,
        response: Mapping[str, Any],
        *,
        expected: int,
        requested_sample_id: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, list[Any]], list[str] | None]:
        if "outputs" not in response:
            raise RewardProtocolError("World-R1 response is missing 'outputs'.")
        values = _float_vector(response["outputs"], expected=expected, field="outputs")
        if self.protocol_mode == REFERENCE_V1:
            valid = np.ones(expected, dtype=bool)
            response_ids = None
        else:
            if response.get("protocol_version") != STRICT_V2:
                raise RewardProtocolError(
                    "strict_v2 response did not echo protocol_version='strict_v2'."
                )
            response_ids = _validate_sample_id(
                response.get("sample_id"), expected=expected
            )
            if response_ids != requested_sample_id:
                raise RewardProtocolError(
                    "strict_v2 response sample_id does not match the request."
                )
            if (
                self.server_revision is not None
                and response.get("server_revision") != self.server_revision
            ):
                raise RewardProtocolError(
                    "strict_v2 response server_revision does not match the configured scorer."
                )
            if "valid_mask" not in response:
                raise RewardProtocolError("strict_v2 response is missing 'valid_mask'.")
            valid = _bool_vector(
                response["valid_mask"], expected=expected, field="valid_mask"
            )
        details = self._parse_details(
            response.get("details"),
            expected=expected,
            outputs=values,
        )
        return values, valid, details, response_ids

    def _encode_media(self, array: np.ndarray) -> tuple[list[Any], dict[str, Any]]:
        raise NotImplementedError

    def _build_payload(
        self,
        encoded: list[Any],
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _parse_details(
        self,
        details: Any,
        *,
        expected: int,
        outputs: np.ndarray | None = None,
    ) -> dict[str, list[Any]]:
        del details, expected, outputs
        return {}


class WorldR1RewardGeneralClient(_WorldR1RewardClient):
    name = "reward_general"
    payload_kind = "images"
    default_batch_size = 64

    def __init__(
        self,
        url: str,
        timeout: float = 1000.0,
        retries: int = 2,
        frame_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.frame_index = None if frame_index is None else int(frame_index)
        super().__init__(url, timeout=timeout, retries=retries, **kwargs)

    def _frame_policy(self) -> Any:
        return {"mode": "fixed", "frame_index": self.frame_index, "default": "middle"}

    def _encode_media(self, array: np.ndarray) -> tuple[list[bytes], dict[str, Any]]:
        shape = tuple(int(item) for item in array.shape)
        layout = _resolve_layout(
            shape,
            allowed=("BCHW", "BHWC", "BFCHW", "BFHWC"),
            requested=self.media_layout,
        )
        selected = None
        if layout == "BCHW":
            images = np.transpose(array, (0, 2, 3, 1))
        elif layout == "BHWC":
            images = array
        else:
            frame_count = shape[1]
            selected = (
                frame_count // 2 if self.frame_index is None else self.frame_index
            )
            if selected < 0 or selected >= frame_count:
                raise ValueError(
                    f"reward_general frame_index {selected} is outside frame range "
                    f"0..{frame_count - 1}."
                )
            if layout == "BFCHW":
                images = np.transpose(array[:, selected], (0, 2, 3, 1))
            else:
                images = array[:, selected]
        encoded = [self._jpeg_encoder(image) for image in images]
        return encoded, {
            "input_shape": list(shape),
            "input_layout": layout,
            "selected_frame_index": selected,
            "selection_policy": "fixed_middle"
            if self.frame_index is None
            else "fixed_index",
            "jpeg_encoding": dict(self._jpeg_encoding),
        }

    def _build_payload(
        self,
        encoded: list[bytes],
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del metadata
        return {"images": encoded, "prompts": prompts}


class WorldR1Reward3DClient(_WorldR1RewardClient):
    name = "reward_3d"
    payload_kind = "videos"
    default_batch_size = 8
    float_quantization = "truncate"

    def __init__(
        self,
        url: str,
        timeout: float = 2000.0,
        retries: int = 2,
        require_camera_trajectory: bool = False,
        frame_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(require_camera_trajectory, bool):
            raise TypeError("require_camera_trajectory must be a boolean.")
        self.require_camera_trajectory = require_camera_trajectory
        if frame_indices is None:
            self.frame_indices: tuple[int, ...] | None = None
        else:
            if isinstance(frame_indices, (str, bytes)) or not isinstance(
                frame_indices, Sequence
            ):
                raise TypeError("frame_indices must be a sequence of integers or None.")
            resolved_indices = list(frame_indices)
            if not resolved_indices:
                raise ValueError("frame_indices must not be empty.")
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in resolved_indices
            ):
                raise TypeError("frame_indices must contain only integers.")
            if any(index < 0 for index in resolved_indices):
                raise ValueError("frame_indices must contain only non-negative values.")
            if len(set(resolved_indices)) != len(resolved_indices):
                raise ValueError("frame_indices must contain unique values.")
            if any(
                current >= following
                for current, following in zip(
                    resolved_indices,
                    resolved_indices[1:],
                    strict=False,
                )
            ):
                raise ValueError("frame_indices must be strictly increasing.")
            self.frame_indices = tuple(resolved_indices)
        super().__init__(url, timeout=timeout, retries=retries, **kwargs)

    @staticmethod
    def _camera_conversion_identity() -> dict[str, str]:
        return {
            "input_pose_convention": "w2c",
            "input_coordinate_system": "opencv",
            "intrinsics_handling": "validated_not_transmitted",
            "wire_camera_payload": "extrinsics_only_4x4_matrix_strings",
        }

    def _frame_policy(self) -> dict[str, Any]:
        return {
            "mode": (
                "all_source_frames" if self.frame_indices is None else "fixed_indices"
            ),
            "source_frame_count": "runtime_media",
            "selected_frame_indices": (
                "all" if self.frame_indices is None else list(self.frame_indices)
            ),
        }

    def cache_fingerprint(self) -> dict[str, Any] | None:
        fingerprint = super().cache_fingerprint()
        if fingerprint is None:
            return None
        fingerprint["require_camera_trajectory"] = self.require_camera_trajectory
        fingerprint["camera_conversion"] = self._camera_conversion_identity()
        return fingerprint

    def _encode_media(
        self, array: np.ndarray
    ) -> tuple[list[list[bytes]], dict[str, Any]]:
        shape = tuple(int(item) for item in array.shape)
        layout = _resolve_layout(
            shape,
            allowed=("BCHW", "BHWC", "BFCHW", "BFHWC"),
            requested=self.media_layout,
        )
        if layout == "BCHW":
            videos = np.transpose(array, (0, 2, 3, 1))[:, None]
        elif layout == "BHWC":
            videos = array[:, None]
        elif layout == "BFCHW":
            videos = np.transpose(array, (0, 1, 3, 4, 2))
        else:
            videos = array
        source_frame_count = int(videos.shape[1])
        if source_frame_count <= 0:
            raise ValueError("World-R1 3D videos must contain at least one frame.")
        selected_indices = (
            list(range(source_frame_count))
            if self.frame_indices is None
            else list(self.frame_indices)
        )
        if selected_indices[-1] >= source_frame_count:
            raise ValueError(
                f"reward_3d frame index {selected_indices[-1]} is outside source "
                f"frame range 0..{source_frame_count - 1}."
            )
        if self.frame_indices is not None:
            videos = videos[:, selected_indices]
        selected_frame_count = int(videos.shape[1])
        encoded = [[self._jpeg_encoder(frame) for frame in video] for video in videos]
        return encoded, {
            "input_shape": list(shape),
            "input_layout": layout,
            "source_frames_per_video": source_frame_count,
            "selected_frame_indices": selected_indices,
            "frames_per_video": selected_frame_count,
            "selection_policy": (
                "all_source_frames" if self.frame_indices is None else "fixed_indices"
            ),
            "promoted_single_frame": len(shape) == 4,
            "jpeg_encoding": dict(self._jpeg_encoding),
            "camera_conversion": self._camera_conversion_identity(),
        }

    def _build_payload(
        self,
        encoded: list[list[bytes]],
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trajectories = []
        for index, (video, item) in enumerate(zip(encoded, metadata, strict=True)):
            self._validate_minwm_frame_alignment(item, video, entry=index)
            trajectory = item.get("camera_trajectory")
            if trajectory is None and not self.require_camera_trajectory:
                trajectories.append(None)
                continue
            trajectories.append(
                _normalize_reward_camera_trajectory(
                    trajectory,
                    expected_frames=(
                        len(video) if self.protocol_mode == STRICT_V2 else None
                    ),
                    entry=index,
                )
            )
        return {
            "videos": encoded,
            "prompts": prompts,
            "camera_trajectories": trajectories,
        }

    def _validate_minwm_frame_alignment(
        self,
        metadata: Mapping[str, Any],
        video: Sequence[bytes],
        *,
        entry: int,
    ) -> None:
        alignment = metadata.get("minwm_reward_frame_alignment")
        if alignment is None:
            return
        if not isinstance(alignment, Mapping) or alignment.get("contract") != (
            "minwm_vae_camera_alignment_v1"
        ):
            raise ValueError(
                f"metadata[{entry}].minwm_reward_frame_alignment is invalid."
            )
        latent_frames = _positive_alignment_int(alignment, "latent_frames", entry=entry)
        decoded_frames = _positive_alignment_int(
            alignment, "decoded_media_frames", entry=entry
        )
        temporal_stride = _positive_alignment_int(
            alignment, "vae_temporal_stride", entry=entry
        )
        if decoded_frames != 1 + temporal_stride * (latent_frames - 1):
            raise ValueError(
                f"metadata[{entry}] MinWM decoded frame geometry is inconsistent."
            )
        expected = tuple(range(0, decoded_frames, temporal_stride))
        if len(expected) != latent_frames:
            raise ValueError(
                f"metadata[{entry}] MinWM camera alignment count is inconsistent."
            )
        if self.frame_indices != expected:
            raise ValueError(
                "reward_3d frame_indices must match MinWM VAE camera alignment: "
                f"expected {list(expected)}."
            )
        if len(video) != latent_frames:
            raise ValueError(
                f"metadata[{entry}] MinWM reward video must contain exactly "
                f"{latent_frames} aligned frames."
            )

    def _parse_details(
        self,
        details: Any,
        *,
        expected: int,
        outputs: np.ndarray | None = None,
    ) -> dict[str, list[Any]]:
        if details is None:
            return {}
        if isinstance(details, (str, bytes)) or not isinstance(details, Sequence):
            raise RewardProtocolError(
                "World-R1 3D response details must be a sequence."
            )
        items = list(details)
        if len(items) != expected:
            raise RewardProtocolError(
                f"World-R1 3D response details length must be {expected}, got {len(items)}."
            )
        numeric_fields = {
            "gs_score": SCORE_RECONSTRUCTION,
            "meta_score": SCORE_META_VIEW,
            "camera_motion_score": SCORE_TRAJECTORY_ALIGNMENT,
        }
        typed_items: list[Mapping[str, Any]] = []
        for response_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise RewardProtocolError(
                    f"World-R1 3D response details[{response_index}] must be a mapping."
                )
            typed_items.append(item)

        has_video_id = ["video_id" in item for item in typed_items]
        if any(has_video_id) and not all(has_video_id):
            raise RewardProtocolError(
                "World-R1 3D response details must either all include video_id or "
                "all use trusted reference order."
            )
        reference_order = not any(has_video_id)
        if reference_order and self.protocol_mode != REFERENCE_V1:
            raise RewardProtocolError(
                "strict_v2 World-R1 3D details must include video_id."
            )

        ordered_items: list[tuple[int, Mapping[str, Any]]] = []
        seen_video_ids: set[int] = set()
        for response_index, item in enumerate(typed_items):
            if reference_order:
                ordered_items.append((response_index, item))
                continue
            video_id = item.get("video_id")
            if isinstance(video_id, bool) or not isinstance(
                video_id, (int, np.integer)
            ):
                raise RewardProtocolError(
                    f"World-R1 3D response details[{response_index}].video_id must be an integer."
                )
            resolved_video_id = int(video_id)
            if resolved_video_id < 0 or resolved_video_id >= expected:
                raise RewardProtocolError(
                    "World-R1 3D response details video_id must be in "
                    f"0..{expected - 1}, got {resolved_video_id}."
                )
            if resolved_video_id in seen_video_ids:
                raise RewardProtocolError(
                    f"World-R1 3D response details has duplicate video_id {resolved_video_id}."
                )
            seen_video_ids.add(resolved_video_id)
            ordered_items.append((resolved_video_id, item))
        expected_video_ids = set(range(expected))
        if not reference_order and seen_video_ids != expected_video_ids:
            missing = sorted(expected_video_ids - seen_video_ids)
            raise RewardProtocolError(
                f"World-R1 3D response details is missing video_id values {missing}."
            )

        result: dict[str, list[Any]] = {name: [] for name in numeric_fields.values()}
        result[TRAJECTORY_COMPARISON_PATHS] = []
        for video_id, item in sorted(ordered_items):
            components: list[float] = []
            for source, target in numeric_fields.items():
                if source not in item:
                    raise RewardProtocolError(
                        f"World-R1 3D response details for video_id {video_id} "
                        f"is missing {source!r}."
                    )
                value = _float_vector([item[source]], expected=1, field=source)[0]
                if not 0.0 <= float(value) <= 1.0:
                    raise RewardProtocolError(
                        f"World-R1 3D response {source!r} must be in [0, 1]."
                    )
                components.append(float(value))
                result[target].append(float(value))
            if reference_order:
                required_reference_fields = {
                    "final_score",
                    "gs_video_path",
                    "meta_view_path",
                }
                missing_reference = sorted(required_reference_fields.difference(item))
                if missing_reference:
                    raise RewardProtocolError(
                        "World-R1 reference_v1 3D details are missing "
                        f"{missing_reference}."
                    )
            if "final_score" in item:
                final_score = float(
                    _float_vector(
                        [item["final_score"]],
                        expected=1,
                        field="final_score",
                    )[0]
                )
                expected_output = None if outputs is None else float(outputs[video_id])
                if abs(sum(components) - final_score) > 1e-6 or (
                    expected_output is not None
                    and abs(final_score - expected_output) > 1e-6
                ):
                    raise RewardProtocolError(
                        "World-R1 3D final_score must equal its components and output."
                    )
            for artifact_field in ("gs_video_path", "meta_view_path"):
                artifact_path = item.get(artifact_field)
                if artifact_field in item and (
                    not isinstance(artifact_path, str) or not artifact_path.strip()
                ):
                    raise RewardProtocolError(
                        "World-R1 3D returned an empty reconstruction artifact path."
                    )
            path = item.get("trajectory_comparison_path", "")
            if path is None:
                path = ""
            if not isinstance(path, str):
                raise RewardProtocolError(
                    "World-R1 3D trajectory_comparison_path must be a string or null."
                )
            result[TRAJECTORY_COMPARISON_PATHS].append(path)
        return result


def _retryable_transport_exception(exc: Exception) -> bool:
    try:
        import requests
    except ModuleNotFoundError:
        requests = None
    if requests is not None:
        exceptions = requests.exceptions
        permanent = (
            exceptions.InvalidURL,
            exceptions.InvalidSchema,
            exceptions.MissingSchema,
            exceptions.TooManyRedirects,
            exceptions.SSLError,
        )
        if isinstance(exc, permanent):
            return False
        if isinstance(exc, (exceptions.Timeout, exceptions.ConnectionError)):
            return True
        if isinstance(exc, exceptions.RequestException):
            return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    transient_os_errors = {
        errno.EAGAIN,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
    return isinstance(exc, OSError) and exc.errno in transient_os_errors


def sample_video_frame_for_reward_general(
    media: Any, *, frame_index: int | None = None
) -> tuple[Any, dict[str, Any]]:
    """Compatibility helper using the same deterministic frame policy as the client."""

    array = _as_numpy(media)
    shape = tuple(int(item) for item in array.shape)
    if len(shape) != 5:
        return media, {
            "input_shape": list(shape),
            "output_shape": list(shape),
            "selected_frame_index": None,
        }
    layout = _resolve_layout(
        shape,
        allowed=("BFCHW", "BFHWC"),
        requested="auto",
    )
    selected = shape[1] // 2 if frame_index is None else int(frame_index)
    if selected < 0 or selected >= shape[1]:
        raise ValueError(
            f"reward_general frame_index {selected} is outside frame range 0..{shape[1] - 1}."
        )
    images = media[:, selected]
    return images, {
        "input_shape": list(shape),
        "output_shape": list(getattr(images, "shape", [])),
        "selected_frame_index": selected,
        "input_layout": layout,
    }


def reward_3d_client(url: str, **kwargs: Any) -> WorldR1Reward3DClient:
    return WorldR1Reward3DClient(url=url, **kwargs)


def reward_general_client(url: str, **kwargs: Any) -> WorldR1RewardGeneralClient:
    return WorldR1RewardGeneralClient(url=url, **kwargs)


REWARD_CLIENTS.register("reward_3d", WorldR1Reward3DClient)
REWARD_CLIENTS.register("reward_general", WorldR1RewardGeneralClient)
