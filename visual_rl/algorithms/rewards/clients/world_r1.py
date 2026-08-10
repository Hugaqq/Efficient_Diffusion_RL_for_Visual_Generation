"""Builtin World-R1 clients for the sole ``strict_v2 + json_v1`` wire."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import ipaddress
import math
import os
from pathlib import Path
import re
from typing import Any, ClassVar
from urllib.parse import urlsplit

import numpy as np

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RuntimeBuildContext,
    ValidationCheck,
    ValidationContext,
)
from visual_rl.algorithms.rewards.clients.input import (
    PointwiseRewardInput,
    pointwise_reward_output,
    resolve_pointwise_reward_input,
)
from visual_rl.algorithms.rewards.clients.mock import (
    RewardTransportError,
    close_http_response,
    read_bounded_http_response,
    redact_error_text,
    requests_session,
    validate_max_response_bytes,
)
from visual_rl.algorithms.rewards.types import PointwiseRewardOutput, RewardBatchView
from visual_rl.algorithms.rewards.input_selection import RewardInputSelectionPolicy
from visual_rl.core.protocols.world_r1 import (
    HEALTH_ROUTE,
    HEALTH_TIMEOUT_S,
    MANAGER_CONTRACT,
    MAX_REQUEST_BYTES,
    MIN_CLIENT_TIMEOUT_S,
    PROTOCOL_VERSION,
    REWARD_3D,
    REWARD_GENERAL,
    SCORE_ROUTE,
    WIRE_FORMAT,
    WorldR1ProtocolError,
    decode_json,
    encode_json,
    validate_camera_trajectory,
    validate_health_payload,
    validate_json_size,
    validate_score_response,
    validate_server_revision,
)

__all__ = [
    "WORLD_R1_RESOURCE_PROTOCOL",
    "WorldR1HealthAttestation",
    "WorldR1Reward3DClient",
    "WorldR1RewardGeneralClient",
]

WORLD_R1_RESOURCE_PROTOCOL = "world_r1_json"
_WORLD_PARAM_KEYS = frozenset(
    {
        "url",
        "timeout_s",
        "trusted_hosts",
        "ca_bundle",
        "max_response_bytes",
        "server_revision",
        "input_selection_policy",
    }
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_JSON_CONTENT_TYPE = "application/json"
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


@dataclass(frozen=True, slots=True)
class WorldR1HealthAttestation:
    """Typed facts independently established by one strict health exchange."""

    endpoint_origin: str
    reward: str
    protocol: str
    protocol_version: str
    wire_format: str
    server_revision: str
    manager_contract: str

    def __post_init__(self) -> None:
        origin, _host = _canonical_origin(self.endpoint_origin)
        if origin != self.endpoint_origin:
            raise ValueError("endpoint_origin must use its canonical origin spelling")
        if self.reward not in {REWARD_3D, REWARD_GENERAL}:
            raise ValueError("attested reward kind is unsupported")
        if self.protocol != WORLD_R1_RESOURCE_PROTOCOL:
            raise ValueError("attested resource protocol is unsupported")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("attested protocol_version is unsupported")
        if self.wire_format != WIRE_FORMAT:
            raise ValueError("attested wire_format is unsupported")
        validate_server_revision(self.server_revision)
        if self.manager_contract != MANAGER_CONTRACT:
            raise ValueError("attested manager_contract is unsupported")


class _WorldR1RewardClient:
    """Shared fail-closed transport; concrete clients only build request data."""

    name: ClassVar[str]

    def __init__(
        self,
        *,
        url: str,
        timeout_s: float,
        trusted_hosts: tuple[str, ...],
        ca_bundle: Path | None,
        max_response_bytes: int,
        server_revision: str,
        input_selection_policy: RewardInputSelectionPolicy | None,
        transport: Any | None = None,
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.trusted_hosts = trusted_hosts
        self.ca_bundle = ca_bundle
        self.max_response_bytes = max_response_bytes
        self.server_revision = server_revision
        if self.name == REWARD_GENERAL:
            if not isinstance(input_selection_policy, RewardInputSelectionPolicy):
                raise TypeError(
                    "reward_general requires a typed input_selection_policy"
                )
        elif input_selection_policy is not None:
            raise ValueError("reward_3d does not accept an input_selection_policy")
        self.input_selection_policy = input_selection_policy
        self._transport = requests_session() if transport is None else transport
        self._closed = False

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        return FrozenMapping(_resolve_world_params(raw, config_dir=context.config_dir))

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        params = _resolved_world_values(resolved)
        ca_issue = _ca_bundle_issue(params["ca_bundle"])
        if ca_issue is not None:
            return (
                ValidationCheck(
                    level="error",
                    code="reward.ca_bundle",
                    path=f"reward.{cls.name}.params.ca_bundle",
                    message=ca_issue,
                    volatile=False,
                ),
            )

        transport = None
        client = None
        try:
            transport = requests_session()
            client = cls(**params, transport=transport)
            client._health()
        except Exception as exc:  # noqa: BLE001 - convert to structured preflight
            return (
                ValidationCheck(
                    level="error",
                    code="reward.endpoint_unhealthy",
                    path=f"reward.{cls.name}.params.url",
                    message=redact_error_text(exc),
                    volatile=True,
                ),
            )
        finally:
            if client is not None:
                client.close()
            elif transport is not None:
                _close_transport(transport)
        del context
        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> _WorldR1RewardClient:
        del context
        params = _resolved_world_values(resolved)
        ca_issue = _ca_bundle_issue(params["ca_bundle"])
        if ca_issue is not None:
            raise ValueError(ca_issue)
        return cls(**params)

    def score(
        self,
        *,
        batch: RewardBatchView,
    ) -> PointwiseRewardOutput:
        if self._closed:
            raise RuntimeError(f"{self.name} client is closed")
        resolved = resolve_pointwise_reward_input(batch)
        payload, sample_metadata = self._build_payload(resolved)
        response = self._request_json(
            method="post",
            route=SCORE_ROUTE,
            payload=payload,
            timeout_s=self.timeout_s,
        )
        outputs = validate_score_response(
            response,
            expected_sample_ids=resolved.sample_ids,
            server_revision=self.server_revision,
        )
        selection_policy = self.input_selection_policy
        return pointwise_reward_output(
            resolved,
            outputs,
            shared_metadata={
                "reward": self.name,
                "protocol_version": PROTOCOL_VERSION,
                "wire_format": WIRE_FORMAT,
                "server_revision": self.server_revision,
                "input_selection_policy_id": (
                    None if selection_policy is None else selection_policy.policy_id
                ),
            },
            sample_metadata=sample_metadata,
        )

    def cache_fingerprint(self) -> FrozenMapping:
        """Canonical transport identity used only for diagnostics."""

        return FrozenMapping(
            {
                "url": self.url,
                "timeout_s": self.timeout_s,
                "trusted_hosts": self.trusted_hosts,
                "ca_bundle": self.ca_bundle,
                "max_response_bytes": self.max_response_bytes,
                "server_revision": self.server_revision,
                "input_selection_policy": (
                    None
                    if self.input_selection_policy is None
                    else self.input_selection_policy.to_payload()
                ),
                "protocol_version": PROTOCOL_VERSION,
                "wire_format": WIRE_FORMAT,
            }
        )

    def close(self) -> None:
        """Close the one owned Requests session exactly once."""

        if self._closed:
            return
        self._closed = True
        _close_transport(self._transport)

    def healthcheck(self) -> WorldR1HealthAttestation:
        """Verify the endpoint and return independent typed protocol evidence."""

        return self._health()

    def _health(self) -> WorldR1HealthAttestation:
        payload = self._request_json(
            method="get",
            route=HEALTH_ROUTE,
            payload=None,
            timeout_s=HEALTH_TIMEOUT_S,
        )
        validated = validate_health_payload(
            payload,
            reward=self.name,
            server_revision=self.server_revision,
        )
        return WorldR1HealthAttestation(
            endpoint_origin=self.url,
            reward=validated["reward"],
            protocol=WORLD_R1_RESOURCE_PROTOCOL,
            protocol_version=validated["protocol_version"],
            wire_format=validated["wire_format"],
            server_revision=validated["server_revision"],
            manager_contract=validated["manager_contract"],
        )

    def _request_json(
        self,
        *,
        method: str,
        route: str,
        payload: Mapping[str, object] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"{self.name} client is closed")
        kwargs: dict[str, Any] = {
            "timeout": timeout_s,
            "verify": True if self.ca_bundle is None else str(self.ca_bundle),
            "allow_redirects": False,
            "stream": True,
        }
        if payload is not None:
            body = encode_json(payload)
            validate_json_size(
                len(body),
                max_bytes=MAX_REQUEST_BYTES,
                what="score request",
            )
            kwargs.update(
                {
                    "data": body,
                    "headers": {"Content-Type": _JSON_CONTENT_TYPE},
                }
            )
        request = getattr(self._transport, method)
        response = None
        try:
            response = request(f"{self.url}{route}", **kwargs)
        except Exception as exc:  # noqa: BLE001 - transport boundary
            raise RewardTransportError(
                f"{self.name} HTTP request failed: {redact_error_text(exc)}"
            ) from exc

        try:
            if getattr(response, "status_code", None) != 200:
                raise RewardTransportError(
                    f"{self.name} HTTP response status must be 200"
                )
            content_type = _content_type(response)
            if content_type != _JSON_CONTENT_TYPE:
                raise WorldR1ProtocolError(
                    f"{self.name} HTTP Content-Type must be application/json"
                )
            body = read_bounded_http_response(
                response,
                max_response_bytes=self.max_response_bytes,
            )
            response = None
            return decode_json(
                body,
                max_bytes=self.max_response_bytes,
                what=f"{self.name} response",
            )
        finally:
            if response is not None:
                close_http_response(response)

    def _build_payload(
        self,
        batch: PointwiseRewardInput,
    ) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
        raise NotImplementedError


class WorldR1RewardGeneralClient(_WorldR1RewardClient):
    """General World-R1 reward using a recipe-owned frame selection policy."""

    name = REWARD_GENERAL

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        # Schema-v1 experiment inputs predate the typed selection contract and
        # implemented the middle-frame behavior directly in this client.  Keep
        # those frozen sources readable by making that historical behavior an
        # explicit extension policy at the legacy resolution boundary.  The
        # v0.8 recipe/resource compiler has its own strict schema and requires
        # the release policy to be present, so this is not a silent v2 default.
        migrated = dict(raw)
        if "input_selection_policy" not in migrated:
            migrated["input_selection_policy"] = (
                RewardInputSelectionPolicy.fixed_middle_extension().to_payload()
            )
        return super().resolve_params(migrated, context)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> WorldR1RewardGeneralClient:
        return super().from_config(resolved, context)

    def _build_payload(
        self,
        batch: PointwiseRewardInput,
    ) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
        if batch.media_layout == "BCHW":
            image_batch = _image_batch(batch)
            frame_count = 1
        else:
            videos = _video_batch(batch)
            frame_count = int(videos.shape[1])
        policy = self.input_selection_policy
        assert isinstance(policy, RewardInputSelectionPolicy)
        selection = policy.select(
            frame_count=frame_count,
            context=batch.context,
            sample_ids=batch.sample_ids,
            invocation_identity=self.name,
        )
        frame_index = selection.selected_frame_index
        if batch.media_layout != "BCHW":
            image_batch = videos[:, frame_index]
        images = [_jpeg_base64(image_batch[row]) for row in range(batch.flat_size)]
        payload: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "server_revision": self.server_revision,
            "sample_id": list(batch.sample_ids),
            "prompts": list(batch.prompts),
            "images": images,
        }
        records = tuple(
            {
                "source_frame_count": frame_count,
                "selected_frame_index": frame_index,
                "input_selection_policy_id": selection.policy_id,
                "input_selection_mode": policy.selection,
                "input_selection_sharing": policy.sharing,
                "selection_key_id": selection.selection_key_id,
            }
            for _ in batch.sample_ids
        )
        return payload, records


class WorldR1Reward3DClient(_WorldR1RewardClient):
    """World-R1 3D reward using all source frames and typed camera data."""

    name = REWARD_3D

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        # The schema-v1 3D reward never had a frame-selection policy because it
        # consumes all frames.  Materialize the typed ``None`` value without
        # mutating the archived source mapping.
        migrated = dict(raw)
        migrated.setdefault("input_selection_policy", None)
        return super().resolve_params(migrated, context)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> WorldR1Reward3DClient:
        return super().from_config(resolved, context)

    def _build_payload(
        self,
        batch: PointwiseRewardInput,
    ) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
        videos = _video_batch(batch)
        camera = batch.camera_trajectory
        if camera is None:
            raise ValueError("reward_3d requires batch.camera_trajectory")
        try:
            import torch

            if isinstance(camera, torch.Tensor):
                camera = camera.detach().to(device="cpu").numpy()
        except ModuleNotFoundError:  # pragma: no cover - core-only install
            pass
        camera_array = np.asarray(camera)
        if not np.issubdtype(camera_array.dtype, np.floating):
            raise TypeError("reward_3d camera trajectory must be floating point")
        if not bool(np.isfinite(camera_array).all()):
            raise ValueError("reward_3d camera trajectory must be finite")
        if camera_array.shape != (
            batch.flat_size,
            videos.shape[1],
            4,
            4,
        ):
            raise ValueError("reward_3d camera trajectory must match [B, F, 4, 4]")
        trajectories = [
            validate_camera_trajectory(
                camera_array[row].tolist(),
                entry=row,
                expected_frames=int(videos.shape[1]),
            )
            for row in range(batch.flat_size)
        ]
        encoded_videos = [
            [_jpeg_base64(frame) for frame in videos[row]]
            for row in range(batch.flat_size)
        ]
        payload: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "server_revision": self.server_revision,
            "sample_id": list(batch.sample_ids),
            "prompts": list(batch.prompts),
            "videos": encoded_videos,
            "camera_trajectories": trajectories,
        }
        records = tuple(
            {
                "source_frame_count": int(videos.shape[1]),
                "camera_frame_count": int(videos.shape[1]),
            }
            for _ in batch.sample_ids
        )
        return payload, records


def _resolve_world_params(
    raw: Mapping[str, object],
    *,
    config_dir: Path,
) -> dict[str, object]:
    _require_exact_keys(raw, set(_WORLD_PARAM_KEYS))
    url, url_host = _canonical_origin(raw["url"])
    timeout_s = _timeout(raw["timeout_s"])
    trusted_hosts = _trusted_hosts(raw["trusted_hosts"])
    if url_host not in trusted_hosts:
        raise ValueError("trusted_hosts must contain the reward URL hostname exactly")
    ca_bundle = _resolve_ca_bundle(raw["ca_bundle"], config_dir=config_dir)
    max_response_bytes = validate_max_response_bytes(raw["max_response_bytes"])
    server_revision = validate_server_revision(raw["server_revision"])
    return {
        "url": url,
        "timeout_s": timeout_s,
        "trusted_hosts": trusted_hosts,
        "ca_bundle": ca_bundle,
        "max_response_bytes": max_response_bytes,
        "server_revision": server_revision,
        "input_selection_policy": _selection_policy_payload(
            raw["input_selection_policy"]
        ),
    }


def _resolved_world_values(
    resolved: Mapping[str, object],
) -> dict[str, Any]:
    _require_exact_keys(resolved, set(_WORLD_PARAM_KEYS))
    url, url_host = _canonical_origin(resolved["url"])
    timeout_s = _timeout(resolved["timeout_s"])
    trusted_hosts = _trusted_hosts(resolved["trusted_hosts"])
    if url_host not in trusted_hosts:
        raise ValueError("trusted_hosts must contain the reward URL hostname exactly")
    ca_bundle = resolved["ca_bundle"]
    if ca_bundle is not None and not isinstance(ca_bundle, Path):
        raise TypeError("resolved ca_bundle must be a Path or None")
    return {
        "url": url,
        "timeout_s": timeout_s,
        "trusted_hosts": trusted_hosts,
        "ca_bundle": ca_bundle,
        "max_response_bytes": validate_max_response_bytes(
            resolved["max_response_bytes"]
        ),
        "server_revision": validate_server_revision(resolved["server_revision"]),
        "input_selection_policy": _selection_policy(resolved["input_selection_policy"]),
    }


def _selection_policy(value: object) -> RewardInputSelectionPolicy | None:
    if value is None:
        return None
    if isinstance(value, RewardInputSelectionPolicy):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("input_selection_policy must be a mapping or None")
    return RewardInputSelectionPolicy.from_mapping(value)


def _selection_policy_payload(value: object) -> dict[str, Any] | None:
    policy = _selection_policy(value)
    return None if policy is None else policy.to_payload()


def _require_exact_keys(
    raw: Mapping[str, object],
    expected: set[str],
) -> None:
    if not isinstance(raw, Mapping):
        raise TypeError("World-R1 params must be a mapping")
    actual = set(raw)
    if actual != expected:
        raise ValueError(
            f"World-R1 params must contain exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _canonical_origin(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise TypeError("World-R1 url must be a non-empty string")
    if (
        "?" in value
        or "#" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError(
            "World-R1 url must be an origin without whitespace, query, or fragment"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("World-R1 url has an invalid port") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("World-R1 url must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("World-R1 url must not contain userinfo")
    if parsed.hostname is None:
        raise ValueError("World-R1 url must contain a hostname")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "World-R1 url must be an origin without path, query, or fragment"
        )
    host = _canonical_host(parsed.hostname)
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        raise ValueError("World-R1 HTTP is allowed only for an exact loopback host")
    display_host = f"[{host}]" if ":" in host else host
    authority = display_host if port is None else f"{display_host}:{port}"
    return f"{parsed.scheme}://{authority}", host


def _canonical_host(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("trusted host values must be non-empty strings")
    if value != value.strip():
        raise ValueError("trusted host values must not contain surrounding whitespace")
    if "%" in value:
        raise ValueError("trusted host must not contain an IPv6 zone identifier")
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        try:
            host = value.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("trusted host is not a valid hostname") from exc
        if host.endswith("."):
            host = host[:-1]
        labels = host.split(".")
        if (
            not host
            or len(host) > 253
            or any(_HOST_LABEL.fullmatch(label) is None for label in labels)
        ):
            raise ValueError("trusted host is not a valid hostname")
        return host


def _trusted_hosts(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("trusted_hosts must be a non-empty sequence of hostnames")
    result = tuple(_canonical_host(item) for item in value)
    if not result:
        raise ValueError("trusted_hosts must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("trusted_hosts must not contain duplicates")
    return result


def _timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_s must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < MIN_CLIENT_TIMEOUT_S:
        raise ValueError(f"timeout_s must be finite and >= {MIN_CLIENT_TIMEOUT_S}")
    return result


def _resolve_ca_bundle(value: object, *, config_dir: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError("ca_bundle must be a path string or null")
    path = Path(value)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve(strict=False)


def _ca_bundle_issue(value: Path | None) -> str | None:
    if value is None:
        return None
    if not value.exists():
        return "World-R1 ca_bundle does not exist"
    if not value.is_file():
        return "World-R1 ca_bundle must be a regular file"
    if not os.access(value, os.R_OK):
        return "World-R1 ca_bundle must be readable"
    return None


def _content_type(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    value = headers.get("Content-Type", "")
    if not isinstance(value, str):
        return ""
    return value.partition(";")[0].strip().lower()


def _close_transport(transport: Any) -> None:
    close = getattr(transport, "close", None)
    if callable(close):
        close()


def _video_batch(batch: PointwiseRewardInput) -> np.ndarray:
    if batch.media_layout not in {"BFCHW", "BFHWC"}:
        raise ValueError("World-R1 rewards require BFCHW or BFHWC video media")
    media = batch.media
    try:
        import torch

        if isinstance(media, torch.Tensor):
            media = media.detach().to(device="cpu").numpy()
    except ModuleNotFoundError:  # pragma: no cover - core-only install
        pass
    array = np.asarray(media)
    if array.ndim != 5 or array.shape[0] != batch.flat_size:
        raise ValueError("World-R1 video media must have shape [B, F, ..., ...]")
    if batch.media_layout == "BFCHW":
        if array.shape[2] != 3:
            raise ValueError("BFCHW World-R1 media must have three RGB channels")
        array = np.moveaxis(array, 2, -1)
    elif array.shape[-1] != 3:
        raise ValueError("BFHWC World-R1 media must have three RGB channels")
    if array.shape[1] < 1 or array.shape[2] < 1 or array.shape[3] < 1:
        raise ValueError("World-R1 video dimensions must be positive")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("World-R1 media must be uint8 or floating point")
    if not np.isfinite(array).all():
        raise ValueError("World-R1 media must be finite")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError("floating World-R1 media must be in [0, 1]")
    return np.ascontiguousarray(np.rint(array * 255.0).astype(np.uint8))


def _image_batch(batch: PointwiseRewardInput) -> np.ndarray:
    if batch.media_layout != "BCHW":
        raise ValueError("general image reward requires BCHW image media")
    media = batch.media
    try:
        import torch

        if isinstance(media, torch.Tensor):
            media = media.detach().to(device="cpu").numpy()
    except ModuleNotFoundError:  # pragma: no cover - core-only install
        pass
    array = np.asarray(media)
    if array.ndim != 4 or array.shape[0] != batch.flat_size:
        raise ValueError("general image reward media must have shape [B, C, H, W]")
    if array.shape[1] != 3:
        raise ValueError("BCHW general image reward media must have three RGB channels")
    if array.shape[2] < 1 or array.shape[3] < 1:
        raise ValueError("general image reward dimensions must be positive")
    array = np.moveaxis(array, 1, -1)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("general image reward media must be uint8 or floating point")
    if not np.isfinite(array).all():
        raise ValueError("general image reward media must be finite")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError("floating general image reward media must be in [0, 1]")
    return np.ascontiguousarray(np.rint(array * 255.0).astype(np.uint8))


def _jpeg_base64(image: np.ndarray) -> str:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("World-R1 JPEG encoding requires Pillow") from exc
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
