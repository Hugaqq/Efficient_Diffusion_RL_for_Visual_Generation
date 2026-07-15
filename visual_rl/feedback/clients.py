"""Reward client implementations."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
import re
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.feedback.cache import stable_hash_text


class RewardTransportError(RuntimeError):
    """A reward request could not complete at the HTTP transport boundary."""


class RewardProtocolError(ValueError):
    """A reward peer sent or received data that violates its wire contract."""


_HTTP_URL_START = re.compile(r"https?://", re.IGNORECASE)
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_ERROR_TOKEN = re.compile(r"\S+")
_REDACTED_URL_COMPONENT = "[REDACTED]"
_MAX_ERROR_LENGTH = 500
_MAX_URL_INPUT_LENGTH = 4096
_MAX_ERROR_INPUT_LENGTH = 4096
_MAX_PERCENT_DECODE_ROUNDS = 5
_ENCODED_URL_DELIMITER = re.compile(r"%(?:25)*(?:3a|2f)", re.IGNORECASE)
_ENCODED_RELATIVE_URL_DELIMITER = re.compile(
    r"%(?:25)*(?:2f|3f|23)", re.IGNORECASE
)
_RELATIVE_URL_START = re.compile(
    r"(?:^|[=:'\"(<\[{])(?:/{1,2}|\./|\.\./|\?|#)"
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?P<prefix>(?P<key_quote>['\"]?)\b(?:"
    r"secret|password|passwd|token|api[_-]?key|"
    r"authorization|auth|cookie|signature|credential|private[_-]?key)\b"
    r"(?P=key_quote)\s*[:=]\s*)"
    r"(?P<value>"
    r"(?P<quote>['\"])[^'\"]*(?P=quote)"
    r"|(?:bearer|basic)\s+[^\s,;)\]}]+"
    r"|[^\s,;&?#)\]}]+"
    r")",
    re.IGNORECASE,
)

JSON_V1 = "json_v1"
LEGACY_PICKLE = "legacy_pickle"
REWARD_WIRE_FORMATS = frozenset({JSON_V1, LEGACY_PICKLE})
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def redact_url(url: str, *, max_length: int = _MAX_ERROR_LENGTH) -> str:
    """Return a bounded host-only URL safe for display or persistence."""

    limit = _display_limit(max_length)
    try:
        value = str(url).strip()
    except Exception:  # noqa: BLE001 - display sanitization must fail closed
        return _bounded_text(_REDACTED_URL_COMPONENT, max_length=limit)
    if not value or len(value) > _MAX_URL_INPUT_LENGTH:
        return _bounded_text(_REDACTED_URL_COMPONENT, max_length=limit)

    decoded = _percent_decode(value)
    if _HTTP_URL_START.match(decoded) is None:
        return _bounded_text(_REDACTED_URL_COMPONENT, max_length=limit)
    try:
        parsed = urlsplit(decoded)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except Exception:  # noqa: BLE001 - malformed URLs must not be echoed
        return _bounded_text(_REDACTED_URL_COMPONENT, max_length=limit)
    if scheme not in {"http", "https"} or not hostname:
        return _bounded_text(_REDACTED_URL_COMPONENT, max_length=limit)

    normalized_host = _normalize_hostname(hostname)
    if normalized_host is None:
        return _bounded_text(_REDACTED_URL_COMPONENT, max_length=limit)
    authority = normalized_host if port is None else f"{normalized_host}:{port}"
    return _bounded_text(
        f"{scheme}://{authority}/{_REDACTED_URL_COMPONENT}",
        max_length=limit,
    )


def redact_error_text(error: BaseException | str, *, max_length: int = _MAX_ERROR_LENGTH) -> str:
    """Redact URLs and credential values in bounded error text."""

    limit = _display_limit(max_length)
    try:
        value = str(error)
    except Exception:  # noqa: BLE001 - malicious exception text must fail closed
        value = f"<{type(error).__name__}>"
    value = value[:_MAX_ERROR_INPUT_LENGTH]
    value = _SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED_URL_COMPONENT}",
        value,
    )
    redacted = _ERROR_TOKEN.sub(
        lambda match: _redact_error_token(match.group(0)),
        value,
    )
    return _bounded_text(redacted, max_length=limit)


def _redact_error_token(token: str) -> str:
    decoded = _percent_decode(token)
    match = _HTTP_URL_START.search(decoded)
    if match is not None:
        candidate = decoded[match.start() :].rstrip(".,;!?)]}")
        return redact_url(candidate)
    lowered = decoded.lower()
    if "http" in lowered and _ENCODED_URL_DELIMITER.search(lowered):
        return _REDACTED_URL_COMPONENT
    if _RELATIVE_URL_START.search(decoded) is not None:
        return _REDACTED_URL_COMPONENT
    if _ENCODED_RELATIVE_URL_DELIMITER.search(lowered) is not None:
        return _REDACTED_URL_COMPONENT
    return token


def _percent_decode(value: str) -> str:
    decoded = value
    for _ in range(_MAX_PERCENT_DECODE_ROUNDS):
        try:
            next_value = unquote(decoded, errors="replace")
        except Exception:  # noqa: BLE001 - decoding is best-effort and bounded
            break
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def _normalize_hostname(hostname: str) -> str | None:
    if ":" in hostname:
        try:
            return f"[{ipaddress.IPv6Address(hostname).compressed}]"
        except ValueError:
            return None
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None
    core_hostname = ascii_hostname[:-1] if ascii_hostname.endswith(".") else ascii_hostname
    if not core_hostname or len(ascii_hostname) > 253:
        return None
    if any(_HOST_LABEL.fullmatch(label) is None for label in core_hostname.split(".")):
        return None
    return ascii_hostname


def validate_max_response_bytes(value: Any) -> int:
    """Return a finite positive response limit without accepting bool coercion."""

    if isinstance(value, bool):
        raise ValueError("max_response_bytes must be a finite positive integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "max_response_bytes must be a finite positive integer."
        ) from None
    if resolved <= 0:
        raise ValueError("max_response_bytes must be a finite positive integer.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("max_response_bytes must be a finite positive integer.")
    return resolved


def close_http_response(response: Any) -> None:
    """Close an HTTP response when the transport exposes a close hook."""

    try:
        close = getattr(response, "close", None)
    except Exception:  # noqa: BLE001 - response cleanup must not mask the request result
        return
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - response cleanup must not mask the request result
            pass


def read_bounded_http_response(response: Any, *, max_response_bytes: int) -> bytes:
    """Read a streaming HTTP response without buffering beyond the configured limit."""

    limit = validate_max_response_bytes(max_response_bytes)
    try:
        headers = getattr(response, "headers", None)
        content_length = None
        if headers is not None:
            try:
                content_length = headers.get("Content-Length")
            except Exception:  # noqa: BLE001 - untrusted response headers are advisory
                content_length = None
        if isinstance(content_length, int) and not isinstance(content_length, bool):
            content_length = str(content_length)
        if isinstance(content_length, str) and content_length.strip().isdigit():
            normalized_length = content_length.strip().lstrip("0") or "0"
            if len(normalized_length) > len(str(limit)) or (
                len(normalized_length) == len(str(limit))
                and normalized_length > str(limit)
            ):
                raise RewardProtocolError(
                    "Remote reward response exceeds max_response_bytes."
                )

        iter_content = getattr(response, "iter_content", None)
        if callable(iter_content):
            chunks = iter_content(chunk_size=min(64 * 1024, limit + 1))
        else:
            read = getattr(response, "read", None)
            if not callable(read):
                raise RewardProtocolError(
                    "Remote reward response does not provide a streaming body."
                )

            def read_chunks():
                remaining = limit + 1
                while remaining > 0:
                    chunk = read(min(64 * 1024, remaining))
                    if not chunk:
                        return
                    remaining -= len(chunk)
                    yield chunk

            chunks = read_chunks()

        content = bytearray()
        for chunk in chunks:
            if not isinstance(chunk, (bytes, bytearray)):
                raise RewardProtocolError(
                    "Remote reward response stream must yield bytes."
                )
            if not chunk:
                continue
            if len(content) + len(chunk) > limit:
                raise RewardProtocolError(
                    "Remote reward response exceeds max_response_bytes."
                )
            content.extend(chunk)
        return bytes(content)
    finally:
        close_http_response(response)


def validate_wire_security_policy(
    url: str,
    *,
    wire_format: str,
    allow_unsafe_pickle: bool,
    trusted_hosts: Any,
) -> tuple[str, tuple[str, ...]]:
    """Validate an exact-host opt-in policy for unsafe legacy pickle wires."""

    resolved_wire = str(wire_format)
    if resolved_wire not in REWARD_WIRE_FORMATS:
        raise ValueError(f"wire_format must be one of {sorted(REWARD_WIRE_FORMATS)}.")
    if not isinstance(allow_unsafe_pickle, bool):
        raise TypeError("allow_unsafe_pickle must be a boolean.")
    if isinstance(trusted_hosts, (str, bytes)) or trusted_hosts is None:
        raise TypeError("trusted_hosts must be a sequence of exact hostnames.")
    try:
        raw_hosts = list(trusted_hosts)
    except TypeError:
        raise TypeError(
            "trusted_hosts must be a sequence of exact hostnames."
        ) from None

    normalized_hosts: list[str] = []
    for raw_host in raw_hosts:
        if not isinstance(raw_host, str):
            raise TypeError("trusted_hosts entries must be exact hostname strings.")
        host = raw_host.strip()
        if (
            not host
            or host != raw_host
            or "*" in host
            or host.startswith(".")
            or any(marker in host for marker in ("/", "?", "#", "@"))
        ):
            raise ValueError(
                "trusted_hosts entries must be exact hostnames without wildcards, "
                "suffix rules, credentials, ports, or paths."
            )
        normalized = _normalize_hostname(host)
        if normalized is None:
            raise ValueError(f"trusted_hosts contains an invalid hostname {raw_host!r}.")
        normalized_hosts.append(normalized)
    normalized_tuple = tuple(sorted(set(normalized_hosts)))

    if resolved_wire != LEGACY_PICKLE:
        return resolved_wire, normalized_tuple
    if not allow_unsafe_pickle:
        raise ValueError(
            "legacy_pickle requires explicit allow_unsafe_pickle=true."
        )
    if not normalized_tuple:
        raise ValueError(
            "legacy_pickle requires a non-empty trusted_hosts exact-hostname list."
        )
    try:
        hostname = urlsplit(str(url)).hostname
    except Exception:
        hostname = None
    normalized_url_host = _normalize_hostname(hostname) if hostname else None
    if normalized_url_host not in normalized_tuple:
        raise ValueError(
            "legacy_pickle URL hostname must exactly match an entry in trusted_hosts."
        )
    return resolved_wire, normalized_tuple


def _display_limit(max_length: int) -> int:
    if max_length < 0:
        raise ValueError("max_length must be non-negative.")
    return min(max_length, _MAX_ERROR_LENGTH)


def _bounded_text(value: str, *, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return f"{value[: max_length - 3]}..."


class RewardClient(Protocol):
    name: str

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        ...


@dataclass
class MockRewardClient:
    name: str = "mock"
    mode: str = "prompt_hash"

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        del metadata
        if self.mode == "constant":
            return np.ones(len(prompts), dtype=np.float32), {"mode": self.mode}
        if self.mode == "prompt_media":
            prompt_values = []
            for prompt in prompts:
                digest = stable_hash_text(prompt)
                prompt_values.append((int(digest[:8], 16) % 1000) / 1000.0)
            try:
                import torch

                if isinstance(media, torch.Tensor):
                    media_values = media.float().flatten(1).mean(dim=1).detach().cpu().numpy()
                else:
                    media_values = np.zeros(len(prompts), dtype=np.float32)
            except Exception:  # noqa: BLE001 - mock fallback should be resilient
                media_values = np.zeros(len(prompts), dtype=np.float32)
            values = 0.7 * np.asarray(prompt_values, dtype=np.float32) + 0.3 * media_values.astype(np.float32)
            return values.astype(np.float32), {"mode": self.mode}
        values = []
        for prompt in prompts:
            digest = stable_hash_text(prompt)
            values.append((int(digest[:8], 16) % 1000) / 1000.0)
        return np.asarray(values, dtype=np.float32), {"mode": self.mode}


@dataclass
class RemotePickleRewardClient:
    """Generic pickle-over-HTTP reward client used by the legacy projects."""

    url: str
    name: str = "remote_pickle"
    payload_kind: str = "images"
    timeout: float = 1000.0
    retries: int = 2
    allow_unsafe_pickle: bool = False
    trusted_hosts: tuple[str, ...] = ()
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        _, self.trusted_hosts = validate_wire_security_policy(
            self.url,
            wire_format=LEGACY_PICKLE,
            allow_unsafe_pickle=self.allow_unsafe_pickle,
            trusted_hosts=self.trusted_hosts,
        )
        self.max_response_bytes = validate_max_response_bytes(self.max_response_bytes)
        try:
            self.timeout = float(self.timeout)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "Remote pickle reward timeout must be finite and positive."
            ) from None
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Remote pickle reward timeout must be finite and positive.")
        if isinstance(self.retries, bool) or self.retries < 0:
            raise ValueError("Remote pickle reward retries must be non-negative.")

    def cache_fingerprint(self) -> dict[str, Any]:
        return {
            "client": f"{type(self).__module__}:{type(self).__qualname__}",
            "url": self.url,
            "wire_format": LEGACY_PICKLE,
            "allow_unsafe_pickle": self.allow_unsafe_pickle,
            "trusted_hosts": list(self.trusted_hosts),
            "max_response_bytes": self.max_response_bytes,
            "payload_kind": self.payload_kind,
        }

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        del metadata
        payload = {self.payload_kind: media, "prompts": prompts}
        data = self._request_payload(payload)
        return self._normalize_response(data, expected_count=len(prompts))

    def _request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        import pickle

        payload_bytes = pickle.dumps(payload)
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                response_bytes = _post_bytes(
                    self.url,
                    payload_bytes,
                    timeout=self.timeout,
                    max_response_bytes=self.max_response_bytes,
                )
                try:
                    data = pickle.loads(response_bytes)
                except Exception:
                    data = None
                if data is None:
                    raise RewardProtocolError(
                        "Remote pickle reward response is not a valid pickle payload."
                    ) from None
                if not isinstance(data, dict):
                    raise RewardProtocolError(
                        "Remote pickle reward response must be a mapping."
                    )
                return data
            except RewardProtocolError:
                raise
            except Exception as exc:  # noqa: BLE001 - attach final error to metadata
                last_error = exc
        raise RuntimeError(
            f"Reward client {self.name} failed: {redact_error_text(last_error or '')}"
        ) from None

    def _normalize_response(
        self,
        data: dict[str, Any],
        *,
        expected_count: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if "outputs" not in data:
            raise RewardProtocolError(
                f"Reward client {self.name} response is missing outputs"
            )
        try:
            values = np.asarray(data["outputs"], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RewardProtocolError(
                f"Reward client {self.name} outputs must be numeric"
            ) from exc
        if values.shape != (expected_count,):
            raise RewardProtocolError(
                f"Reward client {self.name} expected {expected_count} scores, "
                f"got shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RewardProtocolError(
                f"Reward client {self.name} returned non-finite scores"
            )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise RewardProtocolError(
                f"Reward client {self.name} metadata must be a mapping"
            )
        return values, dict(metadata)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        return None


def _post_bytes(
    url: str,
    payload: bytes,
    *,
    timeout: float,
    max_response_bytes: int,
) -> bytes:
    try:
        import requests
    except ModuleNotFoundError as exc:
        if exc.name != "requests":
            raise
    else:
        response = requests.post(
            url,
            data=payload,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
        try:
            response.raise_for_status()
        except Exception:
            close_http_response(response)
            raise
        return read_bounded_http_response(
            response, max_response_bytes=max_response_bytes
        )

    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    opener = build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        return read_bounded_http_response(
            response, max_response_bytes=max_response_bytes
        )


def requests_session() -> Any:
    """Create a session that never inherits ambient proxy configuration."""

    try:
        import requests
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError(
            "World-R1 HTTP rewards require the 'requests' package or an injected transport."
        ) from exc
    session = requests.Session()
    session.trust_env = False
    return session


REWARD_CLIENTS.register("mock", MockRewardClient)
REWARD_CLIENTS.register("remote_pickle", RemotePickleRewardClient)
