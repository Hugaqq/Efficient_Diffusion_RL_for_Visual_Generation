"""Small builtin reward clients and shared HTTP transport primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ipaddress
import re
from typing import Any, ClassVar
from urllib.parse import unquote, urlsplit

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RewardVector,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.feedback.base import RewardClient
from visual_rl.feedback.cache import stable_hash_text

__all__ = [
    "MockRewardClient",
    "RewardTransportError",
    "close_http_response",
    "read_bounded_http_response",
    "redact_error_text",
    "redact_url",
    "requests_session",
    "validate_max_response_bytes",
]


class RewardTransportError(RuntimeError):
    """A reward request failed at the bounded HTTP transport boundary."""


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


def redact_url(url: str, *, max_length: int = _MAX_ERROR_LENGTH) -> str:
    """Return a bounded host-only URL safe for logs and artifacts."""

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
    except Exception:  # noqa: BLE001 - malformed URLs must never be echoed
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


def redact_error_text(
    error: BaseException | str,
    *,
    max_length: int = _MAX_ERROR_LENGTH,
) -> str:
    """Redact URLs and credential values in bounded exception text."""

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
    """Return a positive integer response bound without bool coercion."""

    if type(value) is not int or value <= 0:
        raise ValueError("max_response_bytes must be a positive integer.")
    return value


def close_http_response(response: Any) -> None:
    """Best-effort response cleanup which never masks the primary result."""

    try:
        close = getattr(response, "close", None)
    except Exception:  # noqa: BLE001 - cleanup must not mask the request result
        return
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - cleanup must not mask the request result
            pass


def read_bounded_http_response(response: Any, *, max_response_bytes: int) -> bytes:
    """Read a streaming response without buffering beyond the configured bound."""

    limit = validate_max_response_bytes(max_response_bytes)
    try:
        headers = getattr(response, "headers", None)
        content_length = headers.get("Content-Length") if headers is not None else None
        if isinstance(content_length, int) and not isinstance(content_length, bool):
            content_length = str(content_length)
        if isinstance(content_length, str) and content_length.strip().isdigit():
            normalized_length = content_length.strip().lstrip("0") or "0"
            limit_text = str(limit)
            if len(normalized_length) > len(limit_text) or (
                len(normalized_length) == len(limit_text)
                and normalized_length > limit_text
            ):
                raise RewardTransportError(
                    "Remote reward response exceeds max_response_bytes."
                )

        iter_content = getattr(response, "iter_content", None)
        if not callable(iter_content):
            raise RewardTransportError(
                "Remote reward response does not provide a streaming body."
            )
        content = bytearray()
        for chunk in iter_content(chunk_size=min(64 * 1024, limit + 1)):
            if not isinstance(chunk, (bytes, bytearray)):
                raise RewardTransportError(
                    "Remote reward response stream must yield bytes."
                )
            if not chunk:
                continue
            if len(content) + len(chunk) > limit:
                raise RewardTransportError(
                    "Remote reward response exceeds max_response_bytes."
                )
            content.extend(chunk)
        return bytes(content)
    finally:
        close_http_response(response)


def requests_session() -> Any:
    """Create the sole production HTTP transport without ambient proxies."""

    try:
        import requests
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "World-R1 HTTP rewards require the 'requests' package."
        ) from exc
    session = requests.Session()
    session.trust_env = False
    return session


@dataclass(frozen=True)
class MockRewardClient(RewardClient):
    """Deterministic, dependency-light reward used only by contract tests."""

    mode: str
    name: ClassVar[str] = "mock"
    _MODES: ClassVar[frozenset[str]] = frozenset(
        {"constant", "prompt_hash", "prompt_media"}
    )

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        del context
        _require_exact_keys(raw, {"mode"}, component=cls.name)
        mode = raw["mode"]
        if not isinstance(mode, str) or mode not in cls._MODES:
            raise ValueError(f"mock.mode must be one of {sorted(cls._MODES)}")
        return FrozenMapping({"mode": mode})

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> MockRewardClient:
        del context
        _require_exact_keys(resolved, {"mode"}, component=cls.name)
        mode = resolved["mode"]
        if not isinstance(mode, str) or mode not in cls._MODES:
            raise ValueError(f"mock.mode must be one of {sorted(cls._MODES)}")
        return cls(mode=mode)

    def score(
        self,
        batch: RolloutBatch,
        context: StepContext,
    ) -> RewardVector:
        if batch.context is not context:
            raise ValueError("batch.context must be the identical StepContext")

        import torch

        prompt_values = torch.tensor(
            [
                (int(stable_hash_text(prompt)[:8], 16) % 1000) / 1000.0
                for prompt in batch.prompts
            ],
            dtype=torch.float32,
        )
        if self.mode == "constant":
            values = torch.ones(batch.batch_size, dtype=torch.float32)
        elif self.mode == "prompt_hash":
            values = prompt_values
        else:
            media = torch.as_tensor(batch.media).detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            media_values = media.reshape(batch.batch_size, -1).mean(dim=1)
            values = 0.7 * prompt_values + 0.3 * media_values

        values = values.detach().contiguous()
        return RewardVector(
            sample_id=batch.sample_id,
            values=values,
            shared_metadata={"mode": self.mode},
            sample_metadata=tuple(
                {"prompt_hash": stable_hash_text(prompt)}
                for prompt in batch.prompts
            ),
        )

    def close(self) -> None:
        """Pure client: no owned resource."""


def _require_exact_keys(
    raw: Mapping[str, object],
    expected: set[str],
    *,
    component: str,
) -> None:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{component} params must be a mapping")
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{component} params must contain exactly {sorted(expected)}; "
            f"missing={missing}, unknown={unknown}"
        )


def _display_limit(max_length: int) -> int:
    if type(max_length) is not int or max_length < 0:
        raise ValueError("max_length must be a non-negative integer.")
    return min(max_length, _MAX_ERROR_LENGTH)


def _bounded_text(value: str, *, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return f"{value[: max_length - 3]}..."
