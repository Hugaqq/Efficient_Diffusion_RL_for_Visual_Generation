"""JSON-safe conversion for experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit


_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:"
    r"secret|password|passwd|token|api[_-]?key|authorization|auth|cookie|"
    r"credential|private[_-]?key"
    r")(?:$|[_-])",
    re.IGNORECASE,
)


def to_jsonable(value: Any) -> Any:
    """Detach tensor-like values and recursively convert them to JSON data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__qualname__),
        )
        return {"callable": f"{module}.{qualname}"}
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "tolist"):
        return to_jsonable(converted.tolist())
    if hasattr(converted, "item"):
        try:
            return to_jsonable(converted.item())
        except (TypeError, ValueError):
            pass

    raise TypeError(f"Unsupported artifact value type: {type(value).__name__}")


def redact_artifact_config(value: Any) -> Any:
    """Return a JSON-safe config copy without persisted credentials or URL paths."""

    return _redact_jsonable(to_jsonable(value), key=None)


def _redact_jsonable(value: Any, *, key: str | None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return _REDACTED
    if isinstance(value, dict):
        return {
            str(item_key): _redact_jsonable(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_jsonable(item, key=key) for item in value]
    if isinstance(value, str) and _looks_like_http_url(value):
        return _redact_url(value)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
    if normalized.endswith("_env"):
        return False
    return _SENSITIVE_KEY.search(normalized) is not None


def _looks_like_http_url(value: str) -> bool:
    candidate = value.strip()
    for _ in range(5):
        if candidate.lower().startswith(("http://", "https://")):
            return True
        decoded = unquote(candidate)
        if decoded == candidate:
            break
        candidate = decoded
    return False


def _redact_url(value: str) -> str:
    candidate = value.strip()
    for _ in range(5):
        if candidate.lower().startswith(("http://", "https://")):
            break
        decoded = unquote(candidate)
        if decoded == candidate:
            return _REDACTED
        candidate = decoded
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not hostname:
            return _REDACTED
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme.lower(), host, "", "", ""))
    except ValueError:
        return _REDACTED
