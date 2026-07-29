"""Strict JSON and redacted config helpers for artifact boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from visual_rl.core.types import to_plain_dict


_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:"
    r"secret|password|passwd|token|api[_-]?key|authorization|auth|cookie|"
    r"credential|private[_-]?key"
    r")(?:$|[_-])",
    re.IGNORECASE,
)


def canonical_json_text(value: Any) -> str:
    """Serialize an already validated plain/frozen value deterministically."""

    return json.dumps(
        to_plain_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def strict_json_load(path: str | Path) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    source = Path(path)

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read strict JSON from {source}: {error}") from error


def redact_artifact_config(value: Any) -> Any:
    """Return a plain config copy without credentials or URL path/query data."""

    return _redact_plain(to_plain_dict(value), key=None)


def _redact_plain(value: Any, *, key: str | None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return _REDACTED
    if isinstance(value, dict):
        return {
            item_key: _redact_plain(item, key=item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_plain(item, key=key) for item in value]
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
