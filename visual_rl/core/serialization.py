"""Strict projection of immutable core values to plain artifact values."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

__all__ = (
    "canonical_json_text",
    "redact_artifact_config",
    "strict_json_load",
    "to_plain_dict",
)

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:"
    r"secret|password|passwd|token|api[_-]?key|authorization|auth|cookie|"
    r"credential|private[_-]?key"
    r")(?:$|[_-])",
    re.IGNORECASE,
)


def _reject_non_plain(value: Any) -> None:
    if isinstance(value, (set, frozenset)):
        raise TypeError("to_plain_dict does not accept set/frozenset values")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("to_plain_dict does not accept binary values")
    if callable(value):
        raise TypeError("to_plain_dict does not accept callables")
    root_module = type(value).__module__.partition(".")[0]
    type_name = type(value).__name__
    if root_module == "torch" and type_name == "Tensor":
        raise TypeError("to_plain_dict does not accept torch.Tensor")
    if root_module == "numpy" and type_name == "ndarray":
        raise TypeError("to_plain_dict does not accept numpy.ndarray")


def to_plain_dict(value: Any) -> Any:
    """Project validated immutable values without stringifying unknown types."""

    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(value, "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            raise TypeError("to_plain_dict only accepts frozen dataclass instances")
        projected = {}
        for item in fields(value):
            plain_name = item.metadata.get("plain_name", item.name)
            if not isinstance(plain_name, str) or not plain_name:
                raise TypeError("dataclass field plain_name must be a non-empty string")
            if plain_name in projected:
                raise ValueError(
                    f"dataclass projection contains duplicate key {plain_name!r}"
                )
            projected[plain_name] = to_plain_dict(getattr(value, item.name))
        return projected
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("to_plain_dict mapping keys must be strings")
            projected[key] = to_plain_dict(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("to_plain_dict does not accept non-finite floats")
        return value
    if isinstance(value, Path):
        return str(value)
    _reject_non_plain(value)
    raise TypeError(f"to_plain_dict does not accept {type(value).__name__}")


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
