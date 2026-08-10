"""Structured errors owned by the v0.8 configuration front-end."""

from __future__ import annotations

from typing import Any

import yaml

from visual_rl.errors import ConfigError

__all__ = ("ConfigMigrationError", "ConfigSourceError")


class ConfigMigrationError(ConfigError):
    """A retired config schema that requires an explicit offline migration."""

    def __init__(
        self,
        *,
        source_schema_version: int,
        required_schema_version: int,
        path: str,
    ) -> None:
        for name, value in (
            ("source_schema_version", source_schema_version),
            ("required_schema_version", required_schema_version),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if source_schema_version == required_schema_version:
            raise ValueError("source and required schema versions must differ")
        super().__init__(
            f"schema_version {source_schema_version} is retired and cannot be "
            "executed by the v0.8 production entry; start from a schema-v2 "
            "example in configs/v2/ or migrate this file offline. Runtime "
            "legacy parsing is intentionally unavailable.",
            key="schema_version",
            path=path,
        )
        self.code = "config.schema_v1_migration_required"
        self.source_schema_version = source_schema_version
        self.required_schema_version = required_schema_version
        self.migration_examples = "configs/v2/"
        self.migration_mode = "offline_only"


class ConfigSourceError(ConfigError):
    """A source decoding or YAML parsing error with stable diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        path: str,
        key: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message, key=key, path=path)
        if not isinstance(code, str) or not code:
            raise ValueError("configuration error code must be a non-empty string")
        for name, value in (("line", line), ("column", column)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        self.code = code
        self.line = line
        self.column = column


class _DuplicateKeyYAMLError(yaml.YAMLError):
    """Internal PyYAML control-flow error; converted at the loader boundary."""

    def __init__(self, key: Any, line: int, column: int) -> None:
        super().__init__(
            f"duplicate mapping key {key!r} at line {line}, column {column}"
        )
        self.key = key
        self.line = line
        self.column = column
