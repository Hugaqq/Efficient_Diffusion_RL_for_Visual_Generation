"""Strict, side-effect-free source loader for VisualRL recipes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from visual_rl.composition.config.errors import (
    ConfigSourceError,
    _DuplicateKeyYAMLError,
)
from visual_rl.core.types import FrozenMapping, ResolutionContext

__all__ = ("SourceRecipe", "load_source_recipe")


@dataclass(frozen=True)
class SourceRecipe:
    """One immutable snapshot of the exact user-supplied YAML source."""

    path: Path
    text: str
    values: FrozenMapping
    config_source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("SourceRecipe.path must be an absolute Path")
        if not isinstance(self.text, str):
            raise TypeError("SourceRecipe.text must be a string")
        if not isinstance(self.values, FrozenMapping):
            raise TypeError("SourceRecipe.values must be a FrozenMapping")
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.config_source_id != expected:
            raise ValueError("config_source_id must hash the exact UTF-8 source")

    @property
    def context(self) -> ResolutionContext:
        """Return the sole path-normalization context for this snapshot."""

        return ResolutionContext(config_path=self.path, config_dir=self.path.parent)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise _DuplicateKeyYAMLError(
                key,
                key_node.start_mark.line + 1,
                key_node.start_mark.column + 1,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_source_recipe(path: str | Path) -> SourceRecipe:
    """Read and freeze exactly one complete UTF-8 YAML file.

    This boundary performs source parsing only. It deliberately does not
    select a recipe, resolve components, inspect the environment, or construct
    runtime objects.
    """

    config_path = _absolute_config_path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as exc:
        raise ConfigSourceError(
            f"Cannot read configuration file: {config_path}",
            code="config.source_read",
            path=str(config_path),
        ) from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigSourceError(
            f"Configuration file must be UTF-8: {config_path}",
            code="config.source_encoding",
            path=str(config_path),
        ) from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except _DuplicateKeyYAMLError as exc:
        raise ConfigSourceError(
            f"Invalid YAML in {config_path}: {exc}",
            code="config.duplicate_key",
            key=str(exc.key),
            path=str(config_path),
            line=exc.line,
            column=exc.column,
        ) from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        raise ConfigSourceError(
            f"Invalid YAML in {config_path}: {exc}",
            code="config.invalid_yaml",
            path=str(config_path),
            line=None if mark is None else mark.line + 1,
            column=None if mark is None else mark.column + 1,
        ) from exc
    if not isinstance(raw, Mapping):
        raise ConfigSourceError(
            "Configuration YAML root must be a mapping",
            code="config.root_not_mapping",
            key="<root>",
            path=str(config_path),
        )
    try:
        snapshot = FrozenMapping(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigSourceError(
            f"Configuration must contain plain YAML values: {exc}",
            code="config.non_plain_value",
            path=str(config_path),
        ) from exc
    return SourceRecipe(
        path=config_path,
        text=text,
        values=snapshot,
        config_source_id=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _absolute_config_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("configuration path must be str or Path")
    return Path(value).expanduser().resolve(strict=False)
