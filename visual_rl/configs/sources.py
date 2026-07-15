"""Configuration source models and experiment-spec loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import resources
import os
from pathlib import Path
from typing import Any

import yaml


_MISSING = object()
_SOURCE_LAYERS = ("preset", "recipe", "profile", "user")
_ENVELOPE_KEYS = {
    "config",
    "context_dir",
    "explicit",
    "preset",
    "profile",
    "recipe",
    "set",
    "set_overrides",
    "sources",
    "user",
    "version",
}


def _lexical_absolute(
    path: str | os.PathLike[str], base_dir: Path | None = None
) -> Path:
    value = os.fspath(path)
    if not os.path.isabs(value):
        anchor = os.fspath(base_dir) if base_dir is not None else os.getcwd()
        value = os.path.join(anchor, value)
    return Path(os.path.normpath(value))


@dataclass(frozen=True)
class SourceRef:
    """Identifies one configuration source and its relative-path anchor."""

    kind: str
    name: str | None = None
    base_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("SourceRef.kind must be a non-empty string")
        if self.base_dir is not None:
            object.__setattr__(self, "base_dir", _lexical_absolute(self.base_dir))

    @property
    def label(self) -> str:
        return self.name or self.kind


@dataclass(frozen=True)
class ConfigDocument:
    """A mapping together with the source that supplied its values."""

    values: Mapping[str, Any]
    source: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise TypeError("ConfigDocument.values must be a mapping")
        object.__setattr__(self, "values", deepcopy(dict(self.values)))

    @property
    def data(self) -> Mapping[str, Any]:
        return self.values


@dataclass(frozen=True, init=False)
class KeyOverride:
    """One ordered dotted-key override."""

    key: str
    value: Any
    source: SourceRef | None

    def __init__(
        self,
        key: str | None = None,
        value: Any = _MISSING,
        source: SourceRef | None = None,
        *,
        path: str | None = None,
    ) -> None:
        if key is None:
            key = path
        elif path is not None and path != key:
            raise ValueError("KeyOverride key and path disagree")
        if not isinstance(key, str) or not key:
            raise ValueError("KeyOverride.key must be a non-empty string")
        if value is _MISSING:
            raise TypeError("KeyOverride.value is required")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", deepcopy(value))
        object.__setattr__(self, "source", source)

    @property
    def path(self) -> str:
        return self.key


@dataclass(frozen=True)
class ExperimentSpec:
    """All resolver inputs, already detached from source I/O."""

    preset: ConfigDocument | Mapping[str, Any] | None = None
    recipe: ConfigDocument | Mapping[str, Any] | None = None
    profile: ConfigDocument | Mapping[str, Any] | None = None
    user: ConfigDocument | Mapping[str, Any] | None = None
    set_overrides: tuple[KeyOverride, ...] = field(default_factory=tuple)
    explicit: ConfigDocument | Mapping[str, Any] | None = None
    explicit_documents: tuple[ConfigDocument | Mapping[str, Any], ...] = field(
        default_factory=tuple
    )
    context_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_overrides", tuple(self.set_overrides))
        object.__setattr__(
            self, "explicit_documents", tuple(self.explicit_documents)
        )
        if self.context_dir is not None:
            object.__setattr__(self, "context_dir", _lexical_absolute(self.context_dir))


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise TypeError(f"Configuration document {path} must contain a mapping")
    return dict(values)


def _read_file_document(
    value: str | os.PathLike[str], *, layer: str, reference_dir: Path
) -> ConfigDocument:
    path = _lexical_absolute(value, reference_dir)
    return ConfigDocument(
        _read_yaml_mapping(path),
        SourceRef(kind=layer, name=str(path), base_dir=path.parent),
    )


def list_packaged_presets() -> tuple[str, ...]:
    """Return the stable names of presets shipped with the installed package."""

    package_root = resources.files("visual_rl.configs.presets")
    names = {
        Path(resource.name).stem
        for resource in package_root.iterdir()
        if resource.is_file() and resource.name.endswith((".yaml", ".yml"))
    }
    return tuple(sorted(names))


def read_packaged_preset(name: str) -> ConfigDocument:
    """Read one packaged preset by name without depending on the current cwd."""

    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    if Path(filename).name != filename or filename in {".yaml", ".yml"}:
        raise ValueError(f"Invalid package preset name: {name!r}")

    package_root = resources.files("visual_rl.configs.presets")
    resource = package_root.joinpath(filename)
    if not resource.is_file():
        available = ", ".join(list_packaged_presets()) or "<none>"
        raise FileNotFoundError(
            f"Unknown packaged preset {name!r}. Available: {available}"
        )
    values = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    if not isinstance(values, Mapping):
        raise TypeError(f"Packaged preset {name!r} must contain a mapping")

    try:
        resource_path = Path(os.fspath(resource))
    except TypeError:
        base_dir = None
    else:
        base_dir = _lexical_absolute(resource_path.parent)
    return ConfigDocument(
        dict(values),
        SourceRef(
            kind="preset",
            name=f"visual_rl.configs.presets:{filename}",
            base_dir=base_dir,
        ),
    )


def _looks_like_preset_file(value: str) -> bool:
    return (
        os.path.isabs(value)
        or value.startswith(".")
        or "/" in value
        or "\\" in value
        or value.endswith((".yaml", ".yml"))
    )


def _inline_document(
    values: Mapping[str, Any], *, layer: str, source_path: Path, base_dir: Path
) -> ConfigDocument:
    return ConfigDocument(
        values,
        SourceRef(kind=layer, name=str(source_path), base_dir=base_dir),
    )


def _read_layer(
    value: Any,
    *,
    layer: str,
    source_path: Path,
    reference_dir: Path,
    inline_base_dir: Path,
) -> ConfigDocument | None:
    if value is None:
        return None
    if isinstance(value, ConfigDocument):
        return value
    if isinstance(value, Mapping):
        descriptor_keys = {"base_dir", "name", "path", "values"}
        if "values" in value and set(value).issubset(descriptor_keys):
            inline_values = value["values"]
            if not isinstance(inline_values, Mapping):
                raise TypeError(f"{layer}.values must be a mapping")
            declared_base_dir = value.get("base_dir")
            base_dir = (
                inline_base_dir
                if declared_base_dir is None
                else _lexical_absolute(declared_base_dir, reference_dir)
            )
            source = SourceRef(
                kind=layer,
                name=str(value.get("name", source_path)),
                base_dir=base_dir,
            )
            return ConfigDocument(inline_values, source)
        if "path" in value and set(value).issubset(descriptor_keys):
            return _read_file_document(
                value["path"], layer=layer, reference_dir=reference_dir
            )
        return _inline_document(
            value,
            layer=layer,
            source_path=source_path,
            base_dir=inline_base_dir,
        )
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if layer == "preset" and not _looks_like_preset_file(text):
            return read_packaged_preset(text)
        return _read_file_document(text, layer=layer, reference_dir=reference_dir)
    raise TypeError(f"{layer} must be a mapping or document reference")


def _deep_merge_inline(
    lower: Mapping[str, Any], higher: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(lower))
    for key, value in higher.items():
        current = merged.get(key, _MISSING)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_inline(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_set_overrides(
    raw: Any, *, source_path: Path, context_dir: Path
) -> tuple[KeyOverride, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        entries: Sequence[Any] = [
            {"key": key, "value": value} for key, value in raw.items()
        ]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = raw
    else:
        raise TypeError("set_overrides must be a mapping or ordered sequence")

    overrides: list[KeyOverride] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, KeyOverride):
            overrides.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(f"set_overrides[{index}] must be a mapping")
        if "key" in entry or "path" in entry:
            allowed = {"key", "path", "value"}
            unknown = set(entry).difference(allowed)
            if unknown:
                raise ValueError(
                    f"Unknown set_overrides[{index}] fields: {sorted(unknown)}"
                )
            key = entry.get("key", entry.get("path"))
            if "value" not in entry:
                raise ValueError(f"set_overrides[{index}] is missing value")
            value = entry["value"]
        elif len(entry) == 1:
            key, value = next(iter(entry.items()))
        else:
            raise ValueError(f"set_overrides[{index}] needs key/path and value")
        source = SourceRef(
            kind="set",
            name=f"{source_path}#set_overrides[{index}]",
            base_dir=context_dir,
        )
        overrides.append(KeyOverride(key=key, value=value, source=source))
    return tuple(overrides)


def _is_envelope(values: Mapping[str, Any]) -> bool:
    return bool(set(values).intersection(_ENVELOPE_KEYS))


def read_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Read a new source envelope or adapt a legacy full-config YAML file."""

    source_path = _lexical_absolute(path)
    raw = _read_yaml_mapping(source_path)
    source_dir = source_path.parent

    if not _is_envelope(raw):
        values = deepcopy(raw)
        values.setdefault("run_name", source_path.stem)
        return ExperimentSpec(
            user=_inline_document(
                values,
                layer="user",
                source_path=source_path,
                base_dir=source_dir,
            ),
            context_dir=source_dir,
        )

    declared_context = raw.get("context_dir")
    context_dir = (
        source_dir
        if declared_context is None
        else _lexical_absolute(declared_context, source_dir)
    )
    nested_sources = raw.get("sources") or {}
    if not isinstance(nested_sources, Mapping):
        raise TypeError("sources must be a mapping")

    layer_values: dict[str, Any] = {}
    for layer in _SOURCE_LAYERS:
        top_value = raw.get(layer, _MISSING)
        nested_value = nested_sources.get(layer, _MISSING)
        if top_value is not _MISSING and nested_value is not _MISSING:
            raise ValueError(f"{layer} is defined both directly and under sources")
        layer_values[layer] = top_value if top_value is not _MISSING else nested_value

    config_value = raw.get("config", _MISSING)
    if config_value is not _MISSING:
        if layer_values["user"] is not _MISSING:
            raise ValueError("Use only one of user and config")
        layer_values["user"] = config_value

    documents: dict[str, ConfigDocument | None] = {}
    for layer in _SOURCE_LAYERS:
        value = layer_values[layer]
        documents[layer] = (
            None
            if value is _MISSING
            else _read_layer(
                value,
                layer=layer,
                source_path=source_path,
                reference_dir=source_dir,
                inline_base_dir=source_dir,
            )
        )

    set_value = raw.get("set_overrides", _MISSING)
    set_alias = raw.get("set", _MISSING)
    if set_value is not _MISSING and set_alias is not _MISSING:
        raise ValueError("Use only one of set_overrides and set")
    if set_value is _MISSING:
        set_value = None if set_alias is _MISSING else set_alias
    set_overrides = _parse_set_overrides(
        set_value, source_path=source_path, context_dir=context_dir
    )

    residual = {
        key: deepcopy(value) for key, value in raw.items() if key not in _ENVELOPE_KEYS
    }
    explicit_value = raw.get("explicit")
    if explicit_value is None:
        explicit_values = residual
    else:
        if not isinstance(explicit_value, Mapping):
            raise TypeError("explicit must be a mapping")
        explicit_values = _deep_merge_inline(residual, explicit_value)
    explicit = (
        _inline_document(
            explicit_values,
            layer="explicit",
            source_path=source_path,
            base_dir=context_dir,
        )
        if explicit_values
        else None
    )

    return ExperimentSpec(
        preset=documents["preset"],
        recipe=documents["recipe"],
        profile=documents["profile"],
        user=documents["user"],
        set_overrides=set_overrides,
        explicit=explicit,
        context_dir=context_dir,
    )


__all__ = [
    "ConfigDocument",
    "ExperimentSpec",
    "KeyOverride",
    "SourceRef",
    "list_packaged_presets",
    "read_experiment_spec",
    "read_packaged_preset",
]
