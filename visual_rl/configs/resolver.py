"""Pure layered resolution for VisualRL experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import os
import re
from typing import Any

from visual_rl.configs.schema import (
    VisualRLConfig,
    config_from_dict,
    config_to_dict,
    external_provider_metadata,
)
from visual_rl.configs.sources import (
    ConfigDocument,
    ExperimentSpec,
    KeyOverride,
    SourceRef,
)


_MISSING = object()
_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_REWARD_MAPPING_ROOTS = ("rewards.weights", "rewards.clients")
_PATH_FIELDS = (
    "model.model_path",
    "dataset.path",
    "evaluation.path",
    "train.lora_path",
    "rewards.cache_dir",
    "runner.rollout_cache_dir",
    "paths.output_dir",
    "paths.pretrained_model",
    "paths.resume_from",
    "model.extra.repo_root",
    "model.extra.world_r1_root",
    "model.extra.flash_grpo_root",
    "model.extra.model_path",
    "model.extra.lora_path",
)


@dataclass(frozen=True)
class ResolvedExperiment:
    """Validated config values and leaf-level winning source information."""

    config: VisualRLConfig
    provenance: dict[str, SourceRef]

    @property
    def values(self) -> dict[str, Any]:
        return config_to_dict(self.config)

    @property
    def resolved(self) -> dict[str, Any]:
        return config_to_dict(self.config)


def _join_path(prefix: str, key: object) -> str:
    text = str(key)
    return f"{prefix}.{text}" if prefix else text


def _is_reward_mapping_path(path: str) -> bool:
    return any(
        path == root or path.startswith(f"{root}.") for root in _REWARD_MAPPING_ROOTS
    )


def _clear_tracking(
    prefix: str,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    dotted_prefix = f"{prefix}."
    for key in list(provenance):
        if key == prefix or key.startswith(dotted_prefix):
            provenance.pop(key, None)
            base_dirs.pop(key, None)


def _record_tree(
    value: Any,
    *,
    prefix: str,
    source: SourceRef,
    base_dir: os.PathLike[str] | None,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    if isinstance(value, Mapping) and value:
        for key, child in value.items():
            _record_tree(
                child,
                prefix=_join_path(prefix, key),
                source=source,
                base_dir=base_dir,
                provenance=provenance,
                base_dirs=base_dirs,
            )
        return
    if prefix:
        provenance[prefix] = source
        base_dirs[prefix] = base_dir


def _normalize_tracking(
    values: Mapping[str, Any],
    *,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    """Keep tracking only for winning leaves and semantic empty mappings."""

    winning_paths: set[str] = set()

    def collect(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping) and value:
            for key, child in value.items():
                collect(child, _join_path(prefix, key))
            return
        if prefix:
            winning_paths.add(prefix)

    collect(values, "")
    for path in list(provenance):
        if path not in winning_paths:
            provenance.pop(path, None)
            base_dirs.pop(path, None)
    for path in list(base_dirs):
        if path not in provenance:
            base_dirs.pop(path, None)


def _replace_value(
    target: dict[str, Any],
    key: str,
    value: Any,
    *,
    path: str,
    source: SourceRef,
    base_dir: os.PathLike[str] | None,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    _clear_tracking(path, provenance, base_dirs)
    target[key] = deepcopy(value)
    _record_tree(
        target[key],
        prefix=path,
        source=source,
        base_dir=base_dir,
        provenance=provenance,
        base_dirs=base_dirs,
    )


def _merge_mapping(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    prefix: str,
    source: SourceRef,
    base_dir: os.PathLike[str] | None,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    if not incoming and prefix and _is_reward_mapping_path(prefix):
        provenance[prefix] = source
        base_dirs[prefix] = base_dir
        return
    if incoming and prefix:
        provenance.pop(prefix, None)
        base_dirs.pop(prefix, None)
    for key, value in incoming.items():
        path = _join_path(prefix, key)
        current = target.get(key, _MISSING)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            if not isinstance(current, dict):
                current = deepcopy(dict(current))
                target[key] = current
            _merge_mapping(
                current,
                value,
                prefix=path,
                source=source,
                base_dir=base_dir,
                provenance=provenance,
                base_dirs=base_dirs,
            )
        else:
            _replace_value(
                target,
                key,
                value,
                path=path,
                source=source,
                base_dir=base_dir,
                provenance=provenance,
                base_dirs=base_dirs,
            )


def _coerce_document(
    value: ConfigDocument | Mapping[str, Any] | None,
    *,
    layer: str,
    context_dir: os.PathLike[str] | None,
) -> ConfigDocument | None:
    if value is None:
        return None
    if isinstance(value, ConfigDocument):
        return value
    if isinstance(value, Mapping):
        return ConfigDocument(
            value,
            SourceRef(kind=layer, name=layer, base_dir=context_dir),
        )
    raise TypeError(f"ExperimentSpec.{layer} must be a ConfigDocument or mapping")


def _apply_key_override(
    values: dict[str, Any],
    override: KeyOverride,
    *,
    source: SourceRef,
    base_dir: os.PathLike[str] | None,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    segments = override.key.split(".")
    if any(not segment for segment in segments):
        raise ValueError(f"Invalid dotted override path: {override.key!r}")

    cursor = values
    for index, segment in enumerate(segments[:-1]):
        current = cursor.get(segment, _MISSING)
        if current is _MISSING:
            cursor[segment] = {}
            current = cursor[segment]
        if not isinstance(current, Mapping):
            traversed = ".".join(segments[: index + 1])
            raise TypeError(
                f"Cannot apply {override.key!r}: {traversed!r} is not a mapping"
            )
        if not isinstance(current, dict):
            current = deepcopy(dict(current))
            cursor[segment] = current
        cursor = current

    final_key = segments[-1]
    path = override.key
    current = cursor.get(final_key, _MISSING)
    if isinstance(current, Mapping) and isinstance(override.value, Mapping):
        if not isinstance(current, dict):
            current = deepcopy(dict(current))
            cursor[final_key] = current
        _merge_mapping(
            current,
            override.value,
            prefix=path,
            source=source,
            base_dir=base_dir,
            provenance=provenance,
            base_dirs=base_dirs,
        )
        return
    _replace_value(
        cursor,
        final_key,
        override.value,
        path=path,
        source=source,
        base_dir=base_dir,
        provenance=provenance,
        base_dirs=base_dirs,
    )


def _fill_schema_defaults(
    values: dict[str, Any],
    defaults: Mapping[str, Any],
    *,
    prefix: str,
    source: SourceRef,
    base_dir: os.PathLike[str] | None,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    for key, default in defaults.items():
        path = _join_path(prefix, key)
        if key not in values:
            values[key] = deepcopy(default)
            _record_tree(
                values[key],
                prefix=path,
                source=source,
                base_dir=base_dir,
                provenance=provenance,
                base_dirs=base_dirs,
            )
            continue
        current = values[key]
        if isinstance(current, Mapping) and isinstance(default, Mapping):
            if not isinstance(current, dict):
                current = deepcopy(dict(current))
                values[key] = current
            marker = provenance.get(path)
            preserve_explicit_empty = (
                marker is not None
                and marker.kind != "schema"
                and _is_reward_mapping_path(path)
            )
            if default and not preserve_explicit_empty:
                provenance.pop(path, None)
                base_dirs.pop(path, None)
            _fill_schema_defaults(
                current,
                default,
                prefix=path,
                source=source,
                base_dir=base_dir,
                provenance=provenance,
                base_dirs=base_dirs,
            )


def _lookup(values: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = values
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _assign_existing(values: dict[str, Any], dotted_path: str, value: Any) -> None:
    segments = dotted_path.split(".")
    current = values
    for segment in segments[:-1]:
        current = current[segment]
    current[segments[-1]] = value


def _normalize_paths(
    values: dict[str, Any],
    *,
    provenance: Mapping[str, SourceRef],
    base_dirs: Mapping[str, os.PathLike[str] | None],
) -> None:
    for dotted_path in _PATH_FIELDS:
        value = _lookup(values, dotted_path)
        if value is _MISSING or value is None or value == "":
            continue
        if not isinstance(value, str):
            continue
        if os.path.isabs(value) or _URL_PATTERN.match(value):
            continue
        base_dir = base_dirs.get(dotted_path)
        if base_dir is None:
            source = provenance.get(dotted_path)
            label = source.label if source is not None else "unknown source"
            raise ValueError(
                f"Relative path at {dotted_path!r} from {label!r} has no base_dir"
            )
        normalized = os.path.normpath(os.path.join(os.fspath(base_dir), value))
        _assign_existing(values, dotted_path, normalized)


def _schema_defaults() -> dict[str, Any]:
    defaults = config_to_dict(VisualRLConfig(run_name=""))
    defaults.pop("run_name")
    return defaults


def _drop_schema_reward_defaults(
    values: dict[str, Any],
    *,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    rewards = values.get("rewards")
    if not isinstance(rewards, Mapping) or rewards.get("replace_defaults") is not True:
        return

    def prune(mapping: dict[str, Any], prefix: str) -> bool:
        removed_schema_value = False
        for key in list(mapping):
            path = _join_path(prefix, key)
            value = mapping[key]
            if isinstance(value, Mapping):
                if not isinstance(value, dict):
                    value = deepcopy(dict(value))
                    mapping[key] = value
                removed_descendant = prune(value, path)
                source = provenance.get(path)
                if not value and (
                    (source is not None and source.kind == "schema")
                    or (source is None and removed_descendant)
                ):
                    mapping.pop(key)
                    _clear_tracking(path, provenance, base_dirs)
                    removed_schema_value = True
                elif value:
                    provenance.pop(path, None)
                    base_dirs.pop(path, None)
                continue
            source = provenance.get(path)
            if source is not None and source.kind == "schema":
                mapping.pop(key)
                _clear_tracking(path, provenance, base_dirs)
                removed_schema_value = True
        return removed_schema_value

    for field_name in ("weights", "clients"):
        root = f"rewards.{field_name}"
        entries = rewards.get(field_name)
        if isinstance(entries, Mapping):
            if not isinstance(entries, dict):
                entries = deepcopy(dict(entries))
                rewards[field_name] = entries
            prune(entries, root)
            source = provenance.get(root)
            if source is not None and (entries or source.kind == "schema"):
                provenance.pop(root, None)
                base_dirs.pop(root, None)


def _canonicalize_external_provider_params(
    values: dict[str, Any],
    *,
    provenance: dict[str, SourceRef],
    base_dirs: dict[str, os.PathLike[str] | None],
) -> None:
    rewards = values.get("rewards")
    if not isinstance(rewards, dict) or rewards.get("provider") == "reward_router":
        return

    metadata = external_provider_metadata(
        str(rewards.get("provider", "")),
        rewards.get("provider_params", {}),
        rewards.get("weights", {}),
    )
    canonical = {
        "target": metadata.target,
        "version": metadata.version,
        "source_sha256": metadata.source_sha256,
        "dependencies": list(metadata.dependencies),
        "params": metadata.params,
        "reward_name": metadata.reward_name,
    }
    if metadata.controls is not None:
        canonical["controls"] = metadata.controls
    rewards["provider_params"] = canonical
    _clear_tracking("rewards.provider_params.weight", provenance, base_dirs)


def _canonicalize_inline_evaluation_prompts(values: dict[str, Any]) -> None:
    """Resolve the inline held-out prompt identity without filesystem access."""

    evaluation = values.get("evaluation")
    if not isinstance(evaluation, dict):
        return
    prompts = evaluation.get("prompts", [])
    if not prompts:
        return
    if not isinstance(prompts, list) or any(
        not isinstance(prompt, str) for prompt in prompts
    ):
        return
    from visual_rl.datasets.prompt_dataset import prompt_content_sha256

    actual = prompt_content_sha256(prompts)
    declared = evaluation.get("content_sha256")
    if declared and str(declared) != actual:
        raise ValueError(
            "evaluation prompt content SHA256 mismatch: "
            f"{actual} != {declared}"
        )
    evaluation["content_sha256"] = actual


def resolve_experiment(spec: ExperimentSpec) -> ResolvedExperiment:
    """Resolve a detached experiment spec without filesystem or runtime effects."""

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")

    context_dir = spec.context_dir
    schema_source = SourceRef(
        kind="schema",
        name="VisualRLConfig defaults",
        base_dir=context_dir,
    )
    defaults = _schema_defaults()
    values = deepcopy(defaults)
    provenance: dict[str, SourceRef] = {}
    base_dirs: dict[str, os.PathLike[str] | None] = {}
    _record_tree(
        values,
        prefix="",
        source=schema_source,
        base_dir=context_dir,
        provenance=provenance,
        base_dirs=base_dirs,
    )

    for layer in ("preset", "recipe", "profile", "user"):
        document = _coerce_document(
            getattr(spec, layer), layer=layer, context_dir=context_dir
        )
        if document is None:
            continue
        _merge_mapping(
            values,
            document.values,
            prefix="",
            source=document.source,
            base_dir=document.source.base_dir,
            provenance=provenance,
            base_dirs=base_dirs,
        )

    for index, override in enumerate(spec.set_overrides):
        if not isinstance(override, KeyOverride):
            raise TypeError(f"set_overrides[{index}] must be a KeyOverride")
        source = override.source or SourceRef(
            kind="set",
            name=f"set_overrides[{index}]",
            base_dir=context_dir,
        )
        _apply_key_override(
            values,
            override,
            source=source,
            base_dir=source.base_dir,
            provenance=provenance,
            base_dirs=base_dirs,
        )

    explicit_values = (spec.explicit, *spec.explicit_documents)
    for index, explicit_value in enumerate(explicit_values):
        explicit = _coerce_document(
            explicit_value,
            layer="explicit",
            context_dir=context_dir,
        )
        if explicit is None:
            continue
        _merge_mapping(
            values,
            explicit.values,
            prefix="",
            source=explicit.source,
            base_dir=explicit.source.base_dir,
            provenance=provenance,
            base_dirs=base_dirs,
        )

    _fill_schema_defaults(
        values,
        defaults,
        prefix="",
        source=schema_source,
        base_dir=context_dir,
        provenance=provenance,
        base_dirs=base_dirs,
    )
    _drop_schema_reward_defaults(
        values,
        provenance=provenance,
        base_dirs=base_dirs,
    )
    _canonicalize_external_provider_params(
        values,
        provenance=provenance,
        base_dirs=base_dirs,
    )
    _canonicalize_inline_evaluation_prompts(values)
    _normalize_tracking(
        values,
        provenance=provenance,
        base_dirs=base_dirs,
    )
    _normalize_paths(values, provenance=provenance, base_dirs=base_dirs)
    config = config_from_dict(values)
    return ResolvedExperiment(
        config=config,
        provenance=dict(provenance),
    )


__all__ = ["ResolvedExperiment", "resolve_experiment"]
