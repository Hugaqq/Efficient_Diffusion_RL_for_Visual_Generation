"""Resolve one complete YAML mapping into the canonical VisualRL config."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from visual_rl.configs.schema import VisualRLConfig, config_from_mapping
from visual_rl.core.components import CAPABILITY_OWNER, ComponentSpec
from visual_rl.core.types import FrozenMapping, ResolutionContext
from visual_rl.errors import ComponentError, ConfigError

__all__ = ("resolve_config",)


def resolve_config(
    values: Mapping[str, Any],
    context: ResolutionContext,
) -> VisualRLConfig:
    """Resolve component parameters and return one deeply immutable config.

    ``values`` is the duplicate-key-free mapping frozen by ``vr.load()``.
    This function never reads the YAML again, stats a path, imports training
    frameworks, or constructs runtime objects.
    """

    if not isinstance(context, ResolutionContext):
        raise TypeError("context must be a ResolutionContext")
    if not context.config_path.is_absolute() or not context.config_dir.is_absolute():
        raise ValueError("ResolutionContext paths must be absolute")
    if not isinstance(values, Mapping):
        raise ConfigError("YAML root must be a mapping", key="<root>")

    # Global schema first: unknown/missing/legacy fields fail before component
    # selection. Component params remain immutable raw values at this pass.
    raw_config = config_from_mapping(values, config_dir=context.config_dir)

    # The concrete manifest is imported only when resolution actually begins;
    # importing visual_rl and vr.load() therefore do not import component
    # modules.
    from visual_rl.builtins import get_builtin_component

    model_spec = get_builtin_component("model", raw_config.model.name)
    rollout_spec = get_builtin_component("rollout", raw_config.rollout.name)
    algorithm_spec = get_builtin_component("algorithm", raw_config.algorithm.name)
    reward_specs = tuple(
        get_builtin_component("reward", item.name)
        for item in raw_config.reward.components
    )

    model_params = _resolve_params(
        model_spec, raw_config.model.params, context, key="model.params"
    )
    rollout_params = _resolve_params(
        rollout_spec, raw_config.rollout.params, context, key="rollout.params"
    )
    reward_params = tuple(
        _resolve_params(
            spec,
            item.params,
            context,
            key=f"reward.components[{index}].params",
        )
        for index, (spec, item) in enumerate(
            zip(reward_specs, raw_config.reward.components, strict=True)
        )
    )
    algorithm_params = _resolve_params(
        algorithm_spec,
        raw_config.algorithm.params,
        context,
        key="algorithm.params",
    )

    resolved_values = _replace_component_params(
        values,
        model_params=model_params,
        rollout_params=rollout_params,
        reward_params=reward_params,
        algorithm_params=algorithm_params,
    )
    config = config_from_mapping(resolved_values, config_dir=context.config_dir)
    _validate_capabilities(
        config,
        model_spec=model_spec,
        rollout_spec=rollout_spec,
        reward_specs=reward_specs,
        algorithm_spec=algorithm_spec,
    )
    _validate_group_size(config, rollout_spec, algorithm_spec)
    return config


def _resolve_params(
    spec: ComponentSpec,
    raw: Mapping[str, Any],
    context: ResolutionContext,
    *,
    key: str,
) -> FrozenMapping:
    resolver = getattr(spec.factory, "resolve_params", None)
    if not callable(resolver):
        raise ComponentError(
            f"{spec.kind} component {spec.name!r} does not implement "
            "resolve_params()",
            kind=spec.kind,
            name=spec.name,
        )
    try:
        resolved = resolver(raw, context)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{key}: {exc}",
            key=key,
            path=str(context.config_path),
        ) from exc
    if not isinstance(resolved, Mapping):
        raise ComponentError(
            f"{spec.kind} component {spec.name!r} returned non-mapping params",
            kind=spec.kind,
            name=spec.name,
        )
    try:
        return (
            resolved
            if isinstance(resolved, FrozenMapping)
            else FrozenMapping(resolved)
        )
    except (TypeError, ValueError) as exc:
        raise ComponentError(
            f"{spec.kind} component {spec.name!r} returned invalid params: {exc}",
            kind=spec.kind,
            name=spec.name,
        ) from exc


def _replace_component_params(
    values: Mapping[str, Any],
    *,
    model_params: FrozenMapping,
    rollout_params: FrozenMapping,
    reward_params: tuple[FrozenMapping, ...],
    algorithm_params: FrozenMapping,
) -> FrozenMapping:
    """Defensively rebuild the raw tree with only resolved params replaced."""

    root = dict(values)
    model = dict(root["model"])
    model["params"] = model_params
    root["model"] = model

    rollout = dict(root["rollout"])
    rollout["params"] = rollout_params
    root["rollout"] = rollout

    reward = dict(root["reward"])
    components = []
    for raw_item, resolved in zip(
        reward["components"], reward_params, strict=True
    ):
        item = dict(raw_item)
        item["params"] = resolved
        components.append(item)
    reward["components"] = components
    root["reward"] = reward

    algorithm = dict(root["algorithm"])
    algorithm["params"] = algorithm_params
    root["algorithm"] = algorithm
    return FrozenMapping(root)


def _validate_capabilities(
    config: VisualRLConfig,
    *,
    model_spec: ComponentSpec,
    rollout_spec: ComponentSpec,
    reward_specs: tuple[ComponentSpec, ...],
    algorithm_spec: ComponentSpec,
) -> None:
    selections = (
        (model_spec, config.model.params),
        (rollout_spec, config.rollout.params),
        *(
            (spec, item.params)
            for spec, item in zip(
                reward_specs, config.reward.components, strict=True
            )
        ),
        (algorithm_spec, config.algorithm.params),
    )
    by_kind: dict[str, tuple[ComponentSpec, ...]] = {}
    for spec, _params in selections:
        by_kind.setdefault(spec.kind, ())
        by_kind[spec.kind] = (*by_kind[spec.kind], spec)

    for spec, params in selections:
        requirement_resolver = getattr(
            spec.factory, "required_capabilities", None
        )
        if not callable(requirement_resolver):
            raise ComponentError(
                f"{spec.kind} component {spec.name!r} does not implement "
                "required_capabilities()",
                kind=spec.kind,
                name=spec.name,
            )
        try:
            conditional = requirement_resolver(params)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{spec.kind} component {spec.name!r} has invalid conditional "
                f"capabilities: {exc}",
                key=f"{spec.kind}.params",
            ) from exc
        if not isinstance(conditional, frozenset):
            raise ComponentError(
                f"{spec.kind} component {spec.name!r} "
                "required_capabilities() must return frozenset",
                kind=spec.kind,
                name=spec.name,
            )
        for capability in spec.requires | conditional:
            owner_kind = CAPABILITY_OWNER.get(capability)
            if owner_kind is None:
                raise ComponentError(
                    f"{spec.kind} component {spec.name!r} requires unknown "
                    f"capability {capability!r}",
                    kind=spec.kind,
                    name=spec.name,
                )
            providers = by_kind.get(owner_kind, ())
            if not any(capability in provider.provides for provider in providers):
                raise ComponentError(
                    f"{spec.kind} component {spec.name!r} requires "
                    f"{capability!r}, but selected {owner_kind} component "
                    "does not provide it",
                    kind=spec.kind,
                    name=spec.name,
                )


def _validate_group_size(
    config: VisualRLConfig,
    rollout_spec: ComponentSpec,
    algorithm_spec: ComponentSpec,
) -> None:
    if "rollout.branching" in rollout_spec.provides:
        key = "branch_count"
    else:
        key = "samples_per_prompt"
    try:
        group_size = config.rollout.params[key]
    except KeyError as exc:
        raise ConfigError(
            f"rollout.params.{key} is required to derive group size",
            key=f"rollout.params.{key}",
        ) from exc
    if type(group_size) is not int or group_size <= 0:
        raise ConfigError(
            f"rollout.params.{key} must be a positive integer",
            key=f"rollout.params.{key}",
        )
    minimum = getattr(algorithm_spec.factory, "MIN_GROUP_SIZE", None)
    if type(minimum) is not int or minimum < 1:
        raise ComponentError(
            f"algorithm component {algorithm_spec.name!r} has invalid "
            "MIN_GROUP_SIZE",
            kind="algorithm",
            name=algorithm_spec.name,
        )
    if group_size < minimum:
        raise ConfigError(
            f"resolved group size {group_size} is smaller than "
            f"{algorithm_spec.name}.MIN_GROUP_SIZE={minimum}",
            key=f"rollout.params.{key}",
        )
