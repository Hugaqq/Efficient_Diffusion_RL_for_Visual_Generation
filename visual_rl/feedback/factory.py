from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import inspect
import math
from pathlib import Path
import re
from typing import Any

from visual_rl.configs.schema import RewardExecutorConfig, external_provider_metadata
from visual_rl.core.registry import FEEDBACK_PROVIDERS
from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.external import CallableFeedbackProvider
from visual_rl.feedback.executor import AsyncRewardExecutor, SyncRewardExecutor
from visual_rl.feedback.provider import RewardRouterFeedbackProvider  # noqa: F401


_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BUILTIN_PROVIDER_NAMES = frozenset({"reward_router"})
_EXECUTOR_DEFAULTS = {
    "mode": "sync",
    "max_workers": 4,
    "microbatch_size": 1,
    "timeout_s": 30.0,
    "max_retries": 0,
    "submit_timeout_s": 30.0,
    "max_in_flight": None,
    "require_hard_timeout": False,
}


def build_feedback_provider(rewards_config, cache_dir=None, name=None):
    from visual_rl.builtins import register_builtin_plugins

    register_builtin_plugins()
    provider_name = name or getattr(rewards_config, "provider", None)
    if provider_name is None and isinstance(rewards_config, dict):
        provider_name = rewards_config.get("provider")
    provider_name = provider_name or "reward_router"
    provider_params = getattr(rewards_config, "provider_params", None)
    if provider_params is None and isinstance(rewards_config, dict):
        provider_params = rewards_config.get("provider_params")
    provider_params = {} if provider_params is None else provider_params
    if provider_name in _BUILTIN_PROVIDER_NAMES:
        provider_cls = FEEDBACK_PROVIDERS.get(provider_name)
        provider = provider_cls(
            rewards_config,
            cache_dir=cache_dir,
            **dict(provider_params),
        )
        return _require_provider(provider, provider_name)

    weights = getattr(rewards_config, "weights", None)
    if weights is None and isinstance(rewards_config, dict):
        weights = rewards_config.get("weights")
    metadata = external_provider_metadata(provider_name, provider_params, weights)
    component = _resolve_external_target(metadata.target)
    source_sha256 = _verify_source_sha256(component, metadata.source_sha256)

    if inspect.isclass(component):
        if not issubclass(component, FeedbackProvider):
            raise TypeError("External feedback classes must subclass FeedbackProvider")
        instance = component(
            rewards_config,
            cache_dir=cache_dir,
            **metadata.params,
        )
        _require_provider(instance, provider_name)
        component = instance

    provider = CallableFeedbackProvider(
        component,
        name=metadata.reward_name,
        version=metadata.version,
        params=metadata.params,
        weight=metadata.weight,
        cache_dir=cache_dir,
        target=metadata.target,
        source_sha256=source_sha256,
    )
    return _require_provider(provider, provider_name)


def build_reward_executor(provider: FeedbackProvider, config=None):
    """Build the runtime scoring strategy without changing provider semantics."""

    provider = _require_provider(provider, getattr(provider, "name", "unknown"))
    values = _config_values(config)
    mode = values.pop("mode")
    if mode == "sync":
        require_hard_timeout = values.pop("require_hard_timeout")
        provider_requires_hard_timeout = getattr(
            provider,
            "requires_hard_timeout",
            False,
        )
        if not isinstance(provider_requires_hard_timeout, bool):
            raise TypeError("provider requires_hard_timeout must be a bool")
        if require_hard_timeout or provider_requires_hard_timeout:
            raise ValueError(
                "Synchronous reward execution cannot provide a hard timeout; "
                "use a process-isolated reward provider"
            )
        return SyncRewardExecutor(provider)
    return AsyncRewardExecutor(provider, **values)


def _config_values(config) -> dict[str, Any]:
    fields = frozenset(_EXECUTOR_DEFAULTS)
    if config is None:
        supplied: dict[str, Any] = {}
    if isinstance(config, Mapping):
        unknown = sorted(str(key) for key in set(config).difference(fields))
        if unknown:
            raise ValueError(f"Unknown reward executor fields: {unknown}")
        supplied = dict(config)
    elif isinstance(config, RewardExecutorConfig):
        unknown = sorted(name for name in vars(config) if name not in fields)
        if unknown:
            raise ValueError(f"Unknown reward executor fields: {unknown}")
        missing = sorted(name for name in fields if not hasattr(config, name))
        if missing:
            raise ValueError(f"Missing reward executor fields: {missing}")
        supplied = {name: getattr(config, name) for name in fields}
    elif config is not None:
        raise TypeError("reward executor config must be a mapping or typed config")
    values = {**_EXECUTOR_DEFAULTS, **supplied}
    _validate_executor_values(values)
    return values


def _validate_executor_values(values: Mapping[str, Any]) -> None:
    if values["mode"] not in {"sync", "async"}:
        raise ValueError("reward executor mode must be one of: sync, async")
    for name in ("max_workers", "microbatch_size"):
        _require_positive_int(name, values[name])
    _require_non_negative_int("max_retries", values["max_retries"])
    _require_positive_float("timeout_s", values["timeout_s"])
    _require_non_negative_float("submit_timeout_s", values["submit_timeout_s"])
    max_in_flight = values["max_in_flight"]
    if max_in_flight is not None:
        _require_positive_int("max_in_flight", max_in_flight)
    if not isinstance(values["require_hard_timeout"], bool):
        raise TypeError("require_hard_timeout must be a bool")


def _require_positive_int(name: str, value: Any) -> None:
    _require_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_float(name: str, value: Any) -> None:
    _require_non_negative_float(name, value)
    if float(value) == 0.0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_float(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_provider(provider: Any, provider_name: str) -> FeedbackProvider:
    if not isinstance(provider, FeedbackProvider):
        raise TypeError(
            f"Feedback provider {provider_name!r} must implement FeedbackProvider"
        )
    return provider


def _resolve_external_target(target: Any) -> Any:
    if not isinstance(target, str) or not _TARGET_PATTERN.fullmatch(target):
        raise ValueError(
            f"Invalid external feedback target {target!r}; expected module:attribute"
        )
    module_name, attribute_path = target.split(":", 1)
    component: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        component = getattr(component, attribute)
    if not callable(component):
        raise TypeError(f"External feedback target {target!r} is not callable")
    if inspect.ismethod(component):
        raise TypeError("External feedback target must not be a bound method")
    if inspect.isfunction(component):
        if component.__name__ == "<lambda>":
            raise TypeError("Lambda feedback targets are not supported")
        if "." in component.__qualname__:
            raise TypeError("Feedback functions must be defined at module level")
    elif inspect.isclass(component):
        if "<locals>" in component.__qualname__:
            raise TypeError("Local feedback classes are not supported")
    else:
        qualname = type(component).__qualname__
        if "<locals>" in qualname:
            raise TypeError("Local callable feedback objects are not supported")
    return component


def _verify_source_sha256(component: Any, declared: Any) -> str:
    if not isinstance(declared, str):
        raise ValueError("External feedback provider requires source_sha256")
    normalized = declared.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("External feedback source_sha256 must be 64 hex characters")
    source = _source_path(component)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != normalized:
        raise RuntimeError(f"External feedback source SHA256 mismatch for {source}")
    return actual


def _source_path(component: Any) -> Path:
    inspected = (
        component
        if inspect.isfunction(component) or inspect.isclass(component)
        else type(component)
    )
    source_name = inspect.getsourcefile(inspected)
    if source_name is None:
        raise ValueError("External feedback component has no trusted source file")
    source = Path(source_name).resolve()
    if not source.is_file():
        raise ValueError(f"External feedback source file does not exist: {source}")
    return source
