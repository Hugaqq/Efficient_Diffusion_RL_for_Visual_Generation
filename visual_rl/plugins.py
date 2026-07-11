"""Small public registration API for VisualRL extension points.

Construction contracts are intentionally explicit:

- model adapter class: ``Adapter(model_config)``
- rollout engine class: ``Engine(rollout_config)``
- feedback provider class: ``Provider(reward_config, cache_dir=..., **provider_params)``
- reward client class: ``Client(**client_params)``
- algorithm class: ``from_config(algorithm_config)`` or ``Algorithm(**config)``
- optimizer plugin builder: ``builder(visual_rl_config) -> OptimizerPlugin``
"""

from __future__ import annotations

from typing import Any

from visual_rl.core.registry import (
    ALGORITHMS,
    FEEDBACK_PROVIDERS,
    MODEL_ADAPTERS,
    OPTIMIZER_PLUGINS,
    REWARD_CLIENTS,
    ROLLOUT_ENGINES,
)


def register_model_adapter(name: str, adapter: Any | None = None):
    return MODEL_ADAPTERS.register(name, adapter)


def register_rollout_engine(name: str, engine: Any | None = None):
    return ROLLOUT_ENGINES.register(name, engine)


def register_feedback_provider(name: str, provider: Any | None = None):
    return FEEDBACK_PROVIDERS.register(name, provider)


def register_reward_client(name: str, client: Any | None = None):
    return REWARD_CLIENTS.register(name, client)


def register_algorithm(name: str, algorithm: Any | None = None):
    return ALGORITHMS.register(name, algorithm)


def register_optimizer_plugin(name: str, plugin: Any | None = None):
    return OPTIMIZER_PLUGINS.register(name, plugin)


__all__ = [
    "register_algorithm",
    "register_feedback_provider",
    "register_model_adapter",
    "register_optimizer_plugin",
    "register_reward_client",
    "register_rollout_engine",
]
