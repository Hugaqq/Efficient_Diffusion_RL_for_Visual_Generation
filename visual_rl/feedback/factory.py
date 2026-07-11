from visual_rl.core.registry import FEEDBACK_PROVIDERS
from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.provider import RewardRouterFeedbackProvider  # noqa: F401


def build_feedback_provider(rewards_config, cache_dir=None, name=None):
    from visual_rl.builtins import register_builtin_plugins

    register_builtin_plugins()
    provider_name = name or getattr(rewards_config, "provider", None)
    if provider_name is None and isinstance(rewards_config, dict):
        provider_name = rewards_config.get("provider")
    provider_cls = FEEDBACK_PROVIDERS.get(provider_name or "reward_router")
    provider_params = getattr(rewards_config, "provider_params", None)
    if provider_params is None and isinstance(rewards_config, dict):
        provider_params = rewards_config.get("provider_params")
    provider = provider_cls(
        rewards_config,
        cache_dir=cache_dir,
        **dict(provider_params or {}),
    )
    if not isinstance(provider, FeedbackProvider):
        raise TypeError(
            f"Feedback provider {provider_name!r} must implement FeedbackProvider"
        )
    return provider
