"""Pluggable feedback providers and built-in reward clients."""

from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.factory import build_feedback_provider
from visual_rl.feedback.provider import RewardRouterFeedbackProvider
from visual_rl.feedback.router import RewardRouter

__all__ = [
    "FeedbackProvider",
    "RewardRouter",
    "RewardRouterFeedbackProvider",
    "build_feedback_provider",
]
