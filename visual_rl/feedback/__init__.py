"""Pluggable feedback providers and built-in reward clients."""

from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.executor import (
    AsyncRewardExecutor,
    RewardExecutionError,
    RewardExecutor,
    RewardHandle,
    SyncRewardExecutor,
)
from visual_rl.feedback.external import CallableFeedbackProvider
from visual_rl.feedback.factory import build_feedback_provider, build_reward_executor
from visual_rl.feedback.provider import RewardRouterFeedbackProvider
from visual_rl.feedback.router import RewardRouter

__all__ = [
    "AsyncRewardExecutor",
    "CallableFeedbackProvider",
    "FeedbackProvider",
    "RewardExecutionError",
    "RewardExecutor",
    "RewardHandle",
    "RewardRouter",
    "RewardRouterFeedbackProvider",
    "SyncRewardExecutor",
    "build_feedback_provider",
    "build_reward_executor",
]
