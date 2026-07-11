from visual_rl.core.types import RolloutBatch, RewardBatch
from visual_rl.core.registry import FEEDBACK_PROVIDERS
from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.router import RewardRouter


class RewardRouterFeedbackProvider(FeedbackProvider):
    def __init__(self, reward_config, cache_dir=None):
        self.reward_router = RewardRouter(reward_config, cache_dir=cache_dir)

    def score(self, batch: RolloutBatch) -> RewardBatch:
        return self.reward_router.score(batch.media, batch.prompts, batch.metadata)


FEEDBACK_PROVIDERS.register("reward_router", RewardRouterFeedbackProvider)
