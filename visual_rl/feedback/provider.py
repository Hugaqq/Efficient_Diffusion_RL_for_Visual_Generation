from visual_rl.core.types import RolloutBatch, RewardBatch
from visual_rl.core.registry import FEEDBACK_PROVIDERS
from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.router import RewardRouter


class RewardRouterFeedbackProvider(FeedbackProvider):
    def __init__(self, reward_config, cache_dir=None):
        self.reward_router = RewardRouter(reward_config, cache_dir=cache_dir)

    def score(self, batch: RolloutBatch) -> RewardBatch:
        rewards = self.reward_router.score(
            batch.media,
            batch.prompts,
            batch.metadata,
            sample_id=batch.sample_id,
        )
        if rewards.sample_id is None:
            raise ValueError(
                "RewardRouter returned no sample_id for a rollout-bound reward batch."
            )
        if list(rewards.sample_id) != list(batch.sample_id):
            raise ValueError(
                "RewardRouter sample_id order does not match the rollout batch."
            )
        return rewards


FEEDBACK_PROVIDERS.register("reward_router", RewardRouterFeedbackProvider)
