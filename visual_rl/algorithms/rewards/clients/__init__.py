"""Native reward clients operating on the v0.8 typed reward batch."""

from visual_rl.algorithms.rewards.clients.image import (
    PromptColorGuardedRewardClient,
    PromptColorMarginRewardClient,
    PromptColorRewardClient,
)
from visual_rl.algorithms.rewards.clients.mock import MockRewardClient
from visual_rl.algorithms.rewards.clients.world_r1 import (
    WORLD_R1_RESOURCE_PROTOCOL,
    WorldR1HealthAttestation,
    WorldR1Reward3DClient,
    WorldR1RewardGeneralClient,
)

__all__ = (
    "MockRewardClient",
    "PromptColorGuardedRewardClient",
    "PromptColorMarginRewardClient",
    "PromptColorRewardClient",
    "WORLD_R1_RESOURCE_PROTOCOL",
    "WorldR1HealthAttestation",
    "WorldR1Reward3DClient",
    "WorldR1RewardGeneralClient",
)
