"""Explicit registration of VisualRL's trusted built-in components."""

from __future__ import annotations


def register_builtin_plugins() -> None:
    """Import and register every repository-local built-in explicitly."""

    from visual_rl.core.registry import (
        ALGORITHMS,
        FEEDBACK_PROVIDERS,
        MODEL_ADAPTERS,
        OPTIMIZER_PLUGINS,
        REWARD_CLIENTS,
        ROLLOUT_ENGINES,
    )
    from visual_rl.feedback.clients import MockRewardClient, RemotePickleRewardClient
    from visual_rl.feedback.image_rewards import (
        PromptColorGuardedRewardClient,
        PromptColorMarginRewardClient,
        PromptColorRewardClient,
    )
    from visual_rl.feedback.pickscore import PickScoreRewardClient
    from visual_rl.feedback.provider import RewardRouterFeedbackProvider
    from visual_rl.feedback.video_hpsv3 import VideoHPSv3RewardClient
    from visual_rl.feedback.world_r1_rewards import (
        WorldR1Reward3DClient,
        WorldR1RewardGeneralClient,
    )
    from visual_rl.model_adapters.mock import MockWanAdapter
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
    from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
    from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter
    from visual_rl.optimizers.factory import _build_algorithm_optimizer
    from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
    from visual_rl.optimizers.grpo import GRPOAlgorithm
    from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm
    from visual_rl.rollout.branching import BranchingRollout
    from visual_rl.rollout.full_trajectory import FullTrajectoryRollout
    from visual_rl.rollout.single_step import SingleStepRollout

    registrations = (
        (MODEL_ADAPTERS, "mock_wan", MockWanAdapter),
        (MODEL_ADAPTERS, "tiny_diffusion", TinyDiffusionAdapter),
        (MODEL_ADAPTERS, "sd3_tempflow", SD3TempFlowAdapter),
        (MODEL_ADAPTERS, "tempflow_sd3_legacy", SD3TempFlowAdapter),
        (MODEL_ADAPTERS, "world_r1_wan_legacy", WorldR1WanLegacyAdapter),
        (ALGORITHMS, "grpo", GRPOAlgorithm),
        (ALGORITHMS, "flash_grpo", FlashGRPOAlgorithm),
        (ALGORITHMS, "tempflow_grpo", TempFlowGRPOAlgorithm),
        (REWARD_CLIENTS, "mock", MockRewardClient),
        (REWARD_CLIENTS, "remote_pickle", RemotePickleRewardClient),
        (REWARD_CLIENTS, "prompt_color", PromptColorRewardClient),
        (REWARD_CLIENTS, "prompt_color_margin", PromptColorMarginRewardClient),
        (REWARD_CLIENTS, "prompt_color_guarded", PromptColorGuardedRewardClient),
        (REWARD_CLIENTS, "pickscore", PickScoreRewardClient),
        (REWARD_CLIENTS, "video_hpsv3", VideoHPSv3RewardClient),
        (REWARD_CLIENTS, "reward_3d", WorldR1Reward3DClient),
        (REWARD_CLIENTS, "reward_general", WorldR1RewardGeneralClient),
        (FEEDBACK_PROVIDERS, "reward_router", RewardRouterFeedbackProvider),
        (OPTIMIZER_PLUGINS, "algorithm", _build_algorithm_optimizer),
        (ROLLOUT_ENGINES, "full_trajectory", FullTrajectoryRollout),
        (ROLLOUT_ENGINES, "branching", BranchingRollout),
        (ROLLOUT_ENGINES, "single_step", SingleStepRollout),
        (ROLLOUT_ENGINES, "flash_single_step", SingleStepRollout),
    )
    for registry, name, component in registrations:
        registry.register_builtin(name, component)


__all__ = ["register_builtin_plugins"]
