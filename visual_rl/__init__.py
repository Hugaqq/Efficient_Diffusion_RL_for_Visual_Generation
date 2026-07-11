"""VisualRL integration package."""

__version__ = "0.6.0"

from visual_rl.artifacts import (
    ArtifactManager,
    ManifestBuilder,
    SampleManifest,
    SampleRecord,
)
from visual_rl.configs.schema import VisualRLConfig, load_config
from visual_rl.core.types import RewardBatch, RolloutBatch
from visual_rl.feedback import FeedbackProvider
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.optimizers import OptimizerPlugin
from visual_rl.plugins import (
    register_algorithm,
    register_feedback_provider,
    register_model_adapter,
    register_optimizer_plugin,
    register_reward_client,
    register_rollout_engine,
)
from visual_rl.rollout.base import RolloutEngine
from visual_rl.runner import ExperimentRunner

__all__ = [
    "ArtifactManager",
    "ExperimentRunner",
    "FeedbackProvider",
    "ManifestBuilder",
    "ModelAdapter",
    "OptimizerPlugin",
    "RewardBatch",
    "RolloutEngine",
    "RolloutBatch",
    "SampleManifest",
    "SampleRecord",
    "VisualRLConfig",
    "load_config",
    "register_algorithm",
    "register_feedback_provider",
    "register_model_adapter",
    "register_optimizer_plugin",
    "register_reward_client",
    "register_rollout_engine",
]
