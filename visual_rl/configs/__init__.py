"""Internal canonical configuration types.

Configuration is loaded only through :func:`visual_rl.load`; no loader,
preset, layered source, override, or descriptor is exported here.
"""

from visual_rl.configs.resolver import resolve_config
from visual_rl.configs.schema import (
    AdvantageConfig,
    AlgorithmConfig,
    ArtifactsConfig,
    ComponentSelectionConfig,
    DatasetConfig,
    DistributedConfig,
    ModelConfig,
    OptimizerConfig,
    ResumeConfig,
    RewardComponentConfig,
    RewardConfig,
    RewardExecutionConfig,
    RunConfig,
    RuntimeConfig,
    VisualRLConfig,
)
from visual_rl.core.types import to_plain_dict

__all__ = (
    "AdvantageConfig",
    "AlgorithmConfig",
    "ArtifactsConfig",
    "ComponentSelectionConfig",
    "DatasetConfig",
    "DistributedConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ResumeConfig",
    "RewardComponentConfig",
    "RewardConfig",
    "RewardExecutionConfig",
    "RunConfig",
    "RuntimeConfig",
    "VisualRLConfig",
    "resolve_config",
    "to_plain_dict",
)
