"""Public typed configuration API."""

from visual_rl.configs.schema import (
    VisualRLConfig,
    config_from_dict,
    load_config,
    validate_config,
)
from visual_rl.configs.sources import (
    ConfigDocument,
    ExperimentSpec,
    KeyOverride,
    SourceRef,
    list_packaged_presets,
    read_experiment_spec,
    read_packaged_preset,
)
from visual_rl.configs.resolver import ResolvedExperiment, resolve_experiment

__all__ = [
    "ConfigDocument",
    "ExperimentSpec",
    "KeyOverride",
    "ResolvedExperiment",
    "SourceRef",
    "VisualRLConfig",
    "config_from_dict",
    "list_packaged_presets",
    "load_config",
    "read_experiment_spec",
    "read_packaged_preset",
    "resolve_experiment",
    "validate_config",
]
