"""Runtime wrapper for the import-safe Flow-GRPO declaration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from visual_rl.algorithms.modules.config import FlowGRPOAlgorithmConfig
from visual_rl.algorithms.modules.interface import AlgorithmModule

__all__ = ("FlowGRPOAlgorithmModule",)


class FlowGRPOAlgorithmModule(AlgorithmModule):
    CONFIG_TYPE = "visual_rl.algorithms.modules.config:FlowGRPOAlgorithmConfig"

    def __init__(self, config: FlowGRPOAlgorithmConfig) -> None:
        if not isinstance(config, FlowGRPOAlgorithmConfig):
            raise TypeError("config must be FlowGRPOAlgorithmConfig")
        self._config = config
        self._blueprint = config.describe_blueprint()
        self._requirements = config.describe_requirements()

    @property
    def config(self) -> FlowGRPOAlgorithmConfig:
        return self._config

    @property
    def blueprint(self):
        return self._blueprint

    @property
    def requirements(self):
        return self._requirements

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> FlowGRPOAlgorithmModule:
        if not isinstance(runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")
        if not isinstance(config, FlowGRPOAlgorithmConfig):
            raise TypeError("config must be FlowGRPOAlgorithmConfig")
        return cls(config)
