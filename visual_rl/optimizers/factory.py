from dataclasses import asdict, is_dataclass
from typing import Any

from visual_rl.configs.schema import VisualRLConfig
from visual_rl.core.registry import ALGORITHMS, OPTIMIZER_PLUGINS
from visual_rl.optimizers.advantages import AdvantageComputer
from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin


def build_algorithm(config: Any):
    from visual_rl.builtins import register_builtin_plugins

    register_builtin_plugins()
    config_dict = asdict(config) if is_dataclass(config) else dict(config)
    params = dict(config_dict.pop("params", {}) or {})
    config_dict.update(params)
    algorithm_cls = ALGORITHMS.get(config_dict.get("name", "grpo"))
    if hasattr(algorithm_cls, "from_config"):
        return algorithm_cls.from_config(config_dict)
    return algorithm_cls(**config_dict)


def _build_algorithm_optimizer(config: VisualRLConfig) -> OptimizerPlugin:
    algorithm = build_algorithm(config.algorithm)

    advantage_computer = AdvantageComputer(
        reward_weights=config.rewards.weights,
        per_prompt=config.per_prompt_stat_tracking,
        weight_advantages=config.algorithm.weight_advantages,
        use_global_std=config.sample.global_std,
        max_group_std=config.sample.max_group_std,
        mode=config.algorithm.advantage_mode,
    )

    return AlgorithmOptimizerPlugin(
        algorithm=algorithm,
        advantage_computer=advantage_computer,
        optimizer_config=config.optimizer.params,
    )


OPTIMIZER_PLUGINS.register("algorithm", _build_algorithm_optimizer)


def build_optimizer_plugin(config: VisualRLConfig) -> OptimizerPlugin:
    from visual_rl.builtins import register_builtin_plugins

    register_builtin_plugins()
    builder = OPTIMIZER_PLUGINS.get(config.optimizer.name)
    plugin = builder(config)
    if not isinstance(plugin, OptimizerPlugin):
        raise TypeError(
            f"Optimizer plugin {config.optimizer.name!r} must implement OptimizerPlugin"
        )
    return plugin
