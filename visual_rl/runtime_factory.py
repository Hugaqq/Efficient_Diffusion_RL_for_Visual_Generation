"""The only construction site for train-time VisualRL components."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visual_rl.configs.schema import VisualRLConfig
    from visual_rl.core.types import RuntimeBuildContext
    from visual_rl.datasets.prompt_dataset import PromptDataset
    from visual_rl.feedback.executor import RewardExecutor
    from visual_rl.model_adapters.base import ModelAdapter
    from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
    from visual_rl.rollout.base import RolloutEngine

__all__ = ["RuntimeComponents", "build_runtime_components"]


@dataclass(frozen=True)
class RuntimeComponents:
    """Owned runtime bundle returned to the one Runner setup path.

    The Provider, RewardCache, RewardClients and PolicyAlgorithm remain private
    resources owned by ``_resources``; exposing them here would make it
    possible to bypass RewardExecutor or AlgorithmOptimizerPlugin.
    """

    dataset: PromptDataset
    model: ModelAdapter
    rollout: RolloutEngine
    reward_executor: RewardExecutor
    optimizer_plugin: AlgorithmOptimizerPlugin
    _resources: ExitStack = field(repr=False, compare=False)

    def close(self) -> None:
        """Close every successfully constructed resource exactly once."""

        self._resources.close()


def build_runtime_components(
    config: VisualRLConfig,
    context: RuntimeBuildContext,
) -> RuntimeComponents:
    """Build the fixed runtime graph with transactional resource ownership.

    Every successful constructor is registered before the next constructor is
    attempted.  ``pop_all`` occurs only after the full graph exists, so a
    partial failure closes exactly the prefix that was actually returned.
    """

    from visual_rl.builtins import get_builtin_component
    from visual_rl.datasets.prompt_dataset import PromptDataset
    from visual_rl.optimizers.advantages import AdvantageComputer
    from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin

    with ExitStack() as bundle_stack:
        dataset = PromptDataset.from_config(config.dataset)
        _own(bundle_stack, dataset)

        model_spec = get_builtin_component("model", config.model.name)
        model = model_spec.factory.from_config(config.model.params, context)
        _own(bundle_stack, model)

        rollout_spec = get_builtin_component("rollout", config.rollout.name)
        rollout = rollout_spec.factory.from_config(config.rollout.params, context)
        _own(bundle_stack, rollout)

        reward_executor = _construct_reward_executor(
            config,
            context,
            bundle_stack,
        )

        algorithm_spec = get_builtin_component("algorithm", config.algorithm.name)
        algorithm = algorithm_spec.factory.from_config(
            config.algorithm.params,
            context,
        )
        _own(bundle_stack, algorithm)

        advantage_computer = AdvantageComputer(
            epsilon=config.algorithm.advantage.epsilon,
            output_dtype=algorithm_spec.factory.ADVANTAGE_DTYPE,
        )

        optimizer_plugin = AlgorithmOptimizerPlugin(
            algorithm=algorithm,
            advantage_computer=advantage_computer,
            update_microbatch_size=config.runtime.update_microbatch_size,
            transition_microbatch_size=(
                config.runtime.transition_microbatch_size
            ),
            precision=config.runtime.precision,
            max_grad_norm=config.optimizer.max_grad_norm,
            max_initial_logprob_delta=config.optimizer.max_initial_logprob_delta,
            require_initial_clipfrac_zero=(
                config.optimizer.require_initial_clipfrac_zero
            ),
            require_finite_gradients=config.optimizer.require_finite_gradients,
            require_nonzero_gradients=config.optimizer.require_nonzero_gradients,
        )
        _own(bundle_stack, optimizer_plugin)

        owned_resources = bundle_stack.pop_all()

    return RuntimeComponents(
        dataset=dataset,
        model=model,
        rollout=rollout,
        reward_executor=reward_executor,
        optimizer_plugin=optimizer_plugin,
        _resources=owned_resources,
    )


def _construct_reward_executor(
    config: VisualRLConfig,
    context: RuntimeBuildContext,
    bundle_stack: ExitStack,
) -> RewardExecutor:
    """Build the nested reward ownership graph in canonical component order."""

    from visual_rl.builtins import get_builtin_component
    from visual_rl.feedback.cache import RewardCache
    from visual_rl.feedback.executor import RewardExecutor
    from visual_rl.feedback.provider import (
        RewardClientBinding,
        RewardFeedbackProvider,
    )

    with ExitStack() as reward_stack:
        bindings: list[RewardClientBinding] = []
        for item in config.reward.components:
            reward_spec = get_builtin_component("reward", item.name)
            client = reward_spec.factory.from_config(item.params, context)
            _own(reward_stack, client)
            bindings.append(
                RewardClientBinding(
                    name=item.name,
                    client=client,
                    weight=item.weight,
                    resolved_params=item.params,
                )
            )

        reward_cache = None
        if config.reward.cache_dir is not None:
            reward_cache = RewardCache(
                config.reward.cache_dir / f"rank_{context.rank}"
            )
            _own(reward_stack, reward_cache)

        provider = RewardFeedbackProvider(
            clients=tuple(bindings),
            cache=reward_cache,
        )
        _own(reward_stack, provider)

        executor = RewardExecutor(
            provider=provider,
            microbatch_size=config.reward.execution.microbatch_size,
            max_retries=config.reward.execution.max_retries,
        )
        _own(reward_stack, executor)

        owned_reward_stack = reward_stack.pop_all()

    bundle_stack.callback(owned_reward_stack.close)
    return executor


def _own(stack: ExitStack, resource: Any) -> None:
    """Register one required idempotent ``close`` method immediately."""

    close = resource.close
    if not callable(close):
        raise TypeError(f"{type(resource).__name__}.close must be callable")
    stack.callback(close)
