from __future__ import annotations


def test_feedback_provider_factory_accepts_registered_provider(monkeypatch):
    import torch

    from visual_rl.configs.schema import RewardConfig
    from visual_rl.core.registry import FEEDBACK_PROVIDERS
    from visual_rl.core.types import RewardBatch, RolloutBatch
    from visual_rl.feedback import FeedbackProvider, build_feedback_provider

    class ConstantFeedback(FeedbackProvider):
        def __init__(self, config, cache_dir=None, scale=1.0):
            self.value = float(config.weights["constant"]) * float(scale)
            self.cache_dir = cache_dir

        def score(self, batch):
            values = torch.full((len(batch.prompts),), self.value)
            return RewardBatch(
                raw={"constant": values},
                weighted={"constant": values},
                weighted_total=values,
                valid_mask=torch.ones(len(batch.prompts), dtype=torch.bool),
            )

    monkeypatch.setitem(FEEDBACK_PROVIDERS._items, "constant", ConstantFeedback)  # noqa: SLF001
    provider = build_feedback_provider(
        RewardConfig(
            provider="constant",
            weights={"constant": 2.0},
            clients={},
            provider_params={"scale": 3.0},
        )
    )
    batch = RolloutBatch(
        prompts=["a", "b"],
        metadata=[{}, {}],
        media=torch.zeros(2, 3, 2, 2),
        latents=torch.zeros(2, 1, 3, 2, 2),
        next_latents=torch.zeros(2, 1, 3, 2, 2),
        timesteps=torch.zeros(2, 1),
        old_log_probs=torch.zeros(2, 1),
    )

    assert provider.score(batch).weighted_total.tolist() == [6.0, 6.0]


def test_optimizer_plugin_factory_accepts_registered_builder(monkeypatch):
    from visual_rl.configs.schema import VisualRLConfig
    from visual_rl.core.registry import OPTIMIZER_PLUGINS
    from visual_rl.optimizers import OptimizerPlugin, build_optimizer_plugin

    class NoOpPlugin(OptimizerPlugin):
        def build_optimizer(self, parameters, train_config):
            del parameters, train_config
            return object()

        def step(self, adapter, batch, rewards, optimizer, context):
            del adapter, batch, rewards, optimizer
            return {"seen_step": float(context["step"])}

    monkeypatch.setitem(  # noqa: SLF001
        OPTIMIZER_PLUGINS._items,
        "no_op",
        lambda _config: NoOpPlugin(),
    )
    config = VisualRLConfig(run_name="plugin-test")
    config.optimizer.name = "no_op"

    plugin = build_optimizer_plugin(config)

    assert isinstance(plugin, NoOpPlugin)


def test_builtin_factories_work_in_a_clean_process():
    import subprocess
    import sys

    script = """
from visual_rl.configs.schema import RewardConfig, VisualRLConfig
from visual_rl.feedback import build_feedback_provider
from visual_rl.optimizers import build_optimizer_plugin
from visual_rl.rollout.full_trajectory import build_rollout_engine

config = VisualRLConfig(run_name='clean-factory')
feedback = build_feedback_provider(
    RewardConfig(
        weights={'prompt_color': 1.0},
        clients={'prompt_color': {'name': 'prompt_color'}},
    )
)
plugin = build_optimizer_plugin(config)
rollout = build_rollout_engine({'name': 'full_trajectory'})
print(type(feedback).__name__, type(plugin).__name__, type(rollout).__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == (
        "RewardRouterFeedbackProvider AlgorithmOptimizerPlugin FullTrajectoryRollout"
    )


def test_algorithm_params_are_forwarded_to_registered_algorithm():
    from visual_rl.configs.schema import AlgorithmConfig
    from visual_rl.optimizers import build_algorithm

    algorithm = build_algorithm(
        AlgorithmConfig(name="grpo", params={"clip_range": 0.25})
    )

    assert algorithm.clip_range == 0.25


def test_runner_rejects_optimizer_without_checkpoint_contract():
    import pytest

    from visual_rl.runner import ExperimentRunner

    with pytest.raises(TypeError, match="state_dict"):
        ExperimentRunner._validate_optimizer_contract(object())


def test_model_adapter_requires_checkpoint_methods():
    import pytest

    from visual_rl.model_adapters.base import ModelAdapter

    class IncompleteAdapter(ModelAdapter):
        name = "incomplete"
        media_type = "image"

        def parameters(self):
            return []

        def sample(self, prompts, metadata, rollout_config):
            del prompts, metadata, rollout_config

        def recompute_log_probs(self, batch):
            del batch

    with pytest.raises(TypeError, match="abstract"):
        IncompleteAdapter()
