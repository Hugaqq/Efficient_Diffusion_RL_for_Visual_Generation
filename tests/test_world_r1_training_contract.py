from __future__ import annotations


def test_reward_failure_stops_before_optimizer_step(monkeypatch, tmp_path):
    import pytest
    import torch

    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.feedback.clients  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.runner import ExperimentRunner

    class FailingRewardClient:
        def score(self, media, prompts, metadata):
            del media, prompts, metadata
            raise RuntimeError("reward server unavailable")

    monkeypatch.setitem(REWARD_CLIENTS._items, "world_r1_failing_test", FailingRewardClient)  # noqa: SLF001
    step_calls = {"count": 0}
    original_step = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        step_calls["count"] += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    config = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    config.paths.output_dir = str(tmp_path / "run")
    config.rewards.weights = {"reward_general": 1.0}
    config.rewards.clients = {"reward_general": {"name": "world_r1_failing_test"}}
    config.rewards.fail_policy = "invalid"

    with pytest.raises(RuntimeError, match="Reward failure"):
        ExperimentRunner(config).run(max_steps=1)

    assert step_calls["count"] == 0
    assert not (tmp_path / "run" / "checkpoint_000001").exists()


def test_reward_metadata_is_written_to_rollout_cache(monkeypatch, tmp_path):
    import numpy as np
    import json

    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.feedback.clients  # noqa: F401
    from visual_rl.configs.schema import load_config
    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.runner import ExperimentRunner

    class MetadataRewardClient:
        def score(self, media, prompts, metadata):
            del media, metadata
            return np.ones(len(prompts), dtype=np.float32), {"server": "fake-world-r1"}

    monkeypatch.setitem(REWARD_CLIENTS._items, "world_r1_metadata_test", MetadataRewardClient)  # noqa: SLF001

    config = load_config("visual_rl/configs/presets/world_r1_wan_v02_mock.yaml")
    config.paths.output_dir = str(tmp_path / "run")
    config.rewards.weights = {"reward_general": 1.0}
    config.rewards.clients = {"reward_general": {"name": "world_r1_metadata_test"}}

    ExperimentRunner(config).run(max_steps=1)

    metadata = json.loads((tmp_path / "run" / "rollouts" / "batch_000000.json").read_text(encoding="utf-8"))
    assert metadata["reward_metadata"]["reward_general"] == {"server": "fake-world-r1"}
