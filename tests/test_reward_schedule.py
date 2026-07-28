from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.artifacts.checkpoint import config_fingerprint
from visual_rl.configs.schema import VisualRLConfig, config_to_dict, validate_config
from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.feedback.provider import RewardRouterFeedbackProvider
from visual_rl.feedback.router import RewardRouter


class _ScheduleGeneralClient:
    calls = 0

    def cache_fingerprint(self):
        return None

    def score(self, media, prompts, metadata):
        del media, metadata
        type(self).calls += 1
        return np.ones(len(prompts), dtype=np.float32), {"client": "general"}


class _Schedule3DClient:
    calls = 0

    def cache_fingerprint(self):
        return None

    def score(self, media, prompts, metadata):
        del media, metadata
        type(self).calls += 1
        return np.full(len(prompts), 2.0, dtype=np.float32), {"client": "3d"}


_GENERAL = "test_schedule_general"
_REWARD_3D = "test_schedule_3d"
REWARD_CLIENTS.register(_GENERAL, _ScheduleGeneralClient)
REWARD_CLIENTS.register(_REWARD_3D, _Schedule3DClient)


def _schedule() -> list[dict[str, object]]:
    return [
        {
            "name": "full_initial",
            "start_step": 0,
            "end_step": 100,
            "weights": {_GENERAL: 1.0, _REWARD_3D: 0.5},
        },
        {
            "name": "general_only",
            "start_step": 100,
            "end_step": 150,
            "weights": {_GENERAL: 1.0},
        },
        {
            "name": "full_restored",
            "start_step": 150,
            "end_step": 240,
            "weights": {_GENERAL: 1.0, _REWARD_3D: 0.5},
        },
    ]


def _world_config() -> VisualRLConfig:
    config = VisualRLConfig(run_name="world-schedule")
    config.rewards.weights = {_GENERAL: 1.0, _REWARD_3D: 0.5}
    config.rewards.clients = {
        _GENERAL: {"name": _GENERAL},
        _REWARD_3D: {"name": _REWARD_3D},
    }
    config.rewards.schedule = _schedule()
    config.train.max_steps = 240
    return config


def _router_config() -> dict[str, object]:
    return config_to_dict(_world_config())["rewards"]


def _batch(step: int) -> RolloutBatch:
    return RolloutBatch(
        prompts=["move the camera around the object"],
        metadata=[{}],
        timesteps=np.asarray([[step]], dtype=np.int64),
        context=StepContext(
            step=step,
            seed=2000 + step,
            epoch_tag=step,
            policy_version=step,
        ),
    )


def test_world_reward_schedule_validates_and_is_part_of_fingerprint():
    config = _world_config()
    validate_config(config)

    changed = deepcopy(config)
    changed.rewards.schedule[1]["weights"][_GENERAL] = 2.0
    validate_config(changed)
    assert config_fingerprint(config_to_dict(config)) != config_fingerprint(
        config_to_dict(changed)
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config.rewards.schedule[0].__setitem__("start_step", 1),
            "contiguous and start at step 0",
        ),
        (
            lambda config: config.rewards.schedule[1].__setitem__("start_step", 101),
            "contiguous",
        ),
        (
            lambda config: config.rewards.schedule[1]["weights"].__setitem__(
                "undeclared", 1.0
            ),
            "undeclared rewards.weights",
        ),
        (
            lambda config: config.rewards.clients.pop(_REWARD_3D),
            "undeclared rewards.clients",
        ),
        (
            lambda config: config.rewards.schedule[-1].__setitem__("end_step", 239),
            "cover train.max_steps",
        ),
        (
            lambda config: setattr(config.algorithm, "weight_advantages", True),
            "incompatible with algorithm.weight_advantages=true",
        ),
        (
            lambda config: setattr(config.rewards, "provider", "external"),
            "only supported by rewards.provider='reward_router'",
        ),
    ],
)
def test_world_reward_schedule_rejects_ambiguous_semantics(mutate, message):
    config = _world_config()
    mutate(config)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("step", "phase", "raw_keys", "weighted_total"),
    [
        (0, "full_initial", {_GENERAL, _REWARD_3D}, 2.0),
        (99, "full_initial", {_GENERAL, _REWARD_3D}, 2.0),
        (100, "general_only", {_GENERAL}, 1.0),
        (149, "general_only", {_GENERAL}, 1.0),
        (150, "full_restored", {_GENERAL, _REWARD_3D}, 2.0),
        (239, "full_restored", {_GENERAL, _REWARD_3D}, 2.0),
    ],
)
def test_router_selects_exact_world_phase_boundaries(
    step, phase, raw_keys, weighted_total
):
    router = RewardRouter(_router_config())

    rewards = router.score(
        None,
        ["prompt"],
        [{}],
        sample_id=["sample"],
        step=step,
    )

    assert set(rewards.raw) == raw_keys
    assert rewards.weighted_total.tolist() == pytest.approx([weighted_total])
    assert rewards.metadata["_schedule"] == {
        "name": phase,
        "step": step,
        "start_step": 0 if step < 100 else 100 if step < 150 else 150,
        "end_step": 100 if step < 100 else 150 if step < 150 else 240,
        "effective_weights": (
            {_GENERAL: 1.0} if 100 <= step < 150 else {_GENERAL: 1.0, _REWARD_3D: 0.5}
        ),
    }


def test_general_only_phase_does_not_call_3d_client_and_resume_uses_context_step():
    _ScheduleGeneralClient.calls = 0
    _Schedule3DClient.calls = 0
    provider = RewardRouterFeedbackProvider(_router_config())

    middle = provider.score(_batch(100))
    restored = provider.score(_batch(150))

    assert middle.metadata["_schedule"]["name"] == "general_only"
    assert restored.metadata["_schedule"]["name"] == "full_restored"
    assert _ScheduleGeneralClient.calls == 2
    assert _Schedule3DClient.calls == 1


@pytest.mark.parametrize("step", [-1, 240])
def test_scheduled_router_rejects_steps_outside_frozen_range(step):
    router = RewardRouter(_router_config())
    with pytest.raises(ValueError, match="outside the configured rewards.schedule"):
        router.score(None, ["prompt"], [{}], sample_id=["sample"], step=step)


def test_manifest_preserves_effective_world_reward_phase():
    provider = RewardRouterFeedbackProvider(_router_config())
    batch = _batch(100)
    rewards = provider.score(batch)

    (record,) = ManifestBuilder("world-schedule-run").build_records(
        step=100,
        batch=batch,
        rewards=rewards,
        media_type="video",
    )

    assert record.reward_values["schedule"] == {
        "name": "general_only",
        "step": 100,
        "start_step": 100,
        "end_step": 150,
        "effective_weights": {_GENERAL: 1.0},
    }


def test_empty_schedule_preserves_legacy_fingerprints():
    current = config_to_dict(VisualRLConfig(run_name="fingerprint"))
    legacy = deepcopy(current)
    legacy["rewards"].pop("schedule")

    assert config_fingerprint(current, version=1) == config_fingerprint(
        legacy, version=1
    )
    assert config_fingerprint(current, version=2) == config_fingerprint(
        legacy, version=2
    )
