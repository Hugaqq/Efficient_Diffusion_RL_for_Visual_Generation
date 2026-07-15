"""CPU-only coverage for persistent FP16 GradScaler state."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest
import torch

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.optimizers import AdvantageResult, ObjectiveOutput, UpdateEngine
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin


class _FakeScaler:
    def __init__(self) -> None:
        self.current_scale = 128.0
        self.update_count = 0
        self.unscale_count = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: Any) -> None:
        del optimizer
        self.unscale_count += 1

    def step(self, optimizer: Any) -> None:
        optimizer.step()

    def update(self) -> None:
        self.update_count += 1
        self.current_scale *= 2.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "scale": self.current_scale,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_scale = float(state["scale"])
        self.update_count = int(state["update_count"])


class _Advantage:
    def __init__(self) -> None:
        self.checkpoint_state: dict[str, Any] = {}

    def __call__(self, batch: RolloutBatch, rewards: RewardBatch) -> AdvantageResult:
        del rewards
        return AdvantageResult(
            advantages=torch.ones(batch.batch_size),
            metrics={},
        )

    def state_dict(self) -> dict[str, Any]:
        return dict(self.checkpoint_state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.checkpoint_state = dict(state)


class _Objective:
    def __call__(
        self,
        batch: RolloutBatch,
        advantages: torch.Tensor,
        new_log_probs: torch.Tensor,
    ) -> ObjectiveOutput:
        del batch, advantages
        loss = new_log_probs.mean()
        zero = loss.detach() * 0.0
        return ObjectiveOutput(
            loss=loss,
            policy_loss=loss.detach(),
            approx_kl=zero,
            clipfrac=zero,
            metrics={},
        )


class _Adapter:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    def prepare_for_training(self) -> None:
        return None

    def recompute_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        return self.parameter.expand_as(batch.old_log_probs)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.parameter]


class _Algorithm:
    def compute_loss(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("checkpoint-only test must not compute a loss")


def _batch_and_rewards() -> tuple[RolloutBatch, RewardBatch]:
    context = StepContext(step=1, seed=7, epoch_tag=0)
    batch = RolloutBatch(
        prompts=["a", "b"],
        metadata=[{}, {}],
        media=torch.zeros(2, 1),
        latents=torch.zeros(2, 1, 1),
        next_latents=torch.ones(2, 1, 1),
        timesteps=torch.zeros(2, 1),
        old_log_probs=torch.zeros(2, 1),
        sample_id=["sample-a", "sample-b"],
        prompt_id=["prompt-a", "prompt-b"],
        group_id=["group-a", "group-b"],
        branch_id=[0, 0],
        context=context,
    )
    rewards = RewardBatch(
        raw={"score": torch.ones(2)},
        weighted={"score": torch.ones(2)},
        weighted_total=torch.ones(2),
        valid_mask=torch.ones(2, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    return batch, rewards


def test_fp16_update_engine_reuses_one_scaler_across_steps(monkeypatch) -> None:
    engine = UpdateEngine(_Advantage(), _Objective(), precision="fp16")
    created: list[_FakeScaler] = []

    def create_scaler() -> _FakeScaler:
        scaler = _FakeScaler()
        created.append(scaler)
        return scaler

    monkeypatch.setattr(engine, "_grad_scaler", create_scaler)
    monkeypatch.setattr(engine, "_validate_precision_device", lambda device: None)
    monkeypatch.setattr(engine, "_autocast_context", lambda device: nullcontext())
    adapter = _Adapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    batch, rewards = _batch_and_rewards()

    engine.step(adapter, batch, rewards, optimizer, batch.context)
    engine.step(adapter, batch, rewards, optimizer, batch.context)

    assert len(created) == 1
    assert created[0].unscale_count == 2
    assert created[0].update_count == 2
    assert created[0].current_scale == 512.0


def test_fp16_update_failure_restores_scaler_to_retryable_pre_step_state(
    monkeypatch,
) -> None:
    engine = UpdateEngine(_Advantage(), _Objective(), precision="fp16")
    created = [_FakeScaler(), _FakeScaler()]
    returned: list[_FakeScaler] = []

    def create_scaler() -> _FakeScaler:
        scaler = created.pop(0)
        returned.append(scaler)
        return scaler

    monkeypatch.setattr(engine, "_grad_scaler", create_scaler)
    monkeypatch.setattr(engine, "_validate_precision_device", lambda device: None)
    monkeypatch.setattr(engine, "_autocast_context", lambda device: nullcontext())
    adapter = _Adapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    batch, rewards = _batch_and_rewards()

    with pytest.raises(RuntimeError, match="reject commit"):
        engine.step(
            adapter,
            batch,
            rewards,
            optimizer,
            batch.context,
            before_optimizer_step=lambda: (_ for _ in ()).throw(
                RuntimeError("reject commit")
            ),
        )

    assert returned[0].unscale_count == 1
    assert returned[0].update_count == 0
    engine.step(adapter, batch, rewards, optimizer, batch.context)
    assert len(returned) == 2
    assert returned[1].unscale_count == 1
    assert returned[1].update_count == 1
    assert returned[1].current_scale == 256.0


def test_plugin_checkpoint_restores_scaler_state_lazily(monkeypatch) -> None:
    source = AlgorithmOptimizerPlugin(_Algorithm(), _Advantage(), precision="fp16")
    source_scaler = _FakeScaler()
    source_scaler.current_scale = 4096.0
    source_scaler.update_count = 11
    monkeypatch.setattr(source.update_engine, "_grad_scaler", lambda: source_scaler)
    assert source.update_engine._get_grad_scaler() is source_scaler

    checkpoint_state = source.state_dict()
    assert checkpoint_state == {
        "advantage": {},
        "grad_scaler": {"scale": 4096.0, "update_count": 11},
    }

    restored = AlgorithmOptimizerPlugin(_Algorithm(), _Advantage(), precision="fp16")
    restored_scalers: list[_FakeScaler] = []

    def create_restored_scaler() -> _FakeScaler:
        scaler = _FakeScaler()
        restored_scalers.append(scaler)
        return scaler

    monkeypatch.setattr(restored.update_engine, "_grad_scaler", create_restored_scaler)
    restored.load_state_dict(checkpoint_state)

    assert restored_scalers == []
    assert restored.state_dict() == checkpoint_state
    scaler = restored.update_engine._get_grad_scaler()
    assert restored_scalers == [scaler]
    assert scaler.state_dict() == checkpoint_state["grad_scaler"]


def test_plugin_accepts_legacy_checkpoint_without_scaler_state(monkeypatch) -> None:
    plugin = AlgorithmOptimizerPlugin(_Algorithm(), _Advantage(), precision="fp16")
    existing_scaler = _FakeScaler()
    existing_scaler.update_count = 5
    scalers = [existing_scaler, _FakeScaler()]
    monkeypatch.setattr(plugin.update_engine, "_grad_scaler", lambda: scalers.pop(0))
    assert plugin.update_engine._get_grad_scaler() is existing_scaler

    plugin.load_state_dict({"advantage": {"legacy": True}})

    assert plugin.state_dict() == {"advantage": {"legacy": True}}
    fresh_scaler = plugin.update_engine._get_grad_scaler()
    assert fresh_scaler is not existing_scaler
    assert fresh_scaler.state_dict() == {"scale": 128.0, "update_count": 0}


def test_plugin_state_load_rolls_back_advantage_and_scaler_together(
    monkeypatch,
) -> None:
    advantage = _Advantage()
    advantage.checkpoint_state = {"before": True}
    plugin = AlgorithmOptimizerPlugin(_Algorithm(), advantage, precision="fp16")
    scaler = _FakeScaler()
    scaler.current_scale = 256.0
    monkeypatch.setattr(plugin.update_engine, "_grad_scaler", lambda: scaler)
    assert plugin.update_engine._get_grad_scaler() is scaler
    before = plugin.state_dict()
    real_load = plugin.update_engine.load_scaler_state_dict
    calls = 0

    def fail_first_scaler_load(state):
        nonlocal calls
        calls += 1
        real_load(state)
        if calls == 1:
            raise RuntimeError("injected scaler restore failure")

    monkeypatch.setattr(
        plugin.update_engine,
        "load_scaler_state_dict",
        fail_first_scaler_load,
    )

    with pytest.raises(RuntimeError, match="injected scaler restore failure"):
        plugin.load_state_dict(
            {
                "advantage": {"after": True},
                "grad_scaler": {"scale": 1024.0, "update_count": 9},
            }
        )

    assert plugin.state_dict() == before
    assert calls == 2


@pytest.mark.parametrize(
    "state",
    [None, [], {"advantage": []}, {"advantage": {}, "grad_scaler": []}],
)
def test_plugin_state_load_rejects_malformed_sections_without_mutation(state) -> None:
    advantage = _Advantage()
    advantage.checkpoint_state = {"before": True}
    plugin = AlgorithmOptimizerPlugin(_Algorithm(), advantage, precision="fp16")
    before = plugin.state_dict()

    with pytest.raises(TypeError):
        plugin.load_state_dict(state)

    assert plugin.state_dict() == before
