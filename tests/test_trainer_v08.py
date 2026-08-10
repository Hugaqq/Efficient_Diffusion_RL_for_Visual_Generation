"""Canonical six-stage GRPO trainer lifecycle and identity contracts."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest
import torch

from visual_rl.algorithms.trainer.grpo import BaseTrainer, GRPOTrainer
from visual_rl.algorithms.trainer.interface import (
    IterationIdentity,
    PrepareRunContext,
    StageValue,
    TrainerState,
)


def _identity(step: int) -> IterationIdentity:
    return IterationIdentity(
        optimizer_step=step,
        source_id="main",
        phase_id="main",
        row_identities=("row-0", "row-1"),
        group_ids=("group-0", "group-0"),
        member_ids=(0, 1),
    )


class _Prelude:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.committed: list[IterationIdentity] = []
        self.aborted: list[IterationIdentity] = []

    def build(self, optimizer_step: int) -> StageValue[object]:
        self.events.append("prelude")
        return StageValue(_identity(optimizer_step), {"batch": optimizer_step})

    def commit_iteration(self, identity: IterationIdentity) -> None:
        self.committed.append(identity)

    def abort_iteration(self, identity: IterationIdentity) -> None:
        self.aborted.append(identity)


class _Hook:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def prepare_run(self, context: PrepareRunContext) -> None:
        self.events.append(f"prepare:{self.name}:{context.start_optimizer_step}")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


class _Stage:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail: bool = False,
        replace_identity: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail = fail
        self.replace_identity = replace_identity

    def __call__(self, value: StageValue[object]) -> StageValue[object]:
        self.events.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        identity = value.identity
        if self.replace_identity:
            identity = _identity(identity.optimizer_step)
        return StageValue(identity, {"stage": self.name, "previous": value.payload})


def _trainer(
    events: list[str],
    *,
    failing_stage: str | None = None,
    replacing_stage: str | None = None,
    hooks: tuple[object, ...] = (),
    prelude: _Prelude | None = None,
) -> tuple[GRPOTrainer, _Prelude]:
    owned_prelude = prelude or _Prelude(events)
    stages = {
        name: _Stage(
            name,
            events,
            fail=name == failing_stage,
            replace_identity=name == replacing_stage,
        )
        for name in ("rollout", "reward", "advantage", "credit", "optimize")
    }
    return (
        GRPOTrainer(
            prelude=owned_prelude,
            prepare_hooks=hooks,
            close_hooks=hooks,
            **stages,
        ),
        owned_prelude,
    )


def test_trainer_executes_exactly_one_canonical_six_stage_path() -> None:
    events: list[str] = []
    hook = _Hook("model-preprocess", events)
    trainer, prelude = _trainer(events, hooks=(hook, hook))

    trainer.prepare_run(PrepareRunContext("run-1", "recipe-1", 100))
    result = trainer.run_iteration(100)

    assert events == [
        "prepare:model-preprocess:100",
        "prelude",
        "rollout",
        "reward",
        "advantage",
        "credit",
        "optimize",
    ]
    assert result.stage_order == (
        "prelude",
        "rollout",
        "reward",
        "advantage",
        "credit",
        "optimize",
    )
    assert result.value.payload["stage"] == "optimize"
    assert prelude.committed == [result.value.identity]
    assert prelude.aborted == []
    assert trainer.next_optimizer_step == 101
    assert trainer.state is TrainerState.PREPARED


def test_failure_aborts_reservation_and_does_not_advance_step() -> None:
    events: list[str] = []
    trainer, prelude = _trainer(events, failing_stage="advantage")
    trainer.prepare_run(PrepareRunContext("run", "recipe", 0))

    with pytest.raises(RuntimeError, match="advantage failed"):
        trainer.run_iteration(0)

    assert events == ["prelude", "rollout", "reward", "advantage"]
    assert len(prelude.aborted) == 1
    assert prelude.committed == []
    assert trainer.next_optimizer_step == 0


def test_retained_failure_does_not_keep_last_stage_payload_alive() -> None:
    payload_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    class EphemeralPrelude:
        def build(self, optimizer_step):
            payload = torch.zeros(64, 64)
            payload_refs.append(weakref.ref(payload))
            return StageValue(_identity(optimizer_step), payload)

        def abort_iteration(self, identity):
            assert identity.optimizer_step == 0

    class PassthroughStage:
        def __call__(self, value):
            return value

    class DroppingFailureStage:
        def __call__(self, value):
            del value
            raise RuntimeError("optimize failed")

    def invoke() -> RuntimeError:
        passthrough = PassthroughStage()
        trainer = BaseTrainer(
            prelude=EphemeralPrelude(),
            rollout=passthrough,
            reward=passthrough,
            advantage=passthrough,
            credit=passthrough,
            optimize=DroppingFailureStage(),
        )
        trainer.prepare_run(PrepareRunContext("run", "recipe", 0))
        try:
            trainer.run_iteration(0)
        except RuntimeError as error:
            return error
        raise AssertionError("the injected trainer failure did not propagate")

    error = invoke()
    assert str(error) == "optimize failed"
    gc.collect()
    assert payload_refs[0]() is None


def test_stage_cannot_reconstruct_or_replace_iteration_identity() -> None:
    events: list[str] = []
    trainer, prelude = _trainer(events, replacing_stage="reward")
    trainer.prepare_run(PrepareRunContext("run", "recipe", 0))

    with pytest.raises(ValueError, match="replaced the canonical"):
        trainer.run_iteration(0)

    assert events[-1] == "reward"
    assert len(prelude.aborted) == 1
    assert trainer.next_optimizer_step == 0


def test_prepare_once_requires_contiguous_steps_and_close_is_reverse_deduplicated() -> (
    None
):
    events: list[str] = []
    first = _Hook("first", events)
    second = _Hook("second", events)
    trainer, _ = _trainer(events, hooks=(first, second, first))
    context = PrepareRunContext("run", "recipe", 4)
    trainer.prepare_run(context)

    with pytest.raises(RuntimeError, match="exactly once"):
        trainer.prepare_run(context)
    with pytest.raises(ValueError, match="expected optimizer step 4"):
        trainer.run_iteration(5)

    trainer.close()
    trainer.close()
    assert events[:2] == ["prepare:first:4", "prepare:second:4"]
    assert events[-2:] == ["close:second", "close:first"]
    assert trainer.state is TrainerState.CLOSED
    with pytest.raises(RuntimeError, match="prepared"):
        trainer.run_iteration(4)


def test_trainer_source_contains_no_model_recipe_or_data_prelude_switches() -> None:
    source = (
        Path(__file__).parents[1] / "visual_rl" / "algorithms" / "trainer" / "grpo.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "sd3",
        "wan",
        "tempflow",
        "flash_grpo",
        "world_r1",
        "PeriodicPhaseSchedule",
        "MultiSourceSampler",
    ):
        assert forbidden.lower() not in source.lower()
