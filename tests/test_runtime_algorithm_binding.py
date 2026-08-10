"""Runtime-to-trainer bindings fail closed and bind exactly once."""

from __future__ import annotations

import pytest

from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue
from visual_rl.runtime.algorithm_binding import (
    BindOncePrelude,
    BindOnceStage,
    StageBindingError,
)


def _identity() -> IterationIdentity:
    return IterationIdentity(
        optimizer_step=0,
        source_id="main",
        phase_id="main",
        row_identities=("row-0",),
        group_ids=("group-0",),
        member_ids=(0,),
    )


class _Prelude:
    def __init__(self) -> None:
        self.committed = False

    def build(self, optimizer_step):
        assert optimizer_step == 0
        return StageValue(_identity(), "batch")

    def commit_iteration(self, identity):
        assert identity.optimizer_step == 0
        self.committed = True


def test_stage_port_rejects_use_before_bind_and_rebinding() -> None:
    port = BindOnceStage("reward")
    value = StageValue(_identity(), "payload")
    with pytest.raises(StageBindingError, match="not bound"):
        port(value)

    port.bind(lambda incoming: StageValue(incoming.identity, "next"))
    assert port(value).payload == "next"
    with pytest.raises(StageBindingError, match="already bound"):
        port.bind(lambda incoming: incoming)


def test_prelude_port_preserves_commit_and_rejects_rebinding() -> None:
    port = BindOncePrelude()
    with pytest.raises(StageBindingError, match="not bound"):
        port.build(0)
    prelude = _Prelude()
    port.bind(prelude)
    value = port.build(0)
    port.commit_iteration(value.identity)
    assert prelude.committed
    with pytest.raises(StageBindingError, match="already bound"):
        port.bind(_Prelude())
