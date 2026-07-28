"""GradScaler is Runner-owned and advances inside the atomic boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.update_engine import UpdateEngine


ROOT = Path(__file__).resolve().parents[1]


def _tree(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def test_plugin_and_update_engine_have_no_scaler_or_checkpoint_state_owner() -> None:
    for owner in (AlgorithmOptimizerPlugin, UpdateEngine):
        assert not hasattr(owner, "state_dict")
        assert not hasattr(owner, "load_state_dict")
        assert not hasattr(owner, "scaler_state_dict")
        assert not hasattr(owner, "load_scaler_state_dict")
        assert not hasattr(owner, "_get_grad_scaler")

    for relative in (
        "visual_rl/optimizers/algorithm_plugin.py",
        "visual_rl/optimizers/update_engine.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "_pending_scaler_state" not in source
        assert "self._scaler" not in source


def test_runner_is_the_only_live_optimizer_and_scaler_attribute_owner() -> None:
    owners = {}
    for relative in (
        "visual_rl/runner.py",
        "visual_rl/optimizers/algorithm_plugin.py",
        "visual_rl/optimizers/update_engine.py",
        "visual_rl/distributed.py",
    ):
        attributes = {
            node.attr
            for node in ast.walk(_tree(relative))
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        owners[relative] = attributes & {"optimizer", "scaler"}
    assert owners == {
        "visual_rl/runner.py": {"optimizer", "scaler"},
        "visual_rl/optimizers/algorithm_plugin.py": set(),
        "visual_rl/optimizers/update_engine.py": set(),
        "visual_rl/distributed.py": set(),
    }


def test_scaler_step_and_update_share_the_one_atomic_operation() -> None:
    tree = _tree("visual_rl/optimizers/update_engine.py")
    step = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
        and any(
            isinstance(child, ast.FunctionDef)
            and child.name == "operation"
            for child in node.body
        )
    )
    operation = next(
        child
        for child in step.body
        if isinstance(child, ast.FunctionDef)
        and child.name == "operation"
    )
    calls = [
        node
        for node in ast.walk(operation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    ]
    scaler_calls = [
        node.func.attr
        for node in calls
        if isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scaler"
    ]
    assert scaler_calls == ["step", "update"]
    atomic_calls = [
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_optimizer_step"
    ]
    assert len(atomic_calls) == 1
    assert isinstance(atomic_calls[0].args[0], ast.Name)
    assert atomic_calls[0].args[0].id == "operation"
