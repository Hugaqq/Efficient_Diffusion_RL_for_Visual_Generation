"""AdamW and GradScaler have one live owner and one fixed topology."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.optimizers.advantages import AdvantageComputer
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def _plugin() -> AlgorithmOptimizerPlugin:
    return AlgorithmOptimizerPlugin(
        algorithm=GRPOAlgorithm(),
        advantage_computer=AdvantageComputer(
            epsilon=1e-8,
            output_dtype="float32",
        ),
        update_microbatch_size=2,
        precision="fp32",
        max_grad_norm=None,
        max_initial_logprob_delta=None,
        require_initial_clipfrac_zero=False,
        require_finite_gradients=True,
        require_nonzero_gradients=False,
    )


def test_plugin_builds_exactly_one_adamw_over_named_identity_order() -> None:
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    plugin = _plugin()
    optimizer = plugin.build_optimizer(
        (("first", first), ("second", second)),
        SimpleNamespace(
            learning_rate=3e-4,
            adam_beta1=0.8,
            adam_beta2=0.95,
            adam_weight_decay=0.1,
            adam_epsilon=1e-7,
        ),
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["params"] == [first, second]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0.8, 0.95))
    assert not hasattr(plugin, "state_dict")
    assert not hasattr(plugin, "load_state_dict")


def test_runner_scaler_matrix_has_no_plugin_or_engine_owner(monkeypatch) -> None:
    assert ExperimentRunner._build_gradient_scaler(
        precision="fp32",
        device=torch.device("cpu"),
    ) is None
    assert ExperimentRunner._build_gradient_scaler(
        precision="bf16",
        device=torch.device("cuda"),
    ) is None
    with pytest.raises(ValueError, match="requires a CUDA"):
        ExperimentRunner._build_gradient_scaler(
            precision="fp16",
            device=torch.device("cpu"),
        )

    sentinel = object()
    calls = []

    def fake_scaler(device_type):
        calls.append(device_type)
        return sentinel

    monkeypatch.setattr(torch.amp, "GradScaler", fake_scaler)
    assert ExperimentRunner._build_gradient_scaler(
        precision="fp16",
        device=torch.device("cuda"),
    ) is sentinel
    assert calls == ["cuda"]

    for relative in (
        "visual_rl/optimizers/algorithm_plugin.py",
        "visual_rl/optimizers/update_engine.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        assert "_scaler" not in attributes
        assert "_pending_scaler_state" not in attributes
        assert "_get_grad_scaler" not in attributes


def test_update_engine_does_not_add_a_recompute_only_autocast_context() -> None:
    tree = ast.parse(
        (ROOT / "visual_rl/optimizers/update_engine.py").read_text(
            encoding="utf-8"
        )
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "autocast"
    ]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert isinstance(keywords["enabled"], ast.Constant)
    assert keywords["enabled"].value is False
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_forward_context"
        for node in ast.walk(tree)
    )


def test_runner_initial_seed_is_rank_local_from_the_one_validated_seed() -> None:
    tree = ast.parse(
        (ROOT / "visual_rl/runner.py").read_text(encoding="utf-8")
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "seed_everything"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    assert ast.unparse(calls[0].args[0]) == (
        "self.config.run.seed + strategy.rank"
    )
