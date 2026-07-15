"""CPU-only contract coverage for the C6 optimizer responsibility split."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from visual_rl import load_config
from visual_rl.core.registry import OPTIMIZER_PLUGINS
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.optimizers import (
    AdvantageComputer,
    AdvantageResult,
    AlgorithmOptimizerPlugin,
    AlgorithmPolicyObjective,
    ObjectiveOutput,
    OptimizerPlugin,
    UpdateEngine,
    build_optimizer_plugin,
)
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.optimizers.tempflow_grpo import (
    TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT,
    TempFlowGRPOAlgorithm,
)


ROOT = Path(__file__).resolve().parents[1]


def _batch(
    *,
    context: StepContext | None = StepContext(0, 7, 0),
    metadata: list[dict] | None = None,
    old_log_probs: torch.Tensor | None = None,
    model_metadata: dict | None = None,
) -> RolloutBatch:
    old_log_probs = (
        torch.zeros(2, 2, dtype=torch.float64)
        if old_log_probs is None
        else old_log_probs
    )
    batch_size, steps = old_log_probs.shape
    return RolloutBatch(
        prompts=["same prompt"] * batch_size,
        metadata=metadata
        or [
            {"parent_prompt_index": 4, "branch_step_index": 0}
            for _ in range(batch_size)
        ],
        media=torch.zeros(batch_size, 1),
        latents=torch.zeros(batch_size, steps, 1),
        next_latents=torch.zeros(batch_size, steps, 1),
        timesteps=torch.zeros(batch_size, steps),
        old_log_probs=old_log_probs,
        group_id=["test-group"] * batch_size,
        context=context,
        model_metadata=dict(model_metadata or {}),
    )


def _rewards(batch: RolloutBatch, values: tuple[float, ...] = (1.0, 3.0)) -> RewardBatch:
    total = torch.tensor(values, dtype=torch.float64)
    return RewardBatch(
        raw={"score": total},
        weighted={"score": total},
        weighted_total=total,
        valid_mask=torch.ones(len(values), dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )


def _advantage_computer(*, output_dtype: str = "float64") -> AdvantageComputer:
    return AdvantageComputer(
        reward_weights={"score": 1.0},
        mode="grpo",
        output_dtype=output_dtype,
    )


class _GuardAdapter:
    def __init__(self, logprob_builder=None):
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.logprob_builder = logprob_builder

    def prepare_for_training(self):
        return None

    def recompute_log_probs(self, batch):
        if self.logprob_builder is not None:
            return self.logprob_builder(self.parameter, batch)
        return self.parameter.expand_as(batch.old_log_probs)

    def parameters(self):
        return [self.parameter]


class _GuardOptimizer(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _GuardObjective:
    def __init__(self, output_builder):
        self.output_builder = output_builder

    def __call__(self, batch, advantages, new_log_probs):
        del batch, advantages
        return self.output_builder(new_log_probs)


def _valid_output(new_log_probs, *, loss=None, metrics=None) -> ObjectiveOutput:
    loss = new_log_probs.mean() if loss is None else loss
    zero = torch.zeros((), device=new_log_probs.device)
    return ObjectiveOutput(
        loss=loss,
        policy_loss=zero,
        approx_kl=zero,
        clipfrac=zero,
        metrics=dict(metrics or {}),
    )


def _assert_guarded_failure(
    output_builder,
    *,
    expected_exception,
    match: str,
    engine_options: dict | None = None,
    adapter: _GuardAdapter | None = None,
    rewards: RewardBatch | None = None,
    advantage_function=None,
):
    batch = _batch(old_log_probs=torch.zeros(2, 1))
    adapter = adapter or _GuardAdapter()
    rewards = rewards or _rewards(batch)
    optimizer = _GuardOptimizer(adapter.parameters())
    parameter_before = adapter.parameter.detach().clone()
    optimizer_before = deepcopy(optimizer.state_dict())

    with pytest.raises(expected_exception, match=match):
        UpdateEngine(
            advantage_function or _advantage_computer(),
            _GuardObjective(output_builder),
            **dict(engine_options or {}),
        ).step(adapter, batch, rewards, optimizer, batch.context)

    assert optimizer.step_calls == 0
    torch.testing.assert_close(adapter.parameter.detach(), parameter_before)
    assert optimizer.state_dict() == optimizer_before
    return adapter, optimizer


class _AdvantageWithMetrics:
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, batch, rewards):
        del rewards
        return AdvantageResult(
            torch.ones(batch.batch_size),
            dict(self.metrics),
        )


def test_advantage_function_dataflow_and_legacy_compute_are_equivalent() -> None:
    computer = _advantage_computer()
    batch = _batch()
    rewards = _rewards(batch)

    result = computer(batch, rewards)
    legacy = computer.compute(
        batch.prompts,
        rewards.raw,
        rewards.weighted_total,
        group_ids=list(batch.group_id),
    )

    assert isinstance(result, AdvantageResult)
    torch.testing.assert_close(result.advantages, legacy.advantages)
    assert result.metrics == legacy.metrics


def test_advantage_function_uses_formal_groups_and_legacy_parent_groups() -> None:
    computer = _advantage_computer()
    formal = _batch(
        metadata=[
            {"parent_prompt_index": 0},
            {"parent_prompt_index": 1},
        ]
    )
    formal.group_id = ["declared", "declared"]
    formal_rewards = _rewards(formal)
    formal_expected = computer.compute(
        formal.prompts,
        formal_rewards.raw,
        formal_rewards.weighted_total,
        group_ids=["declared", "declared"],
    )
    torch.testing.assert_close(
        computer(formal, formal_rewards).advantages,
        formal_expected.advantages,
    )

    legacy = _batch(context=None)
    legacy_rewards = _rewards(legacy)
    legacy_expected = computer.compute(
        legacy.prompts,
        legacy_rewards.raw,
        legacy_rewards.weighted_total,
        group_ids=[4, 4],
    )
    torch.testing.assert_close(
        computer(legacy, legacy_rewards).advantages,
        legacy_expected.advantages,
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        GRPOAlgorithm(clip_range=0.2),
        FlashGRPOAlgorithm(
            clip_range=0.2,
            rectification={"enabled": True, "normalize": True},
        ),
        TempFlowGRPOAlgorithm(
            objective_version="legacy",
            clip_range=0.2,
            noise_weighting={"enabled": False},
        ),
    ],
    ids=["grpo", "flash", "tempflow"],
)
def test_algorithm_policy_objective_matches_existing_loss_kernels(algorithm) -> None:
    batch = _batch(
        model_metadata={
            "flash_rectification_weights": torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        }
    )
    advantages = torch.tensor([1.25, -0.5], dtype=torch.float64)
    direct_log_probs = torch.tensor(
        [[0.05, -0.1], [0.2, -0.15]],
        dtype=torch.float64,
        requires_grad=True,
    )
    adapted_log_probs = direct_log_probs.detach().clone().requires_grad_(True)

    direct_loss, direct_metrics = algorithm.compute_loss(
        batch,
        advantages,
        direct_log_probs,
    )
    output = AlgorithmPolicyObjective(algorithm)(
        batch,
        advantages,
        adapted_log_probs,
    )

    assert isinstance(output, ObjectiveOutput)
    assert output.loss.requires_grad
    torch.testing.assert_close(output.loss, direct_loss)
    for name, expected in direct_metrics.items():
        assert name in output.metrics
        torch.testing.assert_close(output.metrics[name], expected)
    torch.testing.assert_close(output.policy_loss, direct_metrics["policy_loss"])
    torch.testing.assert_close(output.approx_kl, direct_metrics["approx_kl"])
    torch.testing.assert_close(output.clipfrac, direct_metrics["clipfrac"])


def test_algorithm_policy_objective_requires_all_standard_metrics() -> None:
    class MissingMetricsAlgorithm:
        def compute_loss(self, batch, advantages, new_log_probs):
            del batch, advantages
            return new_log_probs.mean(), {"approx_kl": new_log_probs.mean()}

    with pytest.raises(ValueError, match="policy_loss, clipfrac"):
        AlgorithmPolicyObjective(MissingMetricsAlgorithm())(
            _batch(),
            torch.ones(2),
            torch.zeros(2, 2, requires_grad=True),
        )


def test_tempflow_reference_objective_keeps_provenance_and_float64_checks() -> None:
    algorithm = TempFlowGRPOAlgorithm(
        objective_version="reference_v1",
        noise_weighting={"enabled": True, "mode": "reference_std_dev_t"},
        advantage_dtype="float64",
        preserve_advantage_dtype=True,
    )
    objective = AlgorithmPolicyObjective(algorithm)
    old_log_probs = torch.zeros(2, 1, dtype=torch.float64)
    invalid_provenance = _batch(
        old_log_probs=old_log_probs,
        model_metadata={"transition_std_dev_t": [0.7]},
    )
    with pytest.raises(ValueError, match="tempflow_reference_mode"):
        objective(
            invalid_provenance,
            torch.ones(2, dtype=torch.float64),
            torch.zeros_like(old_log_probs, requires_grad=True),
        )

    valid_batch = _batch(
        old_log_probs=old_log_probs,
        model_metadata={
            "tempflow_reference_mode": True,
            "trajectory_contract_version": TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT,
            "recompute_transformer_training": True,
            "transition_std_dev_t": [0.7],
        },
    )
    with pytest.raises(TypeError, match="advantages.*dtype=torch.float64"):
        objective(
            valid_batch,
            torch.ones(2, dtype=torch.float32),
            torch.zeros_like(old_log_probs, requires_grad=True),
        )


def test_update_engine_rejects_reward_order_before_objective_or_optimizer() -> None:
    batch = _batch()
    rewards = _rewards(batch)
    rewards.sample_id.reverse()
    calls = {"advantage": 0, "objective": 0, "prepare": 0, "zero": 0, "step": 0}

    class Advantage:
        def __call__(self, batch, rewards):
            del batch, rewards
            calls["advantage"] += 1
            return AdvantageResult(torch.ones(2), {})

    class Objective:
        def __call__(self, batch, advantages, new_log_probs):
            del batch, advantages
            calls["objective"] += 1
            return ObjectiveOutput(
                loss=new_log_probs.mean(),
                policy_loss=new_log_probs.mean(),
                approx_kl=new_log_probs.mean(),
                clipfrac=new_log_probs.mean(),
                metrics={},
            )

    class Adapter:
        def prepare_for_training(self):
            calls["prepare"] += 1

        def recompute_log_probs(self, batch):
            return torch.zeros_like(batch.old_log_probs, requires_grad=True)

        def parameters(self):
            return []

    class Optimizer:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True
            calls["zero"] += 1

        def step(self):
            calls["step"] += 1

    with pytest.raises(ValueError, match="sample_id order"):
        UpdateEngine(Advantage(), Objective()).step(
            Adapter(),
            batch,
            rewards,
            Optimizer(),
            batch.context,
        )
    assert calls == {
        "advantage": 0,
        "objective": 0,
        "prepare": 0,
        "zero": 0,
        "step": 0,
    }


@pytest.mark.parametrize(
    ("loss_builder", "expected_exception", "message"),
    [
        (
            lambda new: new.mean() * 0.0 + float("nan"),
            ValueError,
            "loss must be finite",
        ),
        (
            lambda new: new.mean() * 0.0 + float("inf"),
            ValueError,
            "loss must be finite",
        ),
        (
            lambda new: new[:, 0],
            ValueError,
            "loss must be a scalar tensor",
        ),
        (
            lambda new: torch.tensor(1.0, device=new.device),
            ValueError,
            "loss must require gradients",
        ),
        (
            lambda new: 1.0,
            TypeError,
            "loss must be a floating scalar torch.Tensor with backward",
        ),
        (
            lambda new: torch.tensor(1, device=new.device),
            TypeError,
            "loss must be a floating tensor",
        ),
    ],
    ids=[
        "nan-zero-grad",
        "inf-zero-grad",
        "non-scalar",
        "no-grad",
        "no-backward",
        "non-floating",
    ],
)
def test_loss_validation_fails_before_update(
    loss_builder,
    expected_exception,
    message: str,
) -> None:
    _assert_guarded_failure(
        lambda new: _valid_output(new, loss=loss_builder(new)),
        expected_exception=expected_exception,
        match=message,
    )


class _NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.new_tensor(1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * float("inf")


def test_nonfinite_gradient_fails_without_parameter_or_state_update() -> None:
    _assert_guarded_failure(
        lambda new: _valid_output(
            new,
            loss=_NonFiniteGradient.apply(new.mean()),
        ),
        expected_exception=RuntimeError,
        match="non-finite gradient",
    )


def test_required_nonzero_gradient_fails_without_parameter_or_state_update() -> None:
    _assert_guarded_failure(
        lambda new: _valid_output(new, loss=new.mean() * 0.0 + 1.0),
        expected_exception=RuntimeError,
        match="all gradients are zero",
        engine_options={"require_nonzero_gradients": True},
    )


def test_nonfinite_logprob_metrics_fail_before_update() -> None:
    adapter = _GuardAdapter(
        lambda parameter, batch: (
            parameter * 0.0 + float("inf")
        ).expand_as(batch.old_log_probs)
    )
    _assert_guarded_failure(
        lambda new: _valid_output(new, loss=adapter.parameter),
        expected_exception=ValueError,
        match="logprob.*must be finite",
        adapter=adapter,
    )


def test_nonfinite_reward_metrics_fail_before_update() -> None:
    batch = _batch(old_log_probs=torch.zeros(2, 1))
    values = torch.tensor([3.0e38, 3.0e38], dtype=torch.float32)
    rewards = RewardBatch(
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(2, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    assert torch.isfinite(values).all()
    assert torch.isinf(values.mean())

    _assert_guarded_failure(
        lambda new: _valid_output(new),
        expected_exception=ValueError,
        match="reward.reward_mean must be finite",
        rewards=rewards,
    )


@pytest.mark.parametrize(
    ("gate", "engine_options", "message"),
    [
        (
            "logprob",
            {"max_initial_logprob_delta": 0.1},
            "log-prob parity gate failed",
        ),
        (
            "clipfrac",
            {"require_initial_clipfrac_zero": True},
            "clipfrac gate failed",
        ),
    ],
)
def test_pre_update_gates_leave_parameter_and_optimizer_state_unchanged(
    gate: str,
    engine_options: dict,
    message: str,
) -> None:
    adapter = (
        _GuardAdapter(
            lambda parameter, batch: (parameter + 1.0).expand_as(
                batch.old_log_probs
            )
        )
        if gate == "logprob"
        else _GuardAdapter()
    )

    def output_builder(new):
        output = _valid_output(new)
        if gate == "clipfrac":
            output.clipfrac = torch.tensor(0.5)
        return output

    _assert_guarded_failure(
        output_builder,
        expected_exception=RuntimeError,
        match=message,
        engine_options=engine_options,
        adapter=adapter,
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_initial_logprob_delta", float("nan")),
        ("max_initial_logprob_delta", float("inf")),
        ("max_initial_logprob_delta", float("-inf")),
        ("max_initial_logprob_delta", -1.0),
        ("max_grad_norm", float("nan")),
        ("max_grad_norm", float("inf")),
        ("max_grad_norm", float("-inf")),
        ("max_grad_norm", -1.0),
        ("max_grad_norm", 0.0),
    ],
)
def test_update_engine_rejects_invalid_thresholds_without_update(
    option: str,
    value: float,
) -> None:
    _assert_guarded_failure(
        lambda new: _valid_output(new),
        expected_exception=ValueError,
        match=option,
        engine_options={option: value},
    )


def test_update_engine_accepts_threshold_boundaries() -> None:
    defaults = UpdateEngine(_advantage_computer(), _GuardObjective(_valid_output))
    bounded = UpdateEngine(
        _advantage_computer(),
        _GuardObjective(_valid_output),
        max_initial_logprob_delta=0.0,
        max_grad_norm=0.5,
    )

    assert defaults.max_initial_logprob_delta is None
    assert defaults.max_grad_norm is None
    assert bounded.max_initial_logprob_delta == 0.0
    assert bounded.max_grad_norm == 0.5


@pytest.mark.parametrize(
    ("optimizer_config", "max_grad_norm", "message"),
    [
        ({"max_initial_logprob_delta": float("nan")}, None, "must be finite"),
        ({"max_initial_logprob_delta": float("inf")}, None, "must be finite"),
        ({}, float("nan"), "must be finite"),
        ({}, float("inf"), "must be finite"),
    ],
)
def test_algorithm_plugin_reuses_update_engine_threshold_validation(
    optimizer_config: dict,
    max_grad_norm: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AlgorithmOptimizerPlugin(
            GRPOAlgorithm(),
            _advantage_computer(),
            optimizer_config=optimizer_config,
            max_grad_norm=max_grad_norm,
        )


@pytest.mark.parametrize(
    ("target", "bad_value", "expected_exception", "message"),
    [
        (
            "policy_loss",
            torch.tensor([0.0, 1.0]),
            ValueError,
            "policy_loss must be a scalar tensor",
        ),
        (
            "approx_kl",
            torch.tensor(float("nan")),
            ValueError,
            "approx_kl must be finite",
        ),
        (
            "clipfrac",
            True,
            TypeError,
            "clipfrac must not be bool",
        ),
        (
            "policy_loss",
            "invalid",
            TypeError,
            "policy_loss must be a finite scalar tensor",
        ),
        (
            "extra",
            torch.tensor([0.0, 1.0]),
            ValueError,
            "bad_extra must be a scalar tensor",
        ),
        (
            "extra",
            float("inf"),
            ValueError,
            "bad_extra must be finite",
        ),
        (
            "extra",
            False,
            TypeError,
            "bad_extra must not be bool",
        ),
        (
            "extra",
            object(),
            TypeError,
            "bad_extra must be a finite scalar tensor",
        ),
    ],
    ids=[
        "standard-nonscalar",
        "standard-nonfinite",
        "standard-bool",
        "standard-unsupported",
        "extra-nonscalar",
        "extra-nonfinite",
        "extra-bool",
        "extra-unsupported",
    ],
)
def test_invalid_objective_metrics_fail_before_update(
    target: str,
    bad_value,
    expected_exception,
    message: str,
) -> None:
    def output_builder(new):
        output = _valid_output(new)
        if target == "extra":
            output.metrics["bad_extra"] = bad_value
        else:
            setattr(output, target, bad_value)
        return output

    _assert_guarded_failure(
        output_builder,
        expected_exception=expected_exception,
        match=message,
    )


@pytest.mark.parametrize(
    ("objective_metrics", "advantage_metrics", "conflicting_key"),
    [
        ({"loss": 7.0}, {}, "loss"),
        ({}, {"reward_mean": 7.0}, "reward_mean"),
        ({}, {"approx_kl": 7.0}, "approx_kl"),
        ({"shared_extension": 1.0}, {"shared_extension": 2.0}, "shared_extension"),
    ],
    ids=[
        "objective-canonical",
        "advantage-reward",
        "advantage-objective",
        "extension-extension",
    ],
)
def test_metric_key_collisions_fail_before_update(
    objective_metrics: dict,
    advantage_metrics: dict,
    conflicting_key: str,
) -> None:
    _assert_guarded_failure(
        lambda new: _valid_output(new, metrics=objective_metrics),
        expected_exception=ValueError,
        match=f"Metric key collision for {conflicting_key!r}",
        advantage_function=_AdvantageWithMetrics(advantage_metrics),
    )


@pytest.mark.parametrize(
    ("section", "key", "expected_exception"),
    [
        ("objective", "", ValueError),
        ("objective", 7, TypeError),
        ("advantage", "", ValueError),
        ("advantage", 7, TypeError),
    ],
)
def test_metric_keys_must_be_nonempty_strings(
    section: str,
    key,
    expected_exception,
) -> None:
    objective_metrics = {key: 1.0} if section == "objective" else {}
    advantage_metrics = {key: 1.0} if section == "advantage" else {}
    _assert_guarded_failure(
        lambda new: _valid_output(new, metrics=objective_metrics),
        expected_exception=expected_exception,
        match="metric keys must be non-empty str",
        advantage_function=_AdvantageWithMetrics(advantage_metrics),
    )


@pytest.mark.parametrize("clip_failure", ["return-nonfinite", "produce-nonfinite"])
def test_nonfinite_gradient_clipping_fails_before_update(
    monkeypatch,
    clip_failure: str,
) -> None:
    def bad_clip(parameters, max_norm):
        del max_norm
        if clip_failure == "produce-nonfinite":
            for parameter in parameters:
                parameter.grad.fill_(float("nan"))
            return torch.tensor(1.0)
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", bad_clip)
    message = (
        "Gradient clip gate failed"
        if clip_failure == "produce-nonfinite"
        else "grad_preclip_norm must be finite"
    )
    _assert_guarded_failure(
        lambda new: _valid_output(new),
        expected_exception=(RuntimeError if clip_failure == "produce-nonfinite" else ValueError),
        match=message,
        engine_options={"max_grad_norm": 1.0},
    )


def test_update_engine_runs_one_complete_gradient_update() -> None:
    batch = _batch(old_log_probs=torch.zeros(2, 1))
    rewards = _rewards(batch)
    calls = {"prepare": 0, "recompute": 0, "zero": 0, "step": 0}

    class Adapter:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))

        def prepare_for_training(self):
            calls["prepare"] += 1

        def recompute_log_probs(self, batch):
            calls["recompute"] += 1
            return self.parameter.expand_as(batch.old_log_probs)

        def parameters(self):
            return [self.parameter]

    class Objective:
        def __call__(self, batch, advantages, new_log_probs):
            del batch, advantages
            loss = new_log_probs.mean() * 2.0
            zero = new_log_probs.sum() * 0.0
            return ObjectiveOutput(
                loss=loss,
                policy_loss=loss.detach(),
                approx_kl=zero.detach(),
                clipfrac=zero.detach(),
                metrics={"mock_extra": torch.tensor(3.0)},
            )

    class CountingSGD(torch.optim.SGD):
        def zero_grad(self, *, set_to_none=True):
            calls["zero"] += 1
            return super().zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            calls["step"] += 1
            return super().step(closure=closure)

    adapter = Adapter()
    optimizer = CountingSGD(adapter.parameters(), lr=0.1)
    metrics = UpdateEngine(
        _advantage_computer(),
        Objective(),
        require_nonzero_gradients=True,
        max_grad_norm=1.0,
    ).step(adapter, batch, rewards, optimizer, batch.context)

    assert calls == {"prepare": 1, "recompute": 1, "zero": 1, "step": 1}
    assert adapter.parameter.item() < 1.0
    assert metrics["gradients_finite"] is True
    assert metrics["grad_nonzero_count"] == 1
    assert metrics["grad_tensor_count"] == 1
    assert metrics["grad_preclip_norm"] == pytest.approx(2.0)
    assert metrics["mock_extra"] == 3.0
    assert metrics["zero_std_ratio"] == 0.0


def test_algorithm_plugin_state_dict_remains_advantage_only() -> None:
    plugin = AlgorithmOptimizerPlugin(GRPOAlgorithm(), _advantage_computer())

    assert plugin.state_dict() == {"advantage": {}}
    plugin.load_state_dict({"advantage": {}})


def test_factory_still_allows_a_complete_custom_optimizer_plugin() -> None:
    class CustomPlugin(OptimizerPlugin):
        def build_optimizer(self, parameters, train_config):
            del train_config
            return torch.optim.SGD(parameters, lr=0.1)

        def step(self, adapter, batch, rewards, optimizer, context):
            del adapter, batch, rewards, optimizer, context
            return {"custom": 1.0}

    name = "c6_custom_complete_plugin"
    OPTIMIZER_PLUGINS.register(name, lambda config: CustomPlugin())
    config = load_config(
        ROOT / "visual_rl/configs/presets/flash_tiny_single_step.yaml"
    )
    config.optimizer.name = name

    assert isinstance(build_optimizer_plugin(config), CustomPlugin)
