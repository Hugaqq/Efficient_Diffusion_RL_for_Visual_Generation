"""The one UpdateEngine path from typed inputs through atomic AdamW."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from visual_rl.core.types import (
    FrozenMapping,
    PolicyRecomputeStats,
    RewardBatch,
    RolloutBatch,
    StepContext,
    ValidatedRuntimeEnv,
)
from visual_rl.distributed import build_strategy
from visual_rl.optimizers.advantages import AdvantageComputer
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.optimizers.update_engine import UpdateEngine, UpdateResult


_GRADIENT_NORM_FIELDS = (
    "update/gradient_norm_pre_clip",
    "update/gradient_norm_post_clip",
)


class _TrainModule(torch.nn.Module):
    def __init__(self, value: float = 0.05) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(value))


class _Adapter:
    def __init__(self, value: float = 0.05) -> None:
        self.train_module = _TrainModule(value)
        self.recompute_calls = 0
        self.reference_calls = 0

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        self.recompute_calls += 1
        sample_weight = batch.recompute_payload["sample_weight"]
        features = sample_weight * (
            batch.timesteps.to(dtype=torch.float32) + 1.0
        )
        new_log_probs = self.train_module.delta * features
        if not require_reference:
            return PolicyRecomputeStats(new_log_probs=new_log_probs)
        self.reference_calls += 1
        current = batch.next_latents + self.train_module.delta
        reference = batch.next_latents.detach().clone()
        return PolicyRecomputeStats(
            new_log_probs=new_log_probs,
            current_transition_mean=current,
            transition_std=torch.ones_like(batch.old_log_probs),
            reference_transition_mean=reference,
        )

    def named_parameters(self):
        return tuple(self.train_module.named_parameters())


class _MismatchedDtypeAlgorithm(GRPOAlgorithm):
    ADVANTAGE_DTYPE = "float64"


def _strategy(adapter: _Adapter):
    strategy = build_strategy(
        SimpleNamespace(
            mode="single",
            device="cpu",
            timeout_s=5.0,
            max_snapshot_tensor_bytes=None,
        ),
        ValidatedRuntimeEnv(
            mode="single",
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            group_rank=None,
            group_world_size=None,
            master_addr=None,
            master_port=None,
            visible_gpu_count=0,
            raw_launch_env=FrozenMapping(),
        ),
    )
    strategy.prepare(adapter)
    return strategy


def _batch(
    *,
    context: StepContext | None = None,
    transition_mask: torch.Tensor | None = None,
) -> RolloutBatch:
    context = context or StepContext(step=0, seed=17)
    batch_size, transition_count = 4, 2
    mask = (
        torch.ones(batch_size, transition_count, dtype=torch.bool)
        if transition_mask is None
        else transition_mask
    )
    return RolloutBatch(
        prompts=("a", "a", "b", "b"),
        metadata=({}, {}, {}, {}),
        media=torch.zeros(batch_size, 1, 1, 1),
        latents=torch.zeros(batch_size, transition_count, 1),
        next_latents=torch.ones(batch_size, transition_count, 1),
        timesteps=torch.arange(transition_count).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transition_count),
        transition_mask=mask,
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=("prompt-a", "prompt-a", "prompt-b", "prompt-b"),
        group_id=("group-a", "group-a", "group-b", "group-b"),
        branch_id=None,
        media_layout="BCHW",
        camera_trajectory=None,
        context=context,
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={
            "sample_weight": torch.arange(
                1,
                batch_size + 1,
                dtype=torch.float32,
            ).reshape(batch_size, 1)
        },
        artifact_metadata={},
    )


def _rewards(batch: RolloutBatch) -> RewardBatch:
    values = torch.tensor([1.0, 3.0, 2.0, 4.0])
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": ({}, {}, {}, {})},
    )


def _plugin(
    *,
    microbatch_size: int,
    transition_microbatch_size: int | None = None,
    beta: float = 0.0,
    max_grad_norm: float | None = None,
) -> AlgorithmOptimizerPlugin:
    return AlgorithmOptimizerPlugin(
        algorithm=GRPOAlgorithm(
            clip_range=0.2,
            adv_clip_max=5.0,
            beta=beta,
        ),
        advantage_computer=AdvantageComputer(
            epsilon=1e-8,
            output_dtype="float32",
        ),
        update_microbatch_size=microbatch_size,
        transition_microbatch_size=transition_microbatch_size,
        precision="fp32",
        max_grad_norm=max_grad_norm,
        max_initial_logprob_delta=None,
        require_initial_clipfrac_zero=False,
        require_finite_gradients=True,
        require_nonzero_gradients=True,
    )


def _optimizer(
    plugin: AlgorithmOptimizerPlugin,
    adapter: _Adapter,
) -> torch.optim.AdamW:
    return plugin.build_optimizer(
        adapter.named_parameters(),
        SimpleNamespace(
            learning_rate=1e-2,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_weight_decay=0.0,
            adam_epsilon=1e-8,
        ),
    )


def _run_once(
    *,
    microbatch_size: int,
    transition_microbatch_size: int | None = None,
    beta: float = 0.0,
    max_grad_norm: float | None = None,
):
    adapter = _Adapter()
    strategy = _strategy(adapter)
    plugin = _plugin(
        microbatch_size=microbatch_size,
        transition_microbatch_size=transition_microbatch_size,
        beta=beta,
        max_grad_norm=max_grad_norm,
    )
    optimizer = _optimizer(plugin, adapter)
    batch = _batch()
    try:
        result = plugin.step(
            batch=batch,
            rewards=_rewards(batch),
            optimizer=optimizer,
            scaler=None,
            context=batch.context,
            strategy=strategy,
        )
        return (
            result,
            adapter.train_module.delta.detach().clone(),
            adapter.train_module.delta.grad.detach().clone(),
            adapter.recompute_calls,
            adapter.reference_calls,
        )
    finally:
        strategy.close()


def test_microbatch_and_full_batch_use_one_active_count_weighting() -> None:
    full = _run_once(microbatch_size=4)
    split = _run_once(microbatch_size=3)
    full_result, full_parameter, full_gradient, full_forwards, _ = full
    split_result, split_parameter, split_gradient, split_forwards, _ = split

    assert isinstance(full_result, UpdateResult)
    assert full_result.active_transition_count == 8
    assert split_result.active_transition_count == 8
    for name in (
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
    ):
        assert getattr(split_result, name) == pytest.approx(
            getattr(full_result, name),
            abs=1e-7,
        )
    assert tuple(full_result.diagnostics)[-2:] == _GRADIENT_NORM_FIELDS
    assert split_result.diagnostics == pytest.approx(
        full_result.diagnostics,
        abs=5e-7,
    )
    torch.testing.assert_close(split_parameter, full_parameter)
    torch.testing.assert_close(split_gradient, full_gradient)
    assert full_forwards == 1
    assert split_forwards == 2


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_transition_microbatch_preserves_the_one_objective_and_update(
    beta: float,
) -> None:
    full = _run_once(
        microbatch_size=4,
        transition_microbatch_size=None,
        beta=beta,
    )
    split = _run_once(
        microbatch_size=4,
        transition_microbatch_size=1,
        beta=beta,
    )
    full_result, full_parameter, full_gradient, full_forwards, full_references = (
        full
    )
    (
        split_result,
        split_parameter,
        split_gradient,
        split_forwards,
        split_references,
    ) = split

    assert full_result.active_transition_count == 8
    assert split_result.active_transition_count == 8
    for name in (
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
    ):
        assert getattr(split_result, name) == pytest.approx(
            getattr(full_result, name),
            abs=1e-7,
        )
    assert split_result.diagnostics == pytest.approx(
        full_result.diagnostics,
        abs=5e-7,
    )
    torch.testing.assert_close(split_parameter, full_parameter)
    torch.testing.assert_close(split_gradient, full_gradient)
    assert full_forwards == 1
    assert split_forwards == 2
    assert full_references == (1 if beta > 0.0 else 0)
    assert split_references == (2 if beta > 0.0 else 0)


def test_beta_zero_never_requests_reference_statistics() -> None:
    result, _parameter, _gradient, _forwards, reference_calls = _run_once(
        microbatch_size=2,
        beta=0.0,
    )
    assert result.reference_kl == 0.0
    assert reference_calls == 0


def test_beta_positive_adds_differentiable_reference_loss() -> None:
    control = _run_once(microbatch_size=2, beta=0.0)
    active = _run_once(microbatch_size=2, beta=0.5)
    active_result, _active_parameter, active_gradient, _forwards, reference_calls = (
        active
    )
    assert reference_calls == 2
    assert active_result.reference_kl > 0.0
    assert active_result.loss == pytest.approx(
        active_result.policy_loss + 0.5 * active_result.reference_kl
    )
    assert not torch.equal(active_gradient, control[2])


def test_gradient_norm_diagnostics_report_actual_clipping() -> None:
    max_grad_norm = 0.01
    unclipped = _run_once(microbatch_size=4, max_grad_norm=None)
    result, _parameter, gradient, _forwards, _references = _run_once(
        microbatch_size=4,
        max_grad_norm=max_grad_norm,
    )

    pre_clip = result.diagnostics["update/gradient_norm_pre_clip"]
    post_clip = result.diagnostics["update/gradient_norm_post_clip"]
    assert pre_clip == pytest.approx(float(unclipped[2].abs()), abs=1e-8)
    assert post_clip == pytest.approx(float(gradient.abs()), abs=1e-8)
    assert post_clip == pytest.approx(max_grad_norm, rel=1e-4)


def test_no_clip_norm_diagnostic_does_not_write_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])
    parameters = (first, second)
    before = tuple(parameter.grad.detach().clone() for parameter in parameters)

    pre_clip, post_clip = UpdateEngine._gradient_gate_and_clip(
        parameters,
        require_finite=True,
        require_nonzero=True,
        max_grad_norm=None,
    )

    assert float(pre_clip) == pytest.approx(13.0)
    assert float(post_clip) == pytest.approx(13.0)
    for parameter, expected in zip(parameters, before, strict=True):
        torch.testing.assert_close(
            parameter.grad,
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_no_clip_diagnostic_preserves_disabled_finite_gate() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])

    pre_clip, post_clip = UpdateEngine._gradient_gate_and_clip(
        (parameter,),
        require_finite=False,
        require_nonzero=False,
        max_grad_norm=None,
    )

    assert pre_clip is None
    assert post_clip is None
    assert torch.isinf(parameter.grad).all()
    assert UpdateEngine._gradient_norm_contributions(
        pre_clip,
        post_clip,
        device="cpu",
    ) == {}


def test_update_result_accepts_only_declared_diagnostic_namespaces() -> None:
    accepted = UpdateResult(
        loss=0.0,
        policy_loss=0.0,
        reference_kl=0.0,
        approx_kl=0.0,
        clipfrac=0.0,
        active_transition_count=1,
        diagnostics={"update/gradient_norm_pre_clip": 1.0},
    )
    assert accepted.diagnostics["update/gradient_norm_pre_clip"] == 1.0

    with pytest.raises(ValueError, match="diagnostics keys"):
        UpdateResult(
            loss=0.0,
            policy_loss=0.0,
            reference_kl=0.0,
            approx_kl=0.0,
            clipfrac=0.0,
            active_transition_count=1,
            diagnostics={"optimizer/gradient_norm": 1.0},
        )


def test_context_identity_failure_precedes_forward_and_mutation() -> None:
    adapter = _Adapter()
    strategy = _strategy(adapter)
    plugin = _plugin(microbatch_size=2)
    optimizer = _optimizer(plugin, adapter)
    batch = _batch()
    initial = adapter.train_module.delta.detach().clone()
    equal_but_distinct = StepContext(
        step=batch.context.step,
        seed=batch.context.seed,
        rank=batch.context.rank,
        world_size=batch.context.world_size,
    )
    try:
        with pytest.raises(ValueError, match="RolloutBatch.context object"):
            plugin.step(
                batch=batch,
                rewards=_rewards(batch),
                optimizer=optimizer,
                scaler=None,
                context=equal_but_distinct,
                strategy=strategy,
            )
        assert adapter.recompute_calls == 0
        torch.testing.assert_close(adapter.train_module.delta, initial)
        assert optimizer.state == {}
    finally:
        strategy.close()


def test_each_sample_requires_an_active_transition_before_forward() -> None:
    mask = torch.tensor(
        [
            [True, False],
            [False, False],
            [True, True],
            [False, True],
        ]
    )
    adapter = _Adapter()
    strategy = _strategy(adapter)
    plugin = _plugin(microbatch_size=2)
    optimizer = _optimizer(plugin, adapter)
    batch = _batch(transition_mask=mask)
    try:
        with pytest.raises(ValueError, match="every sample"):
            plugin.step(
                batch=batch,
                rewards=_rewards(batch),
                optimizer=optimizer,
                scaler=None,
                context=batch.context,
                strategy=strategy,
            )
        assert adapter.recompute_calls == 0
        assert optimizer.state == {}
    finally:
        strategy.close()


def test_algorithm_advantage_dtype_mismatch_fails_before_forward() -> None:
    adapter = _Adapter()
    strategy = _strategy(adapter)
    plugin = _plugin(microbatch_size=2)
    plugin.algorithm = _MismatchedDtypeAlgorithm(
        clip_range=0.2,
        adv_clip_max=5.0,
        beta=0.0,
    )
    plugin.update_engine.algorithm = plugin.algorithm
    optimizer = _optimizer(plugin, adapter)
    batch = _batch()
    initial = adapter.train_module.delta.detach().clone()
    try:
        with pytest.raises(TypeError, match="ADVANTAGE_DTYPE=float64"):
            plugin.step(
                batch=batch,
                rewards=_rewards(batch),
                optimizer=optimizer,
                scaler=None,
                context=batch.context,
                strategy=strategy,
            )
        assert adapter.recompute_calls == 0
        torch.testing.assert_close(adapter.train_module.delta, initial)
        assert optimizer.state == {}
    finally:
        strategy.close()


def test_optimizer_plugin_step_has_only_the_six_required_keywords() -> None:
    signature = inspect.signature(OptimizerPlugin.step)
    assert tuple(signature.parameters) == (
        "self",
        "batch",
        "rewards",
        "optimizer",
        "scaler",
        "context",
        "strategy",
    )
    for name, parameter in signature.parameters.items():
        if name != "self":
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is inspect.Parameter.empty
