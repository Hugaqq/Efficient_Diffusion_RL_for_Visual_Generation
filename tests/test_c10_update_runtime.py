"""CPU-only coverage for C10 update microbatching and precision policy."""

from __future__ import annotations

import pytest
import torch

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.optimizers import AdvantageResult, ObjectiveOutput, UpdateEngine
from visual_rl.optimizers import AlgorithmPolicyObjective
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.optimizers.tempflow_grpo import (
    TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT,
    TempFlowGRPOAlgorithm,
)


def _batch(*, transition_mask: torch.Tensor | None = None) -> RolloutBatch:
    batch_size, transitions = 4, 3
    context = StepContext(step=2, seed=17, epoch_tag=1)
    return RolloutBatch(
        prompts=[f"prompt-{index}" for index in range(batch_size)],
        metadata=[{"position": index} for index in range(batch_size)],
        media=torch.arange(batch_size, dtype=torch.float32)[:, None],
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions),
        sample_id=[f"sample-{index}" for index in range(batch_size)],
        prompt_id=[f"prompt-id-{index}" for index in range(batch_size)],
        group_id=["group-a", "group-a", "group-b", "group-b"],
        branch_id=list(range(batch_size)),
        transition_mask=(
            torch.ones(batch_size, transitions, dtype=torch.bool)
            if transition_mask is None
            else transition_mask
        ),
        context=context,
        model_metadata={
            "selected_timestep_indices": [10, 11, 12, 13],
            "nested": {"coefficient": torch.arange(batch_size)[:, None]},
            "scheduler_name": "test-scheduler",
        },
        model_tensors={
            "features": torch.arange(
                1, batch_size * transitions + 1, dtype=torch.float32
            ).reshape(batch_size, transitions),
        },
    )


def _rewards(batch: RolloutBatch) -> RewardBatch:
    values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    return RewardBatch(
        raw={"score": values, "aux": values + 1.0},
        weighted={"score": values, "aux": values + 1.0},
        weighted_total=values * 2.0 + 1.0,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        metadata={"source_rows": [20, 21, 22, 23], "provider": "test"},
        sample_id=list(batch.sample_id),
    )


class _CountingAdvantage:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, batch, rewards):
        del rewards
        self.calls += 1
        return AdvantageResult(
            torch.arange(1, batch.batch_size + 1, dtype=torch.float32),
            {"advantage_calls": float(self.calls)},
        )


class _Adapter:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.25))
        self.prepare_calls = 0
        self.recompute_order: list[str] = []

    def prepare_for_training(self) -> None:
        self.prepare_calls += 1

    def recompute_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        self.recompute_order.extend(batch.sample_id)
        return self.parameter * batch.model_tensors["features"]

    def parameters(self):
        return [self.parameter]


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.01)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *, set_to_none=True):
        self.zero_calls += 1
        return super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _MaskedMeanObjective:
    def __call__(self, batch, advantages, new_log_probs):
        values = new_log_probs * advantages[:, None]
        loss = values.masked_select(batch.transition_mask).mean()
        detached = loss.detach()
        zero = detached * 0.0
        return ObjectiveOutput(
            loss=loss,
            policy_loss=detached,
            approx_kl=zero,
            clipfrac=zero,
            metrics={"objective_mean": detached},
        )


def test_rollout_and_reward_slices_preserve_order_and_metadata() -> None:
    batch = _batch()
    rewards = _rewards(batch)

    selected_batch = batch.slice([3, 1])
    selected_rewards = rewards.select([3, 1])

    assert selected_batch.prompts == ["prompt-3", "prompt-1"]
    assert selected_batch.sample_id == ["sample-3", "sample-1"]
    assert selected_batch.metadata == [{"position": 3}, {"position": 1}]
    assert selected_batch.context == batch.context
    assert selected_batch.model_metadata["scheduler_name"] == "test-scheduler"
    assert selected_batch.model_metadata["selected_timestep_indices"] == [13, 11]
    torch.testing.assert_close(
        selected_batch.model_metadata["nested"]["coefficient"],
        torch.tensor([[3], [1]]),
    )
    torch.testing.assert_close(
        selected_batch.model_tensors["features"],
        batch.model_tensors["features"][[3, 1]],
    )
    assert selected_rewards.sample_id == ["sample-3", "sample-1"]
    assert selected_rewards.metadata["source_rows"] == [23, 21]
    torch.testing.assert_close(
        selected_rewards.raw["aux"],
        rewards.raw["aux"][[3, 1]],
    )

    for invalid in ([], [-1], [4], [1, 1]):
        with pytest.raises((ValueError, IndexError)):
            batch.slice(invalid)
        with pytest.raises((ValueError, IndexError)):
            rewards.slice(invalid)


def _run_update(microbatch_size: int | None):
    mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    batch = _batch(transition_mask=mask)
    rewards = _rewards(batch)
    advantage = _CountingAdvantage()
    adapter = _Adapter()
    optimizer = _CountingSGD(adapter.parameters())
    metrics = UpdateEngine(
        advantage,
        _MaskedMeanObjective(),
        update_microbatch_size=microbatch_size,
    ).step(adapter, batch, rewards, optimizer, batch.context)
    return advantage, adapter, optimizer, metrics


def test_full_advantage_once_and_one_optimizer_step_for_microbatches() -> None:
    advantage, adapter, optimizer, metrics = _run_update(2)

    assert advantage.calls == 1
    assert adapter.prepare_calls == 1
    assert adapter.recompute_order == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    assert optimizer.zero_calls == 1
    assert optimizer.step_calls == 1
    assert metrics["update_microbatches"] == 2
    assert metrics["recompute_time_s"] >= 0.0
    assert metrics["backward_time_s"] >= 0.0
    assert metrics["optimizer_time_s"] >= 0.0


def test_full_and_microbatch_simple_objective_updates_are_equivalent() -> None:
    _, full_adapter, full_optimizer, full_metrics = _run_update(None)
    _, micro_adapter, micro_optimizer, micro_metrics = _run_update(2)

    assert full_optimizer.step_calls == micro_optimizer.step_calls == 1
    torch.testing.assert_close(
        micro_adapter.parameter,
        full_adapter.parameter,
        rtol=1e-6,
        atol=1e-7,
    )
    assert micro_metrics["loss"] == pytest.approx(full_metrics["loss"])
    assert micro_metrics["policy_loss"] == pytest.approx(full_metrics["policy_loss"])


def test_bf16_cpu_recompute_smoke_keeps_objective_float32() -> None:
    batch = _batch()
    rewards = _rewards(batch)

    class MatmulAdapter(_Adapter):
        def __init__(self) -> None:
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(2, 1))

        def recompute_log_probs(self, batch):
            features = batch.model_tensors["features"].unsqueeze(-1).expand(-1, -1, 2)
            return torch.matmul(features, self.parameter).squeeze(-1)

    class Float32Objective(_MaskedMeanObjective):
        def __call__(self, batch, advantages, new_log_probs):
            assert advantages.dtype == torch.float32
            assert new_log_probs.dtype == torch.float32
            return super().__call__(batch, advantages, new_log_probs)

    adapter = MatmulAdapter()
    optimizer = _CountingSGD(adapter.parameters())
    metrics = UpdateEngine(
        _CountingAdvantage(),
        Float32Objective(),
        update_microbatch_size=2,
        precision="bf16",
    ).step(adapter, batch, rewards, optimizer, batch.context)

    assert optimizer.step_calls == 1
    assert metrics["update_microbatches"] == 2


def test_fp16_cpu_fails_before_training_or_optimizer_mutation() -> None:
    batch = _batch()
    rewards = _rewards(batch)
    advantage = _CountingAdvantage()
    adapter = _Adapter()
    optimizer = _CountingSGD(adapter.parameters())
    before = adapter.parameter.detach().clone()

    with pytest.raises(RuntimeError, match="fp16.*CUDA"):
        UpdateEngine(
            advantage,
            _MaskedMeanObjective(),
            precision="fp16",
        ).step(adapter, batch, rewards, optimizer, batch.context)

    assert advantage.calls == 0
    assert adapter.prepare_calls == 0
    assert optimizer.zero_calls == 0
    assert optimizer.step_calls == 0
    torch.testing.assert_close(adapter.parameter, before)


def test_invalid_reward_mask_fails_before_advantage_or_training() -> None:
    batch = _batch()
    rewards = _rewards(batch)
    rewards.valid_mask[2] = False
    advantage = _CountingAdvantage()
    adapter = _Adapter()
    optimizer = _CountingSGD(adapter.parameters())

    with pytest.raises(ValueError, match=r"invalid valid_mask indices: \[2\]"):
        UpdateEngine(advantage, _MaskedMeanObjective()).step(
            adapter,
            batch,
            rewards,
            optimizer,
            batch.context,
        )

    assert advantage.calls == 0
    assert adapter.prepare_calls == 0
    assert optimizer.zero_calls == 0
    assert optimizer.step_calls == 0


class _FixedAdvantage:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def __call__(self, batch, rewards):
        del rewards
        assert batch.batch_size == self.values.shape[0]
        return AdvantageResult(
            self.values.clone(),
            {"fixed_advantage_mean": self.values.mean()},
        )


class _AlgorithmAdapter:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.07, dtype=torch.float64))

    def prepare_for_training(self) -> None:
        return None

    def recompute_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        return self.parameter * batch.model_tensors["features"]

    def parameters(self):
        return [self.parameter]


def _algorithm_batch(case: str) -> RolloutBatch:
    batch_size, transitions = 4, 3
    model_metadata = {
        "selected_timestep_indices": [0, 2, 1, 2],
        "num_steps": transitions,
    }
    model_tensors = {
        "features": torch.tensor(
            [
                [0.2, -0.4, 0.8],
                [1.1, -0.3, 0.5],
                [-0.7, 0.9, 0.4],
                [0.6, -1.2, 0.3],
            ],
            dtype=torch.float64,
        )
    }
    if case == "flash_reference":
        model_tensors["coefficient"] = torch.tensor(
            [[1.0], [9.0], [3.0], [6.0]],
            dtype=torch.float64,
        )
    elif case == "flash_custom":
        model_metadata["flash_rectification_weights"] = torch.tensor(
            [
                [1.0, 2.0, 4.0],
                [8.0, 3.0, 5.0],
                [2.0, 7.0, 1.0],
                [6.0, 9.0, 4.0],
            ],
            dtype=torch.float64,
        )
    elif case == "tempflow_reference":
        model_metadata.update(
            {
                "tempflow_reference_mode": True,
                "trajectory_contract_version": (TEMPFLOW_REFERENCE_TRAJECTORY_CONTRACT),
                "recompute_transformer_training": True,
                "trajectory_step_indices": [0, 1, 2],
                "transition_std_dev_t": torch.tensor(
                    [
                        [0.2, 0.5, 0.9],
                        [1.3, 0.7, 0.4],
                        [0.6, 1.1, 0.3],
                        [0.8, 0.4, 1.5],
                    ],
                    dtype=torch.float64,
                ),
            }
        )
    elif case == "tempflow_custom":
        model_metadata["noise_weights"] = torch.tensor(
            [
                [1.0, 2.0, 8.0],
                [4.0, 9.0, 3.0],
                [7.0, 2.0, 5.0],
                [6.0, 1.0, 4.0],
            ],
            dtype=torch.float64,
        )

    context = StepContext(step=4, seed=23, epoch_tag=2)
    branch_indices = [0, 2, 1, 2]
    return RolloutBatch(
        prompts=[f"algorithm-{index}" for index in range(batch_size)],
        metadata=[{"branch_step_index": index} for index in branch_indices],
        media=torch.zeros(batch_size, 1),
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).expand(batch_size, -1),
        old_log_probs=torch.zeros(batch_size, transitions, dtype=torch.float32),
        transition_mask=torch.tensor(
            [
                [True, True, True],
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        ),
        context=context,
        model_metadata=model_metadata,
        model_tensors=model_tensors,
    )


def _algorithm_for_case(case: str):
    if case == "grpo":
        return GRPOAlgorithm(clip_range=0.15)
    if case == "flash_reference":
        return FlashGRPOAlgorithm(
            objective_version="reference_v1",
            clip_range=0.15,
        )
    if case == "flash_custom":
        return FlashGRPOAlgorithm(
            objective_version="legacy_v0",
            clip_range=0.15,
            rectification={"enabled": True, "normalize": True},
        )
    if case == "tempflow_reference":
        return TempFlowGRPOAlgorithm(
            objective_version="reference_v1",
            clip_range=0.15,
            credit_assignment="all_after_branch",
            noise_weighting={
                "enabled": True,
                "mode": "reference_std_dev_t",
                "scale": 2.25,
            },
            preserve_advantage_dtype=True,
            advantage_dtype="float64",
        )
    if case == "tempflow_custom":
        return TempFlowGRPOAlgorithm(
            objective_version="legacy",
            clip_range=0.15,
            credit_assignment="all_after_branch",
            noise_weighting={
                "enabled": True,
                "mode": "std_dev_t",
                "normalize_custom": True,
            },
            preserve_advantage_dtype=True,
            advantage_dtype="float64",
        )
    raise AssertionError(f"Unknown test case: {case}")


def _run_algorithm_update(case: str, microbatch_size: int | None):
    batch = _algorithm_batch(case)
    rewards = RewardBatch(
        raw={"score": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        weighted={"score": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        weighted_total=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)
    adapter = _AlgorithmAdapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.03)
    metrics = UpdateEngine(
        _FixedAdvantage(advantages),
        AlgorithmPolicyObjective(_algorithm_for_case(case)),
        update_microbatch_size=microbatch_size,
    ).step(adapter, batch, rewards, optimizer, batch.context)
    return adapter.parameter.detach(), metrics


def test_grpo_masks_loss_metrics_gradients_and_reduction_weight() -> None:
    batch = _algorithm_batch("grpo")
    mask = batch.transition_mask
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)
    new_log_probs = torch.tensor(
        [
            [0.02, -0.03, 0.04],
            [0.05, 1.7, -1.4],
            [-0.06, 0.07, 1.2],
            [0.08, -0.09, 0.10],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    literal_new_log_probs = new_log_probs.detach().clone().requires_grad_(True)
    algorithm = GRPOAlgorithm(clip_range=0.15)

    loss, metrics = algorithm.compute_loss(batch, advantages, new_log_probs)

    old_log_probs = batch.old_log_probs.to(dtype=torch.float64)
    expanded_advantages = advantages[:, None].expand_as(literal_new_log_probs)
    ratio = torch.exp(literal_new_log_probs - old_log_probs)
    expected_loss = (
        torch.maximum(
            -expanded_advantages * ratio,
            -expanded_advantages * ratio.clamp(0.85, 1.15),
        )
        .masked_select(mask)
        .mean()
    )
    expected_kl = (
        0.5
        * (literal_new_log_probs - old_log_probs).square().masked_select(mask).mean()
    )
    expected_clipfrac = (
        ((ratio - 1.0).abs() > 0.15).to(new_log_probs.dtype).masked_select(mask).mean()
    )

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(metrics["approx_kl"], expected_kl)
    torch.testing.assert_close(metrics["clipfrac"], expected_clipfrac)
    assert algorithm.reduction_weight(batch, advantages) == int(mask.sum()) == 9

    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(new_log_probs.grad, literal_new_log_probs.grad)
    assert torch.count_nonzero(new_log_probs.grad.masked_select(~mask)).item() == 0


def test_logprob_metrics_ignore_masked_padding_values() -> None:
    batch = _algorithm_batch("grpo")
    mask = batch.transition_mask
    old_log_probs = torch.zeros_like(batch.old_log_probs)
    old_log_probs[~mask] = -50.0
    rollout_kl = torch.full_like(old_log_probs, 0.25)
    rollout_kl[~mask] = 100.0
    batch = batch.replace(old_log_probs=old_log_probs, kl=rollout_kl)
    new_log_probs = torch.full_like(old_log_probs, 0.1)
    new_log_probs[~mask] = 50.0

    metrics = UpdateEngine._logprob_metrics(batch, new_log_probs)

    assert metrics["old_logprob_mean"] == pytest.approx(0.0)
    assert metrics["new_logprob_mean"] == pytest.approx(0.1)
    assert metrics["logprob_delta_mean"] == pytest.approx(0.1)
    assert metrics["logprob_delta_abs_max"] == pytest.approx(0.1)
    assert metrics["rollout_kl_mean"] == pytest.approx(0.25)
    assert metrics["rollout_kl_abs_max"] == pytest.approx(0.25)


@pytest.mark.parametrize("beta", [-0.1, 0.1])
def test_grpo_nonzero_beta_fails_closed(beta: float) -> None:
    with pytest.raises(ValueError, match="requires beta=0"):
        GRPOAlgorithm(beta=beta)


@pytest.mark.parametrize("objective_version", ["legacy_v0", "reference_v1"])
@pytest.mark.parametrize("beta", [-0.1, 0.1])
def test_flash_nonzero_beta_fails_closed(
    objective_version: str,
    beta: float,
) -> None:
    with pytest.raises(ValueError, match="requires beta=0"):
        FlashGRPOAlgorithm(
            objective_version=objective_version,
            beta=beta,
        )


def test_flash_masks_loss_metrics_gradients_and_reduction_weight() -> None:
    batch = _algorithm_batch("flash_reference")
    mask = batch.transition_mask
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)
    new_log_probs = torch.tensor(
        [
            [0.02, -0.03, 0.04],
            [0.05, 1.7, -1.4],
            [-0.06, 0.07, 1.2],
            [0.08, -0.09, 0.10],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    literal_new_log_probs = new_log_probs.detach().clone().requires_grad_(True)
    algorithm = FlashGRPOAlgorithm(
        objective_version="reference_v1",
        clip_range=0.15,
    )

    loss, metrics = algorithm.compute_loss(batch, advantages, new_log_probs)

    old_log_probs = batch.old_log_probs.to(dtype=torch.float64)
    coefficient = batch.model_tensors["coefficient"].to(dtype=torch.float64)
    weights = coefficient / coefficient.mean()
    expanded_advantages = advantages[:, None].expand_as(literal_new_log_probs)
    weighted_advantages = expanded_advantages * weights
    ratio = torch.exp(literal_new_log_probs - old_log_probs)
    expected_loss = (
        torch.maximum(
            -weighted_advantages * ratio,
            -weighted_advantages * ratio.clamp(0.85, 1.15),
        )
        .masked_select(mask)
        .mean()
    )
    expected_kl = (
        0.5
        * (literal_new_log_probs - old_log_probs).square().masked_select(mask).mean()
    )
    expected_clipfrac = (
        ((ratio - 1.0).abs() > 0.15).to(new_log_probs.dtype).masked_select(mask).mean()
    )

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(metrics["approx_kl"], expected_kl)
    torch.testing.assert_close(metrics["clipfrac"], expected_clipfrac)
    assert algorithm.reduction_weight(batch, advantages) == int(mask.sum()) == 9

    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(new_log_probs.grad, literal_new_log_probs.grad)
    assert torch.count_nonzero(new_log_probs.grad.masked_select(~mask)).item() == 0


@pytest.mark.parametrize("case", ["grpo", "flash_reference"])
def test_masked_extreme_logratio_cannot_poison_gradients(case: str) -> None:
    batch = _algorithm_batch(case)
    mask = batch.transition_mask
    new_log_probs = torch.zeros_like(
        batch.old_log_probs,
        dtype=torch.float64,
    )
    new_log_probs[~mask] = 2_000.0
    new_log_probs.requires_grad_(True)
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)

    loss, metrics = _algorithm_for_case(case).compute_loss(
        batch,
        advantages,
        new_log_probs,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["approx_kl"])
    assert torch.isfinite(metrics["clipfrac"])
    assert torch.isfinite(new_log_probs.grad).all()
    assert torch.count_nonzero(new_log_probs.grad.masked_select(~mask)).item() == 0


@pytest.mark.parametrize(
    "case",
    [
        "flash_reference",
        "flash_custom",
        "tempflow_reference",
        "tempflow_custom",
    ],
)
def test_prepared_full_batch_constants_preserve_direct_loss_math(case: str) -> None:
    batch = _algorithm_batch(case)
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)
    new_log_probs = (
        torch.tensor(0.07, dtype=torch.float64, requires_grad=True)
        * batch.model_tensors["features"]
    )
    algorithm = _algorithm_for_case(case)
    direct_loss, direct_metrics = algorithm.compute_loss(
        batch,
        advantages,
        new_log_probs,
    )
    prepared = AlgorithmPolicyObjective(algorithm).prepare_batch(batch, advantages)
    prepared_loss, prepared_metrics = algorithm.compute_loss(
        prepared,
        advantages,
        new_log_probs,
    )

    assert prepared_loss.dtype == direct_loss.dtype == torch.float64
    torch.testing.assert_close(prepared_loss, direct_loss, rtol=1e-12, atol=1e-13)
    assert set(prepared_metrics) == set(direct_metrics)
    for name, direct_value in direct_metrics.items():
        torch.testing.assert_close(
            prepared_metrics[name],
            direct_value,
            rtol=1e-12,
            atol=1e-13,
        )


@pytest.mark.parametrize(
    "case",
    [
        "grpo",
        "flash_reference",
        "flash_custom",
        "tempflow_reference",
        "tempflow_custom",
    ],
)
def test_algorithm_full_and_microbatch_updates_are_equivalent(case: str) -> None:
    full_parameter, full_metrics = _run_algorithm_update(case, None)
    micro_parameter, micro_metrics = _run_algorithm_update(case, 2)

    torch.testing.assert_close(
        micro_parameter,
        full_parameter,
        rtol=1e-11,
        atol=1e-12,
    )
    ignored_metrics = {
        "update_microbatches",
        "recompute_time_s",
        "backward_time_s",
        "optimizer_time_s",
    }
    assert set(micro_metrics) == set(full_metrics)
    for name in sorted(set(full_metrics) - ignored_metrics):
        assert micro_metrics[name] == pytest.approx(
            full_metrics[name],
            rel=1e-6,
            abs=1e-8,
        ), name


def test_flash_global_coefficient_mean_is_prepared_once_before_microbatching() -> None:
    batch = _algorithm_batch("flash_reference")
    rewards = RewardBatch(
        raw={"score": torch.ones(batch.batch_size)},
        weighted={"score": torch.ones(batch.batch_size)},
        weighted_total=torch.ones(batch.batch_size),
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)
    adapter = _AlgorithmAdapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.03)
    reductions: list[tuple[torch.Tensor, int]] = []
    synchronizations: list[BaseException | bool | None] = []

    def reduce_once(local_mean: torch.Tensor, count: int) -> torch.Tensor:
        reductions.append((local_mean.detach().clone(), count))
        return local_mean.new_tensor(8.0)

    def synchronize(failure: BaseException | bool | None) -> bool:
        synchronizations.append(failure)
        if isinstance(failure, BaseException):
            raise failure
        return bool(failure)

    metrics = UpdateEngine(
        _FixedAdvantage(advantages),
        AlgorithmPolicyObjective(_algorithm_for_case("flash_reference")),
        update_microbatch_size=1,
    ).step(
        adapter,
        batch,
        rewards,
        optimizer,
        batch.context,
        reduce_tensor_weighted_mean=reduce_once,
        synchronize_failure=synchronize,
    )

    assert len(reductions) == 1
    local_mean, count = reductions[0]
    assert local_mean.dtype == torch.float64
    torch.testing.assert_close(local_mean, torch.tensor(4.75, dtype=torch.float64))
    assert count == batch.batch_size
    assert synchronizations == [None, None]
    assert metrics["flash_rectification_weight_mean"] == pytest.approx(0.5)
