"""CPU-only contract tests for TempFlow legacy and reference loss semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.configs.schema import load_config
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.sd3 import (
    SD3_TRANSITION_CONTRACT_VERSION,
    SD3TempFlowAdapter,
)
from visual_rl.optimizers.factory import build_algorithm
from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm


ROOT = Path(__file__).resolve().parents[1]


class _NonUniformScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor(
            [1000.0, 700.0, 250.0, 10.0],
            dtype=torch.float64,
        )
        self.sigmas = torch.tensor(
            [1.0, 0.78, 0.36, 0.08, 0.0],
            dtype=torch.float64,
        )

    def index_for_timestep(self, timestep) -> int:
        value = float(torch.as_tensor(timestep).reshape(-1)[0])
        matches = torch.nonzero(self.timesteps == value).reshape(-1)
        if matches.numel() != 1:
            raise ValueError(f"unknown timestep {value}")
        return int(matches[0])


def _reference_adapter() -> SD3TempFlowAdapter:
    adapter = SD3TempFlowAdapter(
        {
            "model_path": "",
            "extra": {
                "defer_load": True,
                "tempflow_reference_mode": True,
            },
        }
    )
    adapter.pipeline = SimpleNamespace(scheduler=_NonUniformScheduler())
    return adapter


def _batch(
    old_log_probs: torch.Tensor,
    *,
    branch_indices: list[int] | None = None,
    model_metadata: dict | None = None,
    kl: torch.Tensor | None = None,
) -> RolloutBatch:
    batch_size, steps = old_log_probs.shape
    branch_indices = branch_indices or [0] * batch_size
    return RolloutBatch(
        prompts=[f"prompt-{index}" for index in range(batch_size)],
        metadata=[
            {"branch_step_index": branch_index} for branch_index in branch_indices
        ],
        media=torch.zeros(batch_size, 1),
        latents=torch.zeros(batch_size, steps, 1),
        next_latents=torch.zeros(batch_size, steps, 1),
        timesteps=torch.zeros(batch_size, steps),
        old_log_probs=old_log_probs,
        kl=kl,
        model_metadata=dict(model_metadata or {}),
    )


def _reference_metadata(**overrides) -> dict:
    metadata = {
        "tempflow_reference_mode": True,
        "trajectory_contract_version": SD3_TRANSITION_CONTRACT_VERSION,
        "recompute_transformer_training": True,
    }
    metadata.update(overrides)
    return metadata


def _reference_algorithm(**overrides) -> TempFlowGRPOAlgorithm:
    options = {
        "objective_version": "reference_v1",
        "beta": 0.0,
        "noise_weighting": {
            "enabled": True,
            "mode": "reference_std_dev_t",
            "scale": 2.25,
        },
        "advantage_dtype": "float64",
        "preserve_advantage_dtype": True,
    }
    options.update(overrides)
    return TempFlowGRPOAlgorithm(**options)


def _policy_identity_metadata(**overrides) -> dict:
    metadata = {
        "tempflow_reference_mode": False,
        "trajectory_contract_version": SD3_TRANSITION_CONTRACT_VERSION,
        "recompute_transformer_training": False,
        "branching_mode": "shared_prefix",
    }
    metadata.update(overrides)
    return metadata


def _policy_identity_algorithm(**overrides) -> TempFlowGRPOAlgorithm:
    options = {
        "objective_version": "policy_identity_v1",
        "beta": 0.0,
        "noise_weighting": {
            "enabled": True,
            "mode": "reference_std_dev_t",
            "scale": 2.25,
        },
        "advantage_dtype": "float64",
        "preserve_advantage_dtype": True,
    }
    options.update(overrides)
    return TempFlowGRPOAlgorithm(**options)


def test_reference_transition_std_matches_nonuniform_sigma_oracle() -> None:
    adapter = _reference_adapter()
    scheduler = adapter.pipeline.scheduler

    step_zero = adapter._reference_transition_std_dev_t(
        scheduler.timesteps[0],
        noise_level=0.7,
    )
    middle = adapter._reference_transition_std_dev_t(
        scheduler.timesteps[1],
        noise_level=0.7,
    )

    sigma = scheduler.sigmas[1]
    sigma_next = scheduler.sigmas[2]
    base_std = 0.7 * torch.sqrt(sigma / (1.0 - sigma))
    expected_middle = base_std * torch.sqrt(sigma - sigma_next)
    torch.testing.assert_close(step_zero, torch.tensor(0.7, dtype=torch.float64))
    torch.testing.assert_close(middle, expected_middle)
    assert not torch.isclose(middle, base_std)


def test_reference_transition_rejects_invalid_step_and_noise_level() -> None:
    adapter = _reference_adapter()
    scheduler = adapter.pipeline.scheduler

    with pytest.raises(ValueError, match="non-terminal"):
        adapter._reference_transition_std_dev_t(
            scheduler.timesteps[-1],
            noise_level=0.7,
        )
    with pytest.raises(ValueError, match="absent"):
        adapter._reference_transition_std_dev_t(123.0, noise_level=0.7)
    with pytest.raises(ValueError, match="pinned"):
        adapter._reference_transition_std_dev_t(
            scheduler.timesteps[1],
            noise_level=0.9,
        )


def test_reference_contract_rejects_old_trajectory_metadata() -> None:
    adapter = _reference_adapter()
    adapter.transformer = SimpleNamespace(training=True)
    batch = _batch(
        torch.zeros(1, 1),
        model_metadata={"trajectory_contract_version": "sd3_tempflow_v2"},
    )

    assert SD3_TRANSITION_CONTRACT_VERSION == "sd3_tempflow_v3"
    with pytest.raises(ValueError, match="requires trajectory contract"):
        adapter._validate_recompute_contract(
            batch,
            batch.latents,
            batch.next_latents,
        )


@pytest.mark.parametrize(
    ("credit_assignment", "expected_mask"),
    [
        (
            "branch_timestep",
            [[False, True, False, False], [False, False, True, False]],
        ),
        (
            "all_after_branch",
            [[False, True, True, True], [False, False, True, True]],
        ),
    ],
)
def test_credit_assignment_preserves_float64_advantages(
    credit_assignment: str,
    expected_mask: list[list[bool]],
) -> None:
    algorithm = TempFlowGRPOAlgorithm(
        credit_assignment=credit_assignment,
        noise_weighting={"enabled": False},
        preserve_advantage_dtype=True,
    )
    new_log_probs = torch.zeros(2, 4, dtype=torch.float32)
    batch = _batch(
        torch.zeros_like(new_log_probs),
        branch_indices=[1, 2],
    )
    advantages = torch.tensor([1.25, -0.5], dtype=torch.float64)

    expanded, active_mask = algorithm._expand_advantages_and_mask(
        batch,
        advantages,
        new_log_probs,
    )

    assert expanded.dtype == torch.float64
    assert active_mask.tolist() == expected_mask


def test_legacy_branch_timestep_matches_literal_full_mean_and_gradient() -> None:
    algorithm = TempFlowGRPOAlgorithm(
        objective_version="legacy",
        clip_range=0.2,
        beta=0.3,
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": True, "mode": "std_dev_t"},
    )
    old_log_probs = torch.zeros(2, 4, dtype=torch.float64)
    noise_weights = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]],
        dtype=torch.float64,
    )
    kl = torch.tensor(
        [[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 2.0, 20.0]],
        dtype=torch.float64,
    )
    batch = _batch(
        old_log_probs,
        branch_indices=[1, 2],
        model_metadata={"noise_weights": noise_weights},
        kl=kl,
    )
    rewards = torch.tensor([1.25, -0.5], dtype=torch.float64)
    actual_new_log_probs = torch.tensor(
        [[0.0, 0.1, 0.5, -0.5], [0.3, -0.3, -0.1, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    literal_new_log_probs = actual_new_log_probs.detach().clone().requires_grad_(True)

    actual_loss, actual_info = algorithm.compute_loss(
        batch,
        rewards,
        actual_new_log_probs,
    )

    literal_advantages = torch.zeros_like(literal_new_log_probs)
    literal_advantages[0, 1] = rewards[0]
    literal_advantages[1, 2] = rewards[1]
    literal_advantages = literal_advantages.clamp(
        -algorithm.adv_clip_max,
        algorithm.adv_clip_max,
    )
    literal_advantages = literal_advantages * noise_weights
    literal_ratio = torch.exp(literal_new_log_probs - old_log_probs)
    literal_unclipped = -literal_advantages * literal_ratio
    literal_clipped = -literal_advantages * literal_ratio.clamp(
        1.0 - algorithm.clip_range,
        1.0 + algorithm.clip_range,
    )
    literal_loss = torch.maximum(literal_unclipped, literal_clipped).mean()
    literal_loss = literal_loss + algorithm.beta * kl.mean()
    literal_approx_kl = 0.5 * ((literal_new_log_probs - old_log_probs) ** 2).mean()
    literal_clipfrac = (
        ((literal_ratio - 1.0).abs() > algorithm.clip_range).float().mean()
    )

    actual_loss.backward()
    literal_loss.backward()

    torch.testing.assert_close(actual_loss, literal_loss)
    torch.testing.assert_close(actual_new_log_probs.grad, literal_new_log_probs.grad)
    torch.testing.assert_close(actual_info["policy_loss"], literal_loss.detach())
    torch.testing.assert_close(actual_info["approx_kl"], literal_approx_kl)
    torch.testing.assert_close(actual_info["clipfrac"], literal_clipfrac)
    torch.testing.assert_close(
        actual_info["tempflow_noise_weight_mean"],
        noise_weights.mean(),
    )
    assert actual_info["tempflow_active_timestep_frac"].item() == 0.25


def test_reference_v1_compact_and_embedded_loss_gradients_match() -> None:
    algorithm = _reference_algorithm(
        clip_range=0.2,
        credit_assignment="branch_timestep",
    )
    advantages = torch.tensor([1.25, 0.0], dtype=torch.float64)
    compact_old = torch.tensor([[0.02], [-0.03]], dtype=torch.float64)
    embedded_old = torch.tensor(
        [[0.4, -0.2, 0.02, 0.1], [-0.3, 0.5, -0.03, -0.4]],
        dtype=torch.float64,
    )
    compact_batch = _batch(
        compact_old,
        branch_indices=[2, 2],
        model_metadata=_reference_metadata(
            trajectory_step_indices=[2],
            transition_std_dev_t=[0.7],
        ),
    )
    embedded_batch = _batch(
        embedded_old,
        branch_indices=[2, 2],
        model_metadata=_reference_metadata(
            transition_std_dev_t=[0.2, 1.3, 0.7, 1.8],
        ),
    )

    compact_parameter = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    embedded_parameter = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    compact_new = compact_parameter * torch.tensor(
        [[0.5], [-0.25]],
        dtype=torch.float64,
    )
    embedded_new = embedded_parameter * torch.tensor(
        [[1.5, -2.0, 0.5, 0.7], [-1.0, 1.25, -0.25, 2.0]],
        dtype=torch.float64,
    )

    compact_loss, compact_info = algorithm.compute_loss(
        compact_batch,
        advantages,
        compact_new,
    )
    embedded_loss, embedded_info = algorithm.compute_loss(
        embedded_batch,
        advantages,
        embedded_new,
    )
    compact_loss.backward()
    embedded_loss.backward()

    torch.testing.assert_close(compact_loss, embedded_loss)
    torch.testing.assert_close(compact_parameter.grad, embedded_parameter.grad)
    torch.testing.assert_close(
        compact_info["approx_kl"],
        embedded_info["approx_kl"],
    )
    torch.testing.assert_close(compact_info["clipfrac"], embedded_info["clipfrac"])
    torch.testing.assert_close(
        compact_info["tempflow_noise_weight_mean"],
        embedded_info["tempflow_noise_weight_mean"],
    )
    assert compact_info["tempflow_active_timestep_frac"].item() == 1.0
    assert embedded_info["tempflow_active_timestep_frac"].item() == 0.25


@pytest.mark.parametrize("failure_mode", ["missing", "wrong"])
@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("tempflow_reference_mode", False),
        ("trajectory_contract_version", "sd3_tempflow_v2"),
        ("recompute_transformer_training", False),
    ],
)
def test_reference_v1_compute_loss_rejects_invalid_batch_provenance(
    field: str,
    wrong_value,
    failure_mode: str,
) -> None:
    metadata = _reference_metadata(transition_std_dev_t=[0.7])
    if failure_mode == "missing":
        metadata.pop(field)
    else:
        metadata[field] = wrong_value
    batch = _batch(torch.zeros(1, 1), model_metadata=metadata)

    with pytest.raises(ValueError, match=field):
        _reference_algorithm().compute_loss(
            batch,
            torch.tensor([1.0]),
            torch.zeros(1, 1),
        )


@pytest.mark.parametrize(
    ("std_dev_t", "message"),
    [
        ([0.7], "shape"),
        ([0.7, float("nan")], "finite"),
        ([0.7, 0.0], "positive"),
    ],
)
def test_reference_std_dev_t_validation(std_dev_t, message: str) -> None:
    algorithm = _reference_algorithm()
    new_log_probs = torch.zeros(2, 2)
    batch = _batch(
        torch.zeros_like(new_log_probs),
        model_metadata={"transition_std_dev_t": std_dev_t},
    )

    with pytest.raises(ValueError, match=message):
        algorithm._noise_weights(batch, new_log_probs)


def test_objective_versions_fail_closed() -> None:
    with pytest.raises(ValueError, match="objective_version"):
        TempFlowGRPOAlgorithm(objective_version="future_v2")
    with pytest.raises(ValueError, match="beta > 0"):
        _reference_algorithm(beta=0.1)


@pytest.mark.parametrize("advantage_dtype", ["float32", "float16", "bfloat16"])
def test_reference_v1_requires_float64_config(advantage_dtype: str) -> None:
    with pytest.raises(ValueError, match="advantage_dtype='float64'"):
        _reference_algorithm(advantage_dtype=advantage_dtype)


@pytest.mark.parametrize("preserve_advantage_dtype", [False, "true", 1])
def test_reference_v1_requires_native_true_preserve_flag(
    preserve_advantage_dtype,
) -> None:
    with pytest.raises(ValueError, match="preserve_advantage_dtype=True"):
        _reference_algorithm(preserve_advantage_dtype=preserve_advantage_dtype)


@pytest.mark.parametrize(
    "advantages",
    [
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float16),
        torch.tensor([1.0], dtype=torch.bfloat16),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([True], dtype=torch.bool),
    ],
    ids=["float32", "float16", "bfloat16", "int64", "bool"],
)
def test_reference_v1_compute_loss_rejects_non_float64_advantages(
    advantages: torch.Tensor,
) -> None:
    new_log_probs = torch.zeros(1, 1, dtype=torch.float64)
    batch = _batch(
        torch.zeros_like(new_log_probs),
        model_metadata=_reference_metadata(transition_std_dev_t=[0.7]),
    )

    with pytest.raises(TypeError, match="advantages.*dtype=torch.float64"):
        _reference_algorithm().compute_loss(batch, advantages, new_log_probs)


def test_reference_v1_compute_loss_accepts_float64_advantages() -> None:
    new_log_probs = torch.zeros(1, 1, dtype=torch.float64)
    batch = _batch(
        torch.zeros_like(new_log_probs),
        model_metadata=_reference_metadata(transition_std_dev_t=[0.7]),
    )

    loss, _info = _reference_algorithm().compute_loss(
        batch,
        torch.tensor([1.0], dtype=torch.float64),
        new_log_probs,
    )

    assert loss.dtype == torch.float64


def test_policy_identity_v1_is_strict_and_separate_from_reference_mode() -> None:
    new_log_probs = torch.zeros(1, 1, dtype=torch.float64)
    batch = _batch(
        torch.zeros_like(new_log_probs),
        model_metadata=_policy_identity_metadata(transition_std_dev_t=[0.7]),
    )

    loss, _info = _policy_identity_algorithm().compute_loss(
        batch,
        torch.tensor([1.0], dtype=torch.float64),
        new_log_probs,
    )
    assert loss.dtype == torch.float64

    incompatible = batch.replace(
        model_metadata={
            **batch.model_metadata,
            "tempflow_reference_mode": True,
        }
    )
    with pytest.raises(ValueError, match="tempflow_reference_mode"):
        _policy_identity_algorithm().compute_loss(
            incompatible,
            torch.tensor([1.0], dtype=torch.float64),
            new_log_probs,
        )


def test_mainline_tempflow_preset_defaults_to_policy_identity() -> None:
    config = load_config(ROOT / "visual_rl/configs/presets/sd3_tempflow_adapter.yaml")
    algorithm = build_algorithm(config.algorithm)

    assert config.model.extra["tempflow_reference_mode"] is False
    assert config.algorithm.objective_version == "policy_identity_v1"
    assert algorithm.objective_version == "policy_identity_v1"


def test_remote_tempflow_smoke_defaults_strict_and_bounds_reference_opt_in() -> None:
    from scripts.remote_smoke import (
        RemoteSd3CliSmokeConfig,
        _bounded_trainer_cli_args,
    )

    default_args = _bounded_trainer_cli_args(RemoteSd3CliSmokeConfig())
    mode_index = default_args.index("--tempflow-execution-mode")
    assert default_args[mode_index + 1] == "policy-identity"
    assert "--allow-initial-clipping" not in default_args

    with pytest.raises(ValueError, match="only valid"):
        _bounded_trainer_cli_args(
            RemoteSd3CliSmokeConfig(allow_initial_clipping=True)
        )
    reference_args = _bounded_trainer_cli_args(
        RemoteSd3CliSmokeConfig(
            tempflow_execution_mode="reference-compatible",
            allow_initial_clipping=True,
        )
    )
    assert "--allow-initial-clipping" in reference_args


def test_reference_preset_parses_objective_version() -> None:
    config = load_config(
        ROOT / "visual_rl/configs/presets/sd3_tempflow_reference_v1.yaml"
    )
    algorithm = TempFlowGRPOAlgorithm.from_config(config.algorithm)

    assert config.run_name == "sd3_tempflow_reference_v1"
    assert config.algorithm.objective_version == "reference_v1"
    assert config.algorithm.noise_weighting["mode"] == "reference_std_dev_t"
    assert config.algorithm.params["preserve_advantage_dtype"] is True
    assert algorithm.objective_version == "reference_v1"
    assert algorithm.advantage_dtype == "float64"
    assert algorithm.preserve_advantage_dtype is True


def test_build_algorithm_rejects_reference_v1_with_float32_advantages() -> None:
    config = load_config(
        ROOT / "visual_rl/configs/presets/sd3_tempflow_reference_v1.yaml"
    )
    config.algorithm.advantage_dtype = "float32"

    with pytest.raises(ValueError, match="advantage_dtype='float64'"):
        build_algorithm(config.algorithm)


def test_build_algorithm_accepts_reference_v1_preset() -> None:
    config = load_config(
        ROOT / "visual_rl/configs/presets/sd3_tempflow_reference_v1.yaml"
    )

    algorithm = build_algorithm(config.algorithm)

    assert algorithm.objective_version == "reference_v1"
    assert algorithm.advantage_dtype == "float64"
    assert algorithm.preserve_advantage_dtype is True
