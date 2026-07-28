"""Executable v0.6 loss characterization for the W02 -> W04 cutover.

The JSON fixture was captured at the frozen v0.6 baseline.  These tests rebuild
the probe from public VisualRL classes, so the fixture is an enforced regression
oracle rather than an unaudited artifact or a dependency on a temporary script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from visual_rl.core.seed import seed_everything
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "characterization" / "v0_6_loss_probe.json"
)


@pytest.fixture(scope="module")
def characterization() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    assert fixture["schema"] == "visual_rl.characterization/v0_6_loss_probe.v1"
    return fixture


def _adapter(config: dict) -> TinyDiffusionAdapter:
    return TinyDiffusionAdapter(
        {
            "name": "tiny_diffusion",
            "extra": {
                "image_size": config["image_size"],
                "device": config["device"],
            },
        }
    )


def _probe_batch(fixture: dict):
    config = fixture["config"]
    seed_everything(config["seed"])
    teacher = _adapter(config)
    target_bias = torch.tensor(
        config["target_bias"],
        device=teacher.device,
        dtype=teacher.color_bias.dtype,
    )
    with torch.no_grad():
        teacher.color_bias.copy_(target_bias)
    prompts = ["a red square" for _ in range(config["batch_size"])]
    metadata = [
        {"source": "tiny_loss_probe", "target_color": "red"}
        for _ in range(config["batch_size"])
    ]
    batch = teacher.sample(
        prompts,
        metadata,
        {"num_steps": config["num_steps"], "seed": config["seed"]},
    )
    mask = torch.tensor(
        fixture["mask"]["transition_mask"],
        dtype=torch.bool,
        device=teacher.device,
    )
    batch = batch.replace(transition_mask=mask)
    batch.validate_strict()
    return batch, target_bias


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _assert_tensor_record(actual: torch.Tensor, expected: dict) -> None:
    assert list(actual.shape) == expected["shape"]
    assert str(actual.dtype) == expected["dtype"]
    torch.testing.assert_close(
        actual.detach().cpu().reshape(-1),
        torch.tensor(expected["values"], dtype=actual.dtype),
        rtol=1e-6,
        atol=1e-7,
    )


def test_v0_6_probe_batch_and_supervised_update_match_fixture(characterization):
    fixture = characterization
    config = fixture["config"]
    batch, target_bias = _probe_batch(fixture)
    recorded_batch = fixture["batch"]

    _assert_tensor_record(batch.old_log_probs, recorded_batch["old_log_probs"])
    _assert_tensor_record(batch.timesteps, recorded_batch["timesteps"])
    _assert_tensor_record(batch.kl, recorded_batch["kl"])
    _assert_tensor_record(target_bias, recorded_batch["teacher_target_bias"])
    assert _tensor_digest(batch.latents) == recorded_batch["latents_sha256"]
    assert (
        _tensor_digest(batch.next_latents)
        == recorded_batch["next_latents_sha256"]
    )
    assert _tensor_digest(batch.media) == recorded_batch["media_sha256"]
    assert int(batch.transition_mask.sum()) == fixture["mask"][
        "active_transition_count"
    ]

    student = _adapter(config)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
    }
    new_log_probs = student.recompute_log_probs(batch)
    _assert_tensor_record(
        new_log_probs, recorded_batch["student_new_log_probs_at_init"]
    )
    old_log_probs = batch.old_log_probs.to(
        new_log_probs.device, dtype=new_log_probs.dtype
    )
    fit_loss = 0.5 * ((new_log_probs - old_log_probs) ** 2).mean()
    transition_nll = (-new_log_probs).mean()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(config["learning_rate"])
    )
    optimizer.zero_grad(set_to_none=True)
    fit_loss.backward()

    recorded_fit = fixture["supervised_fit"]
    assert float(fit_loss.detach()) == pytest.approx(recorded_fit["fit_loss"])
    assert float(transition_nll.detach()) == pytest.approx(
        recorded_fit["transition_nll"]
    )
    for name, parameter in student.named_parameters():
        _assert_tensor_record(parameter.grad, recorded_fit["gradients"][name])
    optimizer.step()
    for name, parameter in student.named_parameters():
        _assert_tensor_record(
            parameter.detach() - initial[name],
            recorded_fit["parameter_delta"][name],
        )


def _algorithm(name: str):
    if name == "grpo":
        return GRPOAlgorithm(clip_range=10.0, adv_clip_max=5.0, beta=0.0)
    if name == "flash_grpo":
        return FlashGRPOAlgorithm(
            objective_version="legacy_v0",
            clip_range=10.0,
            adv_clip_max=5.0,
            beta=0.0,
            rectification=None,
        )
    if name == "tempflow_grpo":
        return TempFlowGRPOAlgorithm(
            objective_version="legacy",
            clip_range=10.0,
            adv_clip_max=5.0,
            beta=0.0,
            credit_assignment="branch_timestep",
            noise_weighting=None,
            preserve_advantage_dtype=False,
            advantage_dtype="float32",
        )
    raise AssertionError(f"unknown characterization algorithm {name!r}")


@pytest.mark.parametrize("algorithm_name", ["grpo", "flash_grpo", "tempflow_grpo"])
def test_v0_6_policy_updates_match_fixture(characterization, algorithm_name):
    fixture = characterization
    config = fixture["config"]
    batch, _target_bias = _probe_batch(fixture)
    student = _adapter(config)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
    }
    advantages = torch.ones_like(batch.old_log_probs, device=student.device)
    algorithm = _algorithm(algorithm_name)
    new_log_probs = student.recompute_log_probs(batch)
    loss, info = algorithm.compute_loss(batch, advantages, new_log_probs)
    reduction_weight = algorithm.reduction_weight(batch, advantages)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(config["learning_rate"])
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    expected = fixture["algorithms"][algorithm_name]
    assert float(loss.detach()) == pytest.approx(expected["loss"])
    assert int(reduction_weight) == expected["active_transition_count"]
    assert int((~batch.transition_mask).sum()) == expected[
        "invalid_transition_count"
    ]
    for metric_name in ("approx_kl", "clipfrac"):
        assert float(info[metric_name].detach()) == pytest.approx(
            expected[metric_name]
        )
    for metric_name, expected_value in expected["diagnostics"].items():
        if isinstance(expected_value, (int, float)):
            assert float(info[metric_name].detach()) == pytest.approx(expected_value)
    for name, parameter in student.named_parameters():
        _assert_tensor_record(parameter.grad, expected["gradients"][name])

    optimizer.step()
    for name, parameter in student.named_parameters():
        _assert_tensor_record(
            parameter.detach() - initial[name],
            expected["parameter_delta"][name],
        )
        _assert_tensor_record(parameter.detach(), expected["updated_parameters"][name])
