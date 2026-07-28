"""Tiny CPU causal-chain proof for MinWM reward-driven LoRA updates."""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from visual_rl import ExperimentRunner
from visual_rl.configs.schema import config_from_dict, config_to_dict
from visual_rl.core.types import RewardBatch, StepContext
from visual_rl.model_adapters.minwm_wan import MinWMWanAdapter
from visual_rl.model_adapters.minwm_wan_native_backend import (
    MinWMWanNativeBackend,
    _promote_trainable_parameters_to_fp32,
)
import visual_rl.model_adapters.minwm_wan_native_backend as minwm_native
from visual_rl.optimizers.advantages import AdvantageComputer
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.preflight import static_preflight
from visual_rl.rollout.full_trajectory import FullTrajectoryRollout


_TIMESTEPS = (997.0, 613.0, 211.0, 7.0)
_SIGMAS = (0.92, 0.61, 0.27, 0.09)


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_base = torch.nn.Parameter(
            torch.tensor(0.11),
            requires_grad=False,
        )
        self.lora_gain = torch.nn.Parameter(torch.tensor(0.17))


class _TrainingBackend:
    checkpoint_identity = "training-fake-checkpoint"
    minwm_commit = "training-fake-commit"
    scheduler_identity = "training-fake-shifted-scheduler"
    causal_state_contract = "temporal_committed_prefix_pure_conditioning_memoized_v2"
    temporal_committed_prefix_pure = True
    conditioning_memoization_contract = (
        "tensor_identity_bound_crossattn_immutable_after_init_v1"
    )
    sampling_contract = "native_bfchw_model_dtype_flowmatch_add_noise_v2"
    media_layout = "BFCHW"
    lora_identity = {
        "target_modules": ["tiny_policy.lora_gain"],
        "rank": 1,
        "alpha": 1.0,
    }
    geometry_identity = {
        "chunk_size": 2,
        "latent_layout": "BCFHW",
        "latent_frames": 2,
        "camera_conditioning_frames": 2,
        "decoded_media_frames": 5,
        "vae_temporal_stride": 4,
        "vae_temporal_downsample": [True, True, False],
        "decoded_media_frame_formula": (
            "1 + vae_temporal_stride * (latent_frames - 1)"
        ),
    }
    implementation_revision = "runner-test-v1"

    def __init__(self):
        self.train_module = _TinyPolicy()

    def implementation_identity(self):
        return {"revision": self.implementation_revision}

    def resolve_camera_control(self, camera_control, *, expected_frames):
        if expected_frames != 2:
            raise ValueError("tiny camera control requires two frames")
        trajectory = camera_control.get("trajectory")
        if trajectory not in {"right", "left"}:
            raise ValueError("unknown tiny camera trajectory")
        translation = 0.25 if trajectory == "right" else -0.25
        return _camera(translation=translation)

    def resolve_schedule(self, nominal_timesteps, *, stage, warp):
        assert stage == "dmd"
        assert warp is True
        return [
            {
                "nominal_timestep": nominal,
                "timestep": timestep,
                "sigma": sigma,
            }
            for nominal, timestep, sigma in zip(
                nominal_timesteps,
                _TIMESTEPS,
                _SIGMAS,
                strict=True,
            )
        ]

    def encode_condition(self, prompts):
        return torch.tensor([[len(prompt) / 50.0] for prompt in prompts])

    def reset_causal_state(self, *, batch_size, total_frames, dtype, device):
        return {
            "history": torch.zeros(batch_size, 1, 1, 1, 1, device=device),
            "initial_noise": None,
            "total_frames": total_frames,
            "dtype": dtype,
        }

    def sample_initial_latent(
        self,
        state,
        *,
        batch_size,
        chunk_id,
        frame_start,
        chunk_size,
        condition,
        viewmats,
        Ks,
        generator,
    ):
        del chunk_id, condition, viewmats, Ks
        if state["initial_noise"] is None:
            state["initial_noise"] = torch.randn(
                batch_size,
                state["total_frames"],
                1,
                2,
                2,
                dtype=state["dtype"],
                device=state["history"].device,
                generator=generator,
            )
        native = state["initial_noise"][:, frame_start : frame_start + chunk_size]
        return native.permute(0, 2, 1, 3, 4).contiguous()

    def sample_x0_renoise(self, x0_pred, *, sigma_next, generator):
        native_x0 = x0_pred.detach().permute(0, 2, 1, 3, 4).contiguous()
        noise = torch.randn(
            native_x0.shape,
            dtype=native_x0.dtype,
            device=native_x0.device,
            generator=generator,
        )
        sigma = torch.as_tensor(
            sigma_next,
            dtype=torch.float32,
            device=native_x0.device,
        ).reshape(1, 1, 1, 1, 1)
        observed = ((1.0 - sigma) * native_x0 + sigma * noise).type_as(noise)
        return observed.permute(0, 2, 1, 3, 4).contiguous()

    def predict_flow_x0(
        self,
        state,
        x_t,
        condition,
        viewmats,
        Ks,
        *,
        timestep,
        frame_start,
        chunk_id,
        step_id,
    ):
        del Ks, frame_start, chunk_id
        condition_term = condition.reshape(-1, 1, 1, 1, 1)
        camera_term = viewmats[:, :, 0, 3].mean(dim=1).reshape(-1, 1, 1, 1, 1)
        step_term = float(step_id + 1) * 0.03
        feature = (
            x_t
            + 0.2 * condition_term
            + 0.05 * state["history"]
            + 0.02 * camera_term
            + step_term
        )
        flow = self.train_module.frozen_base + self.train_module.lora_gain * feature
        lookup = dict(zip(_TIMESTEPS, _SIGMAS, strict=True))
        sigma = torch.tensor(
            [lookup[float(value)] for value in timestep],
            dtype=torch.float32,
            device=x_t.device,
        ).reshape(-1, 1, 1, 1, 1)
        return flow, x_t.float() - sigma * flow.float()

    def commit_clean_context(
        self,
        state,
        final_clean,
        condition,
        viewmats,
        Ks,
        *,
        frame_start,
        context_timestep,
    ):
        del condition, viewmats, Ks, frame_start, context_timestep
        state["history"] = final_clean.detach().mean(
            dim=tuple(range(1, final_clean.ndim)),
            keepdim=True,
        )

    def decode(self, final_clean_chunks, *, condition, viewmats, Ks):
        del condition, viewmats, Ks
        batch, chunks, channels, frames, height, width = final_clean_chunks.shape
        latent_media = (
            final_clean_chunks.permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, chunks * frames, channels, height, width)
            .sigmoid()
        )
        return torch.cat(
            (latent_media[:, :1], latent_media[:, 1:].repeat_interleave(4, dim=1)),
            dim=1,
        )


def build_minwm_runner_test_backend(_config):
    """Importable factory used by the JSON-safe ExperimentRunner config test."""

    return _TrainingBackend()


def _camera(*, translation: float = 0.25):
    viewmats = torch.eye(4).repeat(2, 1, 1)
    viewmats[1, 0, 3] = translation
    return {
        "convention": "w2c",
        "coordinate_system": "opencv",
        "viewmats": viewmats.tolist(),
        "Ks": torch.eye(3).repeat(2, 1, 1).tolist(),
    }


def _adapter():
    return MinWMWanAdapter(
        {
            "backend": _TrainingBackend(),
            "stage": "dmd",
            "transition_kernel": "x0_renoise",
            "replay_credit_assignment": "sample_one",
            "use_lora": True,
            "denoising_timesteps": [1000, 750, 500, 250],
            "num_chunks": 1,
            "chunk_size": 2,
        }
    )


def _rollout(adapter, *, trajectory="right"):
    engine = FullTrajectoryRollout(
        {
            "samples_per_prompt": 4,
            "num_steps": 4,
            "num_chunks": 1,
            "chunk_size": 2,
            "denoising_timesteps": [1000, 750, 500, 250],
            "seed": 321,
            "camera_control": {"trajectory": trajectory},
        }
    )
    metadata = [
        {
            "prompt_id": "tiny-prompt",
            "group_id": "tiny-camera-control",
        }
    ]
    return engine.sample(
        adapter,
        ["orbit around a red block"],
        metadata,
        StepContext(step=0, seed=321, epoch_tag=0),
    )


def test_minwm_global_camera_control_is_bound_before_branch_sampling():
    right = _rollout(_adapter(), trajectory="right")
    left = _rollout(_adapter(), trajectory="left")

    assert len(set(right.group_id)) == 1
    assert len({item["camera_control_sha256"] for item in right.metadata}) == 1
    assert all(
        item["camera_trajectory"] == right.metadata[0]["camera_trajectory"]
        for item in right.metadata
    )
    assert right.metadata[0]["minwm_reward_frame_alignment"] == {
        "contract": "minwm_vae_camera_alignment_v1",
        "latent_frames": 2,
        "decoded_media_frames": 5,
        "vae_temporal_stride": 4,
    }
    assert (
        right.model_metadata["conditioning_identity_sha256"]
        != left.model_metadata["conditioning_identity_sha256"]
    )


def test_minwm_defaults_factory_to_streaming_dmd_sample_one():
    captured = {}

    def factory(config):
        captured.update(config)
        return _TrainingBackend()

    adapter = MinWMWanAdapter(
        {
            "backend_factory": factory,
            "use_lora": True,
            "denoising_timesteps": [1000, 750, 500, 250],
            "num_chunks": 1,
            "chunk_size": 2,
        }
    )
    assert adapter.train_module is not None
    assert captured["replay_credit_assignment"] == "sample_one"

    with pytest.raises(ValueError, match="replay_credit_assignment='sample_one'"):
        MinWMWanAdapter(
            {
                "backend": _TrainingBackend(),
                "stage": "dmd",
                "transition_kernel": "x0_renoise",
                "replay_credit_assignment": "all",
            }
        )
    with pytest.raises(ValueError, match="only stage='dmd'"):
        MinWMWanAdapter(
            {
                "backend": _TrainingBackend(),
                "stage": "causal_ode",
                "transition_kernel": "x0_renoise",
            }
        )


def test_minwm_sample_one_replay_requires_one_row_at_a_time():
    adapter = _adapter()
    batch = _rollout(adapter)

    assert torch.equal(
        batch.transition_mask.sum(dim=1),
        torch.ones(batch.batch_size, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="update_microbatch_size=1"):
        adapter.recompute_log_probs(batch)
    assert adapter.recompute_log_probs(batch.slice([0])).shape == (1, 3)


def test_minwm_bf16_x0_check_accepts_one_native_ulp_but_not_two():
    adapter = _adapter()
    derived = torch.tensor([[0.1], [0.0]], dtype=torch.float32)
    rounded = derived.to(torch.bfloat16)
    one_ulp = torch.nextafter(
        rounded,
        torch.full_like(rounded, float("inf")),
    )
    two_ulp = torch.nextafter(
        one_ulp,
        torch.full_like(one_ulp, float("inf")),
    )

    adapter._validate_backend_x0(one_ulp, derived)
    with pytest.raises(ValueError, match="inconsistent"):
        adapter._validate_backend_x0(two_ulp, derived)


def test_minwm_bf16_native_path_keeps_lora_and_adam_state_fp32():
    module = _TinyPolicy().to(dtype=torch.bfloat16)
    trainable = _promote_trainable_parameters_to_fp32(module)

    assert module.frozen_base.dtype == torch.bfloat16
    assert [(name, parameter.dtype) for name, parameter in trainable] == [
        ("lora_gain", torch.float32)
    ]

    before = module.lora_gain.detach().clone()
    optimizer = torch.optim.AdamW([module.lora_gain], lr=1e-4, weight_decay=0.0)
    module.lora_gain.grad = torch.full_like(module.lora_gain, 0.01)
    optimizer.step()

    assert not torch.equal(module.lora_gain.detach(), before)
    state = optimizer.state[module.lora_gain]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_minwm_native_rejects_wrong_checkpoint_payload_before_model_load(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "minwm"
    base = tmp_path / "base"
    repo.mkdir()
    base.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"")

    class Runtime:
        requires_cuda = False

        def __init__(self):
            self.load_calls = 0

        def load_components(self, **_kwargs):
            self.load_calls += 1
            raise AssertionError("model construction must follow payload validation")

    runtime = Runtime()
    monkeypatch.setattr(
        minwm_native,
        "_validate_source_checkout",
        lambda *_args, **_kwargs: {"files_sha256": {}},
    )
    monkeypatch.setattr(
        minwm_native,
        "_validate_base_model",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {"generator": {}})
    backend = MinWMWanNativeBackend(
        {
            "replay_credit_assignment": "sample_one",
            "stage": "dmd",
            "minwm_repo_root": str(repo),
            "minwm_commit": "a" * 40,
            "base_model_path": str(base),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"").hexdigest(),
            "checkpoint_payload_key": "generator_ema",
            "device": "cpu",
            "dtype": "bfloat16",
            "chunk_size": 4,
            "total_frames": 20,
            "latent_channels": 16,
            "latent_height": 60,
            "latent_width": 104,
            "local_attn_size": 20,
            "lora_rank": 16,
            "lora_alpha": 16.0,
        },
        _runtime=runtime,
    )

    with pytest.raises(RuntimeError, match="missing explicit payload key"):
        backend.load()
    assert runtime.load_calls == 0


def test_minwm_lora_checkpoint_round_trip(tmp_path):
    source = _adapter()
    with torch.no_grad():
        source.train_module.lora_gain.add_(0.125)
    source.save_pretrained(str(tmp_path))

    restored = _adapter()
    restored.load_checkpoint(str(tmp_path))
    torch.testing.assert_close(
        restored.train_module.lora_gain,
        source.train_module.lora_gain,
        rtol=0,
        atol=0,
    )


def _rewards_from_media(batch):
    camera_offsets = torch.tensor(
        [item["camera_trajectory"]["viewmats"][-1][0][3] for item in batch.metadata],
        dtype=torch.float32,
    )
    assert all(item.get("minwm_conditioning_sha256") for item in batch.metadata)
    values = (
        batch.media.float().flatten(start_dim=1).mean(dim=1) + 0.01 * camera_offsets
    ).detach()
    rewards = RewardBatch(
        raw={"fake_quality": values},
        weighted={"fake_quality": values},
        weighted_total=values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    rewards.validate_against(batch)
    return rewards


def _advantages(batch, rewards):
    return AdvantageComputer(
        {"fake_quality": 1.0},
        mode="grpo",
        epsilon=1e-6,
    )(batch, rewards).advantages


def _policy_loss(adapter, batch, advantages):
    selected = batch.slice([0])
    new_log_probs = adapter.recompute_log_probs(selected)
    return GRPOAlgorithm(
        clip_range=0.001,
        adv_clip_max=5.0,
        beta=0.0,
    ).compute_loss(selected, advantages[:1], new_log_probs)


def _optimizer_plugin(*, update_microbatch_size=1):
    return AlgorithmOptimizerPlugin(
        GRPOAlgorithm(
            clip_range=0.001,
            adv_clip_max=5.0,
            beta=0.0,
        ),
        AdvantageComputer(
            {"fake_quality": 1.0},
            mode="grpo",
            epsilon=1e-6,
        ),
        optimizer_config={
            "max_initial_logprob_delta": 1e-6,
            "require_initial_clipfrac_zero": True,
            "require_finite_gradients": True,
            "require_nonzero_gradients": True,
        },
        update_microbatch_size=update_microbatch_size,
    )


def test_minwm_fake_reward_causal_chain_updates_only_lora():
    adapter = _adapter()
    batch = _rollout(adapter)
    rewards = _rewards_from_media(batch)
    advantages = _advantages(batch, rewards)
    selected = batch.slice([0])
    new_log_probs = adapter.recompute_log_probs(selected)
    loss, metrics = GRPOAlgorithm(beta=0.0).compute_loss(
        selected,
        advantages[:1],
        new_log_probs,
    )

    assert torch.isfinite(rewards.weighted_total).all()
    assert float(rewards.weighted_total.var(unbiased=False)) > 1e-8
    assert torch.isfinite(advantages).all()
    assert int(torch.count_nonzero(advantages)) == batch.batch_size
    assert abs(float(advantages.mean())) < 1e-6
    torch.testing.assert_close(new_log_probs, selected.old_log_probs, rtol=0, atol=1e-6)
    assert float(metrics["clipfrac"]) == 0.0
    assert loss.requires_grad

    before_lora = adapter.train_module.lora_gain.detach().clone()
    before_frozen = adapter.train_module.frozen_base.detach().clone()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = adapter.train_module.lora_gain.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 1e-8
    assert adapter.train_module.frozen_base.grad is None
    optimizer.step()

    assert float((adapter.train_module.lora_gain.detach() - before_lora).abs()) > 1e-8
    assert torch.equal(adapter.train_module.frozen_base, before_frozen)


def test_minwm_existing_optimizer_plugin_runs_the_same_causal_chain():
    adapter = _adapter()
    batch = _rollout(adapter)
    rewards = _rewards_from_media(batch)
    before = adapter.train_module.lora_gain.detach().clone()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.05)

    metrics = _optimizer_plugin(update_microbatch_size=1).step(
        adapter,
        batch,
        rewards,
        optimizer,
        batch.context,
    )

    assert metrics["logprob_delta_abs_max"] <= 1e-6
    assert metrics["clipfrac"] == 0.0
    assert metrics["gradients_finite"] is True
    assert metrics["grad_norm"] > 1e-8
    assert float((adapter.train_module.lora_gain.detach() - before).abs()) > 1e-8


def test_minwm_advantage_sign_flips_gradient_and_zero_advantage_zeroes_it():
    adapter = _adapter()
    batch = _rollout(adapter)
    advantages = _advantages(batch, _rewards_from_media(batch))

    positive_loss, _ = _policy_loss(adapter, batch, advantages)
    positive_grad = torch.autograd.grad(
        positive_loss,
        adapter.train_module.lora_gain,
    )[0]
    negative_loss, _ = _policy_loss(adapter, batch, -advantages)
    negative_grad = torch.autograd.grad(
        negative_loss,
        adapter.train_module.lora_gain,
    )[0]
    torch.testing.assert_close(
        negative_grad,
        -positive_grad,
        rtol=1e-5,
        atol=1e-7,
    )

    probe = batch.old_log_probs.detach().clone().requires_grad_(True)
    probe_loss, _ = GRPOAlgorithm(beta=0.0).compute_loss(batch, advantages, probe)
    probe_grad = torch.autograd.grad(probe_loss, probe)[0]
    assert float(probe_grad.norm()) > 1e-8
    zero_loss, _ = GRPOAlgorithm(beta=0.0).compute_loss(
        batch,
        torch.zeros_like(advantages),
        probe,
    )
    zero_grad = torch.autograd.grad(zero_loss, probe)[0]
    assert int(torch.count_nonzero(zero_grad)) == 0


def test_minwm_zero_lr_control_has_gradient_but_exactly_no_parameter_delta():
    source = _adapter()
    batch = _rollout(source).detach()
    rewards = _rewards_from_media(batch)
    advantages = _advantages(batch, rewards)
    control = _adapter()
    before = control.train_module.lora_gain.detach().clone()
    loss, _ = _policy_loss(control, batch, advantages)
    optimizer = torch.optim.SGD(control.parameters(), lr=0.0)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert control.train_module.lora_gain.grad is not None
    assert float(control.train_module.lora_gain.grad.norm()) > 1e-8
    optimizer.step()

    assert torch.equal(control.train_module.lora_gain, before)


def test_minwm_constant_reward_has_exact_zero_advantage_and_no_fake_update():
    source = _adapter()
    batch = _rollout(source).detach()
    constant_values = torch.full((batch.batch_size,), 0.5)
    rewards = RewardBatch(
        raw={"fake_quality": constant_values},
        weighted={"fake_quality": constant_values},
        weighted_total=constant_values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    advantages = _advantages(batch, rewards)
    assert torch.equal(advantages, torch.zeros_like(advantages))

    control = _adapter()
    before = copy.deepcopy(control.train_module.state_dict())
    loss, _ = _policy_loss(control, batch, advantages)
    optimizer = torch.optim.SGD(control.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = control.train_module.lora_gain.grad
    assert gradient is not None
    assert torch.equal(gradient, torch.zeros_like(gradient))
    optimizer.step()

    for name, value in control.train_module.state_dict().items():
        assert torch.equal(value, before[name])


def test_minwm_constant_reward_plugin_gate_rejects_without_update():
    adapter = _adapter()
    batch = _rollout(adapter)
    constant_values = torch.full((batch.batch_size,), 0.5)
    rewards = RewardBatch(
        raw={"fake_quality": constant_values},
        weighted={"fake_quality": constant_values},
        weighted_total=constant_values,
        valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
        sample_id=list(batch.sample_id),
    )
    before = copy.deepcopy(adapter.train_module.state_dict())
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.05)

    try:
        _optimizer_plugin().step(
            adapter,
            batch,
            rewards,
            optimizer,
            batch.context,
        )
    except RuntimeError as exc:
        assert "Gradient gate failed: all gradients are zero" in str(exc)
    else:
        raise AssertionError("constant reward must fail before the optimizer update")

    for name, value in adapter.train_module.state_dict().items():
        assert torch.equal(value, before[name])


def _runner_config(output_dir, *, max_steps, resume_from=None):
    return config_from_dict(
        {
            "run_name": "minwm-tiny-resume",
            "seed": 41,
            "use_lora": True,
            "model": {
                "name": "minwm_wan_rl",
                "model_family": "wan",
                "latent_shape": [1, 2, 2, 2],
                "media_shape": [5, 1, 2, 2],
                "extra": {
                    "backend_factory": (f"{__name__}:build_minwm_runner_test_backend"),
                    "stage": "dmd",
                    "transition_kernel": "x0_renoise",
                    "replay_credit_assignment": "sample_one",
                    "denoising_timesteps": [1000, 750, 500, 250],
                    "num_chunks": 1,
                    "chunk_size": 2,
                    "total_frames": 2,
                    "lora_rank": 1,
                    "lora_alpha": 1.0,
                },
            },
            "dataset": {
                "prompts": [
                    "orbit around a red block",
                    "track a blue sphere from the left",
                ],
                "sampling_strategy": "deterministic_shuffle",
                "sampling_seed": 73,
            },
            "sample": {
                "name": "full_trajectory",
                "batch_size": 1,
                "num_steps": 4,
                "samples_per_prompt": 4,
                "guidance_scale": 1.0,
            },
            "rollout": {
                "num_chunks": 1,
                "chunk_size": 2,
                "denoising_timesteps": [1000, 750, 500, 250],
                "camera_control": {"trajectory": "right"},
            },
            "algorithm": {
                "name": "grpo",
                "beta": 0.0,
                "advantage_mode": "grpo",
            },
            "rewards": {
                "weights": {"mock": 1.0},
                "clients": {
                    "mock": {
                        "name": "mock",
                        "version": "v2",
                        "mode": "prompt_media",
                    }
                },
                "fail_policy": "raise",
            },
            "optimizer": {
                "name": "algorithm",
                "params": {
                    "max_initial_logprob_delta": 1e-6,
                    "require_initial_clipfrac_zero": True,
                    "require_nonzero_gradients": True,
                },
            },
            "train": {
                "learning_rate": 0.01,
                "max_steps": max_steps,
                "save_every": 1,
                "precision": "fp32",
                "update_microbatch_size": 1,
            },
            "runner": {
                "show_progress": False,
                "disable_rollout_cache": True,
            },
            "paths": {
                "output_dir": str(output_dir),
                "resume_from": None if resume_from is None else str(resume_from),
            },
        }
    )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_minwm_runner_resume_matches_continuous_training(tmp_path):
    continuous = ExperimentRunner(_runner_config(tmp_path / "continuous", max_steps=2))
    initial = continuous.adapter.train_module.lora_gain.detach().clone()
    continuous.run()

    split_dir = tmp_path / "split"
    ExperimentRunner(_runner_config(split_dir, max_steps=1)).run()
    resumed = ExperimentRunner(
        _runner_config(
            split_dir,
            max_steps=2,
            resume_from=split_dir / "checkpoint_000001",
        )
    )
    assert resumed.start_step == 1
    resumed.run()

    assert not torch.equal(continuous.adapter.train_module.lora_gain, initial)
    torch.testing.assert_close(
        resumed.adapter.train_module.lora_gain,
        continuous.adapter.train_module.lora_gain,
        rtol=0,
        atol=0,
    )
    _assert_nested_equal(
        resumed.optimizer.state_dict(),
        continuous.optimizer.state_dict(),
    )


def test_minwm_config_rejects_non_streaming_sample_one_update(tmp_path):
    values = config_to_dict(_runner_config(tmp_path, max_steps=1))
    values["train"]["update_microbatch_size"] = None
    with pytest.raises(ValueError, match="update_microbatch_size=1"):
        config_from_dict(values)

    unsupported_stage = config_to_dict(_runner_config(tmp_path, max_steps=1))
    unsupported_stage["model"]["extra"]["stage"] = "causal_ode"
    with pytest.raises(ValueError, match="only model.extra.stage=dmd"):
        config_from_dict(unsupported_stage)


def test_minwm_static_preflight_lists_native_runtime_dependencies(tmp_path):
    report = static_preflight(_runner_config(tmp_path, max_steps=1))
    model = next(item for item in report.components if item.kind == "model")

    assert model.name == "minwm_wan_rl"
    assert set(model.dependencies) >= {
        "torch",
        "diffusers",
        "transformers",
        "peft",
        "omegaconf",
        "easydict",
        "einops",
        "ftfy",
        "regex",
        "flash_attn",
    }


def _native_config_values(tmp_path):
    values = config_to_dict(_runner_config(tmp_path, max_steps=1))
    values["model"]["extra"].update(
        {
            "backend_factory": (
                "visual_rl.model_adapters.minwm_wan_native_backend:"
                "build_minwm_wan_native_backend"
            ),
            "chunk_size": 4,
            "total_frames": 20,
            "latent_channels": 16,
            "latent_height": 60,
            "latent_width": 104,
            "local_attn_size": 20,
            "dtype": "bfloat16",
        }
    )
    values["rollout"].update(
        {
            "num_chunks": 5,
            "chunk_size": 4,
            "denoising_timesteps": [1000, 750, 500, 250],
        }
    )
    values["train"]["precision"] = "bf16"
    return values


def test_minwm_native_config_rejects_geometry_and_scorer_identity_drift(tmp_path):
    values = _native_config_values(tmp_path)
    config_from_dict(values)

    wrong_geometry = copy.deepcopy(values)
    wrong_geometry["model"]["extra"]["latent_height"] = 61
    with pytest.raises(ValueError, match="native geometry mismatch"):
        config_from_dict(wrong_geometry)

    strict_3d = copy.deepcopy(values)
    strict_3d["rewards"] = {
        "weights": {"reward_3d": 1.0},
        "clients": {
            "reward_3d": {
                "name": "reward_3d",
                "url": "http://127.0.0.1:8089",
                "protocol_mode": "strict_v2",
                "server_revision": "frozen-world-r1",
                "frame_indices": list(range(0, 77, 4)),
            }
        },
        "fail_policy": "raise",
    }
    config_from_dict(strict_3d)

    wrong_scorer = copy.deepcopy(strict_3d)
    wrong_scorer["rewards"]["clients"]["reward_3d"]["frame_indices"][1] = 5
    with pytest.raises(ValueError, match="frame_indices"):
        config_from_dict(wrong_scorer)
