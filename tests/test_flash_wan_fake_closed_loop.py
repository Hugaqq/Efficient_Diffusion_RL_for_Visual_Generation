"""CPU fake closed loop for the Flash/Wan reference-v1 contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import torch

from visual_rl.configs import load_config
from visual_rl.configs.schema import section_to_dict
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.model_adapters.wan import (
    DEFAULT_WAN_LORA_TARGETS,
    WorldR1WanLegacyAdapter,
)
from visual_rl.optimizers.advantages import AdvantageComputer, AdvantageFunction
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.preflight import StaticPreflightError, static_preflight
from visual_rl.rollout.base import RolloutEngine
from visual_rl.rollout.full_trajectory import build_rollout_engine


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "visual_rl/configs/presets/flash_wan_reference.yaml"


class _FlashScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([999, 500])
        self.sigmas = torch.tensor([1.0, 0.8, 0.2])

    def index_for_timestep(self, timestep):
        matches = (self.timesteps == int(timestep)).nonzero()
        return int(matches[0].item())


class _FakeLoraTransformer(torch.nn.Module):
    def __init__(self, value: float = 0.2):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(value))
        self.peft_config = {"default": object()}
        self.active_adapter = "default"
        self.config = types.SimpleNamespace(in_channels=1)
        self.dtype = torch.float32

    def forward(self, *, hidden_states, **_kwargs):
        return (torch.zeros_like(hidden_states) + self.lora_weight,)

    def save_pretrained(self, path, *, safe_serialization, selected_adapters):
        assert safe_serialization is True
        assert selected_adapters == ["default"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"target_modules": DEFAULT_WAN_LORA_TARGETS}),
            encoding="utf-8",
        )
        torch.save(
            {"lora_weight": self.lora_weight.detach().cpu()},
            path / "adapter_model.bin",
        )


class _FlashPipeline:
    def __init__(self):
        self.transformer = _FakeLoraTransformer()
        self.scheduler = _FlashScheduler()
        self._execution_device = "cpu"

    def encode_prompt(self, *, prompt, do_classifier_free_guidance, **_kwargs):
        batch = len(prompt)
        embeds = torch.arange(1, batch + 1, dtype=torch.float32).reshape(batch, 1)
        negative = -embeds if do_classifier_free_guidance else None
        return embeds, negative


def _sde_step(
    scheduler,
    model_output,
    timestep,
    sample,
    *,
    prev_sample,
    return_dt_and_std_dev_t=False,
):
    batch = sample.shape[0]
    indices = [scheduler.index_for_timestep(item) for item in timestep]
    sigma = scheduler.sigmas[indices].reshape(batch, 1, 1, 1, 1).to(sample)
    sigma_prev = (
        scheduler.sigmas[[item + 1 for item in indices]]
        .reshape(batch, 1, 1, 1, 1)
        .to(sample)
    )
    dt = sigma_prev - sigma
    sigma_max = scheduler.sigmas[1].to(sample)
    sigma_min = scheduler.sigmas[-1].to(sample)
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    sqrt_neg_dt = torch.sqrt(-dt)
    coefficient = 1 / (
        sqrt_neg_dt / std_dev_t + std_dev_t * sqrt_neg_dt * (1 - sigma) / (2 * sigma)
    )
    log_prob = model_output.reshape(batch, -1).mean(dim=1)
    result = (prev_sample, log_prob, sample, std_dev_t, sqrt_neg_dt, coefficient)
    return result if return_dt_and_std_dev_t else result[:4]


def _flash_adapter(observed: dict):
    def pipeline(pipeline, *, train_cfg, **kwargs):
        assert train_cfg is True
        observed["kwargs"] = kwargs
        observed.setdefault("calls", []).append(
            {
                "index": kwargs["index"],
                "generator_seed": kwargs["generator"].initial_seed(),
            }
        )
        batch = kwargs["prompt_embeds"].shape[0]
        marker = kwargs["prompt_embeds"][:, 0] + 10 * kwargs["index"]
        latent = marker.reshape(batch, 1, 1, 1, 1).expand(-1, 1, 2, 1, 1)
        next_latent = latent + 1
        old = marker.clone()
        videos = marker.reshape(batch, 1, 1, 1, 1).expand(-1, 2, 3, 2, 2)
        return (
            videos,
            [latent, next_latent],
            [old],
            [torch.zeros(batch), torch.ones(batch)],
            kwargs["index"],
        )

    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": "/offline/fake-wan",
            "use_lora": True,
            "extra": {
                "wan_backend": "flash",
                "wan_pipeline_with_logprob": pipeline,
                "sde_step_with_logprob": _sde_step,
            },
        }
    )
    adapter.pipeline = _FlashPipeline()
    adapter.transformer = adapter.pipeline.transformer
    adapter.scheduler = adapter.pipeline.scheduler
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    return adapter


def _rng_flash_adapter():
    def pipeline(_pipeline, *, train_cfg, **kwargs):
        assert train_cfg is True
        batch = kwargs["prompt_embeds"].shape[0]
        shape = (batch, 1, 2, 1, 1)
        latent = torch.randn(shape, generator=kwargs["generator"])
        transition_noise = torch.randn(shape)
        next_latent = latent + transition_noise
        videos = next_latent.mean(dim=(1, 2, 3, 4), keepdim=True).expand(-1, 2, 3, 2, 2)
        old_log_prob = transition_noise.reshape(batch, -1).mean(dim=1)
        return (
            videos,
            [latent, next_latent],
            [old_log_prob],
            [torch.zeros(batch)],
            kwargs["index"],
        )

    observed = {}
    adapter = _flash_adapter(observed)
    adapter.config["wan_pipeline_with_logprob"] = pipeline
    return adapter


def _single_step_engine(samples_per_prompt: int = 2) -> RolloutEngine:
    return build_rollout_engine(
        {
            "name": "single_step",
            "samples_per_prompt": samples_per_prompt,
            "num_steps": 2,
            "guidance_scale": 5.0,
            "selected_step_strategy": "first",
            "timestep_range": [0, 1],
        }
    )


def _install_fake_peft_checkpoint_api(monkeypatch):
    def get_state(model, adapter_name):
        assert adapter_name == "default"
        return {"lora_weight": model.lora_weight}

    def load_weights(path, device):
        assert device == "cpu"
        return torch.load(Path(path) / "adapter_model.bin", weights_only=False)

    def set_state(model, state, adapter_name):
        assert adapter_name == "default"
        with torch.no_grad():
            model.lora_weight.copy_(state["lora_weight"])
        return types.SimpleNamespace(missing_keys=(), unexpected_keys=())

    peft = types.ModuleType("peft")
    peft_utils = types.ModuleType("peft.utils")
    save_load = types.ModuleType("peft.utils.save_and_load")
    save_load.get_peft_model_state_dict = get_state
    save_load.load_peft_weights = load_weights
    save_load.set_peft_model_state_dict = set_state
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.utils", peft_utils)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", save_load)


def test_flash_selected_step_shapes_actual_timestep_and_coefficient_contract():
    observed = {}
    adapter = _flash_adapter(observed)
    batch = adapter.sample_single_step(
        ["fake-a", "fake-b"],
        [{}, {}],
        {
            "num_steps": 2,
            "guidance_scale": 5.0,
            "selected_timestep_indices": [0, 0],
        },
    )

    assert observed["kwargs"]["index"] == 0
    assert "use_camera_trajectory" not in observed["kwargs"]
    assert "noise_level" not in observed["kwargs"]
    assert batch.timesteps.tolist() == [[999], [999]]
    assert batch.old_log_probs.shape == (2, 1)
    assert batch.latents.shape[:2] == (2, 1)
    assert batch.next_latents.shape == batch.latents.shape
    assert batch.media_layout == "BFCHW"
    assert batch.model_tensors["coefficient"].shape == (2, 1)
    assert torch.isfinite(batch.model_tensors["coefficient"]).all()
    assert (batch.model_tensors["coefficient"] > 0).all()
    assert adapter.pipeline.transformer is adapter.transformer
    expected = _sde_step(
        adapter.scheduler,
        torch.zeros_like(batch.latents[:, 0]),
        batch.timesteps[:, 0],
        batch.latents[:, 0],
        prev_sample=batch.next_latents[:, 0],
        return_dt_and_std_dev_t=True,
    )[5].reshape(2, -1)[:, :1]
    torch.testing.assert_close(batch.model_tensors["coefficient"], expected)

    recomputed = adapter.recompute_log_probs(batch)
    assert recomputed.shape == (2, 1)
    bad = batch.replace(
        model_tensors={
            **batch.model_tensors,
            "coefficient": batch.model_tensors["coefficient"] * 2,
        }
    )
    with pytest.raises(ValueError, match="coefficient mismatch"):
        adapter.recompute_log_probs(bad)


def test_flash_groups_two_indices_and_restores_every_batch_field_in_input_order():
    observed = {}
    adapter = _flash_adapter(observed)
    batch = adapter.sample_single_step(
        ["sample-0", "sample-1", "sample-2"],
        [{"row": 0}, {"row": 1}, {"row": 2}],
        {
            "num_steps": 2,
            "guidance_scale": 5.0,
            "selected_timestep_indices": [1, 0, 1],
            "seed": 41,
        },
    )

    assert [call["index"] for call in observed["calls"]] == [0, 1]
    assert len({call["generator_seed"] for call in observed["calls"]}) == 2
    assert batch.metadata == [{"row": 0}, {"row": 1}, {"row": 2}]
    assert batch.timesteps.tolist() == [[500], [999], [500]]
    assert batch.model_metadata["actual_scheduler_timesteps"] == [500, 999, 500]
    assert batch.old_log_probs[:, 0].tolist() == [11.0, 2.0, 13.0]
    assert batch.latents[:, 0, 0, 0, 0, 0].tolist() == [11.0, 2.0, 13.0]
    assert batch.next_latents[:, 0, 0, 0, 0, 0].tolist() == [12.0, 3.0, 14.0]
    assert batch.media[:, 0, 0, 0, 0].tolist() == [11.0, 2.0, 13.0]
    assert batch.kl[:, 0].tolist() == [1.0, 0.0, 1.0]
    assert batch.model_tensors["prompt_embeds"].tolist() == [[1.0], [2.0], [3.0]]
    assert batch.model_tensors["negative_prompt_embeds"].tolist() == [
        [-1.0],
        [-2.0],
        [-3.0],
    ]
    assert batch.model_tensors["coefficient"].shape == (3, 1)
    assert not torch.equal(
        batch.model_tensors["coefficient"][0],
        batch.model_tensors["coefficient"][1],
    )

    repeated_observed = {}
    _flash_adapter(repeated_observed).sample_single_step(
        ["sample-0", "sample-1", "sample-2"],
        [{"row": 0}, {"row": 1}, {"row": 2}],
        {
            "num_steps": 2,
            "guidance_scale": 5.0,
            "selected_timestep_indices": [1, 0, 1],
            "seed": 41,
        },
    )
    assert repeated_observed["calls"] == observed["calls"]


def test_flash_step_context_seed_isolated_from_process_global_rng():
    def sample(seed: int, perturb_seed: int):
        torch.manual_seed(perturb_seed)
        torch.rand(7)
        state_before = torch.random.get_rng_state().clone()
        batch = _single_step_engine().sample(
            _rng_flash_adapter(),
            ["seeded prompt"],
            [{}],
            StepContext(step=3, seed=seed, epoch_tag=3),
        )
        state_after = torch.random.get_rng_state()
        assert torch.equal(state_after, state_before)
        return batch

    first = sample(seed=73, perturb_seed=11)
    repeated = sample(seed=73, perturb_seed=991)
    different = sample(seed=74, perturb_seed=11)

    torch.testing.assert_close(first.latents, repeated.latents)
    torch.testing.assert_close(first.next_latents, repeated.next_latents)
    torch.testing.assert_close(first.old_log_probs, repeated.old_log_probs)
    assert not torch.equal(first.next_latents, different.next_latents)


def test_published_coefficient_is_patch_sixth_item_not_its_inverse():
    # Wan2.1's t=999 flow sigmas produce denominator 0.133743799731.
    sigma = torch.tensor(1.0, dtype=torch.float64)
    sigma_prev = torch.tensor(0.982725258686269, dtype=torch.float64)
    sigma_max = sigma_prev
    sigma_min = torch.tensor(0.0, dtype=torch.float64)
    sqrt_neg_dt = torch.sqrt(sigma - sigma_prev)
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    denominator = sqrt_neg_dt / std_dev_t + (
        std_dev_t * sqrt_neg_dt * (1 - sigma) / (2 * sigma)
    )
    patch_sixth_item = 1 / denominator

    torch.testing.assert_close(
        denominator,
        torch.tensor(0.133743799731, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        patch_sixth_item,
        torch.tensor(7.476982125591, dtype=torch.float64),
        rtol=0,
        atol=3e-11,
    )
    assert round(patch_sixth_item.item(), 4) == 7.4770


def test_recompute_casts_hidden_states_and_prompt_embeds_to_transformer_dtype():
    class StrictBFloat16Transformer(_FakeLoraTransformer):
        def __init__(self):
            super().__init__()
            self.dtype = torch.bfloat16

        def forward(self, *, hidden_states, encoder_hidden_states, **_kwargs):
            assert hidden_states.dtype is self.dtype
            assert encoder_hidden_states.dtype is self.dtype
            return (hidden_states * self.lora_weight.to(self.dtype),)

    adapter = _flash_adapter({})
    transformer = StrictBFloat16Transformer()
    adapter.pipeline.transformer = transformer
    adapter.transformer = transformer
    adapter.dtype = torch.float32
    batch = adapter.sample_single_step(
        ["fake"],
        [{}],
        {
            "num_steps": 2,
            "guidance_scale": 5.0,
            "selected_timestep_indices": [0],
        },
    )

    recomputed = adapter.recompute_log_probs(batch)
    assert recomputed.dtype is torch.float32
    assert torch.isfinite(recomputed).all()


def test_reference_v1_numeric_max_clipped_objective_and_beta_fail_closed():
    batch = RolloutBatch(
        prompts=["a", "b"],
        metadata=[{}, {}],
        old_log_probs=torch.zeros(2, 1),
        model_tensors={"coefficient": torch.tensor([[1.0], [3.0]])},
    )
    algorithm = FlashGRPOAlgorithm(
        objective_version="reference_v1",
        clip_range=0.1,
        beta=0.0,
    )
    new_log_probs = torch.tensor([[0.0], [torch.log(torch.tensor(2.0))]])
    loss, metrics = algorithm.compute_loss(
        batch,
        torch.ones(2),
        new_log_probs,
    )
    torch.testing.assert_close(loss, torch.tensor(-1.075))
    torch.testing.assert_close(
        metrics["flash_rectification_weight_mean"], torch.tensor(1.0)
    )

    for invalid_beta in (-0.1, 0.1):
        with pytest.raises(ValueError, match="requires beta=0"):
            FlashGRPOAlgorithm.from_config(
                {
                    "objective_version": "reference_v1",
                    "beta": invalid_beta,
                }
            )
        with pytest.raises(ValueError, match="requires beta=0"):
            FlashGRPOAlgorithm(
                objective_version="reference_v1", beta=invalid_beta
            ).compute_loss(batch, torch.ones(2), new_log_probs)
    with pytest.raises(ValueError, match="coefficient"):
        algorithm.compute_loss(
            batch.replace(model_tensors={}),
            torch.ones(2),
            new_log_probs,
        )


def test_one_lora_update_save_load_and_resume_contract(tmp_path, monkeypatch):
    _install_fake_peft_checkpoint_api(monkeypatch)
    adapter = _flash_adapter({})
    batch = adapter.sample_single_step(
        ["fake"],
        [{}],
        {
            "num_steps": 2,
            "guidance_scale": 5.0,
            "selected_timestep_indices": [0],
        },
    )
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    algorithm = FlashGRPOAlgorithm(objective_version="reference_v1", beta=0.0)
    before = adapter.transformer.lora_weight.detach().clone()
    new_log_probs = adapter.recompute_log_probs(batch)
    loss, _metrics = algorithm.compute_loss(batch, torch.ones(1), new_log_probs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    updated = adapter.transformer.lora_weight.detach().clone()
    assert not torch.equal(updated, before)

    parameter_id = id(adapter.transformer.lora_weight)
    adapter.save_pretrained(tmp_path)
    with torch.no_grad():
        adapter.transformer.lora_weight.fill_(99.0)
    adapter.load_checkpoint(tmp_path)
    assert id(adapter.transformer.lora_weight) == parameter_id
    torch.testing.assert_close(adapter.transformer.lora_weight, updated)
    resumed_log_probs = adapter.recompute_log_probs(batch)
    assert torch.isfinite(resumed_log_probs).all()


def test_reference_preset_is_explicit_and_passes_static_preflight():
    config = load_config(PRESET)
    assert config.model.extra["wan_backend"] == "flash"
    assert config.algorithm.objective_version == "reference_v1"
    assert config.algorithm.beta == 0.0
    assert config.use_lora is True
    assert config.sample.batch_size == 1
    assert config.sample.samples_per_prompt == 2
    assert config.sample.name == "single_step"
    assert config.rollout["selected_step_strategy"] == "first"
    assert config.model.model_path == ""
    static_preflight(config)

    config.algorithm.beta = 0.01
    with pytest.raises(StaticPreflightError, match="requires beta=0"):
        static_preflight(config)
    config.algorithm.beta = -0.01
    with pytest.raises(StaticPreflightError, match=r"algorithm.beta must be >= 0.0"):
        static_preflight(config)


def test_reference_preset_rollout_satisfies_real_grpo_advantage_contract():
    config = load_config(PRESET)
    rollout_config = section_to_dict(config.sample)
    rollout_config.update(config.rollout)
    rollout: RolloutEngine = build_rollout_engine(rollout_config)
    batch = rollout.sample(
        _flash_adapter({}),
        [config.dataset.prompts[0]],
        [{}],
        StepContext(step=0, seed=config.seed, epoch_tag=0),
    )
    rewards = RewardBatch(
        raw={"mock": torch.tensor([0.0, 1.0])},
        weighted={"mock": torch.tensor([0.0, 1.0])},
        weighted_total=torch.tensor([0.0, 1.0]),
        valid_mask=torch.ones(2, dtype=torch.bool),
        sample_id=batch.sample_id,
    )
    advantage: AdvantageFunction = AdvantageComputer(
        reward_weights=config.rewards.weights,
        mode=config.algorithm.advantage_mode,
    )

    result = advantage(batch, rewards)

    assert batch.batch_size == 2
    assert batch.group_id[0] == batch.group_id[1]
    assert result.advantages.shape == (2,)
    assert result.advantages[0] < 0 < result.advantages[1]
    assert result.metrics["group_size"] == 2.0
