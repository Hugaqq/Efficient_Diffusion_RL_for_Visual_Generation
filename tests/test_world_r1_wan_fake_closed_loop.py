"""CPU fake closed loop for World-R1/Wan full-trajectory training."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import types

import torch

from visual_rl.core.types import StepContext
from visual_rl.feedback.router import RewardRouter
from visual_rl.feedback.world_r1_rewards import (
    STRICT_V2,
    WorldR1Reward3DClient,
    WorldR1RewardGeneralClient,
)
from visual_rl.model_adapters.wan import (
    DEFAULT_WAN_LORA_TARGETS,
    WorldR1WanLegacyAdapter,
)
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.rollout.full_trajectory import build_rollout_engine


def _camera_matrix(translation: float) -> str:
    return f"[1 0 0 0] [0 1 0 0] [0 0 1 0] [{translation:.3f} 0 0 1]"


def _camera_trajectory(frames: int, offset: float) -> dict[str, str]:
    return {
        f"frame{index}": _camera_matrix(offset + index * 0.01)
        for index in range(frames)
    }


class _PeftConfig:
    target_modules = set(DEFAULT_WAN_LORA_TARGETS)


class _FakeWorldTransformer(torch.nn.Module):
    def __init__(self, lora_value: float = 0.2):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.lora_weight = torch.nn.Parameter(torch.tensor(lora_value))
        self.peft_config = {"default": _PeftConfig()}
        self.active_adapter = "default"
        self.config = types.SimpleNamespace(in_channels=1)
        self.dtype = torch.float32

    def forward(self, *, hidden_states, **_kwargs):
        value = self.base_weight + self.lora_weight
        return (torch.zeros_like(hidden_states) + value,)

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


class _FakeWorldPipeline:
    def __init__(self, lora_value: float = 0.2):
        self.transformer = _FakeWorldTransformer(lora_value)
        self.scheduler = types.SimpleNamespace(timesteps=torch.tensor([9, 5]))
        self._execution_device = "cpu"
        self.vae_scale_factor_temporal = 1

    def encode_prompt(self, *, prompt, do_classifier_free_guidance, **_kwargs):
        embeds = torch.arange(1, len(prompt) + 1, dtype=torch.float32).reshape(-1, 1)
        negative = -embeds if do_classifier_free_guidance else None
        return embeds, negative


def _world_adapter(observed: dict, *, lora_value: float = 0.2):
    def get_camera_trajectories(prompts, *, frames_per_trajectory, **_kwargs):
        trajectories = [
            _camera_trajectory(frames_per_trajectory, index * 0.1)
            for index in range(len(prompts))
        ]
        observed["camera_trajectories"] = trajectories
        return trajectories, ["pan"] * len(prompts), list(prompts), [{}] * len(prompts)

    def prepare_latents(*, camera_trajectories, batch_size, num_frames, **_kwargs):
        assert camera_trajectories == observed["camera_trajectories"]
        observed["prepared_frames"] = num_frames
        return torch.zeros(batch_size, 1, 1, 1, 1)

    def pipeline(pipeline, *, train_cfg, **kwargs):
        assert train_cfg is False
        assert kwargs["use_camera_trajectory"] is False
        assert kwargs["latents"].shape[0] == kwargs["prompt_embeds"].shape[0]
        observed.setdefault("pipeline_calls", []).append(kwargs)
        batch_size = kwargs["prompt_embeds"].shape[0]
        policy_value = (
            pipeline.transformer.base_weight + pipeline.transformer.lora_weight
        ).detach()
        latents = [
            torch.full((batch_size, 1, 1, 1, 1), float(index)) for index in range(3)
        ]
        old_log_probs = [policy_value.expand(batch_size) for _ in range(2)]
        videos = torch.linspace(0.0, 1.0, steps=batch_size * 24).reshape(
            batch_size, 2, 3, 2, 2
        )
        return videos, latents, old_log_probs, [], [torch.tensor(9), torch.tensor(5)]

    def sde_step(_scheduler, noise_pred, _timestep, _sample, *, prev_sample):
        del prev_sample
        log_prob = noise_pred.reshape(noise_pred.shape[0], -1).mean(dim=1)
        return None, log_prob

    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": "/offline/fake-wan",
            "device": "cpu",
            "use_lora": True,
            "extra": {
                "wan_backend": "world_r1",
                "wan_pipeline_with_logprob": pipeline,
                "sde_step_with_logprob": sde_step,
                "get_camera_trajectories_for_batch": get_camera_trajectories,
                "prepare_latents_with_camera": prepare_latents,
            },
        }
    )
    adapter.pipeline = _FakeWorldPipeline(lora_value)
    adapter.transformer = adapter.pipeline.transformer
    adapter.scheduler = adapter.pipeline.scheduler
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter.model_path = "/offline/fake-wan"
    return adapter


class _FakeJpegEncoder:
    encoding_metadata = {
        "name": "shape_bytes",
        "format": "test",
        "wire_compatible": False,
    }

    def __call__(self, image):
        return f"fake-jpeg:{'x'.join(str(item) for item in image.shape)}".encode()


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.content = json.dumps(payload).encode("utf-8")

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield self.content

    def close(self):
        return None


class _FakeRewardTransport:
    def __init__(self, outputs: list[float]):
        self.outputs = outputs
        self.calls: list[dict] = []

    def post(self, url, *, data, timeout, headers, allow_redirects, stream):
        assert url.startswith("http://127.0.0.1:")
        assert timeout == 1.0
        assert headers == {"Content-Type": "application/json"}
        assert allow_redirects is False
        assert stream is True
        payload = json.loads(data)
        self.calls.append(payload)
        count = len(payload["prompts"])
        return _FakeResponse(
            {
                "protocol_version": STRICT_V2,
                "sample_id": payload["sample_id"],
                "valid_mask": [True] * count,
                "outputs": self.outputs[:count],
            }
        )


def _reward_router(general_transport, reward_3d_transport):
    encoder = _FakeJpegEncoder()
    router = RewardRouter(
        {
            "weights": {"reward_general": 0.25, "reward_3d": 0.75},
            "fail_policy": "raise",
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": "http://127.0.0.1:8090",
                    "timeout": 1.0,
                    "retries": 0,
                    "protocol_mode": STRICT_V2,
                    "media_layout": "BFCHW",
                    "transport": general_transport,
                    "jpeg_encoder": encoder,
                },
                "reward_3d": {
                    "name": "reward_3d",
                    "url": "http://127.0.0.1:8089",
                    "timeout": 1.0,
                    "retries": 0,
                    "protocol_mode": STRICT_V2,
                    "media_layout": "BFCHW",
                    "require_camera_trajectory": True,
                    "transport": reward_3d_transport,
                    "jpeg_encoder": encoder,
                },
            },
        }
    )
    assert isinstance(router.clients["reward_general"], WorldR1RewardGeneralClient)
    assert isinstance(router.clients["reward_3d"], WorldR1Reward3DClient)
    return router


def _install_fake_peft_checkpoint_api(monkeypatch):
    def get_state(model, adapter_name):
        assert adapter_name == "default"
        return {"lora_weight": model.lora_weight}

    def load_weights(path, device):
        assert device == "cpu"
        return torch.load(
            Path(path) / "adapter_model.bin",
            map_location="cpu",
            weights_only=True,
        )

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


def _rollout(adapter, context):
    engine = build_rollout_engine(
        {
            "name": "full_trajectory",
            "samples_per_prompt": 2,
            "num_steps": 2,
            "frames": 2,
            "height": 2,
            "width": 2,
            "guidance_scale": 1.0,
            "train_cfg": False,
            "use_camera_trajectory": True,
        }
    )
    return engine.sample(
        adapter,
        ["move the camera"],
        [{"prompt_id": "prompt-world-r1"}],
        context,
    )


def _score(router, batch):
    rewards = router.score(
        batch.media,
        batch.prompts,
        batch.metadata,
        sample_id=batch.sample_id,
    ).canonical()
    rewards.validate_against(batch)
    return rewards


def _optimizer_step(adapter, batch, rewards):
    base_before = adapter.transformer.base_weight.detach().clone()
    lora_before = adapter.transformer.lora_weight.detach().clone()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    new_log_probs = adapter.recompute_log_probs(batch)
    loss, metrics = GRPOAlgorithm(clip_range=0.5).compute_loss(
        batch,
        rewards.weighted_total,
        new_log_probs,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["approx_kl"])
    assert adapter.transformer.base_weight.grad is None
    torch.testing.assert_close(adapter.transformer.base_weight, base_before)
    assert not torch.equal(adapter.transformer.lora_weight, lora_before)
    return optimizer


def test_world_r1_wan_full_trajectory_reward_update_checkpoint_resume(
    tmp_path, monkeypatch
):
    _install_fake_peft_checkpoint_api(monkeypatch)
    observed: dict = {}
    adapter = _world_adapter(observed)
    context = StepContext(
        step=7,
        seed=123,
        epoch_tag=2,
        policy_version=4,
    )
    batch = _rollout(adapter, context)

    assert batch.context is context
    assert batch.media_layout == "BFCHW"
    assert batch.media.shape == (2, 2, 3, 2, 2)
    assert batch.old_log_probs.shape == (2, 2)
    assert batch.timesteps.tolist() == [[9, 5], [9, 5]]
    assert batch.model_metadata["rollout"] == "full_trajectory"
    assert batch.model_metadata["wan_backend"] == "world_r1"
    assert "selected_timestep_indices" not in batch.model_metadata
    assert batch.prompt_id == ["prompt-world-r1", "prompt-world-r1"]
    assert batch.group_id[0] == batch.group_id[1]
    assert batch.branch_id == [0, 1]
    assert len(set(batch.sample_id)) == 2
    assert all(item.startswith("step-000007-rank-0000-") for item in batch.sample_id)
    assert all(item["rollout_kind"] == "full_trajectory" for item in batch.metadata)
    assert [item["sample_index"] for item in batch.metadata] == [0, 1]
    assert observed["prepared_frames"] == 2
    assert len(observed["pipeline_calls"]) == 1
    assert "index" not in observed["pipeline_calls"][0]

    general_transport = _FakeRewardTransport([0.2, 0.8])
    reward_3d_transport = _FakeRewardTransport([0.4, 1.0])
    router = _reward_router(general_transport, reward_3d_transport)
    rewards = _score(router, batch)

    assert set(rewards.raw) == {"reward_general", "reward_3d"}
    assert rewards.sample_id == batch.sample_id
    assert rewards.valid_mask.tolist() == [True, True]
    assert rewards.weighted_total.device.type == "cpu"
    assert len(general_transport.calls) == 1
    assert len(reward_3d_transport.calls) == 1
    general_payload = general_transport.calls[0]
    reward_3d_payload = reward_3d_transport.calls[0]
    assert general_payload["sample_id"] == batch.sample_id
    assert reward_3d_payload["sample_id"] == batch.sample_id
    assert len(general_payload["images"]) == 2
    assert all(
        base64.b64decode(image) == b"fake-jpeg:2x2x3"
        for image in general_payload["images"]
    )
    assert [len(video) for video in reward_3d_payload["videos"]] == [2, 2]
    assert reward_3d_payload["camera_trajectories"] == [
        item["camera_trajectory"] for item in batch.metadata
    ]
    assert rewards.metadata["reward_general"]["encoding"]["input_layout"] == "BFCHW"
    assert rewards.metadata["reward_general"]["encoding"]["selected_frame_index"] == 1
    assert rewards.metadata["reward_3d"]["encoding"]["input_layout"] == "BFCHW"
    assert rewards.metadata["reward_3d"]["encoding"]["frames_per_video"] == 2

    assert [name for name, _ in adapter.named_parameters()] == [
        "transformer.lora_weight"
    ]
    optimizer = _optimizer_step(adapter, batch, rewards)
    updated = adapter.transformer.lora_weight.detach().clone()
    parameter = adapter.transformer.lora_weight
    adapter.save_pretrained(tmp_path)
    with torch.no_grad():
        parameter.fill_(99.0)
    adapter.load_checkpoint(tmp_path)
    assert adapter.transformer.lora_weight is parameter
    assert optimizer.param_groups[0]["params"] == [parameter]
    torch.testing.assert_close(parameter, updated)

    resumed_observed: dict = {}
    resumed = _world_adapter(resumed_observed, lora_value=-3.0)
    resumed_parameter = resumed.transformer.lora_weight
    resumed.load_checkpoint(tmp_path)
    assert resumed.transformer.lora_weight is resumed_parameter
    torch.testing.assert_close(resumed_parameter, updated)

    resumed_context = StepContext(
        step=8,
        seed=124,
        epoch_tag=2,
        policy_version=5,
    )
    resumed_batch = _rollout(resumed, resumed_context)
    resumed_rewards = _score(router, resumed_batch)
    assert resumed_batch.context is resumed_context
    assert resumed_batch.sample_id != batch.sample_id
    assert all(
        item.startswith("step-000008-rank-0000-") for item in resumed_batch.sample_id
    )
    assert resumed_rewards.sample_id == resumed_batch.sample_id
    _optimizer_step(resumed, resumed_batch, resumed_rewards)
    assert len(general_transport.calls) == 2
    assert len(reward_3d_transport.calls) == 2
