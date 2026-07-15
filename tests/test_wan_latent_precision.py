"""Regression coverage for Wan recomputation latent precision."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter


def test_wan_recompute_casts_model_input_without_rounding_sde_latents() -> None:
    received: dict[str, torch.Tensor] = {}

    class RecordingTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.dtype = torch.bfloat16

        def forward(self, *, hidden_states, **_kwargs):
            received["transformer_hidden_states"] = hidden_states.detach().clone()
            return (hidden_states + self.anchor,)

    def recording_sde_step(
        _scheduler,
        model_output,
        _timestep,
        sample,
        *,
        prev_sample,
        return_dt_and_std_dev_t=False,
    ):
        assert return_dt_and_std_dev_t is True
        received["sde_current_latent"] = sample.detach().clone()
        received["sde_next_latent"] = prev_sample.detach().clone()
        log_prob = model_output.reshape(model_output.shape[0], -1).mean(dim=1)
        coefficient = torch.ones(
            (sample.shape[0], 1, 1, 1, 1),
            dtype=sample.dtype,
            device=sample.device,
        )
        return prev_sample, log_prob, sample, coefficient, coefficient, coefficient

    transformer = RecordingTransformer()
    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": "/offline/fake-wan",
            "use_lora": False,
            "extra": {
                "wan_backend": "flash",
                "sde_step_with_logprob": recording_sde_step,
            },
        }
    )
    adapter.pipeline = SimpleNamespace(transformer=transformer)
    adapter.transformer = transformer
    adapter.scheduler = object()
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32

    current = torch.tensor(
        [[[[[[1.0001, -2.0003]]]]]],
        dtype=torch.float32,
    )
    next_ = torch.tensor(
        [[[[[[3.0007, -4.0013]]]]]],
        dtype=torch.float32,
    )
    assert not torch.equal(current.to(torch.bfloat16).float(), current)
    assert not torch.equal(next_.to(torch.bfloat16).float(), next_)

    batch = RolloutBatch(
        prompts=["precision regression"],
        metadata=[{}],
        latents=current,
        next_latents=next_,
        timesteps=torch.tensor([[999]]),
        old_log_probs=torch.zeros(1, 1),
        model_metadata={
            "sample_config": {"guidance_scale": 1.0, "train_cfg": False}
        },
        model_tensors={
            "prompt_embeds": torch.ones(1, 1, dtype=torch.float32),
            "coefficient": torch.ones(1, 1, dtype=torch.float32),
        },
    )

    adapter.recompute_log_probs(batch)

    hidden_states = received["transformer_hidden_states"]
    assert hidden_states.dtype is torch.bfloat16
    assert not torch.equal(hidden_states.float(), current[:, 0])

    sde_current = received["sde_current_latent"]
    sde_next = received["sde_next_latent"]
    assert sde_current.dtype is torch.float32
    assert sde_next.dtype is torch.float32
    torch.testing.assert_close(sde_current, current[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(sde_next, next_[:, 0], rtol=0, atol=0)
