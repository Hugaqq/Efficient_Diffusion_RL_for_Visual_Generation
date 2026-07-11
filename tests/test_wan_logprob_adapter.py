from __future__ import annotations


def test_world_r1_wan_recompute_log_probs_matches_old_shape_and_is_finite(tmp_path):
    import torch

    from visual_rl.core.types import RolloutBatch
    from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

    repo_root = tmp_path / "World-R1-main"
    repo_root.mkdir()

    def fake_sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        *,
        prev_sample,
        **kwargs,
    ):
        del scheduler, timestep, kwargs
        prev_sample_mean = sample + 0.1 * model_output
        log_prob = -((prev_sample - prev_sample_mean) ** 2).mean(dim=tuple(range(1, sample.ndim)))
        return prev_sample, log_prob, prev_sample_mean, torch.ones_like(log_prob), torch.ones_like(log_prob), 1.0

    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
            del timestep, kwargs
            embed_scale = encoder_hidden_states.mean(dim=1).view(-1, 1, 1, 1, 1)
            return (hidden_states + self.bias + embed_scale,)

    class FakeScheduler:
        timesteps = torch.tensor([2, 1])

    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": str(tmp_path / "fake-model"),
            "repo_root": str(repo_root),
            "device": "cpu",
            "wan_pipeline_with_logprob": lambda *args, **kwargs: None,
            "sde_step_with_logprob": fake_sde_step_with_logprob,
        }
    )
    adapter.pipeline = object()
    adapter.transformer = FakeTransformer()
    adapter.scheduler = FakeScheduler()
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32

    batch = RolloutBatch(
        prompts=["a", "b"],
        metadata=[{}, {}],
        media=torch.zeros(2, 2, 3, 4, 4),
        latents=torch.zeros(2, 2, 1, 1, 2, 2),
        next_latents=torch.ones(2, 2, 1, 1, 2, 2) * 0.05,
        timesteps=torch.tensor([[2, 1], [2, 1]]),
        old_log_probs=torch.zeros(2, 2),
        kl=torch.zeros(2, 2),
        model_metadata={
            "sample_config": {
                "guidance_scale": 2.0,
                "noise_level": 0.7,
                "sde_type": "flow_sde",
                "diffusion_clip": False,
                "diffusion_clip_value": 0.45,
                "train_cfg": True,
            }
        },
        model_tensors={
            "prompt_embeds": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            "negative_prompt_embeds": torch.zeros(2, 2),
        },
    )

    new_log_probs = adapter.recompute_log_probs(batch)

    assert new_log_probs.shape == batch.old_log_probs.shape
    assert torch.isfinite(new_log_probs).all()
    loss = new_log_probs.mean()
    loss.backward()
    assert adapter.transformer.bias.grad is not None
