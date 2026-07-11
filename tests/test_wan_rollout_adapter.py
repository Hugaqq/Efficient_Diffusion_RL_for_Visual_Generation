from __future__ import annotations


def test_world_r1_wan_sample_builds_strict_rollout_batch(tmp_path):
    import torch

    from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

    repo_root = tmp_path / "World-R1-main"
    repo_root.mkdir()
    calls = []

    class FakeTransformer(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    class FakeScheduler:
        timesteps = torch.tensor([9, 4])

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()
            self.scheduler = FakeScheduler()
            self._execution_device = torch.device("cpu")

        def encode_prompt(self, prompt, negative_prompt, **kwargs):
            del kwargs
            prompt_embeds = torch.arange(len(prompt) * 3, dtype=torch.float32).reshape(len(prompt), 3)
            negative_embeds = -torch.ones(len(negative_prompt), 3)
            return prompt_embeds, negative_embeds

    def fake_pipeline_with_logprob(pipeline, **kwargs):
        calls.append(kwargs)
        batch_size = kwargs["prompt_embeds"].shape[0]
        videos = torch.ones(batch_size, kwargs["num_frames"], 3, kwargs["height"], kwargs["width"])
        all_latents = [
            torch.full((batch_size, 2, 1, 2, 2), float(step), dtype=torch.float32)
            for step in range(kwargs["num_inference_steps"] + 1)
        ]
        all_log_probs = [
            torch.full((batch_size,), -0.1 * (step + 1), dtype=torch.float32)
            for step in range(kwargs["num_inference_steps"])
        ]
        all_kl = [torch.full((batch_size,), 0.01 * step, dtype=torch.float32) for step in range(kwargs["num_inference_steps"])]
        all_timesteps = [torch.tensor(9), torch.tensor(4)]
        return videos, all_latents, all_log_probs, all_kl, all_timesteps

    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": str(tmp_path / "fake-model"),
            "repo_root": str(repo_root),
            "device": "cpu",
            "wan_pipeline_with_logprob": fake_pipeline_with_logprob,
        }
    )
    adapter.pipeline = FakePipeline()
    adapter.transformer = adapter.pipeline.transformer
    adapter.scheduler = adapter.pipeline.scheduler
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter.model_path = str(tmp_path / "fake-model")

    batch = adapter.sample(
        prompts=["orbit left", "push in"],
        metadata=[{"id": 0}, {"id": 1}],
        rollout_config={
            "num_steps": 2,
            "seed": 123,
            "frames": 3,
            "height": 4,
            "width": 5,
            "guidance_scale": 4.5,
            "noise_level": 0.7,
        },
    )

    batch.validate_lightweight(strict=True)
    assert batch.media.shape == (2, 3, 3, 4, 5)
    assert batch.latents.shape == (2, 2, 2, 1, 2, 2)
    assert batch.next_latents.shape == batch.latents.shape
    assert batch.timesteps.tolist() == [[9, 4], [9, 4]]
    assert batch.old_log_probs.shape == (2, 2)
    assert batch.kl.shape == (2, 2)
    assert batch.seed == 123
    assert batch.model_tensors["prompt_embeds"].shape == (2, 3)
    assert batch.model_tensors["negative_prompt_embeds"].shape == (2, 3)
    assert batch.model_metadata["reference_path"] == "<injected>"
    assert batch.model_metadata["sample_config"]["seed"] == 123
    assert calls[0]["return_dict"] is False
    assert calls[0]["output_type"] == "pt"
    assert calls[0]["generator"].initial_seed() == 123
