from __future__ import annotations


def _install_fake_runtime(monkeypatch):
    import sys
    import types

    import torch

    class FakeScheduler:
        timesteps = torch.tensor([2, 1])

    class FakeTransformer(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.15))

        def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
            del timestep, kwargs
            embed_scale = encoder_hidden_states.mean(dim=1).view(-1, 1, 1, 1, 1)
            return (hidden_states + self.bias + embed_scale,)

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()
            self.scheduler = FakeScheduler()
            self._execution_device = torch.device("cpu")

        def to(self, device):
            self._execution_device = torch.device(device)
            return self

        def encode_prompt(self, prompt, negative_prompt, **kwargs):
            del kwargs
            prompt_embeds = torch.arange(len(prompt) * 2, dtype=torch.float32).reshape(len(prompt), 2)
            negative_prompt_embeds = -torch.ones(len(negative_prompt), 2)
            return prompt_embeds, negative_prompt_embeds

    class WanPipeline:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            del model_path, kwargs
            return FakePipeline()

    def fake_wan_pipeline_with_logprob(pipeline, **kwargs):
        del pipeline
        batch_size = kwargs["prompt_embeds"].shape[0]
        steps = int(kwargs["num_inference_steps"])
        videos = torch.ones(batch_size, kwargs["num_frames"], 3, kwargs["height"], kwargs["width"])
        all_latents = [
            torch.full((batch_size, 2, 1, 2, 2), 0.1 * step, dtype=torch.float32)
            for step in range(steps + 1)
        ]
        all_log_probs = [torch.full((batch_size,), -0.2, dtype=torch.float32) for _ in range(steps)]
        all_kl = [torch.zeros(batch_size, dtype=torch.float32) for _ in range(steps)]
        all_timesteps = [torch.tensor(2), torch.tensor(1)]
        return videos, all_latents, all_log_probs, all_kl, all_timesteps

    def fake_sde_step_with_logprob(scheduler, model_output, timestep, sample, *, prev_sample, **kwargs):
        del scheduler, timestep, kwargs
        prev_sample_mean = sample + 0.05 * model_output
        log_prob = -((prev_sample - prev_sample_mean) ** 2).mean(dim=tuple(range(1, sample.ndim)))
        return prev_sample, log_prob, prev_sample_mean, torch.ones_like(log_prob), torch.ones_like(log_prob), 1.0

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(WanPipeline=WanPipeline))
    return fake_wan_pipeline_with_logprob, fake_sde_step_with_logprob


def _bounded_config(tmp_path, run_dir, fake_pipeline, fake_sde):
    from visual_rl.configs.schema import load_config

    repo_root = tmp_path / "World-R1-main"
    model_path = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    repo_root.mkdir(exist_ok=True)
    model_path.mkdir(exist_ok=True)

    config = load_config("visual_rl/configs/presets/world_r1_wan_bounded.yaml")
    config.paths.output_dir = str(run_dir)
    config.model.model_path = str(model_path)
    config.model.extra["repo_root"] = str(repo_root)
    config.model.extra["wan_pipeline_with_logprob"] = fake_pipeline
    config.model.extra["sde_step_with_logprob"] = fake_sde
    config.rewards.weights = {"reward_general": 1.0}
    config.rewards.clients = {"reward_general": {"name": "resume_fake_reward"}}
    config.runner.show_progress = False
    return config


def test_world_r1_bounded_resume_from_latest_runs_one_more_step(monkeypatch, tmp_path):
    import json

    import numpy as np

    import visual_rl.model_adapters.wan  # noqa: F401
    import visual_rl.feedback.clients  # noqa: F401
    import visual_rl.feedback.world_r1_rewards  # noqa: F401
    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.runner import ExperimentRunner

    fake_pipeline, fake_sde = _install_fake_runtime(monkeypatch)

    class FakeRewardClient:
        def score(self, media, prompts, metadata):
            del media, metadata
            return np.linspace(0.2, 0.8, len(prompts), dtype=np.float32), {"server": "resume-fake"}

    monkeypatch.setitem(REWARD_CLIENTS._items, "resume_fake_reward", FakeRewardClient)  # noqa: SLF001
    run_dir = tmp_path / "run"

    first_config = _bounded_config(tmp_path, run_dir, fake_pipeline, fake_sde)
    first_metrics = ExperimentRunner(first_config).run(max_steps=1)
    assert first_metrics[0]["step"] == 0
    assert json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))["step"] == 1

    second_config = _bounded_config(tmp_path, run_dir, fake_pipeline, fake_sde)
    second_config.paths.resume_from = str(run_dir / "latest.json")
    second_metrics = ExperimentRunner(second_config).run(max_steps=2)

    assert [item["step"] for item in second_metrics] == [1]
    assert json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))["step"] == 2
    assert (run_dir / "checkpoint_000002" / "transformer_state.pt").exists()
    metric_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metric_lines) == 2
