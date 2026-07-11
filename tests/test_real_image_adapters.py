def test_real_image_adapters_register_deferred():
    import pytest

    import visual_rl.model_adapters.sd3  # noqa: F401
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.model_adapters.diffusers_common import AdapterNotLoadedError

    adapter = MODEL_ADAPTERS.get("sd3_tempflow")({"name": "sd3_tempflow", "model_path": "", "extra": {"defer_load": True}})
    assert adapter.name
    with pytest.raises(AdapterNotLoadedError):
        adapter.parameters()


def test_real_image_presets_load():
    from visual_rl.configs.schema import load_config

    cfg = load_config("visual_rl/configs/presets/sd3_tempflow_adapter.yaml")
    assert cfg.model.model_family == "sd3"
    assert cfg.runner.strict_rollout_validation is True


def test_sd3_pipeline_helper_filters_unsupported_kwargs_without_return_dict():
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    adapter = SD3TempFlowAdapter({"name": "sd3_tempflow", "extra": {"defer_load": True}})
    adapter.pipeline = object()
    calls = []

    def fake_pipeline(pipe, prompt_embeds, num_inference_steps):
        calls.append(
            {
                "pipe": pipe,
                "prompt_embeds": prompt_embeds,
                "num_inference_steps": num_inference_steps,
            }
        )
        return "ok"

    adapter._pipeline_with_logprob = fake_pipeline

    result = adapter._call_pipeline_with_logprob(
        prompt_embeds="embeds",
        num_inference_steps=2,
        guidance_scale=4.5,
        return_dict=True,
        kl_reward=0.1,
    )

    assert result == "ok"
    assert calls == [{"pipe": adapter.pipeline, "prompt_embeds": "embeds", "num_inference_steps": 2}]


def test_sd3_pipeline_helper_passes_return_dict_false_when_supported():
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    adapter = SD3TempFlowAdapter({"name": "sd3_tempflow", "extra": {"defer_load": True}})
    adapter.pipeline = object()
    calls = []

    def fake_pipeline(pipe, prompt_embeds, return_dict=True):
        calls.append({"pipe": pipe, "prompt_embeds": prompt_embeds, "return_dict": return_dict})
        return "ok"

    adapter._pipeline_with_logprob = fake_pipeline

    result = adapter._call_pipeline_with_logprob(prompt_embeds="embeds", unsupported="drop-me")

    assert result == "ok"
    assert calls == [{"pipe": adapter.pipeline, "prompt_embeds": "embeds", "return_dict": False}]


def test_sd3_sample_accepts_three_tuple_pipeline_result_and_zero_fills_kl(monkeypatch):
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    class FakeScheduler:
        timesteps = torch.tensor([2, 1])

    class FakePipeline:
        scheduler = FakeScheduler()

    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    adapter = SD3TempFlowAdapter(
        {
            "name": "sd3_tempflow",
            "device": "cpu",
            "dtype": "float32",
            "resolution": 8,
            "extra": {"defer_load": True},
        }
    )
    adapter.pipeline = FakePipeline()
    adapter.transformer = FakeTransformer()

    prompt_embeds = torch.ones(1, 4, 3)
    pooled_prompt_embeds = torch.ones(1, 3)
    neg_prompt_embeds = torch.zeros(1, 4, 3)
    neg_pooled_prompt_embeds = torch.zeros(1, 3)

    def fake_encode_text(prompts):
        if prompts == [""]:
            return neg_prompt_embeds, neg_pooled_prompt_embeds
        return prompt_embeds, pooled_prompt_embeds

    monkeypatch.setattr(adapter, "_encode_text", fake_encode_text)

    def fake_pipeline(
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        num_inference_steps,
        guidance_scale,
        generator,
        output_type,
        height,
        width,
        kl_reward,
        return_dict=False,
    ):
        assert pipe is adapter.pipeline
        assert num_inference_steps == 2
        assert guidance_scale == 3.0
        assert generator is not None
        assert output_type == "pt"
        assert height == 8
        assert width == 8
        assert kl_reward == 0.0
        assert return_dict is False
        images = torch.zeros(1, 3, 8, 8)
        latents = [torch.full((1, 4, 2, 2), value, dtype=torch.float32) for value in (0.0, 1.0, 2.0)]
        log_probs = [torch.tensor([0.25]), torch.tensor([0.5])]
        return images, latents, log_probs

    adapter._pipeline_with_logprob = fake_pipeline

    batch = adapter.sample(
        ["a cube"],
        [{"id": 1}],
        {"num_steps": 2, "guidance_scale": 3.0, "seed": 123, "output_type": "pt"},
    )

    assert batch.prompts == ["a cube"]
    assert batch.metadata == [{"id": 1}]
    assert batch.latents.shape == (1, 2, 4, 2, 2)
    assert batch.next_latents.shape == (1, 2, 4, 2, 2)
    assert batch.timesteps.tolist() == [[2, 1]]
    assert batch.old_log_probs.tolist() == [[0.25, 0.5]]
    assert batch.kl.shape == batch.old_log_probs.shape
    assert torch.equal(batch.kl, torch.zeros_like(batch.old_log_probs))


def test_sd3_transformer_dtype_uses_trainable_module_dtype():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    adapter = SD3TempFlowAdapter({"name": "sd3_tempflow", "extra": {"defer_load": True}})
    adapter.transformer = torch.nn.Linear(2, 2, dtype=torch.float16)

    assert adapter._transformer_dtype() == torch.float16


def test_sd3_recompute_casts_latents_to_transformer_dtype_for_guidance_modes():
    import torch

    from visual_rl.core.types import RolloutBatch
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    class FakeScheduler:
        pass

    class FakePipeline:
        scheduler = FakeScheduler()

    class FakeTransformer(torch.nn.Module):
        def __init__(self, base_batch_size):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
            self.base_batch_size = base_batch_size
            self.calls = []

        def forward(
            self,
            *,
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
            return_dict,
        ):
            self.calls.append(
                {
                    "guidance_mode": "cfg" if hidden_states.shape[0] == self.base_batch_size * 2 else "direct",
                    "hidden_states_dtype": hidden_states.dtype,
                    "hidden_states_shape": tuple(hidden_states.shape),
                    "timestep_shape": tuple(timestep.shape),
                    "encoder_hidden_states_shape": tuple(encoder_hidden_states.shape),
                    "pooled_projections_shape": tuple(pooled_projections.shape),
                    "return_dict": return_dict,
                }
            )
            return (torch.ones_like(hidden_states),)

    def run_case(guidance_scale, expected_batch, expected_mode):
        batch_size = 2
        num_steps = 2
        adapter = SD3TempFlowAdapter(
            {
                "name": "sd3_tempflow",
                "device": "cpu",
                "dtype": "float32",
                "extra": {"defer_load": True},
            }
        )
        transformer = FakeTransformer(batch_size)
        adapter.pipeline = FakePipeline()
        adapter.transformer = transformer

        latents = torch.arange(batch_size * num_steps * 4, dtype=torch.float32).reshape(
            batch_size,
            num_steps,
            1,
            2,
            2,
        )
        next_latents = latents + 0.5
        timesteps = torch.tensor([[9, 7], [9, 7]])
        prompt_embeds = torch.ones(batch_size, 3, 5)
        pooled_prompt_embeds = torch.ones(batch_size, 5)
        neg_prompt_embeds = torch.zeros(batch_size, 3, 5)
        neg_pooled_prompt_embeds = torch.zeros(batch_size, 5)

        def fake_sde_step_with_logprob(scheduler, noise_pred, timestep, sample, prev_sample):
            assert scheduler is adapter.pipeline.scheduler
            assert noise_pred.dtype == torch.float32
            assert sample.dtype == torch.float32
            assert prev_sample.dtype == torch.float32
            log_prob = timestep.float() + sample.flatten(start_dim=1).mean(dim=1)
            return prev_sample, log_prob, sample, torch.ones_like(log_prob)

        adapter._sde_step_with_logprob = fake_sde_step_with_logprob

        rollout_batch = RolloutBatch(
            prompts=["a red cube", "a blue cube"],
            metadata=[{"id": 0}, {"id": 1}],
            media=torch.zeros(batch_size, 3, 8, 8),
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            old_log_probs=torch.zeros(batch_size, num_steps),
            kl=torch.zeros(batch_size, num_steps),
            model_metadata={"guidance_scale": guidance_scale},
            model_tensors={
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": neg_prompt_embeds,
                "negative_pooled_prompt_embeds": neg_pooled_prompt_embeds,
            },
        )

        recomputed = adapter.recompute_log_probs(rollout_batch)

        assert recomputed.shape == (batch_size, num_steps)
        assert len(transformer.calls) == num_steps
        for call in transformer.calls:
            assert call["guidance_mode"] == expected_mode
            assert call["hidden_states_dtype"] == torch.float16
            assert call["hidden_states_shape"] == (expected_batch, 1, 2, 2)
            assert call["timestep_shape"] == (expected_batch,)
            assert call["encoder_hidden_states_shape"] == (expected_batch, 3, 5)
            assert call["pooled_projections_shape"] == (expected_batch, 5)
            assert call["return_dict"] is False

    for guidance_scale, expected_batch, expected_mode in [(1.0, 2, "direct"), (3.0, 4, "cfg")]:
        run_case(guidance_scale, expected_batch, expected_mode)


def test_sd3_numeric_smoke_cli_uses_explicit_model_path_and_options(monkeypatch, capsys):
    import json

    import torch

    from scripts import legacy_cli as cli
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.core.types import RolloutBatch

    class FakeSD3Adapter:
        name = "tempflow_sd3_legacy"

        def __init__(self, config):
            assert config["name"] == "sd3_tempflow"
            assert config["model_family"] == "sd3"
            assert config["model_path"] == "/models/sd35"
            assert config["use_lora"] is False
            assert config["extra"]["resolution"] == 80
            assert config["extra"]["dtype"] == "float32"
            assert config["extra"]["device"] == "cpu"
            assert config["extra"]["repo_root"] == "/ref/tempflow"
            assert config["extra"]["lora_rank"] == 8
            assert config["extra"]["lora_alpha"] == 16
            assert config["extra"]["max_sequence_length"] == 77
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.weight = torch.nn.Parameter(torch.ones(3))

        def sample(self, prompts, metadata, rollout_config):
            assert prompts == ["a blue cube"]
            assert metadata == [{"source": "sd3_numeric_smoke", "adapter_key": "sd3_tempflow"}]
            assert rollout_config["num_steps"] == 2
            assert rollout_config["guidance_scale"] == 3.5
            assert rollout_config["seed"] == 99
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=torch.zeros(1, 3, 8, 8),
                latents=torch.zeros(1, 2, 16, 4, 4),
                next_latents=torch.zeros(1, 2, 16, 4, 4),
                timesteps=torch.tensor([[1, 0]]),
                old_log_probs=torch.tensor([[0.25, 0.5]]),
                kl=torch.zeros(1, 2),
                seed=rollout_config["seed"],
                model_metadata={
                    "adapter": self.name,
                    "reference_repo": "/ref/tempflow",
                    "reference_pipeline": "sd3_pipeline_with_logprob",
                    "resolution": 80,
                    "guidance_scale": 3.5,
                },
            )

        def recompute_log_probs(self, batch):
            return batch.old_log_probs.clone()

        def parameters(self):
            return [self.weight]

    monkeypatch.setattr(cli, "_register_builtin_plugins", lambda: None)
    monkeypatch.setitem(MODEL_ADAPTERS._items, "sd3_tempflow", FakeSD3Adapter)  # noqa: SLF001

    exit_code = cli.main(
        [
            "sd3-numeric-smoke",
            "--model-path",
            "/models/sd35",
            "--repo-root",
            "/ref/tempflow",
            "--prompt",
            "a blue cube",
            "--resolution",
            "80",
            "--num-steps",
            "2",
            "--guidance-scale",
            "3.5",
            "--seed",
            "99",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--max-sequence-length",
            "77",
            "--disable-lora",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"] == "tempflow_sd3_legacy"
    assert payload["adapter_key"] == "sd3_tempflow"
    assert payload["model_path"] == "/models/sd35"
    assert payload["repo_root"] == "/ref/tempflow"
    assert payload["reference_repo"] == "/ref/tempflow"
    assert payload["resolution"] == 80
    assert payload["num_steps"] == 2
    assert payload["guidance_scale"] == 3.5
    assert payload["seed"] == 99
    assert payload["max_sequence_length"] == 77
    assert payload["media_finite"] is True
    assert payload["old_log_probs_finite"] is True
    assert payload["recomputed_log_probs_finite"] is True
    assert payload["max_abs_logprob_delta"] == 0.0
    assert payload["trainable_parameters"] == 3
    assert payload["shapes"]["old_log_probs"] == [1, 2]
    assert payload["model_metadata"]["reference_pipeline"] == "sd3_pipeline_with_logprob"


def _install_fake_sd3_branching_adapter(monkeypatch, delta_by_step):
    import torch

    from scripts import legacy_cli as cli
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.core.types import RolloutBatch
    from visual_rl.rollout import full_trajectory as rollout_module
    from visual_rl.rollout.branching import BranchingRollout

    seen = {"sample_steps": []}

    class FakeSD3BranchingAdapter:
        name = "tempflow_sd3_legacy"

        def __init__(self, config):
            seen["model_config"] = config
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.weight = torch.nn.Parameter(torch.ones(4))

        def branch_transition_count(self, rollout_config):
            assert rollout_config["num_steps"] == 3
            return 2

        def sample_branching(self, prompts, metadata, rollout_config):
            branch_count = int(rollout_config["branch_count"])
            transition_index = int(rollout_config["branch_step_index"])
            seen["sample_steps"].append(transition_index)
            seen.setdefault("rollout_configs", []).append(dict(rollout_config))
            expanded_metadata = []
            for branch_id in range(branch_count):
                item = dict(metadata[0])
                item.update(
                    {
                        "parent_prompt_index": 0,
                        "branch_id": branch_id,
                        "branch_step_index": transition_index,
                        "branch_timestep_value": 900 - transition_index * 300,
                        "is_main_branch": False,
                        "rollout_kind": "tempflow_branching",
                    }
                )
                expanded_metadata.append(item)
            return RolloutBatch(
                prompts=prompts * branch_count,
                metadata=expanded_metadata,
                media=torch.zeros(branch_count, 3, 8, 8),
                latents=torch.zeros(branch_count, 1, 16, 4, 4),
                next_latents=torch.zeros(branch_count, 1, 16, 4, 4),
                timesteps=torch.full((branch_count, 1), 900 - transition_index * 300),
                old_log_probs=torch.zeros(branch_count, 1),
                kl=torch.zeros(branch_count, 1),
                branch_ids=torch.arange(branch_count),
                seed=rollout_config["seed"],
                model_metadata={
                    "adapter": self.name,
                    "reference_repo": "/ref/tempflow",
                    "reference_pipeline": "sd3_pipeline_with_logprob_perstep",
                    "guidance_scale": rollout_config["guidance_scale"],
                    "trajectory_step_indices": [transition_index],
                    "transformer_training": False,
                },
                model_tensors={
                    "initial_latents": torch.zeros(1, 16, 4, 4),
                    "scheduler_timesteps": torch.tensor(
                        [900.0, 600.0, 300.0]
                    ),
                    "scheduler_sigmas": torch.tensor(
                        [1.0, 0.7, 0.3, 0.0]
                    ),
                },
            )

        def recompute_log_probs(self, batch):
            transition_index = int(batch.model_metadata["branch_step_index"])
            delta = torch.as_tensor(delta_by_step[transition_index], dtype=torch.float32).reshape(-1)
            if delta.numel() == 1:
                delta = delta.repeat(batch.old_log_probs.shape[0])
            assert delta.numel() == batch.old_log_probs.shape[0]
            return batch.old_log_probs + delta.reshape_as(batch.old_log_probs)

        def parameters(self):
            return [self.weight]

    monkeypatch.setattr(cli, "_register_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        rollout_module,
        "build_rollout_engine",
        lambda config: BranchingRollout(config),
    )
    monkeypatch.setitem(MODEL_ADAPTERS._items, "sd3_tempflow", FakeSD3BranchingAdapter)  # noqa: SLF001
    return seen


def test_sd3_branching_numeric_smoke_cli_auto_validates_every_transition(monkeypatch, capsys):
    import json

    import pytest

    from scripts import legacy_cli as cli

    seen = _install_fake_sd3_branching_adapter(
        monkeypatch,
        {
            0: [0.0, 0.002, -0.002],
            1: [0.0005, 0.0, 0.0],
        },
    )

    exit_code = cli.main(
        [
            "sd3-branching-numeric-smoke",
            "--model-path",
            "/models/sd35",
            "--repo-root",
            "/ref/tempflow",
            "--prompt",
            "a green cube",
            "--resolution",
            "80",
            "--num-steps",
            "3",
            "--guidance-scale",
            "3.5",
            "--seed",
            "99",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--max-sequence-length",
            "77",
            "--disable-lora",
            "--branch-count",
            "3",
            "--branch-step-index",
            "auto",
            "--clip-range",
            "0.001",
            "--logprob-atol",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert seen["sample_steps"] == [0, 1]
    assert seen["model_config"] == {
        "name": "sd3_tempflow",
        "model_family": "sd3",
        "model_path": "/models/sd35",
        "use_lora": False,
        "extra": {
            "resolution": 80,
            "dtype": "float32",
            "lora_rank": 8,
            "lora_alpha": 16,
            "max_sequence_length": 77,
            "device": "cpu",
            "repo_root": "/ref/tempflow",
        },
    }
    for transition_index, rollout_config in enumerate(seen["rollout_configs"]):
        assert rollout_config["name"] == "branching"
        assert rollout_config["branch_count"] == 3
        assert rollout_config["include_main"] is False
        assert rollout_config["branch_step_index"] == transition_index

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["sampling_mode"] == "same_seed_replay_per_transition"
    assert payload["replay_fingerprints_consistent"] is True
    assert payload["branch_step_index"] == "auto"
    assert payload["transition_count"] == 2
    assert payload["evaluated_transition_indices"] == [0, 1]
    assert payload["failed_transition_indices"] == []
    assert payload["shapes"]["old_log_probs"] == [2, 3, 1]
    assert payload["shapes"]["recomputed_log_probs"] == [2, 3, 1]
    assert payload["old_log_probs_finite"] is True
    assert payload["recomputed_log_probs_finite"] is True
    assert payload["logprob_delta_finite"] is True
    assert payload["max_abs_logprob_delta"] == pytest.approx(0.002)
    assert payload["clipfrac"] == pytest.approx(2 / 6)
    assert [item["transition_index"] for item in payload["per_transition"]] == [0, 1]
    assert payload["per_transition"][0]["clipfrac"] == pytest.approx(2 / 3)
    assert payload["per_transition"][1]["clipfrac"] == 0.0
    assert payload["per_transition"][0]["shapes"]["latents"] == [3, 1, 16, 4, 4]
    transition_contract = payload["per_transition"][0]["contract_metadata"]
    assert transition_contract["branch_ids"] == [0, 1, 2]
    assert transition_contract["branch_step_indices"] == [0, 0, 0]
    assert transition_contract["model_metadata"]["rollout"] == "branching"
    assert transition_contract["model_metadata"]["branching_mode"] == "shared_prefix"
    assert payload["contract_metadata"]["strict_rollout_validation"] is True
    assert payload["trainable_parameters"] == 4


def test_sd3_branching_numeric_smoke_cli_hard_fails_explicit_transition_atol(monkeypatch, capsys):
    import json

    import pytest

    from scripts import legacy_cli as cli

    seen = _install_fake_sd3_branching_adapter(monkeypatch, {1: [0.02, 0.0]})

    with pytest.raises(ValueError, match=r"transition\(s\) 1.*atol=0.01"):
        cli.main(
            [
                "sd3-branching-numeric-smoke",
                "--model-path",
                "/models/sd35",
                "--repo-root",
                "/ref/tempflow",
                "--num-steps",
                "3",
                "--branch-count",
                "2",
                "--branch-step-index",
                "1",
                "--logprob-atol",
                "0.01",
            ]
        )

    assert seen["sample_steps"] == [1]
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["branch_step_index"] == 1
    assert payload["evaluated_transition_indices"] == [1]
    assert payload["failed_transition_indices"] == [1]
    assert payload["per_transition"][0]["within_atol"] is False
    assert payload["max_abs_logprob_delta"] == pytest.approx(0.02)


def test_image_preview_cli_writes_png_and_metadata_for_sd3_contract(monkeypatch, tmp_path, capsys):
    import json

    import torch

    from scripts import legacy_cli as cli
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.core.types import RolloutBatch

    class FakeSD3PreviewAdapter:
        name = "tempflow_sd3_legacy"

        def __init__(self, config):
            assert config["name"] == "sd3_tempflow"
            assert config["model_family"] == "sd3"
            assert config["model_path"] == "/models/sd35"
            assert config["extra"]["resolution"] == 12
            assert config["extra"]["device"] == "cpu"
            assert config["extra"]["repo_root"] == "/ref/tempflow"
            self.device = torch.device("cpu")

        def sample(self, prompts, metadata, rollout_config):
            assert prompts == ["a preview cube"]
            assert metadata == [{"source": "image_preview", "adapter_key": "sd3_tempflow"}]
            assert rollout_config == {
                "num_steps": 2,
                "guidance_scale": 3.5,
                "seed": 19,
                "output_type": "pt",
            }
            media = torch.tensor(
                [
                    [
                        [[-1.0, 0.0], [0.5, 2.0]],
                        [[0.25, 0.5], [0.75, 1.0]],
                        [[1.0, 0.75], [0.5, 0.25]],
                    ]
                ]
            )
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=media,
                latents=torch.zeros(1, 2, 4, 2, 2),
                next_latents=torch.zeros(1, 2, 4, 2, 2),
                timesteps=torch.tensor([[9, 7]]),
                old_log_probs=torch.zeros(1, 2),
                kl=torch.zeros(1, 2),
                seed=rollout_config["seed"],
                model_metadata={
                    "adapter": self.name,
                    "reference_repo": "/ref/tempflow",
                    "tensor_summary_only": torch.zeros(2, 3),
                },
            )

        def recompute_log_probs(self, batch):
            return batch.old_log_probs.clone()

        def parameters(self):
            return []

    monkeypatch.setattr(cli, "_register_builtin_plugins", lambda: None)
    monkeypatch.setitem(MODEL_ADAPTERS._items, "sd3_tempflow", FakeSD3PreviewAdapter)  # noqa: SLF001

    exit_code = cli.main(
        [
            "image-preview",
            "--adapter",
            "sd3_tempflow",
            "--model-path",
            "/models/sd35",
            "--repo-root",
            "/ref/tempflow",
            "--prompt",
            "a preview cube",
            "--resolution",
            "12",
            "--num-steps",
            "2",
            "--guidance-scale",
            "3.5",
            "--seed",
            "19",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    png_path = tmp_path / "preview_000.png"
    metadata_path = tmp_path / "metadata.json"
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata_path.exists()
    assert payload["adapter"] == "tempflow_sd3_legacy"
    assert payload["adapter_key"] == "sd3_tempflow"
    assert payload["model_path"] == "/models/sd35"
    assert payload["repo_root"] == "/ref/tempflow"
    assert payload["prompt"] == "a preview cube"
    assert payload["media_shape"] == [1, 3, 2, 2]
    assert payload["latents_shape"] == [1, 2, 4, 2, 2]
    assert payload["timesteps_shape"] == [1, 2]
    assert payload["old_log_probs_shape"] == [1, 2]
    assert payload["png_paths"] == [str(png_path)]
    assert payload["metadata_path"] == str(metadata_path)
    assert payload["model_metadata"]["tensor_summary_only"] == {
        "device": "cpu",
        "dtype": "torch.float32",
        "kind": "torch.Tensor",
        "shape": [2, 3],
    }
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved == payload


def test_image_preview_cli_supports_tiny_diffusion_locally(tmp_path, capsys):
    import json

    from scripts import legacy_cli as cli

    exit_code = cli.main(
        [
            "image-preview",
            "--adapter",
            "tiny_diffusion",
            "--model-path",
            "/unused/tiny",
            "--prompt",
            "a local tiny preview",
            "--resolution",
            "4",
            "--num-steps",
            "2",
            "--guidance-scale",
            "1.0",
            "--seed",
            "5",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter_key"] == "tiny_diffusion"
    assert payload["model_family"] == "image"
    assert payload["media_shape"] == [1, 3, 4, 4]
    assert (tmp_path / "preview_000.png").exists()
    assert (tmp_path / "metadata.json").exists()


def test_sd3_bounded_trainer_smoke_cli_uses_visual_rl_trainer_contract(monkeypatch, tmp_path, capsys):
    import json

    import torch

    from scripts import legacy_cli as cli
    from visual_rl.core.types import RewardBatch, RolloutBatch
    import visual_rl.configs.schema as schema_module
    import visual_rl.runner as runner_module

    seen = {}
    real_load_config = schema_module.load_config

    def load_config_without_optimizer_gates(path):
        config = real_load_config(path)
        config.optimizer.params = {}
        return config

    monkeypatch.setattr(
        schema_module,
        "load_config",
        load_config_without_optimizer_gates,
    )

    class FakeAdapter:
        name = "tempflow_sd3_legacy"

        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        def parameters(self):
            return [self.weight]

        def sample(self, prompts, metadata, rollout_config):
            phase = metadata[0]["phase"]
            seen.setdefault("preview_phases", []).append(phase)
            assert prompts == ["a red cube"]
            assert rollout_config["num_steps"] == 2
            assert rollout_config["guidance_scale"] == 3.5
            assert rollout_config["seed"] == 99
            value = 0.25 if phase == "before" else 0.75
            media = torch.full((1, 3, 2, 2), value)
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=media,
                latents=torch.zeros(1, 2, 4, 2, 2),
                next_latents=torch.ones(1, 2, 4, 2, 2) * value,
                timesteps=torch.tensor([[9, 7]]),
                old_log_probs=torch.tensor([[0.1, 0.2]]),
                kl=torch.tensor([[0.0, 0.01]]),
                seed=rollout_config["seed"],
                model_metadata={"adapter": self.name, "phase": phase},
            )

    class FakeFeedbackProvider:
        def score(self, batch):
            phase = batch.metadata[0]["phase"]
            reward = 0.3 if phase == "before" else 0.9
            return RewardBatch(
                raw={"prompt_color": torch.tensor([reward])},
                weighted={"prompt_color": torch.tensor([reward])},
                weighted_total=torch.tensor([reward]),
                valid_mask=torch.tensor([True]),
                metadata={"prompt_color": {"phase": phase}},
            )

    class FakeExperimentRunner:
        def __init__(self, config):
            seen["config"] = config
            self.output_dir = tmp_path
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.adapter = FakeAdapter()
            self.feedback_provider = FakeFeedbackProvider()
            assert config.paths.output_dir == str(tmp_path)
            assert config.paths.output_dir == str(tmp_path)
            assert config.paths.pretrained_model == "/models/sd35"
            assert config.model.name == "sd3_tempflow"
            assert config.model.model_family == "sd3"
            assert config.model.model_path == "/models/sd35"
            assert config.model.extra["repo_root"] == "/ref/tempflow"
            assert config.model.extra["resolution"] == 80
            assert config.model.extra["dtype"] == "float32"
            assert config.model.extra["device"] == "cpu"
            assert config.model.extra["lora_rank"] == 8
            assert config.model.extra["lora_alpha"] == 16
            assert config.model.extra["max_sequence_length"] == 77
            assert config.dataset.prompts == ["a red cube"]
            assert config.sample.name == "branching"
            assert config.sample.num_steps == 2
            assert config.sample.guidance_scale == 3.5
            assert config.paths.resume_from is None
            assert config.train.lora_path is None
            assert config.train.max_steps == 2
            assert config.train.save_every == 2
            assert config.runner.strict_rollout_validation is True
            assert config.runner.disable_rollout_cache is True
            assert config.optimizer.params["max_initial_logprob_delta"] == 1e-5
            assert config.optimizer.params["require_initial_clipfrac_zero"] is True
            assert config.optimizer.params["require_finite_gradients"] is True
            assert config.optimizer.params["require_nonzero_gradients"] is True

        def run(self, max_steps=None):
            seen["max_steps"] = max_steps
            assert max_steps == 2
            with torch.no_grad():
                self.adapter.weight.add_(torch.tensor([0.5, 0.0]))
            rows = [
                {"step": 0, "reward_mean": 0.25, "approx_kl": 0.0},
                {
                    "step": 1,
                    "reward_mean": 0.75,
                    "reward_std": 0.0,
                    "old_logprob_mean": 0.1,
                    "new_logprob_mean": 0.2,
                    "logprob_delta_abs_max": 0.1,
                    "rollout_kl_mean": 0.0,
                    "approx_kl": 0.01,
                    "clipfrac": 0.0,
                    "tempflow_noise_weight_mean": 1.0,
                    "tempflow_active_timestep_frac": 0.5,
                },
            ]
            (self.output_dir / "metrics.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            checkpoint_dir = self.output_dir / "checkpoint_000002"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "adapter_model.safetensors").write_text("fake", encoding="utf-8")
            (self.output_dir / "latest.json").write_text(json.dumps({"step": 2}), encoding="utf-8")
            return rows

    monkeypatch.setattr(runner_module, "ExperimentRunner", FakeExperimentRunner)

    exit_code = cli.main(
        [
            "sd3-bounded-trainer-smoke",
            "--adapter",
            "sd3_tempflow",
            "--model-path",
            "/models/sd35",
            "--repo-root",
            "/ref/tempflow",
            "--prompt",
            "a red cube",
            "--resolution",
            "80",
            "--num-steps",
            "2",
            "--guidance-scale",
            "3.5",
            "--seed",
            "99",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--max-sequence-length",
            "77",
            "--steps",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen["max_steps"] == 2
    assert seen["preview_phases"] == ["before", "after"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["adapter"] == "tempflow_sd3_legacy"
    assert payload["adapter_key"] == "sd3_tempflow"
    assert payload["model_path"] == "/models/sd35"
    assert payload["repo_root"] == "/ref/tempflow"
    assert payload["output_dir"] == str(tmp_path)
    assert payload["metrics_path"] == str(tmp_path / "metrics.jsonl")
    assert payload["latest_path"] == str(tmp_path / "latest.json")
    assert payload["checkpoint_dirs"] == [str(tmp_path / "checkpoint_000002")]
    assert payload["checkpoint_summary"][0]["files"] == ["adapter_model.safetensors"]
    assert payload["steps"] == 2
    assert payload["resolution"] == 80
    assert payload["num_steps"] == 2
    assert payload["guidance_scale"] == 3.5
    assert payload["seed"] == 99
    assert payload["rollout_cache_disabled"] is True
    assert payload["rollout_cache_path"] is None
    assert payload["resume_from"] is None
    assert payload["resume_loaded"] is False
    assert payload["resume_checkpoint_summary"] is None
    assert payload["source_checkpoint_summary"] is None
    assert payload["resume_base_step"] == 0
    assert payload["resume_steps"] == 0
    assert payload["effective_total_step"] == 2
    assert payload["metrics_line_count"] == 2
    assert "clipfrac" in payload["required_metric_keys"]
    assert payload["trainable_parameters"] == 2
    assert payload["trainable_parameter_tensors"] == 1
    assert payload["parameter_delta_abs_max"] == 0.5
    assert payload["parameter_delta_l2"] == 0.5
    assert payload["parameter_delta_nonzero_count"] == 1
    assert payload["latest"] == {"step": 2}
    before_preview = payload["preview_artifacts"]["before"]
    after_preview = payload["preview_artifacts"]["after"]
    assert before_preview["png_paths"] == [str(tmp_path / "previews" / "before" / "preview_000.png")]
    assert after_preview["png_paths"] == [str(tmp_path / "previews" / "after" / "preview_000.png")]
    assert before_preview["reward_mean"] == 0.30000001192092896
    assert after_preview["reward_mean"] == 0.8999999761581421
    assert before_preview["old_logprob_mean"] == 0.15000000596046448
    assert before_preview["rollout_kl_mean"] == 0.004999999888241291
    assert before_preview["reward_metadata"] == {"prompt_color": {"phase": "before"}}
    assert after_preview["model_metadata"] == {"adapter": "tempflow_sd3_legacy", "phase": "after"}
    assert (tmp_path / "previews" / "before" / "preview_000.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "previews" / "after" / "preview_000.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((tmp_path / "previews" / "before" / "metadata.json").read_text(encoding="utf-8")) == before_preview
    assert json.loads((tmp_path / "previews" / "after" / "metadata.json").read_text(encoding="utf-8")) == after_preview
    assert payload["final_metric_extract"] == {
        "approx_kl": 0.01,
        "clipfrac": 0.0,
        "logprob_delta_abs_max": 0.1,
        "new_logprob_mean": 0.2,
        "old_logprob_mean": 0.1,
        "reward_mean": 0.75,
        "reward_std": 0.0,
        "rollout_kl_mean": 0.0,
        "tempflow_active_timestep_frac": 0.5,
        "tempflow_noise_weight_mean": 1.0,
    }
    assert payload["final_metric_artifact_extract"] == payload["final_metric_extract"]
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved == payload


def test_sd3_bounded_trainer_smoke_requires_evidence_metrics(tmp_path):
    import json

    import pytest

    from scripts import legacy_cli as cli

    metrics_path = tmp_path / "metrics.jsonl"
    latest_path = tmp_path / "latest.json"
    checkpoint_dir = tmp_path / "checkpoint_000001"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.safetensors").write_text("fake", encoding="utf-8")
    latest_path.write_text(json.dumps({"step": 1}), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "step": 0,
                "reward_mean": 0.7,
                "reward_std": 0.0,
                "approx_kl": 0.01,
                "old_logprob_mean": 0.1,
                "new_logprob_mean": 0.2,
                "logprob_delta_abs_max": 0.1,
                "rollout_kl_mean": 0.0,
                "tempflow_active_timestep_frac": 0.5,
                "tempflow_noise_weight_mean": 1.0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoints = cli._checkpoint_summary([checkpoint_dir])  # noqa: SLF001
    with pytest.raises(RuntimeError, match="missing required key\\(s\\): clipfrac"):
        cli._validate_bounded_trainer_artifacts(  # noqa: SLF001
            metrics_path,
            latest_path,
            checkpoints,
            expected_steps=1,
        )


def test_sd3_bounded_trainer_smoke_cli_supports_resume_from_checkpoint(monkeypatch, tmp_path, capsys):
    import json

    import torch

    from scripts import legacy_cli as cli
    from visual_rl.core.types import RewardBatch, RolloutBatch
    import visual_rl.runner as runner_module

    output_dir = tmp_path / "resume_run"
    resume_dir = tmp_path / "checkpoint_000005"
    resume_dir.mkdir()
    (resume_dir / "adapter_model.safetensors").write_text("resume", encoding="utf-8")
    seen = {}

    class FakeAdapter:
        name = "tempflow_sd3_legacy"

        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        def parameters(self):
            return [self.weight]

        def sample(self, prompts, metadata, rollout_config):
            phase = metadata[0]["phase"]
            seen.setdefault("preview_phases", []).append(phase)
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=torch.full((1, 3, 2, 2), 0.5),
                latents=torch.zeros(1, 1, 4, 2, 2),
                next_latents=torch.ones(1, 1, 4, 2, 2),
                timesteps=torch.tensor([[7]]),
                old_log_probs=torch.tensor([[0.2]]),
                kl=torch.zeros(1, 1),
                seed=rollout_config["seed"],
                model_metadata={"adapter": self.name, "phase": phase},
            )

    class FakeFeedbackProvider:
        def score(self, batch):
            return RewardBatch(
                raw={"prompt_color": torch.tensor([0.6])},
                weighted={"prompt_color": torch.tensor([0.6])},
                weighted_total=torch.tensor([0.6]),
                valid_mask=torch.tensor([True]),
                metadata={"prompt_color": {"target": "red"}},
            )

    class FakeExperimentRunner:
        def __init__(self, config):
            seen["config"] = config
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.adapter = FakeAdapter()
            self.feedback_provider = FakeFeedbackProvider()
            assert config.paths.output_dir == str(output_dir)
            assert config.paths.output_dir == str(output_dir)
            assert config.paths.resume_from == str(resume_dir)
            assert config.train.lora_path == str(resume_dir)

        def run(self, max_steps=None):
            seen["max_steps"] = max_steps
            assert max_steps == 1
            with torch.no_grad():
                self.adapter.weight.add_(torch.tensor([0.0, 0.25, 0.0]))
            row = {
                "step": 0,
                "reward_mean": 0.6,
                "reward_std": 0.0,
                "old_logprob_mean": 0.2,
                "new_logprob_mean": 0.2,
                "logprob_delta_abs_max": 0.0,
                "rollout_kl_mean": 0.0,
                "approx_kl": 0.0,
                "clipfrac": 0.0,
                "tempflow_active_timestep_frac": 1.0,
                "tempflow_noise_weight_mean": 1.0,
            }
            (self.output_dir / "metrics.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            checkpoint_dir = self.output_dir / "checkpoint_000001"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "adapter_model.safetensors").write_text("fake", encoding="utf-8")
            (self.output_dir / "latest.json").write_text(json.dumps({"step": 1}), encoding="utf-8")
            return [row]

    monkeypatch.setattr(runner_module, "ExperimentRunner", FakeExperimentRunner)

    exit_code = cli.main(
        [
            "sd3-bounded-trainer-smoke",
            "--adapter",
            "sd3_tempflow",
            "--model-path",
            "/models/sd35",
            "--repo-root",
            "/ref/tempflow",
            "--steps",
            "1",
            "--output-dir",
            str(output_dir),
            "--resume-from",
            str(resume_dir),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert seen["max_steps"] == 1
    assert seen["preview_phases"] == ["before", "after"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["resume_loaded"] is True
    assert payload["resume_from"] == str(resume_dir)
    assert payload["resume_checkpoint_summary"] == {
        "path": str(resume_dir),
        "file_count": 1,
        "files": ["adapter_model.safetensors"],
    }
    assert payload["source_checkpoint_summary"] == payload["resume_checkpoint_summary"]
    assert payload["resume_base_step"] == 5
    assert payload["resume_steps"] == 1
    assert payload["effective_total_step"] == 6
    assert payload["metrics_line_count"] == 1
    assert payload["checkpoint_dirs"] == [str(output_dir / "checkpoint_000001")]
    assert payload["parameter_delta_abs_max"] == 0.25
    assert payload["parameter_delta_nonzero_count"] == 1
    assert payload["preview_artifacts"]["before"]["reward_mean"] == 0.6000000238418579
    assert payload["preview_artifacts"]["after"]["reward_mean"] == 0.6000000238418579


def test_sd3_bounded_trainer_smoke_rejects_resume_without_lora(tmp_path):
    import pytest

    from scripts import legacy_cli as cli

    resume_dir = tmp_path / "checkpoint_000005"
    resume_dir.mkdir()
    (resume_dir / "adapter_model.safetensors").write_text("resume", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot combine --resume-from with --disable-lora"):
        cli.main(
            [
                "sd3-bounded-trainer-smoke",
                "--adapter",
                "sd3_tempflow",
                "--model-path",
                "/models/sd35",
                "--repo-root",
                "/ref/tempflow",
                "--steps",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--resume-from",
                str(resume_dir),
                "--disable-lora",
            ]
        )


def test_sd3_bounded_trainer_smoke_requires_explicit_long_run(tmp_path):
    import pytest

    from scripts import legacy_cli as cli

    base_args = [
        "sd3-bounded-trainer-smoke",
        "--adapter",
        "sd3_tempflow",
        "--model-path",
        "/models/sd35",
        "--repo-root",
        "/ref/tempflow",
        "--output-dir",
        str(tmp_path / "out"),
    ]

    with pytest.raises(ValueError, match="Pass --allow-long-run"):
        cli.main([*base_args, "--steps", "6"])

    with pytest.raises(ValueError, match="capped at --steps <= 100"):
        cli.main([*base_args, "--steps", "101", "--allow-long-run"])


def test_image_preview_help_runs(capsys):
    import pytest

    from scripts import legacy_cli as cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["image-preview", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--adapter" in output
    assert "--model-path" in output
    assert "--repo-root" in output
    assert "--output-dir" in output


def test_sd3_bounded_trainer_smoke_help_runs(capsys):
    import pytest

    from scripts import legacy_cli as cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sd3-bounded-trainer-smoke", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--model-path" in output
    assert "--repo-root" in output
    assert "--output-dir" in output
    assert "--resume-from" in output
    assert "--allow-long-run" in output
    assert "--enable-rollout-cache" in output
