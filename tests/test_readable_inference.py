"""Additional CPU checks, not replacements for the original seven core gates.

All model/scorer/media leaves here are explicitly fake; no GPU/HTTP claim.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests import test_default_composition_e2e as e2e
from tests import test_v2_demo_default_composition as demo
from visual_rl.models.inference import WanGenerationRequest, WanInferenceAdapter
from visual_rl.runtime.composition import create_default_run_controller


def forbidden(*_args, **_kwargs):
    raise AssertionError("inference must not enter training/reward/LoRA preparation")


class FakeNativeWanPipeline:
    """A tiny frozen compute leaf standing in for Diffusers, never GPU evidence."""
    calls = []
    instances = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        obj = cls()
        cls.instances.append(obj)
        return obj

    def __init__(self):
        self.vae = torch.nn.Linear(1, 1)
        self.transformer = torch.nn.Linear(1, 1)
        self.components = {"vae": self.vae, "transformer": self.transformer}
        self.removed = 0
        self.placement = None
        self.requests = []

    def to(self, device):
        self.placement = str(device)
        return self

    def remove_all_hooks(self):
        self.removed += 1

    def __call__(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.requests.append(kwargs)
        return SimpleNamespace(frames=torch.rand(
            (1, kwargs["num_frames"], 3, kwargs["height"], kwargs["width"]),
            generator=kwargs["generator"],
        ))


def test_native_wan_inference_has_no_training_prerequisites(monkeypatch, tmp_path):
    import diffusers
    from visual_rl.models.implementations import common_diffusers
    from visual_rl.models.state.parameters import ParameterStateManager
    from visual_rl.runtime.reward_resources import DefaultRewardResourceFactory
    monkeypatch.setattr(diffusers, "WanPipeline", FakeNativeWanPipeline)
    monkeypatch.setattr(common_diffusers, "apply_peft_lora", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(ParameterStateManager, "__init__", forbidden)
    monkeypatch.setattr(DefaultRewardResourceFactory, "__call__", forbidden)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")
    request = WanGenerationRequest("  original prompt  ", 42, height=16, width=16, frames=5)
    with WanInferenceAdapter.load(model, device="cpu", cpu_offload=False) as adapter:
        leaf = adapter._pipeline
        assert all(not p.requires_grad for m in leaf.components.values() for p in m.parameters())
        assert all(not m.training for m in leaf.components.values())
        rng = torch.get_rng_state().clone()
        first, second = adapter.generate(request), adapter.generate(request)
        assert torch.equal(first.tensor, second.tensor)
        assert torch.equal(torch.get_rng_state(), rng)
        assert first.shape == (1, 5, 3, 16, 16) and first.layout == "BFCHW"
        assert leaf.requests[0]["prompt"] == "  original prompt  "
        assert leaf.requests[0]["guidance_scale"] == 4.5
        assert not first.tensor.requires_grad and first.tensor.device.type == "cpu"
        assert FakeNativeWanPipeline.calls[-1][1]["local_files_only"] is True
        assert not hasattr(adapter, "predict") and not hasattr(adapter, "trainable_parameters")
    adapter.close()
    assert leaf.removed == 1 and leaf.placement == "cpu"
    with pytest.raises(RuntimeError, match="closed"):
        adapter.generate(request)
    with pytest.raises(FileNotFoundError):
        WanInferenceAdapter.load(tmp_path / "missing")
    with pytest.raises(ValueError, match="frames"):
        replace(request, frames=6)
    # Generation/validation errors retain context-manager cleanup.
    with pytest.raises(ValueError, match="native Wan output"):
        with WanInferenceAdapter.load(model, device="cpu", cpu_offload=False) as bad:
            leaf = bad._pipeline
            monkeypatch.setattr(FakeNativeWanPipeline, "__call__", lambda *_a, **_kw:
                                SimpleNamespace(frames=torch.full((1, 5, 3, 16, 16), float("nan"))))
            bad.generate(request)
    assert leaf.removed == 1


@pytest.mark.parametrize("case", demo._DEMO_CASES, ids=lambda case: case.filename)
def test_model_algorithm_families_use_direct_dynamics(monkeypatch, tmp_path, case):
    from visual_rl.runtime.model_binding import DefaultPolicyRuntimePort
    assert not hasattr(DefaultPolicyRuntimePort, "transition")
    assert not hasattr(DefaultPolicyRuntimePort, "preprocess")
    assert not hasattr(DefaultPolicyRuntimePort, "initialize_latents")
    # Even an external old-style method must not intercept the algorithm's
    # sample/evaluate path. Exercise full original assertions for all6 demos,
    # including TempFlow branch topology, SD3 reference and World conditioners.
    monkeypatch.setattr(DefaultPolicyRuntimePort, "transition", forbidden, raising=False)
    demo.test_official_v2_demo_reaches_update_checkpoint_and_completed_restore(tmp_path, monkeypatch, case)


def test_rl5_typed_recipe_retains_real_hps_and_exact_target():
    from visual_rl.composition.config.compiler import compile_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe
    path = Path(__file__).resolve().parents[1] / "configs/v2/wan_rl5.yaml"
    recipe = compile_recipe_v2(load_source_recipe(path))
    training = recipe.training
    assert training.max_optimizer_steps == 5 and not training.flash_evaluation
    assert training.global_prompt_batch_size == 2
    assert training.collection.collection_rows_per_commit == 2
    assert training.collection.collection_commits == 2
    assert training.collection.collection_chunk_rows == 2
    assert training.policy_recompute.row_microbatch_size == 1
    assert training.update_safety.require_finite_gradients
    assert training.update_safety.require_nonzero_gradients
    raw = yaml.safe_load(path.read_text())
    assert raw["overrides"]["execution"]["group_size"] == 2
    assert raw["overrides"]["model"]["params"]["frames"] == 81
    assert raw["overrides"]["model"]["params"]["height"] == 480
    resource = raw["overrides"]["rewards"]["reward_general"]["params"]["resource"]
    assert resource["protocol"] == "flash_hpsv3_json"
    assert resource["semantic_factory_config"]["video_contract"] == {"frames":81,"height":480,"width":832,"top_k":24}
    assert raw["launch"]["checkpoint_every_optimizer_steps"] == 1
    assert raw["launch"]["resume_from"] is None


def test_cpu_rl5_five_changed_updates_odd_terminal_and_completed_restore(monkeypatch, tmp_path):
    from tests.test_flash_collection_runtime import setup
    from scripts.validate_wan_rl5 import observe_updates
    from scripts.launch_flash140 import inspect_checkpoint
    artifacts, code = setup(monkeypatch, tmp_path)
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    output = tmp_path / "out"
    output.mkdir()  # launcher owns observations; real controller owns checkpoints
    path = demo._write_smoke_config(tmp_path, case, output_dir=output,
                                    artifacts=artifacts, max_optimizer_steps=5)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["training"].update({
        "global_prompt_batch_size": 2,
        "collection": {"collection_rows_per_commit": 2, "collection_commits": 2, "collection_chunk_rows": 2},
        "policy_recompute": {"row_microbatch_size": 1, "transition_window_size": 1},
    })
    raw["overrides"]["execution"].update({"group_size": 2, "rollout": {
        "forward_microbatch_size": 1, "decode_microbatch_size": 1, "trajectory_storage_device": "cpu"}})
    path.write_text(yaml.safe_dump(raw))
    factory = demo._FakeRewardResourceFactory()
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                              reward_resource_factory=factory)
    history = observe_updates(controller, output)
    result = controller.run(path)
    assert result.committed_steps == 5 and (output / "SUCCESS").is_file()
    assert [v["committed_steps"] for v in history] == [1, 2, 3, 4, 5]
    assert len({v["trainable_sha256"] for v in history}) == 5
    assert factory.resources[0].score_calls == 6  # three4-row pools, two2-row chunks each
    for step in range(1, 6):
        inspected = inspect_checkpoint(output / f"checkpoints/step-{step}")
        assert inspected["trainable_sha256"] == history[step-1]["trainable_sha256"]
        assert inspected["last_metrics"]["gradient_norm_pre_clip"] > 0
        assert inspected["last_metrics"]["active_transition_count"] == 2
    final = inspect_checkpoint(result.authoritative_checkpoint)
    assert final["committed_steps"] == 5 and final["commit_cursor"] == 1
    assert final["pending_rows"] == 4 and final["evaluations_completed"] == []
    _, saved = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    pool = saved.component_state("data_plane").payload["runtime_state"]["pool"]
    assert pool["execution"].trajectory.media is None
    assert pool["execution"].trajectory.reward_media_released
    # Completed restore must recognize terminal5 without losing its pending
    # state, re-scoring, or trying to round the target to6.
    before_bytes = {p.relative_to(result.authoritative_checkpoint): p.read_bytes()
                    for p in result.authoritative_checkpoint.rglob("*") if p.is_file()}
    raw["launch"]["resume_from"] = str(result.authoritative_checkpoint)
    path.write_text(yaml.safe_dump(raw))
    restored_rewards = demo._FakeRewardResourceFactory()
    restored = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                            reward_resource_factory=restored_rewards)
    completed = restored.run(path)
    assert completed.committed_steps == 5
    assert restored_rewards.resources[0].score_calls == 0
    assert completed.authoritative_checkpoint == result.authoritative_checkpoint
    after_bytes = {p.relative_to(completed.authoritative_checkpoint): p.read_bytes()
                   for p in completed.authoritative_checkpoint.rglob("*") if p.is_file()}
    assert after_bytes == before_bytes  # exact terminal checkpoint, including pending tensors


def test_rl5_cli_fresh_preflight_run_and_no_automatic_repeat(monkeypatch, tmp_path):
    from tests.test_flash_collection_runtime import setup
    from scripts import validate_wan_rl5 as cli
    from scripts import launch_flash140 as utilities
    from scripts import flash_small40_repair as repair
    from visual_rl.runtime import composition
    artifacts, code = setup(monkeypatch, tmp_path)
    (code / "visual_rl").mkdir()
    (code / "visual_rl/identity.py").write_text("CPU_TEST = True\n")
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    output = tmp_path / "fresh"
    path = demo._write_smoke_config(tmp_path, case, output_dir=output,
                                    artifacts=artifacts, max_optimizer_steps=5)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["training"].update({"global_prompt_batch_size": 2,
        "collection": {"collection_rows_per_commit": 2, "collection_commits": 2, "collection_chunk_rows": 2},
        "policy_recompute": {"row_microbatch_size": 1, "transition_window_size": 1}})
    raw["overrides"]["execution"].update({"group_size": 2, "rollout": {
        "forward_microbatch_size": 1, "decode_microbatch_size": 1, "trajectory_storage_device": "cpu"}})
    path.write_text(yaml.safe_dump(raw))
    factory = demo._FakeRewardResourceFactory()
    def cpu_controller(**kwargs):
        assert kwargs["reward_resource_factory"] is factory
        return create_default_run_controller(code_root=kwargs["code_root"],
            model_loader=demo._FakeModelLoader(), reward_resource_factory=factory)
    sample = {"gpu_free_gib": 64, "host_available_gib": 88, "disk_free_gib": 128,
              "torch_peak_reserved_gib": 10, "process_peak_rss_gib": 10}
    monkeypatch.setattr(cli, "CONFIG", path)
    monkeypatch.setattr(cli, "ROOT", code)
    monkeypatch.setattr(cli, "LOCK", tmp_path / "shared.lock")
    monkeypatch.setattr(cli, "git_identity", lambda expected: expected)
    monkeypatch.setattr(cli, "configure_imports", lambda _: {"test": "fake CPU leaves"})
    monkeypatch.setattr(cli, "check_hps", lambda _: None)
    monkeypatch.setattr(cli, "resource_sample", lambda _: sample)
    monkeypatch.setattr(utilities, "resource_sample", lambda _: sample)
    monkeypatch.setattr(repair, "CacheHandoffRewardFactory", lambda: factory)
    monkeypatch.setattr(composition, "create_default_run_controller", cpu_controller)
    args = ["--expected-sha", "a" * 40]
    assert cli.main([*args, "--preflight"]) == 0
    assert not output.exists() and not factory.resources
    assert cli.main(args) == 0
    import json
    record = json.loads((output / "rl5-result.json").read_text())
    assert record["status"] == "verified_five_updates"
    assert len(record["updates"]) == record["checkpoint"]["committed_steps"] == 5
    assert (output / "rl5-resources.json").is_file()
    before = (output / "rl5-result.json").read_bytes()
    with pytest.raises(FileExistsError, match="fresh-only"):
        cli.main(args)
    assert (output / "rl5-result.json").read_bytes() == before
