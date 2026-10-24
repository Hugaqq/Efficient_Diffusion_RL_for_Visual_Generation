"""H3 CPU structure/export tests. Explicit fake native engine, NOT GPU evidence.

Real ffmpeg encodes and fully decodes tiny fake AV outputs; no old videos or
HTTP/ComfyUI, no model downloads, no imported upstream model computation.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from visual_rl.models import h3
from visual_rl.algorithms.rollout.h3 import decode_media, run_h3_rollout


def assets_fixture(tmp_path):
    root = tmp_path / "snapshot"
    partition = root / "FL2VA"
    partition.mkdir(parents=True)
    (partition / "model_index.json").write_text(json.dumps({"_minimax_h3": {"partition": "fl2va"}}))
    for name in ("transformer", "text_encoder", "video_vae", "audio_vae"):
        folder = partition / name
        folder.mkdir()
        # Match the official video VAE layout observed by server at
        # 42ed227ee7df40d41602854ae760620d6eb651fe, not a fake flat duplicate.
        config = {}
        weights = folder
        if name == "video_vae":
            config = {"mode": "standalone", "source_path": "source",
                      "source_safetensors_path": "model.safetensors"}
            weights = folder / "source"
            weights.mkdir()
            (weights / "config.json").write_text('{"_class_name":"AutoencoderKLLegacy"}')
        (folder / "config.json").write_text(json.dumps(config))
        # Inventory-only bytes, deliberately not real model weights.
        (weights / "model.safetensors").write_bytes(b"CPU_TEST_ASSET_NOT_MODEL")
    for name in ("tokenizer", "processor"):
        (partition / name).mkdir()
        (partition / name / "config.json").write_text("{}")
    (partition / "audio_vae/modeling_audio.py").write_text("# CPU inventory fixture; never executed\n")
    return root


def tiny_request(name="tiny", **kwargs):
    return h3.H3Request(name, "  original prompt, with typo  ", 42009, width=32, height=32, frames=107, **kwargs)


def tiny_result(request):
    generator = torch.Generator().manual_seed(request.seed)
    return h3.H3Result(request,
        torch.randint(0, 256, (request.frames, request.height, request.width, 3), dtype=torch.uint8, generator=generator),
        torch.randn((2, round(request.frames / 24 * 40) * 800), generator=generator) * 0.01)


def test_request_result_joint_geometry_and_unsupported_modes(tmp_path):
    request = tiny_request()
    result = tiny_result(request)
    assert result.audio.shape == (2, 142400)
    for change in ({"frames":124+1}, {"width":33}, {"seed":True}, {"task":"ref2va"}, {"steps":False}):
        with pytest.raises(ValueError):
            replace(request, **change)
    with pytest.raises(ValueError, match="first"):
        replace(request, task="fl2va")
    with pytest.raises(ValueError, match="safe"):
        replace(request, sample_id="../old-video")
    for bad in (torch.zeros_like(result.audio), result.audio[:, :-1], torch.full_like(result.audio, float("nan"))):
        with pytest.raises(ValueError, match="audio"):
            replace(result, audio=bad)
    with pytest.raises(ValueError, match="rates"):
        replace(result, fps=30)
    with pytest.raises(ValueError, match="video"):
        replace(result, video=torch.zeros_like(result.video, dtype=torch.float32))


def test_asset_inventory_rejects_wrong_format_missing_shards_and_dmd(tmp_path):
    root = assets_fixture(tmp_path)
    report = h3.inspect_assets(root, "a" * 40)
    assert report["total_bytes"] > 0 and report["model_python_files"]
    assert report["files"]["FL2VA/processor/config.json"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(ValueError, match="revision"):
        h3.inspect_assets(root, "main")
    index = root / "FL2VA/model_index.json"
    old = index.read_text()
    index.write_text(json.dumps({"_minimax_h3":{"partition":"fl2va", "base_schedule":[1, 0.5, 0]}}))
    with pytest.raises(ValueError, match="RES"):
        h3.inspect_assets(root, "a" * 40)
    index.write_text(old)
    config = root / "FL2VA/text_encoder/config.json"
    config.write_text('{"quantization_config":{"quant_method":"nvfp4"}}')
    with pytest.raises(ValueError, match="quantized"):
        h3.inspect_assets(root, "a" * 40)
    config.write_text("{}")
    missing = root / "FL2VA/transformer/model.safetensors.index.json"
    missing.write_text('{"weight_map":{"weight":"missing.safetensors"}}')
    with pytest.raises(ValueError, match="shard"):
        h3.inspect_assets(root, "a" * 40)
    missing.unlink()
    processor = root / "FL2VA/processor/config.json"
    processor.unlink()
    with pytest.raises(ValueError, match="processor"):
        h3.inspect_assets(root, "a" * 40)


def test_official_video_vae_source_layout_and_strict_path_rejections(tmp_path):
    root = assets_fixture(tmp_path)
    revision = "42ed227ee7df40d41602854ae760620d6eb651fe"
    component = root / "FL2VA/video_vae"
    config_path = component / "config.json"
    config = json.loads(config_path.read_text())
    weight = component / "source/model.safetensors"
    report = h3.inspect_assets(root, revision)
    assert not list(component.glob("*.safetensors"))
    key = "FL2VA/video_vae/source/model.safetensors"
    assert report["files"][key] == {"bytes": weight.stat().st_size, "sha256": h3.sha256_file(weight)}
    assert "FL2VA/video_vae/source/config.json" in report["files"]
    assert report["total_bytes"] == sum(p.stat().st_size for p in (root / "FL2VA").rglob("*") if p.is_file())
    for field, invalid in (
        ("source_path", ("../source", "/tmp/source", "./source", "source/../source", "source//inner", "source/", "C:\\source", "", None)),
        ("source_safetensors_path", ("../model.safetensors", "/model.safetensors", "inner/model.safetensors", "..\\model.safetensors", "C:model.safetensors", "model.bin", "", None)),
        ("mode", ("other", None)),
    ):
        for value in invalid:
            config_path.write_text(json.dumps({**config, field: value}))
            with pytest.raises(ValueError, match="unsafe or unsupported"):
                h3.inspect_assets(root, revision)
    config_path.write_text(json.dumps(config))
    # Metadata-owned missing weight must fail even if an unrelated flat file
    # exists. No broad rglob-based acceptance or server copying workaround.
    weight.unlink()
    decoy = component / "decoy.safetensors"
    decoy.write_bytes(b"UNRELATED")
    with pytest.raises(ValueError, match="missing native video VAE source weight"):
        h3.inspect_assets(root, revision)
    weight.write_bytes(b"")
    with pytest.raises(ValueError, match="missing native video VAE source weight"):
        h3.inspect_assets(root, revision)
    weight.write_bytes(b"CPU_TEST_ASSET_NOT_MODEL")
    decoy.unlink()
    source_config = component / "source/config.json"
    source_config.write_text('{"quantization_config":{"quant_method":"nvfp4"}}')
    with pytest.raises(ValueError, match="quantized video VAE source"):
        h3.inspect_assets(root, revision)
    source_config.write_text("{}")
    unsupported = component / "source/pruned.safetensors"
    unsupported.write_bytes(b"UNSUPPORTED")
    with pytest.raises(ValueError, match="unsupported format"):
        h3.inspect_assets(root, revision)
    unsupported.unlink()
    shard_index = component / "source/model.safetensors.index.json"
    shard_index.write_text('{"weight_map":{"weight":"missing.safetensors"}}')
    with pytest.raises(ValueError, match="missing or unsafe native weight shard"):
        h3.inspect_assets(root, revision)
    shard_index.unlink()
    alias = component / "alias"
    alias.symlink_to(component / "source", target_is_directory=True)
    config_path.write_text(json.dumps({**config, "source_path": "alias"}))
    with pytest.raises(ValueError, match="unsafe native video VAE source directory"):
        h3.inspect_assets(root, revision)
    alias.unlink()
    config_path.write_text(json.dumps(config))
    # HF snapshots use file symlinks to cache blobs: retain those exact bytes,
    # while never accepting directory redirects in config-owned source_path.
    blob = tmp_path / "hf-cache-blob"
    weight.replace(blob)
    weight.symlink_to(blob)
    assert h3.inspect_assets(root, revision)["files"][key]["sha256"] == h3.sha256_file(blob)
    # Preserve the previous native flat layout too, without changing config at
    # runtime or manufacturing an extra copy of a real model weight.
    flat = assets_fixture(tmp_path / "flat-case")
    folder = flat / "FL2VA/video_vae"
    (folder / "config.json").write_text("{}")
    (folder / "source/model.safetensors").replace(folder / "model.safetensors")
    assert "FL2VA/video_vae/model.safetensors" in h3.inspect_assets(flat, revision)["files"]


def test_backend_identity_uses_real_import_path_and_full_sha(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    module = ModuleType("vllm_omni")
    module.__file__ = str(root / "vllm_omni/__init__.py")
    monkeypatch.setitem(sys.modules, "vllm_omni", module)
    values = {"sha": h3.BACKEND_SHA, "dirty": ""}
    def git(command, **kwargs):
        assert command[:3] == ["git", "-C", str(root)] and kwargs["text"]
        return values["sha"] if command[3] == "rev-parse" else values["dirty"]
    monkeypatch.setattr(h3.subprocess, "check_output", git)
    assert h3.verify_backend(root) == root
    values["dirty"] = " M native.py"
    with pytest.raises(RuntimeError, match="clean"):
        h3.verify_backend(root)
    values["dirty"], values["sha"] = "", "wrong"
    with pytest.raises(RuntimeError, match="clean"):
        h3.verify_backend(root)
    values["sha"] = h3.BACKEND_SHA
    module.__file__ = str(tmp_path / "shadow/__init__.py")
    with pytest.raises(RuntimeError, match="shadowed"):
        h3.verify_backend(root)


class FakeConfig:
    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(**kwargs)
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def set_tf_model_config(self, config):
        self.transformer_config = config
    def update_multimodal_support(self):
        self.metadata_bound = True


class FakeEngine:
    instances = []
    def __init__(self, config):
        self.config, self.requests, self.close_calls = config, [], 0
        self.instances.append(self)
    def add_req_and_wait_for_response(self, request):
        self.requests.append(request)
        sampling = request.sampling_params
        r = h3.H3Request(request.request_id, request.prompt["prompt"], sampling.seed,
                         width=sampling.width, height=sampling.height, frames=sampling.num_frames)
        result = tiny_result(r)
        def post(output, output_type):
            assert output_type == "pt"
            return {"video": output[0], "audio": output[1], "fps": 24, "audio_sample_rate": 32000}
        return SimpleNamespace(error=None, aborted=False, finished=True,
            post_process_func=post, output=(result.video[None], result.audio[None]),
            peak_memory_mb=0, stage_durations={"fake_cpu":0.01})
    def close(self):
        self.close_calls += 1


def fake_backend_modules(monkeypatch):
    for name in ("vllm_omni", "vllm_omni.diffusion", "vllm_omni.inputs"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name, values in {
        "vllm_omni.diffusion.data": {"OmniDiffusionConfig":FakeConfig, "TransformerConfig":SimpleNamespace(from_dict=lambda v:v)},
        "vllm_omni.diffusion.diffusion_engine": {"DiffusionEngine":FakeEngine},
        "vllm_omni.diffusion.request": {"OmniDiffusionRequest":SimpleNamespace},
        "vllm_omni.inputs.data": {"OmniDiffusionSamplingParams":SimpleNamespace},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)


def test_native_adapter_calls_pinned_joint_engine_without_training(monkeypatch, tmp_path):
    from PIL import Image
    root = assets_fixture(tmp_path)
    assets = h3.inspect_assets(root, "a" * 40)
    fake_backend_modules(monkeypatch)
    monkeypatch.setattr(h3, "verify_backend", lambda _: tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda:True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda:1)
    def forbidden(*_a, **_k):
        raise AssertionError("H3 must not create an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    kwargs = dict(model_root=root, model_revision="a"*40, backend_root=tmp_path)
    with pytest.raises(RuntimeError, match="review"):
        h3.H3InferenceAdapter.load(**kwargs, reviewed_model_code_sha256="unreviewed")
    with h3.H3InferenceAdapter.load(**kwargs, reviewed_model_code_sha256=assets["model_code_review_sha256"]) as adapter:
        engine = adapter._engine
        config = engine.config.kwargs
        assert config["distributed_executor_backend"] == "uni" and config["num_gpus"] == 1
        assert config["model_class_name"] == "MiniMaxH3Pipeline"
        assert config["step_execution"] is False and config["cache_backend"] == "none"
        assert config["diffusion_offload_config"]["components"] == ["dit", "text_encoder"]
        assert config["diffusion_attention_backend"] == "TORCH_SDPA"
        first, second = adapter.generate(tiny_request()), adapter.generate(tiny_request())
        assert torch.equal(first.video, second.video) and torch.equal(first.audio, second.audio)
        native = engine.requests[0]
        assert native.prompt["prompt"] == "  original prompt, with typo  "
        assert native.sampling_params.guidance_scale == 1.0 and native.sampling_params.num_outputs_per_prompt == 1
        assert native.sampling_params.extra_args == {"task":"t2va", "sampler":"res_multistep",
            "flow_shift":12.0, "audio_flow_shift":3.0, "preencode_mp4":False}
        image = tmp_path / "first.png"
        Image.new("RGB", (32, 32)).save(image)
        adapter.generate(tiny_request(task="fl2va", images=(image,)))
        assert len(engine.requests[-1].prompt["multi_modal_data"]["image"]) == 1
        assert engine.requests[-1].sampling_params.extra_args["task"] == "fl2va"
        monkeypatch.setattr(engine, "add_req_and_wait_for_response", lambda _:
                            SimpleNamespace(error="joint denoiser failed", aborted=False, finished=True))
        with pytest.raises(RuntimeError, match="joint denoiser"):
            adapter.generate(tiny_request())
    adapter.close()
    assert engine.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        adapter.generate(tiny_request())


class FakeAdapter:
    assets = {"declared_revision":"a"*40, "assets_sha256":"b"*64}
    def generate(self, request):
        return tiny_result(request)
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        pass


def test_real_ffmpeg_exports_and_full_decodes_three_fake_av_clips(tmp_path):
    requests = tuple(tiny_request(f"scene-{i}") for i in range(3))
    out = tmp_path / "three"
    receipts = run_h3_rollout(FakeAdapter(), requests, out, diagnostic=True)
    assert len(receipts) == 3 and (out / "ROLLOUT_COMPLETE.json").is_file()
    assert not (out / "SUCCESS.json").exists()  # only CLI after close/resources may publish overall SUCCESS
    for receipt in receipts:
        root = out / receipt["sample_id"]
        for name in ("video", "audio"):
            media = receipt["media"][name]
            assert media["sha256"] == h3.sha256_file(root / media["path"])
        assert receipt["media"]["native_audio_samples"] == 142400
        assert receipt["media"]["decoded_mp4"]["video_frames"] == 107
        assert {k: receipt["media"]["decoded_mp4"][k] for k in ("width", "height", "fps", "channels", "sample_rate")} == {
            "width":32, "height":32, "fps":24, "channels":2, "sample_rate":32000}
        assert receipt["media"]["full_decode_verified"]
        assert receipt["diagnostic"] and not receipt["rl_training_supported"]
        assert receipt["request"]["prompt"] == requests[0].prompt
    with pytest.raises(FileExistsError):
        run_h3_rollout(FakeAdapter(), requests, out, diagnostic=True)
    assert not (out / "FAILED.json").exists()  # refusal did not edit old artifacts
    broken = out / requests[0].sample_id / "video.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="decode failed"):
        decode_media(broken, width=32, height=32)


def test_rollout_failure_keeps_previous_receipts_and_no_success(tmp_path):
    class Broken(FakeAdapter):
        def generate(self, request):
            if request.sample_id == "scene-1":
                raise RuntimeError("native audio decode failed")
            return tiny_result(request)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="audio decode"):
        run_h3_rollout(Broken(), tuple(tiny_request(f"scene-{i}") for i in range(3)), out)
    assert not (out / "SUCCESS.json").exists()
    assert (out / "scene-0/receipt.json").is_file()
    assert json.loads((out / "FAILED.json").read_text())["completed_samples"] == ["scene-0"]


def test_configs_preserve_all_original_scene_text_seed_and_reference_geometry():
    from visual_rl.infer_h3 import read_requests
    root = Path(__file__).resolve().parents[1]
    scenes = read_requests(root / "configs/h3/three-scenes.json")
    originals = root.parent / "h3-scene-inference-20261024"
    for scene in scenes:
        original = json.loads((originals / f"{scene.sample_id}-request.json").read_text())["prompt"]
        assert scene.prompt == original["7"]["inputs"]["prompt"]
        assert scene.seed == original["8"]["inputs"]["noise_seed"]
        assert (scene.width,scene.height,scene.frames,scene.steps) == (1344,768,124,20)
        assert scene.task == "t2va" and scene.images == ()
    assert read_requests(root / "configs/h3/one-scene.json") == scenes[:1]
    diagnostic = read_requests(root / "configs/h3/one-scene.json", diagnostic=True)[0]
    assert (diagnostic.width, diagnostic.height, diagnostic.frames) == (384,224,107)
    assert diagnostic.prompt == scenes[0].prompt and diagnostic.seed == scenes[0].seed


def test_cli_preflight_is_not_generation_and_full_fake_run_refuses_repeat(monkeypatch, tmp_path):
    from visual_rl import infer_h3 as cli
    from scripts import launch_flash140 as utilities
    root = assets_fixture(tmp_path)
    config = tmp_path / "one.json"
    request = tiny_request()
    config.write_text(json.dumps({"schema_version":1,"description":"CPU fake leaves", "requests":[asdict(request)]}))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(utilities, "git_identity", lambda sha:sha)
    monkeypatch.setattr(utilities, "LOCK", tmp_path / "shared.lock")
    monkeypatch.setattr(cli, "verify_backend", lambda _:tmp_path)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda _:"CPU_TEST")
    monkeypatch.setattr(utilities, "resource_sample", lambda _:dict(gpu_free_gib=75,host_available_gib=150,
        disk_free_gib=500,torch_peak_reserved_gib=1,process_peak_rss_gib=1))
    called = []
    def load(**kwargs):
        called.append(kwargs)
        return FakeAdapter()
    monkeypatch.setattr(cli.H3InferenceAdapter, "load", load)
    output = tmp_path / "run"
    args = [str(config), "--expected-sha", "a"*40, "--backend-root", str(tmp_path),
            "--model-root", str(root), "--model-revision", "b"*40, "--output", str(output)]
    assert cli.main([*args,"--preflight"]) == 0
    assert not called and not output.exists()
    assert cli.main(args) == 0
    assert len(called) == 1 and (output / "SUCCESS.json").is_file()
    record = json.loads((tmp_path / "run.launch/result.json").read_text())
    assert record["status"] == "native_av_generated_and_decoded"
    assert record["samples"] == 1 and not record["rl5_gpu_passed"]
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(args)
    # Native close failures must not leave an overall SUCCESS marker even
    # though the already encoded/decoded media are preserved for diagnosis.
    class CloseFailure(FakeAdapter):
        def __exit__(self, *_args):
            raise RuntimeError("native close failed")
    monkeypatch.setattr(cli.H3InferenceAdapter, "load", lambda **_: CloseFailure())
    failed_output = tmp_path / "close-failure"
    failed_args = [*args[:-1], str(failed_output)]
    with pytest.raises(RuntimeError, match="native close failed"):
        cli.main(failed_args)
    assert (failed_output / "ROLLOUT_COMPLETE.json").is_file()
    assert not (failed_output / "SUCCESS.json").exists()
    assert json.loads((tmp_path / "close-failure.launch/result.json").read_text())["status"] == "blocked_or_failed"
