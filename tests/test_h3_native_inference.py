"""CPU numerical/ownership tests; fake heavy model components, NOT GPU proof.

The reference kernels are extracted from fixed Diffusers text with attribution.
FFmpeg really encodes and fully decodes the tiny generated CPU test tensors.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch

from visual_rl.models import h3, h3_components as components
from visual_rl.algorithms.rollout.h3 import (
    decode_media, joint_euler_step, run_h3_rollout, sample_h3, sigma_schedule)

spec = importlib.util.spec_from_file_location("h3_pinned_reference", Path(__file__).with_name("h3_pinned_reference.py"))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def assets_fixture(tmp_path):
    root = tmp_path / "snapshot"
    partition = root / "FL2VA"
    partition.mkdir(parents=True)
    (partition / "model_index.json").write_text(json.dumps({"_minimax_h3": {"partition": "fl2va"}}))
    for name in ("transformer", "text_encoder", "video_vae", "audio_vae"):
        folder = partition / name
        folder.mkdir()
        config, weights = {}, folder
        if name == "video_vae":
            config = {"mode": "standalone", "source_path": "source", "source_safetensors_path": "model.safetensors"}
            weights = folder / "source"
            weights.mkdir()
            (weights / "config.json").write_text("{}")
        (folder / "config.json").write_text(json.dumps(config))
        (weights / "model.safetensors").write_bytes(b"CPU_TEST_ASSET_NOT_MODEL")
    for name in ("tokenizer", "processor"):
        (partition / name).mkdir()
        (partition / name / "config.json").write_text("{}")
    (partition / "audio_vae/modeling_audio.py").write_text("raise AssertionError('checkpoint code must never execute')\n")
    return root


def converted_fixture(tmp_path, assets):
    root = tmp_path / "derived"
    for name in ("transformer", "vae", "audio_vae"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
        (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"CPU_FAKE_DERIVED_WEIGHTS")
    files = {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": h3.sha256_file(p)}
             for p in root.rglob("*") if p.is_file()}
    receipt = {"diffusers_commit": h3.DIFFUSERS_SHA, "converter_sha256": components.CONVERTER_SHA256,
               "raw_assets_sha256": assets["assets_sha256"], "files": files}
    (root / "conversion.json").write_text(json.dumps(receipt))
    return root


def tiny_request(name="tiny", **kwargs):
    return h3.H3Request(name, "  original prompt, with typo  ", 42009, width=32, height=32, frames=107, **kwargs)


def tiny_result(request):
    generator = torch.Generator().manual_seed(request.seed)
    return h3.H3Result(request,
        torch.randint(0, 256, (request.frames, request.height, request.width, 3), dtype=torch.uint8, generator=generator),
        torch.randn((2, round(request.frames / 24 * 40) * 800), generator=generator) * 0.01)


def test_request_result_joint_geometry_and_unsupported_modes():
    request, result = tiny_request(), tiny_result(tiny_request())
    assert result.audio.shape == (2, 142400)
    for change in ({"frames":125}, {"width":33}, {"seed":True}, {"task":"ref2va"}, {"steps":False}, {"steps":1}):
        with pytest.raises(ValueError):
            replace(request, **change)
    with pytest.raises(ValueError, match="text-only"):
        replace(request, task="fl2va")
    with pytest.raises(ValueError, match="safe"):
        replace(request, sample_id="../old-video")
    for bad in (torch.zeros_like(result.audio), result.audio[:, :-1], torch.full_like(result.audio, float("nan"))):
        with pytest.raises(ValueError, match="audio"):
            replace(result, audio=bad)
    with pytest.raises(ValueError, match="rates"):
        replace(result, fps=30)
    with pytest.raises(ValueError, match="video"):
        replace(result, video=result.video.float())


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
    with pytest.raises(ValueError, match="schedule"):
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
    (root / "FL2VA/processor/config.json").unlink()
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
    blob = tmp_path / "hf-cache-blob"
    weight.replace(blob)
    weight.symlink_to(blob)
    assert h3.inspect_assets(root, revision)["files"][key]["sha256"] == h3.sha256_file(blob)
    flat = assets_fixture(tmp_path / "flat-case")
    folder = flat / "FL2VA/video_vae"
    (folder / "config.json").write_text("{}")
    (folder / "source/model.safetensors").replace(folder / "model.safetensors")
    assert "FL2VA/video_vae/model.safetensors" in h3.inspect_assets(flat, revision)["files"]


def test_converted_identity_and_bytes_are_independently_checked(tmp_path):
    raw = assets_fixture(tmp_path)
    assets = h3.inspect_assets(raw, "a" * 40)
    converted = converted_fixture(tmp_path, assets)
    assert components.inspect_converted(converted, assets)["diffusers_commit"] == h3.DIFFUSERS_SHA
    with pytest.raises(ValueError, match="identity"):
        components.inspect_converted(converted, {**assets, "assets_sha256":"wrong"})
    weight = converted / "transformer/diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        components.inspect_converted(converted, assets)


def test_adapter_load_is_metadata_only_and_rejects_offline_or_conversion_mismatch(monkeypatch,tmp_path):
    raw = assets_fixture(tmp_path)
    assets = h3.inspect_assets(raw,"a"*40)
    derived = converted_fixture(tmp_path,assets)
    audits = []
    monkeypatch.setattr(components,"audit_runtime",lambda:audits.append("audited"))
    monkeypatch.setattr(components,"release_memory",lambda:None)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:1)
    monkeypatch.delenv("HF_HUB_OFFLINE",raising=False)
    with pytest.raises(RuntimeError,match="offline"):
        h3.H3InferenceAdapter.load(model_root=raw,converted_root=derived,model_revision="a"*40)
    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    with h3.H3InferenceAdapter.load(model_root=raw,converted_root=derived,model_revision="a"*40) as adapter:
        assert adapter.phase is None and adapter.transformer is None and not adapter.closed
        assert not hasattr(adapter,"optimizer") and not hasattr(adapter,"_engine")
    adapter.close()
    assert adapter.closed and audits == ["audited"]
    with pytest.raises(RuntimeError,match="closed"):
        adapter.prepare_latents(tiny_request(),torch.ones(1,3,8))
    (derived / "transformer/diffusion_pytorch_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash"):
        h3.H3InferenceAdapter.load(model_root=raw,converted_root=derived,model_revision="a"*40)


def test_runtime_refuses_installed_and_live_engines_and_checks_source(monkeypatch, tmp_path):
    monkeypatch.setattr(components.importlib.util, "find_spec", lambda _:None)
    root = tmp_path / "diffusers"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "pinned.py").write_text("# pinned component")
    fake = SimpleNamespace(__file__=str(root / "__init__.py"), __version__="CPU_TEST")
    monkeypatch.setattr(components, "COMPONENT_HASHES", {"pinned.py":h3.sha256_file(root / "pinned.py")})
    original = components.importlib.import_module
    monkeypatch.setattr(components.importlib, "import_module", lambda n: fake if n == "diffusers" else original(n))
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    assert components.audit_runtime()["diffusers"]["file"] == fake.__file__
    (root / "pinned.py").write_text("# wrong source")
    with pytest.raises(RuntimeError, match="source mismatch"):
        components.audit_runtime()
    monkeypatch.setattr(components.importlib.util, "find_spec", lambda n: object() if n == "vllm" else None)
    with pytest.raises(RuntimeError, match="without vllm"):
        components.audit_runtime()
    monkeypatch.setattr(components.importlib.util, "find_spec", lambda _:None)
    monkeypatch.setitem(sys.modules, "sglang.hidden", object())
    with pytest.raises(RuntimeError, match="without sglang"):
        components.audit_runtime()


def test_joint_schedule_and_step_match_pinned_scheduler_exactly():
    for points in (2, 20, 40):
        schedulers = {name:reference.MiniMaxH3Scheduler(shift) for name, shift in (("video",12.0),("audio",3.0))}
        for name, shift in (("video",12.0),("audio",3.0)):
            schedulers[name].set_timesteps(points)
            assert torch.equal(sigma_schedule(points, shift), schedulers[name].sigmas)
            assert len(schedulers[name].timesteps) == points - 1
        generator = torch.Generator().manual_seed(17)
        state = {"video":torch.randn((1,5,96), generator=generator), "audio":torch.randn((1,8,32), generator=generator)}
        for i in range(points - 1):
            velocity = {name:torch.randn(x.shape, generator=generator, dtype=torch.bfloat16) for name,x in state.items()}
            sigma = {n:s.sigmas[i] for n,s in schedulers.items()}
            following = {n:s.sigmas[i+1] for n,s in schedulers.items()}
            expected = {n:s.step(velocity[n].float(), s.timesteps[i], state[n], return_dict=False)[0] for n,s in schedulers.items()}
            actual = joint_euler_step(state, velocity, sigma, following)
            for name in actual:
                assert actual[name].dtype == torch.float32
                assert torch.equal(actual[name], expected[name])
            state = actual
    # Positive data-ward velocity must increase x, not use Wan's opposite sign.
    state = {"video":torch.ones(1,2,96), "audio":torch.ones(1,4,32)}
    v = {n:torch.ones_like(x) for n,x in state.items()}
    sigmas = {n:torch.tensor(1.) for n in state}
    following = {n:torch.tensor(0.) for n in state}
    assert all(torch.equal(x, torch.full_like(x,2.)) for x in joint_euler_step(state,v,sigmas,following).values())
    for bad in ({"video":v["video"]}, {**v,"audio":torch.full_like(v["audio"],float("nan"))}, {**v,"audio":v["audio"][:,:,:1]}):
        with pytest.raises(ValueError):
            joint_euler_step(state,bad,sigmas,following)
    with pytest.raises(ValueError):
        joint_euler_step({n:x.bfloat16() for n,x in state.items()},v,sigmas,following)


def test_reference_conversion_preserves_qkv_heads_and_gate_value_order():
    raw = torch.arange(12*3).reshape(12,3).float()
    reordered = reference.reorder_interleaved_qkv(raw, 2, 2)
    outputs = reference.convert_transformer_key("blocks.0.attn.qkv_proj.weight",reordered,
                                               {"num_attention_heads":2,"attention_head_dim":2})
    assert [n for n,_ in outputs] == [f"transformer_blocks.0.attn.to_{c}.weight" for c in "qkv"]
    assert torch.equal(outputs[0][1], torch.cat([raw[:2],raw[6:8]]))
    assert torch.equal(outputs[1][1], torch.cat([raw[2:4],raw[8:10]]))
    assert torch.equal(outputs[2][1], torch.cat([raw[4:6],raw[10:12]]))
    fused = torch.arange(24).reshape(8,3)
    name, value = reference.convert_transformer_key("blocks.0.mlp.fc1.weight", fused, {})[0]
    assert name == "transformer_blocks.0.ff.net.0.proj.weight"
    assert torch.equal(value, torch.cat([fused[4:],fused[:4]]))
    assert reference.convert_transformer_key("rope.inv_freq",torch.ones(3),{}) == []


def install_geometry(monkeypatch):
    module = ModuleType("diffusers.modular_pipelines.minimax_h3.before_denoise")
    module.MiniMaxH3PrepareLayoutStep = SimpleNamespace(build_packed_sequence=reference.build_packed_sequence)
    module.MiniMaxH3SetTimestepsStep = SimpleNamespace(build_row_timesteps=reference.build_row_timesteps)
    module.patchify_video_latents = reference.patchify_video_latents
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_packed_geometry_noise_order_prediction_and_component_release(monkeypatch, tmp_path):
    install_geometry(monkeypatch)
    raw = assets_fixture(tmp_path)
    assets = h3.inspect_assets(raw,"a"*40)
    converted = converted_fixture(tmp_path, assets)
    adapter = h3.H3InferenceAdapter(raw, converted, assets, device="cpu")
    request = tiny_request(steps=4)
    prompt = torch.randn(1,7,12,dtype=torch.bfloat16)
    state, layout, geometry = adapter.prepare_latents(request,prompt)
    again, _, _ = adapter.prepare_latents(request,prompt)
    assert geometry == (32,2,2,178)
    assert state["video"].shape == (1,32,96) and state["audio"].shape == (1,356,32)
    assert layout["position_ids"].dtype == torch.float64
    assert torch.equal(state["video"],again["video"]) and torch.equal(state["audio"],again["audio"])
    generator = torch.Generator().manual_seed(request.seed)
    expected_video = reference.patchify_video_latents(torch.randn((1,24,32,2,2),generator=generator),(1,2,2))[None]
    expected_audio = torch.randn((1,356,32),generator=generator)
    assert torch.equal(state["video"],expected_video) and torch.equal(state["audio"],expected_audio)
    assert set(layout["token_tags"].tolist()) == {0,1,2}
    # Exact packed order: text, channel-major audio, video; never average stereo.
    assert layout["audio_indices"].tolist() == list(range(7,363))
    assert layout["video_indices"].tolist() == list(range(363,395))
    calls, refs = [], []
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(1,1,dtype=torch.float32)
            self.other = torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16))
            self.transformer_blocks = torch.nn.ModuleList()
            self.token_refiner = SimpleNamespace(refiner_blocks=[])
            refs.append(weakref.ref(self))
        @classmethod
        def from_pretrained(cls,path,**kwargs):
            assert kwargs["local_files_only"] and kwargs["low_cpu_mem_usage"]
            assert path == converted / "transformer"
            return cls()
        def enable_group_offload(self,**kwargs):
            assert kwargs["num_blocks_per_group"] == 1 and not kwargs["use_stream"]
        def forward(self,**kwargs):
            assert not torch.is_grad_enabled()
            calls.append(kwargs)
            unique, indices = reference.build_row_timesteps(layout["video_indices"],layout["audio_indices"],0,0,7,0.2,0.4,0.999,1.)
            assert torch.equal(kwargs["timestep"],unique) and torch.equal(kwargs["timestep_indices"],indices)
            return torch.ones_like(kwargs["hidden_states"],dtype=torch.bfloat16),torch.ones_like(kwargs["audio_hidden_states"],dtype=torch.bfloat16)
    import diffusers
    monkeypatch.setattr(diffusers,"MiniMaxH3Transformer3DModel",Transformer,raising=False)
    with pytest.raises(RuntimeError,match="active"):
        adapter.predict(state,prompt,layout,0.2,0.4)
    with torch.inference_mode(), adapter.denoising():
        velocity = adapter.predict(state,prompt,layout,0.2,0.4)
        assert all(x.dtype == torch.float32 for x in velocity.values()) and len(calls) == 1
        with pytest.raises(RuntimeError,match="overlapping"):
            with adapter.denoising():
                pass
    gc.collect()
    assert adapter.phase is None and adapter.transformer is None and refs[0]() is None
    adapter.close()
    with pytest.raises(RuntimeError,match="closed"):
        with adapter.denoising():
            pass


def test_encoder_and_decoders_execute_staged_ports_without_pipeline_or_remote_code(monkeypatch,tmp_path):
    import accelerate
    import diffusers
    import transformers
    install_geometry(monkeypatch)
    raw = assets_fixture(tmp_path)
    assets = h3.inspect_assets(raw,"a"*40)
    derived = converted_fixture(tmp_path,assets)
    adapter = h3.H3InferenceAdapter(raw,derived,assets,device="cpu")
    request = tiny_request(steps=3)
    calls, refs = [], []
    class Backbone(torch.nn.Module):
        def forward(self,**kwargs):
            assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is True
            assert torch.equal(kwargs["attention_mask"],torch.ones_like(kwargs["input_ids"]))
            assert torch.equal(kwargs["mm_token_type_ids"],torch.zeros_like(kwargs["input_ids"]))
            assert not torch.is_grad_enabled()
            return SimpleNamespace(hidden_states=tuple(torch.full((1,7,12),float(i),dtype=torch.bfloat16) for i in range(65)))
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(12,8)
            self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64))
            refs.append(weakref.ref(self))
        @classmethod
        def from_pretrained(cls,path,**kwargs):
            assert path == raw / "FL2VA/text_encoder"
            assert kwargs["trust_remote_code"] is False and kwargs["local_files_only"]
            calls.append("encoder")
            return cls()
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,path,**kwargs):
            assert path == raw / "FL2VA/tokenizer" and kwargs == {"local_files_only":True}
            return cls()
        def __call__(self,text,**kwargs):
            assert text == request.prompt and kwargs == {"add_special_tokens":False}
            return {"input_ids":list(range(7))}
    monkeypatch.setattr(transformers,"Qwen3VLForConditionalGeneration",Encoder)
    monkeypatch.setattr(transformers,"Qwen2TokenizerFast",Tokenizer)
    monkeypatch.setattr(accelerate,"cpu_offload",lambda model,**kwargs:
        calls.append(("offload",type(model).__name__,kwargs["offload_buffers"])))
    def forbidden(*_,**__):
        raise AssertionError("inference must not construct optimizer or call a pipeline")
    monkeypatch.setattr(torch.optim,"AdamW",forbidden)
    prompt = adapter.encode(request)
    assert torch.equal(prompt,torch.full_like(prompt,50)) and prompt.device.type == "cpu"
    assert adapter.phase is None and refs[0]() is None
    assert calls == ["encoder",("offload","Backbone",True)]
    state,layout,geometry = adapter.prepare_latents(request,prompt)
    expected_video = state["video"].reshape(1,32,1,1,24,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,24,32,2,2)
    expected_audio = state["audio"].reshape(2,178,32).permute(0,2,1)
    class VAE(torch.nn.Module):
        name = "vae"
        @classmethod
        def from_pretrained(cls,path,**kwargs):
            assert path == derived / cls.name and kwargs["torch_dtype"] == torch.float32
            model = cls()
            count = 24 if cls.name == "vae" else 32
            model.config = SimpleNamespace(latents_mean=[2.]*count,latents_std=[3.]*count)
            refs.append(weakref.ref(model))
            calls.append(cls.name)
            return model
        def enable_tiling(self):
            calls.append("tiling")
        def decode(self,latents,return_dict):
            assert adapter.phase == "video_decoder" and return_dict is False
            torch.testing.assert_close(latents,expected_video*3+2,rtol=0,atol=0)
            return (torch.zeros(1,3,request.frames,request.height,request.width),)
    class AudioVAE(VAE):
        name = "audio_vae"
        def decode(self,latents,return_dict):
            assert adapter.phase == "audio_decoder" and return_dict is False
            torch.testing.assert_close(latents,expected_audio*3+2,rtol=0,atol=0)
            return (torch.stack([torch.full((1,142400),0.125),torch.full((1,142400),-0.25)]),)
    monkeypatch.setattr(diffusers,"AutoencoderKLMiniMaxH3",VAE,raising=False)
    monkeypatch.setattr(diffusers,"AutoencoderKLMiniMaxH3Audio",AudioVAE,raising=False)
    result = adapter.decode(request,state,geometry)
    assert result.video[0,0,0].tolist() == [124,116,104]
    assert torch.equal(result.audio[0],torch.full((142400,),0.125))
    assert torch.equal(result.audio[1],torch.full((142400,),-0.25))
    assert calls[-3:] == ["vae","tiling","audio_vae"] and all(ref() is None for ref in refs)
    assert adapter.phase is None
    adapter.close()


def test_feed_forward_chunking_keeps_parameters_and_remainder():
    def ff():
        module = torch.nn.Module()
        module.net = torch.nn.ModuleList([torch.nn.Linear(4,8),torch.nn.SiLU(),torch.nn.Linear(8,4)])
        return module
    block = torch.nn.Module()
    block.ff = ff()
    transformer = SimpleNamespace(transformer_blocks=[block],token_refiner=SimpleNamespace(refiner_blocks=[]))
    ids = [id(p) for p in block.ff.parameters()]
    x = torch.randn(2,13,4)
    expected = x
    for layer in block.ff.net:
        expected = layer(expected)
    components.chunk_feed_forwards(transformer,max_tokens=5)
    assert [id(p) for p in block.ff.parameters()] == ids
    torch.testing.assert_close(block.ff(x),expected,rtol=1e-6,atol=1e-7)


class FakeAdapter:
    """Only fake heavy compute; the real VisualRL loop still executes each step."""
    assets = {"declared_revision":"a"*40,"assets_sha256":"b"*64}
    def __init__(self):
        self.calls = []
        self.active = False
    def encode(self,request):
        assert not self.active
        self.calls.append(("encode",request.prompt))
        return torch.ones(1,3,8)
    def prepare_latents(self,request,prompt):
        gen = torch.Generator().manual_seed(request.seed)
        return {"video":torch.randn((1,2,96),generator=gen),"audio":torch.randn((1,4,32),generator=gen)}, {}, None
    @contextmanager
    def denoising(self):
        self.active = True
        self.calls.append(("denoiser_start",))
        try:
            yield
        finally:
            self.active = False
            self.calls.append(("denoiser_end",))
    def predict(self,state,prompt,layout,video_time,audio_time):
        assert self.active and not torch.is_grad_enabled()
        self.calls.append(("predict",float(video_time),float(audio_time)))
        return {name:torch.ones_like(x) for name,x in state.items()}
    def decode(self,request,state,geometry):
        assert not self.active
        self.calls.append(("decode",))
        return tiny_result(request)
    def __enter__(self):
        return self
    def __exit__(self,*_):
        pass


def test_visualrl_loop_owns_every_joint_forward_and_failure_cleanup():
    adapter = FakeAdapter()
    result = sample_h3(adapter,tiny_request(steps=5))
    assert result.metrics["model_evaluations"] == 4
    assert [c[0] for c in adapter.calls] == ["encode","denoiser_start",*("predict",)*4,"denoiser_end","decode"]
    clocks = [c[1:] for c in adapter.calls if c[0] == "predict"]
    assert clocks[0] == (0.,0.) and clocks[-1][0] != clocks[-1][1]
    class Broken(FakeAdapter):
        def predict(self,*_):
            raise RuntimeError("prediction failed")
    broken = Broken()
    with pytest.raises(RuntimeError,match="prediction failed"):
        sample_h3(broken,tiny_request())
    assert not broken.active and broken.calls[-1] == ("denoiser_end",)


def test_real_ffmpeg_exports_and_full_decodes_three_fake_av_clips(tmp_path):
    requests = tuple(tiny_request(f"scene-{i}",steps=3) for i in range(3))
    out = tmp_path / "three"
    receipts = run_h3_rollout(FakeAdapter(),requests,out,diagnostic=True)
    assert len(receipts) == 3 and (out / "ROLLOUT_COMPLETE.json").is_file()
    assert not (out / "SUCCESS.json").exists()
    for receipt in receipts:
        root = out / receipt["sample_id"]
        for name in ("video","audio"):
            media = receipt["media"][name]
            assert media["sha256"] == h3.sha256_file(root / media["path"])
        assert receipt["media"]["native_audio_samples"] == 142400
        assert receipt["media"]["decoded_mp4"]["video_frames"] == 107
        assert {k:receipt["media"]["decoded_mp4"][k] for k in ("width","height","fps","channels","sample_rate")} == {
            "width":32,"height":32,"fps":24,"channels":2,"sample_rate":32000}
        assert receipt["media"]["full_decode_verified"] and receipt["diagnostic"] and not receipt["rl_training_supported"]
        assert receipt["request"]["prompt"] == requests[0].prompt
        assert receipt["implementation"] == "visualrl-pytorch-joint-loop" and receipt["model_evaluations"] == 2
    with pytest.raises(FileExistsError):
        run_h3_rollout(FakeAdapter(),requests,out,diagnostic=True)
    assert not (out / "FAILED.json").exists()
    broken = out / requests[0].sample_id / "video.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError,match="decode failed"):
        decode_media(broken,width=32,height=32)


def test_rollout_failure_keeps_previous_receipts_and_no_success(tmp_path):
    class Broken(FakeAdapter):
        def decode(self,request,*_):
            if request.sample_id == "scene-1":
                raise RuntimeError("native audio decode failed")
            return tiny_result(request)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError,match="audio decode"):
        run_h3_rollout(Broken(),tuple(tiny_request(f"scene-{i}",steps=3) for i in range(3)),out)
    assert not (out / "SUCCESS.json").exists() and (out / "scene-0/receipt.json").is_file()
    assert json.loads((out / "FAILED.json").read_text())["completed_samples"] == ["scene-0"]


def test_configs_preserve_original_scene_text_seed_geometry_and_explicit_new_baseline():
    from visual_rl.infer_h3 import read_requests
    root = Path(__file__).resolve().parents[1]
    scenes = read_requests(root / "configs/h3/three-scenes.json")
    for scene in scenes:
        original = json.loads((root.parent / "h3-scene-inference-20261024" / f"{scene.sample_id}-request.json").read_text())["prompt"]
        assert scene.prompt == original["7"]["inputs"]["prompt"] and scene.seed == original["8"]["inputs"]["noise_seed"]
        assert (scene.width,scene.height,scene.frames,scene.steps) == (1344,768,124,40)
        assert scene.task == "t2va" and scene.images == ()
    assert read_requests(root / "configs/h3/one-scene.json") == scenes[:1]
    diagnostic = read_requests(root / "configs/h3/one-scene.json",diagnostic=True)[0]
    assert (diagnostic.width,diagnostic.height,diagnostic.frames) == (384,224,107)
    assert diagnostic.prompt == scenes[0].prompt and diagnostic.seed == scenes[0].seed


def test_preparation_streams_to_separate_atomic_derived_directory(monkeypatch,tmp_path):
    from scripts import prepare_h3_torch_assets as preparation
    raw = assets_fixture(tmp_path)
    assets = h3.inspect_assets(raw,"a"*40)
    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    monkeypatch.setattr(preparation,"audit_runtime",lambda:{})
    monkeypatch.setattr(preparation.shutil,"disk_usage",lambda _:SimpleNamespace(free=2**40))
    original_read = Path.read_text
    monkeypatch.setattr(Path,"read_text",lambda p,*a,**k:"MemAvailable: 104857600 kB\n" if str(p) == "/proc/meminfo" else original_read(p,*a,**k))
    calls = []
    def weights(name,output):
        folder = Path(output)
        folder.mkdir(parents=True,exist_ok=True)
        (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"CPU_FAKE_CONVERSION")
        (folder / "config.json").write_text("{}")
        calls.append(name)
    converter = SimpleNamespace(
        MINIMAX_H3_TRANSFORMER_CONFIG={},MINIMAX_H3_VIDEO_VAE_CONFIG={},
        convert_transformer=lambda raw,out,config,shard:weights("transformer",out),
        write_transformer_config=lambda *_:None,
        convert_video_vae=lambda raw,out,config,version,shard:weights("vae",out),
        convert_audio_vae=lambda raw,out,version:weights("audio_vae",out),
        write_scheduler_configs=lambda *_:None)
    monkeypatch.setattr(preparation,"load_converter",lambda _:converter)
    output = tmp_path / "converted"
    receipt = preparation.convert(raw,output,"a"*40,tmp_path / "upstream")
    assert calls == ["transformer","vae","audio_vae"] and len(receipt["files"]) == 6
    assert not output.with_name("converted.partial").exists()
    assert components.inspect_converted(output,assets) == json.loads((output / "conversion.json").read_text())
    assert h3.inspect_assets(raw,"a"*40) == assets  # original raw bytes/code never changed
    with pytest.raises(FileExistsError,match="preserve"):
        preparation.convert(raw,output,"a"*40,tmp_path)
    def fail(*_):
        raise RuntimeError("conversion mapping failed")
    converter.convert_video_vae = fail
    failed = tmp_path / "bad-converted"
    with pytest.raises(RuntimeError,match="mapping failed"):
        preparation.convert(raw,failed,"a"*40,tmp_path)
    assert not failed.exists() and (tmp_path / "bad-converted.partial/FAILED.json").is_file()


def test_converter_requires_fixed_clean_source_and_reviewed_script_hash(monkeypatch,tmp_path):
    from scripts import prepare_h3_torch_assets as preparation
    folder = tmp_path / "scripts"
    folder.mkdir()
    script = folder / "convert_minimax_h3_to_diffusers.py"
    script.write_text("VALUE = 'CPU_TEST_ONLY'\n")
    state = {"sha":h3.DIFFUSERS_SHA,"dirty":""}
    monkeypatch.setattr(preparation.subprocess,"check_output",lambda args,**kwargs:state["sha"] if "rev-parse" in args else state["dirty"])
    with pytest.raises(RuntimeError,match="unreviewed"):
        preparation.load_converter(tmp_path)
    monkeypatch.setattr(preparation,"CONVERTER_SHA256",h3.sha256_file(script))
    assert preparation.load_converter(tmp_path).VALUE == "CPU_TEST_ONLY"
    state["dirty"] = "M weights.py"
    with pytest.raises(RuntimeError,match="fixed clean"):
        preparation.load_converter(tmp_path)
    state.update(dirty="",sha="b"*40)
    with pytest.raises(RuntimeError,match="fixed clean"):
        preparation.load_converter(tmp_path)


def test_active_sources_and_requirements_have_no_engine_import_or_pipeline_call():
    import ast
    root = Path(__file__).resolve().parents[1]
    for relative in ("visual_rl/models/h3.py","visual_rl/models/h3_components.py",
                     "visual_rl/algorithms/rollout/h3.py","visual_rl/infer_h3.py"):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert not any(alias.name.split(".")[0] in components.FORBIDDEN_PACKAGES for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in components.FORBIDDEN_PACKAGES
    assert not hasattr(h3.H3InferenceAdapter,"generate")  # not a hidden complete-pipeline facade
    requirements = (root / "configs/h3/requirements-torch.txt").read_text()
    assert h3.DIFFUSERS_SHA in requirements and "transformers==5.19.0" in requirements
    assert not any(package in requirements.lower() for package in components.FORBIDDEN_PACKAGES)


def test_stage_admission_is_largest_live_component_not_asset_sum():
    sample = {"host_available_gib":100,"gpu_free_gib":50}
    assert components.stage_budget(62*1024**3,5*1024**3,sample) == 91
    with pytest.raises(RuntimeError,match="stage admission"):
        components.stage_budget(62*1024**3,5*1024**3,{**sample,"host_available_gib":90})
    with pytest.raises(RuntimeError,match="stage admission"):
        components.stage_budget(62*1024**3,5*1024**3,{**sample,"gpu_free_gib":31})


def test_cli_preflight_is_not_generation_and_full_fake_run_refuses_repeat(monkeypatch,tmp_path):
    from visual_rl import infer_h3 as cli
    from scripts import launch_flash140 as utilities
    root = assets_fixture(tmp_path)
    assets = h3.inspect_assets(root,"b"*40)
    derived = converted_fixture(tmp_path,assets)
    config = tmp_path / "one.json"
    config.write_text(json.dumps({"schema_version":1,"description":"CPU fake leaves","requests":[asdict(tiny_request(steps=3))]}))
    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    monkeypatch.setattr(utilities,"git_identity",lambda sha:sha)
    monkeypatch.setattr(utilities,"LOCK",tmp_path / "shared.lock")
    monkeypatch.setattr(cli,"audit_runtime",lambda:{"torch":{"file":torch.__file__}})
    monkeypatch.setattr(utilities,"resource_sample",lambda _:dict(gpu_free_gib=75,host_available_gib=150,
        disk_free_gib=500,torch_peak_reserved_gib=1,process_peak_rss_gib=1))
    called = []
    def load(**kwargs):
        called.append(kwargs)
        return FakeAdapter()
    monkeypatch.setattr(cli.H3InferenceAdapter,"load",load)
    output = tmp_path / "run"
    args = ["--diagnostic",str(config),"--expected-sha","a"*40,"--converted-root",str(derived),"--model-root",str(root),
            "--model-revision","b"*40,"--output",str(output)]
    with pytest.raises(ValueError,match="normal acceptance"):
        cli.main(args[1:])
    assert cli.main([*args,"--preflight"]) == 0 and not called and not output.exists()
    assert cli.main(args) == 0 and len(called) == 1 and (output / "SUCCESS.json").is_file()
    record = json.loads((tmp_path / "run.launch/result.json").read_text())
    assert record["status"] == "native_av_generated_and_decoded" and record["samples"] == 1
    assert record["no_serving_engines"] and record["diagnostic"] and not record["rl5_gpu_passed"]
    with pytest.raises(FileExistsError,match="already exists"):
        cli.main(args)
    class CloseFailure(FakeAdapter):
        def __exit__(self,*_):
            raise RuntimeError("native close failed")
    monkeypatch.setattr(cli.H3InferenceAdapter,"load",lambda **_:CloseFailure())
    failed_output = tmp_path / "close-failure"
    with pytest.raises(RuntimeError,match="native close failed"):
        cli.main([*args[:-1],str(failed_output)])
    assert (failed_output / "ROLLOUT_COMPLETE.json").is_file() and not (failed_output / "SUCCESS.json").exists()
    assert json.loads((tmp_path / "close-failure.launch/result.json").read_text())["status"] == "blocked_or_failed"
