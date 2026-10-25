"""Concrete H3 PyTorch components, not a pipeline or a serving engine.

Reference: Diffusers aaf57d03 (Apache-2.0), H3 encoders/before_denoise/
decoders; Flow-Factory 3dd5ca4d denoise.py (Apache-2.0). See
`docs/h3_pytorch_sources.md`. No checkpoint-local Python is imported.
"""
from contextlib import contextmanager
import gc
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time

# Hash the actual installed component sources, not just a version string.
COMPONENT_HASHES = {
    "models/transformers/transformer_minimax_h3.py": "92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef",
    "models/autoencoders/autoencoder_kl_minimax_h3.py": "58e0ed47fdee8227d738eec5bb1e7a47acc21c65ee3de01aa3795c97d679d015",
    "models/autoencoders/autoencoder_kl_minimax_h3_audio.py": "b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57",
    "modular_pipelines/minimax_h3/before_denoise.py": "c0c3484b238b7cf529fb494bcfd02c286db8b34ce3e23fd78a17eeda257764d1",
    "schedulers/scheduling_minimax_h3.py": "307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd",
}
CONVERTER_SHA256 = "86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c"
FORBIDDEN_PACKAGES = ("vllm", "vllm_omni", "sglang", "comfy", "comfyui")


def audit_runtime():
    """Fail if an engine is installed OR imported; verify executed model source."""
    from .h3 import sha256_file, DIFFUSERS_SHA
    for name in FORBIDDEN_PACKAGES:
        if any(k == name or k.startswith(name + ".") for k in sys.modules) or importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"H3 PyTorch requires an isolated environment without {name}")
    import diffusers
    root = Path(diffusers.__file__).resolve().parent
    for relative, expected in COMPONENT_HASHES.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"install exact Diffusers {DIFFUSERS_SHA}; source mismatch: {relative}")
    modules = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors"):
        module = importlib.import_module(name)
        modules[name] = {"version": module.__version__, "file": module.__file__}
    return modules


def inspect_converted(root):
    """Check loadable layout/config only; no receipt or weight-content reads."""
    root = Path(root).resolve(strict=True)
    for name in ("transformer", "vae", "audio_vae"):
        folder = root / name
        if folder.is_symlink():
            raise ValueError("unsafe converted component directory")
        config = json.loads((folder / "config.json").read_text())
        if config.get("quantization_config"):
            raise ValueError("quantized converted assets are unsupported")
        weights = list(folder.glob("*.safetensors"))
        if not weights or any(p.stat().st_size == 0 for p in weights):
            raise ValueError(f"missing converted component {name}")
        for index in folder.glob("*.safetensors.index.json"):
            for shard in json.loads(index.read_text())["weight_map"].values():
                if (Path(shard).name != shard or "\\" in shard or ":" in shard
                        or not (folder / shard).is_file()):
                    raise ValueError("missing converted shard")


def stage_budget(weight_bytes, largest_shard_bytes, sample):
    """One live CPU component + one loading shard + 24 GiB host reserve.

    CPU/group offload bounds GPU weight residency; retain the existing 16 GiB
    global GPU floor. Activations are monitored, not claimed to be measured here.
    """
    needed = (weight_bytes + largest_shard_bytes) / 1024**3 + 24
    if sample["host_available_gib"] < needed or sample["gpu_free_gib"] < 32:
        raise RuntimeError(f"H3 stage admission requires host {needed:.3f}GiB and GPU 32GiB free")
    return needed


def release_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def record_component(component, phase):
    module = importlib.import_module(type(component).__module__)
    logging.info(json.dumps({"event": "h3_component_loaded", "phase": phase,
        "class": type(component).__qualname__, "module": type(component).__module__,
        "module_file": str(getattr(module, "__file__", None))}))


def chunk_feed_forwards(transformer, max_tokens=1024):
    """Token-local FF evaluation only; no attention/cache approximation.

    Adapted from Flow-Factory _ChunkedFeedForward. Retains original modules and
    mixed precision. A remainder chunk is allowed; no sequence padding.
    """
    import torch
    class ChunkedFF(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.net = original.net

        def forward(self, hidden):
            outputs = []
            for chunk in hidden.split(max_tokens, dim=1):
                for module in self.net:
                    chunk = module(chunk)
                outputs.append(chunk)
            return torch.cat(outputs, dim=1)
    for block in [*transformer.transformer_blocks, *transformer.token_refiner.refiner_blocks]:
        block.ff = ChunkedFF(block.ff)


class H3InferenceAdapter:
    """Owns a single *phase* of components, never all four models at once.

    Public numerical ports: encode, prepare_latents, predict, decode. Sampling
    and both sigma grids belong to algorithms.rollout.h3.sample_h3.
    """
    def __init__(self, raw_root, converted_root, assets, *, resource_probe=None, device="cuda"):
        import torch
        self.device = torch.device(device)
        self.raw_root = Path(raw_root) / "FL2VA"
        self.converted_root, self.assets = Path(converted_root), assets
        self.resource_probe = resource_probe
        self.closed = False
        self.transformer = None
        self.phase = None

    @classmethod
    def load(cls, *, model_root, converted_root, model_revision, resource_probe=None):
        import os
        from .h3 import inspect_assets
        if any(os.environ.get(n) != "1" for n in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            raise RuntimeError("H3 requires explicit offline environment")
        audit_runtime()
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("H3 requires exactly one visible CUDA GPU")
        assets = inspect_assets(model_root, model_revision)
        inspect_converted(converted_root)
        return cls(model_root, converted_root, assets, resource_probe=resource_probe)

    @contextmanager
    def _phase(self, name, folder):
        if self.closed or self.phase is not None:
            raise RuntimeError("H3 adapter closed or overlapping component phases")
        release_memory()
        files = list(Path(folder).rglob("*.safetensors"))
        if not files:
            raise ValueError(f"missing phase weights: {name}")
        budget = None
        if self.resource_probe:
            budget = stage_budget(sum(p.stat().st_size for p in files), max(p.stat().st_size for p in files), self.resource_probe())
        self.phase = name
        started = time.monotonic()
        logging.info(json.dumps({"event": "h3_phase_start", "phase": name,
            "host_admission_gib": budget, "resources": self._resource_snapshot()}))
        try:
            yield
        finally:
            self.transformer = None
            self.phase = None
            release_memory()
            logging.info(json.dumps({"event": "h3_phase_end", "phase": name,
                "seconds": time.monotonic() - started, "resources": self._resource_snapshot()}))

    def _resource_snapshot(self):
        if self.resource_probe is None:
            return {}  # Explicit tiny CPU constructor used only by tests.
        import torch
        sample = dict(self.resource_probe())
        if self.device.type == "cuda":
            sample.update(torch_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                          torch_reserved_gib=torch.cuda.memory_reserved() / 1024**3)
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    sample["process_rss_gib"] = int(line.split()[1]) / 1024**2
        return sample

    def encode(self, request):
        import torch
        from accelerate import cpu_offload
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
        with self._phase("encoder", self.raw_root / "text_encoder"), torch.inference_mode():
            tokenizer = Qwen2TokenizerFast.from_pretrained(self.raw_root / "tokenizer", local_files_only=True)
            # Native HF Qwen3-VL; never trust_remote_code. No LM-head logits.
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                self.raw_root / "text_encoder", torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                attn_implementation="sdpa")
            try:
                record_component(encoder, "encoder")
                encoder.eval().requires_grad_(False)
                if encoder.config.text_config.num_hidden_layers <= 50:
                    raise ValueError("H3 needs pre-final-norm hidden_states[50], not a truncated final state")
                encoder.lm_head = torch.nn.Identity()
                # Attach to the actual called root, not the unused LM wrapper.
                cpu_offload(encoder.model, execution_device=self.device, offload_buffers=True)
                ids = tokenizer(request.prompt, add_special_tokens=False)["input_ids"]
                if not ids:
                    raise ValueError("empty H3 tokenization")
                inputs = torch.tensor([ids], device=self.device, dtype=torch.long)
                outputs = encoder.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        mm_token_type_ids=torch.zeros_like(inputs),
                                        use_cache=False, output_hidden_states=True)
                prompt = outputs.hidden_states[50].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                del outputs
            finally:
                # Discard hooks with their owner. Detaching them first would
                # rematerialize every meta parameter on CPU during teardown.
                del encoder
            return prompt

    def prepare_latents(self, request, prompt):
        """Native packed geometry and CPU generator draw order, B=1, no padding."""
        import torch
        if self.closed:
            raise RuntimeError("H3 adapter is closed")
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep, patchify_video_latents)
        latent_frames = (request.frames - 5) // 17 * 5 + 2
        height, width = request.height // 16, request.width // 16
        audio_frames = round(request.frames / 24 * 40)
        tags = torch.ones(prompt.shape[1], dtype=torch.long)
        layout_values = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            tags, latent_frames, height, width, audio_frames, (1, 2, 2), 2, 2, 0, ())
        keys = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
        layout = {key: value.to(self.device) for key, value in zip(keys, layout_values[:5])}
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        # Exactly the upstream order: video as BCTHW, then channel-major audio rows.
        video = torch.randn((1, 24, latent_frames, height, width), generator=generator, dtype=torch.float32)
        video = patchify_video_latents(video, (1, 2, 2))[None].to(self.device)
        audio = torch.randn((1, audio_frames * 2, 32), generator=generator, dtype=torch.float32).to(self.device)
        geometry = (latent_frames, height, width, audio_frames)
        return {"video": video, "audio": audio}, layout, geometry

    @contextmanager
    def denoising(self):
        import torch
        from diffusers import MiniMaxH3Transformer3DModel
        with self._phase("denoiser", self.converted_root / "transformer"):
            model = MiniMaxH3Transformer3DModel.from_pretrained(
                self.converted_root / "transformer", torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True)
            try:
                record_component(model, "denoiser")
                model.eval().requires_grad_(False)
                for name, parameter in model.named_parameters():
                    expected = torch.float32 if any(part in name for part in
                        ("proj_in", "proj_out", "time_embedder")) else torch.bfloat16
                    if parameter.dtype != expected:
                        raise ValueError(f"H3 mixed precision mismatch: {name} {parameter.dtype} != {expected}")
                chunk_feed_forwards(model)
                # No stream prefetch or pinned duplicate of the 62GiB CPU weights.
                model.enable_group_offload(onload_device=self.device, offload_device=torch.device("cpu"),
                    offload_type="block_level", num_blocks_per_group=1, use_stream=False, low_cpu_mem_usage=True)
                self.transformer = model
                yield
            finally:
                self.transformer = None
                del model

    def predict(self, state, prompt, layout, video_time, audio_time):
        """One joint velocity forward; time is t=1-sigma, data-ward positive."""
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
        if self.transformer is None or self.closed:
            raise RuntimeError("predict requires the active denoiser phase")
        unique, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            layout["video_indices"].cpu(), layout["audio_indices"].cpu(), 0, 0,
            prompt.shape[1], float(video_time), float(audio_time), 0.999, 1.0)
        # Refuse quadratic math attention fallback on normal full-length sequences.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            video, audio = self.transformer(hidden_states=state["video"], audio_hidden_states=state["audio"],
                encoder_hidden_states=prompt.to(self.device), timestep=unique.to(self.device),
                timestep_indices=indices.to(self.device), return_dict=False, **layout)
        velocity = {"video": video.float(), "audio": audio.float()}
        for name, value in velocity.items():
            if value.shape != state[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid joint {name} velocity")
        return velocity

    def decode(self, request, state, geometry):
        import torch
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from .h3 import H3Result
        frames, height, width, audio_frames = geometry
        rows = state["video"].detach().cpu().reshape(1, frames, height//2, width//2, 24, 1, 2, 2)
        latents = rows.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, frames, height, width).contiguous()
        audio_latents = state["audio"].detach().cpu().reshape(2, audio_frames, 32).permute(0, 2, 1).contiguous()
        with self._phase("video_decoder", self.converted_root / "vae"), torch.inference_mode():
            vae = AutoencoderKLMiniMaxH3.from_pretrained(self.converted_root / "vae", torch_dtype=torch.float32,
                low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True)
            try:
                record_component(vae, "video_decoder")
                vae.eval().requires_grad_(False).to(self.device)
                vae.enable_tiling()
                mean = torch.tensor(vae.config.latents_mean, device=self.device).view(1, -1, 1, 1, 1)
                std = torch.tensor(vae.config.latents_std, device=self.device).view(1, -1, 1, 1, 1)
                decoded = vae.decode(latents.to(self.device) * std + mean, return_dict=False)[0]
                pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1, 1)
                pixel_std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1, 1)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError("nonfinite decoded H3 video")
                video = ((decoded.float() * pixel_std + pixel_mean).clamp(0, 1) * 255).round().to(torch.uint8)
                video = video[0].permute(1, 2, 3, 0).cpu().contiguous()
                del decoded
            finally:
                del vae
        with self._phase("audio_decoder", self.converted_root / "audio_vae"), torch.inference_mode():
            vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(self.converted_root / "audio_vae", torch_dtype=torch.float32,
                low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True)
            try:
                record_component(vae, "audio_decoder")
                vae.eval().requires_grad_(False).to(self.device)
                mean = torch.tensor(vae.config.latents_mean, device=self.device).view(1, -1, 1)
                std = torch.tensor(vae.config.latents_std, device=self.device).view(1, -1, 1)
                decoded = vae.decode(audio_latents.to(self.device) * std + mean, return_dict=False)[0]
                audio = decoded[:, 0].detach().float().cpu().contiguous()
                del decoded
            finally:
                del vae
        return H3Result(request, video, audio)

    def close(self):
        if self.phase is not None:
            raise RuntimeError("cannot close H3 during an active component phase")
        self.closed = True
        self.transformer = None
        release_memory()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("H3 adapter is closed")
        return self

    def __exit__(self, *_):
        self.close()
