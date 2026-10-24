"""H3 native joint AV generate capability; deliberately not an RL policy port.

Pinned vLLM-Omni owns encode -> packed joint video/audio denoise -> AV decode.
VisualRL owns local assets, request/result validation and rollout artifacts.
No HTTP/ComfyUI, optimizer, LoRA injection, reward or training checkpoint.
"""
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import time

BACKEND_SHA = "0f79bd8b0e5ff9379a9a38e0c585db8f8335bf7c"
MODEL_REPO = "MiniMaxAI/MiniMax-H3"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_backend(root):
    """Verify actual imported upstream source, not its package version alone."""
    root = Path(root).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()
    if git("rev-parse", "HEAD") != BACKEND_SHA or git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError(f"H3 requires clean vLLM-Omni checkout {BACKEND_SHA}")
    module = importlib.import_module("vllm_omni")
    if Path(module.__file__).resolve().parent != root / "vllm_omni":
        raise RuntimeError("H3 backend import is shadowed by another installation")
    return root


def _video_vae_weight_directory(component, config):
    """Resolve the native standalone VAE's config-owned source, not a fallback.

    HF snapshot weight files may be symlinks to its blob cache; directory
    symlinks and absolute/traversing config paths are not accepted.
    """
    if not {"source_path", "source_safetensors_path"}.intersection(config):
        return component  # Existing native flat layout.
    source = config.get("source_path")
    weight = config.get("source_safetensors_path")
    if (config.get("mode") != "standalone" or not isinstance(source, str)
            or not source or "\\" in source or ":" in source
            or any(part in {"", ".", ".."} for part in source.split("/"))
            or Path(source).is_absolute()
            or not isinstance(weight, str) or not weight
            or "/" in weight or "\\" in weight or ":" in weight
            or Path(weight).suffix != ".safetensors"):
        raise ValueError("unsafe or unsupported native video VAE source_path/source_safetensors_path")
    directory = component
    for part in source.split("/"):
        directory = directory / part
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("missing or unsafe native video VAE source directory")
    if not (directory / weight).is_file() or (directory / weight).stat().st_size == 0:
        raise ValueError("missing native video VAE source weight")
    source_config = json.loads((directory / "config.json").read_text())
    if source_config.get("quantization_config"):
        raise ValueError("quantized video VAE source format is not validated")
    return directory


def inspect_assets(root, revision):
    """Inventory the complete native FL2VA partition, including processor/code.

    Revision is caller-declared provenance; actual local bytes are independently
    hashed. This does not authenticate a remote repository or audit model code.
    Quantized/pruned ComfyUI/FastH3 formats are NOT accepted by this first path.
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model revision must be an explicit full HF commit")
    root = Path(root).resolve(strict=True)
    partition = root / "FL2VA"
    index = json.loads((partition / "model_index.json").read_text())
    release = index.get("_minimax_h3", {})
    if release.get("partition") != "fl2va" or release.get("base_schedule") is not None:
        raise ValueError("require native FL2VA with uniform base schedule for RES; no Turbo/DMD schedule")
    shifts = release.get("sigma_shift_scales", {"video": 12.0, "audio": 3.0})
    if shifts != {"video": 12.0, "audio": 3.0}:
        raise ValueError("unexpected H3 video/audio shift contract")
    if any((root / name).exists() for name in ("fastvideo_inference.json", "provenance.json")):
        raise ValueError("FastH3/converted checkpoints are not supported by this native path")
    for name in ("transformer", "text_encoder", "video_vae", "audio_vae"):
        config = json.loads((partition / name / "config.json").read_text())
        if config.get("quantization_config"):
            raise ValueError("quantized model format is not validated for this backend/device")
        component = partition / name
        weight_directory = _video_vae_weight_directory(component, config) if name == "video_vae" else component
        weights = list(weight_directory.glob("*.safetensors"))
        if (not weights or any(p.stat().st_size == 0 for p in weights)
                or any("nvfp4" in p.name.lower() or "pruned" in p.name.lower()
                       for p in component.rglob("*.safetensors"))):
            raise ValueError(f"missing native safetensors or unsupported format: {name}")
        for shard_index in weight_directory.glob("*.safetensors.index.json"):
            shard_map = json.loads(shard_index.read_text())["weight_map"]
            for shard in set(shard_map.values()):
                if Path(shard).name != shard or "\\" in shard or ":" in shard or not (weight_directory / shard).is_file():
                    raise ValueError(f"missing or unsafe native weight shard: {name}/{shard}")
    for name in ("tokenizer", "processor"):
        if not (partition / name).is_dir() or not list((partition / name).glob("*.json")):
            raise ValueError(f"H3 requires the native {name}, not a Wan/Comfy replacement")
    files = {}
    for path in sorted(partition.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    code = {name: v["sha256"] for name, v in files.items() if name.endswith(".py")}
    code_hash = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    return {"repo": MODEL_REPO, "declared_revision": revision, "root": str(root),
            "files": files, "total_bytes": sum(v["bytes"] for v in files.values()),
            "model_python_files": code, "model_code_review_sha256": code_hash,
            "assets_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


@dataclass(frozen=True)
class H3Request:
    sample_id: str
    prompt: str
    seed: int
    width: int = 1344
    height: int = 768
    frames: int = 124
    steps: int = 20
    task: str = "t2va"
    images: tuple[Path, ...] = ()

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", self.sample_id):
            raise ValueError("sample_id must be a safe local identifier")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("H3 prompt must be nonempty; original spelling/whitespace are preserved")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("H3 seed must be uint32")
        if any(type(v) is not int or v < 1 for v in (self.width, self.height, self.frames, self.steps)):
            raise ValueError("H3 geometry/steps must be positive integers")
        if self.width % 32 or self.height % 32 or not 96 <= self.frames <= 365 or self.frames % 17 != 5:
            raise ValueError("H3 requires spatial32, >=4s and aligned17n+5 frames; no silent rounding")
        if max(self.width, self.height) > 4 * min(self.width, self.height):
            raise ValueError("H3 aspect ratio must be within [1:4,4:1]")
        if type(self.images) is not tuple or any(not isinstance(p, Path) or not p.is_file() for p in self.images):
            raise ValueError("H3 images must be explicit existing local Paths")
        if self.task not in {"t2va", "fl2va"} or (self.task == "t2va" and self.images) or (self.task == "fl2va" and not 1 <= len(self.images) <= 2):
            raise ValueError("T2VA has no images; FL2VA requires first and optionally last local image")


@dataclass(frozen=True)
class H3Result:
    request: H3Request
    video: object = field(repr=False)  # CPU uint8 [F,H,W,3], not a filename/MP4
    audio: object = field(repr=False)  # CPU float32 [2,N], native nonzero waveform
    fps: int = 24
    sample_rate: int = 32000
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        import torch
        r, video, audio = self.request, self.video, self.audio
        if (not isinstance(video, torch.Tensor) or video.dtype != torch.uint8
                or video.device.type != "cpu" or tuple(video.shape) != (r.frames, r.height, r.width, 3)):
            raise ValueError("H3 native video must be requested CPU uint8 FHWC")
        if (not isinstance(audio, torch.Tensor) or audio.dtype != torch.float32 or audio.device.type != "cpu"
                or audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < 1
                or audio.requires_grad or audio.grad_fn is not None
                or not bool(torch.isfinite(audio).all()) or not bool((audio != 0).any())):
            raise ValueError("H3 native audio must be finite nonzero detached CPU float32 stereo; no silent substitute")
        if self.fps != 24 or self.sample_rate != 32000:
            raise ValueError("unexpected native H3 AV rates")
        # Audio is quantized to the native40Hz latent clock, then decoded800x.
        expected_samples = round(r.frames / self.fps * 40) * 800
        if audio.shape[1] != expected_samples:
            raise ValueError("H3 audio length disagrees with the independent40Hz latent clock")


class H3InferenceAdapter:
    """Single-request native in-process session. H3 RL/logprob is unsupported."""

    def __init__(self, engine, assets):
        self._engine, self.assets = engine, assets

    @classmethod
    def load(cls, *, model_root, model_revision, backend_root, reviewed_model_code_sha256):
        import os
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("H3 native loading requires explicit offline environment")
        verify_backend(backend_root)
        assets = inspect_assets(model_root, model_revision)
        if reviewed_model_code_sha256 != assets["model_code_review_sha256"]:
            raise RuntimeError("review all listed local model Python sources and pass their exact review digest before execution")
        import torch
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("native H3 uni requires exactly one visible CUDA GPU")
        config = OmniDiffusionConfig.from_kwargs(
            model=str(Path(model_root).resolve()), model_class_name="MiniMaxH3Pipeline",
            task_type="fl2va", dtype="bfloat16", num_gpus=1,
            distributed_executor_backend="uni", max_num_seqs=1,
            enforce_eager=True, step_execution=False, cache_backend="none",
            trust_remote_code=True,  # ONLY after explicit local-code digest approval
            diffusion_attention_backend="TORCH_SDPA", vae_use_tiling=True,
            diffusion_offload_config={"mode": "module", "components": ["dit", "text_encoder"], "pin_memory": False},
        )
        # Native root has partition subfolders, not a generic root transformer.
        # Read exact local metadata; never fall back to a Hub model ID.
        raw = json.loads((Path(model_root) / "FL2VA/transformer/config.json").read_text())
        config.set_tf_model_config(TransformerConfig.from_dict(raw))
        config.update_multimodal_support()
        engine = DiffusionEngine(config)
        return cls(engine, assets)

    def generate(self, request: H3Request) -> H3Result:
        if self._engine is None:
            raise RuntimeError("H3 adapter is closed")
        if not isinstance(request, H3Request):
            raise TypeError("request must be H3Request")
        import torch
        from PIL import Image
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        prompt = {"prompt": request.prompt}
        if request.images:
            images = []
            for path in request.images:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            prompt["multi_modal_data"] = {"image": images}
        sampling = OmniDiffusionSamplingParams(
            seed=request.seed, width=request.width, height=request.height,
            num_frames=request.frames, fps=24, num_inference_steps=request.steps,
            num_outputs_per_prompt=1, guidance_scale=1.0, quality="lossless",
            extra_args={"task": request.task, "sampler": "res_multistep",
                        "flow_shift": 12.0, "audio_flow_shift": 3.0, "preencode_mp4": False},
        )
        started = time.monotonic()
        with torch.inference_mode():
            output = self._engine.add_req_and_wait_for_response(OmniDiffusionRequest(
                prompt=prompt, sampling_params=sampling, request_id=request.sample_id))
        if output.error or output.aborted or not output.finished:
            raise RuntimeError(f"native H3 failed: {output.error or output.abort_message or 'incomplete output'}")
        if not callable(output.post_process_func):
            raise ValueError("native H3 must return its joint AV postprocessor, not an external file")
        media = output.post_process_func(output.output, output_type="pt")
        video, audio = media["video"], media["audio"]
        if (not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1
                or not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != 1):
            raise ValueError("native H3 requires B=1 joint video/audio tensor output")
        return H3Result(request, video[0].detach().cpu().contiguous(),
            audio[0].detach().to(device="cpu", dtype=torch.float32).contiguous(),
            fps=media["fps"], sample_rate=media["audio_sample_rate"],
            metrics={"generate_seconds": time.monotonic() - started,
                     "backend_peak_memory_mb": output.peak_memory_mb,
                     "stage_durations": output.stage_durations})

    def close(self):
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def __enter__(self):
        if self._engine is None:
            raise RuntimeError("H3 adapter is closed")
        return self

    def __exit__(self, *_exc):
        self.close()
