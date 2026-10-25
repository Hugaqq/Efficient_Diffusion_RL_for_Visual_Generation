"""H3 asset/request contracts and staged PyTorch component adapter.

The model owns encode/predict/decode, never a sampling loop. Raw official assets
are immutable; the reviewed Diffusers conversion is stored separately.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

DIFFUSERS_SHA = "aaf57d03a3edfbd9de457d2872417a92f149a91d"
MODEL_REPO = "MiniMaxAI/MiniMax-H3"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("require native FL2VA uniform base schedule; no Turbo/DMD schedule")
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
    steps: int = 40  # sigma grid points INCLUDING terminal zero: 39 Euler forwards
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
        if self.task != "t2va" or self.images:
            raise ValueError("this PyTorch release supports text-only T2VA; conditioned inference is not implemented")
        if self.steps < 2:
            raise ValueError("H3 steps counts sigma grid points, including terminal zero; require >=2")


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


# Kept separate so importing value contracts does not import optional model code.
from .h3_components import H3InferenceAdapter  # noqa: E402
