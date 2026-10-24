"""Native Wan generation, without the training composition root.

This is a complete generate capability, NOT the training single-step predict
ABI. Diffusers owns its inference loop; no LoRA, reward, optimizer, parameter
projection, recipe or training checkpoint is constructed here.
"""
from dataclasses import dataclass
from pathlib import Path
import math

from visual_rl.data.media import DecodedMediaBatch


@dataclass(frozen=True)
class WanGenerationRequest:
    prompt: str
    seed: int
    height: int = 480
    width: int = 832
    frames: int = 81
    steps: int = 20
    guidance_scale: float = 4.5

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be nonempty (its original text is preserved)")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be uint32")
        for name in ("height", "width", "frames", "steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.height % 16 or self.width % 16 or (self.frames - 1) % 4:
            raise ValueError("Wan requires spatial multiples of16 and frames=4n+1")
        if (isinstance(self.guidance_scale, bool)
                or not isinstance(self.guidance_scale, (int, float))
                or not math.isfinite(self.guidance_scale) or self.guidance_scale < 1):
            raise ValueError("guidance_scale must be finite and >=1")


class WanInferenceAdapter:
    """One owner for a local, frozen native pipeline. Use as a context manager."""

    def __init__(self, pipeline, device):
        self._pipeline = pipeline
        self.device = device

    @classmethod
    def load(cls, model_path, *, device="cuda:0", cpu_offload=True):
        import torch
        from diffusers import WanPipeline

        path = Path(model_path).resolve(strict=True)
        if not path.is_dir() or not (path / "model_index.json").is_file():
            raise ValueError("Wan inference requires a local Diffusers model directory")
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"} or (cpu_offload and device.type != "cuda"):
            raise ValueError("CPU offload requires a CUDA device; otherwise use cpu/cuda")
        pipeline = WanPipeline.from_pretrained(
            str(path), torch_dtype=torch.bfloat16,
            local_files_only=True, low_cpu_mem_usage=True,
        )
        adapter = cls(pipeline, device)
        try:
            for component in pipeline.components.values():
                if isinstance(component, torch.nn.Module):
                    component.requires_grad_(False)
                    component.eval()
            # Native Wan decode is FP32; do not blanket-cast VAE to BF16.
            pipeline.vae.to(dtype=torch.float32)
            if cpu_offload:
                pipeline.enable_model_cpu_offload(device=device)
            else:
                pipeline.to(device)
        except BaseException as error:
            try:
                adapter.close()
            except BaseException as cleanup:
                error.add_note(f"inference cleanup failed: {type(cleanup).__name__}: {cleanup}")
            raise
        return adapter

    def generate(self, request: WanGenerationRequest) -> DecodedMediaBatch:
        import torch

        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("Wan inference adapter is closed")
        if not isinstance(request, WanGenerationRequest):
            raise TypeError("request must be WanGenerationRequest")
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt, negative_prompt="",
                height=request.height, width=request.width, num_frames=request.frames,
                num_inference_steps=request.steps, guidance_scale=request.guidance_scale,
                generator=generator, output_type="pt",
            ).frames
        expected = (1, request.frames, 3, request.height, request.width)
        if (not isinstance(result, torch.Tensor) or tuple(result.shape) != expected
                or not result.is_floating_point() or not bool(torch.isfinite(result).all())
                or bool((result < 0).any()) or bool((result > 1).any())):
            raise ValueError("native Wan output must be finite [0,1] BFCHW at requested geometry")
        return DecodedMediaBatch(result.detach().to(device="cpu", dtype=torch.float32), "BFCHW")

    def close(self):
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.remove_all_hooks()
            pipeline.to("cpu")

    def __enter__(self):
        if self._pipeline is None:
            raise RuntimeError("Wan inference adapter is closed")
        return self

    def __exit__(self, *_exc):
        self.close()
