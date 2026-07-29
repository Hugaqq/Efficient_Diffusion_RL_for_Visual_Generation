"""Internal model-adapter contract and final builtin implementations."""

from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
from visual_rl.model_adapters.wan import WanFlashAdapter, WanWorldR1Adapter

__all__ = [
    "ModelAdapter",
    "SD3TempFlowAdapter",
    "TinyDiffusionAdapter",
    "WanFlashAdapter",
    "WanWorldR1Adapter",
]
