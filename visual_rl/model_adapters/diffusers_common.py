"""Shared helpers for optional Diffusers image adapters."""

from __future__ import annotations

from typing import Any


class AdapterNotLoadedError(RuntimeError):
    """Raised when a lazy real-model adapter is used before loading weights."""


def resolve_torch_dtype(dtype_name: str | None):
    import torch

    if dtype_name in {None, "auto"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def require_model_path(config: dict[str, Any], adapter_name: str) -> str:
    model_path = config.get("model_path") or config.get("pretrained_model") or config.get("extra", {}).get("model_path")
    if not model_path:
        raise AdapterNotLoadedError(f"{adapter_name} requires model.model_path before loading real weights.")
    return str(model_path)


def trainable_parameters(module) -> list[Any]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def module_or_self(module):
    return getattr(module, "module", module)


def apply_peft_lora(
    module,
    *,
    rank: int,
    alpha: int,
    target_modules: list[str],
    lora_path: str | None = None,
):
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise ImportError("Install visual-rl[train] to use LoRA adapters.") from exc

    module.requires_grad_(False)
    if lora_path:
        wrapped = PeftModel.from_pretrained(module, lora_path)
        wrapped.set_adapter("default")
        for name, parameter in wrapped.named_parameters():
            parameter.requires_grad_("lora" in name.lower())
        return wrapped

    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    return get_peft_model(module, lora_config)


def make_generator(device, seed: int | None):
    import torch

    if seed is None:
        return None
    generator_device = device if torch.device(device).type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def stack_steps(values, *, dim: int = 1):
    import torch

    if isinstance(values, torch.Tensor):
        return values
    return torch.stack(list(values), dim=dim)
