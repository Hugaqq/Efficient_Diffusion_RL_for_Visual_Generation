"""Shared helpers for optional Diffusers image adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class AdapterNotLoadedError(RuntimeError):
    """Raised when a lazy real-model adapter is used before loading weights."""


@dataclass(frozen=True)
class GradientCheckpointingState:
    """Auditable requested/effective state for a Diffusers transformer."""

    requested: bool | None
    effective: bool | None


def gradient_checkpointing_request(config: Mapping[str, Any]) -> bool | None:
    """Return an explicit request without inventing a compatibility default.

    Resolved VisualRL configs keep this optional control in ``model.extra``.
    Direct adapter callers historically pass flattened dictionaries, so both
    shapes remain supported while conflicting declarations fail closed.
    """

    sentinel = object()
    direct = config.get("gradient_checkpointing", sentinel)
    extra = config.get("extra", {})
    if not isinstance(extra, Mapping):
        raise TypeError("model.extra must be a mapping")
    nested = extra.get("gradient_checkpointing", sentinel)
    for label, candidate in (("model", direct), ("model.extra", nested)):
        if candidate is not sentinel and not isinstance(candidate, bool):
            raise TypeError(f"{label}.gradient_checkpointing must be a bool")
    if (
        direct is not sentinel
        and nested is not sentinel
        and direct is not nested
    ):
        raise ValueError(
            "Conflicting gradient_checkpointing declarations in model and model.extra"
        )
    value = direct if direct is not sentinel else nested
    if value is sentinel:
        return None
    if not isinstance(value, bool):
        raise TypeError("gradient_checkpointing must be a bool when provided")
    return value


def _gradient_checkpointing_effective(
    module: Any,
    *,
    context: str,
    required: bool,
) -> bool | None:
    try:
        value = getattr(module, "is_gradient_checkpointing")
    except AttributeError:
        if required:
            raise RuntimeError(
                f"{context} does not expose is_gradient_checkpointing"
            ) from None
        return None
    except Exception as exc:
        raise RuntimeError(
            f"Cannot inspect {context} is_gradient_checkpointing: {exc}"
        ) from exc
    if callable(value):
        try:
            value = value()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot inspect {context} is_gradient_checkpointing: {exc}"
            ) from exc
    if not isinstance(value, bool):
        raise RuntimeError(
            f"{context} is_gradient_checkpointing must be bool, got "
            f"{type(value).__name__}"
        )
    return value


def configure_gradient_checkpointing(
    module: Any,
    requested: bool | None,
    *,
    context: str,
) -> GradientCheckpointingState:
    """Apply an explicit Diffusers request and prove its effective state."""

    if requested is None:
        return GradientCheckpointingState(
            requested=None,
            effective=_gradient_checkpointing_effective(
                module,
                context=context,
                required=False,
            ),
        )
    method_name = (
        "enable_gradient_checkpointing"
        if requested
        else "disable_gradient_checkpointing"
    )
    try:
        method = getattr(module, method_name)
    except AttributeError:
        raise RuntimeError(f"{context} does not support {method_name}()") from None
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect {context} {method_name}(): {exc}") from exc
    if not callable(method):
        raise RuntimeError(f"{context} does not support callable {method_name}()")
    try:
        method()
    except Exception as exc:
        raise RuntimeError(f"{context} {method_name}() failed: {exc}") from exc
    effective = _gradient_checkpointing_effective(
        module,
        context=context,
        required=True,
    )
    if effective is not requested:
        raise RuntimeError(
            f"{context} {method_name}() did not take effect: "
            f"requested={requested}, effective={effective}"
        )
    return GradientCheckpointingState(requested=requested, effective=effective)


def verify_gradient_checkpointing(
    module: Any,
    state: GradientCheckpointingState,
    *,
    context: str,
) -> GradientCheckpointingState:
    """Re-read state after wrapping/moving and reject explicit drift."""

    effective = _gradient_checkpointing_effective(
        module,
        context=context,
        required=state.requested is not None,
    )
    if effective is None and state.requested is None:
        effective = state.effective
    if state.requested is not None and effective is not state.requested:
        raise RuntimeError(
            f"{context} gradient checkpointing state drifted: "
            f"requested={state.requested}, effective={effective}"
        )
    return GradientCheckpointingState(
        requested=state.requested,
        effective=effective,
    )


def gradient_checkpointing_metadata(
    module: Any,
    state: GradientCheckpointingState,
    *,
    context: str,
) -> tuple[GradientCheckpointingState, dict[str, bool | None]]:
    """Refresh and serialize state for rollout/checkpoint provenance."""

    refreshed = verify_gradient_checkpointing(module, state, context=context)
    return refreshed, {
        "gradient_checkpointing_requested": refreshed.requested,
        "gradient_checkpointing_effective": refreshed.effective,
    }


def validate_gradient_checkpointing_checkpoint_metadata(
    metadata: Mapping[str, Any],
    state: GradientCheckpointingState,
    *,
    context: str,
) -> None:
    """Bind new checkpoint provenance to the already verified runtime state.

    Checkpoints written before these fields existed remain loadable. Once one
    field is present, both are required so a truncated or partially upgraded
    declaration cannot silently pass.
    """

    requested_key = "gradient_checkpointing_requested"
    effective_key = "gradient_checkpointing_effective"
    present = {key for key in (requested_key, effective_key) if key in metadata}
    if not present:
        return
    if len(present) != 2:
        raise RuntimeError(
            f"{context} gradient checkpointing metadata must contain both "
            "requested and effective fields"
        )
    recorded_requested = metadata[requested_key]
    recorded_effective = metadata[effective_key]
    for label, value in (
        ("requested", recorded_requested),
        ("effective", recorded_effective),
    ):
        if value is not None and not isinstance(value, bool):
            raise RuntimeError(
                f"{context} gradient checkpointing {label} metadata must be "
                "bool or null"
            )
    if recorded_requested is not None and recorded_effective is None:
        raise RuntimeError(
            f"{context} explicit gradient checkpointing request has no "
            "effective state"
        )
    if (
        recorded_requested is not state.requested
        or recorded_effective is not state.effective
    ):
        raise RuntimeError(
            f"{context} gradient checkpointing metadata mismatch: "
            f"checkpoint requested/effective="
            f"{recorded_requested}/{recorded_effective}, runtime="
            f"{state.requested}/{state.effective}"
        )


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
