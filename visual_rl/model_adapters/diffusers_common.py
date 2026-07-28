"""Small shared implementation helpers for repository-owned Diffusers models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterator


class AdapterNotLoadedError(RuntimeError):
    """Raised when a real-model adapter has not finished construction."""


@contextmanager
def reference_repo_import_path(repo_root: Path) -> Iterator[Path]:
    """Expose exactly one resolved reference repository for a scoped import.

    Reference modules are removed again on exit, so two experiments cannot
    accidentally share helpers imported from different repository identities.
    """

    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise ValueError("reference_repo must be an absolute pathlib.Path")
    if not repo_root.is_dir():
        raise FileNotFoundError(
            f"reference_repo does not exist or is not a directory: {repo_root}"
        )
    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "flow_grpo" or name.startswith("flow_grpo.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(repo_root))
    try:
        yield repo_root
    finally:
        sys.path[:] = previous_path
        for name, module in tuple(sys.modules.items()):
            module_file = getattr(module, "__file__", None)
            if not (name == "flow_grpo" or name.startswith("flow_grpo.")):
                continue
            if module_file is None:
                sys.modules.pop(name, None)
                continue
            resolved_file = Path(module_file).resolve()
            if resolved_file == repo_root or repo_root in resolved_file.parents:
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


@dataclass(frozen=True)
class GradientCheckpointingState:
    """The requested and observed state after model construction."""

    requested: bool
    effective: bool


def _read_gradient_checkpointing(module: Any, *, context: str) -> bool:
    try:
        value = getattr(module, "is_gradient_checkpointing")
    except AttributeError:
        raise RuntimeError(
            f"{context} does not expose is_gradient_checkpointing"
        ) from None
    if callable(value):
        value = value()
    if type(value) is not bool:
        raise RuntimeError(
            f"{context} is_gradient_checkpointing must be bool, "
            f"got {type(value).__name__}"
        )
    return value


def configure_gradient_checkpointing(
    module: Any,
    requested: bool,
    *,
    context: str,
) -> GradientCheckpointingState:
    """Apply the required canonical bool and prove that it took effect."""

    if type(requested) is not bool:
        raise TypeError("gradient_checkpointing must be bool")
    method_name = (
        "enable_gradient_checkpointing"
        if requested
        else "disable_gradient_checkpointing"
    )
    method = getattr(module, method_name, None)
    if not callable(method):
        raise RuntimeError(f"{context} does not support {method_name}()")
    method()
    effective = _read_gradient_checkpointing(module, context=context)
    if effective is not requested:
        raise RuntimeError(
            f"{context} {method_name}() did not take effect: "
            f"requested={requested}, effective={effective}"
        )
    return GradientCheckpointingState(
        requested=requested,
        effective=effective,
    )


def verify_gradient_checkpointing(
    module: Any,
    state: GradientCheckpointingState,
    *,
    context: str,
) -> GradientCheckpointingState:
    effective = _read_gradient_checkpointing(module, context=context)
    if effective is not state.requested:
        raise RuntimeError(
            f"{context} gradient checkpointing state drifted: "
            f"requested={state.requested}, effective={effective}"
        )
    return GradientCheckpointingState(
        requested=state.requested,
        effective=effective,
    )


def resolve_torch_dtype(precision: str):
    """Map the sole runtime precision vocabulary to a torch dtype."""

    import torch

    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[precision]
    except KeyError:
        raise ValueError(f"Unsupported runtime precision: {precision!r}") from None


def apply_peft_lora(
    module: Any,
    *,
    rank: int,
    alpha: int,
    target_modules: tuple[str, ...],
):
    """Freeze the base module and build the one supported LoRA topology."""

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise ImportError("Install visual-rl[train] to use LoRA adapters.") from exc

    module.requires_grad_(False)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
    )
    wrapped = get_peft_model(module, lora_config)
    trainable = tuple(
        parameter for parameter in wrapped.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("PEFT did not create any trainable LoRA parameters")
    return wrapped


def make_generator(device: object, seed: int):
    import torch

    generator_device = device if torch.device(device).type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)
