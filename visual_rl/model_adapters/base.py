"""Final model-adapter contract shared by every builtin model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import ClassVar, Literal, TYPE_CHECKING

from visual_rl.core.types import (
    FrozenMapping,
    PolicyRecomputeStats,
    ResolutionContext,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    ValidationCheck,
    ValidationContext,
)

if TYPE_CHECKING:
    import torch


class ModelAdapter(ABC):
    """One trainable policy implementation behind the typed rollout contract."""

    MEDIA_TYPE: ClassVar[Literal["image", "video"]]

    @property
    @abstractmethod
    def train_module(self) -> "torch.nn.Module":
        """Return the sole module that owns VisualRL-trainable parameters."""

        raise NotImplementedError

    def named_parameters(
        self,
    ) -> tuple[tuple[str, "torch.nn.Parameter"], ...]:
        """Return the canonical unique trainable parameter sequence."""

        selected = tuple(
            (name, parameter)
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        )
        names = tuple(name for name, _parameter in selected)
        identities = tuple(id(parameter) for _name, parameter in selected)
        if any(not name for name in names):
            raise ValueError("trainable parameter names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("trainable parameter names must be unique")
        if len(identities) != len(set(identities)):
            raise ValueError("trainable parameter objects must be unique")
        return selected

    def parameters(self) -> tuple["torch.nn.Parameter", ...]:
        """Return parameters in exactly the same order as ``named_parameters``."""

        return tuple(parameter for _name, parameter in self.named_parameters())

    @abstractmethod
    def sample(self, request: RolloutRequest) -> RolloutBatch:
        """Sample exactly one typed request and echo its identity fields."""

        raise NotImplementedError

    @abstractmethod
    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        """Recompute differentiable policy statistics for one microbatch."""

        raise NotImplementedError

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__}.resolve_params() requires a mapping, "
                f"got {type(raw).__name__}"
            )
        del context
        return FrozenMapping(raw)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        del resolved, context
        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> ModelAdapter:
        del resolved, context
        raise NotImplementedError(f"{cls.__name__} must implement from_config()")

    @classmethod
    def required_capabilities(
        cls,
        resolved_params: Mapping[str, object],
    ) -> frozenset[str]:
        del resolved_params
        return frozenset()

    def close(self) -> None:
        """Release resources owned by this adapter; pure adapters are no-op."""

    def save_checkpoint(self, output_dir: Path) -> None:
        """Write the canonical two-file trainable-parameter checkpoint."""

        import torch

        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink():
            raise ValueError("adapter checkpoint directory must not be a symlink")
        expected_files = {"adapter.json", "adapter_state.pt"}
        existing = {item.name for item in output_dir.iterdir()}
        if existing:
            raise ValueError(
                "adapter checkpoint staging directory must be empty; "
                f"found {sorted(existing)}"
            )

        named = self.named_parameters()
        tensors = {
            name: parameter.detach().to(device="cpu").contiguous().clone()
            for name, parameter in named
        }
        non_finite = [
            name
            for name, tensor in tensors.items()
            if (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all())
        ]
        if non_finite:
            raise ValueError(
                f"adapter checkpoint parameters must be finite: {non_finite}"
            )
        state_path = output_dir / "adapter_state.pt"
        torch.save(
            {"format_version": 1, "parameters": tensors},
            state_path,
        )
        state_sha256 = _sha256_file(state_path)
        metadata = {
            "format_version": 1,
            "state_file": "adapter_state.pt",
            "state_sha256": state_sha256,
            "parameters": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                for name, tensor in tensors.items()
            ],
        }
        metadata_path = output_dir / "adapter.json"
        metadata_path.write_bytes(
            (
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        if {item.name for item in output_dir.iterdir()} != expected_files:
            raise RuntimeError("adapter checkpoint must contain exactly two files")

    def validate_checkpoint(self, checkpoint_dir: Path) -> None:
        """Validate a checkpoint completely without observable state mutation."""

        self._read_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Atomically copy a fully validated checkpoint into existing tensors."""

        import torch

        loaded = self._read_checkpoint(checkpoint_dir)
        named = self.named_parameters()
        staged = tuple(
            loaded[name].to(device=parameter.device, dtype=parameter.dtype).clone()
            for name, parameter in named
        )
        originals = tuple(parameter.detach().clone() for _name, parameter in named)
        try:
            with torch.no_grad():
                for (_name, parameter), value in zip(named, staged, strict=True):
                    parameter.copy_(value)
        except BaseException:
            with torch.no_grad():
                for (_name, parameter), value in zip(
                    named,
                    originals,
                    strict=True,
                ):
                    torch.Tensor.copy_(parameter, value)
            raise

    def _read_checkpoint(self, checkpoint_dir: Path) -> dict[str, "torch.Tensor"]:
        import torch

        if not isinstance(checkpoint_dir, Path):
            raise TypeError("checkpoint_dir must be a pathlib.Path")
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise ValueError("checkpoint_dir must be a real directory")
        children = tuple(checkpoint_dir.iterdir())
        if {item.name for item in children} != {"adapter.json", "adapter_state.pt"}:
            raise ValueError(
                "adapter checkpoint must contain exactly adapter.json and "
                "adapter_state.pt"
            )
        if any(item.is_symlink() or not item.is_file() for item in children):
            raise ValueError("adapter checkpoint files must be regular files")

        metadata_path = checkpoint_dir / "adapter.json"
        state_path = checkpoint_dir / "adapter_state.pt"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("adapter.json is not valid UTF-8 JSON") from exc
        if not isinstance(metadata, dict) or set(metadata) != {
            "format_version",
            "state_file",
            "state_sha256",
            "parameters",
        }:
            raise ValueError("adapter.json has an invalid exact key set")
        if metadata["format_version"] != 1 or type(metadata["format_version"]) is not int:
            raise ValueError("adapter format_version must be integer 1")
        if metadata["state_file"] != "adapter_state.pt":
            raise ValueError("adapter state_file must be adapter_state.pt")
        digest = metadata["state_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("adapter state_sha256 must be lowercase SHA-256")
        if _sha256_file(state_path) != digest:
            raise ValueError("adapter_state.pt SHA-256 mismatch")

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("adapter_state.pt cannot be safely loaded") from exc
        if not isinstance(state, dict) or set(state) != {
            "format_version",
            "parameters",
        }:
            raise ValueError("adapter state has an invalid exact key set")
        if state["format_version"] != 1 or type(state["format_version"]) is not int:
            raise ValueError("adapter state format_version must be integer 1")
        parameters = state["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("adapter state parameters must be an ordered mapping")

        named = self.named_parameters()
        expected_names = tuple(name for name, _parameter in named)
        if tuple(parameters) != expected_names:
            raise ValueError("adapter checkpoint parameter names/order mismatch")
        parameter_metadata = metadata["parameters"]
        if not isinstance(parameter_metadata, list) or len(parameter_metadata) != len(
            named
        ):
            raise ValueError("adapter parameter metadata length mismatch")
        for index, ((name, target), item) in enumerate(
            zip(named, parameter_metadata, strict=True)
        ):
            if not isinstance(item, dict) or set(item) != {
                "name",
                "shape",
                "dtype",
            }:
                raise ValueError(
                    f"adapter parameter metadata {index} has invalid keys"
                )
            value = parameters[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"adapter parameter {name!r} is not a tensor")
            if item != {
                "name": name,
                "shape": list(target.shape),
                "dtype": str(target.dtype),
            }:
                raise ValueError(f"adapter parameter metadata mismatch for {name!r}")
            if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                raise ValueError(f"adapter parameter tensor mismatch for {name!r}")
            if (
                value.is_floating_point() or value.is_complex()
            ) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"adapter parameter {name!r} must be finite")
        return parameters


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
