"""Local, lazy PickScore reward for Flow-GRPO and TempFlow-GRPO."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from visual_rl.artifacts.hashing import tree_sha256
from visual_rl.core.registry import REWARD_CLIENTS


PICKSCORE_MAX_LENGTH = 77
PICKSCORE_NORMALIZATION_DIVISOR = 26.0
PICKSCORE_FORMULA = "exp(logit_scale) * paired_cosine / 26"


def _sha256(value: str, label: str) -> str:
    value = value.strip().lower() if isinstance(value, str) else ""
    if len(value) != 64 or not set(value) <= set("0123456789abcdef"):
        raise ValueError(f"PickScore {label} must be a 64-digit SHA-256.")
    return value


def _local_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"PickScore {label} must be a local directory.")
    return path.resolve()


def _verified_tree_sha256(path: Path, declared: str, label: str) -> str:
    expected = _sha256(declared, label)
    actual = tree_sha256(path)
    if actual != expected:
        raise ValueError(
            f"PickScore {label} does not match the canonical tree hash of {path}."
        )
    return expected


def _load_local_components(
    model_path: Path, processor_path: Path, *, device: str
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(processor_path), local_files_only=True
    )
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    return model.eval().to(device=device, dtype=torch.float32), processor


def _pil_images(media: Any, batch_size: int) -> list[Any]:
    import torch

    if not isinstance(media, torch.Tensor):
        raise TypeError("PickScore media must be a BCHW torch.Tensor.")
    if media.ndim != 4 or media.shape[:2] != (batch_size, 3):
        raise ValueError("PickScore media must be batch-matched RGB BCHW.")
    if not torch.is_floating_point(media) or min(media.shape[2:]) < 1:
        raise ValueError("PickScore images must be non-empty floating point tensors.")
    media = media.detach()
    if not torch.isfinite(media).all().item():
        raise ValueError("PickScore media contains NaN or infinity.")
    if media.min().item() < 0 or media.max().item() > 1:
        raise ValueError("PickScore canonical media must be in [0, 1].")
    pixels = (
        media.mul(255).round().byte().cpu().permute(0, 2, 3, 1).contiguous().numpy()
    )
    from PIL import Image

    return [Image.fromarray(image) for image in pixels]


def _on_device(inputs: Any, device: str) -> dict[str, Any]:
    if not hasattr(inputs, "items"):
        raise TypeError("PickScore processor output must be a tensor mapping.")
    return {
        str(key): value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _nonempty_strings(values: Sequence[Any]) -> bool:
    return all(isinstance(value, str) and bool(value.strip()) for value in values)


class PickScoreRewardClient:
    """Official paired PickScore; manifest hashes are gate-verified declarations."""

    name = "pickscore"

    def __init__(
        self,
        model_path: str | Path,
        processor_path: str | Path,
        *,
        scorer_revision: str,
        checkpoint_manifest_sha256: str,
        processor_manifest_sha256: str,
        device: str = "cuda",
    ) -> None:
        self.model_path = _local_directory(model_path, "model_path")
        self.processor_path = _local_directory(processor_path, "processor_path")
        if not isinstance(scorer_revision, str) or not scorer_revision.strip():
            raise ValueError("PickScore scorer_revision must be non-empty.")
        self.identity = {
            "scorer_revision": scorer_revision.strip(),
            "checkpoint_manifest_sha256": _verified_tree_sha256(
                self.model_path,
                checkpoint_manifest_sha256,
                "checkpoint_manifest_sha256",
            ),
            "processor_manifest_sha256": _verified_tree_sha256(
                self.processor_path,
                processor_manifest_sha256,
                "processor_manifest_sha256",
            ),
        }
        self.device = device.strip()
        if not self.device:
            raise ValueError("PickScore device must be non-empty.")
        self._model: Any = None
        self._processor: Any = None

    def cache_fingerprint(self) -> dict[str, Any]:
        return {
            "client": f"{type(self).__module__}:{type(self).__qualname__}",
            "identity": dict(self.identity),
            "formula": PICKSCORE_FORMULA,
            "max_length": PICKSCORE_MAX_LENGTH,
        }

    def score(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        sample_id: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prompts = list(prompts)
        if not prompts or not _nonempty_strings(prompts):
            raise ValueError("PickScore prompts must be non-empty strings.")
        if len(metadata) != len(prompts):
            raise ValueError("PickScore metadata must match the prompt batch.")
        if isinstance(sample_id, (str, bytes)):
            raise ValueError("PickScore sample_id must be a sequence of identifiers.")
        identifiers = None if sample_id is None else list(sample_id)
        if identifiers is not None and (
            len(identifiers) != len(prompts)
            or not _nonempty_strings(identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("PickScore sample_id must be unique and batch-matched.")

        images = _pil_images(media, len(prompts))
        model, processor = self._components()
        processor_options = {
            "padding": True,
            "truncation": True,
            "max_length": PICKSCORE_MAX_LENGTH,
            "return_tensors": "pt",
        }
        image_inputs = _on_device(
            processor(images=images, **processor_options), self.device
        )
        text_inputs = _on_device(
            processor(text=prompts, **processor_options), self.device
        )
        raw, values = self._score_features(model, image_inputs, text_inputs)
        evidence = [
            {
                **({"sample_id": identifiers[index]} if identifiers else {}),
                "raw_score": raw[index],
                "normalized_score": values[index],
            }
            for index in range(len(values))
        ]
        return np.asarray(values, dtype=np.float32), {
            "identity": dict(self.identity),
            "formula": PICKSCORE_FORMULA,
            "normalization_divisor": PICKSCORE_NORMALIZATION_DIVISOR,
            "sample_evidence": evidence,
            "valid_mask": [True] * len(values),
        }

    def _components(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            self._model, self._processor = _load_local_components(
                self.model_path, self.processor_path, device=self.device
            )
        return self._model, self._processor

    @staticmethod
    def _score_features(
        model: Any, image_inputs: dict[str, Any], text_inputs: dict[str, Any]
    ) -> tuple[list[float], list[float]]:
        import torch

        with torch.no_grad():
            image = model.get_image_features(**image_inputs).float()
            text = model.get_text_features(**text_inputs).float()
            if image.ndim != 2 or text.shape != image.shape:
                raise ValueError(
                    "PickScore image/text features must have equal 2D shape."
                )
            image_norm = torch.linalg.vector_norm(image, dim=-1, keepdim=True)
            text_norm = torch.linalg.vector_norm(text, dim=-1, keepdim=True)
            if (image_norm == 0).any().item() or (text_norm == 0).any().item():
                raise ValueError("PickScore produced a zero-norm embedding.")
            raw = model.logit_scale.float().exp() * (
                (text / text_norm) * (image / image_norm)
            ).sum(dim=-1)
            normalized = raw / PICKSCORE_NORMALIZATION_DIVISOR
            if not torch.isfinite(raw).all().item():
                raise ValueError("PickScore produced a non-finite score.")
        return raw.cpu().tolist(), normalized.cpu().tolist()


REWARD_CLIENTS.register("pickscore", PickScoreRewardClient)
