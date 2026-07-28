from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from visual_rl.artifacts.hashing import tree_sha256
import visual_rl.feedback.pickscore as pickscore_module
from visual_rl.feedback.pickscore import PICKSCORE_FORMULA, PickScoreRewardClient

torch = pytest.importorskip("torch")


def _client(tmp_path: Path, **overrides: Any) -> PickScoreRewardClient:
    model, processor = tmp_path / "model", tmp_path / "processor"
    model.mkdir(exist_ok=True)
    processor.mkdir(exist_ok=True)
    (model / "model.safetensors").write_bytes(b"small frozen checkpoint")
    (processor / "processor_config.json").write_text(
        '{"processor": "frozen"}\n', encoding="utf-8"
    )
    kwargs = {
        "scorer_revision": "pickscore-upstream-commit",
        "checkpoint_manifest_sha256": tree_sha256(model),
        "processor_manifest_sha256": tree_sha256(processor),
        "device": "cpu",
        **overrides,
    }
    return PickScoreRewardClient(model, processor, **kwargs)


def test_official_formula_lazy_load_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor_calls, loads = [], []

    def processor(**kwargs: Any) -> dict[str, Any]:
        processor_calls.append(kwargs)
        values = kwargs.get("images", kwargs.get("text"))
        key = "pixel_values" if "images" in kwargs else "input_ids"
        return {key: torch.arange(len(values))}

    model = SimpleNamespace(
        logit_scale=torch.tensor(math.log(13.0)),
        get_image_features=lambda **kwargs: torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        get_text_features=lambda **kwargs: torch.tensor([[3.0, 4.0], [2.0, 0.0]]),
    )

    def load(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        loads.append((args, kwargs))
        return model, processor

    monkeypatch.setattr(pickscore_module, "_load_local_components", load)
    monkeypatch.setattr(
        pickscore_module, "_pil_images", lambda media, size: [object()] * size
    )
    client = _client(tmp_path)
    media = torch.zeros(2, 3, 2, 2)
    values, metadata = client.score(
        media, ["first", "second"], [{}, {}], sample_id=["a", "b"]
    )
    client.score(media, ["first", "second"], [{}, {}])

    identity = {
        "scorer_revision": "pickscore-upstream-commit",
        "checkpoint_manifest_sha256": tree_sha256(client.model_path),
        "processor_manifest_sha256": tree_sha256(client.processor_path),
    }
    assert len(loads) == 1
    assert values.dtype == np.float32
    assert values.tolist() == pytest.approx([0.5, 0.0])
    assert metadata["identity"] == identity
    assert metadata["formula"] == PICKSCORE_FORMULA
    evidence = metadata["sample_evidence"]
    assert [item["sample_id"] for item in evidence] == ["a", "b"]
    assert [item["raw_score"] for item in evidence] == pytest.approx([13.0, 0.0])
    assert [item["normalized_score"] for item in evidence] == pytest.approx([0.5, 0.0])
    assert client.cache_fingerprint()["identity"] == identity
    assert all(
        call["max_length"] == 77 and call["truncation"] for call in processor_calls
    )


def test_loader_is_local_only_and_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []

    class Model:
        def eval(self) -> Model:
            return self

        def to(self, **kwargs: Any) -> Model:
            calls.append(("to", kwargs))
            return self

    def load_processor(path: str, **kwargs: Any) -> object:
        calls.append(("processor", (path, kwargs)))
        return object()

    def load_model(path: str, **kwargs: Any) -> Model:
        calls.append(("model", (path, kwargs)))
        return Model()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=load_model),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        ),
    )
    model, processor = tmp_path / "model", tmp_path / "processor"
    model.mkdir()
    processor.mkdir()
    pickscore_module._load_local_components(model, processor, device="cpu")

    assert calls[0][1][1] == {"local_files_only": True}
    assert calls[1][1][1] == {"local_files_only": True}
    assert calls[2] == ("to", {"device": "cpu", "dtype": torch.float32})


@pytest.mark.parametrize(
    "media,match",
    [
        (object(), "BCHW"),
        (torch.zeros(1, 2, 3, 4, 4), "BCHW"),
        (torch.zeros(1, 4, 4, 4), "RGB BCHW"),
        (torch.full((1, 3, 2, 2), math.nan), "NaN"),
        (torch.full((1, 3, 2, 2), 1.01), r"\[0, 1\]"),
    ],
)
def test_media_contract_fails_closed(media: Any, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        pickscore_module._pil_images(media, 1)


def test_canonical_media_converts_to_rgb() -> None:
    pytest.importorskip("PIL")
    image = pickscore_module._pil_images(torch.ones(1, 3, 2, 3), 1)[0]
    assert image.mode == "RGB" and image.size == (3, 2)


def test_identity_is_frozen(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scorer_revision"):
        _client(tmp_path, scorer_revision=" ")
    with pytest.raises(ValueError, match="SHA-256"):
        _client(tmp_path, checkpoint_manifest_sha256="bad")


@pytest.mark.parametrize(
    ("directory_name", "manifest_name"),
    [
        ("model", "checkpoint_manifest_sha256"),
        ("processor", "processor_manifest_sha256"),
    ],
)
def test_asset_tree_byte_change_is_rejected(
    tmp_path: Path, directory_name: str, manifest_name: str
) -> None:
    client = _client(tmp_path)
    declared = dict(client.identity)
    asset_directory = tmp_path / directory_name
    asset_file = next(path for path in asset_directory.rglob("*") if path.is_file())
    asset_file.write_bytes(asset_file.read_bytes() + b" changed")

    with pytest.raises(ValueError, match=manifest_name):
        PickScoreRewardClient(
            tmp_path / "model",
            tmp_path / "processor",
            scorer_revision=declared["scorer_revision"],
            checkpoint_manifest_sha256=declared["checkpoint_manifest_sha256"],
            processor_manifest_sha256=declared["processor_manifest_sha256"],
            device="cpu",
        )


def test_sample_identity_must_be_unique(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_id"):
        _client(tmp_path).score(
            torch.zeros(2, 3, 2, 2), ["a", "b"], [{}, {}], sample_id=["same", "same"]
        )


@pytest.mark.parametrize("failure", ["shape", "zero", "nonfinite"])
def test_feature_contract_fails_closed(failure: str) -> None:
    image = torch.ones(2, 2)
    text = torch.ones(2, 2)
    if failure == "shape":
        text = torch.ones(1, 2)
    elif failure == "zero":
        image[0] = 0
    else:
        text[0, 0] = math.inf
    model = SimpleNamespace(
        logit_scale=torch.tensor(0.0),
        get_image_features=lambda **kwargs: image,
        get_text_features=lambda **kwargs: text,
    )
    with pytest.raises(ValueError):
        PickScoreRewardClient._score_features(model, {}, {})
