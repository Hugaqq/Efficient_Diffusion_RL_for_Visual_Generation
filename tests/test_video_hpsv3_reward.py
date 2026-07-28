from __future__ import annotations

import base64
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scripts.serve_video_hpsv3 import build_application
from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.builtins import register_builtin_plugins
from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.core.types import RewardBatch
from visual_rl.feedback.clients import RewardProtocolError
from visual_rl.feedback.video_hpsv3 import (
    VIDEO_HPSV3_PATH,
    VIDEO_HPSV3_PROTOCOL,
    VideoHPSv3JSONApplication,
    VideoHPSv3RewardClient,
    aggregate_flash_grpo_hpsv3,
    video_hpsv3_identity,
    video_hpsv3_runtime_manifest,
    video_hpsv3_runtime_manifest_sha256,
)

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")

_IDENTITY = {
    "scorer_revision": "HPSv3-release",
    "checkpoint_sha256": "3" * 64,
    "runtime_manifest_sha256": "4" * 64,
}


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self.content = json.dumps(payload, allow_nan=True).encode()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Scorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def reward(
        self, *, prompts: list[str], image_paths: list[str]
    ) -> list[list[float]]:
        means = []
        for path in image_paths:
            with PIL.open(path) as image:
                means.append(float(np.asarray(image).mean()))
        self.calls.append(
            {
                "prompts": list(prompts),
                "paths": [Path(path).name for path in image_paths],
                "means": means,
            }
        )
        return [[value / 255.0, 0.0] for value in means]


class _Transport:
    def __init__(self, app: VideoHPSv3JSONApplication, mutate=None) -> None:
        self.app, self.mutate = app, mutate
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    def post(self, url: str, *, data: bytes, **kwargs: Any) -> _Response:
        assert url == f"http://127.0.0.1:8080{VIDEO_HPSV3_PATH}"
        assert kwargs["headers"] == {"Content-Type": "application/json"}
        payload = json.loads(data)
        self.requests.append(payload)
        response = self.app.handle(payload)
        if self.mutate:
            self.mutate(response)
        self.responses.append(response)
        return _Response(response)


def _stack(values: list[list[int]]) -> Any:
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor[:, :, None, None, None].expand(-1, -1, 3, 8, 8) / 255.0


def _client(mutate=None) -> tuple[VideoHPSv3RewardClient, _Scorer, _Transport]:
    scorer = _Scorer()
    app = VideoHPSv3JSONApplication(
        scorer_identity=video_hpsv3_identity(**_IDENTITY), scorer=scorer
    )
    transport = _Transport(app, mutate)
    client = VideoHPSv3RewardClient(
        f"http://127.0.0.1:8080{VIDEO_HPSV3_PATH}",
        transport=transport,
        **_IDENTITY,
    )
    return client, scorer, transport


def _runtime_assets(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "reward-model.safetensors"
    checkpoint.write_bytes(b"checkpoint-v1")
    base_model_root = tmp_path / "base-model"
    base_model_root.mkdir()
    (base_model_root / "config.json").write_text('{"model_type":"qwen2_vl"}')
    source_root = tmp_path / "HPSv3"
    package_root = source_root / "hpsv3"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        "class HPSv3RewardInferencer:\n"
        "    def __init__(self, *, device, config_path, checkpoint_path):\n"
        "        self.device = device\n"
        "        self.config_path = config_path\n"
        "        self.checkpoint_path = checkpoint_path\n"
        "    def reward(self, *, prompts, image_paths):\n"
        "        return [[0.0] for _ in prompts]\n",
        encoding="utf-8",
    )
    config = tmp_path / "HPSv3.yaml"
    config.write_text(
        f"model_name_or_path: {base_model_root}\nrm_head_type: ranknet\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "base_model_root": base_model_root,
        "hps_source_root": source_root,
        "config": config,
    }


@pytest.mark.parametrize("frame_count,selected_count", [(4, 1), (17, 5), (81, 24)])
def test_top_30_percent_matches_flash_release(
    frame_count: int, selected_count: int
) -> None:
    scores = [float((index * 37) % frame_count) for index in range(frame_count)]
    value, selected = aggregate_flash_grpo_hpsv3(scores)
    expected = sorted(scores, reverse=True)[:selected_count]
    assert len(selected) == selected_count
    assert value == math.fsum(expected) / selected_count


def test_top_30_percent_is_stable_and_rejects_bad_input() -> None:
    value, selected = aggregate_flash_grpo_hpsv3([4, 9, 9, 1, 9, 0, -1])
    assert (value, selected) == (9.0, [1, 2])
    with pytest.raises(ValueError, match="at least 4"):
        aggregate_flash_grpo_hpsv3([1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        aggregate_flash_grpo_hpsv3([1, 2, 3, math.nan])


def test_real_client_server_semantics_and_minimal_evidence() -> None:
    client, scorer, transport = _client()
    media = _stack([[0, 31, 63, 95, 127], [159, 191, 223, 239, 255]])

    rewards, details = client.score(
        media,
        ["first prompt", "second prompt"],
        [{}, {}],
        sample_id=["sample-a", "sample-b"],
    )

    assert rewards.shape == (2,)
    assert details.keys() == {
        "protocol",
        "scorer_identity",
        "sample_evidence",
        "valid_mask",
    }
    assert details["protocol"] == VIDEO_HPSV3_PROTOCOL
    assert details["scorer_identity"] == _IDENTITY
    assert [item["sample_id"] for item in details["sample_evidence"]] == [
        "sample-a",
        "sample-b",
    ]
    for reward, item in zip(rewards, details["sample_evidence"], strict=True):
        assert item.keys() == {
            "sample_id",
            "raw_scores",
            "selected_indices",
            "aggregate",
        }
        assert reward == pytest.approx(item["aggregate"])

    assert [len(call["prompts"]) for call in scorer.calls] == [4, 4, 4]
    prompts = [value for call in scorer.calls for value in call["prompts"]]
    assert prompts[:10] == ["first prompt"] * 5 + ["second prompt"] * 5
    assert prompts[10:] == ["second prompt"] * 2
    assert [round(value) for call in scorer.calls for value in call["means"]][:10] == [
        0,
        31,
        63,
        95,
        127,
        159,
        191,
        223,
        239,
        255,
    ]
    assert scorer.calls[-1]["paths"][-2:] == [scorer.calls[-1]["paths"][-3]] * 2

    request = transport.requests[0]
    assert request.keys() == {"protocol", "scorer_identity", "samples"}
    assert [len(item["jpeg_frames"]) for item in request["samples"]] == [5, 5]
    for encoded in request["samples"][0]["jpeg_frames"]:
        data = base64.b64decode(encoded, validate=True)
        with PIL.open(BytesIO(data)) as image:
            assert (image.format, image.mode) == ("JPEG", "RGB")
    response = transport.responses[0]
    assert response.keys() == {"protocol", "scorer_identity", "samples"}
    assert all(
        item.keys() == {"sample_id", "raw_scores"} for item in response["samples"]
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda out: out["scorer_identity"].update(checkpoint_sha256="9" * 64),
            "identity",
        ),
        (
            lambda out: out["scorer_identity"].update(runtime_manifest_sha256="8" * 64),
            "identity",
        ),
        (lambda out: out["samples"][0].update(sample_id="wrong"), "sample_id"),
        (lambda out: out["samples"][0]["raw_scores"].pop(), "count"),
        (
            lambda out: out["samples"][0]["raw_scores"].__setitem__(0, math.nan),
            "raw_scores",
        ),
    ],
)
def test_client_rejects_misaligned_or_invalid_response(mutate, match: str) -> None:
    client, _, _ = _client(mutate)
    with pytest.raises(RewardProtocolError, match=match):
        client.score(_stack([[0, 1, 2, 3]]), ["prompt"], [{}], sample_id=["sample"])


@pytest.mark.parametrize(
    "media,match",
    [
        (torch.zeros(1, 3, 3, 4, 4), "at least 4"),
        (torch.zeros(1, 4, 4, 4, 4), "BFCHW"),
        (torch.full((1, 4, 3, 4, 4), math.nan), "NaN"),
        (torch.full((1, 4, 3, 4, 4), 1.01), r"\[0, 1\]"),
    ],
)
def test_client_rejects_noncanonical_media(media: Any, match: str) -> None:
    client, _, _ = _client()
    with pytest.raises(ValueError, match=match):
        client.score(media, ["prompt"], [{}], sample_id=["sample"])


def test_identity_sample_id_loopback_and_registration_are_required() -> None:
    with pytest.raises(ValueError, match="scorer_revision"):
        video_hpsv3_identity(**(_IDENTITY | {"scorer_revision": ""}))
    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        video_hpsv3_identity(**(_IDENTITY | {"runtime_manifest_sha256": "4" * 63}))
    client, _, _ = _client()
    with pytest.raises(RewardProtocolError, match="explicit sample_id"):
        client.score(_stack([[0, 1, 2, 3]]), ["prompt"], [{}])
    with pytest.raises(ValueError, match="localhost"):
        VideoHPSv3RewardClient(f"http://example.com{VIDEO_HPSV3_PATH}", **_IDENTITY)
    register_builtin_plugins()
    assert REWARD_CLIENTS.get("video_hpsv3") is VideoHPSv3RewardClient


def test_manifest_keeps_scorer_identity_and_per_sample_evidence() -> None:
    evidence = {
        "sample_id": "sample-a",
        "raw_scores": [0.1, 0.4, 0.2, 0.3],
        "selected_indices": [1],
        "aggregate": 0.4,
    }
    rewards = RewardBatch(
        raw={"video_hpsv3": torch.tensor([0.4])},
        weighted={"video_hpsv3": torch.tensor([0.4])},
        weighted_total=torch.tensor([0.4]),
        valid_mask=torch.tensor([True]),
        metadata={
            "video_hpsv3": {
                "protocol": VIDEO_HPSV3_PROTOCOL,
                "scorer_identity": _IDENTITY,
                "sample_evidence": [evidence],
            }
        },
        sample_id=["sample-a"],
    )

    values = ManifestBuilder._reward_values(rewards, 0)

    assert values["provenance"]["video_hpsv3"] == {
        "protocol": VIDEO_HPSV3_PROTOCOL,
        "scorer_identity": _IDENTITY,
        "sample": evidence,
    }


def test_runtime_manifest_canonically_binds_all_inference_assets(
    tmp_path: Path,
) -> None:
    assets = _runtime_assets(tmp_path)
    components = video_hpsv3_runtime_manifest(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )
    assert components.keys() == {
        "checkpoint_file_sha256",
        "hpsv3_package_tree_sha256",
        "config_file_sha256",
        "base_model_tree_sha256",
    }
    assert all(len(value) == 64 for value in components.values())
    assert (
        len(
            video_hpsv3_runtime_manifest_sha256(
                checkpoint_path=assets["checkpoint"],
                hps_source_root=assets["hps_source_root"],
                config_path=assets["config"],
                base_model_root=assets["base_model_root"],
            )
        )
        == 64
    )


@pytest.mark.parametrize(
    "asset,relative",
    [
        ("checkpoint", None),
        ("config", None),
        ("hps_source_root", "hpsv3/__init__.py"),
        ("base_model_root", "config.json"),
    ],
)
def test_runtime_asset_byte_mutation_is_rejected_before_scorer_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset: str,
    relative: str | None,
) -> None:
    assets = _runtime_assets(tmp_path)
    checkpoint_sha256 = video_hpsv3_runtime_manifest(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )["checkpoint_file_sha256"]
    runtime_sha256 = video_hpsv3_runtime_manifest_sha256(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )
    target = assets[asset] / relative if relative else assets[asset]
    target.write_bytes(target.read_bytes() + b"\nmutated")
    monkeypatch.delitem(sys.modules, "hpsv3", raising=False)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_application(
            checkpoint=assets["checkpoint"],
            checkpoint_sha256=checkpoint_sha256,
            runtime_manifest_sha256=runtime_sha256,
            scorer_revision="release",
            config=assets["config"],
            base_model_root=assets["base_model_root"],
            hps_source_root=assets["hps_source_root"],
            device="cpu",
        )
    assert "hpsv3" not in sys.modules


def test_launcher_binds_config_source_and_inferencer_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _runtime_assets(tmp_path)
    components = video_hpsv3_runtime_manifest(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )
    runtime_sha256 = video_hpsv3_runtime_manifest_sha256(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )
    monkeypatch.delitem(sys.modules, "hpsv3", raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    app = build_application(
        checkpoint=assets["checkpoint"],
        checkpoint_sha256=components["checkpoint_file_sha256"],
        runtime_manifest_sha256=runtime_sha256,
        scorer_revision="release",
        config=assets["config"],
        base_model_root=assets["base_model_root"],
        hps_source_root=assets["hps_source_root"],
        device="cpu",
    )

    assert app.identity == {
        "scorer_revision": "release",
        "checkpoint_sha256": components["checkpoint_file_sha256"],
        "runtime_manifest_sha256": runtime_sha256,
    }
    assert Path(app.scorer.config_path) == assets["config"]
    assert Path(app.scorer.checkpoint_path) == assets["checkpoint"]
    assert app.scorer.device == "cpu"


def test_launcher_rejects_wrong_base_model_declaration(tmp_path: Path) -> None:
    assets = _runtime_assets(tmp_path)
    other_base = tmp_path / "other-base"
    other_base.mkdir()
    assets["config"].write_text(f"model_name_or_path: {other_base}\n", encoding="utf-8")
    components = video_hpsv3_runtime_manifest(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )
    runtime_sha256 = video_hpsv3_runtime_manifest_sha256(
        checkpoint_path=assets["checkpoint"],
        hps_source_root=assets["hps_source_root"],
        config_path=assets["config"],
        base_model_root=assets["base_model_root"],
    )
    with pytest.raises(ValueError, match="model_name_or_path"):
        build_application(
            checkpoint=assets["checkpoint"],
            checkpoint_sha256=components["checkpoint_file_sha256"],
            runtime_manifest_sha256=runtime_sha256,
            scorer_revision="release",
            config=assets["config"],
            base_model_root=assets["base_model_root"],
            hps_source_root=assets["hps_source_root"],
            device="cpu",
        )
