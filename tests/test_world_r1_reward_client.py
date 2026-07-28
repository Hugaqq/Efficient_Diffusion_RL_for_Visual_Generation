"""CPU/offline tests for the World-R1 reward wire protocol."""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import errno
import importlib.util
import json
from pathlib import Path
import pickle
import struct
import traceback
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
import yaml

import visual_rl.feedback.clients as feedback_clients
from scripts import legacy_cli
from scripts.world_r1_reward_probe import (
    WorldR1RewardServerProbeConfig,
    run_world_r1_reward_server_probe,
)
from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.feedback.clients import (
    JSON_V1,
    LEGACY_PICKLE,
    RemotePickleRewardClient,
    RewardProtocolError,
    RewardTransportError,
    redact_error_text,
    redact_url,
    requests_session,
)
from visual_rl.feedback.factory import build_feedback_provider
from visual_rl.feedback.router import RewardRouter
import visual_rl.feedback.world_r1_rewards as world_r1_rewards
from visual_rl.feedback.world_r1_rewards import (
    SCORE_META_VIEW,
    SCORE_RECONSTRUCTION,
    SCORE_TRAJECTORY_ALIGNMENT,
    TRAJECTORY_COMPARISON_PATHS,
    WorldR1Reward3DClient,
    WorldR1RewardGeneralClient,
)


@pytest.fixture(autouse=True)
def _jpeg_backend(monkeypatch):
    if importlib.util.find_spec("PIL") is not None:
        return

    def encode_shape(image):
        height, width, channels = image.shape
        return b"TESTJPEG" + struct.pack(">III", width, height, channels)

    monkeypatch.setattr(world_r1_rewards, "_rgb_jpeg", encode_shape)


@dataclass
class FakeResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str] | None = None
    chunks: tuple[bytes, ...] | None = None
    closed: bool = False
    iterated_chunks: int = 0
    content_accessed: bool = False

    @property
    def content(self):
        self.content_accessed = True
        raise AssertionError("bounded HTTP code accessed full response.content")

    def iter_content(self, *, chunk_size):
        chunks = self.chunks
        if chunks is None:
            chunks = tuple(
                self.body[start : start + chunk_size]
                for start in range(0, len(self.body), chunk_size)
            )
        for chunk in chunks:
            self.iterated_chunks += 1
            yield chunk

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code < 200 or self.status_code >= 300:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeTransport:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def post(self, url, *, data, timeout, headers, allow_redirects, stream):
        if headers["Content-Type"] == "application/json":
            wire_payload = json.loads(data)
            payload = dict(wire_payload)
            if "images" in payload:
                payload["images"] = [
                    base64.b64decode(item) for item in payload["images"]
                ]
            if "videos" in payload:
                payload["videos"] = [
                    [base64.b64decode(frame) for frame in video]
                    for video in payload["videos"]
                ]
        else:
            wire_payload = None
            payload = pickle.loads(data)
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "wire_payload": wire_payload,
                "timeout": timeout,
                "headers": headers,
                "allow_redirects": allow_redirects,
                "stream": stream,
            }
        )
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class MaliciousArray:
    def __array__(self, *_args, **_kwargs):
        raise RuntimeError(
            "https://user:password@reward.example/private?token=SECRET#signature"
        )


def _fake_requests_module():
    class RequestException(Exception):
        pass

    class Timeout(RequestException):
        pass

    class RequestsConnectionError(RequestException):
        pass

    class InvalidURL(RequestException):
        pass

    class InvalidSchema(RequestException):
        pass

    class MissingSchema(RequestException):
        pass

    class TooManyRedirects(RequestException):
        pass

    class SSLError(RequestsConnectionError):
        pass

    return SimpleNamespace(
        exceptions=SimpleNamespace(
            RequestException=RequestException,
            Timeout=Timeout,
            ConnectionError=RequestsConnectionError,
            InvalidURL=InvalidURL,
            InvalidSchema=InvalidSchema,
            MissingSchema=MissingSchema,
            TooManyRedirects=TooManyRedirects,
            SSLError=SSLError,
        )
    )


def _response(payload, status=200, *, wire_format=JSON_V1):
    if wire_format == LEGACY_PICKLE:
        content = pickle.dumps(payload)
    else:
        content = json.dumps(payload).encode("utf-8")
    return FakeResponse(status, content)


def _legacy_kwargs(host: str = "localhost") -> dict[str, object]:
    return {
        "wire_format": LEGACY_PICKLE,
        "allow_unsafe_pickle": True,
        "trusted_hosts": [host],
    }


def _jpeg_size(data):
    if data.startswith(b"TESTJPEG"):
        width, height, channels = struct.unpack(">III", data[8:20])
        return (width, height), "RGB" if channels == 3 else "INVALID"

    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        return image.size, image.mode


def _camera_matrix(translation: float = 0.0) -> str:
    return f"[1 0 0 0] [0 1 0 0] [0 0 1 0] [{translation:.6g} 0 0 1]"


def _camera_trajectory(frames: int, *, offset: float = 0.0) -> dict[str, str]:
    return {
        f"frame{index}": _camera_matrix(offset + index * 0.001)
        for index in range(frames)
    }


def _minwm_camera_trajectory(frames: int) -> dict[str, object]:
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    viewmats[:, 0, 3] = np.arange(frames, dtype=np.float32) / 10.0
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0)
    return {
        "viewmats": viewmats.tolist(),
        "Ks": intrinsics.tolist(),
        "convention": "w2c",
        "coordinate_system": "opencv",
    }


def test_default_requests_session_disables_environment_proxies(monkeypatch):
    session = SimpleNamespace(trust_env=True)
    monkeypatch.setitem(
        __import__("sys").modules,
        "requests",
        SimpleNamespace(Session=lambda: session),
    )

    assert requests_session() is session
    assert session.trust_env is False


@pytest.mark.parametrize("encoding_rounds", [1, 3, 5, 8])
def test_url_and_error_redaction_remove_deeply_encoded_url_secrets(
    encoding_rounds,
):
    url = (
        "HTTPS://user:password@Reward.Example:8443/private/TOPSECRET/"
        "%3Ftoken%3DENCODEDSECRET?api_key=QUERYSECRET&x=1#fragment-value"
    )
    encoded_url = url
    for _ in range(encoding_rounds):
        encoded_url = quote(encoded_url, safe="")

    displayed = redact_url(url)
    error = redact_error_text(
        RuntimeError(f"plain={url} encoded={encoded_url} failed: " + "x" * 1000)
    )

    for value in (
        "user",
        "password",
        "private",
        "TOPSECRET",
        "ENCODEDSECRET",
        "QUERYSECRET",
        "=1",
        "fragment-value",
    ):
        assert value not in displayed
        assert value not in error
    assert displayed == "https://reward.example:8443/[REDACTED]"
    assert "TOPSECRET" not in redact_url(encoded_url)
    assert "QUERYSECRET" not in redact_url(encoded_url)
    assert "plain=" not in error
    assert len(error) <= 500


def test_error_redaction_removes_relative_urls_and_sensitive_values():
    encoded_relative_url = quote(
        quote("/score?password=ENCODEDSECRET", safe=""), safe=""
    )
    error = redact_error_text(
        "MissingSchema for reward.example with url: "
        "/score?token=TOPSECRET&api_key=QUERYSECRET&x=1#signature=SIGSECRET "
        "authorization: Bearer AUTHSECRET cookie=COOKIESECRET "
        "password='PASSWORD SECRET' "
        "headers={'Authorization': 'Bearer DICTAUTHSECRET'} "
        f"encoded={encoded_relative_url}"
    )

    assert "MissingSchema" in error
    assert "reward.example" in error
    for value in (
        "TOPSECRET",
        "QUERYSECRET",
        "SIGSECRET",
        "AUTHSECRET",
        "COOKIESECRET",
        "PASSWORD SECRET",
        "DICTAUTHSECRET",
        "ENCODEDSECRET",
    ):
        assert value not in error
    assert len(error) <= 500


def test_reference_general_payload_is_jpeg_and_uses_fixed_middle_frame():
    media = np.zeros((2, 5, 3, 4, 6), dtype=np.float32)
    media[:, 2, 0] = 1.0
    transport = FakeTransport(
        _response({"outputs": [0.25, 0.75]}),
        _response({"outputs": [0.25, 0.75]}),
    )
    client = WorldR1RewardGeneralClient(
        "http://127.0.0.1:8090/", transport=transport, media_layout="BFCHW"
    )

    first, metadata = client.score(media, ["one", "two"], [{}, {}])
    second, _ = client.score(media, ["one", "two"], [{}, {}])

    assert first.tolist() == pytest.approx([0.25, 0.75])
    assert second.tolist() == pytest.approx(first.tolist())
    assert transport.calls[0]["payload"].keys() == {"images", "prompts"}
    assert (
        transport.calls[0]["payload"]["images"]
        == transport.calls[1]["payload"]["images"]
    )
    assert [_jpeg_size(item) for item in transport.calls[0]["payload"]["images"]] == [
        ((6, 4), "RGB"),
        ((6, 4), "RGB"),
    ]
    assert metadata["encoding"]["selected_frame_index"] == 2
    assert metadata["server_revision"] is None
    assert metadata["configured_batch_size"] == 64
    assert metadata["request_count"] == 1
    assert metadata["payload_batch_sizes"] == [2]
    assert metadata["identity_mode"] == "trusted_input_order"
    assert metadata["server_identity_echo"] is False
    assert "sample_id" not in metadata


def test_default_json_wire_base64_encodes_jpegs_and_never_loads_pickle(monkeypatch):
    transport = FakeTransport(_response({"outputs": [0.5]}))
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090", transport=transport, media_layout="BCHW"
    )
    monkeypatch.setattr(
        world_r1_rewards.pickle,
        "loads",
        lambda *_args, **_kwargs: pytest.fail("json_v1 called pickle.loads"),
    )

    values, _ = client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    call = transport.calls[0]
    encoded = call["wire_payload"]["images"][0]
    assert base64.b64decode(encoded) == call["payload"]["images"][0]
    assert call["headers"] == {"Content-Type": "application/json"}
    assert call["allow_redirects"] is False
    assert call["stream"] is True
    assert values.tolist() == [0.5]


def test_uint8_channel_last_and_torch_channel_first_are_supported():
    torch = pytest.importorskip("torch")
    numpy_transport = FakeTransport(_response({"outputs": [1.0]}))
    numpy_client = WorldR1RewardGeneralClient(
        "http://localhost:8090", transport=numpy_transport, media_layout="BHWC"
    )
    numpy_client.score(np.zeros((1, 4, 5, 3), dtype=np.uint8), ["p"], [{}])
    assert _jpeg_size(numpy_transport.calls[0]["payload"]["images"][0]) == (
        (5, 4),
        "RGB",
    )

    torch_transport = FakeTransport(_response({"outputs": [2.0]}))
    torch_client = WorldR1RewardGeneralClient(
        "http://localhost:8090", transport=torch_transport, media_layout="BCHW"
    )
    torch_client.score(torch.zeros(1, 3, 4, 5, dtype=torch.uint8), ["p"], [{}])
    assert _jpeg_size(torch_transport.calls[0]["payload"]["images"][0]) == (
        (5, 4),
        "RGB",
    )


@pytest.mark.parametrize(
    ("media", "match"),
    [
        (np.zeros((1, 3, 4, 3), dtype=np.uint8), "unambiguous"),
        (np.zeros((1, 4, 4, 4), dtype=np.uint8), "unambiguous"),
        (np.zeros((1, 3, 4, 4), dtype=np.int16), "only uint8"),
        (np.full((1, 3, 4, 4), 1.1, dtype=np.float32), r"\[0, 1\]"),
    ],
)
def test_media_layout_dtype_and_range_are_strict(media, match):
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090", transport=FakeTransport()
    )
    with pytest.raises((TypeError, ValueError), match=match):
        client.prepare_payloads(media, ["p"], [{}])


def test_reference_3d_payload_camera_trajectory_and_details_are_batch_aligned():
    details = [
        {
            "video_id": 1,
            "gs_score": 0.4,
            "meta_score": 0.5,
            "camera_motion_score": 0.6,
            "trajectory_comparison_path": None,
        },
        {
            "video_id": 0,
            "gs_score": 0.1,
            "meta_score": 0.2,
            "camera_motion_score": 0.3,
            "trajectory_comparison_path": "/tmp/a.json",
        },
    ]
    transport = FakeTransport(_response({"outputs": [1.0, 2.0], "details": details}))
    client = WorldR1Reward3DClient(
        "http://localhost:8089", transport=transport, media_layout="BFHWC"
    )
    trajectories = [
        _camera_trajectory(3),
        _camera_trajectory(3, offset=0.01),
    ]
    metadata = [{"camera_trajectory": item} for item in trajectories]

    values, result = client.score(
        np.zeros((2, 3, 4, 5, 3), dtype=np.uint8), ["a", "b"], metadata
    )

    payload = transport.calls[0]["payload"]
    assert payload.keys() == {"videos", "prompts", "camera_trajectories"}
    assert payload["camera_trajectories"] == trajectories
    assert [[_jpeg_size(frame) for frame in video] for video in payload["videos"]] == [
        [((5, 4), "RGB")] * 3,
        [((5, 4), "RGB")] * 3,
    ]
    assert values.tolist() == pytest.approx([1.0, 2.0])
    assert result[SCORE_RECONSTRUCTION] == pytest.approx([0.1, 0.4])
    assert result[SCORE_META_VIEW] == pytest.approx([0.2, 0.5])
    assert result[SCORE_TRAJECTORY_ALIGNMENT] == pytest.approx([0.3, 0.6])
    assert result[TRAJECTORY_COMPARISON_PATHS] == ["/tmp/a.json", ""]


def test_reward_3d_aligns_77_decoded_frames_with_20_minwm_camera_poses():
    selected = list(range(0, 77, 4))
    media = np.zeros((1, 77, 1, 1, 3), dtype=np.uint8)
    media[0, :, 0, 0, :] = np.arange(77, dtype=np.uint8)[:, None]
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BFHWC",
        protocol_mode="strict_v2",
        server_revision="world-r1-test",
        require_camera_trajectory=True,
        frame_indices=selected,
        jpeg_encoder=lambda frame: bytes([int(frame[0, 0, 0])]),
    )

    metadata = {
        "camera_trajectory": _minwm_camera_trajectory(20),
        "minwm_reward_frame_alignment": {
            "contract": "minwm_vae_camera_alignment_v1",
            "latent_frames": 20,
            "decoded_media_frames": 77,
            "vae_temporal_stride": 4,
        },
    }
    payloads, evidence = client.prepare_payloads(
        media,
        ["move through the scene"],
        [metadata],
        sample_id=["sample-0"],
    )

    assert payloads[0]["videos"] == [[bytes([index]) for index in selected]]
    assert list(payloads[0]["camera_trajectories"][0]) == [
        f"frame{index}" for index in range(20)
    ]
    assert evidence["selected_frame_indices"] == selected
    fingerprint = client.cache_fingerprint()
    assert fingerprint["frame_policy"]["selected_frame_indices"] == selected
    assert fingerprint["camera_conversion"]["input_pose_convention"] == "w2c"

    wrong = selected.copy()
    wrong[1] = 5
    misaligned = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BFHWC",
        protocol_mode="strict_v2",
        server_revision="world-r1-test",
        require_camera_trajectory=True,
        frame_indices=wrong,
        jpeg_encoder=lambda frame: bytes([int(frame[0, 0, 0])]),
    )
    with pytest.raises(ValueError, match="must match MinWM VAE camera alignment"):
        misaligned.prepare_payloads(
            media,
            ["move through the scene"],
            [metadata],
            sample_id=["sample-0"],
        )


def test_released_reference_3d_details_support_trusted_order_contract():
    details = [
        {
            "gs_score": 0.1,
            "meta_score": 0.2,
            "camera_motion_score": 0.3,
            "final_score": 0.6,
            "gs_video_path": "reconstruction.mp4",
            "meta_view_path": "meta.png",
            "trajectory_comparison_path": "trajectory.png",
        }
    ]
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(_response({"outputs": [0.6], "details": details})),
        media_layout="BFCHW",
    )

    values, metadata = client.score(
        np.zeros((1, 2, 3, 4, 5), dtype=np.uint8),
        ["prompt"],
        [{}],
    )

    assert values.tolist() == pytest.approx([0.6])
    assert metadata[SCORE_RECONSTRUCTION] == pytest.approx([0.1])
    assert metadata[SCORE_META_VIEW] == pytest.approx([0.2])
    assert metadata[SCORE_TRAJECTORY_ALIGNMENT] == pytest.approx([0.3])
    assert metadata[TRAJECTORY_COMPARISON_PATHS] == ["trajectory.png"]


def _detail(video_id):
    return {
        "video_id": video_id,
        "gs_score": 0.1,
        "meta_score": 0.2,
        "camera_motion_score": 0.3,
        "trajectory_comparison_path": "",
    }


@pytest.mark.parametrize(
    ("details", "batch_size", "match"),
    [
        ([_detail(0), _detail(0)], 2, "duplicate video_id"),
        ([_detail(0)], 2, "details length"),
        ([_detail(0), _detail(2)], 2, "must be in 0..1"),
        ([_detail(0), _detail("1")], 2, "must be an integer"),
    ],
    ids=["duplicate", "missing", "out-of-range", "non-integer"],
)
def test_3d_response_video_ids_must_be_a_complete_integer_permutation(
    details, batch_size, match
):
    transport = FakeTransport(
        _response({"outputs": [0.5] * batch_size, "details": details})
    )
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=transport,
        media_layout="BFCHW",
    )
    with pytest.raises(RewardProtocolError, match=match):
        client.score(
            np.zeros((batch_size, 2, 3, 4, 5), dtype=np.uint8),
            ["p"] * batch_size,
            [{} for _ in range(batch_size)],
        )


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("BCHW", (2, 3, 4, 5)),
        ("BHWC", (2, 4, 5, 3)),
    ],
)
def test_reward_3d_promotes_4d_images_to_single_frame_videos(layout, shape):
    transport = FakeTransport(_response({"outputs": [0.1, 0.2]}))
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=transport,
        media_layout=layout,
    )

    _, metadata = client.score(
        np.zeros(shape, dtype=np.uint8),
        ["a", "b"],
        [{}, {}],
    )

    videos = transport.calls[0]["payload"]["videos"]
    assert [len(video) for video in videos] == [1, 1]
    assert [_jpeg_size(video[0]) for video in videos] == [((5, 4), "RGB")] * 2
    assert metadata["encoding"]["frames_per_video"] == 1
    assert metadata["encoding"]["promoted_single_frame"] is True


@pytest.mark.parametrize(
    ("layout", "shape", "match"),
    [
        ("auto", (1, 3, 4, 3), "unambiguous"),
        ("BFCHW", (1, 3, 4, 5), "requires 5 dimensions"),
        ("BCHW", (1, 1, 4, 5), "exactly 3 RGB channels"),
    ],
)
def test_reward_3d_4d_layout_validation_is_strict(layout, shape, match):
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout=layout,
    )
    with pytest.raises(ValueError, match=match):
        client.prepare_payloads(np.zeros(shape, dtype=np.uint8), ["p"], [{}])


@pytest.mark.parametrize(
    "trajectory",
    [None, {}, [], "not-a-mapping"],
)
def test_reward_3d_can_require_non_empty_camera_trajectory_mapping(trajectory):
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BCHW",
        require_camera_trajectory=True,
    )
    with pytest.raises(
        ValueError, match="camera_trajectory must be a non-empty mapping"
    ):
        client.prepare_payloads(
            np.zeros((1, 3, 4, 5), dtype=np.uint8),
            ["p"],
            [{"camera_trajectory": trajectory}],
        )

    payloads, _ = client.prepare_payloads(
        np.zeros((1, 3, 4, 5), dtype=np.uint8),
        ["p"],
        [{"camera_trajectory": _camera_trajectory(1)}],
    )
    assert payloads[0]["camera_trajectories"][0]


class _DuplicateFrameMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        if key != "frame0":
            raise KeyError(key)
        return _camera_matrix()

    def __iter__(self) -> Iterator[str]:
        return iter(("frame0", "frame0"))

    def __len__(self) -> int:
        return 2


@pytest.mark.parametrize(
    ("trajectory", "frames", "match"),
    [
        (
            {"frame0": _camera_matrix(), "frame2": _camera_matrix()},
            2,
            "missing=.*frame1.*extra=.*frame2",
        ),
        (_DuplicateFrameMapping(), 1, "duplicate=.*frame0"),
        ({"frame0": "[1 0 0 0]"}, 1, "finite 4x4 matrix"),
        (
            {
                "frame0": "[1 0 0 0] [0 1 0 0] [0 0 nan 0] [0 0 0 1]",
            },
            1,
            "finite 4x4 matrix",
        ),
    ],
    ids=[
        "non-contiguous-frame",
        "duplicate-frame",
        "bad-matrix",
        "nan-matrix",
    ],
)
def test_reward_3d_camera_trajectory_is_frame_exact_and_finite(
    trajectory, frames, match
):
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BFCHW",
        require_camera_trajectory=True,
    )

    with pytest.raises(ValueError, match=match):
        client.prepare_payloads(
            np.zeros((1, frames, 3, 4, 5), dtype=np.uint8),
            ["p"],
            [{"camera_trajectory": trajectory}],
        )


def test_reference_allows_trajectory_resampling_but_strict_requires_frame_count():
    media = np.zeros((1, 3, 3, 4, 5), dtype=np.uint8)
    metadata = [{"camera_trajectory": _camera_trajectory(4)}]
    reference = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BFCHW",
        protocol_mode="reference_v1",
        require_camera_trajectory=True,
    )
    strict = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BFCHW",
        protocol_mode="strict_v2",
        require_camera_trajectory=True,
    )

    payloads, _ = reference.prepare_payloads(media, ["p"], metadata)
    assert list(payloads[0]["camera_trajectories"][0]) == [
        "frame0",
        "frame1",
        "frame2",
        "frame3",
    ]
    with pytest.raises(ValueError, match="must contain 3 frames"):
        strict.prepare_payloads(media, ["p"], metadata, sample_id=["sample"])


def test_reward_3d_validates_optional_non_none_camera_trajectory():
    client = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BFCHW",
        require_camera_trajectory=False,
    )

    with pytest.raises(ValueError, match="finite 4x4 matrix"):
        client.prepare_payloads(
            np.zeros((1, 1, 3, 4, 5), dtype=np.uint8),
            ["p"],
            [{"camera_trajectory": {"frame0": "[1 0 0 0]"}}],
        )

    payloads, _ = client.prepare_payloads(
        np.zeros((1, 1, 3, 4, 5), dtype=np.uint8),
        ["p"],
        [{}],
    )
    assert payloads[0]["camera_trajectories"] == [None]


def test_general_rounds_but_3d_truncates_float_pixels(monkeypatch):
    encoded_values = []

    def encoder(image):
        encoded_values.append(int(image[0, 0, 0]))
        return bytes([encoded_values[-1]])

    monkeypatch.setattr(world_r1_rewards, "_rgb_jpeg", encoder)
    general = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
        media_layout="BCHW",
        server_revision="quantization-general-v1",
    )
    reward_3d = WorldR1Reward3DClient(
        "http://localhost:8089",
        transport=FakeTransport(),
        media_layout="BCHW",
        server_revision="quantization-3d-v1",
    )
    media = np.full((1, 3, 2, 2), 0.5, dtype=np.float32)

    general_payloads, _ = general.prepare_payloads(media, ["p"], [{}])
    reward_3d_payloads, _ = reward_3d.prepare_payloads(media, ["p"], [{}])

    assert general_payloads[0]["images"] == [bytes([128])]
    assert reward_3d_payloads[0]["videos"] == [[bytes([127])]]
    assert general.cache_fingerprint()["float_quantization"] == "round_to_nearest_even"
    assert reward_3d.cache_fingerprint()["float_quantization"] == "truncate"


def test_strict_v2_requires_echoed_identity_version_and_boolean_mask():
    sample_id = ["sample-a", "sample-b"]
    transport = FakeTransport(
        _response(
            {
                "outputs": [0.1, 0.2],
                "protocol_version": "strict_v2",
                "sample_id": sample_id,
                "valid_mask": [True, False],
            }
        )
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        protocol_mode="strict_v2",
        media_layout="BCHW",
    )

    _, metadata = client.score(
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
        ["a", "b"],
        [{}, {}],
        sample_id=sample_id,
    )

    payload = transport.calls[0]["payload"]
    assert payload["protocol_version"] == "strict_v2"
    assert payload["sample_id"] == sample_id
    assert metadata["sample_id"] == sample_id
    assert metadata["valid_mask"] == [True, False]
    assert metadata["identity_mode"] == "server_echo"
    assert metadata["server_revision_echo"] is False


@pytest.mark.parametrize("echoed_revision", [None, "other-revision"])
def test_strict_v2_rejects_missing_or_mismatched_server_revision(echoed_revision):
    sample_id = ["sample-a"]
    response = {
        "outputs": [0.1],
        "protocol_version": "strict_v2",
        "sample_id": sample_id,
        "valid_mask": [True],
    }
    if echoed_revision is not None:
        response["server_revision"] = echoed_revision
    transport = FakeTransport(_response(response))
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        protocol_mode="strict_v2",
        media_layout="BCHW",
        server_revision="expected-revision",
    )

    with pytest.raises(RewardProtocolError, match="server_revision"):
        client.score(
            np.zeros((1, 3, 4, 5), dtype=np.uint8),
            ["prompt"],
            [{}],
            sample_id=sample_id,
        )

    assert transport.calls[0]["payload"]["server_revision"] == "expected-revision"


def test_strict_v2_accepts_and_records_matching_server_revision():
    sample_id = ["sample-a"]
    transport = FakeTransport(
        _response(
            {
                "outputs": [0.1],
                "protocol_version": "strict_v2",
                "sample_id": sample_id,
                "valid_mask": [True],
                "server_revision": "expected-revision",
            }
        )
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        protocol_mode="strict_v2",
        media_layout="BCHW",
        server_revision="expected-revision",
    )

    _, metadata = client.score(
        np.zeros((1, 3, 4, 5), dtype=np.uint8),
        ["prompt"],
        [{}],
        sample_id=sample_id,
    )

    assert metadata["server_revision"] == "expected-revision"
    assert metadata["server_revision_echo"] is True


def test_strict_v2_chunks_five_items_as_two_two_one_and_preserves_order():
    sample_id = [f"sample-{index}" for index in range(5)]
    prompts = [f"prompt-{index}" for index in range(5)]
    transport = FakeTransport(
        _response(
            {
                "outputs": [0.0, 1.0],
                "protocol_version": "strict_v2",
                "sample_id": sample_id[:2],
                "valid_mask": [True, True],
                "server_revision": "general-hps-v2.1-sha256:fixture",
            }
        ),
        _response(
            {
                "outputs": [2.0, 3.0],
                "protocol_version": "strict_v2",
                "sample_id": sample_id[2:4],
                "valid_mask": [True, True],
                "server_revision": "general-hps-v2.1-sha256:fixture",
            }
        ),
        _response(
            {
                "outputs": [4.0],
                "protocol_version": "strict_v2",
                "sample_id": sample_id[4:],
                "valid_mask": [True],
                "server_revision": "general-hps-v2.1-sha256:fixture",
            }
        ),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        protocol_mode="strict_v2",
        media_layout="BCHW",
        batch_size=2,
        server_revision="general-hps-v2.1-sha256:fixture",
    )

    values, metadata = client.score(
        np.zeros((5, 3, 4, 5), dtype=np.uint8),
        prompts,
        [{} for _ in prompts],
        sample_id=sample_id,
    )

    assert values.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert [call["payload"]["prompts"] for call in transport.calls] == [
        prompts[:2],
        prompts[2:4],
        prompts[4:],
    ]
    assert [call["payload"]["sample_id"] for call in transport.calls] == [
        sample_id[:2],
        sample_id[2:4],
        sample_id[4:],
    ]
    assert metadata["sample_id"] == sample_id
    assert metadata["server_revision"] == "general-hps-v2.1-sha256:fixture"
    assert metadata["configured_batch_size"] == 2
    assert metadata["request_count"] == 3
    assert metadata["payload_batch_sizes"] == [2, 2, 1]


def test_strict_v2_and_router_reject_duplicate_sample_identity():
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
        protocol_mode="strict_v2",
        media_layout="BCHW",
    )
    media = np.zeros((2, 3, 4, 5), dtype=np.uint8)

    with pytest.raises(RewardProtocolError, match="must be unique"):
        client.prepare_payloads(
            media,
            ["a", "b"],
            [{}, {}],
            sample_id=["duplicate", "duplicate"],
        )

    router = RewardRouter({"weights": {}})
    with pytest.raises(ValueError, match="must be unique"):
        router.score(
            media,
            ["a", "b"],
            [{}, {}],
            sample_id=["duplicate", "duplicate"],
        )


@pytest.mark.parametrize(
    "response",
    [
        {"outputs": [1.0], "sample_id": ["sample"], "valid_mask": [True]},
        {
            "outputs": [1.0],
            "protocol_version": "strict_v2",
            "sample_id": ["wrong"],
            "valid_mask": [True],
        },
        {
            "outputs": [1.0],
            "protocol_version": "strict_v2",
            "sample_id": ["sample"],
            "valid_mask": [1],
        },
    ],
)
def test_strict_v2_identity_and_mask_fail_closed(response):
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(_response(response)),
        protocol_mode="strict_v2",
        media_layout="BCHW",
    )
    with pytest.raises(RewardProtocolError):
        client.score(
            np.zeros((1, 3, 4, 5), dtype=np.uint8),
            ["p"],
            [{}],
            sample_id=["sample"],
        )


def test_transport_retries_only_retryable_failures_with_exponential_backoff():
    sleeps = []
    transport = FakeTransport(
        FakeResponse(503, b"busy"),
        TimeoutError("slow"),
        _response({"outputs": [0.5]}),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        retries=2,
        backoff_seconds=0.1,
        sleep=sleeps.append,
        media_layout="BCHW",
    )

    values, _ = client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert values.tolist() == pytest.approx([0.5])
    assert len(transport.calls) == 3
    assert sleeps == pytest.approx([0.1, 0.2])

    no_retry = FakeTransport(FakeResponse(400, b"bad"), _response({"outputs": [1.0]}))
    bad_client = WorldR1RewardGeneralClient(
        "http://localhost:8090", transport=no_retry, retries=2, media_layout="BCHW"
    )
    with pytest.raises(RewardTransportError, match="HTTP 400"):
        bad_client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])
    assert len(no_retry.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("injected timeout"),
        ConnectionError("injected connection failure"),
        OSError(errno.ECONNRESET, "connection reset"),
    ],
    ids=["timeout", "connection-error", "transient-oserror"],
)
def test_injected_transient_transport_errors_are_retried(error):
    transport = FakeTransport(error, _response({"outputs": [0.75]}))
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        retries=1,
        backoff_seconds=0.001,
        sleep=lambda _delay: None,
        media_layout="BCHW",
    )

    values, _ = client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert values.tolist() == pytest.approx([0.75])
    assert len(transport.calls) == 2


def test_non_transient_oserror_is_wrapped_without_retry():
    transport = FakeTransport(
        OSError(errno.EINVAL, "invalid transport operation"),
        _response({"outputs": [1.0]}),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        retries=2,
        media_layout="BCHW",
    )

    with pytest.raises(RewardTransportError, match="permanent transport error"):
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])
    assert len(transport.calls) == 1


def test_router_failure_metadata_is_structured_redacted_and_bounded(tmp_path):
    url = "https://reward.example/score?token=TOPSECRET&x=1#fragment-value"
    transport = FakeTransport(OSError(errno.EINVAL, f"request to {url} failed"))
    router = RewardRouter(
        {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": url,
                    "media_layout": "BCHW",
                    "transport": transport,
                }
            },
            "fail_policy": "invalid",
        },
        cache_dir=tmp_path,
    )

    result = router.score(
        np.zeros((1, 3, 4, 5), dtype=np.uint8),
        ["p"],
        [{}],
        sample_id=["sample"],
    )

    failure = result.metadata["reward_general"]
    assert failure["error_type"] == "RewardTransportError"
    assert len(failure["error"]) <= 500
    for value in ("TOPSECRET", "=1", "fragment-value"):
        assert value not in failure["error"]
    assert result.weighted_total.tolist() == [0.0]
    assert result.valid_mask.tolist() == [False]
    assert transport.calls[0]["url"] == url


def test_router_failure_metadata_redacts_relative_url_secrets(tmp_path):
    message = (
        "MissingSchema for reward.example with url: "
        "/score?token=TOPSECRET&api_key=QUERYSECRET&x=1#signature=SIGSECRET "
        "authorization=Bearer AUTHSECRET cookie=COOKIESECRET"
    )
    transport = FakeTransport(OSError(errno.EINVAL, message))
    router = RewardRouter(
        {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": "https://reward.example/score",
                    "media_layout": "BCHW",
                    "transport": transport,
                }
            },
            "fail_policy": "invalid",
        },
        cache_dir=tmp_path,
    )

    result = router.score(
        np.zeros((1, 3, 4, 5), dtype=np.uint8),
        ["p"],
        [{}],
        sample_id=["sample"],
    )

    failure = result.metadata["reward_general"]
    assert failure["error_type"] == "RewardTransportError"
    assert "MissingSchema" in failure["error"]
    assert "reward.example" in failure["error"]
    assert len(failure["error"]) <= 500
    for value in (
        "TOPSECRET",
        "QUERYSECRET",
        "SIGSECRET",
        "AUTHSECRET",
        "COOKIESECRET",
    ):
        assert value not in failure["error"]
    assert result.weighted_total.tolist() == [0.0]
    assert result.valid_mask.tolist() == [False]


@pytest.mark.parametrize(
    "exception_name",
    ["InvalidURL", "TooManyRedirects", "SSLError"],
)
def test_requests_permanent_errors_are_wrapped_without_retry(
    exception_name, monkeypatch
):
    requests = _fake_requests_module()
    monkeypatch.setitem(__import__("sys").modules, "requests", requests)
    error_type = getattr(requests.exceptions, exception_name)
    transport = FakeTransport(
        error_type("permanent request failure"),
        _response({"outputs": [1.0]}),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        retries=2,
        media_layout="BCHW",
    )

    with pytest.raises(RewardTransportError, match="permanent transport error"):
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])
    assert len(transport.calls) == 1


@pytest.mark.parametrize("exception_name", ["Timeout", "ConnectionError"])
def test_requests_transient_errors_are_retried(exception_name, monkeypatch):
    requests = _fake_requests_module()
    monkeypatch.setitem(__import__("sys").modules, "requests", requests)
    error_type = getattr(requests.exceptions, exception_name)
    transport = FakeTransport(
        error_type("transient request failure"),
        _response({"outputs": [0.25]}),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=transport,
        retries=1,
        backoff_seconds=0.001,
        sleep=lambda _delay: None,
        media_layout="BCHW",
    )

    values, _ = client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert values.tolist() == pytest.approx([0.25])
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "result",
    [
        FakeResponse(200, b"not-pickle"),
        _response({}),
        _response({"outputs": [float("nan")]}),
        _response(
            {
                "outputs": [1.0],
                "details": [{"video_id": 0, "gs_score": 0.1, "meta_score": 0.2}],
            }
        ),
    ],
)
def test_bad_responses_are_protocol_errors_and_are_not_retried(result):
    transport = FakeTransport(result, _response({"outputs": [1.0]}))
    client_class = (
        WorldR1Reward3DClient
        if b"details" in result.body
        else WorldR1RewardGeneralClient
    )
    layout = "BFCHW" if client_class is WorldR1Reward3DClient else "BCHW"
    media = (
        np.zeros((1, 2, 3, 4, 5), dtype=np.uint8)
        if client_class is WorldR1Reward3DClient
        else np.zeros((1, 3, 4, 5), dtype=np.uint8)
    )
    client = client_class(
        "http://localhost:8089", transport=transport, retries=1, media_layout=layout
    )
    with pytest.raises(RewardProtocolError):
        client.score(media, ["p"], [{}])
    assert len(transport.calls) == 1


def test_router_merges_and_restores_invalid_mask_from_cache(tmp_path):
    response = {
        "outputs": [0.4, 0.8],
        "protocol_version": "strict_v2",
        "sample_id": ["a", "b"],
        "valid_mask": [True, False],
        "server_revision": "strict-invalid-mask-v1",
    }
    transport = FakeTransport(_response(response))
    transport.cache_fingerprint = {"fixture": "invalid-mask-v1"}
    router = RewardRouter(
        {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "version": "wire-v2",
                    "url": "http://localhost:8090",
                    "server_revision": "strict-invalid-mask-v1",
                    "protocol_mode": "strict_v2",
                    "media_layout": "BCHW",
                    "transport": transport,
                }
            },
            "fail_policy": "raise",
        },
        cache_dir=tmp_path,
    )
    args = (
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
        ["one", "two"],
        [{}, {}],
    )

    first = router.score(*args, sample_id=["a", "b"])
    second = router.score(*args, sample_id=["a", "b"])

    assert first.valid_mask.tolist() == [True, False]
    assert second.valid_mask.tolist() == [True, False]
    assert len(transport.calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_router_never_swallows_strict_protocol_errors_under_invalid_policy(tmp_path):
    transport = FakeTransport(
        _response(
            {
                "outputs": [1.0],
                "protocol_version": "strict_v2",
                "sample_id": ["wrong"],
                "valid_mask": [True],
            }
        )
    )
    router = RewardRouter(
        {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": "http://localhost:8090",
                    "protocol_mode": "strict_v2",
                    "media_layout": "BCHW",
                    "transport": transport,
                }
            },
            "fail_policy": "invalid",
        },
        cache_dir=tmp_path,
    )

    with pytest.raises(RewardProtocolError, match="sample_id"):
        router.score(
            np.zeros((1, 3, 4, 5), dtype=np.uint8),
            ["p"],
            [{}],
            sample_id=["expected"],
        )
    assert list(tmp_path.glob("*.json")) == []


def test_router_never_swallows_ragged_strict_mask_under_invalid_policy(tmp_path):
    transport = FakeTransport(
        _response(
            {
                "outputs": [1.0, 2.0],
                "protocol_version": "strict_v2",
                "sample_id": ["a", "b"],
                "valid_mask": [[True], [False, True]],
            }
        )
    )
    router = RewardRouter(
        {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": "http://localhost:8090",
                    "protocol_mode": "strict_v2",
                    "media_layout": "BCHW",
                    "batch_size": 2,
                    "transport": transport,
                }
            },
            "fail_policy": "invalid",
        },
        cache_dir=tmp_path,
    )

    with pytest.raises(RewardProtocolError, match="boolean vector"):
        router.score(
            np.zeros((2, 3, 4, 5), dtype=np.uint8),
            ["one", "two"],
            [{}, {}],
            sample_id=["a", "b"],
        )
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("malicious_field", ["outputs", "valid_mask"])
def test_router_wraps_malicious_array_errors_without_leaking_cause(
    tmp_path, malicious_field
):
    response = {
        "outputs": [1.0],
        "protocol_version": "strict_v2",
        "sample_id": ["sample"],
        "valid_mask": [True],
    }
    response[malicious_field] = MaliciousArray()
    transport = FakeTransport(_response(response, wire_format=LEGACY_PICKLE))
    router = RewardRouter(
        {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": "http://localhost:8090",
                    "protocol_mode": "strict_v2",
                    "media_layout": "BCHW",
                    "transport": transport,
                    **_legacy_kwargs(),
                }
            },
            "fail_policy": "invalid",
        },
        cache_dir=tmp_path,
    )

    with pytest.raises(RewardProtocolError) as exc_info:
        router.score(
            np.zeros((1, 3, 4, 5), dtype=np.uint8),
            ["p"],
            [{}],
            sample_id=["sample"],
        )

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert "SECRET" not in str(exc_info.value)
    assert "SECRET" not in rendered
    assert "RuntimeError" not in rendered
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_response_rejects_non_finite_constants_without_context(constant):
    response = FakeResponse(200, f'{{"outputs":[{constant}]}}'.encode())
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(response),
        media_layout="BCHW",
    )

    with pytest.raises(RewardProtocolError) as exc_info:
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_response_limit_is_checked_before_json_or_pickle_deserialization(monkeypatch):
    class OpaqueTransport:
        def __init__(self, response):
            self.response = response

        def post(self, *_args, **_kwargs):
            return self.response

    media = np.zeros((1, 3, 4, 5), dtype=np.uint8)
    json_client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(FakeResponse(200, b"{" + b"x" * 32)),
        media_layout="BCHW",
        max_response_bytes=8,
    )
    with pytest.raises(RewardProtocolError, match="max_response_bytes"):
        json_client.score(media, ["p"], [{}])

    legacy_client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=OpaqueTransport(FakeResponse(200, b"x" * 32)),
        media_layout="BCHW",
        max_response_bytes=8,
        **_legacy_kwargs(),
    )
    monkeypatch.setattr(
        world_r1_rewards.pickle,
        "loads",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized response reached pickle.loads"
        ),
    )
    with pytest.raises(RewardProtocolError, match="max_response_bytes"):
        legacy_client.score(media, ["p"], [{}])


def test_chunked_response_stops_at_limit_closes_and_never_accesses_content():
    response = FakeResponse(
        200,
        b"unused",
        chunks=(b"1234", b"5678", b"9", b"must-not-be-read"),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(response),
        media_layout="BCHW",
        max_response_bytes=8,
    )

    with pytest.raises(RewardProtocolError, match="max_response_bytes"):
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert response.iterated_chunks == 3
    assert response.closed is True
    assert response.content_accessed is False


def test_oversized_content_length_rejects_before_body_iteration():
    response = FakeResponse(
        200,
        b"unused",
        headers={"Content-Length": "9"},
        chunks=(b"must-not-be-read",),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(response),
        media_layout="BCHW",
        max_response_bytes=8,
    )

    with pytest.raises(RewardProtocolError, match="max_response_bytes"):
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert response.iterated_chunks == 0
    assert response.closed is True
    assert response.content_accessed is False


def test_content_length_underreporting_does_not_bypass_stream_limit():
    response = FakeResponse(
        200,
        b"unused",
        headers={"Content-Length": "1"},
        chunks=(b"12345678", b"9"),
    )
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(response),
        media_layout="BCHW",
        max_response_bytes=8,
    )

    with pytest.raises(RewardProtocolError, match="max_response_bytes"):
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert response.iterated_chunks == 2
    assert response.closed is True


def test_streaming_failure_is_closed_and_redacted():
    url = "https://reward.example/private?token=TOPSECRET#signature"
    response = FakeResponse(200, b"unused")

    def fail_stream(*, chunk_size):
        del chunk_size
        raise OSError(errno.EINVAL, f"stream failed for {url}")
        yield b"unreachable"

    response.iter_content = fail_stream
    client = WorldR1RewardGeneralClient(
        url,
        transport=FakeTransport(response),
        media_layout="BCHW",
    )

    with pytest.raises(RewardTransportError) as exc_info:
        client.score(np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    assert "TOPSECRET" not in str(exc_info.value)
    assert "private" not in str(exc_info.value)
    assert response.closed is True
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_generic_requests_http_uses_streaming_bounded_reader(monkeypatch):
    response = FakeResponse(200, b"12345678", headers={})
    call = {}

    def post(url, **kwargs):
        call.update({"url": url, **kwargs})
        return response

    monkeypatch.setitem(
        __import__("sys").modules,
        "requests",
        SimpleNamespace(post=post),
    )

    result = feedback_clients._post_bytes(
        "http://localhost:8090", b"request", timeout=1.0, max_response_bytes=8
    )

    assert result == b"12345678"
    assert call["stream"] is True
    assert response.closed is True
    assert response.content_accessed is False


def test_generic_client_closes_and_does_not_retry_oversized_stream(monkeypatch):
    response = FakeResponse(
        200,
        b"unused",
        chunks=(b"12345678", b"9", b"must-not-be-read"),
    )
    calls = []

    def post(*_args, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setitem(
        __import__("sys").modules,
        "requests",
        SimpleNamespace(post=post),
    )
    client = RemotePickleRewardClient(
        "http://localhost:8090",
        retries=2,
        max_response_bytes=8,
        allow_unsafe_pickle=True,
        trusted_hosts=("localhost",),
    )

    with pytest.raises(RewardProtocolError, match="max_response_bytes"):
        client.score(None, ["p"], [{}])

    assert len(calls) == 1
    assert calls[0]["stream"] is True
    assert response.iterated_chunks == 2
    assert response.closed is True
    assert response.content_accessed is False


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field", ["timeout", "backoff_seconds", "batch_size", "max_response_bytes"]
)
def test_world_r1_positive_config_rejects_zero_and_non_finite(field, value):
    kwargs = {
        "transport": FakeTransport(),
        "media_layout": "BCHW",
        field: value,
    }

    with pytest.raises((TypeError, ValueError, OverflowError)):
        WorldR1RewardGeneralClient("http://localhost:8090", **kwargs)


@pytest.mark.parametrize("server_revision", ["", "   ", 1, True, b"revision"])
def test_world_r1_server_revision_must_be_non_empty_string_or_none(server_revision):
    with pytest.raises((TypeError, ValueError), match="server_revision"):
        WorldR1RewardGeneralClient(
            "http://localhost:8090",
            transport=FakeTransport(),
            server_revision=server_revision,
        )


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), float("-inf")])
def test_remote_pickle_timeout_rejects_zero_and_non_finite(value):
    with pytest.raises(ValueError, match="finite and positive"):
        RemotePickleRewardClient(
            "http://localhost:8090",
            timeout=value,
            allow_unsafe_pickle=True,
            trusted_hosts=("localhost",),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wire_format": LEGACY_PICKLE}, "allow_unsafe_pickle"),
        (
            {"wire_format": LEGACY_PICKLE, "allow_unsafe_pickle": True},
            "non-empty trusted_hosts",
        ),
        (
            {
                "wire_format": LEGACY_PICKLE,
                "allow_unsafe_pickle": True,
                "trusted_hosts": ["example.com"],
            },
            "exactly match",
        ),
        (
            {
                "wire_format": LEGACY_PICKLE,
                "allow_unsafe_pickle": True,
                "trusted_hosts": ["*.localhost"],
            },
            "without wildcards",
        ),
        (
            {
                "wire_format": LEGACY_PICKLE,
                "allow_unsafe_pickle": True,
                "trusted_hosts": [".localhost"],
            },
            "suffix rules",
        ),
    ],
)
def test_legacy_pickle_requires_exact_host_explicit_opt_in(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        WorldR1RewardGeneralClient("http://localhost:8090", **kwargs)


def test_legacy_pickle_policy_and_wire_are_part_of_cache_fingerprint():
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
        server_revision="legacy-general-v1",
        **_legacy_kwargs(),
    )
    fingerprint = client.cache_fingerprint()
    assert fingerprint["wire_format"] == LEGACY_PICKLE
    assert fingerprint["allow_unsafe_pickle"] is True
    assert fingerprint["trusted_hosts"] == ["localhost"]
    assert fingerprint["max_response_bytes"] > 0


def test_remote_pickle_client_is_not_silently_constructible():
    with pytest.raises(ValueError, match="allow_unsafe_pickle"):
        RemotePickleRewardClient("http://localhost:8090")

    client = RemotePickleRewardClient(
        "http://localhost:8090",
        allow_unsafe_pickle=True,
        trusted_hosts=("localhost",),
    )
    assert client.cache_fingerprint()["wire_format"] == LEGACY_PICKLE


def test_protocol_mode_and_batch_policy_are_part_of_cache_fingerprint():
    reference = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
        protocol_mode="reference_v1",
        batch_size=4,
        server_revision="general-v1",
    )
    strict = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
        protocol_mode="strict_v2",
        batch_size=4,
        server_revision="general-v1",
    )
    different_batch = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
        protocol_mode="reference_v1",
        batch_size=8,
        server_revision="general-v1",
    )

    assert reference.cache_fingerprint() != strict.cache_fingerprint()
    assert reference.cache_fingerprint() != different_batch.cache_fingerprint()


def test_missing_server_revision_explicitly_disables_client_cache_fingerprint():
    client = WorldR1RewardGeneralClient(
        "http://localhost:8090",
        transport=FakeTransport(),
    )

    assert client.server_revision is None
    assert client.cache_fingerprint() is None


def test_mock_mode_change_cannot_hit_an_old_router_cache_entry(tmp_path):
    constant = RewardRouter(
        {
            "weights": {"mock": 1.0},
            "clients": {"mock": {"name": "mock", "mode": "constant"}},
        },
        cache_dir=tmp_path,
    )
    prompt_hash = RewardRouter(
        {
            "weights": {"mock": 1.0},
            "clients": {"mock": {"name": "mock", "mode": "prompt_hash"}},
        },
        cache_dir=tmp_path,
    )

    first = constant.score(None, ["same prompt"], [{}], sample_id=["sample"])
    second = prompt_hash.score(None, ["same prompt"], [{}], sample_id=["sample"])

    assert first.weighted_total.tolist() == pytest.approx([1.0])
    assert second.weighted_total.tolist() != pytest.approx([1.0])
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_injected_transport_without_fingerprint_disables_cache(tmp_path):
    class StatefulTransport:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return _response({"outputs": [self.value]})

    def router_for(transport):
        return RewardRouter(
            {
                "weights": {"reward_general": 1.0},
                "clients": {
                    "reward_general": {
                        "name": "reward_general",
                        "url": "http://localhost:8090",
                        "media_layout": "BCHW",
                        "server_revision": "general-v1",
                        "transport": transport,
                    }
                },
                "fail_policy": "raise",
            },
            cache_dir=tmp_path,
        )

    first_transport = StatefulTransport(1.0)
    second_transport = StatefulTransport(2.0)
    first_router = router_for(first_transport)
    second_router = router_for(second_transport)
    args = (np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    first = first_router.score(*args, sample_id=["sample"])
    second = second_router.score(*args, sample_id=["sample"])

    assert first.weighted_total.tolist() == [1.0]
    assert second.weighted_total.tolist() == [2.0]
    assert first_transport.calls == second_transport.calls == 1
    assert first_router.client_fingerprints["reward_general"] is None
    assert second_router.client_fingerprints["reward_general"] is None
    assert list(tmp_path.glob("*.json")) == []


def test_explicit_injected_fingerprint_is_secret_free_cacheable_and_isolated(tmp_path):
    class FingerprintedTransport:
        def __init__(self, state, value, secret):
            self.state = state
            self.value = value
            self.secret = secret
            self.calls = 0

        def cache_fingerprint(self):
            return {"state": self.state}

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return _response({"outputs": [self.value]})

    def router_for(transport):
        return RewardRouter(
            {
                "weights": {"reward_general": 1.0},
                "clients": {
                    "reward_general": {
                        "name": "reward_general",
                        "url": "http://localhost:8090",
                        "media_layout": "BCHW",
                        "server_revision": "general-v1",
                        "transport": transport,
                    }
                },
                "fail_policy": "raise",
            },
            cache_dir=tmp_path,
        )

    state_a_first = FingerprintedTransport("a", 1.0, "transport-secret-a")
    state_a_cached = FingerprintedTransport("a", 99.0, "transport-secret-b")
    state_b = FingerprintedTransport("b", 2.0, "transport-secret-c")
    routers = [
        router_for(state_a_first),
        router_for(state_a_cached),
        router_for(state_b),
    ]
    args = (np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])
    results = [router.score(*args, sample_id=["sample"]) for router in routers]

    assert [result.weighted_total.tolist() for result in results] == [
        [1.0],
        [1.0],
        [2.0],
    ]
    assert [
        transport.calls for transport in (state_a_first, state_a_cached, state_b)
    ] == [
        1,
        0,
        1,
    ]
    identity = routers[0].client_fingerprints["reward_general"]
    assert identity is not None
    assert "transport-secret" not in repr(identity)
    assert "0x" not in repr(identity)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_default_transport_and_same_server_revision_hit_shared_cache(
    tmp_path, monkeypatch
):
    transport = FakeTransport(_response({"outputs": [0.75]}))
    monkeypatch.setattr(world_r1_rewards, "requests_session", lambda: transport)
    config = {
        "weights": {"reward_general": 1.0},
        "clients": {
            "reward_general": {
                "name": "reward_general",
                "url": "http://localhost:8090",
                "media_layout": "BCHW",
                "server_revision": "general-v1",
            }
        },
        "fail_policy": "raise",
    }
    args = (np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    first_router = RewardRouter(config, cache_dir=tmp_path)
    second_router = RewardRouter(config, cache_dir=tmp_path)
    first = first_router.score(*args, sample_id=["sample"])
    second = second_router.score(*args, sample_id=["sample"])

    assert first.weighted_total.tolist() == second.weighted_total.tolist() == [0.75]
    assert transport.calls and len(transport.calls) == 1
    assert first_router.client_fingerprints["reward_general"] is not None
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_missing_server_revision_disables_persistent_cache(tmp_path, monkeypatch):
    transport = FakeTransport(
        _response({"outputs": [0.25]}),
        _response({"outputs": [0.75]}),
    )
    monkeypatch.setattr(world_r1_rewards, "requests_session", lambda: transport)
    config = {
        "weights": {"reward_general": 1.0},
        "clients": {
            "reward_general": {
                "name": "reward_general",
                "url": "http://localhost:8090",
                "media_layout": "BCHW",
            }
        },
        "fail_policy": "raise",
    }
    args = (np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])

    first_router = RewardRouter(config, cache_dir=tmp_path)
    second_router = RewardRouter(config, cache_dir=tmp_path)
    first = first_router.score(*args, sample_id=["sample"])
    second = second_router.score(*args, sample_id=["sample"])

    assert first.weighted_total.tolist() == [0.25]
    assert second.weighted_total.tolist() == [0.75]
    assert len(transport.calls) == 2
    assert first_router.client_fingerprints["reward_general"] is None
    assert second_router.client_fingerprints["reward_general"] is None
    assert list(tmp_path.glob("*.json")) == []


def test_same_url_different_server_revision_cannot_share_cache(tmp_path, monkeypatch):
    transport = FakeTransport(
        _response({"outputs": [0.25]}),
        _response({"outputs": [0.75]}),
    )
    monkeypatch.setattr(world_r1_rewards, "requests_session", lambda: transport)

    def config(server_revision):
        return {
            "weights": {"reward_general": 1.0},
            "clients": {
                "reward_general": {
                    "name": "reward_general",
                    "url": "http://localhost:8090",
                    "media_layout": "BCHW",
                    "server_revision": server_revision,
                }
            },
            "fail_policy": "raise",
        }

    args = (np.zeros((1, 3, 4, 5), dtype=np.uint8), ["p"], [{}])
    revision_a = RewardRouter(config("general-v1"), cache_dir=tmp_path)
    revision_b = RewardRouter(config("general-v2"), cache_dir=tmp_path)

    first = revision_a.score(*args, sample_id=["sample"])
    second = revision_b.score(*args, sample_id=["sample"])

    assert first.weighted_total.tolist() == [0.25]
    assert second.weighted_total.tolist() == [0.75]
    assert len(transport.calls) == 2
    assert (
        revision_a.client_fingerprints["reward_general"]
        != (revision_b.client_fingerprints["reward_general"])
    )
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_router_signature_detection_preserves_old_clients_and_does_not_hide_typeerror():
    calls = []

    class LegacyClient:
        def score(self, media, prompts, metadata):
            calls.append("legacy")
            return np.ones(len(prompts), dtype=np.float32), {}

    class IdentityClient:
        def score(self, media, prompts, metadata, *, sample_id=None):
            calls.append(list(sample_id))
            raise TypeError("client-internal-typeerror")

    class KwargsClient:
        def score(self, media, prompts, metadata, **kwargs):
            calls.append(dict(kwargs))
            return np.ones(len(prompts), dtype=np.float32), {}

    class PositionalIdentityClient:
        def score(self, media, prompts, metadata, sample_id, /):
            calls.append(("positional", list(sample_id)))
            return np.ones(len(prompts), dtype=np.float32), {}

    legacy_name = "c9b_legacy_client"
    identity_name = "c9b_identity_client"
    kwargs_name = "c9b_kwargs_client"
    positional_name = "c9b_positional_identity_client"
    REWARD_CLIENTS.register(legacy_name, LegacyClient)
    REWARD_CLIENTS.register(identity_name, IdentityClient)
    REWARD_CLIENTS.register(kwargs_name, KwargsClient)
    REWARD_CLIENTS.register(positional_name, PositionalIdentityClient)
    legacy = RewardRouter(
        {
            "weights": {legacy_name: 1.0},
            "clients": {legacy_name: {"name": legacy_name}},
            "fail_policy": "raise",
        }
    )
    result = legacy.score(None, ["p"], [{}], sample_id=["sample"])
    assert result.valid_mask.tolist() == [True]

    kwargs_router = RewardRouter(
        {
            "weights": {kwargs_name: 1.0},
            "clients": {kwargs_name: {"name": kwargs_name}},
            "fail_policy": "raise",
        }
    )
    result = kwargs_router.score(None, ["p"], [{}], sample_id=["sample"])
    assert result.valid_mask.tolist() == [True]

    positional = RewardRouter(
        {
            "weights": {positional_name: 1.0},
            "clients": {positional_name: {"name": positional_name}},
            "fail_policy": "raise",
        }
    )
    result = positional.score(None, ["p"], [{}], sample_id=["sample"])
    assert result.valid_mask.tolist() == [True]

    identity = RewardRouter(
        {
            "weights": {identity_name: 1.0},
            "clients": {identity_name: {"name": identity_name}},
            "fail_policy": "raise",
        }
    )
    with pytest.raises(TypeError, match="client-internal-typeerror"):
        identity.score(None, ["p"], [{}], sample_id=["sample"])
    assert calls == [
        "legacy",
        {"sample_id": ["sample"]},
        ("positional", ["sample"]),
        ["sample"],
    ]


def test_packaged_world_preset_constructs_provider_without_requesting(monkeypatch):
    class NoRequestTransport:
        def post(self, *_args, **_kwargs):
            pytest.fail("provider construction sent a reward request")

    transport = NoRequestTransport()
    monkeypatch.setattr(world_r1_rewards, "requests_session", lambda: transport)
    preset_path = (
        Path(world_r1_rewards.__file__).parents[1]
        / "configs"
        / "presets"
        / "world_r1_wan_bounded.yaml"
    )
    preset = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    rewards = preset["rewards"]

    provider = build_feedback_provider(rewards)

    clients = provider.reward_router.clients
    assert clients["reward_general"].url == "http://127.0.0.1:8090/"
    assert clients["reward_3d"].url == "http://127.0.0.1:8089/"
    assert clients["reward_3d"].require_camera_trajectory is True
    for client in clients.values():
        assert client.wire_format == LEGACY_PICKLE
        assert client.allow_unsafe_pickle is True
        assert client.trusted_hosts == ("127.0.0.1", "localhost")
    assert preset["model"]["extra"]["use_camera_trajectory"] is True
    assert (
        preset["model"]["extra"]["use_camera_trajectory"]
        is rewards["clients"]["reward_3d"]["require_camera_trajectory"]
    )
    assert clients["reward_general"].transport is transport
    assert clients["reward_3d"].transport is transport


def test_probe_is_dry_run_by_default_and_summarizes_payload_without_http(
    monkeypatch,
):
    monkeypatch.setattr(
        world_r1_rewards,
        "_rgb_jpeg",
        lambda _image: pytest.fail("dry-run required the Pillow JPEG encoder"),
    )
    transport = SimpleNamespace(
        post=lambda *_args, **_kwargs: pytest.fail("dry-run probe sent HTTP")
    )
    result = run_world_r1_reward_server_probe(
        WorldR1RewardServerProbeConfig(
            reward="reward_3d",
            url="http://localhost:8089",
            batch_size=2,
            frames=3,
            width=3,
            transport=transport,
        )
    )

    assert result["mode"] == "dry_run"
    assert result["http_executed"] is False
    assert result["payload_bytes_sendable"] is False
    assert result["encoding"]["jpeg_encoding"] == {
        "name": "synthetic_dry_run",
        "format": "synthetic_bytes",
        "real_jpeg": False,
        "wire_compatible": False,
        "sendable": False,
        "purpose": "payload_shape_and_length_preview_only",
    }
    assert result["payloads"][0]["keys"] == [
        "camera_trajectories",
        "prompts",
        "videos",
    ]
    assert len(result["payloads"][0]["video_frame_byte_lengths"]) == 2
    trajectories = result["payloads"][0]["camera_trajectories"]
    assert len(trajectories) == 2
    assert list(trajectories[0]) == ["frame0", "frame1", "frame2"]
    assert all(value.count("[") == 4 for value in trajectories[0].values())


def test_probe_redacts_display_url_but_requests_the_original_url(monkeypatch):
    monkeypatch.setattr(world_r1_rewards, "_rgb_jpeg", lambda _image: b"REAL-JPEG")
    url = "https://reward.example/score?token=TOPSECRET&x=1#fragment-value"
    transport = FakeTransport(_response({"outputs": [1.0]}))

    result = run_world_r1_reward_server_probe(
        WorldR1RewardServerProbeConfig(
            reward="reward_general",
            url=url,
            execute_http=True,
            transport=transport,
        )
    )

    assert transport.calls[0]["url"] == url
    assert result["url"] == "https://reward.example/[REDACTED]"
    serialized = str(result)
    assert "TOPSECRET" not in serialized
    assert "=1" not in serialized
    assert "fragment-value" not in serialized


def test_probe_legacy_http_requires_explicit_policy_and_can_use_opted_in_transport(
    monkeypatch,
):
    monkeypatch.setattr(world_r1_rewards, "_rgb_jpeg", lambda _image: b"REAL-JPEG")
    with pytest.raises(ValueError, match="allow_unsafe_pickle"):
        run_world_r1_reward_server_probe(
            WorldR1RewardServerProbeConfig(
                reward="reward_general",
                url="http://localhost:8090",
                execute_http=True,
                wire_format=LEGACY_PICKLE,
                transport=FakeTransport(),
            )
        )

    transport = FakeTransport(_response({"outputs": [1.0]}, wire_format=LEGACY_PICKLE))
    result = run_world_r1_reward_server_probe(
        WorldR1RewardServerProbeConfig(
            reward="reward_general",
            url="http://localhost:8090/private?token=SECRET",
            execute_http=True,
            wire_format=LEGACY_PICKLE,
            allow_unsafe_pickle=True,
            trusted_hosts=("localhost",),
            transport=transport,
        )
    )

    assert transport.calls[0]["url"].endswith("/private?token=SECRET")
    assert result["url"] == "http://localhost:8090/[REDACTED]"
    assert "SECRET" not in str(result)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@reward.example/private/TOPSECRET"
        "?token=QUERYSECRET#fragment-secret",
        "https%3A%2F%2Fuser%3Apassword%40reward.example%2Fprivate%2FTOPSECRET"
        "%3Ftoken%3DQUERYSECRET%23fragment-secret",
    ],
    ids=["plaintext-url", "percent-encoded-url"],
)
def test_legacy_probe_error_json_redacts_url_for_invalid_height(capsys, url):
    exit_code = legacy_cli.main(
        [
            "world-r1-reward-server-probe",
            "--reward",
            "reward_general",
            "--url",
            url,
            "--height",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    serialized = json.dumps(payload, sort_keys=True)
    assert exit_code == 1
    assert payload["valid"] is False
    assert payload["url"] == "https://reward.example/[REDACTED]"
    assert "height must be positive" in payload["errors"][0]
    for secret in (
        "user",
        "password",
        "private",
        "TOPSECRET",
        "QUERYSECRET",
        "fragment-secret",
    ):
        assert secret not in serialized


@pytest.mark.parametrize("reward", ["reward_general", "reward_3d"])
def test_probe_dry_run_width_three_never_uses_real_jpeg(monkeypatch, reward):
    monkeypatch.setattr(
        world_r1_rewards,
        "_rgb_jpeg",
        lambda _image: pytest.fail("dry-run used the real JPEG path"),
    )

    result = run_world_r1_reward_server_probe(
        WorldR1RewardServerProbeConfig(
            reward=reward,
            url="http://localhost:8089",
            frames=2,
            width=3,
        )
    )

    assert result["valid"] is True
    assert result["encoding"]["input_layout"] == (
        "BCHW" if reward == "reward_general" else "BFCHW"
    )
    assert result["encoding"]["jpeg_encoding"]["real_jpeg"] is False


@pytest.mark.parametrize("reward", ["reward_general", "reward_3d"])
def test_probe_http_never_sends_synthetic_dry_run_bytes(monkeypatch, reward):
    monkeypatch.setattr(world_r1_rewards, "_rgb_jpeg", lambda _image: b"REAL-JPEG")
    response = {"outputs": [1.0]}
    transport = FakeTransport(_response(response))

    result = run_world_r1_reward_server_probe(
        WorldR1RewardServerProbeConfig(
            reward=reward,
            url="http://localhost:8089",
            frames=2,
            width=3,
            execute_http=True,
            transport=transport,
        )
    )

    payload = transport.calls[0]["payload"]
    encoded = payload["images"] if reward == "reward_general" else payload["videos"][0]
    assert encoded
    assert all(item == b"REAL-JPEG" for item in encoded)
    assert result["payload_bytes_sendable"] is True
    assert result["encoding"]["jpeg_encoding"]["wire_compatible"] is True
