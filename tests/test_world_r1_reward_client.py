"""Focused tests for the sole World-R1 strict_v2/json_v1 client path."""

from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from visual_rl.core.types import (
    ResolutionContext,
    RuntimeBuildContext,
    StepContext,
    ValidationContext,
)
from visual_rl.algorithms.rewards import (
    RewardBatchIdentity,
    RewardBatchView,
    RewardRuntimeContext,
)
from visual_rl.data.samples import (
    BatchRowContext,
    CameraConditionBatchState,
    NoConditionBatchState,
    SourceItemContext,
    StackedSampleBatch,
    TrajectoryBatch,
    TrajectoryContext,
    camera_condition_identity,
)
from visual_rl.algorithms.rewards.clients.mock import RewardTransportError
from visual_rl.algorithms.rewards.clients import world_r1 as world_r1_rewards
from visual_rl.algorithms.rewards.clients.world_r1 import (
    WORLD_R1_RESOURCE_PROTOCOL,
    WorldR1HealthAttestation,
    WorldR1Reward3DClient,
    WorldR1RewardGeneralClient,
)
from visual_rl.algorithms.rewards.input_selection import RewardInputSelectionPolicy
from visual_rl.core.protocols.world_r1 import (
    MANAGER_CONTRACT,
    PROTOCOL_VERSION,
    WIRE_FORMAT,
    WorldR1ProtocolError,
    WorldR1RevisionError,
    encode_json,
)

REVISION = "world-r1-0123456789ab"


@pytest.fixture(autouse=True)
def _deterministic_jpeg_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep protocol tests independent of the optional Pillow wheel."""

    def encode(image) -> str:
        return base64.b64encode(b"\xff\xd8" + image.tobytes()).decode("ascii")

    monkeypatch.setattr(world_r1_rewards, "_jpeg_base64", encode)


class _Response:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        content_type: str = "application/json",
        content_length: str | None = None,
    ) -> None:
        self.body = encode_json(payload)
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self.close_calls = 0

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self) -> None:
        self.close_calls += 1


class _Session:
    def __init__(
        self,
        *,
        score_payload: dict[str, Any] | None = None,
        health_payload: dict[str, Any] | None = None,
        post_error: Exception | None = None,
    ) -> None:
        self.score_payload = score_payload
        self.health_payload = health_payload
        self.post_error = post_error
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[tuple[str, dict[str, Any]]] = []
        self.close_calls = 0

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        assert self.score_payload is not None
        return _Response(self.score_payload)

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.gets.append((url, kwargs))
        assert self.health_payload is not None
        return _Response(self.health_payload)

    def close(self) -> None:
        self.close_calls += 1


def _context() -> StepContext:
    return StepContext(step=2, seed=9)


def _batch(
    context: StepContext,
    *,
    frames: int = 3,
    camera: bool = False,
    image: bool = False,
    camera_offset: float = 0.0,
) -> RewardBatchView:
    batch_size = 2
    media = torch.zeros(batch_size, frames, 3, 4, 5, dtype=torch.float32)
    for frame in range(frames):
        media[:, frame, frame % 3] = (frame + 1) / (frames + 1)
    camera_tensor = (
        (torch.eye(4, dtype=torch.float32) + camera_offset)
        .reshape(1, 1, 4, 4)
        .repeat(batch_size, frames, 1, 1)
        if camera
        else None
    )
    transitions = 2
    rows = tuple(
        BatchRowContext(
            occurrence_id=f"occurrence-{row}",
            group_id=f"group-{'a' if row == 0 else 'b'}",
            member_id=0,
            phase="main",
            optimizer_step=context.step,
            source_item_id=f"prompt-{'a' if row == 0 else 'b'}",
        )
        for row in range(batch_size)
    )
    sources = tuple(
        SourceItemContext(
            source_item_id=rows[row].source_item_id,
            dataset_source_id="main",
            dataset_index=row,
            dataset_revision="test-v1",
        )
        for row in range(batch_size)
    )
    if camera_tensor is None:
        condition_state = NoConditionBatchState(batch_size)
        condition_ids = ("none",) * batch_size
    else:
        configs = tuple(f"camera-config-{row}" for row in range(batch_size))
        condition_ids = tuple(
            camera_condition_identity(camera_tensor[row], configs[row])
            for row in range(batch_size)
        )
        condition_state = CameraConditionBatchState(
            camera_trajectory=camera_tensor,
            conditioner_config_identity=configs,
            row_condition_identities=condition_ids,
        )
    samples = StackedSampleBatch(
        task_type="t2i" if image else "t2v",
        prompts=("first prompt", "second prompt"),
        metadata=({"row": 0}, {"row": 1}),
        sources=sources,
        rows=rows,
        condition_state=condition_state,
    )
    contexts = tuple(
        TrajectoryContext(
            sample_id=f"sample-{'a' if row == 0 else 'b'}",
            trajectory_id=f"trajectory-{row}",
            batch_row=rows[row],
        )
        for row in range(batch_size)
    )
    reward_media = media[:, 0].contiguous() if image else media
    media_layout = "BCHW" if image else "BFCHW"
    trajectory = TrajectoryBatch(
        kind="full_trajectory",
        contexts=contexts,
        x_t=torch.zeros(batch_size, transitions, 1),
        sampled_action=torch.ones(batch_size, transitions, 1),
        conditioned_next=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).repeat(batch_size, 1),
        next_timesteps=torch.arange(1, transitions + 1).repeat(batch_size, 1),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(batch_size, transitions, dtype=torch.bool),
        transition_index=torch.arange(transitions).repeat(batch_size, 1),
        likelihood_semantics="exact_env_action",
        condition_identity=tuple((item,) * transitions for item in condition_ids),
        guidance_identity=(("cfg",) * transitions,) * batch_size,
        storage_dtype_identity=(("torch.float32",) * transitions,) * batch_size,
        quantization_identity=(("none",) * transitions,) * batch_size,
        media=reward_media,
        media_layout=media_layout,
        condition_state=condition_state,
    )
    payload = {
        "trajectory": trajectory,
        "samples": samples,
        "reward_runtime_context": RewardRuntimeContext(context),
        "media": reward_media,
        "condition_state": condition_state,
    }
    if camera_tensor is not None:
        payload["camera_trajectory_v1"] = camera_tensor
    return RewardBatchView(
        identity=RewardBatchIdentity(
            source_id="main",
            phase_id="main",
            batch_row_ids=tuple(row.identity for row in rows),
            sample_ids=tuple(item.sample_id for item in contexts),
            trajectory_ids=tuple(item.trajectory_id for item in contexts),
            condition_payload_ids=condition_ids,
            group_ids=tuple(row.group_id for row in rows),
        ),
        active_reward_ids=("reward",),
        payload=payload,
    )


def _image_batch(context: StepContext) -> RewardBatchView:
    return _batch(context, frames=1, camera=False, image=True)


def _params(
    tmp_path: Path | None = None,
    *,
    general: bool = True,
) -> dict[str, object]:
    return {
        "url": "http://127.0.0.1:8090/",
        "timeout_s": 1830.0,
        "trusted_hosts": ["127.0.0.1"],
        "ca_bundle": None if tmp_path is None else "ca.pem",
        "max_response_bytes": 1024 * 1024,
        "server_revision": REVISION,
        "input_selection_policy": (
            RewardInputSelectionPolicy.release_world_r1().to_payload()
            if general
            else None
        ),
    }


def _resolution_context(tmp_path: Path) -> ResolutionContext:
    config_path = (tmp_path / "config.yaml").resolve()
    return ResolutionContext(
        config_path=config_path,
        config_dir=config_path.parent,
    )


def _runtime_context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _score_payload(
    *,
    sample_id: list[str] | None = None,
    revision: str = REVISION,
    valid_mask: list[bool] | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "server_revision": revision,
        "sample_id": sample_id or ["sample-a", "sample-b"],
        "outputs": [0.25, 0.75],
        "valid_mask": valid_mask or [True, True],
    }


def _client(
    client_type,
    session: _Session,
    *,
    max_response_bytes: int = 1024 * 1024,
):
    selection_policy = (
        RewardInputSelectionPolicy.release_world_r1()
        if client_type is WorldR1RewardGeneralClient
        else None
    )
    return client_type(
        url="http://127.0.0.1:8090",
        timeout_s=1830.0,
        trusted_hosts=("127.0.0.1",),
        ca_bundle=None,
        max_response_bytes=max_response_bytes,
        server_revision=REVISION,
        input_selection_policy=selection_policy,
        transport=session,
    )


def _request_json(session: _Session) -> dict[str, Any]:
    return json.loads(session.posts[-1][1]["data"].decode("utf-8"))


def test_world_clients_expose_the_native_typed_reward_contract() -> None:
    assert callable(WorldR1RewardGeneralClient.score)
    assert callable(WorldR1Reward3DClient.score)


def test_healthcheck_returns_typed_independent_attestation() -> None:
    health = {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "wire_format": WIRE_FORMAT,
        "reward": "reward_general",
        "server_revision": REVISION,
        "manager_contract": MANAGER_CONTRACT,
    }
    session = _Session(health_payload=health)
    client = _client(WorldR1RewardGeneralClient, session)

    attestation = client.healthcheck()

    assert attestation == WorldR1HealthAttestation(
        endpoint_origin="http://127.0.0.1:8090",
        reward="reward_general",
        protocol=WORLD_R1_RESOURCE_PROTOCOL,
        protocol_version=PROTOCOL_VERSION,
        wire_format=WIRE_FORMAT,
        server_revision=REVISION,
        manager_contract=MANAGER_CONTRACT,
    )
    client.close()
    assert session.close_calls == 1


def test_resolve_params_is_exact_and_canonical(tmp_path: Path) -> None:
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")
    resolved = WorldR1RewardGeneralClient.resolve_params(
        _params(tmp_path),
        _resolution_context(tmp_path),
    )

    assert resolved["url"] == "http://127.0.0.1:8090"
    assert resolved["trusted_hosts"] == ("127.0.0.1",)
    assert resolved["ca_bundle"] == ca_bundle.resolve()
    assert resolved["server_revision"] == REVISION
    assert dict(resolved["input_selection_policy"]) == (
        RewardInputSelectionPolicy.release_world_r1().to_payload()
    )

    invalid = _params()
    invalid["retries"] = 2
    with pytest.raises(ValueError, match="unknown=.*retries"):
        WorldR1RewardGeneralClient.resolve_params(
            invalid,
            _resolution_context(tmp_path),
        )


def test_legacy_resolution_materializes_historical_selection_semantics(
    tmp_path: Path,
) -> None:
    general = _params()
    del general["input_selection_policy"]
    general_before = dict(general)
    resolved_general = WorldR1RewardGeneralClient.resolve_params(
        general,
        _resolution_context(tmp_path),
    )

    assert general == general_before
    assert dict(resolved_general["input_selection_policy"]) == (
        RewardInputSelectionPolicy.fixed_middle_extension().to_payload()
    )

    reward_3d = _params(general=False)
    del reward_3d["input_selection_policy"]
    resolved_3d = WorldR1Reward3DClient.resolve_params(
        reward_3d,
        _resolution_context(tmp_path),
    )
    assert resolved_3d["input_selection_policy"] is None


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("url", "http://example.com", "loopback"),
        ("url", "https://example.com/path", "origin"),
        ("timeout_s", 1829.0, ">= 1830"),
        ("trusted_hosts", [], "must not be empty"),
        ("max_response_bytes", True, "positive integer"),
        ("server_revision", "main", "must match"),
    ],
)
def test_resolve_params_rejects_noncanonical_values(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    params = _params()
    params[field] = value
    if field == "url" and value == "https://example.com/path":
        params["trusted_hosts"] = ["example.com"]
    with pytest.raises((TypeError, ValueError), match=error):
        WorldR1RewardGeneralClient.resolve_params(
            params,
            _resolution_context(tmp_path),
        )


def test_from_config_rechecks_ca_bundle_and_owns_one_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")
    resolved = WorldR1RewardGeneralClient.resolve_params(
        _params(tmp_path),
        _resolution_context(tmp_path),
    )
    session = _Session(score_payload=_score_payload())
    monkeypatch.setattr(
        "visual_rl.algorithms.rewards.clients.world_r1.requests_session",
        lambda: session,
    )

    client = WorldR1RewardGeneralClient.from_config(
        resolved,
        _runtime_context(),
    )
    client.close()
    client.close()

    assert session.close_calls == 1


def test_general_score_uses_keyed_batch_shared_frame_and_no_camera() -> None:
    context = _context()
    batch = _batch(context, frames=3, camera=True)
    session = _Session(score_payload=_score_payload())
    client = _client(WorldR1RewardGeneralClient, session)

    result = client.score(batch=batch)
    request = _request_json(session)

    assert result.values.tolist() == [0.25, 0.75]
    assert result.identity is batch.identity
    shared = result.execution_provenance["shared_metadata"]
    records = result.execution_provenance["sample_metadata"]
    assert shared["protocol_version"] == "strict_v2"
    assert set(request) == {
        "protocol_version",
        "server_revision",
        "sample_id",
        "prompts",
        "images",
    }
    assert request["sample_id"] == list(batch.identity.sample_ids)
    assert len(request["images"]) == batch.batch_size
    assert all(
        base64.b64decode(item).startswith(b"\xff\xd8") for item in request["images"]
    )
    expected_selection = RewardInputSelectionPolicy.release_world_r1().select(
        frame_count=3,
        context=context,
        sample_ids=batch.identity.sample_ids,
        invocation_identity="reward_general",
    )
    assert records[0]["selected_frame_index"] == (
        expected_selection.selected_frame_index
    )
    assert records[1]["selected_frame_index"] == (
        expected_selection.selected_frame_index
    )
    assert records[0]["input_selection_policy_id"] == (
        expected_selection.policy_id
    )
    assert records[0]["selection_key_id"] == (
        expected_selection.selection_key_id
    )
    assert shared["input_selection_policy_id"] == (
        expected_selection.policy_id
    )
    assert session.posts[0][0] == "http://127.0.0.1:8090/v2/reward"
    assert session.posts[0][1]["allow_redirects"] is False
    assert session.posts[0][1]["verify"] is True
    assert session.posts[0][1]["timeout"] == 1830.0

    changed_camera = _batch(context, frames=3, camera=True, camera_offset=5.0)
    second = _Session(score_payload=_score_payload())
    _client(WorldR1RewardGeneralClient, second).score(batch=changed_camera)
    assert _request_json(second)["images"] == request["images"]


def test_general_score_accepts_sd3_bchw_image_media() -> None:
    context = _context()
    batch = _image_batch(context)
    session = _Session(score_payload=_score_payload())

    result = _client(WorldR1RewardGeneralClient, session).score(batch=batch)
    request = _request_json(session)

    assert result.values.tolist() == [0.25, 0.75]
    assert request["sample_id"] == list(batch.identity.sample_ids)
    assert request["prompts"] == list(batch.payload["samples"].prompts)
    assert len(request["images"]) == batch.batch_size
    assert all(
        base64.b64decode(item).startswith(b"\xff\xd8") for item in request["images"]
    )
    assert all(
        item["source_frame_count"] == 1
        and item["selected_frame_index"] == 0
        and item["input_selection_mode"] == "keyed_uniform"
        and item["input_selection_sharing"] == "batch"
        for item in result.execution_provenance["sample_metadata"]
    )


def test_reward_3d_uses_all_frames_and_only_typed_camera() -> None:
    context = _context()
    batch = _batch(context, frames=4, camera=True)
    session = _Session(score_payload=_score_payload())
    client = _client(WorldR1Reward3DClient, session)

    result = client.score(batch=batch)
    request = _request_json(session)

    assert set(request) == {
        "protocol_version",
        "server_revision",
        "sample_id",
        "prompts",
        "videos",
        "camera_trajectories",
    }
    assert len(request["videos"]) == 2
    assert all(len(video) == 4 for video in request["videos"])
    camera = batch.payload["camera_trajectory_v1"]
    assert request["camera_trajectories"] == camera.tolist()
    assert result.execution_provenance["sample_metadata"][0][
        "camera_frame_count"
    ] == 4

    no_camera = _batch(context, frames=4, camera=False)
    with pytest.raises(ValueError, match="requires batch.camera_trajectory"):
        client.score(batch=no_camera)


def test_general_fixed_middle_is_an_explicit_extension_policy() -> None:
    context = _context()
    batch = _batch(context, frames=4)
    session = _Session(score_payload=_score_payload())
    client = WorldR1RewardGeneralClient(
        url="http://127.0.0.1:8090",
        timeout_s=1830.0,
        trusted_hosts=("127.0.0.1",),
        ca_bundle=None,
        max_response_bytes=1024 * 1024,
        server_revision=REVISION,
        input_selection_policy=(
            RewardInputSelectionPolicy.fixed_middle_extension()
        ),
        transport=session,
    )

    result = client.score(batch=batch)

    record = result.execution_provenance["sample_metadata"][0]
    assert record["selected_frame_index"] == 2
    assert record["input_selection_mode"] == "fixed_middle"


@pytest.mark.parametrize(
    "payload,error_type",
    [
        (_score_payload(sample_id=["sample-b", "sample-a"]), WorldR1ProtocolError),
        (
            _score_payload(revision="world-r1-abcdefabcdef"),
            WorldR1RevisionError,
        ),
        (_score_payload(valid_mask=[True, False]), WorldR1ProtocolError),
    ],
)
def test_score_fails_closed_on_echo_revision_or_validity(
    payload: dict[str, Any],
    error_type: type[Exception],
) -> None:
    context = _context()
    client = _client(
        WorldR1RewardGeneralClient,
        _Session(score_payload=payload),
    )
    with pytest.raises(error_type):
        client.score(batch=_batch(context))


def test_client_never_retries_transport_failures() -> None:
    context = _context()
    session = _Session(post_error=OSError("endpoint failed token=secret"))
    client = _client(WorldR1RewardGeneralClient, session)

    with pytest.raises(RewardTransportError):
        client.score(batch=_batch(context))

    assert len(session.posts) == 1


def test_health_check_is_bounded_exact_and_volatile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    health = {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "wire_format": WIRE_FORMAT,
        "reward": "reward_general",
        "server_revision": REVISION,
        "manager_contract": MANAGER_CONTRACT,
    }
    session = _Session(health_payload=health)
    monkeypatch.setattr(
        "visual_rl.algorithms.rewards.clients.world_r1.requests_session",
        lambda: session,
    )
    resolved = WorldR1RewardGeneralClient.resolve_params(
        _params(),
        _resolution_context(tmp_path),
    )
    context = ValidationContext(
        phase="validate",
        config_dir=tmp_path,
        distributed_mode="single",
        world_size=1,
        backend=None,
        device="cpu",
        timeout_s=30.0,
    )

    assert WorldR1RewardGeneralClient.check_environment(resolved, context) == ()
    assert session.gets[0][0] == "http://127.0.0.1:8090/healthz"
    assert session.gets[0][1]["timeout"] == 5.0
    assert session.gets[0][1]["allow_redirects"] is False
    assert session.close_calls == 1

    unhealthy = _Session(
        health_payload={**health, "server_revision": "world-r1-abcdefabcdef"}
    )
    monkeypatch.setattr(
        "visual_rl.algorithms.rewards.clients.world_r1.requests_session",
        lambda: unhealthy,
    )
    checks = WorldR1RewardGeneralClient.check_environment(resolved, context)
    assert len(checks) == 1
    assert checks[0].level == "error"
    assert checks[0].volatile is True


def test_response_body_is_bounded_before_decode() -> None:
    context = _context()
    session = _Session(score_payload=_score_payload())
    client = _client(
        WorldR1RewardGeneralClient,
        session,
        max_response_bytes=8,
    )
    with pytest.raises(RewardTransportError, match="max_response_bytes"):
        client.score(batch=_batch(context))


def test_source_has_no_second_wire_or_legacy_execution_path() -> None:
    root = Path(__file__).parents[1]
    paths = (
        root / "visual_rl" / "algorithms" / "rewards" / "clients" / "mock.py",
        root / "visual_rl" / "algorithms" / "rewards" / "clients" / "world_r1.py",
    )
    forbidden = {
        "RemotePickleRewardClient",
        "backoff_seconds",
        "protocol_mode",
        "frame_indices",
        "require_camera_trajectory",
        "REFERENCE_V1",
        "LEGACY_PICKLE",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        imported_modules = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert "RewardClient" not in class_names
        assert "urllib.request" not in imported_modules
        for token in forbidden:
            assert token not in source
