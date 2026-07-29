"""Contract tests for the World-R1 strict companion service.

Uses the Flask test client with fake strict/legacy managers for protocol and
fail-closed coverage, static artifact checks for the frozen
requirements/env/README/patch files, and one real Gunicorn subprocess test for
the worker_exit -> manager shutdown lifecycle wiring.  No GPU, no real model,
no real World-R1 checkout.
"""

from __future__ import annotations

import ast
import base64
import http.client
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from visual_rl.world_r1_protocol import (
    ERROR_COMPUTE_FAILED,
    ERROR_MANAGER_NOT_READY,
    ERROR_REVISION_MISMATCH,
    HEALTH_ROUTE,
    MANAGER_CONTRACT,
    REWARD_3D,
    REWARD_GENERAL,
    SCORE_ROUTE,
    WorldR1ProtocolError,
    build_health_payload,
    encode_json,
    validate_score_response,
)
from services.world_r1_strict import gunicorn_conf, reference_contract
from services.world_r1_strict import reward_3d_app, reward_general_app

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ModuleNotFoundError:  # pragma: no cover - dev extra provides flask
    _HAS_FLASK = False

_HAS_GUNICORN = subprocess.run(
    [sys.executable, "-c", "import gunicorn"],
    capture_output=True,
).returncode == 0

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "services/world_r1_strict/reference_patches/world_r1_fail_closed_v1.patch"
REQUIREMENTS_PATH = REPO_ROOT / "services/world_r1_strict/requirements-service.txt"
ENV_PATH = REPO_ROOT / "envs/world-r1-reward-cu128.yml"
README_PATH = REPO_ROOT / "services/world_r1_strict/README.md"

REVISION = "world-r1-" + "a" * 12
REVISION_3D = "world-r1-" + "b" * 12
JPEG_BYTES = b"\xff\xd8strict-test-jpeg\xff\xd9"

IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
ASYMMETRIC_4X4 = [
    [0.0, -1.0, 0.0, 10.5],
    [1.0, 0.0, 0.0, -3.25],
    [0.0, 0.0, 1.0, 2.75],
    [0.0, 0.0, 0.0, 1.0],
]

requires_flask = pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")


def _drain_registry() -> None:
    reference_contract.close_registered_manager(expected_pid=os.getpid())


@pytest.fixture(autouse=True)
def _clean_registry():
    _drain_registry()
    yield
    _drain_registry()


class FakeStrictManager:
    """Minimal fail-closed manager double with the strict markers."""

    STRICT_MANAGER_PROTOCOL = MANAGER_CONTRACT
    STRICT_REWARD_KIND = REWARD_GENERAL

    def __init__(
        self,
        *,
        reward: str | None = None,
        ready: bool = True,
        outputs: list[float] | None = None,
        fail_with: BaseException | None = None,
        lock_timeout_s: float | None = None,
        hold_s: float = 0.0,
    ) -> None:
        if reward is not None:
            self.STRICT_REWARD_KIND = reward
        self._ready = ready
        self._poisoned = False
        self._outputs = outputs
        self._fail_with = fail_with
        self._lock_timeout_s = lock_timeout_s
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self.compute_calls = 0
        self.shutdown_count = 0
        self.received: tuple | None = None

    def is_ready(self) -> bool:
        return self._ready and not self._poisoned

    def _poison(self) -> None:
        self._poisoned = True
        self._ready = False
        self.shutdown()

    def compute_batch_scores(self, items, prompts, **kwargs):
        self.compute_calls += 1
        acquired = True
        if self._lock_timeout_s is not None:
            acquired = self._lock.acquire(timeout=self._lock_timeout_s)
            if not acquired:
                self._poison()
                raise TimeoutError("strict request-lock deadline exhausted")
        try:
            self.received = (items, prompts, kwargs)
            if self._hold_s:
                time.sleep(self._hold_s)
            if self._fail_with is not None:
                self._poison()
                raise self._fail_with
            if self._outputs is not None:
                return list(self._outputs)
            return [0.75] * len(items)
        finally:
            if self._lock_timeout_s is not None and acquired:
                self._lock.release()

    def shutdown(self) -> None:
        self.shutdown_count += 1


class LegacyManagerDouble:
    """Unpatched native manager shape: no strict markers."""

    def __init__(self) -> None:
        self.shutdown_count = 0

    def is_ready(self) -> bool:
        return True

    def compute_batch_scores(self, items, prompts, **kwargs):
        del kwargs
        return [0.5] * len(items)

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _general_payload(**overrides):
    payload = {
        "protocol_version": "strict_v2",
        "server_revision": REVISION,
        "sample_id": ["sample-0", "sample-1"],
        "prompts": ["a red cube", "a blue sphere"],
        "images": [_b64(JPEG_BYTES), _b64(JPEG_BYTES)],
    }
    payload.update(overrides)
    return payload


def _trajectories() -> list:
    return [
        [ASYMMETRIC_4X4, IDENTITY_4X4],
        [IDENTITY_4X4, ASYMMETRIC_4X4],
    ]


def _3d_payload(**overrides):
    payload = {
        "protocol_version": "strict_v2",
        "server_revision": REVISION_3D,
        "sample_id": ["sample-0", "sample-1"],
        "prompts": ["orbit left around a vase", "push in toward a cube"],
        "videos": [
            [_b64(JPEG_BYTES), _b64(JPEG_BYTES)],
            [_b64(JPEG_BYTES), _b64(JPEG_BYTES)],
        ],
        "camera_trajectories": _trajectories(),
    }
    payload.update(overrides)
    return payload


def _post(client, payload, *, content_type="application/json", route=SCORE_ROUTE):
    body = payload if isinstance(payload, bytes) else encode_json(payload)
    return client.post(route, data=body, content_type=content_type)


@requires_flask
def test_general_score_round_trip_uses_real_json_encoder():
    manager = FakeStrictManager()
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    client = app.test_client()
    response = _post(client, _general_payload())
    assert response.status_code == 200
    body = json.loads(response.get_data())
    assert validate_score_response(
        body, expected_sample_ids=["sample-0", "sample-1"], server_revision=REVISION
    ) == [0.75, 0.75]
    items, prompts, kwargs = manager.received
    assert list(items) == [JPEG_BYTES, JPEG_BYTES]
    assert prompts == ["a red cube", "a blue sphere"]
    assert kwargs == {}


@requires_flask
def test_3d_score_round_trip_preserves_typed_camera_elements():
    manager = FakeStrictManager(reward=REWARD_3D)
    app = reward_3d_app.create_app(manager=manager, server_revision=REVISION_3D)
    client = app.test_client()
    response = _post(client, _3d_payload())
    assert response.status_code == 200
    body = json.loads(response.get_data())
    assert validate_score_response(
        body, expected_sample_ids=["sample-0", "sample-1"], server_revision=REVISION_3D
    ) == [0.75, 0.75]
    _, _, kwargs = manager.received
    received = kwargs["camera_trajectories"]
    expected = _trajectories()
    for row_expected, row_actual in zip(expected, received, strict=True):
        for matrix_expected, matrix_actual in zip(row_expected, row_actual, strict=True):
            for line_expected, line_actual in zip(matrix_expected, matrix_actual, strict=True):
                assert line_actual == line_expected  # exact per-element, catches transposes
    assert matrix_actual != [list(row) for row in zip(*matrix_actual)]  # asymmetric sanity


@requires_flask
def test_health_exact_schema_and_no_manager_compute():
    manager = FakeStrictManager()
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    client = app.test_client()
    response = client.get(HEALTH_ROUTE)
    assert response.status_code == 200
    assert json.loads(response.get_data()) == build_health_payload(
        reward=REWARD_GENERAL, server_revision=REVISION
    )
    assert manager.compute_calls == 0


@requires_flask
def test_health_503_until_manager_ready():
    manager = FakeStrictManager(ready=False)
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    response = app.test_client().get(HEALTH_ROUTE)
    assert response.status_code == 503
    assert json.loads(response.get_data()) == {"error": ERROR_MANAGER_NOT_READY}


@requires_flask
def test_fail_closed_scorer_exception_poisons_manager_and_rejects_followup():
    manager = FakeStrictManager(fail_with=RuntimeError("hps scorer exploded"))
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    client = app.test_client()

    first = _post(client, _general_payload())
    assert first.status_code == 500
    assert json.loads(first.get_data()) == {"error": ERROR_COMPUTE_FAILED}
    assert manager.shutdown_count == 1  # poison ran the cleanup primitive once
    assert not manager.is_ready()

    health = client.get(HEALTH_ROUTE)
    assert health.status_code == 503

    second = _post(client, _general_payload())
    assert second.status_code == 503
    assert json.loads(second.get_data()) == {"error": ERROR_MANAGER_NOT_READY}
    assert manager.compute_calls == 1  # no further compute after poison


@requires_flask
@pytest.mark.parametrize(
    "outputs",
    [[0.5], [0.5, 0.6, 0.7], [0.5, float("nan")], [0.5, float("inf")]],
)
def test_manager_output_contract_violations_are_500_not_200(outputs):
    manager = FakeStrictManager(outputs=outputs)
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    response = _post(app.test_client(), _general_payload())
    assert response.status_code == 500
    assert json.loads(response.get_data()) == {"error": ERROR_COMPUTE_FAILED}


@requires_flask
@pytest.mark.parametrize(
    ("payload", "content_type", "expected_status", "expected_error"),
    [
        (_general_payload(), "text/plain", 400, "invalid_request"),
        (b"\x80\x05N.", "application/octet-stream", 400, "invalid_request"),
        (_general_payload(extra_key=1), "application/json", 400, "invalid_request"),
        (
            {k: v for k, v in _general_payload().items() if k != "images"},
            "application/json",
            400,
            "invalid_request",
        ),
        (
            _general_payload(sample_id=["sample-0", "sample-0"]),
            "application/json",
            400,
            "invalid_request",
        ),
        (
            _general_payload(prompts=["only one"]),
            "application/json",
            400,
            "invalid_request",
        ),
        (
            _general_payload(protocol_version="reference_v1"),
            "application/json",
            400,
            "invalid_request",
        ),
        (
            _general_payload(server_revision="world-r1-" + "c" * 12),
            "application/json",
            409,
            ERROR_REVISION_MISMATCH,
        ),
    ],
)
def test_schema_and_revision_rejections_happen_before_manager_compute(
    payload, content_type, expected_status, expected_error
):
    manager = FakeStrictManager()
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    response = _post(app.test_client(), payload, content_type=content_type)
    assert response.status_code == expected_status
    assert json.loads(response.get_data()) == {"error": expected_error}
    assert manager.compute_calls == 0


@requires_flask
@pytest.mark.parametrize(
    "payload",
    [
        _3d_payload(videos=[[_b64(JPEG_BYTES)], [_b64(JPEG_BYTES), _b64(JPEG_BYTES)]]),
        _3d_payload(videos=[[], [_b64(JPEG_BYTES), _b64(JPEG_BYTES)]]),
        _3d_payload(
            camera_trajectories=[[IDENTITY_4X4, IDENTITY_4X4], [IDENTITY_4X4, IDENTITY_4X4, IDENTITY_4X4]]
        ),
        _3d_payload(
            camera_trajectories=[
                [IDENTITY_4X4, IDENTITY_4X4],
                [[[1.0, 0.0, 0.0]] * 4, IDENTITY_4X4],
            ]
        ),
        _3d_payload(camera_trajectories=[{"frame0": IDENTITY_4X4, "frame1": IDENTITY_4X4}, _trajectories()[1]]),
        _3d_payload(camera_trajectories=[[[IDENTITY_4X4[0], True, IDENTITY_4X4[2], IDENTITY_4X4[3]], IDENTITY_4X4], _trajectories()[1]]),
    ],
)
def test_3d_frame_and_camera_rejections_happen_before_manager_compute(payload):
    manager = FakeStrictManager(reward=REWARD_3D)
    app = reward_3d_app.create_app(manager=manager, server_revision=REVISION_3D)
    response = _post(app.test_client(), payload)
    assert response.status_code == 400
    assert json.loads(response.get_data()) == {"error": "invalid_request"}
    assert manager.compute_calls == 0


@requires_flask
def test_non_finite_json_constant_rejected_before_compute():
    manager = FakeStrictManager(reward=REWARD_3D)
    app = reward_3d_app.create_app(manager=manager, server_revision=REVISION_3D)
    body_text = json.dumps(_3d_payload())
    assert "10.5" in body_text
    body = body_text.replace("10.5", "NaN").encode("utf-8")
    response = app.test_client().post(
        SCORE_ROUTE, data=body, content_type="application/json"
    )
    assert response.status_code == 400
    assert manager.compute_calls == 0


@requires_flask
def test_post_root_and_wrong_method_rejected_before_compute():
    manager = FakeStrictManager()
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    client = app.test_client()
    root = _post(client, _general_payload(), route="/")
    assert root.status_code == 404
    wrong_method = client.get(SCORE_ROUTE)
    assert wrong_method.status_code == 405
    assert manager.compute_calls == 0


@requires_flask
def test_create_app_rejects_legacy_manager_and_bad_revision():
    with pytest.raises(reference_contract.ManagerContractError):
        reward_general_app.create_app(manager=LegacyManagerDouble(), server_revision=REVISION)
    with pytest.raises(WorldR1ProtocolError):
        reward_general_app.create_app(
            manager=FakeStrictManager(), server_revision="not-a-revision"
        )
    with pytest.raises(reference_contract.ManagerContractError):
        reward_general_app.create_app(
            manager=FakeStrictManager(reward=REWARD_3D), server_revision=REVISION
        )


@requires_flask
def test_pid_change_rejects_request_and_closes_inherited_manager(monkeypatch):
    manager = FakeStrictManager()
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    monkeypatch.setattr(os, "getpid", lambda: 424242)
    client = app.test_client()
    response = client.get(HEALTH_ROUTE)
    assert response.status_code == 503
    score = _post(client, _general_payload())
    assert score.status_code == 503
    assert manager.shutdown_count == 1  # inherited manager closed exactly once
    assert manager.compute_calls == 0


@requires_flask
def test_build_app_runs_gates_constructor_initialize_exactly_once(monkeypatch):
    calls: list = []
    monkeypatch.setenv("WORLD_R1_SERVER_REVISION", REVISION)
    monkeypatch.setattr(
        reference_contract, "require_service_runtime", lambda: calls.append("runtime")
    )

    def fake_gate(manager_class, *, reward):
        calls.append(("native_gate", reward))

    monkeypatch.setattr(reference_contract, "run_native_fault_injection_gate", fake_gate)

    class SpyManager(FakeStrictManager):
        def __init__(self):
            calls.append("construct")
            super().__init__()

        def initialize(self):
            calls.append("initialize")

    def fake_loader():
        calls.append("manager_import")
        return SpyManager

    monkeypatch.setattr(reward_general_app, "_load_manager_class", fake_loader)
    app = reward_general_app.build_app()
    assert app is not None
    assert calls == [
        "runtime",
        "manager_import",
        ("native_gate", REWARD_GENERAL),
        "construct",
        "initialize",
    ]


@requires_flask
def test_build_app_runtime_gate_failure_precedes_manager_import(monkeypatch):
    calls: list = []
    monkeypatch.setenv("WORLD_R1_SERVER_REVISION", REVISION)

    def failing_runtime():
        calls.append("runtime")
        raise reference_contract.ServiceRuntimeError("torch is not +cu128")

    monkeypatch.setattr(reference_contract, "require_service_runtime", failing_runtime)
    monkeypatch.setattr(
        reward_general_app,
        "_load_manager_class",
        lambda: calls.append("manager_import"),
    )
    with pytest.raises(reference_contract.ServiceRuntimeError):
        reward_general_app.build_app()
    assert calls == ["runtime"]  # no manager import/constructor/initialize


@requires_flask
def test_build_app_initialize_failure_cleans_up_once(monkeypatch):
    monkeypatch.setenv("WORLD_R1_SERVER_REVISION", REVISION_3D)
    monkeypatch.setattr(reference_contract, "require_service_runtime", lambda: None)
    monkeypatch.setattr(
        reference_contract, "run_native_fault_injection_gate", lambda cls, *, reward: None
    )

    class InitFailManager(FakeStrictManager):
        STRICT_REWARD_KIND = REWARD_3D

        def __init__(self, scorer_type=None, use_lpips=None):
            assert scorer_type == "qwen"
            assert use_lpips is True
            super().__init__(reward=REWARD_3D)

        def initialize(self):
            raise RuntimeError("worker INIT_ERROR")

    monkeypatch.setattr(reward_3d_app, "_load_manager_class", lambda: InitFailManager)
    manager_instance: list = []
    original_init = InitFailManager.__init__

    def tracking_init(self, scorer_type=None, use_lpips=None):
        original_init(self, scorer_type=scorer_type, use_lpips=use_lpips)
        manager_instance.append(self)

    monkeypatch.setattr(InitFailManager, "__init__", tracking_init)
    with pytest.raises(RuntimeError, match="INIT_ERROR"):
        reward_3d_app.build_app()
    assert manager_instance[0].shutdown_count == 1


@requires_flask
def test_concurrent_second_request_lock_wait_bounded_by_same_deadline():
    manager = FakeStrictManager(lock_timeout_s=0.3, hold_s=1.0)
    app = reward_general_app.create_app(manager=manager, server_revision=REVISION)
    finished: dict[str, float] = {}
    statuses: dict[str, int] = {}

    def call(name):
        client = app.test_client()
        response = _post(client, _general_payload())
        statuses[name] = response.status_code
        finished[name] = time.monotonic()

    first = threading.Thread(target=call, args=("first",))
    first.start()
    time.sleep(0.1)  # first request now holds the manager lock
    second = threading.Thread(target=call, args=("second",))
    second.start()
    first.join(timeout=15)
    second.join(timeout=15)
    assert statuses == {"first": 200, "second": 500}
    assert finished["second"] < finished["first"]  # bounded lock wait, no unbounded queue


# ---------------------------------------------------------------------------
# Lifecycle registry unit tests (hook/atexit orderings share call-once close)
# ---------------------------------------------------------------------------


def test_worker_exit_then_atexit_closes_exactly_once():
    manager = FakeStrictManager()
    reference_contract.register_manager(manager=manager, pid=os.getpid())
    gunicorn_conf.worker_exit(None, SimpleNamespace(pid=os.getpid()))
    assert manager.shutdown_count == 1
    reference_contract._close_at_exit()
    assert manager.shutdown_count == 1
    assert reference_contract.close_registered_manager(expected_pid=os.getpid()) == 0


def test_atexit_then_worker_exit_closes_exactly_once():
    manager = FakeStrictManager()
    reference_contract.register_manager(manager=manager, pid=os.getpid())
    reference_contract._close_at_exit()
    assert manager.shutdown_count == 1
    gunicorn_conf.worker_exit(None, SimpleNamespace(pid=os.getpid()))
    assert manager.shutdown_count == 1


def test_repeated_manual_cleanup_and_wrong_pid_are_idempotent():
    manager = FakeStrictManager()
    reference_contract.register_manager(manager=manager, pid=os.getpid())
    assert reference_contract.close_registered_manager(expected_pid=os.getpid() + 9999) == 0
    assert manager.shutdown_count == 0
    assert reference_contract.close_registered_manager(expected_pid=os.getpid()) == 1
    assert reference_contract.close_registered_manager(expected_pid=os.getpid()) == 0
    assert manager.shutdown_count == 1


# ---------------------------------------------------------------------------
# Static artifact contracts (patch, requirements, env, README, gunicorn_conf)
# ---------------------------------------------------------------------------


def test_reference_patch_targets_exact_native_files_and_fail_closed_surface():
    text = PATCH_PATH.read_text(encoding="utf-8")
    targets = set(re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.MULTILINE))
    assert targets == {
        "reward_server/general_reward.py",
        "reward_server/reward_3d.py",
        "reward_server/reward_3d_backend.py",
    }
    added = "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    for marker in (
        'STRICT_MANAGER_PROTOCOL = "world_r1_fail_closed_v1"',
        "STRICT_REWARD_KIND",
        "def is_ready",
        'get_context("spawn")',
        "ROW_OK",
        "ROW_ERROR",
        "INIT_ERROR",
        "_poison_and_shutdown",
        "STRICT_MANAGER_TIMEOUT_S = 1800.0",
        "STRICT_CLEANUP_TIMEOUT_S = 10.0",
        "STRICT_FATAL_EXIT_CODE = 70",
        "StrictWorkerDeathError",
        "StrictRewardTimeoutError",
        "StrictRewardDecodeError",
    ):
        assert marker in added, f"patch is missing {marker!r}"
    assert "ThreadPoolExecutor" not in added
    assert "= 0.5" not in added
    assert "append(0.0)" not in added
    # All lock/init/score waits consume the same monotonic deadline budget.
    assert added.count("deadline - time.monotonic()") >= 6
    # Removed native fallbacks must actually be deleted by the patch.
    removed = "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    assert "ThreadPoolExecutor" in removed
    assert "CUDA_VISIBLE_DEVICES" in removed


EXPECTED_REQUIREMENTS = [
    "--extra-index-url https://download.pytorch.org/whl/cu128",
    "torch==2.7.1+cu128",
    "torchvision==0.22.1+cu128",
    "numpy>=1.26,<2.3",
    "pillow>=10",
    "flask>=3,<4",
    "gunicorn>=23,<24",
    "transformers==4.57.6",
    "sentencepiece>=0.2,<0.3",
    "protobuf>=4.25,<7",
    "hpsv2==1.2.0",
    "qwen-vl-utils==0.0.14",
    "lpips==0.1.4",
    "scipy>=1.11,<2",
    "opencv-python>=4.10,<5",
    "matplotlib>=3.8,<4",
    "imageio>=2.34,<3",
    "einops>=0.8,<0.9",
    "addict>=2.4,<3",
    "omegaconf>=2.3,<3",
    "huggingface-hub>=0.28,<2",
    "trimesh>=4,<5",
    "plyfile>=1,<2",
    "moviepy==1.0.3",
    "pycolmap>=3.11,<4",
    "gsplat==1.5.3",
    "evo>=1.31,<2",
    "e3nn>=0.5,<1",
    "tqdm>=4.66",
    "ftfy>=6.2,<7",
]


def test_requirements_service_is_the_frozen_single_dependency_owner():
    lines = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines == EXPECTED_REQUIREMENTS
    assert len(lines) == len(set(lines))


def test_env_yml_declares_only_the_service_toolchain():
    env = yaml.safe_load(ENV_PATH.read_text(encoding="utf-8"))
    assert env["name"] == "world-r1-reward"
    assert env["channels"] == ["nvidia", "conda-forge"]
    assert env["dependencies"] == ["python=3.10", "pip", "cuda-nvcc=12.8"]


def test_readme_freezes_install_order_and_both_gunicorn_commands():
    text = README_PATH.read_text(encoding="utf-8")
    assert "python -m pip install -r services/world_r1_strict/requirements-service.txt" in text
    assert text.index("git apply --check") < text.index("pip install --no-deps -e")
    for app, port in (("reward_general_app", 8090), ("reward_3d_app", 8089)):
        command = (
            "WORLD_R1_SERVER_REVISION=world-r1-<patched-commit> \\\n"
            "python -m gunicorn \\\n"
            "  --chdir /absolute/path/to/framecode \\\n"
            "  --config python:services.world_r1_strict.gunicorn_conf \\\n"
            f"  --bind 127.0.0.1:{port} \\\n"
            "  --workers 1 --worker-class gthread --threads 4 \\\n"
            "  --timeout 1860 --graceful-timeout 30 \\\n"
            "  --access-logfile - --error-logfile - --capture-output \\\n"
            f"  'services.world_r1_strict.{app}:build_app()'"
        )
        assert command in text, f"README lost the frozen {app} command"
    # The frozen commands carry no preload/reload; each flag may appear at
    # most once, inside the sentence that forbids it.
    for flag in ("--preload", "--reload"):
        occurrences = [line for line in text.splitlines() if flag in line]
        assert len(occurrences) <= 1
        if occurrences:
            assert "forbidden" in occurrences[0]


def test_gunicorn_conf_ast_defines_only_the_worker_exit_hook():
    path = REPO_ROOT / "services/world_r1_strict/gunicorn_conf.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert [node.name for node in top_defs] == ["worker_exit"]
    # No server settings, app imports or heavy imports anywhere in the AST.
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not (assigned_names & {"bind", "workers", "threads", "timeout"})
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module)
    assert imported_modules == {"os", "services.world_r1_strict"}


# ---------------------------------------------------------------------------
# Real Gunicorn lifecycle integration (worker_exit actually fires)
# ---------------------------------------------------------------------------

_LIFECYCLE_EVENT_ENV = "WORLD_R1_STRICT_LIFECYCLE_EVENT_FILE"


def _append_event(path: str, event: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _lifecycle_child_main(event_file: str) -> None:
    _append_event(event_file, {"event": "child_started", "pid": os.getpid()})
    while True:
        time.sleep(3600)


class _LifecycleFakeManager:
    """Strict fake manager that spawns one recognizable descendant."""

    STRICT_MANAGER_PROTOCOL = MANAGER_CONTRACT
    STRICT_REWARD_KIND = REWARD_GENERAL

    def __init__(self, event_file: str) -> None:
        self._event_file = event_file
        self._ready = True
        context = multiprocessing.get_context("spawn")
        self._child = context.Process(
            target=_lifecycle_child_main, args=(event_file,), daemon=True
        )
        self._child.start()
        _append_event(
            event_file,
            {"event": "registered", "pid": os.getpid(), "child_pid": self._child.pid},
        )

    def is_ready(self) -> bool:
        return self._ready

    def compute_batch_scores(self, items, prompts):
        return [0.5] * len(items)

    def shutdown(self) -> None:
        self._ready = False
        if self._child.is_alive():
            self._child.terminate()
            self._child.join(timeout=5)
        _append_event(self._event_file, {"event": "closed", "pid": os.getpid()})


def build_lifecycle_test_app():
    """Subprocess-only app factory used by the Gunicorn lifecycle test."""

    event_file = os.environ[_LIFECYCLE_EVENT_ENV]
    manager = _LifecycleFakeManager(event_file)
    return reward_general_app.create_app(
        manager=manager, server_revision="world-r1-" + "0" * 12
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _cleanup_process_group(master: subprocess.Popen) -> None:
    if master.poll() is not None:
        return
    try:
        os.killpg(master.pid, signal.SIGCONT)
    except (ProcessLookupError, PermissionError):
        pass
    for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        if master.poll() is not None:
            break
        try:
            os.killpg(master.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            master.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    if master.poll() is None:
        master.kill()
        master.wait(timeout=5)


@pytest.mark.skipif(
    not (_HAS_FLASK and _HAS_GUNICORN and hasattr(os, "fork")),
    reason="requires flask, gunicorn and a fork-capable platform",
)
def test_gunicorn_worker_exit_closes_manager_once_and_leaves_no_descendants(tmp_path):
    event_file = tmp_path / "lifecycle-events.jsonl"
    port = _free_port()
    env = os.environ.copy()
    env[_LIFECYCLE_EVENT_ENV] = str(event_file)
    command = [
        sys.executable,
        "-m",
        "gunicorn",
        "--chdir",
        str(REPO_ROOT),
        "--config",
        "python:services.world_r1_strict.gunicorn_conf",
        "--bind",
        f"127.0.0.1:{port}",
        "--workers",
        "1",
        "--worker-class",
        "gthread",
        "--threads",
        "4",
        "--timeout",
        "1860",
        "--graceful-timeout",
        "30",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "--capture-output",
        "tests.test_world_r1_strict_server:build_lifecycle_test_app()",
    ]
    master = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    worker_pid: int | None = None
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 60
        registered = None
        while time.monotonic() < deadline:
            for event in _read_events(event_file):
                if event.get("event") == "registered":
                    registered = event
                    break
            if registered is not None:
                break
            if master.poll() is not None:
                break
            time.sleep(0.2)
        assert registered is not None, "gunicorn worker never registered its manager"
        worker_pid = int(registered["pid"])
        child_pid = int(registered["child_pid"])

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", HEALTH_ROUTE)
        health = connection.getresponse()
        health.read()
        connection.close()
        assert health.status == 200

        master.send_signal(signal.SIGTERM)
        master.wait(timeout=20)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if (
                not _pid_alive(worker_pid)
                and not _pid_alive(child_pid)
                and not _port_listening(port)
            ):
                break
            time.sleep(0.2)
        assert not _pid_alive(worker_pid), "gunicorn worker survived TERM"
        assert not _pid_alive(child_pid), "manager descendant survived shutdown"
        assert not _port_listening(port), "port still listening after shutdown"

        events = _read_events(event_file)
        close_events = [event for event in events if event.get("event") == "closed"]
        assert len(close_events) == 1, f"expected exactly one close event, got {events}"
        assert close_events[0]["pid"] == worker_pid
    finally:
        _cleanup_process_group(master)
