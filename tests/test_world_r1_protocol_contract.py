"""Contract tests for visual_rl.core.protocols.world_r1 (the single protocol owner)."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from visual_rl.core.protocols import world_r1 as wrp

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_REVISIONS = [
    "world-r1-0123456789ab",
    "world-r1-" + "a" * 12,
    "world-r1-" + "0123456789abcdef" * 2 + "01234567",  # 40 hex chars
    "world-r1-" + "f" * 40,
]

INVALID_REVISIONS = [
    "",
    "world-r1-",
    "world-r1-" + "a" * 11,
    "world-r1-" + "a" * 41,
    "world-r1-" + "A" * 12,
    "world-r1-0123456789AB",
    "WORLD-R1-" + "a" * 12,
    "world-r2-" + "a" * 12,
    "world-r1-" + "g" * 12,
    "https://scorer.example/" + "world-r1-" + "a" * 12,
    "/absolute/path/" + "world-r1-" + "a" * 12,
    "../relative/" + "world-r1-" + "a" * 12,
    "world-r1-" + "a" * 12 + "/suffix",
    " world-r1-" + "a" * 12,
    "world-r1-" + "a" * 12 + " ",
    "world-r1-" + "a" * 11 + "\t",
    "world-r1-" + "a" * 11 + "\n",
    "token-" + "x" * 30,
    "world-r1-" + "a" * 12 + "\x00",
]

IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# Asymmetric matrix with distinct rotation/translation elements so a
# transpose or element smear cannot pass the round-trip equality check.
ASYMMETRIC_4X4 = [
    [0.0, -1.0, 0.0, 10.5],
    [1.0, 0.0, 0.0, -3.25],
    [0.0, 0.0, 1.0, 2.75],
    [0.0, 0.0, 0.0, 1.0],
]


@pytest.mark.parametrize("revision", VALID_REVISIONS)
def test_validate_server_revision_accepts_public_grammar(revision):
    assert wrp.validate_server_revision(revision) == revision


@pytest.mark.parametrize("revision", INVALID_REVISIONS)
def test_validate_server_revision_rejects_invalid_grammar(revision):
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_server_revision(revision)


@pytest.mark.parametrize("value", [None, 0, 12.5, b"world-r1-" + b"a" * 12, ["x"], {"x": 1}])
def test_validate_server_revision_rejects_non_strings(value):
    with pytest.raises(TypeError):
        wrp.validate_server_revision(value)


def _production_python_files():
    roots = [REPO_ROOT / "visual_rl", REPO_ROOT / "services"]
    return [
        path
        for root in roots
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def test_revision_grammar_has_exactly_one_owner():
    offenders = []
    for path in _production_python_files():
        text = path.read_text(encoding="utf-8")
        if "world-r1-[0-9a-f]{12,40}" in text or "SERVER_REVISION_PATTERN" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == ["visual_rl/core/protocols/world_r1.py"]


def test_service_modules_call_the_shared_validator_without_local_copies():
    checked = {
        "services/world_r1_strict/protocol.py",
        "services/world_r1_strict/reward_general_app.py",
        "services/world_r1_strict/reward_3d_app.py",
    }
    for relative in checked:
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = False
        local_defs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "visual_rl.core.protocols.world_r1":
                if any(alias.name == "validate_server_revision" for alias in node.names):
                    imported = True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                "revision" in node.name
            ):
                local_defs.append(node.name)
        assert imported, f"{relative} must import validate_server_revision from the owner"
        assert not local_defs, f"{relative} defines local revision validators: {local_defs}"


def test_sample_id_validation_and_echo():
    ids = wrp.validate_sample_ids(["s0", "s1", "s2"])
    assert ids == ["s0", "s1", "s2"]
    assert wrp.require_sample_id_echo(["s0", "s1", "s2"], ["s0", "s1", "s2"]) == ids
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.require_sample_id_echo(["s0", "s1", "s2"], ["s0", "s2", "s1"])
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_sample_ids([])
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_sample_ids(["s0", "s0"])
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_sample_ids(["s0", "  "])
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_sample_ids(["s0"], expected=2)
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_sample_ids("s0")


def test_revision_echo():
    revision = VALID_REVISIONS[0]
    assert wrp.require_revision_echo(revision, revision) == revision
    with pytest.raises(wrp.WorldR1RevisionError):
        wrp.require_revision_echo(revision, VALID_REVISIONS[1])


def test_json_size_limits_and_finite_decode():
    payload = {"prompts": ["a"], "n": 1}
    body = wrp.encode_json(payload)
    assert wrp.decode_json(body, max_bytes=len(body)) == payload
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.decode_json(body, max_bytes=len(body) - 1)
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.decode_json(b'{"x": NaN}', max_bytes=1024)
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.decode_json(b"[1, 2]", max_bytes=1024)
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.encode_json({"x": float("nan")})
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_json_size(11, max_bytes=10)


def test_camera_typed_numeric_format_round_trip():
    trajectory = [ASYMMETRIC_4X4, IDENTITY_4X4]
    typed = wrp.validate_camera_trajectory(trajectory, entry=0, expected_frames=2)
    for row_expected, row_actual in zip(ASYMMETRIC_4X4, typed[0], strict=True):
        assert row_actual == pytest.approx(row_expected, rel=0, abs=0)
        assert row_actual == row_expected
    assert typed[1] == [[float(v) for v in row] for row in IDENTITY_4X4]


@pytest.mark.parametrize(
    "bad",
    [
        "identity",
        {"frame0": IDENTITY_4X4},
        [[1.0] * 4] * 3,
        [[1.0] * 3] * 4,
        [[1.0] * 4] * 5,
        [[True, 0.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0, 0.0]] * 3,
        [[float("nan"), 0.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0, 0.0]] * 3,
        [[float("inf"), 0.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0, 0.0]] * 3,
        [["1.0", 0.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0, 0.0]] * 3,
    ],
)
def test_camera_matrix_rejects_bool_string_shape_and_nonfinite(bad):
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_camera_matrix(bad, entry=0, frame_index=0)


def test_camera_trajectory_rejects_frame_mapping_and_wrong_frame_count():
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_camera_trajectory({"frame0": IDENTITY_4X4}, entry=0)
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_camera_trajectory([], entry=0)
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_camera_trajectory([IDENTITY_4X4], entry=0, expected_frames=2)


def test_health_schema_build_and_validate():
    revision = VALID_REVISIONS[0]
    payload = wrp.build_health_payload(reward=wrp.REWARD_3D, server_revision=revision)
    assert payload == {
        "status": "ok",
        "protocol_version": "strict_v2",
        "wire_format": "json_v1",
        "reward": "reward_3d",
        "server_revision": revision,
        "manager_contract": "world_r1_fail_closed_v1",
    }
    assert wrp.validate_health_payload(
        payload, reward=wrp.REWARD_3D, server_revision=revision
    ) == payload
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_health_payload(payload, reward=wrp.REWARD_GENERAL, server_revision=revision)
    with pytest.raises(wrp.WorldR1RevisionError):
        wrp.validate_health_payload(payload, reward=wrp.REWARD_3D, server_revision=VALID_REVISIONS[1])
    for key, value in (
        ("status", "degraded"),
        ("protocol_version", "reference_v1"),
        ("wire_format", "legacy_pickle"),
        ("manager_contract", "other"),
    ):
        broken = {**payload, key: value}
        with pytest.raises(wrp.WorldR1ProtocolError):
            wrp.validate_health_payload(broken, reward=wrp.REWARD_3D, server_revision=revision)
    extra = {**payload, "extra": 1}
    with pytest.raises(wrp.WorldR1ProtocolError):
        wrp.validate_health_payload(extra, reward=wrp.REWARD_3D, server_revision=revision)


def test_score_response_requires_exact_keys_echo_and_all_true_mask():
    revision = VALID_REVISIONS[0]
    body = {
        "protocol_version": "strict_v2",
        "server_revision": revision,
        "sample_id": ["a", "b"],
        "outputs": [0.5, 1],
        "valid_mask": [True, True],
    }
    assert wrp.validate_score_response(
        body, expected_sample_ids=["a", "b"], server_revision=revision
    ) == [0.5, 1.0]
    cases = [
        {**body, "sample_id": ["b", "a"]},
        {**body, "outputs": [0.5]},
        {**body, "outputs": [0.5, float("nan")]},
        {**body, "valid_mask": [True, False]},
        {**body, "valid_mask": [True, 1]},
        {k: v for k, v in body.items() if k != "valid_mask"},
    ]
    for broken in cases:
        with pytest.raises(wrp.WorldR1ProtocolError):
            wrp.validate_score_response(
                broken, expected_sample_ids=["a", "b"], server_revision=revision
            )


def test_protocol_modules_import_without_torch():
    code = (
        "import sys\n"
        "import visual_rl.core.protocols.world_r1\n"
        "import services.world_r1_strict.protocol\n"
        "import services.world_r1_strict.reference_contract\n"
        "import services.world_r1_strict.gunicorn_conf\n"
        "forbidden = {'torch', 'torchvision', 'transformers'} & set(sys.modules)\n"
        "assert not forbidden, forbidden\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
