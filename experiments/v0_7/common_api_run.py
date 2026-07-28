"""The sole training-side high-level API callsite for v0.7 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import struct
from typing import Any

import visual_rl as vr

MAX_CONTROL_PAYLOAD_BYTES = 4096
ACK = b"ACK"


@dataclass(frozen=True)
class PreparedRunMessage:
    """Validated, non-secret readiness projection sent to the controller."""

    output_dir: str
    checks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PreparedRunEnvelope:
    """One rank's authenticated-by-possession readiness message."""

    attempt_id: str
    nonce: str
    rank: int
    world_size: int
    prepared: PreparedRunMessage


@dataclass(frozen=True)
class CompletedRun:
    """Public-API-only projection returned after a successful audited run."""

    run_id: str
    output_dir: str
    committed_steps: int
    audit_ok: bool


def canonical_json_bytes(value: object) -> bytes:
    """Encode one canonical finite JSON value."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_frame(value: object) -> bytes:
    """Frame one bounded control value with a four-byte big-endian length."""

    payload = canonical_json_bytes(value)
    if not 1 <= len(payload) <= MAX_CONTROL_PAYLOAD_BYTES:
        raise ValueError("control payload must contain 1..4096 bytes")
    return struct.pack(">I", len(payload)) + payload


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Receive and strictly decode one bounded framed JSON object."""

    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    if not 1 <= length <= MAX_CONTROL_PAYLOAD_BYTES:
        raise ValueError("control payload must contain 1..4096 bytes")
    payload = _recv_exact(sock, length)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("control payload must be canonical UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("control payload root must be an object")
    if canonical_json_bytes(value) != payload:
        raise ValueError("control payload is not canonical JSON")
    return value


def resolve_only(config_path: str | Path) -> object:
    """Resolve a full YAML through the sole public loader without validation."""

    return vr.load(config_path).resolve()


def run_config(
    config_path: str | Path,
    *,
    barrier_path: str | Path,
    attempt_id: str,
    nonce: str,
    repo_root: str | Path,
) -> CompletedRun:
    """Validate, cross the controller barrier, run once, inspect, and audit."""

    experiment = vr.load(config_path)
    resolved = experiment.resolve()
    report = experiment.validate()
    if not report.ok:
        messages = "; ".join(check.message for check in report.errors)
        raise RuntimeError(f"experiment validation failed: {messages}")
    if report.runtime_rank is None or report.runtime_world_size is None:
        raise RuntimeError("validation did not provide runtime rank/world size")

    output_dir = _repo_relative(resolved.artifacts.output_dir, Path(repo_root))
    checks = tuple(
        {
            "code": check.code,
            "level": check.level,
            "path": check.path,
            "volatile": check.volatile,
        }
        for check in report.checks
    )
    envelope = PreparedRunEnvelope(
        attempt_id=_nonempty("attempt_id", attempt_id),
        nonce=_nonempty("nonce", nonce),
        rank=report.runtime_rank,
        world_size=report.runtime_world_size,
        prepared=PreparedRunMessage(output_dir=output_dir, checks=checks),
    )
    _cross_barrier(Path(barrier_path), envelope)

    result = experiment.run()
    status = vr.inspect_run(result.output_dir)
    audit = vr.audit_run(result.output_dir)
    if not status.ok or not audit.ok:
        raise RuntimeError("completed run failed public status/audit")
    if (
        status.run_id != result.run_id
        or audit.run_id != result.run_id
        or status.committed_steps != result.committed_steps
        or audit.committed_steps != result.committed_steps
    ):
        raise RuntimeError("run result/status/audit identity mismatch")
    return CompletedRun(
        run_id=result.run_id,
        output_dir=_repo_relative(result.output_dir, Path(repo_root)),
        committed_steps=result.committed_steps,
        audit_ok=True,
    )


def _cross_barrier(path: Path, envelope: PreparedRunEnvelope) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(path))
        sock.sendall(encode_frame(asdict(envelope)))
        if _recv_exact(sock, len(ACK)) != ACK:
            raise RuntimeError("controller barrier did not return exact ACK")
        if sock.recv(1) != b"":
            raise RuntimeError("controller barrier must close after ACK")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("unexpected EOF in framed control message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _repo_relative(path: Path, repo_root: Path) -> str:
    root = repo_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("experiment output_dir must be inside the repository") from exc


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
