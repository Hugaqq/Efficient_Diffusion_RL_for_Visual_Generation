"""Fixed single-node, two-GPU NCCL correctness and Tiny training suite."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
from typing import Any
from collections.abc import Mapping

import visual_rl as vr

from experiments.v0_7 import common_api_run
from experiments.v0_7.interrupt_resume import (
    CONFIG_ROOT,
    EVIDENCE_ROOT,
    REPO_ROOT,
    ROLE_BY_NAME,
    ProcessIdentity,
    RunObservation,
    _accept_ready,
    _ack_connections,
    _capture_owned_group,
    _capture_owned_process,
    _execution_candidate,
    _append_attempt_environment,
    _invalidate_family_evidence,
    _interrupt_at_marker,
    _semantic_gate,
    _terminate_owned_process,
    _record_family_evidence,
    assert_config_family,
    roles_for_family,
)

INTERNAL_NCCL_COMMANDS = (
    (
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "pytest",
        "-q",
        "-s",
        "tests/test_distributed_update.py::test_nccl_fixed_batch_matches_single_process",
    ),
    (
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "pytest",
        "-q",
        "-s",
        "tests/test_distributed_update.py::test_nccl_one_rank_update_failure_synchronizes",
    ),
    (
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "pytest",
        "-q",
        "-s",
        "tests/test_distributed_runner.py::test_nccl_root_commit_failure_synchronizes",
    ),
)

ROLE_MODULES = {
    "mg1_tiny_grpo_c20_continuous": "experiments.v0_7.mg1_nccl_continuous",
    "mg1_tiny_grpo_c20_interrupted": "experiments.v0_7.mg1_nccl_interrupted",
    "mg1_tiny_grpo_c20_resume": "experiments.v0_7.mg1_nccl_resume",
}


@dataclass(frozen=True)
class MG1Control:
    attempt_id: str
    nonce: str
    barrier_path: str


def run_mg1_suite() -> tuple[RunObservation, ...]:
    """Run fixed internal NCCL gates, then the fixed three-role MG1 lineage."""

    candidate = _execution_candidate()
    _invalidate_family_evidence("mg1_tiny_grpo", candidate)
    internal_path = EVIDENCE_ROOT / "mg1_internal.json"
    internal_path.unlink(missing_ok=True)
    assert_config_family("mg1_tiny_grpo")
    internal_results: list[dict[str, object]] = []
    for command in INTERNAL_NCCL_COMMANDS:
        passed = _run_fixed_command(command, timeout_s=900.0) == 0
        internal_results.append({"nodeid": command[-1], "passed": passed})
        _write_internal_evidence(candidate, internal_results)
        if not passed:
            raise RuntimeError(f"MG1 internal NCCL gate failed: {command[-1]}")
    observations = tuple(
        _run_role(spec.name, candidate=candidate)
        for spec in roles_for_family("mg1_tiny_grpo")
    )
    if not _semantic_gate(observations):
        raise RuntimeError("MG1 continuous/resume semantic parity failed")
    _record_family_evidence(
        "mg1_tiny_grpo",
        observations,
        semantic_pass=True,
        candidate=candidate,
    )
    return observations


def run_mg1_role(role_name: str) -> common_api_run.CompletedRun:
    """Worker entry used only by one of the three literal MG1 modules."""

    if role_name not in ROLE_MODULES:
        raise ValueError(f"unknown MG1 role: {role_name}")
    control = _load_control(role_name)
    return common_api_run.run_config(
        CONFIG_ROOT / ROLE_BY_NAME[role_name].config,
        barrier_path=control.barrier_path,
        attempt_id=control.attempt_id,
        nonce=control.nonce,
        repo_root=REPO_ROOT,
    )


def _run_role(
    role_name: str,
    *,
    candidate: Mapping[str, object],
) -> RunObservation:
    spec = ROLE_BY_NAME[role_name]
    config = CONFIG_ROOT / spec.config
    resolved = common_api_run.resolve_only(config)
    output_dir = resolved.artifacts.output_dir
    attempt_id = f"{role_name}-{secrets.token_hex(8)}"
    nonce = secrets.token_hex(16)
    _append_attempt_environment(
        role_name,
        attempt_id=attempt_id,
        candidate=candidate,
    )
    control_path = _control_path(role_name)
    if control_path.exists():
        raise RuntimeError(f"stale MG1 control file exists: {control_path}")
    with tempfile.TemporaryDirectory(prefix="visualrl-v07-mg1-") as temporary:
        barrier_path = Path(temporary) / "ready.sock"
        process: subprocess.Popen[bytes] | None = None
        identity: ProcessIdentity | None = None
        captured_members: tuple[ProcessIdentity, ...] = ()
        connections: tuple[socket.socket, ...] = ()
        try:
            control_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            control_path.write_bytes(
                common_api_run.canonical_json_bytes(
                    {
                        "attempt_id": attempt_id,
                        "barrier_path": str(barrier_path),
                        "nonce": nonce,
                    }
                )
            )
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(barrier_path))
                server.listen(2)
                server.settimeout(60.0)
                process = subprocess.Popen(
                    (
                        "torchrun",
                        "--standalone",
                        "--nproc-per-node=2",
                        "--module",
                        ROLE_MODULES[role_name],
                    ),
                    cwd=REPO_ROOT,
                    start_new_session=True,
                )
                identity = _capture_owned_process(process.pid)
                connections = _accept_ready(
                    server,
                    attempt_id=attempt_id,
                    nonce=nonce,
                    world_size=2,
                    expected_output_dir=output_dir.relative_to(REPO_ROOT).as_posix(),
                )
                captured_members = _capture_owned_group(identity)
                _ack_connections(
                    connections,
                    terminate_before_eof=lambda: _terminate_owned_process(
                        identity,
                        captured_members,
                    ),
                )
                if spec.phase == "interrupted":
                    _interrupt_at_marker(
                        identity,
                        output_dir,
                        spec.target_steps,
                    )
                try:
                    exit_code = process.wait(timeout=60.0)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(
                        f"{role_name} did not exit after its run"
                    ) from exc
        finally:
            try:
                if process is not None:
                    if identity is None:
                        if process.poll() is None:
                            process.kill()
                    else:
                        _terminate_owned_process(identity, captured_members)
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError(
                            f"could not reap MG1 process group for {role_name}"
                        ) from exc
            finally:
                for connection in connections:
                    connection.close()
                control_path.unlink(missing_ok=True)

    status = vr.inspect_run(output_dir)
    audit = vr.audit_run(output_dir)
    if exit_code != spec.expected_exit:
        raise RuntimeError(f"{role_name} exit {exit_code}, expected {spec.expected_exit}")
    if status.committed_steps != spec.target_steps:
        raise RuntimeError(f"{role_name} did not reach its target marker")
    if spec.phase != "interrupted" and not audit.ok:
        raise RuntimeError(f"{role_name} did not finish audit-ok")
    return RunObservation(
        role=role_name,
        exit_code=exit_code,
        committed_steps=status.committed_steps,
        audit_ok=audit.ok,
        output_dir=output_dir.relative_to(REPO_ROOT).as_posix(),
    )


def _control_path(role_name: str) -> Path:
    return Path("/tmp") / f"visualrl-v07-{os.getuid()}-{role_name}.json"


def _load_control(role_name: str) -> MG1Control:
    path = _control_path(role_name)
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read MG1 control file: {path}") from exc
    if not isinstance(value, dict) or set(value) != {
        "attempt_id",
        "barrier_path",
        "nonce",
    }:
        raise RuntimeError("invalid MG1 control object")
    if any(not isinstance(value[key], str) or not value[key] for key in value):
        raise RuntimeError("invalid MG1 control value")
    return MG1Control(**value)


def _run_fixed_command(
    command: tuple[str, ...],
    *,
    timeout_s: float,
) -> int:
    """Run one fixed torchrun command with an owned process-group boundary."""

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        start_new_session=True,
    )
    identity: ProcessIdentity | None = None
    captured_members: tuple[ProcessIdentity, ...] = ()
    try:
        identity = _capture_owned_process(process.pid)
        captured_members = _capture_fixed_group_members(
            process,
            identity,
            minimum_members=3,
            timeout_s=min(timeout_s, 30.0),
        )
        if len(captured_members) < 3:
            raise RuntimeError(
                "fixed MG1 command did not expose its complete process group: "
                f"captured {len(captured_members)} of 3 members"
            )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"fixed MG1 command timed out: {command[-1]}") from exc
    finally:
        if identity is None:
            if process.poll() is None:
                process.kill()
        else:
            _terminate_owned_process(identity, captured_members)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"could not reap fixed MG1 process group: {command[-1]}"
            ) from exc


def _capture_fixed_group_members(
    process: subprocess.Popen[bytes],
    identity: ProcessIdentity,
    *,
    minimum_members: int,
    timeout_s: float,
) -> tuple[ProcessIdentity, ...]:
    """Capture torchrun workers before waiting on a fallible group leader."""

    deadline = time.monotonic() + timeout_s
    captured: dict[tuple[int, str], ProcessIdentity] = {}
    while True:
        try:
            current = _capture_owned_group(identity)
        except (ProcessLookupError, RuntimeError):
            break
        captured.update(
            {(member.pid, member.create_time): member for member in current}
        )
        if len(captured) >= minimum_members or process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    return tuple(
        sorted(
            captured.values(),
            key=lambda member: (member.pid, member.create_time),
        )
    )


def _write_internal_evidence(
    candidate: Mapping[str, object],
    results: list[dict[str, object]],
) -> None:
    from experiments.v0_7.environment_report import atomic_write_bytes

    atomic_write_bytes(
        EVIDENCE_ROOT / "mg1_internal.json",
        common_api_run.canonical_json_bytes(
            {
                "candidate": dict(candidate),
                "schema_version": 1,
                "tests": results,
            }
        )
        + b"\n",
    )


if __name__ == "__main__":
    run_mg1_suite()
