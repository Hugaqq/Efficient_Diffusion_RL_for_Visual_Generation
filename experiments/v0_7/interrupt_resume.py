"""Fixed v0.7 role table, family validation, and interruption controller."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import secrets
import signal
import socket
import subprocess
import tempfile
import time

import visual_rl as vr

from experiments.v0_7 import common_api_run
from experiments.v0_7.environment_report import (
    append_environment_report,
    atomic_write_bytes,
    current_environment_report,
    evidence_candidate,
    probe_git_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
EVIDENCE_ROOT = Path(__file__).resolve().parent / "evidence"
FLOW_NATIVE_ITEMS = {
    "checkpoint_resume",
    "current_log_prob",
    "gradient",
    "group_advantage",
    "initial_latent",
    "old_log_prob",
    "parameter_delta",
    "policy_loss",
    "prompt_encoding",
    "reference_kl",
    "rollout_latent",
    "timestep",
    "total_loss",
    "transition_statistics",
}


@dataclass(frozen=True)
class RoleSpec:
    """The sole role/config/step/lineage contract for one fixed run."""

    name: str
    family: str
    phase: str
    config: str
    target_steps: int
    expected_exit: int
    lineage_head: str
    world_size: int


@dataclass(frozen=True)
class RunObservation:
    """Public evidence projection produced by one controller role."""

    role: str
    exit_code: int
    committed_steps: int
    audit_ok: bool
    output_dir: str


@dataclass(frozen=True)
class ProcessIdentity:
    """Identity captured after a child becomes its own process-group leader."""

    pid: int
    create_time: str
    pgid: int


FAMILY_ORDER = (
    "tiny_s100",
    "flow_grpo_sd3",
    "tempflow_sd3",
    "flash_wan",
    "world_r1_wan",
    "mg1_tiny_grpo",
)


def _role(
    name: str,
    family: str,
    phase: str,
    target_steps: int,
    *,
    expected_exit: int = 0,
    lineage_head: str | None = None,
    world_size: int = 1,
) -> RoleSpec:
    return RoleSpec(
        name=name,
        family=family,
        phase=phase,
        config=f"{name}.yaml",
        target_steps=target_steps,
        expected_exit=expected_exit,
        lineage_head=lineage_head or name,
        world_size=world_size,
    )


ROLE_SPECS = (
    _role("tiny_s100_continuous", "tiny_s100", "continuous", 100),
    _role(
        "tiny_s100_interrupted",
        "tiny_s100",
        "interrupted",
        50,
        expected_exit=-signal.SIGKILL,
    ),
    _role(
        "tiny_s100_resume",
        "tiny_s100",
        "resume",
        100,
        lineage_head="tiny_s100_interrupted",
    ),
    _role("flow_grpo_sd3_c20_continuous", "flow_grpo_sd3", "continuous", 20),
    _role(
        "flow_grpo_sd3_c20_interrupted",
        "flow_grpo_sd3",
        "interrupted",
        10,
        expected_exit=-signal.SIGKILL,
    ),
    _role(
        "flow_grpo_sd3_c20_resume",
        "flow_grpo_sd3",
        "resume",
        20,
        lineage_head="flow_grpo_sd3_c20_interrupted",
    ),
    _role("flow_grpo_sd3_q100_seed17", "flow_grpo_sd3", "q100", 100),
    _role("flow_grpo_sd3_q100_seed29", "flow_grpo_sd3", "q100", 100),
    _role("flow_grpo_sd3_q100_seed43", "flow_grpo_sd3", "q100", 100),
    _role("tempflow_sd3_c20_continuous", "tempflow_sd3", "continuous", 20),
    _role(
        "tempflow_sd3_c20_interrupted",
        "tempflow_sd3",
        "interrupted",
        10,
        expected_exit=-signal.SIGKILL,
    ),
    _role(
        "tempflow_sd3_c20_resume",
        "tempflow_sd3",
        "resume",
        20,
        lineage_head="tempflow_sd3_c20_interrupted",
    ),
    _role("tempflow_sd3_q100_seed17", "tempflow_sd3", "q100", 100),
    _role("tempflow_sd3_q100_seed29", "tempflow_sd3", "q100", 100),
    _role("tempflow_sd3_q100_seed43", "tempflow_sd3", "q100", 100),
    _role("flash_wan_c20_continuous", "flash_wan", "continuous", 20),
    _role(
        "flash_wan_c20_interrupted",
        "flash_wan",
        "interrupted",
        10,
        expected_exit=-signal.SIGKILL,
    ),
    _role(
        "flash_wan_c20_resume",
        "flash_wan",
        "resume",
        20,
        lineage_head="flash_wan_c20_interrupted",
    ),
    _role("flash_wan_q100_seed17", "flash_wan", "q100", 100),
    _role("flash_wan_q100_seed29", "flash_wan", "q100", 100),
    _role("flash_wan_q100_seed43", "flash_wan", "q100", 100),
    _role("world_r1_wan_c20_continuous", "world_r1_wan", "continuous", 20),
    _role(
        "world_r1_wan_c20_interrupted",
        "world_r1_wan",
        "interrupted",
        10,
        expected_exit=-signal.SIGKILL,
    ),
    _role(
        "world_r1_wan_c20_resume",
        "world_r1_wan",
        "resume",
        20,
        lineage_head="world_r1_wan_c20_interrupted",
    ),
    _role("world_r1_wan_q100_seed17", "world_r1_wan", "q100", 100),
    _role("world_r1_wan_q100_seed29", "world_r1_wan", "q100", 100),
    _role("world_r1_wan_q100_seed43", "world_r1_wan", "q100", 100),
    _role(
        "mg1_tiny_grpo_c20_continuous",
        "mg1_tiny_grpo",
        "continuous",
        20,
        world_size=2,
    ),
    _role(
        "mg1_tiny_grpo_c20_interrupted",
        "mg1_tiny_grpo",
        "interrupted",
        10,
        expected_exit=-signal.SIGKILL,
        world_size=2,
    ),
    _role(
        "mg1_tiny_grpo_c20_resume",
        "mg1_tiny_grpo",
        "resume",
        20,
        lineage_head="mg1_tiny_grpo_c20_interrupted",
        world_size=2,
    ),
)

ROLE_BY_NAME = {spec.name: spec for spec in ROLE_SPECS}
if len(ROLE_BY_NAME) != len(ROLE_SPECS):
    raise RuntimeError("duplicate v0.7 role name")


def roles_for_family(family: str) -> tuple[RoleSpec, ...]:
    if family not in FAMILY_ORDER:
        raise ValueError(f"unknown experiment family: {family}")
    return tuple(spec for spec in ROLE_SPECS if spec.family == family)


def assert_config_family(
    family: str,
    *,
    resolver: Callable[[Path], object] = common_api_run.resolve_only,
) -> dict[str, object]:
    """Resolve every role before launch and enforce the frozen family contract."""

    specs = roles_for_family(family)
    resolved = {spec.name: resolver(CONFIG_ROOT / spec.config) for spec in specs}
    for spec in specs:
        _assert_role_invariants(spec, resolved[spec.name])

    continuous = next(spec for spec in specs if spec.phase == "continuous")
    baseline = resolved[continuous.name]
    for spec in specs:
        candidate = resolved[spec.name]
        if spec.phase in {"continuous", "interrupted", "resume"}:
            allowed = {"artifacts.output_dir", "resume.from_"}
        elif spec.phase == "q100":
            allowed = {
                "run.seed",
                "runtime.max_steps",
                "artifacts.output_dir",
            }
        else:
            raise RuntimeError(f"unsupported role phase: {spec.phase}")
        differences = _diff_paths(
            _canonical_projection(baseline),
            _canonical_projection(candidate),
        )
        unexpected = differences - allowed
        if unexpected:
            raise ValueError(
                f"{family} role {spec.name} drifted outside allowlist: "
                f"{sorted(unexpected)}"
            )
    _assert_lineage(specs, resolved)
    return resolved


def prompt_window_counts(
    prompt_count: int,
    *,
    group_size: int,
    start: int,
    stop: int,
) -> tuple[int, ...]:
    """Return absolute-position sequential record counts for one step window."""

    if prompt_count <= 0 or group_size <= 0 or start < 0 or stop < start:
        raise ValueError("invalid prompt balance inputs")
    counts = [0] * prompt_count
    for step in range(start, stop):
        counts[step % prompt_count] += group_size
    return tuple(counts)


def assert_prompt_balance(config: object) -> None:
    prompts = _prompts(config)
    count = len(prompts)
    if count not in {2, 36}:
        raise ValueError("evidence prompt set must contain exactly 2 or 36 prompts")
    group_size = _group_size(config)
    for start, stop in ((0, 36), (64, 100)):
        counts = prompt_window_counts(
            count,
            group_size=group_size,
            start=start,
            stop=stop,
        )
        if len(set(counts)) != 1:
            raise ValueError(f"prompt window {start}..{stop - 1} is imbalanced")


def compare_semantic_projections(
    continuous: Mapping[str, object],
    resumed: Mapping[str, object],
) -> bool:
    """Compare already-public/audited projections without loading checkpoints."""

    required = {"checkpoint_digest", "manifest", "core_metrics", "audit_ok"}
    if set(continuous) != required or set(resumed) != required:
        raise ValueError("semantic projections have unexpected fields")
    if continuous["audit_ok"] is not True or resumed["audit_ok"] is not True:
        return False
    left = dict(continuous)
    right = dict(resumed)
    left["manifest"] = _normalize_run_id(left["manifest"])
    right["manifest"] = _normalize_run_id(right["manifest"])
    return left == right


def run_family(family: str) -> tuple[RunObservation, ...]:
    """Run one fixed family after an all-config pre-launch assertion."""

    if family == "mg1_tiny_grpo":
        from experiments.v0_7.mg1_nccl import run_mg1_suite

        return run_mg1_suite()
    candidate = _execution_candidate()
    _invalidate_family_evidence(family, candidate)
    if family == "flow_grpo_sd3":
        (EVIDENCE_ROOT / "flow_native.json").unlink(missing_ok=True)
    resolved = assert_config_family(family)
    del resolved
    specs = roles_for_family(family)
    c20 = tuple(spec for spec in specs if spec.phase != "q100")
    observations = tuple(_run_role(spec, candidate=candidate) for spec in c20)
    if not _mechanical_gate(observations, c20):
        raise RuntimeError(f"{family} C20 mechanical gate failed")
    semantic_pass = _semantic_gate(observations)
    if not semantic_pass:
        raise RuntimeError(f"{family} continuous/resume semantic parity failed")
    if family == "flow_grpo_sd3" and not _flow_native_gate(candidate):
        raise RuntimeError("Flow-GRPO native parity gate failed")
    q100 = tuple(spec for spec in specs if spec.phase == "q100")
    all_observations = observations + tuple(
        _run_role(spec, candidate=candidate) for spec in q100
    )
    _record_family_evidence(
        family,
        all_observations,
        semantic_pass=semantic_pass,
        candidate=candidate,
    )
    return all_observations


def _run_role(
    spec: RoleSpec,
    *,
    candidate: Mapping[str, object],
) -> RunObservation:
    config = CONFIG_ROOT / spec.config
    resolved = common_api_run.resolve_only(config)
    output_dir = resolved.artifacts.output_dir
    attempt_id = f"{spec.name}-{secrets.token_hex(8)}"
    nonce = secrets.token_hex(16)
    _append_attempt_environment(
        spec.name,
        attempt_id=attempt_id,
        candidate=candidate,
    )
    with tempfile.TemporaryDirectory(prefix="visualrl-v07-") as temporary:
        barrier_path = Path(temporary) / "ready.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(barrier_path))
            server.listen(spec.world_size)
            server.settimeout(60.0)
            context = multiprocessing.get_context("spawn")
            child = context.Process(
                target=_child_entry,
                args=(config, barrier_path, attempt_id, nonce),
            )
            identity: ProcessIdentity | None = None
            connections: tuple[socket.socket, ...] = ()
            child_started = False
            try:
                child.start()
                child_started = True
                identity = _capture_owned_process(child.pid)
                connections = _accept_ready(
                    server,
                    attempt_id=attempt_id,
                    nonce=nonce,
                    world_size=spec.world_size,
                    expected_output_dir=output_dir.relative_to(REPO_ROOT).as_posix(),
                )
                _ack_connections(
                    connections,
                    terminate_before_eof=lambda: _terminate_owned_process(identity),
                )

                if spec.phase == "interrupted":
                    _interrupt_at_marker(identity, output_dir, spec.target_steps)
                child.join(timeout=30.0)
                if child.is_alive():
                    raise TimeoutError(f"{spec.name} did not exit after its run")
                exit_code = int(
                    child.exitcode if child.exitcode is not None else 1
                )
            finally:
                try:
                    if child_started:
                        if child.is_alive():
                            if identity is None:
                                child.kill()
                            else:
                                _terminate_owned_process(identity)
                        child.join(timeout=5.0)
                        if child.is_alive():
                            raise RuntimeError(
                                f"could not reap process group for {spec.name}"
                            )
                finally:
                    for connection in connections:
                        connection.close()

    status = vr.inspect_run(output_dir)
    audit = vr.audit_run(output_dir)
    observation = RunObservation(
        role=spec.name,
        exit_code=exit_code,
        committed_steps=status.committed_steps,
        audit_ok=audit.ok,
        output_dir=output_dir.relative_to(REPO_ROOT).as_posix(),
    )
    if exit_code != spec.expected_exit:
        raise RuntimeError(
            f"{spec.name} exit {exit_code}, expected {spec.expected_exit}"
        )
    if status.committed_steps != spec.target_steps:
        raise RuntimeError(
            f"{spec.name} committed {status.committed_steps}, "
            f"expected {spec.target_steps}"
        )
    if spec.phase != "interrupted" and not audit.ok:
        raise RuntimeError(f"{spec.name} did not finish with an audit-ok head")
    return observation


def _child_entry(
    config: Path,
    barrier_path: Path,
    attempt_id: str,
    nonce: str,
) -> None:
    os.setsid()
    common_api_run.run_config(
        config,
        barrier_path=barrier_path,
        attempt_id=attempt_id,
        nonce=nonce,
        repo_root=REPO_ROOT,
    )


def _accept_ready(
    server: socket.socket,
    *,
    attempt_id: str,
    nonce: str,
    world_size: int,
    expected_output_dir: str,
) -> tuple[socket.socket, ...]:
    if (
        not isinstance(attempt_id, str)
        or not attempt_id
        or not isinstance(nonce, str)
        or not nonce
    ):
        raise RuntimeError("readiness identity must be non-empty")
    if type(world_size) is not int or world_size <= 0:
        raise RuntimeError("readiness world_size must be a positive integer")
    _validate_relative_output_dir(expected_output_dir)
    connections: dict[int, socket.socket] = {}
    ranks: set[int] = set()
    prepared_projection: Mapping[str, object] | None = None
    try:
        while len(connections) < world_size:
            connection, _ = server.accept()
            try:
                value = common_api_run.recv_frame(connection)
                if set(value) != {
                    "attempt_id",
                    "nonce",
                    "prepared",
                    "rank",
                    "world_size",
                }:
                    raise RuntimeError("readiness outer fields do not match schema")
                if (
                    value["attempt_id"] != attempt_id
                    or value["nonce"] != nonce
                ):
                    raise RuntimeError("readiness identity mismatch")
                rank = value["rank"]
                if (
                    type(rank) is not int
                    or not 0 <= rank < world_size
                    or rank in ranks
                ):
                    raise RuntimeError("invalid or duplicate readiness rank")
                if (
                    type(value["world_size"]) is not int
                    or value["world_size"] != world_size
                ):
                    raise RuntimeError("readiness world-size mismatch")
                prepared = _validate_prepared_message(
                    value["prepared"],
                    expected_output_dir=expected_output_dir,
                )
                if prepared_projection is None:
                    prepared_projection = prepared
                elif prepared != prepared_projection:
                    raise RuntimeError(
                        "DDP ranks disagree on non-topology prepared projection"
                    )
                ranks.add(rank)
                connections[rank] = connection
            except BaseException:
                connection.close()
                raise
        if ranks != set(range(world_size)):
            raise RuntimeError("readiness did not provide the complete rank set")
        return tuple(connections[rank] for rank in range(world_size))
    except BaseException:
        for connection in connections.values():
            connection.close()
        raise


def _validate_prepared_message(
    value: object,
    *,
    expected_output_dir: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"checks", "output_dir"}:
        raise RuntimeError("readiness prepared fields do not match schema")
    if value["output_dir"] != expected_output_dir:
        raise RuntimeError("readiness output_dir disagrees with the role config")
    checks = value["checks"]
    if not isinstance(checks, list):
        raise RuntimeError("readiness checks must be a list")
    normalized: list[dict[str, object]] = []
    for raw in checks:
        if not isinstance(raw, Mapping) or set(raw) != {
            "code",
            "level",
            "path",
            "volatile",
        }:
            raise RuntimeError("readiness check fields do not match schema")
        if raw["level"] != "warning":
            raise RuntimeError("readiness cannot contain a validation error")
        for field in ("code", "path"):
            if not isinstance(raw[field], str) or not raw[field]:
                raise RuntimeError(f"readiness check {field} must be non-empty")
        if type(raw["volatile"]) is not bool:
            raise RuntimeError("readiness check volatile must be bool")
        normalized.append(dict(raw))
    return {
        "checks": normalized,
        "output_dir": expected_output_dir,
    }


def _validate_relative_output_dir(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("expected output_dir must be repo-relative POSIX")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError("expected output_dir must be repo-relative POSIX")
    return value


def _ack_connections(
    connections: Sequence[socket.socket],
    *,
    terminate_before_eof: Callable[[], None],
) -> None:
    """Commit the barrier only after every ACK write succeeds.

    If any write fails, terminate the owned group before callers close sockets;
    already-written ACK bytes therefore cannot be followed by EOF while a child
    remains able to enter ``run()``.
    """

    try:
        for connection in connections:
            connection.sendall(common_api_run.ACK)
    except BaseException:
        terminate_before_eof()
        raise
    for connection in connections:
        connection.shutdown(socket.SHUT_WR)


def _interrupt_at_marker(
    identity: ProcessIdentity,
    output_dir: Path,
    target: int,
    timeout_s: float = 900.0,
    poll_interval_s: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not output_dir.exists() and not output_dir.is_symlink():
            time.sleep(poll_interval_s)
            continue
        status = vr.inspect_run(output_dir)
        if status.errors:
            _terminate_owned_process(identity)
            raise RuntimeError("interrupted run status is corrupt")
        if status.committed_steps > target:
            _terminate_owned_process(identity)
            raise RuntimeError("interrupted role overshot its target marker")
        if status.committed_steps == target:
            _signal_owned_process(identity, signal.SIGSTOP)
            stopped = vr.inspect_run(output_dir)
            if stopped.errors:
                _terminate_owned_process(identity)
                raise RuntimeError("stopped interrupted run status is corrupt")
            if stopped.committed_steps != target:
                _terminate_owned_process(identity)
                raise RuntimeError("marker changed after SIGSTOP")
            _terminate_owned_process(identity)
            return
        time.sleep(poll_interval_s)
    _terminate_owned_process(identity)
    raise TimeoutError("timed out waiting for interruption marker")


def _capture_owned_process(
    pid: int | None,
    *,
    timeout_s: float = 2.0,
) -> ProcessIdentity:
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("child has an invalid pid")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            pgid = os.getpgid(pid)
            create_time = _process_create_time(pid)
        except ProcessLookupError as exc:
            raise RuntimeError("child exited before identity capture") from exc
        if pgid == pid:
            return ProcessIdentity(pid=pid, create_time=create_time, pgid=pgid)
        if time.monotonic() >= deadline:
            raise RuntimeError("child did not become its own process-group leader")
        time.sleep(0.01)


def _capture_owned_group(
    identity: ProcessIdentity,
) -> tuple[ProcessIdentity, ...]:
    """Snapshot current members after proving the session leader identity."""

    if not isinstance(identity, ProcessIdentity):
        raise TypeError("identity must be ProcessIdentity")
    if (
        identity.pid <= 1
        or identity.pgid != identity.pid
        or os.getpgid(identity.pid) != identity.pgid
        or _process_create_time(identity.pid) != identity.create_time
    ):
        raise RuntimeError("cannot capture a reused or non-owned process group")
    completed = subprocess.run(
        ("ps", "-axo", "pid=,pgid=,lstart="),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot enumerate the owned process group")
    members: list[ProcessIdentity] = []
    for raw_line in completed.stdout.splitlines():
        parts = raw_line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        if pgid == identity.pgid:
            members.append(
                ProcessIdentity(
                    pid=pid,
                    create_time=parts[2].strip(),
                    pgid=pgid,
                )
            )
    members.sort(key=lambda member: member.pid)
    if identity not in members:
        raise RuntimeError("owned process-group snapshot lost its leader")
    return tuple(members)


def _process_create_time(pid: int) -> str:
    completed = subprocess.run(
        ("ps", "-o", "lstart=", "-p", str(pid)),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise ProcessLookupError(pid)
    return value


def _signal_owned_process(identity: ProcessIdentity, sig: int) -> None:
    if not isinstance(identity, ProcessIdentity):
        raise TypeError("identity must be ProcessIdentity")
    if (
        identity.pid <= 1
        or identity.pgid != identity.pid
        or os.getpgid(identity.pid) != identity.pgid
        or _process_create_time(identity.pid) != identity.create_time
    ):
        raise RuntimeError("refusing to signal a reused or non-owned process group")
    os.killpg(identity.pgid, sig)


def _terminate_owned_process(
    identity: ProcessIdentity,
    captured_members: Sequence[ProcessIdentity] = (),
) -> None:
    """Kill an owned group, including after its leader has already exited."""

    try:
        _signal_owned_process(identity, signal.SIGKILL)
    except ProcessLookupError:
        for member in captured_members:
            if (
                not isinstance(member, ProcessIdentity)
                or member.pgid != identity.pgid
                or member.pid == identity.pid
            ):
                continue
            try:
                matches = (
                    os.getpgid(member.pid) == identity.pgid
                    and _process_create_time(member.pid) == member.create_time
                )
            except ProcessLookupError:
                continue
            if matches:
                os.killpg(identity.pgid, signal.SIGKILL)
                return


def _mechanical_gate(
    observations: Sequence[RunObservation],
    specs: Sequence[RoleSpec],
) -> bool:
    if len(observations) != len(specs):
        return False
    return all(
        item.role == spec.name
        and item.exit_code == spec.expected_exit
        and item.committed_steps == spec.target_steps
        and (item.audit_ok or spec.phase == "interrupted")
        for item, spec in zip(observations, specs, strict=True)
    )


def _execution_candidate() -> dict[str, object]:
    return evidence_candidate(probe_git_identity(REPO_ROOT))


def _append_attempt_environment(
    role: str,
    *,
    attempt_id: str,
    candidate: Mapping[str, object],
) -> None:
    report = current_environment_report(
        repo_root=REPO_ROOT,
        attempt_id=attempt_id,
        role=role,
        tested=True,
    )
    if (
        report.commit != candidate.get("commit")
        or report.clean is not True
        or candidate.get("clean") is not True
        or candidate.get("tested") is not True
    ):
        raise RuntimeError("attempt environment disagrees with clean tested candidate")
    append_environment_report(
        EVIDENCE_ROOT / "environment.jsonl",
        report,
    )


def _invalidate_family_evidence(
    family: str,
    candidate: Mapping[str, object],
) -> None:
    document = _role_evidence_document(candidate)
    names = {spec.name for spec in roles_for_family(family)}
    document["roles"] = [
        role for role in document["roles"] if role.get("role") not in names
    ]
    document["families"].pop(family, None)
    _write_role_evidence(document)


def _record_family_evidence(
    family: str,
    observations: Sequence[RunObservation],
    *,
    semantic_pass: bool,
    candidate: Mapping[str, object],
) -> None:
    if semantic_pass is not True:
        raise RuntimeError("cannot record a failed semantic parity result")
    expected_names = {spec.name for spec in roles_for_family(family)}
    if {item.role for item in observations} != expected_names:
        raise RuntimeError(f"{family} observations do not match RoleSpec")
    document = _role_evidence_document(candidate)
    existing = {
        role["role"]: role
        for role in document["roles"]
        if role["role"] not in expected_names
    }
    existing.update(
        {
            item.role: {
                "audit_ok": item.audit_ok,
                "committed_steps": item.committed_steps,
                "exit_code": item.exit_code,
                "output_dir": item.output_dir,
                "role": item.role,
            }
            for item in observations
        }
    )
    document["roles"] = [
        existing[spec.name]
        for spec in ROLE_SPECS
        if spec.name in existing
    ]
    document["families"][family] = {"semantic_parity": True}
    _write_role_evidence(document)


def _role_evidence_document(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    path = EVIDENCE_ROOT / "role_results.json"
    if not path.exists():
        return {
            "candidate": dict(candidate),
            "families": {},
            "roles": [],
            "schema_version": 1,
        }
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("role evidence path must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"candidate", "families", "roles", "schema_version"}
        or value["schema_version"] != 1
        or value["candidate"] != dict(candidate)
        or not isinstance(value["families"], dict)
        or not isinstance(value["roles"], list)
    ):
        raise RuntimeError("existing role evidence has invalid schema/candidate")
    return value


def _write_role_evidence(document: Mapping[str, object]) -> None:
    atomic_write_bytes(
        EVIDENCE_ROOT / "role_results.json",
        common_api_run.canonical_json_bytes(document) + b"\n",
    )


def _semantic_gate(observations: Sequence[RunObservation]) -> bool:
    continuous = next(
        (item for item in observations if "_continuous" in item.role),
        None,
    )
    resumed = next(
        (item for item in observations if item.role.endswith("_resume")),
        None,
    )
    if continuous is None or resumed is None:
        return False
    return compare_semantic_projections(
        _read_semantic_projection(REPO_ROOT / continuous.output_dir),
        _read_semantic_projection(REPO_ROOT / resumed.output_dir),
    )


def _read_semantic_projection(output_dir: Path) -> dict[str, object]:
    """Read only public audited projections and the head marker digest promise."""

    status = vr.inspect_run(output_dir)
    audit = vr.audit_run(output_dir)
    if not status.ok or not audit.ok or status.committed_steps <= 0:
        raise RuntimeError("semantic comparison requires a completed audited run")
    manifest = json.loads(
        (output_dir / "sample_manifest.json").read_text(encoding="utf-8")
    )
    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    marker_path = (
        output_dir
        / "commits"
        / f"commit_{status.committed_steps:06d}.json"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint = marker.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("head marker is missing its checkpoint promise")
    digest = checkpoint.get("tree_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("head marker checkpoint digest is invalid")
    return {
        "audit_ok": True,
        "checkpoint_digest": digest,
        "core_metrics": metrics,
        "manifest": manifest,
    }


def _flow_native_gate(candidate: Mapping[str, object]) -> bool:
    """Run the fixed W04 oracle without importing it into experiment code."""

    import subprocess
    import sys

    script = REPO_ROOT / "tests/native_parity/run_flow_grpo_sd3.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False
    envelope = {
        "candidate": dict(candidate),
        "report": payload,
        "schema_version": 1,
    }
    atomic_write_bytes(
        EVIDENCE_ROOT / "flow_native.json",
        common_api_run.canonical_json_bytes(envelope) + b"\n",
    )
    return completed.returncode == 0 and _native_gate_report_passes(payload)


def _native_gate_report_passes(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "case",
        "config_path",
        "items",
        "overall_pass",
        "precision",
        "schema_version",
    }:
        return False
    if (
        value["schema_version"] != 1
        or value["overall_pass"] is not True
        or value["case"] != "flow_grpo_sd3_case_v1"
        or value["config_path"] != "configs/flow_grpo_sd3.yaml"
        or value["precision"] != "fp32"
    ):
        return False
    items = value["items"]
    if not isinstance(items, Mapping) or set(items) != FLOW_NATIVE_ITEMS:
        return False
    return all(
        isinstance(item, Mapping)
        and set(item) == {"comparisons", "passed"}
        and item["passed"] is True
        for item in items.values()
    )


def _assert_role_invariants(spec: RoleSpec, config: object) -> None:
    if config.runtime.deterministic is not True:
        raise ValueError(f"{spec.name} must enable deterministic runtime")
    if config.runtime.max_steps not in {20, 100}:
        raise ValueError(f"{spec.name} has unexpected final max_steps")
    if spec.family == "tiny_s100" and config.runtime.max_steps != 100:
        raise ValueError("Tiny-S100 final max_steps must be 100")
    if spec.family not in {"tiny_s100", "mg1_tiny_grpo"}:
        if config.runtime.batch_size != 1:
            raise ValueError(f"{spec.name} runtime.batch_size must be 1")
        if config.dataset.repeat_per_prompt != 1:
            raise ValueError(f"{spec.name} repeat_per_prompt must be 1")
        if config.dataset.sampling_strategy != "sequential":
            raise ValueError(f"{spec.name} sampling must be sequential")
        assert_prompt_balance(config)
    if spec.phase == "q100":
        expected_seed = int(spec.name.rsplit("seed", 1)[1])
        if expected_seed not in {17, 29, 43} or config.run.seed != expected_seed:
            raise ValueError(f"{spec.name} has an invalid Q100 seed")
        if config.runtime.max_steps != 100 or config.resume.from_ is not None:
            raise ValueError(f"{spec.name} is not a fresh 100-step run")
    elif config.run.seed != _family_seed(spec.family):
        raise ValueError(f"{spec.name} changed the C20/S100 seed")


def _assert_lineage(specs: Sequence[RoleSpec], resolved: Mapping[str, object]) -> None:
    interrupted = next(
        (spec for spec in specs if spec.phase == "interrupted"),
        None,
    )
    resume = next((spec for spec in specs if spec.phase == "resume"), None)
    if interrupted is None or resume is None:
        return
    interrupted_config = resolved[interrupted.name]
    resume_config = resolved[resume.name]
    if interrupted_config.resume.from_ is not None:
        raise ValueError("interrupted role must start fresh")
    if resume_config.resume.from_ is None:
        raise ValueError("resume role must name its authoritative checkpoint")
    if interrupted_config.artifacts.output_dir != resume_config.artifacts.output_dir:
        raise ValueError("interrupted and resume roles must share one output directory")
    if resume.lineage_head != interrupted.name:
        raise ValueError("resume lineage head must name the interrupted role")


def _family_seed(family: str) -> int:
    return {
        "tiny_s100": 42,
        "flow_grpo_sd3": 41,
        "tempflow_sd3": 23,
        "flash_wan": 29,
        "world_r1_wan": 31,
        "mg1_tiny_grpo": 42,
    }[family]


def _prompts(config: object) -> tuple[str, ...]:
    if config.dataset.prompts is not None:
        prompts = tuple(config.dataset.prompts)
    else:
        path = config.dataset.path
        prompts = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if len(prompts) != len(set(prompts)) or any(not prompt for prompt in prompts):
        raise ValueError("evidence prompts must be non-empty and unique")
    return prompts


def _group_size(config: object) -> int:
    params = config.rollout.params
    key = "branch_count" if config.rollout.name == "branching" else "samples_per_prompt"
    value = params[key]
    if type(value) is not int or value <= 0:
        raise ValueError("rollout group size must be a positive integer")
    return value


def _canonical_projection(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _canonical_projection(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_projection(item) for item in value)
    if isinstance(value, Path):
        return value.resolve(strict=False).as_posix()
    return value


def _diff_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(_diff_paths(left[key], right[key], path))
        return paths
    if left != right:
        return {prefix}
    return set()


def _normalize_run_id(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: ("<run-id>" if key == "run_id" else _normalize_run_id(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_run_id(item) for item in value]
    return value


def digest_public_projection(value: Mapping[str, object]) -> str:
    """Stable digest helper for controller tests and append-only evidence."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
