"""Contracts for the fixed v0.7 experiment source preparation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import threading

import pytest

from experiments.v0_7 import common_api_run
from experiments.v0_7 import interrupt_resume
from experiments.v0_7 import mg1_nccl
from experiments.v0_7 import verify_evidence

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments/v0_7"
TEST_CANDIDATE = {"clean": True, "commit": "a" * 40, "tested": True}


def test_exact_role_table_and_thirty_full_yaml_configs_resolve() -> None:
    expected = {
        spec.config for spec in interrupt_resume.ROLE_SPECS
    }
    actual = {
        path.name for path in (EXPERIMENT_ROOT / "configs").glob("*.yaml")
    }
    assert len(interrupt_resume.ROLE_SPECS) == 30
    assert len({spec.name for spec in interrupt_resume.ROLE_SPECS}) == 30
    assert actual == expected
    for family in interrupt_resume.FAMILY_ORDER:
        resolved = interrupt_resume.assert_config_family(family)
        assert set(resolved) == {
            spec.name
            for spec in interrupt_resume.ROLE_SPECS
            if spec.family == family
        }


def test_experiment_configs_freeze_bounded_storage_policy() -> None:
    no_preview_families = {"tiny_s100", "mg1_tiny_grpo"}
    for family in interrupt_resume.FAMILY_ORDER:
        expected_preview_count = 0 if family in no_preview_families else 2
        for config in interrupt_resume.assert_config_family(family).values():
            assert config.artifacts.preview_samples_per_event == (
                expected_preview_count
            )
            assert config.artifacts.checkpoint_keep_last == 1
            assert not hasattr(config.runtime, "rollout_cache")


def test_four_real_families_freeze_batching_prompt_balance_and_q100_seeds() -> None:
    for family in (
        "flow_grpo_sd3",
        "tempflow_sd3",
        "flash_wan",
        "world_r1_wan",
    ):
        configs = interrupt_resume.assert_config_family(family)
        for name, config in configs.items():
            assert config.runtime.batch_size == 1
            assert config.runtime.deterministic is True
            assert config.dataset.repeat_per_prompt == 1
            assert config.dataset.sampling_strategy == "sequential"
            interrupt_resume.assert_prompt_balance(config)
            if "_q100_seed" in name:
                assert config.run.seed in {17, 29, 43}
                assert config.runtime.max_steps == 100
                assert config.resume.from_ is None


def test_prompt_window_formula_is_absolute_position_balanced() -> None:
    assert interrupt_resume.prompt_window_counts(
        36,
        group_size=8,
        start=0,
        stop=36,
    ) == (8,) * 36
    assert interrupt_resume.prompt_window_counts(
        2,
        group_size=2,
        start=64,
        stop=100,
    ) == (36, 36)


def test_only_common_api_run_calls_public_load_and_run() -> None:
    callsites: list[tuple[str, str]] = []
    for path in EXPERIMENT_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr in {"load", "run"}
                and isinstance(function.value, ast.Name)
                and function.value.id == "vr"
            ):
                callsites.append((path.name, function.attr))
    assert sorted(callsites) == [
        ("common_api_run.py", "load"),
        ("common_api_run.py", "load"),
    ]
    source = (EXPERIMENT_ROOT / "common_api_run.py").read_text(encoding="utf-8")
    assert "experiment.run()" in source


def test_framing_accepts_4096_and_rejects_4097_payload_bytes() -> None:
    exact = {"value": "x" * (4096 - len(b'{"value":""}'))}
    frame = common_api_run.encode_frame(exact)
    assert int.from_bytes(frame[:4], "big") == 4096
    too_large = {"value": exact["value"] + "x"}
    with pytest.raises(ValueError, match="1..4096"):
        common_api_run.encode_frame(too_large)


def test_accept_ready_validates_exact_schema_and_returns_complete_rank_order() -> None:
    class Connection:
        def __init__(self, value: dict[str, object]) -> None:
            self.payload = bytearray(common_api_run.encode_frame(value))
            self.closed = False

        def recv(self, size: int) -> bytes:
            payload = bytes(self.payload[:size])
            del self.payload[:size]
            return payload

        def close(self) -> None:
            self.closed = True

    prepared = {
        "checks": [
            {
                "code": "runtime.warning",
                "level": "warning",
                "path": "runtime",
                "volatile": False,
            }
        ],
        "output_dir": "experiments/v0_7/runs/mg1",
    }

    def envelope(rank: int) -> dict[str, object]:
        return {
            "attempt_id": "attempt",
            "nonce": "nonce",
            "prepared": prepared,
            "rank": rank,
            "world_size": 2,
        }

    rank1 = Connection(envelope(1))
    rank0 = Connection(envelope(0))

    class Server:
        connections = [rank1, rank0]

        def accept(self):
            return self.connections.pop(0), None

    accepted = interrupt_resume._accept_ready(
        Server(),
        attempt_id="attempt",
        nonce="nonce",
        world_size=2,
        expected_output_dir="experiments/v0_7/runs/mg1",
    )
    assert accepted == (rank0, rank1)
    assert not rank0.closed and not rank1.closed


def test_accept_ready_rejects_rank_projection_drift_without_ack() -> None:
    class Connection:
        def __init__(self, rank: int, output_dir: str) -> None:
            self.payload = bytearray(
                common_api_run.encode_frame(
                    {
                        "attempt_id": "attempt",
                        "nonce": "nonce",
                        "prepared": {
                            "checks": [],
                            "output_dir": output_dir,
                        },
                        "rank": rank,
                        "world_size": 2,
                    }
                )
            )
            self.closed = False

        def recv(self, size: int) -> bytes:
            payload = bytes(self.payload[:size])
            del self.payload[:size]
            return payload

        def close(self) -> None:
            self.closed = True

    rank0 = Connection(0, "runs/expected")
    rank1 = Connection(1, "runs/drifted")

    class Server:
        connections = [rank0, rank1]

        def accept(self):
            return self.connections.pop(0), None

    with pytest.raises(RuntimeError, match="output_dir"):
        interrupt_resume._accept_ready(
            Server(),
            attempt_id="attempt",
            nonce="nonce",
            world_size=2,
            expected_output_dir="runs/expected",
        )
    assert rank0.closed and rank1.closed


@pytest.mark.parametrize(
    "case",
    (
        "outer_field",
        "prepared_field",
        "check_field",
        "check_projection",
        "duplicate_rank",
    ),
)
def test_accept_ready_rejects_schema_rank_and_non_topology_drift(
    case: str,
) -> None:
    class Connection:
        def __init__(self, value: dict[str, object]) -> None:
            self.payload = bytearray(common_api_run.encode_frame(value))
            self.closed = False

        def recv(self, size: int) -> bytes:
            payload = bytes(self.payload[:size])
            del self.payload[:size]
            return payload

        def close(self) -> None:
            self.closed = True

    def envelope(rank: int) -> dict[str, object]:
        return {
            "attempt_id": "attempt",
            "nonce": "nonce",
            "prepared": {
                "checks": [
                    {
                        "code": "runtime.warning",
                        "level": "warning",
                        "path": "runtime",
                        "volatile": False,
                    }
                ],
                "output_dir": "runs/expected",
            },
            "rank": rank,
            "world_size": 2,
        }

    first_value = envelope(0)
    second_value = envelope(1)
    if case == "outer_field":
        second_value["unexpected"] = True
    elif case == "prepared_field":
        second_value["prepared"]["unexpected"] = True
    elif case == "check_field":
        second_value["prepared"]["checks"][0]["unexpected"] = True
    elif case == "check_projection":
        second_value["prepared"]["checks"][0]["path"] = "model"
    else:
        second_value["rank"] = 0
    first = Connection(first_value)
    second = Connection(second_value)

    class Server:
        connections = [first, second]

        def accept(self):
            return self.connections.pop(0), None

    with pytest.raises(RuntimeError):
        interrupt_resume._accept_ready(
            Server(),
            attempt_id="attempt",
            nonce="nonce",
            world_size=2,
            expected_output_dir="runs/expected",
        )
    assert first.closed and second.closed


@pytest.mark.parametrize(
    "output_dir",
    (
        "/absolute/run",
        "../escaped/run",
        "runs/../other",
        r"runs\windows",
    ),
)
def test_accept_ready_requires_a_normalized_repo_relative_output(
    output_dir: str,
) -> None:
    class Server:
        def accept(self):
            raise AssertionError("invalid expected output must fail before accept")

    with pytest.raises(RuntimeError, match="repo-relative POSIX"):
        interrupt_resume._accept_ready(
            Server(),
            attempt_id="attempt",
            nonce="nonce",
            world_size=1,
            expected_output_dir=output_dir,
        )


def test_run_config_crosses_ack_eof_before_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    barrier = Path("/tmp") / f"visualrl-v07-test-{os.getpid()}.sock"
    barrier.unlink(missing_ok=True)
    events: list[str] = []

    @dataclass(frozen=True)
    class Check:
        code: str = "runtime.ok"
        level: str = "warning"
        path: str = "runtime"
        volatile: bool = False
        message: str = "ok"

    @dataclass(frozen=True)
    class Report:
        ok: bool = True
        runtime_rank: int = 0
        runtime_world_size: int = 1
        checks: tuple[Check, ...] = (Check(),)
        errors: tuple[Check, ...] = ()

    @dataclass(frozen=True)
    class Result:
        run_id: str
        output_dir: Path
        committed_steps: int

    @dataclass(frozen=True)
    class Status:
        ok: bool = True
        run_id: str = "run-1"
        committed_steps: int = 1

    class Experiment:
        def resolve(self):
            return type(
                "Resolved",
                (),
                {"artifacts": type("Artifacts", (), {"output_dir": output_dir})()},
            )()

        def validate(self):
            events.append("validate")
            return Report()

        def run(self):
            events.append("run")
            return Result("run-1", output_dir, 1)

    monkeypatch.setattr(common_api_run.vr, "load", lambda _path: Experiment())
    monkeypatch.setattr(common_api_run.vr, "inspect_run", lambda _path: Status())
    monkeypatch.setattr(common_api_run.vr, "audit_run", lambda _path: Status())

    received: list[dict[str, object]] = []

    def server() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(barrier))
            listener.listen(1)
            ready.set()
            connection, _ = listener.accept()
            with connection:
                received.append(common_api_run.recv_frame(connection))
                events.append("ack")
                connection.sendall(common_api_run.ACK)
                connection.shutdown(socket.SHUT_WR)

    ready = threading.Event()
    thread = threading.Thread(target=server)
    thread.start()
    assert ready.wait(timeout=2.0)
    result = common_api_run.run_config(
        tmp_path / "config.yaml",
        barrier_path=barrier,
        attempt_id="attempt",
        nonce="nonce",
        repo_root=tmp_path,
    )
    thread.join(timeout=2.0)
    barrier.unlink(missing_ok=True)
    assert not thread.is_alive()
    assert events == ["validate", "ack", "run"]
    assert received[0]["rank"] == 0
    assert received[0]["world_size"] == 1
    assert result.audit_ok is True


def test_validation_failure_happens_before_barrier_and_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailureReport:
        ok = False
        runtime_rank = None
        runtime_world_size = None
        checks = ()
        errors = (type("Error", (), {"message": "missing CUDA"})(),)

    class Experiment:
        def resolve(self):
            return object()

        def validate(self):
            return FailureReport()

        def run(self):
            raise AssertionError("run must not be called")

    monkeypatch.setattr(common_api_run.vr, "load", lambda _path: Experiment())
    with pytest.raises(RuntimeError, match="missing CUDA"):
        common_api_run.run_config(
            tmp_path / "config.yaml",
            barrier_path=tmp_path / "missing.sock",
            attempt_id="attempt",
            nonce="nonce",
            repo_root=tmp_path,
        )


def test_cleanup_refuses_non_session_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(interrupt_resume.os, "getpgid", lambda _pid: 99)
    monkeypatch.setattr(
        interrupt_resume,
        "_process_create_time",
        lambda _pid: "captured",
    )
    monkeypatch.setattr(
        interrupt_resume.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    identity = interrupt_resume.ProcessIdentity(
        pid=100,
        create_time="captured",
        pgid=100,
    )
    with pytest.raises(RuntimeError, match="reused or non-owned"):
        interrupt_resume._terminate_owned_process(identity)
    assert killed == []


def test_cleanup_uses_captured_worker_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = interrupt_resume.ProcessIdentity(100, "leader-created", 100)
    worker = interrupt_resume.ProcessIdentity(101, "worker-created", 100)
    killed: list[tuple[int, int]] = []

    def getpgid(pid: int) -> int:
        if pid == leader.pid:
            raise ProcessLookupError(pid)
        assert pid == worker.pid
        return 100

    monkeypatch.setattr(interrupt_resume.os, "getpgid", getpgid)
    monkeypatch.setattr(
        interrupt_resume,
        "_process_create_time",
        lambda pid: "worker-created" if pid == worker.pid else "reused",
    )
    monkeypatch.setattr(
        interrupt_resume.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    interrupt_resume._terminate_owned_process(leader, (leader, worker))
    assert killed == [(100, interrupt_resume.signal.SIGKILL)]


def test_cleanup_does_not_kill_reused_captured_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = interrupt_resume.ProcessIdentity(100, "leader-created", 100)
    worker = interrupt_resume.ProcessIdentity(101, "worker-created", 100)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        interrupt_resume.os,
        "getpgid",
        lambda pid: (_ for _ in ()).throw(ProcessLookupError(pid))
        if pid == leader.pid
        else 100,
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_process_create_time",
        lambda _pid: "reused-worker",
    )
    monkeypatch.setattr(
        interrupt_resume.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    interrupt_resume._terminate_owned_process(leader, (worker,))
    assert killed == []


def test_partial_ack_terminates_before_any_socket_eof() -> None:
    events: list[str] = []

    class Connection:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def sendall(self, _payload: bytes) -> None:
            events.append(f"send:{self.name}")
            if self.fail:
                raise OSError("broken ACK")

        def shutdown(self, _direction: int) -> None:
            events.append(f"eof:{self.name}")

    with pytest.raises(OSError, match="broken ACK"):
        interrupt_resume._ack_connections(
            (Connection("rank0"), Connection("rank1", fail=True)),
            terminate_before_eof=lambda: events.append("terminate"),
        )
    assert events == ["send:rank0", "send:rank1", "terminate"]


@pytest.mark.parametrize(
    ("exits_before_failure", "expected_events"),
    (
        (False, ["start", "terminate", "join:5.0"]),
        (True, ["start", "join:5.0"]),
    ),
)
def test_single_parent_accept_failure_always_reaps_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exits_before_failure: bool,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    child: object

    class Child:
        pid = 123
        exitcode = None
        alive = False

        def start(self) -> None:
            self.alive = True
            events.append("start")

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            events.append(f"join:{timeout}")

        def kill(self) -> None:
            self.alive = False
            events.append("fallback-kill")

    child = Child()

    class Context:
        def Process(self, **_kwargs):
            return child

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def bind(self, _path: str) -> None:
            pass

        def listen(self, _world_size: int) -> None:
            pass

        def settimeout(self, _timeout: float) -> None:
            pass

    resolved = type(
        "Resolved",
        (),
        {"artifacts": type("Artifacts", (), {"output_dir": tmp_path / "run"})()},
    )()
    identity = interrupt_resume.ProcessIdentity(123, "created", 123)
    monkeypatch.setattr(
        interrupt_resume.common_api_run,
        "resolve_only",
        lambda _path: resolved,
    )
    monkeypatch.setattr(interrupt_resume, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        interrupt_resume,
        "_append_attempt_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        interrupt_resume.multiprocessing,
        "get_context",
        lambda _method: Context(),
    )
    monkeypatch.setattr(interrupt_resume.socket, "socket", lambda *_args: Server())
    monkeypatch.setattr(
        interrupt_resume,
        "_capture_owned_process",
        lambda _pid: identity,
    )
    def fail_accept(*_args, **_kwargs):
        if exits_before_failure:
            child.alive = False
        raise TimeoutError("accept timeout")

    monkeypatch.setattr(interrupt_resume, "_accept_ready", fail_accept)

    def terminate(received: interrupt_resume.ProcessIdentity) -> None:
        assert received == identity
        child.alive = False
        events.append("terminate")

    monkeypatch.setattr(interrupt_resume, "_terminate_owned_process", terminate)
    spec = interrupt_resume.ROLE_BY_NAME["tiny_s100_continuous"]
    with pytest.raises(TimeoutError, match="accept timeout"):
        interrupt_resume._run_role(spec, candidate=TEST_CANDIDATE)
    assert events == expected_events


def test_interrupt_marker_allows_output_directory_startup_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "late-run"
    identity = interrupt_resume.ProcessIdentity(789, "created", 789)
    events: list[str] = []

    class Status:
        committed_steps = 10
        errors = ()

    monkeypatch.setattr(
        interrupt_resume.time,
        "sleep",
        lambda _seconds: output_dir.mkdir(exist_ok=True),
    )
    monkeypatch.setattr(
        interrupt_resume.vr,
        "inspect_run",
        lambda _path: Status(),
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_signal_owned_process",
        lambda received, _sig: events.append(f"stop:{received.pid}"),
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_terminate_owned_process",
        lambda received: events.append(f"kill:{received.pid}"),
    )
    interrupt_resume._interrupt_at_marker(
        identity,
        output_dir,
        10,
        timeout_s=1.0,
        poll_interval_s=0.0,
    )
    assert events == ["stop:789", "kill:789"]


def test_interrupt_marker_polls_an_existing_run_before_first_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    identity = interrupt_resume.ProcessIdentity(790, "created", 790)
    events: list[str] = []
    statuses = iter(
        (
            type("Status", (), {"committed_steps": 0, "errors": ()})(),
            type("Status", (), {"committed_steps": 10, "errors": ()})(),
            type("Status", (), {"committed_steps": 10, "errors": ()})(),
        )
    )
    monkeypatch.setattr(interrupt_resume.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        interrupt_resume.vr,
        "inspect_run",
        lambda _path: next(statuses),
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_signal_owned_process",
        lambda received, _sig: events.append(f"stop:{received.pid}"),
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_terminate_owned_process",
        lambda received: events.append(f"kill:{received.pid}"),
    )
    interrupt_resume._interrupt_at_marker(
        identity,
        output_dir,
        10,
        timeout_s=1.0,
        poll_interval_s=0.0,
    )
    assert events == ["stop:790", "kill:790"]


def test_interrupt_marker_fails_closed_on_explicit_status_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    identity = interrupt_resume.ProcessIdentity(791, "created", 791)
    events: list[str] = []
    corrupt = type(
        "Status",
        (),
        {"committed_steps": 0, "errors": ("bad marker",)},
    )()
    monkeypatch.setattr(
        interrupt_resume.vr,
        "inspect_run",
        lambda _path: corrupt,
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_terminate_owned_process",
        lambda received: events.append(f"kill:{received.pid}"),
    )
    with pytest.raises(RuntimeError, match="status is corrupt"):
        interrupt_resume._interrupt_at_marker(
            identity,
            output_dir,
            10,
            timeout_s=1.0,
            poll_interval_s=0.0,
        )
    assert events == ["kill:791"]


def test_mg1_fixed_parent_timeout_finally_reaps_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    identity = interrupt_resume.ProcessIdentity(456, "created", 456)
    worker_one = interrupt_resume.ProcessIdentity(457, "worker-1", 456)
    worker_two = interrupt_resume.ProcessIdentity(458, "worker-2", 456)
    captured = (identity, worker_one, worker_two)

    class Process:
        pid = 456
        returncode = None
        waits = 0

        def wait(self, timeout: float) -> int:
            self.waits += 1
            events.append(f"wait:{timeout}")
            if self.waits == 1:
                raise mg1_nccl.subprocess.TimeoutExpired(("torchrun",), timeout)
            return -9

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9
            events.append("fallback-kill")

    process = Process()
    monkeypatch.setattr(
        mg1_nccl.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_capture_owned_process",
        lambda _pid: identity,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_capture_fixed_group_members",
        lambda *_args, **_kwargs: captured,
    )

    def terminate(
        received: interrupt_resume.ProcessIdentity,
        members: tuple[interrupt_resume.ProcessIdentity, ...],
    ) -> None:
        assert received == identity
        assert members == captured
        process.returncode = -9
        events.append("terminate")

    monkeypatch.setattr(mg1_nccl, "_terminate_owned_process", terminate)
    with pytest.raises(TimeoutError, match="fixed MG1 command timed out"):
        mg1_nccl._run_fixed_command(("torchrun", "nodeid"), timeout_s=1.0)
    assert events == ["wait:1.0", "terminate", "wait:5.0"]


def test_mg1_fixed_parent_checks_group_after_leader_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = interrupt_resume.ProcessIdentity(456, "leader", 456)
    worker_one = interrupt_resume.ProcessIdentity(457, "worker-1", 456)
    worker_two = interrupt_resume.ProcessIdentity(458, "worker-2", 456)
    captured = (leader, worker_one, worker_two)
    events: list[object] = []

    class Process:
        pid = 456

        def wait(self, timeout: float) -> int:
            events.append(("wait", timeout))
            return 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            raise AssertionError("numeric leader fallback must not be used")

    process = Process()
    monkeypatch.setattr(
        mg1_nccl.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_capture_owned_process",
        lambda _pid: leader,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_capture_fixed_group_members",
        lambda *_args, **_kwargs: captured,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_terminate_owned_process",
        lambda identity, members: events.append(
            ("terminate", identity, members)
        ),
    )
    assert mg1_nccl._run_fixed_command(("torchrun", "nodeid"), timeout_s=1.0) == 0
    assert events == [
        ("wait", 1.0),
        ("terminate", leader, captured),
        ("wait", 5.0),
    ]


def test_mg1_fixed_group_capture_fails_closed_when_workers_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = interrupt_resume.ProcessIdentity(456, "leader", 456)
    events: list[object] = []

    class Process:
        pid = 456

        def wait(self, timeout: float) -> int:
            events.append(("wait", timeout))
            return 1

        def poll(self) -> int:
            return 1

    monkeypatch.setattr(
        mg1_nccl.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_capture_owned_process",
        lambda _pid: leader,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_capture_fixed_group_members",
        lambda *_args, **_kwargs: (leader,),
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_terminate_owned_process",
        lambda identity, members: events.append(
            ("terminate", identity, members)
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="captured 1 of 3 members",
    ):
        mg1_nccl._run_fixed_command(
            ("torchrun", "nodeid"),
            timeout_s=1.0,
        )
    assert events == [
        ("terminate", leader, (leader,)),
        ("wait", 5.0),
    ]


def test_mg1_partial_control_write_is_removed_before_process_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.json"
    output_dir = tmp_path / "run"
    resolved = type(
        "Resolved",
        (),
        {"artifacts": type("Artifacts", (), {"output_dir": output_dir})()},
    )()
    monkeypatch.setattr(mg1_nccl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mg1_nccl.common_api_run,
        "resolve_only",
        lambda _config: resolved,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_append_attempt_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_control_path",
        lambda _role: control_path,
    )

    def partial_write(path: Path, payload: bytes) -> int:
        with path.open("wb") as stream:
            stream.write(payload[:4])
        raise OSError("partial control write")

    monkeypatch.setattr(Path, "write_bytes", partial_write)
    monkeypatch.setattr(
        mg1_nccl.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "torchrun must not launch after a partial control write"
        ),
    )
    with pytest.raises(OSError, match="partial control write"):
        mg1_nccl._run_role(
            "mg1_tiny_grpo_c20_continuous",
            candidate=TEST_CANDIDATE,
        )
    assert not control_path.exists()


@pytest.mark.parametrize("failed_gate", ("mechanical", "semantic", "native"))
def test_family_gate_failure_raises_instead_of_returning_partial_success(
    monkeypatch: pytest.MonkeyPatch,
    failed_gate: str,
) -> None:
    launched: list[str] = []
    monkeypatch.setattr(
        interrupt_resume,
        "_execution_candidate",
        lambda: TEST_CANDIDATE,
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_invalidate_family_evidence",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        interrupt_resume,
        "assert_config_family",
        lambda _family: {},
    )

    def run_role(spec, *, candidate):
        assert candidate == TEST_CANDIDATE
        launched.append(spec.name)
        return interrupt_resume.RunObservation(
            role=spec.name,
            exit_code=spec.expected_exit,
            committed_steps=spec.target_steps,
            audit_ok=True,
            output_dir=f"runs/{spec.name}",
        )

    monkeypatch.setattr(interrupt_resume, "_run_role", run_role)
    monkeypatch.setattr(
        interrupt_resume,
        "_mechanical_gate",
        lambda *_args: failed_gate != "mechanical",
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_semantic_gate",
        lambda *_args: failed_gate != "semantic",
    )
    monkeypatch.setattr(
        interrupt_resume,
        "_flow_native_gate",
        lambda _candidate: failed_gate != "native",
    )
    with pytest.raises(RuntimeError, match="gate failed|semantic parity failed"):
        interrupt_resume.run_family("flow_grpo_sd3")
    assert all("_q100_" not in role for role in launched)


def test_native_gate_records_report_and_rejects_a_failed_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = {
        name: {"comparisons": [], "passed": True}
        for name in interrupt_resume.FLOW_NATIVE_ITEMS
    }
    report = {
        "case": "flow_grpo_sd3_case_v1",
        "config_path": "configs/flow_grpo_sd3.yaml",
        "items": items,
        "overall_pass": True,
        "precision": "fp32",
        "schema_version": 1,
    }

    def completed(payload: dict[str, object]):
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(payload),
            },
        )()

    monkeypatch.setattr(interrupt_resume, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(
        interrupt_resume.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(report),
    )
    assert interrupt_resume._flow_native_gate(TEST_CANDIDATE)
    envelope = json.loads(
        (tmp_path / "flow_native.json").read_text(encoding="utf-8")
    )
    assert envelope == {
        "candidate": TEST_CANDIDATE,
        "report": report,
        "schema_version": 1,
    }

    report["items"]["gradient"]["passed"] = False
    assert not interrupt_resume._flow_native_gate(TEST_CANDIDATE)


def test_controller_atomically_records_all_roles_and_six_semantic_families(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(interrupt_resume, "EVIDENCE_ROOT", tmp_path)
    for family in interrupt_resume.FAMILY_ORDER:
        observations = tuple(
            interrupt_resume.RunObservation(
                role=spec.name,
                exit_code=spec.expected_exit,
                committed_steps=spec.target_steps,
                audit_ok=True,
                output_dir=f"runs/{spec.name}",
            )
            for spec in interrupt_resume.roles_for_family(family)
        )
        interrupt_resume._record_family_evidence(
            family,
            observations,
            semantic_pass=True,
            candidate=TEST_CANDIDATE,
        )
    value = json.loads(
        (tmp_path / "role_results.json").read_text(encoding="utf-8")
    )
    assert value["candidate"] == TEST_CANDIDATE
    assert value["families"] == {
        family: {"semantic_parity": True}
        for family in interrupt_resume.FAMILY_ORDER
    }
    assert [item["role"] for item in value["roles"]] == [
        spec.name for spec in interrupt_resume.ROLE_SPECS
    ]
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_failed_retry_cannot_fall_back_to_old_family_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(interrupt_resume, "EVIDENCE_ROOT", tmp_path)
    for family in interrupt_resume.FAMILY_ORDER:
        observations = tuple(
            interrupt_resume.RunObservation(
                role=spec.name,
                exit_code=spec.expected_exit,
                committed_steps=spec.target_steps,
                audit_ok=True,
                output_dir=f"runs/{spec.name}",
            )
            for spec in interrupt_resume.roles_for_family(family)
        )
        interrupt_resume._record_family_evidence(
            family,
            observations,
            semantic_pass=True,
            candidate=TEST_CANDIDATE,
        )

    interrupt_resume._invalidate_family_evidence(
        "flow_grpo_sd3",
        TEST_CANDIDATE,
    )
    stale_after_failed_retry = json.loads(
        (tmp_path / "role_results.json").read_text(encoding="utf-8")
    )
    assert "flow_grpo_sd3" not in stale_after_failed_retry["families"]
    assert not any(
        role["role"].startswith("flow_grpo_sd3_")
        for role in stale_after_failed_retry["roles"]
    )
    with pytest.raises(
        ValueError,
        match="role family semantic parity fields do not match schema",
    ):
        verify_evidence._verify_roles(stale_after_failed_retry)


def test_role_attempt_collects_and_appends_candidate_bound_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = type(
        "Report",
        (),
        {"clean": True, "commit": "a" * 40},
    )()
    calls: list[tuple[object, ...]] = []

    def collect(**kwargs):
        calls.append(("collect", kwargs))
        return report

    def append(path: Path, received: object) -> None:
        calls.append(("append", path, received))

    monkeypatch.setattr(interrupt_resume, "current_environment_report", collect)
    monkeypatch.setattr(interrupt_resume, "append_environment_report", append)
    interrupt_resume._append_attempt_environment(
        "tiny_s100_continuous",
        attempt_id="attempt-1",
        candidate=TEST_CANDIDATE,
    )
    assert calls == [
        (
            "collect",
            {
                "repo_root": interrupt_resume.REPO_ROOT,
                "attempt_id": "attempt-1",
                "role": "tiny_s100_continuous",
                "tested": True,
            },
        ),
        (
            "append",
            interrupt_resume.EVIDENCE_ROOT / "environment.jsonl",
            report,
        ),
    ]


def test_mg1_internal_failure_records_failure_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[list[dict[str, object]]] = []
    monkeypatch.setattr(mg1_nccl, "_execution_candidate", lambda: TEST_CANDIDATE)
    monkeypatch.setattr(
        mg1_nccl,
        "_invalidate_family_evidence",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(mg1_nccl, "assert_config_family", lambda _family: {})
    monkeypatch.setattr(
        mg1_nccl,
        "_run_fixed_command",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_write_internal_evidence",
        lambda _candidate, results: writes.append(
            [dict(item) for item in results]
        ),
    )
    monkeypatch.setattr(
        mg1_nccl,
        "_run_role",
        lambda *_args, **_kwargs: pytest.fail(
            "MG1 roles must not launch after an internal gate failure"
        ),
    )
    with pytest.raises(RuntimeError, match="internal NCCL gate failed"):
        mg1_nccl.run_mg1_suite()
    assert writes == [
        [
            {
                "nodeid": mg1_nccl.INTERNAL_NCCL_COMMANDS[0][-1],
                "passed": False,
            }
        ]
    ]


def test_mg1_module_has_a_no_argument_main_entry() -> None:
    source = (EXPERIMENT_ROOT / "mg1_nccl.py").read_text(encoding="utf-8")
    assert (
        'if __name__ == "__main__":\n'
        "    run_mg1_suite()\n"
    ) in source


def test_semantic_comparison_normalizes_only_run_id() -> None:
    left = {
        "audit_ok": True,
        "checkpoint_digest": "abc",
        "core_metrics": [{"step": 0, "loss": 1.0}],
        "manifest": {"run_id": "continuous", "records": [{"run_id": "continuous"}]},
    }
    right = json.loads(json.dumps(left))
    right["manifest"]["run_id"] = "resume"
    right["manifest"]["records"][0]["run_id"] = "resume"
    assert interrupt_resume.compare_semantic_projections(left, right)
    right["core_metrics"][0]["loss"] = 2.0
    assert not interrupt_resume.compare_semantic_projections(left, right)
