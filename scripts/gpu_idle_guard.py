#!/usr/bin/env python3
"""Guard a single-GPU training command on a shared server.

The guard starts a command only when the selected physical GPU is idle. While
the command is running it watches for foreign compute processes on that same
GPU. If one appears, the guard first SIGSTOPs its own process group. If the
foreign process persists, the guard SIGTERMs its own group to release VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int
    compute_pids: list[int]
    foreign_pids: list[int]


def run_text(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def gpu_uuid_by_index() -> dict[int, str]:
    output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    mapping: dict[int, str] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            mapping[int(parts[0])] = parts[1]
    return mapping


def gpu_snapshot(index: int, own_pgid: int | None) -> GpuSnapshot:
    gpu_output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    target: GpuSnapshot | None = None
    for line in gpu_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6 or int(parts[0]) != index:
            continue
        target = GpuSnapshot(
            index=index,
            uuid=parts[1],
            name=parts[2],
            memory_used_mb=int(parts[3]),
            memory_total_mb=int(parts[4]),
            utilization_pct=int(parts[5]),
            compute_pids=[],
            foreign_pids=[],
        )
        break
    if target is None:
        raise RuntimeError(f"GPU index {index} was not found by nvidia-smi")

    app_output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in app_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] != target.uuid:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        target.compute_pids.append(pid)
        if own_pgid is None or process_group(pid) != own_pgid:
            target.foreign_pids.append(pid)
    return target


def process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None
    except PermissionError:
        return None


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp_path.replace(path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def is_idle(snapshot: GpuSnapshot, max_memory_mb: int, max_util_pct: int) -> bool:
    return (
        snapshot.memory_used_mb <= max_memory_mb
        and snapshot.utilization_pct <= max_util_pct
        and not snapshot.compute_pids
    )


def signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index to guard.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--idle-memory-mb", type=int, default=1024)
    parser.add_argument("--idle-util-pct", type=int, default=10)
    parser.add_argument("--foreign-stop-grace-seconds", type=float, default=15.0)
    parser.add_argument("--foreign-terminate-seconds", type=float, default=60.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=120.0)
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("A guarded command is required after --")

    status_file = Path(args.status_file)
    event_log = Path(args.event_log)
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    restarts = 0
    process: subprocess.Popen | None = None
    pgid: int | None = None
    stopped_since: float | None = None
    stop_requested_at: float | None = None

    append_event(event_log, {"time": time.time(), "event": "guard_started", "gpu": args.gpu, "command": args.command})

    while True:
        if process is not None and process.poll() is not None:
            code = process.returncode
            append_event(event_log, {"time": time.time(), "event": "process_exited", "returncode": code})
            if code == 0:
                write_status(
                    status_file,
                    {"state": "completed", "returncode": code, "gpu": args.gpu, "time": time.time()},
                )
                return 0
            restarts += 1
            if restarts > args.max_restarts:
                write_status(
                    status_file,
                    {"state": "failed", "returncode": code, "restarts": restarts, "gpu": args.gpu, "time": time.time()},
                )
                return code or 1
            process = None
            pgid = None
            stopped_since = None
            stop_requested_at = None
            time.sleep(args.restart_delay_seconds)

        snapshot = gpu_snapshot(args.gpu, pgid)
        status_payload = {
            "state": "waiting" if process is None else ("paused" if stopped_since else "running"),
            "gpu": asdict(snapshot),
            "pid": process.pid if process is not None else None,
            "pgid": pgid,
            "restarts": restarts,
            "time": time.time(),
        }
        write_status(status_file, status_payload)

        if process is None:
            if is_idle(snapshot, args.idle_memory_mb, args.idle_util_pct):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                stdout_handle = stdout_log.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    args.command,
                    cwd=args.cwd,
                    env=env,
                    stdout=stdout_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                pgid = os.getpgid(process.pid)
                stopped_since = None
                stop_requested_at = None
                append_event(
                    event_log,
                    {"time": time.time(), "event": "process_started", "pid": process.pid, "pgid": pgid, "gpu": args.gpu},
                )
            time.sleep(args.poll_seconds)
            continue

        if snapshot.foreign_pids:
            now = time.time()
            if stopped_since is None and pgid is not None:
                signal_group(pgid, signal.SIGSTOP)
                stopped_since = now
                stop_requested_at = now
                append_event(
                    event_log,
                    {
                        "time": now,
                        "event": "process_stopped_for_foreign_gpu_user",
                        "foreign_pids": snapshot.foreign_pids,
                    },
                )
                time.sleep(args.foreign_stop_grace_seconds)
                continue
            if stop_requested_at and now - stop_requested_at >= args.foreign_terminate_seconds and pgid is not None:
                signal_group(pgid, signal.SIGTERM)
                append_event(
                    event_log,
                    {
                        "time": now,
                        "event": "process_terminated_to_release_gpu",
                        "foreign_pids": snapshot.foreign_pids,
                    },
                )
                time.sleep(20)
                if process.poll() is None:
                    signal_group(pgid, signal.SIGKILL)
                    append_event(event_log, {"time": time.time(), "event": "process_killed_after_term_timeout"})
            time.sleep(args.poll_seconds)
            continue

        if stopped_since is not None and pgid is not None:
            resumed_snapshot = gpu_snapshot(args.gpu, pgid)
            if not resumed_snapshot.foreign_pids:
                signal_group(pgid, signal.SIGCONT)
                append_event(event_log, {"time": time.time(), "event": "process_resumed"})
                stopped_since = None
                stop_requested_at = None

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
