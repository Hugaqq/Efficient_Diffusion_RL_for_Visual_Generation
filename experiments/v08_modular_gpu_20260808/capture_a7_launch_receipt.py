"""Capture one live frozen A7 trainer launch from Linux procfs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path("/dev/shm/v-qiaoqifan/visualrl-v08-candidate-56507f6e-source")
EVIDENCE_ROOT = Path("/dev/shm/v-qiaoqifan/visualrl-v08-a7-final-56507f-6f1533ef")
RELEASE_ROOT = Path(
    "/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/"
    "release_candidates/code-56507f6e-wheel-6f1533ef"
)
TRAINER_PYTHON = "/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python"
ROUTE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class LaunchReceiptError(RuntimeError):
    """The live process does not prove the requested frozen launch."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchReceiptError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchReceiptError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise LaunchReceiptError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _proc_bytes(pid: int, name: str) -> bytes:
    try:
        return (Path("/proc") / str(pid) / name).read_bytes()
    except OSError as exc:
        raise LaunchReceiptError(
            f"cannot read live trainer /proc/{pid}/{name}: {exc}"
        ) from exc


def _nul_fields(payload: bytes) -> list[str]:
    try:
        return [item.decode("utf-8") for item in payload.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        raise LaunchReceiptError("live trainer procfs fields are not UTF-8") from exc


def _parent_pid(pid: int) -> int:
    try:
        lines = (
            (Path("/proc") / str(pid) / "status")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except (OSError, UnicodeError) as exc:
        raise LaunchReceiptError(f"cannot read live trainer parent PID: {exc}") from exc
    for line in lines:
        if line.startswith("PPid:"):
            try:
                parent = int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise LaunchReceiptError("trainer PPid field is malformed") from exc
            if parent > 0:
                return parent
    raise LaunchReceiptError("trainer has no positive parent PID")


def _environment(pid: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in _nul_fields(_proc_bytes(pid, "environ")):
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        values[name] = value
    return values


def _atomic_create(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise LaunchReceiptError(
            f"refusing to overwrite launch receipt: {path}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capture(*, route: str, gpu_index: int, config_path: Path) -> dict[str, Any]:
    if ROUTE.fullmatch(route) is None:
        raise LaunchReceiptError("route must use lowercase dash-separated tokens")
    if type(gpu_index) is not int or gpu_index < 0:
        raise LaunchReceiptError("GPU index must be a non-negative integer")
    config_path = config_path.resolve(strict=True)
    freeze_path = RELEASE_ROOT / "a7-freeze-identity.json"
    freeze = _json(freeze_path)
    config_sha = _sha256(config_path)
    expected_config_sha = freeze.get("configs", {}).get(config_path.name)
    if config_sha != expected_config_sha:
        raise LaunchReceiptError("config does not match the A7 freeze record")

    pid_path = EVIDENCE_ROOT / "logs" / f"{route}.pid"
    if not pid_path.is_file() or pid_path.is_symlink():
        raise LaunchReceiptError(f"missing canonical trainer PID file: {pid_path}")
    try:
        trainer_pid = int(pid_path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise LaunchReceiptError("trainer PID file is malformed") from exc
    if trainer_pid <= 0:
        raise LaunchReceiptError("trainer PID must be positive")

    expected_command = [TRAINER_PYTHON, "-m", "visual_rl.train", str(config_path)]
    trainer_command = _nul_fields(_proc_bytes(trainer_pid, "cmdline"))
    if trainer_command != expected_command:
        raise LaunchReceiptError("live trainer command differs from the frozen launch")
    try:
        working_directory = os.readlink(f"/proc/{trainer_pid}/cwd")
    except OSError as exc:
        raise LaunchReceiptError(f"cannot read live trainer CWD: {exc}") from exc
    if working_directory != str(SOURCE_ROOT):
        raise LaunchReceiptError("live trainer CWD differs from the frozen source root")
    environment = _environment(trainer_pid)
    if environment.get("PYTHONPATH") != ".":
        raise LaunchReceiptError(
            "live trainer PYTHONPATH is not the frozen source root"
        )
    if environment.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise LaunchReceiptError(
            "live trainer GPU binding differs from the requested GPU"
        )

    supervisor_pid = _parent_pid(trainer_pid)
    supervisor_command = _nul_fields(_proc_bytes(supervisor_pid, "cmdline"))
    if not (
        supervisor_command
        and supervisor_command[0].endswith("bash")
        and str(gpu_index) in supervisor_command
        and route in supervisor_command
        and str(config_path) in supervisor_command
    ):
        raise LaunchReceiptError("trainer parent is not the expected route launcher")

    return {
        "schema_version": 1,
        "kind": "visual_rl_a7_launch_receipt",
        "route": route,
        "physical_gpu_index": gpu_index,
        "supervisor_pid": supervisor_pid,
        "trainer_pid": trainer_pid,
        "trainer_command": trainer_command,
        "working_directory": working_directory,
        "config": {"path": str(config_path), "sha256": config_sha},
        "output_dir": str(EVIDENCE_ROOT / "runs" / route),
        "stdout_log": str(EVIDENCE_ROOT / "logs" / f"{route}.log"),
        "memory_csv": str(EVIDENCE_ROOT / "logs" / f"{route}.gpu-memory.csv"),
        "code_content_sha256": freeze["code"]["content_sha256"],
        "wheel_sha256": freeze["wheel"]["sha256"],
        "freeze_record_sha256": _sha256(freeze_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route")
    parser.add_argument("gpu_index", type=int)
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()
    receipt = capture(
        route=args.route,
        gpu_index=args.gpu_index,
        config_path=args.config_path,
    )
    target = EVIDENCE_ROOT / "launch-receipts" / f"{args.route}.json"
    _atomic_create(target, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
