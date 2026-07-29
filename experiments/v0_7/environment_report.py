"""Pure environment projection helpers for v0.7 experiment evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


@dataclass(frozen=True)
class EnvironmentReport:
    """Bounded environment identity supplied by an outer authorized runner."""

    python: str
    platform: str
    attempt_id: str
    role: str
    commit: str
    clean: bool
    tested: bool
    cuda: str | None
    devices: tuple[str, ...]
    packages: tuple[tuple[str, str], ...]


def collect_environment(
    *,
    attempt_id: str,
    role: str,
    commit: str,
    clean: bool,
    tested: bool,
    cuda: str | None,
    devices: tuple[str, ...],
    packages: Mapping[str, str],
) -> EnvironmentReport:
    """Collect only deterministic process facts plus explicitly supplied probes."""

    for name, value in (
        ("attempt_id", attempt_id),
        ("role", role),
        ("commit", commit),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if type(clean) is not bool or type(tested) is not bool:
        raise TypeError("clean/tested must be bool")
    if cuda is not None and (not isinstance(cuda, str) or not cuda):
        raise ValueError("cuda must be None or a non-empty string")
    if any(not isinstance(item, str) or not item for item in devices):
        raise ValueError("devices must contain non-empty strings")
    normalized = tuple(sorted((str(name), str(version)) for name, version in packages.items()))
    return EnvironmentReport(
        python=platform.python_version(),
        platform=platform.platform(),
        attempt_id=attempt_id,
        role=role,
        commit=commit,
        clean=clean,
        tested=tested,
        cuda=cuda,
        devices=devices,
        packages=normalized,
    )


def canonical_report_bytes(report: EnvironmentReport) -> bytes:
    """Return a stable JSON line suitable for append-only evidence."""

    payload: dict[str, Any] = {
        "clean": report.clean,
        "commit": report.commit,
        "cuda": report.cuda,
        "devices": list(report.devices),
        "packages": dict(report.packages),
        "platform": report.platform,
        "python": report.python,
        "attempt_id": report.attempt_id,
        "role": report.role,
        "tested": report.tested,
    }
    _reject_nonfinite(payload)
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def probe_git_identity(repo_root: Path) -> dict[str, object]:
    """Read current HEAD and tracked/untracked cleanliness without mutation."""

    root = Path(repo_root).resolve(strict=True)
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if re_full_sha(head) is False:
        raise RuntimeError("git rev-parse did not return a full lowercase SHA")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    return {"commit": head, "clean": status == ""}


def current_environment_report(
    *,
    repo_root: Path,
    attempt_id: str,
    role: str,
    tested: bool,
) -> EnvironmentReport:
    identity = probe_git_identity(repo_root)
    packages: dict[str, str] = {}
    for name in ("visual-rl", "numpy", "pyyaml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return collect_environment(
        attempt_id=attempt_id,
        role=role,
        commit=str(identity["commit"]),
        clean=bool(identity["clean"]),
        tested=tested,
        cuda=None,
        devices=(),
        packages=packages,
    )


def append_environment_report(path: Path, report: EnvironmentReport) -> None:
    """Append one canonical attempt line with one O_APPEND write."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_report_bytes(report)
    descriptor = os.open(
        target,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short environment evidence append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace one generated evidence file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def evidence_candidate(identity: Mapping[str, object]) -> dict[str, object]:
    if set(identity) != {"clean", "commit"}:
        raise ValueError("git identity must contain exact clean/commit fields")
    commit = identity["commit"]
    clean = identity["clean"]
    if not isinstance(commit, str) or not re_full_sha(commit):
        raise ValueError("git identity commit must be a full lowercase SHA")
    if clean is not True:
        raise RuntimeError("real evidence requires a clean working tree")
    return {"clean": True, "commit": commit, "tested": True}


def re_full_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git probe failed: {' '.join(arguments)}")
    return completed.stdout.rstrip("\n")


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("environment report cannot contain non-finite values")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)
