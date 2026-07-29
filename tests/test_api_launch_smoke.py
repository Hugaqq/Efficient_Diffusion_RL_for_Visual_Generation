"""Dedicated non-repository-cwd smoke for the sole public Python API."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import socket
import stat
import subprocess
import sys
from textwrap import dedent
from typing import Any

import pytest
import torch.distributed as dist
import yaml


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"
_RESULT_PREFIX = "VISUALRL_API_SMOKE="


def _write_config(
    root: Path,
    *,
    name: str,
    max_steps: int,
    resume: bool,
    distributed: bool = False,
) -> Path:
    payload = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    output_dir = (root / name).resolve()
    payload["runtime"]["max_steps"] = max_steps
    payload["runtime"]["distributed"]["mode"] = (
        "ddp" if distributed else "single"
    )
    payload["runtime"]["distributed"]["device"] = "cpu"
    payload["runtime"]["distributed"]["timeout_s"] = 30.0
    payload["runtime"]["distributed"]["max_snapshot_tensor_bytes"] = (
        1 << 20 if distributed else None
    )
    payload["artifacts"]["output_dir"] = str(output_dir)
    payload["artifacts"]["checkpoint_every"] = 1
    payload["resume"]["from"] = str(output_dir) if resume else None
    config_path = root / f"{name}-{max_steps}-{int(resume)}.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _source_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not existing
        else os.pathsep.join((str(ROOT), existing))
    )
    environment["OMP_NUM_THREADS"] = "1"
    return environment


def _public_run_script() -> str:
    return dedent(
        f"""
        import json
        from pathlib import Path
        import sys

        import visual_rl as vr

        config_path = Path(sys.argv[1]).resolve()
        experiment = vr.load(config_path)
        resolved = experiment.resolve()
        report = experiment.validate()
        if not report.ok:
            raise RuntimeError(
                "public validation failed: "
                + repr([(item.code, item.message) for item in report.errors])
            )
        result = experiment.run()
        status = vr.inspect_run(result.output_dir)
        audit = vr.audit_run(result.output_dir)
        payload = {{
            "cwd": str(Path.cwd()),
            "config_path": str(config_path),
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "committed_steps": result.committed_steps,
            "checkpoint": str(result.authoritative_checkpoint),
            "manifest": str(result.manifest_path),
            "marker": str(result.marker_path),
            "last_metrics": dict(result.last_metrics),
            "status_ok": status.ok,
            "status_steps": status.committed_steps,
            "audit_ok": audit.ok,
            "audit_commits": audit.checked_commit_count,
        }}
        print(
            {_RESULT_PREFIX!r}
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        """
    )


def _run_public_process(
    config_path: Path,
    *,
    outside_dir: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        (sys.executable, "-c", _public_run_script(), str(config_path)),
        cwd=outside_dir,
        env=_source_environment(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, (
        f"public API child failed\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    rows = tuple(
        line.removeprefix(_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    )
    assert len(rows) == 1, completed.stdout
    payload = json.loads(rows[0])
    assert isinstance(payload, dict)
    return payload


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        assert not stat.S_ISLNK(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(b"D\0" + relative + b"\0")
            continue
        assert stat.S_ISREG(metadata.st_mode)
        digest.update(b"F\0" + relative + b"\0")
        digest.update(entry.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "<run>"
    for record in payload["records"]:
        record["run_id"] = "<run>"
    return payload


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def test_public_api_from_outside_repo_continuous_and_fresh_resume(
    tmp_path: Path,
) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    split_one = _run_public_process(
        _write_config(
            tmp_path,
            name="split",
            max_steps=1,
            resume=False,
        ),
        outside_dir=outside_dir,
    )
    split_two = _run_public_process(
        _write_config(
            tmp_path,
            name="split",
            max_steps=2,
            resume=True,
        ),
        outside_dir=outside_dir,
    )
    continuous = _run_public_process(
        _write_config(
            tmp_path,
            name="continuous",
            max_steps=2,
            resume=False,
        ),
        outside_dir=outside_dir,
    )

    assert {
        split_one["cwd"],
        split_two["cwd"],
        continuous["cwd"],
    } == {str(outside_dir)}
    assert split_one["committed_steps"] == 1
    assert split_two["committed_steps"] == continuous["committed_steps"] == 2
    assert split_one["run_id"] == split_two["run_id"]
    assert split_two["run_id"] != continuous["run_id"]
    assert split_two["last_metrics"] == pytest.approx(
        continuous["last_metrics"],
        abs=1e-9,
    )
    assert all(
        item["status_ok"] and item["audit_ok"]
        for item in (split_one, split_two, continuous)
    )
    assert split_two["status_steps"] == continuous["status_steps"] == 2
    assert split_two["audit_commits"] == continuous["audit_commits"] == 2
    assert _tree_digest(Path(split_two["checkpoint"])) == _tree_digest(
        Path(continuous["checkpoint"])
    )
    assert _normalized_manifest(
        Path(split_two["manifest"])
    ) == _normalized_manifest(Path(continuous["manifest"]))
    assert _metric_rows(
        Path(split_two["output_dir"]) / "metrics.jsonl"
    ) == _metric_rows(
        Path(continuous["output_dir"]) / "metrics.jsonl"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _public_gloo_worker(
    rank: int,
    config_text: str,
    outside_text: str,
    observer_text: str,
    port: int,
    output: Any,
) -> None:
    outside_dir = Path(outside_text)
    observer_dir = Path(observer_text)
    os.chdir(outside_dir)
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "GROUP_RANK": "0",
            "GROUP_WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    try:
        import visual_rl as vr

        config_path = Path(config_text).resolve()
        experiment = vr.load(config_path)
        experiment.resolve()
        report = experiment.validate()
        if not report.ok:
            raise RuntimeError(
                "public DDP validation failed: "
                + repr([(item.code, item.message) for item in report.errors])
            )
        result = experiment.run()
        status = vr.inspect_run(result.output_dir)
        audit = vr.audit_run(result.output_dir)
        payload = {
            "rank": rank,
            "cwd": str(Path.cwd()),
            "run_id": result.run_id,
            "committed_steps": result.committed_steps,
            "output_dir": str(result.output_dir),
            "manifest": str(result.manifest_path),
            "status_ok": status.ok,
            "audit_ok": audit.ok,
        }
        target = observer_dir / f"rank-{rank}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        output.put({"rank": rank, "ok": True})
    except BaseException as exc:
        output.put(
            {
                "rank": rank,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


@pytest.mark.distributed
def test_two_rank_gloo_uses_public_api_from_outside_repo(
    tmp_path: Path,
) -> None:
    assert dist.is_available() and dist.is_gloo_available(), (
        "the dedicated API smoke requires CPU/Gloo"
    )
    outside_dir = tmp_path / "outside-gloo"
    observer_dir = tmp_path / "observer"
    outside_dir.mkdir()
    observer_dir.mkdir()
    config_path = _write_config(
        tmp_path,
        name="gloo",
        max_steps=1,
        resume=False,
        distributed=True,
    )
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = tuple(
        context.Process(
            target=_public_gloo_worker,
            args=(
                rank,
                str(config_path),
                str(outside_dir),
                str(observer_dir),
                port,
                output,
            ),
        )
        for rank in range(2)
    )
    started = []
    timed_out = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        for process in started:
            process.join(timeout=60)
            if process.is_alive():
                timed_out.append(process.name)
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=5)
    assert not timed_out, f"public Gloo API workers timed out: {timed_out}"
    assert all(process.exitcode == 0 for process in started), tuple(
        (process.name, process.exitcode) for process in started
    )
    statuses = []
    for _ in processes:
        try:
            statuses.append(output.get(timeout=5))
        except queue.Empty:
            pytest.fail("public Gloo API worker returned no status")
    statuses.sort(key=lambda item: item["rank"])
    assert all(item["ok"] for item in statuses), statuses
    rows = tuple(
        json.loads((observer_dir / f"rank-{rank}.json").read_text("utf-8"))
        for rank in range(2)
    )
    assert [row["rank"] for row in rows] == [0, 1]
    assert {row["cwd"] for row in rows} == {str(outside_dir)}
    assert len({row["run_id"] for row in rows}) == 1
    assert len({row["output_dir"] for row in rows}) == 1
    assert len({row["manifest"] for row in rows}) == 1
    assert all(row["committed_steps"] == 1 for row in rows)
    assert all(row["status_ok"] and row["audit_ok"] for row in rows)

    output_dir = Path(rows[0]["output_dir"])
    assert [path.name for path in (output_dir / "commits").glob("*.json")] == [
        "commit_000001.json"
    ]
    manifest = json.loads(Path(rows[0]["manifest"]).read_text("utf-8"))
    assert {record["rank"] for record in manifest["records"]} == {0, 1}
