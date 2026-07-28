"""Two-rank CPU/Gloo integration for the one shared Runner lifecycle."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import socket
import tempfile
import time
from typing import Any

import pytest
import torch
import torch.distributed as dist
import yaml


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _config(
    tmp_path: Path,
    *,
    name: str,
    max_steps: int,
    resume: bool,
) -> Path:
    payload = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    output_dir = (tmp_path / name).resolve()
    payload["runtime"]["max_steps"] = max_steps
    payload["runtime"]["distributed"]["mode"] = "ddp"
    payload["runtime"]["distributed"]["device"] = "cpu"
    payload["runtime"]["distributed"]["timeout_s"] = 30.0
    payload["runtime"]["distributed"]["max_snapshot_tensor_bytes"] = 1 << 20
    payload["artifacts"]["output_dir"] = str(output_dir)
    payload["artifacts"]["checkpoint_every"] = 1
    payload["resume"]["from"] = str(output_dir) if resume else None
    path = tmp_path / f"{name}-{max_steps}-{int(resume)}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _worker(
    rank: int,
    config_path: str,
    port: int,
    output: Any,
    failure_rank: int | None,
) -> None:
    try:
        import visual_rl as vr
        from visual_rl.core.types import FrozenMapping, ValidatedRuntimeEnv
        from visual_rl.runner import ExperimentRunner

        config = vr.load(config_path).resolve()
        if rank == failure_rank:
            import visual_rl.runtime_factory as runtime_factory

            original_build = runtime_factory.build_runtime_components

            def build_with_reward_failure(config, context):
                components = original_build(config, context)

                def fail_reward(_batch, _step_context):
                    raise RuntimeError("injected rank-one reward failure")

                components.reward_executor.score = fail_reward
                return components

            runtime_factory.build_runtime_components = build_with_reward_failure
        environment = ValidatedRuntimeEnv(
            mode="ddp",
            rank=rank,
            local_rank=rank,
            world_size=2,
            local_world_size=2,
            group_rank=0,
            group_world_size=1,
            master_addr="127.0.0.1",
            master_port=port,
            visible_gpu_count=0,
            raw_launch_env=FrozenMapping({}),
        )
        runner = ExperimentRunner(config, environment)
        result = runner.run()
        output.put(
            {
                "rank": rank,
                "ok": True,
                "run_id": result.run_id,
                "committed_steps": result.committed_steps,
                "checkpoint": result.authoritative_checkpoint.name,
                "last_metrics": dict(result.last_metrics),
                "owned_artifact_manager": runner.artifact_manager is not None,
            }
        )
    except BaseException as exc:
        output.put(
            {
                "rank": rank,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _exception_chain(exc: BaseException) -> tuple[str, ...]:
    values = []
    current: BaseException | None = exc
    while current is not None:
        values.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return tuple(values)


def _public_post_marker_worker(
    rank: int,
    config_path: str,
    port: int,
    output: Any,
    method_name: str,
    phase: str,
) -> None:
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
        from visual_rl.artifacts.manager import ArtifactManager
        from visual_rl.errors import ArtifactError

        original = getattr(ArtifactManager, method_name)

        def fail_after_marker(
            manager: ArtifactManager,
            *args,
            **kwargs,
        ) -> None:
            if method_name == "rebuild_projections" and manager.head is None:
                original(manager, *args, **kwargs)
                return
            raise ArtifactError(
                f"injected distributed post-marker {phase} failure"
            )

        setattr(ArtifactManager, method_name, fail_after_marker)
        try:
            vr.load(config_path).run()
        finally:
            setattr(ArtifactManager, method_name, original)
        output.put(
            {
                "rank": rank,
                "ok": False,
                "error": "public DDP run unexpectedly succeeded",
            }
        )
    except BaseException as exc:
        output.put(
            {
                "rank": rank,
                "ok": True,
                "error_type": type(exc).__name__,
                "chain": _exception_chain(exc),
            }
        )


def _run_two_ranks(
    config: Path,
    port: int,
    *,
    failure_rank: int | None = None,
    expect_failure: bool = False,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, str(config), port, output, failure_rank),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("two-rank Runner worker timed out")
        assert process.exitcode == 0

    rows = []
    for _ in range(2):
        try:
            rows.append(output.get(timeout=5))
        except queue.Empty:
            pytest.fail("two-rank Runner worker returned no result")
    rows.sort(key=lambda row: row["rank"])
    assert [row["rank"] for row in rows] == [0, 1]
    assert all(row["ok"] is not expect_failure for row in rows), rows
    return rows


def _run_public_post_marker_failure(
    config: Path,
    port: int,
    *,
    method_name: str,
    phase: str,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_public_post_marker_worker,
            args=(
                rank,
                str(config),
                port,
                output,
                method_name,
                phase,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("public post-marker DDP worker timed out")
        assert process.exitcode == 0

    rows = []
    for _ in processes:
        try:
            rows.append(output.get(timeout=5))
        except queue.Empty:
            pytest.fail("public post-marker DDP worker returned no result")
    rows.sort(key=lambda row: row["rank"])
    assert [row["rank"] for row in rows] == [0, 1]
    return rows


def _normalized_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "<run>"
    for record in payload["records"]:
        record["run_id"] = "<run>"
    return payload


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _nccl_torchrun_rank() -> int:
    required = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    if any(name not in os.environ for name in required):
        pytest.skip("requires a complete two-rank torchrun environment")
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("requires a PyTorch build with NCCL")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except ValueError as exc:
        pytest.skip(f"requires integer torchrun topology values: {exc}")
    if (
        world_size != 2
        or local_world_size != 2
        or rank not in (0, 1)
        or local_rank not in (0, 1)
        or local_rank >= torch.cuda.device_count()
    ):
        pytest.skip(
            "requires one single-node two-rank torchrun process per visible GPU"
        )
    return rank


def _wait_for_path(path: Path, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.05)


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo is unavailable",
)
def test_two_rank_gloo_shares_lifecycle_rank_zero_writes_and_resume(
    tmp_path: Path,
) -> None:
    first = _run_two_ranks(
        _config(
            tmp_path,
            name="split",
            max_steps=1,
            resume=False,
        ),
        _free_port(),
    )
    resumed = _run_two_ranks(
        _config(
            tmp_path,
            name="split",
            max_steps=2,
            resume=True,
        ),
        _free_port(),
    )
    continuous = _run_two_ranks(
        _config(
            tmp_path,
            name="continuous",
            max_steps=2,
            resume=False,
        ),
        _free_port(),
    )

    assert {row["run_id"] for row in first} == {
        row["run_id"] for row in resumed
    }
    assert [row["owned_artifact_manager"] for row in first] == [True, False]
    assert [row["owned_artifact_manager"] for row in resumed] == [True, False]
    assert all(row["committed_steps"] == 1 for row in first)
    assert all(row["committed_steps"] == 2 for row in resumed)
    assert all(row["checkpoint"] == "checkpoint_000002" for row in resumed)
    assert first[0]["last_metrics"] == first[1]["last_metrics"]
    assert resumed[0]["last_metrics"] == resumed[1]["last_metrics"]
    assert continuous[0]["last_metrics"] == continuous[1]["last_metrics"]
    assert resumed[0]["last_metrics"] == continuous[0]["last_metrics"]
    assert resumed[0]["last_metrics"]["sample_count"] == 8

    split_dir = tmp_path / "split"
    continuous_dir = tmp_path / "continuous"
    commits = sorted((split_dir / "commits").glob("commit_*.json"))
    assert [path.name for path in commits] == [
        "commit_000001.json",
        "commit_000002.json",
    ]
    assert (split_dir / "checkpoint_000002").is_dir()
    manifest = json.loads(
        (split_dir / "sample_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["records"]) == 16
    assert {record["rank"] for record in manifest["records"]} == {0, 1}
    assert len({record["sample_id"] for record in manifest["records"]}) == 16
    from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256

    assert checkpoint_tree_sha256(
        split_dir / "checkpoint_000002"
    ) == checkpoint_tree_sha256(
        continuous_dir / "checkpoint_000002"
    )
    assert _normalized_manifest(
        split_dir / "sample_manifest.json"
    ) == _normalized_manifest(
        continuous_dir / "sample_manifest.json"
    )
    assert _metric_rows(split_dir / "metrics.jsonl") == _metric_rows(
        continuous_dir / "metrics.jsonl"
    )

    import visual_rl as vr

    status = vr.inspect_run(split_dir)
    audit = vr.audit_run(split_dir)
    assert status.ok and status.committed_steps == 2
    assert audit.ok and audit.checked_commit_count == 2


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo is unavailable",
)
def test_rank_one_pre_update_failure_aborts_both_ranks_without_commit(
    tmp_path: Path,
) -> None:
    rows = _run_two_ranks(
        _config(
            tmp_path,
            name="failure",
            max_steps=1,
            resume=False,
        ),
        _free_port(),
        failure_rank=1,
        expect_failure=True,
    )

    assert all("injected rank-one reward failure" in row["error"] for row in rows)
    output_dir = tmp_path / "failure"
    assert not list((output_dir / "commits").glob("commit_*.json"))
    assert not list(output_dir.glob("checkpoint_*"))
    assert list((output_dir / ".staging").iterdir()) == []


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo is unavailable",
)
@pytest.mark.parametrize(
    ("method_name", "phase"),
    (
        ("rebuild_projections", "projection"),
        ("cleanup_published_staging", "cleanup"),
        ("apply_checkpoint_retention", "retention"),
    ),
)
def test_two_rank_public_post_marker_failure_is_run_error_with_head_preserved(
    tmp_path: Path,
    method_name: str,
    phase: str,
) -> None:
    config = _config(
        tmp_path,
        name=f"post-marker-ddp-{phase}",
        max_steps=1,
        resume=False,
    )
    rows = _run_public_post_marker_failure(
        config,
        _free_port(),
        method_name=method_name,
        phase=phase,
    )

    assert all(row["ok"] for row in rows), rows
    assert [row["error_type"] for row in rows] == ["RunError", "RunError"]
    assert all(
        any(
            f"injected distributed post-marker {phase} failure" in item
            for item in row["chain"]
        )
        for row in rows
    )
    output_dir = tmp_path / f"post-marker-ddp-{phase}"
    assert (output_dir / "commits" / "commit_000001.json").is_file()
    assert (output_dir / "checkpoint_000001").is_dir()

    import visual_rl as vr

    status = vr.inspect_run(output_dir)
    assert status.committed_steps == 1
    assert status.authoritative_checkpoint == output_dir / "checkpoint_000001"


@pytest.mark.distributed
def test_nccl_root_commit_failure_synchronizes() -> None:
    """Run one real torchrun/NCCL root-only pre-marker commit failure."""

    rank = _nccl_torchrun_rank()
    master_port = int(os.environ["MASTER_PORT"])
    root = (
        Path(tempfile.gettempdir()).resolve()
        / f"visualrl-nccl-root-commit-{os.getuid()}-{master_port}"
    )
    ready = root / "ready"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(mode=0o700, parents=True)
        ready.write_text("ready\n", encoding="utf-8")
    else:
        _wait_for_path(ready, timeout_s=30.0)

    payload = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    output_dir = root / "run"
    payload["runtime"]["max_steps"] = 1
    payload["runtime"]["distributed"]["mode"] = "ddp"
    payload["runtime"]["distributed"]["device"] = "cuda"
    payload["runtime"]["distributed"]["timeout_s"] = 30.0
    payload["runtime"]["distributed"]["max_snapshot_tensor_bytes"] = 1 << 20
    payload["artifacts"]["output_dir"] = str(output_dir)
    payload["artifacts"]["checkpoint_every"] = 1
    payload["resume"]["from"] = None
    config = root / f"config-rank-{rank}.yaml"
    config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    import visual_rl as vr
    from visual_rl.artifacts.manager import ArtifactManager
    from visual_rl.errors import ArtifactError, RunError

    original = ArtifactManager.commit

    def fail_before_marker(self, transaction, *, checkpoint_path):
        del self, transaction, checkpoint_path
        raise ArtifactError("injected rank-zero pre-marker commit failure")

    ArtifactManager.commit = fail_before_marker
    caught: BaseException | None = None
    try:
        vr.load(config).run()
    except BaseException as exc:
        caught = exc
    finally:
        ArtifactManager.commit = original

    assert isinstance(caught, RunError)
    chain = _exception_chain(caught)
    assert any(
        "injected rank-zero pre-marker commit failure" in item
        for item in chain
    )
    result_path = root / f"result-rank-{rank}.json"
    result_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "error_type": type(caught).__name__,
                "injected_failure_observed": any(
                    "injected rank-zero pre-marker commit failure" in item
                    for item in chain
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if rank == 0:
        rank_paths = tuple(root / f"result-rank-{item}.json" for item in range(2))
        for path in rank_paths:
            _wait_for_path(path, timeout_s=30.0)
        results = tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in rank_paths
        )
        assert [item["rank"] for item in results] == [0, 1]
        assert all(item["error_type"] == "RunError" for item in results)
        assert all(item["injected_failure_observed"] for item in results)
        assert not list((output_dir / "commits").glob("commit_*.json"))
        assert not list(output_dir.glob("checkpoint_*"))
        assert list((output_dir / ".staging").iterdir()) == []
        status = vr.inspect_run(output_dir)
        assert status.committed_steps == 0
        sentinel = {
            "nodeid": (
                "tests/test_distributed_runner.py::"
                "test_nccl_root_commit_failure_synchronizes"
            ),
            "world_size": 2,
            "ranks_entered": [0, 1],
            "ranks_exited": [0, 1],
            "marker_advanced": False,
            "passed": True,
        }
        print(
            "VISUALRL_NCCL_RESULT="
            + json.dumps(sentinel, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
