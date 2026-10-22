"""Single-process admission/observability wrapper around the production controller.

No training loop, model/reward replacement, dependency installation or service
management. `validate` pauses ONLY after a real durable non-terminal checkpoint;
the recipe endpoint stays140, including its LR schedule and all evaluations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/flash_wan_capacity_20261022/approved140-runtime.yaml"
OVERLAY = Path("/home/ubuntu/qifan/visualrl-flash140-20261022/dependency-overlay")
PYTHON = Path("/home/ubuntu/qifan/envs/visual-rl/bin/python")
LOCK = Path("/mnt/data/qifan/runs/flash140-20261022/.flash140-launch.lock")
GIB = 1024 ** 3
GPU_WORKLOAD_GIB = 52.03  # observed batch4 reserved, NOT an end-to-end peak
GPU_MARGIN_GIB = 16.0
HOST_MARGIN_GIB = 24.0
HOST_WORKLOAD_GIB = 64.0  # conservative admission budget, not a measurement
DISK_MARGIN_GIB = 64.0


def publish(path, value):
    from visual_rl.artifacts.terminal import atomic_write
    atomic_write(Path(path), (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def git_identity(expected):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=30).strip()
    observed = git("rev-parse", "HEAD")
    if len(expected) != 40 or observed != expected:
        raise RuntimeError(f"expected full Git SHA {expected}; checkout is {observed}")
    # Includes untracked files; do not hide edits or ignore dirty imports.
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("source checkout is dirty; preserve it and deploy a fresh Git worktree")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"launch from the clean checkout root: {ROOT}")
    return observed


def configure_imports(overlay):
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        raise RuntimeError(f"use the training interpreter {PYTHON}, not {sys.executable}")
    overlay = Path(overlay).resolve(strict=True)
    if (overlay / "visual_rl").exists() or (overlay / "visual_rl.py").exists():
        raise RuntimeError("dependency overlay must not shadow training source")
    sys.path[:0] = [str(ROOT), str(overlay)]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    import visual_rl
    from visual_rl.algorithms.trainer import collection
    if (Path(visual_rl.__file__).resolve() != ROOT / "visual_rl/__init__.py"
            or Path(collection.__file__).resolve() != ROOT / "visual_rl/algorithms/trainer/collection.py"):
        raise RuntimeError("actual imports do not match the clean candidate source")
    modules = {}
    for name in ("torch", "diffusers", "transformers", "peft", "accelerate", "numpy",
                 "yaml", "PIL", "requests", "imageio", "imageio_ffmpeg", "ftfy", "wcwidth"):
        module = importlib.import_module(name)
        version = str(getattr(module, "__version__", "unknown"))
        origin = Path(module.__file__).resolve()
        modules[name] = {"version": version, "file": str(origin)}
        if name in ("ftfy", "wcwidth") and overlay not in origin.parents:
            raise RuntimeError(f"{name} must import from the declared dependency overlay")
    for name, version in (("torch", "2.7.1+cu128"), ("diffusers", "0.33.1"),
                          ("ftfy", "6.3.1"), ("wcwidth", "0.2.14")):
        if modules[name]["version"] != version:
            raise RuntimeError(f"dependency mismatch: {name} {modules[name]['version']} != {version}; do not upgrade shared env")
    return modules


@contextmanager
def exclusive_launch(path, owner):
    """Persistent flock inode: never unlink it, never kill another owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Flash140 launcher owns the lock; do not start a duplicate") from error
        stream.seek(0)
        stream.truncate()
        json.dump(owner, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_budget(sample, *, admission=False, disk_required_gib=DISK_MARGIN_GIB):
    floors = {"gpu_free_gib": GPU_MARGIN_GIB + (GPU_WORKLOAD_GIB if admission else 0),
              "host_available_gib": HOST_MARGIN_GIB + (HOST_WORKLOAD_GIB if admission else 0),
              "disk_free_gib": disk_required_gib}
    for name, minimum in floors.items():
        observed = sample[name]
        if not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed < minimum:
            raise RuntimeError(f"resource safety: {name}={observed} below required {minimum:.2f}GiB")


def resource_sample(output_parent):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("require exactly one visible CUDA GPU (set CUDA_VISIBLE_DEVICES explicitly)")
    free, total = torch.cuda.mem_get_info(0)
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"gpu_free_gib": free / GIB, "gpu_total_gib": total / GIB,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_peak_reserved_gib": torch.cuda.max_memory_reserved(0) / GIB,
            "host_available_gib": int(memory["MemAvailable"].split()[0]) * 1024 / GIB,
            "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB,
            "disk_free_gib": shutil.disk_usage(output_parent).free / GIB}


class ResourceSafetyError(RuntimeError):
    pass


class ResourceMonitor:
    """15s sampled heartbeat; terminate ONLY this launcher on a safety breach.

    Native CUDA/HTTP calls can delay delivery until Python regains control.
    Sampled minima are not a proof of the unsampled device-wide peak.
    """
    def __init__(self, output, report, *, gpu_reserved_limit=None):
        self.output, self.report = Path(output), Path(report)
        self.gpu_reserved_limit = gpu_reserved_limit
        self.stop = threading.Event()
        self.error = None
        self.minima = {}
        self.peaks = {}
        self.thread = threading.Thread(target=self._run, name="flash140-resource-heartbeat", daemon=True)

    def sample(self):
        from visual_rl.runtime.flash140_logging import progress
        observed = resource_sample(self.output)
        for name in ("gpu_free_gib", "host_available_gib", "disk_free_gib"):
            self.minima[name] = min(observed[name], self.minima.get(name, observed[name]))
        for name in ("torch_peak_reserved_gib", "process_peak_rss_gib"):
            self.peaks[name] = max(observed[name], self.peaks.get(name, observed[name]))
        publish(self.report, {"sample": observed, "sampled_minima": self.minima, "peaks": self.peaks,
                              "time_unix": time.time(), "full_gpu_validation": False})
        progress("resource_heartbeat", **observed)
        check_budget(observed)
        if self.gpu_reserved_limit is not None and self.peaks["torch_peak_reserved_gib"] > self.gpu_reserved_limit:
            raise RuntimeError("resource safety: Torch peak reservation exceeded admitted GPU budget")
        return observed

    def _run(self):
        while not self.stop.wait(15):
            try:
                self.sample()
            except Exception as error:
                self.error = str(error)
                if not self.stop.is_set():
                    os.kill(os.getpid(), signal.SIGUSR1)
                return


class SafePointPause(RuntimeError):
    def __init__(self, record):
        super().__init__("paused after durable non-terminal checkpoint (NOT training SUCCESS)")
        self.record = record


def observe_checkpoints(controller, *, pause_after=None, on_checkpoint=None):
    """Observe the existing sole sink; do not replace checkpoint/update ownership."""
    from visual_rl.runtime.flash140_logging import progress
    sink = controller._backend.checkpoint_sink
    original = sink.checkpoint_safe_point
    def observed(request):
        step = request.summary.committed_steps
        progress("checkpoint_start", committed_steps=step)
        receipt = original(request)  # pause only AFTER atomic commit returns
        update = request.summary.last_iteration.value.payload.update
        metrics = {name: getattr(update, name) for name in (
            "loss", "policy_loss", "approx_kl", "clipfrac", "active_transition_count",
            "logprob_delta_max", "gradient_norm_pre_clip", "gradient_norm_post_clip")}
        record = {"status": "durable_non_terminal", "committed_steps": step,
                  "max_optimizer_steps": request.bound.prepared.training.max_optimizer_steps,
                  "checkpoint": str(receipt.checkpoint_path), "metrics": metrics,
                  "full_gpu_validation": False}
        progress("checkpoint_durable", **record)
        if on_checkpoint is not None:
            on_checkpoint(record)
        if pause_after is not None and step >= pause_after:
            raise SafePointPause({**record, "status": "paused_not_success"})
        return receipt
    sink.checkpoint_safe_point = observed


def prepare_output(output, *, identity, resume=None, retry_incomplete=False):
    """Fail closed on unknown outputs, old code/config, and ambiguous recovery."""
    output = Path(output).absolute()
    if output == ROOT or ROOT in output.parents:
        raise RuntimeError("runtime outputs must stay outside the clean source checkout")
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise RuntimeError("output path must not contain symlinks")
    owner_path = output / "flash140-owner.json"
    if resume is None and not retry_incomplete:
        if output.exists():
            raise FileExistsError("fresh output already exists; choose a new path or explicit resume/retry")
        output.mkdir(parents=True)
        publish(owner_path, identity)
    else:
        if not output.is_dir() or owner_path.is_symlink() or not owner_path.is_file():
            raise RuntimeError("resume/retry requires a launcher-owned output")
        if json.loads(owner_path.read_text()) != identity:
            raise RuntimeError("output belongs to different source/config; no old candidate conversion")
        if (output / "SUCCESS").exists():
            raise RuntimeError("output is already terminal; do not launch again")
        # Orphan atomic step-N.tmp-* directories are not checkpoints; leave
        # them untouched. A damaged final step-N is never silently skipped.
        checkpoints = [p for p in (output / "checkpoints").glob("step-*")
                       if p.name.removeprefix("step-").isdigit()]
        if resume is not None:
            resume = Path(resume).absolute()
            if (resume.parent != output / "checkpoints" or resume.is_symlink() or not resume.is_dir()
                    or not resume.name.removeprefix("step-").isdigit()):
                raise RuntimeError("resume must name an owned authoritative checkpoints/step-N")
            if resume != max(checkpoints, key=lambda p: int(p.name.removeprefix("step-"))):
                raise RuntimeError("resume must select the latest checkpoint; do not overwrite newer history")
        elif checkpoints:
            raise RuntimeError("checkpoints exist; explicit --resume is required")
    return output


def inspect_checkpoint(path):
    """Read a TRUSTED local atomic checkpoint, one full pool at a time."""
    import torch
    from visual_rl.artifacts.checkpoint import AtomicCheckpointManager, RankCheckpointReader
    manager = AtomicCheckpointManager(Path(path).parent)
    inspection = manager.inspect_complete(Path(path))
    snapshot = RankCheckpointReader(manager).read_rank_snapshot(inspection, expected_world_size=1, expected_rank=0)
    state = snapshot.component_state("data_plane").payload["runtime_state"]
    if state["schema_version"] != 2 or "behavior" in state:
        raise RuntimeError("checkpoint is not a simplified schema2 candidate")
    def fingerprint(tensors):
        result = hashlib.sha256()
        count = 0
        for name, tensor in tensors:
            if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("checkpoint parameter must be finite CPU data")
            result.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
            result.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            count += tensor.numel()
        return result.hexdigest(), count
    model = snapshot.component_state("model").payload
    parameter_hash, parameter_count = fingerprint(sorted(model["tensors"].items()))
    ema_hash, ema_count = fingerprint(enumerate(state["ema"]))
    pool = state["pool"]
    return {"checkpoint": str(path), "committed_steps": state["committed_steps"],
            "commit_cursor": state["commit_cursor"], "pending_rows": 0 if pool is None else pool["execution"].trajectory.batch_size,
            "ema_parameter_count": ema_count, "ema_sha256": ema_hash,
            "trainable_parameter_count": parameter_count, "trainable_sha256": parameter_hash,
            "evaluations_completed": state["evaluations_completed"],
            "last_metrics": dict(snapshot.component_state("run_checkpoint_summary").payload.last_metrics),
            "full_gpu_validation": False}


def render_config(raw, output, resume):
    import copy
    rendered = copy.deepcopy(raw)
    if raw["overrides"]["training"]["max_optimizer_steps"] != 140:
        raise RuntimeError("formal endpoint must remain140, including validation")
    rendered["launch"]["output_dir"] = str(output)
    rendered["launch"]["resume_from"] = None if resume is None else str(Path(resume).absolute())
    return rendered


def check_artifacts(raw):
    """Read-only admission; the real controller still hashes all model contents."""
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract, decode_json, validate_health, MAX_HEALTH_BYTES
    artifacts = raw["launch"]["artifacts"]
    manifest = json.loads((CONFIG.parent / "workload-140.json").read_text())
    for key, path in (("train", artifacts["datasets"]["main"]), ("test", artifacts["evaluation_prompts"])):
        data = Path(path).read_bytes()
        expected = manifest["author_datasets"][key]
        if (hashlib.sha256(data).hexdigest() != expected["sha256"]
                or len(data.decode().splitlines()) != expected["nonempty_rows"]):
            raise RuntimeError(f"author {key} data hash/count mismatch")
    model = Path(artifacts["model"])
    for name in ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json",
                 "vae/config.json", "text_encoder/config.json", "tokenizer/tokenizer_config.json"):
        if not (model / name).is_file():
            raise RuntimeError(f"missing local model asset: {model / name}")
    for component in ("transformer", "vae", "text_encoder"):
        if not list((model / component).glob("*.safetensors")):
            raise RuntimeError(f"missing local {component} safetensors; no downloads allowed")
    semantic = raw["overrides"]["rewards"]["reward_general"]["params"]["resource"]["semantic_factory_config"]
    contract = VideoContract(81, 480, 832)
    artifact_path = Path(artifacts["rewards"]["reward_general"])
    with artifact_path.open("rb") as stream:
        artifact = decode_json(stream.read(MAX_HEALTH_BYTES + 1), MAX_HEALTH_BYTES)
    validate_health(artifact, semantic["service_identity"], contract)
    binding = raw["launch"]["reward_runtime_bindings"]["reward_general"]
    client = Hpsv3VideoClient(origin=binding["endpoint"], service_identity=semantic["service_identity"],
                            contract=contract, timeout_s=30, max_response_bytes=MAX_HEALTH_BYTES)
    try:
        if client.healthcheck() != artifact:
            raise RuntimeError("live HPSv3 identity differs from the pinned artifact; main must handle ownership")
    finally:
        client.close()  # does not restart/move/change the existing worker
    return {"train_rows": 19700, "eval_rows": 300, "service_identity": semantic["service_identity"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "validate", "run", "inspect"))
    parser.add_argument("--expected-sha", required=True, help="full40-character pulled Git commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, default=OVERLAY)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", type=Path)
    recovery.add_argument("--retry-incomplete", action="store_true", help="only pre-first-checkpoint receipts/chunks")
    parser.add_argument("--until", type=int, choices=(1, 2, 8), help="validate only; absolute safe-point count, default2")
    parser.add_argument("--compare", type=Path, help="inspect only: compare another trusted checkpoint's trainable hash")
    args = parser.parse_args(argv)
    if args.mode != "validate" and args.until is not None:
        parser.error("--until is only for bounded validate; formal run stays140")
    if args.compare is not None and args.mode != "inspect":
        parser.error("--compare is only for read-only checkpoint inspection")
    if args.mode == "inspect" and (args.resume is None or args.retry_incomplete):
        parser.error("inspect requires --resume pointing to a trusted owned checkpoint")
    sha = git_identity(args.expected_sha)
    modules = configure_imports(args.overlay)
    import yaml
    from visual_rl.runtime.composition import create_default_run_controller
    from visual_rl.runtime.flash140_logging import progress
    raw = yaml.safe_load(CONFIG.read_text())
    output = args.output.absolute()
    if output == ROOT or ROOT in output.parents or output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise RuntimeError("outputs must be outside source and must not contain symlinks")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    identity = {"git_sha": sha, "source_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
    owner = {**identity, "pid": os.getpid(), "uid": os.getuid(), "source": str(ROOT),
             "output": str(output), "mode": args.mode, "started_unix": time.time()}
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    with exclusive_launch(LOCK, owner):
        progress("launch_admission_start", **owner, imports=modules)
        if args.mode == "inspect":
            if json.loads((output / "flash140-owner.json").read_text()) != identity:
                raise RuntimeError("inspection requires the same source/config owned output")
            for path in (args.resume, args.compare):
                if path is not None and (path.absolute().parent != output / "checkpoints" or path.is_symlink()):
                    raise RuntimeError("inspect only trusted checkpoints inside the owned output")
            first = inspect_checkpoint(args.resume.absolute())
            if args.compare is not None:
                second = inspect_checkpoint(args.compare.absolute())
                first["comparison"] = second
                first["trainable_parameters_differ"] = first["trainable_sha256"] != second["trainable_sha256"]
            progress("checkpoint_inspection", **first)
            return 0
        sample = resource_sample(parent)
        # All checkpoints are retained. Formal admission reserves1TiB; validate
        # reserves64GiB. Measured checkpoint sizes are reported for replanning.
        check_budget(sample, admission=True, disk_required_gib=1024 if args.mode == "run" else 64)
        artifacts = check_artifacts(raw)
        if args.mode == "preflight":
            progress("preflight_only", sample=sample, artifacts=artifacts, full_gpu_validation=False)
            return 0
        if args.mode == "validate" and args.resume is not None:
            if int(args.resume.name.removeprefix("step-")) >= (args.until or 2):
                raise RuntimeError("validation --until must be beyond the explicitly resumed checkpoint")
        output = prepare_output(output, identity=identity, resume=args.resume, retry_incomplete=args.retry_incomplete)
        attempt = output / "launcher" / str(time.time_ns())
        attempt.mkdir(parents=True)
        handler = logging.FileHandler(attempt / "events.jsonl")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("visual_rl.flash140").addHandler(handler)
        publish(attempt / "owner.json", {**owner, "imports": modules, "admission": sample, "artifacts": artifacts})
        publish(output / "launcher-pid.json", owner)
        launch = render_config(raw, output, args.resume)
        config_path = attempt / "launch.yaml"
        config_path.write_text(yaml.safe_dump(launch))
        monitor = ResourceMonitor(output, attempt / "resources.json",
                                  gpu_reserved_limit=sample["gpu_free_gib"] - GPU_MARGIN_GIB)
        def safety_signal(*_):
            raise ResourceSafetyError(monitor.error)
        def term_signal(*_):
            raise KeyboardInterrupt("launcher SIGTERM")
        old_usr1 = signal.signal(signal.SIGUSR1, safety_signal)
        old_term = signal.signal(signal.SIGTERM, term_signal)
        def checkpoint_record(record):
            checkpoint = Path(record["checkpoint"])
            record["checkpoint_disk_bytes"] = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
            publish(attempt / f"checkpoint-{record['committed_steps']}.json", record)
            monitor.sample()  # includes compact-pool/checkpoint peak, before pause
        try:
            controller = create_default_run_controller(code_root=ROOT / "visual_rl")
            observe_checkpoints(controller, pause_after=(args.until or 2) if args.mode == "validate" else None,
                                on_checkpoint=checkpoint_record)
            monitor.sample()
            monitor.thread.start()
            result = controller.run(config_path)
            monitor.sample()
            if result.committed_steps != 140:
                raise RuntimeError("formal endpoint did not commit140")
            publish(attempt / "result.json", {"status": "training_terminal", "committed_steps": 140,
                    "checkpoint": str(result.authoritative_checkpoint), "full_gpu_validation": False})
        except SafePointPause as pause:
            notes = list(getattr(pause, "__notes__", ()))
            if notes:
                publish(attempt / "result.json", {"status": "blocked_or_failed", "cleanup_notes": notes,
                        "git_sha": sha, "full_gpu_validation": False})
                raise RuntimeError(f"safe-point pause teardown failed: {notes}") from pause
            record = {**pause.record, "git_sha": sha, "sampled_minima": dict(monitor.minima), "peaks": dict(monitor.peaks)}
            publish(attempt / "result.json", record)
            progress("validation_paused_not_success", **record)
        except BaseException as error:
            publish(attempt / "result.json", {"status": "blocked_or_failed", "error": str(error),
                    "error_type": type(error).__name__, "git_sha": sha, "full_gpu_validation": False})
            raise
        finally:
            monitor.stop.set()
            if monitor.thread.is_alive():
                monitor.thread.join(timeout=30)
            signal.signal(signal.SIGUSR1, old_usr1)
            signal.signal(signal.SIGTERM, old_term)
            logging.getLogger("visual_rl.flash140").removeHandler(handler)
            handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
