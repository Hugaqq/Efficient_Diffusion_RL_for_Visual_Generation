"""python -m visual_rl.train_h3: fresh-only real joint H3 LoRA diagnostic.

No inference engines, no automatic capacity fallback/retry or service changes.
Keep stderr/stdout/RC outside the source. A failure after optimizer_start is
fatal; never attempt to continue from an uncommitted in-memory policy.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/h3/training-diagnostic.json"))
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--converted-root", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, choices=(1, 5), default=1)
    parser.add_argument("--reward-origin", default="http://127.0.0.1:18820")
    parser.add_argument("--reward-identity", default="4c831491cb7b413839ad07459f777bd674cbd59aadbc7d7d3e2ce25b11c993e9")
    args = parser.parse_args(argv)
    import torch
    from visual_rl.algorithms.trainer.h3 import H3Diagnostic, H3GRPODiagnostic, H3VideoReward
    from visual_rl.models.h3_training import H3TrainingAdapter
    from visual_rl.models.h3_components import audit_runtime, stage_budget
    from scripts.launch_flash140 import (ROOT, LOCK, ResourceMonitor, ResourceSafetyError,
        check_budget, exclusive_launch, git_identity, publish, resource_sample)
    import visual_rl
    from visual_rl.models import h3_training
    if (Path(visual_rl.__file__).resolve().parent != ROOT / "visual_rl"
            or Path(h3_training.__file__).resolve() != ROOT / "visual_rl/models/h3_training.py"):
        raise RuntimeError("training imports do not match the candidate checkout")
    sha = git_identity(args.expected_sha)
    config = H3Diagnostic.read(args.config)
    output = args.output.absolute()
    if output == ROOT or ROOT in output.parents or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must be external, with no symlink parents")
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    owner = {"source": str(ROOT), "git_sha": sha, "pid": os.getpid(), "output": str(output),
             "updates_requested": args.updates, "diagnostic": True, "config": asdict(config),
             "started_unix": time.time(), "resume_supported": False}
    with exclusive_launch(LOCK, owner):
        if output.exists():
            raise FileExistsError("fresh output already exists; no resume/retry is supported")
        output.mkdir()
        publish(output / "owner.json", owner)
        monitor = ResourceMonitor(output.parent, output / "resources.json")
        algorithm = adapter = reward = None
        started = time.monotonic()
        def abort_safety(*_):
            raise ResourceSafetyError(monitor.error or "resource monitor signal")
        previous = signal.signal(signal.SIGUSR1, abort_safety)
        try:
            initial = resource_sample(output.parent)
            check_budget(initial)
            modules = audit_runtime()
            if (torch.__version__.split('+')[0] != "2.10.0"
                    or importlib.metadata.version("peft") != "0.18.1"
                    or importlib.metadata.version("transformers") != "5.19.0"):
                raise RuntimeError("use the pinned isolated H3 training stack")
            # Staged host admission, not a134GiB whole-asset sum. GPU DiT is
            # fully resident with activations/checkpointing and a16GiB floor.
            # No claim this unmeasured training workload fits an A10080GB.
            budgets = {}
            for name, folder in (("encoder", args.model_root / "FL2VA/text_encoder"),
                                 ("policy", args.converted_root / "transformer"),
                                 ("video_decoder", args.converted_root / "vae"),
                                 ("audio_decoder", args.converted_root / "audio_vae")):
                files = list(folder.rglob("*.safetensors"))
                if not files:
                    raise ValueError(f"missing component layout: {name}")
                budgets[name] = stage_budget(sum(p.stat().st_size for p in files),
                                             max(p.stat().st_size for p in files), initial)
            import peft
            modules["peft"] = {"version": importlib.metadata.version("peft"), "file": peft.__file__}
            publish(output / "environment.json", {"modules": modules, "initial_resources": initial,
                "host_stage_admission_gib": budgets, "model_revision_declared": args.model_revision,
                "model_root": str(args.model_root), "converted_root": str(args.converted_root),
                "placement": "full_DiT_GPU_during_rollout_replay_backward_CPU_park_during_decode",
                "checkpointing": "native_nonreentrant", "weight_integrity_scan": False})
            monitor.sample()
            monitor.thread.start()
            random.seed(config.seed)
            import numpy as np
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            # No ambient download or checkpoint Python; explicit offline gate in load.
            reward = H3VideoReward(args.reward_origin, args.reward_identity)
            adapter = H3TrainingAdapter.load(model_root=args.model_root, converted_root=args.converted_root,
                        model_revision=args.model_revision, resource_probe=lambda: resource_sample(output.parent))
            algorithm = H3GRPODiagnostic(adapter, reward, config, output,
                {"git_sha": sha, "model_revision_declared": args.model_revision, "source": str(ROOT)},
                publish, monitor.sample)
            result = algorithm.run(args.updates)
            adapter.close()
            adapter = None
            reward.close()
            reward = None
            monitor.sample()
            publish(output / "SUCCESS.json", {**result, "git_sha": sha, "seconds": time.monotonic()-started,
                "sampled_minima": monitor.minima, "peaks": monitor.peaks,
                "status": "real_H3_diagnostic_optimizer_updates_complete"})
        except BaseException as error:
            try:
                sample = resource_sample(output.parent)
            except Exception as telemetry_error:
                sample = {"telemetry_error": repr(telemetry_error)}
            capacity = isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError, ResourceSafetyError)) or "resource safety" in str(error) or "stage admission" in str(error)
            failure_kind = ("cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else
                            "host_memory_error" if isinstance(error, MemoryError) else
                            "safety_or_admission_blocked" if capacity else "code_or_external_error")
            publish(output / "FAILED.json", {"status": "capacity_or_safety_blocked" if capacity else "failed",
                "failure_kind": failure_kind,
                "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                "phase": algorithm.phase if algorithm else "environment_or_load",
                "committed_steps": algorithm.committed_steps if algorithm else 0,
                "resources": sample, "torch_allocated_gib": torch.cuda.memory_allocated()/1024**3,
                "torch_reserved_gib": torch.cuda.memory_reserved()/1024**3,
                "config": asdict(config), "git_sha": sha, "diagnostic": True,
                "notify_user_capacity": capacity, "automatic_retry": False})
            logging.exception("H3 training failed; preserve output and notify main immediately if capacity/safety")
            raise
        finally:
            monitor.stop.set()
            if monitor.thread.is_alive():
                monitor.thread.join(timeout=30)
            signal.signal(signal.SIGUSR1, previous)
            # A CUDA failure may also break cleanup. Preserve the original error
            # and its saved traceback instead of replacing it with a close error.
            original_error = sys.exc_info()[0] is not None
            for resource in (adapter, reward):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        if not original_error:
                            raise
                        logging.exception("cleanup failed after the preserved H3 failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
