"""python -m visual_rl.infer_h3: native H3 model + serial AV rollout.

Use a separate pinned H3 environment, never upgrade or embed into a live Wan
trainer. No HTTP server, external generator process, reward or optimizer.
"""
import argparse
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

from visual_rl.models.h3 import H3InferenceAdapter, H3Request, inspect_assets
from visual_rl.models.h3_components import audit_runtime, inspect_converted, stage_budget
from visual_rl.algorithms.rollout.h3 import run_h3_rollout


def read_requests(path, *, diagnostic=False):
    path = Path(path)
    raw = json.loads(path.read_text())
    if set(raw) != {"schema_version", "description", "requests"} or raw["schema_version"] != 1:
        raise ValueError("invalid native H3 scene config")
    if not isinstance(raw["requests"], list) or len(raw["requests"]) not in {1, 3}:
        raise ValueError("only one or three H3 validation requests are authorized")
    requests = []
    for row in raw["requests"]:
        row = dict(row)
        images = row.pop("images", [])
        if not isinstance(images, list):
            raise ValueError("H3 images must be an ordered local path list")
        request = H3Request(**row, images=tuple((path.parent / p).resolve(strict=True) for p in images))
        if diagnostic:
            request = replace(request, width=384, height=224, frames=107)
        requests.append(request)
    if len({r.sample_id for r in requests}) != len(requests):
        raise ValueError("duplicate H3 sample id")
    return tuple(requests)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--converted-root", required=True, type=Path, help="separate verified Diffusers-derived components")
    parser.add_argument("--model-root", required=True, type=Path, help="local official snapshot containing FL2VA/")
    parser.add_argument("--model-revision", required=True, help="full40-character HF revision used to download assets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true", help="hash/inspect assets and environment, never load a model")
    parser.add_argument("--diagnostic", action="store_true", help="clearly labeled384x224x107 capacity request, NOT final scenes")
    args = parser.parse_args(argv)
    from scripts.launch_flash140 import (
        ROOT, LOCK, ResourceMonitor, ResourceSafetyError, check_budget, exclusive_launch,
        git_identity, publish, resource_sample,
    )
    import visual_rl
    from visual_rl.models import h3
    if (Path(visual_rl.__file__).resolve().parent != ROOT / "visual_rl"
            or Path(h3.__file__).resolve() != ROOT / "visual_rl/models/h3.py"):
        raise RuntimeError("native H3 imports do not match the agreed candidate")
    sha = git_identity(args.expected_sha)
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before importing the backend")
    output = args.output.absolute()
    if ROOT in output.parents or output == ROOT or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("native H3 outputs must be outside source, without symlink parents")
    requests = read_requests(args.config, diagnostic=args.diagnostic)
    if not args.diagnostic and any((r.width, r.height, r.frames, r.steps) != (1344, 768, 124, 40) for r in requests):
        raise ValueError("normal acceptance requires1344x768x124/40-point baseline; use --diagnostic for small probes")
    output.parent.mkdir(parents=True, exist_ok=True)
    launch = output.with_name(output.name + ".launch")
    owner = {"git_sha": sha, "pid": os.getpid(), "source": str(ROOT), "output": str(output),
             "started_unix": time.time(), "rl5_gpu_passed": False, "diagnostic": args.diagnostic}
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    with exclusive_launch(LOCK, owner):
        if output.exists() or launch.exists():
            raise FileExistsError("native H3 run/attempt already exists; preserve it, no automatic restart")
        resources = resource_sample(output.parent)
        check_budget(resources)
        modules = audit_runtime()
        assets = inspect_assets(args.model_root, args.model_revision)
        converted = inspect_converted(args.converted_root, assets)
        stages = {}
        for name, folder in (("encoder", args.model_root / "FL2VA/text_encoder"),
                             ("denoiser", args.converted_root / "transformer"),
                             ("video_decoder", args.converted_root / "vae"),
                             ("audio_decoder", args.converted_root / "audio_vae")):
            files = list(folder.rglob("*.safetensors"))
            if not files:
                raise ValueError(f"missing phase weights: {name}")
            stages[name] = stage_budget(sum(p.stat().st_size for p in files),
                                       max(p.stat().st_size for p in files), resources)
        environment = {**owner, "modules": modules, "resources": resources, "stage_host_admission_gib": stages,
                       "config": str(args.config.resolve()), "model_root": str(args.model_root.resolve()),
                       "model_revision_declared": args.model_revision, "converted_root": str(args.converted_root.resolve()),
                       "conversion": converted}
        if args.preflight:
            publish(output.with_name(output.name + ".preflight.json"), {**environment, "assets": assets})
            print(json.dumps({"status": "preflight_only_not_generation", "stage_host_admission_gib": stages,
                              "modules": modules, "checkpoint_python_executed": False}, sort_keys=True))
            return 0
        launch.mkdir()
        publish(launch / "owner.json", environment)
        monitor = ResourceMonitor(output.parent, launch / "resources.json")
        def safety_signal(*_):
            raise ResourceSafetyError(monitor.error)
        previous = signal.signal(signal.SIGUSR1, safety_signal)
        started = time.monotonic()
        try:
            monitor.sample()
            monitor.thread.start()
            with H3InferenceAdapter.load(model_root=args.model_root, model_revision=args.model_revision,
                    converted_root=args.converted_root, resource_probe=lambda: resource_sample(output.parent)) as adapter:
                publish(launch / "assets.json", adapter.assets)
                publish(launch / "load.json", {"load_seconds": time.monotonic() - started})
                receipts = run_h3_rollout(adapter, requests, output, diagnostic=args.diagnostic)
            monitor.sample()
            final_modules = audit_runtime()
            publish(launch / "result.json", {"status": "native_av_generated_and_decoded", "samples": len(receipts),
                    "modules": final_modules, "no_serving_engines": True,
                    "elapsed_seconds": time.monotonic() - started, "rl5_gpu_passed": False,
                    "diagnostic": args.diagnostic, "resources_sampled_minima": monitor.minima, "peaks": monitor.peaks})
            # Overall SUCCESS only after native adapter.close and final resource
            # sampling return. ROLLOUT_COMPLETE alone is not full-run success.
            publish(output / "SUCCESS.json", {"status": "native_av_generated_and_decoded",
                    "git_sha": sha, "samples": len(receipts), "diagnostic": args.diagnostic,
                    "rl5_gpu_passed": False})
        except BaseException as error:
            publish(launch / "result.json", {"status": "blocked_or_failed", "error": str(error),
                    "error_type": type(error).__name__, "rl5_gpu_passed": False})
            raise
        finally:
            monitor.stop.set()
            if monitor.thread.is_alive():
                monitor.thread.join(timeout=30)
            signal.signal(signal.SIGUSR1, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
