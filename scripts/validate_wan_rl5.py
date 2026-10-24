"""Fresh RL5 config through the existing controller; observe, do not schedule.

No recovery/reset/loop/model/reward implementation lives here. Reuse the proven
identity/lock/resource utilities and real HPS cache handoff from SMALL40.
"""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time

from scripts.launch_flash140 import (
    ROOT, OVERLAY, LOCK, ResourceMonitor, ResourceSafetyError,
    check_budget, configure_imports, exclusive_launch, git_identity,
    inspect_checkpoint, prepare_output, publish, resource_sample,
)

CONFIG = ROOT / "configs/v2/wan_rl5.yaml"


def model_digest(bound):
    """Hash real saved trainables, not a scalar proxy or the full frozen base."""
    from visual_rl.models.state.io import ModelStateAdapter
    import torch
    payload = ModelStateAdapter(bound.prepared.manager.parameter_state).capture().to_payload()
    digest = hashlib.sha256()
    for name, tensor in sorted(payload["tensors"].items()):
        cpu = tensor.detach().to("cpu").contiguous()
        if not bool(torch.isfinite(cpu).all()):
            raise RuntimeError("nonfinite trainable parameter")
        digest.update(str((name, tuple(cpu.shape), str(cpu.dtype))).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def observe_updates(controller, output):
    """Read-only observations at existing prepare/checkpoint boundaries.

    Refuse terminal SUCCESS before all five real updates have passed. This
    never invokes optimizer.step, creates a model, or changes a training target.
    """
    backend = controller._backend
    history = []
    original_prepare = backend.prepare_run
    initial = []

    def prepared(bound):
        original_prepare(bound)
        initial.append(model_digest(bound))
        publish(output / "rl5-initial.json", {"trainable_sha256": initial[0]})

    backend.prepare_run = prepared
    sink = backend.checkpoint_sink

    def observe(original):
        def checkpoint(request):
            step = request.summary.committed_steps
            if step != len(history) + 1 or step > 5 or not initial:
                raise RuntimeError("RL5 checkpoint/commit order mismatch")
            update = request.summary.last_iteration.value.payload.update
            metrics = {name: getattr(update, name) for name in (
                "loss", "policy_loss", "approx_kl", "clipfrac", "logprob_delta_max",
                "active_transition_count", "gradient_norm_pre_clip", "gradient_norm_post_clip")}
            if (update.optimizer_step + 1 != step or not update.transaction.committed
                    or metrics["active_transition_count"] != 2
                    or any(not math.isfinite(v) for v in metrics.values())
                    or metrics["gradient_norm_pre_clip"] <= 0
                    or metrics["gradient_norm_post_clip"] <= 0):
                raise RuntimeError("RL5 requires real finite nonzero-gradient optimizer commits")
            digest = model_digest(request.bound)
            previous = initial[0] if not history else history[-1]["trainable_sha256"]
            if digest == previous:
                raise RuntimeError("RL5 optimizer commit did not change saved trainables")
            receipt = original(request)  # existing owner atomically commits/finalizes
            record = {"committed_steps": step, "trainable_sha256": digest, "metrics": metrics}
            history.append(record)
            publish(output / f"rl5-update-{step}.json", record)
            logging.getLogger("visual_rl.flash140").info(json.dumps({"event": "rl5_update_verified", **record}))
            return receipt
        return checkpoint

    sink.checkpoint_safe_point = observe(sink.checkpoint_safe_point)
    sink.checkpoint = observe(sink.checkpoint)
    return history


def check_hps(raw):
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract, validate_health
    artifact = json.loads(Path(raw["launch"]["artifacts"]["rewards"]["reward_general"]).read_text())
    identity = raw["overrides"]["rewards"]["reward_general"]["params"]["resource"]["semantic_factory_config"]["service_identity"]
    contract = VideoContract(81, 480, 832)
    validate_health(artifact, identity, contract)
    client = Hpsv3VideoClient(origin=raw["launch"]["reward_runtime_bindings"]["reward_general"]["endpoint"],
                            service_identity=identity, contract=contract, timeout_s=30, max_response_bytes=262144)
    try:
        if client.healthcheck() != artifact:
            raise RuntimeError("existing HPS health/artifact mismatch; do not change the service")
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--preflight", action="store_true", help="read-only; never starts training")
    args = parser.parse_args(argv)
    sha = git_identity(args.expected_sha)
    imports = configure_imports(OVERLAY)
    import yaml
    from visual_rl.composition.config.compiler import compile_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe
    from visual_rl.runtime.composition import create_default_run_controller
    from scripts.flash_small40_repair import CacheHandoffRewardFactory
    compile_recipe_v2(load_source_recipe(CONFIG))
    raw = yaml.safe_load(CONFIG.read_text())
    output = Path(raw["launch"]["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    owner = {"git_sha": sha, "pid": os.getpid(), "source": str(ROOT),
             "output": str(output), "started_unix": time.time()}
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    with exclusive_launch(LOCK, owner):
        if output.exists():
            raise FileExistsError("RL5 is fresh-only; preserve the existing output, do not repeat launch")
        sample = resource_sample(output.parent)
        check_budget(sample)  # shared 16GiB GPU/24GiB host/64GiB disk margins
        if sample["gpu_free_gib"] < 32 or sample["host_available_gib"] < 48:
            raise RuntimeError("RL5 admission needs >=32GiB GPU and >=48GiB host free (unmeasured budget)")
        check_hps(raw)
        if args.preflight:
            print(json.dumps({"status": "preflight_only", **owner, "imports": imports, "resources": sample}))
            return 0
        prepare_output(output, identity={"git_sha": sha, "source_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest()})
        publish(output / "rl5-owner.json", {**owner, "imports": imports, "admission": sample})
        monitor = ResourceMonitor(output, output / "rl5-resources.json")
        def safety(*_):
            raise ResourceSafetyError(monitor.error)
        previous_signal = signal.signal(signal.SIGUSR1, safety)
        controller = create_default_run_controller(code_root=ROOT / "visual_rl",
                                                   reward_resource_factory=CacheHandoffRewardFactory())
        history = observe_updates(controller, output)
        try:
            monitor.sample()
            monitor.thread.start()
            result = controller.run(CONFIG)
            final = inspect_checkpoint(result.authoritative_checkpoint)
            if (result.committed_steps != 5 or len(history) != 5
                    or final["committed_steps"] != 5 or final["commit_cursor"] != 1
                    or final["pending_rows"] != 4 or final["evaluations_completed"]):
                raise RuntimeError("RL5 terminal checkpoint lost the odd pending pool or target")
            monitor.sample()
            publish(output / "rl5-result.json", {"status": "verified_five_updates", "git_sha": sha,
                    "updates": history, "checkpoint": final})
        except BaseException as error:
            publish(output / "rl5-result.json", {"status": "blocked_or_failed", "error": str(error),
                    "error_type": type(error).__name__, "git_sha": sha, "verified_updates": len(history)})
            raise
        finally:
            monitor.stop.set()
            if monitor.thread.is_alive():
                monitor.thread.join(timeout=30)
            signal.signal(signal.SIGUSR1, previous_signal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
