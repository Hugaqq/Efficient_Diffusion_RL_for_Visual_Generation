"""Run one frozen Flow/Pick-a-Pic YAML through the sole public Python API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import visual_rl as vr


class ProgressPrinter(vr.Callback):
    """Print tensor-free lifecycle observations for the remote log."""

    @staticmethod
    def _emit(event: vr.CallbackEvent) -> None:
        payload = {
            "kind": event.kind,
            "run_id": event.run_id,
            "step": event.step,
            "target_steps": event.target_steps,
            "committed_steps": event.committed_steps,
            "metrics": dict(event.metrics),
        }
        print(json.dumps(payload, sort_keys=True), flush=True)

    on_run_start = _emit
    on_step_end = _emit
    on_commit = _emit
    on_run_end = _emit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    experiment = vr.load(config_path)
    report = experiment.validate()
    if not report.ok:
        errors = [
            {"code": check.code, "path": check.path, "message": check.message}
            for check in report.errors
        ]
        raise RuntimeError(
            "experiment preflight failed: " + json.dumps(errors, sort_keys=True)
        )

    result = experiment.run(callbacks=[ProgressPrinter()])
    status = vr.inspect_run(result.output_dir)
    audit = vr.audit_run(result.output_dir)
    summary = {
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "committed_steps": result.committed_steps,
        "status_ok": status.ok,
        "audit_ok": audit.ok,
        "checked_commit_count": audit.checked_commit_count,
    }
    print(json.dumps({"completed": summary}, sort_keys=True), flush=True)
    if not status.ok or not audit.ok:
        raise RuntimeError("completed run failed authoritative status/audit")


if __name__ == "__main__":
    main()
