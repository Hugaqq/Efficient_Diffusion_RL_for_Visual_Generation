"""Aggregate independent SD3 bounded-run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_rl.evaluation.cross_run import aggregate_sd3_run_summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-summary", action="append", required=True)
    parser.add_argument("--control-summary", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--min-training-seeds", type=int, default=3)
    parser.add_argument("--min-positive-fraction", type=float, default=0.8)
    args = parser.parse_args(argv)
    payload = aggregate_sd3_run_summaries(
        args.active_summary,
        args.control_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        min_training_seeds=args.min_training_seeds,
        min_positive_fraction=args.min_positive_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
