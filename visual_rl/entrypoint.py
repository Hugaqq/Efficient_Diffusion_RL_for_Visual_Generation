"""Minimal installed command-line entry for config-driven training."""

from __future__ import annotations

import argparse

from visual_rl.configs.schema import load_config
from visual_rl.runner import ExperimentRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visual-rl")
    parser.add_argument(
        "--config", required=True, help="Path to a VisualRL YAML config"
    )
    args = parser.parse_args(argv)
    ExperimentRunner(load_config(args.config)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
