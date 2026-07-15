"""Deprecated repository-local ``train.py --config`` compatibility entry."""

from __future__ import annotations

import argparse
import sys

from visual_rl.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--config", required=True)
    args, remainder = parser.parse_known_args(argv)
    print(
        "warning: train.py --config is deprecated; use visual-rl run CONFIG",
        file=sys.stderr,
    )
    return cli_main(["run", args.config, *remainder])


if __name__ == "__main__":
    raise SystemExit(main())
