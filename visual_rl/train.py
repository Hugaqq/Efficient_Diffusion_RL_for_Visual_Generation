"""Thin ``python -m visual_rl.train`` entry for schema-v2 execution."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from visual_rl.errors import (
    ComponentError,
    ConfigError,
    ValidationError,
    VisualRLError,
)

__all__ = ("main",)

_USAGE = "usage: python -m visual_rl.train CONFIG"


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one schema-v2 config through the sole production controller.

    Return values are stable process exit codes so tests and launch wrappers do
    not need to intercept ``SystemExit``. Only the module guard raises it.
    """

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    if arguments in {("-h",), ("--help",)}:
        print(_USAGE, file=out)
        return 0
    if len(arguments) != 1:
        print(_USAGE, file=err)
        return 2

    try:
        result = _create_controller().run(arguments[0])
    except KeyboardInterrupt:
        print(
            json.dumps(
                {"error": "KeyboardInterrupt", "status": "interrupted"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=err,
        )
        return 130
    except VisualRLError as exc:
        print(
            json.dumps(
                _error_payload(exc),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=err,
        )
        return (
            2 if isinstance(exc, (ConfigError, ComponentError, ValidationError)) else 1
        )

    print(
        json.dumps(
            {
                "authoritative_checkpoint": str(result.authoritative_checkpoint),
                "committed_steps": result.committed_steps,
                "output_dir": str(result.output_dir),
                "run_id": result.run_id,
                "status": "ok",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=out,
    )
    return 0


def _create_controller():
    """Delay runtime imports until after argument validation/help handling."""

    from visual_rl.runtime.composition import create_default_run_controller

    return create_default_run_controller()


def _error_payload(error: VisualRLError) -> dict[str, object]:
    payload: dict[str, object] = {
        "error": type(error).__name__,
        "message": str(error),
        "status": "error",
    }
    for name in (
        "code",
        "key",
        "path",
        "kind",
        "name",
        "line",
        "column",
        "source_schema_version",
        "required_schema_version",
        "migration_examples",
        "migration_mode",
    ):
        value = getattr(error, name, None)
        if isinstance(value, Path):
            value = str(value)
        if value is not None:
            payload[name] = value
    if isinstance(error, ValidationError) and error.checks is not None:
        payload["checks"] = [
            {
                "code": check.code,
                "level": check.level,
                "message": check.message,
                "path": check.path,
            }
            for check in error.checks
        ]
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
