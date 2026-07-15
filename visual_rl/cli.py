"""Stable command-line interface for VisualRL preflight and execution."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

from visual_rl.configs import (
    ConfigDocument,
    ExperimentSpec,
    KeyOverride,
    SourceRef,
    list_packaged_presets,
    read_experiment_spec,
    read_packaged_preset,
    resolve_experiment,
)
from visual_rl.preflight import (
    ResumePreflightError,
    StaticPreflightError,
    TrustedComponentError,
    static_preflight,
    trusted_component_load,
)
EXIT_SUCCESS = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_TRUSTED = 3
EXIT_RESUME = 4
EXIT_EXECUTION = 5
EXIT_ARTIFACT = 6
_CONFIG_COMMANDS = ("validate", "inspect", "run")
_OPERATIONAL_COMMANDS = ("status", "audit")
_COMMANDS = (*_CONFIG_COMMANDS, "presets", *_OPERATIONAL_COMMANDS)
_PACKAGED_PRESET_PREFIX = "preset:"


class _UsageError(ValueError):
    pass


class _ExecutionError(RuntimeError):
    pass


class _ResumeCLIError(RuntimeError):
    pass


class ArtifactCheckError(RuntimeError):
    """A safe, stable CLI diagnostic for an unavailable or invalid run."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="visual-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in _CONFIG_COMMANDS:
        command = subparsers.add_parser(name)
        command.add_argument(
            "config",
            metavar="CONFIG",
            help="configuration file or explicit preset:NAME reference",
        )
        command.add_argument("--set", dest="sets", action="append", default=[])
        command.add_argument("--json", action="store_true")
        if name in {"validate", "inspect"}:
            command.add_argument("--trusted-components", action="store_true")
        else:
            command.add_argument("--resume", metavar="PATH")
    presets = subparsers.add_parser(
        "presets",
        help="list presets shipped with the installed VisualRL package",
    )
    presets.add_argument("--json", action="store_true")
    for name in _OPERATIONAL_COMMANDS:
        command = subparsers.add_parser(
            name,
            help=f"inspect a run's {'lifecycle status' if name == 'status' else 'artifacts'}",
        )
        command.add_argument("run_dir", metavar="RUN_DIR")
        command.add_argument("--json", action="store_true")
    return parser


def _parse_set(raw: str, index: int, cwd: Path) -> KeyOverride:
    if "=" not in raw:
        raise _UsageError(f"--set expects KEY=VALUE, got {raw!r}")
    key, raw_value = raw.split("=", 1)
    if not key or any(not segment for segment in key.split(".")):
        raise _UsageError(f"--set has an invalid dotted key: {key!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise _UsageError(f"Invalid YAML value for --set {key!r}: {exc}") from exc
    return KeyOverride(
        key,
        value,
        SourceRef("set", f"CLI --set[{index}]", cwd),
    )


def _read_config_argument(value: str, cwd: Path) -> ExperimentSpec:
    if value.startswith(_PACKAGED_PRESET_PREFIX):
        name = value.removeprefix(_PACKAGED_PRESET_PREFIX)
        return ExperimentSpec(
            preset=read_packaged_preset(name),
            context_dir=cwd,
        )
    return read_experiment_spec(value)


def _resolve_from_args(args: argparse.Namespace, cwd: Path):
    spec = _read_config_argument(args.config, cwd)
    cli_sets = tuple(
        _parse_set(raw, index, cwd) for index, raw in enumerate(args.sets)
    )
    explicit_documents = spec.explicit_documents
    if getattr(args, "resume", None) is not None:
        explicit_documents += (
            ConfigDocument(
                {"paths": {"resume_from": args.resume}},
                SourceRef("explicit", "CLI --resume", cwd),
            ),
        )
    return resolve_experiment(
        replace(
            spec,
            set_overrides=spec.set_overrides + cli_sets,
            explicit_documents=explicit_documents,
        )
    )


def _provenance_payload(provenance: dict[str, SourceRef]) -> dict[str, Any]:
    return {
        key: {
            "kind": source.kind,
            "name": source.name,
            "base_dir": (
                None if source.base_dir is None else os.fspath(source.base_dir)
            ),
        }
        for key, source in sorted(provenance.items())
    }


def _execute(args: argparse.Namespace, cwd: Path) -> dict[str, Any]:
    if args.command == "presets":
        return {"presets": list(list_packaged_presets())}
    if args.command in _OPERATIONAL_COMMANDS:
        from visual_rl.artifacts import audit_run_artifacts, inspect_run_status

        run_dir = Path(args.run_dir)
        try:
            if args.command == "status":
                status = inspect_run_status(run_dir / "run_status.json")
                return {"run_dir": os.fspath(run_dir), "status": status}
            audit = audit_run_artifacts(run_dir)
            return {"run_dir": os.fspath(run_dir), "audit": audit}
        except Exception as exc:
            label = "Run status" if args.command == "status" else "Run audit"
            raise ArtifactCheckError(
                f"{label} is missing or invalid; inspect trusted process logs."
            ) from exc
    resolved = _resolve_from_args(args, cwd)
    report = static_preflight(resolved.config)
    if args.command == "run":
        report = trusted_component_load(resolved.config, report)
        from visual_rl.runner import (
            ExperimentRunner,
            ResumeError,
            prepare_resume_source,
        )

        try:
            prepare_resume_source(resolved.config.paths.resume_from)
            runner = ExperimentRunner(resolved.config)
            metrics = runner.run()
        except (ResumeError, ResumePreflightError) as exc:
            raise _ResumeCLIError(str(exc)) from exc
        except Exception as exc:
            raise _ExecutionError(str(exc)) from exc
        return {
            "preflight": report.to_dict(),
            "output_dir": os.fspath(runner.output_dir),
            "steps": len(metrics),
            "metrics": metrics,
        }
    if args.trusted_components:
        report = trusted_component_load(resolved.config, report)
    payload: dict[str, Any] = {"preflight": report.to_dict()}
    if args.command == "inspect":
        payload["config"] = resolved.values
        payload["provenance"] = _provenance_payload(resolved.provenance)
    return payload


def _envelope(
    command: str | None,
    *,
    code: int,
    data: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "command": command,
        "ok": code == 0,
        "status": "ok" if code == 0 else "error",
        "exit_code": code,
    }
    if data is not None:
        payload["data"] = data
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def _classify_error(exc: Exception) -> int:
    if isinstance(exc, ArtifactCheckError):
        return EXIT_ARTIFACT
    if isinstance(exc, (ResumePreflightError, _ResumeCLIError)):
        return EXIT_RESUME
    if isinstance(exc, TrustedComponentError):
        return EXIT_TRUSTED
    if isinstance(exc, _ExecutionError):
        return EXIT_EXECUTION
    if isinstance(
        exc,
        (
            _UsageError,
            StaticPreflightError,
            yaml.YAMLError,
            OSError,
            TypeError,
            ValueError,
        ),
    ):
        return EXIT_USAGE
    return EXIT_INTERNAL


def _result_exit_code(command: str, data: dict[str, Any]) -> int:
    if command == "status":
        return (
            EXIT_SUCCESS
            if data["status"]["ready_for_aggregation"]
            else EXIT_ARTIFACT
        )
    if command == "audit":
        return EXIT_SUCCESS if data["audit"]["valid"] else EXIT_ARTIFACT
    return EXIT_SUCCESS


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in arguments
    command = next(
        (item for item in arguments if item in _COMMANDS),
        None,
    )
    if json_mode and any(item in {"-h", "--help"} for item in arguments):
        data = {"help": _build_parser().format_help()}
        print(json.dumps(_envelope(command, code=0, data=data), sort_keys=True))
        return EXIT_SUCCESS
    try:
        args = _build_parser().parse_args(arguments)
        json_mode = bool(args.json)
        command = args.command
        if json_mode:
            with redirect_stdout(sys.stderr):
                data = _execute(args, Path.cwd())
        else:
            data = _execute(args, Path.cwd())
        code = _result_exit_code(command, data)
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:  # CLI owns stable error mapping and no-traceback output.
        code = _classify_error(exc)
        if json_mode:
            print(
                json.dumps(
                    _envelope(command, code=code, error=exc),
                    sort_keys=True,
                    default=str,
                )
            )
        else:
            print(f"visual-rl: {exc}", file=sys.stderr)
        return code

    if json_mode:
        print(
            json.dumps(
                _envelope(command, code=code, data=data),
                sort_keys=True,
                default=str,
            )
        )
    elif command == "inspect":
        print(yaml.safe_dump(data, sort_keys=False).rstrip())
    elif command == "validate":
        level = "trusted" if data["preflight"]["trusted"] else "static"
        print(f"validation successful ({level})")
    elif command == "presets":
        print("\n".join(data["presets"]))
    elif command == "status":
        status = data["status"]
        print(
            "run status: "
            f"{status['observed_state']} "
            f"(committed steps: {status['authoritative_completed_steps']})"
        )
    elif command == "audit":
        audit = data["audit"]
        outcome = "valid" if audit["valid"] else "invalid"
        print(
            "artifact audit: "
            f"{outcome} "
            f"(markers: {audit['commit_markers']}, steps: {audit['metric_rows']})"
        )
    else:
        print(f"run completed: {data['output_dir']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
