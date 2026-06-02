"""Remote staging helpers for non-invasive SD3 TempFlow CLI smokes."""

from __future__ import annotations

import json
import posixpath
import shlex
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_ROOT = "/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke"
DEFAULT_SERVER = "v-qiaoqifan@10.130.140.73"
DEFAULT_LEGACY_REPO_ROOT = "/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke"
DEFAULT_CONDA_BIN = "/home/v-qiaoqifan/miniconda3/bin/conda"
DEFAULT_CONDA_ENV = "visual-rl-sd35"
ARCHIVE_NAME = "visual_rl_source.tar.gz"
REMOTE_SCRIPT_NAME = "run_remote_sd3_cli_smoke.sh"

_EXCLUDED_PARTS = {
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "reference_code",
    "runs",
}
_EXCLUDED_NAMES = {".DS_Store"}
_ROOT_FILES = ("pyproject.toml", "README.md")


def _default_stage_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"sd3_cli_smoke_{timestamp}"


@dataclass
class RemoteSd3CliSmokeConfig:
    server: str = DEFAULT_SERVER
    remote_root: str = DEFAULT_REMOTE_ROOT
    gpu: int = 2
    model_path: str = ""
    repo_root: str = DEFAULT_LEGACY_REPO_ROOT
    conda_env: str = DEFAULT_CONDA_ENV
    conda_bin: str = DEFAULT_CONDA_BIN
    resolution: int = 256
    num_steps: int = 3
    guidance_scale: float = 4.5
    seed: int = 23
    dtype: str = "bfloat16"
    lora_rank: int = 32
    lora_alpha: int = 64
    max_sequence_length: int = 128
    bounded_steps: int = 1
    resume_steps: int = 1
    idle_memory_mb: int = 1024
    idle_util_pct: int = 5
    stage_name: str = field(default_factory=_default_stage_name)
    prompt: str = "a red square"
    run_bounded_trainer: bool = True
    run_resume_validation: bool = True
    allow_long_run: bool = False
    dry_run: bool = True

    @property
    def remote_stage_dir(self) -> str:
        return posixpath.join(self.remote_root.rstrip("/"), self.stage_name.strip("/"))

    @property
    def remote_archive_path(self) -> str:
        return posixpath.join(self.remote_stage_dir, ARCHIVE_NAME)

    @property
    def remote_script_path(self) -> str:
        return posixpath.join(self.remote_stage_dir, REMOTE_SCRIPT_NAME)

    @property
    def remote_preview_dir(self) -> str:
        return posixpath.join(self.remote_stage_dir, "preview")

    @property
    def remote_bounded_dir(self) -> str:
        return posixpath.join(self.remote_stage_dir, f"bounded_{self.bounded_steps}step")

    @property
    def remote_bounded_checkpoint_dir(self) -> str:
        return posixpath.join(self.remote_bounded_dir, f"checkpoint_{self.bounded_steps:06d}")

    @property
    def remote_resume_dir(self) -> str:
        return posixpath.join(self.remote_stage_dir, f"resume_from_{self.bounded_steps}step_{self.resume_steps}step")


def _is_excluded(path: Path) -> bool:
    if path.name in _EXCLUDED_NAMES or path.suffix == ".pyc":
        return True
    return any(part in _EXCLUDED_PARTS for part in path.parts)


def iter_source_archive_members(source_root: str | Path) -> list[Path]:
    """Return source files that should be staged for the remote smoke."""

    root = Path(source_root)
    members: list[Path] = []
    for root_file in _ROOT_FILES:
        path = root / root_file
        if path.is_file():
            members.append(Path(root_file))

    package_root = root / "visual_rl"
    if package_root.is_dir():
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root)
            if not _is_excluded(rel_path):
                members.append(rel_path)
    return members


def create_source_archive(source_root: str | Path, archive_path: str | Path) -> list[str]:
    """Create a tar.gz archive containing only the package source needed remotely."""

    root = Path(source_root)
    members = iter_source_archive_members(root)
    if not members:
        raise ValueError(f"No visual_rl source files found under {root}.")

    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for rel_path in members:
            tar.add(root / rel_path, arcname=rel_path.as_posix(), recursive=False)
    return [path.as_posix() for path in members]


def _shell_array(items: list[str]) -> str:
    return "(" + " ".join(shlex.quote(str(item)) for item in items) + ")"


def _tempflow_cli_args(config: RemoteSd3CliSmokeConfig) -> list[str]:
    return [
        "tempflow-image-numeric-smoke",
        "--adapter",
        "sd3_tempflow",
        "--model-path",
        config.model_path,
        "--repo-root",
        config.repo_root,
        "--prompt",
        config.prompt,
        "--resolution",
        str(config.resolution),
        "--num-steps",
        str(config.num_steps),
        "--guidance-scale",
        str(config.guidance_scale),
        "--seed",
        str(config.seed),
        "--device",
        "cuda",
        "--dtype",
        config.dtype,
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--max-sequence-length",
        str(config.max_sequence_length),
    ]


def _image_preview_cli_args(config: RemoteSd3CliSmokeConfig) -> list[str]:
    return [
        "image-preview",
        "--adapter",
        "sd3_tempflow",
        "--model-path",
        config.model_path,
        "--repo-root",
        config.repo_root,
        "--prompt",
        config.prompt,
        "--resolution",
        str(config.resolution),
        "--num-steps",
        str(config.num_steps),
        "--guidance-scale",
        str(config.guidance_scale),
        "--seed",
        str(config.seed),
        "--device",
        "cuda",
        "--output-dir",
        config.remote_preview_dir,
    ]


def _bounded_trainer_cli_args(config: RemoteSd3CliSmokeConfig) -> list[str]:
    args = [
        "sd3-bounded-trainer-smoke",
        "--adapter",
        "sd3_tempflow",
        "--model-path",
        config.model_path,
        "--repo-root",
        config.repo_root,
        "--prompt",
        config.prompt,
        "--resolution",
        str(config.resolution),
        "--num-steps",
        str(config.num_steps),
        "--guidance-scale",
        str(config.guidance_scale),
        "--seed",
        str(config.seed),
        "--device",
        "cuda",
        "--dtype",
        config.dtype,
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--max-sequence-length",
        str(config.max_sequence_length),
        "--steps",
        str(config.bounded_steps),
        "--output-dir",
        config.remote_bounded_dir,
        "--disable-rollout-cache",
    ]
    if config.allow_long_run:
        args.append("--allow-long-run")
    return args


def _resume_trainer_cli_args(config: RemoteSd3CliSmokeConfig) -> list[str]:
    args = _bounded_trainer_cli_args(config)
    output_index = args.index("--output-dir") + 1
    steps_index = args.index("--steps") + 1
    args[output_index] = config.remote_resume_dir
    args[steps_index] = str(config.resume_steps)
    args.extend(["--resume-from", config.remote_bounded_checkpoint_dir])
    return args


def build_remote_script(config: RemoteSd3CliSmokeConfig) -> str:
    """Build the bash script that runs entirely inside the remote stage dir."""

    help_cmd = [config.conda_bin, "run", "-n", config.conda_env, "python", "-m", "visual_rl.cli", "--help"]
    preview_help_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        "image-preview",
        "--help",
    ]
    tempflow_help_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        "tempflow-image-numeric-smoke",
        "--help",
    ]
    bounded_help_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        "sd3-bounded-trainer-smoke",
        "--help",
    ]
    preview_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        *_image_preview_cli_args(config),
    ]
    smoke_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        *_tempflow_cli_args(config),
    ]
    bounded_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        *_bounded_trainer_cli_args(config),
    ]
    resume_cmd = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        "visual_rl.cli",
        *_resume_trainer_cli_args(config),
    ]
    bounded_block = ""
    if config.run_bounded_trainer:
        bounded_block = """run_logged bounded "$STAGE_DIR/sd3_bounded_trainer.stdout.log" "${BOUNDED_CMD[@]}"
test -s "$STAGE_DIR/bounded_${BOUNDED_STEPS}step/summary.json"
test -s "$STAGE_DIR/bounded_${BOUNDED_STEPS}step/metrics.jsonl"
test -s "$STAGE_DIR/bounded_${BOUNDED_STEPS}step/latest.json"
test -s "$STAGE_DIR/bounded_${BOUNDED_STEPS}step/previews/before/preview_000.png"
test -s "$STAGE_DIR/bounded_${BOUNDED_STEPS}step/previews/after/preview_000.png"
if ! find "$STAGE_DIR/bounded_${BOUNDED_STEPS}step" -maxdepth 1 -type d -name 'checkpoint_*' | grep -q .; then
  echo "[remote_smoke] bounded trainer did not create checkpoint_*" | tee "$STAGE_DIR/artifact_status.log"
  exit 78
fi
"""
        if config.run_resume_validation:
            bounded_block += """echo "[remote_smoke] GPU before resume" | tee "$STAGE_DIR/gpu_before_resume.log"
nvidia-smi -i "$GPU" | tee -a "$STAGE_DIR/gpu_before_resume.log"
nvidia-smi pmon -i "$GPU" -c 1 | tee "$STAGE_DIR/gpu_pmon_before_resume.log" || true
GPU_LINE="$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
MEM_USED_MB="$(printf '%s\\n' "$GPU_LINE" | awk -F, 'NR==1 {gsub(/ /,"",$1); print $1}')"
UTIL_PCT="$(printf '%s\\n' "$GPU_LINE" | awk -F, 'NR==1 {gsub(/ /,"",$2); print $2}')"
echo "[remote_smoke] resume_idle_guard gpu=$GPU memory_used_mb=$MEM_USED_MB util_pct=$UTIL_PCT thresholds=${IDLE_MEMORY_MB}MB/${IDLE_UTIL_PCT}%" | tee "$STAGE_DIR/gpu_idle_guard_before_resume.log"
if [ "$MEM_USED_MB" -gt "$IDLE_MEMORY_MB" ] || [ "$UTIL_PCT" -gt "$IDLE_UTIL_PCT" ]; then
  echo "[remote_smoke] GPU $GPU is not idle enough before resume; exiting 77." | tee -a "$STAGE_DIR/gpu_idle_guard_before_resume.log"
  exit 77
fi
PMON_PIDS="$(awk -v gpu="$GPU" '$1 == gpu && $2 ~ /^[0-9]+$/ {print $2}' "$STAGE_DIR/gpu_pmon_before_resume.log" | tr '\\n' ' ')"
if [ -n "$PMON_PIDS" ]; then
  echo "[remote_smoke] GPU $GPU has pmon process(es) before resume: $PMON_PIDS; exiting 77." | tee -a "$STAGE_DIR/gpu_idle_guard_before_resume.log"
  exit 77
fi
run_logged resume "$STAGE_DIR/sd3_bounded_trainer_resume.stdout.log" "${RESUME_CMD[@]}"
test -s "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step/summary.json"
test -s "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step/metrics.jsonl"
test -s "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step/latest.json"
test -s "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step/previews/before/preview_000.png"
test -s "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step/previews/after/preview_000.png"
if ! find "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step" -maxdepth 1 -type d -name 'checkpoint_*' | grep -q .; then
  echo "[remote_smoke] resume validation did not create checkpoint_*" | tee "$STAGE_DIR/artifact_status.log"
  exit 78
fi
grep -q -- '"resume_loaded": true' "$STAGE_DIR/resume_from_${BOUNDED_STEPS}step_${RESUME_STEPS}step/summary.json"
"""

    return f"""#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR={shlex.quote(config.remote_stage_dir)}
ARCHIVE_PATH={shlex.quote(config.remote_archive_path)}
LOG_PATH="$STAGE_DIR/sd3_cli_smoke.stdout.log"
GPU={shlex.quote(str(config.gpu))}
IDLE_MEMORY_MB={shlex.quote(str(config.idle_memory_mb))}
IDLE_UTIL_PCT={shlex.quote(str(config.idle_util_pct))}
BOUNDED_STEPS={shlex.quote(str(config.bounded_steps))}
RESUME_STEPS={shlex.quote(str(config.resume_steps))}
HELP_CMD={_shell_array(help_cmd)}
PREVIEW_HELP_CMD={_shell_array(preview_help_cmd)}
TEMPFLOW_HELP_CMD={_shell_array(tempflow_help_cmd)}
BOUNDED_HELP_CMD={_shell_array(bounded_help_cmd)}
PREVIEW_CMD={_shell_array(preview_cmd)}
SMOKE_CMD={_shell_array(smoke_cmd)}
BOUNDED_CMD={_shell_array(bounded_cmd)}
RESUME_CMD={_shell_array(resume_cmd)}

run_logged() {{
  local label="$1"
  local log_path="$2"
  shift 2
  echo "[remote_smoke] running $label: $*" | tee -a "$LOG_PATH"
  set +e
  "$@" 2>&1 | tee "$log_path"
  local status=${{PIPESTATUS[0]}}
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[remote_smoke] $label failed with status=$status" | tee -a "$LOG_PATH"
    exit "$status"
  fi
}}

mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"
tar -xzf "$ARCHIVE_PATH"

echo "[remote_smoke] hostname=$(hostname)" | tee "$STAGE_DIR/hostname.log"
echo "[remote_smoke] GPU before" | tee "$STAGE_DIR/gpu_before.log"
nvidia-smi -i "$GPU" | tee -a "$STAGE_DIR/gpu_before.log"
nvidia-smi pmon -i "$GPU" -c 1 | tee "$STAGE_DIR/gpu_pmon_before.log" || true

echo "[remote_smoke] visual_rl/cli.py hash" | tee "$STAGE_DIR/source_hash_cli.txt"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum visual_rl/cli.py | tee -a "$STAGE_DIR/source_hash_cli.txt"
else
  shasum -a 256 visual_rl/cli.py | tee -a "$STAGE_DIR/source_hash_cli.txt"
fi

"${{HELP_CMD[@]}}" 2>&1 | tee "$STAGE_DIR/cli_help.log"
grep -q -- "remote-sd3-cli-smoke" "$STAGE_DIR/cli_help.log"
"${{PREVIEW_HELP_CMD[@]}}" 2>&1 | tee "$STAGE_DIR/image_preview_help.log"
grep -q -- "--output-dir" "$STAGE_DIR/image_preview_help.log"
"${{TEMPFLOW_HELP_CMD[@]}}" 2>&1 | tee "$STAGE_DIR/tempflow_image_numeric_smoke_help.log"
grep -q -- "--repo-root" "$STAGE_DIR/tempflow_image_numeric_smoke_help.log"
"${{BOUNDED_HELP_CMD[@]}}" 2>&1 | tee "$STAGE_DIR/sd3_bounded_trainer_smoke_help.log"
grep -q -- "--disable-rollout-cache" "$STAGE_DIR/sd3_bounded_trainer_smoke_help.log"

GPU_LINE="$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
MEM_USED_MB="$(printf '%s\\n' "$GPU_LINE" | awk -F, 'NR==1 {{gsub(/ /,"",$1); print $1}}')"
UTIL_PCT="$(printf '%s\\n' "$GPU_LINE" | awk -F, 'NR==1 {{gsub(/ /,"",$2); print $2}}')"
echo "[remote_smoke] idle_guard gpu=$GPU memory_used_mb=$MEM_USED_MB util_pct=$UTIL_PCT thresholds=${{IDLE_MEMORY_MB}}MB/${{IDLE_UTIL_PCT}}%" | tee "$STAGE_DIR/gpu_idle_guard.log"
if [ "$MEM_USED_MB" -gt "$IDLE_MEMORY_MB" ] || [ "$UTIL_PCT" -gt "$IDLE_UTIL_PCT" ]; then
  echo "[remote_smoke] GPU $GPU is not idle enough; exiting 77 before model load." | tee -a "$STAGE_DIR/gpu_idle_guard.log"
  exit 77
fi
PMON_PIDS="$(awk -v gpu="$GPU" '$1 == gpu && $2 ~ /^[0-9]+$/ {{print $2}}' "$STAGE_DIR/gpu_pmon_before.log" | tr '\\n' ' ')"
if [ -n "$PMON_PIDS" ]; then
  echo "[remote_smoke] GPU $GPU has pmon process(es) before model load: $PMON_PIDS; exiting 77." | tee -a "$STAGE_DIR/gpu_idle_guard.log"
  exit 77
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$STAGE_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
touch "$LOG_PATH"
run_logged preview "$STAGE_DIR/image_preview.stdout.log" "${{PREVIEW_CMD[@]}}"
test -s "$STAGE_DIR/preview/preview_000.png"
test -s "$STAGE_DIR/preview/metadata.json"
run_logged numeric "$STAGE_DIR/tempflow_image_numeric_smoke.stdout.log" "${{SMOKE_CMD[@]}}"
{bounded_block}
echo "[remote_smoke] artifact_status=ok" | tee "$STAGE_DIR/artifact_status.log"

echo "[remote_smoke] GPU after" | tee "$STAGE_DIR/gpu_after.log"
nvidia-smi -i "$GPU" | tee -a "$STAGE_DIR/gpu_after.log" || true
nvidia-smi pmon -i "$GPU" -c 1 | tee "$STAGE_DIR/gpu_pmon_after.log" || true
exit 0
"""


def build_dry_run_payload(config: RemoteSd3CliSmokeConfig, source_root: str | Path | None = None) -> dict[str, Any]:
    source = Path.cwd() if source_root is None else Path(source_root)
    members = [path.as_posix() for path in iter_source_archive_members(source)]
    return {
        "dry_run": True,
        "config": asdict(config),
        "local_source_root": str(source),
        "archive_name": ARCHIVE_NAME,
        "archive_members": members,
        "remote_stage_dir": config.remote_stage_dir,
        "remote_archive_path": config.remote_archive_path,
        "remote_script_path": config.remote_script_path,
        "remote_preview_dir": config.remote_preview_dir,
        "remote_bounded_dir": config.remote_bounded_dir,
        "remote_bounded_checkpoint_dir": config.remote_bounded_checkpoint_dir,
        "remote_resume_dir": config.remote_resume_dir,
        "run_bounded_trainer": config.run_bounded_trainer,
        "run_resume_validation": config.run_resume_validation,
        "ssh_mkdir_command": ["ssh", config.server, f"mkdir -p {shlex.quote(config.remote_stage_dir)}"],
        "scp_archive_command": ["scp", "<local-archive>", f"{config.server}:{config.remote_archive_path}"],
        "scp_script_command": ["scp", "<local-script>", f"{config.server}:{config.remote_script_path}"],
        "ssh_run_command": ["ssh", config.server, f"bash {shlex.quote(config.remote_script_path)}"],
        "remote_script": build_remote_script(config),
    }


def run_remote_sd3_cli_smoke(config: RemoteSd3CliSmokeConfig, source_root: str | Path | None = None) -> dict[str, Any]:
    """Dry-run or execute the remote staging smoke."""

    if config.dry_run:
        return build_dry_run_payload(config, source_root=source_root)
    if not config.model_path:
        raise ValueError("--model-path is required when running remote-sd3-cli-smoke with --execute.")

    source = Path.cwd() if source_root is None else Path(source_root)
    with tempfile.TemporaryDirectory(prefix="visualrl_remote_smoke_") as tmpdir:
        tmp = Path(tmpdir)
        archive_path = tmp / ARCHIVE_NAME
        script_path = tmp / REMOTE_SCRIPT_NAME
        members = create_source_archive(source, archive_path)
        script_path.write_text(build_remote_script(config), encoding="utf-8")

        commands = [
            ["ssh", config.server, f"mkdir -p {shlex.quote(config.remote_stage_dir)}"],
            ["scp", str(archive_path), f"{config.server}:{config.remote_archive_path}"],
            ["scp", str(script_path), f"{config.server}:{config.remote_script_path}"],
            ["ssh", config.server, f"bash {shlex.quote(config.remote_script_path)}"],
        ]
        results = []
        for command in commands:
            completed = subprocess.run(command, check=False, text=True, capture_output=True)  # noqa: S603
            results.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            if completed.returncode != 0:
                return {
                    "dry_run": False,
                    "ok": False,
                    "remote_stage_dir": config.remote_stage_dir,
                    "archive_members": members,
                    "results": results,
                }

    return {
        "dry_run": False,
        "ok": True,
        "remote_stage_dir": config.remote_stage_dir,
        "archive_members": members,
        "results": results,
    }


def dumps_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
