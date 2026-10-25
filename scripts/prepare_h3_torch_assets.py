"""Convert verified raw FL2VA components without touching the original snapshot.

Executes ONLY the SHA-checked Apache-2.0 Diffusers conversion tool from its
fixed clean Git checkout. No model checkpoint Python, downloads or engines.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from visual_rl.models.h3 import DIFFUSERS_SHA, inspect_assets, sha256_file
from visual_rl.models.h3_components import CONVERTER_SHA256, audit_runtime


def load_converter(source):
    source = Path(source).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True, timeout=30).strip()
    if git("rev-parse", "HEAD") != DIFFUSERS_SHA or git("status", "--porcelain"):
        raise RuntimeError("conversion requires the fixed clean Diffusers source checkout")
    script = source / "scripts/convert_minimax_h3_to_diffusers.py"
    if sha256_file(script) != CONVERTER_SHA256:
        raise RuntimeError("unreviewed conversion script hash")
    spec = importlib.util.spec_from_file_location("visualrl_reviewed_h3_converter", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(raw_root, output, revision, source):
    if any(os.environ.get(n) != "1" for n in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("conversion requires offline mode")
    audit_runtime()
    raw_root, output = Path(raw_root).resolve(strict=True), Path(output).absolute()
    if (raw_root == output or raw_root in output.parents or output in raw_root.parents
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("derived assets must be separate from raw snapshot, without symlink parents")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError("preserve prior conversion; choose a fresh derived path")
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = inspect_assets(raw_root, revision)
    # Conservative disk bound includes both newly converted components and64GiB.
    if shutil.disk_usage(output.parent).free < assets["total_bytes"] + 64 * 1024**3:
        raise RuntimeError("insufficient disk for bounded conversion and64GiB reserve")
    available = int(dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())["MemAvailable"].split()[0]) * 1024
    if available < 48 * 1024**3:
        raise RuntimeError("conversion requires48GiB host available (bounded shards +24GiB reserve)")
    converter = load_converter(source)
    staging = output.with_name(output.name + ".partial")
    staging.mkdir()
    import diffusers
    raw = str(raw_root / "FL2VA")
    try:
        # Calls the reviewed streaming converter explicitly, not a model pipeline.
        # Per-head QKV deinterleave, gate/value swaps, FP32 islands and key-plan
        # completeness checks are all implemented in this fixed upstream script.
        transformer = str(staging / "transformer")
        converter.convert_transformer(raw, transformer, converter.MINIMAX_H3_TRANSFORMER_CONFIG, 2 * 1024**3)
        converter.write_transformer_config(transformer, converter.MINIMAX_H3_TRANSFORMER_CONFIG, diffusers.__version__)
        converter.convert_video_vae(raw, str(staging / "vae"), converter.MINIMAX_H3_VIDEO_VAE_CONFIG,
                                    diffusers.__version__, 2 * 1024**3)
        converter.convert_audio_vae(raw, str(staging / "audio_vae"), diffusers.__version__)
        converter.write_scheduler_configs(raw, str(staging), diffusers.__version__)
        files = {p.relative_to(staging).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in sorted(staging.rglob("*")) if p.is_file()}
        receipt = {"schema_version": 1, "diffusers_commit": DIFFUSERS_SHA, "converter_sha256": CONVERTER_SHA256,
                   "raw_assets_sha256": assets["assets_sha256"], "raw_revision": revision, "files": files}
        (staging / "conversion.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
        os.replace(staging, output)
    except BaseException as error:
        (staging / "FAILED.json").write_text(json.dumps({"error": str(error), "error_type": type(error).__name__}) + "\n")
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--diffusers-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = convert(args.model_root, args.output, args.model_revision, args.diffusers_source)
    print(json.dumps({"status": "converted_not_generated", "files": len(receipt["files"])}), flush=True)


if __name__ == "__main__":
    main()
