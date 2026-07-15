"""Load the pinned real Wan model and run one bounded rollout on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback


def tensor_bundle_sha256(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def finite_tensor(value) -> bool:
    import torch

    return bool(torch.isfinite(value.detach().float()).all())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--world-r1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"

    try:
        import torch

        from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        torch.manual_seed(1400)
        torch.cuda.manual_seed_all(1400)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        adapter = WorldR1WanLegacyAdapter(
            {
                "name": "world_r1_wan_legacy",
                "model_path": str(args.model),
                "world_r1_root": str(args.world_r1_root),
                "device": "cuda",
                "dtype": "bfloat16",
                "local_files_only": True,
                "low_cpu_mem_usage": True,
                "train_cfg": False,
                "use_lora": True,
                "lora_rank": 16,
                "lora_alpha": 32,
                "gradient_checkpointing": True,
            }
        ).load()
        trainable = list(adapter.named_parameters())
        loaded_memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        batch = adapter.sample(
            ["a vivid red cube slowly rotating on a white table"],
            [{"source_index": 0, "target_color": "red"}],
            {
                "num_steps": 2,
                "guidance_scale": 1.0,
                "num_videos_per_prompt": 1,
                "frames": 5,
                "height": 64,
                "width": 64,
                "max_sequence_length": 128,
                "seed": 1401,
                "train_cfg": False,
                "kl_reward": 0.0,
            },
        )
        batch.validate_lightweight(strict=True)
        if not all(
            finite_tensor(value)
            for value in (
                batch.media,
                batch.latents,
                batch.next_latents,
                batch.timesteps,
                batch.old_log_probs,
                batch.kl,
            )
        ):
            raise RuntimeError("Wan rollout returned a non-finite tensor")

        checkpoint = args.output / "peft_source"
        adapter.save_pretrained(str(checkpoint))
        tensor_hash = tensor_bundle_sha256(trainable)
        evidence = {
            "media": batch.media.detach().cpu(),
            "latents": batch.latents.detach().cpu(),
            "next_latents": batch.next_latents.detach().cpu(),
            "timesteps": batch.timesteps.detach().cpu(),
            "old_log_probs": batch.old_log_probs.detach().cpu(),
            "kl": batch.kl.detach().cpu(),
        }
        torch.save(evidence, args.output / "sample_evidence.pt")
        checkpoint_files = {
            str(path.relative_to(checkpoint)): path.stat().st_size
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        }
        has_full_transformer = any(
            "transformer_state" in name or size > 1_000_000_000
            for name, size in checkpoint_files.items()
        )
        result = {
            "valid": not has_full_transformer,
            "cuda": {
                "device_name": torch.cuda.get_device_name(0),
                "device_count_visible": torch.cuda.device_count(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "model_path": str(args.model),
            "runtime_metadata": adapter.runtime_metadata(),
            "trainable_parameter_count": sum(item.numel() for _, item in trainable),
            "trainable_tensor_count": len(trainable),
            "trainable_sha256": tensor_hash,
            "loaded_memory": loaded_memory,
            "peak_after_sample": {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "batch_shapes": {
                name: list(value.shape) for name, value in evidence.items()
            },
            "checkpoint_files": checkpoint_files,
            "checkpoint_total_bytes": sum(checkpoint_files.values()),
            "has_full_transformer": has_full_transformer,
        }
        write_json(result_path, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    except Exception as exc:  # noqa: BLE001 - preserve the full remote failure
        result = {
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(result_path, result)
        print(json.dumps(result, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
