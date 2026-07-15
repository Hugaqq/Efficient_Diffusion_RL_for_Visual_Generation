"""Load a Wan PEFT checkpoint in a fresh process and compare its rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback


def tensor_bundle_sha256(named_tensors) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--world-r1-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"

    try:
        import torch

        from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

        source_result = json.loads((args.source / "result.json").read_text())
        expected = torch.load(
            args.source / "sample_evidence.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.cuda.set_device(0)
        torch.manual_seed(9999)
        torch.cuda.manual_seed_all(9999)
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
        parameter_ids_before = {
            name: id(parameter) for name, parameter in adapter.named_parameters()
        }
        adapter.load_checkpoint(str(args.source / "peft_source"))
        parameter_ids_after = {
            name: id(parameter) for name, parameter in adapter.named_parameters()
        }
        loaded_hash = tensor_bundle_sha256(adapter.named_parameters())
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
        actual = {
            "media": batch.media.detach().cpu(),
            "latents": batch.latents.detach().cpu(),
            "next_latents": batch.next_latents.detach().cpu(),
            "timesteps": batch.timesteps.detach().cpu(),
            "old_log_probs": batch.old_log_probs.detach().cpu(),
            "kl": batch.kl.detach().cpu(),
        }
        comparisons = {}
        for name in sorted(expected):
            difference = actual[name].float() - expected[name].float()
            comparisons[name] = {
                "shape_equal": list(actual[name].shape) == list(expected[name].shape),
                "exact": bool(torch.equal(actual[name], expected[name])),
                "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
            }
        gates = {
            "source_valid": bool(source_result.get("valid")),
            "trainable_hash_exact": loaded_hash
            == source_result["trainable_sha256"],
            "parameter_objects_preserved": parameter_ids_before
            == parameter_ids_after,
            "sample_exact": all(item["exact"] for item in comparisons.values()),
        }
        result = {
            "valid": all(gates.values()),
            "gates": gates,
            "loaded_trainable_sha256": loaded_hash,
            "comparisons": comparisons,
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
