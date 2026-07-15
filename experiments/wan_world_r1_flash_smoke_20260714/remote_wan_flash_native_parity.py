"""Compare the real Wan Flash native sampler with the VisualRL adapter path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


FLASH_REFERENCE_RECTIFICATION = {
    999: 7.4770,
    982: 7.0414,
    963: 6.6112,
    944: 6.1867,
    922: 5.7682,
    899: 5.3559,
    874: 4.9502,
    847: 4.5513,
    817: 4.1596,
    785: 3.7754,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tensor_comparison(reference, actual) -> dict[str, object]:
    import torch

    if reference.shape != actual.shape:
        return {
            "equal": False,
            "finite": False,
            "max_abs": None,
            "shape_actual": list(actual.shape),
            "shape_reference": list(reference.shape),
        }
    reference_float = reference.detach().float()
    actual_float = actual.detach().float()
    difference = (reference_float - actual_float).abs()
    return {
        "equal": bool(torch.equal(reference.detach(), actual.detach())),
        "finite": bool(
            torch.isfinite(reference_float).all() and torch.isfinite(actual_float).all()
        ),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "shape_actual": list(actual.shape),
        "shape_reference": list(reference.shape),
    }


def gradient_snapshot(adapter) -> list[tuple[str, object]]:
    result = []
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        result.append((name, parameter.grad.detach().cpu().clone()))
    return result


def zero_grad(adapter) -> None:
    for parameter in adapter.parameters():
        parameter.grad = None


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--world-r1-root", type=Path, required=True)
    parser.add_argument("--flash-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"
    started = time.perf_counter()

    try:
        import torch

        from visual_rl.model_adapters.diffusers_common import make_generator
        from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter
        from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
        from visual_rl.rollout.single_step import SingleStepRollout
        from visual_rl.third_party.legacy import legacy_repo_path

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"expected exactly one visible GPU, got {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        torch.manual_seed(1707)
        torch.cuda.manual_seed_all(1707)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.use_deterministic_algorithms(True)

        adapter = WorldR1WanLegacyAdapter(
            {
                "name": "world_r1_wan_legacy",
                "model_path": str(args.model),
                "world_r1_root": str(args.world_r1_root),
                "flash_grpo_root": str(args.flash_root),
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
        initial_hash = tensor_bundle_sha256(adapter.named_parameters())

        prompt = "a vivid blue sphere slowly rotating on a white table"
        expanded_prompts = [prompt, prompt]
        negative_prompts = ["", ""]
        prompt_embeds, negative_prompt_embeds = adapter._encode_prompt_embeds(
            expanded_prompts,
            negative_prompts=negative_prompts,
            num_videos_per_prompt=1,
            max_sequence_length=128,
        )
        with legacy_repo_path(args.flash_root):
            from flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample import (
                sde_step_with_logprob,
                wan_pipeline_with_logprob,
            )

        selected_index = 3
        num_steps = 20
        sample_seed = 1708
        reference_generator = make_generator(adapter.device, sample_seed)
        with torch.no_grad():
            reference_result = wan_pipeline_with_logprob(
                adapter.pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=1.0,
                num_videos_per_prompt=1,
                output_type="pt",
                return_dict=False,
                num_frames=5,
                height=64,
                width=64,
                kl_reward=0.0,
                generator=reference_generator,
                index=selected_index,
                determistic=False,
            )
        (
            reference_media,
            reference_latent_list,
            reference_logprob_list,
            reference_kl_by_step,
            reference_index,
        ) = reference_result
        if int(reference_index) != selected_index:
            raise RuntimeError("reference sampler returned the wrong selected index")
        if len(reference_latent_list) != 2 or len(reference_logprob_list) != 1:
            raise RuntimeError("reference sampler did not retain exactly one transition")
        reference_latents = torch.stack(reference_latent_list, dim=1)
        reference_old_log_probs = torch.stack(reference_logprob_list, dim=1)
        reference_kl = reference_kl_by_step[selected_index].reshape(-1, 1)
        reference_timestep = adapter.scheduler.timesteps[selected_index]
        reference_timesteps = reference_timestep.reshape(1, 1).repeat(2, 1)

        rollout = SingleStepRollout(
            {
                "name": "single_step",
                "num_steps": num_steps,
                "samples_per_prompt": 2,
                "selected_step_strategy": "first",
                "timestep_range": [selected_index, selected_index],
                "rectification_mode": "flash_reference_table",
                "guidance_scale": 1.0,
                "num_videos_per_prompt": 1,
                "frames": 5,
                "height": 64,
                "width": 64,
                "max_sequence_length": 128,
                "train_cfg": False,
                "kl_reward": 0.0,
                "seed": sample_seed,
            }
        )
        batch = rollout.sample(adapter, [prompt], [{"source_index": 0}])

        sample_comparisons = {
            "media": tensor_comparison(reference_media, batch.media),
            "latents": tensor_comparison(reference_latents[:, :1], batch.latents),
            "next_latents": tensor_comparison(
                reference_latents[:, 1:2], batch.next_latents
            ),
            "timesteps": tensor_comparison(reference_timesteps, batch.timesteps),
            "old_log_probs": tensor_comparison(
                reference_old_log_probs, batch.old_log_probs
            ),
            "kl": tensor_comparison(reference_kl, batch.kl),
        }

        adapter.prepare_for_training()
        latent = batch.latents[:, 0].to(dtype=adapter.dtype)
        timestep = batch.timesteps[:, 0]
        reference_noise_pred = adapter.transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=batch.model_tensors["prompt_embeds"].to(
                dtype=adapter.dtype
            ),
            attention_kwargs=None,
            return_dict=False,
        )[0]
        _, reference_new_log_probs, *_ = sde_step_with_logprob(
            adapter.scheduler,
            reference_noise_pred.float(),
            timestep,
            latent.float(),
            prev_sample=batch.next_latents[:, 0].float(),
        )
        reference_new_log_probs = reference_new_log_probs[:, None]
        timestep_value = int(batch.timesteps[0, 0])
        if timestep_value not in FLASH_REFERENCE_RECTIFICATION:
            raise RuntimeError(
                f"selected timestep {timestep_value} is absent from the reference table"
            )
        advantages = torch.tensor([-1.0, 1.0], device="cuda")
        reference_value = torch.full(
            (2,),
            FLASH_REFERENCE_RECTIFICATION[timestep_value],
            device="cuda",
        )
        reference_weight = 1.0 / reference_value.mean()
        reference_ratio = torch.exp(
            reference_new_log_probs[:, 0] - batch.old_log_probs[:, 0]
        )
        clip_range = 0.001
        reference_loss = torch.maximum(
            -reference_weight * reference_value * advantages * reference_ratio,
            -reference_weight
            * reference_value
            * advantages
            * reference_ratio.clamp(1.0 - clip_range, 1.0 + clip_range),
        ).mean()
        zero_grad(adapter)
        reference_loss.backward()
        reference_gradients = gradient_snapshot(adapter)

        zero_grad(adapter)
        infra_new_log_probs = adapter.recompute_log_probs(batch)
        algorithm = FlashGRPOAlgorithm(
            clip_range=clip_range,
            adv_clip_max=5.0,
            beta=0.0,
            rectification={
                "enabled": True,
                "mode": "flash_reference_table",
                "normalize": True,
            },
        )
        infra_loss, infra_metrics = algorithm.compute_loss(
            batch,
            advantages,
            infra_new_log_probs,
        )
        infra_loss.backward()
        infra_gradients = gradient_snapshot(adapter)

        gradient_comparisons = {}
        for (reference_name, reference_gradient), (infra_name, infra_gradient) in zip(
            reference_gradients,
            infra_gradients,
            strict=True,
        ):
            if reference_name != infra_name:
                raise RuntimeError(
                    f"gradient name mismatch: {reference_name} != {infra_name}"
                )
            gradient_comparisons[reference_name] = tensor_comparison(
                reference_gradient,
                infra_gradient,
            )

        final_hash = tensor_bundle_sha256(adapter.named_parameters())
        single_state_bytes = (
            batch.latents[:, 0].numel() * batch.latents.element_size()
        )
        storage_contract = {
            "native_retained_state_count": 2,
            "native_retained_state_bytes": 2 * single_state_bytes,
            "full_trajectory_state_count": num_steps + 1,
            "full_trajectory_state_bytes_equivalent": (num_steps + 1)
            * single_state_bytes,
            "state_storage_reduction_ratio": (num_steps + 1) / 2,
            "scope": "retained transition tensors only; native still executes all denoising forwards",
        }
        evidence = {
            "reference": {
                "media": reference_media.detach().cpu(),
                "latents": reference_latents.detach().cpu(),
                "old_log_probs": reference_old_log_probs.detach().cpu(),
                "timesteps": reference_timesteps.detach().cpu(),
            },
            "infra": {
                "media": batch.media.detach().cpu(),
                "latents": batch.latents.detach().cpu(),
                "next_latents": batch.next_latents.detach().cpu(),
                "old_log_probs": batch.old_log_probs.detach().cpu(),
                "timesteps": batch.timesteps.detach().cpu(),
            },
        }
        torch.save(evidence, args.output / "parity_tensors.pt")

        source_paths = {
            "adapter": args.source_root / "visual_rl/model_adapters/wan.py",
            "rollout": args.source_root / "visual_rl/rollout/single_step.py",
            "rectification": args.source_root
            / "visual_rl/rollout/rectification.py",
            "flash_sampler": args.flash_root
            / "flow_grpo/diffusers_patch/wan2_1_pipeline_with_logprob_sample.py",
        }
        gradients_exact = all(
            bool(item["equal"]) and bool(item["finite"])
            for item in gradient_comparisons.values()
        )
        gates = {
            "sample_tensors_exact": all(
                bool(item["equal"]) and bool(item["finite"])
                for item in sample_comparisons.values()
            ),
            "selected_index_exact": batch.model_metadata.get(
                "native_selected_index"
            )
            == selected_index,
            "retained_one_transition": batch.old_log_probs.shape[1] == 1
            and batch.latents.shape[1] == 1
            and batch.next_latents.shape[1] == 1,
            "rectification_reference_exact": batch.model_metadata.get(
                "flash_rectification_weights"
            )
            == [[1.0], [1.0]],
            "recomputed_logprob_exact": bool(
                torch.equal(reference_new_log_probs, infra_new_log_probs)
            ),
            "loss_exact": bool(torch.equal(reference_loss, infra_loss)),
            "gradients_exact": gradients_exact,
            "parameters_unchanged": initial_hash == final_hash,
        }
        result = {
            "valid": all(gates.values()),
            "gates": gates,
            "cuda": {
                "device_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "runtime": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_steps": num_steps,
                "selected_index": selected_index,
                "selected_timestep": timestep_value,
                "sample_seed": sample_seed,
            },
            "source_sha256": {
                name: file_sha256(path) for name, path in source_paths.items()
            },
            "sample_comparisons": sample_comparisons,
            "new_logprob_comparison": tensor_comparison(
                reference_new_log_probs,
                infra_new_log_probs,
            ),
            "loss": {
                "reference": float(reference_loss.detach()),
                "infra": float(infra_loss.detach()),
                "abs_difference": abs(
                    float(reference_loss.detach()) - float(infra_loss.detach())
                ),
            },
            "gradient": {
                "tensor_count": len(gradient_comparisons),
                "reference_sha256": tensor_bundle_sha256(reference_gradients),
                "infra_sha256": tensor_bundle_sha256(infra_gradients),
                "max_abs": max(
                    float(item["max_abs"] or 0.0)
                    for item in gradient_comparisons.values()
                ),
            },
            "infra_metrics": {
                name: float(value.detach())
                if isinstance(value, torch.Tensor)
                else value
                for name, value in infra_metrics.items()
            },
            "storage_contract": storage_contract,
            "parameter_sha256_before": initial_hash,
            "parameter_sha256_after": final_hash,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(result_path, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    except Exception as exc:  # noqa: BLE001 - preserve real-model failure evidence
        result = {
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }
        write_json(result_path, result)
        print(json.dumps(result, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
