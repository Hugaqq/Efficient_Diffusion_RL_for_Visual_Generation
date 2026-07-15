"""Validate heterogeneous Flash selected-index batching with the real Wan model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from remote_wan_flash_native_parity import (
    file_sha256,
    gradient_snapshot,
    tensor_bundle_sha256,
    tensor_comparison,
    write_json,
    zero_grad,
)


def tensor_difference_diagnostic(reference, actual) -> dict[str, object]:
    """Describe a finite tensor difference without turning it into a gate."""

    import torch

    result = tensor_comparison(reference, actual)
    result.update(
        {
            "dtype_actual": str(actual.dtype),
            "dtype_reference": str(reference.dtype),
        }
    )
    if reference.shape == actual.shape:
        difference = (reference.detach().float() - actual.detach().float()).abs()
        result.update(
            {
                "mean_abs": float(difference.mean())
                if difference.numel()
                else 0.0,
                "nonzero_count": int(torch.count_nonzero(difference)),
            }
        )
    return result


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
        torch.manual_seed(1717)
        torch.cuda.manual_seed_all(1717)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.use_deterministic_algorithms(True)

        adapter = WorldR1WanLegacyAdapter(
            {
                "name": "world_r1_wan_legacy",
                "model_path": str(args.model),
                "wan_backend": "flash",
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
        adapter.prepare_for_sampling()

        prompts = [
            "a vivid blue sphere slowly rotating on a white table",
            "a matte red cube slowly rotating on a black table",
        ]
        metadata = [{"source_index": 0}, {"source_index": 1}]
        selected_indices = [2, 1]
        num_steps = 20
        sample_seed = 1718

        # W7b isolates selected-index grouping from the text encoder's normal
        # BF16 scalar-vs-batch numerical variation. The scalar upstream calls
        # below consume rows from this one shared batch conditioning tensor.
        shared_prompt_embeds, shared_negative_prompt_embeds = (
            adapter._encode_prompt_embeds(
                prompts,
                negative_prompts=[""] * len(prompts),
                num_videos_per_prompt=1,
                max_sequence_length=128,
                train_cfg=False,
            )
        )
        if shared_negative_prompt_embeds is not None:
            raise RuntimeError("train_cfg=false unexpectedly produced negative embeds")
        singleton_prompt_embeds = []
        for prompt in prompts:
            scalar_prompt_embeds, scalar_negative_prompt_embeds = (
                adapter._encode_prompt_embeds(
                    [prompt],
                    negative_prompts=[""],
                    num_videos_per_prompt=1,
                    max_sequence_length=128,
                    train_cfg=False,
                )
            )
            if scalar_negative_prompt_embeds is not None:
                raise RuntimeError(
                    "scalar train_cfg=false unexpectedly produced negative embeds"
                )
            singleton_prompt_embeds.append(scalar_prompt_embeds)
        singleton_prompt_embeds = torch.cat(singleton_prompt_embeds, dim=0)
        prompt_batch_composition_diagnostic = tensor_difference_diagnostic(
            singleton_prompt_embeds,
            shared_prompt_embeds,
        )

        with legacy_repo_path(args.flash_root):
            from flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample import (
                sde_step_with_logprob,
                wan_pipeline_with_logprob,
            )

        reference_rows = []
        for position, selected_index in enumerate(selected_indices):
            prompt_embeds = shared_prompt_embeds[position : position + 1]
            negative_prompt_embeds = None
            group_seed = adapter._flash_group_seed(sample_seed, selected_index)
            generator = make_generator(adapter.device, group_seed)
            with adapter._fork_flash_group_rng(adapter.device, group_seed):
                with torch.no_grad():
                    result = adapter._call_pipeline_with_logprob(
                        wan_pipeline_with_logprob,
                        injected=False,
                        train_cfg=False,
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
                        max_sequence_length=128,
                        attention_kwargs=None,
                        kl_reward=0.0,
                        generator=generator,
                        index=selected_index,
                    )
            media, latent_list, logprob_list, kl_by_step, returned_index = result
            if int(returned_index) != selected_index:
                raise RuntimeError("scalar reference returned the wrong selected index")
            if len(latent_list) != 2 or len(logprob_list) != 1:
                raise RuntimeError("scalar reference retained the wrong transition count")
            timestep = adapter.scheduler.timesteps[selected_index].reshape(1, 1)
            old_log_probs = torch.stack(logprob_list, dim=1)
            coefficient_result = sde_step_with_logprob(
                adapter.scheduler,
                torch.zeros_like(latent_list[0]),
                timestep[:, 0],
                latent_list[0].float(),
                prev_sample=latent_list[1].float(),
                return_dt_and_std_dev_t=True,
            )
            if len(coefficient_result) != 6:
                raise RuntimeError("reference coefficient call did not return six items")
            coefficient = adapter._normalize_flash_coefficient(
                coefficient_result[5],
                batch_size=1,
            )
            reference_rows.append(
                {
                    "media": media,
                    "latents": torch.stack(latent_list, dim=1),
                    "old_log_probs": old_log_probs,
                    "kl": adapter._flash_selected_kl(
                        kl_by_step,
                        reference_index=selected_index,
                        batch_size=1,
                        like=old_log_probs,
                    ),
                    "timesteps": timestep,
                    "prompt_embeds": prompt_embeds,
                    "coefficient": coefficient,
                    "group_seed": group_seed,
                }
            )

        def concatenate(name: str):
            return torch.cat([row[name] for row in reference_rows], dim=0)

        reference = {
            name: concatenate(name)
            for name in (
                "media",
                "latents",
                "old_log_probs",
                "kl",
                "timesteps",
                "prompt_embeds",
                "coefficient",
            )
        }

        rollout = SingleStepRollout(
            {
                "name": "single_step",
                "num_steps": num_steps,
                "samples_per_prompt": 1,
                "selected_step_strategy": "cycle",
                "timestep_range": [1, 2],
                "epoch_tag": 1,
                "rectification_mode": "scheduler_formula",
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
        batch = rollout.sample(adapter, prompts, metadata)

        sample_comparisons = {
            "media": tensor_comparison(reference["media"], batch.media),
            "latents": tensor_comparison(
                reference["latents"][:, :1], batch.latents
            ),
            "next_latents": tensor_comparison(
                reference["latents"][:, 1:2], batch.next_latents
            ),
            "timesteps": tensor_comparison(
                reference["timesteps"], batch.timesteps
            ),
            "old_log_probs": tensor_comparison(
                reference["old_log_probs"], batch.old_log_probs
            ),
            "kl": tensor_comparison(reference["kl"], batch.kl),
            "prompt_embeds": tensor_comparison(
                reference["prompt_embeds"], batch.model_tensors["prompt_embeds"]
            ),
            "coefficient": tensor_comparison(
                reference["coefficient"], batch.model_tensors["coefficient"]
            ),
        }

        adapter.prepare_for_training()
        # Build both training graphs from the exact RolloutBatch tensor objects
        # before either backward pass. This isolates formula parity from CUDA
        # allocator history or state changes caused by a preceding backward.
        transformer_device, transformer_dtype = adapter._transformer_device_dtype()
        transformer_hidden_states = batch.latents[:, 0].to(
            device=transformer_device,
            dtype=transformer_dtype,
        )
        sde_current_latent = batch.latents[:, 0].to(device=transformer_device)
        timestep = batch.timesteps[:, 0].to(device=transformer_device)
        prompt_embeds_for_training = batch.model_tensors["prompt_embeds"].to(
            device=transformer_device,
            dtype=transformer_dtype,
        )
        sde_next_latent = batch.next_latents[:, 0].to(device=transformer_device)
        if (
            sde_current_latent.dtype != torch.float32
            or sde_next_latent.dtype != torch.float32
        ):
            raise RuntimeError(
                "W7b Flash SDE reference requires original FP32 current/next "
                "latents"
            )
        detached_training_inputs = {
            "latents": not batch.latents.requires_grad
            and batch.latents.grad_fn is None,
            "next_latents": not batch.next_latents.requires_grad
            and batch.next_latents.grad_fn is None,
            "timesteps": not batch.timesteps.requires_grad
            and batch.timesteps.grad_fn is None,
            "prompt_embeds": not batch.model_tensors[
                "prompt_embeds"
            ].requires_grad
            and batch.model_tensors["prompt_embeds"].grad_fn is None,
            "old_log_probs": not batch.old_log_probs.requires_grad
            and batch.old_log_probs.grad_fn is None,
            "coefficient": not batch.model_tensors["coefficient"].requires_grad
            and batch.model_tensors["coefficient"].grad_fn is None,
        }
        if not all(detached_training_inputs.values()):
            raise RuntimeError("W7b RolloutBatch training inputs must be detached")
        reference_noise_pred = adapter.transformer(
            hidden_states=transformer_hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds_for_training,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        reference_step_result = sde_step_with_logprob(
            adapter.scheduler,
            reference_noise_pred.float(),
            timestep,
            sde_current_latent,
            prev_sample=sde_next_latent,
            return_dt_and_std_dev_t=True,
        )
        if len(reference_step_result) != 6:
            raise RuntimeError("reference log-prob call did not return six items")
        reference_new_log_probs = reference_step_result[1]
        reference_new_log_probs = reference_new_log_probs[:, None]

        timestep_values = [int(value) for value in timestep.detach().cpu().tolist()]
        recomputed_coefficient = adapter._normalize_flash_coefficient(
            reference_step_result[5],
            batch_size=len(prompts),
        ).to(
            device=reference_new_log_probs.device,
            dtype=reference_new_log_probs.dtype,
        )
        if not torch.allclose(
            recomputed_coefficient,
            batch.model_tensors["coefficient"].to(
                device=recomputed_coefficient.device,
                dtype=recomputed_coefficient.dtype,
            ),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise RuntimeError("sampled and recomputed Flash coefficients disagree")
        rectification = (
            recomputed_coefficient / recomputed_coefficient.mean()
        )[:, 0]
        advantages = torch.tensor([-1.0, 1.0], device="cuda")
        ratio = torch.exp(
            reference_new_log_probs[:, 0] - batch.old_log_probs[:, 0]
        )
        clip_range = 0.001
        reference_loss = torch.maximum(
            -rectification * advantages * ratio,
            -rectification
            * advantages
            * ratio.clamp(1.0 - clip_range, 1.0 + clip_range),
        ).mean()
        infra_new_log_probs = adapter.recompute_log_probs(batch)
        algorithm = FlashGRPOAlgorithm(
            objective_version="reference_v1",
            clip_range=clip_range,
            adv_clip_max=5.0,
            beta=0.0,
        )
        infra_loss, infra_metrics = algorithm.compute_loss(
            batch,
            advantages,
            infra_new_log_probs,
        )

        zero_grad(adapter)
        reference_loss.backward()
        reference_gradients = gradient_snapshot(adapter)

        zero_grad(adapter)
        infra_loss.backward()
        infra_gradients = gradient_snapshot(adapter)

        gradient_comparisons = {}
        for (reference_name, reference_gradient), (infra_name, infra_gradient) in zip(
            reference_gradients, infra_gradients, strict=True
        ):
            if reference_name != infra_name:
                raise RuntimeError(
                    f"gradient name mismatch: {reference_name} != {infra_name}"
                )
            gradient_comparisons[reference_name] = tensor_comparison(
                reference_gradient, infra_gradient
            )

        final_hash = tensor_bundle_sha256(adapter.named_parameters())
        torch.save(
            {
                "reference": {
                    key: value.detach().cpu()
                    for key, value in reference.items()
                },
                "infra": {
                    "media": batch.media.detach().cpu(),
                    "latents": batch.latents.detach().cpu(),
                    "next_latents": batch.next_latents.detach().cpu(),
                    "old_log_probs": batch.old_log_probs.detach().cpu(),
                    "timesteps": batch.timesteps.detach().cpu(),
                    "coefficient": batch.model_tensors["coefficient"]
                    .detach()
                    .cpu(),
                },
            },
            args.output / "parity_tensors.pt",
        )

        gradients_exact = all(
            bool(item["equal"]) and bool(item["finite"])
            for item in gradient_comparisons.values()
        )
        scheduler_metadata = batch.model_metadata.get("scheduler")
        expected_group_seeds = {
            str(index): adapter._flash_group_seed(sample_seed, index)
            for index in sorted(set(selected_indices))
        }
        gates = {
            "sample_order_and_tensors_exact": all(
                bool(item["equal"]) and bool(item["finite"])
                for item in sample_comparisons.values()
            ),
            "selected_indices_exact": batch.model_metadata.get(
                "selected_timestep_indices"
            )
            == selected_indices
            and batch.model_metadata.get("actual_scheduler_timesteps")
            == timestep_values,
            "grouped_seed_contract_exact": {
                str(selected_index): row["group_seed"]
                for row, selected_index in zip(
                    reference_rows, selected_indices, strict=True
                )
            }
            == expected_group_seeds,
            "scheduler_metadata_exact": isinstance(scheduler_metadata, dict)
            and isinstance(scheduler_metadata.get("timesteps"), list)
            and [
                scheduler_metadata["timesteps"][index] for index in selected_indices
            ]
            == timestep_values,
            "flash_reference_contract": adapter.wan_backend == "flash"
            and batch.model_metadata.get("wan_backend") == "flash"
            and batch.model_metadata.get("sample_config", {}).get("train_cfg")
            is False
            and batch.model_tensors.get("negative_prompt_embeds") is None,
            "retained_one_transition": batch.old_log_probs.shape[1] == 1
            and batch.latents.shape[1] == 1
            and batch.next_latents.shape[1] == 1,
            "recomputed_logprob_exact": bool(
                torch.equal(reference_new_log_probs, infra_new_log_probs)
            ),
            "loss_exact": bool(torch.equal(reference_loss, infra_loss)),
            "all_480_gradients_exact": len(gradient_comparisons) == 480
            and gradients_exact,
            "parameters_unchanged": initial_hash == final_hash,
        }
        source_paths = {
            "harness": Path(__file__).resolve(),
            "adapter": args.source_root / "visual_rl/model_adapters/wan.py",
            "rollout": args.source_root / "visual_rl/rollout/single_step.py",
            "rectification": args.source_root
            / "visual_rl/rollout/rectification.py",
            "flash_sampler": args.flash_root
            / "flow_grpo/diffusers_patch/wan2_1_pipeline_with_logprob_sample.py",
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
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "num_steps": num_steps,
                "selected_indices": selected_indices,
                "selected_timesteps": timestep_values,
                "base_sample_seed": sample_seed,
                "group_seeds_by_selected_index": expected_group_seeds,
                "train_cfg": False,
                "objective_version": "reference_v1",
                "rectification_source": "dynamic_sde_coefficient",
                "scheduler": scheduler_metadata,
            },
            "training_input_contract": {
                "adapter_dtype": str(adapter.dtype),
                "transformer_dtype": str(transformer_dtype),
                "transformer_device": str(transformer_device),
                "detached": detached_training_inputs,
                "latent_dtype_before": str(batch.latents.dtype),
                "latent_dtype_after": str(transformer_hidden_states.dtype),
                "latent_contiguous_before": batch.latents[:, 0].is_contiguous(),
                "latent_contiguous_after": transformer_hidden_states.is_contiguous(),
                "latent_stride_before": list(batch.latents[:, 0].stride()),
                "latent_stride_after": list(transformer_hidden_states.stride()),
                "sde_current_latent_dtype_before": str(batch.latents.dtype),
                "sde_current_latent_dtype_after": str(sde_current_latent.dtype),
                "sde_next_latent_dtype_before": str(batch.next_latents.dtype),
                "sde_next_latent_dtype_after": str(sde_next_latent.dtype),
                "prompt_dtype_before": str(
                    batch.model_tensors["prompt_embeds"].dtype
                ),
                "prompt_dtype_after": str(prompt_embeds_for_training.dtype),
                "prompt_contiguous_before": batch.model_tensors[
                    "prompt_embeds"
                ].is_contiguous(),
                "prompt_contiguous_after": prompt_embeds_for_training.is_contiguous(),
            },
            "prompt_batch_composition_diagnostic": {
                **prompt_batch_composition_diagnostic,
                "gating": False,
                "batch_composition_bitwise_invariance_claimed": False,
                "scope": (
                    "singleton encoding concatenation versus one shared batch "
                    "encoding; heterogeneous grouping parity uses the shared "
                    "batch conditioning"
                ),
            },
            "source_sha256": {
                name: file_sha256(path) for name, path in source_paths.items()
            },
            "sample_comparisons": sample_comparisons,
            "new_logprob_comparison": tensor_comparison(
                reference_new_log_probs, infra_new_log_probs
            ),
            "coefficient_comparison": tensor_comparison(
                reference["coefficient"],
                batch.model_tensors["coefficient"],
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
