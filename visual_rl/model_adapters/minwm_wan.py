"""Stage-aware MinWM Wan adapter for reward-driven LoRA fine-tuning.

This module deliberately does not reimplement MinWM's native training stages.
It consumes a causal few-step checkpoint through a narrow backend contract and
adapts its stochastic x0 re-noising sampler to VisualRL's rollout/log-prob
contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.minwm_transition import X0RenoiseKernel


_SUPPORTED_STAGE = "dmd"
_CHECKPOINT_FILENAME = "minwm_lora.pt"
_SERIAL_SEED_ALGORITHM = "sha256_branch_seed_v1"
_TRANSITION_SELECTION_ALGORITHM = "sha256_uniform_v1"
_DECODED_MEDIA_FRAME_FORMULA = "1 + vae_temporal_stride * (latent_frames - 1)"
_SAMPLING_CONTRACT = "native_bfchw_model_dtype_flowmatch_add_noise_v2"
_CAUSAL_STATE_CONTRACT = "temporal_committed_prefix_pure_conditioning_memoized_v2"
_CONDITIONING_MEMOIZATION_CONTRACT = (
    "tensor_identity_bound_crossattn_immutable_after_init_v1"
)
_INITIAL_LATENT_CONTRACT = "native_full_draw_chunk_views_v1"


@dataclass(frozen=True)
class _ResolvedStep:
    nominal_timestep: float
    timestep: float
    sigma: float


class MinWMWanAdapter(ModelAdapter):
    """RL adapter for an already-trained causal MinWM Wan checkpoint.

    The injected backend remains the authority for model loading, prompt
    conditioning, camera-aware causal state, scheduler resolution, and decode.
    ``predict_flow_x0`` may overwrite only the uncommitted temporal scratch
    tail; the committed prefix remains pure, and cross-attention conditioning is
    memoized immutably for one tensor identity. Only ``commit_clean_context``
    may advance the temporal prefix. This adapter owns rollout recording,
    sampler log probabilities, sequential replay, and LoRA checkpointing.
    """

    name = "minwm_wan_rl"
    media_type = "video"

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        extra = dict(config.get("extra") or {})
        self.config = {
            **extra,
            **{key: value for key, value in config.items() if key != "extra"},
        }
        self.stage = str(self.config.get("stage", "dmd"))
        self.transition_kernel = str(self.config.get("transition_kernel", "x0_renoise"))
        self._validate_stage_kernel()
        if self.config.get("use_lora", True) is not True:
            raise ValueError("MinWM RL MVP requires use_lora=true")

        self.kernel = X0RenoiseKernel(
            eps=float(self.config.get("logprob_epsilon", 1e-8))
        )
        self.warp_schedule = bool(self.config.get("warp_denoising_step", True))
        self.context_timestep = self._finite_float(
            "context_timestep",
            self.config.get("context_timestep", 0.0),
        )
        self.camera_convention = str(self.config.get("camera_convention", "w2c_opencv"))
        if self.camera_convention != "w2c_opencv":
            raise ValueError("MinWM RL MVP requires camera_convention='w2c_opencv'")
        self.x0_consistency_atol = self._positive_float(
            "x0_consistency_atol",
            self.config.get("x0_consistency_atol", 1e-3),
        )
        self.x0_consistency_rtol = self._positive_float(
            "x0_consistency_rtol",
            self.config.get("x0_consistency_rtol", 1e-3),
            allow_zero=True,
        )
        self.replay_credit_assignment = str(
            self.config.get("replay_credit_assignment", "sample_one")
        )
        if self.replay_credit_assignment != "sample_one":
            raise ValueError(
                "MinWM native replay requires replay_credit_assignment='sample_one'"
            )
        self.config["replay_credit_assignment"] = self.replay_credit_assignment
        self._backend: Any | None = None
        self._train_module: Any | None = None
        self._checkpoint_identity: str | None = None
        self._minwm_commit: str | None = None
        self._scheduler_identity: str | None = None
        self._lora_identity: dict[str, Any] | None = None
        self._geometry_identity: dict[str, Any] | None = None
        self._backend_implementation: dict[str, Any] | None = None
        self._causal_state_contract: str | None = None
        self._sampling_contract: str | None = None
        self._conditioning_memoization_contract: str | None = None

    @property
    def train_module(self) -> Any:
        self._ensure_backend()
        return self._train_module

    def prepare_for_sampling(self) -> None:
        """Keep stochastic-policy rollout and replay in the same network mode."""

        self.train_module.eval()

    def prepare_for_training(self) -> None:
        """Disable dropout while retaining autograd for log-prob replay."""

        self.train_module.eval()

    def implementation_identity(self) -> dict[str, Any]:
        """Return the backend, policy, and transition identities used by replay."""

        self._ensure_backend()
        transition_source = Path(__file__).with_name("minwm_transition.py")
        if not transition_source.is_file():
            raise RuntimeError("MinWM transition kernel source is missing")
        return {
            "integration_kind": "rl_finetune_of_minwm_stage_checkpoint",
            "minwm_stage": self.stage,
            "transition_kernel": self.transition_kernel,
            "replay_credit_assignment": self.replay_credit_assignment,
            "serial_seed_algorithm": _SERIAL_SEED_ALGORITHM,
            "transition_selection_algorithm": _TRANSITION_SELECTION_ALGORITHM,
            "transition_source_sha256": hashlib.sha256(
                transition_source.read_bytes()
            ).hexdigest(),
            "checkpoint_identity": self._checkpoint_identity,
            "minwm_commit": self._minwm_commit,
            "scheduler_identity": self._scheduler_identity,
            "causal_state_contract": self._causal_state_contract,
            "sampling_contract": self._sampling_contract,
            "conditioning_memoization_contract": (
                self._conditioning_memoization_contract
            ),
            "initial_latent_contract": _INITIAL_LATENT_CONTRACT,
            "lora_identity": self._lora_identity,
            "geometry_identity": self._geometry_identity,
            "backend_implementation": self._backend_implementation,
        }

    def branch_transition_count(self, rollout_config: dict[str, Any]) -> int:
        num_chunks = int(
            rollout_config.get("num_chunks", self.config.get("num_chunks", 1))
        )
        num_steps = int(rollout_config.get("num_steps", 0))
        if num_chunks < 1 or num_steps < 2:
            raise ValueError("MinWM rollout requires num_chunks>=1 and num_steps>=2")
        return num_chunks * (num_steps - 1)

    def sample(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        self._ensure_backend()
        if not prompts or len(prompts) != len(metadata):
            raise ValueError(
                "MinWM sample requires equally sized non-empty prompts and metadata"
            )
        metadata = self._resolve_camera_metadata(metadata, rollout_config)
        max_batch_size = getattr(self._backend, "max_batch_size", None)
        if max_batch_size is None:
            return self._sample_batch(prompts, metadata, rollout_config)
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size < 1
        ):
            raise ValueError("MinWM backend max_batch_size must be a positive integer")
        if len(prompts) <= max_batch_size:
            return self._sample_batch(prompts, metadata, rollout_config)
        if max_batch_size != 1:
            raise ValueError(
                "MinWM backend max_batch_size must be a positive integer; "
                "automatic serialization currently requires max_batch_size=1"
            )
        return self._sample_serially(prompts, metadata, rollout_config)

    def _resolve_camera_metadata(
        self,
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Bind one frozen Action2V camera control to every GRPO branch.

        Per-record camera metadata remains supported for injected/test backends.
        The minimal real experiment path uses ``rollout.camera_control`` so the
        camera identity is already covered by the checkpoint config fingerprint
        without expanding the prompt dataset format.
        """

        existing = ["camera_trajectory" in item for item in metadata]
        if any(existing):
            if not all(existing):
                raise ValueError(
                    "MinWM batch cannot mix camera-aware and camera-free rows"
                )
            if rollout_config.get("camera_control") is not None:
                raise ValueError(
                    "MinWM camera control is ambiguous: provide metadata cameras "
                    "or rollout.camera_control, not both"
                )
            return [dict(item) for item in metadata]

        camera_control = rollout_config.get("camera_control")
        if not isinstance(camera_control, Mapping):
            raise ValueError(
                "MinWM Action2V requires rollout.camera_control or per-record "
                "camera_trajectory metadata"
            )
        num_chunks, chunk_size, _num_steps = self._rollout_geometry(rollout_config)
        resolver = self._backend_method("resolve_camera_control")
        camera = resolver(
            dict(camera_control),
            expected_frames=num_chunks * chunk_size,
        )
        if not isinstance(camera, Mapping):
            raise TypeError(
                "MinWM backend resolve_camera_control() must return a mapping"
            )
        canonical = json.loads(
            json.dumps(
                dict(camera),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        camera_sha256 = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if self._geometry_identity is None:
            raise RuntimeError("MinWM backend geometry identity is not initialized")
        latent_frames = num_chunks * chunk_size
        temporal_stride = int(self._geometry_identity["vae_temporal_stride"])
        decoded_frames = int(self._geometry_identity["decoded_media_frames"])
        expected_decoded_frames = 1 + temporal_stride * (latent_frames - 1)
        if decoded_frames != expected_decoded_frames:
            raise ValueError(
                "MinWM decoded/camera frame geometry does not match the VAE stride"
            )
        reward_alignment = {
            "contract": "minwm_vae_camera_alignment_v1",
            "latent_frames": latent_frames,
            "decoded_media_frames": decoded_frames,
            "vae_temporal_stride": temporal_stride,
        }
        resolved = []
        for item in metadata:
            copied = dict(item)
            copied["camera_trajectory"] = canonical
            copied["camera_control_sha256"] = camera_sha256
            copied["minwm_reward_frame_alignment"] = reward_alignment
            resolved.append(copied)
        return resolved

    def _sample_batch(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        import torch

        self._ensure_backend()
        if not prompts or len(prompts) != len(metadata):
            raise ValueError(
                "MinWM sample requires equally sized non-empty prompts and metadata"
            )
        runtime_kernel = rollout_config.get("transition_kernel")
        if runtime_kernel is not None and str(runtime_kernel) != self.transition_kernel:
            raise ValueError(
                "rollout transition_kernel must match the MinWM model adapter"
            )
        guidance_scale = self._finite_float(
            "MinWM guidance_scale",
            rollout_config.get("guidance_scale", 1.0),
        )
        if guidance_scale != 1.0:
            raise ValueError(
                "MinWM RL adapter does not implement classifier-free guidance; "
                "guidance_scale must be 1.0"
            )
        num_chunks, chunk_size, num_steps = self._rollout_geometry(rollout_config)
        schedule = self._resolved_schedule(rollout_config, num_steps=num_steps)
        conditioning_identities = self._conditioning_identities(prompts, metadata)
        rollout_metadata = []
        for row_index, (item, identity) in enumerate(
            zip(metadata, conditioning_identities, strict=True)
        ):
            copied = dict(item)
            copied["minwm_conditioning_sha256"] = identity
            copied.setdefault("minwm_seed_derivation_index", row_index)
            rollout_metadata.append(copied)
        device, dtype = self._parameter_device_dtype()
        viewmats, Ks = self._camera_tensors(
            metadata,
            expected_frames=num_chunks * chunk_size,
            device=device,
        )

        seed = int(rollout_config.get("seed", self.config.get("seed", 0)))
        rollout_seeds = []
        for item in rollout_metadata:
            item_seed = item.get("minwm_rollout_seed", seed)
            if isinstance(item_seed, bool) or not isinstance(item_seed, int):
                raise ValueError("MinWM rollout seed must be an integer")
            item["minwm_rollout_seed"] = item_seed
            rollout_seeds.append(item_seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        batch_size = len(prompts)
        latent_frames = num_chunks * chunk_size

        self.prepare_for_sampling()
        with torch.no_grad():
            condition = self._encode_condition(prompts)
            self._validate_condition(condition, batch_size)
            state = self._reset_causal_state(
                batch_size=batch_size,
                total_frames=latent_frames,
                dtype=dtype,
                device=device,
            )
            recorded_x_t: list[Any] = []
            recorded_x_next: list[Any] = []
            recorded_timestep: list[Any] = []
            recorded_sigma_t: list[Any] = []
            recorded_sigma_next: list[Any] = []
            recorded_log_prob: list[Any] = []
            final_clean_chunks: list[Any] = []
            latent_shape: tuple[int, ...] | None = None

            for chunk_id in range(num_chunks):
                frame_start = chunk_id * chunk_size
                chunk_viewmats = viewmats[:, frame_start : frame_start + chunk_size]
                chunk_Ks = Ks[:, frame_start : frame_start + chunk_size]
                x_t = self._sample_initial_latent(
                    state,
                    batch_size=batch_size,
                    chunk_id=chunk_id,
                    frame_start=frame_start,
                    chunk_size=chunk_size,
                    condition=condition,
                    viewmats=chunk_viewmats,
                    Ks=chunk_Ks,
                    generator=generator,
                )
                x_t = self._validated_latent("initial latent", x_t, batch_size).detach()
                if latent_shape is None:
                    latent_shape = tuple(x_t.shape[1:])
                elif tuple(x_t.shape[1:]) != latent_shape:
                    raise ValueError("all MinWM chunks must use the same latent shape")
                final_clean = None
                for step_id, resolved in enumerate(schedule):
                    timestep = torch.full(
                        (batch_size,),
                        resolved.timestep,
                        dtype=torch.float32,
                        device=device,
                    )
                    flow_pred, backend_x0 = self._predict_flow_x0(
                        state,
                        x_t,
                        condition,
                        chunk_viewmats,
                        chunk_Ks,
                        timestep=timestep,
                        frame_start=frame_start,
                        chunk_id=chunk_id,
                        step_id=step_id,
                    )
                    flow_pred = self._validated_prediction("flow_pred", flow_pred, x_t)
                    derived_x0 = self.kernel.x0_from_flow(
                        x_t,
                        flow_pred,
                        resolved.sigma,
                    )
                    self._validate_backend_x0(backend_x0, derived_x0)

                    if step_id == num_steps - 1:
                        # The official wrapper returns its clean prediction in
                        # model dtype after the native conversion. Preserve that
                        # value for context commit and VAE decode; the fp32
                        # derivation above is a consistency check, not a second
                        # sampler implementation.
                        final_clean = backend_x0.detach()
                        break

                    sigma_next = schedule[step_id + 1].sigma
                    observed_next = self._sample_x0_renoise(
                        backend_x0,
                        sigma_next=sigma_next,
                        generator=generator,
                    )
                    observed_next = self._validated_prediction(
                        "native re-noised latent",
                        observed_next,
                        x_t,
                    )
                    transition = self.kernel.from_observation(
                        backend_x0,
                        sigma_next,
                        observed_next,
                    )
                    recorded_x_t.append(x_t.detach())
                    recorded_x_next.append(transition.next_latent.detach())
                    recorded_timestep.append(timestep.detach())
                    recorded_sigma_t.append(torch.full_like(timestep, resolved.sigma))
                    recorded_sigma_next.append(torch.full_like(timestep, sigma_next))
                    recorded_log_prob.append(transition.log_prob.detach())
                    x_t = transition.next_latent.detach()

                if final_clean is None:
                    raise RuntimeError(
                        "MinWM terminal step did not produce a clean latent"
                    )
                final_clean_chunks.append(final_clean)
                self._commit_clean_context(
                    state,
                    final_clean,
                    condition,
                    chunk_viewmats,
                    chunk_Ks,
                    frame_start=frame_start,
                )

            final_clean_tensor = torch.stack(final_clean_chunks, dim=1)
            media = self._decode(
                final_clean_tensor,
                condition=condition,
                viewmats=viewmats,
                Ks=Ks,
            )
            media = self._validated_media(
                media,
                batch_size,
                expected_decoded_frames=self._expected_decoded_media_frames(
                    latent_frames
                ),
            )

        latents = torch.stack(recorded_x_t, dim=1)
        next_latents = torch.stack(recorded_x_next, dim=1)
        timesteps = torch.stack(recorded_timestep, dim=1)
        sigma_t = torch.stack(recorded_sigma_t, dim=1)
        sigma_next = torch.stack(recorded_sigma_next, dim=1)
        old_log_probs = torch.stack(recorded_log_prob, dim=1).float()
        expected_transitions = num_chunks * (num_steps - 1)
        if old_log_probs.shape != (batch_size, expected_transitions):
            raise RuntimeError("MinWM terminal transition accounting is inconsistent")
        transition_mask, selected_transition_indices = self._credit_mask(
            old_log_probs,
            rollout_seeds=rollout_seeds,
            metadata=rollout_metadata,
        )
        for item, selected_index in zip(
            rollout_metadata,
            selected_transition_indices,
            strict=True,
        ):
            item["minwm_selected_transition_index"] = selected_index

        return RolloutBatch(
            prompts=list(prompts),
            metadata=rollout_metadata,
            media=media,
            media_layout="BFCHW",
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=torch.zeros_like(old_log_probs),
            transition_mask=transition_mask,
            model_tensors={
                "sigma_t": sigma_t,
                "sigma_next": sigma_next,
                "viewmats": viewmats.detach(),
                "Ks": Ks.detach(),
                "final_clean_chunks": final_clean_tensor.detach(),
                "prompt_embeddings": self._detach_tree(condition),
            },
            model_metadata=self._model_metadata(
                schedule,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                conditioning_identities=conditioning_identities,
                selected_transition_indices=selected_transition_indices,
                rollout_seeds=rollout_seeds,
            ),
        )

    def _sample_serially(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        """Run branch rollouts one at a time for the bounded native backend."""

        import torch

        base_seed = int(rollout_config.get("seed", self.config.get("seed", 0)))
        batches = []
        for row_index, (prompt, item) in enumerate(zip(prompts, metadata, strict=True)):
            row_metadata = dict(item)
            row_metadata["minwm_seed_derivation_index"] = row_index
            row_seed = self._serial_rollout_seed(
                base_seed=base_seed,
                prompt=prompt,
                metadata=row_metadata,
            )
            row_metadata["minwm_rollout_seed"] = row_seed
            row_config = dict(rollout_config)
            row_config["seed"] = row_seed
            batches.append(self._sample_batch([prompt], [row_metadata], row_config))

        batch_metadata_fields = {
            "conditioning_identity_sha256",
            "selected_transition_indices",
            "rollout_seeds",
        }
        model_metadata = {
            key: value
            for key, value in batches[0].model_metadata.items()
            if key not in batch_metadata_fields
        }
        for batch in batches[1:]:
            candidate = {
                key: value
                for key, value in batch.model_metadata.items()
                if key not in batch_metadata_fields
            }
            if candidate != model_metadata:
                raise RuntimeError("serialized MinWM rollouts changed policy metadata")
        for key in batch_metadata_fields:
            model_metadata[key] = [
                value for batch in batches for value in batch.model_metadata[key]
            ]

        return RolloutBatch(
            prompts=[prompt for batch in batches for prompt in batch.prompts],
            metadata=[item for batch in batches for item in batch.metadata],
            media=torch.cat([batch.media for batch in batches], dim=0),
            media_layout="BFCHW",
            latents=torch.cat([batch.latents for batch in batches], dim=0),
            next_latents=torch.cat([batch.next_latents for batch in batches], dim=0),
            timesteps=torch.cat([batch.timesteps for batch in batches], dim=0),
            old_log_probs=torch.cat([batch.old_log_probs for batch in batches], dim=0),
            kl=torch.cat([batch.kl for batch in batches], dim=0),
            transition_mask=torch.cat(
                [batch.transition_mask for batch in batches], dim=0
            ),
            model_tensors=self._concatenate_tensor_trees(
                [batch.model_tensors for batch in batches]
            ),
            model_metadata=model_metadata,
        )

    def _credit_mask(
        self,
        old_log_probs: Any,
        *,
        rollout_seeds: list[int],
        metadata: list[dict[str, Any]],
    ) -> tuple[Any, list[int]]:
        import torch

        if len(rollout_seeds) != int(old_log_probs.shape[0]) or len(metadata) != int(
            old_log_probs.shape[0]
        ):
            raise ValueError("MinWM credit metadata must match rollout batch size")
        mask = torch.zeros_like(old_log_probs, dtype=torch.bool)
        selected = []
        transition_count = int(old_log_probs.shape[1])
        for row, (rollout_seed, item) in enumerate(
            zip(rollout_seeds, metadata, strict=True)
        ):
            payload = {
                "algorithm": _TRANSITION_SELECTION_ALGORITHM,
                "rollout_seed": rollout_seed,
                "conditioning_sha256": item.get("minwm_conditioning_sha256"),
                "group_id": item.get("group_id"),
                "prompt_id": item.get("prompt_id"),
                "parent_prompt_index": item.get("parent_prompt_index"),
                "sample_index": item.get("sample_index"),
                "branch_id": item.get("branch_id"),
                "seed_derivation_index": item.get("minwm_seed_derivation_index"),
            }
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).digest()
            index = int.from_bytes(digest[:8], "big") % transition_count
            mask[row, index] = True
            selected.append(index)
        return mask, selected

    @staticmethod
    def _serial_rollout_seed(
        *,
        base_seed: int,
        prompt: str,
        metadata: dict[str, Any],
    ) -> int:
        payload = {
            "algorithm": _SERIAL_SEED_ALGORITHM,
            "base_seed": base_seed,
            "prompt": prompt,
            "camera_trajectory": metadata.get("camera_trajectory"),
            "group_id": metadata.get("group_id"),
            "prompt_id": metadata.get("prompt_id"),
            "parent_prompt_index": metadata.get("parent_prompt_index"),
            "sample_index": metadata.get("sample_index"),
            "branch_id": metadata.get("branch_id"),
            "seed_derivation_index": metadata.get("minwm_seed_derivation_index"),
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)

    @classmethod
    def _concatenate_tensor_trees(cls, values: list[Any]) -> Any:
        import torch

        first = values[0]
        if all(isinstance(value, torch.Tensor) for value in values):
            return torch.cat(values, dim=0)
        if all(isinstance(value, Mapping) for value in values):
            keys = set(first)
            if any(set(value) != keys for value in values[1:]):
                raise RuntimeError("serialized MinWM tensor trees changed keys")
            return type(first)(
                (
                    key,
                    cls._concatenate_tensor_trees([value[key] for value in values]),
                )
                for key in first
            )
        raise TypeError("serialized MinWM model tensors must be tensors or mappings")

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        import torch

        self._ensure_backend()
        batch.validate_lightweight(strict=True)
        self._validate_replay_identity(batch)
        tensors = batch.model_tensors
        required = {
            "sigma_t",
            "sigma_next",
            "viewmats",
            "Ks",
            "final_clean_chunks",
            "prompt_embeddings",
        }
        missing = sorted(required.difference(tensors))
        if missing:
            raise ValueError(f"MinWM replay tensors missing: {', '.join(missing)}")

        num_chunks = int(batch.model_metadata["num_chunks"])
        chunk_size = int(batch.model_metadata["chunk_size"])
        num_steps = int(batch.model_metadata["num_steps"])
        expected_transitions = num_chunks * (num_steps - 1)
        if batch.old_log_probs.shape != (batch.batch_size, expected_transitions):
            raise ValueError(
                "MinWM replay transition count does not match chunk geometry"
            )

        device, dtype = self._parameter_device_dtype()
        viewmats = tensors["viewmats"]
        Ks = tensors["Ks"]
        condition = tensors["prompt_embeddings"]
        final_clean_chunks = tensors["final_clean_chunks"]
        self._validate_condition(condition, batch.batch_size)
        expected_viewmats, expected_Ks = self._camera_tensors(
            batch.metadata,
            expected_frames=num_chunks * chunk_size,
            device=viewmats.device,
        )
        if not torch.equal(viewmats.float(), expected_viewmats) or not torch.equal(
            Ks.float(), expected_Ks
        ):
            raise ValueError(
                "saved MinWM viewmats/Ks do not match rollout camera metadata"
            )
        if tuple(final_clean_chunks.shape[:2]) != (batch.batch_size, num_chunks):
            raise ValueError(
                "final_clean_chunks must have shape [batch, num_chunks, ...]"
            )
        return self._recompute_sampled_transition(
            batch,
            viewmats=viewmats,
            Ks=Ks,
            condition=condition,
            final_clean_chunks=final_clean_chunks,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
        )

    def _recompute_sampled_transition(
        self,
        batch: RolloutBatch,
        *,
        viewmats: Any,
        Ks: Any,
        condition: Any,
        final_clean_chunks: Any,
        num_chunks: int,
        chunk_size: int,
        num_steps: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        """Replay one stochastic transition with a truncated recurrent graph.

        The native backend keeps one large mutable causal cache. Prefix chunks
        are therefore committed before the differentiable prediction, while
        the selected and later chunks are left uncommitted until backward has
        consumed the graph.
        """

        import torch

        if batch.batch_size != 1:
            raise ValueError(
                "MinWM sample_one replay requires update_microbatch_size=1"
            )
        selected_values = batch.model_metadata.get("selected_transition_indices")
        if not isinstance(selected_values, list) or len(selected_values) != 1:
            raise ValueError("MinWM replay selected transition identity is invalid")
        selected_index = selected_values[0]
        if isinstance(selected_index, bool) or not isinstance(selected_index, int):
            raise ValueError("MinWM selected transition index must be an integer")

        transitions_per_chunk = num_steps - 1
        expected_transitions = num_chunks * transitions_per_chunk
        if not 0 <= selected_index < expected_transitions:
            raise ValueError("MinWM selected transition index is out of range")
        selected_chunk, selected_step = divmod(
            selected_index,
            transitions_per_chunk,
        )
        tensors = batch.model_tensors
        state = self._reset_causal_state(
            batch_size=1,
            total_frames=num_chunks * chunk_size,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            for chunk_id in range(selected_chunk):
                frame_start = chunk_id * chunk_size
                self._commit_clean_context(
                    state,
                    final_clean_chunks[:, chunk_id].detach(),
                    condition,
                    viewmats[:, frame_start : frame_start + chunk_size],
                    Ks[:, frame_start : frame_start + chunk_size],
                    frame_start=frame_start,
                )

        frame_start = selected_chunk * chunk_size
        x_t = batch.latents[:, selected_index].detach()
        x_next = batch.next_latents[:, selected_index].detach()
        timestep = batch.timesteps[:, selected_index]
        flow_pred, backend_x0 = self._predict_flow_x0(
            state,
            x_t,
            condition,
            viewmats[:, frame_start : frame_start + chunk_size],
            Ks[:, frame_start : frame_start + chunk_size],
            timestep=timestep,
            frame_start=frame_start,
            chunk_id=selected_chunk,
            step_id=selected_step,
        )
        flow_pred = self._validated_prediction("flow_pred", flow_pred, x_t)
        sigma_t = tensors["sigma_t"][:, selected_index]
        sigma_next = tensors["sigma_next"][:, selected_index]
        derived_x0 = self.kernel.x0_from_flow(x_t, flow_pred, sigma_t)
        self._validate_backend_x0(backend_x0, derived_x0)
        selected_log_prob = self.kernel.from_observation(
            backend_x0,
            sigma_next,
            x_next,
        ).log_prob.float()
        result = (
            batch.old_log_probs.detach()
            .to(
                device=selected_log_prob.device,
                dtype=torch.float32,
            )
            .clone()
        )
        result[:, selected_index] = selected_log_prob
        return result

    def save_pretrained(self, output_dir: str) -> None:
        import torch

        self._ensure_backend()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        trainable = {
            name: parameter.detach().cpu()
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        }
        torch.save(
            {
                "format_version": 2,
                "adapter": self.name,
                "stage": self.stage,
                "transition_kernel": self.transition_kernel,
                "checkpoint_identity": self._checkpoint_identity,
                "minwm_commit": self._minwm_commit,
                "scheduler_identity": self._scheduler_identity,
                "causal_state_contract": self._causal_state_contract,
                "sampling_contract": self._sampling_contract,
                "conditioning_memoization_contract": (
                    self._conditioning_memoization_contract
                ),
                "initial_latent_contract": _INITIAL_LATENT_CONTRACT,
                "lora_identity": self._lora_identity,
                "geometry_identity": self._geometry_identity,
                "backend_implementation": self._backend_implementation,
                "policy_contract_sha256": self._policy_contract_sha256(),
                "trainable_state": trainable,
            },
            path / _CHECKPOINT_FILENAME,
        )

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        import torch

        self._ensure_backend()
        payload = torch.load(
            Path(checkpoint_dir) / _CHECKPOINT_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("adapter") != self.name:
            raise ValueError("checkpoint is not a MinWM Wan RL adapter checkpoint")
        if payload.get("format_version") != 2:
            raise ValueError("MinWM checkpoint format_version is unsupported")
        for key, expected in (
            ("stage", self.stage),
            ("transition_kernel", self.transition_kernel),
            ("checkpoint_identity", self._checkpoint_identity),
            ("minwm_commit", self._minwm_commit),
            ("scheduler_identity", self._scheduler_identity),
            ("causal_state_contract", self._causal_state_contract),
            ("sampling_contract", self._sampling_contract),
            (
                "conditioning_memoization_contract",
                self._conditioning_memoization_contract,
            ),
            ("initial_latent_contract", _INITIAL_LATENT_CONTRACT),
            ("lora_identity", self._lora_identity),
            ("geometry_identity", self._geometry_identity),
            ("backend_implementation", self._backend_implementation),
            ("policy_contract_sha256", self._policy_contract_sha256()),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"MinWM checkpoint {key} does not match the adapter")
        saved = payload.get("trainable_state")
        if not isinstance(saved, dict):
            raise ValueError("MinWM checkpoint is missing trainable_state")
        current = {
            name: parameter
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        }
        if set(saved) != set(current):
            raise ValueError("MinWM checkpoint LoRA parameter names do not match")
        with torch.no_grad():
            for name, parameter in current.items():
                value = saved[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"MinWM checkpoint shape mismatch for {name}")
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        backend = self.config.get("backend")
        if backend is None:
            factory = self.config.get("backend_factory")
            if factory is None:
                raise RuntimeError(
                    "MinWM adapter requires model.extra.backend_factory='module:factory' "
                    "or an injected backend"
                )
            if isinstance(factory, str):
                factory = self._import_factory(factory)
            if not callable(factory):
                raise TypeError(
                    "MinWM backend_factory must be callable or 'module:factory'"
                )
            backend = factory(dict(self.config))
        load = getattr(backend, "load", None)
        if load is not None:
            if not callable(load):
                raise TypeError("MinWM backend load must be callable")
            load()
        train_module = getattr(backend, "train_module", None)
        if callable(train_module) and not hasattr(train_module, "parameters"):
            train_module = train_module()
        if train_module is None or not callable(
            getattr(train_module, "parameters", None)
        ):
            raise TypeError("MinWM backend must expose an nn.Module train_module")
        if getattr(backend, "temporal_committed_prefix_pure", None) is not True:
            raise ValueError(
                "MinWM backend must declare temporal_committed_prefix_pure=true; "
                "predict_flow_x0 may overwrite only the uncommitted temporal tail"
            )
        if getattr(backend, "media_layout", None) != "BFCHW":
            raise ValueError("MinWM backend must declare media_layout='BFCHW'")
        trainable = [
            (name, parameter)
            for name, parameter in train_module.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError(
                "MinWM backend must expose at least one trainable LoRA parameter"
            )
        non_lora = [name for name, _ in trainable if "lora" not in name.casefold()]
        if non_lora:
            raise ValueError(
                "MinWM RL MVP permits LoRA-only trainable parameters; found "
                + ", ".join(non_lora)
            )
        non_fp32 = [
            name
            for name, parameter in trainable
            if str(parameter.dtype) != "torch.float32"
        ]
        if non_fp32:
            raise ValueError(
                "MinWM trainable LoRA parameters must remain float32 so small "
                "optimizer updates are not lost to low-precision rounding; found "
                + ", ".join(non_fp32)
            )
        self._backend = backend
        self._train_module = train_module
        self._checkpoint_identity = self._identity_value(
            "checkpoint_identity",
            getattr(backend, "checkpoint_identity", None),
        )
        self._minwm_commit = self._identity_value(
            "minwm_commit",
            getattr(backend, "minwm_commit", None),
        )
        self._scheduler_identity = self._identity_value(
            "scheduler_identity",
            getattr(backend, "scheduler_identity", None),
        )
        self._causal_state_contract = self._identity_value(
            "causal_state_contract",
            getattr(backend, "causal_state_contract", None),
        )
        if self._causal_state_contract != _CAUSAL_STATE_CONTRACT:
            raise ValueError(
                "MinWM backend causal_state_contract must be "
                f"{_CAUSAL_STATE_CONTRACT!r}"
            )
        self._sampling_contract = self._identity_value(
            "sampling_contract",
            getattr(backend, "sampling_contract", None),
        )
        if self._sampling_contract != _SAMPLING_CONTRACT:
            raise ValueError(
                f"MinWM backend sampling_contract must be {_SAMPLING_CONTRACT!r}"
            )
        self._conditioning_memoization_contract = self._identity_value(
            "conditioning_memoization_contract",
            getattr(backend, "conditioning_memoization_contract", None),
        )
        if (
            self._conditioning_memoization_contract
            != _CONDITIONING_MEMOIZATION_CONTRACT
        ):
            raise ValueError(
                "MinWM backend conditioning_memoization_contract must be "
                f"{_CONDITIONING_MEMOIZATION_CONTRACT!r}"
            )
        if not callable(getattr(backend, "sample_x0_renoise", None)):
            raise TypeError("MinWM backend must define callable sample_x0_renoise()")
        self._lora_identity = self._resolve_lora_identity(backend, trainable)
        self._geometry_identity = self._resolve_geometry_identity(backend)
        self._backend_implementation = self._backend_implementation_identity(backend)

    def _validate_stage_kernel(self) -> None:
        if self.stage != _SUPPORTED_STAGE:
            raise ValueError("MinWM RL currently supports only stage='dmd'")
        if self.transition_kernel != "x0_renoise":
            raise ValueError("MinWM DMD requires transition_kernel='x0_renoise'")

    def _rollout_geometry(self, rollout_config: dict[str, Any]) -> tuple[int, int, int]:
        num_chunks = int(
            rollout_config.get("num_chunks", self.config.get("num_chunks", 1))
        )
        chunk_size = int(
            rollout_config.get("chunk_size", self.config.get("chunk_size", 1))
        )
        num_steps = int(rollout_config.get("num_steps", 0))
        if num_chunks < 1:
            raise ValueError("MinWM num_chunks must be positive")
        if chunk_size < 1:
            raise ValueError("MinWM chunk_size must be positive")
        if self._geometry_identity is None:
            raise RuntimeError("MinWM backend geometry identity is not initialized")
        if chunk_size != self._geometry_identity["chunk_size"]:
            raise ValueError(
                "MinWM rollout chunk_size does not match the backend checkpoint geometry"
            )
        backend_latent_frames = self._geometry_identity.get("latent_frames")
        if (
            backend_latent_frames is not None
            and num_chunks * chunk_size != backend_latent_frames
        ):
            raise ValueError(
                "MinWM rollout latent frame count does not match the backend "
                "checkpoint geometry"
            )
        if num_steps < 2:
            raise ValueError("MinWM num_steps must be at least 2")
        return num_chunks, chunk_size, num_steps

    def _resolved_schedule(
        self,
        rollout_config: dict[str, Any],
        *,
        num_steps: int,
    ) -> tuple[_ResolvedStep, ...]:
        nominal = rollout_config.get(
            "denoising_timesteps",
            self.config.get("denoising_timesteps"),
        )
        if isinstance(nominal, (str, bytes)) or not isinstance(nominal, Sequence):
            raise ValueError("MinWM denoising_timesteps must be an explicit sequence")
        nominal_values = [
            self._finite_float("nominal timestep", value) for value in nominal
        ]
        if len(nominal_values) != num_steps:
            raise ValueError(
                "num_steps must match the explicit denoising_timesteps length"
            )
        resolver = self._backend_method("resolve_schedule")
        raw = resolver(
            nominal_values,
            stage=self.stage,
            warp=self.warp_schedule,
        )
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("MinWM backend resolve_schedule must return a sequence")
        if len(raw) != num_steps:
            raise ValueError("resolved MinWM schedule length does not match num_steps")
        resolved: list[_ResolvedStep] = []
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                timestep = item.get("timestep")
                sigma = item.get("sigma")
                item_nominal = item.get("nominal_timestep", nominal_values[index])
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) != 2:
                    raise ValueError(
                        "resolved schedule tuple must be (timestep, sigma)"
                    )
                timestep, sigma = item
                item_nominal = nominal_values[index]
            else:
                timestep = getattr(item, "timestep", None)
                sigma = getattr(item, "sigma", None)
                item_nominal = getattr(item, "nominal_timestep", nominal_values[index])
            resolved_step = _ResolvedStep(
                nominal_timestep=self._finite_float("nominal timestep", item_nominal),
                timestep=self._finite_float("resolved timestep", timestep),
                sigma=self._finite_float("resolved sigma", sigma),
            )
            if resolved_step.sigma < 0:
                raise ValueError("resolved MinWM sigma must be non-negative")
            if resolved_step.nominal_timestep != nominal_values[index]:
                raise ValueError(
                    "MinWM backend changed the requested nominal timestep identity"
                )
            resolved.append(resolved_step)
        for current, following in zip(resolved, resolved[1:]):
            if not 0 < following.sigma < current.sigma:
                raise ValueError(
                    "resolved MinWM sigma must be strictly decreasing with "
                    "0 < sigma_next < sigma_t"
                )
        return tuple(resolved)

    def _camera_tensors(
        self,
        metadata: list[dict[str, Any]],
        *,
        expected_frames: int,
        device: Any,
    ) -> tuple[Any, Any]:
        import torch

        all_viewmats = []
        all_Ks = []
        for index, item in enumerate(metadata):
            camera = item.get("camera_trajectory")
            if not isinstance(camera, Mapping):
                raise ValueError(
                    f"metadata[{index}].camera_trajectory must be a mapping"
                )
            convention = str(camera.get("convention", "")).casefold()
            coordinate_system = str(camera.get("coordinate_system", "")).casefold()
            if convention not in {"w2c", "world_to_camera"}:
                raise ValueError("MinWM camera convention must be w2c/world_to_camera")
            if coordinate_system != "opencv":
                raise ValueError("MinWM camera coordinate_system must be opencv")
            viewmats = torch.as_tensor(camera.get("viewmats"), dtype=torch.float32)
            Ks = torch.as_tensor(camera.get("Ks"), dtype=torch.float32)
            if tuple(viewmats.shape) != (expected_frames, 4, 4):
                raise ValueError(
                    "camera viewmats must have shape "
                    f"({expected_frames}, 4, 4), got {tuple(viewmats.shape)}"
                )
            if tuple(Ks.shape) != (expected_frames, 3, 3):
                raise ValueError(
                    f"camera Ks must have shape ({expected_frames}, 3, 3), "
                    f"got {tuple(Ks.shape)}"
                )
            if not bool(torch.isfinite(viewmats).all().item()) or not bool(
                torch.isfinite(Ks).all().item()
            ):
                raise ValueError("MinWM camera tensors must be finite")
            all_viewmats.append(viewmats)
            all_Ks.append(Ks)
        return (
            torch.stack(all_viewmats).to(device=device),
            torch.stack(all_Ks).to(device=device),
        )

    def _conditioning_identities(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> list[str]:
        identities: dict[str, str] = {}
        digests: list[str] = []
        for prompt, item in zip(prompts, metadata, strict=True):
            group_id = item.get("group_id")
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(
                    "MinWM metadata requires an explicit non-empty group_id"
                )
            camera = item.get("camera_trajectory")
            payload = json.dumps(
                {"prompt": prompt, "camera_trajectory": camera},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            previous = identities.setdefault(group_id, payload)
            if previous != payload:
                raise ValueError(
                    "all samples in a MinWM GRPO group must share prompt and camera"
                )
            digests.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
        return digests

    def _encode_condition(self, prompts: list[str]) -> Any:
        return self._backend_method("encode_condition")(list(prompts))

    def _reset_causal_state(self, **kwargs: Any) -> Any:
        return self._backend_method("reset_causal_state")(**kwargs)

    def _sample_initial_latent(self, state: Any, **kwargs: Any) -> Any:
        return self._backend_method("sample_initial_latent")(state, **kwargs)

    def _predict_flow_x0(
        self,
        state: Any,
        x_t: Any,
        condition: Any,
        viewmats: Any,
        Ks: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        result = self._backend_method("predict_flow_x0")(
            state,
            x_t,
            condition,
            viewmats,
            Ks,
            **kwargs,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "MinWM backend predict_flow_x0 must return (flow_pred, x0_pred)"
            )
        return result

    def _commit_clean_context(
        self,
        state: Any,
        final_clean: Any,
        condition: Any,
        viewmats: Any,
        Ks: Any,
        *,
        frame_start: int,
    ) -> None:
        self._backend_method("commit_clean_context")(
            state,
            final_clean.detach(),
            condition,
            viewmats,
            Ks,
            frame_start=frame_start,
            context_timestep=self.context_timestep,
        )

    def _decode(self, final_clean_chunks: Any, **kwargs: Any) -> Any:
        return self._backend_method("decode")(final_clean_chunks, **kwargs)

    def _sample_x0_renoise(
        self,
        x0_pred: Any,
        *,
        sigma_next: Any,
        generator: Any,
    ) -> Any:
        return self._backend_method("sample_x0_renoise")(
            x0_pred,
            sigma_next=sigma_next,
            generator=generator,
        )

    def _validate_condition(self, value: Any, batch_size: int) -> None:
        import torch

        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or int(value.shape[0]) != batch_size:
                raise ValueError(
                    "prompt embeddings must have a leading batch dimension"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("prompt embeddings must be finite")
            return
        if isinstance(value, Mapping):
            if not value:
                raise ValueError("prompt embedding mapping must not be empty")
            for item in value.values():
                self._validate_condition(item, batch_size)
            return
        raise TypeError("prompt embeddings must be a tensor or mapping of tensors")

    @staticmethod
    def _validated_latent(name: str, value: Any, batch_size: int) -> Any:
        import torch

        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
        if value.ndim < 2 or int(value.shape[0]) != batch_size:
            raise ValueError(f"{name} must have a leading batch dimension")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _validated_prediction(cls, name: str, value: Any, reference: Any) -> Any:
        value = cls._validated_latent(name, value, int(reference.shape[0]))
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"{name} must have the same shape as x_t")
        return value

    def _validate_backend_x0(self, value: Any, derived: Any) -> None:
        import torch

        value = self._validated_prediction("x0_pred", value, derived)
        if value.dtype in {torch.bfloat16, torch.float16}:
            rounded = derived.detach().to(dtype=value.dtype)
            lower = torch.nextafter(
                rounded,
                torch.full_like(rounded, float("-inf")),
            )
            upper = torch.nextafter(
                rounded,
                torch.full_like(rounded, float("inf")),
            )
            native = value.detach()
            matches = bool(((native >= lower) & (native <= upper)).all().item())
        else:
            matches = torch.allclose(
                value.detach().float(),
                derived.detach().float(),
                rtol=self.x0_consistency_rtol,
                atol=self.x0_consistency_atol,
            )
        if not matches:
            raise ValueError(
                "MinWM backend x0_pred is inconsistent with x_t - sigma_t * flow_pred"
            )

    @staticmethod
    def _validated_media(
        value: Any,
        batch_size: int,
        *,
        expected_decoded_frames: int,
    ) -> Any:
        import torch

        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("MinWM backend decode must return a floating torch.Tensor")
        if value.ndim != 5 or int(value.shape[0]) != batch_size:
            raise ValueError("MinWM decoded media must have BFCHW shape")
        if int(value.shape[1]) != expected_decoded_frames:
            raise ValueError(
                "MinWM decoded media frame count must match the frozen VAE "
                "temporal geometry"
            )
        if int(value.shape[2]) not in {1, 3, 4}:
            raise ValueError("MinWM decoded media has an invalid BFCHW channel axis")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("MinWM decoded media must be finite")
        return value.detach()

    def _validate_replay_identity(self, batch: RolloutBatch) -> None:
        expected = {
            "adapter": self.name,
            "minwm_stage": self.stage,
            "transition_kernel": self.transition_kernel,
            "replay_credit_assignment": self.replay_credit_assignment,
            "serial_seed_algorithm": _SERIAL_SEED_ALGORITHM,
            "transition_selection_algorithm": _TRANSITION_SELECTION_ALGORITHM,
            "checkpoint_identity": self._checkpoint_identity,
            "minwm_commit": self._minwm_commit,
            "camera_convention": self.camera_convention,
            "guidance_scale": 1.0,
            "logprob_reduction": "latent_mean",
            "scheduler_identity": self._scheduler_identity,
            "warp_denoising_step": self.warp_schedule,
            "context_timestep": self.context_timestep,
            "logprob_epsilon": self.kernel.eps,
            "sampling_contract": self._sampling_contract,
            "causal_state_contract": self._causal_state_contract,
            "conditioning_memoization_contract": (
                self._conditioning_memoization_contract
            ),
            "initial_latent_contract": _INITIAL_LATENT_CONTRACT,
            "policy_contract_sha256": self._policy_contract_sha256(),
        }
        for key, value in expected.items():
            if batch.model_metadata.get(key) != value:
                raise ValueError(f"MinWM replay identity mismatch for {key}")
        actual_conditioning = self._conditioning_identities(
            batch.prompts,
            batch.metadata,
        )
        saved_conditioning = batch.model_metadata.get("conditioning_identity_sha256")
        if list(saved_conditioning or ()) != actual_conditioning:
            raise ValueError(
                "MinWM replay prompt/camera conditioning identity mismatch"
            )
        if any(
            item.get("minwm_conditioning_sha256") != identity
            for item, identity in zip(
                batch.metadata,
                actual_conditioning,
                strict=True,
            )
        ):
            raise ValueError("MinWM rollout metadata conditioning identity mismatch")
        self._validate_credit_assignment(batch)
        self._validate_replay_schedule(batch)

    def _validate_credit_assignment(self, batch: RolloutBatch) -> None:
        import torch

        selected = batch.model_metadata.get("selected_transition_indices")
        rollout_seeds = batch.model_metadata.get("rollout_seeds")
        if not isinstance(selected, list) or len(selected) != batch.batch_size:
            raise ValueError("MinWM replay selected transition identity is invalid")
        if (
            not isinstance(rollout_seeds, list)
            or len(rollout_seeds) != batch.batch_size
        ):
            raise ValueError("MinWM replay rollout seed identity is invalid")
        mask = torch.as_tensor(batch.transition_mask, dtype=torch.bool)
        for row, (index, rollout_seed, item) in enumerate(
            zip(selected, rollout_seeds, batch.metadata, strict=True)
        ):
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("MinWM selected transition index must be an integer")
            if isinstance(rollout_seed, bool) or not isinstance(rollout_seed, int):
                raise ValueError("MinWM rollout seed identity must be an integer")
            if item.get("minwm_selected_transition_index") != index:
                raise ValueError("MinWM rollout metadata selection identity mismatch")
            if item.get("minwm_rollout_seed") != rollout_seed:
                raise ValueError("MinWM rollout metadata seed identity mismatch")
            if not 0 <= index < int(mask.shape[1]):
                raise ValueError("MinWM selected transition index is out of range")
            if int(mask[row].sum().item()) != 1 or not bool(mask[row, index].item()):
                raise ValueError("MinWM sample_one transition mask is invalid")
        expected_mask, expected_selected = self._credit_mask(
            batch.old_log_probs,
            rollout_seeds=rollout_seeds,
            metadata=batch.metadata,
        )
        if selected != expected_selected or not torch.equal(
            mask.cpu(),
            expected_mask.detach().cpu(),
        ):
            raise ValueError(
                "MinWM replay transition selection does not match its frozen identity"
            )

    def _validate_replay_schedule(self, batch: RolloutBatch) -> None:
        import torch

        metadata = batch.model_metadata
        for name in ("num_chunks", "num_steps", "chunk_size"):
            value = metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MinWM replay {name} must be a positive integer")
        num_steps = int(metadata["num_steps"])
        if num_steps < 2:
            raise ValueError("MinWM replay num_steps must be at least 2")
        latent_frames = int(metadata["num_chunks"]) * int(metadata["chunk_size"])
        frame_identity = {
            "latent_frames": latent_frames,
            "camera_conditioning_frames": latent_frames,
            "decoded_media_frames": self._expected_decoded_media_frames(latent_frames),
            "camera_frame_domain": "latent_frames",
            "media_frame_domain": "decoded_pixel_frames",
        }
        for name, value in frame_identity.items():
            if metadata.get(name) != value:
                raise ValueError(f"MinWM replay frame identity mismatch for {name}")
        schedule_mapping = metadata.get("denoising_schedule")
        if not isinstance(schedule_mapping, Mapping) or set(schedule_mapping) != {
            str(index) for index in range(num_steps)
        }:
            raise ValueError("MinWM replay denoising_schedule is invalid")
        schedule = [schedule_mapping[str(index)] for index in range(num_steps)]
        nominal = []
        for item in schedule:
            if not isinstance(item, Mapping):
                raise ValueError("MinWM replay schedule entries must be mappings")
            nominal.append(item.get("nominal_timestep"))
        expected = self._resolved_schedule(
            {"denoising_timesteps": nominal},
            num_steps=num_steps,
        )
        for index, (saved, resolved) in enumerate(zip(schedule, expected, strict=True)):
            for key, value in (
                ("nominal_timestep", resolved.nominal_timestep),
                ("timestep", resolved.timestep),
                ("sigma", resolved.sigma),
            ):
                saved_value = self._finite_float(
                    f"schedule[{index}].{key}",
                    saved.get(key),
                )
                if saved_value != value:
                    raise ValueError(
                        f"MinWM replay denoising_schedule mismatch at {index}.{key}"
                    )

        tensors = batch.model_tensors
        transitions_per_chunk = num_steps - 1
        expected_transitions = int(metadata["num_chunks"]) * transitions_per_chunk
        expected_shape = (batch.batch_size, expected_transitions)
        for name in ("sigma_t", "sigma_next"):
            if tuple(getattr(tensors.get(name), "shape", ())) != expected_shape:
                raise ValueError(
                    f"MinWM replay {name} must have shape {expected_shape}"
                )
        if tuple(getattr(batch.timesteps, "shape", ())) != expected_shape:
            raise ValueError("MinWM replay timesteps shape does not match geometry")
        for chunk_id in range(int(metadata["num_chunks"])):
            for step_id in range(transitions_per_chunk):
                transition_index = chunk_id * transitions_per_chunk + step_id
                checks = (
                    (batch.timesteps[:, transition_index], expected[step_id].timestep),
                    (tensors["sigma_t"][:, transition_index], expected[step_id].sigma),
                    (
                        tensors["sigma_next"][:, transition_index],
                        expected[step_id + 1].sigma,
                    ),
                )
                if any(
                    not bool(
                        torch.allclose(
                            torch.as_tensor(values).float(),
                            torch.full_like(torch.as_tensor(values).float(), target),
                            rtol=0,
                            atol=0,
                        )
                    )
                    for values, target in checks
                ):
                    raise ValueError(
                        "saved MinWM timestep/sigma tensors do not match the "
                        "resolved denoising schedule"
                    )

    def _model_metadata(
        self,
        schedule: tuple[_ResolvedStep, ...],
        *,
        num_chunks: int,
        chunk_size: int,
        conditioning_identities: list[str],
        selected_transition_indices: list[int],
        rollout_seeds: list[int],
    ) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "integration_kind": "rl_finetune_of_minwm_stage_checkpoint",
            "minwm_stage": self.stage,
            "transition_kernel": self.transition_kernel,
            "checkpoint_identity": self._checkpoint_identity,
            "minwm_commit": self._minwm_commit,
            # A step-keyed mapping cannot be mistaken for batch-shaped metadata
            # when RolloutBatch slices a four-sample GRPO group into microbatches.
            "denoising_schedule": {
                str(index): {
                    "nominal_timestep": item.nominal_timestep,
                    "timestep": item.timestep,
                    "sigma": item.sigma,
                }
                for index, item in enumerate(schedule)
            },
            "scheduler_identity": self._scheduler_identity,
            "warp_denoising_step": self.warp_schedule,
            "num_chunks": num_chunks,
            "num_steps": len(schedule),
            "chunk_size": chunk_size,
            "latent_frames": num_chunks * chunk_size,
            "camera_conditioning_frames": num_chunks * chunk_size,
            "decoded_media_frames": self._expected_decoded_media_frames(
                num_chunks * chunk_size
            ),
            "camera_frame_domain": "latent_frames",
            "media_frame_domain": "decoded_pixel_frames",
            "camera_convention": self.camera_convention,
            "guidance_scale": 1.0,
            "conditioning_identity_sha256": list(conditioning_identities),
            "replay_credit_assignment": self.replay_credit_assignment,
            "serial_seed_algorithm": _SERIAL_SEED_ALGORITHM,
            "transition_selection_algorithm": _TRANSITION_SELECTION_ALGORITHM,
            "selected_transition_indices": list(selected_transition_indices),
            "rollout_seeds": list(rollout_seeds),
            "context_timestep": self.context_timestep,
            "terminal_transition": "deterministic_x0_excluded_from_logprob",
            "logprob_reduction": "latent_mean",
            "logprob_epsilon": self.kernel.eps,
            "logprob_interpretation": "dimension_normalized_tempered_log_ratio",
            "recurrent_gradient": "truncated recurrent semi-gradient",
            "causal_state_contract": self._causal_state_contract,
            "sampling_contract": self._sampling_contract,
            "conditioning_memoization_contract": (
                self._conditioning_memoization_contract
            ),
            "initial_latent_contract": _INITIAL_LATENT_CONTRACT,
            "policy_contract_sha256": self._policy_contract_sha256(),
        }

    def _backend_method(self, name: str) -> Any:
        method = getattr(self._backend, name, None)
        if not callable(method):
            raise TypeError(f"MinWM backend must define callable {name}()")
        return method

    def _parameter_device_dtype(self) -> tuple[Any, Any]:
        import torch

        parameters = list(self.train_module.parameters())
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("MinWM train_module parameters must be on one device")
        floating = [
            parameter for parameter in parameters if parameter.is_floating_point()
        ]
        dtype = floating[0].dtype if floating else torch.float32
        return next(iter(devices)), dtype

    def _identity_value(self, name: str, backend_value: Any) -> str:
        config_provides_value = name in self.config

        def _normalize(source: str, value: Any) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"MinWM {source} must provide non-empty {name}")
            return value.strip()

        configured = (
            _normalize("config", self.config[name]) if config_provides_value else None
        )
        reported = (
            _normalize("backend", backend_value) if backend_value is not None else None
        )
        if configured is not None and reported is not None:
            if configured != reported:
                raise ValueError(f"MinWM {name} mismatch between config and backend")
            return configured
        if configured is not None:
            return configured
        if reported is not None:
            return reported
        raise ValueError(f"MinWM backend/config must provide non-empty {name}")

    def _policy_contract_sha256(self) -> str:
        payload = {
            "stage": self.stage,
            "transition_kernel": self.transition_kernel,
            "checkpoint_identity": self._checkpoint_identity,
            "minwm_commit": self._minwm_commit,
            "scheduler_identity": self._scheduler_identity,
            "causal_state_contract": self._causal_state_contract,
            "sampling_contract": self._sampling_contract,
            "conditioning_memoization_contract": (
                self._conditioning_memoization_contract
            ),
            "initial_latent_contract": _INITIAL_LATENT_CONTRACT,
            "lora_identity": self._lora_identity,
            "geometry_identity": self._geometry_identity,
            "backend_implementation": self._backend_implementation,
            "replay_credit_assignment": self.replay_credit_assignment,
            "serial_seed_algorithm": _SERIAL_SEED_ALGORITHM,
            "transition_selection_algorithm": _TRANSITION_SELECTION_ALGORITHM,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _resolve_lora_identity(
        self,
        backend: Any,
        trainable: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        raw = getattr(backend, "lora_identity", None)
        if callable(raw):
            raw = raw()
        if not isinstance(raw, Mapping):
            raise TypeError(
                "MinWM backend must expose lora_identity with target_modules, "
                "rank, and alpha"
            )
        targets = raw.get("target_modules")
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise TypeError(
                "MinWM backend lora_identity.target_modules must be a sequence"
            )
        normalized_targets = sorted({str(value).strip() for value in targets})
        if not normalized_targets or any(not value for value in normalized_targets):
            raise ValueError(
                "MinWM backend lora_identity.target_modules must be non-empty strings"
            )
        rank = raw.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(
                "MinWM backend lora_identity.rank must be a positive integer"
            )
        alpha = self._positive_float(
            "MinWM backend lora_identity.alpha", raw.get("alpha")
        )

        configured_rank = self.config.get("lora_rank")
        if configured_rank is not None and configured_rank != rank:
            raise ValueError("MinWM lora_rank mismatch between config and backend")
        configured_alpha = self.config.get("lora_alpha")
        if configured_alpha is not None and float(configured_alpha) != alpha:
            raise ValueError("MinWM lora_alpha mismatch between config and backend")
        configured_targets = self.config.get("lora_target_modules")
        if configured_targets is not None:
            if isinstance(configured_targets, (str, bytes)) or not isinstance(
                configured_targets, Sequence
            ):
                raise TypeError("MinWM lora_target_modules must be a sequence")
            normalized_configured = sorted(
                {str(value).strip() for value in configured_targets}
            )
            if normalized_configured != normalized_targets:
                raise ValueError(
                    "MinWM lora_target_modules mismatch between config and backend"
                )
        return {
            "target_modules": normalized_targets,
            "rank": rank,
            "alpha": alpha,
            "trainable_parameters": [name for name, _ in trainable],
            "trainable_parameter_dtype": "float32",
        }

    def _resolve_geometry_identity(self, backend: Any) -> dict[str, Any]:
        raw = getattr(backend, "geometry_identity", None)
        if callable(raw):
            raw = raw()
        if not isinstance(raw, Mapping):
            raise TypeError(
                "MinWM backend must expose geometry_identity with chunk_size and "
                "latent_layout"
            )
        chunk_size = raw.get("chunk_size")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
        ):
            raise ValueError(
                "MinWM backend geometry_identity.chunk_size must be a positive integer"
            )
        latent_layout = raw.get("latent_layout")
        if latent_layout != "BCFHW":
            raise ValueError(
                "MinWM backend geometry_identity.latent_layout must be 'BCFHW'"
            )
        configured_chunk_size = self.config.get("chunk_size")
        if (
            configured_chunk_size is not None
            and int(configured_chunk_size) != chunk_size
        ):
            raise ValueError("MinWM chunk_size mismatch between config and backend")
        identity = {
            "chunk_size": chunk_size,
            "latent_layout": latent_layout,
        }
        temporal_keys = {
            "latent_frames",
            "camera_conditioning_frames",
            "decoded_media_frames",
            "vae_temporal_stride",
            "vae_temporal_downsample",
            "decoded_media_frame_formula",
        }
        present = temporal_keys.intersection(raw)
        if not present:
            return identity
        if present != temporal_keys:
            missing = ", ".join(sorted(temporal_keys - present))
            raise ValueError(
                "MinWM backend temporal geometry identity is incomplete; missing "
                f"{missing}"
            )

        integer_values: dict[str, int] = {}
        for key in (
            "latent_frames",
            "camera_conditioning_frames",
            "decoded_media_frames",
            "vae_temporal_stride",
        ):
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"MinWM backend geometry_identity.{key} must be a positive integer"
                )
            integer_values[key] = value
        downsample = raw["vae_temporal_downsample"]
        if (
            isinstance(downsample, (str, bytes))
            or not isinstance(downsample, Sequence)
            or not downsample
            or any(not isinstance(value, bool) for value in downsample)
        ):
            raise ValueError(
                "MinWM backend geometry_identity.vae_temporal_downsample must "
                "be a non-empty boolean sequence"
            )
        if integer_values["vae_temporal_stride"] != 2 ** sum(downsample):
            raise ValueError(
                "MinWM backend VAE temporal stride does not match its "
                "downsample identity"
            )
        if raw["decoded_media_frame_formula"] != _DECODED_MEDIA_FRAME_FORMULA:
            raise ValueError("MinWM backend decoded media frame formula is unsupported")
        latent_frames = integer_values["latent_frames"]
        if integer_values["camera_conditioning_frames"] != latent_frames:
            raise ValueError(
                "MinWM camera conditioning frames must match latent frames"
            )
        expected_decoded = 1 + integer_values["vae_temporal_stride"] * (
            latent_frames - 1
        )
        if integer_values["decoded_media_frames"] != expected_decoded:
            raise ValueError(
                "MinWM decoded media frames do not match the frozen VAE formula"
            )
        configured_total_frames = self.config.get("total_frames")
        if (
            configured_total_frames is not None
            and int(configured_total_frames) != latent_frames
        ):
            raise ValueError(
                "MinWM total_frames must identify latent frames and match the backend"
            )
        return {
            **identity,
            **integer_values,
            "vae_temporal_downsample": list(downsample),
            "decoded_media_frame_formula": _DECODED_MEDIA_FRAME_FORMULA,
        }

    def _expected_decoded_media_frames(self, latent_frames: int) -> int:
        if self._geometry_identity is None:
            raise RuntimeError("MinWM backend geometry identity is not initialized")
        expected_latent_frames = self._geometry_identity.get("latent_frames")
        if expected_latent_frames is None:
            # Compatibility for existing custom backends whose decoders are
            # explicitly one-to-one. The production Wan backend always exposes
            # the complete temporal VAE identity above.
            return latent_frames
        if latent_frames != expected_latent_frames:
            raise ValueError(
                "MinWM runtime latent frames do not match backend geometry"
            )
        return int(self._geometry_identity["decoded_media_frames"])

    @staticmethod
    def _backend_implementation_identity(backend: Any) -> dict[str, Any]:
        backend_type = type(backend)
        class_name = f"{backend_type.__module__}.{backend_type.__qualname__}"
        try:
            class_source = inspect.getsource(backend_type).encode("utf-8")
        except (OSError, TypeError):
            class_source = class_name.encode("utf-8")
        module = inspect.getmodule(backend_type)
        module_path = Path(getattr(module, "__file__", "")) if module else Path()
        module_sha256 = (
            hashlib.sha256(module_path.read_bytes()).hexdigest()
            if module_path.is_file()
            else hashlib.sha256(class_source).hexdigest()
        )
        reported = getattr(backend, "implementation_identity", None)
        if callable(reported):
            reported = reported()
        if reported is None:
            reported = {}
        if not isinstance(reported, Mapping):
            raise TypeError("MinWM backend implementation_identity must be a mapping")
        try:
            normalized = json.loads(
                json.dumps(dict(reported), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "MinWM backend implementation_identity must be JSON-safe"
            ) from exc
        return {
            "class": class_name,
            "class_source_sha256": hashlib.sha256(class_source).hexdigest(),
            "module_sha256": module_sha256,
            "reported": normalized,
        }

    @staticmethod
    def _import_factory(target: str) -> Any:
        module_name, separator, attribute = target.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("backend_factory must use 'module:factory' syntax")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise ImportError(f"backend factory {target!r} does not exist") from exc

    @staticmethod
    def _detach_tree(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach()
        if isinstance(value, Mapping):
            return type(value)(
                (key, MinWMWanAdapter._detach_tree(item)) for key, item in value.items()
            )
        return value

    @staticmethod
    def _finite_float(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite number")
        try:
            resolved = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite")
        return resolved

    @classmethod
    def _positive_float(
        cls,
        name: str,
        value: Any,
        *,
        allow_zero: bool = False,
    ) -> float:
        resolved = cls._finite_float(name, value)
        if resolved < 0 or (resolved == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")
        return resolved


MODEL_ADAPTERS.register("minwm_wan_rl", MinWMWanAdapter)


__all__ = ["MinWMWanAdapter"]
