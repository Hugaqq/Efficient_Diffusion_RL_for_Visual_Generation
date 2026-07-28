"""Lazy production boundary for the official MinWM Wan Action2V runtime.

The module intentionally contains no MinWM model implementation.  It loads the
official checkout lazily, validates its identity, and adapts the native causal
generator primitives to :class:`MinWMWanAdapter`'s backend contract.

This first bounded backend supports inference/rollout and one differentiable
prediction probe.  Full-trajectory GRPO replay must remain disabled until the
optimizer can consume sampled transitions or backpropagate them in a streaming
fashion; mutating a shared native KV cache while several graphs are alive is
not a valid substitute for that gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


_SOURCE_FILES = (
    "Wan21/wan/modules/attention.py",
    "Wan21/wan/modules/causal_model.py",
    "Wan21/wan/modules/model.py",
    "Wan21/wan/modules/prope.py",
    "Wan21/wan/modules/vae.py",
    "Wan21/wan_utils/scheduler.py",
    "Wan21/wan_utils/camera_trajectory.py",
    "Wan21/wan_utils/wan_wrapper.py",
)
_BASE_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)
_BASE_TOKENIZER_DIR = "google/umt5-xxl"
_KNOWN_FSDP_PREFIX = "model._fsdp_wrapped_module."
_SUPPORTED_STAGES = frozenset({"dmd"})
_SUPPORTED_PAYLOAD_KEYS = frozenset({"generator", "generator_ema", "model"})
_SELF_ATTN_PROJECTIONS = frozenset({"q", "k", "v", "o"})
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DECODED_MEDIA_FRAME_FORMULA = "1 + vae_temporal_stride * (latent_frames - 1)"
_OFFICIAL_BOUNDED_GEOMETRY = {
    "chunk_size": 4,
    "latent_frames": 20,
    "latent_channels": 16,
    "latent_height": 60,
    "latent_width": 104,
    "local_attn_size": 20,
}
_SAMPLING_CONTRACT = "native_bfchw_model_dtype_flowmatch_add_noise_v2"
_CAUSAL_STATE_CONTRACT = "temporal_committed_prefix_pure_conditioning_memoized_v2"
_CONDITIONING_MEMOIZATION_CONTRACT = (
    "tensor_identity_bound_crossattn_immutable_after_init_v1"
)
_OFFICIAL_IMPORT_NAMESPACES = ("wan", "wan_utils", "sp")


@dataclass
class _NativeCausalState:
    kv_cache: list[dict[str, Any]]
    prope_kv_cache: list[dict[str, Any]]
    crossattn_cache: list[dict[str, Any]]
    latent_frames: int
    committed_frames: int = 0
    initial_noise: Any | None = None
    pending_differentiable_prediction: bool = False
    conditioning_tensor_identity: tuple[Any, ...] | None = None
    crossattn_memo_identity: tuple[Any, ...] | None = None


def _promote_trainable_parameters_to_fp32(module: Any) -> list[tuple[str, Any]]:
    """Keep LoRA weights and optimizer state above BF16 update resolution."""

    import torch

    trainable: list[tuple[str, Any]] = []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if not parameter.is_floating_point():
                raise RuntimeError(
                    f"MinWM trainable parameter {name!r} must be floating point"
                )
            parameter.data = parameter.data.to(dtype=torch.float32)
            if parameter.grad is not None:
                parameter.grad = parameter.grad.to(dtype=torch.float32)
            trainable.append((name, parameter))
    if not trainable:
        raise RuntimeError("MinWM LoRA injection produced no trainable parameters")
    if any(parameter.dtype != torch.float32 for _, parameter in trainable):
        raise RuntimeError("MinWM trainable LoRA parameters must remain float32")
    return trainable


class _OfficialMinWMRuntime:
    """Import and construct official MinWM primitives only when requested."""

    requires_cuda = True

    def __init__(self, repo_root: Path, *, expected_commit: str):
        self.repo_root = repo_root
        self.expected_commit = expected_commit
        self._asset_overlay: tempfile.TemporaryDirectory[str] | None = None
        self.imported_modules_identity: dict[str, dict[str, str]] = {}

    def load_components(
        self,
        *,
        base_model_path: Path,
        model_name: str,
        timestep_shift: float,
        local_attn_size: int,
    ) -> tuple[Any, Any, Any]:
        _validate_official_module_origins(
            self.repo_root,
            expected_commit=self.expected_commit,
            failure_subject="cached official MinWM module",
        )
        self._install_import_paths()
        try:
            from wan_utils.wan_wrapper import (  # type: ignore[import-not-found]
                WanDiffusionWrapper,
                WanTextEncoder,
                WanVAEWrapper,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Official MinWM dependencies are unavailable. Create a Linux CUDA "
                "environment from the checked-out MinWM requirements, including "
                "easydict, diffusers, transformers, peft, omegaconf, and einops. "
                f"Original import error: {exc}"
            ) from exc

        # The official wrappers use Wan21/wan_models/... relative paths.  A
        # temporary overlay preserves the pristine source checkout and keeps the
        # configured base-model path explicit.
        self._asset_overlay = tempfile.TemporaryDirectory(
            prefix="visualrl-minwm-assets-"
        )
        overlay = Path(self._asset_overlay.name)
        model_parent = overlay / "Wan21" / "wan_models"
        model_parent.mkdir(parents=True)
        (model_parent / model_name).symlink_to(
            base_model_path,
            target_is_directory=True,
        )
        previous_cwd = Path.cwd()
        try:
            os.chdir(overlay)
            generator = WanDiffusionWrapper(
                model_name=model_name,
                timestep_shift=timestep_shift,
                is_causal=True,
                local_attn_size=local_attn_size,
                sink_size=0,
                use_camera=True,
            )
            text_encoder = WanTextEncoder()
            vae = WanVAEWrapper()
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct official MinWM Wan components from "
                f"base_model_path={base_model_path}: {exc}"
            ) from exc
        finally:
            os.chdir(previous_cwd)
        self.imported_modules_identity = _validate_official_module_origins(
            self.repo_root,
            expected_commit=self.expected_commit,
            failure_subject="imported official MinWM module",
        )
        return generator, text_encoder, vae

    @staticmethod
    def attach_lora(
        model: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: float,
    ) -> Any:
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "PEFT is required for the MinWM native backend. Install the "
                "version pinned by the official MinWM checkout."
            ) from exc
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=True,
        )
        return inject_adapter_in_model(config, model, adapter_name="default")

    def _install_import_paths(self) -> None:
        values = [
            str((self.repo_root / relative).resolve())
            for relative in ("Wan21", "shared")
        ]
        sys.path[:] = [value for value in sys.path if value not in values]
        sys.path[0:0] = values


class MinWMWanNativeBackend:
    """Official MinWM Wan Action2V backend with fail-closed identities."""

    media_layout = "BFCHW"
    max_batch_size = 1
    temporal_committed_prefix_pure = True
    causal_state_contract = _CAUSAL_STATE_CONTRACT
    conditioning_memoization_contract = _CONDITIONING_MEMOIZATION_CONTRACT
    sampling_contract = _SAMPLING_CONTRACT

    def __init__(self, config: Mapping[str, Any], *, _runtime: Any | None = None):
        self.config = dict(config)
        self.replay_credit_assignment = _required_choice(
            "replay_credit_assignment",
            self.config,
            frozenset({"sample_one"}),
        )
        self.stage = _required_choice("stage", self.config, _SUPPORTED_STAGES)
        self.repo_root = _required_directory("minwm_repo_root", self.config)
        self.minwm_commit = _required_digest(
            "minwm_commit",
            self.config,
            pattern=_HEX40,
        )
        self.base_model_path = _required_directory(
            "base_model_path",
            self.config,
        )
        self.checkpoint_path = _required_file("checkpoint_path", self.config)
        self._expected_checkpoint_sha256 = _required_digest(
            "checkpoint_sha256",
            self.config,
            pattern=_HEX64,
        )
        self.checkpoint_payload_key = _required_choice(
            "checkpoint_payload_key",
            self.config,
            _SUPPORTED_PAYLOAD_KEYS,
        )
        self.model_name = _string_value(
            "model_name",
            self.config.get("model_name", "Wan2.1-T2V-1.3B"),
        )
        if self.model_name != "Wan2.1-T2V-1.3B":
            raise ValueError(
                "The bounded MinWM native backend supports only "
                "model_name='Wan2.1-T2V-1.3B'"
            )
        self.device_name = _string_value("device", self.config.get("device"))
        self.dtype_name = _required_choice(
            "dtype",
            self.config,
            frozenset({"bfloat16", "float16", "float32"}),
        )
        self.offload_text_encoder = _bool_value(
            "offload_text_encoder",
            self.config.get("offload_text_encoder", True),
        )
        self.offload_vae = _bool_value(
            "offload_vae",
            self.config.get("offload_vae", True),
        )
        self.timestep_shift = _finite_float(
            "timestep_shift",
            self.config.get("timestep_shift", 5.0),
        )
        if self.timestep_shift != 5.0:
            raise ValueError("MinWM Wan Action2V requires timestep_shift=5.0")
        if self.config.get("warp_denoising_step", True) is not True:
            raise ValueError("MinWM Wan Action2V requires warp_denoising_step=true")

        self.chunk_size = _positive_int("chunk_size", self.config.get("chunk_size"))
        self.latent_frames = _positive_int(
            "total_frames",
            self.config.get("total_frames"),
        )
        self.latent_channels = _positive_int(
            "latent_channels",
            self.config.get("latent_channels"),
        )
        self.latent_height = _positive_int(
            "latent_height",
            self.config.get("latent_height"),
        )
        self.latent_width = _positive_int(
            "latent_width",
            self.config.get("latent_width"),
        )
        self.local_attn_size = _positive_int(
            "local_attn_size",
            self.config.get("local_attn_size"),
        )
        if self.latent_frames % self.chunk_size:
            raise ValueError("latent total_frames must be divisible by chunk_size")
        if self.latent_frames > self.local_attn_size:
            raise ValueError(
                "latent total_frames must not exceed local_attn_size; cache "
                "eviction is not supported by the bounded native backend"
            )

        self.lora_rank = _positive_int("lora_rank", self.config.get("lora_rank"))
        self.lora_alpha = _positive_float(
            "lora_alpha",
            self.config.get("lora_alpha"),
        )
        self.context_timestep = _finite_float(
            "context_timestep",
            self.config.get("context_timestep", 0.0),
        )
        self.checkpoint_identity = (
            f"sha256:{self._expected_checkpoint_sha256};"
            f"payload:{self.checkpoint_payload_key}"
        )

        self._runtime = _runtime or _OfficialMinWMRuntime(
            self.repo_root,
            expected_commit=self.minwm_commit,
        )
        self._geometry_profile = self._validate_configured_geometry_profile()
        self._loaded = False
        self._generator: Any | None = None
        self._text_encoder: Any | None = None
        self._vae: Any | None = None
        self._train_module: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._source_identity: dict[str, Any] | None = None
        self._base_identity: dict[str, Any] | None = None
        self._geometry_identity: dict[str, Any] | None = None
        self._lora_identity: dict[str, Any] | None = None
        self._implementation_identity: dict[str, Any] | None = None
        self.scheduler_identity = "uninitialized"

    @property
    def train_module(self) -> Any:
        self.load()
        return self._train_module

    @property
    def geometry_identity(self) -> dict[str, Any]:
        self.load()
        return dict(self._geometry_identity or {})

    @property
    def lora_identity(self) -> dict[str, Any]:
        self.load()
        return dict(self._lora_identity or {})

    def implementation_identity(self) -> dict[str, Any]:
        self.load()
        return dict(self._implementation_identity or {})

    def load(self) -> None:
        if self._loaded:
            return
        import torch

        if self.replay_credit_assignment != "sample_one":
            raise RuntimeError(
                "The bounded MinWM native backend requires "
                "replay_credit_assignment='sample_one' before model loading"
            )

        self._source_identity = _validate_source_checkout(
            self.repo_root,
            expected_commit=self.minwm_commit,
        )
        self._base_identity = _validate_base_model(self.base_model_path)
        actual_checkpoint_sha = _file_sha256(self.checkpoint_path)
        if actual_checkpoint_sha != self._expected_checkpoint_sha256:
            raise RuntimeError(
                "MinWM stage checkpoint SHA256 mismatch: expected "
                f"{self._expected_checkpoint_sha256}, got {actual_checkpoint_sha}. "
                "Do not load or resume from this checkpoint."
            )

        self._device = torch.device(self.device_name)
        self._dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]
        if getattr(self._runtime, "requires_cuda", False):
            if self._device.type != "cuda" or self._device.index is None:
                raise RuntimeError(
                    "Official MinWM execution requires an explicit CUDA device such "
                    "as device='cuda:0'."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable; the official MinWM Wan backend cannot run "
                    "on this host."
                )
            if (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() != 1
            ):
                raise RuntimeError(
                    "The bounded MinWM native backend currently supports one model "
                    "process only; sequence/model sharding is not implemented."
                )

        # Validate the release payload before constructing CUDA components so a
        # wrong outer key or malformed state fails without occupying a GPU.
        payload = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise RuntimeError("MinWM stage checkpoint must contain a mapping payload")
        if self.checkpoint_payload_key not in payload:
            available = ", ".join(sorted(str(key) for key in payload))
            raise RuntimeError(
                "MinWM stage checkpoint is missing explicit payload key "
                f"{self.checkpoint_payload_key!r}; available keys: {available or '<none>'}"
            )
        state_dict = payload[self.checkpoint_payload_key]
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("MinWM checkpoint payload key must contain a state dict")
        normalized = _normalize_state_dict(state_dict)

        with _device_context(self._device):
            generator, text_encoder, vae = self._runtime.load_components(
                base_model_path=self.base_model_path,
                model_name=self.model_name,
                timestep_shift=self.timestep_shift,
                local_attn_size=self.local_attn_size,
            )
        imported_modules_identity = getattr(
            self._runtime,
            "imported_modules_identity",
            None,
        )
        if imported_modules_identity:
            self._source_identity["imported_modules"] = {
                str(name): dict(identity)
                for name, identity in imported_modules_identity.items()
            }
        generator.requires_grad_(False)
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        generator.eval()
        text_encoder.eval()
        vae.eval()

        try:
            generator.load_state_dict(normalized, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Strict MinWM stage checkpoint load failed after known FSDP prefix "
                f"normalization: {exc}"
            ) from exc

        geometry = self._derive_geometry(generator.model, vae)
        target_modules = _self_attention_lora_targets(generator.model)
        injected = self._runtime.attach_lora(
            generator.model,
            target_modules=target_modules,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )
        if injected is not None:
            generator.model = injected
        trainable = [
            (name, parameter)
            for name, parameter in generator.model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any("lora" not in name.casefold() for name, _ in trainable):
            raise RuntimeError(
                "MinWM LoRA injection must leave only LoRA parameters trainable"
            )
        generator.model.num_frame_per_block = self.chunk_size
        generator.model.independent_first_frame = False
        generator.to(device=self._device, dtype=self._dtype)
        trainable = _promote_trainable_parameters_to_fp32(generator.model)
        if not self.offload_text_encoder:
            text_encoder.to(device=self._device, dtype=self._dtype)
        if not self.offload_vae:
            vae.to(device=self._device, dtype=self._dtype)

        self._generator = generator
        self._text_encoder = text_encoder
        self._vae = vae
        self._train_module = generator.model
        self._geometry_identity = geometry
        self._lora_identity = {
            "target_modules": target_modules,
            "rank": self.lora_rank,
            "alpha": self.lora_alpha,
            "trainable_parameters": [name for name, _ in trainable],
            "trainable_parameter_dtype": "float32",
        }
        scheduler_source = self._source_identity["files_sha256"][
            "Wan21/wan_utils/scheduler.py"
        ]
        self.scheduler_identity = (
            "official_minwm_flowmatch_v1:train_steps=1000;shift=5;"
            "sigma_min=0;extra_one_step=true;warp=true;"
            f"source_sha256={scheduler_source}"
        )
        self._implementation_identity = {
            "kind": "official_minwm_wan_action2v_native_backend_v2",
            "source": self._source_identity,
            "base_model": self._base_identity,
            "checkpoint_sha256": actual_checkpoint_sha,
            "checkpoint_payload_key": self.checkpoint_payload_key,
            "stage": self.stage,
            "scheduler_identity": self.scheduler_identity,
            "geometry": geometry,
            "lora": self._lora_identity,
            "trainable_parameter_dtype": "float32",
            "geometry_profile": self._geometry_profile,
            "sampling_contract": self.sampling_contract,
            "causal_state_contract": self.causal_state_contract,
            "conditioning_memoization_contract": (
                self.conditioning_memoization_contract
            ),
            "cache_strategy": "temporal_tail_scratch_prefix_pure_commit_only_v2",
            "temporal_committed_state": "valid_prefix_and_indices_only",
            "temporal_uncommitted_state": "tail_is_scratch_and_may_be_overwritten",
            "cross_attention_cache": (
                "condition_tensor_identity_bound_immutable_memo_after_init"
            ),
            "transition_precision": "fp32_policy_kernel_native_model_dtype_input",
            "cuda_bfloat16_native_parity": (
                "required_real_experiment_gate_not_yet_proven"
            ),
            "max_batch_size": self.max_batch_size,
            "all_transition_replay_supported": False,
            "sampled_transition_replay_supported": True,
            "sampled_transition_replay_mode": "one_uniform_transition_per_sample",
            "required_replay_gate": "sample_one_v1",
            "offload": {
                "text_encoder": self.offload_text_encoder,
                "vae": self.offload_vae,
            },
        }
        self._loaded = True

    def resolve_schedule(
        self,
        nominal_timesteps: Sequence[Any],
        *,
        stage: str,
        warp: bool,
    ) -> list[dict[str, float]]:
        if stage != self.stage:
            raise ValueError("requested MinWM stage does not match the loaded backend")
        if warp is not True:
            raise ValueError("official MinWM Wan Action2V schedule requires warp=true")
        resolved = []
        for raw_value in nominal_timesteps:
            value = _finite_float("nominal timestep", raw_value)
            if not value.is_integer() or not 1 <= value <= 1000:
                raise ValueError(
                    "official MinWM nominal timesteps must be integers in [1, 1000]"
                )
            raw_sigma = value / 1000.0
            sigma = 5.0 * raw_sigma / (1.0 + 4.0 * raw_sigma)
            resolved.append(
                {
                    "nominal_timestep": value,
                    "timestep": 1000.0 * sigma,
                    "sigma": sigma,
                }
            )
        return resolved

    def resolve_camera_control(
        self,
        camera_control: Mapping[str, Any],
        *,
        expected_frames: int,
    ) -> dict[str, Any]:
        """Resolve the official compact Action2V trajectory into model inputs."""

        import numpy as np

        self.load()
        unknown = sorted(set(camera_control).difference({"trajectory", "intrinsics"}))
        if unknown:
            raise ValueError(f"unknown MinWM camera_control fields: {unknown}")
        trajectory = camera_control.get("trajectory")
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise ValueError("MinWM camera_control.trajectory must be non-empty")
        try:
            from wan_utils.camera_trajectory import parse_trajectory
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Official MinWM camera trajectory parser is unavailable"
            ) from exc
        viewmats = np.asarray(parse_trajectory(trajectory.strip()), dtype=np.float32)
        if viewmats.shape != (expected_frames, 4, 4):
            raise ValueError(
                "MinWM camera trajectory must resolve to "
                f"{expected_frames} poses, got {tuple(viewmats.shape)}"
            )
        if not np.isfinite(viewmats).all():
            raise ValueError("MinWM camera trajectory contains non-finite values")

        raw_intrinsics = camera_control.get("intrinsics") or {}
        if not isinstance(raw_intrinsics, Mapping):
            raise TypeError("MinWM camera_control.intrinsics must be a mapping")
        unknown_intrinsics = sorted(
            set(raw_intrinsics).difference({"fx", "fy", "cx", "cy"})
        )
        if unknown_intrinsics:
            raise ValueError(
                f"unknown MinWM camera intrinsics fields: {unknown_intrinsics}"
            )
        values = {}
        for name in ("fx", "fy", "cx", "cy"):
            value = _finite_float(name, raw_intrinsics.get(name, 0.5))
            if name in {"fx", "fy"} and value <= 0.0:
                raise ValueError(f"MinWM camera intrinsic {name} must be positive")
            values[name] = value
        K = np.asarray(
            [
                [values["fx"], 0.0, values["cx"]],
                [0.0, values["fy"], values["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        Ks = np.repeat(K[None], expected_frames, axis=0)
        return {
            "viewmats": viewmats.tolist(),
            "Ks": Ks.tolist(),
            "convention": "w2c",
            "coordinate_system": "opencv",
        }

    def encode_condition(self, prompts: list[str]) -> Any:
        import torch

        self.load()
        if not prompts:
            raise ValueError("MinWM encode_condition requires at least one prompt")
        with _device_context(self._device):
            self._move_component(self._text_encoder, self._device, self._dtype)
            with torch.no_grad():
                condition = self._text_encoder(text_prompts=list(prompts))
        condition = _detach_tree(condition)
        if self.offload_text_encoder:
            self._move_component(self._text_encoder, torch.device("cpu"), None)
            _empty_cuda_cache(self._device)
        return condition

    def reset_causal_state(
        self,
        *,
        batch_size: int,
        total_frames: int,
        dtype: Any,
        device: Any,
    ) -> _NativeCausalState:
        import torch

        self.load()
        if batch_size != 1:
            raise ValueError("bounded MinWM native backend requires batch_size=1")
        if total_frames != self.latent_frames:
            raise ValueError(
                "runtime latent total_frames does not match backend geometry"
            )
        if torch.device(device) != self._device:
            raise ValueError("runtime device does not match MinWM backend device")
        if dtype != self._dtype:
            raise ValueError("runtime dtype does not match MinWM backend dtype")
        cache_tokens = total_frames * self._geometry_identity["frame_seq_length"]
        layers = self._geometry_identity["num_layers"]
        heads = self._geometry_identity["num_heads"]
        head_dim = self._geometry_identity["head_dim"]
        text_len = self._geometry_identity["text_len"]

        def temporal_cache() -> list[dict[str, Any]]:
            return [
                {
                    "k": torch.zeros(
                        1,
                        cache_tokens,
                        heads,
                        head_dim,
                        dtype=self._dtype,
                        device=self._device,
                    ),
                    "v": torch.zeros(
                        1,
                        cache_tokens,
                        heads,
                        head_dim,
                        dtype=self._dtype,
                        device=self._device,
                    ),
                    "global_end_index": torch.zeros(
                        1,
                        dtype=torch.long,
                        device=self._device,
                    ),
                    "local_end_index": torch.zeros(
                        1,
                        dtype=torch.long,
                        device=self._device,
                    ),
                }
                for _ in range(layers)
            ]

        crossattn_cache = [
            {
                "k": torch.zeros(
                    1,
                    text_len,
                    heads,
                    head_dim,
                    dtype=self._dtype,
                    device=self._device,
                ),
                "v": torch.zeros(
                    1,
                    text_len,
                    heads,
                    head_dim,
                    dtype=self._dtype,
                    device=self._device,
                ),
                "is_init": False,
            }
            for _ in range(layers)
        ]
        return _NativeCausalState(
            kv_cache=temporal_cache(),
            prope_kv_cache=temporal_cache(),
            crossattn_cache=crossattn_cache,
            latent_frames=total_frames,
        )

    def sample_initial_latent(
        self,
        state: _NativeCausalState,
        *,
        batch_size: int,
        chunk_id: int,
        frame_start: int,
        chunk_size: int,
        condition: Any,
        viewmats: Any,
        Ks: Any,
        generator: Any,
    ) -> Any:
        import torch

        del condition, viewmats, Ks
        self._validate_state_request(
            state,
            batch_size=batch_size,
            frame_start=frame_start,
            chunk_size=chunk_size,
        )
        if chunk_id != frame_start // self.chunk_size:
            raise ValueError("chunk_id and frame_start are inconsistent")
        if state.initial_noise is None:
            # Official MinWM draws the complete trajectory once in native
            # BFCHW layout.  Drawing BCFHW and permuting afterwards maps the
            # same RNG stream to different frame/channel coordinates.
            state.initial_noise = torch.randn(
                1,
                self.latent_frames,
                self.latent_channels,
                self.latent_height,
                self.latent_width,
                dtype=self._dtype,
                device=self._device,
                generator=generator,
            )
        native_chunk = state.initial_noise[:, frame_start : frame_start + chunk_size]
        return _bfchw_to_bcfhw(native_chunk).clone()

    def sample_x0_renoise(
        self,
        x0_pred: Any,
        *,
        sigma_next: Any,
        generator: Any,
    ) -> Any:
        """Apply the official native-layout FlowMatch ``add_noise`` path.

        The returned tensor is the observed policy transition in adapter
        BCFHW layout.  Distribution scoring remains the adapter kernel's fp32
        responsibility; no second noise draw is permitted there.
        """

        import torch

        self.load()
        if not isinstance(x0_pred, torch.Tensor) or not x0_pred.is_floating_point():
            raise TypeError("MinWM native x0_pred must be a floating tensor")
        expected_shape = (
            1,
            self.latent_channels,
            self.chunk_size,
            self.latent_height,
            self.latent_width,
        )
        if tuple(x0_pred.shape) != expected_shape:
            raise ValueError(
                "MinWM native x0_pred does not match bounded chunk geometry"
            )
        native_x0 = _bcfhw_to_bfchw(x0_pred.detach()).to(
            device=self._device,
            dtype=self._dtype,
        )
        noise = torch.randn(
            native_x0.shape,
            dtype=self._dtype,
            device=self._device,
            generator=generator,
        )
        sigma = torch.as_tensor(
            sigma_next,
            dtype=torch.float32,
            device=self._device,
        )
        if sigma.numel() != 1 or not bool(torch.isfinite(sigma).all().item()):
            raise ValueError("MinWM native sigma_next must be one finite scalar")
        if not bool((sigma > 0).all().item()):
            raise ValueError("MinWM native sigma_next must be greater than zero")
        sigma = sigma.reshape(1, 1, 1, 1, 1)
        # This mirrors FlowMatchScheduler.add_noise exactly, including the
        # final model-dtype cast performed by type_as(noise).
        observed_native = ((1.0 - sigma) * native_x0 + sigma * noise).type_as(noise)
        return _bfchw_to_bcfhw(observed_native).detach()

    def predict_flow_x0(
        self,
        state: _NativeCausalState,
        x_t: Any,
        condition: Any,
        viewmats: Any,
        Ks: Any,
        *,
        timestep: Any,
        frame_start: int,
        chunk_id: int,
        step_id: int,
    ) -> tuple[Any, Any]:
        import torch

        del chunk_id, step_id
        self._validate_state_request(
            state,
            batch_size=int(x_t.shape[0]),
            frame_start=frame_start,
            chunk_size=int(x_t.shape[2]),
        )
        if torch.is_grad_enabled():
            if state.pending_differentiable_prediction:
                raise RuntimeError(
                    "MinWM native cache already backs a live differentiable "
                    "prediction. Use sampled-transition or streaming backward before "
                    "requesting another replay transition."
                )
            state.pending_differentiable_prediction = True

        committed_end = frame_start * self._geometry_identity["frame_seq_length"]
        expected_end = (frame_start + int(x_t.shape[2])) * self._geometry_identity[
            "frame_seq_length"
        ]
        self._assert_temporal_indices(state, committed_end)
        memo_was_initialized = self._bind_conditioning(state, condition)
        native_x = _bcfhw_to_bfchw(x_t).to(device=self._device, dtype=self._dtype)
        frames = int(native_x.shape[1])
        native_timestep = (
            torch.as_tensor(
                timestep,
                dtype=torch.float32,
                device=self._device,
            )
            .reshape(1, 1)
            .expand(1, frames)
        )
        condition = _move_tree(condition, device=self._device, dtype=self._dtype)
        viewmats = torch.as_tensor(
            viewmats,
            dtype=self._dtype,
            device=self._device,
        )
        Ks = torch.as_tensor(Ks, dtype=self._dtype, device=self._device)
        with _device_context(self._device):
            try:
                flow, x0 = self._generator(
                    noisy_image_or_video=native_x,
                    conditional_dict=condition,
                    timestep=native_timestep,
                    kv_cache=state.kv_cache,
                    crossattn_cache=state.crossattn_cache,
                    current_start=committed_end,
                    viewmats=viewmats,
                    Ks=Ks,
                    prope_kv_cache=state.prope_kv_cache,
                )
                self._assert_temporal_indices(state, expected_end)
                self._finalize_conditioning_memo(
                    state,
                    was_initialized=memo_was_initialized,
                )
            finally:
                self._set_temporal_indices(state, committed_end)
        return _bfchw_to_bcfhw(flow), _bfchw_to_bcfhw(x0)

    def commit_clean_context(
        self,
        state: _NativeCausalState,
        final_clean: Any,
        condition: Any,
        viewmats: Any,
        Ks: Any,
        *,
        frame_start: int,
        context_timestep: float,
    ) -> None:
        import torch

        if state.pending_differentiable_prediction:
            raise RuntimeError(
                "Cannot mutate MinWM committed cache while a differentiable "
                "prediction graph is alive. Complete streaming backward before commit."
            )
        self._validate_state_request(
            state,
            batch_size=int(final_clean.shape[0]),
            frame_start=frame_start,
            chunk_size=int(final_clean.shape[2]),
        )
        committed_end = frame_start * self._geometry_identity["frame_seq_length"]
        expected_end = (
            frame_start + int(final_clean.shape[2])
        ) * self._geometry_identity["frame_seq_length"]
        self._assert_temporal_indices(state, committed_end)
        memo_was_initialized = self._bind_conditioning(state, condition)
        native_clean = _bcfhw_to_bfchw(final_clean.detach()).to(
            device=self._device,
            dtype=self._dtype,
        )
        frames = int(native_clean.shape[1])
        native_timestep = torch.full(
            (1, frames),
            _finite_float("context_timestep", context_timestep),
            dtype=torch.float32,
            device=self._device,
        )
        condition = _move_tree(condition, device=self._device, dtype=self._dtype)
        viewmats = torch.as_tensor(
            viewmats,
            dtype=self._dtype,
            device=self._device,
        )
        Ks = torch.as_tensor(Ks, dtype=self._dtype, device=self._device)
        with _device_context(self._device):
            with torch.no_grad():
                self._generator(
                    noisy_image_or_video=native_clean,
                    conditional_dict=condition,
                    timestep=native_timestep,
                    kv_cache=state.kv_cache,
                    crossattn_cache=state.crossattn_cache,
                    current_start=committed_end,
                    viewmats=viewmats,
                    Ks=Ks,
                    prope_kv_cache=state.prope_kv_cache,
                )
        self._finalize_conditioning_memo(
            state,
            was_initialized=memo_was_initialized,
        )
        self._assert_temporal_indices(state, expected_end)
        state.committed_frames += frames

    def decode(
        self,
        final_clean_chunks: Any,
        *,
        condition: Any,
        viewmats: Any,
        Ks: Any,
    ) -> Any:
        import torch

        del condition, viewmats, Ks
        self.load()
        if final_clean_chunks.ndim != 6:
            raise ValueError(
                "MinWM decode expects [B, chunks, C, F, H, W] clean latents"
            )
        batch, chunks, channels, frames, height, width = final_clean_chunks.shape
        if batch != 1 or channels != self.latent_channels:
            raise ValueError("MinWM decode latent batch/channel geometry mismatch")
        if (
            chunks * frames != self.latent_frames
            or frames != self.chunk_size
            or height != self.latent_height
            or width != self.latent_width
        ):
            raise ValueError("MinWM decode latent frame/spatial geometry mismatch")
        native = final_clean_chunks.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            chunks * frames,
            channels,
            height,
            width,
        )
        with _device_context(self._device):
            self._move_component(self._vae, self._device, self._dtype)
            with torch.no_grad():
                media = self._vae.decode_to_pixel(
                    native.to(device=self._device, dtype=self._dtype),
                    use_cache=False,
                )
                media = (media.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        expected_media_frames = self._geometry_identity["decoded_media_frames"]
        if (
            not isinstance(media, torch.Tensor)
            or media.ndim != 5
            or int(media.shape[0]) != batch
            or int(media.shape[1]) != expected_media_frames
        ):
            raise RuntimeError(
                "Official MinWM VAE decode must return BFCHW media with "
                f"{expected_media_frames} decoded frames for "
                f"{self.latent_frames} latent frames"
            )
        if self.offload_vae:
            self._move_component(self._vae, torch.device("cpu"), None)
            _empty_cuda_cache(self._device)
        return media

    def _derive_geometry(self, model: Any, vae: Any) -> dict[str, Any]:
        patch_size = tuple(int(value) for value in model.patch_size)
        if len(patch_size) != 3 or patch_size[0] != 1:
            raise RuntimeError("MinWM native backend requires temporal patch_size=1")
        if self.latent_height % patch_size[1] or self.latent_width % patch_size[2]:
            raise RuntimeError("latent H/W must be divisible by the native patch size")
        if int(model.in_dim) != self.latent_channels:
            raise RuntimeError("latent_channels does not match the native MinWM model")
        if int(model.local_attn_size) != self.local_attn_size:
            raise RuntimeError("local_attn_size does not match the loaded MinWM model")
        num_heads = int(model.num_heads)
        dim = int(model.dim)
        if dim % num_heads:
            raise RuntimeError("native MinWM dim must be divisible by num_heads")
        frame_seq_length = (self.latent_height // patch_size[1]) * (
            self.latent_width // patch_size[2]
        )
        vae_temporal_downsample = self._vae_temporal_downsample(vae)
        vae_temporal_stride = 2 ** sum(vae_temporal_downsample)
        if vae_temporal_stride != 4:
            raise RuntimeError(
                "The bounded MinWM Wan backend requires official VAE temporal stride 4"
            )
        decoded_media_frames = 1 + vae_temporal_stride * (self.latent_frames - 1)
        return {
            "latent_layout": "BCFHW",
            "native_latent_layout": "BFCHW",
            "chunk_size": self.chunk_size,
            "latent_frames": self.latent_frames,
            "camera_conditioning_frames": self.latent_frames,
            "decoded_media_frames": decoded_media_frames,
            "vae_temporal_stride": vae_temporal_stride,
            "vae_temporal_downsample": vae_temporal_downsample,
            "decoded_media_frame_formula": _DECODED_MEDIA_FRAME_FORMULA,
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "patch_size": list(patch_size),
            "frame_seq_length": frame_seq_length,
            "num_layers": int(model.num_layers),
            "num_heads": num_heads,
            "head_dim": dim // num_heads,
            "text_len": int(model.text_len),
            "local_attn_size": self.local_attn_size,
            "independent_first_frame": False,
            "batch_size": 1,
            "cache_eviction": False,
        }

    def _validate_configured_geometry_profile(self) -> str:
        configured = {
            "chunk_size": self.chunk_size,
            "latent_frames": self.latent_frames,
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "local_attn_size": self.local_attn_size,
        }
        if getattr(self._runtime, "requires_cuda", False):
            if configured != _OFFICIAL_BOUNDED_GEOMETRY:
                expected = ", ".join(
                    f"{key}={value}"
                    for key, value in _OFFICIAL_BOUNDED_GEOMETRY.items()
                )
                raise ValueError(
                    "Official MinWM production runtime requires the frozen bounded "
                    f"geometry ({expected})"
                )
            return "official_bounded_action2v_chunk4_f20_c16_h60_w104_local20_v1"
        # Small shapes exist solely to exercise wiring on CPU without official
        # weights.  The identity prevents that evidence from being presented as
        # production geometry validation.
        return "test_only_non_cuda_geometry_compatibility_v1"

    @staticmethod
    def _vae_temporal_downsample(vae: Any) -> list[bool]:
        vae_model = getattr(vae, "model", None)
        raw = getattr(vae_model, "temperal_downsample", None)
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or not raw
            or any(not isinstance(value, bool) for value in raw)
        ):
            raise RuntimeError(
                "Official MinWM Wan VAE must expose boolean "
                "model.temperal_downsample geometry"
            )
        return list(raw)

    def _validate_state_request(
        self,
        state: _NativeCausalState,
        *,
        batch_size: int,
        frame_start: int,
        chunk_size: int,
    ) -> None:
        if not isinstance(state, _NativeCausalState):
            raise TypeError("invalid MinWM native causal state")
        if batch_size != 1:
            raise ValueError("bounded MinWM native backend requires batch_size=1")
        if chunk_size != self.chunk_size:
            raise ValueError("native chunk_size does not match backend geometry")
        if frame_start != state.committed_frames:
            raise ValueError("frame_start must equal the committed causal prefix")
        if frame_start + chunk_size > state.latent_frames:
            raise ValueError("MinWM chunk exceeds the configured latent total_frames")

    @staticmethod
    def _move_component(component: Any, device: Any, dtype: Any | None) -> None:
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        component.to(**kwargs)

    @staticmethod
    def _temporal_caches(state: _NativeCausalState) -> tuple[list[Any], list[Any]]:
        return state.kv_cache, state.prope_kv_cache

    def _bind_conditioning(
        self,
        state: _NativeCausalState,
        condition: Any,
    ) -> bool:
        identity = _conditioning_tensor_identity(condition)
        if state.conditioning_tensor_identity is None:
            state.conditioning_tensor_identity = identity
        elif state.conditioning_tensor_identity != identity:
            raise ValueError(
                "MinWM causal state is bound to one conditioning tensor identity; "
                "reset state before using different prompt conditioning"
            )
        was_initialized = state.crossattn_memo_identity is not None
        if was_initialized:
            current = self._crossattn_memo_identity(state)
            if current != state.crossattn_memo_identity:
                raise RuntimeError(
                    "MinWM cross-attention conditioning memo changed after initialization"
                )
        return was_initialized

    def _finalize_conditioning_memo(
        self,
        state: _NativeCausalState,
        *,
        was_initialized: bool,
    ) -> None:
        if not all(item.get("is_init") is True for item in state.crossattn_cache):
            raise RuntimeError(
                "MinWM native generator did not initialize every cross-attention memo"
            )
        current = self._crossattn_memo_identity(state)
        if not was_initialized:
            state.crossattn_memo_identity = current
            return
        if current != state.crossattn_memo_identity:
            raise RuntimeError(
                "MinWM cross-attention conditioning memo must be immutable after init"
            )

    @staticmethod
    def _crossattn_memo_identity(state: _NativeCausalState) -> tuple[Any, ...]:
        identity: list[Any] = []
        for layer_index, item in enumerate(state.crossattn_cache):
            for key in ("k", "v"):
                tensor = item.get(key)
                if tensor is None or not hasattr(tensor, "data_ptr"):
                    raise RuntimeError(
                        "MinWM cross-attention memo must contain tensor k/v entries"
                    )
                identity.append(
                    (
                        layer_index,
                        key,
                        id(tensor),
                        int(tensor.data_ptr()),
                        int(tensor._version),
                        tuple(int(value) for value in tensor.shape),
                        str(tensor.dtype),
                        str(tensor.device),
                    )
                )
        return tuple(identity)

    def _assert_temporal_indices(
        self,
        state: _NativeCausalState,
        expected: int,
    ) -> None:
        for family in self._temporal_caches(state):
            for item in family:
                if (
                    int(item["global_end_index"].item()) != expected
                    or int(item["local_end_index"].item()) != expected
                ):
                    raise RuntimeError(
                        "MinWM native cache index mismatch; eviction or unintended "
                        "causal-state mutation was detected"
                    )

    def _set_temporal_indices(self, state: _NativeCausalState, value: int) -> None:
        for family in self._temporal_caches(state):
            for item in family:
                item["global_end_index"].fill_(value)
                item["local_end_index"].fill_(value)


def build_minwm_wan_native_backend(config: Mapping[str, Any]) -> MinWMWanNativeBackend:
    """Importable factory used by ``model.extra.backend_factory``."""

    return MinWMWanNativeBackend(config)


def _validate_source_checkout(root: Path, *, expected_commit: str) -> dict[str, Any]:
    try:
        commit = _git(root, "rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"minwm_repo_root must be a readable Git checkout: {root}: {exc}"
        ) from exc
    if commit != expected_commit:
        raise RuntimeError(
            f"MinWM source commit mismatch: expected {expected_commit}, got {commit}"
        )
    files = {}
    for relative in _SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"MinWM source file is missing: {path}")
        actual_sha256 = _file_sha256(path)
        committed_sha256 = _committed_file_sha256(
            root,
            commit=commit,
            relative_path=relative,
        )
        if actual_sha256 != committed_sha256:
            raise RuntimeError(
                "frozen MinWM source/config file differs from the configured "
                f"commit: {relative}"
            )
        files[relative] = actual_sha256
    return {"commit": commit, "files_sha256": files}


def _validate_official_module_origins(
    root: Path,
    *,
    expected_commit: str,
    failure_subject: str,
) -> dict[str, dict[str, str]]:
    root = root.resolve()
    identities: dict[str, dict[str, str]] = {}
    for name, module in sorted(sys.modules.items()):
        if not _is_official_module_name(name):
            continue
        try:
            source_path = _module_source_path(module)
            relative = source_path.relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"{failure_subject} {name!r} is not from the configured "
                f"MinWM checkout {root}: {exc}"
            ) from exc
        actual_sha256 = _file_sha256(source_path)
        try:
            committed_sha256 = _committed_file_sha256(
                root,
                commit=expected_commit,
                relative_path=relative,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{failure_subject} {name!r} does not resolve to source tracked "
                f"by configured commit {expected_commit}: {relative}"
            ) from exc
        if actual_sha256 != committed_sha256:
            raise RuntimeError(
                f"{failure_subject} {name!r} source differs from configured "
                f"commit {expected_commit}: {relative}"
            )
        identities[name] = {
            "relative_path": relative,
            "sha256": actual_sha256,
        }
    return identities


def _is_official_module_name(name: str) -> bool:
    return any(
        name == namespace or name.startswith(f"{namespace}.")
        for namespace in _OFFICIAL_IMPORT_NAMESPACES
    )


def _module_source_path(module: Any) -> Path:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or not origin or origin in {"built-in", "frozen"}:
        raise RuntimeError("module has no filesystem source origin")
    path = Path(origin)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError as exc:
            raise RuntimeError("module bytecode has no canonical source path") from exc
    return path.resolve(strict=True)


def _committed_file_sha256(
    root: Path,
    *,
    commit: str,
    relative_path: str,
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"MinWM source/config file is not tracked by {commit}: {relative_path}"
        ) from exc
    return hashlib.sha256(completed.stdout).hexdigest()


def _validate_base_model(root: Path) -> dict[str, Any]:
    paths = []
    for relative in _BASE_REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(
                "Wan2.1 base model is incomplete; missing required file "
                f"{relative!r} under {root}"
            )
        paths.append(path)
    tokenizer_root = root / _BASE_TOKENIZER_DIR
    tokenizer_files = sorted(
        path for path in tokenizer_root.rglob("*") if path.is_file()
    )
    if not tokenizer_files:
        raise RuntimeError(
            "Wan2.1 base model is missing the google/umt5-xxl tokenizer files"
        )
    paths.extend(tokenizer_files)

    entries = []
    digest = hashlib.sha256(b"visualrl-minwm-base-v1\0")
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        sha256 = _file_sha256(path)
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        entries.append({"path": relative, "size": size, "sha256": sha256})
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def _normalize_state_dict(state_dict: Mapping[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, value in state_dict.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError(
                "MinWM checkpoint state-dict keys must be non-empty strings"
            )
        name = raw_name
        if name.startswith(_KNOWN_FSDP_PREFIX):
            name = "model." + name[len(_KNOWN_FSDP_PREFIX) :]
        if name in normalized:
            raise RuntimeError(
                f"MinWM checkpoint key collision after FSDP normalization: {name}"
            )
        normalized[name] = value
    return normalized


def _self_attention_lora_targets(model: Any) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if (
            len(parts) == 4
            and parts[0] == "blocks"
            and parts[1].isdigit()
            and parts[2] == "self_attn"
            and parts[3] in _SELF_ATTN_PROJECTIONS
            and hasattr(module, "weight")
        ):
            targets.append(name)
    expected = int(model.num_layers) * len(_SELF_ATTN_PROJECTIONS)
    if len(targets) != expected:
        raise RuntimeError(
            "Could not resolve exactly q/k/v/o for every native MinWM causal "
            f"self-attention block: expected {expected}, found {len(targets)}"
        )
    return sorted(targets)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bcfhw_to_bfchw(value: Any) -> Any:
    if getattr(value, "ndim", None) != 5:
        raise ValueError("MinWM latent must have BCFHW rank 5")
    return value.permute(0, 2, 1, 3, 4).contiguous()


def _bfchw_to_bcfhw(value: Any) -> Any:
    if getattr(value, "ndim", None) != 5:
        raise ValueError("native MinWM latent must have BFCHW rank 5")
    return value.permute(0, 2, 1, 3, 4).contiguous()


def _move_tree(value: Any, *, device: Any, dtype: Any) -> Any:
    if hasattr(value, "to"):
        kwargs = {"device": device}
        if getattr(value, "is_floating_point", lambda: False)():
            kwargs["dtype"] = dtype
        return value.to(**kwargs)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_tree(item, device=device, dtype=dtype))
            for key, item in value.items()
        )
    return value


def _conditioning_tensor_identity(value: Any) -> tuple[Any, ...]:
    """Return a cheap object/storage identity for every conditioning tensor."""

    entries: list[Any] = []

    def visit(item: Any, path: tuple[str, ...]) -> None:
        if hasattr(item, "data_ptr") and hasattr(item, "shape"):
            entries.append(
                (
                    path,
                    id(item),
                    int(item.data_ptr()),
                    int(item._version),
                    tuple(int(size) for size in item.shape),
                    str(item.dtype),
                    str(item.device),
                )
            )
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda raw: str(raw)):
                visit(item[key], (*path, str(key)))
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item):
                visit(child, (*path, str(index)))

    visit(value, ())
    if not entries:
        raise TypeError("MinWM conditioning must contain at least one tensor")
    return tuple(entries)


def _detach_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach()
    if isinstance(value, Mapping):
        return type(value)((key, _detach_tree(item)) for key, item in value.items())
    return value


def _empty_cuda_cache(device: Any) -> None:
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


def _device_context(device: Any) -> Any:
    import torch

    if getattr(device, "type", None) == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def _required_directory(name: str, config: Mapping[str, Any]) -> Path:
    value = _string_value(name, config.get(name))
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{name} must point to an existing directory: {path}")
    return path


def _required_file(name: str, config: Mapping[str, Any]) -> Path:
    value = _string_value(name, config.get(name))
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name} must point to an existing file: {path}")
    return path


def _required_choice(
    name: str,
    config: Mapping[str, Any],
    choices: frozenset[str],
) -> str:
    value = _string_value(name, config.get(name))
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")
    return value


def _required_digest(
    name: str,
    config: Mapping[str, Any],
    *,
    pattern: re.Pattern[str],
) -> str:
    value = _string_value(name, config.get(name)).casefold()
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has an invalid lowercase hexadecimal digest")
    return value


def _string_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _bool_value(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def _positive_float(name: str, value: Any) -> float:
    resolved = _finite_float(name, value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


__all__ = ["MinWMWanNativeBackend", "build_minwm_wan_native_backend"]
