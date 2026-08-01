"""Executable, test-only Flow-GRPO SD3 one-shot parity harness.

This file is deliberately outside :mod:`visual_rl` and is not installed in
the wheel.  It has no dataset, Runner, training loop, ArtifactManager, command
line options, or environment-variable overrides.  The one fixed JSON case and
its one resolved YAML are the only inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any


_DEPENDENCIES = (
    "absl",
    "accelerate",
    "ml_collections",
    "diffusers",
    "numpy",
    "torch",
    "wandb",
    "PIL",
    "peft",
)
_ITEM_KEYS = (
    "prompt_encoding",
    "initial_latent",
    "timestep",
    "rollout_latent",
    "old_log_prob",
    "current_log_prob",
    "transition_statistics",
    "group_advantage",
    "policy_loss",
    "reference_kl",
    "total_loss",
    "gradient",
    "parameter_delta",
    "checkpoint_resume",
)
_RESUME_KEYS = (
    "adapter_tensors",
    "optimizer_state",
    "grad_scaler_state",
    "rng_state",
    "next_step_inputs",
    "global_step",
    "non_timing_metrics",
)
_CASE_KEYS = frozenset(
    {
        "schema_version",
        "config_path",
        "prompt",
        "seed",
        "logical_step",
        "reward_values",
        "expected_advantages",
    }
)
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "case",
        "config_path",
        "precision",
        "items",
        "overall_pass",
    }
)
_COMPARISON_KEYS = frozenset(
    {
        "tensor_name",
        "shape",
        "dtype",
        "rtol",
        "atol",
        "max_abs_error",
        "max_rel_error",
        "passed",
    }
)
_CUDA_RTOL = 1.0e-5
_CUDA_ATOL = 1.0e-6
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _configure_cuda_determinism(torch_module: Any) -> None:
    """Freeze CUDA math choices for this standalone FP32 parity process."""

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "native parity requires CUBLAS_WORKSPACE_CONFIG="
            f"{_CUBLAS_WORKSPACE_CONFIG}, got {configured!r}"
        )
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False


def _seed_native_case(seed: int) -> None:
    """Seed every model-construction RNG from the frozen case."""

    from visual_rl.core.seed import seed_everything

    seed_everything(seed)


@dataclass(frozen=True)
class _NativeComputeLogProbView:
    """The exact config attributes read by native ``compute_log_prob()``."""

    @dataclass(frozen=True)
    class Sample:
        guidance_scale: float
        noise_level: float

    @dataclass(frozen=True)
    class Train:
        cfg: bool

    sample: Sample
    train: Train

    @classmethod
    def from_resolved(cls, config: Any) -> _NativeComputeLogProbView:
        guidance_scale = float(config.model.params["guidance_scale"])
        return cls(
            sample=cls.Sample(
                guidance_scale=guidance_scale,
                noise_level=0.7,
            ),
            train=cls.Train(cfg=guidance_scale > 1.0),
        )


@dataclass(frozen=True)
class _RngSnapshot:
    """Complete process RNG state used to isolate the two oracle branches."""

    python_state: tuple[Any, ...]
    numpy_state: tuple[Any, ...]
    torch_cpu: Any
    torch_cuda: tuple[Any, ...]

    @classmethod
    def capture(cls) -> _RngSnapshot:
        import numpy as np
        import torch

        return cls(
            python_state=random.getstate(),
            numpy_state=deepcopy(np.random.get_state()),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=tuple(
                state.clone() for state in torch.cuda.get_rng_state_all()
            ),
        )

    def restore(self) -> None:
        import numpy as np
        import torch

        random.setstate(self.python_state)
        np.random.set_state(deepcopy(self.numpy_state))
        torch.set_rng_state(self.torch_cpu.clone())
        if self.torch_cuda:
            if len(self.torch_cuda) != torch.cuda.device_count():
                raise RuntimeError("CUDA RNG topology changed during native parity")
            torch.cuda.set_rng_state_all(
                [state.clone() for state in self.torch_cuda]
            )

    def exactly_equal(self, other: _RngSnapshot) -> bool:
        import numpy as np
        import torch

        if not isinstance(other, _RngSnapshot):
            return False
        return (
            self.python_state == other.python_state
            and self.numpy_state[0] == other.numpy_state[0]
            and np.array_equal(self.numpy_state[1], other.numpy_state[1])
            and self.numpy_state[2:] == other.numpy_state[2:]
            and torch.equal(self.torch_cpu, other.torch_cpu)
            and len(self.torch_cuda) == len(other.torch_cuda)
            and all(
                torch.equal(left, right)
                for left, right in zip(
                    self.torch_cuda,
                    other.torch_cuda,
                    strict=True,
                )
            )
        )


@dataclass(frozen=True)
class _NativeHelpers:
    compute_text_embeddings: Callable[..., Any]
    compute_log_prob: Callable[..., Any]
    pipeline_with_logprob: Callable[..., Any]
    tracker_type: type[Any]


class _NativeVaeProxy:
    """Decode through the Adapter's existing frozen-module lifecycle."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter.pipeline.vae, name)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        self._adapter._activate_vae_for_decode()
        try:
            return self._adapter.pipeline.vae.decode(*args, **kwargs)
        finally:
            self._adapter._offload_vae_after_decode()


class _NativePipelineProxy:
    """Expose the training CUDA device while frozen modules remain offloaded."""

    def __init__(self, adapter: Any) -> None:
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_vae_proxy", _NativeVaeProxy(adapter))

    @property
    def _execution_device(self) -> Any:
        return self._adapter.device

    @property
    def vae(self) -> _NativeVaeProxy:
        return self._vae_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter.pipeline, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._adapter.pipeline, name, value)


@dataclass(frozen=True)
class _UpdateTrace:
    current_log_prob: Any
    current_mean: Any
    reference_mean: Any
    transition_std: Any
    policy_loss: Any
    reference_kl: Any
    total_loss: Any
    pre_clip_gradients: Mapping[str, Any]
    post_clip_gradients: Mapping[str, Any]
    parameter_delta: Mapping[str, Any]
    slot_rng_before: tuple[_RngSnapshot, ...]
    slot_rng_after: tuple[_RngSnapshot, ...]
    backward_count: int
    clip_count: int
    step_count: int
    zero_grad_count: int


@dataclass(frozen=True)
class _ResumeProjection:
    adapter_tensors: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    grad_scaler_state: Any
    start_rng: _RngSnapshot
    end_rng: _RngSnapshot
    next_step_inputs: Mapping[str, Any]
    global_step: int
    non_timing_metrics: Mapping[str, float]


class NativeFlowReferenceOracle:
    """Independent statement of the native first-update loss."""

    @staticmethod
    def group_advantages(reward_values: Any, epsilon: float):
        import torch

        rewards = torch.as_tensor(reward_values, dtype=torch.float64)
        if rewards.ndim != 1 or rewards.numel() < 2:
            raise ValueError("reward_values must be a one-dimensional group")
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("reward_values must be finite")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        return (rewards - rewards.mean()) / (
            rewards.std(unbiased=False) + epsilon
        )

    @staticmethod
    def evaluate(
        *,
        old_log_probs: Any,
        new_log_probs: Any,
        advantages: Any,
        current_mean: Any,
        reference_mean: Any,
        std_dev: Any,
        clip_range: float,
        beta: float,
    ) -> dict[str, Any]:
        import torch

        def tensor(value: Any):
            return (
                value
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value, dtype=torch.float64)
            )

        old = tensor(old_log_probs)
        new = tensor(new_log_probs)
        advantage = tensor(advantages)
        current = tensor(current_mean)
        reference = tensor(reference_mean)
        std = tensor(std_dev)
        if old.ndim != 2 or new.shape != old.shape or advantage.shape != old.shape:
            raise ValueError("log-probabilities and advantages must share [B,T]")
        if current.shape != reference.shape or current.shape[:2] != old.shape:
            raise ValueError("transition means must share [B,T,...X]")
        if std.shape == old.shape:
            std = std.reshape(*old.shape, *((1,) * (current.ndim - 2)))
        elif (
            std.ndim != current.ndim
            or std.shape[:2] != old.shape
            or any(size != 1 for size in std.shape[2:])
        ):
            raise ValueError("std_dev must be [B,T] or [B,T,1,...,1]")
        if not 0.0 < clip_range < 1.0:
            raise ValueError("clip_range must satisfy 0 < value < 1")
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        tensors = (old, new, advantage, current, reference, std)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("oracle tensors must be finite")
        if not bool((std > 0.0).all()):
            raise ValueError("std_dev must be strictly positive")

        ratio = torch.exp(new - old)
        clipped_ratio = ratio.clamp(
            min=1.0 - clip_range,
            max=1.0 + clip_range,
        )
        policy_loss = -torch.minimum(
            ratio * advantage,
            clipped_ratio * advantage,
        ).mean()
        per_element_kl = (
            (current - reference).square() / (2.0 * std.square())
        )
        reference_kl = per_element_kl.flatten(start_dim=2).mean(dim=2).mean()
        total_loss = policy_loss + beta * reference_kl
        return {
            "policy_loss": policy_loss,
            "reference_kl": reference_kl,
            "total_loss": total_loss,
        }


def _failure_report(
    *,
    case_name: str,
    config_path: str,
    precision: str,
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name in _ITEM_KEYS:
        if name == "checkpoint_resume":
            items[name] = {
                "passed": False,
                "comparisons": {key: False for key in _RESUME_KEYS},
            }
        else:
            items[name] = {"passed": False, "comparisons": []}
    return {
        "schema_version": 1,
        "case": case_name,
        "config_path": config_path,
        "precision": precision,
        "items": items,
        "overall_pass": False,
    }


def _canonical_report_json(report: Mapping[str, Any]) -> str:
    """Validate and serialize the only stdout payload."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise ValueError("native report has an invalid top-level key set")
    if report["schema_version"] != 1 or type(report["schema_version"]) is not int:
        raise ValueError("native report schema_version must be integer 1")
    for key in ("case", "config_path", "precision"):
        if not isinstance(report[key], str) or not report[key]:
            raise ValueError(f"native report {key} must be a non-empty string")
    if type(report["overall_pass"]) is not bool:
        raise TypeError("native report overall_pass must be bool")
    items = report["items"]
    if not isinstance(items, Mapping) or set(items) != set(_ITEM_KEYS):
        raise ValueError("native report has an invalid item key set")
    for name, item in items.items():
        if not isinstance(item, Mapping) or set(item) != {
            "passed",
            "comparisons",
        }:
            raise ValueError(f"native report item {name} has invalid keys")
        if type(item["passed"]) is not bool:
            raise TypeError(f"native report item {name}.passed must be bool")
        comparisons = item["comparisons"]
        if name == "checkpoint_resume":
            if (
                not isinstance(comparisons, Mapping)
                or set(comparisons) != set(_RESUME_KEYS)
                or any(type(value) is not bool for value in comparisons.values())
            ):
                raise ValueError("checkpoint_resume comparisons are invalid")
            expected_pass = all(comparisons.values())
            if item["passed"] is not expected_pass:
                raise ValueError(
                    "checkpoint_resume passed disagrees with comparisons"
                )
            continue
        if not isinstance(comparisons, list):
            raise TypeError(f"native report item {name}.comparisons must be a list")
        names: list[str] = []
        for comparison in comparisons:
            if not isinstance(comparison, Mapping) or set(comparison) != _COMPARISON_KEYS:
                raise ValueError("native tensor comparison has invalid keys")
            tensor_name = comparison["tensor_name"]
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError("tensor_name must be a non-empty string")
            names.append(tensor_name)
            if (
                type(comparison["shape"]) is not list
                or any(type(size) is not int or size < 0 for size in comparison["shape"])
            ):
                raise ValueError("comparison shape must contain non-negative integers")
            if not isinstance(comparison["dtype"], str):
                raise TypeError("comparison dtype must be a string")
            for scalar in ("rtol", "atol", "max_abs_error", "max_rel_error"):
                value = comparison[scalar]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"comparison {scalar} must be finite/non-negative")
            if type(comparison["passed"]) is not bool:
                raise TypeError("comparison passed must be bool")
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("comparisons must be uniquely sorted by tensor_name")
        expected_pass = bool(comparisons) and all(
            comparison["passed"] for comparison in comparisons
        )
        if item["passed"] is not expected_pass:
            raise ValueError(f"native report item {name}.passed is inconsistent")
    expected_overall = all(item["passed"] for item in items.values())
    if report["overall_pass"] is not expected_overall:
        raise ValueError("native report overall_pass is inconsistent with items")
    return json.dumps(
        dict(report),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _comparison(
    tensor_name: str,
    left: Any,
    right: Any,
    *,
    rtol: float = _CUDA_RTOL,
    atol: float = _CUDA_ATOL,
) -> dict[str, Any]:
    import torch

    left_tensor = torch.as_tensor(left).detach().cpu().contiguous()
    right_tensor = torch.as_tensor(right).detach().cpu().contiguous()
    shape_equal = tuple(left_tensor.shape) == tuple(right_tensor.shape)
    dtype_equal = left_tensor.dtype == right_tensor.dtype
    finite = True
    if left_tensor.is_floating_point() or left_tensor.is_complex():
        finite = bool(torch.isfinite(left_tensor).all())
    if right_tensor.is_floating_point() or right_tensor.is_complex():
        finite = finite and bool(torch.isfinite(right_tensor).all())
    if shape_equal and dtype_equal and finite:
        if left_tensor.is_floating_point() or left_tensor.is_complex():
            delta = (left_tensor - right_tensor).abs()
            max_abs = float(delta.max()) if delta.numel() else 0.0
            denominator = right_tensor.abs().clamp_min(torch.finfo(right_tensor.dtype).tiny)
            max_rel = float((delta / denominator).max()) if delta.numel() else 0.0
            passed = bool(torch.allclose(left_tensor, right_tensor, rtol=rtol, atol=atol))
        else:
            passed = bool(torch.equal(left_tensor, right_tensor))
            max_abs = 0.0 if passed else 1.0
            max_rel = max_abs
            rtol = 0.0
            atol = 0.0
    else:
        passed = False
        max_abs = 0.0 if finite else float(sys.float_info.max)
        max_rel = max_abs
    return {
        "tensor_name": tensor_name,
        "shape": list(left_tensor.shape),
        "dtype": str(left_tensor.dtype),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": float(max_abs),
        "max_rel_error": float(max_rel),
        "passed": passed,
    }


def _comparison_item(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    rtol: float = _CUDA_RTOL,
    atol: float = _CUDA_ATOL,
) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    comparisons: list[dict[str, Any]] = []
    for key in keys:
        if key not in left or key not in right:
            comparisons.append(
                {
                    "tensor_name": key,
                    "shape": [],
                    "dtype": "missing",
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "max_abs_error": float(sys.float_info.max),
                    "max_rel_error": float(sys.float_info.max),
                    "passed": False,
                }
            )
        else:
            comparisons.append(
                _comparison(key, left[key], right[key], rtol=rtol, atol=atol)
            )
    return {
        "passed": bool(comparisons) and all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }


def _load_case(case_path: Path) -> dict[str, Any]:
    value = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _CASE_KEYS:
        raise ValueError("native case has an invalid key set")
    if value["schema_version"] != 1:
        raise ValueError("native case schema_version must be 1")
    if not isinstance(value["prompt"], str) or not value["prompt"].strip():
        raise ValueError("native case prompt must be non-empty")
    if (
        type(value["seed"]) is not int
        or not 0 <= value["seed"] <= 0xFFFF_FFFF
    ):
        raise ValueError("native case seed must be uint32")
    if value["logical_step"] != 0:
        raise ValueError("native case logical_step must be zero")
    for key in ("reward_values", "expected_advantages"):
        values = value[key]
        if (
            type(values) is not list
            or len(values) < 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in values
            )
        ):
            raise ValueError(
                f"native case {key} must be a finite scalar list"
            )
    rewards = tuple(float(item) for item in value["reward_values"])
    reward_mean = sum(rewards) / len(rewards)
    if sum((item - reward_mean) ** 2 for item in rewards) == 0.0:
        raise ValueError("native case rewards must have non-zero variance")
    return value


def _missing_dependencies() -> tuple[str, ...]:
    return tuple(
        name
        for name in _DEPENDENCIES
        if importlib.util.find_spec(name) is None
    )


def _resolve_setup(repo_root: Path, case: dict[str, Any]):
    import visual_rl as vr

    relative_config = Path(case["config_path"])
    if relative_config.is_absolute() or ".." in relative_config.parts:
        raise ValueError("config_path must be repository-relative")
    config_path = (repo_root / relative_config).resolve()
    if not config_path.is_relative_to(repo_root.resolve()):
        raise ValueError("config_path escapes the repository root")
    config = vr.load(config_path).resolve()
    if config.model.name != "sd3_tempflow":
        raise ValueError("native case requires model sd3_tempflow")
    if config.rollout.name != "full_trajectory":
        raise ValueError("native case requires full_trajectory")
    if config.algorithm.name != "grpo":
        raise ValueError("native case requires grpo")
    if float(config.algorithm.params["beta"]) <= 0.0:
        raise ValueError("native case requires beta > 0")
    if config.algorithm.advantage.epsilon != 1.0e-4:
        raise ValueError("native case requires advantage epsilon 1e-4")
    if config.runtime.precision != "fp32":
        raise ValueError("native case requires fp32")
    if config.optimizer.max_grad_norm != 1.0:
        raise ValueError("native case requires max_grad_norm 1.0")
    if (
        config.runtime.distributed.mode != "single"
        or config.runtime.distributed.device != "cuda"
    ):
        raise ValueError("native case requires single CUDA")
    group_size = int(config.rollout.params["samples_per_prompt"])
    transition_count = int(config.rollout.params["num_steps"])
    microbatch_size = config.runtime.update_microbatch_size
    if len(case["reward_values"]) != group_size:
        raise ValueError("native reward vector does not match group size")
    if len(case["expected_advantages"]) != group_size:
        raise ValueError("native advantage vector does not match group size")
    if microbatch_size is None or group_size % microbatch_size:
        raise ValueError("group size must divide into update microbatches")
    if transition_count < 1:
        raise ValueError("native case requires at least one transition")
    reference_repo = (repo_root / "reference_code/TempFlow-GRPO-main").resolve()
    required_files = (
        reference_repo / "scripts" / "train_sd3.py",
        reference_repo / "flow_grpo" / "stat_tracking.py",
        reference_repo
        / "flow_grpo"
        / "diffusers_patch"
        / "sd3_pipeline_with_logprob.py",
        reference_repo
        / "flow_grpo"
        / "diffusers_patch"
        / "sd3_sde_with_logprob.py",
    )
    missing_files = tuple(path for path in required_files if not path.is_file())
    if missing_files:
        raise FileNotFoundError(
            "reference repo is missing required native modules: "
            + ", ".join(path.relative_to(reference_repo).as_posix() for path in missing_files)
        )
    return config, _NativeComputeLogProbView.from_resolved(config), reference_repo


@contextmanager
def _scoped_native_helpers(reference_repo: Path) -> Iterator[_NativeHelpers]:
    """Import the actual reference script without retaining its modules."""

    module_name = "_visualrl_native_flow_grpo_train_sd3"
    script = reference_repo / "scripts" / "train_sd3.py"
    previous = sys.modules.pop(module_name, None)
    with _reference_repo_import_path(reference_repo):
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot create native train_sd3 import spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            try:
                compute_text = module.compute_text_embeddings
                compute_log = module.compute_log_prob
                pipeline = module.pipeline_with_logprob
                tracker = module.PerPromptStatTracker
            except AttributeError as exc:
                raise RuntimeError(
                    "native train_sd3 callable surface is incomplete"
                ) from exc
            for name, value in (
                ("compute_text_embeddings", compute_text),
                ("compute_log_prob", compute_log),
                ("pipeline_with_logprob", pipeline),
                ("PerPromptStatTracker", tracker),
            ):
                if not callable(value):
                    raise RuntimeError(f"native train_sd3 is missing callable {name}")
            if tuple(inspect.signature(compute_text).parameters) != (
                "prompt",
                "text_encoders",
                "tokenizers",
                "max_sequence_length",
                "device",
            ):
                raise RuntimeError("native compute_text_embeddings signature drifted")
            if tuple(inspect.signature(compute_log).parameters) != (
                "transformer",
                "pipeline",
                "sample",
                "j",
                "embeds",
                "pooled_embeds",
                "config",
            ):
                raise RuntimeError("native compute_log_prob signature drifted")
            yield _NativeHelpers(
                compute_text_embeddings=compute_text,
                compute_log_prob=compute_log,
                pipeline_with_logprob=pipeline,
                tracker_type=tracker,
            )
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous


@contextmanager
def _reference_repo_import_path(repo_root: Path) -> Iterator[Path]:
    if not repo_root.is_absolute() or not repo_root.is_dir():
        raise ValueError("native reference repo must be one absolute directory")
    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "flow_grpo" or name.startswith("flow_grpo.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(repo_root))
    try:
        yield repo_root
    finally:
        sys.path[:] = previous_path
        for name, module in tuple(sys.modules.items()):
            module_file = getattr(module, "__file__", None)
            if not (name == "flow_grpo" or name.startswith("flow_grpo.")):
                continue
            if module_file is None:
                sys.modules.pop(name, None)
                continue
            resolved = Path(module_file).resolve()
            if resolved == repo_root or repo_root in resolved.parents:
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def _native_tracker_advantages(
    tracker_type: type[Any],
    *,
    prompt: str,
    rewards: Sequence[float],
) -> Any:
    import numpy as np

    tracker = tracker_type(False)
    advantages = tracker.update([prompt] * len(rewards), list(rewards))
    tracker.clear()
    value = np.asarray(advantages, dtype=np.float64)
    if value.shape != (len(rewards),) or not np.isfinite(value).all():
        raise RuntimeError("native PerPromptStatTracker returned invalid advantages")
    try:
        stats = tracker.stats
    except AttributeError as exc:
        raise RuntimeError(
            "native PerPromptStatTracker does not expose stats"
        ) from exc
    if stats != {}:
        raise RuntimeError("native PerPromptStatTracker.clear() did not clear stats")
    return value


def _canonical_permutation(
    *,
    batch_size: int,
    seed: int,
    device: Any,
):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randperm(batch_size, generator=generator, device=device)


def _transition_slice(batch: Any, rows: Sequence[int], step: int):
    micro = batch.slice(tuple(int(row) for row in rows))
    if not 0 <= step < micro.transition_count:
        raise IndexError("transition step is out of range")
    updates = {
        name: getattr(micro, name)[:, step : step + 1]
        for name in (
            "latents",
            "next_latents",
            "timesteps",
            "old_log_probs",
            "transition_mask",
        )
    }
    if micro.transition_std_dev is not None:
        updates["transition_std_dev"] = micro.transition_std_dev[
            :, step : step + 1
        ]
    return micro.replace(**updates)


def _loss_input_slice(inputs: Any, rows: Sequence[int], step: int):
    sliced = inputs.slice(tuple(int(row) for row in rows))
    return replace(
        sliced,
        base_advantage=sliced.base_advantage[:, step : step + 1],
        algorithm_weight=sliced.algorithm_weight[:, step : step + 1],
        active_mask=sliced.active_mask[:, step : step + 1],
    )


def _set_policy_residency(adapter: Any, *, active: bool) -> None:
    """Keep only the parity branch being evaluated resident on CUDA."""

    import torch

    if torch.device(adapter.device).type != "cuda":
        raise RuntimeError("native parity policy swapping requires CUDA")
    if active:
        adapter._activate_policy_module()
        if not adapter._policy_active:
            raise RuntimeError("native parity policy activation did not complete")
        return
    if adapter._text_encoders_active or adapter._vae_active:
        raise RuntimeError("cannot offload parity policy during a frozen-module phase")
    if adapter._policy_active:
        adapter.train_module.to("cpu")
        adapter._policy_active = False
        torch.cuda.empty_cache()
    if adapter._policy_active:
        raise RuntimeError("native parity policy offload did not complete")


def _activate_exclusive_policy(active: Any, inactive: Any) -> None:
    """Rebuild the same single-policy CUDA residency for every branch."""

    _set_policy_residency(inactive, active=False)
    _set_policy_residency(active, active=False)
    _set_policy_residency(active, active=True)


def _clone_named_parameters(
    source: Sequence[tuple[str, Any]],
    target: Sequence[tuple[str, Any]],
) -> None:
    import torch

    source_names = tuple(name for name, _parameter in source)
    target_names = tuple(name for name, _parameter in target)
    if source_names != target_names:
        raise RuntimeError("VisualRL/native trainable parameter names/order differ")
    with torch.no_grad():
        for (name, left), (_right_name, right) in zip(
            source,
            target,
            strict=True,
        ):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"trainable parameter topology differs at {name}")
            right.copy_(left)


def _named_tensor_snapshot(named: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in named
    }


def _restore_named_parameters(
    named: Sequence[tuple[str, Any]],
    snapshot: Mapping[str, Any],
) -> None:
    import torch

    if tuple(name for name, _parameter in named) != tuple(snapshot):
        raise RuntimeError("parameter snapshot topology changed")
    with torch.no_grad():
        for name, parameter in named:
            value = snapshot[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype:
                raise RuntimeError(f"parameter snapshot drifted at {name}")
            parameter.copy_(value.to(device=parameter.device))


def _build_adamw(named: Sequence[tuple[str, Any]], optimizer_config: Any):
    import torch

    return torch.optim.AdamW(
        [parameter for _name, parameter in named],
        lr=float(optimizer_config.learning_rate),
        betas=(
            float(optimizer_config.adam_beta1),
            float(optimizer_config.adam_beta2),
        ),
        eps=float(optimizer_config.adam_epsilon),
        weight_decay=float(optimizer_config.adam_weight_decay),
    )


def _run_update_window(
    *,
    named_parameters: Sequence[tuple[str, Any]],
    optimizer: Any,
    initial_parameters: Mapping[str, Any],
    slots: Sequence[tuple[Sequence[int], int]],
    evaluate_slot: Callable[[Sequence[int], int], Mapping[str, Any]],
    batch_size: int,
    transition_count: int,
    max_grad_norm: float,
) -> _UpdateTrace:
    """Execute exactly one native K-slot synchronized AdamW window."""

    import torch

    if not slots:
        raise ValueError("update window requires at least one slot")
    optimizer.zero_grad(set_to_none=True)
    zero_grad_count = 1
    records: list[tuple[tuple[int, ...], int, Mapping[str, Any]]] = []
    slot_rng_before: list[_RngSnapshot] = []
    slot_rng_after: list[_RngSnapshot] = []
    for rows, step in slots:
        slot_rng_before.append(_RngSnapshot.capture())
        output = evaluate_slot(rows, step)
        loss = output["total_loss"]
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise RuntimeError("slot total_loss must be a scalar tensor")
        (loss / len(slots)).backward()
        slot_rng_after.append(_RngSnapshot.capture())
        records.append((tuple(int(row) for row in rows), step, output))

    pre_clip: dict[str, Any] = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for trainable parameter {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"non-finite gradient for {name}")
        pre_clip[f"pre_clip/{name}"] = parameter.grad.detach().cpu().clone()
    torch.nn.utils.clip_grad_norm_(
        [parameter for _name, parameter in named_parameters],
        max_grad_norm,
    )
    post_clip = {
        f"post_clip/{name}": parameter.grad.detach().cpu().clone()
        for name, parameter in named_parameters
    }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    zero_grad_count += 1

    deltas = {
        name: parameter.detach().cpu() - initial_parameters[name]
        for name, parameter in named_parameters
    }
    new_log = torch.empty(
        (batch_size, transition_count),
        dtype=records[0][2]["new_log_prob"].dtype,
        device=records[0][2]["new_log_prob"].device,
    )
    first_mean = records[0][2]["current_mean"]
    current_mean = torch.empty(
        (batch_size, transition_count, *first_mean.shape[2:]),
        dtype=first_mean.dtype,
        device=first_mean.device,
    )
    reference_mean = torch.empty_like(current_mean)
    first_std = records[0][2]["transition_std"]
    transition_std = torch.empty(
        (batch_size, transition_count, *first_std.shape[2:]),
        dtype=first_std.dtype,
        device=first_std.device,
    )
    policy_losses = []
    reference_kls = []
    total_losses = []
    for rows, step, output in records:
        index = torch.tensor(rows, dtype=torch.long, device=new_log.device)
        new_log[index, step] = output["new_log_prob"][:, 0]
        current_mean[index, step] = output["current_mean"][:, 0]
        reference_mean[index, step] = output["reference_mean"][:, 0]
        transition_std[index, step] = output["transition_std"][:, 0]
        policy_losses.append(output["policy_loss"].detach())
        reference_kls.append(output["reference_kl"].detach())
        total_losses.append(output["total_loss"].detach())
    return _UpdateTrace(
        current_log_prob=new_log.detach(),
        current_mean=current_mean.detach(),
        reference_mean=reference_mean.detach(),
        transition_std=transition_std.detach(),
        policy_loss=torch.stack(policy_losses).mean(),
        reference_kl=torch.stack(reference_kls).mean(),
        total_loss=torch.stack(total_losses).mean(),
        pre_clip_gradients=pre_clip,
        post_clip_gradients=post_clip,
        parameter_delta=deltas,
        slot_rng_before=tuple(slot_rng_before),
        slot_rng_after=tuple(slot_rng_after),
        backward_count=len(slots),
        clip_count=1,
        step_count=1,
        zero_grad_count=zero_grad_count,
    )


def _run_isolated_branches(
    snapshot: _RngSnapshot,
    branches: Mapping[str, Callable[[], Any]],
    order: Sequence[str],
) -> dict[str, tuple[Any, _RngSnapshot]]:
    """Run branch callables from one RNG state and restore caller state."""

    if set(order) != set(branches) or len(order) != len(branches):
        raise ValueError("branch order must contain every branch exactly once")
    outer = _RngSnapshot.capture()
    results: dict[str, tuple[Any, _RngSnapshot]] = {}
    try:
        for name in order:
            snapshot.restore()
            value = branches[name]()
            results[name] = (value, _RngSnapshot.capture())
        return results
    finally:
        outer.restore()


def _deep_equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_deep_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_deep_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _prompt_payload_native(
    helpers: _NativeHelpers,
    adapter: Any,
    prompts: tuple[str, ...],
) -> dict[str, Any]:
    adapter._activate_text_encoders_for_prompt()
    try:
        encoders = [
            adapter.pipeline.text_encoder,
            adapter.pipeline.text_encoder_2,
            adapter.pipeline.text_encoder_3,
        ]
        tokenizers = [
            adapter.pipeline.tokenizer,
            adapter.pipeline.tokenizer_2,
            adapter.pipeline.tokenizer_3,
        ]
        positive, pooled = helpers.compute_text_embeddings(
            list(prompts),
            encoders,
            tokenizers,
            adapter.max_sequence_length,
            adapter.device,
        )
        negative, negative_pooled = helpers.compute_text_embeddings(
            [""] * len(prompts),
            encoders,
            tokenizers,
            adapter.max_sequence_length,
            adapter.device,
        )
    finally:
        adapter._offload_text_encoders()
        adapter._activate_policy_module()
    return {
        "prompt_embeds": positive,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": negative,
        "negative_pooled_prompt_embeds": negative_pooled,
    }


def _prepare_initial_latents(
    *,
    visual_adapter: Any,
    native_adapter: Any,
    visual_prompt_embeds: Any,
    native_prompt_embeds: Any,
    batch_size: int,
    seed: int,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    """Consume each explicit stream once and freeze its shared post-state."""

    import torch

    generators = (
        torch.Generator(device=device).manual_seed(seed),
        torch.Generator(device=device).manual_seed(seed),
    )

    def prepare(adapter: Any, embeds: Any, generator: Any):
        return adapter.pipeline.prepare_latents(
            batch_size,
            adapter.pipeline.transformer.config.in_channels,
            adapter.resolution,
            adapter.resolution,
            embeds.dtype,
            device,
            generator,
            None,
        )

    visual_latent = prepare(
        visual_adapter,
        visual_prompt_embeds,
        generators[0],
    )
    native_latent = prepare(
        native_adapter,
        native_prompt_embeds,
        generators[1],
    )
    visual_state = generators[0].get_state().clone()
    native_state = generators[1].get_state().clone()
    return visual_latent, native_latent, visual_state, native_state


def _run_with_default_torch_rng(
    *,
    device: Any,
    state: Any,
    callback: Callable[[], Any],
) -> tuple[Any, Any]:
    """Temporarily bridge one explicit-generator state to the default stream."""

    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda":
        index = (
            torch.cuda.current_device()
            if resolved.index is None
            else resolved.index
        )
        with torch.random.fork_rng(devices=[index]):
            torch.cuda.set_rng_state(state.clone(), device=resolved)
            value = callback()
            post = torch.cuda.get_rng_state(resolved).clone()
    elif resolved.type == "cpu":
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state.clone())
            value = callback()
            post = torch.get_rng_state().clone()
    else:
        raise ValueError("native RNG bridge supports only CPU/CUDA")
    return value, post


def _pipeline_sde_globals(pipeline_function: Any) -> tuple[dict[str, Any], str]:
    seen: set[int] = set()
    current = pipeline_function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        function = getattr(current, "__func__", current)
        namespace = getattr(function, "__globals__", None)
        if isinstance(namespace, dict):
            for key in ("sde_step_with_logprob", "sd3_sde_step_with_logprob"):
                if key in namespace and callable(namespace[key]):
                    return namespace, key
        try:
            current = function.__wrapped__
        except AttributeError:
            break
    raise RuntimeError("native pipeline does not expose its SDE helper globals")


@contextmanager
def _trace_sde_draws(
    pipeline_function: Any,
) -> Iterator[list[Any]]:
    """Record every stochastic SDE standard-normal draw from real outputs."""

    import torch

    namespace, key = _pipeline_sde_globals(pipeline_function)
    original = namespace[key]
    draws: list[Any] = []

    def traced(*args: Any, **kwargs: Any):
        deterministic = bool(
            kwargs.get("deterministic", kwargs.get("determistic", False))
        )
        previous = (
            args[4] if len(args) >= 5 else kwargs.get("prev_sample")
        )
        output = original(*args, **kwargs)
        if not deterministic and previous is None:
            if not isinstance(output, (tuple, list)) or len(output) < 4:
                raise RuntimeError("native SDE helper did not expose mean/std")
            sample = torch.as_tensor(output[0])
            mean = torch.as_tensor(output[2])
            std = torch.as_tensor(output[3])
            draw = (sample - mean) / std
            if not bool(torch.isfinite(draw).all()):
                raise RuntimeError("native SDE trace reconstructed non-finite draw")
            draws.append(draw.detach().cpu().contiguous().clone())
        return output

    namespace[key] = traced
    try:
        yield draws
    finally:
        if namespace[key] is traced:
            namespace[key] = original


def _rollout_visual(
    *,
    adapter: Any,
    rollout: Any,
    prompt: str,
    context: Any,
    canonical_latent: Any,
    transition_rng_state: Any,
) -> tuple[Any, Any, tuple[Any, ...]]:
    """Use the real RolloutEngine while injecting the frozen latent/stream."""

    import torch

    original_prepare = adapter._prepare_native_sd3_rollout
    original_run = adapter._run_full_pipeline
    generator_post_state: list[Any] = []

    def fixed_prepare(*, prompt_embeds: Any, num_steps: int, generator: Any):
        if generator is None:
            raise RuntimeError("VisualRL SD3 rollout did not provide a generator")
        timesteps, prepared = original_prepare(
            prompt_embeds=prompt_embeds,
            num_steps=num_steps,
            generator=generator,
        )
        if not torch.equal(generator.get_state(), transition_rng_state):
            raise RuntimeError("VisualRL initial-latent RNG state drifted")
        if not torch.equal(prepared, canonical_latent):
            raise RuntimeError("VisualRL rollout initial latent drifted")
        return timesteps, canonical_latent.clone()

    def fixed_run(**kwargs: Any):
        generator = kwargs.get("generator")
        if generator is None:
            raise RuntimeError("VisualRL SD3 rollout did not provide a generator")
        output = original_run(**kwargs)
        generator_post_state.append(generator.get_state().clone())
        return output

    adapter._prepare_native_sd3_rollout = fixed_prepare
    adapter._run_full_pipeline = fixed_run
    try:
        with _trace_sde_draws(original_run) as draws:
            batch = rollout.sample(
                adapter=adapter,
                prompts=(prompt,),
                metadata=(
                    {
                        "dataset_epoch": 0,
                        "dataset_index": 0,
                        "prompt_id": "native-parity-prompt",
                        "group_id": "native-parity-group",
                    },
                ),
                context=context,
            )
    finally:
        adapter._prepare_native_sd3_rollout = original_prepare
        adapter._run_full_pipeline = original_run
    if len(generator_post_state) != 1:
        raise RuntimeError("VisualRL rollout did not expose one generator post-state")
    return batch, generator_post_state[0], tuple(draws)


def _rollout_native(
    *,
    helpers: _NativeHelpers,
    adapter: Any,
    payload: Mapping[str, Any],
    canonical_latent: Any,
    transition_rng_state: Any,
    batch_size: int,
    transition_count: int,
    guidance_scale: float,
) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    """Call the actual native pipeline with its CUDA-default RNG bridge."""

    import torch

    adapter.train_module.eval()
    pipeline = _NativePipelineProxy(adapter)

    def run():
        with torch.no_grad():
            return helpers.pipeline_with_logprob(
                pipeline,
                prompt_embeds=payload["prompt_embeds"],
                pooled_prompt_embeds=payload["pooled_prompt_embeds"],
                negative_prompt_embeds=payload["negative_prompt_embeds"],
                negative_pooled_prompt_embeds=payload[
                    "negative_pooled_prompt_embeds"
                ],
                num_inference_steps=transition_count,
                guidance_scale=guidance_scale,
                latents=canonical_latent.clone(),
                output_type="pt",
                height=adapter.resolution,
                width=adapter.resolution,
                return_dict=False,
                max_sequence_length=adapter.max_sequence_length,
                kl_reward=0.0,
            )

    with _trace_sde_draws(helpers.pipeline_with_logprob) as draws:
        output, post_state = _run_with_default_torch_rng(
            device=adapter.device,
            state=transition_rng_state,
            callback=run,
        )
    if not isinstance(output, (tuple, list)) or len(output) != 4:
        raise RuntimeError("native SD3 pipeline returned an invalid result")
    _media, states_raw, log_probs_raw, _kl = output
    if (
        not isinstance(states_raw, (tuple, list))
        or len(states_raw) != transition_count + 1
        or not isinstance(log_probs_raw, (tuple, list))
        or len(log_probs_raw) != transition_count
    ):
        raise RuntimeError("native rollout has an invalid transition count")
    states = torch.stack(tuple(states_raw), dim=1)
    old_log_probs = torch.stack(
        tuple(torch.as_tensor(value).reshape(batch_size) for value in log_probs_raw),
        dim=1,
    )
    timesteps = (
        torch.as_tensor(adapter.pipeline.scheduler.timesteps)
        .to(device=old_log_probs.device)
        .reshape(1, transition_count)
        .expand(batch_size, transition_count)
        .clone()
    )
    return (
        {
            "latents": states[:, :-1].detach(),
            "next_latents": states[:, 1:].detach(),
            "old_log_probs": old_log_probs.detach(),
            "timesteps": timesteps.detach(),
            "transition_mask": torch.ones_like(
                old_log_probs,
                dtype=torch.bool,
            ),
            **{name: value.detach() for name, value in payload.items()},
        },
        post_state,
        tuple(draws),
    )


def _reward_batch(batch: Any, rewards: Sequence[float]):
    import torch

    from visual_rl.core.types import RewardBatch

    values = torch.tensor(rewards, dtype=torch.float32).contiguous()
    empty_rows = tuple({} for _ in rewards)
    return RewardBatch(
        sample_id=batch.sample_id,
        raw={"native_case": values},
        weighted={"native_case": values.clone()},
        weighted_total=values.clone(),
        valid_mask=torch.ones(len(rewards), dtype=torch.bool),
        shared_metadata={"native_case": {}},
        sample_metadata={"native_case": empty_rows},
    )


def _aligned_visual_objective(
    *,
    adapter: Any,
    objective: Any,
    batch: Any,
    inputs: Any,
) -> Mapping[str, Any]:
    from visual_rl.optimizers.update_engine import UpdateEngine

    stats = adapter.recompute_policy_stats(batch, require_reference=True)
    aligned_batch, aligned_inputs, aligned_stats = (
        UpdateEngine._aligned_objective_views(batch, inputs, stats)
    )
    output = objective(aligned_batch, aligned_inputs, aligned_stats)
    return {
        "new_log_prob": aligned_stats.new_log_probs,
        "current_mean": aligned_stats.current_transition_mean,
        "reference_mean": aligned_stats.reference_transition_mean,
        "transition_std": aligned_stats.transition_std,
        "policy_loss": output.policy_loss,
        "reference_kl": output.reference_kl,
        "total_loss": output.loss,
    }


def _native_slot_objective(
    *,
    helpers: _NativeHelpers,
    adapter: Any,
    native_data: Mapping[str, Any],
    view: _NativeComputeLogProbView,
    algorithm: Any,
    native_advantage: Any,
    rows: Sequence[int],
    step: int,
) -> Mapping[str, Any]:
    import torch

    index = torch.tensor(rows, dtype=torch.long, device=adapter.device)
    sample = {
        "latents": native_data["latents"].index_select(0, index)[
            :, step : step + 1
        ],
        "next_latents": native_data["next_latents"].index_select(0, index)[
            :, step : step + 1
        ],
        "timesteps": native_data["timesteps"].index_select(0, index)[
            :, step : step + 1
        ],
        "log_probs": native_data["old_log_probs"].index_select(0, index)[
            :, step : step + 1
        ],
    }
    positive = native_data["prompt_embeds"].index_select(0, index)
    pooled = native_data["pooled_prompt_embeds"].index_select(0, index)
    if view.train.cfg:
        positive = torch.cat(
            (
                native_data["negative_prompt_embeds"].index_select(0, index),
                positive,
            )
        )
        pooled = torch.cat(
            (
                native_data["negative_pooled_prompt_embeds"].index_select(
                    0,
                    index,
                ),
                pooled,
            )
        )
    _prev, new_log, current_mean, std = helpers.compute_log_prob(
        adapter.train_module,
        adapter.pipeline,
        sample,
        0,
        positive,
        pooled,
        view,
    )
    with torch.no_grad(), adapter._disable_lora_reference():
        _ref_prev, _ref_log, reference_mean, reference_std = (
            helpers.compute_log_prob(
                adapter.train_module,
                adapter.pipeline,
                sample,
                0,
                positive,
                pooled,
                view,
            )
        )
    if not torch.equal(std.detach(), reference_std.detach()):
        raise RuntimeError("native current/reference transition std differs")
    new_log = torch.as_tensor(new_log).reshape(len(rows), 1)
    current_mean = torch.as_tensor(current_mean).unsqueeze(1)
    reference_mean = torch.as_tensor(reference_mean).unsqueeze(1).detach()
    std = torch.as_tensor(std).unsqueeze(1).detach()
    advantage = (
        native_advantage.index_select(0, index)
        .clamp(-algorithm.adv_clip_max, algorithm.adv_clip_max)
        .reshape(len(rows), 1)
    )
    oracle = NativeFlowReferenceOracle.evaluate(
        old_log_probs=sample["log_probs"],
        new_log_probs=new_log,
        advantages=advantage,
        current_mean=current_mean,
        reference_mean=reference_mean,
        std_dev=std,
        clip_range=float(algorithm.clip_range),
        beta=float(algorithm.beta),
    )
    return {
        "new_log_prob": new_log,
        "current_mean": current_mean,
        "reference_mean": reference_mean,
        "transition_std": std,
        **oracle,
    }


def _trace_items(
    visual: _UpdateTrace,
    native: _UpdateTrace,
) -> dict[str, dict[str, Any]]:
    if (
        visual.backward_count != native.backward_count
        or visual.clip_count != native.clip_count
        or visual.step_count != native.step_count
        or visual.zero_grad_count != native.zero_grad_count
    ):
        raise RuntimeError("VisualRL/native update cadence differs")
    for field in ("slot_rng_before", "slot_rng_after"):
        visual_states = getattr(visual, field)
        native_states = getattr(native, field)
        if (
            len(visual_states) != len(native_states)
            or not all(
                left.exactly_equal(right)
                for left, right in zip(
                    visual_states,
                    native_states,
                    strict=True,
                )
            )
        ):
            raise RuntimeError(
                f"VisualRL/native per-slot RNG trace differs at {field}"
            )
    return {
        "current_log_prob": _comparison_item(
            {"current_log_prob": visual.current_log_prob},
            {"current_log_prob": native.current_log_prob},
        ),
        "transition_statistics": _comparison_item(
            {
                "current_transition_mean": visual.current_mean,
                "reference_transition_mean": visual.reference_mean,
                "transition_std": visual.transition_std,
            },
            {
                "current_transition_mean": native.current_mean,
                "reference_transition_mean": native.reference_mean,
                "transition_std": native.transition_std,
            },
        ),
        "policy_loss": _comparison_item(
            {"policy_loss": visual.policy_loss},
            {"policy_loss": native.policy_loss},
        ),
        "reference_kl": _comparison_item(
            {"reference_kl": visual.reference_kl},
            {"reference_kl": native.reference_kl},
        ),
        "total_loss": _comparison_item(
            {"total_loss": visual.total_loss},
            {"total_loss": native.total_loss},
        ),
        "gradient": _comparison_item(
            {
                **visual.pre_clip_gradients,
                **visual.post_clip_gradients,
            },
            {
                **native.pre_clip_gradients,
                **native.post_clip_gradients,
            },
        ),
        "parameter_delta": _comparison_item(
            visual.parameter_delta,
            native.parameter_delta,
        ),
    }


def _update_traces_exactly_equal(
    left: _UpdateTrace,
    right: _UpdateTrace,
) -> bool:
    scalar_and_tensor_fields_equal = all(
        _deep_equal(getattr(left, field), getattr(right, field))
        for field in (
            "current_log_prob",
            "current_mean",
            "reference_mean",
            "transition_std",
            "policy_loss",
            "reference_kl",
            "total_loss",
            "pre_clip_gradients",
            "post_clip_gradients",
            "parameter_delta",
            "backward_count",
            "clip_count",
            "step_count",
            "zero_grad_count",
        )
    )
    rng_fields_equal = all(
        len(getattr(left, field)) == len(getattr(right, field))
        and all(
            a.exactly_equal(b)
            for a, b in zip(
                getattr(left, field),
                getattr(right, field),
                strict=True,
            )
        )
        for field in ("slot_rng_before", "slot_rng_after")
    )
    return scalar_and_tensor_fields_equal and rng_fields_equal


def _update_trace_difference_summary(
    left: _UpdateTrace,
    right: _UpdateTrace,
    *,
    limit: int = 12,
) -> dict[str, Any]:
    """Summarize order sensitivity without discarding numerical magnitude."""

    if limit < 1:
        raise ValueError("update trace summary limit must be positive")
    tensor_values: list[dict[str, Any]] = []
    for field in (
        "current_log_prob",
        "current_mean",
        "reference_mean",
        "transition_std",
        "policy_loss",
        "reference_kl",
        "total_loss",
    ):
        tensor_values.append(
            _comparison(field, getattr(left, field), getattr(right, field))
        )
    for field in (
        "pre_clip_gradients",
        "post_clip_gradients",
        "parameter_delta",
    ):
        comparison = _comparison_item(
            {
                f"{field}/{name}": value
                for name, value in getattr(left, field).items()
            },
            {
                f"{field}/{name}": value
                for name, value in getattr(right, field).items()
            },
        )
        tensor_values.extend(comparison["comparisons"])

    changed = [
        item
        for item in tensor_values
        if item["max_abs_error"] != 0.0 or not item["passed"]
    ]
    changed.sort(
        key=lambda item: (
            not item["passed"],
            item["max_abs_error"],
            item["max_rel_error"],
            item["tensor_name"],
        ),
        reverse=True,
    )
    groups: dict[str, dict[str, Any]] = {}
    for item in changed:
        group = item["tensor_name"].split("/", 1)[0]
        summary = groups.setdefault(
            group,
            {
                "changed_tensor_count": 0,
                "failed_tolerance_count": 0,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
            },
        )
        summary["changed_tensor_count"] += 1
        summary["failed_tolerance_count"] += int(not item["passed"])
        summary["max_abs_error"] = max(
            summary["max_abs_error"],
            item["max_abs_error"],
        )
        summary["max_rel_error"] = max(
            summary["max_rel_error"],
            item["max_rel_error"],
        )
    counter_fields = (
        "backward_count",
        "clip_count",
        "step_count",
        "zero_grad_count",
    )
    counters_equal = all(
        getattr(left, field) == getattr(right, field)
        for field in counter_fields
    )
    rng_equal = all(
        len(getattr(left, field)) == len(getattr(right, field))
        and all(
            a.exactly_equal(b)
            for a, b in zip(
                getattr(left, field),
                getattr(right, field),
                strict=True,
            )
        )
        for field in ("slot_rng_before", "slot_rng_after")
    )
    return {
        "exact": _update_traces_exactly_equal(left, right),
        "within_cuda_tolerance": bool(tensor_values)
        and all(item["passed"] for item in tensor_values)
        and counters_equal
        and rng_equal,
        "changed_tensor_count": len(changed),
        "failed_tolerance_count": sum(not item["passed"] for item in tensor_values),
        "counters_equal": counters_equal,
        "rng_equal": rng_equal,
        "groups": groups,
        "largest_differences": changed[:limit],
    }


def _rank_state_from_current_rng(device: Any):
    import numpy as np
    import torch

    from visual_rl.artifacts.checkpoint import RankState

    return RankState.from_rng(
        rank=0,
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state(device),
    )


def _next_step_input_projection(
    rollout_batch: Any,
    loss_inputs: Any,
) -> dict[str, Any]:
    return {
        "latents": rollout_batch.latents.detach().cpu().contiguous().clone(),
        "next_latents": (
            rollout_batch.next_latents.detach().cpu().contiguous().clone()
        ),
        "timesteps": rollout_batch.timesteps.detach().cpu().contiguous().clone(),
        "old_log_probs": (
            rollout_batch.old_log_probs.detach().cpu().contiguous().clone()
        ),
        "advantages": (
            loss_inputs.base_advantage.detach().cpu().contiguous().clone()
        ),
    }


def _compare_resume_projections(
    continuous: _ResumeProjection,
    resumed: _ResumeProjection,
) -> dict[str, bool]:
    if not isinstance(continuous, _ResumeProjection) or not isinstance(
        resumed,
        _ResumeProjection,
    ):
        raise TypeError("resume comparison requires two projections")
    return {
        "adapter_tensors": _deep_equal(
            continuous.adapter_tensors,
            resumed.adapter_tensors,
        ),
        "optimizer_state": _deep_equal(
            continuous.optimizer_state,
            resumed.optimizer_state,
        ),
        "grad_scaler_state": _deep_equal(
            continuous.grad_scaler_state,
            resumed.grad_scaler_state,
        ),
        "rng_state": (
            continuous.start_rng.exactly_equal(resumed.start_rng)
            and continuous.end_rng.exactly_equal(resumed.end_rng)
        ),
        "next_step_inputs": _deep_equal(
            continuous.next_step_inputs,
            resumed.next_step_inputs,
        ),
        "global_step": (
            continuous.global_step == resumed.global_step
        ),
        "non_timing_metrics": _deep_equal(
            continuous.non_timing_metrics,
            resumed.non_timing_metrics,
        ),
    }


def _resume_stage(label: str, callback: Callable[[], Any]) -> Any:
    """Attach one stable checkpoint/resume stage to low-level failures."""

    try:
        return callback()
    except BaseException as exc:
        raise RuntimeError(
            f"checkpoint/resume stage {label} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _resume_comparison(
    *,
    config: Any,
    adapter: Any,
    optimizer: Any,
    rollout_batch: Any,
    loss_inputs: Any,
    objective: Any,
    slots: Sequence[tuple[Sequence[int], int]],
    build_fresh_adapter: Callable[[], Any],
) -> dict[str, bool]:
    """Compare VisualRL two-step continuous with format-v5 1+resume+1."""

    import torch

    from visual_rl.artifacts.checkpoint import (
        TrainingContract,
        apply_training_state,
        read_and_validate_training_state,
        save_training_state,
    )

    contract = TrainingContract(algorithm="grpo", version=2)
    rank_state = _rank_state_from_current_rng(adapter.device)

    with tempfile.TemporaryDirectory(prefix="visualrl-native-parity-") as temp:
        checkpoint = Path(temp) / "checkpoint_000001"
        save_training_state(
            checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            global_step=1,
            training_contract=contract,
            rank_states=(rank_state,),
            writer_rank=0,
            writer_device=adapter.device,
        )
        first_step_rng = _RngSnapshot.capture()
        continuous_next_inputs = _next_step_input_projection(
            rollout_batch,
            loss_inputs,
        )
        continuous_start_step = 1
        continuous_initial = _named_tensor_snapshot(adapter.named_parameters())
        continuous_trace = _run_update_window(
            named_parameters=adapter.named_parameters(),
            optimizer=optimizer,
            initial_parameters=continuous_initial,
            slots=slots,
            evaluate_slot=lambda rows, step: _aligned_visual_objective(
                adapter=adapter,
                objective=objective,
                batch=_transition_slice(rollout_batch, rows, step),
                inputs=_loss_input_slice(loss_inputs, rows, step),
            ),
            batch_size=rollout_batch.batch_size,
            transition_count=rollout_batch.transition_count,
            max_grad_norm=float(config.optimizer.max_grad_norm),
        )
        continuous_parameters = _named_tensor_snapshot(adapter.named_parameters())
        continuous_optimizer = deepcopy(optimizer.state_dict())
        continuous_rng = _RngSnapshot.capture()
        continuous_metrics = {
            "policy_loss": float(continuous_trace.policy_loss),
            "reference_kl": float(continuous_trace.reference_kl),
            "total_loss": float(continuous_trace.total_loss),
        }
        continuous_global_step = continuous_start_step + 1
        continuous_scaler_state = None

        _set_policy_residency(adapter, active=False)
        fresh = _resume_stage("build_fresh_adapter", build_fresh_adapter)
        try:
            fresh_optimizer = _build_adamw(
                fresh.named_parameters(),
                config.optimizer,
            )
            validated = _resume_stage(
                "read_and_validate_training_state",
                lambda: read_and_validate_training_state(
                    checkpoint,
                    adapter=fresh,
                    optimizer=fresh_optimizer,
                    scaler=None,
                    expected_global_step=1,
                    expected_world_size=1,
                    expected_training_contract=contract,
                ),
            )
            _resume_stage(
                "apply_training_state",
                lambda: apply_training_state(
                    validated,
                    adapter=fresh,
                    optimizer=fresh_optimizer,
                    scaler=None,
                    optimizer_config=config.optimizer,
                    rank=0,
                ),
            )
            resumed_start_rng = _RngSnapshot.capture()
            resumed_next_inputs = _next_step_input_projection(
                rollout_batch,
                loss_inputs,
            )
            resumed_start_step = validated.global_step
            resumed_initial = _named_tensor_snapshot(fresh.named_parameters())
            resumed_trace = _resume_stage(
                "resumed_update",
                lambda: _run_update_window(
                    named_parameters=fresh.named_parameters(),
                    optimizer=fresh_optimizer,
                    initial_parameters=resumed_initial,
                    slots=slots,
                    evaluate_slot=lambda rows, step: _aligned_visual_objective(
                        adapter=fresh,
                        objective=objective,
                        batch=_transition_slice(rollout_batch, rows, step),
                        inputs=_loss_input_slice(loss_inputs, rows, step),
                    ),
                    batch_size=rollout_batch.batch_size,
                    transition_count=rollout_batch.transition_count,
                    max_grad_norm=float(config.optimizer.max_grad_norm),
                ),
            )
            resumed_parameters = _named_tensor_snapshot(fresh.named_parameters())
            resumed_optimizer = deepcopy(fresh_optimizer.state_dict())
            resumed_rng = _RngSnapshot.capture()
            resumed_metrics = {
                "policy_loss": float(resumed_trace.policy_loss),
                "reference_kl": float(resumed_trace.reference_kl),
                "total_loss": float(resumed_trace.total_loss),
            }
            resumed_global_step = resumed_start_step + 1
            resumed_scaler_state = None
        finally:
            fresh.close()
            torch.cuda.empty_cache()

    continuous_projection = _ResumeProjection(
        adapter_tensors=continuous_parameters,
        optimizer_state=continuous_optimizer,
        grad_scaler_state=continuous_scaler_state,
        start_rng=first_step_rng,
        end_rng=continuous_rng,
        next_step_inputs=continuous_next_inputs,
        global_step=continuous_global_step,
        non_timing_metrics=continuous_metrics,
    )
    resumed_projection = _ResumeProjection(
        adapter_tensors=resumed_parameters,
        optimizer_state=resumed_optimizer,
        grad_scaler_state=resumed_scaler_state,
        start_rng=resumed_start_rng,
        end_rng=resumed_rng,
        next_step_inputs=resumed_next_inputs,
        global_step=resumed_global_step,
        non_timing_metrics=resumed_metrics,
    )
    flags = _compare_resume_projections(
        continuous_projection,
        resumed_projection,
    )
    if continuous_global_step != 2 or resumed_global_step != 2:
        flags["global_step"] = False
    return flags


def _run_real_parity(
    *,
    repo_root: Path,
    case_path: Path,
    case: Mapping[str, Any],
    config: Any,
    view: _NativeComputeLogProbView,
    reference_repo: Path,
) -> dict[str, Any]:
    """Execute the fixed real-CUDA 14-item comparison."""

    import numpy as np
    import torch

    from visual_rl.core.types import RuntimeBuildContext, StepContext
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
    from visual_rl.optimizers.advantages import AdvantageComputer
    from visual_rl.optimizers.grpo import GRPOAlgorithm
    from visual_rl.optimizers.objective import PolicyObjective
    from visual_rl.rollout.full_trajectory import FullTrajectoryRollout

    _configure_cuda_determinism(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("real native parity requires an available CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("native parity requires the default float32 dtype")
    context = RuntimeBuildContext(
        rank=0,
        local_rank=device.index or 0,
        world_size=1,
        backend=None,
        device=device,
        precision="fp32",
    )
    batch_size = int(config.rollout.params["samples_per_prompt"])
    transition_count = int(config.rollout.params["num_steps"])
    microbatch_size = int(config.runtime.update_microbatch_size)
    if batch_size % microbatch_size:
        raise RuntimeError("native update batch is not microbatch-divisible")
    if float(config.model.params["guidance_scale"]) != view.sample.guidance_scale:
        raise RuntimeError("native guidance-scale view drifted")
    if view.sample.noise_level != 0.7:
        raise RuntimeError("native noise level must be exactly 0.7")

    def build_adapter() -> Any:
        return SD3TempFlowAdapter.from_config(config.model.params, context)

    _seed_native_case(int(case["seed"]))
    visual_adapter = build_adapter()
    _seed_native_case(int(case["seed"]))
    native_adapter = build_adapter()
    _set_policy_residency(native_adapter, active=False)
    native_closed = False
    try:
        visual_named = visual_adapter.named_parameters()
        native_named = native_adapter.named_parameters()
        _clone_named_parameters(visual_named, native_named)
        construction_rng = _RngSnapshot.capture()

        with _scoped_native_helpers(reference_repo) as helpers:
            prompts = (str(case["prompt"]),) * batch_size
            visual_prompt_values = visual_adapter._prompt_payload(prompts)
            visual_prompt = {
                name: value
                for name, value in zip(
                    (
                        "prompt_embeds",
                        "pooled_prompt_embeds",
                        "negative_prompt_embeds",
                        "negative_pooled_prompt_embeds",
                    ),
                    visual_prompt_values,
                    strict=True,
                )
            }
            _set_policy_residency(visual_adapter, active=False)
            native_prompt = _prompt_payload_native(
                helpers,
                native_adapter,
                prompts,
            )
            _set_policy_residency(native_adapter, active=False)
            prompt_item = _comparison_item(visual_prompt, native_prompt)

            (
                visual_initial,
                native_initial,
                visual_prepare_post,
                native_prepare_post,
            ) = _prepare_initial_latents(
                visual_adapter=visual_adapter,
                native_adapter=native_adapter,
                visual_prompt_embeds=visual_prompt["prompt_embeds"],
                native_prompt_embeds=native_prompt["prompt_embeds"],
                batch_size=batch_size,
                seed=int(case["seed"]),
                device=device,
            )
            initial_item = _comparison_item(
                {
                    "initial_latent": visual_initial,
                    "prepare_generator_post_state": visual_prepare_post,
                },
                {
                    "initial_latent": native_initial,
                    "prepare_generator_post_state": native_prepare_post,
                },
            )
            transition_rng_state = visual_prepare_post.clone()

            rollout = FullTrajectoryRollout.from_config(
                config.rollout.params,
                context,
            )
            step_context = StepContext(
                step=int(case["logical_step"]),
                seed=int(case["seed"]),
                rank=0,
                world_size=1,
            )
            construction_rng.restore()
            _activate_exclusive_policy(visual_adapter, native_adapter)
            (
                visual_batch,
                visual_rollout_post,
                visual_sde_draws,
            ) = _rollout_visual(
                adapter=visual_adapter,
                rollout=rollout,
                prompt=str(case["prompt"]),
                context=step_context,
                canonical_latent=visual_initial,
                transition_rng_state=transition_rng_state,
            )
            visual_global_post = _RngSnapshot.capture()

            construction_rng.restore()
            _activate_exclusive_policy(native_adapter, visual_adapter)
            (
                native_data,
                native_rollout_post,
                native_sde_draws,
            ) = _rollout_native(
                helpers=helpers,
                adapter=native_adapter,
                payload=native_prompt,
                canonical_latent=visual_initial,
                transition_rng_state=transition_rng_state,
                batch_size=batch_size,
                transition_count=transition_count,
                guidance_scale=view.sample.guidance_scale,
            )
            native_global_post = _RngSnapshot.capture()
            if not visual_global_post.exactly_equal(native_global_post):
                raise RuntimeError(
                    "rollout branches consumed different process-global RNG"
                )
            if (
                len(visual_sde_draws) != transition_count
                or len(native_sde_draws) != transition_count
            ):
                raise RuntimeError(
                    "rollout SDE draw count does not match transition count"
                )

            timestep_item = _comparison_item(
                {
                    "timesteps": visual_batch.timesteps,
                    "transition_mask": visual_batch.transition_mask,
                },
                {
                    "timesteps": native_data["timesteps"],
                    "transition_mask": native_data["transition_mask"],
                },
                rtol=0.0,
                atol=0.0,
            )
            rollout_item = _comparison_item(
                {
                    "latents": visual_batch.latents,
                    "next_latents": visual_batch.next_latents,
                    "transition_generator_post_state": visual_rollout_post,
                    **{
                        f"sde_draw/{index:03d}": draw
                        for index, draw in enumerate(visual_sde_draws)
                    },
                },
                {
                    "latents": native_data["latents"],
                    "next_latents": native_data["next_latents"],
                    "transition_generator_post_state": native_rollout_post,
                    **{
                        f"sde_draw/{index:03d}": draw
                        for index, draw in enumerate(native_sde_draws)
                    },
                },
            )
            old_log_item = _comparison_item(
                {"old_log_prob": visual_batch.old_log_probs},
                {"old_log_prob": native_data["old_log_probs"]},
            )
            if (
                visual_batch.transition_count != transition_count
                or not bool(visual_batch.transition_mask.all())
                or not bool(native_data["transition_mask"].all())
            ):
                raise RuntimeError("native v1 requires T matching YAML and all-active mask")

            algorithm = GRPOAlgorithm.from_config(
                config.algorithm.params,
                context,
            )
            objective = PolicyObjective()
            advantage_computer = AdvantageComputer(
                epsilon=float(config.algorithm.advantage.epsilon),
                output_dtype=GRPOAlgorithm.ADVANTAGE_DTYPE,
            )
            visual_advantage_result = advantage_computer(
                visual_batch,
                _reward_batch(visual_batch, case["reward_values"]),
            )
            visual_loss_inputs = algorithm.prepare_loss_inputs(
                visual_batch,
                visual_advantage_result,
                normalization_mean=None,
            )
            native_advantage_np = _native_tracker_advantages(
                helpers.tracker_type,
                prompt=str(case["prompt"]),
                rewards=case["reward_values"],
            )
            expected_advantage = np.asarray(
                case["expected_advantages"],
                dtype=np.float64,
            )
            if not np.allclose(
                native_advantage_np,
                expected_advantage,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "actual native PerPromptStatTracker drifted from case constants"
                )
            native_advantage = torch.as_tensor(
                native_advantage_np,
                dtype=torch.float32,
                device=device,
            )
            advantage_item = _comparison_item(
                {"group_advantage": visual_advantage_result.base_advantage},
                {"group_advantage": native_advantage.cpu()},
            )

            perm = _canonical_permutation(
                batch_size=batch_size,
                seed=int(case["seed"]) ^ 0x5A17_31C9,
                device=device,
            )
            slots: list[tuple[tuple[int, ...], int]] = []
            for start in range(0, batch_size, microbatch_size):
                rows = tuple(
                    int(value)
                    for value in perm[start : start + microbatch_size].tolist()
                )
                for step in range(transition_count):
                    slots.append((rows, step))
            expected_k = (batch_size // microbatch_size) * transition_count
            if len(slots) != expected_k:
                raise RuntimeError("native K-slot schedule is inconsistent")

            visual_initial_parameters = _named_tensor_snapshot(visual_named)
            native_initial_parameters = _named_tensor_snapshot(native_named)
            # The update stream is independent of either rollout's final
            # global state.  No global random draw is allowed between this
            # restore and the snapshot.
            construction_rng.restore()
            update_rng = _RngSnapshot.capture()

            def visual_update(optimizer: Any) -> _UpdateTrace:
                _activate_exclusive_policy(visual_adapter, native_adapter)
                visual_adapter.train_module.train(True)
                return _run_update_window(
                    named_parameters=visual_named,
                    optimizer=optimizer,
                    initial_parameters=visual_initial_parameters,
                    slots=slots,
                    evaluate_slot=lambda rows, step: _aligned_visual_objective(
                        adapter=visual_adapter,
                        objective=objective,
                        batch=_transition_slice(visual_batch, rows, step),
                        inputs=_loss_input_slice(
                            visual_loss_inputs,
                            rows,
                            step,
                        ),
                    ),
                    batch_size=batch_size,
                    transition_count=transition_count,
                    max_grad_norm=float(config.optimizer.max_grad_norm),
                )

            def native_update(optimizer: Any) -> _UpdateTrace:
                _activate_exclusive_policy(native_adapter, visual_adapter)
                native_adapter.train_module.train(True)
                return _run_update_window(
                    named_parameters=native_named,
                    optimizer=optimizer,
                    initial_parameters=native_initial_parameters,
                    slots=slots,
                    evaluate_slot=lambda rows, step: _native_slot_objective(
                        helpers=helpers,
                        adapter=native_adapter,
                        native_data=native_data,
                        view=view,
                        algorithm=algorithm,
                        native_advantage=native_advantage,
                        rows=rows,
                        step=step,
                    ),
                    batch_size=batch_size,
                    transition_count=transition_count,
                    max_grad_norm=float(config.optimizer.max_grad_norm),
                )

            # First order: VisualRL then native, each from the same update RNG
            # snapshot and the same trainable-parameter state.
            visual_optimizer_forward = _build_adamw(
                visual_named,
                config.optimizer,
            )
            native_optimizer_forward = _build_adamw(
                native_named,
                config.optimizer,
            )
            update_rng.restore()
            visual_trace_forward = visual_update(visual_optimizer_forward)
            visual_post_forward = _RngSnapshot.capture()
            update_rng.restore()
            native_trace_forward = native_update(native_optimizer_forward)
            native_post_forward = _RngSnapshot.capture()

            # Reset both objects and AdamW state, then run the reverse order.
            _restore_named_parameters(
                visual_named,
                visual_initial_parameters,
            )
            _restore_named_parameters(
                native_named,
                native_initial_parameters,
            )
            del visual_optimizer_forward, native_optimizer_forward
            native_optimizer = _build_adamw(
                native_named,
                config.optimizer,
            )
            visual_optimizer = _build_adamw(
                visual_named,
                config.optimizer,
            )
            update_rng.restore()
            native_trace = native_update(native_optimizer)
            native_post_reverse = _RngSnapshot.capture()
            update_rng.restore()
            visual_trace = visual_update(visual_optimizer)
            visual_post_reverse = _RngSnapshot.capture()

            order_flags = {
                "forward_branch_rng": visual_post_forward.exactly_equal(
                    native_post_forward
                ),
                "native_forward_reverse_rng": visual_post_forward.exactly_equal(
                    native_post_reverse
                ),
                "visual_forward_reverse_rng": visual_post_forward.exactly_equal(
                    visual_post_reverse
                ),
                "visual_forward_reverse_trace": _update_traces_exactly_equal(
                    visual_trace_forward,
                    visual_trace,
                ),
                "native_forward_reverse_trace": _update_traces_exactly_equal(
                    native_trace_forward,
                    native_trace,
                ),
            }
            if not all(order_flags.values()):
                order_flags["visual_forward_reverse_difference"] = (
                    _update_trace_difference_summary(
                        visual_trace_forward,
                        visual_trace,
                    )
                )
                order_flags["native_forward_reverse_difference"] = (
                    _update_trace_difference_summary(
                        native_trace_forward,
                        native_trace,
                    )
                )
                raise RuntimeError(
                    "update result depends on VisualRL/native branch order: "
                    + json.dumps(order_flags, sort_keys=True)
                )

            items = {
                "prompt_encoding": prompt_item,
                "initial_latent": initial_item,
                "timestep": timestep_item,
                "rollout_latent": rollout_item,
                "old_log_prob": old_log_item,
                "group_advantage": advantage_item,
                **_trace_items(visual_trace, native_trace),
            }

            native_adapter.close()
            native_closed = True
            del native_optimizer
            torch.cuda.empty_cache()
            resume_flags = _resume_comparison(
                config=config,
                adapter=visual_adapter,
                optimizer=visual_optimizer,
                rollout_batch=visual_batch,
                loss_inputs=visual_loss_inputs,
                objective=objective,
                slots=slots,
                build_fresh_adapter=build_adapter,
            )
            items["checkpoint_resume"] = {
                "passed": all(resume_flags.values()),
                "comparisons": resume_flags,
            }

    finally:
        visual_adapter.close()
        if not native_closed:
            native_adapter.close()
        torch.cuda.empty_cache()

    if set(items) != set(_ITEM_KEYS):
        raise RuntimeError("real native parity did not produce all fourteen items")
    overall = all(bool(item["passed"]) for item in items.values())
    return {
        "schema_version": 1,
        "case": case_path.stem,
        "config_path": str(case["config_path"]),
        "precision": str(config.runtime.precision),
        "items": items,
        "overall_pass": overall,
    }


def main() -> int:
    # This is a fixed test process setting, not a user-configurable input.  It
    # must be installed before visual_rl imports torch and initializes cuBLAS.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    repo_root = Path(__file__).resolve().parents[2]
    case_path = (
        repo_root
        / "tests"
        / "fixtures"
        / "native_parity"
        / "flow_grpo_sd3_case_v1.json"
    )
    case_name = case_path.stem
    config_path = "configs/flow_grpo_sd3.yaml"
    precision = "fp32"
    report: dict[str, Any]
    try:
        # Third-party reference imports/forwards sometimes print directly.
        # Redirect the whole execution so stdout remains one canonical object.
        with redirect_stdout(sys.stderr):
            case = _load_case(case_path)
            config_path = str(case["config_path"])
            missing = _missing_dependencies()
            if missing:
                raise RuntimeError(
                    "missing native dependencies: " + ", ".join(missing)
                )
            config, view, reference_repo = _resolve_setup(repo_root, case)
            precision = config.runtime.precision
            report = _run_real_parity(
                repo_root=repo_root,
                case_path=case_path,
                case=case,
                config=config,
                view=view,
                reference_repo=reference_repo,
            )
    except BaseException as exc:
        report = _failure_report(
            case_name=case_name,
            config_path=config_path,
            precision=precision,
        )
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
    try:
        payload = _canonical_report_json(report)
    except BaseException as exc:
        sys.stderr.write(
            "native report validation failed: "
            f"{type(exc).__name__}: {exc}\n"
        )
        report = _failure_report(
            case_name=case_name,
            config_path=config_path,
            precision=precision,
        )
        payload = _canonical_report_json(report)
    sys.stdout.write(payload + "\n")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
