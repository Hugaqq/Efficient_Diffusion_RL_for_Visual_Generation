from __future__ import annotations

import hashlib
import importlib
import importlib.util
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

import visual_rl.runtime.lifecycle as runtime_defaults
from visual_rl.artifacts.checkpoint import AtomicCheckpointManager, RankCheckpointReader
from visual_rl.models import SchedulerArtifactBlueprint
from visual_rl.models.implementations.sd3 import SD3RuntimeParts
from visual_rl.runtime import (
    ControllerStage,
    ControllerState,
)
from visual_rl.runtime.composition import create_default_run_controller
from visual_rl.runtime.types import RunResult


class _SchedulerConfig(dict):
    def __getattr__(self, name: str):
        return self[name]


class _FakeScheduler:
    scheduler_identity = "fake-sd3-scheduler.v1"

    def __init__(self, config=None) -> None:
        self.config = _SchedulerConfig({} if config is None else config)
        self.set_timesteps(num_inference_steps=2, device="cpu")

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps: int, device: object) -> None:
        self.timesteps = torch.linspace(
            900.0,
            100.0,
            num_inference_steps,
            dtype=torch.float32,
            device=device,
        )
        self.sigmas = torch.linspace(
            1.0,
            0.1,
            num_inference_steps + 1,
            dtype=torch.float32,
            device=device,
        )


class _FakePromptEncoder:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def to(self, device: object):
        self.device = torch.device(device)
        return self

    def encode(
        self,
        prompts: tuple[str, ...],
        max_sequence_length: int,
        guidance_scale: float,
    ):
        del max_sequence_length, guidance_scale
        batch_size = len(prompts)
        positive = (
            torch.linspace(
                0.25,
                0.75,
                batch_size,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(batch_size, 1, 1)
            .expand(-1, 3, 2)
        )
        negative = torch.full_like(positive, -0.25)
        pooled = positive.mean(dim=1)
        negative_pooled = negative.mean(dim=1)
        return positive, negative, pooled, negative_pooled


class _FakeDecoder:
    def to(self, device: object):
        del device
        return self

    def decode(self, latents: torch.Tensor, latent_spec: object) -> torch.Tensor:
        assert tuple(latents.shape) == latent_spec.shape
        return latents[:, :3].detach().clone()


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_scale = torch.nn.Parameter(torch.tensor(0.2))
        self.register_buffer("base_scale", torch.tensor(0.45))
        self._adapter_disabled = 0

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool,
        pooled_projections: torch.Tensor | None = None,
    ):
        del timestep, return_dict, pooled_projections
        conditioning = encoder_hidden_states.mean(
            dim=tuple(range(1, encoder_hidden_states.ndim))
        ).reshape(hidden_states.shape[0], *([1] * (hidden_states.ndim - 1)))
        scale = self.base_scale
        if self._adapter_disabled == 0:
            scale = scale + self.policy_scale
        return (hidden_states * scale + conditioning,)

    @contextmanager
    def disable_adapter(self):
        self._adapter_disabled += 1
        try:
            yield
        finally:
            self._adapter_disabled -= 1


class _FakeSD3Loader:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.transformers: list[_FakeTransformer] = []

    def __call__(self, family, artifact_path, config, precision):
        self.calls.append((family, artifact_path, config, precision))
        if family != "sd3":
            raise AssertionError(f"unexpected model family: {family}")
        transformer = _FakeTransformer()
        self.transformers.append(transformer)
        return SD3RuntimeParts(
            prompt_encoder=_FakePromptEncoder(),
            transformer=transformer,
            decoder=_FakeDecoder(),
            reference_context=transformer.disable_adapter,
            latent_channels=4,
            scheduler_artifact_blueprint=(
                SchedulerArtifactBlueprint.from_scheduler(_FakeScheduler())
            ),
        )


class _FakeAccelerator:
    instances: ClassVar[list[_FakeAccelerator]] = []

    def __init__(
        self,
        *,
        mixed_precision: str,
        gradient_accumulation_steps: int,
    ) -> None:
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_processes = 1
        self.process_index = 0
        self.local_process_index = 0
        self.distributed_type = "NO"
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.optimizer_step_was_skipped = False
        self.scaler = None
        self.end_training_calls = 0
        self.instances.append(self)

    def prepare(self, *values: object):
        return values

    @contextmanager
    def accumulate(self, root: object):
        del root
        yield

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def unscale_gradients(optimizer: object) -> None:
        del optimizer

    def end_training(self) -> None:
        self.end_training_calls += 1


def _set_seed(seed: int, *, device_specific: bool) -> None:
    assert device_specific is True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_artifacts(root: Path) -> tuple[Path, Path, Path]:
    model = root / "model"
    model.mkdir()
    (model / "weights.fake").write_bytes(b"fake-sd3")
    dataset = root / "prompts.txt"
    dataset.write_text("a red cube\na blue sphere\n", encoding="utf-8")
    reward = root / "reward"
    reward.mkdir()
    (reward / "revision.txt").write_text("reward-v1\n", encoding="utf-8")
    return model, dataset, reward


def _write_recipe(
    root: Path,
    *,
    filename: str,
    output: Path,
    artifacts: tuple[Path, Path, Path],
    max_optimizer_steps: int,
    resume_from: Path | None = None,
) -> Path:
    model, dataset, reward = artifacts
    config = root / filename
    resume_value = "null" if resume_from is None else resume_from.as_posix()
    config.write_text(
        f"""schema_version: 2
recipe: flow_grpo_v1
overrides:
  algorithm:
    params:
      num_steps: 2
  model:
    params:
      artifact_ref: main
      resolution: 16
      guidance_scale: 1.0
      gradient_checkpointing: false
  training:
    seed: 17
    global_prompt_batch_size: 1
    max_optimizer_steps: {max_optimizer_steps}
    gradient_accumulation_steps: 1
    adamw:
      learning_rate: 0.05
      weight_decay: 0.0
    lr_schedule:
      warmup_steps: 0
    update_safety:
      require_finite_gradients: true
      require_nonzero_gradients: true
      max_grad_norm: 1.0
  execution:
    precision: fp32
    distribution_mode: single
    group_size: 2
launch:
  output_dir: {output.as_posix()}
  resume_from: {resume_value}
  checkpoint_every_optimizer_steps: 1
  artifacts:
    model: {model.as_posix()}
    datasets:
      main: {dataset.as_posix()}
    rewards:
      reward_quality: {reward.as_posix()}
""",
        encoding="utf-8",
    )
    return config


def _write_one_update_recipe(root: Path) -> tuple[Path, Path]:
    artifacts = _write_artifacts(root)
    output = root / "output"
    config = _write_recipe(
        root,
        filename="one-update.yaml",
        output=output,
        artifacts=artifacts,
        max_optimizer_steps=1,
    )
    return config, output


def _patch_default_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    original_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object):
        if name in {"diffusers", "imageio_ffmpeg", "peft", "transformers"}:
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(runtime_defaults, "_create_accelerator", _FakeAccelerator)
    monkeypatch.setattr(runtime_defaults, "_set_seed", _set_seed)
    _FakeAccelerator.instances.clear()


def _checkpoint_snapshot(checkpoint_path: Path):
    manager = AtomicCheckpointManager(checkpoint_path.parent)
    inspection = manager.inspect_complete(checkpoint_path)
    snapshot = RankCheckpointReader(manager).read_rank_snapshot(
        inspection,
        expected_world_size=1,
        expected_rank=0,
    )
    return inspection, snapshot


def _file_tree_identity(root: Path) -> tuple[tuple[object, ...], ...]:
    values: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            values.append(
                (
                    path.relative_to(root).as_posix(),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ino,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return tuple(values)


def _assert_exact(left: object, right: object, *, path: str) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor), path
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype, path
        assert tuple(left.shape) == tuple(right.shape), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray), path
        assert isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype, path
        assert np.array_equal(left, right), path
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping), path
        assert isinstance(right, Mapping), path
        assert set(left) == set(right), path
        for key in sorted(left, key=str):
            _assert_exact(left[key], right[key], path=f"{path}.{key}")
        return
    sequence_types = (tuple, list)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        assert type(left) is type(right), path
        assert isinstance(left, Sequence), path
        assert isinstance(right, Sequence), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_exact(left_item, right_item, path=f"{path}[{index}]")
        return
    assert type(left) is type(right), path
    assert left == right, path


def _assert_checkpoint_state_exact(
    left,
    right,
    *,
    include_run_summary: bool,
) -> None:
    left_inspection, left_snapshot = left
    right_inspection, right_snapshot = right
    assert left_inspection.contract.recipe_id == right_inspection.contract.recipe_id
    assert left_inspection.contract.checkpoint_contract_id == (
        right_inspection.contract.checkpoint_contract_id
    )
    assert left_inspection.progress.to_payload() == (
        right_inspection.progress.to_payload()
    )
    if include_run_summary:
        assert left_inspection.committed.state_tree_id == (
            right_inspection.committed.state_tree_id
        )
    assert (
        left_snapshot.safe_point.to_payload() == right_snapshot.safe_point.to_payload()
    )
    assert left_snapshot.rng_state == right_snapshot.rng_state
    assert left_snapshot.dynamics_selection_policy == (
        right_snapshot.dynamics_selection_policy
    )
    assert left_snapshot.component_names == right_snapshot.component_names
    for name in left_snapshot.component_names:
        if name == "run_checkpoint_summary" and not include_run_summary:
            continue
        left_state = left_snapshot.component_state(name)
        right_state = right_snapshot.component_state(name)
        _assert_exact(left_state.payload, right_state.payload, path=name)


def test_default_composition_runs_one_real_update_and_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_default_runtime(monkeypatch)

    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text("FAKE_E2E = True\n", encoding="utf-8")
    config, output = _write_one_update_recipe(tmp_path)
    loader = _FakeSD3Loader()
    controller = create_default_run_controller(
        code_root=code_root,
        model_loader=loader,
    )

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    try:
        result = controller.run(config)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)

    assert isinstance(result, RunResult)
    assert result.committed_steps == 1
    assert result.output_dir == output
    assert result.authoritative_checkpoint == output / "checkpoints" / "step-1"
    assert result.marker_path.read_text(encoding="utf-8").endswith("\n")
    inspection, snapshot = _checkpoint_snapshot(result.authoritative_checkpoint)
    assert inspection.progress.next_optimizer_step == 1
    assert snapshot.safe_point.committed_optimizer_step == 1
    assert snapshot.component_names == (
        "data_plane",
        "lr_scheduler",
        "model",
        "optimizer",
        "run_checkpoint_summary",
    )
    assert len(loader.calls) == 1
    assert loader.calls[0][0] == "sd3"
    assert len(loader.transformers) == 1
    assert loader.transformers[0].policy_scale.item() != pytest.approx(0.2)
    assert controller.state is ControllerState.CLOSED
    assert controller.completed_stages == tuple(ControllerStage)
    assert len(_FakeAccelerator.instances) == 1
    assert _FakeAccelerator.instances[0].end_training_calls == 1


class _InjectedCrash(RuntimeError):
    pass


def test_default_composition_continuation_matches_continuous_two_step_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_default_runtime(monkeypatch)
    # The SD3 adapter lazily imports Diffusers.  Its HTTP/Rich dependency chain
    # initializes a module-level style id with Python's global
    # ``random.getrandbits``.  Production comparisons use independent CLI
    # processes, where both sides take the same import path.  This unit test
    # deliberately runs two fresh controllers in one process, so prime that
    # one-time import before either seeded run rather than making the first
    # controller's checkpoint depend on test order.
    importlib.import_module("diffusers")
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text("FAKE_E2E = True\n", encoding="utf-8")
    artifacts = _write_artifacts(tmp_path)
    continuous_output = tmp_path / "continuous-output"
    interrupted_output = tmp_path / "interrupted-output"
    continuous_config = _write_recipe(
        tmp_path,
        filename="continuous.yaml",
        output=continuous_output,
        artifacts=artifacts,
        max_optimizer_steps=2,
    )
    interrupted_config = _write_recipe(
        tmp_path,
        filename="interrupted.yaml",
        output=interrupted_output,
        artifacts=artifacts,
        max_optimizer_steps=2,
    )

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    try:
        continuous_loader = _FakeSD3Loader()
        continuous_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=continuous_loader,
        )
        continuous_result = continuous_controller.run(continuous_config)
        continuous_step1 = _checkpoint_snapshot(
            continuous_output / "checkpoints" / "step-1"
        )
        continuous_step2 = _checkpoint_snapshot(
            continuous_result.authoritative_checkpoint
        )

        interrupted_loader = _FakeSD3Loader()
        interrupted_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=interrupted_loader,
        )
        sink = interrupted_controller._backend.checkpoint_sink
        real_safe_point = sink.checkpoint_safe_point
        committed_receipts: list[object] = []

        def commit_then_crash(request):
            receipt = real_safe_point(request)
            committed_receipts.append(receipt)
            raise _InjectedCrash("injected after durable step-1 checkpoint")

        sink.checkpoint_safe_point = commit_then_crash
        with pytest.raises(_InjectedCrash, match="durable step-1"):
            interrupted_controller.run(interrupted_config)

        step1_path = interrupted_output / "checkpoints" / "step-1"
        interrupted_step1 = _checkpoint_snapshot(step1_path)
        step1_files_before_resume = _file_tree_identity(step1_path)
        assert len(committed_receipts) == 1
        assert committed_receipts[0].checkpoint_path == step1_path
        assert interrupted_controller.state is ControllerState.FAILED
        assert not (interrupted_output / "SUCCESS").exists()
        assert not (interrupted_output / "checkpoints" / "step-2").exists()

        _assert_checkpoint_state_exact(
            continuous_step1,
            interrupted_step1,
            include_run_summary=True,
        )
        continuous_step1_progress = continuous_step1[0].progress
        interrupted_step1_progress = interrupted_step1[0].progress
        assert continuous_step1_progress.next_source_id == (
            interrupted_step1_progress.next_source_id
        )
        assert continuous_step1_progress.next_prompt_batch_id == (
            interrupted_step1_progress.next_prompt_batch_id
        )
        assert continuous_step1_progress.source_cursors == (
            interrupted_step1_progress.source_cursors
        )
        assert continuous_step1_progress.dynamics_selection_policy == (
            interrupted_step1_progress.dynamics_selection_policy
        )
        assert continuous_step1_progress.rng_state_id == (
            interrupted_step1_progress.rng_state_id
        )

        resume_config = _write_recipe(
            tmp_path,
            filename="resume.yaml",
            output=interrupted_output,
            artifacts=artifacts,
            max_optimizer_steps=2,
            resume_from=step1_path,
        )
        assert interrupted_config.read_text(encoding="utf-8").replace(
            "resume_from: null",
            f"resume_from: {step1_path.as_posix()}",
        ) == resume_config.read_text(encoding="utf-8")
        resume_loader = _FakeSD3Loader()
        resume_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=resume_loader,
        )
        resumed_result = resume_controller.run(resume_config)

        assert _file_tree_identity(step1_path) == step1_files_before_resume
        assert {
            item.name
            for item in (interrupted_output / "checkpoints").iterdir()
            if item.is_dir()
        } == {"step-1", "step-2"}
        assert resumed_result.authoritative_checkpoint == (
            interrupted_output / "checkpoints" / "step-2"
        )
        assert resumed_result.marker_path == interrupted_output / "SUCCESS"
        assert resumed_result.marker_path.is_file()

        resumed_step2 = _checkpoint_snapshot(resumed_result.authoritative_checkpoint)
        _assert_checkpoint_state_exact(
            continuous_step2,
            resumed_step2,
            include_run_summary=False,
        )
        assert continuous_step2[0].contract.recipe_id == (
            interrupted_step1[0].contract.recipe_id
        )
        assert interrupted_step1[0].contract.recipe_id == (
            resumed_step2[0].contract.recipe_id
        )
        assert continuous_result.last_metrics == resumed_result.last_metrics
        assert continuous_controller.completed_stages == tuple(ControllerStage)
        assert resume_controller.completed_stages == tuple(ControllerStage)
        assert len(_FakeAccelerator.instances) == 3
        assert all(
            accelerator.end_training_calls == 1
            for accelerator in _FakeAccelerator.instances
        )
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
