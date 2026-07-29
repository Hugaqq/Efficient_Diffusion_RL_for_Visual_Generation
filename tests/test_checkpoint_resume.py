"""Focused format-v5 mechanical checkpoint and resume contracts."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch

from visual_rl.artifacts.audit import audit_run_artifacts
from visual_rl.artifacts.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    RankState,
    TrainingContract,
    apply_training_state,
    checkpoint_tree_sha256,
    read_and_validate_training_state,
    save_training_state,
)
from visual_rl.artifacts.manifest import SampleManifest, SampleRecord
from visual_rl.artifacts.status import inspect_run_status
from visual_rl.configs.schema import OptimizerConfig
from visual_rl.core.types import FrozenMapping
from visual_rl.errors import ResumeError
from visual_rl.model_adapters.base import ModelAdapter


class _TrainModule(torch.nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


class _Adapter(ModelAdapter):
    MEDIA_TYPE = "image"

    def __init__(self, values: tuple[float, ...]) -> None:
        self._module = _TrainModule(values)
        self.validate_calls = 0
        self.load_calls = 0

    @property
    def train_module(self) -> torch.nn.Module:
        return self._module

    def sample(self, request):
        raise NotImplementedError

    def recompute_policy_stats(self, batch, *, require_reference=False):
        raise NotImplementedError

    def validate_checkpoint(self, checkpoint_dir: Path) -> None:
        self.validate_calls += 1
        super().validate_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        self.load_calls += 1
        super().load_checkpoint(checkpoint_dir)


class _RNGConsumingAdapter(_Adapter):
    def __init__(self, values: tuple[float, ...], *, fail: bool = False) -> None:
        super().__init__(values)
        self.fail = fail

    def save_checkpoint(self, output_dir: Path) -> None:
        random.random()
        np.random.random()
        torch.rand(3)
        if self.fail:
            raise RuntimeError("injected adapter save failure")
        super().save_checkpoint(output_dir)


def _optimizer(adapter: _Adapter, *, lr: float = 0.01) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        adapter.parameters(),
        lr=lr,
        betas=(0.8, 0.9),
        eps=1e-7,
        weight_decay=0.02,
    )


def _initialize_moments(
    adapter: _Adapter,
    optimizer: torch.optim.AdamW,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in adapter.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _rank_state(rank: int = 0) -> RankState:
    return RankState.from_rng(
        rank=rank,
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu=torch.get_rng_state().cpu().contiguous(),
        torch_cuda=None,
    )


def _contract() -> TrainingContract:
    return TrainingContract(algorithm="grpo", version=1)


def _optimizer_config(*, learning_rate: float) -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=learning_rate,
        adam_beta1=0.7,
        adam_beta2=0.95,
        adam_weight_decay=0.03,
        adam_epsilon=1e-6,
        max_grad_norm=None,
        max_initial_logprob_delta=None,
        require_initial_clipfrac_zero=True,
        require_finite_gradients=True,
        require_nonzero_gradients=True,
    )


def _save_fixture(
    root: Path,
    *,
    values: tuple[float, ...] = (1.0, 2.0),
    global_step: int = 3,
) -> tuple[Path, _Adapter, torch.optim.AdamW, RankState]:
    root.mkdir(parents=True, exist_ok=True)
    adapter = _Adapter(values)
    optimizer = _optimizer(adapter)
    _initialize_moments(adapter, optimizer)
    rank_state = _rank_state()
    checkpoint = root / "checkpoint_000003"
    save_training_state(
        checkpoint,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        global_step=global_step,
        training_contract=_contract(),
        rank_states=(rank_state,),
        writer_rank=0,
        writer_device=torch.device("cpu"),
    )
    return checkpoint, adapter, optimizer, rank_state


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_v5_producer_writes_exact_tree_keys_and_stable_bytes(
    tmp_path: Path,
) -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    adapter = _Adapter((1.0, 2.0))
    optimizer = _optimizer(adapter)
    _initialize_moments(adapter, optimizer)
    state = _rank_state()

    first = tmp_path / "first" / "checkpoint_000003"
    second = tmp_path / "second" / "checkpoint_000003"
    first.parent.mkdir()
    second.parent.mkdir()
    first_metadata = save_training_state(
        first,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        global_step=3,
        training_contract=_contract(),
        rank_states=(state,),
        writer_rank=0,
        writer_device=torch.device("cpu"),
    )
    second_metadata = save_training_state(
        second,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        global_step=3,
        training_contract=_contract(),
        rank_states=(state,),
        writer_rank=0,
        writer_device=torch.device("cpu"),
    )

    assert {path.name for path in first.iterdir()} == {
        "adapter",
        "training_state.pt",
        "checkpoint.json",
    }
    metadata = json.loads((first / "checkpoint.json").read_text(encoding="utf-8"))
    assert set(metadata) == {
        "format_version",
        "global_step",
        "world_size",
        "training_contract",
        "adapter_dir",
        "adapter_tree_sha256",
        "training_state",
        "training_state_sha256",
    }
    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION == 5
    assert metadata["training_contract"] == {
        "algorithm": "grpo",
        "version": 1,
    }
    assert _tree_bytes(first) == _tree_bytes(second)
    assert first_metadata.tree_sha256 == second_metadata.tree_sha256
    assert first_metadata.tree_sha256 == checkpoint_tree_sha256(first)


@pytest.mark.parametrize("fail", [False, True])
def test_save_restores_captured_writer_rng_on_success_and_failure(
    tmp_path: Path,
    fail: bool,
) -> None:
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    adapter = _RNGConsumingAdapter((1.0, 2.0), fail=fail)
    optimizer = _optimizer(adapter)
    captured = _rank_state()
    random.random()
    np.random.random()
    torch.rand(5)

    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            save_training_state(
                tmp_path / "checkpoint_000001",
                adapter=adapter,
                optimizer=optimizer,
                scaler=None,
                global_step=1,
                training_contract=_contract(),
                rank_states=(captured,),
                writer_rank=0,
                writer_device=torch.device("cpu"),
            )
    else:
        save_training_state(
            tmp_path / "checkpoint_000001",
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            global_step=1,
            training_contract=_contract(),
            rank_states=(captured,),
            writer_rank=0,
            writer_device=torch.device("cpu"),
        )

    assert random.getstate() == captured.python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == captured.numpy_bit_generator
    assert tuple(current_numpy[1].tolist()) == captured.numpy_state
    assert torch.equal(torch.get_rng_state(), captured.torch_cpu)


def test_reader_uses_weights_only_and_rejects_unknown_fields_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _saved_adapter, _saved_optimizer, _state = _save_fixture(tmp_path)
    adapter = _Adapter((9.0, 10.0))
    optimizer = _optimizer(adapter)
    original_weights = adapter.train_module.weight.detach().clone()
    original_optimizer = optimizer.state_dict()
    calls: list[tuple[Path, dict[str, Any]]] = []
    original_load = torch.load

    def observed_load(path, *args, **kwargs):
        calls.append((Path(getattr(path, "name", path)), dict(kwargs)))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    validated = read_and_validate_training_state(
        checkpoint,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        expected_global_step=3,
        expected_world_size=1,
        expected_training_contract=_contract(),
    )
    assert validated.global_step == 3
    training_calls = [
        kwargs
        for path, kwargs in calls
        if path.name == "training_state.pt"
    ]
    assert training_calls == [{"map_location": "cpu", "weights_only": True}]
    assert torch.equal(adapter.train_module.weight, original_weights)
    assert optimizer.state_dict() == original_optimizer

    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["legacy_identity"] = "forbidden"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    second_adapter = _Adapter((7.0, 8.0))
    second_optimizer = _optimizer(second_adapter)
    before = second_adapter.train_module.weight.detach().clone()
    with pytest.raises(ResumeError, match="exact key set"):
        read_and_validate_training_state(
            checkpoint,
            adapter=second_adapter,
            optimizer=second_optimizer,
            scaler=None,
            expected_global_step=3,
            expected_world_size=1,
            expected_training_contract=_contract(),
        )
    assert torch.equal(second_adapter.train_module.weight, before)
    assert second_adapter.validate_calls == 0


def test_read_rejects_v1_to_v4_and_contract_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    checkpoint, _saved_adapter, _saved_optimizer, _state = _save_fixture(tmp_path)
    adapter = _Adapter((9.0, 10.0))
    optimizer = _optimizer(adapter)
    before = adapter.train_module.weight.detach().clone()

    with pytest.raises(ResumeError, match="training_contract"):
        read_and_validate_training_state(
            checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            expected_global_step=3,
            expected_world_size=1,
            expected_training_contract=TrainingContract(
                algorithm="grpo",
                version=2,
            ),
        )
    assert torch.equal(adapter.train_module.weight, before)
    assert adapter.validate_calls == 0

    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ResumeError, match="format_version"):
        read_and_validate_training_state(
            checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            expected_global_step=3,
            expected_world_size=1,
            expected_training_contract=_contract(),
        )
    assert torch.equal(adapter.train_module.weight, before)
    assert adapter.validate_calls == 0


def test_apply_preserves_moments_reapplies_current_hyperparameters_and_rng(
    tmp_path: Path,
) -> None:
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    checkpoint, saved_adapter, saved_optimizer, saved_rng = _save_fixture(tmp_path)
    saved_weight = saved_adapter.train_module.weight.detach().clone()
    saved_state = saved_optimizer.state_dict()
    saved_exp_avg = saved_state["state"][0]["exp_avg"].clone()

    adapter = _Adapter((20.0, 30.0))
    optimizer = _optimizer(adapter, lr=0.5)
    validated = read_and_validate_training_state(
        checkpoint,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        expected_global_step=3,
        expected_world_size=1,
        expected_training_contract=_contract(),
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    apply_training_state(
        validated,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        optimizer_config=_optimizer_config(learning_rate=0.123),
        rank=0,
    )

    assert torch.equal(adapter.train_module.weight, saved_weight)
    restored_state = optimizer.state_dict()
    assert torch.equal(restored_state["state"][0]["exp_avg"], saved_exp_avg)
    assert optimizer.param_groups[0]["lr"] == 0.123
    assert optimizer.param_groups[0]["betas"] == (0.7, 0.95)
    assert optimizer.param_groups[0]["weight_decay"] == 0.03
    assert optimizer.param_groups[0]["eps"] == 1e-6
    assert random.getstate() == saved_rng.python_state
    assert tuple(np.random.get_state()[1].tolist()) == saved_rng.numpy_state
    assert torch.equal(torch.get_rng_state(), saved_rng.torch_cpu)
    with pytest.raises(ResumeError, match="already been applied"):
        apply_training_state(
            validated,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            optimizer_config=_optimizer_config(learning_rate=0.123),
            rank=0,
        )


def test_digest_changes_for_adapter_moment_rng_or_step(tmp_path: Path) -> None:
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    baseline, _adapter, _optimizer_value, _rng = _save_fixture(
        tmp_path / "baseline"
    )
    baseline_digest = checkpoint_tree_sha256(baseline)

    variants: list[str] = []
    adapter_changed, *_ = _save_fixture(
        tmp_path / "adapter",
        values=(1.0, 2.5),
    )
    variants.append(checkpoint_tree_sha256(adapter_changed))

    moment_adapter = _Adapter((1.0, 2.0))
    moment_optimizer = _optimizer(moment_adapter)
    _initialize_moments(moment_adapter, moment_optimizer)
    moment_optimizer.state[next(iter(moment_adapter.parameters()))]["exp_avg"][0] += 1
    moment_root = tmp_path / "moment" / "checkpoint_000003"
    moment_root.parent.mkdir()
    save_training_state(
        moment_root,
        adapter=moment_adapter,
        optimizer=moment_optimizer,
        scaler=None,
        global_step=3,
        training_contract=_contract(),
        rank_states=(_rank_state(),),
        writer_rank=0,
        writer_device=torch.device("cpu"),
    )
    variants.append(checkpoint_tree_sha256(moment_root))

    rng = _rank_state()
    changed_torch_rng = rng.torch_cpu.clone()
    changed_torch_rng[0] ^= 1
    changed_rng = RankState(
        rank=0,
        python_state=rng.python_state,
        numpy_bit_generator=rng.numpy_bit_generator,
        numpy_state=rng.numpy_state,
        numpy_position=rng.numpy_position,
        numpy_has_gauss=rng.numpy_has_gauss,
        numpy_cached_gaussian=rng.numpy_cached_gaussian,
        torch_cpu=changed_torch_rng,
        torch_cuda=None,
    )
    rng_adapter = _Adapter((1.0, 2.0))
    rng_optimizer = _optimizer(rng_adapter)
    _initialize_moments(rng_adapter, rng_optimizer)
    rng_root = tmp_path / "rng" / "checkpoint_000003"
    rng_root.parent.mkdir()
    save_training_state(
        rng_root,
        adapter=rng_adapter,
        optimizer=rng_optimizer,
        scaler=None,
        global_step=3,
        training_contract=_contract(),
        rank_states=(changed_rng,),
        writer_rank=0,
        writer_device=torch.device("cpu"),
    )
    variants.append(checkpoint_tree_sha256(rng_root))

    step_changed, *_ = _save_fixture(
        tmp_path / "step",
        global_step=4,
    )
    variants.append(checkpoint_tree_sha256(step_changed))

    assert all(digest != baseline_digest for digest in variants)


def test_status_and_audit_project_only_authoritative_v2_v3_v5_artifacts(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / ".staging").mkdir()
    commits = run_root / "commits"
    commits.mkdir()
    adapter = _Adapter((1.0, 2.0))
    optimizer = _optimizer(adapter)
    _initialize_moments(adapter, optimizer)
    checkpoint = run_root / "checkpoint_000001"
    metadata = save_training_state(
        checkpoint,
        adapter=adapter,
        optimizer=optimizer,
        scaler=None,
        global_step=1,
        training_contract=_contract(),
        rank_states=(_rank_state(),),
        writer_rank=0,
        writer_device=torch.device("cpu"),
    )
    record = SampleRecord(
        run_id="run-test",
        sample_id="sample-0",
        sample_index=0,
        step=0,
        rank=0,
        prompt="prompt",
        media_type="image",
        prompt_metadata=FrozenMapping({}),
        seed=7,
        rollout_type="full_trajectory",
        timestep_summary=FrozenMapping({"values": [1], "count": 1}),
        reward_values=FrozenMapping(
            {
                "raw": {"score": 1.0},
                "weighted": {"score": 1.0},
                "weighted_total": 1.0,
                "valid": True,
                "shared_metadata": {"score": {}},
                "sample_metadata": {"score": {}},
            }
        ),
        media_path=None,
        rollout_cache_path=None,
        checkpoint_path="checkpoint_000001",
        model_metadata=FrozenMapping({}),
        prompt_id="prompt-0",
        group_id="group-0",
        branch_id=None,
    )
    metric = {
        "schema_version": "3",
        "step": 0,
        "sample_count": 1,
        "active_transition_count": 1,
        "loss": 0.5,
    }
    marker = {
        "schema_version": "2",
        "kind": "artifact_commit",
        "run_id": "run-test",
        "transaction_id": "a" * 32,
        "completed_steps": 1,
        "staged_steps": [0],
        "checkpoint": {
            "completed_steps": 1,
            "path": "checkpoint_000001",
            "tree_sha256": metadata.tree_sha256,
        },
        "steps": [
            {
                "artifact_step": 0,
                "manifest_records": [record.to_plain_dict()],
                "core_metric_row": metric,
            }
        ],
    }
    (commits / "commit_000001.json").write_text(
        json.dumps(marker, allow_nan=False),
        encoding="utf-8",
    )
    (run_root / "sample_manifest.json").write_text(
        json.dumps(
            SampleManifest(run_id="run-test", records=(record,)).to_dict(),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (run_root / "metrics.jsonl").write_text(
        json.dumps(metric, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    status = inspect_run_status(run_root)
    audit = audit_run_artifacts(run_root)

    assert status["run_id"] == "run-test"
    assert status["committed_steps"] == 1
    assert status["authoritative_checkpoint"] == "checkpoint_000001"
    assert status["resumable"] is True
    assert status["checks"] == ()
    assert audit["run_id"] == "run-test"
    assert audit["committed_steps"] == 1
    assert audit["checked_commit_count"] == 1
    assert audit["checks"] == ()
    assert audit["checked_artifact_paths"] == (
        "commits/commit_000001.json",
        "checkpoint_000001",
        "sample_manifest.json",
        "metrics.jsonl",
    )
