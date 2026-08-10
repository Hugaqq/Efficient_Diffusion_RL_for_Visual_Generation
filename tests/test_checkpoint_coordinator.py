"""All-rank safe-point and shard coordination for v0.8 checkpoints."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.artifacts.checkpoint import (
    AtomicCheckpointManager,
    CheckpointConsensusError,
    CheckpointContract,
    CheckpointCoordinator,
    CheckpointProgress,
    CheckpointSafePoint,
    CheckpointSafetyError,
    CheckpointStateCollector,
    ComponentContractRef,
    OptimizerGroupContract,
    ParameterContract,
    RankCheckpointSnapshot,
    RankRNGSnapshot,
    SingleProcessCheckpointBackend,
    StrategyCheckpointBackend,
)
from visual_rl.models import (
    ComponentBinding,
    ComponentRole,
    ExecutionMode,
    ForwardAutocastPolicy,
    ModelComponents,
    ModelExecutionNumericsEvidence,
    ParameterDTypePolicy,
    ParameterStateManager,
    ParameterViewEvidence,
    ParameterViewMode,
)
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.algorithms.optimization.execution import (
    UpdateDisposition,
    UpdateTransactionResult,
)


def _digest(character: str) -> str:
    return character * 64


def _identity(namespace: str, character: str) -> str:
    return f"{namespace}:{_digest(character)}"


def _projection():
    transformer = torch.nn.Module()
    transformer.register_parameter(
        "lora_A",
        torch.nn.Parameter(torch.zeros(2, 2)),
    )
    return ParameterStateManager(
        ModelComponents(
            (
                ComponentBinding(
                    "transformer",
                    transformer,
                    (ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
                ),
                ComponentBinding(
                    "decoder",
                    torch.nn.Identity(),
                    (ComponentRole.DECODER,),
                ),
            )
        )
    ).state_projection


def _numerics(projection) -> ModelExecutionNumericsEvidence:
    return ModelExecutionNumericsEvidence(
        parameter_dtype_policy=ParameterDTypePolicy("float32"),
        forward_autocast_policies=(
            ForwardAutocastPolicy(
                stage=ExecutionMode.TRAIN,
                parameter_view=ParameterView.CURRENT,
                device_type="cpu",
                compute_dtype="float32",
                enabled=False,
            ),
        ),
        parameter_view_evidence=(
            ParameterViewEvidence(
                parameter_view=ParameterView.CURRENT,
                mode=ParameterViewMode.CURRENT,
                owner_component_names=(projection.standalone_saved_component_names),
                restorable_state_names=projection.standalone_parameter_names,
                source_projection_id=projection.projection_id,
                mutates_parameters_in_place=False,
            ),
        ),
    )


def _contract(*, world_size: int = 1) -> CheckpointContract:
    projection = _projection()
    numerics = _numerics(projection)
    parameter = ParameterContract("transformer.lora_A", (2, 2), "torch.float32")
    components = tuple(
        sorted(
            (
                ComponentContractRef(
                    slot=kind,
                    kind=kind,
                    component_declaration_id=_identity(
                        "component-declaration.v1", str(index + 1)
                    ),
                    artifact_binding_id=_identity(
                        "component-artifact-binding.v1", str((index + 6) % 10)
                    ),
                    runtime_bound_contract_id=_digest(chr(ord("a") + index)),
                )
                for index, kind in enumerate(
                    ("model", "trainer", "dynamics", "rollout", "credit")
                )
            ),
            key=lambda item: item.slot,
        )
    )
    return CheckpointContract(
        recipe_id=_identity("materialized-recipe.v2", "a"),
        resolved_fingerprint=_identity("resolved-recipe.v2", "b"),
        algorithm_materialization_spec_id=_identity(
            "algorithm-materialization-spec.v1", "c"
        ),
        execution_policy_id=_identity("execution-policy.v1", "d"),
        reward_plan_id=_identity("reward-plan-spec.v1", "e"),
        source_content_binding_id=_identity("source-content-binding.v1", "f"),
        component_artifact_binding_set_id=_identity(
            "component-artifact-binding-set.v1", "1"
        ),
        runtime_bound_contract_id=_digest("b"),
        immutable_model_revision="revision-1",
        code_identity=_digest("c"),
        components=components,
        model_state_projection=projection,
        model_state_projection_id=projection.projection_id,
        model_execution_numerics=numerics,
        model_execution_numerics_id=numerics.execution_numerics_id,
        trainable_parameters=(parameter,),
        optimizer_groups=(
            OptimizerGroupContract(
                "lora",
                (parameter.name,),
                _digest("d"),
            ),
        ),
        scaler_schema="none.v1",
        lr_scheduler_schema="constant.v1",
        precision="fp32",
        preprocess_identity=_digest("f"),
        preprocess_requirement_set_id=_digest("3"),
        group_size=2,
        global_batch_size=2 * world_size,
        gradient_accumulation_steps=1,
        data_sharding_version="strict-rank-shard.v1",
        sampler_state_schema="per-source-cursor.v1",
        rng_policy="rank-local-explicit-generator.v1",
        dynamics_state_schema="iteration-keyed-selection-policy.v2",
        progress_state_schema="logical-safe-point.v2",
        ema_state_schema="none.v1",
        reference_state_schema="optional-reference.v1",
        execution_transform_plan_id=_digest("4"),
        execution_transform_chain=(),
        world_size=world_size,
        world_size_policy="strict",
        state_schema_versions=(
            ("dynamics_selection_policy", 2),
            ("model", 2),
            ("optimizer", 1),
            ("progress", 2),
            ("rng", 1),
        ),
    )


def _selection_policy(seed: int = 17) -> DynamicsSelectionPolicyState:
    return DynamicsSelectionPolicyState(base_seed=seed)


def _progress(
    step: int,
    policy: DynamicsSelectionPolicyState,
) -> CheckpointProgress:
    return CheckpointProgress(
        global_step=step,
        iteration=step,
        next_optimizer_step=step,
        next_source_id="main",
        next_prompt_batch_id=_digest("6"),
        next_phase_id="main",
        active_reward_ids=("quality",),
        source_cursors=(("main", step * 2),),
        dynamics_selection_policy=policy,
        gradient_accumulation_position=0,
        ema_state_saved=False,
        reference_state_saved=True,
        execution_transform_plan_id=_digest("4"),
        rng_state_id=_digest("5"),
    )


def _update_result(
    *,
    optimizer_step: int = 0,
    disposition: UpdateDisposition = UpdateDisposition.COMMITTED,
) -> UpdateTransactionResult:
    norms = (
        (None, None) if disposition is UpdateDisposition.ACCUMULATING else (1.0, 1.0)
    )
    return UpdateTransactionResult(
        optimizer_step=optimizer_step,
        disposition=disposition,
        payload={"loss": 1.0},
        gradient_norm_pre_clip=norms[0],
        gradient_norm_post_clip=norms[1],
        trace=("logical_commit",),
    )


def _safe_point(
    *,
    rank: int = 0,
    world_size: int = 1,
    step: int = 1,
    disposition: UpdateDisposition = UpdateDisposition.COMMITTED,
) -> CheckpointSafePoint:
    optimizer_step = step - 1 if disposition is UpdateDisposition.COMMITTED else step
    return CheckpointSafePoint.from_update_result(
        rank=rank,
        world_size=world_size,
        update_result=_update_result(
            optimizer_step=optimizer_step,
            disposition=disposition,
        ),
        group_geometry_id=_digest("7"),
    )


def _collector(
    policy: DynamicsSelectionPolicyState,
    *,
    rng: RankRNGSnapshot | None = None,
    calls: list[str] | None = None,
) -> CheckpointStateCollector:
    trace = calls if calls is not None else []

    def model_state():
        trace.append("model")
        return {"weight": torch.tensor([1.0, 2.0])}

    def optimizer_state():
        trace.append("optimizer")
        return {"step": 1, "moment": torch.tensor([0.25])}

    captured_rng = rng or RankRNGSnapshot.capture_current(0)
    return CheckpointStateCollector(
        component_state_sources={
            "optimizer": optimizer_state,
            "model": model_state,
        },
        dynamics_selection_policy_source=lambda: policy,
        rng_state_source=lambda rank: replace(captured_rng, rank=rank),
    )


class _TracingSingleBackend(SingleProcessCheckpointBackend):
    def __init__(self) -> None:
        self.events = []

    def failure_gate(self, phase, failure):
        self.events.append(("failure_gate", phase))
        return super().failure_gate(phase, failure)

    def gather_object(self, value, *, dst=0):
        self.events.append(("gather", dst))
        return super().gather_object(value, dst=dst)

    def broadcast_object(self, value, *, src=0):
        self.events.append(("broadcast", src))
        return super().broadcast_object(value, src=src)

    def barrier(self, phase):
        self.events.append(("barrier", phase))
        return super().barrier(phase)


def test_single_process_safe_point_writes_and_loads_complete_rank_state(tmp_path):
    dynamics = _selection_policy()
    backend = _TracingSingleBackend()
    capture_calls = []
    coordinator = CheckpointCoordinator(
        manager=AtomicCheckpointManager(tmp_path / "checkpoints"),
        backend=backend,
        collector=_collector(dynamics, calls=capture_calls),
    )
    contract = _contract()
    progress = _progress(1, dynamics)

    committed = coordinator.checkpoint(
        contract=contract,
        progress=progress,
        safe_point=_safe_point(),
    )

    assert committed.step == 1
    assert capture_calls == ["model", "optimizer"]
    assert (committed.path / "complete.json").is_file()
    assert (committed.path / "rank_shards" / "rank-0.pt").is_file()
    manifest = json.loads(
        (committed.path / "coordinator_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["world_size"] == 1
    assert manifest["checkpoint_contract_id"] == contract.checkpoint_contract_id
    assert manifest["progress_id"] == progress.progress_id
    assert [item["rank"] for item in manifest["shards"]] == [0]

    restored = coordinator.load_rank_snapshot(committed)
    assert restored.rank == 0
    assert restored.safe_point.update_disposition == "committed"
    assert restored.dynamics_selection_policy == dynamics
    assert restored.component_names == ("model", "optimizer")
    assert torch.equal(
        restored.component_state("model")["weight"],
        torch.tensor([1.0, 2.0]),
    )
    assert backend.events == [
        ("failure_gate", "checkpoint.prepare"),
        ("barrier", "checkpoint.safe_point"),
        ("failure_gate", "checkpoint.capture_state"),
        ("failure_gate", "checkpoint.prepare_staging"),
        ("broadcast", 0),
        ("failure_gate", "checkpoint.broadcast_staging"),
        ("failure_gate", "checkpoint.write_rank_shard"),
        ("barrier", "checkpoint.rank_shards_written"),
        ("gather", 0),
        ("failure_gate", "checkpoint.gather_shards"),
        ("failure_gate", "checkpoint.validate_consensus"),
        ("failure_gate", "checkpoint.atomic_commit"),
        ("broadcast", 0),
        ("failure_gate", "checkpoint.broadcast_commit"),
        ("barrier", "checkpoint.committed"),
    ]


def test_captured_selection_policy_must_match_progress_before_staging(tmp_path):
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=SingleProcessCheckpointBackend(),
        collector=_collector(_selection_policy(18)),
    )

    with pytest.raises(CheckpointConsensusError, match="selection policy"):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, _selection_policy(17)),
            safe_point=_safe_point(),
        )

    assert manager.latest_complete() is None
    assert tuple(manager.root.glob("step-*")) == ()


def test_rank_snapshot_rejects_legacy_mutable_selection_state_field() -> None:
    policy = _selection_policy()
    snapshot = _collector(policy).capture(_safe_point())
    payload = snapshot.to_checkpoint_payload()
    payload["schema_version"] = 1
    payload["dynamics_selection_state"] = payload.pop("dynamics_selection_policy")

    with pytest.raises(ValueError, match="invalid fields or version"):
        RankCheckpointSnapshot.from_checkpoint_payload(payload)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"update_disposition": "accumulating"}, "did not commit"),
        ({"open_data_reservations": 1}, "data reservation"),
        ({"active_reward_futures": 1}, "reward future"),
        ({"active_dynamics_sessions": 1}, "Dynamics session"),
        ({"gradients_synchronized": False}, "not synchronized"),
        ({"gradient_accumulation_position": 1}, "accumulation"),
        ({"poisoned": True}, "poisoned"),
    ),
)
def test_unsafe_rank_is_rejected_before_state_capture_or_directory_creation(
    tmp_path,
    overrides,
    message,
):
    dynamics = _selection_policy()
    calls = []
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=SingleProcessCheckpointBackend(),
        collector=_collector(dynamics, calls=calls),
    )
    safe_point = replace(_safe_point(), **overrides)

    with pytest.raises(CheckpointSafetyError, match=message):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=safe_point,
        )

    assert calls == []
    assert manager.latest_complete() is None
    assert tuple(manager.root.glob("step-*")) == ()


def test_scaler_skipped_update_cannot_authorize_checkpoint(tmp_path):
    dynamics = _selection_policy()
    coordinator = CheckpointCoordinator(
        manager=AtomicCheckpointManager(tmp_path / "checkpoints"),
        backend=SingleProcessCheckpointBackend(),
        collector=_collector(dynamics),
    )
    safe_point = _safe_point(disposition=UpdateDisposition.SCALER_SKIPPED)

    with pytest.raises(CheckpointSafetyError, match="did not commit"):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=safe_point,
        )


class _ScriptedRootBackend:
    rank = 0
    world_size = 2
    is_main_process = True

    def __init__(self, root, mode: str = "complete") -> None:
        self.root = root
        self.mode = mode
        self.barriers = []

    def failure_gate(self, _phase, failure):
        if failure is not None:
            raise failure

    def gather_object(self, value, *, dst=0):
        assert dst == 0
        if self.mode == "missing":
            return [value]
        other_safe = replace(value.safe_point, rank=1)
        if self.mode == "unsafe":
            other_safe = replace(other_safe, active_reward_futures=1)
        if self.mode == "geometry_mismatch":
            other_safe = replace(other_safe, group_geometry_id=_digest("8"))
        descriptor = replace(
            value.descriptor,
            rank=1,
            staging_file="rank-1.pt",
            safe_point_id=other_safe.safe_point_id,
            group_geometry_id=other_safe.group_geometry_id,
        )
        if self.mode == "complete":
            staging = self.root / ".coordinator-staging" / value.descriptor.staging_name
            payload = torch.load(
                staging / "rank-0.pt",
                map_location="cpu",
                weights_only=False,
            )
            rank_zero = RankCheckpointSnapshot.from_checkpoint_payload(payload)
            other_rng = replace(rank_zero.rng_state, rank=1)
            other_snapshot = replace(
                rank_zero,
                rank=1,
                safe_point=other_safe,
                rng_state=other_rng,
            )
            target = staging / "rank-1.pt"
            torch.save(other_snapshot.to_checkpoint_payload(), target)
            data = target.read_bytes()
            descriptor = replace(
                descriptor,
                shard_sha256=hashlib.sha256(data).hexdigest(),
                shard_size=len(data),
                rng_state_id=other_rng.state_identity,
            )
        other = replace(
            value,
            safe_point=other_safe,
            descriptor=descriptor,
        )
        if self.mode == "progress_mismatch":
            other = replace(other, progress_id=_digest("9"))
        return [value, other]

    def broadcast_object(self, value, *, src=0):
        assert src == 0
        return value

    def barrier(self, phase):
        self.barriers.append(phase)


def _two_rank_coordinator(tmp_path, mode="complete"):
    dynamics = _selection_policy()
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")
    backend = _ScriptedRootBackend(manager.root, mode)
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=backend,
        collector=_collector(dynamics),
    )
    return coordinator, manager, backend, dynamics


def test_root_requires_the_exact_complete_rank_shard_set(tmp_path):
    coordinator, manager, _backend, dynamics = _two_rank_coordinator(
        tmp_path,
        mode="missing",
    )

    with pytest.raises(CheckpointConsensusError, match="shard count"):
        coordinator.checkpoint(
            contract=_contract(world_size=2),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(world_size=2),
        )

    assert manager.latest_complete() is None
    assert tuple(manager.root.glob("step-*")) == ()


def test_root_revalidates_every_rank_safe_point(tmp_path):
    coordinator, manager, _backend, dynamics = _two_rank_coordinator(
        tmp_path,
        mode="unsafe",
    )

    with pytest.raises(CheckpointSafetyError, match="reward future"):
        coordinator.checkpoint(
            contract=_contract(world_size=2),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(world_size=2),
        )

    assert manager.latest_complete() is None


def test_contract_and_progress_are_frozen_by_all_rank_consensus(tmp_path):
    coordinator, manager, _backend, dynamics = _two_rank_coordinator(
        tmp_path,
        mode="progress_mismatch",
    )

    with pytest.raises(CheckpointConsensusError, match="progress differs"):
        coordinator.checkpoint(
            contract=_contract(world_size=2),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(world_size=2),
        )

    assert manager.latest_complete() is None


def test_group_geometry_must_reach_the_same_all_rank_safe_point(tmp_path):
    coordinator, manager, _backend, dynamics = _two_rank_coordinator(
        tmp_path,
        mode="geometry_mismatch",
    )

    with pytest.raises(CheckpointConsensusError, match="geometry differs"):
        coordinator.checkpoint(
            contract=_contract(world_size=2),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(world_size=2),
        )

    assert manager.latest_complete() is None


def test_two_rank_manifest_contains_every_validated_shard(tmp_path):
    coordinator, _manager, backend, dynamics = _two_rank_coordinator(tmp_path)
    committed = coordinator.checkpoint(
        contract=_contract(world_size=2),
        progress=_progress(1, dynamics),
        safe_point=_safe_point(world_size=2),
    )

    manifest = json.loads(
        (committed.path / "coordinator_manifest.json").read_text(encoding="utf-8")
    )
    assert [item["rank"] for item in manifest["shards"]] == [0, 1]
    assert all((committed.path / item["path"]).is_file() for item in manifest["shards"])
    assert backend.barriers == [
        "checkpoint.safe_point",
        "checkpoint.rank_shards_written",
        "checkpoint.committed",
    ]
    rank_one = coordinator.load_rank_snapshot(committed, rank=1)
    assert rank_one.rank == 1
    assert rank_one.world_size == 2


@pytest.mark.parametrize(
    "fault_stage",
    (
        "after_staging_prepare",
        "before_rank_state_write.rank-0",
        "after_rank_state_write.rank-0",
        "before_rank_state_copy.rank-0",
        "after_rank_state_copy.rank-0",
        "after_coordinator_manifest",
        "after_writer",
        "after_complete_marker",
    ),
)
def test_pre_rename_failures_never_publish_a_half_checkpoint(
    tmp_path,
    fault_stage,
):
    dynamics = _selection_policy()
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=SingleProcessCheckpointBackend(),
        collector=_collector(dynamics),
    )

    def fail(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError("injected checkpoint crash")

    with pytest.raises(RuntimeError, match="injected checkpoint crash"):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(),
            fault_injector=fail,
        )

    assert manager.latest_complete() is None
    assert tuple(manager.root.glob("step-1*")) == ()
    assert not (manager.root / "latest.json").exists()
    staging_root = manager.root / ".coordinator-staging"
    assert not staging_root.exists() or tuple(staging_root.iterdir()) == ()


def test_component_capture_failure_happens_before_any_filesystem_transaction(tmp_path):
    dynamics = _selection_policy()
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")

    def fail_capture():
        raise RuntimeError("component state failed")

    collector = CheckpointStateCollector(
        component_state_sources={"model": fail_capture},
        dynamics_selection_policy_source=lambda: dynamics,
    )
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=SingleProcessCheckpointBackend(),
        collector=collector,
    )

    with pytest.raises(RuntimeError, match="component state failed"):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(),
        )

    assert manager.latest_complete() is None
    assert tuple(manager.root.glob("step-*")) == ()


class _CorruptingSingleBackend(SingleProcessCheckpointBackend):
    def __init__(self, root) -> None:
        self.root = root

    def gather_object(self, value, *, dst=0):
        staging = self.root / ".coordinator-staging" / value.descriptor.staging_name
        with (staging / value.descriptor.staging_file).open("ab") as handle:
            handle.write(b"corruption")
        return super().gather_object(value, dst=dst)


def test_rank_shard_is_rehashed_and_validated_before_atomic_commit(tmp_path):
    dynamics = _selection_policy()
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=_CorruptingSingleBackend(manager.root),
        collector=_collector(dynamics),
    )

    with pytest.raises(ValueError, match="digest or size changed"):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(),
        )

    assert manager.latest_complete() is None
    assert tuple(manager.root.glob("step-*")) == ()


@pytest.mark.parametrize("fault_stage", ("after_checkpoint_rename", "after_latest"))
def test_post_rename_failure_leaves_only_a_fully_valid_checkpoint(
    tmp_path,
    fault_stage,
):
    dynamics = _selection_policy()
    manager = AtomicCheckpointManager(tmp_path / "checkpoints")
    coordinator = CheckpointCoordinator(
        manager=manager,
        backend=SingleProcessCheckpointBackend(),
        collector=_collector(dynamics),
    )

    def fail(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError("injected post-rename crash")

    with pytest.raises(RuntimeError, match="injected post-rename crash"):
        coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(),
            fault_injector=fail,
        )

    discovered = manager.latest_complete(expected_contract=_contract())
    assert discovered is not None
    assert discovered.step == 1
    assert (discovered.path / "coordinator_manifest.json").is_file()
    assert (discovered.path / "rank_shards" / "rank-0.pt").is_file()


def test_loaded_rng_snapshot_restores_python_numpy_and_torch_streams(tmp_path):
    original = RankRNGSnapshot.capture_current(0)
    dynamics = _selection_policy()
    frozen_rng = RankRNGSnapshot.capture_current(0)
    coordinator = CheckpointCoordinator(
        manager=AtomicCheckpointManager(tmp_path / "checkpoints"),
        backend=SingleProcessCheckpointBackend(),
        collector=_collector(dynamics, rng=frozen_rng),
    )
    try:
        committed = coordinator.checkpoint(
            contract=_contract(),
            progress=_progress(1, dynamics),
            safe_point=_safe_point(),
        )
        restored = coordinator.load_rank_snapshot(committed)

        frozen_rng.restore_current()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        random.random()
        np.random.random()
        torch.rand(9)
        restored.rng_state.restore_current()
        observed = (random.random(), float(np.random.random()), torch.rand(3))

        assert observed[0] == expected[0]
        assert observed[1] == expected[1]
        assert torch.equal(observed[2], expected[2])
    finally:
        original.restore_current()


class _FakeStrategy:
    rank = 0
    world_size = 2

    def __init__(self):
        self.events = []

    def gather_object(self, value, *, dst=0):
        self.events.append(("gather", value, dst))
        phase = value[0]
        return [(phase, 0), (phase, 1)]

    def broadcast_object(self, value, *, src=0):
        self.events.append(("broadcast", value, src))
        return value

    def failure_gate(self, phase, failure):
        self.events.append(("failure", phase, failure))
        if failure is not None:
            raise failure


def test_strategy_backend_provides_barrier_through_injected_collectives():
    strategy = _FakeStrategy()
    backend = StrategyCheckpointBackend(strategy)

    backend.barrier("safe")

    assert backend.rank == 0
    assert backend.world_size == 2
    assert backend.is_main_process
    assert strategy.events == [
        ("gather", ("safe", 0), 0),
        ("failure", "safe.barrier", None),
    ]
