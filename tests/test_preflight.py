"""Focused tests for the single CPU-only preflight path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from visual_rl.api_types import ValidationReport
from visual_rl.core.types import (
    FrozenMapping,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
)
from visual_rl.preflight import (
    _validate_capabilities,
    _validate_runtime_environment,
    backend_for,
)


def _runtime(
    *,
    mode: str = "single",
    device: str = "cpu",
    precision: str = "fp32",
) -> SimpleNamespace:
    return SimpleNamespace(
        precision=precision,
        distributed=SimpleNamespace(mode=mode, device=device),
    )


def test_backend_mapping_has_one_explicit_v07_matrix() -> None:
    assert backend_for("single", "cpu") is None
    assert backend_for("single", "cuda") is None
    assert backend_for("ddp", "cpu") == "gloo"
    assert backend_for("ddp", "cuda") == "nccl"


def test_single_environment_snapshot_is_complete_and_frozen() -> None:
    environment = {"CUDA_VISIBLE_DEVICES": ""}

    validated, checks = _validate_runtime_environment(_runtime(), environment)

    assert checks == ()
    assert validated is not None
    assert (
        validated.mode,
        validated.rank,
        validated.local_rank,
        validated.world_size,
        validated.local_world_size,
    ) == ("single", 0, 0, 1, 1)
    assert validated.group_rank is None
    assert validated.group_world_size is None
    assert validated.master_addr is None
    assert validated.master_port is None
    assert validated.visible_gpu_count == 0
    assert validated.raw_launch_env["RANK"] is None


def test_single_rejects_any_torchrun_launch_variable() -> None:
    validated, checks = _validate_runtime_environment(
        _runtime(),
        {"RANK": "0"},
    )

    assert validated is None
    assert [item.code for item in checks] == ["runtime.unexpected_launch_env"]
    assert checks[0].volatile is True


def test_ddp_cpu_accepts_only_complete_single_node_two_rank_shape() -> None:
    environment = {
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }

    validated, checks = _validate_runtime_environment(
        _runtime(mode="ddp"),
        environment,
    )

    assert checks == ()
    assert validated is not None
    assert (validated.rank, validated.local_rank) == (1, 1)
    assert (validated.world_size, validated.local_world_size) == (2, 2)
    assert (validated.group_rank, validated.group_world_size) == (0, 1)
    assert (validated.master_addr, validated.master_port) == ("127.0.0.1", 29500)


def test_ddp_rejects_partial_and_invalid_topology_without_snapshot() -> None:
    partial, partial_checks = _validate_runtime_environment(
        _runtime(mode="ddp"),
        {"RANK": "0", "WORLD_SIZE": "2"},
    )
    assert partial is None
    assert [item.code for item in partial_checks] == [
        "runtime.incomplete_launch_env"
    ]

    invalid, invalid_checks = _validate_runtime_environment(
        _runtime(mode="ddp"),
        {
            "RANK": "2",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "70000",
        },
    )
    assert invalid is None
    assert {item.code for item in invalid_checks} == {
        "runtime.unsupported_world_size",
        "runtime.invalid_rank_topology",
        "runtime.invalid_master_port",
    }


def test_cuda_visibility_is_checked_before_runtime_construction() -> None:
    validated, checks = _validate_runtime_environment(
        _runtime(device="cuda"),
        {"CUDA_VISIBLE_DEVICES": ""},
    )

    assert validated is None
    assert [item.code for item in checks] == [
        "runtime.insufficient_visible_gpus"
    ]


def test_capability_matching_uses_each_reward_items_own_params() -> None:
    observed: list[object] = []

    class Factory:
        @classmethod
        def required_capabilities(cls, params):
            observed.append(params["token"])
            return frozenset({params["require"]})

    model = SimpleNamespace(
        factory=Factory,
        provides=frozenset({"media.image"}),
        requires=frozenset(),
    )
    reward_a = SimpleNamespace(
        factory=Factory,
        provides=frozenset(),
        requires=frozenset(),
    )
    reward_b = SimpleNamespace(
        factory=Factory,
        provides=frozenset(),
        requires=frozenset(),
    )
    selected = (
        (
            "model",
            "tiny",
            {"token": "model", "require": "media.image"},
            model,
        ),
        (
            "reward",
            "first",
            {"token": "first", "require": "media.image"},
            reward_a,
        ),
        (
            "reward",
            "second",
            {"token": "second", "require": "media.video"},
            reward_b,
        ),
    )

    checks = _validate_capabilities(selected)

    assert observed == ["model", "first", "second"]
    assert [item.code for item in checks] == ["component.missing_capability"]
    assert checks[0].path == "reward.second"


def test_component_environment_checks_keep_structured_order(tmp_path: Path) -> None:
    first = ValidationCheck(
        level="warning",
        code="first",
        path="model.params",
        message="first",
        volatile=False,
    )
    second = ValidationCheck(
        level="error",
        code="second",
        path="model.params",
        message="second",
        volatile=True,
    )

    class Factory:
        @classmethod
        def check_environment(cls, params, context):
            assert params == {"value": 1}
            assert context == ValidationContext(
                phase="validate",
                config_dir=tmp_path,
                distributed_mode="single",
                world_size=1,
                backend=None,
                device="cpu",
                timeout_s=10.0,
            )
            return (first, second)

    from visual_rl.preflight import _validate_component_environment

    spec = SimpleNamespace(factory=Factory)
    selected = (("model", "tiny", {"value": 1}, spec),)
    context = ValidationContext(
        phase="validate",
        config_dir=tmp_path,
        distributed_mode="single",
        world_size=1,
        backend=None,
        device="cpu",
        timeout_s=10.0,
    )

    assert _validate_component_environment(selected, context) == (first, second)


def test_run_phase_replaces_volatile_checks_and_rejects_snapshot_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import visual_rl.preflight as preflight

    static = ValidationCheck("error", "static", "model", "static", False)
    old_volatile = ValidationCheck("error", "old", "dataset", "old", True)
    fresh_volatile = ValidationCheck(
        "error",
        "fresh",
        "dataset",
        "fresh",
        True,
    )
    cached_report = ValidationReport(
        checks=(static, old_volatile),
        runtime_rank=0,
        runtime_world_size=1,
    )
    cached_env = ValidatedRuntimeEnv(
        mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        group_rank=None,
        group_world_size=None,
        master_addr=None,
        master_port=None,
        visible_gpu_count=0,
        raw_launch_env=FrozenMapping({"RANK": None}),
    )
    fresh_env = ValidatedRuntimeEnv(
        mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        group_rank=None,
        group_world_size=None,
        master_addr=None,
        master_port=None,
        visible_gpu_count=1,
        raw_launch_env=FrozenMapping({"RANK": None}),
    )
    config = SimpleNamespace(
        runtime=SimpleNamespace(
            distributed=SimpleNamespace(
                mode="single",
                device="cpu",
                timeout_s=10.0,
            )
        )
    )
    monkeypatch.setattr(preflight, "_selected_components", lambda config: ())
    monkeypatch.setattr(
        preflight,
        "_validate_component_contracts",
        lambda selected: (),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_capabilities",
        lambda selected: (),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_group_size",
        lambda selected, config: (),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_global_paths",
        lambda config: (fresh_volatile,),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_runtime_environment",
        lambda runtime, environment: (fresh_env, ()),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_component_environment",
        lambda selected, context: (),
    )

    report, runtime_env = preflight.run_preflight(
        config,
        config_dir=tmp_path,
        phase="run",
        cached_report=cached_report,
        cached_env=cached_env,
        environ={},
    )

    assert runtime_env is None
    assert [item.code for item in report.checks] == [
        "static",
        "fresh",
        "runtime.launch_environment_drift",
    ]
    assert report.runtime_rank is None
    assert report.runtime_world_size is None
