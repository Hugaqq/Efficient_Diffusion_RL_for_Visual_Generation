"""C3 preflight contracts, all CPU-only and offline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from visual_rl.artifacts.checkpoint import (
    adapter_payload_sha256,
    checkpoint_tree_sha256,
    read_and_validate_training_state,
)
from visual_rl.configs import (
    ExperimentSpec,
    read_experiment_spec,
    read_packaged_preset,
    resolve_experiment,
)
from visual_rl.preflight import (
    PreflightReport,
    ResumePreflightError,
    StaticPreflightError,
    TrustedComponentError,
    latest_committed_step,
    resolve_resume_checkpoint,
    static_preflight,
    trusted_component_load,
    validate_resume_path,
)


def _resolved(tmp_path: Path):
    return resolve_experiment(
        ExperimentSpec(
            user={
                "run_name": "preflight",
                "dataset": {"prompts": ["offline prompt"]},
                "paths": {"output_dir": "not-created"},
                "runner": {"show_progress": False},
            },
            context_dir=tmp_path,
        )
    ).config


def test_static_preflight_has_no_runtime_import_or_directory_side_effect(
    tmp_path, monkeypatch
):
    config = _resolved(tmp_path)
    output_dir = Path(config.paths.output_dir)

    def fail_import(*args, **kwargs):
        raise AssertionError("static preflight attempted a runtime import")

    monkeypatch.setattr("importlib.import_module", fail_import)
    report = static_preflight(config)

    assert report.trusted is False
    assert not output_dir.exists()
    assert {item.kind for item in report.components} == {
        "model",
        "rollout",
        "algorithm",
        "provider",
        "optimizer",
        "reward",
    }
    assert all(item.version for item in report.components)
    assert all(item["version"] for item in report.to_dict()["components"])
    assert report.to_dict()["resolved_config_sha256"] == (report.resolved_config_sha256)


def test_static_preflight_config_fingerprint_and_dict_are_stable(tmp_path):
    config = _resolved(tmp_path)

    first = static_preflight(config)
    second = static_preflight(config)

    assert len(first.resolved_config_sha256) == 64
    assert first.resolved_config_sha256 == second.resolved_config_sha256
    assert first.to_dict() == second.to_dict()


def test_grpo_nonzero_beta_fails_before_runtime_import(tmp_path) -> None:
    config = _resolved(tmp_path)
    config.algorithm.beta = 0.1

    with pytest.raises(
        StaticPreflightError,
        match="GRPO requires beta=0 until differentiable current/reference KL",
    ):
        static_preflight(config)


def test_flash_wan_reference_yaml_still_passes_generic_flash_contract(tmp_path):
    config = resolve_experiment(
        ExperimentSpec(
            preset=read_packaged_preset("flash_wan_reference"),
            context_dir=tmp_path,
        )
    ).config

    report = static_preflight(config)

    assert config.model.extra["wan_backend"] == "flash"
    assert config.sample.name == "single_step"
    assert config.algorithm.objective_version == "reference_v1"
    assert config.algorithm.beta == 0
    assert any(
        item.kind == "algorithm" and item.name == "flash_grpo"
        for item in report.components
    )


def test_flash_reference_yaml_rejects_builtin_tiny_without_coefficient(tmp_path):
    config_path = tmp_path / "tiny-reference.yaml"
    config_path.write_text(
        """\
preset: flash_tiny_single_step
explicit:
  algorithm:
    objective_version: reference_v1
    beta: 0.0
""",
        encoding="utf-8",
    )
    config = resolve_experiment(read_experiment_spec(config_path)).config

    with pytest.raises(
        StaticPreflightError,
        match="Flash-GRPO reference_v1 requires model capability",
    ):
        static_preflight(config)


def test_flash_reference_keeps_external_model_capability_extensible(tmp_path):
    config = resolve_experiment(
        ExperimentSpec(
            preset=read_packaged_preset("flash_tiny_single_step"),
            explicit={
                "model": {
                    "name": "external_flash_model",
                    "extra": {
                        "target": "external_flash_model:ExternalFlashModel",
                        "version": "v1",
                    },
                },
                "algorithm": {
                    "objective_version": "reference_v1",
                    "beta": 0.0,
                },
            },
            context_dir=tmp_path,
        )
    ).config

    report = static_preflight(config)

    model = next(item for item in report.components if item.kind == "model")
    assert model.name == "external_flash_model"
    assert model.target == "external_flash_model:ExternalFlashModel"


def test_static_preflight_rejects_unknown_name_relative_path_and_target(tmp_path):
    config = _resolved(tmp_path)
    config.model.name = "not-built-in"
    config.paths.output_dir = "relative"
    config.rollout["target"] = "elsewhere:Rollout"

    with pytest.raises(StaticPreflightError) as caught:
        static_preflight(config)

    message = str(caught.value)
    assert "not an absolute resolved path" in message
    assert "requires an explicit target" in message
    assert "does not match trusted catalog target" in message


def test_trusted_load_does_not_instantiate_components_or_load_model(
    tmp_path, monkeypatch
):
    config = _resolved(tmp_path)
    report = static_preflight(config)

    from visual_rl.builtins import register_builtin_plugins
    from visual_rl.model_adapters.mock import MockWanAdapter

    register_builtin_plugins()
    monkeypatch.setattr(
        MockWanAdapter,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("adapter instantiated during trusted load")
        ),
    )
    monkeypatch.setattr(
        MockWanAdapter,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model.load called during trusted load")
        ),
        raising=False,
    )

    assert trusted_component_load(config, report).trusted is True


def test_trusted_load_reuses_report_or_computes_one_once(tmp_path, monkeypatch):
    import visual_rl.preflight as preflight_module

    config = _resolved(tmp_path)
    report = static_preflight(config)
    real_static = preflight_module.static_preflight
    calls = 0

    def count_static(candidate):
        nonlocal calls
        calls += 1
        return real_static(candidate)

    monkeypatch.setattr(preflight_module, "static_preflight", count_static)

    trusted = trusted_component_load(config, report)
    assert trusted.trusted is True
    assert trusted.resolved_config_sha256 == report.resolved_config_sha256
    assert calls == 0
    assert trusted_component_load(config).trusted is True
    assert calls == 1


def test_trusted_load_rejects_report_from_another_config_before_import(
    tmp_path, monkeypatch
):
    import visual_rl.preflight as preflight_module

    report = static_preflight(_resolved(tmp_path / "first"))
    other_config = _resolved(tmp_path / "second")
    monkeypatch.setattr(
        preflight_module,
        "_resolve_target",
        lambda target: (_ for _ in ()).throw(
            AssertionError("trusted component import was reached")
        ),
    )

    with pytest.raises(TrustedComponentError, match="resolved_config_sha256"):
        trusted_component_load(other_config, report)


def test_trusted_load_rejects_config_mutated_after_static_before_import(
    tmp_path, monkeypatch
):
    import visual_rl.preflight as preflight_module

    config = _resolved(tmp_path)
    report = static_preflight(config)
    config.runner.show_progress = not config.runner.show_progress
    monkeypatch.setattr(
        preflight_module,
        "_resolve_target",
        lambda target: (_ for _ in ()).throw(
            AssertionError("trusted component import was reached")
        ),
    )

    with pytest.raises(TrustedComponentError, match="resolved_config_sha256"):
        trusted_component_load(config, report)


def test_trusted_load_rejects_report_without_config_fingerprint_before_import(
    tmp_path, monkeypatch
):
    import visual_rl.preflight as preflight_module

    config = _resolved(tmp_path)
    complete_report = static_preflight(config)
    incomplete_report = PreflightReport(complete_report.components)
    monkeypatch.setattr(
        preflight_module,
        "_resolve_target",
        lambda target: (_ for _ in ()).throw(
            AssertionError("trusted component import was reached")
        ),
    )

    with pytest.raises(TrustedComponentError, match="missing resolved_config_sha256"):
        trusted_component_load(config, incomplete_report)


def test_trusted_load_checks_unselected_builtin_interface(tmp_path, monkeypatch):
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    config = _resolved(tmp_path)
    monkeypatch.setattr(SD3TempFlowAdapter, "sample", None)

    with pytest.raises(TrustedComponentError, match="sample"):
        trusted_component_load(config)


def test_unselected_optional_dependency_does_not_block_mock(tmp_path, monkeypatch):
    original_find_spec = importlib.util.find_spec

    def without_diffusers(name, *args, **kwargs):
        if name == "diffusers":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_diffusers)

    assert trusted_component_load(_resolved(tmp_path)).trusted is True


def test_external_target_is_static_without_import_and_trusted_via_registry(
    tmp_path, monkeypatch
):
    module_name = "c3_external_model"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """\
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.plugins import register_model_adapter

class ExternalModel(ModelAdapter):
    def __init__(self, config):
        self.config = config
    @property
    def train_module(self):
        return None
    def parameters(self):
        return []
    def sample(self, prompts, metadata, rollout_config):
        return RolloutBatch(prompts=prompts, metadata=metadata)
    def recompute_log_probs(self, batch):
        return None
    def save_pretrained(self, output_dir):
        return None
    def load_checkpoint(self, checkpoint_dir):
        return None

register_model_adapter("external_model_c3", ExternalModel)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _resolved(tmp_path)
    config.model.name = "external_model_c3"
    config.model.extra.update(
        {"target": f"{module_name}:ExternalModel", "version": "2026.1"}
    )

    sys.modules.pop(module_name, None)
    static_report = static_preflight(config)
    external = next(item for item in static_report.components if item.kind == "model")
    assert module_name not in sys.modules
    assert external.version == "2026.1"
    assert external.source_sha256 is None

    trusted_report = trusted_component_load(config, static_report)
    trusted_external = next(
        item for item in trusted_report.components if item.kind == "model"
    )
    assert trusted_external.source_sha256


def test_selected_legacy_model_without_train_module_fails_trusted_preflight(
    tmp_path, monkeypatch
):
    module_name = "c4_legacy_model"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """\
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.plugins import register_model_adapter

class LegacyModel(ModelAdapter):
    def __init__(self, config):
        self.config = config
    def parameters(self):
        return []
    def sample(self, prompts, metadata, rollout_config):
        return RolloutBatch(prompts=prompts, metadata=metadata)
    def recompute_log_probs(self, batch):
        return None
    def save_pretrained(self, output_dir):
        return None
    def load_checkpoint(self, checkpoint_dir):
        return None

register_model_adapter("c4_legacy_model", LegacyModel)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _resolved(tmp_path)
    config.model.name = "c4_legacy_model"
    config.model.extra.update(
        {"target": f"{module_name}:LegacyModel", "version": "legacy-v1"}
    )

    with pytest.raises(
        TrustedComponentError,
        match="abstract .*train_module|train_module.*abstract",
    ):
        trusted_component_load(config)


def test_selected_legacy_rollout_without_context_fails_trusted_preflight(
    tmp_path, monkeypatch
):
    from visual_rl.configs import schema

    module_name = "c4_legacy_rollout"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """\
from visual_rl.plugins import register_rollout_engine
from visual_rl.rollout.base import RolloutEngine

class LegacyRollout(RolloutEngine):
    def sample(self, adapter, prompts, metadata):
        raise NotImplementedError

register_rollout_engine("c4_legacy_rollout", LegacyRollout)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        schema._ALGORITHM_SAMPLE_PAIRS,
        "grpo",
        {"full_trajectory", "c4_legacy_rollout"},
    )
    config = _resolved(tmp_path)
    config.sample.name = "c4_legacy_rollout"
    config.rollout.update(
        {"target": f"{module_name}:LegacyRollout", "version": "legacy-v1"}
    )

    with pytest.raises(
        TrustedComponentError,
        match=r"sample must accept .*context.*StepContext",
    ):
        trusted_component_load(config)


def test_static_rejects_unknown_without_target_and_wrong_builtin_version(tmp_path):
    unknown = _resolved(tmp_path)
    unknown.model.name = "external_without_target"
    with pytest.raises(StaticPreflightError, match="requires an explicit target"):
        static_preflight(unknown)

    wrong_version = _resolved(tmp_path)
    wrong_version.model.extra["version"] = "wrong"
    with pytest.raises(StaticPreflightError, match="version .* does not match"):
        static_preflight(wrong_version)

    single_argument_config = _resolved(tmp_path)
    single_argument_config.model.extra["version"] = "wrong"
    with pytest.raises(StaticPreflightError, match="version .* does not match"):
        trusted_component_load(single_argument_config)


def test_wan_backend_matrix_does_not_reject_external_rollout(tmp_path):
    config = _resolved(tmp_path)
    config.model.name = "world_r1_wan_legacy"
    config.model.model_family = "wan"
    config.model.extra = {"world_r1_root": str(tmp_path / "world-r1")}
    config.sample.name = "external_wan_rollout"
    config.rollout = {
        "target": "external_wan_rollout:ExternalWanRollout",
        "version": "v1",
    }
    config.algorithm.name = "external_wan_algorithm"
    config.algorithm.params = {
        "target": "external_wan_algorithm:ExternalWanAlgorithm",
        "version": "v1",
    }

    report = static_preflight(config)

    rollout = next(item for item in report.components if item.kind == "rollout")
    assert rollout.name == "external_wan_rollout"
    assert rollout.target == "external_wan_rollout:ExternalWanRollout"


def test_external_metadata_must_be_serializable(tmp_path):
    config = _resolved(tmp_path)
    config.model.name = "external_bad_metadata"
    config.model.extra.update(
        {
            "target": "external_bad_metadata:ExternalModel",
            "version": "v1",
            "metadata": object(),
        }
    )

    with pytest.raises(StaticPreflightError, match="JSON serializable"):
        static_preflight(config)


def test_extra_public_registration_does_not_count_as_catalog_drift(tmp_path):
    from visual_rl.plugins import register_rollout_engine
    from visual_rl.rollout.base import RolloutEngine

    class UserRollout(RolloutEngine):
        def sample(self, adapter, prompts, metadata):
            raise NotImplementedError

    register_rollout_engine("test_user_rollout_c3", UserRollout)
    config = _resolved(tmp_path)

    assert trusted_component_load(config).trusted is True


def test_resume_shape_is_checked_read_only(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ResumePreflightError, match="does not exist"):
        validate_resume_path(missing)

    incomplete = tmp_path / "checkpoint_000001"
    incomplete.mkdir()
    with pytest.raises(ResumePreflightError, match="training_state.pt"):
        validate_resume_path(incomplete)


def _complete_checkpoint(run_root: Path, step: int) -> Path:
    checkpoint = run_root / f"checkpoint_{step:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_state.pt").write_bytes(b"offline-test")
    return checkpoint


def _commit_checkpoint(run_root: Path, step: int) -> Path:
    checkpoint = _complete_checkpoint(run_root, step)
    commits = run_root / "commits"
    commits.mkdir(exist_ok=True)
    expected_name = checkpoint.name
    (commits / f"commit_{step:06d}.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "kind": "artifact_commit",
                "run_id": "preflight-run",
                "transaction_id": f"txn-{step}",
                "commit_id": step,
                "completed_steps": step,
                "checkpoint": {
                    "completed_steps": step,
                    "path": expected_name,
                    "final_path": expected_name,
                    "sha256": checkpoint_tree_sha256(
                        checkpoint,
                        trusted_root=run_root,
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_resume_manifest_and_runner_share_safe_checkpoint_resolution(tmp_path):
    from visual_rl.runner import ExperimentRunner

    run_root = tmp_path / "run"
    checkpoint = _complete_checkpoint(run_root, 3)
    latest = run_root / "latest.json"
    latest.write_text(
        json.dumps({"step": 3, "checkpoint": "checkpoint_000003"}),
        encoding="utf-8",
    )

    expected = (checkpoint.resolve(), 3)
    assert resolve_resume_checkpoint(run_root) == expected
    assert resolve_resume_checkpoint(latest) == expected
    assert resolve_resume_checkpoint(checkpoint) == expected
    assert ExperimentRunner._resolve_resume_checkpoint(latest) == expected
    validate_resume_path(run_root)


@pytest.mark.parametrize("transaction_directory", ["commits", ".staging"])
@pytest.mark.parametrize("resume_shape", ["run", "latest", "checkpoint"])
def test_transactionized_layout_without_marker_never_uses_legacy_fallback(
    tmp_path,
    transaction_directory,
    resume_shape,
):
    run_root = tmp_path / "run"
    checkpoint = _complete_checkpoint(run_root, 1)
    latest = run_root / "latest.json"
    latest.write_text(
        json.dumps({"step": 1, "checkpoint": checkpoint.name}),
        encoding="utf-8",
    )
    (run_root / transaction_directory).mkdir()
    resume_path = {
        "run": run_root,
        "latest": latest,
        "checkpoint": checkpoint,
    }[resume_shape]

    with pytest.raises(
        ResumePreflightError,
        match="Transactionized artifact layout.*authoritative commit marker",
    ):
        resolve_resume_checkpoint(resume_path)


def test_commit_marker_is_authoritative_when_latest_is_stale_or_missing(tmp_path):
    run_root = tmp_path / "run"
    older = _commit_checkpoint(run_root, 1)
    newest = _commit_checkpoint(run_root, 2)
    latest = run_root / "latest.json"
    latest.write_text(
        json.dumps({"step": 1, "checkpoint": older.name}),
        encoding="utf-8",
    )

    assert resolve_resume_checkpoint(run_root) == (newest.resolve(), 2)
    assert resolve_resume_checkpoint(latest) == (newest.resolve(), 2)
    latest.unlink()
    assert resolve_resume_checkpoint(latest) == (newest.resolve(), 2)


def test_commit_resolution_uses_highest_existing_recovery_point(tmp_path):
    run_root = tmp_path / "run"
    older = _commit_checkpoint(run_root, 1)
    newest = _commit_checkpoint(run_root, 2)
    for path in newest.iterdir():
        path.unlink()
    newest.rmdir()

    assert resolve_resume_checkpoint(run_root) == (older.resolve(), 1)
    assert latest_committed_step(run_root) == 2


def test_commit_resolution_does_not_skip_corrupt_authoritative_marker(tmp_path):
    run_root = tmp_path / "run"
    _commit_checkpoint(run_root, 1)
    _commit_checkpoint(run_root, 2)
    (run_root / "commits" / "commit_000002.json").write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(ResumePreflightError, match="Authoritative commit marker"):
        latest_committed_step(run_root)
    with pytest.raises(ResumePreflightError, match="Authoritative commit marker"):
        resolve_resume_checkpoint(run_root)


def test_commit_resolution_rejects_post_commit_checkpoint_tamper(tmp_path):
    run_root = tmp_path / "run"
    checkpoint = _commit_checkpoint(run_root, 1)
    (checkpoint / "training_state.pt").write_bytes(b"tampered after marker")

    with pytest.raises(ResumePreflightError, match="tree SHA256 mismatch"):
        resolve_resume_checkpoint(run_root)
    with pytest.raises(ResumePreflightError, match="tree SHA256 mismatch"):
        resolve_resume_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "checkpoint_value",
    [
        "/tmp/checkpoint_000001",
        "../checkpoint_000001",
        "nested/checkpoint_000001",
        "checkpoint_000002",
    ],
)
def test_resume_manifest_rejects_unsafe_or_wrong_checkpoint_names(
    tmp_path, checkpoint_value
):
    run_root = tmp_path / "run"
    _complete_checkpoint(run_root, 1)
    latest = run_root / "latest.json"
    latest.write_text(
        json.dumps({"step": 1, "checkpoint": checkpoint_value}),
        encoding="utf-8",
    )

    with pytest.raises(ResumePreflightError, match="relative name"):
        validate_resume_path(latest)


def test_resume_manifest_rejects_checkpoint_symlink_escape(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = _complete_checkpoint(tmp_path / "outside", 1)
    (run_root / "checkpoint_000001").symlink_to(outside, target_is_directory=True)
    latest = run_root / "latest.json"
    latest.write_text(
        json.dumps({"step": 1, "checkpoint": "checkpoint_000001"}),
        encoding="utf-8",
    )

    with pytest.raises(ResumePreflightError, match="escapes run root"):
        validate_resume_path(run_root)


def test_resume_rejects_direct_checkpoint_with_wrong_name(tmp_path):
    checkpoint = tmp_path / "checkpoint_1"
    checkpoint.mkdir()
    (checkpoint / "training_state.pt").write_bytes(b"offline-test")

    with pytest.raises(ResumePreflightError, match="Unsupported resume path shape"):
        validate_resume_path(checkpoint)


def test_checkpoint_payload_hash_rejects_file_and_directory_symlinks(tmp_path):
    checkpoint = tmp_path / "checkpoint_000001"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    external_file = tmp_path / "external.safetensors"
    external_file.write_bytes(b"external")
    payload_link = checkpoint / "adapter_model.safetensors"
    payload_link.symlink_to(external_file)

    with pytest.raises(RuntimeError, match="payload must not contain symlinks"):
        adapter_payload_sha256(checkpoint)

    payload_link.unlink()
    external_dir = tmp_path / "external_adapter"
    external_dir.mkdir()
    (external_dir / "adapter_model.safetensors").write_bytes(b"external")
    (checkpoint / "nested_adapter").symlink_to(
        external_dir,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="payload must not contain symlinks"):
        adapter_payload_sha256(checkpoint)


@pytest.mark.parametrize("control_name", ["training_state.pt", "checkpoint.json"])
def test_training_state_read_rejects_symlinked_control_file(tmp_path, control_name):
    checkpoint = tmp_path / "checkpoint_000001"
    checkpoint.mkdir()
    (checkpoint / "training_state.pt").write_bytes(b"not-read")
    (checkpoint / "checkpoint.json").write_text("{}", encoding="utf-8")
    external = tmp_path / f"external-{control_name}"
    external.write_bytes(b"external")
    control = checkpoint / control_name
    control.unlink()
    control.symlink_to(external)

    with pytest.raises(RuntimeError, match=rf"{control_name}.*symlink"):
        read_and_validate_training_state(checkpoint, config={})
