"""Public, CPU-only acceptance tests for the VisualRL research workflow.

This file intentionally tests user-visible behavior through shipped presets and
public APIs. Real SD3/Wan checkpoints, remote reward servers, and GPU numerical
validation belong to explicit validation runs rather than the default suite.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from visual_rl import ExperimentRunner, RewardBatch, RolloutBatch, load_config
from visual_rl.configs.schema import config_to_dict


ROOT = Path(__file__).resolve().parents[1]

WORKFLOWS = [
    pytest.param(
        "visual_rl/configs/presets/world_r1_wan_v02_mock.yaml",
        "full_trajectory",
        "video",
        "rollout_kl_mean",
        id="grpo-full-trajectory",
    ),
    pytest.param(
        "visual_rl/configs/presets/flash_tiny_single_step.yaml",
        "single_step",
        "image",
        "flash_selected_timestep_mean",
        id="flash-single-step",
    ),
    pytest.param(
        "visual_rl/configs/presets/tempflow_tiny_branching.yaml",
        "branching",
        "image",
        "tempflow_active_timestep_frac",
        id="tempflow-branching",
    ),
]

RUN_ARTIFACTS = {
    "config.resolved.json",
    "prompt_set.json",
    "sample_manifest.json",
    "reward_table.json",
    "metrics.jsonl",
    "visual_report.md",
    "latest.json",
    "run_status.json",
}


def _config(preset: str, output_dir: Path, *, steps: int):
    config = load_config(ROOT / preset)
    config.paths.output_dir = str(output_dir)
    config.train.max_steps = steps
    config.train.save_every = 1
    config.runner.show_progress = False
    config.runner.strict_rollout_validation = True
    return config


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_backed_config(
    output_dir: Path,
    train_path: Path,
    evaluation_path: Path,
    *,
    steps: int,
):
    config = _config(
        "visual_rl/configs/presets/flash_tiny_single_step.yaml",
        output_dir,
        steps=steps,
    )
    config.dataset.path = str(train_path)
    config.dataset.prompts = []
    config.dataset.content_sha256 = None
    config.evaluation.path = str(evaluation_path)
    config.evaluation.content_sha256 = None
    return config


def _assert_finite_metrics(metrics: dict) -> None:
    for name in (
        "loss",
        "reward_mean",
        "reward_std",
        "approx_kl",
        "clipfrac",
        "old_logprob_mean",
        "new_logprob_mean",
        "logprob_delta_abs_max",
        "grad_norm",
    ):
        assert name in metrics
        assert math.isfinite(float(metrics[name])), name


@pytest.mark.parametrize(
    ("preset", "rollout_name", "media_type", "workflow_metric"),
    WORKFLOWS,
)
def test_supported_workflows_run_record_and_resume(
    tmp_path,
    preset,
    rollout_name,
    media_type,
    workflow_metric,
):
    """All supported lightweight workflows complete the same public lifecycle."""

    run_dir = tmp_path / rollout_name
    first_runner = ExperimentRunner(_config(preset, run_dir, steps=1))
    first_metrics = first_runner.run()

    assert [row["step"] for row in first_metrics] == [0]
    _assert_finite_metrics(first_metrics[0])
    assert workflow_metric in first_metrics[0]
    assert RUN_ARTIFACTS.issubset({path.name for path in run_dir.iterdir()})
    assert (run_dir / "checkpoint_000001" / "training_state.pt").is_file()
    assert (run_dir / "rollouts" / "batch_000000.pt").is_file()

    manifest = _json(run_dir / "sample_manifest.json")
    assert manifest["records"]
    assert {record["step"] for record in manifest["records"]} == {0}
    assert {record["media_type"] for record in manifest["records"]} == {
        media_type
    }
    assert all(record["rollout_type"] == rollout_name for record in manifest["records"])
    assert all(
        "weighted_total" in record["reward_values"]
        for record in manifest["records"]
    )

    reward_table = _json(run_dir / "reward_table.json")
    assert len(reward_table["records"]) == len(manifest["records"])
    assert _json(run_dir / "latest.json")["step"] == 1
    first_status = _json(run_dir / "run_status.json")
    assert first_status["state"] == "completed"
    assert first_status["valid"] is True
    assert first_status["target_steps"] == 1

    resume_config = _config(preset, run_dir, steps=2)
    resume_config.paths.resume_from = str(run_dir / "latest.json")
    resumed_metrics = ExperimentRunner(resume_config).run()

    assert [row["step"] for row in resumed_metrics] == [1]
    _assert_finite_metrics(resumed_metrics[0])
    assert workflow_metric in resumed_metrics[0]
    assert _json(run_dir / "latest.json")["step"] == 2
    resumed_status = _json(run_dir / "run_status.json")
    assert resumed_status["state"] == "completed"
    assert resumed_status["valid"] is True
    assert resumed_status["start_step"] == 1
    assert resumed_status["target_steps"] == 2
    assert (run_dir / "checkpoint_000002" / "training_state.pt").is_file()
    assert (run_dir / "rollouts" / "batch_000001.pt").is_file()
    resumed_manifest = _json(run_dir / "sample_manifest.json")
    assert {record["step"] for record in resumed_manifest["records"]} == {0, 1}
    assert [
        json.loads(line)["step"]
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ] == [0, 1]


def test_resume_matches_an_uninterrupted_training_run(tmp_path):
    """A one-step checkpoint followed by resume is equivalent to two steps."""

    import torch

    preset = "visual_rl/configs/presets/flash_tiny_single_step.yaml"
    continuous = ExperimentRunner(_config(preset, tmp_path / "continuous", steps=2))
    continuous_metrics = continuous.run()
    expected_parameter = continuous.adapter.color_bias.detach().cpu().clone()

    split_dir = tmp_path / "split"
    ExperimentRunner(_config(preset, split_dir, steps=1)).run()
    resume_config = _config(preset, split_dir, steps=2)
    resume_config.paths.resume_from = str(split_dir / "latest.json")
    resumed = ExperimentRunner(resume_config)
    resumed_metrics = resumed.run()

    assert torch.equal(resumed.adapter.color_bias.detach().cpu(), expected_parameter)
    for name in ("loss", "reward_mean", "approx_kl", "clipfrac"):
        assert resumed_metrics[0][name] == continuous_metrics[1][name]


def test_checkpoint_v2_allows_same_data_content_at_new_paths(tmp_path):
    """V2 records source paths but compares verified content identity."""

    train_a = tmp_path / "source-a" / "train.txt"
    eval_a = tmp_path / "source-a" / "heldout.txt"
    train_b = tmp_path / "source-b" / "train.txt"
    eval_b = tmp_path / "source-b" / "heldout.txt"
    for path in (train_a, eval_a, train_b, eval_b):
        path.parent.mkdir(parents=True, exist_ok=True)
    train_a.write_text("a red cube\na green bus\n", encoding="utf-8")
    eval_a.write_text("a blue vase\n", encoding="utf-8")
    train_b.write_text(train_a.read_text(encoding="utf-8"), encoding="utf-8")
    eval_b.write_text(eval_a.read_text(encoding="utf-8"), encoding="utf-8")

    run_dir = tmp_path / "portable-resume"
    ExperimentRunner(
        _file_backed_config(run_dir, train_a, eval_a, steps=1)
    ).run()
    checkpoint = run_dir / "checkpoint_000001"
    metadata = _json(checkpoint / "checkpoint.json")
    assert metadata["config_fingerprint_version"] == 2
    assert metadata["data_source"] == {
        "train_path": str(train_a),
        "evaluation_path": str(eval_a),
    }
    assert len(metadata["data_identity"]["train"]["content_sha256"]) == 64
    assert len(metadata["data_identity"]["evaluation"]["content_sha256"]) == 64

    resume = _file_backed_config(run_dir, train_b, eval_b, steps=2)
    resume.paths.resume_from = str(checkpoint)
    resumed = ExperimentRunner(resume)
    assert resumed.start_step == 1
    assert [row["step"] for row in resumed.run()] == [1]


def test_checkpoint_v2_rejects_changed_content_and_reports_field(tmp_path):
    """A replaced file cannot inherit optimizer/RNG state from old content."""

    train_path = tmp_path / "train.txt"
    eval_path = tmp_path / "heldout.txt"
    train_path.write_text("original prompt\n", encoding="utf-8")
    eval_path.write_text("heldout prompt\n", encoding="utf-8")
    run_dir = tmp_path / "content-change"
    ExperimentRunner(
        _file_backed_config(run_dir, train_path, eval_path, steps=1)
    ).run()
    checkpoint = run_dir / "checkpoint_000001"

    train_path.write_text("replacement prompt\n", encoding="utf-8")
    changed = _file_backed_config(run_dir, train_path, eval_path, steps=2)
    changed.paths.resume_from = str(checkpoint)
    with pytest.raises(
        RuntimeError,
        match=r"data_identity\.train\.content_sha256",
    ):
        ExperimentRunner(changed)

    train_path.write_text("original prompt\n", encoding="utf-8")
    changed_semantics = _file_backed_config(
        run_dir,
        train_path,
        eval_path,
        steps=2,
    )
    changed_semantics.paths.resume_from = str(checkpoint)
    changed_semantics.algorithm.clip_range = 0.5
    with pytest.raises(
        RuntimeError,
        match=r"training_semantics\.algorithm\.clip_range",
    ):
        ExperimentRunner(changed_semantics)

    missing = _file_backed_config(
        run_dir,
        tmp_path / "missing-train.txt",
        eval_path,
        steps=2,
    )
    missing.paths.resume_from = str(checkpoint)
    with pytest.raises(RuntimeError, match="Cannot validate dataset data source"):
        ExperimentRunner(missing)


def test_checkpoint_fingerprint_versions_fail_closed(tmp_path):
    """V1 stays path-bound; unknown or inconsistent versions are rejected."""

    import torch

    from visual_rl.artifacts.checkpoint import (
        checkpoint_tree_sha256,
        save_json,
        save_training_state,
    )

    train_path = tmp_path / "v1" / "train.txt"
    eval_path = tmp_path / "v1" / "heldout.txt"
    train_path.parent.mkdir(parents=True)
    train_path.write_text("v1 prompt\n", encoding="utf-8")
    eval_path.write_text("v1 heldout\n", encoding="utf-8")
    run_dir = tmp_path / "v1-run"
    runner = ExperimentRunner(
        _file_backed_config(run_dir, train_path, eval_path, steps=1)
    )
    checkpoint = run_dir / "checkpoint_000001"
    runner.adapter.save_pretrained(str(checkpoint))
    legacy_metadata = save_training_state(
        checkpoint,
        optimizer=runner.optimizer,
        plugin=runner.optimizer_plugin,
        step=1,
        config=config_to_dict(runner.config),
        implementation=runner.checkpoint_identity,
        config_fingerprint_version=1,
    )
    assert "config_fingerprint_version" not in legacy_metadata

    same_path = _file_backed_config(run_dir, train_path, eval_path, steps=2)
    same_path.paths.resume_from = str(checkpoint)
    assert ExperimentRunner(same_path).start_step == 1

    moved_train = tmp_path / "v1-moved" / "train.txt"
    moved_eval = tmp_path / "v1-moved" / "heldout.txt"
    moved_train.parent.mkdir(parents=True)
    moved_train.write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")
    moved_eval.write_text(eval_path.read_text(encoding="utf-8"), encoding="utf-8")
    moved = _file_backed_config(run_dir, moved_train, moved_eval, steps=2)
    moved.paths.resume_from = str(checkpoint)
    with pytest.raises(RuntimeError, match="fingerprint v1"):
        ExperimentRunner(moved)

    v2_dir = tmp_path / "version-tamper"
    ExperimentRunner(
        _file_backed_config(v2_dir, train_path, eval_path, steps=1)
    ).run()
    v2_checkpoint = v2_dir / "checkpoint_000001"
    metadata_path = v2_checkpoint / "checkpoint.json"
    metadata = _json(metadata_path)
    original_format_version = metadata["format_version"]
    metadata["format_version"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    unknown = _file_backed_config(v2_dir, train_path, eval_path, steps=2)
    unknown.paths.resume_from = str(v2_checkpoint)
    with pytest.raises(RuntimeError, match="Committed checkpoint tree SHA256 mismatch"):
        ExperimentRunner(unknown)

    marker_path = next((v2_dir / "commits").glob("commit_*.json"))
    marker = _json(marker_path)
    marker["checkpoint"]["sha256"] = checkpoint_tree_sha256(
        v2_checkpoint,
        trusted_root=v2_dir,
    )
    save_json(marker_path, marker)
    with pytest.raises(RuntimeError, match="Unsupported checkpoint format_version"):
        ExperimentRunner(unknown)

    state_path = v2_checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["config_fingerprint_version"] = 99
    torch.save(state, state_path)
    metadata["format_version"] = original_format_version
    metadata["config_fingerprint_version"] = 99
    metadata["training_state_sha256"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    marker["checkpoint"]["sha256"] = checkpoint_tree_sha256(
        v2_checkpoint,
        trusted_root=v2_dir,
    )
    save_json(marker_path, marker)
    with pytest.raises(RuntimeError, match="Unsupported config fingerprint version"):
        ExperimentRunner(unknown)

    state["config_fingerprint_version"] = 2
    torch.save(state, state_path)
    metadata["training_state_sha256"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    marker["checkpoint"]["sha256"] = checkpoint_tree_sha256(
        v2_checkpoint,
        trusted_root=v2_dir,
    )
    save_json(marker_path, marker)
    inconsistent = _file_backed_config(
        v2_dir,
        train_path,
        eval_path,
        steps=2,
    )
    inconsistent.paths.resume_from = str(v2_checkpoint)
    with pytest.raises(RuntimeError, match="version does not match"):
        ExperimentRunner(inconsistent)


def test_checkpoint_integrity_is_validated_before_adapter_load(tmp_path, monkeypatch):
    """Corrupt adapter bytes are rejected before they can mutate model state."""

    from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter

    run_dir = tmp_path / "integrity-before-load"
    ExperimentRunner(
        _config(
            "visual_rl/configs/presets/flash_tiny_single_step.yaml",
            run_dir,
            steps=1,
        )
    ).run()
    checkpoint = run_dir / "checkpoint_000001"
    adapter_path = checkpoint / "tiny_diffusion.pt"
    adapter_path.write_bytes(adapter_path.read_bytes() + b"corrupt")

    load_called = False

    def forbidden_load(self, checkpoint_dir):
        nonlocal load_called
        load_called = True
        raise AssertionError("adapter load must not run before integrity validation")

    monkeypatch.setattr(TinyDiffusionAdapter, "load_checkpoint", forbidden_load)
    resume = _config(
        "visual_rl/configs/presets/flash_tiny_single_step.yaml",
        run_dir,
        steps=2,
    )
    resume.paths.resume_from = str(checkpoint)
    with pytest.raises(RuntimeError, match="Committed checkpoint tree SHA256 mismatch"):
        ExperimentRunner(resume)
    assert load_called is False


def test_checkpoint_v1_migration_and_manifest_schema_are_explicit(tmp_path):
    """Legacy artifacts require named migration; unknown schemas fail closed."""

    from visual_rl.artifacts.checkpoint import (
        migrate_legacy_checkpoint_to_v4,
        save_training_state,
    )
    from visual_rl.artifacts.manifest import SampleManifest

    run_dir = tmp_path / "legacy-source"
    runner = ExperimentRunner(
        _config(
            "visual_rl/configs/presets/flash_tiny_single_step.yaml",
            run_dir,
            steps=1,
        )
    )
    source = tmp_path / "checkpoint-v1"
    runner.adapter.save_pretrained(str(source))
    legacy_metadata = save_training_state(
        source,
        optimizer=runner.optimizer,
        plugin=runner.optimizer_plugin,
        step=1,
        config=config_to_dict(runner.config),
        implementation=runner.checkpoint_identity,
    )
    import torch

    legacy_state_path = source / "training_state.pt"
    legacy_state = torch.load(
        legacy_state_path,
        map_location="cpu",
        weights_only=True,
    )
    legacy_state["format_version"] = 1
    torch.save(legacy_state, legacy_state_path)
    legacy_metadata["format_version"] = 1
    legacy_metadata.pop("training_state_sha256", None)
    (source / "checkpoint.json").write_text(
        json.dumps(legacy_metadata),
        encoding="utf-8",
    )
    destination_root = tmp_path / "migrated"
    destination_root.mkdir()
    destination = destination_root / "checkpoint_000001"
    migrated = migrate_legacy_checkpoint_to_v4(
        source,
        destination,
        config=config_to_dict(runner.config),
        implementation=runner.checkpoint_identity,
        trusted_root=tmp_path,
        destination_root=destination_root,
    )
    assert migrated["format_version"] == 4
    assert migrated["migrated_from_format_version"] == 1
    assert len(migrated["training_state_sha256"]) == 64
    assert len(migrated["adapter_payload_sha256"]) == 64
    assert len(migrated["checkpoint_tree_sha256"]) == 64
    (destination_root / "trigger_decision.json").write_bytes(
        (run_dir / "trigger_decision.json").read_bytes()
    )

    resume = _config(
        "visual_rl/configs/presets/flash_tiny_single_step.yaml",
        tmp_path / "migrated-resume",
        steps=2,
    )
    resume.paths.resume_from = str(destination)
    assert ExperimentRunner(resume).start_step == 1

    legacy_manifest = {"run_id": "legacy", "records": []}
    with pytest.raises(ValueError, match="missing schema_version"):
        SampleManifest.from_dict(legacy_manifest)
    migrated_manifest = SampleManifest.migrate_legacy_to_v2(legacy_manifest)
    assert migrated_manifest.schema_version == "2"
    with pytest.raises(ValueError, match="Unsupported SampleManifest"):
        SampleManifest.from_dict(
            {"run_id": "future", "schema_version": "999", "records": []}
        )


def test_reward_scoring_is_valid_and_cached(tmp_path):
    """The built-in feedback path scores image batches and writes a stable cache."""

    import torch

    from visual_rl.configs.schema import RewardConfig
    from visual_rl.feedback import build_feedback_provider

    media = torch.zeros(2, 3, 4, 4)
    media[0, 0] = 1.0
    media[1, 2] = 1.0
    batch = RolloutBatch(
        prompts=["a red square", "a blue square"],
        metadata=[{}, {}],
        media=media,
        latents=torch.zeros(2, 1, 3, 4, 4),
        next_latents=torch.zeros(2, 1, 3, 4, 4),
        timesteps=torch.zeros(2, 1),
        old_log_probs=torch.zeros(2, 1),
    )
    cache_dir = tmp_path / "reward_cache"
    provider = build_feedback_provider(
        RewardConfig(
            weights={"prompt_color": 1.0},
            clients={"prompt_color": {"name": "prompt_color", "version": "v1"}},
            fail_policy="raise",
        ),
        cache_dir=cache_dir,
    )

    first = provider.score(batch)
    second = provider.score(batch)

    assert isinstance(first, RewardBatch)
    assert first.valid_mask.tolist() == [True, True]
    assert first.weighted_total.tolist() == pytest.approx([1.0, 1.0])
    assert second.weighted_total.tolist() == first.weighted_total.tolist()
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_invalid_contracts_fail_before_training(tmp_path, monkeypatch):
    """Common configuration, batch, grouping, and process errors fail early."""

    import torch

    from visual_rl.configs.schema import VisualRLConfig
    from visual_rl.optimizers.advantages import AdvantageComputer

    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        yaml.safe_dump(
            {
                "run_name": "invalid-pair",
                "sample": {"name": "single_step"},
                "algorithm": {"name": "grpo"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Incompatible config"):
        load_config(invalid_config)

    invalid_batch = RolloutBatch(
        prompts=["one", "two"],
        metadata=[{}],
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(2, 1, 3, 4, 4),
        next_latents=torch.zeros(2, 1, 3, 4, 4),
        timesteps=torch.zeros(2, 1),
        old_log_probs=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="prompts and metadata"):
        invalid_batch.validate_lightweight(strict=True)

    advantages = AdvantageComputer(reward_weights={"reward": 1.0})
    with pytest.raises(ValueError, match="at least two samples"):
        advantages.compute(
            ["only prompt"],
            {"reward": torch.tensor([1.0])},
            torch.tensor([1.0]),
        )

    output_dir = tmp_path / "multi-process-must-not-exist"
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = VisualRLConfig(run_name="multi-process")
    config.paths.output_dir = str(output_dir)
    with pytest.raises(ValueError, match="Incomplete distributed environment"):
        ExperimentRunner(config)
    assert not output_dir.exists()

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    deterministic_config = VisualRLConfig(run_name="deterministic")
    deterministic_config.seed = 123
    deterministic_config.runner.deterministic_runtime = True
    deterministic_config.paths.output_dir = str(tmp_path / "deterministic")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=123"):
        ExperimentRunner(deterministic_config)


def test_deterministic_runtime_is_explicit_and_checkpoint_bound(tmp_path):
    """The validated runtime is opt-in, fail-fast, and part of identity."""

    script = """
import json
import sys
from visual_rl import ExperimentRunner, load_config

config = load_config(sys.argv[1])
config.paths.output_dir = sys.argv[2]
config.runner.deterministic_runtime = True
config.runner.show_progress = False
runner = ExperimentRunner(config)
print(json.dumps({
    "runtime": runner.runtime_identity,
    "identity_runtime": runner.checkpoint_identity["runtime"],
}, sort_keys=True))
"""
    preset = ROOT / "visual_rl/configs/presets/tempflow_tiny_branching.yaml"
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "7"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(preset),
            str(tmp_path / "deterministic-subprocess"),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    runtime = payload["runtime"]
    assert runtime == payload["identity_runtime"]
    assert runtime["enabled"] is True
    assert runtime["pythonhashseed"] == "7"
    assert runtime["python_ignore_environment"] is False
    assert runtime["cublas_workspace_config"] == ":4096:8"
    assert runtime["deterministic_algorithms"] is True
    assert runtime["cudnn_deterministic"] is True
    assert runtime["cudnn_benchmark"] is False
    assert runtime["matmul_allow_tf32"] is False
    assert runtime["cudnn_allow_tf32"] is False


def test_deterministic_runtime_rejects_interpreter_that_ignores_environment():
    """-E/-I must not turn a visible env string into false hash determinism."""

    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from visual_rl.core.determinism import configure_runtime; "
        "configure_runtime(enabled=True, seed=7)"
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "7"

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "started with -E or -I" in completed.stderr


def test_installed_style_cli_runs_in_a_clean_process(tmp_path):
    """The repository entrypoint works without test-only registration side effects."""

    run_dir = tmp_path / "cli-run"
    config = _config(
        "visual_rl/configs/presets/flash_tiny_single_step.yaml",
        run_dir,
        steps=1,
    )
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(
        yaml.safe_dump(config_to_dict(config), sort_keys=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(config_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_dir / "sample_manifest.json").is_file()
    assert _json(run_dir / "latest.json")["step"] == 1


def test_heavy_adapters_register_without_loading_models():
    """Importing built-ins exposes SD3/Wan adapters without checkpoints or CUDA."""

    from visual_rl.builtins import register_builtin_plugins
    from visual_rl.core.registry import MODEL_ADAPTERS

    register_builtin_plugins()

    assert {"tiny_diffusion", "sd3_tempflow", "world_r1_wan_legacy"}.issubset(
        MODEL_ADAPTERS.keys()
    )
    sd3 = MODEL_ADAPTERS.get("sd3_tempflow")(
        {"name": "sd3_tempflow", "model_path": "", "extra": {"defer_load": True}}
    )
    wan = MODEL_ADAPTERS.get("world_r1_wan_legacy")(
        {"name": "world_r1_wan_legacy", "model_path": ""}
    )
    assert sd3.pipeline is None
    assert wan.pipeline is None
    assert wan.repo_root.name == "World-R1-main"
    assert wan.flash_repo_root.name == "Flash-GRPO-main"


def test_tempflow_policy_identity_preset_keeps_upstream_training_math():
    """The strict SD3 preset keeps TempFlow math without reference-mode drift."""

    import numpy as np
    import torch

    from visual_rl.optimizers.advantages import AdvantageComputer
    from visual_rl.optimizers.factory import build_algorithm

    config = load_config(
        ROOT / "visual_rl/configs/presets/sd3_tempflow_adapter.yaml"
    )
    assert config.algorithm.clip_range == pytest.approx(1e-4)
    assert config.algorithm.advantage_epsilon == pytest.approx(1e-4)
    assert config.algorithm.advantage_dtype == "float64"
    assert config.algorithm.noise_weighting == {
        "enabled": True,
        "mode": "reference_std_dev_t",
        "scale": 2.25,
    }
    assert config.train.max_grad_norm == pytest.approx(1.0)
    assert config.model.extra["tempflow_reference_mode"] is False
    assert config.algorithm.objective_version == "policy_identity_v1"

    advantage = AdvantageComputer(
        reward_weights={"reward": 1.0},
        epsilon=config.algorithm.advantage_epsilon,
        output_dtype=config.algorithm.advantage_dtype,
    ).compute(
        ["same", "same"],
        {"reward": torch.tensor([0.0, 1.0])},
        torch.tensor([0.0, 1.0]),
    ).advantages
    expected = (np.asarray([0.0, 1.0]) - 0.5) / (0.5 + 1e-4)
    assert advantage.dtype == torch.float64
    assert advantage.numpy() == pytest.approx(expected)

    batch = RolloutBatch(
        prompts=["same", "same"],
        metadata=[{}, {}],
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(2, 1, 3, 4, 4),
        next_latents=torch.zeros(2, 1, 3, 4, 4),
        timesteps=torch.zeros(2, 1),
        old_log_probs=torch.zeros(2, 1),
        model_metadata={
            "transition_std_dev_t": [0.7],
            "tempflow_reference_mode": False,
            "trajectory_contract_version": "sd3_tempflow_v3",
            "recompute_transformer_training": False,
            "branching_mode": "shared_prefix",
        },
    )
    algorithm = build_algorithm(config.algorithm)
    weights = algorithm._noise_weights(batch, torch.zeros(2, 1))
    assert weights.flatten().tolist() == pytest.approx([1.575, 1.575])
    loss, _metrics = algorithm.compute_loss(
        batch,
        advantage,
        torch.zeros(2, 1),
    )
    assert loss.dtype == torch.float64


def test_flash_reference_rectification_uses_actual_scheduler_timesteps():
    """Flash's published table is keyed by scheduler values, not list positions."""

    from visual_rl.rollout.rectification import scheduler_rectification_weights

    weights = scheduler_rectification_weights(
        [0, 1, 9],
        num_steps=20,
        mode="flash_reference_table",
        timestep_values=[999, 982, 785],
    )
    expected = [7.4770, 7.0414, 3.7754]
    expected = [value / (sum(expected) / len(expected)) for value in expected]
    assert weights == pytest.approx(expected)

    with pytest.raises(ValueError, match="Unsupported Flash reference timestep"):
        scheduler_rectification_weights(
            [0],
            num_steps=20,
            mode="flash_reference_table",
            timestep_values=[998],
        )


def test_wan_flash_native_single_step_contract_and_recompute():
    """The Wan adapter retains and trains exactly Flash's stochastic transition."""

    import torch

    from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter

    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))

        @property
        def dtype(self):
            return self.scale.dtype

        @property
        def is_gradient_checkpointing(self):
            return False

        def forward(self, hidden_states, **_kwargs):
            return (hidden_states * self.scale,)

    scheduler = SimpleNamespace(timesteps=torch.tensor([999, 982, 963]))

    def flash_pipeline(_pipeline, *, index, prompt_embeds, train_cfg, **_kwargs):
        assert train_cfg is False
        batch_size = prompt_embeds.shape[0]
        prompt_value = prompt_embeds[:, 0, 0].reshape(batch_size, 1, 1, 1, 1)
        before = prompt_value + float(index + 1)
        after = before - 0.5
        return (
            before.reshape(batch_size, 1, 1, 1, 1).repeat(1, 3, 5, 4, 4),
            [before, after],
            [torch.full((batch_size,), float(index) / 10.0)],
            [
                torch.full((batch_size,), float(step + 1) / 8.0)
                for step in range(3)
            ],
            index,
        )

    def flash_sde(
        _scheduler,
        model_output,
        _timestep,
        _sample,
        *,
        prev_sample,
        return_dt_and_std_dev_t=False,
    ):
        log_prob = model_output.mean(dim=(1, 2, 3, 4))
        base = (prev_sample, log_prob, model_output, torch.ones_like(model_output))
        if not return_dt_and_std_dev_t:
            return base
        return (*base, torch.ones_like(model_output), torch.ones_like(model_output))

    adapter = WorldR1WanLegacyAdapter(
        {
            "world_r1_root": str(ROOT),
            "flash_grpo_root": str(ROOT),
            "wan_backend": "flash",
            "device": "cpu",
            "train_cfg": False,
            "guidance_scale": 1.0,
            "gradient_checkpointing": False,
            "wan_pipeline_with_logprob": flash_pipeline,
            "sde_step_with_logprob": flash_sde,
        }
    )
    adapter.pipeline = SimpleNamespace()
    adapter.transformer = FakeTransformer()
    adapter.pipeline.transformer = adapter.transformer
    adapter.scheduler = scheduler
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32

    def encode_prompt_embeds(prompt_values, **_kwargs):
        values = torch.tensor(
            [0.0 if prompt == "same" else float(len(prompt)) for prompt in prompt_values]
        ).reshape(-1, 1, 1)
        return values, None

    adapter._encode_prompt_embeds = encode_prompt_embeds

    batch = adapter.sample_single_step(
        ["same", "same"],
        [{}, {}],
        {
            "selected_timestep_indices": [1, 1],
            "num_steps": 3,
            "num_videos_per_prompt": 1,
            "frames": 5,
            "height": 4,
            "width": 4,
            "seed": 7,
        },
    )
    assert batch.latents.shape == (2, 1, 1, 1, 1, 1)
    assert batch.next_latents.shape == batch.latents.shape
    assert batch.old_log_probs.shape == (2, 1)
    assert batch.timesteps.tolist() == [[982], [982]]
    assert batch.kl.flatten().tolist() == pytest.approx([0.25, 0.25])
    assert batch.model_metadata["wan_backend"] == "flash"
    assert batch.model_metadata["actual_scheduler_timesteps"] == [982, 982]

    new_log_probs = adapter.recompute_log_probs(batch)
    assert new_log_probs.shape == (2, 1)
    assert new_log_probs.flatten().tolist() == pytest.approx([0.5, 0.5])
    new_log_probs.mean().backward()
    assert adapter.transformer.scale.grad.item() == pytest.approx(2.0)

    heterogeneous = adapter.sample_single_step(
        ["one", "two"],
        [{"row": 0}, {"row": 1}],
        {
            "selected_timestep_indices": [0, 1],
            "num_steps": 3,
            "num_videos_per_prompt": 1,
            "seed": 11,
        },
    )
    scalar = [
        adapter.sample_single_step(
            [prompt],
            [{"row": row}],
            {
                "selected_timestep_indices": [index],
                "num_steps": 3,
                "num_videos_per_prompt": 1,
                "seed": 11,
            },
        )
        for row, (prompt, index) in enumerate(zip(["one", "two"], [0, 1]))
    ]
    for field in (
        "media",
        "latents",
        "next_latents",
        "timesteps",
        "old_log_probs",
        "kl",
    ):
        expected = torch.cat([getattr(item, field) for item in scalar], dim=0)
        assert torch.equal(getattr(heterogeneous, field), expected)
    assert heterogeneous.prompts == ["one", "two"]
    assert heterogeneous.metadata == [{"row": 0}, {"row": 1}]
    assert heterogeneous.model_metadata["selected_timestep_indices"] == [0, 1]
    assert heterogeneous.model_metadata["actual_scheduler_timesteps"] == [999, 982]
    assert torch.equal(
        adapter.recompute_log_probs(heterogeneous),
        torch.cat([adapter.recompute_log_probs(item) for item in scalar], dim=0),
    )


def test_prompt_split_validation_rejects_normalized_and_near_duplicates():
    from visual_rl.datasets.prompt_dataset import validate_prompt_splits

    with pytest.raises(ValueError, match="normalized=1"):
        validate_prompt_splits(["A red square!"], ["a red square"])
    with pytest.raises(ValueError, match="near_duplicate=1"):
        validate_prompt_splits(
            ["a red square with one small tree"],
            ["a red square with a small tree"],
        )


def test_prompt_split_validation_reports_clean_audit():
    from visual_rl.datasets.prompt_dataset import validate_prompt_splits

    result = validate_prompt_splits(
        ["a red square in a studio"],
        ["a blue whale swimming in the ocean"],
    )
    assert result["overlap_count"] == 0
    assert result["leakage_audit"] == {
        "near_duplicate_threshold": 0.92,
        "exact_overlap_count": 0,
        "exact_overlap_prompt_ids": [],
        "normalized_overlap_count": 0,
        "normalized_overlap_pairs": [],
        "near_duplicate_count": 0,
        "near_duplicate_pairs": [],
        "leakage_detected": False,
    }


def test_cross_run_aggregation_separates_execution_from_pixel_guardrails():
    from visual_rl.evaluation.cross_run import aggregate_sd3_run_summaries

    def summary(seed, condition, values, *, pixel_guardrail=True):
        cluster_means = {
            str(eval_seed): value
            for eval_seed, value in zip((1701, 1702, 1703), values, strict=True)
        }
        delta = {
            "eval_seed_cluster_means": cluster_means,
            "per_color": {
                color: {"eval_seed_cluster_means": cluster_means}
                for color in ("red", "green", "blue")
            },
        }
        return {
            "valid": True,
            "condition": condition,
            "seed": seed,
            "target_step": 10,
            "prompt_splits": {"heldout": {"content_sha256": "heldout"}},
            "preview_artifacts": {
                "after": {
                    "seeds": [1701, 1702, 1703],
                    "sample_count": 3,
                }
            },
            "model_path": "/models/sd3",
            "resolution": 256,
            "num_steps": 10,
            "guidance_scale": 4.5,
            "branch_count": 6,
            "sample_batch_size": 1,
            "heldout_paired_delta": delta,
            "gates": {"pixel_diversity_guardrail": pixel_guardrail},
        }

    active = [
        summary(201, "active", [0.1, 0.2, 0.3]),
        summary(307, "active", [0.2, 0.3, 0.4], pixel_guardrail=False),
        summary(419, "active", [0.3, 0.4, 0.5]),
    ]
    controls = [
        summary(seed, "zero_lr_control", [0.0, 0.0, 0.0])
        for seed in (201, 307, 419)
    ]
    result = aggregate_sd3_run_summaries(
        active,
        controls,
        bootstrap_samples=100,
        bootstrap_seed=20260714,
    )
    assert result["gates"]["all_runs_execution_valid"] is True
    assert result["gates"]["all_runs_pixel_guardrails_valid"] is False
    assert result["eligible_for_effectiveness_claim"] is False


def test_prompt_dataset_empty_rows_require_an_explicit_policy(tmp_path):
    from visual_rl.datasets.prompt_dataset import PromptDataset

    source = tmp_path / "prompts.txt"
    source.write_text("first\n\nsecond\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty prompt row at line 2"):
        PromptDataset.from_config({"path": str(source)})
    dataset = PromptDataset.from_config(
        {"path": str(source), "empty_prompt_policy": "skip"}
    )
    assert dataset.source_prompts == ["first", "second"]
    assert dataset.empty_prompt_policy == "skip"


def test_rollout_cache_validates_triplet_and_rejects_corrupt_media(tmp_path):
    import torch

    from visual_rl.rollout.cache import RolloutCache

    cache = RolloutCache(tmp_path)
    batch = RolloutBatch(
        prompts=["prompt"],
        metadata=[{"prompt_id": "id"}],
        media=torch.zeros(1, 3, 4, 4),
        latents=torch.zeros(1, 1, 2),
        next_latents=torch.zeros(1, 1, 2),
        timesteps=torch.zeros(1, 1),
        old_log_probs=torch.zeros(1, 1),
    )
    cache.save(0, batch)
    assert cache.validate_step(0)["valid"] is True
    (tmp_path / "batch_000000.media.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="media payload is corrupt"):
        cache.validate_step(0)


def test_default_empty_prompt_policy_preserves_v2_fingerprint_compatibility():
    from visual_rl.artifacts.checkpoint import config_fingerprint

    legacy = {
        "seed": 1,
        "dataset": {"prompts": ["clean prompt"], "split_name": "train"},
        "evaluation": {},
    }
    strict_default = json.loads(json.dumps(legacy))
    strict_default["dataset"]["empty_prompt_policy"] = "error"
    explicit_skip = json.loads(json.dumps(legacy))
    explicit_skip["dataset"]["empty_prompt_policy"] = "skip"
    assert config_fingerprint(legacy) == config_fingerprint(strict_default)
    assert config_fingerprint(legacy) != config_fingerprint(explicit_skip)


def test_reward_cache_quarantines_corruption_and_rebuilds(tmp_path):
    from visual_rl.feedback.cache import RewardCache

    cache = RewardCache(tmp_path)
    cache.set("key", {"values": [1.0], "metadata": {"version": "v1"}})
    assert cache.get("key")["values"] == [1.0]
    (tmp_path / "key.json").write_text('{"values": [', encoding="utf-8")
    assert cache.get("key") is None
    quarantined = list(tmp_path.glob("key.corrupt-*.json"))
    assert len(quarantined) == 1
    cache.set("key", {"values": [2.0], "metadata": {"version": "v1"}})
    assert cache.get("key")["values"] == [2.0]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("execution_mode", "reference_mode"),
    [
        ("reference-compatible", True),
        ("policy-identity", False),
    ],
)
def test_sd3_bounded_trainer_execution_mode_is_explicit(
    tmp_path,
    monkeypatch,
    execution_mode,
    reference_mode,
):
    """The bounded trainer keeps parity and policy-identity modes separate."""

    monkeypatch.syspath_prepend(str(ROOT))
    from scripts.legacy_cli import _sd3_bounded_trainer_config

    args = SimpleNamespace(
        adapter="sd3_tempflow",
        model_path="/models/sd3",
        repo_root="/reference/tempflow",
        output_dir=str(tmp_path / execution_mode),
        resume_from=None,
        seed=201,
        resolution=256,
        dtype="bfloat16",
        lora_rank=8,
        lora_alpha=16,
        max_sequence_length=128,
        device="cuda",
        disable_lora=False,
        train_prompts_file=None,
        heldout_prompts_file=None,
        prompt="a red cube",
        eval_seeds=[1701, 1702, 1703],
        eval_max_prompts=None,
        sample_batch_size=1,
        branch_count=6,
        num_steps=10,
        guidance_scale=4.5,
        reward_name="prompt_color_guarded",
        steps=10,
        condition="active",
        disable_rollout_cache=True,
        deterministic_runtime=True,
        allow_initial_clipping=False,
        logprob_atol=0.0,
        tempflow_execution_mode=execution_mode,
    )
    config = _sd3_bounded_trainer_config(args)

    assert config.model.extra["tempflow_reference_mode"] is reference_mode
    assert config.optimizer.params["max_initial_logprob_delta"] == 0.0
    assert config.optimizer.params["require_initial_clipfrac_zero"] is True


def test_legacy_numeric_helpers_accept_categorical_branch_id_lists():
    from scripts.legacy_cli import _shape_list, _tensor_finite

    assert _shape_list([0, 1]) == [2]
    assert _tensor_finite([0, 1]) is True
    assert _tensor_finite([0, float("inf")]) is False


def test_world_r1_reward_3d_client_matches_reference_wire_format(monkeypatch):
    import io
    import pickle

    import torch
    pil_image = pytest.importorskip("PIL.Image")

    from visual_rl.feedback.world_r1_rewards import WorldR1Reward3DClient

    captured = {}

    def post_bytes(url, payload, *, timeout):
        captured.update(
            {
                "url": url,
                "payload": pickle.loads(payload),
                "timeout": timeout,
            }
        )
        return pickle.dumps(
            {
                "outputs": [0.6, 1.5],
                "details": [
                    {
                        "gs_score": 0.1,
                        "meta_score": 0.2,
                        "camera_motion_score": 0.3,
                        "final_score": 0.6,
                        "gs_video_path": "first.mp4",
                        "meta_view_path": "first.png",
                        "trajectory_comparison_path": "first.png",
                    },
                    {
                        "gs_score": 0.4,
                        "meta_score": 0.5,
                        "camera_motion_score": 0.6,
                        "final_score": 1.5,
                        "gs_video_path": "second.mp4",
                        "meta_view_path": "second.png",
                        "trajectory_comparison_path": "second.png",
                    },
                ],
            }
        )

    monkeypatch.setattr("visual_rl.feedback.clients._post_bytes", post_bytes)
    media = torch.zeros(2, 3, 3, 4, 4)
    media[0, :, 0] = 1.0
    media[1, :, 2] = 1.0
    client = WorldR1Reward3DClient(
        url="http://127.0.0.1:18089",
        timeout=7.0,
        retries=0,
    )
    values, metadata = client.score(
        media,
        ["red", "blue"],
        [{"camera_trajectory": [1, 2]}, {}],
    )

    assert values.tolist() == pytest.approx([0.6, 1.5])
    assert captured["payload"]["prompts"] == ["red", "blue"]
    assert captured["payload"]["camera_trajectories"] == [[1, 2], None]
    assert len(captured["payload"]["videos"]) == 2
    assert all(len(video) == 3 for video in captured["payload"]["videos"])
    decoded = pil_image.open(io.BytesIO(captured["payload"]["videos"][0][0]))
    assert decoded.size == (4, 4)
    assert metadata["score_reconstruction"] == pytest.approx([0.1, 0.4])
    assert metadata["score_meta_view"] == pytest.approx([0.2, 0.5])
    assert metadata["score_trajectory_alignment"] == pytest.approx(
        [0.3, 0.6]
    )

    monkeypatch.setattr(
        "visual_rl.feedback.clients._post_bytes",
        lambda *_args, **_kwargs: pickle.dumps(
            {
                "outputs": [0.0, 0.0],
                "details": [
                    {
                        "gs_score": 0.0,
                        "meta_score": 0.0,
                        "camera_motion_score": 0.0,
                        "final_score": 0.0,
                        "gs_video_path": "",
                        "meta_view_path": "",
                    },
                    {
                        "gs_score": 0.0,
                        "meta_score": 0.0,
                        "camera_motion_score": 0.0,
                        "final_score": 0.0,
                        "gs_video_path": "",
                        "meta_view_path": "",
                    },
                ],
            }
        ),
    )
    with pytest.raises(RuntimeError, match="empty reconstruction artifacts"):
        client.score(media, ["red", "blue"], [{}, {}])


def test_world_r1_general_client_samples_and_encodes_one_frame(monkeypatch):
    import io
    import pickle

    import torch
    pil_image = pytest.importorskip("PIL.Image")

    from visual_rl.feedback.world_r1_rewards import WorldR1RewardGeneralClient

    captured = {}

    def post_bytes(_url, payload, *, timeout):
        captured["payload"] = pickle.loads(payload)
        captured["timeout"] = timeout
        return pickle.dumps({"outputs": [0.25, 0.75]})

    monkeypatch.setattr("visual_rl.feedback.clients._post_bytes", post_bytes)
    media = torch.zeros(2, 3, 3, 4, 4)
    media[0, 1, 0] = 1.0
    media[1, 1, 2] = 1.0
    client = WorldR1RewardGeneralClient(
        url="http://127.0.0.1:18090",
        frame_index=1,
        timeout=5.0,
        retries=0,
    )
    values, metadata = client.score(media, ["red", "blue"], [{}, {}])

    assert values.tolist() == pytest.approx([0.25, 0.75])
    assert captured["payload"]["prompts"] == ["red", "blue"]
    assert len(captured["payload"]["images"]) == 2
    decoded = pil_image.open(io.BytesIO(captured["payload"]["images"][0]))
    assert decoded.size == (4, 4)
    assert metadata["payload_kind"] == "images"
    assert metadata["frame_sampling"]["selected_frame_index"] == 1


def test_remote_reward_client_rejects_malformed_response(monkeypatch):
    import pickle

    from visual_rl.feedback.clients import RemotePickleRewardClient

    monkeypatch.setattr(
        "visual_rl.feedback.clients._post_bytes",
        lambda *_args, **_kwargs: pickle.dumps({"outputs": [1.0]}),
    )
    client = RemotePickleRewardClient(
        url="http://127.0.0.1:18091",
        retries=0,
        allow_unsafe_pickle=True,
        trusted_hosts=["127.0.0.1"],
    )
    with pytest.raises(ValueError, match="expected 2 scores"):
        client.score(None, ["one", "two"], [{}, {}])
