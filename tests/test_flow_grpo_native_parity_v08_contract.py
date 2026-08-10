"""CPU contracts for the fail-closed v0.8 SD3/Flow-GRPO parity entrypoint."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "tests/native_parity/run_flow_grpo_sd3_v08.py"
CASE_PATH = ROOT / "tests/fixtures/native_parity/flow_grpo_sd3_v08_case_v1.json"
CONFIG_PATH = ROOT / "configs/v2/flow_grpo_sd3.yaml"


def _load_harness():
    module_name = "_visualrl_flow_grpo_native_v08_harness"
    spec = importlib.util.spec_from_file_location(module_name, HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v0.8 native parity harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _reference_repo(tmp_path: Path, harness):
    root = tmp_path / "reference"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Native Contract")
    _git(root, "config", "user.email", "native@example.invalid")
    for relative in harness._REFERENCE_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"# {relative}\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "pinned reference")
    revision = _git(root, "rev-parse", "HEAD")
    digest = harness.compute_reference_digest(root)
    return root, revision, digest


def _v08_config(tmp_path: Path) -> tuple[Path, Path]:
    model_artifact = tmp_path / "sd3-artifact"
    model_artifact.mkdir()
    config = tmp_path / "flow_grpo_sd3_v08.yaml"
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "../../checkpoints/stable-diffusion-3.5-medium",
        str(model_artifact),
    )
    config.write_text(text, encoding="utf-8")
    return config, model_artifact


def _arguments(tmp_path: Path, harness, *, preflight_only: bool = False):
    reference, revision, digest = _reference_repo(tmp_path, harness)
    config, _model = _v08_config(tmp_path)
    return harness.HarnessArguments(
        repo_root=ROOT,
        config_path=config,
        case_path=CASE_PATH,
        reference_repo=reference,
        reference_revision=revision,
        reference_digest=digest,
        preflight_only=preflight_only,
    )


def _all_pass_comparisons(harness):
    return {name: {"passed": True} for name in harness._NATIVE_COMPARISON_KEYS}


def test_v08_fixture_preserves_the_independent_group_advantage_oracle() -> None:
    harness = _load_harness()
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    observed = harness.NativeFlowReferenceOracle.group_advantages(
        case["reward_values"],
        1.0e-4,
    )
    torch.testing.assert_close(
        observed,
        torch.tensor(case["expected_advantages"], dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_v08_flow_objective_oracle_preserves_upstream_reduction_semantics() -> None:
    harness = _load_harness()
    old = torch.tensor(
        [[-0.4, -0.1], [-0.3, -0.2]],
        dtype=torch.float64,
    )
    new = torch.tensor(
        [[-0.35, -0.25], [-0.28, -0.05]],
        dtype=torch.float64,
    )
    advantages = torch.tensor(
        [[1.2, -0.7], [0.4, -1.1]],
        dtype=torch.float64,
    )
    current_mean = torch.tensor(
        [
            [[[1.0, 2.0]], [[2.0, 0.5]]],
            [[[0.0, 1.0]], [[-1.0, 2.5]]],
        ],
        dtype=torch.float64,
    )
    reference_mean = current_mean.detach() - 0.2
    transition_std = torch.tensor(
        [[0.5, 0.25], [0.4, 0.8]],
        dtype=torch.float64,
    )
    clip_range = 0.1
    beta = 0.004

    observed = harness.NativeFlowReferenceOracle.evaluate(
        old_log_probs=old,
        new_log_probs=new,
        advantages=advantages,
        current_mean=current_mean,
        reference_mean=reference_mean,
        std_dev=transition_std,
        clip_range=clip_range,
        beta=beta,
    )

    ratio = torch.exp(new - old)
    expected_policy = -torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages,
    ).mean()
    expanded_std = transition_std[:, :, None, None]
    expected_reference_kl = (
        ((current_mean - reference_mean).square() / (2.0 * expanded_std.square()))
        .flatten(start_dim=2)
        .mean(dim=2)
        .mean()
    )
    torch.testing.assert_close(observed["policy_loss"], expected_policy)
    torch.testing.assert_close(observed["reference_kl"], expected_reference_kl)
    torch.testing.assert_close(
        observed["total_loss"],
        expected_policy + beta * expected_reference_kl,
    )


def test_reference_identity_is_explicit_and_has_no_repo_relative_fallback() -> None:
    harness = _load_harness()
    with pytest.raises(
        harness.HarnessArgumentError,
        match="reference identity must be explicit",
    ):
        harness.parse_arguments((), repo_root=ROOT)

    revision = "a" * 40
    digest = "b" * 64
    parsed = harness.parse_arguments(
        (
            "--reference-repo",
            "../code_base/TempFlow-GRPO",
            "--reference-revision",
            revision,
            "--reference-digest",
            digest,
            "--preflight-only",
        ),
        repo_root=ROOT,
    )
    assert parsed.reference_repo == (ROOT / "../code_base/TempFlow-GRPO").resolve()
    assert parsed.reference_revision == revision
    assert parsed.reference_digest == digest
    assert parsed.config_path == CONFIG_PATH
    assert parsed.preflight_only is True


def test_reference_digest_binds_tracked_bytes_and_clean_worktree(tmp_path) -> None:
    harness = _load_harness()
    reference, revision, digest = _reference_repo(tmp_path, harness)
    identity = harness.inspect_reference_identity(reference)
    assert identity.revision == revision
    assert identity.digest == digest

    target = reference / harness._REFERENCE_FILES[0]
    target.write_text("# mutated\n", encoding="utf-8")
    assert harness.compute_reference_digest(reference) != digest
    with pytest.raises(ValueError, match="must be clean"):
        harness.inspect_reference_identity(reference)


def test_missing_execution_dependencies_do_not_block_descriptor_binding(
    tmp_path,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness)

    checks, _case, reference, composition = harness.run_preflight(
        arguments,
        dependency_probe=lambda name: name not in {"accelerate", "peft"},
        cuda_probe=lambda: (False, "cuda_devices=0"),
    )
    by_code = {item.code: item for item in checks}
    assert by_code["python_dependencies"].status == "error"
    assert by_code["python_dependencies"].observed == ["accelerate", "peft"]
    assert by_code["cuda"].status == "error"
    assert by_code["v08_composition"].status == "pass"
    assert reference is not None
    assert composition is not None
    assert composition.model_requested_id == "sd3"
    assert composition.algorithm_requested_id == "flow-grpo"


def test_v08_composition_binds_sd3_flow_module_and_policy_port_without_weights(
    tmp_path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness)

    from visual_rl.models.implementations.sd3 import SD3Adapter

    monkeypatch.setattr(
        SD3Adapter,
        "from_config",
        classmethod(
            lambda cls, config, *, runtime_context: (_ for _ in ()).throw(
                AssertionError("static parity preflight constructed model weights")
            )
        ),
    )
    checks, case, reference, composition = harness.run_preflight(
        arguments,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "cuda:0"),
    )
    failures = tuple(item for item in checks if item.status != "pass")
    assert failures == ()
    assert case is not None
    assert reference is not None
    assert composition is not None
    assert composition.model_requested_id == "sd3"
    assert (
        composition.model_class_path
        == "visual_rl.models.implementations.sd3:SD3Adapter"
    )
    assert composition.algorithm_requested_id == "flow-grpo"
    assert composition.algorithm_class_path.endswith(":FlowGRPOAlgorithmModule")
    assert composition.reference_kl_weight == pytest.approx(0.004)
    assert composition.policy_port_protocol.endswith(":PolicyRuntimePort")
    assert composition.policy_port_implementation.endswith(":DefaultPolicyRuntimePort")
    prefix, digest = composition.algorithm_binding_id.split(":", 1)
    assert prefix == "model-algorithm-binding.v1"
    assert len(digest) == 64
    assert len(composition.algorithm_module_identity) == 64
    assert int(composition.algorithm_module_identity, 16) >= 0


def test_default_execute_mode_is_blocked_without_a_real_sd3_artifact(
    tmp_path,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness)
    report = harness.run_harness(
        arguments,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "cuda:0"),
    )
    assert report["preflight"]["passed"] is True
    assert report["parity_protocol"] == {
        "protocol_id": (
            "flow-grpo-sd3.v08-native-kernel.full-trajectory-single-commit.v1"
        ),
        "reference_surface": "pinned_upstream_numerical_kernels",
        "configuration_source": "resolved_v08_recipe",
        "profile_interpretation": (
            "chosen_kernel_parity_profile_not_upstream_experiment_defaults"
        ),
        "schedule_source": "resolved_v08_recipe",
        "update_cadence": "v08_full_trajectory_single_adamw_commit",
        "excluded_claims": [
            "upstream_train_sd3_end_to_end_update_cadence",
            "upstream_geneval_experiment_default_hyperparameters",
        ],
    }
    assert report["execution"]["status"] == "blocked"
    assert "lacks model_index.json" in report["execution"]["reason"]
    assert report["native_parity_passed"] is False


def test_cpu_contract_executor_closes_all_fourteen_items_without_native_claim(
    tmp_path,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness)
    checks, case, reference, composition = harness.run_preflight(
        arguments,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "contract-only-cuda-probe"),
    )
    assert all(item.status == "pass" for item in checks)
    assert case is not None
    assert reference is not None
    assert composition is not None

    result = harness.run_cpu_contract_executor(
        harness.NativeExecutionRequest(
            arguments=arguments,
            case=case,
            reference=reference,
            composition=composition,
        )
    )

    assert isinstance(result, harness.ContractParityExecution)
    assert result.passed is True
    assert result.evidence_scope == "cpu_fake_contract"
    assert result.executor_identity == "flow-grpo-sd3.v08.cpu-fake-contract.v1"
    assert set(result.comparisons) == set(harness._NATIVE_COMPARISON_KEYS)
    assert all(item["passed"] is True for item in result.comparisons.values())
    assert result.algorithm_binding_id == composition.algorithm_binding_id
    with pytest.raises(TypeError, match="NativeParityExecution"):
        harness._validate_execution_identity(result, reference, composition)


def test_executor_result_must_echo_binding_revision_and_digest(
    tmp_path,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness)

    def mismatched(request):
        return harness.NativeParityExecution(
            passed=True,
            recipe_id="1" * 64,
            bound_contract_id="2" * 64,
            algorithm_module_identity=request.composition.algorithm_module_identity,
            algorithm_binding_id="wrong-binding",
            reference_revision=request.reference.revision,
            reference_digest=request.reference.digest,
            comparisons=_all_pass_comparisons(harness),
            evidence_scope="pinned_upstream_native",
        )

    report = harness.run_harness(
        arguments,
        executor=mismatched,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "cuda:0"),
    )
    assert report["execution"]["status"] == "failed"
    assert "different algorithm binding identity" in report["execution"]["reason"]
    assert report["native_parity_passed"] is False

    def matching(request):
        return harness.NativeParityExecution(
            passed=True,
            recipe_id="1" * 64,
            bound_contract_id="2" * 64,
            algorithm_module_identity=request.composition.algorithm_module_identity,
            algorithm_binding_id=request.composition.algorithm_binding_id,
            reference_revision=request.reference.revision,
            reference_digest=request.reference.digest,
            comparisons=_all_pass_comparisons(harness),
            evidence_scope="pinned_upstream_native",
        )

    report = harness.run_harness(
        arguments,
        executor=matching,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "cuda:0"),
    )
    assert report["execution"]["status"] == "passed"
    assert report["execution"]["identity"] == {
        "recipe_id": "1" * 64,
        "bound_contract_id": "2" * 64,
        "algorithm_module_identity": (
            report["composition"]["algorithm_module_identity"]
        ),
        "algorithm_binding_id": report["composition"]["algorithm_binding_id"],
        "evidence_scope": "pinned_upstream_native",
    }
    assert report["native_parity_passed"] is True


def test_executor_result_must_echo_algorithm_module_identity(
    tmp_path,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness)

    def mismatched(request):
        return harness.NativeParityExecution(
            passed=True,
            recipe_id="1" * 64,
            bound_contract_id="2" * 64,
            algorithm_module_identity="3" * 64,
            algorithm_binding_id=request.composition.algorithm_binding_id,
            reference_revision=request.reference.revision,
            reference_digest=request.reference.digest,
            comparisons=_all_pass_comparisons(harness),
            evidence_scope="pinned_upstream_native",
        )

    report = harness.run_harness(
        arguments,
        executor=mismatched,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "cuda:0"),
    )
    assert report["execution"]["status"] == "failed"
    assert "different algorithm module identity" in report["execution"]["reason"]
    assert report["native_parity_passed"] is False


def test_executor_cannot_claim_pass_with_an_incomplete_comparison_set() -> None:
    harness = _load_harness()
    with pytest.raises(ValueError, match="pinned_upstream_native"):
        harness.NativeParityExecution(
            passed=True,
            recipe_id="1" * 64,
            bound_contract_id="2" * 64,
            algorithm_module_identity="3" * 64,
            algorithm_binding_id="model-algorithm-binding.v1:" + "4" * 64,
            reference_revision="5" * 40,
            reference_digest="6" * 64,
            comparisons=_all_pass_comparisons(harness),
            evidence_scope="cpu_fake_contract",
        )

    with pytest.raises(ValueError, match="exact v0.8 parity item set"):
        harness.NativeParityExecution(
            passed=True,
            recipe_id="1" * 64,
            bound_contract_id="2" * 64,
            algorithm_module_identity="3" * 64,
            algorithm_binding_id="model-algorithm-binding.v1:" + "4" * 64,
            reference_revision="5" * 40,
            reference_digest="6" * 64,
            comparisons={"loss": {"passed": True}},
            evidence_scope="pinned_upstream_native",
        )

    comparisons = _all_pass_comparisons(harness)
    comparisons["gradient"] = {"passed": False}
    with pytest.raises(ValueError, match="conjunction"):
        harness.NativeParityExecution(
            passed=True,
            recipe_id="1" * 64,
            bound_contract_id="2" * 64,
            algorithm_module_identity="3" * 64,
            algorithm_binding_id="model-algorithm-binding.v1:" + "4" * 64,
            reference_revision="5" * 40,
            reference_digest="6" * 64,
            comparisons=comparisons,
            evidence_scope="pinned_upstream_native",
        )


def test_preflight_only_never_claims_native_parity(
    tmp_path,
) -> None:
    harness = _load_harness()
    arguments = _arguments(tmp_path, harness, preflight_only=True)
    report = harness.run_harness(
        arguments,
        dependency_probe=lambda _name: True,
        cuda_probe=lambda: (True, "cuda:0"),
    )
    assert report["preflight"]["passed"] is True
    assert report["mode"] == "preflight"
    assert report["execution"] == {
        "status": "not_run",
        "reason": "explicit preflight-only mode",
        "comparisons": {},
    }
    assert report["native_parity_passed"] is False


def test_cli_missing_identity_is_one_structured_json_and_nonzero() -> None:
    completed = subprocess.run(
        [sys.executable, str(HARNESS_PATH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode != 0
    assert completed.stdout.count("\n") == 1
    report = json.loads(completed.stdout)
    assert report["mode"] == "argument_error"
    assert report["parity_protocol"]["update_cadence"] == (
        "v08_full_trajectory_single_adamw_commit"
    )
    assert report["parity_protocol"]["excluded_claims"] == [
        "upstream_train_sd3_end_to_end_update_cadence",
        "upstream_geneval_experiment_default_hyperparameters",
    ]
    assert report["preflight"]["passed"] is False
    assert report["preflight"]["checks"][0]["code"] == "arguments"
    assert report["native_parity_passed"] is False


def test_v08_harness_uses_canonical_model_and_has_no_legacy_import_path() -> None:
    source = HARNESS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden_import_prefixes = (
        "visual_rl.algorithm_modules",
        "visual_rl.model_adapters",
        "visual_rl.registry",
    )
    assert not any(
        name.startswith(prefix)
        for name in imported
        for prefix in forbidden_import_prefixes
    )
    assert not any(name.startswith("visual_rl.configs") for name in imported)
    for retired_symbol in (
        "ALL_REGISTRIES",
        "ComponentResolver",
        "semantic_config",
    ):
        assert retired_symbol not in source
    assert "SD3TempFlowAdapter" not in source
    assert "ComponentLoader" not in source
    assert "SD3Adapter.from_config(" in source
    assert "visual_rl.models.implementations.sd3:SD3Adapter" in source
    assert "visual_rl.algorithms.modules.flow_grpo:FlowGRPOAlgorithmModule" in source
    assert "CanonicalAlgorithmMaterializer" in source
    assert "AlgorithmExecutionPlan.from_spec(" in source
    assert "compile_recipe_v2(" in source
    assert "PolicyRuntimePort" in source
    assert "cpu_fake_contract" in source
    assert "pinned_upstream_native" in source

    assert not (ROOT / "tests/native_parity/run_flow_grpo_sd3.py").exists()
    assert not (ROOT / "tests/test_flow_grpo_native_parity_contract.py").exists()
