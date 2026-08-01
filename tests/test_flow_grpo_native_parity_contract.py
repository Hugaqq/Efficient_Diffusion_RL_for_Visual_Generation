"""CPU contract for the test-only Flow-GRPO native parity assets."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
import torch

import visual_rl as vr


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "native_parity"
    / "flow_grpo_sd3_case_v1.json"
)
ORACLE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "native_parity"
    / "flow_grpo_oracle_math_v1.json"
)
HARNESS_PATH = ROOT / "tests" / "native_parity" / "run_flow_grpo_sd3.py"
CASE_KEYS = {
    "schema_version",
    "config_path",
    "prompt",
    "seed",
    "logical_step",
    "reward_values",
    "expected_advantages",
}
ORACLE_KEYS = {
    "old_log_probs",
    "new_log_probs",
    "advantages",
    "current_mean",
    "reference_mean",
    "std_dev",
    "clip_range",
    "beta",
    "expected_policy_loss",
    "expected_reference_kl",
    "expected_total_loss",
}
ITEM_KEYS = {
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
}
RESUME_KEYS = {
    "adapter_tensors",
    "optimizer_state",
    "grad_scaler_state",
    "rng_state",
    "next_step_inputs",
    "global_step",
    "non_timing_metrics",
}
TARGET_MODULES = (
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
)


def _load_harness():
    module_name = "_visualrl_flow_grpo_native_harness"
    spec = importlib.util.spec_from_file_location(module_name, HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native parity harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_flow_case_is_frozen_and_matches_resolved_group_contract():
    case = _json(CASE_PATH)
    assert set(case) == CASE_KEYS
    assert case["schema_version"] == 1
    assert case["config_path"] == "configs/flow_grpo_sd3.yaml"
    assert isinstance(case["prompt"], str) and case["prompt"].strip()
    assert type(case["seed"]) is int
    assert 0 <= case["seed"] <= 0xFFFF_FFFF
    assert case["logical_step"] == 0

    rewards = torch.tensor(case["reward_values"], dtype=torch.float64)
    expected = torch.tensor(
        case["expected_advantages"],
        dtype=torch.float64,
    )
    assert rewards.ndim == 1
    assert rewards.numel() == expected.numel() == 8
    assert bool(torch.isfinite(rewards).all())
    assert bool(torch.isfinite(expected).all())
    assert float(rewards.std(unbiased=False)) > 0.0

    harness = _load_harness()
    actual = harness.NativeFlowReferenceOracle.group_advantages(
        rewards,
        1.0e-4,
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("reward_values", [1.0, float("nan")], "finite scalar list"),
        ("expected_advantages", [0.0, float("inf")], "finite scalar list"),
        ("reward_values", [0.5, 0.5], "non-zero variance"),
        ("reward_values", [[0.1], [0.2]], "finite scalar list"),
    ),
)
def test_case_runtime_loader_rejects_nonfinite_nested_or_zero_variance(
    tmp_path,
    field,
    value,
    message,
):
    harness = _load_harness()
    case = _json(CASE_PATH)
    case[field] = value
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        harness._load_case(path)


def test_public_flow_yaml_resolves_to_the_single_canonical_component_chain():
    config = vr.load(ROOT / "configs" / "flow_grpo_sd3.yaml").resolve()
    assert config.run.seed == 41
    assert config.model.name == "sd3_tempflow"
    assert config.model.adapter_checkpoint is None
    assert config.rollout.name == "full_trajectory"
    assert config.algorithm.name == "grpo"
    assert tuple(config.model.params["lora_target_modules"]) == TARGET_MODULES
    assert config.model.params["lora_rank"] == 32
    assert config.model.params["lora_alpha"] == 64
    assert config.model.params["guidance_scale"] == 4.5
    assert config.model.params["resolution"] == 512
    assert config.model.params["max_sequence_length"] == 128
    assert config.rollout.params["num_steps"] == 20
    assert config.rollout.params["samples_per_prompt"] == 8
    assert config.algorithm.params["clip_range"] == 0.001
    assert config.algorithm.params["beta"] == 0.004
    assert config.algorithm.advantage.epsilon == 1.0e-4
    assert config.optimizer.max_grad_norm == 1.0
    assert config.runtime.precision == "fp32"
    assert config.runtime.update_microbatch_size == 8
    assert config.runtime.distributed.mode == "single"
    assert config.runtime.distributed.device == "cuda"


def test_tiny_flow_yaml_uses_the_same_grpo_reference_contract():
    config = vr.load(
        ROOT / "tests" / "fixtures" / "configs" / "flow_grpo_sd3_tiny.yaml"
    ).resolve()
    assert config.model.name == "tiny_diffusion"
    assert config.rollout.name == "full_trajectory"
    assert config.algorithm.name == "grpo"
    assert config.algorithm.params["beta"] == 0.004
    assert config.algorithm.advantage.epsilon == 1.0e-4
    assert config.runtime.precision == "fp32"
    assert config.runtime.distributed.device == "cpu"


def test_independent_oracle_matches_precomputed_float64_scalars():
    fixture = _json(ORACLE_PATH)
    assert set(fixture) == ORACLE_KEYS
    harness = _load_harness()
    output = harness.NativeFlowReferenceOracle.evaluate(
        old_log_probs=fixture["old_log_probs"],
        new_log_probs=fixture["new_log_probs"],
        advantages=fixture["advantages"],
        current_mean=fixture["current_mean"],
        reference_mean=fixture["reference_mean"],
        std_dev=fixture["std_dev"],
        clip_range=fixture["clip_range"],
        beta=fixture["beta"],
    )
    assert float(output["policy_loss"]) == pytest.approx(
        fixture["expected_policy_loss"],
        rel=1.0e-8,
        abs=1.0e-10,
    )
    assert float(output["reference_kl"]) == pytest.approx(
        fixture["expected_reference_kl"],
        rel=1.0e-8,
        abs=1.0e-10,
    )
    assert float(output["total_loss"]) == pytest.approx(
        fixture["expected_total_loss"],
        rel=1.0e-8,
        abs=1.0e-10,
    )


def test_native_attribute_view_has_only_the_frozen_required_surface():
    config = vr.load(ROOT / "configs" / "flow_grpo_sd3.yaml").resolve()
    harness = _load_harness()
    view_type = harness._NativeComputeLogProbView
    view = view_type.from_resolved(config)
    assert dataclasses.is_dataclass(view)
    assert tuple(field.name for field in dataclasses.fields(view)) == (
        "sample",
        "train",
    )
    assert tuple(field.name for field in dataclasses.fields(view.sample)) == (
        "guidance_scale",
        "noise_level",
    )
    assert tuple(field.name for field in dataclasses.fields(view.train)) == (
        "cfg",
    )
    assert view.sample.guidance_scale == 4.5
    assert view.sample.noise_level == 0.7
    assert view.train.cfg is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.sample.noise_level = 0.6


def test_setup_failure_is_nonzero_and_preserves_the_fourteen_item_schema():
    completed = subprocess.run(
        [sys.executable, str(HARNESS_PATH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode != 0
    report = json.loads(completed.stdout)
    assert set(report) == {
        "schema_version",
        "case",
        "config_path",
        "precision",
        "items",
        "overall_pass",
    }
    assert report["schema_version"] == 1
    assert report["overall_pass"] is False
    assert set(report["items"]) == ITEM_KEYS
    for name, item in report["items"].items():
        assert item["passed"] is False
        if name == "checkpoint_resume":
            assert set(item["comparisons"]) == RESUME_KEYS
            assert not any(item["comparisons"].values())
        else:
            assert item["comparisons"] == []


def _all_pass_report(harness):
    items = {}
    for name in ITEM_KEYS:
        if name == "checkpoint_resume":
            items[name] = {
                "passed": True,
                "comparisons": {key: True for key in RESUME_KEYS},
            }
        else:
            items[name] = harness._comparison_item(
                {"value": torch.tensor([1.0])},
                {"value": torch.tensor([1.0])},
            )
    return {
        "schema_version": 1,
        "case": "flow_grpo_sd3_case_v1",
        "config_path": "configs/flow_grpo_sd3.yaml",
        "precision": "fp32",
        "items": items,
        "overall_pass": True,
    }


def test_main_success_dispatch_is_reachable_and_stdout_is_one_json(
    monkeypatch,
    capsys,
):
    harness = _load_harness()
    config = SimpleNamespace(runtime=SimpleNamespace(precision="fp32"))
    view = harness._NativeComputeLogProbView(
        sample=harness._NativeComputeLogProbView.Sample(4.5, 0.7),
        train=harness._NativeComputeLogProbView.Train(True),
    )
    calls = []
    monkeypatch.setattr(harness, "_missing_dependencies", lambda: ())
    monkeypatch.setattr(
        harness,
        "_resolve_setup",
        lambda root, _case: (config, view, root / "reference_code"),
    )

    def run(**kwargs):
        calls.append(kwargs)
        return _all_pass_report(harness)

    monkeypatch.setattr(harness, "_run_real_parity", run)
    assert harness.main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.count("\n") == 1
    assert json.loads(output.out)["overall_pass"] is True
    assert len(calls) == 1
    assert calls[0]["config"] is config
    assert calls[0]["view"] is view
    assert calls[0]["reference_repo"].name == "reference_code"


def test_main_serializer_self_check_degrades_to_failure_schema(
    monkeypatch,
    capsys,
):
    harness = _load_harness()
    config = SimpleNamespace(runtime=SimpleNamespace(precision="fp32"))
    view = harness._NativeComputeLogProbView(
        sample=harness._NativeComputeLogProbView.Sample(4.5, 0.7),
        train=harness._NativeComputeLogProbView.Train(True),
    )
    invalid = _all_pass_report(harness)
    invalid["overall_pass"] = False
    monkeypatch.setattr(harness, "_missing_dependencies", lambda: ())
    monkeypatch.setattr(
        harness,
        "_resolve_setup",
        lambda root, _case: (config, view, root / "reference_code"),
    )
    monkeypatch.setattr(
        harness,
        "_run_real_parity",
        lambda **_kwargs: invalid,
    )
    assert harness.main() == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.out.count("\n") == 1
    assert "native report validation failed" in output.err
    assert report["overall_pass"] is False
    assert set(report["items"]) == ITEM_KEYS
    assert not any(item["passed"] for item in report["items"].values())


def test_canonical_serializer_freezes_schema_order_and_finite_numbers():
    harness = _load_harness()
    report = harness._failure_report(
        case_name="flow_grpo_sd3_case_v1",
        config_path="configs/flow_grpo_sd3.yaml",
        precision="fp32",
    )
    payload = harness._canonical_report_json(report)
    assert payload == json.dumps(
        report,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert payload.startswith('{"case":')
    assert "\n" not in payload

    malformed = dict(report)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="top-level key set"):
        harness._canonical_report_json(malformed)

    success = {
        **report,
        "items": {
            name: (
                {
                    "passed": True,
                    "comparisons": {key: True for key in RESUME_KEYS},
                }
                if name == "checkpoint_resume"
                else harness._comparison_item(
                    {"value": torch.tensor([1.0])},
                    {"value": torch.tensor([1.0])},
                )
            )
            for name in ITEM_KEYS
        },
        "overall_pass": True,
    }
    encoded = harness._canonical_report_json(success)
    assert json.loads(encoded)["overall_pass"] is True
    success["items"]["policy_loss"]["comparisons"][0][
        "max_abs_error"
    ] = math.inf
    with pytest.raises(ValueError, match="finite/non-negative"):
        harness._canonical_report_json(success)

    inconsistent = harness._failure_report(
        case_name="flow_grpo_sd3_case_v1",
        config_path="configs/flow_grpo_sd3.yaml",
        precision="fp32",
    )
    inconsistent["overall_pass"] = True
    with pytest.raises(ValueError, match="overall_pass is inconsistent"):
        harness._canonical_report_json(inconsistent)


def test_tensor_comparator_rejects_missing_nonfinite_shape_and_dtype():
    harness = _load_harness()
    good = harness._comparison_item(
        {"a": torch.tensor([1.0, 2.0])},
        {"a": torch.tensor([1.0, 2.0])},
    )
    assert good["passed"] is True
    assert good["comparisons"][0]["tensor_name"] == "a"

    missing = harness._comparison_item(
        {"a": torch.tensor([1.0])},
        {"b": torch.tensor([1.0])},
    )
    assert missing["passed"] is False
    assert [item["tensor_name"] for item in missing["comparisons"]] == ["a", "b"]
    assert all(math.isfinite(item["max_abs_error"]) for item in missing["comparisons"])

    nonfinite = harness._comparison_item(
        {"a": torch.tensor([float("nan")])},
        {"a": torch.tensor([float("nan")])},
    )
    assert nonfinite["passed"] is False
    harness._canonical_report_json(
        {
            "schema_version": 1,
            "case": "case",
            "config_path": "config.yaml",
            "precision": "fp32",
            "items": {
                name: (
                    {
                        "passed": False,
                        "comparisons": {key: False for key in RESUME_KEYS},
                    }
                    if name == "checkpoint_resume"
                    else (
                        nonfinite
                        if name == "policy_loss"
                        else {"passed": False, "comparisons": []}
                    )
                )
                for name in ITEM_KEYS
            },
            "overall_pass": False,
        }
    )

    assert harness._comparison(
        "shape",
        torch.ones(2),
        torch.ones(3),
    )["passed"] is False
    assert harness._comparison(
        "dtype",
        torch.ones(2, dtype=torch.float32),
        torch.ones(2, dtype=torch.float64),
    )["passed"] is False


def test_explicit_to_default_rng_bridge_is_exact_and_restores_caller():
    harness = _load_harness()
    torch.manual_seed(7103)
    outer = harness._RngSnapshot.capture()
    explicit = torch.Generator(device="cpu").manual_seed(41)
    initial = explicit.get_state().clone()
    expected = torch.randn(5, generator=explicit)
    expected_post = explicit.get_state().clone()

    actual, actual_post = harness._run_with_default_torch_rng(
        device=torch.device("cpu"),
        state=initial,
        callback=lambda: torch.randn(5),
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(actual_post, expected_post)
    assert harness._RngSnapshot.capture().exactly_equal(outer)


@pytest.mark.parametrize(
    "kernel_name",
    ("sde_step_with_logprob", "sd3_sde_step_with_logprob"),
)
def test_sde_trace_supports_reference_and_internal_kernel_names(kernel_name):
    harness = _load_harness()

    def kernel(_scheduler, _model, _timestep, sample, **_kwargs):
        mean = sample + 1.0
        std = torch.ones_like(sample)
        next_sample = mean + 2.0 * std
        return next_sample, torch.zeros(sample.shape[0]), mean, std

    def reference_template():
        sample = torch.zeros(2, 1)
        return sde_step_with_logprob(None, None, None, sample)  # noqa: F821

    def internal_template():
        sample = torch.zeros(2, 1)
        return sd3_sde_step_with_logprob(None, None, None, sample)  # noqa: F821

    namespace = {"torch": torch, kernel_name: kernel}
    template = (
        reference_template
        if kernel_name == "sde_step_with_logprob"
        else internal_template
    )
    pipeline = FunctionType(template.__code__, namespace)
    with harness._trace_sde_draws(pipeline) as draws:
        pipeline()
    assert namespace[kernel_name] is kernel
    assert len(draws) == 1
    torch.testing.assert_close(draws[0], torch.full((2, 1), 2.0))


def test_update_rng_isolation_is_order_independent_and_detects_extra_draw():
    harness = _load_harness()
    random.seed(410)
    np.random.seed(411)
    torch.manual_seed(412)
    outer = harness._RngSnapshot.capture()
    update = harness._RngSnapshot.capture()

    def branch():
        return (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
        )

    forward = harness._run_isolated_branches(
        update,
        {"visual": branch, "native": branch},
        ("visual", "native"),
    )
    reverse = harness._run_isolated_branches(
        update,
        {"visual": branch, "native": branch},
        ("native", "visual"),
    )
    assert harness._deep_equal(forward["visual"][0], forward["native"][0])
    assert forward["visual"][1].exactly_equal(forward["native"][1])
    assert harness._deep_equal(forward["visual"][0], reverse["visual"][0])
    assert forward["visual"][1].exactly_equal(reverse["visual"][1])
    assert harness._RngSnapshot.capture().exactly_equal(outer)

    def mutated():
        torch.rand(())
        return branch()

    control = harness._run_isolated_branches(
        update,
        {"visual": branch, "native": mutated},
        ("visual", "native"),
    )
    assert not harness._deep_equal(control["visual"][0], control["native"][0])
    assert not control["visual"][1].exactly_equal(control["native"][1])
    assert harness._RngSnapshot.capture().exactly_equal(outer)


def _exercise_k_slot_update_rng_isolation(device_name):
    harness = _load_harness()
    device = torch.device(device_name)
    random.seed(7201)
    np.random.seed(7202)
    torch.manual_seed(7203)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7204)
    outer = harness._RngSnapshot.capture()
    try:
        construction = harness._RngSnapshot.capture()
        construction.restore()
        visual_rollout_post = harness._RngSnapshot.capture()
        construction.restore()
        torch.rand((), device=device)
        native_rollout_post = harness._RngSnapshot.capture()
        assert not visual_rollout_post.exactly_equal(native_rollout_post)

        construction.restore()
        update = harness._RngSnapshot.capture()
        batch_size = 4
        microbatch_size = 2
        transition_count = 3
        perm = harness._canonical_permutation(
            batch_size=batch_size,
            seed=991,
            device=device,
        )
        slots = []
        for start in range(0, batch_size, microbatch_size):
            rows = tuple(
                int(value)
                for value in perm[start : start + microbatch_size].tolist()
            )
            for step in range(transition_count):
                slots.append((rows, step))
        expected_k = (batch_size // microbatch_size) * transition_count
        assert len(slots) == expected_k == 6

        def branch(*, extra_cpu=False, extra_device=False):
            if extra_cpu:
                torch.rand(())
            if extra_device:
                torch.rand((), device=device)
            parameter = torch.nn.Parameter(
                torch.tensor([0.125], device=device)
            )
            named = (("adapter.weight", parameter),)
            optimizer = torch.optim.AdamW([parameter], lr=0.01)
            initial = harness._named_tensor_snapshot(named)
            random_values = []
            losses = []

            def evaluate(rows, step):
                cpu_draw = torch.rand(())
                if device.type == "cuda":
                    device_draw = torch.rand((), device=device)
                else:
                    device_draw = cpu_draw
                random_values.append(
                    (
                        cpu_draw.detach().clone(),
                        device_draw.detach().cpu().clone(),
                    )
                )
                coefficient = float(sum(rows) + step + 1)
                scalar = (
                    parameter * coefficient
                    + cpu_draw.to(device)
                    + device_draw * 0.5
                ).square().mean()
                losses.append(scalar.detach().cpu().clone())
                rows_count = len(rows)
                mean = (parameter + scalar).reshape(1, 1, 1).expand(
                    rows_count,
                    1,
                    1,
                )
                return {
                    "new_log_prob": mean[:, :, 0],
                    "current_mean": mean,
                    "reference_mean": torch.zeros_like(mean),
                    "transition_std": torch.ones_like(mean),
                    "policy_loss": scalar,
                    "reference_kl": scalar * 0.0,
                    "total_loss": scalar,
                }

            trace = harness._run_update_window(
                named_parameters=named,
                optimizer=optimizer,
                initial_parameters=initial,
                slots=tuple(slots),
                evaluate_slot=evaluate,
                batch_size=batch_size,
                transition_count=transition_count,
                max_grad_norm=0.5,
            )
            assert trace.backward_count == expected_k
            return trace, tuple(random_values), tuple(losses)

        forward = harness._run_isolated_branches(
            update,
            {
                "visual": lambda: branch(),
                "native": lambda: branch(),
            },
            ("visual", "native"),
        )
        reverse = harness._run_isolated_branches(
            update,
            {
                "visual": lambda: branch(),
                "native": lambda: branch(),
            },
            ("native", "visual"),
        )
        for result in (forward, reverse):
            visual_value, visual_post = result["visual"]
            native_value, native_post = result["native"]
            assert harness._update_traces_exactly_equal(
                visual_value[0],
                native_value[0],
            )
            assert harness._deep_equal(visual_value[1:], native_value[1:])
            assert visual_post.exactly_equal(native_post)
        assert harness._update_traces_exactly_equal(
            forward["visual"][0][0],
            reverse["visual"][0][0],
        )
        assert harness._deep_equal(
            forward["visual"][0][1:],
            reverse["visual"][0][1:],
        )
        assert forward["visual"][1].exactly_equal(reverse["visual"][1])

        cpu_mutation = harness._run_isolated_branches(
            update,
            {
                "visual": lambda: branch(),
                "native": lambda: branch(extra_cpu=True),
            },
            ("visual", "native"),
        )
        assert not harness._update_traces_exactly_equal(
            cpu_mutation["visual"][0][0],
            cpu_mutation["native"][0][0],
        )
        assert not cpu_mutation["visual"][1].exactly_equal(
            cpu_mutation["native"][1]
        )
        if device.type == "cuda":
            device_mutation = harness._run_isolated_branches(
                update,
                {
                    "visual": lambda: branch(),
                    "native": lambda: branch(extra_device=True),
                },
                ("visual", "native"),
            )
            assert not harness._update_traces_exactly_equal(
                device_mutation["visual"][0][0],
                device_mutation["native"][0][0],
            )
            assert not device_mutation["visual"][1].exactly_equal(
                device_mutation["native"][1]
            )
    finally:
        outer.restore()
    assert harness._RngSnapshot.capture().exactly_equal(outer)


def test_k_slot_update_rng_isolation_cpu():
    _exercise_k_slot_update_rng_isolation("cpu")


def test_update_trace_difference_summary_separates_roundoff_from_failure():
    harness = _load_harness()
    rng = harness._RngSnapshot.capture()
    value = torch.tensor([1.0], dtype=torch.float32)
    trace = harness._UpdateTrace(
        current_log_prob=value.clone(),
        current_mean=value.clone(),
        reference_mean=value.clone(),
        transition_std=value.clone(),
        policy_loss=value.clone(),
        reference_kl=value.clone(),
        total_loss=value.clone(),
        pre_clip_gradients={"weight": value.clone()},
        post_clip_gradients={"weight": value.clone()},
        parameter_delta={"weight": value.clone()},
        slot_rng_before=(rng,),
        slot_rng_after=(rng,),
        backward_count=1,
        clip_count=1,
        step_count=1,
        zero_grad_count=2,
    )

    identical = harness._update_trace_difference_summary(trace, trace)
    assert identical == {
        "exact": True,
        "within_cuda_tolerance": True,
        "changed_tensor_count": 0,
        "failed_tolerance_count": 0,
        "counters_equal": True,
        "rng_equal": True,
        "groups": {},
        "largest_differences": [],
    }

    roundoff = dataclasses.replace(
        trace,
        current_log_prob=trace.current_log_prob + 5.0e-7,
    )
    roundoff_summary = harness._update_trace_difference_summary(
        trace,
        roundoff,
    )
    assert roundoff_summary["exact"] is False
    assert roundoff_summary["within_cuda_tolerance"] is True
    assert roundoff_summary["changed_tensor_count"] == 1
    assert roundoff_summary["failed_tolerance_count"] == 0
    assert roundoff_summary["groups"]["current_log_prob"] == {
        "changed_tensor_count": 1,
        "failed_tolerance_count": 0,
        "max_abs_error": pytest.approx(4.76837158203125e-7),
        "max_rel_error": pytest.approx(4.7683693082944956e-7),
    }
    assert roundoff_summary["largest_differences"][0]["tensor_name"] == (
        "current_log_prob"
    )

    mismatch = dataclasses.replace(
        trace,
        current_log_prob=trace.current_log_prob + 1.0e-3,
    )
    mismatch_summary = harness._update_trace_difference_summary(
        trace,
        mismatch,
    )
    assert mismatch_summary["within_cuda_tolerance"] is False
    assert mismatch_summary["failed_tolerance_count"] == 1


def test_cuda_determinism_configuration_is_fixed_and_rejects_drift(monkeypatch):
    harness = _load_harness()
    calls = []
    fake = SimpleNamespace(
        use_deterministic_algorithms=lambda enabled: calls.append(enabled),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                benchmark=True,
                deterministic=False,
                allow_tf32=True,
            ),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=True),
            ),
        ),
    )
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG",
        harness._CUBLAS_WORKSPACE_CONFIG,
    )
    harness._configure_cuda_determinism(fake)
    assert calls == [True]
    assert fake.backends.cudnn.benchmark is False
    assert fake.backends.cudnn.deterministic is True
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.backends.cudnn.allow_tf32 is False

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="requires CUBLAS_WORKSPACE_CONFIG"):
        harness._configure_cuda_determinism(fake)


def test_resume_stage_preserves_result_and_identifies_failure():
    harness = _load_harness()
    assert harness._resume_stage("read", lambda: 17) == 17

    def fail():
        raise ValueError("mixed device")

    with pytest.raises(
        RuntimeError,
        match=(
            "checkpoint/resume stage resumed_update failed: "
            "ValueError: mixed device"
        ),
    ) as captured:
        harness._resume_stage("resumed_update", fail)
    assert isinstance(captured.value.__cause__, ValueError)


def test_native_parity_policy_residency_is_exclusive_and_idempotent(monkeypatch):
    harness = _load_harness()
    calls = []

    class TrainModule:
        def to(self, device):
            calls.append(("to", device))
            return self

    class Adapter:
        device = "cuda:0"
        train_module = TrainModule()
        _policy_active = True
        _text_encoders_active = False
        _vae_active = False

        def _activate_policy_module(self):
            if not self._policy_active:
                calls.append(("activate", self.device))
                self._policy_active = True

    adapter = Adapter()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty",)))

    harness._set_policy_residency(adapter, active=False)
    harness._set_policy_residency(adapter, active=False)
    assert adapter._policy_active is False
    assert calls == [("to", "cpu"), ("empty",)]

    harness._set_policy_residency(adapter, active=True)
    harness._set_policy_residency(adapter, active=True)
    assert adapter._policy_active is True
    assert calls[-1] == ("activate", "cuda:0")


def test_exclusive_policy_recycles_both_branches_before_activation(monkeypatch):
    harness = _load_harness()
    calls = []

    class TrainModule:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            calls.append((self.name, "to", device))
            return self

    class Adapter:
        device = "cuda:0"
        _text_encoders_active = False
        _vae_active = False

        def __init__(self, name):
            self.name = name
            self.train_module = TrainModule(name)
            self._policy_active = True

        def _activate_policy_module(self):
            if not self._policy_active:
                calls.append((self.name, "activate", self.device))
                self._policy_active = True

    active = Adapter("active")
    inactive = Adapter("inactive")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    harness._activate_exclusive_policy(active, inactive)
    assert calls == [
        ("inactive", "to", "cpu"),
        ("active", "to", "cpu"),
        ("active", "activate", "cuda:0"),
    ]
    assert active._policy_active is True
    assert inactive._policy_active is False


def test_native_prompt_payload_uses_the_adapter_offload_lifecycle():
    harness = _load_harness()
    calls = []
    pipeline = SimpleNamespace(
        text_encoder=object(),
        text_encoder_2=object(),
        text_encoder_3=object(),
        tokenizer=object(),
        tokenizer_2=object(),
        tokenizer_3=object(),
    )

    class Adapter:
        device = "cuda:0"
        max_sequence_length = 128
        _text_encoders_active = False
        _policy_active = True

        def __init__(self):
            self.pipeline = pipeline

        def _activate_text_encoders_for_prompt(self):
            calls.append("activate_text")
            self._text_encoders_active = True
            self._policy_active = False

        def _offload_text_encoders(self):
            calls.append("offload_text")
            self._text_encoders_active = False

        def _activate_policy_module(self):
            calls.append("activate_policy")
            self._policy_active = True

    adapter = Adapter()

    def compute(prompts, encoders, tokenizers, max_length, device):
        assert adapter._text_encoders_active is True
        assert len(encoders) == len(tokenizers) == 3
        assert max_length == 128
        assert device == "cuda:0"
        value = 1.0 if prompts[0] else 0.0
        return torch.tensor([value]), torch.tensor([value + 1.0])

    payload = harness._prompt_payload_native(
        SimpleNamespace(compute_text_embeddings=compute),
        adapter,
        ("prompt",),
    )
    assert tuple(payload) == (
        "prompt_embeds",
        "pooled_prompt_embeds",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
    )
    assert calls == ["activate_text", "offload_text", "activate_policy"]
    assert adapter._text_encoders_active is False
    assert adapter._policy_active is True


def test_native_pipeline_proxy_pins_cuda_and_decodes_through_offload_lifecycle():
    harness = _load_harness()
    calls = []

    class Vae:
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

        def decode(self, value, return_dict=False):
            assert return_dict is False
            assert adapter.vae_active is True
            calls.append("decode")
            return (value + 1.0,)

    class Adapter:
        device = torch.device("cuda:0")
        vae_active = False

        def __init__(self):
            self.pipeline = SimpleNamespace(vae=Vae(), marker="original")

        def _activate_vae_for_decode(self):
            calls.append("activate_vae")
            self.vae_active = True

        def _offload_vae_after_decode(self):
            calls.append("offload_vae")
            self.vae_active = False

    adapter = Adapter()
    proxy = harness._NativePipelineProxy(adapter)
    assert proxy._execution_device == torch.device("cuda:0")
    assert proxy.marker == "original"
    proxy.marker = "updated"
    assert adapter.pipeline.marker == "updated"
    decoded = proxy.vae.decode(torch.tensor([2.0]), return_dict=False)
    torch.testing.assert_close(decoded[0], torch.tensor([3.0]))
    assert calls == ["activate_vae", "decode", "offload_vae"]
    assert adapter.vae_active is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_k_slot_update_rng_isolation_cuda():
    _exercise_k_slot_update_rng_isolation("cuda")


def test_k_slot_helper_runs_scaled_backward_then_one_clip_step_and_zero():
    harness = _load_harness()

    class SpyAdamW(torch.optim.AdamW):
        def __init__(self, parameters):
            super().__init__(parameters, lr=0.01)
            self.step_calls = 0
            self.zero_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

        def zero_grad(self, set_to_none=True):
            self.zero_calls += 1
            return super().zero_grad(set_to_none=set_to_none)

    parameter = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
    optimizer = SpyAdamW([parameter])
    named = (("adapter.weight", parameter),)
    initial = harness._named_tensor_snapshot(named)
    slots = (((0,), 0), ((1,), 0), ((2,), 0))
    evaluated = []

    def evaluate(rows, step):
        assert optimizer.step_calls == 0
        assert optimizer.zero_calls == 1
        evaluated.append((tuple(rows), step))
        row = float(rows[0] + 1)
        scalar = (parameter * row).square().mean()
        mean = (parameter * row).reshape(1, 1, 1)
        return {
            "new_log_prob": (parameter * row).reshape(1, 1),
            "current_mean": mean,
            "reference_mean": torch.zeros_like(mean),
            "transition_std": torch.ones_like(mean),
            "policy_loss": scalar,
            "reference_kl": scalar * 0.0,
            "total_loss": scalar,
        }

    trace = harness._run_update_window(
        named_parameters=named,
        optimizer=optimizer,
        initial_parameters=initial,
        slots=slots,
        evaluate_slot=evaluate,
        batch_size=3,
        transition_count=1,
        max_grad_norm=0.25,
    )
    assert evaluated == [((0,), 0), ((1,), 0), ((2,), 0)]
    assert trace.backward_count == 3
    assert trace.clip_count == trace.step_count == 1
    assert trace.zero_grad_count == 2
    assert optimizer.step_calls == 1
    assert optimizer.zero_calls == 2
    assert set(trace.pre_clip_gradients) == {"pre_clip/adapter.weight"}
    assert set(trace.post_clip_gradients) == {"post_clip/adapter.weight"}
    assert set(trace.parameter_delta) == {"adapter.weight"}
    assert trace.current_log_prob.shape == (3, 1)


def test_native_tracker_and_compute_log_prob_call_surface_are_frozen():
    harness = _load_harness()

    class SpyTracker:
        seen_global_std = None

        def __init__(self, global_std):
            SpyTracker.seen_global_std = global_std
            self.stats = {}

        def update(self, prompts, rewards):
            assert len(set(prompts)) == 1
            values = np.asarray(rewards, dtype=np.float64)
            self.stats[prompts[0]] = list(values)
            return (values - values.mean()) / (values.std() + 1.0e-4)

        def clear(self):
            self.stats = {}

    rewards = [0.2, 0.7, -0.1, 1.0]
    advantages = harness._native_tracker_advantages(
        SpyTracker,
        prompt="p",
        rewards=rewards,
    )
    assert SpyTracker.seen_global_std is False
    expected = np.asarray(rewards, dtype=np.float64)
    expected = (expected - expected.mean()) / (expected.std() + 1.0e-4)
    np.testing.assert_allclose(advantages, expected, rtol=0.0, atol=0.0)

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.2))
            self.disabled = False

    class Adapter:
        def __init__(self):
            self.device = torch.device("cpu")
            self.train_module = Transformer()
            self.pipeline = SimpleNamespace(scheduler=object())

        def _disable_lora_reference(self):
            adapter = self

            class Disable:
                def __enter__(self):
                    adapter.train_module.disabled = True

                def __exit__(self, *_exc):
                    adapter.train_module.disabled = False

            return Disable()

    calls = []

    def compute_log_prob(
        transformer,
        pipeline,
        sample,
        j,
        embeds,
        pooled_embeds,
        config,
    ):
        assert pipeline.scheduler is not None
        assert j == 0
        assert config.sample.guidance_scale == 4.5
        assert config.sample.noise_level == 0.7
        assert config.train.cfg is True
        assert embeds.shape[0] == pooled_embeds.shape[0] == 4
        calls.append(transformer.disabled)
        coefficient = 0.0 if transformer.disabled else transformer.weight
        mean = sample["latents"][:, j] + coefficient
        std = torch.ones_like(mean)
        log_prob = mean.flatten(start_dim=1).mean(dim=1)
        return mean, log_prob, mean, std

    adapter = Adapter()
    native_data = {
        "latents": torch.zeros(2, 1, 1, 1),
        "next_latents": torch.zeros(2, 1, 1, 1),
        "timesteps": torch.ones(2, 1, dtype=torch.int64),
        "old_log_probs": torch.zeros(2, 1),
        "prompt_embeds": torch.ones(2, 2),
        "pooled_prompt_embeds": torch.ones(2, 2),
        "negative_prompt_embeds": torch.zeros(2, 2),
        "negative_pooled_prompt_embeds": torch.zeros(2, 2),
    }
    helpers = harness._NativeHelpers(
        compute_text_embeddings=lambda *_args, **_kwargs: None,
        compute_log_prob=compute_log_prob,
        pipeline_with_logprob=lambda *_args, **_kwargs: None,
        tracker_type=SpyTracker,
    )
    output = harness._native_slot_objective(
        helpers=helpers,
        adapter=adapter,
        native_data=native_data,
        view=harness._NativeComputeLogProbView(
            sample=harness._NativeComputeLogProbView.Sample(4.5, 0.7),
            train=harness._NativeComputeLogProbView.Train(True),
        ),
        algorithm=SimpleNamespace(
            adv_clip_max=5.0,
            clip_range=0.2,
            beta=0.004,
        ),
        native_advantage=torch.tensor([1.0, -1.0]),
        rows=(0, 1),
        step=0,
    )
    assert calls == [False, True]
    assert output["new_log_prob"].shape == (2, 1)
    assert output["current_mean"].shape[:2] == (2, 1)
    assert output["reference_mean"].requires_grad is False


def test_resume_projection_mutation_controls_detect_every_semantic_field():
    harness = _load_harness()
    snapshot = harness._RngSnapshot.capture()
    base = harness._ResumeProjection(
        adapter_tensors={"weight": torch.tensor([1.0])},
        optimizer_state={"step": torch.tensor(1)},
        grad_scaler_state=None,
        start_rng=snapshot,
        end_rng=snapshot,
        next_step_inputs={"latents": torch.tensor([2.0])},
        global_step=2,
        non_timing_metrics={"loss": 0.5},
    )
    assert all(harness._compare_resume_projections(base, base).values())

    controls = {
        "adapter_tensors": dataclasses.replace(
            base,
            adapter_tensors={"weight": torch.tensor([1.1])},
        ),
        "optimizer_state": dataclasses.replace(
            base,
            optimizer_state={"step": torch.tensor(2)},
        ),
        "grad_scaler_state": dataclasses.replace(
            base,
            grad_scaler_state={},
        ),
        "next_step_inputs": dataclasses.replace(
            base,
            next_step_inputs={"latents": torch.tensor([3.0])},
        ),
        "global_step": dataclasses.replace(base, global_step=3),
        "non_timing_metrics": dataclasses.replace(
            base,
            non_timing_metrics={"loss": 0.6},
        ),
    }
    for key, mutated in controls.items():
        flags = harness._compare_resume_projections(base, mutated)
        assert flags[key] is False
    outer = harness._RngSnapshot.capture()
    try:
        torch.rand(())
        changed_rng = harness._RngSnapshot.capture()
    finally:
        outer.restore()
    rng_flags = harness._compare_resume_projections(
        base,
        dataclasses.replace(base, end_rng=changed_rng),
    )
    assert rng_flags["rng_state"] is False


def test_native_harness_is_test_only_and_has_no_production_import_path():
    tree = ast.parse(HARNESS_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "visual_rl.runner" not in imported
    assert "visual_rl.artifacts.manager" not in imported
    assert "argparse" not in imported
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert "real CUDA native parity is not run" not in source
    assert "_run_real_parity(" in source
    assert "_pipeline_full" not in source
    assert "raise RuntimeError(\n            \"real CUDA" not in source

    for source_path in (ROOT / "visual_rl").rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "run_flow_grpo_sd3" not in source
        assert "NativeFlowReferenceOracle" not in source
