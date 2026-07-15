"""Static, CPU-only preflight coverage for strict TempFlow contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_rl.configs.schema import load_config
from visual_rl.preflight import StaticPreflightError, static_preflight


ROOT = Path(__file__).resolve().parents[1]
POLICY_PRESET = ROOT / "visual_rl/configs/presets/sd3_tempflow_adapter.yaml"
REFERENCE_PRESET = (
    ROOT / "visual_rl/configs/presets/sd3_tempflow_reference_v1.yaml"
)


@pytest.mark.parametrize("preset", [POLICY_PRESET, REFERENCE_PRESET])
def test_strict_tempflow_presets_pass_static_preflight(preset: Path) -> None:
    report = static_preflight(load_config(preset))

    assert report.trusted is False
    assert any(
        component.kind == "algorithm" and component.name == "tempflow_grpo"
        for component in report.components
    )


def test_policy_identity_keeps_false_adapter_default() -> None:
    config = load_config(POLICY_PRESET)
    config.model.extra.pop("tempflow_reference_mode")

    static_preflight(config)


@pytest.mark.parametrize("reference_mode", [False, None])
def test_reference_v1_requires_explicit_reference_adapter_mode(
    reference_mode: bool | None,
) -> None:
    config = load_config(REFERENCE_PRESET)
    if reference_mode is None:
        config.model.extra.pop("tempflow_reference_mode")
    else:
        config.model.extra["tempflow_reference_mode"] = reference_mode

    with pytest.raises(
        StaticPreflightError,
        match=r"reference_v1 requires explicit .*tempflow_reference_mode=true",
    ):
        static_preflight(config)


def test_policy_identity_rejects_reference_adapter_mode() -> None:
    config = load_config(POLICY_PRESET)
    config.model.extra["tempflow_reference_mode"] = True

    with pytest.raises(
        StaticPreflightError,
        match=r"policy_identity_v1 requires .*tempflow_reference_mode=false",
    ):
        static_preflight(config)


def test_reference_adapter_mode_cannot_hide_behind_legacy_objective() -> None:
    config = load_config(REFERENCE_PRESET)
    config.algorithm.objective_version = "legacy"

    with pytest.raises(
        StaticPreflightError,
        match=r"reference mode requires explicit .*objective_version='reference_v1'",
    ):
        static_preflight(config)


def test_tempflow_reference_mode_requires_native_bool() -> None:
    config = load_config(POLICY_PRESET)
    config.model.extra["tempflow_reference_mode"] = "false"

    with pytest.raises(
        StaticPreflightError,
        match=r"tempflow_reference_mode must be a bool",
    ):
        static_preflight(config)


@pytest.mark.parametrize(
    ("preset", "wrong_recompute", "expected"),
    [
        (POLICY_PRESET, True, "false"),
        (REFERENCE_PRESET, False, "true"),
    ],
)
def test_explicit_recompute_declaration_must_match_derived_mode(
    preset: Path,
    wrong_recompute: bool,
    expected: str,
) -> None:
    config = load_config(preset)
    config.model.extra["recompute_transformer_training"] = wrong_recompute

    with pytest.raises(
        StaticPreflightError,
        match=rf"derived recompute contract .*recompute_transformer_training={expected}",
    ):
        static_preflight(config)


def test_explicit_recompute_declaration_requires_native_bool() -> None:
    config = load_config(POLICY_PRESET)
    config.model.extra["recompute_transformer_training"] = 0

    with pytest.raises(
        StaticPreflightError,
        match=r"recompute_transformer_training must be a bool",
    ):
        static_preflight(config)


@pytest.mark.parametrize(
    ("beta", "message"),
    [
        (-0.1, r"algorithm.beta must be >= 0.0"),
        (0.1, r"requires beta=0"),
    ],
)
def test_strict_tempflow_rejects_nonzero_beta(beta: float, message: str) -> None:
    config = load_config(POLICY_PRESET)
    config.algorithm.beta = beta

    with pytest.raises(StaticPreflightError, match=message):
        static_preflight(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("advantage_dtype", "float32", "advantage_dtype='float64'"),
        ("preserve_advantage_dtype", False, "preserve_advantage_dtype=true"),
        ("noise_enabled", False, "enabled reference_std_dev_t"),
        ("noise_mode", "std_dev_t", "enabled reference_std_dev_t"),
        ("noise_scale", 0.0, "scale must be finite and positive"),
    ],
)
def test_strict_tempflow_math_contract_fails_in_static_preflight(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(POLICY_PRESET)
    if field == "advantage_dtype":
        config.algorithm.advantage_dtype = value
    elif field == "preserve_advantage_dtype":
        config.algorithm.params["preserve_advantage_dtype"] = value
    elif field == "noise_enabled":
        config.algorithm.noise_weighting["enabled"] = value
    elif field == "noise_mode":
        config.algorithm.noise_weighting["mode"] = value
    else:
        config.algorithm.noise_weighting["scale"] = value

    with pytest.raises(StaticPreflightError, match=message):
        static_preflight(config)


def test_reference_v1_rejects_nonreference_noise_level() -> None:
    config = load_config(REFERENCE_PRESET)
    config.sample.noise_level = 0.8

    with pytest.raises(
        StaticPreflightError,
        match=r"frozen SD3 kernel noise_level=0.7",
    ):
        static_preflight(config)


@pytest.mark.parametrize("noise_level", [None, 0.0, -0.1])
def test_strict_tempflow_rejects_invalid_noise_level(
    noise_level: float | None,
) -> None:
    config = load_config(POLICY_PRESET)
    config.sample.noise_level = noise_level

    with pytest.raises(
        StaticPreflightError,
        match=r"noise_level must be a finite positive number",
    ):
        static_preflight(config)


def test_unknown_tempflow_objective_fails_before_runtime_import() -> None:
    config = load_config(POLICY_PRESET)
    config.algorithm.objective_version = "future_v2"

    with pytest.raises(
        StaticPreflightError,
        match=r"objective_version must be one of",
    ):
        static_preflight(config)
