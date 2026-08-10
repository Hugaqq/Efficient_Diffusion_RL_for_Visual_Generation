"""Typed training semantics and location-only schema-v2 launch contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_rl.composition.config import (
    LaunchSpec,
    RewardRuntimeBindingSpec,
    SpecValidationError,
    TrainingSpec,
    bootstrap_recipe_v2,
    compile_recipe_v2,
    load_source_recipe,
)
from visual_rl.composition.recipes import (
    apply_recipe_overrides,
    builtin_recipe_definitions,
    get_recipe_definition,
)
from visual_rl.errors import ConfigError


def _valid_launch_yaml(*, output_dir: str = "runs/demo") -> str:
    return f"""launch:
  output_dir: {output_dir}
  resume_from: checkpoints/step-10
  checkpoint_every_optimizer_steps: 25
  artifacts:
    model: artifacts/model
    datasets:
      main: datasets/main
    rewards:
      reward_quality: rewards/quality
"""


def _remote_reward_binding_yaml(*, endpoint: str = "http://127.0.0.1:8090") -> str:
    return _valid_launch_yaml().replace(
        "  artifacts:\n",
        f"""  reward_runtime_bindings:
    reward_quality:
      execution_domain: remote
      device: cpu
      dtype: fp32
      endpoint: {endpoint}
      timeout_s: 30.0
      trusted_hosts: [127.0.0.1]
      ca_bundle: certs/reward-ca.pem
      max_response_bytes: 1048576
  artifacts:
""",
        1,
    )


def _write_source(tmp_path: Path, text: str, *, name: str = "recipe.yaml"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_source_recipe(path)


def test_all_six_builtins_publish_the_exact_upstream_training_baseline() -> None:
    generic_default = TrainingSpec.default()

    assert generic_default.to_payload() == {
        "schema_version": 1,
        "seed": 42,
        "global_prompt_batch_size": 1,
        "max_optimizer_steps": 1_000,
        "gradient_accumulation_steps": 1,
        "adamw": {
            "learning_rate": 1.0e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
        },
        "lr_schedule": {
            "kind": "cosine",
            "warmup_steps": 10,
            "min_lr_ratio": 0.0,
        },
        "update_safety": {
            "require_finite_gradients": True,
            "require_nonzero_gradients": True,
            "max_grad_norm": 1.0,
            "max_initial_logprob_delta": 1.0e-4,
            "require_initial_clipfrac_zero": True,
            "zero_grad_set_to_none": True,
            "scaler_skip_policy": "do_not_commit",
            "post_optimizer_failure_policy": "poison_and_restore",
        },
        "policy_recompute": {
            "row_microbatch_size": None,
            "transition_window_size": 1,
        },
    }
    expected_payload = generic_default.to_payload()
    adamw = dict(expected_payload["adamw"])
    adamw["weight_decay"] = 1.0e-4
    expected_payload["adamw"] = adamw
    expected = TrainingSpec.from_mapping(expected_payload)
    for definition in builtin_recipe_definitions():
        assert definition.training == expected


def test_training_change_changes_canonical_resolved_identity(tmp_path) -> None:
    baseline_source = _write_source(
        tmp_path / "baseline",
        "schema_version: 2\nrecipe: flow_grpo_v1\n",
    )
    changed_source = _write_source(
        tmp_path / "changed",
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        "overrides:\n  training:\n    seed: 43\n",
    )
    baseline = compile_recipe_v2(baseline_source)
    changed = compile_recipe_v2(changed_source)

    assert baseline.training.seed == 42
    assert changed.training.seed == 43
    assert baseline.resolved_fingerprint != changed.resolved_fingerprint


def test_policy_recompute_geometry_is_typed_and_recipe_identified() -> None:
    definition = get_recipe_definition("flow_grpo_v1")
    changed = apply_recipe_overrides(
        definition,
        {
            "training": {
                "policy_recompute": {
                    "row_microbatch_size": 2,
                    "transition_window_size": 1,
                }
            }
        },
    )

    assert definition.training.policy_recompute.row_microbatch_size is None
    assert changed.training.policy_recompute.row_microbatch_size == 2
    assert definition != changed


def test_launch_location_changes_do_not_change_resolved_or_recipe_identity(
    tmp_path,
) -> None:
    first_source = _write_source(
        tmp_path / "first",
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _valid_launch_yaml(output_dir="runs/first"),
    )
    second_source = _write_source(
        tmp_path / "second",
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _valid_launch_yaml(output_dir="runs/second"),
    )
    first_bootstrap = bootstrap_recipe_v2(first_source)
    second_bootstrap = bootstrap_recipe_v2(second_source)
    first = compile_recipe_v2(first_source)
    second = compile_recipe_v2(second_source)

    assert first_bootstrap.require_launch() != second_bootstrap.require_launch()
    assert first_source.config_source_id != second_source.config_source_id
    assert first.resolved_fingerprint == second.resolved_fingerprint
    assert "launch" not in first.canonical_semantic_payload()


def test_launch_is_optional_for_static_compile_but_explicitly_required_for_run(
    tmp_path,
) -> None:
    source = _write_source(
        tmp_path,
        "schema_version: 2\nrecipe: flow_grpo_v1\n",
    )
    bootstrap = bootstrap_recipe_v2(source)

    assert bootstrap.launch is None
    with pytest.raises(ConfigError, match="requires an explicit launch") as caught:
        bootstrap.require_launch()
    assert caught.value.key == "launch"
    assert caught.value.path == str(source.path)


def test_launch_paths_resolve_only_against_the_source_config_directory(
    tmp_path,
) -> None:
    source = _write_source(
        tmp_path / "configs" / "nested",
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _valid_launch_yaml(output_dir="../../runs/demo"),
    )

    launch = bootstrap_recipe_v2(source).require_launch()

    assert isinstance(launch, LaunchSpec)
    assert launch.output_dir == tmp_path / "runs" / "demo"
    assert launch.resume_from == source.context.config_dir / "checkpoints" / "step-10"
    assert launch.artifacts.model == source.context.config_dir / "artifacts" / "model"
    assert launch.artifacts.dataset("main") == (
        source.context.config_dir / "datasets" / "main"
    )
    assert launch.artifacts.reward("reward_quality") == (
        source.context.config_dir / "rewards" / "quality"
    )


def test_remote_reward_runtime_binding_is_typed_launch_only_input(
    tmp_path: Path,
) -> None:
    first_source = _write_source(
        tmp_path / "first",
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _remote_reward_binding_yaml(endpoint="http://127.0.0.1:8090"),
    )
    second_source = _write_source(
        tmp_path / "second",
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _remote_reward_binding_yaml(endpoint="http://127.0.0.1:9090"),
    )

    first_launch = bootstrap_recipe_v2(first_source).require_launch()
    second_launch = bootstrap_recipe_v2(second_source).require_launch()
    binding = first_launch.reward_runtime_binding("reward_quality")

    assert binding is not None
    assert binding.execution_domain == "remote"
    assert binding.endpoint == "http://127.0.0.1:8090"
    assert binding.ca_bundle == first_source.context.config_dir / "certs/reward-ca.pem"
    assert first_launch.reward_runtime_binding("missing") is None
    assert first_launch != second_launch
    first_recipe = compile_recipe_v2(first_source)
    second_recipe = compile_recipe_v2(second_source)
    assert first_recipe.resolved_fingerprint == (second_recipe.resolved_fingerprint)
    assert "reward_runtime_bindings" not in first_recipe.canonical_semantic_payload()


def test_reward_runtime_binding_cannot_reference_an_unavailable_artifact(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _remote_reward_binding_yaml().replace(
            "    reward_quality:\n",
            "    unknown_reward:\n",
            1,
        ),
    )

    with pytest.raises(ConfigError, match="unknown reward artifacts") as caught:
        bootstrap_recipe_v2(source)
    assert caught.value.key == "launch.reward_runtime_bindings"


@pytest.mark.parametrize("device", ("cpu:garbage", "cuda::0", "not-a-device"))
def test_reward_runtime_binding_rejects_noncanonical_torch_devices(
    tmp_path: Path,
    device: str,
) -> None:
    source = _write_source(
        tmp_path,
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _remote_reward_binding_yaml().replace(
            "      device: cpu",
            f"      device: {device}",
        ),
    )

    with pytest.raises(ConfigError, match="canonical torch device") as caught:
        bootstrap_recipe_v2(source)
    assert caught.value.key == ("launch.reward_runtime_bindings.reward_quality.device")


@pytest.mark.parametrize(
    "trusted_hosts_yaml",
    (
        "127.0.0.1",
        "[bad host]",
        "[host%en0]",
    ),
)
def test_reward_runtime_binding_rejects_nonsequence_or_invalid_trusted_hosts(
    tmp_path: Path,
    trusted_hosts_yaml: str,
) -> None:
    source = _write_source(
        tmp_path,
        "schema_version: 2\nrecipe: flow_grpo_v1\n"
        + _remote_reward_binding_yaml().replace(
            "      trusted_hosts: [127.0.0.1]",
            f"      trusted_hosts: {trusted_hosts_yaml}",
        ),
    )

    with pytest.raises(ConfigError) as caught:
        bootstrap_recipe_v2(source)
    assert caught.value.key.startswith(
        "launch.reward_runtime_bindings.reward_quality.trusted_hosts"
    )


def test_reward_runtime_binding_canonicalizes_and_sorts_trusted_hosts() -> None:
    binding = RewardRuntimeBindingSpec(
        artifact_ref="reward_quality",
        execution_domain="remote",
        device="cpu",
        dtype="fp32",
        endpoint="https://example.com",
        timeout_s=30.0,
        trusted_hosts=("LOCALHOST.", "Example.COM"),
        max_response_bytes=1024,
    )

    assert binding.trusted_hosts == ("example.com", "localhost")
    with pytest.raises(SpecValidationError, match="unique after canonicalization"):
        RewardRuntimeBindingSpec(
            artifact_ref="reward_quality",
            execution_domain="remote",
            device="cpu",
            dtype="fp32",
            endpoint="https://example.com",
            timeout_s=30.0,
            trusted_hosts=("example.com", "EXAMPLE.COM."),
            max_response_bytes=1024,
        )


@pytest.mark.parametrize(
    ("launch_yaml", "expected_key"),
    (
        (
            _valid_launch_yaml() + "  surprise: true\n",
            "launch.surprise",
        ),
        (
            _valid_launch_yaml().replace(
                "checkpoint_every_optimizer_steps: 25",
                "checkpoint_every_optimizer_steps: true",
            ),
            "launch.checkpoint_every_optimizer_steps",
        ),
        (
            _valid_launch_yaml().replace("output_dir: runs/demo", 'output_dir: ""'),
            "launch.output_dir",
        ),
        (
            _valid_launch_yaml().replace(
                "      main: datasets/main",
                "      Bad.Id: datasets/main",
            ),
            "launch.artifacts.datasets.Bad.Id",
        ),
        (
            _valid_launch_yaml().replace(
                "    model: artifacts/model",
                "    model: true",
            ),
            "launch.artifacts.model",
        ),
        (
            _valid_launch_yaml().replace(
                "    datasets:\n      main: datasets/main",
                "    datasets: {}",
            ),
            "launch.artifacts.datasets",
        ),
    ),
)
def test_invalid_launch_unknown_type_and_path_inputs_fail_closed(
    tmp_path,
    launch_yaml: str,
    expected_key: str,
) -> None:
    source = _write_source(
        tmp_path,
        "schema_version: 2\nrecipe: flow_grpo_v1\n" + launch_yaml,
    )

    with pytest.raises(ConfigError) as caught:
        bootstrap_recipe_v2(source)
    assert caught.value.key == expected_key
    assert caught.value.path == str(source.path)


@pytest.mark.parametrize(
    "override",
    (
        {"training": {"unknown": 1}},
        {"training": {"seed": True}},
        {"training": {"gradient_accumulation_steps": 2}},
        {"training": {"adamw": {"beta1": 1.0}}},
        {"training": {"lr_schedule": {"warmup_steps": 1_001}}},
        {"training": {"update_safety": {"max_grad_norm": False}}},
        {"training": {"policy_recompute": {"row_microbatch_size": 0}}},
        {"training": {"policy_recompute": {"transition_window_size": True}}},
    ),
)
def test_invalid_training_unknown_and_type_inputs_fail_closed(override) -> None:
    definition = get_recipe_definition("flow_grpo_v1")

    with pytest.raises(ConfigError, match="invalid typed training override"):
        apply_recipe_overrides(definition, override)
