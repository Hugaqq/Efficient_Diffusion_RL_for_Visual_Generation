"""Production YAML launch-audit and canonical compiler contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.world_r1_strict.service_revision import BUNDLED_SERVICE_REVISION
from visual_rl.composition.config import bootstrap_recipe_v2, load_source_recipe
from visual_rl.composition.config.compiler import compile_recipe_v2
from visual_rl.errors import ConfigError

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "v2"
_SD3_UPSTREAM_LORA_TARGETS = (
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
)
_EXPECTED = {
    "flow_grpo_sd3.yaml": ("flow_grpo_v1", ("main",), (), True, True),
    "flow_grpo_wan.yaml": (
        "flow_grpo_v1",
        ("main",),
        ("reward_general",),
        False,
        False,
    ),
    "tempflow_sd3.yaml": ("tempflow_grpo_v1", ("main",), (), False, True),
    "flash_wan.yaml": (
        "flash_grpo_v1",
        ("main",),
        ("reward_general",),
        False,
        False,
    ),
    "world_r1_core_wan.yaml": (
        "world_r1_core_v1",
        ("main",),
        ("reward_3d", "reward_general"),
        False,
        False,
    ),
    "world_r1_release_surrogate_wan.yaml": (
        "world_r1_release_surrogate_v1",
        ("dynamic", "main"),
        ("reward_3d", "reward_general"),
        False,
        False,
    ),
}
_EXPECTED_MAX_STEPS = {
    "flow_grpo_sd3.yaml": 20,
    "flow_grpo_wan.yaml": 20,
    "tempflow_sd3.yaml": 20,
    "flash_wan.yaml": 20,
    "world_r1_core_wan.yaml": 20,
    "world_r1_release_surrogate_wan.yaml": 150,
}


@pytest.mark.parametrize("filename", tuple(_EXPECTED))
def test_official_v2_config_has_typed_recipe_and_separate_launch_audit(
    filename: str,
) -> None:
    (
        definition_id,
        dataset_refs,
        remote_reward_refs,
        expects_reference_state,
        model_provides_reference,
    ) = _EXPECTED[filename]
    source = load_source_recipe(_CONFIG_ROOT / filename)
    bootstrap = bootstrap_recipe_v2(source)
    launch = bootstrap.require_launch()
    resolved = compile_recipe_v2(source)
    model = resolved.model.declaration

    assert resolved.definition_id == definition_id
    assert resolved.compatibility.status == "compatible"
    assert (
        resolved.algorithm_spec.requires_reference_statistics is expects_reference_state
    )
    assert (
        model.declared_contract.model.provides_reference_policy
        is model_provides_reference
    )
    assert tuple(key for key, _path in launch.artifacts.datasets) == dataset_refs
    assert tuple(key for key, _spec in launch.reward_runtime_bindings) == (
        remote_reward_refs
    )
    assert tuple(
        sorted(resource.artifact_ref for resource in resolved.reward_plan.resources)
    ) == tuple(sorted(key for key, _path in launch.artifacts.rewards))

    for reward in resolved.reward_components:
        resource = reward.declaration.config.resource
        if resource.factory_class not in {"reward_general", "reward_3d"}:
            continue
        assert (
            resource.semantic_factory_config["server_revision_expectation"]
            == BUNDLED_SERVICE_REVISION
        )
        revision_suffix = BUNDLED_SERVICE_REVISION.removeprefix("world-r1-")
        prefix = (
            "world-r1-general-"
            if resource.factory_class == "reward_general"
            else "world-r1-3d-"
        )
        assert launch.artifacts.reward(resource.artifact_ref).name == (
            prefix + revision_suffix
        )

    assert resolved.training.adamw.weight_decay == pytest.approx(1.0e-4)
    assert launch.checkpoint_every_optimizer_steps <= (
        resolved.training.max_optimizer_steps
    )
    if model.alias == "sd3":
        assert tuple(model.config.lora_target_modules) == _SD3_UPSTREAM_LORA_TARGETS
    else:
        assert model.config.max_sequence_length == 512

    # Source and launch locations are separate by construction.
    payload = str(resolved.canonical_semantic_payload())
    assert str(source.path) not in payload
    assert str(launch.output_dir) not in payload


def test_v2_demo_directory_contains_only_the_declared_slices() -> None:
    assert tuple(path.name for path in sorted(_CONFIG_ROOT.glob("*.yaml"))) == tuple(
        sorted(_EXPECTED)
    )


def test_official_configs_lock_the_real_gpu_acceptance_envelope() -> None:
    for filename, expected_steps in _EXPECTED_MAX_STEPS.items():
        resolved = compile_recipe_v2(load_source_recipe(_CONFIG_ROOT / filename))
        recompute = resolved.training.policy_recompute

        assert resolved.training.max_optimizer_steps == expected_steps
        assert resolved.training.max_optimizer_steps >= 20
        assert recompute.transition_window_size == 1
        if resolved.model.declaration.alias == "wan-t2v":
            assert recompute.row_microbatch_size == 1


def test_ddp_request_is_rejected_by_compiler_before_runtime(tmp_path: Path) -> None:
    source_text = (_CONFIG_ROOT / "tempflow_sd3.yaml").read_text(encoding="utf-8")
    ddp_text = source_text.replace(
        "distribution_mode: single",
        "distribution_mode: ddp",
    )
    path = tmp_path / "tempflow_sd3_ddp.yaml"
    path.write_text(ddp_text, encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match="execution distribution mode is not accepted by the algorithm",
    ):
        compile_recipe_v2(load_source_recipe(path))


def test_internal_slot_override_is_rejected_at_config_boundary(tmp_path: Path) -> None:
    path = tmp_path / "internal_override.yaml"
    path.write_text(
        """\
schema_version: 2
recipe: flow_grpo_v1
overrides:
  components:
    rollout:
      id: single-step
      params: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unsupported override roots"):
        compile_recipe_v2(load_source_recipe(path))
