"""C5 config, trust-boundary, and checkpoint identity contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import visual_rl as vr
from visual_rl.artifacts.checkpoint import (
    build_implementation_identity,
    config_fingerprint,
)
from visual_rl.configs import read_experiment_spec, resolve_experiment
from visual_rl.configs.schema import config_to_dict
from visual_rl.feedback import FeedbackProvider, build_feedback_provider
from visual_rl.preflight import (
    StaticPreflightError,
    TrustedComponentError,
    static_preflight,
    trusted_component_load,
)


def external_score(batch, *, scale=1.0):
    return [scale] * batch.batch_size


def wrong_signature():
    return []


class CountingProvider(FeedbackProvider):
    constructions = 0

    def __init__(
        self,
        rewards_config,
        *,
        cache_dir=None,
        scale=1.0,
    ):
        type(self).constructions += 1
        self.rewards_config = rewards_config
        self.cache_dir = cache_dir
        self.scale = scale

    def score(self, batch):
        raise NotImplementedError


class _Adapter:
    def named_parameters(self):
        return []


class _Plugin:
    advantage_computer = None


class _MethodIdentity:
    def implementation_identity(self):
        return {"target": "example:score", "version": "method-v1", "params": {}}


class _InvalidMethodIdentity:
    def implementation_identity(self):
        return {"component": external_score}


def _experiment(tmp_path: Path, reward) -> vr.Experiment:
    return vr.Experiment(
        model=vr.models.MockWan(),
        rollout=vr.rollouts.FullTrajectory(),
        reward=reward,
        advantage=vr.advantages.GroupNormalize(),
        objective=vr.objectives.GRPO(),
        train=vr.Train(),
        output_dir=tmp_path / "run",
        show_progress=False,
    )


def test_external_object_resolves_to_auditable_json_only_config(tmp_path):
    descriptor = vr.rewards.External(
        external_score,
        version="score-v1",
        name="quality",
        params={"scale": 2.5},
        weight=0.25,
        dependencies=("json",),
    )

    values = _experiment(tmp_path, descriptor).to_config()
    metadata = values["rewards"]["provider_params"]

    assert values["rewards"]["provider"] == "external"
    assert values["rewards"]["weights"] == {"quality": 0.25}
    assert values["rewards"]["clients"] == {}
    assert metadata == {
        "target": f"{__name__}:external_score",
        "version": "score-v1",
        "source_sha256": descriptor.source_sha256,
        "params": {"scale": 2.5},
        "dependencies": ["json"],
        "reward_name": "quality",
    }
    assert len(metadata["source_sha256"]) == 64
    json.dumps(values, sort_keys=True, allow_nan=False)


def test_python_and_real_yaml_have_identical_config_and_fingerprint(tmp_path):
    object_descriptor = vr.rewards.External(
        external_score,
        version="score-v2",
        name="alignment",
        params={"scale": 3.0},
        weight=0.75,
    )
    yaml_path = tmp_path / "external.yaml"
    python_values = config_to_dict(_experiment(tmp_path, object_descriptor).resolve())
    yaml_path.write_text(
        f"""\
run_name: experiment
model:
  name: mock_wan
  model_family: wan
  latent_shape: [4, 2, 2, 2]
  media_shape: [4, 3, 16, 16]
sample:
  name: full_trajectory
  num_steps: 2
  batch_size: 1
  samples_per_prompt: 2
  guidance_scale: 4.5
  noise_level: 0.7
algorithm:
  name: grpo
  advantage_mode: grpo
  advantage_epsilon: 1.0e-6
  advantage_dtype: float32
  weight_advantages: false
  clip_range: 0.001
  beta: 0.0
  adv_clip_max: 5.0
rewards:
  provider: external
  provider_params:
    target: {object_descriptor.target}
    version: {object_descriptor.version}
    source_sha256: {object_descriptor.source_sha256}
    params:
      scale: 3.0
    dependencies: []
    reward_name: alignment
  replace_defaults: true
  weights:
    alignment: 0.75
  clients: {{}}
train:
  max_steps: 1
  learning_rate: 1.0e-4
  save_every: 1
  max_grad_norm: null
paths:
  output_dir: {str(tmp_path / "run")}
runner:
  show_progress: false
""",
        encoding="utf-8",
    )
    yaml_values = config_to_dict(vr.load_config(yaml_path))

    assert yaml_values == python_values
    assert config_fingerprint(yaml_values) == config_fingerprint(python_values)


def test_reward_replace_defaults_is_leaf_provenance_aware(tmp_path):
    yaml_path = tmp_path / "layered-rewards.yaml"
    yaml_path.write_text(
        """\
run_name: layered-rewards
preset:
  rewards:
    weights:
      preset_score: 0.25
    clients:
      preset_score:
        name: preset_score
        version: preset-v1
user:
  rewards:
    replace_defaults: true
    weights:
      user_score: 0.75
    clients:
      user_score:
        name: user_score
        version: user-v1
""",
        encoding="utf-8",
    )

    resolved = resolve_experiment(read_experiment_spec(yaml_path))

    assert resolved.config.rewards.weights == {
        "preset_score": 0.25,
        "user_score": 0.75,
    }
    assert resolved.config.rewards.clients == {
        "preset_score": {"name": "preset_score", "version": "preset-v1"},
        "user_score": {"name": "user_score", "version": "user-v1"},
    }
    assert "mock" not in resolved.config.rewards.weights
    assert "mock" not in resolved.config.rewards.clients
    assert resolved.provenance["rewards.weights.preset_score"].kind == "preset"
    assert resolved.provenance["rewards.weights.user_score"].kind == "user"
    assert resolved.provenance["rewards.clients.preset_score.name"].kind == "preset"
    assert resolved.provenance["rewards.clients.user_score.version"].kind == "user"
    assert "rewards.weights.mock" not in resolved.provenance
    assert not any(
        key.startswith("rewards.clients.mock") for key in resolved.provenance
    )


def test_reward_replace_defaults_preserves_explicit_empty_mappings(tmp_path):
    yaml_path = tmp_path / "empty-reward-mappings.yaml"
    yaml_path.write_text(
        """\
run_name: empty-reward-mappings
preset:
  rewards:
    clients:
      mock: {}
      preset_empty: {}
      preset_nested:
        params: {}
user:
  rewards:
    replace_defaults: true
    clients:
      user_empty: {}
      user_nested:
        params: {}
""",
        encoding="utf-8",
    )

    resolved = resolve_experiment(read_experiment_spec(yaml_path))

    assert resolved.config.rewards.weights == {}
    assert resolved.config.rewards.clients == {
        "mock": {},
        "preset_empty": {},
        "preset_nested": {"params": {}},
        "user_empty": {},
        "user_nested": {"params": {}},
    }
    expected_sources = {
        "rewards.clients.mock": "preset",
        "rewards.clients.preset_empty": "preset",
        "rewards.clients.preset_nested.params": "preset",
        "rewards.clients.user_empty": "user",
        "rewards.clients.user_nested.params": "user",
    }
    assert {
        path: resolved.provenance[path].kind for path in expected_sources
    } == expected_sources
    assert not any(
        source.kind == "schema"
        for path, source in resolved.provenance.items()
        if path.startswith(("rewards.weights", "rewards.clients"))
    )

    values = config_to_dict(resolved.config)
    for path in resolved.provenance:
        if not path.startswith(("rewards.weights", "rewards.clients")):
            continue
        current = values
        for segment in path.split("."):
            assert segment in current, f"stale provenance path: {path}"
            current = current[segment]


def test_reward_replace_defaults_false_keeps_schema_merge(tmp_path):
    yaml_path = tmp_path / "merged-rewards.yaml"
    yaml_path.write_text(
        """\
run_name: merged-rewards
preset:
  rewards:
    weights:
      preset_score: 0.25
    clients:
      mock: {}
      preset_score:
        name: preset_score
        version: preset-v1
user:
  rewards:
    replace_defaults: false
    weights:
      user_score: 0.75
    clients:
      mock: {}
      user_score:
        name: user_score
        version: user-v1
""",
        encoding="utf-8",
    )

    resolved = resolve_experiment(read_experiment_spec(yaml_path))

    assert resolved.config.rewards.weights == {
        "mock": 1.0,
        "preset_score": 0.25,
        "user_score": 0.75,
    }
    assert resolved.config.rewards.clients["mock"] == {"name": "mock"}
    assert resolved.provenance["rewards.weights.mock"].kind == "schema"
    assert "rewards.clients.mock" not in resolved.provenance
    assert resolved.provenance["rewards.clients.mock.name"].kind == "schema"


def test_external_descriptor_json_canonicalizes_tuple_params():
    descriptor = vr.rewards.External(
        external_score,
        version="tuple-v1",
        params={"shape": (2, 3), "nested": {"axes": (0, 1)}},
    )

    assert descriptor.params == {
        "shape": [2, 3],
        "nested": {"axes": [0, 1]},
    }
    assert descriptor.to_config()["rewards"]["provider_params"]["params"] == (
        descriptor.params
    )


def test_static_validation_does_not_import_or_construct_external_code(
    tmp_path, monkeypatch
):
    module_name = "c5_static_only_missing_module"
    sys.modules.pop(module_name, None)
    string_descriptor = vr.rewards.External(
        f"{module_name}:score",
        version="v1",
        source_sha256="0" * 64,
    )
    report = _experiment(tmp_path, string_descriptor).validate()
    external = next(item for item in report.components if item.kind == "provider")

    assert module_name not in sys.modules
    assert external.trust_boundary == "trusted_local_code"
    assert external.source_sha256 == "0" * 64

    CountingProvider.constructions = 0
    class_experiment = _experiment(
        tmp_path,
        vr.rewards.External(
            CountingProvider,
            version="provider-v1",
            name="direct-provider",
            params={"scale": 2.5},
        ),
    )
    class_report = class_experiment.validate()
    trusted_component_load(class_experiment.resolve(), class_report)
    assert CountingProvider.constructions == 0

    provider = build_feedback_provider(class_experiment.resolve().rewards)
    assert provider.__class__.__name__ == "CallableFeedbackProvider"
    assert isinstance(provider.component, CountingProvider)
    assert provider.component.scale == 2.5
    assert provider.params == {"scale": 2.5}
    assert CountingProvider.constructions == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target", "not-a-target", "invalid target"),
        ("version", "", "non-empty version"),
        ("source_sha256", "abc", "64-character source_sha256"),
        ("params", [], "JSON-safe mapping"),
        ("dependencies", "json", "invalid dependency declaration"),
    ],
)
def test_static_preflight_rejects_invalid_external_metadata(
    tmp_path, field, value, message
):
    config = _experiment(
        tmp_path,
        vr.rewards.External(external_score, version="v1"),
    ).resolve()
    config.rewards.provider_params[field] = value

    with pytest.raises(StaticPreflightError, match=message):
        static_preflight(config)


def test_external_provider_rejects_unknown_top_level_runtime_params(tmp_path):
    config = _experiment(
        tmp_path,
        vr.rewards.External(
            external_score,
            version="v1",
            params={"scale": 2.0},
        ),
    ).resolve()
    config.rewards.provider_params["scale"] = 3.0

    with pytest.raises(StaticPreflightError, match="Unknown external provider_params"):
        static_preflight(config)
    with pytest.raises(ValueError, match="Unknown external provider_params"):
        build_feedback_provider(config.rewards)


@pytest.mark.parametrize("legacy_weight", [0.5, 2.0])
def test_external_legacy_weight_must_match_canonical_weight(tmp_path, legacy_weight):
    config = _experiment(
        tmp_path,
        vr.rewards.External(
            external_score,
            version="v1",
            name="quality",
            weight=0.25,
        ),
    ).resolve()
    config.rewards.provider_params["weight"] = legacy_weight

    with pytest.raises(StaticPreflightError, match="does not match canonical"):
        static_preflight(config)
    with pytest.raises(ValueError, match="does not match canonical"):
        build_feedback_provider(config.rewards)


def test_matching_legacy_weight_is_accepted_but_not_used_as_source(tmp_path):
    config = _experiment(
        tmp_path,
        vr.rewards.External(
            external_score,
            version="legacy-compatible-v1",
            name="quality",
            weight=0.25,
        ),
    ).resolve()
    config.rewards.provider_params["weight"] = 0.25

    static_preflight(config)
    provider = build_feedback_provider(config.rewards)

    assert provider.weight == pytest.approx(config.rewards.weights["quality"])


def test_external_factory_reads_weight_only_from_rewards_weights(tmp_path):
    descriptor = vr.rewards.External(
        external_score,
        version="canonical-weight-v1",
        name="quality",
        params={"scale": 2.0},
        weight=0.375,
    )
    config = _experiment(tmp_path, descriptor).resolve()

    provider = build_feedback_provider(config.rewards)

    assert "weight" not in config.rewards.provider_params
    assert provider.name == "quality"
    assert provider.weight == pytest.approx(0.375)


def test_trusted_load_rejects_wrong_hash_and_function_signature(tmp_path):
    wrong_hash = _experiment(
        tmp_path,
        vr.rewards.External(
            f"{__name__}:external_score",
            version="v1",
            source_sha256="0" * 64,
        ),
    )
    wrong_hash_config = wrong_hash.resolve()
    with pytest.raises(TrustedComponentError, match="SHA256 mismatch"):
        trusted_component_load(wrong_hash_config, wrong_hash.validate())

    wrong_callable = _experiment(
        tmp_path,
        vr.rewards.External(wrong_signature, version="v1"),
    )
    wrong_callable_config = wrong_callable.resolve()
    with pytest.raises(TrustedComponentError, match=r"accept \(batch, \*\*params\)"):
        trusted_component_load(wrong_callable_config, wrong_callable.validate())


def test_external_descriptor_rejects_unauditable_or_non_json_metadata():
    with pytest.raises(ValueError, match="lambdas and local definitions"):
        vr.rewards.External(lambda batch: [], version="v1")
    with pytest.raises(ValueError, match="source_sha256 is required"):
        vr.rewards.External("example_rewards:score", version="v1")
    with pytest.raises(TypeError, match="JSON-safe"):
        vr.rewards.External(
            external_score,
            version="v1",
            params={"bad": object()},
        )
    with pytest.raises(TypeError, match="iterable of dependency names"):
        vr.rewards.External(
            external_score,
            version="v1",
            dependencies="numpy",
        )


def test_world_r1_and_external_configs_do_not_retain_mock_reward(tmp_path):
    world_r1 = _experiment(
        tmp_path,
        vr.rewards.WorldR1(
            general_url="http://127.0.0.1:18080/general",
            geometry_url="http://127.0.0.1:18080/geometry",
        ),
    ).resolve()
    external = _experiment(
        tmp_path,
        vr.rewards.External(external_score, version="v1", name="quality"),
    ).resolve()

    assert set(world_r1.rewards.weights) == {"reward_general", "reward_3d"}
    assert set(world_r1.rewards.clients) == {"reward_general", "reward_3d"}
    assert world_r1.rewards.provider == "reward_router"
    assert all(
        client["wire_format"] == "json_v1"
        and client["allow_unsafe_pickle"] is False
        for client in world_r1.rewards.clients.values()
    )
    assert external.rewards.weights == {"quality": 1.0}
    assert external.rewards.clients == {}
    assert "mock" not in config_to_dict(world_r1)["rewards"]


def test_world_r1_python_api_requires_explicit_legacy_pickle_policy():
    with pytest.raises(ValueError, match="allow_unsafe_pickle"):
        vr.rewards.WorldR1(
            general_url="http://localhost:8090",
            geometry_url="http://localhost:8089",
            wire_format="legacy_pickle",
        )

    reward = vr.rewards.WorldR1(
        general_url="http://localhost:8090",
        geometry_url="http://127.0.0.1:8089",
        wire_format="legacy_pickle",
        allow_unsafe_pickle=True,
        trusted_hosts=("localhost", "127.0.0.1"),
    )
    clients = reward.to_config()["rewards"]["clients"]
    assert clients["reward_general"]["wire_format"] == "legacy_pickle"
    assert clients["reward_3d"]["allow_unsafe_pickle"] is True
    assert clients["reward_3d"]["trusted_hosts"] == ["localhost", "127.0.0.1"]


def test_world_r1_python_api_supports_general_only_legacy_loopback(tmp_path):
    reward = vr.rewards.WorldR1(
        "http://127.0.0.1:8090/",
        wire_format="legacy_pickle",
        allow_unsafe_pickle=True,
        trusted_hosts=("127.0.0.1",),
        retries=0,
    )
    experiment = _experiment(tmp_path, reward)
    config = config_to_dict(experiment.resolve())["rewards"]

    assert config["weights"] == {"reward_general": 1.0}
    assert set(config["clients"]) == {"reward_general"}
    assert experiment.validate().trusted is False
    assert config["clients"]["reward_general"] == {
        "name": "reward_general",
        "version": "v1",
        "url": "http://127.0.0.1:8090/",
        "timeout": 1000.0,
        "retries": 0,
        "wire_format": "legacy_pickle",
        "allow_unsafe_pickle": True,
        "trusted_hosts": ["127.0.0.1"],
        "max_response_bytes": 16 * 1024 * 1024,
    }

    dual = vr.rewards.WorldR1(
        "http://127.0.0.1:8090/",
        "http://127.0.0.1:8089/",
    ).to_config()["rewards"]
    assert set(dual["weights"]) == {"reward_general", "reward_3d"}
    assert set(dual["clients"]) == {"reward_general", "reward_3d"}


def test_packaged_world_r1_preset_does_not_retain_schema_mock(tmp_path):
    envelope = tmp_path / "world-r1.yaml"
    envelope.write_text("preset: world_r1_wan_bounded\n", encoding="utf-8")

    config = vr.load_config(envelope)

    assert config.rewards.replace_defaults is True
    assert set(config.rewards.weights) == {"reward_general", "reward_3d"}
    assert set(config.rewards.clients) == {"reward_general", "reward_3d"}
    assert "mock" not in config.rewards.weights


def test_external_runtime_details_enter_checkpoint_implementation_identity(tmp_path):
    descriptor = vr.rewards.External(
        external_score,
        version="identity-v1",
        name="identity-score",
        params={"scale": 4.0},
    )
    config = _experiment(tmp_path, descriptor).resolve()
    provider = build_feedback_provider(config.rewards, cache_dir=None)
    identity = build_implementation_identity(
        _Adapter(),
        _Plugin(),
        feedback=provider,
    )["feedback"]

    assert identity["target"] == descriptor.target
    assert identity["version"] == "identity-v1"
    assert identity["source_sha256"] == descriptor.source_sha256
    assert identity["params"]["scale"] == 4.0
    assert identity["class"].endswith("CallableFeedbackProvider")
    assert len(identity["module_sha256"]) == 64


def test_checkpoint_identity_accepts_only_json_safe_method_details():
    identity = build_implementation_identity(
        _Adapter(),
        _Plugin(),
        feedback=_MethodIdentity(),
    )["feedback"]
    assert identity["target"] == "example:score"
    assert identity["version"] == "method-v1"

    with pytest.raises(TypeError, match="JSON-safe"):
        build_implementation_identity(
            _Adapter(),
            _Plugin(),
            feedback=_InvalidMethodIdentity(),
        )
