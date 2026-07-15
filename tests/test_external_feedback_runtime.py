from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import textwrap
import uuid

import numpy as np
import pytest
import torch

from visual_rl.configs.schema import RewardConfig
from visual_rl.core.registry import FEEDBACK_PROVIDERS
from visual_rl.core.types import RolloutBatch, RewardBatch, StepContext
from visual_rl.feedback import (
    CallableFeedbackProvider,
    FeedbackProvider,
    build_feedback_provider,
)
from visual_rl.feedback.cache import RewardCache, stable_hash_json
from visual_rl.feedback.external import (
    _reward_batch_from_payload,
    _reward_batch_to_payload,
)


def _batch() -> RolloutBatch:
    return RolloutBatch(
        prompts=["red square", "blue circle"],
        metadata=[{"index": 0}, {"index": 1}],
        media=torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
        sample_id=["sample-a", "sample-b"],
    )


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0],
        np.asarray([1.0, 2.0], dtype=np.float32),
        torch.tensor([1.0, 2.0]),
    ],
)
def test_callable_vector_outputs_are_wrapped_in_input_order(values):
    def score_fn(batch, *, offset):
        assert batch.sample_id == ["sample-a", "sample-b"]
        return values, {"external_note": "kept"}

    provider = CallableFeedbackProvider(
        score_fn,
        name="external_score",
        version="v1",
        params={"offset": 3},
        weight=0.5,
    )

    rewards = provider.score(_batch())

    assert rewards.sample_id == ["sample-a", "sample-b"]
    assert rewards.raw["external_score"].tolist() == pytest.approx([1.0, 2.0])
    assert rewards.weighted_total.tolist() == pytest.approx([0.5, 1.0])
    assert rewards.valid_mask.dtype == torch.bool
    assert rewards.metadata["external_note"] == "kept"
    assert rewards.metadata["trusted_input_order_callable"] is True
    assert (
        rewards.metadata["sample_id_provenance"]
        == "trusted_input_order_callable"
    )


def test_callable_object_and_reward_batch_output_are_supported():
    class ScoreObject:
        def __call__(self, batch, **params):
            assert params == {"scale": 2}
            return torch.tensor([0.25, 0.75])

    callable_rewards = CallableFeedbackProvider(
        ScoreObject(),
        name="object_score",
        version="v1",
        params={"scale": 2},
    ).score(_batch())
    assert callable_rewards.weighted_total.tolist() == pytest.approx([0.25, 0.75])

    expected = RewardBatch(
        raw={"custom": torch.tensor([2.0, 4.0])},
        weighted={"custom": torch.tensor([200.0, 400.0])},
        weighted_total=torch.tensor([200.0, 400.0]),
        valid_mask=[True, False],
        metadata={"owned_by": "external"},
        sample_id=["sample-a", "sample-b"],
    )
    provider = CallableFeedbackProvider(
        lambda batch: expected,
        name="reward_batch",
        version="v1",
        weight=0.25,
    )

    rewards = provider.score(_batch())
    assert rewards is not expected
    assert set(rewards.raw) == {"reward_batch"}
    assert rewards.raw["reward_batch"].tolist() == [2.0, 4.0]
    assert rewards.weighted["reward_batch"].tolist() == [0.5, 1.0]
    assert rewards.weighted_total.tolist() == [0.5, 1.0]
    assert rewards.valid_mask.tolist() == [True, False]
    assert rewards.metadata["owned_by"] == "external"
    assert "rewards.weights" in rewards.metadata["weight_source"]
    assert rewards.metadata["configured_weight"] == 0.25


def test_callable_rejects_multi_reward_batch():
    result = RewardBatch(
        raw={"a": [1.0, 2.0], "b": [3.0, 4.0]},
        weighted={},
        weighted_total=[0.0, 0.0],
        valid_mask=[True, True],
        sample_id=["sample-a", "sample-b"],
    )
    provider = CallableFeedbackProvider(
        lambda batch: result,
        name="configured",
        version="v1",
    )

    with pytest.raises(ValueError, match="exactly one raw reward"):
        provider.score(_batch())


def test_feedback_provider_component_delegates_without_runtime_params():
    class Provider(FeedbackProvider):
        def __init__(self):
            self.calls = 0

        def score(self, batch):
            self.calls += 1
            return RewardBatch(
                raw={"delegate": [1.0, 3.0]},
                weighted={"ignored": [99.0, 99.0]},
                weighted_total=[99.0, 99.0],
                valid_mask=[True, True],
                sample_id=list(batch.sample_id),
            )

    component = Provider()
    provider = CallableFeedbackProvider(
        component,
        name="configured",
        version="v1",
        params={"constructor_only": 7},
        weight=0.5,
    )

    rewards = provider.score(_batch())

    assert component.calls == 1
    assert provider.params == {"constructor_only": 7}
    assert rewards.weighted_total.tolist() == [0.5, 1.5]


@pytest.mark.parametrize(
    ("score_fn", "match"),
    [
        (lambda batch: [1.0], "length 2"),
        (lambda batch: [[1.0], [2.0]], "1D vector"),
        (lambda batch: [1.0, float("nan")], "finite"),
        (
            lambda batch: RewardBatch(
                raw={"x": torch.tensor([1.0, 2.0])},
                weighted={"x": torch.tensor([1.0, 2.0])},
                weighted_total=torch.tensor([1.0, 2.0]),
                valid_mask=torch.tensor([True, True]),
                sample_id=["sample-b", "sample-a"],
            ),
            "sample_id order",
        ),
        (
            lambda batch: RewardBatch(
                raw={"x": torch.tensor([1.0, 2.0])},
                weighted={"x": torch.tensor([1.0, 2.0])},
                weighted_total=torch.tensor([1.0, 2.0]),
                valid_mask=torch.tensor([1.0, 1.0]),
                sample_id=["sample-a", "sample-b"],
            ),
            "bool dtype",
        ),
    ],
)
def test_invalid_external_outputs_fail_before_use(score_fn, match):
    provider = CallableFeedbackProvider(
        score_fn,
        name="invalid",
        version="v1",
    )
    with pytest.raises((TypeError, ValueError), match=match):
        provider.score(_batch())


class _CountingCallable:
    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        return [0.125, 0.875], {"call": self.calls}


def test_external_cache_round_trip_preserves_data_and_identity(tmp_path):
    component = _CountingCallable()
    cache_dir = tmp_path / "cache"
    provider = CallableFeedbackProvider(
        component,
        name="cached",
        version="2026.1",
        params={},
        cache_dir=cache_dir,
        target="example_rewards:score",
        source_sha256="a" * 64,
    )

    first = provider.score(_batch())
    second = provider.score(_batch())

    assert component.calls == 1
    assert second.sample_id == first.sample_id
    assert second.weighted_total.tolist() == first.weighted_total.tolist()
    assert second.metadata == first.metadata
    cache_file = next(cache_dir.glob("*.json"))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["provider_identity"] == provider._visual_rl_identity
    assert payload["input_identity"]["ordered_sample_id"] == [
        "sample-a",
        "sample-b",
    ]
    assert payload["input_identity"]["media_sha256"]
    assert set(payload["reward_batch"]) == {
        "raw",
        "weighted",
        "weighted_total",
        "valid_mask",
        "metadata",
        "sample_id",
    }

    payload["provider_identity"]["version"] = "tampered"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="failed validation"):
        provider.score(_batch())
    assert component.calls == 1


def test_external_cache_miss_and_hit_are_cpu_detached_and_support_bfloat16(
    tmp_path,
):
    source = torch.tensor(
        [0.25, 0.75], dtype=torch.bfloat16, requires_grad=True
    )
    provider = CallableFeedbackProvider(
        lambda batch: source,
        name="bfloat",
        version="v1",
        cache_dir=tmp_path,
    )

    miss = provider.score(_batch())
    hit = provider.score(_batch())

    for rewards in (miss, hit):
        assert rewards.raw["bfloat"].dtype == torch.float32
        assert rewards.raw["bfloat"].device.type == "cpu"
        assert rewards.raw["bfloat"].requires_grad is False
        assert rewards.weighted_total.device.type == "cpu"
        assert rewards.weighted_total.requires_grad is False
        assert rewards.valid_mask.device.type == "cpu"
        assert rewards.valid_mask.dtype == torch.bool
    assert torch.equal(miss.weighted_total, hit.weighted_total)


def test_external_cache_identity_covers_complete_rollout_batch(tmp_path):
    component = _CountingCallable()
    provider = CallableFeedbackProvider(
        component,
        name="complete-input",
        version="v1",
        cache_dir=tmp_path,
    )
    base = RolloutBatch(
        prompts=["p0", "p1"],
        metadata=[{"index": 0}, {"index": 1}],
        media=torch.zeros(2, 3, 2, 2),
        latents=torch.zeros(2, 1, 1),
        next_latents=torch.ones(2, 1, 1),
        timesteps=torch.zeros(2, 1),
        old_log_probs=torch.zeros(2, 1),
        kl=torch.zeros(2, 1),
        sample_id=["s0", "s1"],
        prompt_id=["p-id-0", "p-id-1"],
        group_id=["g0", "g1"],
        branch_id=[0, 1],
        transition_mask=torch.ones(2, 1, dtype=torch.bool),
        context=StepContext(step=0, seed=1, epoch_tag=0),
        model_metadata={"scheduler": "base"},
        model_tensors={"conditioning": torch.zeros(2, 1)},
    )
    variants = [
        base.replace(sample_id=["s2", "s3"]),
        base.replace(prompt_id=["p-id-2", "p-id-3"]),
        base.replace(group_id=["g2", "g3"]),
        base.replace(branch_id=[2, 3]),
        base.replace(context=StepContext(step=1, seed=1, epoch_tag=0)),
        base.replace(prompts=["changed", "p1"]),
        base.replace(metadata=[{"index": 9}, {"index": 1}]),
        base.replace(media=torch.ones_like(base.media)),
        base.replace(latents=torch.ones_like(base.latents)),
        base.replace(next_latents=torch.zeros_like(base.next_latents)),
        base.replace(timesteps=torch.ones_like(base.timesteps)),
        base.replace(old_log_probs=torch.ones_like(base.old_log_probs)),
        base.replace(kl=torch.ones_like(base.kl)),
        base.replace(
            transition_mask=torch.tensor([[False], [True]])
        ),
        base.replace(model_metadata={"scheduler": "changed"}),
        base.replace(model_tensors={"conditioning": torch.ones(2, 1)}),
        base.replace(
            media=torch.zeros(2, 1, 3, 2, 2),
            media_layout="BFCHW",
        ),
    ]

    provider.score(base)
    provider.score(base)
    for variant in variants:
        provider.score(variant)

    assert component.calls == 1 + len(variants)
    assert len(list(tmp_path.glob("*.json"))) == 1 + len(variants)


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("media", (2, 3, 2, 2)),
        ("latents", (2, 1, 1)),
        ("old_log_probs", (2, 1)),
        ("model_tensors", (2, 1)),
    ],
)
@pytest.mark.parametrize(
    "make_value",
    [
        lambda shape: torch.full(shape, float("nan")),
        lambda shape: np.full(shape, float("inf"), dtype=np.float32),
        lambda shape: torch.full(shape, complex(float("nan"), 0), dtype=torch.complex64),
        lambda shape: np.full(shape, complex(0, float("inf")), dtype=np.complex64),
    ],
    ids=["torch-float", "numpy-float", "torch-complex", "numpy-complex"],
)
def test_external_cache_rejects_nonfinite_identity_before_provider_or_write(
    tmp_path, field, shape, make_value
):
    component = _CountingCallable()
    provider = CallableFeedbackProvider(
        component,
        name="nonfinite-input",
        version="v1",
        cache_dir=tmp_path,
    )
    value = make_value(shape)
    batch = (
        _batch().replace(model_tensors={"nested": {"value": value}})
        if field == "model_tensors"
        else _batch().replace(**{field: value})
    )

    with pytest.raises(ValueError, match="Cache hashes reject NaN and infinity"):
        provider.score(batch)

    assert component.calls == 0
    assert not list(tmp_path.glob("*.json"))


def test_external_cache_payload_mappings_are_deterministic_and_round_trip():
    def reward_batch(reverse: bool) -> RewardBatch:
        pairs = [
            ("z_score", torch.tensor([3.0, 4.0])),
            ("a_score", torch.tensor([1.0, 2.0])),
        ]
        metadata_pairs = [
            (("tuple", 1), {2: "integer", "2": "string"}),
            (5, "integer key"),
            ("5", "string key"),
        ]
        if reverse:
            pairs.reverse()
            metadata_pairs.reverse()
        raw = dict(pairs)
        weighted = {name: values * 0.5 for name, values in pairs}
        return RewardBatch(
            raw=raw,
            weighted=weighted,
            weighted_total=torch.tensor([2.0, 3.0]),
            valid_mask=torch.tensor([True, True]),
            metadata=dict(metadata_pairs),
            sample_id=["sample-a", "sample-b"],
        )

    first_payload = _reward_batch_to_payload(reward_batch(reverse=False))
    second_payload = _reward_batch_to_payload(reward_batch(reverse=True))
    first_bytes = json.dumps(
        first_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    second_bytes = json.dumps(
        second_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert first_bytes == second_bytes
    assert stable_hash_json(first_payload) == stable_hash_json(second_payload)
    restored = _reward_batch_from_payload(first_payload)
    assert list(restored.raw) == ["a_score", "z_score"]
    assert list(restored.weighted) == ["a_score", "z_score"]
    assert restored.metadata[("tuple", 1)] == {2: "integer", "2": "string"}
    assert restored.metadata[5] == "integer key"
    assert restored.metadata["5"] == "string key"


def test_external_cache_rejects_bad_reward_payload(tmp_path):
    provider = CallableFeedbackProvider(
        _CountingCallable(),
        name="cached",
        version="v1",
        cache_dir=tmp_path,
        target="example_rewards:score",
    )
    provider.score(_batch())
    cache_file = next(tmp_path.glob("*.json"))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["reward_batch"] = {}
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="failed validation"):
        provider.score(_batch())


def test_external_cache_rejects_non_finite_json_tamper(tmp_path):
    provider = CallableFeedbackProvider(
        _CountingCallable(),
        name="cached",
        version="v1",
        cache_dir=tmp_path,
    )
    provider.score(_batch())
    cache_file = next(tmp_path.glob("*.json"))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["reward_batch"]["weighted_total"]["data"][0] = float("nan")
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        provider.score(_batch())


def test_reward_cache_rejects_nonfinite_writes_and_duplicate_json_keys(tmp_path):
    cache = RewardCache(tmp_path)
    with pytest.raises(ValueError, match="Non-finite"):
        cache.set("nan", {"value": float("nan")})

    (tmp_path / "duplicate.json").write_text(
        '{"value": 1, "value": 2}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        cache.get("duplicate")
    assert not (tmp_path / "duplicate.json").exists()
    assert len(list(tmp_path.glob("duplicate.corrupt-*.json"))) == 1


def _write_external_module(
    tmp_path: Path, monkeypatch, source: str
) -> tuple[str, str]:
    module_name = f"visual_rl_test_external_{uuid.uuid4().hex}"
    source_path = tmp_path / f"{module_name}.py"
    source_path.write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return module_name, digest


def test_factory_loads_external_function_without_registry(tmp_path, monkeypatch):
    module_name, digest = _write_external_module(
        tmp_path,
        monkeypatch,
        """
        def score(batch, offset=0.0):
            return [offset + index for index in range(batch.batch_size)]
        """,
    )
    provider_name = f"external-{uuid.uuid4().hex}"
    with pytest.raises(KeyError):
        FEEDBACK_PROVIDERS.get(provider_name)

    provider = build_feedback_provider(
        RewardConfig(
            provider=provider_name,
            weights={"quality": 0.5},
            provider_params={
                "target": f"{module_name}:score",
                "version": "v2",
                "source_sha256": digest,
                "params": {"offset": 2.0},
                "reward_name": "quality",
                "controls": {"trusted": True},
            },
        ),
        cache_dir=tmp_path / "cache",
    )

    assert isinstance(provider, CallableFeedbackProvider)
    assert provider.score(_batch()).weighted_total.tolist() == pytest.approx(
        [1.0, 1.5]
    )
    assert provider.name == "quality"
    assert provider.params == {"offset": 2.0}
    assert "controls" not in provider._visual_rl_identity["params"]
    with pytest.raises(KeyError):
        FEEDBACK_PROVIDERS.get(provider_name)


def test_factory_rechecks_external_source_hash(tmp_path, monkeypatch):
    module_name, _ = _write_external_module(
        tmp_path,
        monkeypatch,
        """
        def score(batch):
            return [0.0] * batch.batch_size
        """,
    )
    provider_name = f"external-{uuid.uuid4().hex}"
    config = RewardConfig(
        provider=provider_name,
        weights={provider_name: 1.0},
        provider_params={
            "target": f"{module_name}:score",
            "version": "v1",
            "source_sha256": "0" * 64,
        },
    )

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        build_feedback_provider(config)


def test_factory_instantiates_external_provider_class(tmp_path, monkeypatch):
    module_name, digest = _write_external_module(
        tmp_path,
        monkeypatch,
        """
        import torch
        from visual_rl.core.types import RewardBatch
        from visual_rl.feedback.base import FeedbackProvider

        class Provider(FeedbackProvider):
            def __init__(self, rewards_config, cache_dir=None, scale=1.0):
                self.rewards_config = rewards_config
                self.cache_dir = cache_dir
                self.scale = scale

            def score(self, batch):
                values = torch.arange(batch.batch_size, dtype=torch.float32)
                values = values * self.scale
                return RewardBatch(
                    raw={"class": values},
                    weighted={"class": values},
                    weighted_total=values,
                    valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
                    sample_id=list(batch.sample_id),
                )
        """,
    )
    provider_name = f"class-{uuid.uuid4().hex}"
    provider = build_feedback_provider(
        RewardConfig(
            provider=provider_name,
            weights={"quality": 0.25},
            provider_params={
                "target": f"{module_name}:Provider",
                "version": "class-v1",
                "source_sha256": digest,
                "params": {"scale": 3.0},
                "reward_name": "quality",
                "controls": {"ignored": True},
            },
        ),
        cache_dir=tmp_path / "class-cache",
    )

    assert isinstance(provider, CallableFeedbackProvider)
    assert isinstance(provider.component, FeedbackProvider)
    assert provider.component.scale == 3.0
    rewards = provider.score(_batch())
    assert rewards.raw["quality"].tolist() == [0.0, 3.0]
    assert rewards.weighted_total.tolist() == [0.0, 0.75]
    assert provider._visual_rl_identity == {
        "name": "quality",
        "target": f"{module_name}:Provider",
        "version": "class-v1",
        "source_sha256": digest,
        "params": {"scale": 3.0},
        "weight": 0.25,
    }


@pytest.mark.parametrize(
    ("source", "attribute", "match"),
    [
        ("score = lambda batch: [0.0] * batch.batch_size", "score", "Lambda"),
        (
            """
            def make_score():
                def score(batch):
                    return [0.0] * batch.batch_size
                return score
            score = make_score()
            """,
            "score",
            "module level",
        ),
        ("score = 3", "score", "not callable"),
    ],
)
def test_factory_rejects_unstable_external_targets(
    tmp_path, monkeypatch, source, attribute, match
):
    module_name, digest = _write_external_module(tmp_path, monkeypatch, source)
    provider_name = f"rejected-{uuid.uuid4().hex}"
    config = RewardConfig(
        provider=provider_name,
        weights={provider_name: 1.0},
        provider_params={
            "target": f"{module_name}:{attribute}",
            "version": "v1",
            "source_sha256": digest,
        },
    )

    with pytest.raises(TypeError, match=match):
        build_feedback_provider(config)


@pytest.mark.parametrize("source_sha256", [None, "not-a-sha256"])
def test_factory_rejects_untrusted_source_hash(
    tmp_path, monkeypatch, source_sha256
):
    module_name, _ = _write_external_module(
        tmp_path,
        monkeypatch,
        """
        def score(batch):
            return [0.0] * batch.batch_size
        """,
    )
    provider_name = f"untrusted-{uuid.uuid4().hex}"
    config = RewardConfig(
        provider=provider_name,
        weights={provider_name: 1.0},
        provider_params={
            "target": f"{module_name}:score",
            "version": "v1",
            "source_sha256": source_sha256,
        },
    )

    with pytest.raises(ValueError, match="source_sha256"):
        build_feedback_provider(config)
