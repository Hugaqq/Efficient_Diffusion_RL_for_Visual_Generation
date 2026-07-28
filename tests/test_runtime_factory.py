"""Transactional ownership tests for the one runtime construction site."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from visual_rl.runtime_factory import build_runtime_components


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(prompts=("one", "two")),
        model=SimpleNamespace(name="model", params={"model": 1}),
        rollout=SimpleNamespace(name="rollout", params={"rollout": 2}),
        reward=SimpleNamespace(
            components=(
                SimpleNamespace(name="reward_a", weight=1.0, params={"a": 1}),
                SimpleNamespace(name="reward_b", weight=2.0, params={"b": 2}),
            ),
            cache_dir=tmp_path / "reward-cache",
            execution=SimpleNamespace(microbatch_size=3, max_retries=1),
        ),
        algorithm=SimpleNamespace(
            name="algorithm",
            params={"algorithm": 3},
            advantage=SimpleNamespace(epsilon=1e-5),
        ),
        optimizer=SimpleNamespace(
            max_grad_norm=1.0,
            max_initial_logprob_delta=0.5,
            require_initial_clipfrac_zero=True,
            require_finite_gradients=True,
            require_nonzero_gradients=False,
        ),
        runtime=SimpleNamespace(update_microbatch_size=4, precision="fp32"),
    )


def _install_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: str | None,
    constructed: list[str],
    closed: list[str],
) -> None:
    def resource(name: str):
        if fail_at == name:
            raise RuntimeError(f"failed at {name}")

        class Resource:
            def __init__(self) -> None:
                self.name = name
                self.close_count = 0
                constructed.append(name)

            def close(self) -> None:
                self.close_count += 1
                if self.close_count > 1:
                    raise AssertionError(f"{name} closed more than once")
                closed.append(name)

        return Resource()

    class Dataset:
        @classmethod
        def from_config(cls, config):
            assert config.prompts == ("one", "two")
            return resource("dataset")

    def component_factory(name: str, expected: dict[str, int]):
        class Factory:
            @classmethod
            def from_config(cls, params, context):
                assert params == expected
                assert context.rank == 1
                return resource(name)

        return Factory

    model_factory = component_factory("model", {"model": 1})
    rollout_factory = component_factory("rollout", {"rollout": 2})
    reward_a_factory = component_factory("reward_a", {"a": 1})
    reward_b_factory = component_factory("reward_b", {"b": 2})
    algorithm_factory = component_factory("algorithm", {"algorithm": 3})
    algorithm_factory.ADVANTAGE_DTYPE = "float64"

    factories = {
        ("model", "model"): model_factory,
        ("rollout", "rollout"): rollout_factory,
        ("reward", "reward_a"): reward_a_factory,
        ("reward", "reward_b"): reward_b_factory,
        ("algorithm", "algorithm"): algorithm_factory,
    }

    def get_builtin_component(kind, name):
        return SimpleNamespace(factory=factories[(kind, name)])

    class Binding:
        def __init__(self, *, name, client, weight, resolved_params):
            self.name = name
            self.client = client
            self.weight = weight
            self.resolved_params = resolved_params

    class Cache:
        def __new__(cls, path):
            assert path.name == "rank_1"
            return resource("cache")

    class Provider:
        def __new__(cls, *, clients, cache):
            assert [item.name for item in clients] == ["reward_a", "reward_b"]
            assert [item.weight for item in clients] == [1.0, 2.0]
            assert cache.name == "cache"
            return resource("provider")

    class Executor:
        def __new__(cls, *, provider, microbatch_size, max_retries):
            assert provider.name == "provider"
            assert (microbatch_size, max_retries) == (3, 1)
            return resource("executor")

    class Advantage:
        def __init__(self, *, epsilon, output_dtype):
            assert epsilon == 1e-5
            assert output_dtype == "float64"

    class Plugin:
        def __new__(cls, **kwargs):
            assert kwargs["algorithm"].name == "algorithm"
            assert kwargs["update_microbatch_size"] == 4
            assert kwargs["precision"] == "fp32"
            assert kwargs["max_grad_norm"] == 1.0
            assert kwargs["max_initial_logprob_delta"] == 0.5
            assert kwargs["require_initial_clipfrac_zero"] is True
            assert kwargs["require_finite_gradients"] is True
            assert kwargs["require_nonzero_gradients"] is False
            return resource("plugin")

    modules = {
        "visual_rl.builtins": {"get_builtin_component": get_builtin_component},
        "visual_rl.datasets.prompt_dataset": {"PromptDataset": Dataset},
        "visual_rl.feedback.cache": {"RewardCache": Cache},
        "visual_rl.feedback.executor": {"RewardExecutor": Executor},
        "visual_rl.feedback.provider": {
            "RewardClientBinding": Binding,
            "RewardFeedbackProvider": Provider,
        },
        "visual_rl.optimizers.advantages": {"AdvantageComputer": Advantage},
        "visual_rl.optimizers.algorithm_plugin": {
            "AlgorithmOptimizerPlugin": Plugin
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        rank=1,
        local_rank=1,
        world_size=2,
        backend="gloo",
        device="cpu",
        precision="fp32",
    )


def test_successful_bundle_closes_exact_graph_in_reverse_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[str] = []
    closed: list[str] = []
    _install_runtime_fakes(
        monkeypatch,
        fail_at=None,
        constructed=constructed,
        closed=closed,
    )

    bundle = build_runtime_components(_config(tmp_path), _context())
    assert constructed == [
        "dataset",
        "model",
        "rollout",
        "reward_a",
        "reward_b",
        "cache",
        "provider",
        "executor",
        "algorithm",
        "plugin",
    ]

    bundle.close()
    bundle.close()

    assert closed == [
        "plugin",
        "algorithm",
        "executor",
        "provider",
        "cache",
        "reward_b",
        "reward_a",
        "rollout",
        "model",
        "dataset",
    ]


@pytest.mark.parametrize(
    ("fail_at", "expected_closed"),
    (
        ("dataset", ()),
        ("model", ("dataset",)),
        ("rollout", ("model", "dataset")),
        ("reward_a", ("rollout", "model", "dataset")),
        ("reward_b", ("reward_a", "rollout", "model", "dataset")),
        (
            "cache",
            ("reward_b", "reward_a", "rollout", "model", "dataset"),
        ),
        (
            "provider",
            ("cache", "reward_b", "reward_a", "rollout", "model", "dataset"),
        ),
        (
            "executor",
            (
                "provider",
                "cache",
                "reward_b",
                "reward_a",
                "rollout",
                "model",
                "dataset",
            ),
        ),
        (
            "algorithm",
            (
                "executor",
                "provider",
                "cache",
                "reward_b",
                "reward_a",
                "rollout",
                "model",
                "dataset",
            ),
        ),
        (
            "plugin",
            (
                "algorithm",
                "executor",
                "provider",
                "cache",
                "reward_b",
                "reward_a",
                "rollout",
                "model",
                "dataset",
            ),
        ),
    ),
)
def test_partial_construction_closes_only_returned_prefix_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
    expected_closed: tuple[str, ...],
) -> None:
    constructed: list[str] = []
    closed: list[str] = []
    _install_runtime_fakes(
        monkeypatch,
        fail_at=fail_at,
        constructed=constructed,
        closed=closed,
    )

    with pytest.raises(RuntimeError, match=f"failed at {fail_at}"):
        build_runtime_components(_config(tmp_path), _context())

    assert tuple(closed) == expected_closed
