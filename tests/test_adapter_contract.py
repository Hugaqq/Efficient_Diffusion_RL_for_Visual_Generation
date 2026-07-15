from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from visual_rl.model_adapters.diffusers_common import AdapterNotLoadedError
from visual_rl.model_adapters.mock import MockWanAdapter
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter


def _light_adapters():
    return [
        (MockWanAdapter({}), "policy_bias", "mock_adapter.pt"),
        (
            TinyDiffusionAdapter({"device": "cpu"}),
            "color_bias",
            "tiny_diffusion.pt",
        ),
    ]


@pytest.mark.parametrize(
    ("adapter", "parameter_name", "_checkpoint_name"),
    _light_adapters(),
)
def test_light_adapters_expose_module_and_stable_trainable_parameters(
    adapter,
    parameter_name,
    _checkpoint_name,
):
    frozen = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    adapter.train_module.register_parameter("frozen", frozen)

    assert isinstance(adapter.train_module, torch.nn.Module)
    assert [name for name, _ in adapter.named_parameters()] == [parameter_name]
    assert list(adapter.parameters()) == [getattr(adapter, parameter_name)]
    assert all(parameter.requires_grad for parameter in adapter.parameters())


@pytest.mark.parametrize(
    ("adapter", "_parameter_name", "_checkpoint_name"),
    _light_adapters(),
)
def test_train_eval_and_compatibility_aliases_delegate_to_train_module(
    adapter,
    _parameter_name,
    _checkpoint_name,
):
    assert adapter.train() is adapter
    assert adapter.train_module.training is True
    assert adapter.eval() is adapter
    assert adapter.train_module.training is False

    adapter.prepare_for_training()
    assert adapter.train_module.training is True
    adapter.prepare_for_sampling()
    assert adapter.train_module.training is False


@pytest.mark.parametrize(
    ("adapter", "parameter_name", "_checkpoint_name"),
    _light_adapters(),
)
def test_state_dict_round_trip(adapter, parameter_name, _checkpoint_name):
    parameter = getattr(adapter, parameter_name)
    state = {name: value.detach().clone() for name, value in adapter.state_dict().items()}

    with torch.no_grad():
        parameter.add_(3.0)
    adapter.load_state_dict(state)

    assert set(state) == {parameter_name}
    torch.testing.assert_close(parameter, state[parameter_name])


@pytest.mark.parametrize(
    ("adapter", "parameter_name", "checkpoint_name"),
    _light_adapters(),
)
def test_legacy_checkpoint_methods_round_trip(
    tmp_path,
    adapter,
    parameter_name,
    checkpoint_name,
):
    parameter = getattr(adapter, parameter_name)
    with torch.no_grad():
        parameter.fill_(1.25)
    adapter.save_pretrained(tmp_path)

    assert (tmp_path / checkpoint_name).is_file()
    with torch.no_grad():
        parameter.zero_()
    adapter.load_checkpoint(tmp_path)

    torch.testing.assert_close(parameter, torch.full_like(parameter, 1.25))


@pytest.mark.parametrize(
    ("adapter", "_parameter_name", "_checkpoint_name"),
    _light_adapters(),
)
def test_light_checkpoint_loads_use_weights_only(
    tmp_path,
    monkeypatch,
    adapter,
    _parameter_name,
    _checkpoint_name,
):
    adapter.save_pretrained(tmp_path)
    original_load = torch.load
    observed = []

    def checked_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    adapter.load_checkpoint(tmp_path)

    assert observed == [True]


def test_sd3_full_checkpoint_load_uses_weights_only(tmp_path, monkeypatch):
    adapter = SD3TempFlowAdapter(
        {
            "device": "cpu",
            "use_lora": False,
            "extra": {"defer_load": True},
        }
    )
    adapter.pipeline = object()
    adapter.transformer = torch.nn.Linear(2, 2)
    torch.save(adapter.transformer.state_dict(), tmp_path / "transformer_state.pt")
    original_load = torch.load
    observed = []

    def checked_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    adapter.load_checkpoint(tmp_path)

    assert observed == [True]


def test_adapter_rollout_leaves_context_for_engine_finalization():
    adapter = MockWanAdapter({})

    batch = adapter.sample(
        ["prompt"],
        [{}],
        {"num_steps": 2, "seed": 17, "epoch_tag": 4},
    )

    assert batch.context is None


def test_tiny_branching_rollout_uses_singular_branch_id():
    adapter = TinyDiffusionAdapter({"device": "cpu"})

    batch = adapter.sample_branching(
        ["prompt"],
        [{}],
        {
            "num_steps": 3,
            "branch_step_index": 1,
            "branch_count": 2,
            "seed": 17,
        },
    )

    assert batch.branch_id.tolist() == [0, 1]
    assert batch.context is None


def _unloaded_heavy_adapters():
    return [
        SD3TempFlowAdapter({"device": "cpu", "extra": {"defer_load": True}}),
        WorldR1WanLegacyAdapter({}),
    ]


def test_world_r1_wan_exposes_read_only_flash_root_compatibility_alias() -> None:
    adapter = WorldR1WanLegacyAdapter({})

    assert adapter.wan_backend == "world_r1"
    assert adapter.flash_repo_root.name == "Flash-GRPO-main"
    with pytest.raises(AttributeError):
        adapter.flash_repo_root = Path("/tmp/other")


@pytest.mark.parametrize("adapter", _unloaded_heavy_adapters())
@pytest.mark.parametrize(
    "operation",
    [
        lambda adapter, _path: adapter.train_module,
        lambda adapter, _path: adapter.parameters(),
        lambda adapter, _path: adapter.named_parameters(),
        lambda adapter, _path: adapter.train(),
        lambda adapter, _path: adapter.eval(),
        lambda adapter, _path: adapter.state_dict(),
        lambda adapter, _path: adapter.load_state_dict({}),
        lambda adapter, path: adapter.save_pretrained(path),
        lambda adapter, path: adapter.load_checkpoint(path),
    ],
)
def test_unloaded_heavy_adapters_fail_closed(
    tmp_path,
    adapter,
    operation: Callable,
):
    with pytest.raises(AdapterNotLoadedError, match="load"):
        operation(adapter, tmp_path)


@pytest.mark.parametrize(
    "adapter",
    [
        SD3TempFlowAdapter({"device": "cpu", "extra": {"defer_load": True}}),
        WorldR1WanLegacyAdapter({}),
    ],
)
def test_heavy_adapters_expose_transformer_with_stable_filtered_names(adapter):
    transformer = torch.nn.Module()
    transformer.register_parameter("policy", torch.nn.Parameter(torch.ones(())))
    transformer.register_parameter(
        "frozen",
        torch.nn.Parameter(torch.zeros(()), requires_grad=False),
    )
    adapter.transformer = transformer

    assert adapter.train_module is transformer
    assert [name for name, _ in adapter.named_parameters()] == [
        "transformer.policy"
    ]
    assert list(adapter.parameters()) == [transformer.policy]


def test_sd3_reference_recompute_keeps_transformer_train_contract():
    adapter = SD3TempFlowAdapter(
        {
            "device": "cpu",
            "extra": {"defer_load": True, "tempflow_reference_mode": True},
        }
    )
    adapter.transformer = torch.nn.Linear(1, 1)

    adapter.prepare_for_sampling()
    assert adapter.transformer.training is False
    adapter.prepare_for_training()
    assert adapter.transformer.training is True


def test_sd3_non_reference_training_alias_preserves_sampling_mode():
    adapter = SD3TempFlowAdapter(
        {"device": "cpu", "extra": {"defer_load": True}}
    )
    adapter.transformer = torch.nn.Linear(1, 1)
    adapter.prepare_for_sampling()

    adapter.prepare_for_training()

    assert adapter.transformer.training is False
