"""CPU-only contract tests for the bounded Wan LoRA adapter.

These use fake Diffusers and PEFT modules. They verify the local adapter
contract only; they do not establish compatibility with real Wan or PEFT.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import types
from contextlib import nullcontext

import pytest
import torch

from visual_rl.configs.schema import VisualRLConfig
from visual_rl.model_adapters.wan import (
    DEFAULT_WAN_LORA_TARGETS,
    WAN_LORA_CHECKPOINT_SUBDIR,
    WorldR1WanLegacyAdapter,
)
from visual_rl.preflight import StaticPreflightError, static_preflight
from visual_rl.runner import ExperimentRunner


class _BaseTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        for name in (
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_k",
            "to_q",
            "to_v",
        ):
            setattr(self.block, name, torch.nn.Linear(1, 1, bias=False))
        self.block.to_out = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))

    def forward(self, *, hidden_states, **_kwargs):
        return (hidden_states * self.block.add_k_proj.weight.reshape(()),)


class _CoreWanTransformer(torch.nn.Module):
    """Wan2.1-shaped attention surface without joint-attention projections."""

    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        for name in ("to_k", "to_q", "to_v"):
            setattr(self.block, name, torch.nn.Linear(1, 1, bias=False))
        self.block.to_out = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))


class _PeftConfig:
    def __init__(self, target_modules):
        self.target_modules = target_modules


class _PeftTransformer(torch.nn.Module):
    def __init__(
        self,
        base,
        adapter_name="default",
        target_modules=None,
    ):
        super().__init__()
        self.base_model = base
        self.lora_weight = torch.nn.Parameter(torch.tensor(0.5))
        if target_modules is None:
            target_modules = set(DEFAULT_WAN_LORA_TARGETS)
        self.peft_config = {adapter_name: _PeftConfig(target_modules)}
        self.active_adapter = None
        self.forward_calls = 0
        self.saved_selected_adapters = None

    def forward(self, **kwargs):
        self.forward_calls += 1
        return (self.base_model(**kwargs)[0] + self.lora_weight,)

    def set_adapter(self, name):
        self.active_adapter = name

    def save_pretrained(
        self,
        path,
        safe_serialization=True,
        selected_adapters=None,
    ):
        del safe_serialization
        self.saved_selected_adapters = selected_adapters
        adapter_name = self.active_adapter or "default"
        if selected_adapters is not None:
            assert selected_adapters == [adapter_name]
        output = path if adapter_name == "default" else path / adapter_name
        output.mkdir(parents=True, exist_ok=True)
        target_modules = self.peft_config[adapter_name].target_modules
        if isinstance(target_modules, (set, frozenset, tuple)):
            target_modules = sorted(target_modules)
        (output / "adapter_config.json").write_text(
            json.dumps({"target_modules": target_modules}),
            encoding="utf-8",
        )
        (output / "adapter_model.safetensors").write_bytes(b"fake")


class _EmptyPeftTransformer(torch.nn.Module):
    def __init__(self, base, target_modules):
        super().__init__()
        self.base_model = base
        self.peft_config = {"default": _PeftConfig(target_modules)}

    def set_adapter(self, _name):
        return None


class _Pipeline:
    def __init__(self):
        self.transformer = _BaseTransformer()
        self.vae = torch.nn.Linear(1, 1, bias=False)
        self.text_encoder = torch.nn.Linear(1, 1, bias=False)
        self.scheduler = types.SimpleNamespace(timesteps=torch.tensor([1]))
        self._execution_device = "cpu"
        self.encoded_train_cfg = None
        self.sampled_train_cfg = None

    def to(self, _device):
        return self

    def encode_prompt(self, *, do_classifier_free_guidance, **_kwargs):
        self.encoded_train_cfg = do_classifier_free_guidance
        return torch.ones(1, 1), torch.zeros(1, 1)


@pytest.fixture
def fake_runtime(monkeypatch):
    state = types.SimpleNamespace(
        adapter_state={"lora_weight": torch.tensor(3.0)},
        empty_wrapper=False,
        expected_state=None,
        load_result_shape="object",
        loaded_target_modules=set(DEFAULT_WAN_LORA_TARGETS),
        missing=(),
        replace_parameter=False,
        unexpected=(),
    )

    class FakeWanPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return _Pipeline()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, _path, **kwargs):
            return _PeftTransformer(
                base,
                kwargs.get("adapter_name", "default"),
                state.loaded_target_modules,
            )

    def get_peft_model(base, config, **_kwargs):
        if state.empty_wrapper:
            return _EmptyPeftTransformer(base, config.target_modules)
        return _PeftTransformer(
            base,
            _kwargs.get("adapter_name", "default"),
            config.target_modules,
        )

    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.target_modules = kwargs["target_modules"]

    def load_peft_weights(_path, device):
        assert device == "cpu"
        return state.adapter_state

    def get_peft_model_state_dict(model, adapter_name):
        assert adapter_name == model.active_adapter
        if state.expected_state is not None:
            return state.expected_state
        return {"lora_weight": model.lora_weight}

    def set_peft_model_state_dict(model, values, adapter_name):
        assert adapter_name == model.active_adapter
        if state.replace_parameter:
            model.lora_weight = torch.nn.Parameter(torch.tensor(9.0))
        else:
            with torch.no_grad():
                model.lora_weight.copy_(values["lora_weight"])
        if state.load_result_shape == "none":
            return None
        if state.load_result_shape == "tuple":
            return state.missing, state.unexpected
        if state.load_result_shape == "object":
            return types.SimpleNamespace(
                missing_keys=state.missing,
                unexpected_keys=state.unexpected,
            )
        return object()

    diffusers = types.ModuleType("diffusers")
    diffusers.WanPipeline = FakeWanPipeline
    peft = types.ModuleType("peft")
    peft.LoraConfig = LoraConfig
    peft.PeftModel = FakePeftModel
    peft.get_peft_model = get_peft_model
    peft_utils = types.ModuleType("peft.utils")
    peft_save = types.ModuleType("peft.utils.save_and_load")
    peft_save.get_peft_model_state_dict = get_peft_model_state_dict
    peft_save.load_peft_weights = load_peft_weights
    peft_save.set_peft_model_state_dict = set_peft_model_state_dict
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.utils", peft_utils)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", peft_save)
    monkeypatch.setattr(
        "visual_rl.model_adapters.wan.legacy_repo_path",
        lambda _repo_root: nullcontext(),
    )
    return state


def _config(**overrides):
    config = {
        "model_path": "/offline/fake-wan",
        "device": "cpu",
        "use_lora": True,
        "extra": {},
    }
    config.update(overrides)
    return config


def _pipeline_with_logprob(pipeline, *, train_cfg, **_kwargs):
    assert pipeline.transformer is not None
    pipeline.sampled_train_cfg = train_cfg
    return (
        torch.zeros(1, 1),
        [torch.ones(1, 1), torch.ones(1, 1)],
        [torch.zeros(1)],
        [],
        [torch.tensor(1)],
    )


def _sde_step(_scheduler, noise_pred, *_args, **_kwargs):
    return None, noise_pred.reshape(noise_pred.shape[0], -1).sum(dim=1)


def test_defaults_and_config_validation_use_top_level_values():
    adapter = WorldR1WanLegacyAdapter(
        _config(use_lora=False, extra={"use_lora": True, "lora_rank": 9})
    )

    assert adapter.use_lora is False
    assert adapter.lora_rank == 9
    assert adapter.lora_alpha == 64
    assert adapter.lora_targets == DEFAULT_WAN_LORA_TARGETS
    assert adapter.lora_targets == ["to_k", "to_out.0", "to_q", "to_v"]
    assert adapter.adapter_name == "default"
    for key, value in (
        ("use_lora", "yes"),
        ("lora_rank", 0),
        ("lora_alpha", False),
        ("lora_path", ""),
        ("adapter_name", ""),
    ):
        with pytest.raises(ValueError):
            WorldR1WanLegacyAdapter(_config(**{key: value}))
    with pytest.raises(ValueError, match="unique"):
        WorldR1WanLegacyAdapter(_config(lora_target_modules=["to_q", "to_q"]))
    for adapter_name in (
        ".",
        "..",
        "/tmp/adapter",
        "bad/name",
        "bad\\name",
        "é",
        "a" * 65,
    ):
        with pytest.raises(ValueError, match="safe ASCII identifier"):
            WorldR1WanLegacyAdapter(_config(adapter_name=adapter_name))


def test_wan21_core_targets_are_default_but_explicit_extensions_remain_strict():
    core = _CoreWanTransformer()
    default_adapter = WorldR1WanLegacyAdapter(_config())
    default_adapter._validate_lora_targets(core)

    legacy_extended_targets = [
        "add_k_proj",
        "add_q_proj",
        "add_v_proj",
        "to_add_out",
        *DEFAULT_WAN_LORA_TARGETS,
    ]
    explicit_adapter = WorldR1WanLegacyAdapter(
        _config(lora_target_modules=legacy_extended_targets)
    )
    with pytest.raises(RuntimeError, match="missing targets: add_k_proj"):
        explicit_adapter._validate_lora_targets(core)
    explicit_adapter._validate_lora_targets(_BaseTransformer())


def test_lora_load_freezes_base_exposes_only_lora_and_shares_pipeline(fake_runtime):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()

    assert adapter.pipeline.transformer is adapter.transformer
    assert all("lora" in name for name, _parameter in adapter.named_parameters())
    assert all(parameter.requires_grad for parameter in adapter.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in adapter.transformer.base_model.parameters()
    )
    assert adapter.runtime_metadata()["checkpoint_format"] == "wan_peft_adapter_v1"


def test_lora_path_load_accepts_direct_and_stable_adapter_layout(
    tmp_path, fake_runtime
):
    direct = tmp_path / "direct"
    _PeftTransformer(_BaseTransformer()).save_pretrained(direct)
    stable = tmp_path / "stable" / WAN_LORA_CHECKPOINT_SUBDIR
    _PeftTransformer(_BaseTransformer()).save_pretrained(stable)

    for path in (direct, stable.parent):
        adapter = WorldR1WanLegacyAdapter(_config(lora_path=str(path)))
        adapter.load()
        assert adapter.transformer.active_adapter == "default"


@pytest.mark.parametrize(
    ("expected_targets", "actual_targets"),
    [
        (["to_q", "to_k"], {"to_q", "to_k"}),
        (["to_q", "to_k"], ["to_k", "to_q"]),
        (["to_q"], "to_q"),
    ],
)
def test_lora_path_target_modules_exact_match_is_recorded(
    tmp_path,
    fake_runtime,
    expected_targets,
    actual_targets,
):
    checkpoint = tmp_path / "adapter"
    _PeftTransformer(
        _BaseTransformer(),
        target_modules=actual_targets,
    ).save_pretrained(checkpoint)
    fake_runtime.loaded_target_modules = actual_targets

    adapter = WorldR1WanLegacyAdapter(
        _config(lora_path=str(checkpoint), lora_target_modules=expected_targets)
    ).load()

    assert adapter.runtime_metadata()["lora_targets"] == sorted(expected_targets)


@pytest.mark.parametrize(
    ("expected_targets", "actual_targets", "error"),
    [
        (["to_q", "to_k"], {"to_q"}, "missing targets: to_k"),
        (["to_q"], ["to_q", "to_k"], "unexpected targets: to_k"),
        (
            ["to_q"],
            "to_k",
            "missing targets: to_q; unexpected targets: to_k",
        ),
    ],
)
def test_lora_path_target_modules_mismatch_fails_closed(
    tmp_path,
    fake_runtime,
    expected_targets,
    actual_targets,
    error,
):
    checkpoint = tmp_path / "adapter"
    _PeftTransformer(
        _BaseTransformer(),
        target_modules=actual_targets,
    ).save_pretrained(checkpoint)
    fake_runtime.loaded_target_modules = actual_targets

    with pytest.raises(RuntimeError, match=error):
        WorldR1WanLegacyAdapter(
            _config(
                lora_path=str(checkpoint),
                lora_target_modules=expected_targets,
            )
        ).load()


def test_lora_failures_are_explicit_and_fail_closed(
    tmp_path, monkeypatch, fake_runtime
):
    with pytest.raises(RuntimeError, match="missing targets: missing"):
        WorldR1WanLegacyAdapter(_config(lora_target_modules=["missing"])).load()
    with pytest.raises(RuntimeError, match="missing targets: missing"):
        WorldR1WanLegacyAdapter(
            _config(lora_target_modules=["add_k_proj", "missing"])
        ).load()
    with pytest.raises(RuntimeError, match="lora_path"):
        WorldR1WanLegacyAdapter(_config(lora_path=str(tmp_path))).load()

    monkeypatch.delitem(sys.modules, "peft", raising=False)
    with pytest.raises(ImportError, match="Wan LoRA"):
        WorldR1WanLegacyAdapter(_config()).load()


def test_wan_component_freeze_and_dtype_contract(fake_runtime):
    adapter = WorldR1WanLegacyAdapter(_config(use_lora=False, dtype="bfloat16")).load()

    assert next(adapter.pipeline.vae.parameters()).dtype is torch.float32
    assert not any(
        parameter.requires_grad for parameter in adapter.pipeline.vae.parameters()
    )
    assert next(adapter.pipeline.text_encoder.parameters()).dtype is torch.bfloat16
    assert not any(
        parameter.requires_grad
        for parameter in adapter.pipeline.text_encoder.parameters()
    )
    assert next(adapter.transformer.parameters()).dtype is torch.bfloat16
    assert all(
        parameter.requires_grad for parameter in adapter.transformer.parameters()
    )


def test_lora_attach_rejects_empty_trainable_state(fake_runtime):
    fake_runtime.empty_wrapper = True

    with pytest.raises(RuntimeError, match="no trainable LoRA"):
        WorldR1WanLegacyAdapter(_config()).load()


def test_lora_save_layout_and_non_lora_legacy_layout(tmp_path, fake_runtime):
    lora = WorldR1WanLegacyAdapter(_config())
    lora.load()
    lora.save_pretrained(tmp_path / "lora")
    assert (
        tmp_path / "lora" / WAN_LORA_CHECKPOINT_SUBDIR / "adapter_config.json"
    ).is_file()
    assert (
        tmp_path / "lora" / WAN_LORA_CHECKPOINT_SUBDIR / "adapter_model.safetensors"
    ).is_file()
    assert not (tmp_path / "lora" / "transformer_state.pt").exists()
    assert lora.transformer.saved_selected_adapters == ["default"]

    full = WorldR1WanLegacyAdapter(_config(use_lora=False))
    full.load()
    full.save_pretrained(tmp_path / "full")
    assert (tmp_path / "full" / "transformer_state.pt").is_file()


def test_wan_full_checkpoint_load_uses_weights_only(
    tmp_path,
    monkeypatch,
    fake_runtime,
):
    adapter = WorldR1WanLegacyAdapter(_config(use_lora=False))
    adapter.load()
    adapter.save_pretrained(tmp_path)
    original_load = torch.load
    observed = []

    def checked_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    adapter.load_checkpoint(tmp_path)

    assert observed == [True]


def test_lora_checkpoint_load_is_in_place_and_rejects_empty_unexpected_and_mode_mismatch(
    tmp_path, fake_runtime
):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    adapter.save_pretrained(tmp_path)
    before = {
        name: id(parameter)
        for name, parameter in adapter.transformer.named_parameters()
    }
    adapter.load_checkpoint(tmp_path)
    assert {
        name: id(parameter)
        for name, parameter in adapter.transformer.named_parameters()
    } == before
    assert adapter.transformer.lora_weight.item() == 3.0

    fake_runtime.adapter_state = {}
    with pytest.raises(RuntimeError, match="contains no tensors"):
        adapter.load_checkpoint(tmp_path)
    fake_runtime.adapter_state = {"lora_weight": torch.tensor(2.0)}
    fake_runtime.unexpected = ("base_model.block.lora_A.default.weight",)
    with pytest.raises(RuntimeError, match="unexpected active adapter keys"):
        adapter.load_checkpoint(tmp_path)
    fake_runtime.unexpected = ()
    fake_runtime.missing = ("base_model.block.lora_B.default.weight",)
    fake_runtime.load_result_shape = "tuple"
    with pytest.raises(RuntimeError, match="missing keys"):
        adapter.load_checkpoint(tmp_path)
    fake_runtime.missing = ()
    fake_runtime.load_result_shape = "object"
    fake_runtime.replace_parameter = True
    with pytest.raises(RuntimeError, match="replaced Parameter"):
        adapter.load_checkpoint(tmp_path)

    non_lora = WorldR1WanLegacyAdapter(_config(use_lora=False))
    non_lora.load()
    with pytest.raises(RuntimeError, match="LoRA-only"):
        non_lora.load_checkpoint(tmp_path)
    full_state = tmp_path / "full-state"
    full_state.mkdir()
    torch.save({}, full_state / "transformer_state.pt")
    with pytest.raises(RuntimeError, match="full-transformer"):
        adapter.load_checkpoint(full_state)


def test_peft_state_contract_accepts_base_missing_and_rejects_adapter_drift(
    tmp_path, fake_runtime
):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    adapter.save_pretrained(tmp_path)

    fake_runtime.missing = ("base_model.model.block.add_k_proj.weight",)
    adapter.load_checkpoint(tmp_path)

    fake_runtime.missing = ()
    fake_runtime.expected_state = {
        "lora_A.weight": torch.zeros(()),
        "lora_B.weight": torch.zeros(()),
    }
    fake_runtime.adapter_state = {"lora_A.weight": torch.zeros(())}
    with pytest.raises(RuntimeError, match="missing active adapter keys.*lora_B"):
        adapter.load_checkpoint(tmp_path)

    fake_runtime.expected_state = {"lora_weight": torch.zeros(())}
    fake_runtime.adapter_state = {
        "lora_weight": torch.zeros(()),
        "other_adapter.lora_weight": torch.zeros(()),
    }
    with pytest.raises(RuntimeError, match="extra adapter keys.*other_adapter"):
        adapter.load_checkpoint(tmp_path)

    fake_runtime.adapter_state = {"lora_weight": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="shape mismatch.*lora_weight"):
        adapter.load_checkpoint(tmp_path)


def test_peft_checkpoint_target_modules_mismatch_fails_before_weight_load(
    tmp_path,
    fake_runtime,
):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    adapter.save_pretrained(tmp_path)
    config_path = tmp_path / WAN_LORA_CHECKPOINT_SUBDIR / "adapter_config.json"
    config_path.write_text(
        json.dumps({"target_modules": ["to_q"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing targets"):
        adapter.load_checkpoint(tmp_path)

    assert adapter.transformer.lora_weight.item() == 0.5


def test_legacy_eight_target_checkpoint_is_rejected_by_core_wan_config(
    tmp_path,
    fake_runtime,
):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    adapter.save_pretrained(tmp_path)
    config_path = tmp_path / WAN_LORA_CHECKPOINT_SUBDIR / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_modules": [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                    *DEFAULT_WAN_LORA_TARGETS,
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected targets: add_k_proj, add_q_proj, add_v_proj, to_add_out",
    ):
        adapter.load_checkpoint(tmp_path)

    assert adapter.transformer.lora_weight.item() == 0.5


def test_non_default_adapter_save_and_restore_layout(tmp_path, fake_runtime):
    adapter = WorldR1WanLegacyAdapter(_config(adapter_name="policy.v2"))
    adapter.load()
    adapter.save_pretrained(tmp_path)

    nested = tmp_path / WAN_LORA_CHECKPOINT_SUBDIR / "policy.v2"
    assert (nested / "adapter_config.json").is_file()
    assert (nested / "adapter_model.safetensors").is_file()
    assert adapter.transformer.saved_selected_adapters == ["policy.v2"]

    before = {
        name: id(parameter)
        for name, parameter in adapter.transformer.named_parameters()
    }
    adapter.load_checkpoint(tmp_path)
    assert {
        name: id(parameter)
        for name, parameter in adapter.transformer.named_parameters()
    } == before


def test_lora_save_rejects_adapter_written_below_supported_layout(
    tmp_path,
    monkeypatch,
    fake_runtime,
):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()

    def save_nested(path, **_kwargs):
        nested = Path(path) / WAN_LORA_CHECKPOINT_SUBDIR
        _PeftTransformer(_BaseTransformer()).save_pretrained(nested)

    monkeypatch.setattr(adapter.transformer, "save_pretrained", save_nested)

    with pytest.raises(RuntimeError, match="did not write a complete adapter"):
        adapter.save_pretrained(tmp_path)


def test_wan_checkpoint_rejects_symlinked_payload_files(tmp_path, fake_runtime):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    checkpoint = tmp_path / "lora"
    adapter.save_pretrained(checkpoint)
    weights = checkpoint / WAN_LORA_CHECKPOINT_SUBDIR / "adapter_model.safetensors"
    external_weights = tmp_path / "external.safetensors"
    external_weights.write_bytes(b"external")
    weights.unlink()
    weights.symlink_to(external_weights)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        adapter.load_checkpoint(checkpoint)

    full = WorldR1WanLegacyAdapter(_config(use_lora=False))
    full.load()
    full_checkpoint = tmp_path / "full"
    full.save_pretrained(full_checkpoint)
    state_path = full_checkpoint / "transformer_state.pt"
    external_state = tmp_path / "external_state.pt"
    torch.save({}, external_state)
    state_path.unlink()
    state_path.symlink_to(external_state)

    with pytest.raises(RuntimeError, match="transformer_state.pt.*symlink"):
        full.load_checkpoint(full_checkpoint)


def test_wan_checkpoint_rejects_symlinked_adapter_directory(tmp_path, fake_runtime):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    external_adapter = tmp_path / "external_adapter"
    _PeftTransformer(_BaseTransformer()).save_pretrained(external_adapter)
    (checkpoint / WAN_LORA_CHECKPOINT_SUBDIR).symlink_to(
        external_adapter,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="adapter directory.*symlink"):
        adapter.load_checkpoint(checkpoint)


def test_sample_and_recompute_use_same_wrapped_transformer_with_gradient(fake_runtime):
    adapter = WorldR1WanLegacyAdapter(
        _config(
            wan_pipeline_with_logprob=_pipeline_with_logprob,
            sde_step_with_logprob=_sde_step,
        )
    )
    adapter.load()
    batch = adapter.sample(["offline"], [{}], {"num_steps": 1, "train_cfg": False})
    loss = adapter.recompute_log_probs(batch).sum()
    loss.backward()

    assert adapter.pipeline.transformer is adapter.transformer
    assert adapter.pipeline.encoded_train_cfg is False
    assert adapter.pipeline.sampled_train_cfg is False
    assert batch.model_metadata["sample_config"]["train_cfg"] is False
    assert adapter.transformer.forward_calls == 1
    assert adapter.transformer.lora_weight.grad is not None
    adapter.pipeline.transformer = _BaseTransformer()
    with pytest.raises(RuntimeError, match="same wrapped module"):
        adapter.recompute_log_probs(batch)


@pytest.mark.parametrize("result_shape", ["object", "tuple", "none"])
def test_peft_checkpoint_accepts_clean_real_api_result_shapes(
    tmp_path, fake_runtime, result_shape
):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    adapter.save_pretrained(tmp_path)
    fake_runtime.load_result_shape = result_shape

    adapter.load_checkpoint(tmp_path)

    assert adapter.transformer.lora_weight.item() == 3.0


def test_peft_checkpoint_rejects_unknown_load_result_shape(tmp_path, fake_runtime):
    adapter = WorldR1WanLegacyAdapter(_config())
    adapter.load()
    adapter.save_pretrained(tmp_path)
    fake_runtime.load_result_shape = "unknown"

    with pytest.raises(RuntimeError, match="unsupported compatibility result"):
        adapter.load_checkpoint(tmp_path)


def test_train_cfg_false_rejects_legacy_pipeline_without_explicit_cfg(fake_runtime):
    def fixed_cfg_pipeline(_pipeline, **_kwargs):
        raise AssertionError("unsupported train_cfg=False pipeline was called")

    fixed_cfg_pipeline.__signature__ = inspect.Signature(
        [inspect.Parameter("pipeline", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    adapter = WorldR1WanLegacyAdapter(
        _config(wan_pipeline_with_logprob=fixed_cfg_pipeline)
    )
    adapter.load()

    with pytest.raises(RuntimeError, match="only supports.*train_cfg=True"):
        adapter.sample(["offline"], [{}], {"num_steps": 1, "train_cfg": False})


def test_runner_preserves_model_lora_path_unless_train_path_is_explicit():
    config = VisualRLConfig(run_name="wan-lora-path")
    config.model.extra["lora_path"] = "/model/adapter"

    inherited = ExperimentRunner._resolved_model_config(config)
    assert WorldR1WanLegacyAdapter(inherited).lora_path == "/model/adapter"

    config.train.lora_path = "/train/adapter"
    overridden = ExperimentRunner._resolved_model_config(config)
    assert WorldR1WanLegacyAdapter(overridden).lora_path == "/train/adapter"


def test_preflight_declares_peft_only_for_selected_wan_lora_and_stays_static(
    monkeypatch,
):
    original_find_spec = __import__("importlib").util.find_spec

    def reject_peft_import(name, *args, **kwargs):
        if name == "peft":
            raise AssertionError("static preflight must not inspect-import peft")
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("importlib.util.find_spec", reject_peft_import)
    config = VisualRLConfig(run_name="wan-static")
    config.paths.output_dir = "/tmp/wan-static"
    config.model.name = "world_r1_wan_legacy"
    config.model.extra = {"lora_rank": 32, "lora_alpha": 64}
    report = static_preflight(config)
    model = next(item for item in report.components if item.kind == "model")
    assert "peft" in model.dependencies

    config.use_lora = False
    report = static_preflight(config)
    assert (
        "peft"
        not in next(
            item for item in report.components if item.kind == "model"
        ).dependencies
    )
    config.model.extra["lora_rank"] = 0
    with pytest.raises(StaticPreflightError, match="lora_rank"):
        static_preflight(config)
    config.model.extra["lora_rank"] = 32
    config.model.name = "mock_wan"
    config.use_lora = True
    report = static_preflight(config)
    assert (
        "peft"
        not in next(
            item for item in report.components if item.kind == "model"
        ).dependencies
    )

    config.model.name = "world_r1_wan_legacy"
    config.model.extra["lora_rank"] = 0
    with pytest.raises(StaticPreflightError, match="lora_rank"):
        static_preflight(config)

    config.model.extra["lora_rank"] = 32
    config.model.extra["adapter_name"] = "../policy"
    with pytest.raises(StaticPreflightError, match="adapter_name"):
        static_preflight(config)


def test_preflight_ignores_shadowed_model_extra_lora_path():
    config = VisualRLConfig(run_name="wan-shadowed-lora")
    config.paths.output_dir = "/tmp/wan-shadowed-lora"
    config.model.name = "world_r1_wan_legacy"
    config.use_lora = True
    config.train.lora_path = "/tmp/effective-lora"
    config.model.extra = {"lora_path": "ignored-relative-path"}

    static_preflight(config)
