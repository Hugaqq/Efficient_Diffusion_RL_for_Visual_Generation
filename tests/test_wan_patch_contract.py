"""CPU-only contracts for explicit Wan reference patch selection."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch

from visual_rl.configs.schema import VisualRLConfig
from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter
from visual_rl.preflight import StaticPreflightError, static_preflight
from visual_rl.third_party import legacy as legacy_helpers


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = types.SimpleNamespace(in_channels=2)

    def forward(self, *, hidden_states, **_kwargs):
        return (hidden_states * self.weight,)


class _Pipeline:
    def __init__(self):
        self.transformer = _Transformer()
        self.scheduler = types.SimpleNamespace(timesteps=torch.tensor([900]))
        self._execution_device = "cpu"
        self.vae_scale_factor_temporal = 4

    def encode_prompt(self, *, prompt, do_classifier_free_guidance, **_kwargs):
        batch = len(prompt)
        negative = torch.zeros(batch, 1) if do_classifier_free_guidance else None
        return torch.ones(batch, 1), negative


def _adapter_config(**extra):
    return {
        "model_path": "/offline/fake-wan",
        "device": "cpu",
        "use_lora": False,
        "extra": extra,
    }


def _loaded_adapter(**extra):
    adapter = WorldR1WanLegacyAdapter(_adapter_config(**extra))
    adapter.pipeline = _Pipeline()
    adapter.transformer = adapter.pipeline.transformer
    adapter.scheduler = adapter.pipeline.scheduler
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    return adapter


def _preflight_config() -> VisualRLConfig:
    config = VisualRLConfig(run_name="wan-contract")
    config.paths.output_dir = "/tmp/wan-contract"
    config.model.name = "world_r1_wan_legacy"
    config.model.model_path = "/offline/fake-wan"
    config.model.extra = {"wan_backend": "world_r1"}
    return config


def _install_reference_modules(monkeypatch, world_pipeline, flash_pipeline):
    flow_grpo = types.ModuleType("flow_grpo")
    patch = types.ModuleType("flow_grpo.diffusers_patch")
    world = types.ModuleType("flow_grpo.diffusers_patch.wan_pipeline_with_logprob")
    flash = types.ModuleType(
        "flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample"
    )
    world.wan_pipeline_with_logprob = world_pipeline
    world.sde_step_with_logprob = lambda *_args, **_kwargs: None
    flash.wan_pipeline_with_logprob = flash_pipeline
    flash.sde_step_with_logprob = lambda *_args, **_kwargs: None
    for name, module in (
        ("flow_grpo", flow_grpo),
        ("flow_grpo.diffusers_patch", patch),
        (world.__name__, world),
        (flash.__name__, flash),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_backend_selects_exact_reference_import_path(monkeypatch):
    def world_pipeline(*_args, **_kwargs):
        return None

    def flash_pipeline(*_args, **_kwargs):
        return None

    _install_reference_modules(monkeypatch, world_pipeline, flash_pipeline)
    monkeypatch.setattr(
        "visual_rl.model_adapters.wan.legacy_repo_path",
        lambda _root: nullcontext(),
    )

    world = WorldR1WanLegacyAdapter(_adapter_config(wan_backend="world_r1"))
    flash = WorldR1WanLegacyAdapter(_adapter_config(wan_backend="flash"))

    assert world._load_wan_pipeline_with_logprob()[0] is world_pipeline
    assert flash._load_wan_pipeline_with_logprob()[0] is flash_pipeline
    assert world.runtime_metadata()["wan_backend"] == "world_r1"
    assert flash.runtime_metadata()["wan_backend"] == "flash"


def test_backend_resolves_only_its_reference_root_with_repo_root_compatibility(
    monkeypatch,
):
    seen = []

    def resolve(value):
        seen.append(value)
        return Path(value)

    monkeypatch.setattr("visual_rl.model_adapters.wan.resolve_legacy_repo", resolve)
    WorldR1WanLegacyAdapter(
        _adapter_config(
            wan_backend="world_r1",
            world_r1_root="/references/world",
            flash_grpo_root="/ignored/flash",
        )
    )
    WorldR1WanLegacyAdapter(
        _adapter_config(
            wan_backend="flash",
            world_r1_root="/ignored/world",
            flash_grpo_root="/references/flash",
        )
    )
    WorldR1WanLegacyAdapter(_adapter_config(repo_root="/legacy/compatible"))

    assert seen == [
        "/references/world",
        "/references/flash",
        "/legacy/compatible",
    ]


def _write_fake_reference_repo(root: Path, marker: str) -> Path:
    module_dir = root / "Flash-GRPO-main/flow_grpo/diffusers_patch"
    module_dir.mkdir(parents=True)
    (module_dir.parent / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "wan2_1_pipeline_with_logprob_sample.py").write_text(
        f"MARKER = {marker!r}\n",
        encoding="utf-8",
    )
    return module_dir.parents[1]


def test_reference_resolver_imports_exact_current_layout_and_env_override(
    tmp_path, monkeypatch
):
    project = tmp_path / "workspace/framecode"
    project.mkdir(parents=True)
    current_repo = _write_fake_reference_repo(
        project.parent / "code_base", "current-layout"
    )
    env_root = tmp_path / "explicit-reference-root"
    env_repo = _write_fake_reference_repo(env_root, "env-override")
    monkeypatch.setattr(legacy_helpers, "project_root", lambda: project)

    monkeypatch.delenv("VISUAL_RL_REFERENCE_CODE_ROOT", raising=False)
    assert legacy_helpers.resolve_legacy_repo("Flash-GRPO-main") == current_repo
    with legacy_helpers.legacy_repo_path(current_repo):
        module = importlib.import_module(
            "flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample"
        )
        assert module.MARKER == "current-layout"
        assert Path(module.__file__).is_relative_to(current_repo)

    monkeypatch.setenv("VISUAL_RL_REFERENCE_CODE_ROOT", str(env_root))
    assert legacy_helpers.resolve_legacy_repo("Flash-GRPO-main") == env_repo
    with legacy_helpers.legacy_repo_path(env_repo):
        module = importlib.import_module(
            "flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample"
        )
        assert module.MARKER == "env-override"
        assert Path(module.__file__).is_relative_to(env_repo)


def test_backend_specific_pipeline_allowlist_and_train_cfg_fail_closed():
    adapter = _loaded_adapter(wan_backend="flash")
    called = {}

    def pipeline(_pipeline, **kwargs):
        called.update(kwargs)
        return None

    adapter._call_pipeline_with_logprob(
        pipeline,
        injected=False,
        train_cfg=True,
        index=2,
        guidance_scale=5.0,
    )
    assert called == {"index": 2, "guidance_scale": 5.0}
    with pytest.raises(RuntimeError, match="does not allow.*use_camera_trajectory"):
        adapter._call_pipeline_with_logprob(
            pipeline,
            injected=False,
            train_cfg=True,
            guidance_scale=5.0,
            use_camera_trajectory=True,
        )
    with pytest.raises(RuntimeError, match="must equal"):
        adapter._call_pipeline_with_logprob(
            pipeline,
            injected=False,
            train_cfg=False,
            guidance_scale=5.0,
        )


@pytest.mark.parametrize(
    ("train_cfg", "guidance_scale"),
    [(False, 5.0), (True, 1.0)],
)
def test_wan_train_cfg_matches_guidance_in_preflight_and_runtime(
    train_cfg, guidance_scale
):
    config = _preflight_config()
    config.model.extra["train_cfg"] = train_cfg
    config.sample.guidance_scale = guidance_scale
    with pytest.raises(StaticPreflightError, match="must equal"):
        static_preflight(config)

    adapter = _loaded_adapter(wan_backend="world_r1", train_cfg=train_cfg)
    with pytest.raises(ValueError, match="must equal"):
        adapter.sample(
            ["fake"],
            [{}],
            {"num_steps": 1, "guidance_scale": guidance_scale},
        )


def test_wan_num_videos_per_prompt_is_typed_one_with_rollout_priority():
    config = _preflight_config()
    config.model.extra["num_videos_per_prompt"] = 2
    with pytest.raises(StaticPreflightError, match="Wan v1 requires"):
        static_preflight(config)

    config.rollout["num_videos_per_prompt"] = 1
    static_preflight(config)
    adapter = _loaded_adapter(
        wan_backend="world_r1",
        num_videos_per_prompt=2,
    )
    assert (
        adapter._runtime_options({"guidance_scale": 5.0, "num_videos_per_prompt": 1})[
            "num_videos_per_prompt"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="Wan v1 requires"):
        adapter.sample(
            ["fake"],
            [{}],
            {"num_steps": 1, "num_videos_per_prompt": 2},
        )

    config.rollout["num_videos_per_prompt"] = "1"
    with pytest.raises(StaticPreflightError, match="must resolve to an integer"):
        static_preflight(config)
    with pytest.raises(ValueError, match="must resolve to an integer"):
        adapter._runtime_options({"guidance_scale": 5.0, "num_videos_per_prompt": "1"})


def test_world_camera_trajectory_object_is_shared_with_latents_and_metadata():
    trajectory = {"poses": "same-object"}
    trajectories = [trajectory]
    prepared = torch.randn(1, 2, 2, 1, 1)
    observed = {}

    def get_trajectories(prompts, **kwargs):
        assert prompts == ["pan left"]
        assert kwargs["batch_size"] == 1
        return trajectories, [["pan_left"]], list(prompts), [None]

    def prepare_latents(*, camera_trajectories, **kwargs):
        observed["prepare_trajectories"] = camera_trajectories
        observed["prepare_kwargs"] = kwargs
        return prepared

    def pipeline(_pipeline, *, train_cfg, **kwargs):
        observed["pipeline_kwargs"] = kwargs
        assert train_cfg is True
        return (
            torch.zeros(1, 2, 3, 2, 2),
            [torch.zeros(1, 2, 2, 1, 1), torch.ones(1, 2, 2, 1, 1)],
            [torch.zeros(1)],
            [],
            [torch.tensor(900)],
        )

    adapter = _loaded_adapter(
        wan_backend="world_r1",
        use_camera_trajectory=True,
        get_camera_trajectories_for_batch=get_trajectories,
        prepare_latents_with_camera=prepare_latents,
        wan_pipeline_with_logprob=pipeline,
    )
    source_metadata = {"source": "fake"}
    batch = adapter.sample(
        ["pan left"],
        [source_metadata],
        {"num_steps": 1, "use_camera_trajectory": True},
    )

    assert observed["prepare_trajectories"] is trajectories
    assert observed["pipeline_kwargs"]["latents"] is prepared
    assert observed["pipeline_kwargs"]["use_camera_trajectory"] is False
    assert batch.metadata[0]["camera_trajectory"] is trajectory
    assert source_metadata == {"source": "fake"}
    with pytest.raises(ValueError, match="num_videos_per_prompt=1"):
        adapter.sample(
            ["pan left"],
            [{}],
            {
                "num_steps": 1,
                "use_camera_trajectory": True,
                "num_videos_per_prompt": 2,
            },
        )


@pytest.mark.parametrize("value", ["https://example.test/lora", "relative/lora"])
def test_wan_lora_path_rejects_nonlocal_or_relative_values_early(value):
    with pytest.raises(ValueError, match="local absolute path"):
        WorldR1WanLegacyAdapter(
            {
                "model_path": "/offline/fake-wan",
                "use_lora": True,
                "lora_path": value,
            }
        )

    config = _preflight_config()
    config.train.lora_path = value
    with pytest.raises(
        StaticPreflightError, match="effective lora_path.*local absolute"
    ):
        static_preflight(config)


def test_train_lora_path_fully_shadows_legacy_extra_value():
    config = _preflight_config()
    config.train.lora_path = "/tmp/effective-wan-lora"
    for stale in ("https://example.test/stale", "stale/relative"):
        config.model.extra["lora_path"] = stale
        static_preflight(config)
        adapter = WorldR1WanLegacyAdapter(
            {
                "model_path": "/offline/fake-wan",
                "use_lora": True,
                "lora_path": config.train.lora_path,
                "extra": {"lora_path": stale},
            }
        )
        assert adapter.lora_path == "/tmp/effective-wan-lora"

    config.train.lora_path = None
    with pytest.raises(StaticPreflightError, match="effective lora_path"):
        static_preflight(config)


def test_preflight_validates_backend_selected_root_and_dependencies(tmp_path: Path):
    config = _preflight_config()
    config.model.extra.update(
        {
            "wan_backend": "flash",
            "flash_grpo_root": str(tmp_path / "flash"),
            "world_r1_root": "ignored/relative",
        }
    )
    config.algorithm.name = "flash_grpo"
    config.sample.name = "single_step"
    config.algorithm.objective_version = "reference_v1"
    config.algorithm.beta = 0.0
    report = static_preflight(config)
    model = next(item for item in report.components if item.kind == "model")
    assert set(("torch", "diffusers", "numpy", "PIL", "peft")) <= set(
        model.dependencies
    )

    config.algorithm.beta = 0.1
    with pytest.raises(StaticPreflightError, match="reference_v1 requires beta=0"):
        static_preflight(config)
    config.algorithm.beta = 0.0
    config.model.extra["wan_backend"] = "mixed"
    with pytest.raises(StaticPreflightError, match="wan_backend"):
        static_preflight(config)
