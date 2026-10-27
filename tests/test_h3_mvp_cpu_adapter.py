"""Registered H3 adapter CPU ports; no default runtime/GPU readiness claim."""

import builtins
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch
from safetensors.torch import save_file

from tests.test_h3_mvp_cpu import _tiny_loading_class
from tests.test_h3_mvp_cpu_decode import _artifacts, _leaf_setup
from tests.test_h3_mvp_cpu_text_cache import _fixture as _cache_files, _identity
from visual_rl.composition.registry import DeclarationResolver, Registry
from visual_rl.core.contracts import LatentLayout, MediaKind, TaskKind
from visual_rl.models.catalog import MODEL_CATALOG_FRAGMENT
from visual_rl.models import h3_config, h3_interface
from visual_rl.models.h3_config import H3Config
from visual_rl.models.h3_interface import H3PromptInput, H3ModelInput, H3Generators, H3Conditioning, H3_GUIDANCE_ID
from visual_rl.models.implementations import h3, h3_loading
from visual_rl.models.interface import ModelAdapter, ModelInput, ModelLatentSpec, ModelPortError
from visual_rl.models.lifecycle.components import ComponentManager, ComponentLoadSession, ComponentRole
from visual_rl.models.lifecycle.prepared import PreparedModelBundle, PreparedComponentHandle, PreparedBundleError


def _params():
    return dict(artifact_ref='h3-model', text_cache_ref='h3-text', height=32, width=32,
                max_tokens=4, max_embedding_bytes=100000, max_media_bytes=10000000,
                max_decode_bytes=50000000, lora_rank=2, lora_alpha=4)


def _adapter(tmp_path, monkeypatch):
    model_dir, cache_dir = tmp_path / 'model', tmp_path / 'cache'
    model_dir.mkdir()
    _cache_files(cache_dir)
    (model_dir / 'transformer').mkdir()
    config = dict(_class_name='MiniMaxH3Transformer3DModel', in_channels=24, audio_in_channels=32,
                  patch_size=[1, 2, 2], text_dim=5120, hidden_size=4, num_layers=1)
    (model_dir / 'transformer/config.json').write_text(json.dumps(config))
    base = _tiny_loading_class()

    class HeavyTransformerLeaf(base):
        calls = []
        def forward(self, *, hidden_states, audio_hidden_states, encoder_hidden_states, **kwargs):
            type(self).calls.append((kwargs['timestep'].detach().clone(), torch.is_grad_enabled()))
            context = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype)).mean().float()
            v, a = super().forward(hidden_states, audio_hidden_states)
            return v + context, a + context

    monkeypatch.setattr(h3_loading, '_resolve_h3_class', lambda: HeavyTransformerLeaf)
    registry = Registry('model', MODEL_CATALOG_FRAGMENT.descriptors)
    declaration = DeclarationResolver().resolve(registry, 'h3-t2va', _params())
    module, name = declaration.implementation_class_path.split(':')
    cls = getattr(importlib.import_module(module), name)
    context = dict(artifact_paths={'h3-model': model_dir, 'h3-text': cache_dir}, expected_cache_identity=_identity())
    adapter = cls.from_config(declaration.config, runtime_context=context)
    return adapter, context, HeavyTransformerLeaf


def _cpu_materialize_and_bind(adapter, components):
    # Real full loader + explicit CPU setter, not FSDP or default runtime.
    meta = adapter.meta_model
    generator = torch.Generator().manual_seed(91)
    source = {key: torch.randn(meta.state_specs[name].shape, generator=generator) * .02
              for key, name in meta.base_to_model.items()}
    save_file(source, str(meta.transformer_dir / 'diffusion_pytorch_model.safetensors'))
    state = h3_loading.load_h3_cpu_state(meta, is_init_owner=True, host_weight_limit_bytes=10000000, init_seed=42)
    meta.model.load_state_dict(state.state, strict=True, assign=True)
    bundle = PreparedModelBundle(components)
    optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=1e-4)
    handle = PreparedComponentHandle(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                                     scheduler=None, accelerator=SimpleNamespace())
    adapter._bind_prepared_components(handle)
    return handle


def _input(adapter, condition, x=None):
    spec = adapter.latent_spec_for_batch(H3PromptInput(condition.prompt), device='cpu', dtype=torch.float16)
    if x is None:
        x = adapter.prepare_latents(spec, generator=H3Generators(torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)))
    return H3ModelInput(latents=x, timestep=torch.tensor(.8), audio_sigma=torch.tensor(.4), conditioning=condition,
                        guidance=None, latent_spec=spec, condition_identity=condition.condition_identity,
                        guidance_identity=(H3_GUIDANCE_ID,))


def test_h3_adapter_catalog_declaration_import_safe_and_old_contracts(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for module, relative in ((h3, 'models/implementations/h3.py'), (h3_config, 'models/h3_config.py'),
                             (h3_interface, 'models/h3_interface.py')):
        assert Path(module.__file__).resolve() == root / 'visual_rl' / relative
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in ('torch', 'diffusers', 'peft', 'transformers', 'requests'), name
        assert name != 'visual_rl.models.implementations.h3'
        return real_import(name, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, '__import__', guarded)
        spec = importlib.util.spec_from_file_location('_h3_config_import_probe', h3_config.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            module.H3Config.from_mapping(_params()).describe_contract()
        finally:
            sys.modules.pop(spec.name)
    registry = Registry('model', MODEL_CATALOG_FRAGMENT.descriptors)
    declaration = DeclarationResolver().resolve(registry, 'h3-t2va', _params())
    contract = declaration.declared_contract.model
    assert contract.tasks == (TaskKind.T2VA,)
    assert contract.output_media == (MediaKind.VIDEO, MediaKind.AUDIO)
    assert contract.latent_layouts == (LatentLayout.JOINT_AV_FLAT,) and contract.latent_ranks == (2,)
    assert contract.provides_reference_policy is False and not contract.declares_scheduler_binding
    sd3 = DeclarationResolver().resolve(registry, 'sd3', {'artifact_ref': 'sd3-model'}).declared_contract.model
    wan = DeclarationResolver().resolve(registry, 'wan-t2v', {'artifact_ref': 'wan-model'}).declared_contract.model
    assert sd3.tasks == (TaskKind.T2I,) and sd3.latent_ranks == (4,) and sd3.provides_reference_policy
    assert wan.tasks == (TaskKind.T2V,) and wan.output_media == (MediaKind.VIDEO,) and wan.latent_ranks == (5,)
    assert sd3.declares_scheduler_binding and wan.declares_scheduler_binding
    defaults = H3Config.from_mapping({k: v for k, v in _params().items() if k not in ('height', 'width', 'lora_rank', 'lora_alpha')})
    assert (defaults.height, defaults.width, defaults.frames, defaults.fps, defaults.guidance_scale) == (256, 384, 124, 24, 1)
    assert (defaults.lora_rank, defaults.lora_alpha) == (32, 64)


def test_h3_adapter_real_load_encode_initial_predict_backward_decode_and_close(tmp_path, monkeypatch):
    adapter, _, leaf = _adapter(tmp_path, monkeypatch)
    assert isinstance(adapter, ModelAdapter) and not adapter.__abstractmethods__
    manager = ComponentManager(adapter, execution_device='cpu')
    components = manager.load()  # real ComponentLoadSession register/freeze/commit
    assert components.binding('transformer').roles == (ComponentRole.INFERENCE, ComponentRole.TRAINABLE)
    assert not components.binding('transformer').managed_residency
    assert all(p.is_meta for p in adapter.meta_model.model.parameters())
    assert leaf.constructed_on_meta == [True]
    assert adapter._aux._cache is adapter._aux._decoder is None
    adapter.verify_coverage(['cat', ' cat'], ['heldout\n'])
    condition = adapter.encode(H3PromptInput('cat'))
    text = condition.embeddings
    assert text.shape == (1, 2, 5120) and text.device.type == 'cpu' and not text.requires_grad
    text.fill_(13)
    assert not torch.equal(condition.embeddings, text)
    request = _input(adapter, condition)
    with pytest.raises(ModelPortError, match='not been prepared'):
        adapter.predict(request)
    handle = _cpu_materialize_and_bind(adapter, components)
    with pytest.raises(PreparedBundleError, match='bypassed'):
        adapter.meta_model.model(hidden_states=None, audio_hidden_states=None, encoder_hidden_states=None)
    prediction = adapter.predict(request)
    prediction.validate_against(request)
    assert prediction.value.shape == request.latents.shape and prediction.value.requires_grad
    trainable = tuple(p for p in adapter.meta_model.model.parameters() if p.requires_grad)
    for stream in adapter.geometry.joint_layout.unpack(prediction.value):
        gradients = torch.autograd.grad(stream.square().mean(), trainable, retain_graph=True)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    prediction.value.square().mean().backward()
    params = dict(adapter.meta_model.model.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in params.items() if '.lora_B.' in n)
    assert all(p.grad is None for p in params.values() if not p.requires_grad)
    with torch.no_grad():
        sampled = adapter.predict(request)
    assert torch.equal(sampled.value, prediction.value.detach())
    assert leaf.calls[-2][1] is True and leaf.calls[-1][1] is False
    torch.testing.assert_close(leaf.calls[-1][0], torch.tensor([.2, .6]))
    _artifacts(adapter.meta_model.transformer_dir.parent)
    events, refs, _ = _leaf_setup(monkeypatch, adapter.geometry)
    with adapter.decode(request.latents, request.latent_spec) as media:
        assert media.video.shape == (124, 3, 32, 32) and media.audio.shape == (165600, 2)
        assert media.metadata.sample_rate == 32000 and all(ref() is None for ref in refs)
    assert events.count('load:vae') == events.count('load:audio_vae') == 1
    queued_lease = adapter.decode(request.latents, request.latent_spec)
    aux = components.binding('h3_aux').component
    cache_ref, decoder_ref = weakref.ref(aux._cache), weakref.ref(aux._decoder)
    adapter.close(); adapter.close(); manager.close(); manager.close()
    assert aux.closed and cache_ref() is None and decoder_ref() is None
    with pytest.raises(ModelPortError, match='closed'):
        adapter.predict(request)
    with pytest.raises(ModelPortError, match='closed'):
        with queued_lease:
            pytest.fail('closed adapter queued lease bypass')
    with pytest.raises(PreparedBundleError, match='closed'):
        handle.forward('transformer')
    with pytest.raises(ModelPortError, match='closed'):
        adapter.encode(H3PromptInput('cat'))


def test_h3_adapter_independent_initial_generators_and_old_rank_guard(tmp_path, monkeypatch):
    adapter, _, _ = _adapter(tmp_path, monkeypatch)
    components = adapter.load_components(ComponentLoadSession())
    spec = adapter.latent_spec_for_batch(H3PromptInput('cat'), device='cpu', dtype=torch.float16)
    before = torch.random.get_rng_state().clone()
    def draw(v=4, a=7):
        return adapter.prepare_latents(spec, generator=H3Generators(torch.Generator().manual_seed(v), torch.Generator().manual_seed(a)))
    x, same, changed = draw(), draw(), draw(a=8)
    assert x.dtype == torch.float16 and torch.equal(x, same)
    assert torch.equal(before, torch.random.get_rng_state())
    layout = adapter.geometry.joint_layout
    v, a = layout.unpack(x); v2, a2 = layout.unpack(changed)
    assert torch.equal(v, v2) and not torch.equal(a, a2)
    ev = torch.randn(v.shape, generator=torch.Generator().manual_seed(4)).half()
    ea = torch.randn(a.shape, generator=torch.Generator().manual_seed(7)).half()
    assert torch.equal(v, ev) and torch.equal(a, ea)
    assert adapter.initial_state_version == 'h3.av-normal-fp32-to-fp16.v1'
    with pytest.raises(TypeError, match='H3Generators'):
        adapter.prepare_latents(spec, generator=torch.Generator())
    shared = torch.Generator()
    with pytest.raises(ValueError, match='independent'):
        H3Generators(shared, shared)
    with pytest.raises(ModelPortError, match='at least 3'):
        ModelLatentSpec((1, 4), LatentLayout.PACKED_SEQUENCE, ('batch', 'joint'), 'cpu', torch.float16)
    for shape, axes in (((1, 2, 3), ('batch', 'rows', 'channels')), ((1, 4), ('batch', 'channel'))):
        with pytest.raises(ModelPortError, match='rank2'):
            ModelLatentSpec(shape, LatentLayout.JOINT_AV_FLAT, axes, 'cpu', torch.float16)
    adapter.close()


def test_h3_adapter_typed_sigma_condition_config_and_unavailable_capabilities(tmp_path, monkeypatch):
    adapter, context, leaf = _adapter(tmp_path, monkeypatch)
    adapter.load_components(ComponentLoadSession())
    condition = adapter.encode(H3PromptInput('cat'))
    request = _input(adapter, condition)
    for key in ('timestep', 'audio_sigma'):
        for value in (torch.tensor([.5]), torch.tensor(.5).half(), torch.tensor(float('nan')), torch.tensor(1.5),
                      torch.tensor(.5, requires_grad=True), torch.tensor(.5, device='meta')):
            with pytest.raises((ValueError, RuntimeError)):
                replace(request, **{key: value})
    with pytest.raises(TypeError, match='audio_sigma'):
        H3ModelInput(latents=request.latents, timestep=request.timestep, conditioning=condition, guidance=None,
                     latent_spec=request.latent_spec, condition_identity=condition.condition_identity, guidance_identity=request.guidance_identity)
    plain = ModelInput(request.latents, request.timestep, condition, None, request.latent_spec,
                       condition.condition_identity, request.guidance_identity)
    with pytest.raises(TypeError, match='independent-sigma'):
        adapter.predict(plain)
    for change in (dict(guidance=4.5), dict(condition_identity=('wrong',)), dict(guidance_identity=('negative',))):
        with pytest.raises(ModelPortError):
            replace(request, **change)
    foreign = H3Conditioning(condition.prompt, '0' * 64, condition.embeddings)
    with pytest.raises(ModelPortError, match='cache identity'):
        adapter.predict(replace(request, conditioning=foreign, condition_identity=foreign.condition_identity))
    with pytest.raises(ValueError, match='missing eval'):
        adapter.verify_coverage(['cat'], ['not cached'])
    with pytest.raises(ValueError, match='missing prompt'):
        adapter.encode(H3PromptInput('cat '))
    with pytest.raises(TypeError, match='H3PromptInput'):
        adapter.encode('cat')
    with pytest.raises(ModelPortError, match='one attempt'):
        adapter.load_components(ComponentLoadSession())
    for method in (lambda: adapter.scheduler_artifact_blueprint, adapter.describe_runtime_numerics,
                   lambda: adapter.predict_reference(request)):
        with pytest.raises(ModelPortError):
            method()  # fail closed, no fabricated scheduler/G3/reference receipt
    for change in (dict(sp=2), dict(training_mode='full'), dict(guidance_scale=4.5), dict(frames=121),
                   dict(artifact_ref='/some/path'), dict(max_tokens=0), dict(lora_rank=True)):
        with pytest.raises((ValueError, TypeError)):
            H3Config.from_mapping(_params() | change)
    with pytest.raises(ValueError, match='exactly'):
        h3.H3Adapter.from_config(adapter.config, runtime_context=context | {'model_loader': lambda: None})
    with pytest.raises(ValueError, match='expected cache identity'):
        h3.H3Adapter.from_config(adapter.config, runtime_context=context | {'expected_cache_identity': {}})
    assert leaf.calls == []
    adapter.close()
