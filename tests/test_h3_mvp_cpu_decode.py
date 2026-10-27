"""H3 CPU decode control/math tests; only the two heavy VAE leaves are tiny."""

import json
from pathlib import Path
import weakref

import pytest
import torch

from visual_rl.models.implementations import h3_decode
from visual_rl.models.implementations.h3_decode import H3CPUDecoder
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry


def _artifacts(root):
    configs = {}
    for component, channels, name in (("vae", 24, "AutoencoderKLMiniMaxH3"),
                                       ("audio_vae", 32, "AutoencoderKLMiniMaxH3Audio")):
        directory = root / component
        directory.mkdir(parents=True, exist_ok=True)
        configs[component] = dict(_class_name=name, latents_mean=[c / 100 for c in range(channels)],
                                 latents_std=[.5 + c / 50 for c in range(channels)])
        if component == 'audio_vae':
            configs[component]['sampling_rate'] = 32000
        (directory / 'config.json').write_text(json.dumps(configs[component]))
    return configs


def _terminal(geometry):
    layout = geometry.joint_layout
    video = ((torch.arange(layout.video_numel) % 17 - 8) / 4).reshape(1, layout.video_rows, 96).half()
    video[0, 0, 0], video[0, 0, 1] = 20, -20
    audio = ((torch.arange(layout.audio_numel) % 23 - 11) / 2).reshape(1, layout.audio_rows, 32).half()
    return layout.pack(video, audio)


def _leaf_setup(monkeypatch, geometry, *, defect=None):
    events, refs, seen = [], [], {}

    def resolve(component):
        class TinyVAE(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.register_buffer('filter', torch.tensor([.1234567]))
                self.config = config
                self.hop_length = 800
                if defect == component + ':buffer':
                    self.filter = self.filter.bfloat16()
                if defect == 'audio_vae:hop':
                    self.hop_length = 400
                if defect == 'audio_vae:rate':
                    self.config['sampling_rate'] = 16000
                if defect == component + ':stats':
                    self.config['latents_mean'][0] += 1

            @classmethod
            def from_pretrained(cls, path, *, subfolder, torch_dtype, local_files_only):
                assert subfolder == component and torch_dtype == torch.float32 and local_files_only is True
                assert not torch.is_grad_enabled()
                if component == 'audio_vae':
                    assert refs and refs[0]() is None, 'video model must be released before audio load'
                events.append('load:' + component)
                model = cls(json.loads((Path(path) / component / 'config.json').read_text()))
                refs.append(weakref.ref(model))
                weakref.finalize(model, events.append, 'release:' + component)
                return model

            def decode(self, latents, *, return_dict):
                assert return_dict is False and not self.training and not torch.is_grad_enabled()
                assert not torch.is_autocast_enabled('cpu')
                assert all(not p.requires_grad and p.dtype == torch.float32 and p.device.type == 'cpu' for p in self.parameters())
                assert latents.dtype == torch.float32 and latents.device.type == 'cpu'
                events.append('decode:' + component)
                seen[component] = latents.clone()
                if defect == component + ':raise':
                    raise RuntimeError('leaf decode failure')
                if component == 'vae':
                    t = torch.arange(geometry.num_frames) % latents.shape[2]
                    h = torch.arange(geometry.height) % latents.shape[3]
                    w = torch.arange(geometry.width) % latents.shape[4]
                    output = latents[:, :3].index_select(2, t).index_select(3, h).index_select(4, w) * self.weight
                    if defect == 'vae:frames':
                        output = output[:, :, :121]
                else:
                    output = latents[:, :1].repeat_interleave(800, dim=2) * self.weight
                    if defect == 'audio_vae:length':
                        output = output[:, :, :-1]
                    if defect == 'audio_vae:stereo':
                        output = output[:1]
                if defect == component + ':finite':
                    output.flatten()[0] = float('nan')
                if defect == component + ':dtype':
                    output = output.half()
                if defect == component + ':tuple':
                    return output
                return (output,)
        return TinyVAE

    monkeypatch.setattr(h3_decode, '_resolve_vae_class', resolve)
    return events, refs, seen


def _decoder(root, geometry, **kwargs):
    media, workspace = H3CPUDecoder.estimate_bytes(geometry)
    return H3CPUDecoder(root, **(dict(max_media_bytes=media, max_decode_bytes=workspace) | kwargs))


def test_decode_unpatchify_denormalization_stereo_pixels_and_actual_durations(tmp_path, monkeypatch):
    assert Path(h3_decode.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/models/implementations/h3_decode.py'
    geometry = H3T2VAGeometry(64, 96, 124)
    configs = _artifacts(tmp_path)
    events, refs, seen = _leaf_setup(monkeypatch, geometry)
    terminal = _terminal(geometry)
    original = terminal.clone()
    decoder = _decoder(tmp_path, geometry)
    with decoder.decode_one(terminal, geometry) as media:
        assert torch.is_grad_enabled()  # no_grad is scoped to decoding, not consumer
        assert media.video.shape == (124, 3, 64, 96)
        assert media.audio.shape == (165600, 2)
        assert media.video.dtype == media.audio.dtype == torch.float32
        assert media.video.is_contiguous() and media.audio.is_contiguous()
        assert not media.video.requires_grad and not media.audio.requires_grad
        assert media.metadata.frame_count == 124 and media.metadata.sample_count == 165600
        assert media.metadata.fps == 24 and media.metadata.sample_rate == 32000
        assert media.metadata.video_duration_seconds == 124 / 24
        assert media.metadata.audio_duration_seconds == 5.175
        assert media.metadata.video_duration_seconds != media.metadata.audio_duration_seconds
        # Independent element-wise unpatchify reference, NOT inverse production reshape.
        video_rows, audio_rows = geometry.joint_layout.unpack(terminal)
        expected = torch.empty(1, 24, 37, 4, 6)
        for c in range(24):
            for t in range(37):
                for h in range(4):
                    for w in range(6):
                        row = t * 6 + (h // 2) * 3 + w // 2
                        feature = c * 4 + (h % 2) * 2 + w % 2
                        expected[0, c, t, h, w] = (video_rows[0, row, feature].float()
                            * configs['vae']['latents_std'][c] + configs['vae']['latents_mean'][c])
        assert torch.equal(seen['vae'], expected)
        expected_audio = torch.empty(2, 32, 207)
        for stereo in range(2):
            for c in range(32):
                expected_audio[stereo, c] = (audio_rows[0, stereo*207:(stereo+1)*207, c].float()
                    * configs['audio_vae']['latents_std'][c] + configs['audio_vae']['latents_mean'][c])
        assert torch.equal(seen['audio_vae'], expected_audio)
        for t, h, w in ((0, 0, 0), (0, 0, 1), (123, 63, 95), (37, 4, 6)):
            raw = expected[0, :3, t % 37, h % 4, w % 6]
            pixel = (raw * torch.tensor([.229, .224, .225]) + torch.tensor([.485, .456, .406])).clamp(0, 1)
            torch.testing.assert_close(media.video[t, :, h, w], pixel, rtol=0, atol=0)
        assert media.video[0, 0, 0, 0] == 1 and media.video[0, 0, 0, 1] == 0
        for sample in (0, 799, 800, 165599):
            assert torch.equal(media.audio[sample], expected_audio[:, 0, sample // 800])
        assert media.audio.abs().max() > 1  # no extra clipping of decoder amplitude
        assert torch.equal(terminal, original)
        assert events == ['load:vae', 'decode:vae', 'release:vae', 'load:audio_vae', 'decode:audio_vae', 'release:audio_vae']
        assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match='expired'):
        _ = media.video


def test_decode_one_lease_reentry_release_and_consumer_exception(tmp_path, monkeypatch):
    geometry = H3T2VAGeometry(32, 32, 124)
    _artifacts(tmp_path)
    events, _, _ = _leaf_setup(monkeypatch, geometry)
    decoder = _decoder(tmp_path, geometry)
    lease = decoder.decode_one(_terminal(geometry), geometry)
    with pytest.raises(RuntimeError, match='consumer failure'):
        with lease as media:
            vref, aref = weakref.ref(media.video), weakref.ref(media.audio)
            with pytest.raises(RuntimeError, match='reentered'):
                lease.__enter__()
            with pytest.raises(RuntimeError, match='one media lease'):
                with decoder.decode_one(_terminal(geometry), geometry):
                    pytest.fail('second lease accepted')
            assert events.count('load:vae') == 1
            raise RuntimeError('consumer failure')
    assert vref() is None and aref() is None
    with pytest.raises(RuntimeError, match='expired'):
        _ = media.audio
    with pytest.raises(RuntimeError, match='reentered'):
        lease.__enter__()
    with decoder.decode_one(_terminal(geometry), geometry) as next_media:
        assert next_media.video.shape[0] == 124


def test_decode_budgets_and_terminal_rejections_before_any_load(tmp_path, monkeypatch):
    geometry = H3T2VAGeometry(32, 32, 124)
    _artifacts(tmp_path)
    events, _, _ = _leaf_setup(monkeypatch, geometry)
    terminal = _terminal(geometry)
    media_bytes, workspace = H3CPUDecoder.estimate_bytes(geometry)
    assert media_bytes == 4 * (124 * 3 * 32 * 32 + 165600 * 2)
    for change in (dict(max_media_bytes=media_bytes-1), dict(max_decode_bytes=workspace-1)):
        with pytest.raises(ValueError, match='budget exceeded'):
            with _decoder(tmp_path, geometry, **change).decode_one(terminal, geometry):
                pytest.fail('budget not enforced')
    for change in (dict(max_media_bytes=0), dict(max_decode_bytes=True)):
        with pytest.raises(ValueError, match='positive integer'):
            _decoder(tmp_path, geometry, **change)
    decoder = _decoder(tmp_path, geometry)
    for bad in (terminal.float(), terminal[:, :-1], terminal.expand(2, -1), terminal.to('meta'),
                terminal.clone().requires_grad_(), torch.full_like(terminal, float('nan'))):
        with pytest.raises(ValueError):
            with decoder.decode_one(bad, geometry):
                pytest.fail('malformed terminal accepted')
    with pytest.raises(TypeError, match='geometry'):
        with decoder.decode_one(terminal, None):
            pytest.fail('malformed geometry accepted')
    assert events == []


def test_decode_invalid_configs_and_artifact_paths_fail_before_load(tmp_path, monkeypatch):
    geometry = H3T2VAGeometry(32, 32, 124)
    events, _, _ = _leaf_setup(monkeypatch, geometry)
    changes = [('_class_name', 'Wan'), ('latents_mean', None), ('latents_mean', [0.] * 23),
               ('latents_std', [0.] * 24), ('latents_std', [-1.] * 24), ('latents_mean', [float('nan')] * 24)]
    for i, (key, value) in enumerate(changes):
        root = tmp_path / str(i)
        config = _artifacts(root)
        config['vae'][key] = value
        (root / 'vae/config.json').write_text(json.dumps(config['vae']))
        with pytest.raises(ValueError):
            with _decoder(root, geometry).decode_one(_terminal(geometry), geometry):
                pytest.fail('invalid config accepted')
    root = tmp_path / 'audio'
    configs = _artifacts(root)
    del configs['audio_vae']['sampling_rate']
    (root / 'audio_vae/config.json').write_text(json.dumps(configs['audio_vae']))
    with pytest.raises(ValueError, match='sampling_rate'):
        with _decoder(root, geometry).decode_one(_terminal(geometry), geometry):
            pytest.fail('missing sampling rate accepted')
    (root / 'audio_vae/config.json').unlink()
    with pytest.raises(ValueError, match='missing VAE config'):
        with _decoder(root, geometry).decode_one(_terminal(geometry), geometry):
            pytest.fail('missing config accepted')
    _artifacts(root)
    outside = tmp_path / 'outside-config.json'
    (root / 'audio_vae/config.json').rename(outside)
    (root / 'audio_vae/config.json').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink/path escape'):
        with _decoder(root, geometry).decode_one(_terminal(geometry), geometry):
            pytest.fail('symlink accepted')
    root = tmp_path / 'index'
    _artifacts(root)
    (root / 'vae/model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight': '../outside'})))
    with pytest.raises(ValueError, match='shard missing or path escape'):
        with _decoder(root, geometry).decode_one(_terminal(geometry), geometry):
            pytest.fail('index escape accepted')
    with pytest.raises(ValueError, match='local artifact'):
        _decoder(tmp_path / 'missing-repo-id', geometry)
    assert not events


def test_decode_leaf_failures_drop_models_and_never_publish_half_media(tmp_path, monkeypatch):
    geometry = H3T2VAGeometry(32, 32, 124)
    _artifacts(tmp_path)
    terminal = _terminal(geometry)
    for defect in ('vae:frames', 'vae:finite', 'vae:dtype', 'vae:tuple', 'vae:buffer', 'vae:stats',
                   'audio_vae:hop', 'audio_vae:rate', 'audio_vae:length', 'audio_vae:stereo',
                   'audio_vae:finite', 'audio_vae:dtype', 'audio_vae:raise', 'audio_vae:buffer'):
        events, refs, _ = _leaf_setup(monkeypatch, geometry, defect=defect)
        decoder = _decoder(tmp_path, geometry)
        with pytest.raises((ValueError, RuntimeError)):
            with decoder.decode_one(terminal, geometry):
                pytest.fail('partial or invalid AV result published')
        assert all(ref() is None for ref in refs)
        assert not decoder._lock.locked()
        if defect.startswith('vae:'):
            assert 'load:audio_vae' not in events
    # Finite FP32 statistics can still overflow with finite FP16 state.
    configs = _artifacts(tmp_path)
    configs['vae']['latents_std'] = [3e38] * 24
    (tmp_path / 'vae/config.json').write_text(json.dumps(configs['vae']))
    events, refs, _ = _leaf_setup(monkeypatch, geometry)
    with pytest.raises(ValueError, match='denormalization must be finite'):
        with _decoder(tmp_path, geometry).decode_one(terminal, geometry):
            pytest.fail('overflow accepted')
    assert 'decode:vae' not in events and all(ref() is None for ref in refs)


def test_decode_lazy_public_dependency_failure_is_explicit(monkeypatch):
    real_import = h3_decode.importlib.import_module
    for component in ('vae', 'audio_vae'):
        def missing(name, *args, **kwargs):
            if name.startswith('diffusers.models.autoencoders.autoencoder_kl_minimax_h3'):
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(h3_decode.importlib, 'import_module', missing)
        with pytest.raises(ImportError, match='requires optional.*no reference/fake fallback'):
            h3_decode._resolve_vae_class(component)
