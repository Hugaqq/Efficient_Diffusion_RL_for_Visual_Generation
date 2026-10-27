"""Real local cache I/O; only the heavy transformer leaf is tiny. CPU only."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from visual_rl.models.implementations import h3_text_cache
from visual_rl.models.implementations.h3_text_cache import H3TextCache
from visual_rl.models.implementations.h3_forward import h3_t2va_forward
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _key(prompt):
    return _sha(prompt.encode("utf-8"))


def _identity(dtype="float32"):
    return dict(
        **{name + "_sha256": _sha(("test artifact: " + name).encode()) for name in
           ("model", "encoder", "tokenizer", "processor", "template", "truncation")},
        target_layer=50, normalized=False, add_special_tokens=False, feature_dim=5120, dtype=dtype,
    )


def _metadata(identity):
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {"h3_text_identity_sha256": _sha(raw)}


def _write_index(root, data):
    (root / "index.json").write_text(json.dumps(data), encoding="utf-8")


def _fixture(root, dtype="float32"):
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity(dtype)
    tensors = {}
    entries = {}
    for i, (prompt, length) in enumerate((("cat", 2), (" cat", 3), ("heldout\n", 1))):
        tensor = torch.linspace(-1, 1, length * 5120).reshape(length, 5120).to(getattr(torch, dtype))
        tensor[0] = 0  # A legal zero row must NOT be guessed to be padding.
        shard = f"embeddings_{i // 2:05d}.safetensors"
        tensors.setdefault(shard, {})[_key(prompt)] = tensor
        entries[_key(prompt)] = dict(shard=shard, tensor=_key(prompt), tokens=length)
    data = dict(schema_version="visual_rl.h3-text-cache.v1", identity=identity, shards={}, entries=entries)
    for shard, values in tensors.items():
        save_file(values, str(root / shard), metadata=_metadata(identity))
        data['shards'][shard] = _sha((root / shard).read_bytes())
    _write_index(root, data)
    return data, tensors


def _reader(root, dtype="float32", **kwargs):
    return H3TextCache(root, **(dict(expected_identity=_identity(dtype), max_tokens=4,
                                   max_embedding_bytes=4 * 5120 * 4) | kwargs))


def _replace_shard(root, data, tensors, name, *, metadata=None):
    save_file(tensors[name], str(root / name), metadata=_metadata(data['identity']) if metadata is None else metadata)
    data['shards'][name] = _sha((root / name).read_bytes())
    _write_index(root, data)


def test_text_cache_import_coverage_dtypes_ownership_and_exact_prompts(tmp_path):
    assert Path(h3_text_cache.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/models/implementations/h3_text_cache.py'
    for dtype in ('float32', 'bfloat16', 'float16'):
        root = tmp_path / dtype
        data, tensors = _fixture(root, dtype)
        expected = _identity(dtype)
        cache = _reader(root, dtype, expected_identity=expected)
        identity = cache.identity_sha256
        assert identity == _sha((root / 'index.json').read_bytes())
        assert cache.embedding_identity_sha256 == _metadata(expected)['h3_text_identity_sha256']
        cache.verify_coverage(['cat', 'cat', ' cat'], ['heldout\n'])
        expected['encoder_sha256'] = '0' * 64  # reader owns its identity, not this dict
        first, second = cache.get('cat'), cache.get('cat')
        assert first.shape == (1, 2, 5120) and first.dtype == getattr(torch, dtype)
        assert first.device.type == 'cpu' and not first.requires_grad and first.grad_fn is None
        assert first.data_ptr() != second.data_ptr()
        assert torch.equal(first[0], tensors['embeddings_00000.safetensors'][_key('cat')])
        assert not first[:, 0].any()  # zero token preserved, not trimmed
        first.fill_(99)
        assert torch.equal(cache.get('cat'), second)
        assert cache.get(' cat').shape == (1, 3, 5120)
        assert cache.get('heldout\n').shape == (1, 1, 5120)  # all-zero finite input is legal
        assert cache.identity_sha256 == identity
        for train, heldout, label in ((['dog'], ['heldout\n'], 'train'), (['cat'], ['heldout'], 'eval')):
            with pytest.raises(ValueError, match=f'missing {label}'):
                cache.verify_coverage(train, heldout)
        for prompt in ('cat ', 'heldout', 'CAT'):
            with pytest.raises(ValueError, match='missing prompt'):
                cache.get(prompt)
        with pytest.raises(TypeError):
            cache.get(7)
        with pytest.raises(TypeError):
            cache.verify_coverage('cat', [])
        # Outputs do not alias file storage, and the reader retains no handles.
        (root / 'embeddings_00000.safetensors').unlink()
        assert torch.isfinite(second).all()
        with pytest.raises(ValueError, match='missing'):
            cache.get('cat')


def test_text_cache_real_forward_consumer(tmp_path):
    _fixture(tmp_path)
    cache = _reader(tmp_path)
    cache.verify_coverage(['cat', ' cat'], ['heldout\n'])

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(.2, .8, 5120))
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            text = kwargs['encoder_hidden_states']
            assert text.shape == (1, 2, 5120)
            condition = (text * self.weight).mean()
            return kwargs['hidden_states'].float() + condition, kwargs['audio_hidden_states'].float() + 2 * condition

    geometry = H3T2VAGeometry(32, 32, 124)
    text = cache.get('cat')
    leaf = TinyTransformer()
    result = h3_t2va_forward(leaf, x=torch.ones(1, geometry.joint_layout.packed_width, dtype=torch.float16),
                             geometry=geometry, text_embeddings=text, video_sigma=torch.tensor(.8),
                             audio_sigma=torch.tensor(.4))
    assert len(leaf.calls) == 1 and leaf.calls[0]['encoder_hidden_states'] is text
    assert leaf.calls[0]['text_indices'].numel() == 2
    condition = (text * leaf.weight).mean()
    video, audio = geometry.joint_layout.unpack(result)
    torch.testing.assert_close(video, (-1-condition).expand_as(video))
    torch.testing.assert_close(audio, (-1-2*condition).expand_as(audio))
    result.square().mean().backward()
    assert leaf.weight.grad is not None and leaf.weight.grad.abs().sum() > 0
    assert text.grad is None


def test_text_cache_identity_schema_and_expected_identity_rejections(tmp_path):
    mismatches = [dict(target_layer=49), dict(normalized=True), dict(add_special_tokens=True),
                  dict(feature_dim=4096), dict(dtype='float16')]
    mismatches += [{name: 'a' * 64} for name in _identity() if name.endswith('_sha256')]
    for i, change in enumerate(mismatches):
        root = tmp_path / str(i)
        data, _ = _fixture(root)
        data['identity'].update(change)
        _write_index(root, data)
        with pytest.raises(ValueError, match='identity mismatch'):
            _reader(root)
    root = tmp_path / 'expected'
    _fixture(root)
    for change in (dict(target_layer=True), dict(target_layer=49), dict(normalized=True),
                   dict(add_special_tokens=True), dict(feature_dim=2), dict(dtype='float64'),
                   dict(encoder_sha256='unversioned/path')):
        with pytest.raises(ValueError, match='identity|identities'):
            _reader(root, expected_identity=_identity() | change)
    with pytest.raises(ValueError, match='complete'):
        _reader(root, expected_identity={})
    _write_index(root, dict(schema_version='1.0', extractor='minimax_h3_text_embed', target_layer=50,
                           feature_dim=5120, entries={_key('cat')[:16]: 'embeddings_00000.safetensors'}))
    with pytest.raises(ValueError, match='schema.*raw UniRL'):
        _reader(root)
    with pytest.raises(ValueError, match='local directory'):
        _reader(tmp_path / 'no-repo-fallback')


def test_text_cache_duplicate_keys_mapping_and_metadata_rejections(tmp_path):
    root = tmp_path / 'index'
    data, _ = _fixture(root)
    original = (root / 'index.json').read_text()
    (root / 'index.json').write_text('{"schema_version":"x",' + original[1:])
    with pytest.raises(ValueError, match='duplicate JSON key'):
        _reader(root)
    _write_index(root, data)
    data['entries'][_key('duplicate')] = data['entries'][_key('cat')].copy()
    _write_index(root, data)
    with pytest.raises(ValueError, match='duplicate shard/tensor'):
        _reader(root)
    for i, defect in enumerate(('metadata', 'missing', 'extra', 'tokens', 'dtype', 'nan', 'inf')):
        root = tmp_path / defect
        data, tensors = _fixture(root)
        name = 'embeddings_00000.safetensors'
        key = _key('cat')
        if defect == 'missing':
            del tensors[name][key]
        elif defect == 'extra':
            tensors[name]['unknown'] = torch.ones(1, 5120)
        elif defect == 'tokens':
            tensors[name][key] = torch.ones(1, 5120)
        elif defect == 'dtype':
            tensors[name][key] = tensors[name][key].half()
        elif defect in ('nan', 'inf'):
            tensors[name][key][0, 0] = float(defect)
        _replace_shard(root, data, tensors, name,
                       metadata={'h3_text_identity_sha256': 'a' * 64} if defect == 'metadata' else None)
        if defect in ('nan', 'inf'):
            cache = _reader(root)
            with pytest.raises(ValueError, match='finite'):
                cache.get('cat')
        else:
            with pytest.raises(ValueError, match='metadata|keys|shape/dtype'):
                _reader(root)
    root = tmp_path / 'header'
    data, _ = _fixture(root)
    file = root / 'embeddings_00000.safetensors'
    raw = file.read_bytes()
    size = int.from_bytes(raw[:8], 'little')
    header = raw[8:8+size].rstrip()
    # Add a duplicate __metadata__ field; this is real corrupt safetensors data.
    header = b'{"__metadata__":{},' + header[1:]
    file.write_bytes(len(header).to_bytes(8, 'little') + header + raw[8+size:])
    with pytest.raises(ValueError, match='duplicate JSON key'):
        _reader(root)


def test_text_cache_bounds_precede_payload_hash_or_finite_read(tmp_path):
    data, tensors = _fixture(tmp_path)
    name = 'embeddings_00000.safetensors'
    # Corrupt payload and leave stale hash: bound admission must fail first,
    # without needing tensor materialization or even a payload content hash.
    tensors[name][_key('cat')][0, 0] = float('nan')
    save_file(tensors[name], str(tmp_path / name), metadata=_metadata(data['identity']))
    for limits, match in ((dict(max_tokens=1), 'token'), (dict(max_embedding_bytes=1), 'byte'),
                          (dict(max_tokens=True), 'positive integer'), (dict(max_embedding_bytes=0), 'positive integer')):
        with pytest.raises(ValueError, match=match):
            _reader(tmp_path, **limits)
    # Header shape/dtype admission also precedes payload/hash access.
    for shape in ((2, 5119), (1, 2, 5120), (0, 5120)):
        tensors[name][_key('cat')] = torch.zeros(shape)
        save_file(tensors[name], str(tmp_path / name), metadata=_metadata(data['identity']))
        with pytest.raises(ValueError, match='header shape/dtype'):
            _reader(tmp_path)


def test_text_cache_paths_missing_shards_and_content_drift(tmp_path):
    for i, path in enumerate(('../outside.safetensors', '/tmp/outside.safetensors', 'sub/x.safetensors')):
        root = tmp_path / str(i)
        data, _ = _fixture(root)
        old = 'embeddings_00000.safetensors'
        data['shards'][path] = data['shards'].pop(old)
        for entry in data['entries'].values():
            if entry['shard'] == old:
                entry['shard'] = path
        _write_index(root, data)
        with pytest.raises(ValueError, match='path escape'):
            _reader(root)
    root = tmp_path / 'symlink'
    _fixture(root)
    shard = root / 'embeddings_00000.safetensors'
    outside = tmp_path / 'outside.safetensors'
    shard.rename(outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink/path escape'):
        _reader(root)
    shard.unlink()
    with pytest.raises(ValueError, match='missing'):
        _reader(root)
    root = tmp_path / 'drift'
    _fixture(root)
    cache = _reader(root)
    shard = root / 'embeddings_00000.safetensors'
    stat = shard.stat()
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 1
    shard.write_bytes(raw)
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert shard.stat().st_size == stat.st_size and shard.stat().st_mtime_ns == stat.st_mtime_ns
    for call in (lambda: cache.get('cat'), lambda: cache.verify_coverage(['cat'], ['heldout\n'])):
        with pytest.raises(ValueError, match='content/hash changed'):
            call()
    root = tmp_path / 'index_drift'
    _fixture(root)
    cache = _reader(root)
    with (root / 'index.json').open('a') as stream:
        stream.write(' ')
    for call in (lambda: cache.get('cat'), lambda: cache.verify_coverage(['cat'], ['heldout\n'])):
        with pytest.raises(ValueError, match='index changed'):
            call()
