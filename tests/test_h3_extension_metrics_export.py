from dataclasses import asdict, replace
import math

import pytest
import torch

from visual_rl.runtime.h3_training_metrics import local_metrics, world_metrics, metric_keys
from visual_rl.composition.config.h3_profile import H3EMAConfig
from tests.test_h3_ema import make_ema


def test_world_diagnostics_shard_sum_not_mean(monkeypatch):
    owners = [make_ema(monkeypatch, rank=r, config=H3EMAConfig(enabled=True, update_interval=2)) for r in range(8)]
    rows = []
    for p, ema in owners:
        p.data.add_(1); ema.commit(1)
        rows.append(local_metrics(ema.named_parameters(), rank=ema.rank, process_group=None,
                                  kl=dict(kl_video=.3, kl_audio=.1, kl=.2, policy_loss=-.5, kl_loss=.01), ema=ema, loss=-.49))
    result = world_metrics(rows, step=0, ema_config=owners[0][1].config)
    assert set(result) == metric_keys(ema=True)
    assert result['ema_live_distance'] == math.sqrt(8*6)
    assert result['lora_norm'] == math.sqrt(sum(float(p.double().square().sum()) for p,_ in owners))
    assert result['ema_committed_steps'] == 1 and result['ema_updates'] == 0
    assert result['policy_loss'] == -.5 and result['kl_loss'] == .01
    disabled = local_metrics(owners[0][1].named_parameters(), rank=0, process_group=None, kl=None, ema=None, loss=-.7)
    assert disabled['policy_loss'] == -.7 and disabled['kl_loss'] == disabled['kl'] == 0.
    for bad in (float('nan'), float('inf'), -.1, True):
        broken = [dict(row) for row in rows]; broken[3]['kl'] = bad
        with pytest.raises(ValueError):
            world_metrics(broken, step=0, ema_config=owners[0][1].config)
    broken = [dict(row) for row in rows]; broken[3]['ema_updates'] = 2
    with pytest.raises(ValueError, match='counter'):
        world_metrics(broken, step=0, ema_config=owners[0][1].config)


def test_packed_diagnostics_exact_legacy_numbers_single_inventory_and_copy(monkeypatch):
    from visual_rl.algorithms.trainer import h3_ema
    from tests.test_h3_ema import cpu_local
    tensors = [torch.empty(0,3), torch.arange(6).reshape(2,3).float(),
               torch.tensor([[1e10, 1e-10, -2e10, -3., 4., 5., -6.]])]
    parameters = [(f'transformer_blocks.{i}.attn.to_q.lora_A.default.weight', torch.nn.Parameter(t))
                  for i,t in enumerate(tensors)]
    monkeypatch.setattr(h3_ema, 'local_parameter', cpu_local)
    ema = h3_ema.H3LoRAEMA(lambda: parameters, config=H3EMAConfig(enabled=True), rank=0, process_group=None)
    for _,p in parameters:
        p.data.add_(.25)
    expected_norm = expected_distance = 0.
    for name,p in parameters:
        value = p.detach().to(device='cpu', dtype=torch.float64)
        expected_norm += float(value.square().sum())
        expected_distance += float((value-ema.shadow[name].double()).square().sum())
    counts = dict(inventory=0, copy=0)
    def counted_local(*args, **kwargs):
        counts['inventory'] += 1
        return cpu_local(*args, **kwargs)
    to = torch.Tensor.to
    def counted_to(self, *args, **kwargs):
        if kwargs.get('device') == 'cpu' and kwargs.get('dtype') == torch.float64:
            counts['copy'] += 1
        return to(self, *args, **kwargs)
    monkeypatch.setattr(h3_ema, 'local_parameter', counted_local)
    monkeypatch.setattr(torch.Tensor, 'to', counted_to)
    for owner in (ema, None):
        counts.update(inventory=0, copy=0)
        result = local_metrics(parameters, rank=0, process_group=None, kl=None, ema=owner, loss=-1.)
        assert result['lora_squared_norm'] == expected_norm
        if owner is not None:
            assert result['ema_live_squared_distance'] == expected_distance
        assert counts == dict(inventory=len(parameters), copy=1)
    parameters[-1][1].data.fill_(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        local_metrics(parameters, rank=0, process_group=None, kl=None, ema=ema, loss=-1.)


def test_complete_cpu_checkpoint_ema_export_and_conversion(tmp_path):
    import torch.distributed.checkpoint as dcp
    from scripts.convert_h3_lora_to_comfyui import TARGETS, load_dcp_lora, main, load_training_file, normalize_training, convert
    from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager
    from visual_rl.algorithms.trainer.h3_ema import EMA_STATE_SCHEMA
    from tests.test_checkpoint_coordinator import _contract
    from tests.test_h3_mvp_cpu_checkpoint_state import _progress
    contract = replace(_contract(world_size=8), ema_state_schema=EMA_STATE_SCHEMA)
    progress = replace(_progress(), execution_transform_plan_id=contract.execution_transform_plan_id)
    manager = AtomicCheckpointManager(tmp_path/'checkpoints')
    config = H3EMAConfig(enabled=True, update_interval=2)
    model = {f'transformer_blocks.0.{target}.lora_{side}.default.weight': torch.arange(8).reshape(2,4).float()
             if side == 'A' else torch.arange(8).reshape(4,2).float()
             for target in TARGETS for side in ('A','B')}
    def writer(path):
        dcp.save({'model':model}, checkpoint_id=path/'h3-state'/'dcp', no_dist=True)
        root = path/'h3-ema'; root.mkdir()
        for rank in range(8):
            specs, shadow = {}, {}
            for name,value in model.items():
                chunk = math.ceil(value.shape[0]/8); offset = min(rank*chunk, value.shape[0])
                local = value[offset:offset+chunk].clone() + 1
                specs[name] = dict(shape=list(value.shape), local_shape=list(local.shape), offset=offset, dtype='torch.float32')
                shadow[name] = local
            torch.save(dict(schema=EMA_STATE_SCHEMA, rank=rank, world_size=8, config=asdict(config),
                committed_steps=2, updates=1, specs=specs, shadow=shadow), root/f'rank-{rank}.pt')
    checkpoint = manager.commit(2, contract, writer, progress=progress).path
    live,_ = load_dcp_lora(checkpoint)
    ema,_ = load_dcp_lora(checkpoint, weights='ema')
    assert set(live) == set(ema) == set(model)
    assert all(torch.equal(live[n], model[n]) and torch.equal(ema[n], model[n]+1) for n in live)
    export = tmp_path/'ema.safetensors'; comfy = tmp_path/'comfy.safetensors'
    assert main(['export','--checkpoint',str(checkpoint),'--output',str(export),'--weights','ema']) == 0
    state,meta = load_training_file(str(export)); assert meta['weights'] == 'ema'
    assert main(['convert','--input',str(export),'--output',str(comfy)]) == 0
    converted,meta = load_training_file(str(comfy)); assert meta['weights'] == 'ema'
    blocks, rank = normalize_training(ema)
    expected = convert(blocks,rank,64)
    assert all(torch.equal(expected[n],converted[n]) for n in expected)
    with pytest.raises(ValueError, match='view'):
        main(['convert','--input',str(export),'--output',str(comfy),'--weights','live'])
    # Complete-tree verification precedes pickle/tensor reads.
    (checkpoint/'h3-ema'/'rank-7.pt').unlink()
    with pytest.raises(Exception):
        load_dcp_lora(checkpoint, weights='ema')


def test_live_export_matches_baseline_in_independent_interpreters(tmp_path):
    import json
    import struct
    import subprocess
    import sys
    import tarfile
    from pathlib import Path
    import torch.distributed.checkpoint as dcp
    from scripts.convert_h3_lora_to_comfyui import TARGETS, load_training_file

    root = Path(__file__).resolve().parents[1]
    archive = tmp_path/'baseline.tar'
    with archive.open('wb') as output:
        subprocess.run(['git', 'archive', '7c85751'], cwd=root, stdout=output, check=True)
    baseline = tmp_path/'baseline'
    baseline.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(baseline, filter='data')
    checkpoint = tmp_path/'checkpoint'
    model = {f'transformer_blocks.0.{target}.lora_{side}.default.weight':
             torch.arange(8).reshape((2, 4) if side == 'A' else (4, 2)).float()
             for target in TARGETS for side in ('A', 'B')}
    dcp.save({'model': model}, checkpoint_id=checkpoint, no_dist=True)
    def header(path):
        raw = path.read_bytes()
        size = struct.unpack('<Q', raw[:8])[0]
        return json.loads(raw[8:8+size])

    for mode in ('export', 'convert', 'convert-input'):
        outputs = [tmp_path/f'{mode}-{name}.safetensors' for name in ('old', 'old-repeat', 'current')]
        arguments = ['--input', str(tmp_path/'export-old.safetensors')] if mode == 'convert-input' else [
            '--checkpoint', str(checkpoint)]
        for source, output in zip((baseline, baseline, root), outputs):
            subprocess.run([sys.executable, str(source/'scripts/convert_h3_lora_to_comfyui.py'),
                            mode.split('-')[0], *arguments, '--output', str(output)],
                           cwd=source, check=True, timeout=60)
        old_state, old_meta = load_training_file(str(outputs[0]))
        assert 'weights' not in old_meta
        for output in outputs[1:]:
            state, meta = load_training_file(str(output))
            # 基线自身 metadata 键的序列化顺序不稳定；比较完整 header，而非文件 SHA。
            assert header(outputs[0]) == header(output)
            assert old_meta == meta
            assert old_state.keys() == state.keys()
            assert all(old_state[key].dtype == state[key].dtype and torch.equal(old_state[key], state[key])
                       for key in old_state)
