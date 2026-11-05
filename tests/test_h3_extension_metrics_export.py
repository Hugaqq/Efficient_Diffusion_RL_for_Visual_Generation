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
                                  kl=dict(kl_video=.3, kl_audio=.1, kl=.2), ema=ema))
    result = world_metrics(rows, step=0, ema_config=owners[0][1].config)
    assert set(result) == metric_keys(ema=True)
    assert result['ema_live_distance'] == math.sqrt(8*6)
    assert result['lora_norm'] == math.sqrt(sum(float(p.double().square().sum()) for p,_ in owners))
    assert result['ema_committed_steps'] == 1 and result['ema_updates'] == 0
    for bad in (float('nan'), float('inf'), -.1, True):
        broken = [dict(row) for row in rows]; broken[3]['kl'] = bad
        with pytest.raises(ValueError):
            world_metrics(broken, step=0, ema_config=owners[0][1].config)
    broken = [dict(row) for row in rows]; broken[3]['ema_updates'] = 2
    with pytest.raises(ValueError, match='counter'):
        world_metrics(broken, step=0, ema_config=owners[0][1].config)


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
