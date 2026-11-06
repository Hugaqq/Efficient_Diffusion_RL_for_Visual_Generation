"""真实 WORLD8/gloo/DTensor；不替换 local_parameter 或 _local。"""
from datetime import timedelta
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, Replicate, distribute_tensor

from visual_rl.algorithms.trainer.h3_ema import H3LoRAEMA, local_parameter
from visual_rl.artifacts.checkpoint.h3_ema import save_ema, load_ema
from visual_rl.composition.config.h3_profile import H3EMAConfig


def _worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=8,
                            timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh('cpu', (8,))
        group = mesh.get_group(0)
        parameters, initial = {}, {}
        for i, rows in enumerate((2, 3, 9, 10)):
            full = torch.arange(rows*3, dtype=torch.float32).reshape(rows, 3)
            name = f'transformer_blocks.{i}.attn.to_q.lora_A.default.weight'
            parameter = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
            local, spec = local_parameter(parameter, rank=rank, process_group=group)
            # 用真实 torch.chunk 验证 ceil/offset，尾部 rank 必须是 (0,3)。
            chunks = full.chunk(8, dim=0)
            expected = chunks[rank] if rank < len(chunks) else full[:0]
            offset = sum(chunk.shape[0] for chunk in chunks[:rank])
            assert torch.equal(local, expected)
            assert spec == dict(shape=[rows,3], local_shape=list(expected.shape), offset=offset, dtype='torch.float32')
            parameters[name], initial[name] = parameter, expected.clone()
        if rank >= 3:
            assert all(parameters[n].to_local().shape == (0,3) for n in list(parameters)[:2])
        config = H3EMAConfig(enabled=True, update_interval=1)
        owner = H3LoRAEMA(lambda: parameters.items(), config=config, rank=rank, process_group=group)
        for name, p in parameters.items():
            with torch.no_grad():
                p.to_local().add_(2)
        owner.commit(1)
        for name, p in parameters.items():
            assert torch.equal(owner.shadow[name], initial[name].lerp(p.to_local().detach(), .9))
        from visual_rl.runtime.h3_training_metrics import local_metrics
        metrics = local_metrics(parameters.items(), rank=rank, process_group=group, kl=None, ema=owner, loss=-1.)
        expected_norm = sum(float(p.to_local().detach().double().square().sum()) for p in parameters.values())
        assert metrics['lora_squared_norm'] == expected_norm
        save_ema(owner, root, step=1)
        dist.barrier()  # 所有 rank 文件持久化后，才进入完整 checkpoint 的 restore。
        from scripts.launch_h3_formal import ema_attachment_budget
        actual = sum((root / 'h3-ema' / f'rank-{r}.pt').stat().st_size for r in range(8))
        assert actual <= ema_attachment_budget([(rows, 3) for rows in (2, 3, 9, 10)])
        restored = H3LoRAEMA(lambda: parameters.items(), config=config, rank=rank, process_group=group)
        load_ema(restored, root, step=1)
        live = {n: p.to_local().detach().clone() for n,p in parameters.items()}
        with pytest.raises(RuntimeError, match='evaluation failure'):
            with restored.evaluation_weights():
                for name, p in parameters.items():
                    assert torch.equal(p.to_local(), owner.shadow[name])
                raise RuntimeError('evaluation failure')
        for name, p in parameters.items():
            assert torch.equal(p.to_local(), live[name])
        owner.commit(2); restored.commit(2)
        assert all(torch.equal(owner.shadow[n], restored.shadow[n]) for n in parameters)
        parameter = next(iter(parameters.values()))
        with pytest.raises(ValueError, match='this rank'):
            local_parameter(parameter, rank=(rank+1)%8, process_group=group)
        with pytest.raises(ValueError, match='WORLD8'):
            local_parameter(parameter, rank=rank, process_group=None)
        replicated = torch.nn.Parameter(distribute_tensor(torch.ones(2,3), mesh, [Replicate()]))
        with pytest.raises(ValueError, match='Shard'):
            local_parameter(replicated, rank=rank, process_group=group)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_eight_rank_gloo_ema_geometry_empty_shards_and_restore(tmp_path):
    workers = mp.spawn(_worker, args=((tmp_path/'rendezvous').as_uri(), tmp_path), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 120
        while not workers.join(timeout=1):
            assert time.monotonic() < deadline, 'gloo EMA test exceeded bounded runtime'
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
