"""Two-process Gloo collective evidence, not NCCL/WORLD8 or GPU evidence."""
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from visual_rl.algorithms.optimization.h3_dual_credit import global_stream_means


def _worker(rank,path,queue):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        local=((torch.tensor(float(rank+1)),torch.tensor(float(3+2*rank))),)
        means=global_stream_means(local,process_group=dist.group.WORLD,device='cpu')
        failed=False
        try:
            bad=((torch.tensor(float('nan') if rank==1 else 1.),torch.tensor(2.)),)
            global_stream_means(bad,process_group=dist.group.WORLD,device='cpu')
        except ValueError:failed=True
        queue.put((rank,means[0].item(),means[1].item(),failed))
    finally:dist.destroy_process_group()


def test_two_real_ranks_get_same_means_and_both_reject_bad_rank():
    ctx=mp.get_context('spawn');queue=ctx.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        path=os.path.join(directory,'rendezvous')
        ps=[ctx.Process(target=_worker,args=(r,path,queue)) for r in range(2)]
        for p in ps:p.start()
        for p in ps:p.join(30)
        for p in ps:
            if p.is_alive():p.terminate();p.join();raise AssertionError('distributed hang')
            assert p.exitcode==0
        results=sorted(queue.get() for _ in range(2))
    assert results==[(0,1.5,4.0,True),(1,1.5,4.0,True)]
