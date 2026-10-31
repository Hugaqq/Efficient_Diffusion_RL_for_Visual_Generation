"""Bounded all-rank wait for LOCAL CPU/file work; PG operations stay on main.

This is not suitable for work that itself calls distributed/CUDA APIs.
"""
from __future__ import annotations
import json
import threading
import time
import torch
import torch.distributed as dist


def file_phase(action, *, rank, group, label, timeout_s=3600., interval_s=1.):
    if not 0 <= rank < 8 or timeout_s <= 0 or interval_s <= 0:
        raise ValueError('invalid file phase bounds/rank')
    result = error = None
    done = threading.Event()
    def work():
        nonlocal result, error
        try: result = action()
        except BaseException as exc: error = exc
        finally: done.set()
    worker = threading.Thread(target=work, name='h3-file-'+label.replace('/','-'), daemon=True)
    started=time.monotonic(); worker.start(); tick=0
    signal=torch.zeros((),dtype=torch.int32,device=torch.device('cuda',rank))
    while True:
        elapsed=time.monotonic()-started
        # SUM complete peers, and a value >= 16 if any local timeout occurs.
        signal.fill_(1 if done.is_set() else (16 if elapsed>=timeout_s else 0))
        dist.all_reduce(signal,op=dist.ReduceOp.SUM,group=group)
        state=int(signal.item())
        if rank==0 and tick%30==0:
            print(json.dumps(dict(event='h3_file_phase_wait',phase=label,seconds=elapsed,
                                  completion_sum=state,timeout_seconds=timeout_s)),flush=True)
        if state>=16:
            raise TimeoutError('file phase exceeded bound: '+label)
        if state==8:break
        if not 0<=state<8:raise RuntimeError('invalid file phase heartbeat')
        time.sleep(interval_s);tick+=1
    worker.join()
    if error is not None:raise error
    return result
