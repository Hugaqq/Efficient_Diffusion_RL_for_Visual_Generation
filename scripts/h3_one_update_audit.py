"""Observation for a fresh single-update test; no optimizer or RNG operations."""
import hashlib,json,time,weakref
import torch


def weight_fingerprints(model):
    from torch.distributed.tensor import DTensor
    sums={key:hashlib.sha256() for key in ('trainable','frozen')}
    counts={key:0 for key in sums}
    for name,parameter in model.named_parameters():
        key='trainable' if parameter.requires_grad else 'frozen';digest=sums[key]
        digest.update(json.dumps([name,list(parameter.shape),str(parameter.dtype)]).encode())
        local=parameter.to_local() if isinstance(parameter,DTensor) else parameter
        flat=local.detach().reshape(-1)
        # Only local shards, no full parameter gather; CPU copy is bounded32MiB.
        n=max(1,32*1024**2//flat.element_size())
        for begin in range(0,flat.numel(),n):
            cpu=flat[begin:begin+n].to('cpu').contiguous().view(torch.uint8)
            digest.update(memoryview(cpu.numpy()))
        counts[key]+=1
    return dict(hashes={key:h.hexdigest() for key,h in sums.items()},tensor_counts=counts)


class PredictAudit:
    def __init__(self,original):
        self.original=original;self.last_prediction=None
        self.grad_forward_indices=[];self.nograd_forwards=0
    def __call__(self,*args,**kwargs):
        enabled=torch.is_grad_enabled()
        if enabled and self.last_prediction is not None and self.last_prediction() is not None:
            raise RuntimeError('previous differentiable prediction is retained across policy cells')
        value=self.original(*args,**kwargs)
        if enabled:
            index=kwargs['index']
            if not 0<=index<5:raise RuntimeError('differentiable replay outside first-five window')
            self.grad_forward_indices.append(index);self.last_prediction=weakref.ref(value)
        else:self.nograd_forwards+=1
        return value
    def batched(self,original):
        """S1: one batched rollout call is ONE no-grad transformer forward."""
        def call(*args,**kwargs):
            if torch.is_grad_enabled():raise RuntimeError('batched prediction must be no-grad rollout')
            value=original(*args,**kwargs);self.nograd_forwards+=1
            return value
        return call
    def payload(self):
        return dict(grad_forward_indices=self.grad_forward_indices,nograd_forwards=self.nograd_forwards,
                    final_prediction_released=self.last_prediction is None or self.last_prediction() is None)
