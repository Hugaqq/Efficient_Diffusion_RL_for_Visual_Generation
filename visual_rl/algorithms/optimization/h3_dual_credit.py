"""Experimental dual-SDE per-stream credit primitives, not a trainer binding."""
from __future__ import annotations

from types import SimpleNamespace
import math
import torch
import torch.distributed as dist

from visual_rl.algorithms.optimization.objective import ClippedSurrogateObjective, PolicyLossInputs


def global_stream_means(coefficients, *, process_group, device):
    """Exactly one collective even on invalid local input; all ranks fail closed.

    Counts are reduced separately, so uneven local row counts cannot bias the
    two component means. The caller must gate failures before this call and
    use a live process group with a backend compatible with ``device``.
    """
    device = torch.device(device)
    invalid = False
    total = torch.zeros(4, dtype=torch.float64, device=device)
    try:
        if type(coefficients) is not tuple or not coefficients:
            raise ValueError('nonempty coefficient rows required')
        for row in coefficients:
            if type(row) is not tuple or len(row) != 2:
                raise ValueError('each row must contain two coefficients')
            for index, value in enumerate(row):
                if not isinstance(value, torch.Tensor) or value.numel()!=1 or value.requires_grad or not bool(torch.isfinite(value).all()) or value.item() <= 0:
                    raise ValueError('finite positive detached coefficients required')
                total[index] += value.detach().to(device=device,dtype=torch.float64).reshape(())
        total[2] = len(coefficients)
    except (ValueError,TypeError,RuntimeError):
        invalid = True
    if invalid:
        total.zero_();total[3] = 1
    dist.all_reduce(total,op=dist.ReduceOp.SUM,group=process_group)
    if total[3].item() != 0 or total[2].item() <= 0 or not bool(torch.isfinite(total).all()):
        raise ValueError('invalid coefficient table on at least one rank')
    means=tuple((total[i]/total[2]).to(torch.float32).detach() for i in (0,1))
    if any(not bool(torch.isfinite(v)) or v.item()<=0 for v in means):
        raise ValueError('invalid global coefficient mean')
    return means


def policy_loss(*, dynamics, record, prediction, old_joint, advantage, mode, world_means, clip_range, scale):
    """A single globally-scaled cell, with independent Flash stream gradients."""
    if mode not in ('flow','flash'):
        raise ValueError('explicit flow or flash mode required')
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:
        raise ValueError('explicit positive global loss scale required')
    if not isinstance(advantage,torch.Tensor) or advantage.numel()!=1 or advantage.requires_grad or not bool(torch.isfinite(advantage).all()):
        raise ValueError('detached finite scalar advantage required')
    if not isinstance(old_joint,torch.Tensor) or old_joint.numel()!=1 or old_joint.requires_grad or not bool(torch.isfinite(old_joint).all()):
        raise ValueError('detached finite frozen old joint score required')
    joint, components, coefficients=dynamics.evaluate(
        record.x.to(prediction.device), record.action.to(prediction.device), prediction, record.index)
    if joint.numel()!=1:
        raise ValueError('one sample per FSDP policy cell required')
    if mode=='flash':
        if world_means is None: raise ValueError('Flash needs independently reduced stream means')
        current=dynamics.rectified_score(joint,components,coefficients,world_means)
    else:
        if world_means is not None: raise ValueError('Flow must not apply Flash correction')
        current=joint
    adv=advantage.to(device=current.device,dtype=current.dtype).reshape(1,1)
    inputs=PolicyLossInputs(adv,torch.ones_like(adv),torch.ones_like(adv,dtype=torch.bool),clip_range,0.)
    output=ClippedSurrogateObjective().compute(old_log_probs=old_joint.to(current.device).reshape(1,1),
        policy_stats=SimpleNamespace(current_log_probs=current.reshape(1,1)),loss_inputs=inputs)
    loss=output.loss*scale
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite dual-SDE loss')
    return loss
