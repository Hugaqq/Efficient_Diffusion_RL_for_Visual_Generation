"""Small rank-local diagnostics, reduced by the existing lifecycle exchange."""
import math

import torch

KL_KEYS = ('kl_video', 'kl_audio', 'kl')


def metric_keys(*, ema):
    return {*KL_KEYS, 'lora_norm'} | ({'ema_live_distance', 'ema_committed_steps', 'ema_updates'} if ema else set())


def local_metrics(named_parameters, *, rank, process_group, kl, ema):
    from visual_rl.algorithms.trainer.h3_ema import local_parameter
    squared_norm = distance = 0.
    for name, parameter in named_parameters:
        local, _ = local_parameter(parameter, rank=rank, process_group=process_group)
        value = local.detach().to(device='cpu', dtype=torch.float64)
        squared_norm += float(value.square().sum())
        if ema is not None:
            distance += float((value - ema.shadow[name].double()).square().sum())
    result = dict(kl or dict.fromkeys(KL_KEYS, 0.), lora_squared_norm=squared_norm)
    if ema is not None:
        ema.validate()
        result.update(ema_live_squared_distance=distance, ema_committed_steps=ema.committed_steps, ema_updates=ema.updates)
    return result


def world_metrics(rows, *, step, ema_config):
    enabled = ema_config.enabled
    keys = {*KL_KEYS, 'lora_squared_norm'} | (
        {'ema_live_squared_distance', 'ema_committed_steps', 'ema_updates'} if enabled else set())
    if len(rows) != 8 or any(type(row) is not dict or set(row) != keys for row in rows):
        raise ValueError('H3 diagnostics must cover all eight ranks with exact fields')
    for row in rows:
        for key, value in row.items():
            if key in ('ema_committed_steps', 'ema_updates'):
                expected = step+1 if key == 'ema_committed_steps' else (step+1)//ema_config.update_interval
                if type(value) is not int or value != expected:
                    raise ValueError('EMA diagnostic counter differs from committed progress')
            elif type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError('H3 diagnostics must be finite nonnegative floats')
    result = {key: sum(row[key] for row in rows)/8 for key in KL_KEYS}
    result['lora_norm'] = math.sqrt(sum(row['lora_squared_norm'] for row in rows))
    if enabled:
        result.update(ema_live_distance=math.sqrt(sum(row['ema_live_squared_distance'] for row in rows)),
                      ema_committed_steps=step+1, ema_updates=(step+1)//ema_config.update_interval)
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError('H3 world diagnostics overflow')
    return result
