"""Small rank-local diagnostics, reduced by the existing lifecycle exchange."""
import math

import torch

KL_KEYS = ('kl_video', 'kl_audio', 'kl')
LOSS_KEYS = ('policy_loss', 'kl_loss')


def metric_keys(*, ema):
    return {*KL_KEYS, *LOSS_KEYS, 'lora_norm'} | ({'ema_live_distance', 'ema_committed_steps', 'ema_updates'} if ema else set())


def local_metrics(named_parameters, *, rank, process_group, kl, ema, loss):
    from visual_rl.algorithms.trainer.h3_ema import local_parameter
    parameters = tuple(named_parameters)
    if ema is None:
        shards = {name: local_parameter(p, rank=rank, process_group=process_group)[0] for name,p in parameters}
    else:
        shards = ema.validate()
        if ({name: id(p) for name,p in parameters} != ema._ids
                or len(parameters) != len(shards) or rank != ema.rank or process_group is not ema.group):
            raise ValueError('diagnostic parameters differ from EMA ownership')
    # 一次设备到 CPU 拷贝；在 CPU 保留原逐张量 double 求和顺序，避免改变诊断数值。
    packed = torch.cat([p.detach().reshape(-1) for p in shards.values()]).to(device='cpu', dtype=torch.float64)
    squared_norm = distance = 0.
    values = packed.split([p.numel() for p in shards.values()])
    for (name, local), flat in zip(shards.items(), values):
        value = flat.reshape(local.shape)
        squared_norm += float(value.square().sum())
        if ema is not None:
            distance += float((value - ema.shadow[name].double()).square().sum())
    result = dict(kl if kl is not None else dict(dict.fromkeys(KL_KEYS, 0.), policy_loss=loss, kl_loss=0.),
                  lora_squared_norm=squared_norm)
    if ema is not None:
        result.update(ema_live_squared_distance=distance, ema_committed_steps=ema.committed_steps, ema_updates=ema.updates)
    return result


def validate_diagnostic(key, value):
    """有符号 policy loss；KL、范数和距离均不得为负。计数由调用方严格检查。"""
    if type(value) is not float or not math.isfinite(value) or (key != 'policy_loss' and value < 0):
        raise ValueError('H3 diagnostics must be finite floats; only policy_loss may be negative')


def world_metrics(rows, *, step, ema_config):
    enabled = ema_config.enabled
    keys = {*KL_KEYS, *LOSS_KEYS, 'lora_squared_norm'} | (
        {'ema_live_squared_distance', 'ema_committed_steps', 'ema_updates'} if enabled else set())
    if len(rows) != 8 or any(type(row) is not dict or set(row) != keys for row in rows):
        raise ValueError('H3 diagnostics must cover all eight ranks with exact fields')
    for row in rows:
        for key, value in row.items():
            if key in ('ema_committed_steps', 'ema_updates'):
                expected = step+1 if key == 'ema_committed_steps' else (step+1)//ema_config.update_interval
                if type(value) is not int or value != expected:
                    raise ValueError('EMA diagnostic counter differs from committed progress')
            else:
                validate_diagnostic(key, value)
    result = {key: sum(row[key] for row in rows)/8 for key in (*KL_KEYS, *LOSS_KEYS)}
    result['lora_norm'] = math.sqrt(sum(row['lora_squared_norm'] for row in rows))
    if enabled:
        result.update(ema_live_distance=math.sqrt(sum(row['ema_live_squared_distance'] for row in rows)),
                      ema_committed_steps=float(step+1), ema_updates=float((step+1)//ema_config.update_interval))
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError('H3 world diagnostics overflow')
    return result
