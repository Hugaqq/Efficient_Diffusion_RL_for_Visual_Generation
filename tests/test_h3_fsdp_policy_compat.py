"""FSDP state-layout compatibility; metadata tests, not CUDA acceptance."""
from types import SimpleNamespace

import pytest
import torch

from visual_rl.runtime.h3_binding import _block_policy


@pytest.mark.parametrize('layout', ['single', 'multiple'])
@pytest.mark.parametrize('divisor', [8.0, 1.0])
def test_h3_policy_observes_both_fsdp_state_layouts(monkeypatch, layout, divisor):
    import torch.distributed.fsdp as fsdp

    group = SimpleNamespace(mp_policy=SimpleNamespace(param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, cast_forward_inputs=False), gradient_divide_factor=divisor)
    state = SimpleNamespace(**({'_fsdp_param_group': group} if layout == 'single'
                               else {'_fsdp_param_groups': [group]}))

    class Block:
        def _get_fsdp_state(self):
            return state

    monkeypatch.setattr(fsdp, 'FSDPModule', Block)
    if divisor == 8:
        assert _block_policy(Block()) == dict(param_dtype='bfloat16', reduce_dtype='float32',
                                              cast_forward_inputs=False, divisor=8)
    else:
        with pytest.raises(ValueError, match='mixed precision/reducer drift'):
            _block_policy(Block())


@pytest.mark.parametrize('state', [SimpleNamespace(_fsdp_param_group=None),
                                   SimpleNamespace(_fsdp_param_groups=[])])
def test_h3_policy_does_not_accept_missing_groups(monkeypatch, state):
    import torch.distributed.fsdp as fsdp

    class Block:
        def _get_fsdp_state(self):
            return state

    monkeypatch.setattr(fsdp, 'FSDPModule', Block)
    with pytest.raises(ValueError, match='not FSDP2 prepared'):
        _block_policy(Block())
