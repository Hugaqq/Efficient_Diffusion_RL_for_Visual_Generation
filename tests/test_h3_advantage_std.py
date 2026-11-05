import torch

from tests.test_h3_mvp_cpu_world import _plans, _simulate, _rewards
from tests.test_h3_training_extensions import legacy_module
from visual_rl.algorithms.optimization import h3_world as world


def test_std_modes_manual_world_values_and_exact_default(monkeypatch):
    plans = _plans(monkeypatch)
    def credits(module, mode=None):
        return _simulate(monkeypatch, lambda pg,r: module.gather_h3_credit(plans[r], process_group=pg,
            collection_id='collection-0', behavior_id='policy-0', rewards=_rewards(plans[r],r),
            **({} if mode is None else dict(advantage_std_domain=mode))))
    batch = credits(world)[0]
    group = credits(world, 'group')[0]
    old = legacy_module('visual_rl/algorithms/optimization/h3_world.py')
    # The baseline type check must see its own structurally identical immutable plan.
    old.H3WorldPlan = world.H3WorldPlan
    legacy = credits(old)[0]
    assert all(torch.equal(batch[k], legacy[k]) for k in batch)
    values = torch.tensor([row[3] for r in range(8) for row in _rewards(plans[r],r)], dtype=torch.float64)
    for r in range(8):
        for g in range(2):
            i = r*4 + g*2
            pair = values[i:i+2]
            for j in range(2):
                sid = plans[0].samples[i+j][0]
                assert torch.equal(batch[sid], ((pair[j]-pair.mean())/(values.std(correction=0)+1e-4)).float())
                assert torch.equal(group[sid], ((pair[j]-pair.mean())/(pair.std(correction=0)+1e-4)).float())
    _simulate(monkeypatch, lambda pg,r: world.gather_h3_credit(plans[r], process_group=pg,
        collection_id='collection-0', behavior_id='policy-0', rewards=_rewards(plans[r],r),
        advantage_std_domain='group' if r == 1 else 'batch'), error='reward plan/collection/behavior mismatch')


def test_group_constant_huge_values_exact_zero(monkeypatch):
    plans = _plans(monkeypatch)
    rows = _simulate(monkeypatch, lambda pg,r: world.gather_h3_credit(plans[r], process_group=pg,
        collection_id='collection-0', behavior_id='policy-0',
        rewards=tuple((*row, 1e308 if r < 4 else -1e308, True) for row in plans[r].samples_by_rank[r]),
        advantage_std_domain='group'))
    assert all(value.item() == 0 for value in rows[0].values())
