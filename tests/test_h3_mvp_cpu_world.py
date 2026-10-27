"""CPU world-table/credit/loss evidence; fake only WORLD/collective leaves."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visual_rl.algorithms.optimization import h3_world as world
from visual_rl.algorithms.optimization.advantage import AdvantageGrouping, GroupZScoreAdvantageProcessor
from visual_rl.algorithms.optimization.objective import ClippedSurrogateObjective, PolicyLossInputs
from visual_rl.algorithms.rewards.types import RewardBatchIdentity, RewardResult
from visual_rl.models.lifecycle.h3_fsdp import H3_REDUCER_DIVISOR


def _samples(rank, *, k=2, groups=2):
    # Same prompt may recur anywhere: groups are explicit occurrence identities,
    # not the text "same prompt" shared by all of these synthetic samples.
    return tuple((f's{rank}-{g}-{m}', f'same-prompt:occ-{rank}-{g}', m)
                 for g in range(groups) for m in range(k))


def _args(rank, **changes):
    return dict(collection_id='collection-0', behavior_id='policy-0', samples=_samples(rank),
                group_size=2, num_steps=3, geometry_id='geometry-1', schedule_id='dual-grid-1',
                score_profile='cps-elements-v1') | changes


def _simulate(monkeypatch, call, *, error=None, mutate=None):
    """Capture eight actual locally validated receipts, then replay their gather.

    Each function invocation reaches exactly ONE fake collective. Two passes
    avoid threads/PG/network: capture stops at the collective, replay supplies
    the same eight receipts to every simulated rank. No algorithm is mocked.
    """
    class Capture(Exception):
        pass
    rank, group = [0], object()
    monkeypatch.setattr(world.dist, 'group', SimpleNamespace(WORLD=group))
    monkeypatch.setattr(world.dist, 'is_available', lambda: True)
    monkeypatch.setattr(world.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(world.dist, 'get_backend', lambda pg: 'nccl')
    monkeypatch.setattr(world.dist, 'get_world_size', lambda pg: 8)
    monkeypatch.setattr(world.dist, 'get_rank', lambda pg: rank[0])
    receipts = []
    def capture(output, value, *, group):
        assert type(value) is bytes and len(value) <= world.MAX_RECEIPT_BYTES
        assert len(output) == 8
        receipts.append(value)
        raise Capture
    monkeypatch.setattr(world.dist, 'all_gather_object', capture)
    for r in range(8):
        rank[0] = r
        with pytest.raises(Capture):
            call(group, r)
    assert len(receipts) == 8
    if mutate:
        mutate(receipts)
    counts = [0] * 8
    def replay(output, value, *, group):
        counts[rank[0]] += 1
        output[:] = receipts
    monkeypatch.setattr(world.dist, 'all_gather_object', replay)
    results = []
    for r in range(8):
        rank[0] = r
        if error:
            with pytest.raises(ValueError, match=error) as raised:
                call(group, r)
            results.append(str(raised.value))
        else:
            results.append(call(group, r))
    assert counts == [1] * 8
    if error:
        assert len(set(results)) == 1  # all surviving ranks refuse together
    return results


def _plans(monkeypatch, change=lambda r: {}):
    return _simulate(monkeypatch, lambda pg, r: world.admit_h3_world(process_group=pg, **_args(r, **change(r))))


def _rewards(plan, rank):
    return tuple((*row, float(rank * 10 + i * (rank + 1)), True)
                 for i, row in enumerate(plan.samples_by_rank[rank]))


def _credits(monkeypatch, plans, rewards=None):
    return _simulate(monkeypatch, lambda pg, r: world.gather_h3_credit(
        plans[r], process_group=pg, collection_id='collection-0', behavior_id='policy-0',
        rewards=_rewards(plans[r], r) if rewards is None else rewards[r]))


def test_h3_world_import_plan_order_identity_and_bounds(monkeypatch):
    assert Path(world.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/algorithms/optimization/h3_world.py'
    plans = _plans(monkeypatch)
    plan = plans[0]
    assert all(p == plan and p.plan_id == plan.plan_id for p in plans)
    assert plan.samples == tuple(row for r in range(8) for row in _samples(r))
    assert len(plan.samples) == 32 and plan.global_count == 96
    assert len({gid for _, gid, _ in plan.samples}) == 16
    with pytest.raises(FrozenInstanceError):
        plan.num_steps = 2
    with pytest.raises(TypeError):
        plan.samples_by_rank[0][0] = ('other', 'other', 0)
    assert world.MAX_WORLD_SAMPLES == 4096 and world.MAX_RECEIPT_BYTES == 256 * 1024


def test_h3_world_cooperative_local_rejection_before_model(monkeypatch):
    calls = []
    bad = [dict(samples=()), dict(samples=_samples(3)[:-1]),
           dict(samples=(_samples(3)[0],) * 4), dict(samples=(("x", "g", True),) * 2),
           dict(group_size=1), dict(num_steps=0), dict(rollout_microbatch=2), dict(replay_microbatch=2),
           dict(chunk_size=2), dict(sp=2), dict(tp=2), dict(guidance=4.5), dict(beta=.1),
           dict(local_error=True), dict(geometry_id='x' * 257), dict(schedule_id=torch.ones(1)),
           dict(samples=_samples(3, k=2, groups=257))]
    for changes in bad:
        def call(pg, r):
            plan = world.admit_h3_world(process_group=pg, **_args(r, **(changes if r == 3 else {})))
            calls.append(plan)  # stands for a downstream model phase
        _simulate(monkeypatch, call, error='rejected local input on ranks \\[3\\]')
    assert calls == []
    # Each identity is within its limit, but serialized Unicode expansion is
    # bounded as well; oversized valid metadata becomes an error receipt.
    big = tuple((f'{i}-' + '界' * 80, f'g{i//2}-' + '界' * 80, i % 2) for i in range(512))
    _simulate(monkeypatch, lambda pg, r: world.admit_h3_world(
        process_group=pg, **_args(r, **({'samples': big} if r == 3 else {}))),
        error='rejected local input on ranks \\[3\\]')


def test_h3_world_global_uneven_group_duplicate_and_profile_rejection(monkeypatch):
    cases = [dict(samples=_samples(3, groups=1)), dict(samples=_samples(0)),
             dict(samples=tuple((row[0], _samples(0)[i][1], row[2]) for i, row in enumerate(_samples(3)))),
             dict(collection_id='other'), dict(behavior_id='other'), dict(num_steps=4),
             dict(schedule_id='different-dual-grid'), dict(geometry_id='other'), dict(score_profile='gaussian')]
    for changes in cases:
        _simulate(monkeypatch, lambda pg, r: world.admit_h3_world(
            process_group=pg, **_args(r, **(changes if r == 3 else {}))),
            error='uneven|duplicate|split/shared|mismatch')
    for index, value in ((2, 0), (3, 'other-node'), (1, 'credit')):
        def mutate(receipts):
            row = json.loads(receipts[7])
            row[index] = value
            receipts[7] = json.dumps(row).encode()
        _simulate(monkeypatch, lambda pg, r: world.admit_h3_world(process_group=pg, **_args(r)),
                  mutate=mutate, error='inconsistent')


def test_h3_world_credit_global_std_hand_math_processor_and_reordering(monkeypatch):
    plans = _plans(monkeypatch)
    plan = plans[0]
    credits = _credits(monkeypatch, plans)
    rows = tuple(row for r in range(8) for row in _rewards(plan, r))
    rewards = np.array([row[3] for row in rows], dtype=np.float64)
    # Independent full-world population std and group means.
    mean = sum(float(v) for v in rewards) / len(rewards)
    std = (sum((float(v) - mean)**2 for v in rewards) / len(rewards))**.5
    expected = []
    for sid, gid, member, value, valid in rows:
        group = [row[3] for row in rows if row[1] == gid]
        expected.append((value - sum(group)/len(group)) / (std + 1e-4))
    actual = torch.stack([credits[0][row[0]] for row in rows])
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=1e-7)
    assert std != pytest.approx(rewards[:4].std(ddof=0))
    assert all(torch.equal(credit[sid], credits[0][sid]) for credit in credits for sid in credits[0])
    assert all(v.device.type == 'cpu' and v.dtype == torch.float32 and not v.requires_grad
               for v in credits[0].values())
    # Only this comparison fixture supplies complete synthetic reward identity;
    # production does not invent trajectory/condition IDs for the old processor.
    ids = tuple(row[0] for row in rows)
    gids = tuple(row[1] for row in rows)
    identity = RewardBatchIdentity('collection-0', 'reward', ids, ids,
                                   tuple('trajectory:' + sid for sid in ids), gids, gids)
    valid = np.ones(len(rows), dtype=np.bool_)
    result = RewardResult(identity, {'composed': rewards}, {'composed': rewards}, {'composed': valid},
                          rewards, valid, {'composed': 'synthetic-test-scorer'})
    grouping = AdvantageGrouping(identity.batch_row_ids, ids, identity.trajectory_ids, gids)
    reference = GroupZScoreAdvantageProcessor(epsilon=1e-4, std_domain='batch').normalize(result, grouping)
    torch.testing.assert_close(actual, reference.values, rtol=0, atol=1e-7)
    # Reorder BOTH local admission and reward table rows; sample credit stays aligned.
    reordered = _plans(monkeypatch, lambda r: {'samples': tuple(reversed(_samples(r)))})
    shuffled_rewards = [tuple(reversed(_rewards(plan, r))) for r in range(8)]
    reordered_credit = _credits(monkeypatch, reordered, shuffled_rewards)[0]
    for sid in credits[0]:
        torch.testing.assert_close(reordered_credit[sid], credits[0][sid], rtol=0, atol=1e-7)
    credits[0][ids[0]].fill_(999)
    assert credits[1][ids[0]] != 999


def test_h3_world_reward_coverage_validity_binding_and_peer_corruption(monkeypatch):
    plans = _plans(monkeypatch)
    rows = _rewards(plans[0], 2)
    bad_rows = [rows[:-1], rows + (rows[0],), (rows[0],) * len(rows),
                ((rows[0][0], 'other-group', *rows[0][2:]),) + rows[1:],
                ((rows[0][0], rows[0][1], 1, *rows[0][3:]),) + rows[1:],
                ((*rows[0][:3], float('nan'), True),) + rows[1:],
                ((*rows[0][:3], float('inf'), True),) + rows[1:],
                ((*rows[0][:3], 1., False),) + rows[1:],
                ((*rows[0][:3], 1., 1),) + rows[1:],
                ((*rows[0][:3], torch.tensor(1.), True),) + rows[1:], _rewards(plans[0], 1)]
    for invalid in bad_rows:
        _simulate(monkeypatch, lambda pg, r: world.gather_h3_credit(
            plans[r], process_group=pg, collection_id='collection-0', behavior_id='policy-0',
            rewards=invalid if r == 2 else _rewards(plans[r], r)), error='rejected local input on ranks \\[2\\]')
    for change in ({'collection_id': 'other'}, {'behavior_id': 'other'}, {'local_error': True}):
        def call(pg, r):
            kwargs = dict(collection_id='collection-0', behavior_id='policy-0', rewards=_rewards(plans[r], r))
            return world.gather_h3_credit(plans[r], process_group=pg, **(kwargs | (change if r == 2 else {})))
        _simulate(monkeypatch, call, error='rejected local input on ranks \\[2\\]')
    def mutate(receipts):
        data = json.loads(receipts[7])
        data[4][0] = 'wrong-plan'
        receipts[7] = json.dumps(data).encode()
    _simulate(monkeypatch, lambda pg, r: world.gather_h3_credit(
        plans[r], process_group=pg, collection_id='collection-0', behavior_id='policy-0',
        rewards=_rewards(plans[r], r)), mutate=mutate, error='plan/collection/behavior mismatch')


def _objective(current, advantage):
    return ClippedSurrogateObjective().compute(
        old_log_probs=torch.zeros_like(current).detach(),
        policy_stats=SimpleNamespace(current_log_probs=current),
        loss_inputs=PolicyLossInputs(advantage, torch.ones_like(advantage),
                                    torch.ones_like(advantage, dtype=torch.bool), .2, 0.),
    ).loss


def _compare_global_update(plan, credit):
    # Real objectives and tiny trainable rank replicas. Only the reduction is
    # simulated here: sum gradients / the explicitly configured backend divisor.
    features = torch.arange(plan.global_count * 2, dtype=torch.float64).reshape(-1, plan.num_steps, 2)
    features = (features % 13 - 6) * .17
    initial = torch.tensor([[.12, -.2]], dtype=torch.float64)
    global_model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        global_model.weight.copy_(initial)
    advantages = torch.stack([credit[sid] for sid, _, _ in plan.samples]).double()[:, None].expand(-1, plan.num_steps)
    global_loss = _objective(global_model(features).squeeze(-1), advantages)
    global_loss.backward()
    models, losses, cells = [], [], []
    offset = 0
    for rank, rows in enumerate(plan.samples_by_rank):
        model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(initial)
        loss_value = 0.
        for local, (sid, _, _) in enumerate(rows):
            for index in range(plan.num_steps):
                cell = offset + local
                current = model(features[cell, index]).reshape(1, 1)
                loss = _objective(current, advantages[cell, index].reshape(1, 1))
                scaled = loss * plan.loss_scale(n_slot=1, reducer_divisor=H3_REDUCER_DIVISOR)
                scaled.backward()
                loss_value += scaled.detach()
                cells.append((rank, sid, index))
        offset += len(rows)
        assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
        models.append(model)
        losses.append(loss_value)
    assert len(cells) == plan.global_count and sum(i == plan.num_steps-1 for _, _, i in cells) == len(plan.samples)
    torch.testing.assert_close(sum(losses) / H3_REDUCER_DIVISOR, global_loss, rtol=1e-12, atol=1e-12)
    reduced = torch.stack([m.weight.grad for m in models]).sum(0) / H3_REDUCER_DIVISOR
    torch.testing.assert_close(reduced, global_model.weight.grad, rtol=1e-12, atol=1e-12)
    torch.optim.SGD(global_model.parameters(), lr=.03).step()
    for model in models:
        model.weight.grad = reduced.clone()
        torch.optim.SGD(model.parameters(), lr=.03).step()
        torch.testing.assert_close(model.weight, global_model.weight, rtol=1e-12, atol=1e-12)
    return reduced, global_model.weight, initial


def test_h3_world_real_ppo_cell_accumulation_matches_global_loss_gradient_update(monkeypatch):
    plans = _plans(monkeypatch)
    credits = _credits(monkeypatch, plans)
    gradient, weight, initial = _compare_global_update(plans[0], credits[0])
    assert gradient.abs().sum() > 0 and not torch.equal(weight, initial)


def test_h3_world_zero_credit_still_connected_backward_and_no_advantage_clip(monkeypatch):
    plans = _plans(monkeypatch)
    for value in (.1, 1e308):
        rewards = [tuple((*row, value, True) for row in plans[0].samples_by_rank[r]) for r in range(8)]
        credits = _credits(monkeypatch, plans, rewards)
        assert all(v.item() == 0 for v in credits[0].values())
        gradient, weight, initial = _compare_global_update(plans[0], credits[0])
        assert torch.equal(gradient, torch.zeros_like(gradient)) and torch.equal(weight, initial)
    large = _plans(monkeypatch, lambda r: {'samples': _samples(r, groups=8)})
    rows = [tuple((*row, float(r == 0 and i == 0), True) for i, row in enumerate(large[0].samples_by_rank[r]))
            for r in range(8)]
    credit = _credits(monkeypatch, large, rows)[0]
    assert max(v.item() for v in credit.values()) > 5  # no inherited +/-5 clip


def test_h3_world_loss_scale_counts_and_backend_divisor(monkeypatch):
    plans = _plans(monkeypatch, lambda r: dict(samples=_samples(r, k=8, groups=1), group_size=8, num_steps=10))
    plan = plans[0]
    assert plan.global_count == 640 and plan.loss_scale(n_slot=1, reducer_divisor=H3_REDUCER_DIVISOR) == 1/80
    assert plan.loss_scale(n_slot=10, reducer_divisor=H3_REDUCER_DIVISOR) == 1/8
    for n in (0, -1, True, 1., 81):
        with pytest.raises(ValueError):
            plan.loss_scale(n_slot=n, reducer_divisor=H3_REDUCER_DIVISOR)
    for change in (dict(global_count=0), dict(global_count=True), dict(n_slot=641),
                   dict(reducer_divisor=0), dict(reducer_divisor=True), dict(reducer_divisor=float('inf'))):
        with pytest.raises(ValueError):
            world.h3_slot_loss_scale(**(dict(n_slot=1, global_count=640, reducer_divisor=8.) | change))
    assert world.h3_slot_loss_scale(n_slot=1, global_count=640, reducer_divisor=4.) == 1/160
