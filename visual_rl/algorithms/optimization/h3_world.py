"""WORLD8 small-table admission and complete-collection H3 credit.

No model/latent/media communication, PG ownership, phase scheduler, or optimizer.
Call admission before any model collective. Cooperative local input failures
join the same bounded receipt exchange; dead ranks/transport failures still
require the caller's timeout and whole-attempt teardown.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import socket

import torch
import torch.distributed as dist

WORLD_SIZE = 8
MAX_WORLD_SAMPLES = 4096
MAX_RECEIPT_BYTES = 256 * 1024  # at most 2 MiB of serialized receipts per exchange
MAX_ID_BYTES = 256
MAX_POLICY_STEPS = 4096


def _identity(value):
    if (type(value) is not str or not value or value.strip() != value
            or '\n' in value or '\r' in value or len(value.encode('utf-8')) > MAX_ID_BYTES):
        raise ValueError('expected a bounded nonempty canonical identity')
    return value


def _positive_int(value, limit):
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError('invalid positive integer/count bound')
    return value


def _profile(collection_id, behavior_id, group_size, num_steps, geometry_id, schedule_id, score_profile):
    for value in (collection_id, behavior_id, geometry_id, schedule_id, score_profile):
        _identity(value)
    _positive_int(group_size, MAX_WORLD_SAMPLES // WORLD_SIZE)
    if group_size < 2:
        raise ValueError('group_size must be at least two')
    _positive_int(num_steps, MAX_POLICY_STEPS)
    return (collection_id, behavior_id, group_size, num_steps, geometry_id, schedule_id, score_profile)


def _samples(rows, group_size, group_layout='rank_local'):
    if group_layout not in ('rank_local', 'paired_4x8'):
        raise ValueError('invalid group layout')
    if type(rows) not in (tuple, list) or not 1 <= len(rows) <= MAX_WORLD_SAMPLES // WORLD_SIZE:
        raise ValueError('empty or oversized local sample table')
    result, seen, groups = [], set(), {}
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 3:
            raise ValueError('sample row must be (sample_id, group_id, member_index)')
        sid, gid, member = row
        _identity(sid)
        _identity(gid)
        if type(member) is not int or not 0 <= member < group_size or sid in seen:
            raise ValueError('duplicate sample or invalid member')
        seen.add(sid)
        groups.setdefault(gid, []).append(member)
        result.append((sid, gid, member))
    if group_layout == 'paired_4x8':
        if group_size != 8 or len(result) != 4 or len(groups) != 1 or sorted(next(iter(groups.values()))) not in (list(range(4)), list(range(4,8))):
            raise ValueError('paired_4x8 requires one contiguous four-member half')
    elif any(sorted(members) != list(range(group_size)) for members in groups.values()):
        raise ValueError('each occurrence group must be local_complete, members 0..K-1 exactly once')
    return tuple(result)


@dataclass(frozen=True, slots=True)
class H3WorldPlan:
    """Immutable rank-major, original local-order plan (groups are occurrences)."""
    collection_id: str
    behavior_id: str
    group_size: int
    num_steps: int
    geometry_id: str
    schedule_id: str
    score_profile: str
    samples_by_rank: tuple[tuple[tuple[str, str, int], ...], ...]
    group_layout: str = 'rank_local'

    def __post_init__(self):
        _profile(self.collection_id, self.behavior_id, self.group_size, self.num_steps,
                 self.geometry_id, self.schedule_id, self.score_profile)
        if type(self.samples_by_rank) is not tuple or len(self.samples_by_rank) != WORLD_SIZE:
            raise ValueError('plan requires eight rank tables')
        tables = tuple(_samples(rows, self.group_size, self.group_layout) for rows in self.samples_by_rank)
        if self.group_layout == 'paired_4x8':
            paired_groups = set()
            for rank in range(0, WORLD_SIZE, 2):
                left, right = tables[rank:rank+2]
                gids = {row[1] for row in left + right}
                if (len(gids) != 1 or paired_groups & gids or
                        tuple(row[2] for row in left) != tuple(range(4)) or
                        tuple(row[2] for row in right) != tuple(range(4,8))):
                    raise ValueError('paired_4x8 requires four distinct groups on adjacent rank pairs')
                paired_groups.update(gids)
        if len({len(rows) for rows in tables}) != 1:
            raise ValueError('uneven rank sample counts')
        seen_samples, seen_groups = set(), set()
        for rows in tables:
            ids = {row[0] for row in rows}
            groups = {row[1] for row in rows}
            if seen_samples & ids or (self.group_layout == 'rank_local' and seen_groups & groups):
                raise ValueError('world sample duplicate or occurrence group split/shared across ranks')
            seen_samples.update(ids)
            seen_groups.update(groups)
        object.__setattr__(self, 'samples_by_rank', tables)

    @property
    def samples(self):
        return tuple(row for rows in self.samples_by_rank for row in rows)

    @property
    def global_count(self) -> int:
        """All unique samples times all policy steps, INCLUDING terminal q=0."""
        return len(self.samples) * self.num_steps

    @property
    def plan_id(self) -> str:
        profile = _profile(self.collection_id, self.behavior_id, self.group_size, self.num_steps,
                           self.geometry_id, self.schedule_id, self.score_profile)
        if self.group_layout != 'rank_local':
            profile = (*profile, self.group_layout)
        return hashlib.sha256(json.dumps((profile, self.samples_by_rank), ensure_ascii=True).encode()).hexdigest()

    def loss_scale(self, *, n_slot: int, reducer_divisor: float) -> float:
        _positive_int(n_slot, len(self.samples_by_rank[0]) * self.num_steps)
        return h3_slot_loss_scale(n_slot=n_slot, global_count=self.global_count,
                                  reducer_divisor=reducer_divisor)


def h3_slot_loss_scale(*, n_slot: int, global_count: int, reducer_divisor: float) -> float:
    """Scale a slot's local cell MEAN by d*n/N, with no second GA/world divisor.

    d is supplied from the prepared backend's explicitly configured reduction,
    not guessed from world size. Caller must include every planned cell once.
    """
    _positive_int(global_count, MAX_WORLD_SAMPLES * MAX_POLICY_STEPS)
    _positive_int(n_slot, global_count)
    if (type(reducer_divisor) not in (int, float) or not math.isfinite(reducer_divisor)
            or reducer_divisor <= 0):
        raise ValueError('reducer_divisor must be finite and positive')
    scale = reducer_divisor * (n_slot / global_count)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('nonfinite or underflowed slot loss scale')
    return scale


def _world_rank(process_group):
    # A missing/dead PG cannot participate in a cooperative input-error receipt.
    if (not dist.is_available() or not dist.is_initialized() or process_group is not dist.group.WORLD
            or dist.get_backend(process_group) != 'nccl' or dist.get_world_size(process_group) != WORLD_SIZE):
        raise ValueError('requires caller-initialized NCCL WORLD8')
    rank = dist.get_rank(process_group)
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise ValueError('invalid actual WORLD rank')
    return rank


def _exchange(kind, rank, process_group, payload, valid):
    """One collective, even for a bad local table; never serialize error text."""
    host = hashlib.sha256(socket.gethostname().encode()).hexdigest()
    receipt = [1, kind, rank, host, payload, valid]
    try:
        encoded = json.dumps(receipt, ensure_ascii=True, allow_nan=False).encode('ascii')
        if len(encoded) > MAX_RECEIPT_BYTES:
            raise ValueError('receipt exceeds byte budget')
    except (ValueError, TypeError, OverflowError):
        encoded = json.dumps([1, kind, rank, host, None, False]).encode('ascii')
    received = [None] * WORLD_SIZE
    dist.all_gather_object(received, encoded, group=process_group)
    tables, errors = [], []
    for peer, raw in enumerate(received):
        if type(raw) is not bytes or len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError('invalid or oversized peer receipt')
        entry = json.loads(raw)
        if (type(entry) is not list or len(entry) != 6 or type(entry[0]) is not int
                or entry[:2] != [1, kind] or type(entry[2]) is not int or entry[2] != peer
                or entry[3] != host or type(entry[5]) is not bool):
            raise ValueError('inconsistent phase, node, rank or receipt schema')
        if not entry[5]:
            errors.append(peer)
        tables.append(entry[4])
    if errors:
        raise ValueError(f'H3 {kind} rejected local input on ranks {errors}')
    return tables


def admit_h3_world(
    *, process_group, collection_id, behavior_id, samples, group_size, num_steps,
    geometry_id, schedule_id, score_profile, sp=1, tp=1, rollout_microbatch=1,
    replay_microbatch=1, chunk_size=1, guidance=1., beta=0., local_error=False, group_layout='rank_local',
) -> H3WorldPlan:
    """All ranks call before rollout: equal positive work and complete local groups.

    samples: ordered (sample_id, prompt-occurrence group_id, member_index).
    IDs must identify the actual common geometry, dual schedule and score
    profile. The caller owns those bindings; this routine never calls a model.
    local_error=True is the caller's cooperative preflight-failure signal.
    """
    rank = _world_rank(process_group)
    payload, valid = None, False
    try:
        if local_error is not False:
            raise ValueError('caller preflight failed')
        if any(type(v) is not int or v != 1 for v in (sp, tp, rollout_microbatch, replay_microbatch, chunk_size)):
            raise ValueError('H3 requires SP1/TP1/B1/chunk1')
        if (type(guidance) not in (float, int) or guidance != 1
                or type(beta) not in (float, int) or beta != 0):
            raise ValueError('H3 requires guidance1/beta0')
        profile = _profile(collection_id, behavior_id, group_size, num_steps, geometry_id, schedule_id, score_profile)
        if group_layout != 'rank_local':
            profile = (*profile, group_layout)
        payload = [profile, _samples(samples, group_size, group_layout)]
        valid = True
    except (ValueError, TypeError, OverflowError):
        pass
    tables = _exchange('admission', rank, process_group, payload, valid)
    if any(type(table) is not list or len(table) != 2 for table in tables):
        raise ValueError('invalid admission table')
    profile = tables[0][0]
    if type(profile) is not list or len(profile) not in (7,8) or any(table[0] != profile for table in tables):
        raise ValueError('collection/behavior/schedule/geometry/score profile mismatch')
    return H3WorldPlan(*profile[:7], samples_by_rank=tuple(table[1] for table in tables),
                       group_layout=profile[7] if len(profile) == 8 else 'rank_local')


def _reward_rows(plan, rank, rows):
    expected = plan.samples_by_rank[rank]
    if type(rows) not in (tuple, list) or len(rows) != len(expected):
        raise ValueError('reward rows must cover every local sample')
    by_id = {}
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 5:
            raise ValueError('reward row must be (sample_id, group_id, member_index, reward, valid)')
        sid, gid, member, value, valid = row
        _identity(sid)
        _identity(gid)
        if (sid in by_id or type(member) is not int or type(value) is not float
                or not math.isfinite(value) or valid is not True):
            raise ValueError('duplicate/invalid/nonfinite reward row')
        by_id[sid] = (sid, gid, member, value, valid)
    if set(by_id) != {row[0] for row in expected}:
        raise ValueError('reward sample identity differs from plan/rank')
    ordered = tuple(by_id[sid] for sid, _, _ in expected)
    if any(row[:3] != identity for row, identity in zip(ordered, expected)):
        raise ValueError('reward occurrence/member identity differs from plan')
    return ordered


def gather_h3_credit(plan: H3WorldPlan, *, process_group, collection_id, behavior_id,
                     rewards, local_error=False) -> dict[str, torch.Tensor]:
    """Gather only already-composed scalar rewards; return owned CPU FP32 scalars.

    Uses group means and the full unique world's population std in float64.
    No fabricated RewardResult trajectory/condition IDs just to adapt the old
    processor, no invalid-row dropping, clipping, or hidden reward weighting.
    """
    rank = _world_rank(process_group)
    payload, valid = None, False
    try:
        if local_error is not False or not isinstance(plan, H3WorldPlan):
            raise ValueError('caller preflight/plan invalid')
        if (collection_id, behavior_id) != (plan.collection_id, plan.behavior_id):
            raise ValueError('reward collection/behavior identity mismatch')
        payload = [plan.plan_id, collection_id, behavior_id, _reward_rows(plan, rank, rewards)]
        valid = True
    except (ValueError, TypeError, OverflowError):
        pass
    tables = _exchange('credit', rank, process_group, payload, valid)
    rows = []
    for peer, table in enumerate(tables):
        if (type(table) is not list or len(table) != 4
                or table[:3] != [plan.plan_id, plan.collection_id, plan.behavior_id]):
            raise ValueError('reward plan/collection/behavior mismatch')
        rows.extend(_reward_rows(plan, peer, table[3]))
    values = torch.tensor([row[3] for row in rows], dtype=torch.float64, device='cpu')
    # Exact zero credit for identical finite rewards, including magnitudes for
    # which an unnecessary sum/mean would overflow float64.
    if bool((values == values[0]).all()):
        return {sid: torch.zeros((), dtype=torch.float32, device='cpu') for sid, _, _ in plan.samples}
    std = values.std(correction=0)
    centered = torch.empty_like(values)
    groups = {}
    for index, (_, group_id, _) in enumerate(plan.samples):
        groups.setdefault(group_id, []).append(index)
    for indices in groups.values():
        centered[indices] = values[indices] - values[indices].mean()
    credit = (centered / (std + 1e-4)).to(torch.float32).detach()
    if not torch.isfinite(std) or not torch.isfinite(credit).all():
        raise ValueError('world reward statistics/credit overflow')
    return {sid: credit[index].clone() for index, (sid, _, _) in enumerate(plan.samples)}
