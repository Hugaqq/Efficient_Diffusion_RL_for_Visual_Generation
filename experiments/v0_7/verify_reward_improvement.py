"""Pure reward-improvement verifier over canonical Q100 rows/status."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import statistics

from experiments.v0_7.offline_aggregate import (
    ALGORITHMS,
    SEEDS,
    RewardRow,
    SourceStatus,
    validate_rows,
)

EARLY_STEPS = frozenset(range(0, 36))
LATE_STEPS = frozenset(range(64, 100))


def verify_reward_improvement(
    rows: Iterable[RewardRow | Mapping[str, object]],
    source_status: Sequence[SourceStatus],
) -> dict[str, dict[str, object]]:
    """Return a separate evidence/reward verdict for each algorithm."""

    normalized = validate_rows(rows)
    return {
        algorithm: _verify_algorithm(
            algorithm,
            tuple(row for row in normalized if row.algorithm == algorithm),
            tuple(
                status
                for status in source_status
                if status.algorithm == algorithm
            ),
        )
        for algorithm in ALGORITHMS
    }


def _verify_algorithm(
    algorithm: str,
    rows: Sequence[RewardRow],
    statuses: Sequence[SourceStatus],
) -> dict[str, object]:
    by_seed = {seed: tuple(row for row in rows if row.seed == seed) for seed in SEEDS}
    status_by_seed = {status.seed: status for status in statuses}
    balance_error = _balance_error(by_seed)
    complete = (
        len(statuses) == 3
        and
        set(status_by_seed) == set(SEEDS)
        and all(
            status.inspect_ok
            and status.audit_ok
            and status.committed_steps == 100
            for status in status_by_seed.values()
        )
        and all(_has_complete_steps(by_seed[seed]) for seed in SEEDS)
        and balance_error is None
    )
    if not complete:
        return {
            "algorithm": algorithm,
            "evidence_complete": False,
            "reward_pass": False,
            "reason": balance_error
            or "three complete audited 100-step seeds are required",
        }

    early_cells: list[float] = []
    late_cells: list[float] = []
    seed_deltas: dict[str, float] = {}
    for seed in SEEDS:
        prompt_groups: dict[str, list[RewardRow]] = defaultdict(list)
        for row in by_seed[seed]:
            prompt_groups[row.prompt_id].append(row)
        early_prompt = [
            _mean(row.weighted_total for row in prompt_rows if row.step in EARLY_STEPS)
            for prompt_rows in prompt_groups.values()
        ]
        late_prompt = [
            _mean(row.weighted_total for row in prompt_rows if row.step in LATE_STEPS)
            for prompt_rows in prompt_groups.values()
        ]
        early_cells.extend(early_prompt)
        late_cells.extend(late_prompt)
        seed_deltas[str(seed)] = _mean(late_prompt) - _mean(early_prompt)

    pooled_early = _mean(early_cells)
    pooled_late = _mean(late_cells)
    pooled_delta = pooled_late - pooled_early
    pooled_early_std = statistics.pstdev(early_cells)
    threshold = max(0.0, 0.1 * pooled_early_std)
    positive_seed_count = sum(delta > 0.0 for delta in seed_deltas.values())
    median_delta = statistics.median(seed_deltas.values())

    step_points = [
        _mean(
            row.weighted_total
            for row in rows
            if row.step == step
        )
        for step in range(100)
    ]
    slope = theil_sen_slope(step_points)
    reward_pass = (
        pooled_delta > threshold
        and positive_seed_count >= 2
        and median_delta > 0.0
        and slope > 0.0
    )
    return {
        "algorithm": algorithm,
        "evidence_complete": True,
        "median_seed_delta": median_delta,
        "pooled_delta": pooled_delta,
        "pooled_early_std": pooled_early_std,
        "positive_seed_count": positive_seed_count,
        "reward_pass": reward_pass,
        "seed_deltas": seed_deltas,
        "theil_sen_pair_count": 4950,
        "theil_sen_slope": slope,
        "threshold": threshold,
    }


def theil_sen_slope(points: Sequence[float]) -> float:
    """Return the median of all 4,950 pair slopes for 100 finite points."""

    if len(points) != 100:
        raise ValueError("Theil-Sen input must contain exactly 100 points")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in points
    ):
        raise ValueError("Theil-Sen points must be finite")
    slopes = [
        (float(points[right]) - float(points[left])) / (right - left)
        for left in range(99)
        for right in range(left + 1, 100)
    ]
    if len(slopes) != 4950:
        raise RuntimeError("unexpected Theil-Sen pair count")
    return statistics.median(slopes)


def _has_complete_steps(rows: Sequence[RewardRow]) -> bool:
    return {row.step for row in rows} == set(range(100))


def _balance_error(
    by_seed: Mapping[int, Sequence[RewardRow]],
) -> str | None:
    prompt_sets = {
        seed: {row.prompt_id for row in by_seed[seed]}
        for seed in SEEDS
    }
    if any(not prompts for prompts in prompt_sets.values()) or len(
        {frozenset(prompts) for prompts in prompt_sets.values()}
    ) != 1:
        return "three seeds must use the same non-empty prompt set"
    prompts = next(iter(prompt_sets.values()))
    expected_samples_per_step: int | None = None
    expected_windows: dict[str, tuple[int, ...]] | None = None
    for seed in SEEDS:
        rows = by_seed[seed]
        counts_by_step = {
            step: sum(row.step == step for row in rows)
            for step in range(100)
        }
        counts = set(counts_by_step.values())
        if len(counts) != 1 or next(iter(counts), 0) <= 0:
            return "each seed must have one constant positive sample count per step"
        samples_per_step = next(iter(counts))
        if expected_samples_per_step is None:
            expected_samples_per_step = samples_per_step
        elif samples_per_step != expected_samples_per_step:
            return "sample count per step must be identical across seeds"
        for step in range(100):
            if len({row.prompt_id for row in rows if row.step == step}) != 1:
                return "each artifact step must contain exactly one prompt group"
        windows = {
            "early": tuple(
                sum(
                    row.prompt_id == prompt and row.step in EARLY_STEPS
                    for row in rows
                )
                for prompt in sorted(prompts)
            ),
            "late": tuple(
                sum(
                    row.prompt_id == prompt and row.step in LATE_STEPS
                    for row in rows
                )
                for prompt in sorted(prompts)
            ),
        }
        if any(len(set(counts)) != 1 or counts[0] <= 0 for counts in windows.values()):
            return "early/late windows must be exactly balanced across prompts"
        if expected_windows is None:
            expected_windows = windows
        elif windows != expected_windows:
            return "early/late prompt counts must be identical across seeds"
    return None


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average an empty reward cell")
    return math.fsum(items) / len(items)
