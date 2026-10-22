#!/usr/bin/env python3
"""Freeze quantity schedules, NOT a trainer or proof of completed rollouts.

The original loop evaluates before training an epoch. Epochs0/1 collect no
training videos; subsequent epochs collect384 and perform two192-row commits.
Prompt assignments/noise and execution evidence must be bound by the real
runtime later; this script deliberately does not fabricate them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_DATA = {
    'train': (19700, '610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016'),
    'test': (300, '96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5'),
}


def build_schedule(updates: int = 140) -> dict:
    if type(updates) is not int or updates <= 0 or updates % 2:
        raise ValueError('two-commit collections require a positive even update target')
    collections = [
        {
            'collection_index': i,
            'source_epoch': i + 2,
            'expected_behavior_policy_after_commits': i * 2,
            'training_video_slots': 384,
            'nominal_groups': 48,
            'expected_commit_numbers': [i * 2 + 1, i * 2 + 2],
            'nominal_rows_per_commit': 192,
        }
        for i in range(updates // 2)
    ]
    evaluations = [
        {
            'source_epoch': epoch,
            'after_optimizer_commits': max(0, (epoch - 2) * 2),
            'prompt_slots': 300,
            'sampling_steps': 50,
            'before_training_collection': True,
        }
        for epoch in range(0, updates // 2 + 2, 20)
    ]
    train = len(collections) * 384
    evaluation = len(evaluations) * 300
    return {
        'schema_version': 1,
        'manifest_kind': 'quantity_schedule_not_execution_evidence',
        'paper_exact_endpoint_confirmed': False,
        'sampling_plan_materialized': False,
        'optimizer_commits_target': updates,
        'active_collection_rounds': len(collections),
        'group_size': 8,
        'training_videos_target': train,
        'evaluation_videos_target': evaluation,
        'logical_hps_frame_scores_target': (train + evaluation) * 81,
        'train_sampling_steps': 20,
        'evaluation_sampling_steps': 50,
        'frames_per_video': 81,
        'additional_final_evaluation': False,
        'collections': collections,
        'evaluations': evaluations,
    }


def validate_source_data(root: Path) -> dict:
    records = {}
    for name, (count, expected_sha) in EXPECTED_DATA.items():
        path = root / 'dataset' / 'video' / f'{name}.txt'
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        lines = raw.decode('utf-8').splitlines()
        if digest != expected_sha or sum(bool(line.strip()) for line in lines) != count:
            raise ValueError(f'author {name} dataset does not match the approved source')
        records[name] = {'sha256': digest, 'nonempty_rows': count, 'deduplicated': False}
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=140)
    a = p.parse_args()
    result = build_schedule(a.updates)
    result['author_datasets'] = validate_source_data(a.source_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'collections', 'evaluations'}}, indent=2))


if __name__ == '__main__':
    main()
