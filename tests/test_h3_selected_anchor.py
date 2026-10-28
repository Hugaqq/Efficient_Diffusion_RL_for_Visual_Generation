"""Selected-step H3 replay anchor unit evidence, not GPU/Flash validation."""
import pytest
import torch

from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization.h3_anchor import H3ReplayAnchor
from visual_rl.models.h3_joint import H3JointLayout


def test_selected_anchor_exact_per_sample_coverage_and_frozen_scores():
    dynamics = H3CPSDynamics(H3JointLayout(1, 2, 1, 3), H3ScheduleSnapshot.build(3))
    ids = ('a', 'b')
    selected = {'a': 0, 'b': 2}
    records = []
    for j, (sid, index) in enumerate(selected.items()):
        x = torch.full((1, 5), 0.25 + j, dtype=torch.float16)
        record = dynamics.sample(x, x.float() * 0.1, index,
            video_generator=torch.Generator().manual_seed(11+j),
            audio_generator=torch.Generator().manual_seed(21+j))
        records.append((sid, record))
    def owner():
        return H3ReplayAnchor(collection_id='c', behavior_id='b', sample_ids=ids,
            dynamics=dynamics, max_cells=2, max_trajectory_bytes=1024,
            selected_indices=selected)
    def capture(anchor, values):
        anchor.capture(tuple(values), collection_id='c', behavior_id='b', device='cpu',
            predict=lambda r: r.x.float()*0.1, read_behavior_id=lambda:'b')
    a = owner()
    capture(a, records)
    for sid, record in records:
        score = a.score(collection_id='c', behavior_id='b', sample_id=sid, index=record.index)
        assert torch.equal(score, dynamics.evaluate(record, record.x.float()*0.1))
        assert not score.requires_grad
    with pytest.raises(ValueError, match='unexpected sample/index'):
        a.score(collection_id='c', behavior_id='b', sample_id='a', index=2)
    with pytest.raises(ValueError, match='coverage'):
        capture(owner(), records[:1])
    with pytest.raises(ValueError, match='unexpected sample/index'):
        capture(owner(), [('a', records[1][1]), records[0]])
    with pytest.raises(ValueError, match='sample IDs'):
        H3ReplayAnchor(collection_id='c', behavior_id='b', sample_ids=ids,
            dynamics=dynamics, max_cells=2, max_trajectory_bytes=1024,
            selected_indices={'a':0})
