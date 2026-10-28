"""Real trainer control flow with explicit CPU model/reward/world leaves only."""
import torch
import pytest
from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
from visual_rl.algorithms.trainer.interface import PrepareRunContext
from tests.test_h3_mvp_cpu_stages import _setup


@pytest.mark.parametrize('mode,cells', [('dual_flow',6),('dual_flash',2)])
def test_dual_full_iteration_and_optimizer_step(tmp_path,monkeypatch,mode,cells):
    h=_setup(tmp_path,monkeypatch)
    # This simulates eight identical local ranks, not an NCCL test.
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda tensor,**kw:tensor.mul_(8))
    def commit(request):
        h.data.commits+=1;h.data.position+=2;h.data.behavior=request.next_behavior_id
    monkeypatch.setattr(h.data,'commit',commit)
    stages=H3DualIterationStages(mode=mode,seed=42,data=h.data,prepared=h.prepared,
        predict=h.predict,decode=h.decode,score_one=h.score_one,coordination_factory=h.coordination_factory,
        process_group=h.group,rank=0,device='cpu',read_behavior_id=lambda:h.data.behavior)
    trainer=stages.trainer();trainer.prepare_run(PrepareRunContext('dual-test','h3',0))
    before=[p.detach().clone() for _,p in h.prepared.trainable_named_parameters()]
    for step in range(5):
        result=trainer.run_iteration(step)
        assert result.stage_order==('prelude','rollout','reward','advantage','credit','optimize')
        assert result.value.payload.local_cells==cells
    assert h.data.commits==5 and h.data.behavior=='policy-5'
    assert len([event for event in h.events if event[0]=='step'])==5
    assert any(not torch.equal(x,p) for x,(_,p) in zip(before,h.prepared.trainable_named_parameters()))
