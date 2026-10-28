"""Dual rollout CPU tests only; no trainer/world/GPU attestation."""
import pytest
import torch
from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.trainer.h3_dual_rollout import collect,selected_step
from visual_rl.models.h3_joint import H3JointLayout


def test_stable_group_selection_and_full_vs_single_step():
    d=H3DualFlowSDE(H3JointLayout(1,2,1,3),H3ScheduleSnapshot.build(4))
    samples=(('a','g'),('b','g')); states={sid:torch.full((1,5),.25,dtype=torch.float16) for sid,_ in samples}
    def run(mode):
        gs={sid:(torch.Generator().manual_seed(1+i),torch.Generator().manual_seed(5+i)) for i,(sid,_) in enumerate(samples)}
        gates=[]
        result=collect(dynamics=d,samples=samples,initial_states=states,generators=gs,mode=mode,seed=42,optimizer_step=2,
            device='cpu',predict=lambda sid,x,index:x.float()*.1,read_behavior_id=lambda:'b',behavior_id='b',gate=lambda event,**kw:gates.append(event))
        return result,gates
    (flash,terminal,selection),fgates=run('flash'); (flow,_,_),ggates=run('flow')
    assert fgates==ggates and len(fgates)==16
    assert selection['a']==selection['b']==selected_step(seed=42,optimizer_step=2,group_id='g',num_steps=4)
    assert len(flash)==2 and len(flow)==8
    assert all(r.index==selection[r.sample_id] for r in flash)
    assert set(terminal)=={'a','b'} and all(t.dtype==torch.float16 for t in terminal.values())
    (again,term2,selected2),_=run('flash')
    assert selected2==selection and all(torch.equal(a.action,b.action) for a,b in zip(flash,again))
    assert all(torch.equal(terminal[s],term2[s]) for s in terminal)


def test_selection_validates_and_behavior_mismatch_rejected():
    with pytest.raises(ValueError):selected_step(seed=True,optimizer_step=0,group_id='x',num_steps=3)
    with pytest.raises(ValueError):selected_step(seed=1,optimizer_step=0,group_id='',num_steps=3)
    d=H3DualFlowSDE(H3JointLayout(1,2,1,3),H3ScheduleSnapshot.build(3))
    with pytest.raises(ValueError,match='behavior version'):
        collect(dynamics=d,samples=(('a','g'),),initial_states={'a':torch.zeros((1,5),dtype=torch.float16)},
            generators={'a':(torch.Generator(),torch.Generator())},mode='flash',seed=1,optimizer_step=0,
            device='cpu',predict=lambda s,x,i:x,read_behavior_id=lambda:'other',behavior_id='b',gate=lambda event:None)
