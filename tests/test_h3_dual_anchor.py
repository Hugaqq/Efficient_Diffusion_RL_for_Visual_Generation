import pytest
import torch
from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization.h3_dual_anchor import H3DualReplayAnchor
from visual_rl.algorithms.trainer.h3_dual_rollout import H3DualRecord
from visual_rl.models.h3_joint import H3JointLayout


def test_anchor_freezes_real_replayed_joint_and_rejects_reordering():
    d=H3DualFlowSDE(H3JointLayout(1,2,1,3),H3ScheduleSnapshot.build(3))
    x=torch.full((1,5),.25,dtype=torch.float16)
    action=d.sample(x,x,1,video_generator=torch.Generator().manual_seed(3),audio_generator=torch.Generator().manual_seed(4))
    rec=H3DualRecord('a',1,x,action)
    def owner(): return H3DualReplayAnchor(dynamics=d,behavior_id='b',expected_keys=(('a',1),),max_records=1)
    anchor=owner();anchor.capture((rec,),device='cpu',predict=lambda sid,x,index:x.float(),read_behavior_id=lambda:'b',gate=lambda e:None)
    score,coef=anchor.get('a',1);expected,_,cs=d.evaluate(x,action,x.float(),1)
    assert torch.equal(score,expected) and all(torch.equal(a,b) for a,b in zip(coef,cs))
    score.fill_(99);assert torch.equal(anchor.get('a',1)[0],expected)
    with pytest.raises(RuntimeError):anchor.capture((rec,),device='cpu',predict=lambda *a:x,read_behavior_id=lambda:'b',gate=lambda e:None)
    broken=owner()
    with pytest.raises(ValueError,match='identity/order'):broken.capture((H3DualRecord('z',1,x,action),),device='cpu',predict=lambda *a:x,read_behavior_id=lambda:'b',gate=lambda e:None)
    with pytest.raises(RuntimeError,match='incomplete'):broken.get('a',1)
