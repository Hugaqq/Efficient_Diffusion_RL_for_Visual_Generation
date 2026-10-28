import pytest
import torch
from unittest.mock import patch
from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization.h3_dual_credit import global_stream_means,policy_loss
from visual_rl.algorithms.trainer.h3_dual_rollout import H3DualRecord
from visual_rl.models.h3_joint import H3JointLayout


def test_world_means_uses_global_sums_and_count_and_collective_on_bad_rows():
    calls=[]
    def reduce(t,**kwargs):
        calls.append(t.clone())
        t += torch.tensor([9.,15.,3.,0.],dtype=torch.float64)
    with patch('visual_rl.algorithms.optimization.h3_dual_credit.dist.all_reduce',reduce):
        means=global_stream_means(((torch.tensor(1.),torch.tensor(3.)),),process_group=None,device='cpu')
    assert means[0].item()==2.5 and means[1].item()==4.5 and len(calls)==1
    calls.clear()
    with patch('visual_rl.algorithms.optimization.h3_dual_credit.dist.all_reduce',reduce),pytest.raises(ValueError,match='at least one rank'):
        global_stream_means(((torch.tensor(float('nan')),torch.tensor(3.)),),process_group=None,device='cpu')
    assert len(calls)==1 and calls[0][3].item()==1


def test_policy_loss_corrects_per_stream_gradients_not_forward_ratio():
    d=H3DualFlowSDE(H3JointLayout(1,2,1,3),H3ScheduleSnapshot.build(4))
    x=torch.full((1,5),.25,dtype=torch.float16);p=torch.full((1,5),.1,requires_grad=True)
    action=d.sample(x,p,1,video_generator=torch.Generator().manual_seed(7),audio_generator=torch.Generator().manual_seed(8))
    record=H3DualRecord('a',1,x,action)
    old,_,coeff=d.evaluate(x,action,p.detach(),1); old=old.detach()
    flow=policy_loss(dynamics=d,record=record,prediction=p,old_joint=old,advantage=torch.tensor(1.),mode='flow',world_means=None,clip_range=.005,scale=1.)
    fgrad=torch.autograd.grad(flow,p,retain_graph=True)[0]
    flash=policy_loss(dynamics=d,record=record,prediction=p,old_joint=old,advantage=torch.tensor(1.),mode='flash',world_means=(coeff[0]/2,coeff[1]/3),clip_range=.005,scale=1.)
    sgrad=torch.autograd.grad(flash,p)[0]
    assert torch.equal(flow,flash)
    assert torch.allclose(sgrad[:,:2],2*fgrad[:,:2],atol=1e-6)
    assert torch.allclose(sgrad[:,2:],3*fgrad[:,2:],atol=1e-6)
