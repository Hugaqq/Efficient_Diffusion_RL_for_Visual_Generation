"""CPU numerical contract; no claim of end-to-end or GPU validation."""
import pytest
import torch
from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.models.h3_joint import H3JointLayout


def fixture():
    d = H3DualFlowSDE(H3JointLayout(1,2,1,3),H3ScheduleSnapshot.build(4))
    x = torch.tensor([[.1,.2,.3,.4,.5]],dtype=torch.float16)
    return d,x


def test_sampling_replay_joint_density_and_stream_gradients():
    d,x=fixture()
    prediction=torch.tensor([[.1,-.2,.3,-.1,.2]],requires_grad=True)
    def sample():
        return d.sample(x,prediction,1,video_generator=torch.Generator().manual_seed(42),audio_generator=torch.Generator().manual_seed(43))
    action=sample()
    assert torch.equal(action,sample()) and not action.requires_grad
    joint, (video,audio), (cv,ca)=d.evaluate(x,action,prediction,1)
    assert torch.allclose(joint,video*.4+audio*.6)
    assert not torch.equal(cv,ca)
    raw=torch.autograd.grad(joint.sum(),prediction,retain_graph=True)[0]
    corrected=d.rectified_score(joint,(video,audio),(cv,ca),(cv/2,ca/3))
    assert torch.equal(corrected,joint)
    grad=torch.autograd.grad(corrected.sum(),prediction)[0]
    assert torch.allclose(grad[:,:2],raw[:,:2]*2,atol=1e-6)
    assert torch.allclose(grad[:,2:],raw[:,2:]*3,atol=1e-6)


def test_deterministic_no_rng_and_last_transition_stochastic_valid():
    d,x=fixture(); pred=torch.ones_like(x);before=torch.random.get_rng_state().clone()
    y=d.deterministic_step(x,pred,3)
    assert torch.equal(before,torch.random.get_rng_state())
    v,u,a,b=d.schedule.transition(3)
    assert torch.allclose(y.float(),torch.cat((x[:,:2].float()+u-v,x[:,2:].float()+b-a),dim=1),atol=0.001)
    action=d.sample(x,pred,3,video_generator=torch.Generator().manual_seed(1),audio_generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(d.evaluate(x,action,pred,3)[0]).all()
    with pytest.raises(IndexError): d.evaluate(x,action,pred,4)
    with pytest.raises(ValueError): H3DualFlowSDE(d.layout,H3ScheduleSnapshot.build(1))


def test_reject_invalid_generator_and_normalization():
    d,x=fixture();g=torch.Generator().manual_seed(1)
    with pytest.raises(ValueError):d.sample(x,x,0,video_generator=g,audio_generator=g)
    a=d.sample(x,x,0,video_generator=g,audio_generator=torch.Generator().manual_seed(2))
    joint,parts,coefs=d.evaluate(x,a,x,0)
    with pytest.raises(ValueError):d.rectified_score(joint,parts,coefs,(torch.tensor(0.),torch.tensor(1.)))
