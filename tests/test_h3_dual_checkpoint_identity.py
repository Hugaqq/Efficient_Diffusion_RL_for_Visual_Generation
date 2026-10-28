"""CPU contract metadata isolation; not a real checkpoint/NCCL test."""
from dataclasses import replace
from tests.test_h3_mvp_cpu_binding import _fixture


def test_dual_checkpoint_contract_has_separate_dynamics_and_algorithm_identity(tmp_path,monkeypatch):
    h=_fixture(tmp_path,monkeypatch);b=h.binding
    original=b._contract()
    assert original.dynamics_state_schema=='h3.cps-dual-all-n.v1'
    b.profile=replace(b.profile,training=replace(b.profile.training,algorithm='dual_flow'))
    flow=b._contract()
    b.profile=replace(b.profile,training=replace(b.profile.training,algorithm='dual_flash'))
    flash=b._contract()
    assert flow.dynamics_state_schema==flash.dynamics_state_schema=='h3.dual-flow-sde.v1'
    assert len({original.algorithm_materialization_spec_id,flow.algorithm_materialization_spec_id,flash.algorithm_materialization_spec_id})==3
