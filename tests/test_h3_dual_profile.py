import pytest
from visual_rl.composition.config.h3_profile import H3Training


def test_experimental_modes_have_distinct_checkpoint_semantics():
    cps=H3Training().payload();flow=H3Training(algorithm='dual_flow').payload();flash=H3Training(algorithm='dual_flash').payload()
    assert cps['score']=='cps_proxy_negative_mse' and 'algorithm' not in cps
    assert flow['score']==flash['score']=='dual_gaussian_element_mean_v1'
    assert flow['policy_steps']=='all_stochastic' and flash['policy_steps']=='one_group_selected'
    assert flow['rectification']=='none' and flash['rectification']=='per_stream_world_mean_gradient_v1'
    assert flow != flash != cps
    with pytest.raises(ValueError):H3Training(algorithm='typo')
    with pytest.raises(ValueError):H3Training(algorithm='dual_flash',num_steps=1)
