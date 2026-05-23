def test_epoch_k_repeat_sampler_tags_epoch():
    from visual_rl.datasets.samplers import EpochKRepeatSampler

    sampler = EpochKRepeatSampler(dataset_size=10, batch_size=4, k=2, seed=123)
    sampler.set_epoch(7)
    indices = sampler.next_indices()
    assert len(indices) == 4
    assert all(epoch == 7 for epoch, _idx in indices)

