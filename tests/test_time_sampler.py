
def test_time_sampler_issue1():
    """
    Test an issue where our sampler was sampling the same frame multiple times.
    """
    from geowatch.tasks.fusion.datamodules.temporal_sampling.sampler import TimeWindowSampler
    import numpy as np
    nan = np.nan

    kwargs = {
        'time_kernel': np.array([-31536000. ,         0. ,   2629756.8,  31536000. ]),
        'sensors': [None, None, None, None, None, None, None, None, None, None],
        'unixtimes': np.array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]),
        'time_window': 4,
        'update_rule': 'distribute',
        'affinity_type': 'uniform',
        'deterministic': False,
        'gamma': 1,
        'name': 'toy_video_2',
        'time_span': None,
        'allow_fewer': True,
    }

    sampler = TimeWindowSampler(**kwargs)
    sampled_idxs, info = sampler.sample(0, return_info=True)

    # There was a bug where we had samples returning the same index multiple
    # times, which should not be allowed. Test for this.
    assert len(sampled_idxs) == len(set(sampled_idxs)), (
        'Sampled indexes are not unique!'
    )
