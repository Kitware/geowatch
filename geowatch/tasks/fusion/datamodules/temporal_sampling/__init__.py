"""
Temporal sampling submodule

mkinit ~/code/watch/geowatch/tasks/fusion/datamodules/temporal_sampling/__init__.py --lazy_loader -w
"""

import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'affinity',
        'exceptions',
        'plots',
        'sampler',
        'utils',
    },
    submod_attrs={
        'affinity': [
            'affinity_sample',
            'cython_aff_samp_mod',
            'hard_frame_affinity',
            'hard_time_sample_pattern',
            'soft_frame_affinity',
        ],
        'exceptions': [
            'TimeSampleError',
        ],
        'plots': [
            'plot_dense_sample_indices',
            'plot_temporal_sample',
            'plot_temporal_sample_indices',
            'show_affinity_sample_process',
        ],
        'sampler': [
            'CommonSamplerMixin',
            'MultiTimeWindowSampler',
            'TimeWindowSampler',
        ],
        'utils': [
            'guess_missing_unixtimes',
        ],
    },
)

__all__ = ['CommonSamplerMixin', 'MultiTimeWindowSampler', 'TimeSampleError',
           'TimeWindowSampler', 'affinity', 'affinity_sample',
           'cython_aff_samp_mod', 'exceptions', 'guess_missing_unixtimes',
           'hard_frame_affinity', 'hard_time_sample_pattern',
           'plot_dense_sample_indices', 'plot_temporal_sample',
           'plot_temporal_sample_indices', 'plots', 'sampler',
           'show_affinity_sample_process', 'soft_frame_affinity', 'utils']
