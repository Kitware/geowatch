import torch
import ubelt as ub
import numpy as np


def debug_shapes(data):
    # Ported from netharn _debug_inbatch_shapes
    print('len(inbatch) = {}'.format(len(data)))
    extensions = ub.util_format.FormatterExtensions()

    @extensions.register((torch.Tensor, np.ndarray))
    def format_shape(data, **kwargs):
        return ub.urepr(dict(type=str(type(data).__name__), shape=data.shape), nl=0, sv=1)
    print('data = ' + ub.urepr(data, extensions=extensions, nl=-1, sort=0))


def shape_summary(data, flat=0):
    # Alternative
    walker = ub.IndexableWalker(data, list_cls=(list,))
    summary = {}
    for path, value in walker:
        if not isinstance(value, (list, dict)):
            if isinstance(value, np.ndarray):
                path = path + ['shape']
                value = value.shape
            elif isinstance(value, torch.Tensor):
                path = path + ['shape']
                value = value.shape
            key = '.'.join(map(str, path))
            summary[key] = value
    return summary
