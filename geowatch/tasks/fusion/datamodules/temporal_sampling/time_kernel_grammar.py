"""
A grammar to allow the user to define one or more time kernels.
Because we use "," as the main separator, groups must be enclosed in parans.

Example:
    >>> time_kernel = '-1y,-30d,-1d,0,1d,30d,1y'
    >>> time_kernel = '(-1y,-30d,-1d,0,1d,30d,1y),(0),(-1,0,1)'
"""
import ubelt as ub
import functools


try:
    cache = functools.cache
except AttributeError:
    cache = ub.memoize

try:
    from lark import Transformer
except ImportError:
    class Transformer:
        pass


# For common constructs see:
# https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark
MULTI_TIME_KERNEL_GRAMMAR = ub.codeblock(
    '''
    // TIME_KERNEL_GRAMMAR
    ?start: multi_kernel

    // A delta is a number optionally followed by some unit
    DELTA: (NUMBER | SIGNED_NUMBER) LETTER*

    // A bare kernel is a sequence of deltas
    bare_kernel: DELTA ("," DELTA)*

    paren_kernel: "(" bare_kernel ")"

    multi_kernel : bare_kernel | (paren_kernel ("," paren_kernel)*)

    %import common.NUMBER
    %import common.SIGNED_NUMBER
    %import common.LETTER
    %import common.CNAME
    ''')


class MultiTimeKernelTransformer(Transformer):
    """
    """

    def bare_kernel(self, items):
        from kwutil.util_time import coerce_timedelta
        import numpy as np
        kernel = [coerce_timedelta(item.value).total_seconds() for item in items]
        kernel = np.array(kernel)
        diffs = np.diff(kernel)
        if not np.all(diffs >= 0):
            print(f'parse error items={items}')
            print(f'kernel={kernel}')
            print(f'diffs={diffs}')
            raise ValueError('time_kernel inputs must be in ascending order')
        return kernel

    def paren_kernel(self, items):
        return items[0]

    def multi_kernel(self, items):
        return items


@cache
def _global_multi_time_kernel_parser():
    # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf
    import lark
    try:
        import lark_cython
        parser = lark.Lark(MULTI_TIME_KERNEL_GRAMMAR, start='start', parser='lalr', _plugins=lark_cython.plugins)
    except ImportError:
        parser = lark.Lark(MULTI_TIME_KERNEL_GRAMMAR, start='start', parser='lalr')
    return parser


@cache
def parse_multi_time_kernel(time_kernel):
    """
    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel_grammar import *  # NOQA
        >>> time_kernel = '-3h,-1h,-1min,0,1min,1h,3h'
        >>> multi_kernel = parse_multi_time_kernel(time_kernel)
        >>> print('multi_kernel = {}'.format(ub.urepr(multi_kernel, nl=1)))
        multi_kernel = [
            np.array([-10800.,  -3600.,    -60.,      0.,     60.,   3600.,  10800.], dtype=np.float64),
        ]
        >>> time_kernel = '(-1d,-3h,-1h,0,1h,3h,1d),(0),(-1,0,1)'
        >>> multi_kernel = parse_multi_time_kernel(time_kernel)
        >>> print('multi_kernel = {}'.format(ub.urepr(multi_kernel, nl=1)))
        multi_kernel = [
            np.array([-86400., -10800.,  -3600.,      0.,   3600.,  10800.,  86400.], dtype=np.float64),
            np.array([0.], dtype=np.float64),
            np.array([-1.,  0.,  1.], dtype=np.float64),
        ]
    """
    parser = _global_multi_time_kernel_parser()
    tree = parser.parse(time_kernel)
    multi_kernel = MultiTimeKernelTransformer().transform(tree)
    return multi_kernel
