import ubelt as ub
import functools


try:
    from lark import Transformer
except ImportError:
    class Transformer:
        pass


try:
    cache = functools.cache
except AttributeError:
    cache = ub.memoize


# For common constructs see:
# https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark
RESOLUTION_GRAMMAR_PARTS = ub.codeblock(
    '''
    // Resolution parts of the grammar
    magnitude: NUMBER

    unit: WORD

    resolution: magnitude WS* unit

    %import common.NUMBER
    %import common.WS
    %import common.WORD
    ''')


RESOLVED_WINDOW_GRAMMAR = ub.codeblock(
    r'''
    // RESOLVED WINDOW GRAMMAR
    ?start: resolved_window

    window_1d_dim: NUMBER

    window_2d_dim: NUMBER WS* ("x" | ",") WS* NUMBER

    window: window_1d_dim | window_2d_dim

    resolved_window: window WS* "@" WS* resolution

    %import common.NUMBER
    %import common.WS
    ''') + '\n' + RESOLUTION_GRAMMAR_PARTS


RESOLVED_SCALAR_GRAMMAR = ub.codeblock(
    r'''
    // RESOLVED WINDOW GRAMMAR
    ?start: resolved_scalar

    resolved_scalar: NUMBER WS* "@" WS* resolution

    %import common.NUMBER
    %import common.WS
    ''') + '\n' + RESOLUTION_GRAMMAR_PARTS


def _int_or_float(x):
    try:
        return int(x)
    except Exception:
        return float(x)


class ExtendedTransformer(Transformer):
    """
    Enriches the Transformer with parse and parser classmethods which rely on a
    __grammar__ attribute
    """

    @classmethod
    @cache
    def parser(cls):
        # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf
        import lark
        try:
            import lark_cython
            parser = lark.Lark(cls.__grammar__,  start='start', parser='lalr', _plugins=lark_cython.plugins)
        except ImportError:
            parser = lark.Lark(cls.__grammar__,  start='start', parser='lalr')
        return parser

    @classmethod
    @cache
    def parse(cls, text):
        parser = cls.parser()
        tree = parser.parse(text)
        transformed = cls().transform(tree)
        return transformed


class ResolvedTransformer(ExtendedTransformer):

    def magnitude(self, items):
        d = _int_or_float(items[0].value)
        return d

    def unit(self, items):
        return items[0].value

    def resolution(self, items):
        info = {
            'mag': items[0],
            'unit': items[-1],
        }
        return info


class ResolvedWindowTransformer(ResolvedTransformer):
    __grammar__ = RESOLVED_WINDOW_GRAMMAR

    def window_1d_dim(self, items):
        d1 = _int_or_float(items[0].value)
        info = (d1, d1)
        return info

    def window_2d_dim(self, items):
        d1 = _int_or_float(items[0].value)
        d2 = _int_or_float(items[-1].value)
        info = (d1, d2)
        return info

    def window(self, items):
        return items[0]

    def resolved_window(self, items):
        info = {
            'window': items[0],
            'resolution': items[-1],
        }
        return info


class ResolvedScalarTransformer(ResolvedTransformer):
    __grammar__ = RESOLVED_SCALAR_GRAMMAR

    def resolved_scalar(self, items):
        info = {
            'scalar': _int_or_float(items[0].value),
            'resolution': items[-1],
        }
        return info


class Resolved:
    """
    Base class for all resolved objects
    """

    @classmethod
    def parse(cls, data):
        if isinstance(data, str):
            return cls(**cls.__transformer__.parse(data))
        else:
            raise TypeError(type(data))

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            return data
        elif isinstance(data, str):
            return cls.parse(data)
        elif isinstance(data, dict):
            return cls(**data)
        else:
            raise TypeError(type(data))


class ResolvedScalar(Resolved, ub.NiceRepr):
    """
    Example:
        >>> from watch.utils.util_resolution import *  # NOQA
        >>> resolved_scalar = "128@10GSD"
        >>> resolved_scalar = ResolvedScalar.parse(resolved_scalar)
        >>> print('resolved_scalar = {}'.format(ub.urepr(resolved_scalar, sv=1, nl=1)))
        resolved_scalar = <ResolvedScalar(128 @ {'mag': 10, 'unit': 'GSD'})>
        >>> resolved_scalar = "128  @  10  GSD"
        >>> resolved_scalar = ResolvedScalar.parse(resolved_scalar)
        >>> print('resolved_scalar = {}'.format(ub.urepr(resolved_scalar, sv=1, nl=1)))
        resolved_scalar = <ResolvedScalar(128 @ {'mag': 10, 'unit': 'GSD'})>
    """
    __transformer__ = ResolvedScalarTransformer

    def __init__(self, scalar, resolution):
        self.scalar = scalar
        self.resolution = resolution

    def __nice__(self):
        return (f'{self.scalar} @ {self.resolution}')

    def at_resolution(self, new_resolution):
        '''
        Update the resolution

        Example:
            >>> new_resolution = {'mag': 1, 'unit': 'GSD'}
            >>> self = ResolvedScalar.parse("128@10GSD")
            >>> print(self.at_resolution(new_resolution))
            >>> print(self.at_resolution({'mag': 20, 'unit': 'GSD'}))
            <ResolvedScalar(1280.0 @ {'mag': 1, 'unit': 'GSD'})>
            <ResolvedScalar(64.0 @ {'mag': 20, 'unit': 'GSD'})>
        '''
        scale_factor = self.resolution['mag'] / new_resolution['mag']
        new = self.__class__(self.scalar * scale_factor, new_resolution)
        return new


class ResolvedWindow(Resolved, ub.NiceRepr):
    """
    Parse a window size at a particular resolution

    Example:
        >>> from watch.utils.util_resolution import *  # NOQA
        >>> resolved_window = "128x128@10GSD"
        >>> resolved_window = ResolvedWindow.parse(resolved_window)
        >>> print('resolved_window = {}'.format(ub.urepr(resolved_window, nl=1, sv=1)))
        resolved_window = <ResolvedWindow((128, 128) @ {'mag': 10, 'unit': 'GSD'})>
        >>> resolved_window = "128  ,  128  @  10  GSD"
        >>> resolved_window = ResolvedWindow.parse(resolved_window)
        >>> print('resolved_window = {}'.format(ub.urepr(resolved_window, nl=1, sv=1)))
        resolved_window = <ResolvedWindow((128, 128) @ {'mag': 10, 'unit': 'GSD'})>
        >>> from watch.utils.util_resolution import *  # NOQA
        >>> resolved_window = "128@10GSD"
        >>> resolved_window = ResolvedWindow.parse(resolved_window)
        >>> print('resolved_window = {}'.format(ub.urepr(resolved_window, nl=1, sv=1)))
        resolved_window = <ResolvedWindow((128, 128) @ {'mag': 10, 'unit': 'GSD'})>

    """
    __transformer__ = ResolvedWindowTransformer

    def __init__(self, window, resolution):
        self.window = window
        self.resolution = resolution

    def at_resolution(self, new_resolution):
        '''
        Update the resolution

        Example:
            >>> from watch.utils.util_resolution import *  # NOQA
            >>> new_resolution = {'mag': 1, 'unit': 'GSD'}
            >>> self = ResolvedWindow.parse("128x64@10GSD")
            >>> print(self.at_resolution(new_resolution))
            >>> print(self.at_resolution({'mag': 20, 'unit': 'GSD'}))
            <ResolvedWindow((1280.0, 640.0) @ {'mag': 1, 'unit': 'GSD'})>
            <ResolvedWindow((64.0, 32.0) @ {'mag': 20, 'unit': 'GSD'})>
        '''
        scale_factor = self.resolution['mag'] / new_resolution['mag']
        w, h = self.window
        new_window = (w * scale_factor, h * scale_factor)
        new = self.__class__(new_window, new_resolution)
        return new

    def __nice__(self):
        return (f'{self.window} @ {self.resolution}')
