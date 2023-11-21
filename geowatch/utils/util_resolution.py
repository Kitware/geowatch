import ubelt as ub
import functools
import numbers
from kwcoco.util import dict_proxy2


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
    // Resolution parts of the grammar.
    magnitude: NUMBER

    unit: WORD

    resolved_unit: magnitude WS* unit

    %import common.NUMBER
    %import common.WS
    %import common.WORD
    ''')

RESOLVED_UNIT_GRAMMAR = ub.codeblock(
    r'''
    // RESOLVED WINDOW GRAMMAR. Eg. 2GSD
    ?start: resolved_unit
    ''') + '\n' + RESOLUTION_GRAMMAR_PARTS


RESOLVED_SCALAR_GRAMMAR = ub.codeblock(
    r'''
    // RESOLVED WINDOW GRAMMAR. 128 @ 2GSD
    ?start: resolved_scalar

    resolved_scalar: NUMBER WS* "@" WS* resolved_unit

    ''') + '\n' + RESOLUTION_GRAMMAR_PARTS


RESOLVED_WINDOW_GRAMMAR = ub.codeblock(
    r'''
    // RESOLVED WINDOW GRAMMAR. E.g. 128x128 @ 2GSD
    ?start: resolved_window

    window_1d_dim: NUMBER

    window_2d_dim: NUMBER WS* ("x" | ",") WS* NUMBER

    window: window_1d_dim | window_2d_dim

    resolved_window: window WS* "@" WS* resolved_unit
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
    __grammar__ = NotImplemented

    @classmethod
    @cache
    def parser(cls):
        # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf
        import lark
        try:
            import lark_cython
            parser = lark.Lark(cls.__grammar__, start='start', parser='lalr', _plugins=lark_cython.plugins)
        except ImportError:
            parser = lark.Lark(cls.__grammar__, start='start', parser='lalr')
        return parser

    @classmethod
    @cache
    def parse(cls, text):
        """
        Parses the text and transforms the output tree based on
        __grammar__
        """
        parser = cls.parser()
        tree = parser.parse(text)
        self = cls()
        transformed = self.transform(tree)
        return transformed


class ResolvedTransformer(ExtendedTransformer):
    """
    Base class for resolving a resolution 1D scalar or 2D window.
    """

    def magnitude(self, items):
        d = _int_or_float(items[0].value)
        return d

    def unit(self, items):
        return items[0].value

    def resolved_unit(self, items):
        info = {
            'mag': items[0],
            'unit': items[-1],
        }
        return info


class ResolvedUnitTransformer(ResolvedTransformer):
    """
    Transform for :class:`ResolvedUnit`
    """
    __grammar__ = RESOLVED_UNIT_GRAMMAR


class ResolvedScalarTransformer(ResolvedTransformer):
    """
    Transform for :class:`ResolvedScalar`
    """
    __grammar__ = RESOLVED_SCALAR_GRAMMAR

    def resolved_scalar(self, items):
        info = {
            'scalar': _int_or_float(items[0].value),
            'resolution': items[-1],
        }
        return info


class ResolvedWindowTransformer(ResolvedTransformer):
    """
    Transform for :class:`ResolvedWindow`
    """
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


class Resolved(dict_proxy2.DictProxy2):
    """
    Base class for all resolved objects.
    Must define the ``__transformer__`` attribute.
    """
    __transformer__ = NotImplemented

    @classmethod
    def parse(cls, data):
        if isinstance(data, str):
            text = data
            transformer_cls = cls.__transformer__
            attrs = transformer_cls.parse(text)
            return cls(**attrs)
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


class ResolvedUnit(Resolved, ub.NiceRepr):
    """
    Holds just the unit information (e.g. X GSD)

    Example:
        >>> from geowatch.utils.util_resolution import *  # NOQA
        >>> self = ResolvedUnit.parse('8GSD')
        >>> print('self = {}'.format(ub.urepr(self, nl=1, si=1)))
        self = <ResolvedUnit(8 GSD)>
    """
    __transformer__ = ResolvedUnitTransformer

    def __init__(self, mag, unit):
        self.mag = mag
        self.unit = unit
        self._proxy = {
            'mag': mag,
            'unit': unit,
        }

    def __eq__(self, other):
        if self.unit != other.unit:
            raise TypeError(f'incomparable units: {self.unit}, {other.unit}')
        return self.mag == other.mag

    def __nice__(self):
        return (f'{self.mag} {self.unit}')

    @classmethod
    def coerce(cls, data, default_unit=None):
        """
        Example:
            >>> from geowatch.utils.util_resolution import *  # NOQA
            >>> self1 = ResolvedUnit.coerce(8, default_unit='GSD')
            >>> self2 = ResolvedUnit.coerce('8', default_unit='GSD')
            >>> self3 = ResolvedUnit.coerce('8GSD')
            >>> assert self1 == self2
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     ResolvedUnit.coerce(8)
        """
        is_string = isinstance(data, str)
        if is_string:
            # Allow the input to be given as a numeric string
            try:
                mag = _int_or_float(data)
            except Exception:
                ...
            else:
                data = mag
                is_string = False

        if isinstance(data, str):
            self = cls.parse(data)
        elif isinstance(data, numbers.Number):
            if default_unit is None:
                raise ValueError(
                    'must provide a default unit if numberic input is given')
            self = cls(data, default_unit)
        else:
            raise TypeError(type(data))
        return self


class ResolvedScalar(Resolved, ub.NiceRepr):
    """
    Example:
        >>> from geowatch.utils.util_resolution import *  # NOQA
        >>> self1 = ResolvedScalar.parse("128@10GSD")
        >>> self2 = ResolvedScalar.parse("128  @  10  GSD")
        >>> print('self1 = {}'.format(ub.urepr(self1, sv=1, nl=1)))
        >>> print('self2 = {}'.format(ub.urepr(self2, sv=1, nl=1)))
        self1 = <ResolvedScalar(128 @ 10 GSD)>
        self2 = <ResolvedScalar(128 @ 10 GSD)>
    """
    __transformer__ = ResolvedScalarTransformer

    def __init__(self, scalar, resolution):
        self.scalar = scalar
        self.resolution = ResolvedUnit(**resolution)
        self._proxy = {
            'scalar': scalar,
            'resolution': resolution,
        }

    def __nice__(self):
        return (f'{self.scalar} @ {self.resolution.__nice__()}')

    def at_resolution(self, new_resolution):
        '''
        Update the resolution

        Args:
            new_resolution (dict | ResolvedUnit):
                new base resolution unit to use.

        Returns:
            ResolvedScalar:
                The same scalar but in terms of the new resolution.

        Example:
            >>> new_resolution = {'mag': 1, 'unit': 'GSD'}
            >>> self = ResolvedScalar.parse("128@10GSD")
            >>> print(self.at_resolution(new_resolution))
            >>> print(self.at_resolution({'mag': 20, 'unit': 'GSD'}))
            <ResolvedScalar(1280.0 @ 1 GSD)>
            <ResolvedScalar(64.0 @ 20 GSD)>
        '''
        scale_factor = self.resolution['mag'] / new_resolution['mag']
        new = self.__class__(self.scalar * scale_factor, new_resolution)
        return new


class ResolvedWindow(Resolved, ub.NiceRepr):
    """
    Parse a window size at a particular resolution

    Example:
        >>> from geowatch.utils.util_resolution import *  # NOQA
        >>> data = "128x128@10GSD"
        >>> self1 = ResolvedWindow.parse(data)
        >>> self2 = ResolvedWindow.parse("128  ,  128  @  10  GSD")
        >>> self3 = ResolvedWindow.parse("128@10GSD")
        >>> print('self1 = {}'.format(ub.urepr(self1, nl=1, sv=1)))
        >>> print('self2 = {}'.format(ub.urepr(self2, nl=1, sv=1)))
        >>> print('self3 = {}'.format(ub.urepr(self3, nl=1, sv=1)))
        self1 = <ResolvedWindow((128, 128) @ 10 GSD)>
        self2 = <ResolvedWindow((128, 128) @ 10 GSD)>
        self3 = <ResolvedWindow((128, 128) @ 10 GSD)>
    """
    __transformer__ = ResolvedWindowTransformer

    def __init__(self, window, resolution):
        self.window = window
        self.resolution = ResolvedUnit(**resolution)
        self._proxy = {
            'window': window,
            'resolution': resolution,
        }

    def at_resolution(self, new_resolution):
        """
        Update the resolution

        Args:
            new_resolution (dict | ResolvedUnit):
                new base resolution unit to use.

        Returns:
            ResolvedWindow:
                The same window but in terms of the new resolution.

        Example:
            >>> from geowatch.utils.util_resolution import *  # NOQA
            >>> new_resolution = {'mag': 1, 'unit': 'GSD'}
            >>> self = ResolvedWindow.parse("128x64@10GSD")
            >>> print(self.at_resolution(new_resolution))
            >>> print(self.at_resolution({'mag': 20, 'unit': 'GSD'}))
            <ResolvedWindow((1280.0, 640.0) @ 1 GSD)>
            <ResolvedWindow((64.0, 32.0) @ 20 GSD)>
        """
        scale_factor = self.resolution['mag'] / new_resolution['mag']
        w, h = self.window
        new_window = (w * scale_factor, h * scale_factor)
        new = self.__class__(new_window, new_resolution)
        return new

    def __nice__(self):
        return (f'{self.window} @ {self.resolution.__nice__()}')
