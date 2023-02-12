import datetime as datetime_mod
import numbers
import dateutil
import time
import ubelt as ub
import math
from datetime import datetime as datetime_cls

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class TimeValueError(ValueError):
    ...


class TimeTypeError(TypeError):
    ...


def isoformat(dt, sep='T', timespec='seconds', pathsafe=True):
    """
    A path-safe version of datetime_cls.isotime() that returns a
    path-friendlier version of a ISO 8601 timestamp.

    Args:
        dt (datetime_cls): datetime to format
        pathsafe (bool):

    References:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

    SeeAlso:
        ubelt.timestamp

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> import datetime
        >>> items = []
        >>> dt = datetime_cls.now()
        >>> dt = ensure_timezone(dt, datetime_mod.timezone(datetime_mod.timedelta(hours=+5)))
        >>> items.append(dt)
        >>> dt = datetime_cls.utcnow()
        >>> items.append(dt)
        >>> dt = dt.replace(tzinfo=datetime_mod.timezone.utc)
        >>> items.append(dt)
        >>> dt = ensure_timezone(datetime_cls.now(), datetime_mod.timezone(datetime_mod.timedelta(hours=-5)))
        >>> items.append(dt)
        >>> dt = ensure_timezone(datetime_cls.now(), datetime_mod.timezone(datetime_mod.timedelta(hours=+5)))
        >>> items.append(dt)
        >>> print('items = {!r}'.format(items))
        >>> for dt in items:
        >>>     print('----')
        >>>     print('dt = {!r}'.format(dt))
        >>>     # ISO format is cool, but it doesnt give much control
        >>>     print(dt.isoformat())
        >>>     # Need a better version
        >>>     print(isoformat(dt))
        >>>     print(isoformat(dt, pathsafe=False))
    """
    if not pathsafe:
        return dt.isoformat(sep=sep, timespec=timespec)

    date_fmt = '%Y%m%d'
    if timespec == 'seconds':
        time_tmf = '%H%M%S'
    else:
        raise NotImplementedError(timespec)

    text = dt.strftime(''.join([date_fmt, sep, time_tmf]))
    if dt.tzinfo is not None:
        off = dt.utcoffset()
        off_seconds = off.total_seconds()
        if off_seconds == 0:
            # TODO: use codes for offsets to remove the plus sign if possible
            suffix = 'Z'
        elif off_seconds % 3600 == 0:
            tz_hour = int(off_seconds // 3600)
            suffix = '{:02d}'.format(tz_hour) if tz_hour < 0 else '+{:02d}'.format(tz_hour)
        else:
            suffix = _format_offset(off)
        text += suffix
    return text


def _format_offset(off):
    """
    Taken from CPython:
        https://github.com/python/cpython/blob/main/Lib/datetime_mod.py
    """
    s = ''
    if off is not None:
        if off.days < 0:
            sign = "-"
            off = -off
        else:
            sign = "+"
        hh, mm = divmod(off, datetime_mod.timedelta(hours=1))
        mm, ss = divmod(mm, datetime_mod.timedelta(minutes=1))
        s += "%s%02d:%02d" % (sign, hh, mm)
        if ss or ss.microseconds:
            s += ":%02d" % ss.seconds

            if ss.microseconds:
                s += '.%06d' % ss.microseconds
    return s


@profile
def coerce_datetime(data, default_timezone='utc', strict=False):
    """
    Parses a timestamp and always returns a timestamp with a timezone.
    If only a date is specified, the time is defaulted to 00:00:00
    If one is not discoverable a specified default is used.
    A None input is returned as-is unless strict is True.

    Args:
        data (None | str | datetime.datetime | datetime.date)
        default_timezone (str): defaults to utc.
        strict (bool): if True we error if the input data is None.

    Returns:
        None | datetime.datetime

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> assert coerce_datetime(None) is None
        >>> assert coerce_datetime('2020-01-01') == datetime_cls(2020, 1, 1, 0, 0, tzinfo=datetime_mod.timezone.utc)
        >>> assert coerce_datetime(datetime_cls(2020, 1, 1, 0, 0)) == datetime_cls(2020, 1, 1, 0, 0, tzinfo=datetime_mod.timezone.utc)
        >>> assert coerce_datetime(datetime_cls(2020, 1, 1, 0, 0).date()) == datetime_cls(2020, 1, 1, 0, 0, tzinfo=datetime_mod.timezone.utc)
    """
    if data is None:
        return data
    elif isinstance(data, str):
        # Canse use ubelt.timeparse(data, default_timezone=default_timezone) here.
        if data == 'now':
            dt = datetime_cls.utcnow()
        else:
            dt = dateutil.parser.parse(data)
    elif isinstance(data, datetime_cls):
        dt = data
    elif isinstance(data, datetime_mod.date):
        dt = dateutil.parser.parse(data.isoformat())
    elif isinstance(data, (float, int)):
        dt = datetime_cls.fromtimestamp(data)
    else:
        raise TimeTypeError('unhandled {}'.format(data))
    dt = ensure_timezone(dt, default=default_timezone)
    return dt


@profile
def ensure_timezone(dt, default='utc'):
    """
    Gives a datetime_mod a timezone (utc by default) if it doesnt have one

    Arguments:
        dt (datetime.datetime): the datetime to fix
        default (str): the timezone to use if it does not have one.

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> dt = ensure_timezone(datetime_cls.now(), datetime_mod.timezone(datetime_mod.timedelta(hours=+5)))
        >>> print('dt = {!r}'.format(dt))
        >>> dt = ensure_timezone(datetime_cls.utcnow())
        >>> print('dt = {!r}'.format(dt))
        >>> ensure_timezone(datetime_cls.utcnow(), 'utc')
        >>> ensure_timezone(datetime_cls.utcnow(), 'local')
    """
    if dt.tzinfo is not None:
        return dt
    else:
        if isinstance(default, datetime_mod.timezone):
            tzinfo = default
        else:
            if default == 'utc':
                tzinfo = datetime_mod.timezone.utc
            elif default == 'local':
                tzinfo = datetime_mod.timezone(datetime_mod.timedelta(seconds=-time.timezone))
            else:
                raise NotImplementedError
        return dt.replace(tzinfo=tzinfo)


@ub.memoize
def _time_unit_registery():
    import pint
    # Empty registry
    ureg = pint.UnitRegistry(None)
    ureg.define('second = []')
    ureg.define('minute = 60 * second')
    ureg.define('hour = 60 * minute')

    ureg.define('day = 24 * hour')
    ureg.define('month = 30.437 * day')
    ureg.define('year = 365 * day')

    ureg.define('min = minute')
    ureg.define('mon = month')
    ureg.define('sec = second')

    ureg.define('S = second')
    ureg.define('M = minute')
    ureg.define('H = hour')

    ureg.define('d = day')
    ureg.define('m = month')
    ureg.define('y = year')

    ureg.define('s = second')

    ureg.define('millisecond = second / 1000')
    ureg.define('microsecond = second / 1000000')

    ureg.define('ms = millisecond')
    ureg.define('us = microsecond')
    return ureg


_time_unit_registery()


@profile
def coerce_timedelta(delta):
    """
    TODO:
        move to a util

    Args:
        delta (str | int | float):
            If given as a string, attempt to parse out a time duration.
            Otherwise, interpret pure magnitudes in seconds.

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> variants = [
        >>>     ['year', 'y'],
        >>>     ['month', 'm', 'mon'],
        >>>     ['day', 'd', 'days'],
        >>>     ['hours', 'hour', 'h'],
        >>>     ['minutes', 'min', 'M'],
        >>>     ['second', 'S', 's', 'secs'],
        >>> ]
        >>> for vs in variants:
        >>>     print('vs = {!r}'.format(vs))
        >>>     ds = []
        >>>     for v in vs:
        >>>         d = coerce_timedelta(f'1{v}')
        >>>         ds.append(d)
        >>>         d = coerce_timedelta(f'1 {v}')
        >>>         ds.append(d)
        >>>     assert ub.allsame(ds)
        >>>     print('ds = {!r}'.format(ds))
        >>> print(coerce_timedelta(10.3))
        >>> print(coerce_timedelta('1y'))
        >>> print(coerce_timedelta('1m'))
        >>> print(coerce_timedelta('1d'))
        >>> print(coerce_timedelta('1H'))
        >>> print(coerce_timedelta('1M'))
        >>> print(coerce_timedelta('1S'))
        >>> print(coerce_timedelta('1year'))
        >>> print(coerce_timedelta('1month'))
        >>> print(coerce_timedelta('1day'))
        >>> print(coerce_timedelta('1hour'))
        >>> print(coerce_timedelta('1min'))
        >>> print(coerce_timedelta('1sec'))
        >>> print(coerce_timedelta('1microsecond'))
        >>> print(coerce_timedelta('1milliseconds'))
        >>> print(coerce_timedelta('1ms'))
        >>> print(coerce_timedelta('1us'))

    References:
        https://docs.python.org/3.4/library/datetime_mod.html#strftime-strptime-behavior
    """
    if isinstance(delta, str):
        try:
            delta = float(delta)
        except ValueError:
            ...

    if isinstance(delta, datetime_mod.timedelta):
        ...
    elif isinstance(delta, numbers.Number):
        try:
            delta = datetime_mod.timedelta(seconds=delta)
        except ValueError:
            if isinstance(delta, float) and math.isnan(delta):
                raise TimeValueError('Cannot coerce nan to a timedelta')
            raise
    elif isinstance(delta, str):

        try:
            ureg = _time_unit_registery()
            seconds = ureg.parse_expression(delta).to('seconds').m
            # timedelta apparently does not have resolution higher than
            # microseconds.
            # https://stackoverflow.com/questions/10611328/strings-ns
            # https://bugs.python.org/issue15443
            delta = datetime_mod.timedelta(seconds=seconds)
        except Exception:
            # Separate the expression into a magnitude and a unit
            import re
            expr_pat = re.compile(
                r'^(?P<magnitude>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
                '(?P<spaces> *)'
                '(?P<unit>.*)$')
            match = expr_pat.match(delta.strip())
            if match:
                parsed = match.groupdict()
                unit = parsed.get('unit', '')
                magnitude = parsed.get('magnitude', '')
            else:
                unit = None
                magnitude = None

            if unit == 'y':
                delta = datetime_mod.timedelta(days=365 * float(magnitude))
            elif unit == 'd':
                delta = datetime_mod.timedelta(days=1 * float(magnitude))
            elif unit == 'm':
                delta = datetime_mod.timedelta(days=30.437 * float(magnitude))
            elif unit == 'H':
                delta = datetime_mod.timedelta(hours=float(magnitude))
            elif unit == 'M':
                delta = datetime_mod.timedelta(minutes=float(magnitude))
            elif unit == 'S':
                delta = datetime_mod.timedelta(seconds=float(magnitude))
            else:
                import pytimeparse  #
                print('warning: pytimeparse fallback')
                seconds = pytimeparse.parse(delta)
                if seconds is None:
                    raise Exception(delta)
                delta = datetime_mod.timedelta(seconds=seconds)
                return delta
    else:
        raise TimeTypeError(type(delta))
    return delta
