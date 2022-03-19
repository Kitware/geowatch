import dateutil
import datetime


def isoformat(dt, sep='T', timespec='seconds', pathsafe=True):
    """
    A path-safe version of datetime.datetime.isotime() that returns a
    path-friendlier version of a ISO 8601 timestamp.

    Args:
        dt (datetime.datetime): datetime to format
        pathsafe (bool):

    References:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> import datetime
        >>> items = []
        >>> dt = datetime.datetime.now()
        >>> dt = ensure_timezone(dt, datetime.timezone(datetime.timedelta(hours=+5)))
        >>> items.append(dt)
        >>> dt = datetime.datetime.utcnow()
        >>> items.append(dt)
        >>> dt = dt.replace(tzinfo=datetime.timezone.utc)
        >>> items.append(dt)
        >>> dt = ensure_timezone(datetime.datetime.now(), datetime.timezone(datetime.timedelta(hours=-5)))
        >>> items.append(dt)
        >>> dt = ensure_timezone(datetime.datetime.now(), datetime.timezone(datetime.timedelta(hours=+5)))
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
        https://github.com/python/cpython/blob/main/Lib/datetime.py
    """
    s = ''
    if off is not None:
        if off.days < 0:
            sign = "-"
            off = -off
        else:
            sign = "+"
        hh, mm = divmod(off, datetime.timedelta(hours=1))
        mm, ss = divmod(mm, datetime.timedelta(minutes=1))
        s += "%s%02d:%02d" % (sign, hh, mm)
        if ss or ss.microseconds:
            s += ":%02d" % ss.seconds

            if ss.microseconds:
                s += '.%06d' % ss.microseconds
    return s


def coerce_datetime(data, default_timezone='utc'):
    """
    Parses a timestamp and always returns a timestamp with a timezone

    If only a date is specified, the time is defaulted to 00:00:00

    If one is not discoverable a specified default is used.

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> assert coerce_datetime(None) is None
        >>> assert coerce_datetime('2020-01-01') == datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        >>> assert coerce_datetime(datetime.datetime(2020, 1, 1, 0, 0)) == datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        >>> assert coerce_datetime(datetime.datetime(2020, 1, 1, 0, 0).date()) == datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    if data is None:
        return data
    elif isinstance(data, str):
        dt = dateutil.parser.parse(data)
    elif isinstance(data, datetime.datetime):
        dt = data
    elif isinstance(data, datetime.date):
        dt = dateutil.parser.parse(data.isoformat())
    else:
        raise TypeError('unhandled {}'.format(data))
    dt = ensure_timezone(dt, default=default_timezone)
    return dt


def ensure_timezone(dt, default='utc'):
    """
    Gives a datetime a timezone (utc by default) if it doesnt have one

    Example:
        >>> from watch.utils.util_time import *  # NOQA
        >>> dt = ensure_timezone(datetime.datetime.now(), datetime.timezone(datetime.timedelta(hours=+5)))
        >>> print('dt = {!r}'.format(dt))
        >>> dt = ensure_timezone(datetime.datetime.utcnow())
        >>> print('dt = {!r}'.format(dt))
    """
    if dt.tzinfo is not None:
        return dt
    else:
        if isinstance(default, datetime.timezone):
            tzinfo = default
        else:
            if default == 'utc':
                tzinfo = datetime.timezone.utc
            else:
                raise NotImplementedError
        return dt.replace(tzinfo=tzinfo)


def coerce_timedelta(delta):
    """
    TODO:
        move to a util
    Example:
        coerce_timedelta('1y')
        coerce_timedelta('1m')
        coerce_timedelta('1d')
        coerce_timedelta('1H')
        coerce_timedelta('1M')
        coerce_timedelta('1S')
        coerce_timedelta(10.3)

    References:
        https://docs.python.org/3.4/library/datetime.html#strftime-strptime-behavior
    """
    if isinstance(delta, (int, float)):
        delta = datetime.timedelta(seconds=delta)
    elif isinstance(delta, str):
        # TODO: better coercion function
        if delta.endswith('y'):
            delta = datetime.timedelta(days=365 * float(delta[:-1]))
        elif delta.endswith('d'):
            delta = datetime.timedelta(days=1 * float(delta[:-1]))
        elif delta.endswith('m'):
            delta = datetime.timedelta(days=30.437 * float(delta[:-1]))
        elif delta.endswith('H'):
            delta = datetime.timedelta(hours=float(delta[:-1]))
        elif delta.endswith('M'):
            delta = datetime.timedelta(minutes=float(delta[:-1]))
        elif delta.endswith('S'):
            delta = datetime.timedelta(seconds=float(delta[:-1]))
        else:
            import pytimeparse  #
            pytimeparse.parse(delta)
    elif isinstance(delta, datetime.timedelta):
        pass
    else:
        raise TypeError(type(delta))
    return delta
