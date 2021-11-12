import dateutil
import datetime


def isoformat(dt, sep='T', timespec='seconds', pathsafe=True):
    """
    A path-safe version of datetime.datetime.isotime() that returns a
    path-friendlier version of a ISO 8601 timestamp.

    Args:
        dt (datetime.datetime): datetime to format
        pathsafe (bool):

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
        time_tmf = '%Y%m%d'
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


def coerce_datetime(data):
    if data is None:
        return data
    else:
        dt = dateutil.parser.parse(data)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
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
