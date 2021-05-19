import datetime
import ubelt as ub


class Timestamp(ub.NiceRepr):
    """
    Example:
        self = Timestamp.coerce('1970/01/01')
    """
    def __init__(self, time):
        self.time = time

    def __nice__(self):
        return str(self.time)

    def __json__(self):
        return self.to_iso8601()

    @classmethod
    def coerce(cls, data):
        if isinstance(data, str):
            # https://en.wikipedia.org/wiki/ISO_8601
            # Handle different known formats
            accepted_formats = [
                '%Y-%m-%dT%H%M%S',
                '%Y/%m/%dT%H%M%S',
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%Y%m%d',
            ]
            found = None
            for fmt in accepted_formats:
                try:
                    found = datetime.datetime.strptime(data, fmt)
                    break
                except Exception:
                    pass
            if found is None:
                raise ValueError('Unknown timestamp format {}'.format(data))
            time = found
        elif isinstance(data, datetime.datetime):
            time = data
        else:
            raise Exception
        self = cls(time)
        return self

    def to_datetime(self):
        return self.time

    def to_iso8601(self):
        return self.time.strftime('%Y-%m-%dT%H%M%S')
