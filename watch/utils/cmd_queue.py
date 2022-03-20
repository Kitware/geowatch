import ubelt as ub


class Job(ub.NiceRepr):
    """
    Base class for a job
    """
    def __init__(self, command=None, name=None, depends=None, **kwargs):
        if depends is not None and not ub.iterable(depends):
            depends = [depends]
        self.name = name
        self.command = command
        self.depends = depends
        self.kwargs = kwargs

    def __nice__(self):
        return self.name


class Queue(ub.NiceRepr):
    """
    Base class for a queue
    """

    def submit(self, command, depends=None, **kwargs) -> Job:
        job = Job(...)
        return job

    def synchronize(self):
        """
        Force all subsequent commands
        """
        pass
