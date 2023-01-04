"""
POC for a better ProgIter with rich support
"""
import ubelt as ub


class RichProgIter:
    """
    Ducktypes ProgIter
    """
    def __init__(self, prog_manager, iterable, total=None, desc=None):
        self.prog_manager = prog_manager
        self.iterable = iterable
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                ...
        self.task_id = self.prog_manager.add_task(desc, total=total)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.prog_manager.update(self.task_id, advance=1)
        task = self.prog_manager._tasks[self.task_id]
        if task.total is None:
            self.prog_manager.update(self.task_id, total=task.completed)


class MultiProgress:
    """
    Manage multiple progress bars, either with rich or ProgIter.

    Example:
        >>> from watch.utils.util_progress import *  # NOQA
        >>> multi_prog = MultiProgress(use_rich=0)
        >>> with multi_prog:
        >>>     for i in multi_prog.new(range(100), desc='outer loop'):
        >>>         for i in multi_prog.new(range(100), desc='inner loop'):
        >>>             pass
        >>> #
        >>> self = multi_prog = MultiProgress(use_rich=1)
        >>> with multi_prog:
        >>>     for i in multi_prog.new(range(10), desc='outer loop'):
        >>>         for i in multi_prog.new(iter(range(1000)), desc='inner loop'):
        >>>             pass
    """

    def __init__(self, use_rich=1):
        self.use_rich = use_rich
        self.sub_progs = []
        if self.use_rich:
            self.setup_rich()

    def new(self, iterable, total=None, desc=None):
        self.prog_iters = []
        if self.use_rich:
            prog = RichProgIter(
                prog_manager=self.prog_manager, iterable=iterable, total=total,
                desc=desc)
        else:
            prog = ub.ProgIter(iterable, total=total, desc=desc)
        self.prog_iters.append(prog)
        return prog

    def setup_rich(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.live import Live
        import rich
        import rich.progress
        from rich.progress import (BarColumn, Progress, TextColumn)
        self.prog_manager = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            rich.progress.MofNCompleteColumn(),
            # "[progress.percentage]{task.percentage:>3.0f}%",
            rich.progress.TimeRemainingColumn(),
            rich.progress.TimeElapsedColumn(),
        )
        self.info_panel = Panel('')
        self.progress_group = Group(
            self.info_panel,
            self.prog_manager,
        )
        self.live_context = Live(self.progress_group)

    def update_info(self, text):
        if self.use_rich:
            self.info_panel.renderable = text

    def __enter__(self):
        if self.use_rich:
            return self.live_context.__enter__()

    def __exit__(self, *args, **kw):
        if self.use_rich:
            return self.live_context.__exit__(*args, **kw)
