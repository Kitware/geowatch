"""
POC for a better ProgIter with rich support
"""
import ubelt as ub


class RichProgIter:
    """
    Ducktypes ProgIter
    """

    def __init__(self, prog_manager, iterable, total=None, desc=None,
                 transient=False):
        from rich.progress import Progress
        self.prog_manager: Progress = prog_manager
        self.iterable = iterable
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                ...
        self.total = total
        self.desc = desc
        self.task_id = self.prog_manager.add_task(desc, total=self.total)
        self.transient = transient
        self.extra = None

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.prog_manager.update(self.task_id, advance=1)
        if self.total is None:
            task = self.prog_manager._tasks[self.task_id]
            self.prog_manager.update(self.task_id, total=task.completed)
        if self.transient:
            self.remove()

    def remove(self):
        """
        Remove this progress bar
        """
        self.prog_manager.remove_task(self.task_id)

    def set_postfix_str(self, text, refresh=True):
        self.extra = text
        parts = [self.desc]
        if self.extra is not None:
            parts.append(self.extra)
        description = ' '.join(parts)
        self.prog_manager.update(self.task_id, description=description,
                                 refresh=refresh)


class MultiProgress:
    """
    Manage multiple progress bars, either with rich or ProgIter.

    Example:
        >>> from watch.utils.util_progress import *  # NOQA
        >>> # Can use plain progiter or rich
        >>> # The usecase for plain progiter is when threads / live output
        >>> # is not desirable and you just want plain stdout progress
        >>> multi_prog = MultiProgress(use_rich=0)
        >>> with multi_prog:
        >>>     oprog = multi_prog.progiter(range(20), desc='outer loop', verbose=3)
        >>>     for i in oprog:
        >>>         oprog.set_postfix_str(f'Doing step {i}', refresh=False)
        >>>         for i in multi_prog.progiter(range(100), desc=f'inner loop {i}'):
        >>>             pass
        >>> #
        >>> self = multi_prog = MultiProgress(use_rich=1)
        >>> slowness = 0.001
        >>> multi_prog = MultiProgress(use_rich=1)
        >>> with multi_prog:
        >>>     oprog = multi_prog.progiter(range(20), desc='outer loop', verbose=3)
        >>>     for i in oprog:
        >>>         oprog.set_postfix_str(f'Doing step {i}', refresh=False)
        >>>         for i in multi_prog.progiter(range(100), desc=f'inner loop {i}'):
        >>>             pass

    Example:
        >>> # A fairly complex example
        >>> from watch.utils.util_progress import *  # NOQA
        >>> self = multi_prog = MultiProgress(use_rich=1)
        >>> slowness = 0.0005
        >>> import time
        >>> N_inner = 1000
        >>> N_outer = 11
        >>> with multi_prog:
        >>>     oprog = multi_prog.progiter(range(N_outer), desc='outer loop')
        >>>     for i in oprog:
        >>>         if i > 7:
        >>>             self.update_info(f'The info panel gives detailed updates\nWe are now at step {i}\nWe are just about done now')
        >>>         elif i > 5:
        >>>             self.update_info(f'The info panel gives detailed updates\nWe are now at step {i}')
        >>>         oprog.set_postfix_str(f'Doing step {i}')
        >>>         N = 1000
        >>>         for j in multi_prog.progiter(iter(range(N_inner)), total=None if i % 2 == 0 else N_inner, desc=f'inner loop {i}', transient=i < 4):
        >>>             time.sleep(slowness)
    """

    def __init__(self, use_rich=1):
        self.use_rich = use_rich
        self.sub_progs = []
        if self.use_rich:
            self.setup_rich()

    def progiter(self, iterable, total=None, desc=None, transient=False, verbose=1):
        self.prog_iters = []
        if self.use_rich:
            prog = RichProgIter(
                prog_manager=self.prog_manager, iterable=iterable, total=total,
                desc=desc, transient=transient)
        else:
            prog = ub.ProgIter(iterable, total=total, desc=desc,
                               verbose=verbose)
        self.prog_iters.append(prog)
        return prog

    def new(self, *args, **kw):
        return self.progiter(*args, **kw)

    def setup_rich(self):
        from rich.console import Group
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
        self.info_panel = None
        # Panel('')
        self.progress_group = Group(
            # self.info_panel,
            self.prog_manager,
        )
        self.live_context = Live(self.progress_group)

    def update_info(self, text):
        if self.use_rich:
            from rich.panel import Panel
            if self.info_panel is None:
                self.info_panel = Panel(text)
                self.progress_group.renderables.insert(0, self.info_panel)
            else:
                self.info_panel.renderable = text

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        if self.use_rich:
            return self.live_context.__enter__()

    def __exit__(self, *args, **kw):
        if self.use_rich:
            return self.live_context.__exit__(*args, **kw)
