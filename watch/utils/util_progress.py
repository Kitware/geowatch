"""
POC for a better ProgIter with rich support
"""
import ubelt as ub
import os
from progiter import ProgIter


# If truthy disable all threaded rich options
PROGITER_NOTHREAD = os.environ.get('PROGITER_NOTHREAD', '')


class ProgIter2(ProgIter):

    def update_info(self, text):
        self.info_text = text
        info_text = getattr(self, 'info_text', None)
        self.ensure_newline()
        if info_text is not None:
            # if self._cursor_at_newline:
            print('+ --- Info --- +')
            print(info_text)
            print('+ ------------ +')
        # self.display_message()

    def display_message(self):
        super().display_message()


class RichProgIter:
    """
    Ducktypes ProgIter

    TODO: enhance with the ability to have a update info panel that removes
    the circular reference
    """

    def __init__(self, iterable=None, desc=None, total=None, freq=1, initial=0,
                 eta_window=64, clearline=True, adjust=True, time_thresh=2.0,
                 show_times=True, show_wall=False, enabled=True, verbose=None,
                 stream=None, chunksize=None, rel_adjust_limit=4.0,
                 transient=False, prog_manager=None, **kwargs):

        unhandled = {
            'eta_window', 'clearline', 'adjust', 'time_thresh', 'show_times',
            'show_wall', 'stream', 'chunksize', 'rel_adjust_limit',
        }
        kwargs = ub.udict(kwargs) - unhandled

        from rich.progress import Progress as richProgress
        self.prog_manager: richProgress = prog_manager
        self.iterable = iterable
        self.enabled = enabled
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
        if not self.enabled:
            yield from self.iterable
        else:
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
        Remove this progress task from its rich manager
        """
        if self.enabled:
            self.prog_manager.remove_task(self.task_id)

    def update_info(self, text):
        if self.enabled:
            # FIXME: remove circular reference
            self.prog_manager.pman.update_info(text)

    def set_postfix_str(self, text, refresh=True):
        self.extra = text
        parts = [self.desc]
        if self.extra is not None:
            parts.append(self.extra)
        if self.enabled:
            description = ' '.join(parts)
            self.prog_manager.update(self.task_id, description=description,
                                     refresh=refresh)


class ProgressManager:
    r"""
    A progress manager.

    Manage multiple progress bars, either with rich or ProgIter.

    CommandLine:
        xdoctest -m watch.utils.util_progress ProgressManager:1
        xdoctest -m watch.utils.util_progress ProgressManager:2

    Example:
        >>> from watch.utils.util_progress import ProgressManager
        >>> from progiter import progiter
        >>> # Can use plain progiter or rich
        >>> # The usecase for plain progiter is when threads / live output
        >>> # is not desirable and you just want plain stdout progress
        >>> pman = ProgressManager(backend='progiter')
        >>> with pman:
        >>>     oprog = pman.progiter(range(20), desc='outer loop', verbose=3)
        >>>     for i in oprog:
        >>>         oprog.set_postfix_str(f'Doing step {i}', refresh=False)
        >>>         for i in pman.progiter(range(100), desc=f'inner loop {i}'):
        >>>             pass
        >>> #
        >>> self = pman = ProgressManager(backend='rich')
        >>> pman = ProgressManager(backend='rich')
        >>> with pman:
        >>>     oprog = pman.progiter(range(20), desc='outer loop', verbose=3)
        >>>     for i in oprog:
        >>>         oprog.set_postfix_str(f'Doing step {i}', refresh=False)
        >>>         for i in pman.progiter(range(100), desc=f'inner loop {i}'):
        >>>             pass

    Example:
        >>> # A fairly complex example
        >>> from watch.utils.util_progress import ProgressManager
        >>> import time
        >>> delay = 0.000005
        >>> N_inner = 1000
        >>> N_outer = 11
        >>> self = pman = ProgressManager(backend='rich')
        >>> with pman:
        >>>     oprog = pman(range(N_outer), desc='outer loop')
        >>>     for i in oprog:
        >>>         if i > 7:
        >>>             self.update_info(f'The info panel gives detailed updates\nWe are now at step {i}\nWe are just about done now')
        >>>         elif i > 5:
        >>>             self.update_info(f'The info panel gives detailed updates\nWe are now at step {i}')
        >>>         oprog.set_postfix_str(f'Doing step {i}')
        >>>         N = 1000
        >>>         for j in pman(iter(range(N_inner)), total=None if i % 2 == 0 else N_inner, desc=f'inner loop {i}', transient=i < 4):
        >>>             time.sleep(delay)

    Example:
        >>> # Test complex example over a grid of parameters
        >>> from watch.utils.util_progress import ProgressManager, ProgIter2
        >>> import time
        >>> delay = 0.000005
        >>> N_inner = 1000
        >>> N_outer = 11
        >>> basis = {
        >>>     'with_info': [0, 1],
        >>>     'backend': ['progiter', 'rich'],
        >>>     'enabled': [0, 1],
        >>>     #'with_info': [1],
        >>> }
        >>> grid = list(ub.named_product(basis))
        >>> grid_prog = ProgIter2(grid, desc='Test cases over grid', verbose=3)
        >>> grid_prog.update_info('Here we go')
        >>> for item in grid:
        >>>     grid_prog.ensure_newline()
        >>>     grid_prog.update_info(f'Running grid test {ub.urepr(item, nl=1)}')
        >>>     print('\n\n')
        >>>     self = ProgressManager(backend=item['backend'], enabled=item['enabled'])
        >>>     with self:
        >>>         outer_prog = self.progiter(range(N_outer), desc='outer loop')
        >>>         for i in outer_prog:
        >>>             if item['with_info']:
        >>>                 if i > 7:
        >>>                     outer_prog.update_info(f'The info panel gives detailed updates\nWe are now at step {i}\nWe are just about done now')
        >>>                 elif i > 5:
        >>>                     outer_prog.update_info(f'The info panel gives detailed updates\nWe are now at step {i}')
        >>>             outer_prog.set_postfix_str(f'Doing step {i}')
        >>>             inner_kwargs = dict(
        >>>                 total=None if i % 2 == 0 else N_inner,
        >>>                 transient=i < 4,
        >>>                 time_thresh=delay * 2.3,
        >>>                 desc=f'inner loop {i}',
        >>>             )
        >>>             for j in self.progiter(iter(range(N_inner)), **inner_kwargs):
        >>>                 time.sleep(delay)
        >>>     grid_prog.update_info(f'Finished test item')
    """

    def __init__(self, backend='rich', **kwargs):
        self.backend = backend
        self.sub_progs = []
        self.enabled = kwargs.get('enabled', True)
        # Default arguments for new progiters
        self.default_progkw = ub.udict({
            'time_thresh': 2.0,
        }) & ub.udict(kwargs)
        if self.backend == 'rich':
            self.setup_rich()

    def progiter(self, iterable, total=None, desc=None, transient=False, verbose='auto', **kw):
        self.prog_iters = []
        backend = self.backend
        if PROGITER_NOTHREAD:
            backend = 'progiter'
        if backend == 'rich':
            # Fixme remove circular ref
            self.prog_manager.pman = self
            prog = RichProgIter(
                prog_manager=self.prog_manager, iterable=iterable, total=total,
                desc=desc, transient=transient, **kw)
        elif backend == 'progiter':
            progkw = self.default_progkw.copy()
            progkw.update(kw)
            progkw['verbose'] = verbose
            if verbose == 'auto':
                # Change all other - now outer - progiters to verbose=3 mode
                for other in self.prog_iters:
                    other.ensure_newline()
                    if 0 < other.verbose < 3:
                        other.clearline = False
                        other.adjust = False
                        other.freq = 1
                        other.verbose = 3
                progkw['verbose'] = 1 if len(self.prog_iters) == 0 else 3
            # progkw['verbose'] = 1
            prog = ProgIter2(iterable, total=total, desc=desc, **progkw)
        else:
            raise KeyError(backend)
        self.prog_iters.append(prog)
        return prog

    def new(self, *args, **kw):
        return self.progiter(*args, **kw)

    def __call__(self, *args, **kw):
        return self.progiter(*args, **kw)

    def setup_rich(self):
        import rich
        import rich.progress
        from rich.console import Group
        from rich.live import Live
        from rich.progress import BarColumn, TextColumn
        from rich.progress import Progress as richProgress
        self.prog_manager = richProgress(
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
        # TODO: make a backend RichProgIterManager, TQDMProgIterManager and ProgIterManager
        if self.backend == 'rich':
            from rich.panel import Panel
            if self.info_panel is None:
                self.info_panel = Panel(text)
                self.progress_group.renderables.insert(0, self.info_panel)
            else:
                self.info_panel.renderable = text
        else:
            self.prog_iters[0].update_info(text)

    def start(self):
        self.__enter__()
        return self

    def begin(self):
        return self.start()

    def stop(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        if self.backend == 'rich':
            if self.enabled:
                return self.live_context.__enter__()

    def __exit__(self, *args, **kw):
        if self.backend == 'rich':
            if self.enabled:
                return self.live_context.__exit__(*args, **kw)
