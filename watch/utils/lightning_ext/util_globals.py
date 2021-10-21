"""
Utilities for handling global resources
"""


def configure_hacks(**config):
    """
    Configures hacks to fix global settings in external modules

    Args:
        config (dict): exected to contain certain special keys.

            * "workers" with an integer value equal to the number of dataloader
                processes.

            * "torch_sharing_strategy" to specify the torch multiprocessing backend

        **kw: can also be used to specify config items

    Modules we currently hack:
        * cv2 - fix thread count
        * torch sharing strategy

    References:
        https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.get_all_sharing_strategies
    """

    num_workers = config.get('num_workers', None)
    num_workers = coerce_num_workers(num_workers)
    if num_workers is not None and num_workers > 0:
        import cv2
        cv2.setNumThreads(0)

    key = 'torch_sharing_strategy'
    value = config.get(key, None)
    if value is not None and value != 'default':
        import torch
        # TODO: can we add a better auto test?
        valid = torch.multiprocessing.get_all_sharing_strategies()
        if value not in valid:
            raise KeyError('value={} for {} is not in valid={}'.format(value, key, valid))
        torch.multiprocessing.set_sharing_strategy(value)
        print('SET torch.multiprocessing.set_sharing_strategy to = {!r}'.format(value))

    key = 'torch_start_method'
    value = config.get(key, None)
    if value is not None and value != 'default':
        import torch
        # TODO: can we add a better auto test?
        valid = torch.multiprocessing.get_all_start_methods()
        if value not in valid:
            raise KeyError('value={} for {} is not in valid={}'.format(value, key, valid))
        torch.multiprocessing.set_start_method(value)
        print('SET torch.multiprocessing.set_start_method to = {!r}'.format(value))

    if 0:
        """
        References:
            https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        """
        # torch.multiprocessing.get_all_start_methods()
        # torch_multiprocessing.get_context()
        torch.multiprocessing.set_start_method('spawn')


def request_nofile_limits(requested_limit='auto'):
    """
    Attempts to increase the limit of open file descriptors on the system.

    Args:
        requested_limit (int | str):
            attempt to increase rlimit_nofile to this number of open files,
            if auto, uses a hueristic

    Ignore:
        # Helpful file descriptor monitor script:
        watch -x bash -c '
            PROC_ID_LIST=($(ps -a | grep python | awk '"'"'{print $1}'"'"' ))
            for PROC_ID in "${PROC_ID_LIST[@]}"; do
                NUM_OPEN_FILES=$(lsof -p $PROC_ID | wc -l)
                echo "PROC_ID=$PROC_ID, NUM_OPEN_FILES=$NUM_OPEN_FILES"
            done
        '

        # Query current soft limit
        ulimit -S -n

        # Query current hard limit
        ulimit -H -n

        # Request higher soft limit
        ulimit -S -n 8192

    Example:
        >>> from watch.utils.lightning_ext.util_globals import *  # NOQA
        >>> request_nofile_limits()
    """
    import ubelt as ub
    if ub.LINUX:
        import resource
        if isinstance(requested_limit, str):
            if requested_limit == 'auto':
                requested_limit = 8192
            else:
                raise KeyError(requested_limit)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if requested_limit > soft:
            print('Requesting FileLimit = {!r}'.format(requested_limit))
            print(' * Before FileLimit: soft={}, hard={}'.format(soft, hard))
            resource.setrlimit(resource.RLIMIT_NOFILE, (requested_limit, hard))
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(' * After FileLimit: soft={}, hard={}'.format(soft, hard))


def coerce_num_workers(num_workers='auto', minimum=0):
    """
    Return some number of CPUs based on a chosen hueristic

    Args:
        num_workers (int | str):
            A special string code, or an exact number of cpus

        minimum (int): minimum workers we are allowed to return

    Returns:
        int : number of available cpus based on request parameters

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/utils/lightning_ext/util_globals.py coerce_num_workers

    Example:
        >>> from watch.utils.lightning_ext.util_globals import *  # NOQA
        >>> print(coerce_num_workers('all'))
        >>> print(coerce_num_workers('avail'))
        >>> print(coerce_num_workers('auto'))
        >>> print(coerce_num_workers('all-2'))
        >>> print(coerce_num_workers('avail-2'))
        >>> print(coerce_num_workers('all/2'))
        >>> import pytest
        >>> with pytest.raises(Exception):
        >>>     print(coerce_num_workers('all + 1' + (' + 1' * 100)))
        >>> total_cpus = coerce_num_workers('all')
        >>> assert coerce_num_workers('all-2') == max(total_cpus - 2, 0)
        >>> assert coerce_num_workers('all-100') == max(total_cpus - 100, 0)
        >>> assert coerce_num_workers('avail') <= coerce_num_workers('all')
        >>> assert coerce_num_workers(3) == max(3, 0)
    """
    import numpy as np
    import psutil

    try:
        num_workers = int(num_workers)
    except Exception:
        pass

    if isinstance(num_workers, str):

        num_workers = num_workers.lower()

        if num_workers == 'auto':
            num_workers = 'avail-2'

        # input normalization
        num_workers = num_workers.replace('available', 'avail')

        local_dict = {}

        prefix = 'avail'
        if 'avail' in num_workers:
            current_load = np.array(psutil.cpu_percent(percpu=True)) / 100
            local_dict['avail'] = np.sum(current_load < 0.5)
        local_dict['all_'] = psutil.cpu_count()

        if num_workers == 'none':
            num_workers = None
        else:
            expr = num_workers.replace('all', 'all_')
            if len(expr) > 32:
                raise Exception(
                    'num-workers-hueristic should be small text. '
                    'We want to disallow attempts at crashing python '
                    'by feeding nasty input into eval'
                )
            # note: eval is not safe, using numexpr instead
            # limit chars even futher if eval is used
            # evaluated = eval(expr, local_dict, local_dict)
            import numexpr
            num_workers = numexpr.evaluate(expr, local_dict=local_dict,
                                           global_dict=local_dict)

    if num_workers is not None:
        num_workers = max(int(num_workers), minimum)

    return num_workers
