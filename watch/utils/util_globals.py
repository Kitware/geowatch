"""
Utilities for handling global resources
"""
from watch.utils.util_eval import restricted_eval, RestrictedSyntaxError  # NOQA
from watch.utils.util_resources import request_nofile_limits, check_shm_limits  # NOQA


def configure_global_attributes(**config):
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
        request_nofile_limits()
        want_shm_per_worker = 0.5
        want_shm = want_shm_per_worker * num_workers
        check_shm_limits(want_shm)

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


def coerce_num_workers(num_workers='auto', minimum=0):
    """
    Return some number of CPUs based on a chosen hueristic

    Note:
        Moved to kwcoco.util.util_resources

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
        >>> print(coerce_num_workers('min(all,2)'))
        >>> print(coerce_num_workers('[max(all,2)][0]'))
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

        # prefix = 'avail'
        if 'avail' in num_workers:
            current_load = np.array(psutil.cpu_percent(percpu=True)) / 100
            local_dict['avail'] = np.sum(current_load < 0.5)
        local_dict['all_'] = psutil.cpu_count()

        if num_workers == 'none':
            num_workers = None
        else:
            expr = num_workers.replace('all', 'all_')
            # note: eval is not safe, using numexpr instead
            # limit chars even futher if eval is used
            if 1:
                # Mitigate attack surface by restricting builtin usage
                max_chars = 32
                builtins_passlist = ['min', 'max', 'round', 'sum']
                num_workers = restricted_eval(expr, max_chars, local_dict,
                                              builtins_passlist)
            else:
                import numexpr
                num_workers = numexpr.evaluate(expr, local_dict=local_dict,
                                               global_dict=local_dict)

    if num_workers is not None:
        num_workers = max(int(num_workers), minimum)

    return num_workers
