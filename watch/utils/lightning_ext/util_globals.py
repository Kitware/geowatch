"""
Utilities for handling global resources
"""


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


def check_shm_limits(threshold_gibibytes=1):
    """
    When running training in docker you may need more shared memory than
    its default provide. Warns if shared memory is not above a threshold.

    Notes:
        Running docker run with `--shm-size=32768m` increases shm to 32GB

    References:
        https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container
    """
    import ubelt as ub
    import psutil

    shm = ub.Path('/dev/shm')
    if shm.exists():
        usage = psutil.disk_usage(shm)
        shm_bytes = usage.total
    else:
        raise NotImplementedError('Dont know how to check shm size on this system')

    # shm_gigabytes = (shm_bytes / 10 ** 9)
    shm_gibibytes = (shm_bytes / 2 ** 30)

    if shm_gibibytes < threshold_gibibytes:
        import warnings
        print(f'shm_gibibytes={shm_gibibytes}')
        warnings.warn('Likely do not have enough /dev/shm space')

    if 0:
        disk_parts = psutil.disk_partitions(all=True)
        # Note sure if there is a way to find candidates that could be
        # /dev/shm or if it is ever the case that it doesnt exist.
        import pandas as pd
        rows = []
        for part in disk_parts:
            part_dict = {k: getattr(part, k) for k in dir(part) if not k.startswith('_')}
            part_dict = {k: v for k, v in part_dict.items() if isinstance(v, (str, float, int))}
            part_dict['opts'] = part_dict['opts'][0:128]
            rows.append(part_dict)
        df = pd.DataFrame(rows)
        df = df.sort_values('fstype')
        print(df.to_string())


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


class RestrictedSyntaxError(Exception):
    """
    An exception raised by restricted_eval if a disallowed expression is given
    """
    pass


def restricted_eval(expr, max_chars=32, local_dict=None, builtins_passlist=None):
    """
    A restricted form of Python's eval that is meant to be slightly safer

    Args:
        expr (str): the expression to evaluate
        max_char (int): expression cannot be more than this many characters
        local_dict (Dict[str, Any]): a list of variables allowed to be used
        builtins_passlist : if specified, only allow use of certain builtins

    References:
        https://realpython.com/python-eval-function/#minimizing-the-security-issues-of-eval

    Notes:
        This function may not be safe, but it has as many mitigation measures
        that I know about. This function should be audited and possibly made
        even more restricted. The idea is that this should just be used to
        evaluate numeric expressions.

    Example:
        >>> from watch.utils.lightning_ext.util_globals import *  # NOQA
        >>> builtins_passlist = ['min', 'max', 'round', 'sum']
        >>> local_dict = {}
        >>> max_chars = 32
        >>> expr = 'max(3 + 2, 9)'
        >>> result = restricted_eval(expr, max_chars, local_dict, builtins_passlist)
        >>> expr = '3 + 2'
        >>> result = restricted_eval(expr, max_chars, local_dict, builtins_passlist)
        >>> expr = '3 + 2'
        >>> result = restricted_eval(expr, max_chars)
        >>> import pytest
        >>> with pytest.raises(RestrictedSyntaxError):
        >>>     expr = 'max(a + 2, 3)'
        >>>     result = restricted_eval(expr, max_chars, dict(a=3))
    """
    import builtins
    import ubelt as ub
    if len(expr) > max_chars:
        raise RestrictedSyntaxError(
            'num-workers-hueristic should be small text. '
            'We want to disallow attempts at crashing python '
            'by feeding nasty input into eval. But this may still '
            'be dangerous.'
        )
    if local_dict is None:
        local_dict = {}

    if builtins_passlist is None:
        builtins_passlist = []

    allowed_builtins = ub.dict_isect(builtins.__dict__, builtins_passlist)

    local_dict['__builtins__'] = allowed_builtins
    allowed_names = list(allowed_builtins.keys()) + list(local_dict.keys())
    code = compile(expr, "<string>", "eval")
    # Step 3
    for name in code.co_names:
        if name not in allowed_names:
            raise RestrictedSyntaxError(f"Use of {name} not allowed")
    result = eval(code, local_dict, local_dict)
    return result
