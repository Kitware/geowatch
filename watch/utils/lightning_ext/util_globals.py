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
    """

    num_workers = config.get('num_workers', None)
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


def request_cpus(max_load=0.5):
    """
    Return some number of CPUs based on a chosen hueristic

    Args:
        max_load (float): only consider CPUs with a load less than this

    Returns:
        int : number of available cpus based on request parameters

    Example:
        >>> from watch.utils.lightning_ext.util_globals import *  # NOQA
        >>> request_cpus()
    """
    import psutil
    import numpy as np
    # num_cores = psutil.cpu_count()
    current_load = np.array(psutil.cpu_percent(percpu=True)) / 100
    num_available = np.sum(current_load < 0.5)
    return num_available
