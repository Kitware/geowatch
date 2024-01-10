"""
Utilities for handling global resources
"""
from kwutil.util_eval import restricted_eval, RestrictedSyntaxError  # NOQA
from kwutil.util_resources import request_nofile_limits, check_shm_limits  # NOQA
from kwutil.util_parallel import coerce_num_workers  # NOQA


def configure_global_attributes(**config):
    """
    Configures hacks to fix global settings in external modules

    Args:
        config (dict): exected to contain certain special keys.

            * "workers" with an integer value equal to the number of dataloader
                processes.

            * "torch_sharing_strategy" to specify the torch multiprocessing backend

            * "request_rlimit_nofile"
                the maximum number of open files to request ulimit raise the
                soft limit to.

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
        request_rlimit_nofile = config.get('request_rlimit_nofile', 'auto')
        request_nofile_limits(request_rlimit_nofile=request_rlimit_nofile)
        want_shm_per_worker = 0.5  # gibibytes
        want_shm = want_shm_per_worker * num_workers
        try:
            check_shm_limits(want_shm)
        except IOError as ex:
            import warnings
            warnings.warn(str(ex))

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
