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
