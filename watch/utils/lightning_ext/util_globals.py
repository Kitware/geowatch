def configure_hacks(num_workers=None, sharing_strategy='default'):
    """
    Configures hacks to fix global settings in external modules

    Args:
        config (dict): exected to contain certain special keys.

            * "workers" with an integer value equal to the number of dataloader
                processes.

            * "sharing_strategy" to specify the torch multiprocessing backend

        **kw: can also be used to specify config items

    Modules we currently hack:
        * cv2 - fix thread count
        * torch sharing strategy
    """

    if num_workers is not None and num_workers > 0:
        import cv2
        cv2.setNumThreads(0)

    strat = sharing_strategy
    if strat is not None and strat != 'default':
        import torch
        if strat == 'auto':
            # TODO: can we add a better auto test?
            strat = torch.multiprocessing.get_sharing_strategy()
        valid_strats = torch.multiprocessing.get_all_sharing_strategies()
        if strat not in valid_strats:
            raise KeyError('start={} is not in valid_strats={}'.format(strat, valid_strats))
        torch.multiprocessing.set_sharing_strategy(strat)

    if 0:
        """
        References:
            https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        """
        # torch.multiprocessing.get_all_start_methods()
        # torch_multiprocessing.get_context()
        torch.multiprocessing.set_start_method('spawn')
