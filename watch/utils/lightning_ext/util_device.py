
def coerce_gpus(gpus, auto_select_gpus=False):
    """
    Args:
        gpus (List[int] | str, int):

    References:
        https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
    """
    from pytorch_lightning.utilities import device_parser
    from pytorch_lightning.tuner import auto_gpu_select
    if auto_select_gpus and isinstance(gpus, int):
        gpus = auto_gpu_select.pick_multiple_gpus(gpus)
    gpu_ids = device_parser.parse_gpu_ids(gpus)
    return gpu_ids


def coerce_devices(gpus, auto_select_gpus=False):
    """
    Coerce a command line argument or GPUs into a valid set of torch devices

    Args:
        gpus (List[int] | str | int | None): adds ability to parse
            "cpu", "auto", "auto:N".

    Returns:
        List[torch.device]

    Example:
        >>> from watch.utils.lightning_ext import util_device
        >>> util_device.coerce_devices('cpu')
        >>> util_device.coerce_devices(None)
        >>> # xdoctest: +SKIP
        >>> # breaks without a cuda machine
        >>> from watch.utils.lightning_ext import util_device
        >>> util_device.coerce_devices("1")
        >>> util_device.coerce_devices(1)
        >>> util_device.coerce_devices([0, 1])
        >>> util_device.coerce_devices("0, 1")
        >>> util_device.coerce_devices("auto")
        >>> util_device.coerce_devices("auto:1")
        >>> util_device.coerce_devices("auto:2")
        >>> util_device.coerce_devices("auto:3")
    """
    import torch

    needs_gpu_coerce = True

    if isinstance(gpus, str):
        if gpus == 'cpu':
            gpu_ids = None
            needs_gpu_coerce = False
        elif gpus.startswith('auto'):
            auto_select_gpus = True
            parts = gpus.split(':')
            if len(parts) == 1:
                gpus = -1
            else:
                gpus = int(parts[1])

    if needs_gpu_coerce:
        gpu_ids = coerce_gpus(gpus, auto_select_gpus=auto_select_gpus)

    if gpu_ids is None:
        devices = [torch.device('cpu')]
    else:
        devices = [torch.device(_id) for _id in gpu_ids]
    return devices
