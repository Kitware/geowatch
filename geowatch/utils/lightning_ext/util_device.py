
def coerce_devices(gpus):
    """
    Coerce a command line argument for GPUs into a valid set of torch devices

    This is a wrapper around lightning
    :func:`pytorch_lightning.utilities.parse_gpu_ids` (which was deprecated in
    lighting 1.8 so we have to vendor it)

    It extends the cases
    that it can handle and is specific to torch devices. As of lightning 1.6
    their own device parsing is pretty good, so this may not be necessary.

    If `gpus` is a list of integers, then those devices are used.

    If `gpus` is None or "cpu", then the CPU is used.

    If `gpus` is "cuda", that is equivalent to `gpus=[0]`.

    If `gpus` is a string without commas, then the string should be of a number
        indicating how many gpus should be used.

    If `gpus` is a string with commas separating integers, then that
        indicates the device indexes that should be used.

    Args:
        gpus (List[int] | str | int | None):
            adds ability to parse "cpu", "auto", "auto:N".

    Returns:
        List[torch.device]

    Example:
        >>> from geowatch.utils.lightning_ext import util_device
        >>> print(util_device.coerce_devices('cpu'))
        >>> print(util_device.coerce_devices(None))
        >>> # xdoctest: +SKIP
        >>> # breaks without a cuda machine
        >>> #print(util_device.coerce_devices("0"))
        >>> print(util_device.coerce_devices("1"))
        >>> print(util_device.coerce_devices("0"))
        >>> print(util_device.coerce_devices("0,"))
        >>> print(util_device.coerce_devices(1))
        >>> print(util_device.coerce_devices([0, 1]))
        >>> print(util_device.coerce_devices("0, 1"))
        >>> print(util_device.coerce_devices("auto"))
        >>> if torch.cuda.device_count() > 0:
        >>>     print(util_device.coerce_devices("auto:1"))
        >>> if torch.cuda.device_count() > 1:
        >>>     print(util_device.coerce_devices("auto:2"))
        >>> if torch.cuda.device_count() > 2:
        >>>     print(util_device.coerce_devices("auto:3"))
    """
    import torch
    needs_gpu_coerce = True
    auto_select_gpus = False

    if isinstance(gpus, str):
        if gpus == 'cpu':
            gpu_ids = None
            needs_gpu_coerce = False
        elif gpus == 'cuda':
            gpu_ids = [0]
            needs_gpu_coerce = False
        elif gpus.startswith('auto'):
            auto_select_gpus = True
            parts = gpus.split(':')
            if len(parts) == 1:
                gpus = -1
            else:
                gpus = int(parts[1])
        else:
            try:
                gpus = [int(p.strip()) for p in gpus.split(',') if p.strip()]
            except Exception:
                pass
            needs_gpu_coerce = True

    print(f'gpus={gpus}')
    print(f'auto_select_gpus={auto_select_gpus}')
    if auto_select_gpus:
        from pytorch_lightning.tuner import auto_gpu_select
        gpu_ids = auto_gpu_select.pick_multiple_gpus(gpus)
    elif needs_gpu_coerce:
        try:
            from geowatch.utils.lightning_ext import old_parser_devices
            # from pytorch_lightning.utilities import device_parser
            gpu_ids = old_parser_devices.parse_gpu_ids(gpus)
        except Exception as ex:
            print(f'WARNING. Ignoring ex={ex}')
            gpu_ids = gpus
            import ubelt as ub
            if gpu_ids is not None and ub.iterable(gpu_ids):
                assert all(isinstance(g, int) for g in gpu_ids)

    if gpu_ids is None:
        devices = [torch.device('cpu')]
    else:
        devices = [torch.device(_id) for _id in gpu_ids]
    return devices


def _test_lightning_is_sane():
    from geowatch.utils.lightning_ext import old_parser_devices as device_parser
    # from pytorch_lightning.utilities import device_parser
    import torch
    num_devices = torch.cuda.device_count()

    assert device_parser.parse_gpu_ids('0') is None
    assert device_parser.parse_gpu_ids('[]') is None
    assert device_parser.parse_gpu_ids(0) is None

    if num_devices > 0:
        assert device_parser.parse_gpu_ids('1') == [0]
        assert device_parser.parse_gpu_ids(1) == [0]
        assert device_parser.parse_gpu_ids('0,') == [0]

    if num_devices > 1:
        assert device_parser.parse_gpu_ids('2') == [0, 1]
        assert device_parser.parse_gpu_ids(2) == [0, 1]
        assert device_parser.parse_gpu_ids([0 , 1]) == [0, 1]
        assert device_parser.parse_gpu_ids('0, 1') == [0, 1]

    if num_devices:
        assert device_parser.parse_gpu_ids(-1) == list(range(num_devices))


def _test_coerce_is_sane():
    import torch
    num_devices = torch.cuda.device_count()

    if num_devices:
        all_devices = [torch.device(i) for i in range(num_devices)]
        assert coerce_devices('-1') == all_devices
        assert coerce_devices(-1) == all_devices
        assert coerce_devices('auto') == all_devices

    assert coerce_devices('0') == [torch.device('cpu')]
    assert coerce_devices('[]') == [torch.device('cpu')]
    assert coerce_devices(0) == [torch.device('cpu')]
    if num_devices > 0:
        assert coerce_devices('1') == [torch.device(0)]
        assert coerce_devices(1) == [torch.device(0)]
        assert coerce_devices('0,') == [torch.device(0)]
        assert coerce_devices('auto:1') == [torch.device(0)]
    if num_devices > 1:
        assert coerce_devices('2') == [torch.device(0), torch.device(1)]
        assert coerce_devices(2) == [torch.device(0), torch.device(1)]
        assert coerce_devices([0 , 1]) == [torch.device(0), torch.device(1)]
        assert coerce_devices('0, 1') == [torch.device(0), torch.device(1)]
        assert coerce_devices('auto:2') == [torch.device(0), torch.device(1)]
