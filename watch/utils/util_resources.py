"""
Utilities to configure / inspect system resources
"""


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
        >>> from watch.utils import util_resources
        >>> util_resources.request_nofile_limits()
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
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (requested_limit, hard))
            except Exception as ex:
                print(f'ERROR ex={ex}')
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
