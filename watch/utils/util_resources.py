"""
Utilities to configure / inspect system resources
"""


def request_nofile_limits(request_rlimit_nofile='auto', requested_limit=None):
    """
    Attempts to increase the limit of open file descriptors on the system.

    Does nothing on non-linux operating systems.

    Args:
        request_rlimit_nofile (int | str):
            attempt to increase rlimit_nofile to this number of open files,
            if auto, uses a hueristic, which currently defaults to 8192.

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

    if requested_limit is not None:
        ub.schedule_deprecation(
            'watch', 'requested_limit', 'arg',
            migration='Use request_rlimit_nofile instead',
            deprecate='0.6.4', error='0.7.0', remove='0.8.0')
        request_rlimit_nofile = requested_limit

    if ub.LINUX:
        import resource
        if isinstance(request_rlimit_nofile, str):
            if request_rlimit_nofile == 'auto':
                request_rlimit_nofile = 8192
            else:
                raise KeyError(request_rlimit_nofile)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if request_rlimit_nofile > soft:
            print('Requesting FileLimit = {!r}'.format(request_rlimit_nofile))
            print(' * Before FileLimit: soft={}, hard={}'.format(soft, hard))
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (request_rlimit_nofile, hard))
            except Exception as ex:
                print(f'ERROR ex={ex}')
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(' * After FileLimit: soft={}, hard={}'.format(soft, hard))


def check_shm_limits(threshold_gibibytes=1):
    """
    When running training in docker you may need more shared memory than
    its default provide. Warns if shared memory is not above a threshold.

    Args:
        threshold_gibibytes (float):
            The amount of shared memory we want to have access to.

    Notes:
        Running docker run with ``--shm-size=32768m`` increases shm to 32GB

    Raises:
        IOError: if the system does not have a /dev/shm device.

    References:
        .. [SO30210362] https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container
        .. [SO58804022] https://stackoverflow.com/questions/58804022/how-to-resize-dev-shm
    """
    import ubelt as ub
    import psutil

    shm = ub.Path('/dev/shm')
    if shm.exists():
        usage = psutil.disk_usage(shm)
        shm_bytes = usage.total
    else:
        raise IOError('Dont know how to check shm size on this system')

    # shm_gigabytes = (shm_bytes / 10 ** 9)
    shm_gibibytes = (shm_bytes / 2 ** 30)

    if shm_gibibytes < threshold_gibibytes:
        import warnings
        import math
        thresh_mb = math.ceil(threshold_gibibytes * (2 ** 30 / 10 ** 6))
        print(f'shm_gibibytes={shm_gibibytes}')
        warnings.warn(ub.paragraph(
            f'''
            Likely do not have enough /dev/shm space.
            Current /dev/shm space is {shm_gibibytes} GiB,
            but we requested {threshold_gibibytes} GiB.
            Changing the size of /dev/shm of a host machine requires system
            level configuration. For docker images you can call docker run
            with --shm-size={thresh_mb}m.
            '''))

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
