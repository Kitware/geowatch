"""
See: ~/code/watch/geowatch/utils/process_context.py
"""


def get_cpu_mem_info():
    cpu_info = get_cpu_info()
    mem_info = get_mem_info()
    system_info = {
        'cpu_info': cpu_info,
        'mem_info': mem_info,
    }
    return system_info


def get_cpu_info():
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
    return cpu_info


def get_mem_info():
    """
    Memory info is returned in bytes.

    TODO:
        - [ ] Should we use pint to give these numbers units?

    References:
        https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory
    """
    import psutil
    svmem_info = psutil.virtual_memory()
    mem_info = dict(zip(svmem_info._fields, svmem_info))
    return mem_info


def disk_info_of_path(path):
    """
    Get disk info wrt where a file lives

    WIP - needs more work

    CommandLine:
        xdoctest -m geowatch.utils.util_hardware disk_info_of_path

    Returns:
        dict - dictionary of information

    Example:
        >>> from geowatch.utils.util_hardware import *  # NOQA
        >>> path = '.'
        >>> x = disk_info_of_path(path)
        >>> import ubelt as ub
        >>> print(ub.urepr(x))

    TODO:
        - [ ] Handle btrfs
        - [ ] Handle whatever AWS uses
        - [ ] Use udisksctl or udevadm

    Ignore:
        lsblk  /dev/nvme1n1
        lsblk -afs /dev/mapper/vgubuntu-root
        lsblk -nasd /dev/mapper/vgubuntu-root

        df $HOME --output=source,fstype
        df $HOME/data/dvc-repos/smart_watch_dvc --output=source,fstype
        df $HOME/data/dvc-repos/smart_watch_dvc-hdd --output=source,fstype

        df . --output=source,fstype,itotal,iused,iavail,ipcent,size,used,avail,pcent,file,target

    References:
        https://askubuntu.com/questions/609708/how-to-find-hard-drive-brand-name-or-model
        https://stackoverflow.com/questions/38615464/how-to-get-device-name-on-which-a-file-is-located-from-its-path-in-c
    """
    import ubelt as ub
    import os
    path = ub.Path(path)
    path = path.resolve()
    # Returns the lvm name: e.g.
    # Filesystem     Type
    # data           zfs
    # /dev/sde1      ext4
    # /dev/md0       ext4
    # /dev/mapper/vgubuntu-root ext4
    # /dev/nvme1n1              f2fs
    info = ub.cmd(f'df {path} --output=source,fstype', check=True)
    parts = info['out'].split('\n')[1].rsplit(' ', 1)
    source, filesystem = [p.strip() for p in parts]

    hwinfo = {
        'path': os.fspath(path),
        'source': source,
        'filesystem': filesystem,
    }

    if filesystem == 'zfs':
        # Use ZFS to get more information
        # info = ub.cmd(f'zpool list {source}', verbose=3)
        # info = ub.cmd(f'zpool iostat {source}', verbose=3)
        # info = ub.cmd(f'zpool list -H {source}', verbose=3)
        # info = ub.cmd(f'zpool iostat -H {source}', verbose=3)
        try:
            zfs_status = _zfs_status(source)
            hwinfo['hwtype'] = zfs_status['coarse_type']
        except Exception as ex:
            print('error in zfs stuff: ex = {}'.format(ub.urepr(ex, nl=1)))
    elif filesystem == 'overlay':
        # This is the case on AWS. lsblk isnt able to provide us with more info
        # I'm not sure how to determine more info.
        # References:
        # https://docs.kernel.org/filesystems/overlayfs.html
        ...
    else:
        try:
            if _device_is_hdd(source):
                hwinfo['hwtype'] = 'hdd'
            else:
                hwinfo['hwtype'] = 'ssd'
        except Exception as ex:
            print('warning: unable to infer disk info: ex = {}'.format(ub.urepr(ex, nl=1)))

    try:
        import json
        info = ub.cmd(f'lsblk -as {source} --json')
        lsblk_info = json.loads(info['out'])
        walker = ub.IndexableWalker(lsblk_info)
        names = []
        for path, item in walker:
            if isinstance(item, dict):
                if 'name' in item:
                    names.append(item['name'])
        hwinfo['names'] = names
    except Exception:
        pass
    # print(ub.Path('/proc/partitions').read_text())
    return hwinfo


def _device_is_hdd(path):
    import ubelt as ub
    import json
    info = ub.cmd(f'lsblk -as {path} --json -o name,rota')
    info.check_returncode()
    lsblk_info = json.loads(info['out'])
    walker = ub.IndexableWalker(lsblk_info)
    rotas = []
    for path, item in walker:
        if isinstance(item, dict):
            if 'rota' in item:
                rotas.append(item['rota'])
    return any(rotas)


def _zfs_status(pool, verbose=0):
    """
    Semi-parsable zfs status output.  This is a proof-of-concept and needs some
    work to handle the nested pool structure.
    """
    import ubelt as ub
    import re
    info = ub.cmd(f'zpool status {pool} -P', verbose=verbose)
    info.check_returncode()

    splitter = re.compile(r'\s+')

    config = ub.ddict(list)
    context = None
    state = None
    header = None
    # stack = [] todo

    for line in info.stdout.split('\n'):
        indentation = line[:len(line) - len(line.lstrip())]
        if not line.strip():
            continue
        if state is None:
            if line.strip() == 'config:':
                state = 'CONFIG'
        elif state == 'CONFIG':
            parts = splitter.split(line.strip())
            if parts[0] == 'NAME':
                state = 'NAME'
                header = parts
        elif state == 'NAME':
            if len(indentation) == 1:
                parts = splitter.split(line.strip())
                row = ub.dzip(header, parts)
                name = parts[0]
                row['children'] = []
                context = config[name] = row
            elif len(indentation) > 1:
                parts = splitter.split(line.strip())
                row = ub.dzip(header, parts)
                context['children'].append(row)
            else:
                state = None

    # hack to get the data we currently need
    dev_paths = []
    for row in config[pool]['children']:
        if row['NAME'].startswith('/dev'):
            dev_paths.append(row['NAME'])

    is_part_hdd = []
    for path in dev_paths:
        flag = _device_is_hdd(path)
        is_part_hdd.append(flag)

    if all(is_part_hdd):
        coarse_type = 'hdd'
    elif not any(is_part_hdd):
        coarse_type = 'ssd'
    else:
        coarse_type = 'mixed'

    output = {
        'coarse_type': coarse_type,
        'poc_config': config,
    }

    # print('records = {}'.format(ub.urepr(config, nl=True)))
    return output
