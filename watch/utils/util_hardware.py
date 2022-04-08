def get_cpu_mem_info():
    import ubelt as ub
    import cpuinfo
    import psutil
    cpu_info = cpuinfo.get_cpu_info()
    svmem_info = psutil.virtual_memory()
    mem_info = ub.dzip(svmem_info._fields, svmem_info)
    system_info = {
        'cpu_info': cpu_info,
        'mem_info': mem_info,
    }
    return system_info


def disk_info_of_path(path):
    """
    Get disk info wrt where a file lives

    WIP - needs more work

    Example:
        >>> path = '.'
        >>> disk_info_of_path(path)

    Ignore:
        lsblk  /dev/nvme1n1
        lsblk -afs /dev/mapper/vgubuntu-root
        lsblk -nasd /dev/mapper/vgubuntu-root

        df $HOME --output=source,fstype
        df $HOME/data/dvc-repos/smart_watch_dvc --output=source,fstype
        df $HOME/data/dvc-repos/smart_watch_dvc-hdd --output=source,fstype
    """
    # https://stackoverflow.com/questions/38615464/how-to-get-device-name-on-which-a-file-is-located-from-its-path-in-c
    import ubelt as ub
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
        'source': source,
        'filesystem': filesystem,
    }

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
