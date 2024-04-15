"""
Script to handle setting local and global remotes for our DVC directories.
"""
import platform
import ubelt as ub
import pandas as pd
import rich

addr_lut = {
    'horologic': 'horologic.kitware.com',
    'yardrat': 'yardrat.kitware.com',
    'namek': 'namek.kitware.com',
}

# Explicitly list the machines the data might exist on with absolute paths
# to the dvc caches wrt to those specific machines.
remotes = [
    {
        'tag': 'phase2_data',
        'hardware': 'ssd',
        'repo_path': '/home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc-ssd',
        'host': 'horologic',
        'cache_path': '/flash/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'hdd',
        'repo_path': '/home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc',
        'host': 'horologic',
        'cache_path': '/data/dvc-caches/smart_watch_dvc',
    },
    {
        'tag': 'drop7_data',
        'hardware': 'auto',
        'repo_path': '/flash/smart_drop7',
        'host': 'horologic',
        'cache_path': '/flash/smart_drop7/.dvc/cache',
    },
    {
        'tag': 'phase2_expt',
        'hardware': 'auto',
        'repo_path': '/home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_expt_dvc',
        'host': 'horologic',
        'cache_path': '/data/dvc-caches/smart_expt_dvc_cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'ssd',
        'repo_path': '/home/joncrall/remote/Ooo/data/dvc-repos/smart_data_dvc-ssd',
        'host': 'ooo',
        'cache_path': '/flash/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'hdd',
        'repo_path': '/home/joncrall/remote/Ooo/data/dvc-repos/smart_data_dvc',
        'host': 'ooo',
        'cache_path': '/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'drop7_data',
        'hardware': 'auto',
        'repo_path': '/flash/smart_drop7',
        'host': 'ooo',
        'cache_path': '/flash/smart_drop7/.dvc/cache',
    },
    {
        'tag': 'phase2_expt',
        'hardware': 'auto',
        'repo_path': '/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc',
        'host': 'ooo',
        'cache_path': '/data/joncrall/dvc-repos/smart_expt_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'ssd',
        'repo_path': '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd',
        'host': 'toothbrush',
        'cache_path': '/media/joncrall/flash1/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'hdd',
        'repo_path': '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc',
        'host': 'toothbrush',
        'cache_path': '/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'drop7_data',
        'hardware': 'auto',
        'repo_path': '/media/joncrall/flash1/smart_drop7',
        'host': 'toothbrush',
        'cache_path': '/media/joncrall/flash1/smart_drop7/.dvc/cache',
    },
    {
        'tag': 'phase2_expt',
        'hardware': 'auto',
        'repo_path': '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc',
        'host': 'toothbrush',
        'cache_path': '/data/joncrall/dvc-repos/smart_expt_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'ssd',
        'repo_path': '/home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_data_dvc-ssd',
        'host': 'yardrat',
        'cache_path': '/data2/dvc-repos/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'hdd',
        'repo_path': '/home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_data_dvc-hdd',
        'host': 'yardrat',
        'cache_path': '/data/dvc-repos/smart_data_dvc-hdd/.dvc/cache',
    },
    {
        'tag': 'drop7_data',
        'hardware': 'auto',
        'repo_path': '/data2/dvc-repos/smart_drop7',
        'host': 'yardrat',
        'cache_path': '/data2/dvc-repos/smart_drop7/.dvc/cache',
    },
    {
        'tag': 'phase2_expt',
        'hardware': 'auto',
        'repo_path': '/home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc',
        'host': 'yardrat',
        'cache_path': '/data/dvc-repos/smart_expt_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'ssd',
        'repo_path': '/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc-ssd',
        'host': 'namek',
        'cache_path': '/flash/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'phase2_data',
        'hardware': 'hdd',
        'repo_path': '/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc',
        'host': 'namek',
        'cache_path': '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_data_dvc/.dvc/cache',
    },
    {
        'tag': 'drop7_data',
        'hardware': 'auto',
        'repo_path': '/flash/smart_drop7',
        'host': 'namek',
        'cache_path': '/flash/smart_drop7/.dvc/cache',
    },
    {
        'tag': 'phase2_expt',
        'hardware': 'auto',
        'repo_path': '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc',
        'host': 'namek',
        'cache_path': '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_expt_dvc/.dvc/cache',
    },
]


def check_paths_are_ok():
    """
    Dev function to make sure things have not changed / explore paths on
    different machines.
    """
    unique_hosts = set()
    for remote in remotes:
        unique_hosts.add(remote['host'])

    host_to_fs = {}
    for host in unique_hosts:
        print(f'connect to host={host}')
        host_to_fs[host] = fsspec_shh_connect(host)

    # Use simple_dvc registry / geowatch_dvc to find the paths to the DVC
    # repos on other machines.
    items = []
    for host, fs in host_to_fs.items():
        fs = host_to_fs[host]
        print('host = {}'.format(ub.urepr(host, nl=1)))
        tags = ['phase2_data']
        for tag in tags:
            for hw in ['ssd', 'hdd']:
                command = f'geowatch_dvc --tags "{tag}" --hardware="{hw}"'
                stdin, stdout, stderr = fs.client.exec_command(command)
                stdout_text = stdout.read().decode()
                path = stdout_text.strip()
                items.append({'tag': tag, 'hardware': hw, 'repo_path': path, 'host': host, 'command': command})

        tags = ['drop7_data', 'phase2_expt']
        for tag in tags:
            hw = 'auto'
            command = f'geowatch_dvc --tags "{tag}" --hardware="{hw}"'
            stdin, stdout, stderr = fs.client.exec_command(command)
            stdout_text = stdout.read().decode()
            path = stdout_text.strip()
            items.append({'tag': tag, 'hardware': hw, 'repo_path': path, 'host': host, 'command': command})

    for item in items:
        item.pop('path', None)
        item.pop('command', None)

    # Now find the cache in each case
    for item in items:
        host = item['host']
        fs = host_to_fs[host]
        print('host = {}'.format(ub.urepr(host, nl=1)))
        command = f'cd {item["repo_path"]} && dvc cache dir'
        stdin, stdout, stderr = fs.client.exec_command(command)
        stdout_text = stdout.read().decode()
        cache_path = stdout_text.strip()
        item['cache_path'] = cache_path

    # Hacks to fixup hardware
    for item in items:
        if 'hardware' == 'auto':
            if item['tag'] == 'phase2_expt':
                item['hardware'] = 'hdd'
            else:
                item['hardware'] = 'ssd'
    print('items = {}'.format(ub.urepr(items, nl=3)))


def fsspec_shh_connect(host):
    # This is not as easy as it could be
    # Paramiko does not respect the ssh config by default, but it does
    # give us tools to parse it. However, it is still not straightforward
    # Might look into "fabric"?
    import paramiko
    import os
    ssh_config = paramiko.SSHConfig()
    user_config_file = os.path.expanduser("~/.ssh/config")
    if os.path.exists(user_config_file):
        with open(user_config_file) as f:
            ssh_config.parse(f)
    user_config = ssh_config.lookup(host)
    ssh_kwargs = {
        'username': user_config['user'],
        'key_filename': user_config['identityfile'][0],
    }
    # import fsspec
    from fsspec.implementations.sftp import SFTPFileSystem
    fs = SFTPFileSystem(host=user_config['hostname'], **ssh_kwargs)
    return fs
    # client = paramiko.SSHClient()


def main():
    # Run this inside your DVC repo to setup the local links
    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce('.')

    this_cache_dir = ub.Path(dvc.cache_dir).resolve()

    host = platform.node()

    chosen_remotes = [r for r in remotes if r['tag'] == 'phase2_data']

    for remote in chosen_remotes:
        remote['is_local'] = (host == remote['host'])
        remote['is_this'] = ub.Path(remote['cache_path']).resolve() == this_cache_dir

    df = pd.DataFrame(chosen_remotes)
    rich.print(df.to_string())

    main_tasks = []
    local_tasks = []
    for remote in chosen_remotes:
        host = remote['host']
        cache_path = remote['cache_path']

        if remote['hardware'] == 'ssd':
            names = [host + '_ssd']
        else:
            names = [host, host + '_hdd']

        for name in names:
            if remote['is_local'] and not remote['is_this']:
                hardware = remote['hardware']
                local_tasks.append({'cmd': f'dvc remote add --local -f {name} {cache_path}'})
                local_tasks.append({'cmd': f'dvc remote add --local -f local_{hardware} {cache_path}'})

            addr = addr_lut.get(host, host)
            main_tasks.append({'cmd': f'dvc remote add -f {name} ssh://{addr}{cache_path}'})

    import cmd_queue
    queue = cmd_queue.Queue.create('serial')

    for task in main_tasks:
        queue.submit(task['cmd'])
    for task in local_tasks:
        queue.submit(task['cmd'])

    queue.print_commands()
    queue.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/devsetup/dvc_update_remotes.py
    """
    main()
