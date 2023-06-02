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

remotes = []
remotes += [{'host': 'horologic',  'hardware': 'hdd', 'path': '/data/dvc-caches/smart_watch_dvc'}]
remotes += [{'host': 'horologic',  'hardware': 'ssd', 'path': '/flash/smart_data_dvc/.dvc/cache'}]
remotes += [{'host': 'namek',      'hardware': 'hdd', 'path': '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/.dvc/cache'}]
remotes += [{'host': 'namek',      'hardware': 'ssd', 'path': '/flash/smart_data_dvc/.dvc/cache'}]
remotes += [{'host': 'toothbrush', 'hardware': 'hdd', 'path': '/data/joncrall/dvc-repos/smart_data_dvc-hdd/.dvc/cache'}]
remotes += [{'host': 'toothbrush', 'hardware': 'ssd', 'path': '/data/joncrall/dvc-repos/smart_data_dvc-ssd/.dvc/cache'}]
remotes += [{'host': 'ooo',        'hardware': 'hdd', 'path': '/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache'}]
remotes += [{'host': 'ooo',        'hardware': 'ssd', 'path': '/flash/smart_data_dvc/.dvc/cache'}]
remotes += [{'host': 'yardrat',    'hardware': 'hdd', 'path': '/data/dvc-repos/smart_data_dvc-hdd/.dvc/cache'}]
remotes += [{'host': 'yardrat',    'hardware': 'ssd', 'path': '/data2/dvc-repos/smart_data_dvc/.dvc/cache'}]


def main():
    # Run this inside your DVC repo to setup the local links
    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce('.')

    this_cache_dir = ub.Path(dvc.cache_dir).resolve()

    host = platform.node()

    for remote in remotes:
        remote['is_local'] = (host == remote['host'])
        remote['is_this'] = ub.Path(remote['path']).resolve() == this_cache_dir

    df = pd.DataFrame(remotes)
    rich.print(df.to_string())

    main_tasks = []
    local_tasks = []
    for remote in remotes:
        host = remote['host']
        path = remote['path']

        if remote['hardware'] == 'ssd':
            names = [host + '_ssd']
        else:
            names = [host, host + '_hdd']

        for name in names:
            if remote['is_local'] and not remote['is_this']:
                hardware = remote['hardware']
                local_tasks.append({'cmd': f'dvc remote add --local -f {name} {path}'})
                local_tasks.append({'cmd': f'dvc remote add --local -f local_{hardware} {path}'})

            addr = addr_lut.get(host, host)
            main_tasks.append({'cmd': f'dvc remote add -f {name} ssh://{addr}{path}'})

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
