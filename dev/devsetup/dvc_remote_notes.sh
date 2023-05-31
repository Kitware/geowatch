REMOTE=toothbrush
dvc remote add $REMOTE ssh://$REMOTE/data/dvc-repos/smart_watch_dvc/.dvc/cache

REMOTE=namek
dvc remote add $REMOTE ssh://$REMOTE/data/dvc-repos/smart_watch_dvc/.dvc/cache

cd Cropped

dvc pull -R . -r toothbrush
dvc pull -R . -r namek



### Data Remotes


add_data_remotes(){
    python -c "if 1:

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

    import platform
    host = platform.node()

    for remote in remotes:
        remote['is_local'] = (host == remote['host'])

    local_remotes = [r for r in remotes if r['is_local']]

    tasks = []
    for local in local_remotes:
        cwd = local['path']
        for remote in remotes:
            if remote == local:
                continue
            host = remote['host']
            path = remote['path']

            if remote['hardware'] == 'ssd':
                names = [host + '_ssd']
            else:
                names = [host, host + '_hdd']

            for name in names:
                if remote['is_local']:
                    command = f'dvc remote add --local -f {name} {path}'
                else:
                    addr = addr_lut.get(host, host)
                    command = f'dvc remote add -f {name} ssh://{addr}{path}'
                tasks.append({
                    'cmd': command,
                    'cwd': cwd,
                })


    cwd_to_task = ub.group_items(tasks, lambda x: x['cwd'])
    for cwd, subtasks in cwd_to_task.items():
        print('---')
        print(cwd)
        print('---')
        for task in subtasks:
            print(task['cmd'])

    "
}



dvc remote add -f horologic     ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc
dvc remote add -f horologic_hdd ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc
dvc remote add -f horologic_ssd ssh://horologic.kitware.com/flash/smart_data_dvc/.dvc/cache

dvc remote add -f namek     ssh://namek/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/.dvc/cache
dvc remote add -f namek_hdd ssh://namek/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/.dvc/cache
dvc remote add -f namek_ssd ssh://namek/flash/smart_data_dvc/.dvc/cache

dvc remote add -f toothbrush     ssh://toothbrush/data/joncrall/dvc-repos/smart_data_dvc-hdd/.dvc/cache
dvc remote add -f toothbrush_hdd ssh://toothbrush/data/joncrall/dvc-repos/smart_data_dvc-hdd/.dvc/cache
dvc remote add -f toothbrush_ssd ssh://toothbrush/data/joncrall/dvc-repos/smart_data_dvc-ssd/.dvc/cache

dvc remote add -f ooo     ssh://ooo/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache -f
dvc remote add -f ooo_hdd ssh://ooo/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache -f
dvc remote add -f ooo_ssd ssh://ooo/flash/smart_data_dvc/.dvc/cache -f







### Expt Remotes

dvc remote add --local toothbrush ssh://toothbrush/data/joncrall/dvc-repos/smart_expt_dvc/.dvc/cache


# On horologic
dvc remote add --local local_store /data/dvc-caches/smart_watch_dvc


### See ALso:
"$HOME/data/dvc-repos/smart_data_dvc/Drop6/unpack.py"
