"""
Check how long it takes to iterate through an entire dataset on different disk
hardware / filesystems.
"""


def main():
    import ubelt as ub
    import kwcoco
    import platform
    from watch.utils import util_hardware

    hostname = platform.node()
    print('hostname = {!r}'.format(hostname))

    # dvc_dpath = ub.Path('$HOME/data/dvc-repos/smart_watch_dvc').expand()
    dvc_dpath = ub.Path('$HOME/data/dvc-repos/smart_watch_dvc-hdd').expand()
    # dvc_dpath = ub.Path('$HOME/data/dvc-repos/smart_watch_dvc-ssd').expand()

    coco_fpath = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json'

    system_info = util_hardware.get_cpu_mem_info()
    disk_info = util_hardware.disk_info_of_path(coco_fpath)
    print('disk_info = {!r}'.format(disk_info))

    start_timestamp = ub.timestamp()

    measures = []

    import sys
    args = repr(sys.argv)

    benchmark_info = {
        'type': 'process',
        'properties': {
            'name': 'benchmark',
            'args': args,
            'system_info': system_info,
            'disk_info': disk_info,
            'hostname': hostname,
            'start_timestamp': start_timestamp,
            'coco_fpath': coco_fpath,
        }
    }

    with ub.Timer('load_coco') as timer:
        coco_dset = kwcoco.CocoDataset(coco_fpath)

    measures.append({
        'type': 'measure',
        'properties': {
            'label': 'load_coco',
            'seconds': timer.elapsed,
            'size': coco_fpath.stat().st_size,
        }
    })

    # import kwarray
    gids = list(coco_dset.images())

    coco_img_list = list(coco_dset.images().coco_images)

    if 1:
        import kwarray
        gids = kwarray.shuffle(gids, rng=0)[0:100]

    fpaths = []
    for gid in gids:
        coco_img = coco_dset.coco_image(gid)
        for fpath in coco_img.iter_image_filepaths():
            fpath = ub.Path(fpath)
            fpaths.append(fpath)

    ##
    with ub.Timer('iter_image_exists') as timer:
        prog = ub.ProgIter(fpaths)
        for fpath in prog:
            fpath.exists()
    measures.append({
        'type': 'measure',
        'properties': {
            'label': timer.label,
            'seconds': timer.elapsed,
            'num_images': len(gids),
        }
    })

    ##
    with ub.Timer('iter_image_stat') as timer:
        prog = ub.ProgIter(fpaths)
        for fpath in prog:
            fpath.stat()
    measures.append({
        'type': 'measure',
        'properties': {
            'label': timer.label,
            'seconds': timer.elapsed,
            'num_images': len(gids),
            'Hz': prog._iters_per_second,
        }
    })

    ##
    with ub.Timer('iter_image_load') as timer:
        prog = ub.ProgIter(coco_img_list)
        errors = []
        per_img_times = []
        for coco_img in prog:
            for stream in coco_img.channels.streams():
                with ub.Timer() as t2:
                    delayed = coco_img.delay(stream, space='asset')
                    try:
                        delayed.finalize()
                    except Exception as ex:
                        errors.append(ex)
                        pass
                per_img_times.append(t2.elapsed)

    measures.append({
        'type': 'measure',
        'properties': {
            'label': timer.label,
            'seconds': timer.elapsed,
            'num_images': len(gids),
            'num_errors': len(errors),
            'per_img_times': per_img_times,
        }
    })

    ##
    with ub.Timer('iter_image_load_second_pass') as timer:
        prog = ub.ProgIter(coco_img_list)
        errors = []
        per_img_times = []
        for coco_img in prog:
            for stream in coco_img.channels.streams():
                with ub.Timer() as t2:
                    delayed = coco_img.delay(stream, space='asset')
                    try:
                        delayed.finalize()
                    except Exception as ex:
                        errors.append(ex)
                        pass
                per_img_times.append(t2.elapsed)

    measures.append({
        'type': 'measure',
        'properties': {
            'label': timer.label,
            'seconds': timer.elapsed,
            'num_images': len(gids),
            'num_errors': len(errors),
            'per_img_times': per_img_times,
        }
    })

    end_timestamp = ub.timestamp()
    benchmark_info['end_timestamp'] = end_timestamp

    benchmark = {
        'info': [benchmark_info],
        'measures':  measures,
    }

    for m in measures:
        if m['properties']['label'] == 'iter_image_load':
            print(m['properties']['label'])
            import pandas as pd
            try:
                print(pd.DataFrame({'seconds': m['properties']['per_img_times']}).describe().T)
            except Exception:
                pass
        if m['properties']['label'] == 'iter_image_load_second_pass':
            print(m['properties']['label'])
            import pandas as pd
            try:
                print(pd.DataFrame({'seconds': m['properties']['per_img_times']}).describe().T)
            except Exception:
                pass

    print('disk_info = {!r}'.format(disk_info))

    # Save based on timestamp
    import safer
    import json
    from kwcoco.util import util_json
    benchmark = util_json.ensure_json_serializable(benchmark)
    fpath = ub.Path(f'benchmark-filesys-iteration-crop-{ub.timestamp()}.json')
    with safer.open(fpath, 'w', temp_file=True) as f:
        json.dump(benchmark, f)



def gathering():
    import ubelt as ub

    import kwplot
    kwplot.autompl()
    sns = kwplot.autosns()
    sns.set()

    suffixes = [
        '',
        'data/dvc-repos/smart_watch_dvc',
        'data/dvc-repos/smart_watch_dvc-hdd',
        'code/watch',
    ]

    remotes = [
        ub.Path('$HOME/remote/namek').expand(),
        ub.Path('$HOME/remote/toothbrush').expand(),
        ub.Path('$HOME/remote/ooo').expand(),
        ub.Path('$HOME/remote/horologic').expand(),
    ]
    candidates = []
    for remote in remotes:
        for suff in suffixes:
            dpath = remote / suff
            print(f'{dpath=}')
            candidates += list(dpath.glob('benchmark-filesys-*'))

    datas = []
    for fpath in ub.ProgIter(candidates):
        import json
        with open(fpath, 'r') as file:
            data = json.load(file)
        data['fpath'] = str(fpath)
        datas.append(data)

    rows = []
    trial = 0
    for data in ub.ProgIter(datas):
        data['info']
        try:
            info_prop = data['info'][0]['properties']
        except KeyError:
            info_prop = data['info'][0]

        fs = info_prop['disk_info']['filesystem']
        disks = '-'.join(info_prop['disk_info']['names'])
        cpu = info_prop['system_info']['cpu_info']['brand_raw']

        import pint
        ureg = pint.UnitRegistry()
        ram = int((info_prop['system_info']['mem_info']['total'] * ureg.bytes).to('GiB').m)

        common = {
            'cpu': cpu,
            'filesystem': fs,
            'hostname': info_prop['hostname'],
            'disks': disks,
            'ram': ram,
        }

        for m in data['measures']:
            trial += 1
            prop = m['properties']
            label = prop[ 'label']
            if 'per_img_times' in prop:
                for time in m['properties']['per_img_times']:
                    row = {
                        'label': label,
                        'trial': trial,
                        'per_img_time': time,
                        **common,
                    }
                    rows.append(row)
            else:
                row = {
                    'label': label,
                    'trial': trial,
                    'seconds': prop['seconds'],
                    **common,
                }
                rows.append(row)

    for row in ub.ProgIter(rows):
        key = '{}-{}-{}'.format(row['filesystem'], row['disks'], row['ram'])
        row['key'] = key

    import pandas as pd
    df = pd.DataFrame(rows)
    load_df = df[(df['label'] == 'iter_image_load') | (df['label'] == 'iter_image_load_second_pass')]
    ax = sns.violinplot(data=load_df, x='trial', y='per_img_time',
                        # scale="width",
                        scale="area",
                        inner="quartile", hue='label', split=True, cut=0)
    ax.set_yscale('log')
    ax.set_title('Per Image Loading Time')







if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/check_filesys_iteration_time.py
    """
    main()
