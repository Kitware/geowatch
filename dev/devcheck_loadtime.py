def check_loadtime():
    import ubelt as ub
    ub.codeblock(
        r'''

        DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
        KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
        BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
        RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021-10-01-16:27:07.pth"
        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_RAW128/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=RAW --blocksize=128

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_RAW64/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=RAW --blocksize=64

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_RAW256/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=RAW --blocksize=256

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_DEFLATE128/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=DEFLATE --blocksize=128

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_DEFLATE64/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=DEFLATE --blocksize=64

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_DEFLATE256/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=DEFLATE --blocksize=256

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_LZW128/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=LZW --blocksize=128

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_LZW64/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=LZW --blocksize=64

        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset=$BASE_COCO_FPATH \
            --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
            --default_config_key=iarpa \
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_LZW256/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=LZW --blocksize=256
        ''')

    import watch
    from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataset
    import kwcoco
    import numpy as np
    import timerit
    import ndsampler
    import os
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combined_data.kwcoco.json'
    bundle_dpath = dvc_dpath / 'drop1-S2-L8-aligned'

    coco_fpath = bundle_dpath / 'combo_data.kwcoco.json'

    coco_fpath = bundle_dpath / 'rutgers_imwrite_test_RAW128/data.kwcoco.json'

    num_samples = 10
    ti = timerit.Timerit(5, bestof=1, verbose=3)

    coco_fpaths = [
        bundle_dpath / 'rutgers_imwrite_test_RAW64/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_RAW128/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_LZW64/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_LZW128/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_DEFLATE64/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_DEFLATE128/data.kwcoco.json',
        # bundle_dpath / 'combo_data.kwcoco.json',
    ]

    @ub.memoize
    def memo_torch_dset(coco_fpath, chip_size, time_steps):
        with ub.CaptureStdout():
            coco_dset = kwcoco.CocoDataset(coco_fpath)
            sampler = ndsampler.CocoSampler(coco_dset)
            torch_dset = KWCocoVideoDataset(
                sampler=sampler, channels=channels, neg_to_pos_ratio=0.0,
                time_sampling='soft+distribute',
                time_span='1y', sample_shape=(time_steps, chip_size, chip_size), mode='test',
            )
        return torch_dset

    channels = 'matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9'

    sample_grid = ub.named_product({
        'chip_size': [32, 64, 96],
        'time_steps': [1, 3, 5, 7],
    })

    rows = []
    for sample_kw in sample_grid:
        sample_key = ub.repr2(sample_kw, compact=1)

        for coco_fpath in coco_fpaths:
            data_key = coco_fpath.parent.name

            # watch.utils.kwcoco_extensions.CocoImage(coco_dset.imgs[2], coco_dset).channels
            chip_size = sample_kw['chip_size']
            time_steps = sample_kw['time_steps']
            memo_torch_dset(coco_fpath, chip_size, time_steps)

            candidate_gids = coco_dset.images().lookup('id')
            sample_gids = []
            for gid in candidate_gids:
                coco_img = watch.utils.kwcoco_extensions.CocoImage(coco_dset.imgs[gid], coco_dset)
                if (coco_img.channels & channels).numel():
                    sample_gids.append(gid)

            sample_gids = sample_gids[::max(1, len(sample_gids) // num_samples)]
            total = len(torch_dset)
            indexes = list(range(0, total, total // num_samples))

            key = sample_key + '_' + data_key
            for timer in ti.reset(key):
                with timer:
                    for index in indexes:
                        # tr = torch_dset.new_sample_grid['targets'][index].copy()
                        # tr['gids'] = sample_gids[0:2]
                        # torch_dset[tr]
                        torch_dset[index]

            file_bytes = []
            for gid in sample_gids:
                coco_img = watch.utils.kwcoco_extensions.CocoImage(coco_dset.imgs[gid], coco_dset)
                delayed = coco_img.delay(channels)
                for p, v in ub.IndexableWalker(delayed.__json__()):
                    if p[-1] == 'fpath':
                        fpath = v
                        num_bytes = os.stat(fpath).st_size
                        file_bytes.append(num_bytes)
            disk_bytes = np.mean(file_bytes)

            data_kw = {}
            if 'DEFLATE' in data_key:
                data_kw['compress'] = 'DEFLATE'
            elif 'LZW' in data_key:
                data_kw['compress'] = 'LZW'
            elif 'RAW' in data_key:
                data_kw['compress'] = 'RAW'
            else:
                data_kw['compress'] = 'DEFLATE'

            if '128' in data_key:
                data_kw['blocksize'] = 128
            elif '256' in data_key:
                data_kw['blocksize'] = 256
            elif '64' in data_key:
                data_kw['blocksize'] = 64
            else:
                data_kw['blocksize'] = 256

            rows.append({
                'key': key,
                'sample_key': sample_key,
                'data_key': data_key,
                'min': ti.min(),
                'mean': ti.mean(),
                'disk_bytes': disk_bytes,
                **sample_kw,
                **data_kw,
            })

    for row in rows:
        num_pixels = row['time_steps'] * (row['chip_size'] ** 2)
        row['num_pixels'] = num_pixels
        data_key = row['data_key']
        data_kw = {}
        if 'DEFLATE' in data_key:
            data_kw['compress'] = 'DEFLATE'
        elif 'LZW' in data_key:
            data_kw['compress'] = 'LZW'
        elif 'RAW' in data_key:
            data_kw['compress'] = 'RAW'
        else:
            data_kw['compress'] = 'DEFLATE'
        row.update(data_kw)

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[df.data_key != 'drop1-S2-L8-aligned']
    print(df.sort_values('mean'))

    piv = df.pivot(['compress', 'blocksize'], ['chip_size', 'time_steps', 'num_pixels'], ['mean'])

    df.set_index(['compress', 'blocksize'])
    print(df.set_index(['chip_size', 'time_steps', 'num_pixels']).to_string())

    z = df.melt(id_vars=['compress', 'blocksize', 'time_steps', 'chip_size'], value_vars=['mean'], var_name='var', value_name='val')
    z.pivot(['compress', 'blocksize'], ['chip_size', 'time_steps'], ['val'])
    z.pivot(
    import kwplot
    kwplot.autompl()
    sns = kwplot.autosns()
    # from matplotlib.colors import LogNorm
    fig = kwplot.figure(fnum=1)
    # fig.set_size_inches(8.69, 4.93)
    ax = fig.gca()
    annot = piv.applymap(lambda x: '{:0.2f}'.format(x))
    sns.heatmap(piv,
                annot=annot,
                ax=ax, fmt='s',
                # norm=LogNorm(vmin=1, vmax=24),
                annot_kws={'size': 8},
                cbar_kws={'label': 'seconds', 'pad': 0.001})
    ax.figure.subplots_adjust(bottom=0.2)

    fig = kwplot.figure(fnum=2)
    ax = fig.gca()
    sns.lineplot(ax=ax, data=df, x='num_pixels', y='mean', hue='compress', style='blocksize')

    fig = kwplot.figure(fnum=3)
    ax = fig.gca()
    sns.lineplot(ax=ax, data=df, x='compress', y='disk_bytes', hue='blocksize')
    # hue='compress', style='blocksize')

    fig = kwplot.figure(fnum=4)
    ax = fig.gca()
    sns.lineplot(ax=ax, data=df, x='num_pixels', y='mean', hue='data_key', style='chip_size')

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.set_title(title)
    # ax.set_xlabel('Number of Modes (M)')
    # ax.set_ylabel('Space (S) Time (S) dims')

    # item = torch_dset[0]
    # import kwplot
    # kwplot.autompl()
    # torch_dset.default_combinable_channels.extend(list(map(ub.oset, ub.chunks(channels.split('|'), 3))))
    # canvas = torch_dset.draw_item(item, overlay_on_image=0, norm_over_time=0, max_channels=5)
    # kwplot.imshow(canvas)
