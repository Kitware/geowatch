
def setup_data_real():
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
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_RAW256/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=RAW --blocksize=256

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
            --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_imwrite_test_LZW256/data.kwcoco.json \
            --num_workers="avail/2" \
            --batch_size=32 --gpus "0" \
            --compress=LZW --blocksize=256
        ''')
    import watch
    from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataset
    import ubelt as ub
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    bundle_dpath = dvc_dpath / 'drop1-S2-L8-aligned'
    coco_fpaths = [
        bundle_dpath / 'rutgers_imwrite_test_RAW64/data.kwcoco.json',
        # bundle_dpath / 'rutgers_imwrite_test_RAW128/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_LZW64/data.kwcoco.json',
        # bundle_dpath / 'rutgers_imwrite_test_LZW128/data.kwcoco.json',
        bundle_dpath / 'rutgers_imwrite_test_DEFLATE64/data.kwcoco.json',
        # bundle_dpath / 'rutgers_imwrite_test_DEFLATE128/data.kwcoco.json',
        # bundle_dpath / 'rutgers_imwrite_test_RAW256/data.kwcoco.json',
        # bundle_dpath / 'rutgers_imwrite_test_LZW256/data.kwcoco.json',
        # bundle_dpath / 'rutgers_imwrite_test_DEFLATE256/data.kwcoco.json',
    ]
    return coco_fpaths


def setup_data_demo():
    import kwcoco
    import kwarray
    import ubelt as ub
    from kwarray import distributions

    size_distri = distributions.Uniform(600, 2000)
    size_distri = distributions.Uniform(600, 2000)

    w_distri = distributions.Uniform.coerce((400, 600))
    h_distri = distributions.Uniform.coerce((400, 600))
    # image_size = (w_distri, h_distri)
    image_size = (600, 600)

    channels = kwcoco.ChannelSpec.coerce(9)
    coco_fpaths = []

    imwrite_grid = list(ub.named_product({
        'blocksize': [32, 64, 128, 256],
        'compress': ['DEFLATE', 'RAW'],
        'interleave': ['PIXEL', 'BAND'],
    }))
    coco_fpaths = []
    for imwrite_kw in imwrite_grid:
        dset = kwcoco.CocoDataset.demo(
            'vidshapes2', num_frames=5, image_size=image_size,
            channels='a|b|c|d|e,f|g|h,i,j,k,l,m,n',
            render=imwrite_kw, use_cache=0)
        print('imwrite_kw = {!r}'.format(imwrite_kw))
        # x = dset.imgs[1]['auxiliary'][0]['file_name']
        _ = ub.cmd('gdalinfo ' + dset.imgs[1]['auxiliary'][0]['file_name'], verbose=3)
        coco_fpaths.append(dset.fpath)

    _ = ub.cmd('gdalinfo ' + dset.imgs[1]['auxiliary'][0]['file_name'], verbose=3)
    return coco_fpaths


def check_loadtime():
    import watch
    from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataset
    import kwcoco
    import timerit
    import ubelt as ub
    import ndsampler
    import os
    # dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    # coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combined_data.kwcoco.json'
    # bundle_dpath = dvc_dpath / 'drop1-S2-L8-aligned'
    # coco_fpath = bundle_dpath / 'combo_data.kwcoco.json'
    # coco_fpath = bundle_dpath / 'rutgers_imwrite_test_RAW128/data.kwcoco.json'

    num_samples = 10
    ti = timerit.Timerit(5, bestof=1, verbose=3)

    coco_fpaths = setup_data_demo()

    # coco_fpaths = [
    #     bundle_dpath / 'rutgers_imwrite_test_RAW64/data.kwcoco.json',
    #     # bundle_dpath / 'rutgers_imwrite_test_RAW128/data.kwcoco.json',
    #     bundle_dpath / 'rutgers_imwrite_test_LZW64/data.kwcoco.json',
    #     # bundle_dpath / 'rutgers_imwrite_test_LZW128/data.kwcoco.json',
    #     bundle_dpath / 'rutgers_imwrite_test_DEFLATE64/data.kwcoco.json',
    #     # bundle_dpath / 'rutgers_imwrite_test_DEFLATE128/data.kwcoco.json',
    #     # bundle_dpath / 'rutgers_imwrite_test_RAW256/data.kwcoco.json',
    #     # bundle_dpath / 'rutgers_imwrite_test_LZW256/data.kwcoco.json',
    #     # bundle_dpath / 'rutgers_imwrite_test_DEFLATE256/data.kwcoco.json',
    # ]

    @ub.memoize
    def memo_coco_dset(coco_fpath):
        coco_dset = kwcoco.CocoDataset(coco_fpath)
        return coco_dset

    @ub.memoize
    def memo_torch_dset(coco_fpath, chip_size, time_steps):
        with ub.CaptureStdout():
            coco_dset = memo_coco_dset(coco_fpath)
            sampler = ndsampler.CocoSampler(coco_dset)
            torch_dset = KWCocoVideoDataset(
                sampler=sampler, channels=channels, neg_to_pos_ratio=0.0,
                time_sampling='soft+distribute',
                time_span='1y', sample_shape=(time_steps, chip_size, chip_size), mode='test',
            )
        return torch_dset

    # channels = 'matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9'
    # channels = 'B1|B8|B8a|B11'
    channels = 'a|b|e'

    disk_stat_rows = []
    import xdev
    from osgeo import gdal
    import pandas as pd
    import kwarray
    import pathlib
    for coco_fpath in coco_fpaths:
        coco_fpath = pathlib.Path(coco_fpath)
        data_key = coco_fpath.parent.name
        coco_dset = memo_coco_dset(coco_fpath)
        candidate_gids = coco_dset.images().lookup('id')

        # Find images that have the requested channels
        sample_gids = []
        for gid in candidate_gids:
            coco_img = coco_dset._coco_image(gid)
            if (coco_img.channels & channels).numel():
                sample_gids.append(gid)

        sample_gids = sample_gids[::max(1, len(sample_gids) // num_samples)]
        sample_fpaths = []
        for gid in sample_gids:
            coco_img = watch.utils.kwcoco_extensions.CocoImage(coco_dset.imgs[gid], coco_dset)
            delayed = coco_img.delay(channels)
            for p, v in ub.IndexableWalker(delayed.__json__()):
                if p[-1] == 'fpath':
                    fpath = v
                    sample_fpaths.append(fpath)

        file_bytes = []
        for fpath in sample_fpaths:
            gdal_ds = gdal.Open(fpath, gdal.GA_ReadOnly)
            structure = gdal_ds.GetMetadata("IMAGE_STRUCTURE")
            compression = structure.get("COMPRESSION", None)
            interleave = structure.get("INTERLEAVE", None)
            if interleave is None:
                interleave = "PIXEL"
            if compression is None:
                compression = "RAW"
            main_band = gdal_ds.GetRasterBand(1)
            blocksize = main_band.GetBlockSize()[0]
            print('compression = {!r}'.format(compression))
            print('blocksize = {!r}'.format(blocksize))
            from ndsampler.utils.validate_cog import validate as _validate_cog
            warn, err, details = _validate_cog(fpath)
            # print('err = {!r}'.format(err))
            # print('details = {}'.format(ub.repr2(details, nl=1)))
            # print('warn = {}'.format(ub.repr2(warn, nl=1)))
            assert not err
            num_bytes = os.stat(fpath).st_size
            file_bytes.append(num_bytes)

        disk_stats = kwarray.stats_dict(file_bytes)
        disk_stats = ub.map_keys(lambda x: x + '_bytes', disk_stats)
        disk_stats_str = ub.map_keys(lambda x: x + '_str', ub.dict_diff(disk_stats, {'shape_bytes'}))
        disk_stats_str = ub.map_vals(xdev.byte_str, disk_stats_str)
        data_key = coco_fpath.parent.name
        disk_stats_row = {
            'data_key': data_key,
            'compress': compression,
            'interleave': interleave,
            'blocksize': blocksize,
            **disk_stats,
            **disk_stats_str,
        }
        coco_dset.disk_stats_row = disk_stats_row
        disk_stat_rows.append(disk_stats_row)
    disk_df = pd.DataFrame(disk_stat_rows)
    disk_df = disk_df.sort_values('max_bytes')
    disk_df['total_ave_bytes_str'] = (disk_df.mean_bytes * coco_dset.n_images).apply(xdev.byte_str)
    print(disk_df)

    import kwplot
    kwplot.autompl()
    sns = kwplot.autosns()

    disk_piv = disk_df.pivot(['compress'], ['blocksize'], ['mean_bytes'])
    disk_piv = disk_piv.droplevel(0, axis=1)
    fig = kwplot.figure(fnum=9001)
    # fig.set_size_inches(8.69, 4.93)
    ax = fig.gca()
    annot = disk_piv.applymap(lambda x: '{:0.2f}'.format(x))
    disk_annot = disk_piv.applymap(xdev.byte_str)
    sns.heatmap(disk_piv,
                annot=disk_annot,
                ax=ax, fmt='s',
                # norm=LogNorm(vmin=1, vmax=24),
                annot_kws={'size': 8},
                cbar_kws={'label': 'bytes', 'pad': 0.001})
    ax.set_title('Disk usage based on imwrite settings')

    # ----
    sample_grid = list(ub.named_product({
        'chip_size': [32, 64, 96],
        # 'chip_size': [32, 96],
        # 'time_steps': [1, 3, 5, 7],
        'time_steps': [
            1,
            # 2,
            3,
            5,
            # 7,
            # 11,
            # 13,
            # 17,
            # 19
        ],
    }))

    sample_time_rows = []
    for sample_kw in sample_grid:
        sample_key = ub.repr2(sample_kw, compact=1)

        for coco_fpath in coco_fpaths:
            coco_fpath = pathlib.Path(coco_fpath)
            data_key = coco_fpath.parent.name
            chip_size = sample_kw['chip_size']
            time_steps = sample_kw['time_steps']
            torch_dset = memo_torch_dset(coco_fpath, chip_size, time_steps)
            coco_dset = torch_dset.sampler.dset

            total = len(torch_dset)
            indexes = list(range(0, total, total // num_samples))

            key = sample_key + '_' + data_key
            for timer in ti.reset(key):
                with timer:
                    for index in indexes:
                        torch_dset[index]

            data_kw = {}
            # if 'DEFLATE' in data_key:
            #     data_kw['compress'] = 'DEFLATE'
            # elif 'LZW' in data_key:
            #     data_kw['compress'] = 'LZW'
            # elif 'RAW' in data_key:
            #     data_kw['compress'] = 'RAW'
            # else:
            #     data_kw['compress'] = 'DEFLATE'

            # if '128' in data_key:
            #     data_kw['blocksize'] = 128
            # elif '256' in data_key:
            #     data_kw['blocksize'] = 256
            # elif '64' in data_key:
            #     data_kw['blocksize'] = 64
            # else:
            #     data_kw['blocksize'] = 256

            data_kw['compress'] = coco_dset.disk_stats_row['compress']
            data_kw['interleave'] = coco_dset.disk_stats_row['interleave']
            data_kw['blocksize'] = coco_dset.disk_stats_row['blocksize']

            # Double check we are measureing what we think we are measuring
            # common = ub.dict_isect(coco_dset.disk_stats_row, data_kw)
            # assert common == data_kw

            sample_time_rows.append({
                'key': key,
                'sample_key': sample_key,
                'data_key': data_key,
                'mean_bytes': coco_dset.disk_stats_row['mean_bytes'],
                'min': ti.min(),
                'mean': ti.mean(),
                **sample_kw,
                **data_kw,
            })

    for row in sample_time_rows:
        num_pixels = row['time_steps'] * (row['chip_size'] ** 2)
        row['num_pixels'] = num_pixels

    import pandas as pd
    df = pd.DataFrame(sample_time_rows)
    # df = df[df.data_key != 'drop1-S2-L8-aligned']
    # print(df.sort_values('mean'))
    # df.set_index(['compress', 'blocksize'])
    # print(df.set_index(['chip_size', 'time_steps', 'num_pixels']).to_string())

    # z = df.melt(id_vars=['compress', 'blocksize', 'time_steps', 'chip_size'], value_vars=['mean'], var_name='var', value_name='val')
    # z.pivot(['compress', 'blocksize'], ['chip_size', 'time_steps'], ['val'])
    # z.pivot(
    df = df.rename({'mean': 'mean_seconds'}, axis=1)
    from matplotlib.colors import LogNorm
    piv = df.pivot(['compress', 'blocksize', 'interleave'], ['time_steps', 'chip_size', 'num_pixels'], ['mean_seconds'])
    piv = piv.droplevel(0, axis=1)

    piv = piv.sort_index(axis=1)
    fig = kwplot.figure(fnum=1)
    # fig.set_size_inches(8.69, 4.93)
    ax = fig.gca()
    annot = piv.applymap(lambda x: '{:0.2f}'.format(x))
    sns.heatmap(piv,
                annot=annot,
                ax=ax, fmt='s',
                norm=LogNorm(vmin=0.2, vmax=5),
                annot_kws={'size': 8},
                cbar_kws={'label': 'seconds', 'pad': 0.001})
    ax.set_title('Load times based on COG and Sampler settings')
    ax.figure.subplots_adjust(bottom=0.2)

    fig = kwplot.figure(fnum=2)

    fig.clf()
    groups = dict(list(df.groupby('chip_size')))
    pnum_ = kwplot.PlotNums(nSubplots=len(groups))
    for chip_size, subdf in groups.items():
        ax = kwplot.figure(pnum=pnum_(), fnum=fig.number).gca()
        sns.lineplot(ax=ax, data=subdf, x='time_steps', y='mean_seconds', hue='compress', style='blocksize')
        ax.set_title('sampler chip_size={}'.format(chip_size))
        ax.set_ylim(0, df['mean_seconds'].max())

    fig = kwplot.figure(fnum=3)
    ax = fig.gca()
    ax.cla()
    sns.lineplot(ax=ax, data=df, x='compress', y='mean_bytes', hue='blocksize')
    # hue='compress', style='blocksize')

    fig = kwplot.figure(fnum=4)
    ax = fig.gca()
    sns.lineplot(ax=ax, data=df, x='num_pixels', y='mean_seconds', hue='data_key', style='chip_size')

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
