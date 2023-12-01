r"""
Transfering cold features

CommandLine:

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m geowatch.tasks.cold.transfer_features \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-HTR.kwcoco.zip" \
        --combine_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip" \
        --new_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip"

"""
import os
import ubelt as ub
import kwcoco
import kwimage
from os.path import join
from kwutil import util_time
import scriptconfig as scfg

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class TransferCocoConfig(scfg.DataConfig):
    """
    Transfer channels to one kwcoco file to the nearest image in the future.
    """

    src_kwcoco = scfg.Value(None, position=1, help=ub.paragraph(
        '''
        a path to a file to input kwcoco file (to predict on)
        '''), alias=['coco_fpath'])
    dst_kwcoco = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined input kwcoco file (to merge with)
        '''), alias=['combine_fpath'])
    new_coco_fpath = scfg.Value(None, help='file path for modified output coco json')
    channels_to_transfer = scfg.Value(None, help='COLD channels for transfer')

    io_workers = scfg.Value(0, help='number of workers for copy-asset jobs')

    copy_assets = scfg.Value(False, help='if True copy the assests to the new bundle directory')

    respect_sensors = scfg.Value(
        True, help='if True only transfer features to images that share a sensor')

    # TODO: propogate strategy?
    max_propogate = scfg.Value(1, help='maximum number of future images a src image can transfer onto.')

    allow_affine_approx = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        if True allow a transfer between different CRSs by approximating an affine transform. This should only be used for quick-and-dirty analysis, and never in production
        '''))


@profile
def transfer_features_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m geowatch.tasks.cold.transfer_features --help
        TEST_COLD=1 xdoctest -m geowatch.tasks.cold.transfer_features transfer_features_main

     Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from geowatch.tasks.cold.transfer_features import transfer_features_main
        >>> from geowatch.tasks.cold.transfer_features import *
        >>> kwargs= dict(
        >>>   coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip'),
        >>>   combine_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'),
        >>>   new_coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold_test.kwcoco.zip'),
        >>>   #workermode = 'process',
        >>> )
        >>> cmdline=0
        >>> transfer_features_main(cmdline, **kwargs)

    Example:
        >>> from geowatch.tasks.cold.transfer_features import transfer_features_main
        >>> from geowatch.tasks.cold.transfer_features import *
        >>> import geowatch
        >>> dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, heatmap=True, dates=True)
        >>> dset2 = dset1.copy()
        >>> # Remove saliency assets from dset2
        >>> for img in dset2.images().coco_images:
        >>>     asset = img.find_asset_obj('salient')
        >>>     img['auxiliary'].remove(asset)
        >>> dset2_orig = dset2.copy()
        >>> # Transfer the saliency from dset1 onto dset2
        >>> kwargs = TransferCocoConfig(**{
        >>>     'src_kwcoco': dset1,
        >>>     'dst_kwcoco': dset2,
        >>>     'new_coco_fpath': 'return',
        >>>     'channels_to_transfer': 'salient',
        >>>     'copy_assets': False,
        >>> })
        >>> cmdline = False
        >>> new_dset = transfer_features_main(cmdline, **kwargs)
        >>> assert new_dset is dset2, 'modifies combine_fpath inplace'
        >>> from geowatch.utils import kwcoco_extensions
        >>> stats1 = kwcoco_extensions.coco_channel_stats(dset1)
        >>> stats2 = kwcoco_extensions.coco_channel_stats(dset2)
        >>> stats2_orig = kwcoco_extensions.coco_channel_stats(dset2_orig)
        >>> assert stats2['single_chan_hist']['salient'] == stats1['single_chan_hist']['salient'], 'all assets should transfer'
        >>> assert stats2['single_chan_hist']['salient'] == dset2.n_images, 'all dst images should get saliency here'
        >>> assert stats2_orig['single_chan_hist']['salient'] == 0
    """
    #NOTE: This script doesn't consider timestamp = True
    config = TransferCocoConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    from geowatch.cli.reproject_annotations import keyframe_interpolate
    from geowatch.utils import process_context
    from kwutil import util_json

    resolved_config = config.to_dict()
    resolved_config = util_json.ensure_json_serializable(resolved_config)
    proc_context = process_context.ProcessContext(
        name='geowatch.tasks.cold.transfer_features',
        type='process',
        config=resolved_config,
    )
    proc_context.start()

    # Assign variables
    default_channels = [
        'blue_COLD_cv', 'green_COLD_cv', 'red_COLD_cv', 'nir_COLD_cv',
        'swir16_COLD_cv', 'swir22_COLD_cv', 'blue_COLD_a0', 'green_COLD_a0',
        'red_COLD_a0', 'nir_COLD_a0', 'swir16_COLD_a0', 'swir22_COLD_a0',
        'blue_COLD_a1', 'green_COLD_a1', 'red_COLD_a1', 'nir_COLD_a1',
        'swir16_COLD_a1', 'swir22_COLD_a1', 'blue_COLD_b1', 'green_COLD_b1',
        'red_COLD_b1', 'nir_COLD_b1', 'swir16_COLD_b1', 'swir22_COLD_b1',
        'blue_COLD_c1', 'green_COLD_c1', 'red_COLD_c1', 'nir_COLD_c1',
        'swir16_COLD_c1', 'swir22_COLD_c1', 'blue_COLD_rmse',
        'green_COLD_rmse', 'red_COLD_rmse', 'nir_COLD_rmse',
        'swir16_COLD_rmse', 'swir22_COLD_rmse'
    ]

    do_return = (config['new_coco_fpath'] == 'return')
    if not do_return:
        new_coco_fpath = ub.Path(config['new_coco_fpath'])

    if config['channels_to_transfer'] is None:
        channels_to_transfer = kwcoco.FusedChannelSpec.coerce(default_channels)
    else:
        channels_to_transfer = kwcoco.FusedChannelSpec.coerce(config['channels_to_transfer'])

    print('Loading source kwcoco file')
    src = kwcoco.CocoDataset.coerce(config['src_kwcoco'])

    print('Loading destination kwcoco file')
    dst = kwcoco.CocoDataset.coerce(config['dst_kwcoco'])

    print(f'src={src}')
    print(f'dst={dst}')

    if not do_return:
        _update_coco_fpath(dst, new_coco_fpath)

    src_video_names = src.videos().lookup('name')

    dst_vidnames = sorted(set(dst.index.name_to_video))
    src_vidnames = sorted(set(src.index.name_to_video))
    if src_vidnames != dst_vidnames:
        # If video names do not agree, we need to check for overlaps
        from geowatch.utils import kwcoco_extensions
        from geowatch.utils import util_gis
        src_vid_gdf = kwcoco_extensions.covered_video_geo_regions(src)
        dst_vid_gdf = kwcoco_extensions.covered_video_geo_regions(dst)
        dst_to_src_idxs = util_gis.geopandas_pairwise_overlaps(dst_vid_gdf, src_vid_gdf)
        # For each site, chose a single video assign to Note: this only works
        # well when the dst are smaller than the src, which is the case it is
        # being written for. If larger dst are needed the logic will need to
        # change.
        vidname_pairs = []
        for dst_idx, src_idxs in dst_to_src_idxs.items():
            if len(src_idxs) == 0:
                continue
            elif len(src_idxs) == 1:
                idx = 0
            else:
                qshape = dst_vid_gdf.iloc[dst_idx]['geometry']
                candidates = src_vid_gdf.iloc[src_idxs]
                overlaps = []
                for dshape in candidates['geometry']:
                    iarea = qshape.intersection(dshape).area
                    uarea = qshape.area
                    iofa = iarea / uarea
                    overlaps.append(iofa)
                idx = ub.argmax(overlaps)

            dst_vidname = dst_vid_gdf.iloc[dst_idx].video_name
            src_vidname = src_vid_gdf.iloc[src_idxs[idx]].video_name
            vidname_pairs.append((src_vidname, dst_vidname))
    else:
        vidname_pairs = list(zip(src_video_names, dst_vidnames))
        ...

    # We will build a list containing all of the assignments to make
    assignments = []
    for src_vidname, dst_vidname in vidname_pairs:
        # For each corresponding video
        src_video = src.index.name_to_video[src_vidname]
        dst_video = dst.index.name_to_video[dst_vidname]
        # Look at each sequence of images
        all_src_images = src.images(video_id=src_video['id'])
        dst_images = dst.images(video_id=dst_video['id'])

        # Filter out the source images missing the channels we want to transfer
        keep_flags = [
            coco_img.channels.intersection(channels_to_transfer).numel() > 0
            for coco_img in all_src_images.coco_images
        ]
        src_images = all_src_images.compress(keep_flags)

        # Build a list of columns used as group-ids which will be used to
        # ensure src images only transfer to dst images with the same group-id.
        src_groupers = []
        dst_groupers = []

        if config.respect_sensors:
            # Use sensors in group-ids, so we only transfer between similar
            # sensors
            src_sensors = src_images.lookup('sensor_coarse')
            dst_sensors = dst_images.lookup('sensor_coarse')
            src_groupers.append(src_sensors)
            dst_groupers.append(dst_sensors)

        if src_groupers:
            src_groupids = zip(*src_groupers)
            dst_groupids = zip(*dst_groupers)
            groupid_to_src_gids = ub.group_items(src_images, src_groupids)
            groupid_to_dst_gids = ub.group_items(dst_images, dst_groupids)
        else:
            # No groupers were given, so any src can transfer to any dst.
            groupid_to_src_gids = {'__nogroup__': src_images}
            groupid_to_dst_gids = {'__nogroup__': dst_images}

        for groupid in groupid_to_src_gids.keys():
            src_image_ids = groupid_to_src_gids[groupid]
            dst_image_ids = groupid_to_dst_gids[groupid]

            src_subimgs = src.images(src_image_ids)
            dst_subimgs = dst.images(dst_image_ids)

            # Find the timestamps for each image in both sequences
            src_timestamps = [util_time.coerce_datetime(d) for d in src_subimgs.lookup('date_captured')]
            dst_timestamps = [util_time.coerce_datetime(d) for d in dst_subimgs.lookup('date_captured')]
            assert sorted(src_timestamps) == src_timestamps, 'data should be in order'
            assert sorted(dst_timestamps) == dst_timestamps, 'data should be in order'

            # We will use logic in reproject annotations to assign images with COLD
            # features to the nearest image in the future. In this case the
            # keyframes are the images with COLD features and the target sequence
            # are the images we are transfering onto.
            target_times = dst_timestamps

            key_infos = [{'time': d, 'applies': 'future'} for d in src_timestamps]
            assigned_indexes = keyframe_interpolate(target_times, key_infos)
            # The above function returns a list for each key-frame with the
            # assigned indexes but we are only going to assign it to one of them,
            # so choose the earliest one in each group. The minimum index will be
            # the earliest because dates are sorted.
            for src_idx, dst_idxs in enumerate(assigned_indexes):
                if dst_idxs:
                    dst_idxs = sorted(dst_idxs)
                    chosen_dst_idxs = dst_idxs[:config.max_propogate]
                    for dst_idx in chosen_dst_idxs:
                        assignments.append({
                            'src_image_id': src_image_ids[src_idx],
                            'dst_image_id': dst_image_ids[dst_idx],
                            'src_vidname': src_vidname,
                            'dst_vidname': dst_vidname,
                        })

    print(f'Found {len(assignments)} image-to-image assignments to transfer')
    assets_to_transfer = []

    # Now we have image assignments, find the assets to transfer
    vidname_to_assignments = ub.group_items(assignments, lambda x: (x['src_vidname'], x['dst_vidname']))
    for vidnames, vid_assignments in vidname_to_assignments.items():
        src_vidname, dst_vidname = vidnames
        src_video = src.index.name_to_video[src_vidname]
        dst_video = dst.index.name_to_video[dst_vidname]

        if 0:
            # OVERSIMPLIFIED LOGIC
            # In most cases the source and destination videos will have the exact same
            # width / height, but in some cases they may differ if one is downsampled.
            # Just to be safe, assume the videos align and compute a scale factor
            # transform to account for this case (its just the identity if width/height
            # are the same).
            fx = dst_video['width'] / src_video['width']
            fy = dst_video['height'] / src_video['height']
            warp_dstvid_from_srcvid = kwimage.Affine.scale((fx, fy))
        else:
            # Better logic
            dst_crs = dst_video['wld_crs_info']
            src_crs = src_video['wld_crs_info']

            warp_src_from_wld = kwimage.Affine.coerce(src_video['warp_wld_to_vid'])
            warp_dst_from_wld = kwimage.Affine.coerce(dst_video['warp_wld_to_vid'])
            warp_wld_from_src = warp_src_from_wld.inv()

            warp_dstwld_from_srcwld = None

            if dst_crs != src_crs:
                msg = ub.codeblock(
                    f'''
                    Expected the same CRS but got:
                    dst_crs={dst_crs}
                    src_crs={src_crs}
                    ''')
                ALLOW_AFFINE_APPROXIMATE = config.allow_affine_approx
                if not ALLOW_AFFINE_APPROXIMATE:
                    raise AssertionError(msg)
                else:
                    rich.print('[yellow]WARNING: ' + msg)
                    rich.print('[yellow]WARNING: Estimating an approximate affine transform')
                    # We can construct a non-affine transform between the two CRS
                    # values, but because we are only writing metadata and not
                    # rewriting the pixel values, we cant use it. (kwcoco metadata
                    # only allows for affine transforms).
                    import pyproj
                    import numpy as np
                    # crs1 = pyproj.CRS.from_epsg(32639)
                    # crs2 = pyproj.CRS.from_epsg(32638)
                    crs1 = pyproj.CRS.from_authority(*src_crs['auth'])
                    crs2 = pyproj.CRS.from_authority(*dst_crs['auth'])
                    crs_tf = pyproj.Transformer.from_crs(crs_from=crs1, crs_to=crs2)

                    src_pxl_valid_region = kwimage.MultiPolygon.coerce(src_video['valid_region'])
                    src_wld_valid_region = src_pxl_valid_region.warp(warp_wld_from_src)
                    src_pts = np.concatenate([p.data['exterior'].data for p in src_wld_valid_region.data], axis=0)

                    # We now try be far too clever and estimate an affine
                    # approximation that gets does a good job in the region of
                    # the source image.
                    import numpy as np
                    xx1 = src_pts.T[0]
                    yy1 = src_pts.T[1]
                    xx2, yy2 = crs_tf.transform(xx1, yy1)
                    pts1 = np.stack([xx1, yy1], axis=1)
                    pts2 = np.stack([xx2, yy2], axis=1)
                    approx = kwimage.Affine.fit(pts1, pts2)
                    pts2_hat = kwimage.Points(xy=pts1).warp(approx)
                    error = np.abs(pts2_hat.xy - pts2)
                    max_error = error.max(axis=0)
                    ave_error = error.mean(axis=0)
                    print(f'ave_error={ave_error}')
                    print(f'max_error={max_error}')
                    warp_dstwld_from_srcwld = approx

            if warp_dstwld_from_srcwld is not None:
                # Using the hack
                warp_dstvid_from_srcvid = warp_dst_from_wld @ warp_dstwld_from_srcwld @ warp_wld_from_src
            else:
                warp_dstvid_from_srcvid = warp_dst_from_wld @ warp_wld_from_src

        for assignment in vid_assignments:
            src_image_id = assignment['src_image_id']
            dst_image_id = assignment['dst_image_id']
            src_coco_img = src.coco_image(src_image_id)
            dst_coco_img = dst.coco_image(dst_image_id)

            # Compute the alignment between assigned images.
            warp_srcvid_from_srcimg = src_coco_img.warp_vid_from_img
            warp_dstimg_from_dstvid = dst_coco_img.warp_img_from_vid
            warp_dstimg_from_srcimg = (
                warp_dstimg_from_dstvid @
                warp_dstvid_from_srcvid @
                warp_srcvid_from_srcimg
            )

            # For each asset in the src, check if we want to transfer it.
            img_assets_to_transfer = []
            for asset in src_coco_img.iter_asset_objs():
                asset_channels = kwcoco.FusedChannelSpec.coerce(asset['channels'])
                if asset_channels.intersection(channels_to_transfer):
                    img_assets_to_transfer.append(asset)

            for old_asset in img_assets_to_transfer:
                new_asset = old_asset.copy()
                # Ensure the filename points to the correct place (just use the
                # absolute path for simplicity for now)

                # Handle the case where the transform is not identity
                warp_srcimg_from_aux = kwimage.Affine.coerce(old_asset['warp_aux_to_img'])
                warp_dstimg_from_aux = warp_dstimg_from_srcimg @ warp_srcimg_from_aux
                new_asset['warp_aux_to_img'] = warp_dstimg_from_aux.concise()
                assets_to_transfer.append((dst_coco_img, old_asset, new_asset))

    print(f'Found {len(assets_to_transfer)} assets to tranfer')

    # Update the destination kwcoco file and prepare any copy jobs that need to
    # be done.
    src_bundle_dpath = src.bundle_dpath
    dst_bundle_dpath = dst.bundle_dpath
    copy_assets = config.copy_assets
    copy_tasks = []
    for dst_coco_img, src_asset, new_asset in assets_to_transfer:
        new_fname, copy_task = _make_new_asset_fname(
            src_asset, src_bundle_dpath, dst_bundle_dpath, copy_assets)
        if copy_task:
            copy_tasks.append(copy_task)

        new_asset['file_name'] = new_fname
        new_asset['image_id'] = dst_coco_img['id']
        dst_coco_img.add_asset(**new_asset)

    if not copy_assets:
        assert len(copy_tasks) == 0
        print('Copy assets is False. Only transfering reference to assets.')
    else:
        print(f'Found {len(copy_tasks)} assets to copy')

    rich.print(f'Dest Bundle: [link={dst.bundle_dpath}]{dst.bundle_dpath}[/link]')
    if copy_tasks:
        from kwutil import copy_manager
        copyman = copy_manager.CopyManager(workers=config.io_workers)
        for task in copy_tasks:
            copyman.submit(src=task['src'], dst=task['dst'],
                           overwrite=True)
        copyman.run(desc='Copy Assets')

    obj = proc_context.stop()
    dst.dataset['info'].append(obj)

    dst._ensure_json_serializable()
    if not do_return:
        print('Writing new coco file')
        dst.dump()
        print(f'Wrote transfered features to: {dst.fpath}')
    else:
        return dst


def _update_coco_fpath(self, new_fpath):
    # New method for more robustly updating the file path and bundle
    # directory, still a WIP
    if new_fpath is None:
        # Bundle directory is clobbered, so we should make everything
        # absolute
        self.reroot(absolute=True)
    else:
        old_fpath = self.fpath
        if old_fpath is not None:
            old_fpath_ = ub.Path(old_fpath)
            new_fpath_ = ub.Path(new_fpath)

            same_bundle = (
                (old_fpath_.parent == new_fpath_.parent) or
                (old_fpath_.resolve() == new_fpath_.resolve())
            )
            if not same_bundle:
                # The bundle directory has changed, so we need to reroot
                new_root = new_fpath_.parent
                self.reroot(new_root)

        self._fpath = new_fpath
        self._infer_dirs()


def _make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, copy_assets):
    """
    Find a new path for the transfered asset relative to the destination bundle.

    Args:
        src_asset (Dict): The asset dictionary we are transfering from src to dst
        src_bundle_dpath (str | PathLike): source bundle
        dst_bundle_dpath (str | PathLike): dest bundle
        copy_assets (bool): if True, build a new fname to copy to if possible

    Returns:
        Tuple[str, Dict | None]:
            the new fname and a dictionary containing the copy task
            if we need to perform one.

    TODO:
        port to kwcoco proper and move most of these tests to a unit test

    Example:
        >>> from geowatch.tasks.cold.transfer_features import _make_new_asset_fname

        >>> # Case: asset is absolute and inside src bundle
        >>> src_asset = {'file_name': '/my/src/rel/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('/my/src/rel/img.tif', None)
        ('rel/img.tif', {'src': '/my/src/rel/img.tif', 'dst': '/my/dst/rel/img.tif'})

        >>> # Case: asset is absolute and inside dst bundle
        >>> src_asset = {'file_name': '/my/dst/rel/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('rel/img.tif', None)
        ('rel/img.tif', None)

        >>> # Case: asset is absolute and inside neither bundle
        >>> src_asset = {'file_name': '/abs/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('/abs/img.tif', None)
        ('/abs/img.tif', None)

        >>> # Case: asset is relative and inside src bundle
        >>> src_asset = {'file_name': 'rel/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('/my/src/rel/img.tif', None)
        ('rel/img.tif', {'src': '/my/src/rel/img.tif', 'dst': '/my/dst/rel/img.tif'})

        >>> # Case: asset is relative and inside src bundle but with ..
        >>> src_asset = {'file_name': '../src/rel/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('/my/src/../src/rel/img.tif', None)
        ('rel/img.tif', {'src': '/my/src/../src/rel/img.tif', 'dst': '/my/dst/rel/img.tif'})

        >>> # Case: asset is relative and inside dst bundle
        >>> src_asset = {'file_name': '../dst/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('img.tif', None)
        ('img.tif', None)

        >>> # Case: asset is relative and inside neither bundle
        >>> src_asset = {'file_name': '../neither/img.tif'}
        >>> src_bundle_dpath = '/my/src'
        >>> dst_bundle_dpath = '/my/dst'
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, False))
        >>> print(_make_new_asset_fname(src_asset, src_bundle_dpath, dst_bundle_dpath, True))
        ('/my/src/../neither/img.tif', None)
        ('/my/neither/img.tif', None)
    """
    import warnings
    old_asset_fpath = join(src_bundle_dpath, src_asset['file_name'])
    dst_rel_old_asset_fpath = os.path.relpath(old_asset_fpath, dst_bundle_dpath)
    old_points_outside_dst_bundle = ('..' in ub.Path(dst_rel_old_asset_fpath).parts)

    copy_task = None

    if copy_assets:
        src_rel_old_asset_fpath = os.path.relpath(old_asset_fpath, src_bundle_dpath)
        old_points_outside_src_bundle = ('..' in ub.Path(src_rel_old_asset_fpath).parts)
        if not old_points_outside_dst_bundle:
            warnings.warn('An asset to copy already points inside of the dst bundle, not copying')
            new_fname = dst_rel_old_asset_fpath
        elif old_points_outside_src_bundle:
            # Dont want to deal with the corner case of figuring out a
            # good internal-to-dst-bundle location for an asset that
            # doesnt have a defined internal-to-src-bundle location.
            warnings.warn('An asset to copy points outside of the src bundle, not copying')
            new_fname = os.path.normpath(old_asset_fpath)
        else:
            # This case is safer to copy
            new_fname = src_rel_old_asset_fpath
            copy_task = {
                'src': old_asset_fpath,
                'dst': join(dst_bundle_dpath, new_fname),
            }
    else:
        if old_points_outside_dst_bundle:
            # The asset is outside the dst bundle, so we need to point at
            # the absolute path to it.
            new_fname = old_asset_fpath
        else:
            # Not copying, but safe to use relative paths because the asset
            # is already inside the destination bundle.
            new_fname = dst_rel_old_asset_fpath

    return new_fname, copy_task


def _test_cases():
    """
    Embedding test cases as doctests for now

    Example:
        >>> # Test case: transfer from temporal-lores to temporal-hires
        >>> from geowatch.tasks.cold.transfer_features import transfer_features_main
        >>> from geowatch.tasks.cold.transfer_features import *
        >>> import geowatch
        >>> dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, heatmap=True, dates=True)
        >>> dset2 = dset1.copy()
        >>> # Remove half of the images from dset1
        >>> dset1.remove_images(list(dset1.images())[::2])
        >>> # Remove saliency assets from dset2
        >>> for img in dset2.images().coco_images:
        >>>     asset = img.find_asset_obj('salient')
        >>>     img['auxiliary'].remove(asset)
        >>> dset2_orig = dset2.copy()
        >>> # Transfer the saliency from dset1 onto dset2
        >>> kwargs = TransferCocoConfig(**{
        >>>     'src_kwcoco': dset1,
        >>>     'dst_kwcoco': dset2,
        >>>     'new_coco_fpath': 'return',
        >>>     'channels_to_transfer': 'salient',
        >>>     'copy_assets': False,
        >>> })
        >>> cmdline = False
        >>> new_dset = transfer_features_main(cmdline, **kwargs)
        >>> assert new_dset is dset2, 'modifies combine_fpath inplace'
        >>> from geowatch.utils import kwcoco_extensions
        >>> stats1 = kwcoco_extensions.coco_channel_stats(dset1)
        >>> stats2 = kwcoco_extensions.coco_channel_stats(dset2)
        >>> stats2_orig = kwcoco_extensions.coco_channel_stats(dset2_orig)
        >>> assert stats2['single_chan_hist']['salient'] == stats1['single_chan_hist']['salient'], 'all assets should transfer'
        >>> assert stats2['single_chan_hist']['salient'] == dset2.n_images // 2, 'only some dst images get saliency'
        >>> assert stats2_orig['single_chan_hist']['salient'] == 0

    Example:
        >>> # Test case: transfer from temporal-lores to temporal-hires multiple transfer
        >>> from geowatch.tasks.cold.transfer_features import transfer_features_main
        >>> from geowatch.tasks.cold.transfer_features import *
        >>> import geowatch
        >>> dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, heatmap=True, dates=True)
        >>> dset2 = dset1.copy()
        >>> # Remove half of the images from dset1
        >>> dset1.remove_images(list(dset1.images())[::2])
        >>> # Remove saliency assets from dset2
        >>> for img in dset2.images().coco_images:
        >>>     asset = img.find_asset_obj('salient')
        >>>     img['auxiliary'].remove(asset)
        >>> dset2_orig = dset2.copy()
        >>> # Transfer the saliency from dset1 onto dset2
        >>> kwargs = TransferCocoConfig(**{
        >>>     'src_kwcoco': dset1,
        >>>     'dst_kwcoco': dset2,
        >>>     'new_coco_fpath': 'return',
        >>>     'respect_sensors': False,
        >>>     'max_propogate': None,
        >>>     'channels_to_transfer': 'salient',
        >>>     'copy_assets': False,
        >>> })
        >>> cmdline = False
        >>> new_dset = transfer_features_main(cmdline, **kwargs)
        >>> assert new_dset is dset2, 'modifies combine_fpath inplace'
        >>> from geowatch.utils import kwcoco_extensions
        >>> stats1 = kwcoco_extensions.coco_channel_stats(dset1)
        >>> stats2 = kwcoco_extensions.coco_channel_stats(dset2)
        >>> stats2_orig = kwcoco_extensions.coco_channel_stats(dset2_orig)
        >>> assert stats2['single_chan_hist']['salient'] > stats1['single_chan_hist']['salient'], 'some assets should transfer multiple times'
        >>> assert stats2_orig['single_chan_hist']['salient'] == 0

    Example:
        >>> # Test case: transfer from temporal-hires to temporal-lores
        >>> from geowatch.tasks.cold.transfer_features import transfer_features_main
        >>> from geowatch.tasks.cold.transfer_features import *
        >>> import geowatch
        >>> dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, heatmap=True, dates=True)
        >>> dset2 = dset1.copy()
        >>> # Remove half of the images from dset2
        >>> dset2.remove_images(list(dset1.images())[::2])
        >>> # Remove saliency assets from dset2
        >>> for img in dset2.images().coco_images:
        >>>     asset = img.find_asset_obj('salient')
        >>>     img['auxiliary'].remove(asset)
        >>> dset2_orig = dset2.copy()
        >>> # Transfer the saliency from dset1 onto dset2
        >>> kwargs = TransferCocoConfig(**{
        >>>     'src_kwcoco': dset1,
        >>>     'dst_kwcoco': dset2,
        >>>     'new_coco_fpath': 'return',
        >>>     'channels_to_transfer': 'salient',
        >>>     'copy_assets': False,
        >>> })
        >>> cmdline = False
        >>> new_dset = transfer_features_main(cmdline, **kwargs)
        >>> assert new_dset is dset2, 'modifies combine_fpath inplace'
        >>> from geowatch.utils import kwcoco_extensions
        >>> stats1 = kwcoco_extensions.coco_channel_stats(dset1)
        >>> stats2 = kwcoco_extensions.coco_channel_stats(dset2)
        >>> stats2_orig = kwcoco_extensions.coco_channel_stats(dset2_orig)
        >>> assert stats2['single_chan_hist']['salient'] < stats1['single_chan_hist']['salient'], 'more saliency in dset1 than dset2'
        >>> assert stats2['single_chan_hist']['salient'] == dset2.n_images, 'all dst images get saliency'
        >>> assert stats2_orig['single_chan_hist']['salient'] == 0
    """


if __name__ == '__main__':
    transfer_features_main()
