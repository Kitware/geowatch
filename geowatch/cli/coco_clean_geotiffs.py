#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CleanGeotiffConfig(scfg.DataConfig):
    r"""
    Clean geotiff files inplace by masking bad pixels with NODATA.

    Replaces large contiguous regions of specific same-valued pixels as NODATA.

    Note:
        This is a destructive operation and overwrites the geotiff image data
        inplace. Make a copy of your dataset if there is any chance you need to
        go back. The underlying kwcoco file is not modified.

    Usage:
        # It is a good idea to do a dry run first to check for issues
        # This can be done at a smaller scale for speed.
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch clean_geotiffs \
            --src "$DVC_DATA_DPATH/Drop4-BAS/data.kwcoco.json" \
            --channels="red|green|blue|nir|swir16|swir22" \
            --prefilter_channels="red" \
            --min_region_size=256 \
            --nodata_value=-9999 \
            --workers="min(2,avail)" \
            --probe_scale=0.5 \
            --dry=True

        # Then execute a real run at full scale - optionally with a probe scale
        geowatch clean_geotiffs \
            --src "$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
            --channels="red|green|blue|nir|swir16|swir22" \
            --prefilter_channels="red" \
            --min_region_size=256 \
            --nodata_value=-9999 \
            --workers="min(2,avail)" \
            --probe_scale=None \
            --dry=False


    Ignore:

        geowatch clean_geotiffs \
            --src=data.kwcoco.zip --dry=True --workers=8  \
            --probe_scale=0.25 --prefilter_channels=pan --channels=pan

        geowatch clean_geotiffs \
            --dry=False --workers=2  \
            --probe_scale=0.0625 --prefilter_channels="pan" \
            --channels="pan" \
            --src=data.kwcoco.zip

        geowatch clean_geotiffs \
            --dry=False --workers=avail  \
            --probe_scale=0.0625 --prefilter_channels="red" \
            --channels="red|green|blue|nir|swir16|swir22" \
            --src=data.kwcoco.zip

        geowatch clean_geotiffs \
            --dry=False --workers=avail  \
            --probe_scale=0.25 --prefilter_channels="swir22" \
            --channels="swir16|swir22" \
            --src=imgonly-BR_R005.kwcoco.json

    """
    src = scfg.Value(None, position=1, help='input coco dataset')

    workers = scfg.Value(0, type=str, help='number of workers')

    channels = scfg.Value('*', help=ub.paragraph(
        '''
        The channels to apply nodata fixes to. A value of * means all channels
        except the excluded ones.
        '''))

    exclude_channels = scfg.Value('quality|cloudmask', help=ub.paragraph(
        '''
        Channels to never apply fixes to.
        '''))

    prefilter_channels = scfg.Value('red', help=ub.paragraph(
        '''
        The channels to use when checking for bad nodata values.
        If unspecified, uses channels.
        '''))

    possible_nodata_values = scfg.Value([0], help=ub.paragraph(
        '''
        List of integer values that might represent nodata, but may also failed
        to be labeled as nodata.
        '''))

    min_region_size = scfg.Value(256, help=ub.paragraph(
        '''
        Minimum size of a connected region to be considered as a nodata
        candidate.
        '''))

    scale = scfg.Value(None, help=ub.paragraph(
        '''
        Scale at which to perform the check. E.g. 0.5 for half resolution.
        Speeds up checks, but should only be used in dry runs.
        TODO: we could fix the real run to work with scale checks, and that
        would speed it up and probably not generate many false negatives.
        '''))

    nodata_value = scfg.Value(-9999, help='the real nodata value to use')

    probe_scale = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, we probe the image at smaller scale for bad values and if
        any are found we do a full scale computation. Speeds up computation if
        there are not many images with bad values, but may result in false
        negatives.

        Should be given as a float less than 1.0
        '''))

    use_fix_stamps = scfg.Value(False, help=ub.paragraph(
        '''
        if True, write a file next to every file that was fixed so we dont need
        to check it again.
        '''))

    export_bad_fpath = scfg.Value(None, help='if True, export paths to bad files to this newline separated file, also works in dry mode')

    dry = scfg.Value(False, help='if True, only do a dry run. Report issues but do not fix them')


__config__ = CleanGeotiffConfig


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        xdoctest -m geowatch.cli.coco_clean_geotiffs main

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTESTS)
        >>> # Generate a dataset that has bad nodata values
        >>> from geowatch.cli.coco_clean_geotiffs import *  # NOQA
        >>> import kwimage
        >>> import geowatch
        >>> import kwarray
        >>> import numpy as np
        >>> # Create a copy of the test dataset to clean inplace
        >>> orig_dset = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, bad_nodata=True, num_videos=1, num_frames=2)
        >>> orig_dpath = ub.Path(orig_dset.bundle_dpath)
        >>> dpath = orig_dpath.augment(stemsuffix='_cleaned')
        >>> dpath.delete()
        >>> orig_dpath.copy(dpath)
        >>> dset = geowatch.coerce_kwcoco(dpath / 'data.kwcoco.json')
        >>> coco_img = dset.images().coco_images[0]
        >>> kwargs = {
        >>>     'src': dset,
        >>>     'workers': 0,
        >>>     'channels': 'B11',
        >>>     'prefilter_channels': 'B11',
        >>>     'min_region_size': 32,
        >>>     'nodata_value': 2,  # because toydata is uint16
        >>> }
        >>> cmdline = 0
        >>> # Do a dry run first
        >>> main(cmdline=cmdline, **kwargs, dry=True)
        >>> # Then a real run.
        >>> main(cmdline=cmdline, **kwargs)
        >>> coco_img1 = orig_dset.images().coco_images[0]
        >>> coco_img2 = dset.coco_image(coco_img1.img['id'])
        >>> print(ub.urepr(list(coco_img1.iter_image_filepaths())))
        >>> print(ub.urepr(list(coco_img2.iter_image_filepaths())))
        >>> imdata1 = coco_img1.imdelay('B11', nodata_method='float').finalize()
        >>> imdata2 = coco_img2.imdelay('B11', nodata_method='float').finalize()
        >>> print(np.isnan(imdata1).sum())
        >>> print(np.isnan(imdata2).sum())
        >>> canvas1 = kwarray.robust_normalize(imdata1)
        >>> canvas2 = kwarray.robust_normalize(imdata2)
        >>> canvas1 = kwimage.nodata_checkerboard(canvas1)
        >>> canvas2 = kwimage.nodata_checkerboard(canvas2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas1, pnum=(1, 2, 1), title='before')
        >>> kwplot.imshow(canvas2, pnum=(1, 2, 2), title='after')
    """
    from kwutil import util_parallel
    import kwcoco

    config = CleanGeotiffConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    print('Loading dataset')
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])

    workers = util_parallel.coerce_num_workers(config['workers'])
    print('workers = {}'.format(ub.urepr(workers, nl=1)))
    jobs = ub.JobPool(mode='process', max_workers=workers, transient=True)
    print('Created job pool')

    if config['channels'] is None or config['channels'] == '*':
        channels = None
    else:
        channels = kwcoco.ChannelSpec.coerce(config['channels'])
    if config['prefilter_channels'] is None:
        prefilter_channels = None
    else:
        prefilter_channels = kwcoco.ChannelSpec.coerce(config['prefilter_channels'])

    possible_nodata_values = set(config['possible_nodata_values'])

    if config['scale'] is not None:
        assert config['dry'], 'only scale in dry runs'

    probe_kwargs = {
        'channels': channels,
        'exclude_channels': kwcoco.ChannelSpec.coerce(config['exclude_channels']).fuse(),
        'scale': config['scale'],
        'possible_nodata_values': possible_nodata_values,
        'prefilter_channels': prefilter_channels,
        'min_region_size': config['min_region_size'],
        'probe_scale': config['probe_scale'],
        'use_fix_stamps': config['use_fix_stamps'],
        'nodata_value': config['nodata_value'],
    }

    print('Grabbing coco images')
    coco_imgs = coco_dset.images().coco_images
    print('About to start looping')

    if config.export_bad_fpath is not None:
        export_fpath = ub.Path(config.export_bad_fpath)
        with open(export_fpath, 'w'):
            ...
        export_file = open(export_fpath, 'a')
    else:
        export_file = None

    from kwutil import util_progress
    mprog = util_progress.ProgressManager(backend='progiter')
    # mprog = util_progress.ProgressManager(backend='rich')
    with mprog, jobs:
        print('In the with statement')
        mprog.update_info('Looking for geotiff issues')

        # TODO: may need to use the blocking job queue to limit the maximum
        # number of unhandled but populated results
        for coco_img in mprog.new(coco_imgs, desc='Submit probe jobs'):
            coco_img.detach()
            jobs.submit(probe_image_issues, coco_img, **probe_kwargs)

        def collect_jobs(jobs):
            for job in mprog.new(jobs.as_completed(), total=len(jobs), desc='Collect probe jobs'):
                image_summary = job.result()
                yield image_summary

        def filter_fixable(summaries):

            num_asset_issues = 0
            num_images_issues = 0
            num_incorrect_nodata = 0
            seen_bad_values = set()

            mprog.update_info(ub.codeblock(
                f'''
                Discovered Issues
                -----------------
                Found num_images_issues={num_images_issues}
                Found num_asset_issues={num_asset_issues}

                num_incorrect_nodata={num_incorrect_nodata}
                seen_bad_values={seen_bad_values}
                '''
            ))

            for image_summary in summaries:
                if image_summary['needs_fix']:
                    num_images_issues += 1
                    for asset_summary in image_summary['chans']:
                        if asset_summary['needs_fix']:
                            asset_summary['coco_img'] = image_summary['coco_img']
                            seen_bad_values.update(asset_summary['bad_values'])
                            if not asset_summary.get('has_incorrect_nodata_value', False):
                                num_incorrect_nodata += 1
                            num_asset_issues += 1
                            mprog.update_info(ub.codeblock(
                                f'''
                                Discovered Issues
                                -----------------
                                Found num_images_issues={num_images_issues}
                                Found num_asset_issues={num_asset_issues}

                                seen_bad_values={seen_bad_values}
                                '''
                            ))
                            yield asset_summary

        EAGER = 0

        summaries = collect_jobs(jobs)
        if EAGER:
            summaries = list(summaries)
        else:
            summaries = iter(summaries)

        needs_fix = filter_fixable(summaries)
        if EAGER:
            needs_fix = list(needs_fix)
        else:
            needs_fix = iter(needs_fix)

        if not config['dry']:
            correct_nodata_value = config['nodata_value']
            for asset_summary in mprog.new(needs_fix, desc='Cleaning identified issues'):
                if export_file is not None:
                    export_file.write(asset_summary['fpath'] + '\n')
                fpath = ub.Path(asset_summary['fpath'])
                fix_geotiff_ondisk(asset_summary,
                                   correct_nodata_value=correct_nodata_value)
                if config['use_fix_stamps']:
                    # Mark that fixes have been applied to this asset
                    import json
                    fix_fpath = fpath.augment(tail='.fixes.stamp')
                    if fix_fpath.exists():
                        fixes = json.loads(fix_fpath.read_text())
                    else:
                        fixes = []
                    new_fixes = ub.udict(asset_summary) & {'band_idxs'}
                    fixes.append(new_fixes)
                    fix_fpath.write_text(json.dumps(fixes))
        else:
            for asset_summary in mprog.new(needs_fix, desc='Dry run: identifying issues'):
                if export_file is not None:
                    export_file.write(asset_summary['fpath'] + '\n')
                print(asset_summary['fpath'])
                ...

    if export_file is not None:
        export_file.close()

    # if 0:
    #     import xdev
    #     import kwplot
    #     kwplot.autompl()

    #     nonzero_chans = []
    #     for image_summary in has_nonzeros:
    #         coco_img = image_summary['coco_img']
    #         for asset_summary in image_summary['chans']:
    #             bad_values = asset_summary['bad_values']
    #             if len(set(bad_values) - {0}):
    #                 asset_summary['coco_img'] = coco_img
    #                 nonzero_chans.append(asset_summary)

    #     iter_ = xdev.InteractiveIter(nonzero_chans)
    #     for asset_summary in iter_:
    #         coco_img = asset_summary['coco_img']
    #         chan_canvas = draw_asset_summary(coco_img, asset_summary)
    #         kwplot.imshow(chan_canvas, doclf=1)
    #         xdev.InteractiveIter.draw()


def probe_image_issues(coco_img, channels=None, prefilter_channels=None, scale=None,
                       possible_nodata_values=None, min_region_size=256,
                       exclude_channels=None, probe_scale=None,
                       use_fix_stamps=False, nodata_value=-9999):
    """
    Inspect a single image, possibily with multiple assets, each with possibily
    multiple bands for fixable nodata values.

    Args:
        coco_img : the coco image to check

        channels : the channels to check

        prefilter_channels :
            the channels to check first for efficiency. If they do not exist,
            then the all channels are checked.

        possible_nodata_values (set[int]):
            the values that may be nodata if known

        scale :
            use a downscaled overview to speed up the computation via
            approximation. Returns results at this scale. DO NOT USE RIGHT NOW.

        probe_scale :
            use a downscaled overview to speed up the computation via
            approximation. If the probe identifies an issue a full scale probe
            is done.

    Ignore:
        globals().update((ub.udict(xdev.get_func_kwargs(probe_image_issues)) | probe_kwargs))

    Example:
        >>> import geowatch
        >>> from geowatch.cli.coco_clean_geotiffs import *  # NOQA
        >>> import numpy as np
        >>> dset = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, bad_nodata=True)
        >>> coco_img = dset.images().coco_images[4]
        >>> channels = 'B11|B10|X.1'
        >>> prefilter_channels = 'B11'
        >>> scale = None
        >>> possible_nodata_values = {0}
        >>> min_region_size = 128
        >>> image_summary = probe_image_issues(
        >>>     coco_img, channels=channels, prefilter_channels=prefilter_channels,
        >>>     scale=scale, possible_nodata_values=possible_nodata_values,
        >>>     min_region_size=min_region_size)
        >>> print(f'image_summary={image_summary}')
    """
    import kwcoco
    if channels is None:
        request_channels = coco_img.channels.fuse()
    else:
        request_channels = kwcoco.ChannelSpec.coerce(channels).fuse()
    if prefilter_channels is not None:
        prefilter_channels = kwcoco.ChannelSpec.coerce(prefilter_channels).fuse()

    if exclude_channels is not None:
        request_channels = request_channels.difference(exclude_channels)
        if prefilter_channels is not None:
            prefilter_channels = prefilter_channels.difference(exclude_channels)

    image_summary = {}
    asset_summaries = []

    prefilter_assets = []
    requested_assets = []
    for obj in coco_img.iter_asset_objs():
        chans = kwcoco.ChannelSpec.coerce(obj['channels']).fuse()
        request_common = (chans & request_channels).fuse()
        if request_common.numel():
            if prefilter_channels is None:
                prefilter_common = None
            else:
                prefilter_common = (chans & prefilter_channels).fuse()
            chans_ = chans.to_oset()
            band_idxs = [chans_.index(c) for c in request_common.to_list()]
            if prefilter_common is not None and prefilter_common.numel():
                prefilter_assets.append({
                    'obj': obj,
                    'band_idxs': band_idxs,
                })
            else:
                requested_assets.append({
                    'obj': obj,
                    'band_idxs': band_idxs,
                })

    should_continue = True
    for item in prefilter_assets:
        obj = item['obj']
        band_idxs = item['band_idxs']

        if probe_scale is not None:
            # Check if there is anything obvious at a smaller scale first if
            # requested.
            probe_asset_summary = probe_asset(
                coco_img, obj, band_idxs=band_idxs, scale=probe_scale,
                possible_nodata_values=possible_nodata_values,
                min_region_size=min_region_size, use_fix_stamps=use_fix_stamps,
                nodata_value=nodata_value)
            if not probe_asset_summary['bad_values']:
                should_continue = False

        if should_continue:
            # Check if there is anything obvious in a specific channel first if
            # requested.
            asset_summary = probe_asset(coco_img, obj, band_idxs=band_idxs,
                                        scale=scale,
                                        possible_nodata_values=possible_nodata_values,
                                        min_region_size=min_region_size,
                                        use_fix_stamps=use_fix_stamps,
                                        nodata_value=nodata_value)
            if not asset_summary['bad_values']:
                should_continue = False
            asset_summaries.append(asset_summary)

    if len(prefilter_assets) == 0:
        should_continue = True

    if should_continue:
        for item in requested_assets:
            obj = item['obj']
            band_idxs = item['band_idxs']
            asset_summary = probe_asset(coco_img, obj, band_idxs=band_idxs,
                                        scale=scale,
                                        possible_nodata_values=possible_nodata_values,
                                        min_region_size=min_region_size,
                                        use_fix_stamps=use_fix_stamps,
                                        nodata_value=nodata_value)
            asset_summaries.append(asset_summary)

    image_summary['chans'] = asset_summaries
    image_summary['coco_img'] = coco_img
    image_summary['bad_values'] = list(ub.unique(ub.flatten([c['bad_values'] for c in asset_summaries])))
    image_summary['needs_fix'] = any([c['needs_fix'] for c in asset_summaries])
    return image_summary


def probe_asset(coco_img, obj, band_idxs=None, scale=None,
                possible_nodata_values=None, min_region_size=256,
                use_fix_stamps=False, nodata_value=-9999):
    """
    Inspect a specific single-file asset possibily with multiple bands for
    fixable nodata values.
    """
    import kwcoco
    asset_channels = kwcoco.FusedChannelSpec.coerce(obj['channels'])
    if band_idxs is None:
        band_idxs = list(range(asset_channels.numel()))

    bundle_dpath = ub.Path(coco_img.bundle_dpath)
    fpath = bundle_dpath / obj['file_name']

    asset_summary = {
        'channels': asset_channels,
        'fpath': fpath,
        'needs_fix': False,
    }

    if use_fix_stamps:
        import json
        fix_fpath = fpath.augment(tail='.fixes.stamp')
        if fix_fpath.exists():
            fixes = json.loads(fix_fpath.read_text())
            assert fixes
            return asset_summary
            # TODO: actually check the requested fix was applied, but for now
            # assume if the file exists we should skip the check.

    check_channels = asset_channels[band_idxs]
    delayed = coco_img.imdelay(
        channels=check_channels, space='asset', nodata_method='float')
    if scale is not None:
        delayed.prepare()
        delayed = delayed.scale(scale)
        delayed = delayed.optimize()

    imdata = delayed.finalize(interpolation='nearest')

    if scale is not None:
        min_region_size_ = int(min_region_size * scale)
    else:
        min_region_size_ = min_region_size

    if possible_nodata_values is not None:
        possible_nodata_values = list(possible_nodata_values)
        possible_nodata_values.append(nodata_value)

    asset_summary.update(probe_asset_imdata(
        imdata, band_idxs, min_region_size_=min_region_size_,
        possible_nodata_values=possible_nodata_values))

    asset_summary.update(_probe_correct_nodata_value(
        fpath, band_idxs, nodata_value))

    return asset_summary


def _probe_correct_nodata_value(fpath, band_idxs, nodata_value=-9999):
    asset_summary = {}
    from geowatch.utils import util_gdal
    gdal_dset = util_gdal.GdalDataset.open(fpath)
    band_infos = gdal_dset.info()['bands']
    gdal_dset = None
    asset_summary['has_incorrect_nodata_value'] = False
    for band_idx in band_idxs:
        band_info = band_infos[band_idx]
        if band_info.get('noDataValue', None) != nodata_value:
            asset_summary['has_incorrect_nodata_value'] = True
            asset_summary['needs_fix'] = True
    return asset_summary


def probe_asset_imdata(imdata, band_idxs, min_region_size_=256,
                       possible_nodata_values=None):
    import kwarray
    asset_summary = {}
    asset_summary['band_idxs'] = band_idxs
    band_summaries = []
    imdata = kwarray.atleast_nd(imdata, n=3)
    for idx in range(len(band_idxs)):
        band_imdata = imdata[..., idx]
        band_summary = probe_band_imdata(
            band_imdata, min_region_size_=min_region_size_,
            possible_nodata_values=possible_nodata_values)
        band_summaries.append(band_summary)
    asset_summary['band_summaries'] = band_summaries
    asset_summary['bad_values'] = list(ub.unique(ub.flatten([
        c['bad_values'] for c in band_summaries])))
    if len(asset_summary['bad_values']) > 0:
        asset_summary['needs_fix'] = True
    return asset_summary


def probe_band_imdata(band_imdata, min_region_size_=256,
                      possible_nodata_values=None):
    from geowatch.utils import util_kwimage
    import numpy as np

    band_imdata = np.ascontiguousarray(band_imdata)
    is_samecolor = util_kwimage.find_samecolor_regions(
        band_imdata, min_region_size=min_region_size_,
        values=possible_nodata_values, PRINT_STEPS=0)

    band_summary = {}
    if np.any(is_samecolor):
        unique_labels, unique_first_index, unique_counts = np.unique(is_samecolor, return_index=True, return_counts=True)
        flags = unique_labels > 0
        bad_labels = unique_labels[flags]
        first_index = unique_first_index[flags]
        bad_counts = unique_counts[flags]
        bad_values = band_imdata.ravel()[first_index]

        band_summary['is_samecolor'] = is_samecolor
        band_summary['bad_labels'] = bad_labels
        band_summary['bad_values'] = bad_values
        band_summary['bad_counts'] = bad_counts
    else:
        band_summary['is_samecolor'] = False
        band_summary['bad_values'] = []
    return band_summary


def fix_single_asset(fpath, dry=False):
    from delayed_image import DelayedLoad
    asset_summary = {
        'fpath': fpath,
        'needs_fix': False,
    }

    min_region_size_ = 256
    nodata_value = -9999
    possible_nodata_values = [0, nodata_value]

    delayed = DelayedLoad(fpath)
    delayed.prepare()
    band_idxs = list(range(delayed.num_channels))

    imdata = delayed.finalize(nodata_method='nan', interpolation='nearest')

    asset_summary.update(probe_asset_imdata(
        imdata, band_idxs, min_region_size_=min_region_size_,
        possible_nodata_values=possible_nodata_values))

    asset_summary.update(_probe_correct_nodata_value(
        fpath, band_idxs, nodata_value))

    print('asset_summary = {}'.format(ub.urepr(asset_summary, nl=True)))
    if not dry:
        fix_geotiff_ondisk(asset_summary, correct_nodata_value=nodata_value)


def fix_geotiff_ondisk(asset_summary, correct_nodata_value=-9999):
    """
    Updates the nodata value based on a mask inplace on disk.
    Attempts to preserve all other metadata, but this is not guarenteed or
    always possible.

    Args:
        asset_summary (Dict): an item from :func:`probe_asset`.

        correct_nodata_value (int): the nodata value to use in the
            modified geotiff.

    Assumptions:
        * The input image uses AVERAGE overview resampling
        * The input image is a tiled geotiff (ideally a COG)

    TODO:
        - [ ] Can restructure this as a more general context manager.

    Example:
        >>> from geowatch.cli.coco_clean_geotiffs import *  # NOQA
        >>> from geowatch.demo.metrics_demo.demo_rendering import write_demo_geotiff
        >>> import kwimage
        >>> import numpy as np
        >>> dpath = ub.Path.appdir('geowatch/tests/clean_geotiff').ensuredir()
        >>> fpath1 = dpath / 'test_geotiff.tif'
        >>> fpath2 = fpath1.augment(stemsuffix='_fixed')
        >>> fpath1.delete()
        >>> fpath2.delete()
        >>> imdata = kwimage.grab_test_image('amazon', dsize=(512, 512))
        >>> poly = kwimage.Polygon.random().scale(imdata.shape[0:2][::-1])
        >>> imdata = poly.draw_on(imdata, color='black')
        >>> imdata = imdata.astype(np.int16)
        >>> #imdata = poly.fill(imdata, value=(0, 0, 0), pixels_are='areas')
        >>> imdata = poly.fill(imdata, value=0, pixels_are='areas')
        >>> imdata[:256, :256, 0] = 0
        >>> write_demo_geotiff(img_fpath=fpath1, imdata=imdata)
        >>> fpath1.copy(fpath2)
        >>> asset_summary = probe_asset_imdata(imdata, band_idxs=[0, 2], possible_nodata_values={0})
        >>> asset_summary['fpath'] = fpath2
        >>> assert fpath1.stat().st_size == fpath2.stat().st_size
        >>> fix_geotiff_ondisk(asset_summary)
        >>> assert fpath1.stat().st_size != fpath2.stat().st_size
        >>> imdata1 = kwimage.imread(fpath1, nodata_method='ma')
        >>> imdata2 = kwimage.imread(fpath2, nodata_method='ma')
        >>> canvas1 = kwimage.normalize_intensity(imdata1)
        >>> canvas2 = kwimage.normalize_intensity(imdata2)
        >>> canvas1 = kwimage.nodata_checkerboard(canvas1)
        >>> canvas2 = kwimage.nodata_checkerboard(canvas2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas1.data, pnum=(2, 2, 1), title='norm imdata1 vals')
        >>> kwplot.imshow(canvas2.data, pnum=(2, 2, 2), title='norm imdata2 vals')
        >>> kwplot.imshow(imdata1.mask.any(axis=2), pnum=(2, 2, 3), title='imdata1.mask')
        >>> kwplot.imshow(imdata2.mask.any(axis=2), pnum=(2, 2, 4), title='imdata2.mask')
        >>> #kwplot.imshow((asset_summary['is_samecolor'] > 0), pnum=(3, 2, 5), title='is samecolor mask')

    Example:
        >>> from geowatch.cli.coco_clean_geotiffs import *  # NOQA
        >>> from geowatch.demo.metrics_demo.demo_rendering import write_demo_geotiff
        >>> import kwimage
        >>> import numpy as np
        >>> dpath = ub.Path.appdir('geowatch/tests/clean_geotiff').ensuredir()
        >>> fpath1 = dpath / 'test_geotiff.tif'
        >>> fpath2 = fpath1.augment(stemsuffix='_fixed')
        >>> fpath1.delete()
        >>> fpath2.delete()
        >>> imdata = kwimage.grab_test_image('amazon', dsize=(512, 512))[..., 0]
        >>> poly = kwimage.Polygon.random().scale(imdata.shape[0:2][::-1])
        >>> imdata = imdata.astype(np.int16)
        >>> imdata = poly.fill(imdata, value=0, pixels_are='areas')
        >>> imdata[:256, :256] = 0
        >>> write_demo_geotiff(img_fpath=fpath1, imdata=imdata)
        >>> fpath1.copy(fpath2)
        >>> asset_summary = probe_asset_imdata(imdata, band_idxs=[0], possible_nodata_values={0})
        >>> asset_summary['fpath'] = fpath2
        >>> assert fpath1.stat().st_size == fpath2.stat().st_size
        >>> fix_geotiff_ondisk(asset_summary)
        >>> assert fpath1.stat().st_size != fpath2.stat().st_size
        >>> imdata1 = kwimage.imread(fpath1, nodata_method='ma')
        >>> imdata2 = kwimage.imread(fpath2, nodata_method='ma')
        >>> canvas1 = kwimage.normalize_intensity(imdata1)
        >>> canvas2 = kwimage.normalize_intensity(imdata2)
        >>> canvas1 = kwimage.nodata_checkerboard(canvas1)
        >>> canvas2 = kwimage.nodata_checkerboard(canvas2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas1.data, pnum=(2, 2, 1), title='norm imdata1 vals')
        >>> kwplot.imshow(canvas2.data, pnum=(2, 2, 2), title='norm imdata2 vals')
        >>> kwplot.imshow(imdata1.mask, pnum=(2, 2, 3), title='imdata1.mask')
        >>> kwplot.imshow(imdata2.mask, pnum=(2, 2, 4), title='imdata2.mask')
        >>> #kwplot.imshow((asset_summary['is_samecolor'] > 0), pnum=(3, 2, 5), title='is samecolor mask')
    """
    import os
    from geowatch.utils import util_gdal
    from osgeo import gdal
    import numpy as np
    import tempfile
    fpath = ub.Path(asset_summary['fpath'])
    band_idxs = asset_summary['band_idxs']

    # We will write a modified version to a temporay file and then overwrite
    # the destination file to avoid race conditions.
    tmp_fpath = ub.Path(tempfile.mktemp(
        dir=fpath.parent, prefix='.tmp.' + fpath.stem, suffix='.tiff'))
    assert tmp_fpath.parent.exists()

    # Assume average overview resampling, there does not seem to be standard
    # way that a geotiff encodes what algorithm was used.
    overview_resample = 'AVERAGE'

    ### Read in source data and introspect it
    src_dset = util_gdal.GdalDataset.open(os.fspath(fpath))
    orig_driver_name = src_dset.GetDriver().ShortName
    assert orig_driver_name == 'GTiff'

    band_idx_to_orig_props = {}
    for band_idx in band_idxs:
        band = src_dset.GetRasterBand(band_idx + 1)
        # Have to read this here because it is clobbered in the copy
        band_idx_to_orig_props[band_idx] = {
            'blocksize': band.GetBlockSize(),
            'num_overviews': band.GetOverviewCount(),
        }

    ### Copy the source data to memory and modify it
    driver1 = gdal.GetDriverByName(str('MEM'))
    copy1 = driver1.CreateCopy(str(''), src_dset)
    src_dset.FlushCache()
    src_dset = None

    ### Pixel Modification ###
    # Modify the pixel contents
    for band_idx, band_summary in zip(band_idxs, asset_summary['band_summaries']):
        band = copy1.GetRasterBand(band_idx + 1)
        band_data = band.ReadAsArray()
        mask = band_summary['is_samecolor'] > 0

        curr_nodat_value = band.GetNoDataValue()

        if curr_nodat_value is not None:
            if curr_nodat_value != correct_nodata_value:
                is_curr_nodata = (band_data == curr_nodat_value)
                mask = (mask | is_curr_nodata)

        band_data[mask] = correct_nodata_value

        band.WriteArray(band_data)
        if curr_nodat_value != correct_nodata_value:
            band.SetNoDataValue(correct_nodata_value)

    if band_idx_to_orig_props:
        assert ub.allsame(band_idx_to_orig_props.values()), (
            'We expect input bands to have the same blocksize')
        bandprop = ub.peek(band_idx_to_orig_props.values())
        blocksize = bandprop['blocksize']
        num_overviews = bandprop['num_overviews']
    else:
        raise AssertionError('should have this info')

    ### Rebuild overviews
    overviewlist = (2 ** np.arange(1, num_overviews + 1)).tolist()
    copy1.BuildOverviews(overview_resample, overviewlist)

    ### Flush the in-memory dataset to an on-disk GeoTiff
    xsize, ysize = blocksize
    _options = [
        'BIGTIFF=YES',
        'TILED=YES',
        'BLOCKXSIZE={}'.format(xsize),
        'BLOCKYSIZE={}'.format(ysize),
    ]
    _options += ['COMPRESS={}'.format('DEFLATE')]
    _options.append('COPY_SRC_OVERVIEWS=YES')
    driver1 = None
    # driver2 = gdal.GetDriverByName(str('GTiff'))
    driver2 = gdal.GetDriverByName(str(orig_driver_name))
    copy2 = driver2.CreateCopy(os.fspath(tmp_fpath), copy1, options=_options)

    if copy2 is None:
        last_gdal_error = gdal.GetLastErrorMsg()
        if 'No such file or directory' in last_gdal_error:
            ex_cls = IOError
        else:
            ex_cls = Exception
        raise ex_cls(
            'Unable to create gtiff driver for fpath={}, options={}, last_gdal_error={}'.format(
                fpath, _options, last_gdal_error))
    copy2.FlushCache()
    copy1 = None
    copy2 = None  # NOQA
    driver2 = None

    assert ub.Path(tmp_fpath).exists()
    os.rename(tmp_fpath, fpath)


def draw_asset_summary(coco_img, asset_summary):
    from geowatch.utils import util_kwimage
    import kwimage
    import numpy as np
    is_samecolor = asset_summary['is_samecolor']
    data = coco_img.imdelay(channels=asset_summary['channels'], space='asset',
                            nodata_method='float').finalize()

    is_samecolor = np.ascontiguousarray(is_samecolor)
    unique_labels, first_index = np.unique(is_samecolor, return_index=True)
    data_values = data.ravel()[first_index]

    # is_samecolor.ravel()[first_index]
    label_mapping = dict(zip(unique_labels, data_values))
    label_mapping[0] = 'valid data'
    labels = is_samecolor
    label_to_color = {0: 'black'}
    label_canvas = util_kwimage.colorize_label_image(
        labels, with_legend=True,
        label_mapping=label_mapping, label_to_color=label_to_color)

    poly = kwimage.Mask.coerce(is_samecolor > 0).to_multi_polygon()
    boxes = kwimage.Boxes.concatenate([p.to_boxes() for p in poly.data])

    im_canvas = kwimage.normalize_intensity(data)
    im_canvas = kwimage.fill_nans_with_checkers(im_canvas, on_value=0.3)
    im_canvas = boxes.draw_on(im_canvas)
    canvas = kwimage.stack_images([im_canvas, label_canvas], axis=1)
    title = coco_img.video['name'] + ' ' + asset_summary['channels'] + '\n' + coco_img.img['name']
    canvas = kwimage.draw_header_text(canvas, title)

    if 0:
        import kwplot
        kwplot.imshow(canvas)
    return canvas


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/coco_clean_geotiffs.py
    """
    main()
