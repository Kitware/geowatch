import scriptconfig as scfg
import ubelt as ub
import kwimage
import kwcoco
import kwarray
import numpy as np
from watch.utils import util_kwimage


class CleanGeotiffConfig(scfg.DataConfig):
    r"""
    A preprocessing step for geotiff datasets.

    Replaces large contiguous regions of specific same-valued pixels as nodata.

    Note:
        This is a destructive operation and overwrites the geotiff image data
        inplace. Make a copy of your dataset if there is any chance you need to
        go back. The underlying kwcoco file is not modified.

    Usage:
        # It is a good idea to do a dry run first to check for issues
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        COCO_FPATH="$DVC_DATA_DPATH/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/data.kwcoco.json"
        python -m watch.cli.coco_clean_geotiffs \
            --src "$COCO_FPATH" \
            --channels="red|green|blue|nir|swir16|swir22" \
            --prefilter_channels="red" \
            --min_region_size=256 \
            --nodata_value=-9999 \
            --dry=True
    """
    src = scfg.Value(None, help='input coco dataset')

    workers = scfg.Value(0, help='number of workers')

    channels = scfg.Value('red|green|blue|nir|swir16|swir22', help=ub.paragraph(
        '''
        The channels to apply nodata fixes to.
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

    nodata_value = scfg.Value(-9999, help='the real nodata value to use')

    dry = scfg.Value(False, help='if True, only do a dry run. Report issues but do not fix them')


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        xdoctest -m watch.cli.coco_clean_geotiffs main

    Example:
        >>> # Generate a dataset that has bad nodata values
        >>> from watch.cli.coco_clean_geotiffs import *  # NOQA
        >>> import watch
        >>> # Create a copy of the test dataset to clean inplace
        >>> orig_dset = watch.coerce_kwcoco('watch-msi', geodata=True, bad_nodata=True)
        >>> orig_dpath = ub.Path(orig_dset.bundle_dpath)
        >>> dpath = orig_dpath.augment(stemsuffix='_cleaned')
        >>> dpath.delete()
        >>> orig_dpath.copy(dpath)
        >>> dset = watch.coerce_kwcoco(dpath / 'data.kwcoco.json')
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
        >>> imdata1 = coco_img1.delay('B11', nodata_method='float').finalize()
        >>> imdata2 = coco_img2.delay('B11', nodata_method='float').finalize()
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

    Ignore:
        import watch
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch'))
        from watch.cli.coco_clean_geotiffs import *  # NOQA
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        coco_fpath = dvc_dpath / 'Drop4-BAS' / 'data_vali.kwcoco.json'
        coco_fpath = dvc_dpath / 'Drop4-BAS' / 'data_train.kwcoco.json'
        # coco_fpath = dvc_dpath / 'Drop4-BAS' / 'combo_train_I2.kwcoco.json'
        cmdline = 0
        kwargs = {'src': coco_fpath, 'workers': 'avail'}
    """
    from watch.utils import util_globals

    config = CleanGeotiffConfig.legacy(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.urepr(dict(config), nl=1)))

    print('Loading dataset')
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])

    workers = util_globals.coerce_num_workers(config['workers'])
    print('workers = {}'.format(ub.urepr(workers, nl=1)))
    jobs = ub.JobPool(mode='process', max_workers=workers)

    # channels = kwcoco.ChannelSpec.coerce('red')
    # channels = kwcoco.ChannelSpec.coerce('red|blue|nir')
    # channels = kwcoco.ChannelSpec.coerce('nir')
    channels = kwcoco.ChannelSpec.coerce(config['channels'])
    if config['prefilter_channels'] is None:
        prefilter_channels = kwcoco.ChannelSpec.coerce(config['prefilter_channels'])
    else:
        prefilter_channels = channels

    possible_nodata_values = set(config['possible_nodata_values'])
    probe_kwargs = {
        'channels': channels,
        'scale': None,
        'possible_nodata_values': possible_nodata_values,
        'prefilter_channels': prefilter_channels,
        'channels': channels,
        'min_region_size': config['min_region_size'],
    }

    coco_imgs = coco_dset.images().coco_images

    mprog = MultiProgress()
    with mprog:
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
            seen_bad_values = set()

            for image_summary in summaries:
                if len(image_summary['bad_values']):
                    num_images_issues += 1
                    for asset_summary in image_summary['chans']:
                        if asset_summary['bad_values']:
                            asset_summary['coco_img'] = image_summary['coco_img']
                            seen_bad_values.update(asset_summary['bad_values'])
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

        needs_fix = filter_fixable(summaries)
        if EAGER:
            needs_fix = list(needs_fix)
            total = len(needs_fix)
        else:
            total = len(coco_imgs)

        if not config['dry']:
            correct_nodata_value = config['nodata_value']
            for asset_summary in mprog.new(needs_fix, total=total, desc='Cleaning identified issues'):
                fix_geotiff_ondisk(asset_summary,
                                   correct_nodata_value=correct_nodata_value)
        else:
            _ = list(needs_fix)

    # for k, g in bad_groups.items():
    #     if all([p[2] == [0] for p in g])
    # print(len(investigate_images))
    # has_nonzeros = []
    # all_zeros = []
    # for image_summary in investigate_images:
    #     if len(set(image_summary['bad_values']) - {0}):
    #         has_nonzeros.append(image_summary)
    #     else:
    #         all_zeros.append(image_summary)
    # print(len(all_zeros))
    # print(len(has_nonzeros))
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
    #         chan_canvas = draw_channel_summary(coco_img, asset_summary)
    #         kwplot.imshow(chan_canvas, doclf=1)
    #         xdev.InteractiveIter.draw()


class RichProgIter:
    """
    Ducktypes ProgIter
    """
    def __init__(self, prog_manager, iterable, total=None, desc=None):
        self.prog_manager = prog_manager
        self.iterable = iterable
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                ...
        self.task_id = self.prog_manager.add_task(desc, total=total)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.prog_manager.update(self.task_id, advance=1)
        task = self.prog_manager._tasks[self.task_id]
        if task.total is None:
            self.prog_manager.update(self.task_id, total=task.completed)


class MultiProgress:
    """
    Manage multiple progress bars, either with rich or ProgIter.

    Example:
        >>> from watch.cli.coco_clean_geotiffs import *  # NOQA
        >>> multi_prog = MultiProgress(use_rich=0)
        >>> with multi_prog:
        >>>     for i in multi_prog.new(range(100), desc='outer loop'):
        >>>         for i in multi_prog.new(range(100), desc='inner loop'):
        >>>             pass
        >>> #
        >>> self = multi_prog = MultiProgress(use_rich=1)
        >>> with multi_prog:
        >>>     for i in multi_prog.new(range(10), desc='outer loop'):
        >>>         for i in multi_prog.new(iter(range(1000)), desc='inner loop'):
        >>>             pass
    """

    def __init__(self, use_rich=1):
        self.use_rich = use_rich
        self.sub_progs = []
        if self.use_rich:
            self.setup_rich()

    def new(self, iterable, total=None, desc=None):
        self.prog_iters = []
        if self.use_rich:
            prog = RichProgIter(
                prog_manager=self.prog_manager, iterable=iterable, total=total,
                desc=desc)
        else:
            prog = ub.ProgIter(iterable, total=total, desc=desc)
        self.prog_iters.append(prog)
        return prog

    def setup_rich(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.live import Live
        import rich
        import rich.progress
        from rich.progress import (BarColumn, Progress, TextColumn)
        self.prog_manager = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            rich.progress.MofNCompleteColumn(),
            # "[progress.percentage]{task.percentage:>3.0f}%",
            rich.progress.TimeRemainingColumn(),
            rich.progress.TimeElapsedColumn(),
        )
        self.info_panel = Panel('')
        self.progress_group = Group(
            self.info_panel,
            self.prog_manager,
        )
        self.live_context = Live(self.progress_group)

    def update_info(self, text):
        if self.use_rich:
            self.info_panel.renderable = text

    def __enter__(self):
        if self.use_rich:
            return self.live_context.__enter__()

    def __exit__(self, *args, **kw):
        if self.use_rich:
            return self.live_context.__exit__(*args, **kw)


def probe_image_issues(coco_img, channels=None, prefilter_channels=None, scale=None,
                       possible_nodata_values=None, min_region_size=256):
    """
    Inspect a single image, possibily with multiple assets, each with possibily
    multiple bands for fixable nodata values.

    Args:
        coco_img : the coco image to check
        channels : the channels to check
        prefilter_channels : the channels to check first for efficiency
        possible_nodata_values (set[int]):
            the values that may be nodata if known
        scale :
            use a downscaled overview to speed up the computation via
            approximation.

    Ignore:
        globals().update((ub.udict(xdev.get_func_kwargs(probe_image_issues)) | probe_kwargs))

    Example:
        >>> import watch
        >>> from watch.cli.coco_clean_geotiffs import *  # NOQA
        >>> dset = watch.coerce_kwcoco('watch-msi', geodata=True, bad_nodata=True)
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
    if channels is None:
        request_channels = coco_img.channels
    else:
        request_channels = kwcoco.ChannelSpec.coerce(channels).fuse()
    if prefilter_channels is not None:
        prefilter_channels = kwcoco.ChannelSpec.coerce(prefilter_channels).fuse()

    image_summary = {}
    asset_summaries = []

    prefilter_assets = []
    requested_assets = []
    for obj in coco_img.iter_asset_objs():
        chans = kwcoco.ChannelSpec.coerce(obj['channels']).fuse()
        request_common = (chans & request_channels).fuse()
        if request_common.numel():
            prefilter_common = (chans & prefilter_channels).fuse()
            chans_ = chans.to_oset()
            band_idxs = [chans_.index(c) for c in request_common.to_list()]
            if prefilter_common.numel():
                prefilter_assets.append({
                    'obj': obj,
                    'band_idxs': band_idxs,
                })
            else:
                requested_assets.append({
                    'obj': obj,
                    'band_idxs': band_idxs,
                })

    should_continue = len(prefilter_assets) == 0
    for item in prefilter_assets:
        obj = item['obj']
        band_idxs = item['band_idxs']
        asset_summary = probe_asset(coco_img, obj, band_idxs=band_idxs,
                                    scale=scale,
                                    possible_nodata_values=possible_nodata_values,
                                    min_region_size=min_region_size)
        if asset_summary['bad_values']:
            should_continue = True
        asset_summaries.append(asset_summary)

    if should_continue:
        for item in requested_assets:
            obj = item['obj']
            band_idxs = item['band_idxs']
            asset_summary = probe_asset(coco_img, obj, band_idxs=band_idxs,
                                        scale=scale,
                                        possible_nodata_values=possible_nodata_values,
                                        min_region_size=min_region_size)
            asset_summaries.append(asset_summary)

    image_summary['chans'] = asset_summaries
    image_summary['coco_img'] = coco_img
    image_summary['bad_values'] = list(ub.unique(ub.flatten([c['bad_values'] for c in asset_summaries])))
    return image_summary


def probe_asset(coco_img, obj, band_idxs=None, scale=None,
                possible_nodata_values=None, min_region_size=256):
    """
    Inspect a specific single-file asset possibily with multiple bands for
    fixable nodata values.
    """
    asset_channels = kwcoco.FusedChannelSpec.coerce(obj['channels'])
    if band_idxs is None:
        band_idxs = list(range(asset_channels.numel()))

    check_channels = asset_channels[band_idxs]

    asset_summary = {
        'channels': asset_channels,
    }
    delayed = coco_img.delay(
        channels=check_channels, space='asset', nodata_method='float')
    if scale is not None:
        delayed.prepare()
        delayed = delayed.scale(scale)
        delayed = delayed.optimize()

    imdata = delayed.finalize(interpolation='nearest')

    bundle_dpath = ub.Path(coco_img.bundle_dpath)
    fpath = bundle_dpath / obj['file_name']

    if scale is not None:
        min_region_size_ = int(min_region_size * scale)
    else:
        min_region_size_ = min_region_size

    asset_summary = probe_asset_imdata(
        imdata, band_idxs, min_region_size_=min_region_size_,
        possible_nodata_values=possible_nodata_values)

    asset_summary['fpath'] = fpath
    return asset_summary


def probe_asset_imdata(imdata, band_idxs, min_region_size_=256,
                       possible_nodata_values=None):
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
    return asset_summary


def probe_band_imdata(band_imdata, min_region_size_=256,
                      possible_nodata_values=None):

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
        band_summary['bad_values'] = []
    return band_summary


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
        >>> from watch.cli.coco_clean_geotiffs import *  # NOQA
        >>> from watch.demo.metrics_demo.demo_rendering import write_demo_geotiff
        >>> dpath = ub.Path.appdir('watch/tests/clean_geotiff').ensuredir()
        >>> fpath1 = dpath / 'test_geotiff.tif'
        >>> fpath2 = fpath1.augment(stemsuffix='_fixed')
        >>> fpath1.delete()
        >>> fpath2.delete()
        >>> imdata = kwimage.grab_test_image('amazon', dsize=(512, 512))
        >>> poly = kwimage.Polygon.random().scale(imdata.shape[0:2][::-1])
        >>> imdata = poly.draw_on(imdata, color='black')
        >>> imdata = imdata.astype(np.int16)
        >>> imdata = poly.fill(imdata, value=(0, 0, 0), pixels_are='areas')
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
        >>> from watch.cli.coco_clean_geotiffs import *  # NOQA
        >>> from watch.demo.metrics_demo.demo_rendering import write_demo_geotiff
        >>> dpath = ub.Path.appdir('watch/tests/clean_geotiff').ensuredir()
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
    from watch.utils import util_gdal
    from osgeo import gdal
    import tempfile
    fpath = ub.Path(asset_summary['fpath'])

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

    ### Copy the source data to memory and modify it
    driver1 = gdal.GetDriverByName(str('MEM'))
    copy1 = driver1.CreateCopy(str(''), src_dset)
    src_dset.FlushCache()
    src_dset = None

    bandmeta_candidates = []
    ### Pixel Modification ###
    # Modify the pixel contents
    band_idxs = asset_summary['band_idxs']
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
        bandmeta_candidates.append((
            band.GetBlockSize(),
            band.GetOverviewCount(),
        ))

    if bandmeta_candidates:
        assert ub.allsame(bandmeta_candidates), (
            'We expect input bands to have the same blocksize')
        blocksize = bandmeta_candidates[0][0]
        num_overviews = bandmeta_candidates[0][1]
    else:
        src_band = src_dset.GetRasterBand(1)
        blocksize = src_band.GetBlockSize()
        num_overviews = src_band.GetOverviewCount()

    # FIXME: the returned band size from the gdal calls doesnt seem correct.
    # For multi-band images I get 256x1 blocksizes when gdalinfo reports
    # 256,256. For now just always use a 256x256 blocksize and at least two
    # overviews.
    HACK_FIX_BAND_METADATA = True
    if HACK_FIX_BAND_METADATA:
        blocksize = (256, 256)
        num_overviews = max(num_overviews, 2)

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


def draw_channel_summary(coco_img, asset_summary):
    is_samecolor = asset_summary['is_samecolor']
    data = coco_img.delay(channels=asset_summary['channels'], space='asset',
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
