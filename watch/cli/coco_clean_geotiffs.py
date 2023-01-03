import scriptconfig as scfg
import ubelt as ub
import kwimage
import kwcoco
import numpy as np
from watch.utils import util_kwimage


class CleanGeotiffConfig(scfg.DataConfig):
    """
    A preprocessing step for a geotiff dataset that corrects several issues.

    Replaces large contiguous regions of specific same-valued pixels as nodata.
    """
    src = scfg.Value(None, help='input coco dataset')
    nodata_value = scfg.Value(-9999, help='the real nodata value to use')


def main(cmdline=1, **kwargs):
    """
    Ignore:
        import watch
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch'))
        from watch.cli.coco_clean_geotiffs import *  # NOQA
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        coco_fpath = dvc_dpath / 'Drop4-BAS' / 'combo_vali_I2.kwcoco.json'
        # coco_fpath = dvc_dpath / 'Drop4-BAS' / 'combo_train_I2.kwcoco.json'
        kwargs = {'src': coco_fpath}
    """
    from watch.utils import util_globals

    config = CleanGeotiffConfig.legacy(cmdline=1, data=kwargs)
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])

    workers = util_globals.coerce_num_workers('avail')
    # workers = 0
    jobs = ub.JobPool(mode='process', max_workers=workers)

    # channels = kwcoco.ChannelSpec.coerce('red')
    prefilter_channels = kwcoco.ChannelSpec.coerce('red')
    channels = kwcoco.ChannelSpec.coerce('red|green|blue|nir|swir16|swir22')
    # channels = kwcoco.ChannelSpec.coerce('red|blue|nir')
    # channels = kwcoco.ChannelSpec.coerce('nir')

    possible_nodata_values = set([0])
    probe_kwargs = {
        'channels': channels,
        'scale': None,
        'possible_nodata_values': possible_nodata_values,
        'prefilter_channels': prefilter_channels,
        'channels': channels,
    }

    coco_imgs = coco_dset.images().coco_images
    for coco_img in ub.ProgIter(coco_imgs):
        coco_img.detach()
        job = jobs.submit(probe_image_issues, coco_img, **probe_kwargs)
        job.result()

    summaries = []
    for job in jobs.as_completed(desc='Collect jobs'):
        summary = job.result()
        summaries.append(summary)

    needs_fix = []
    for summary in summaries:
        for chan_summary in summary['chans']:
            if chan_summary['bad_values']:
                chan_summary['coco_img'] = summary['coco_img']
                needs_fix.append(chan_summary)

    if 0:
        for chan_summary in ub.ProgIter(needs_fix, desc='fixing'):
            ...
            fix_geotiff(chan_summary)

    # for k, g in bad_groups.items():
    #     if all([p[2] == [0] for p in g])
    # print(len(investigate_images))
    # has_nonzeros = []
    # all_zeros = []
    # for summary in investigate_images:
    #     if len(set(summary['bad_values']) - {0}):
    #         has_nonzeros.append(summary)
    #     else:
    #         all_zeros.append(summary)
    # print(len(all_zeros))
    # print(len(has_nonzeros))
    # if 0:
    #     import xdev
    #     import kwplot
    #     kwplot.autompl()

    #     nonzero_chans = []
    #     for summary in has_nonzeros:
    #         coco_img = summary['coco_img']
    #         for chan_summary in summary['chans']:
    #             bad_values = chan_summary['bad_values']
    #             if len(set(bad_values) - {0}):
    #                 chan_summary['coco_img'] = coco_img
    #                 nonzero_chans.append(chan_summary)

    #     iter_ = xdev.InteractiveIter(nonzero_chans)
    #     for chan_summary in iter_:
    #         coco_img = chan_summary['coco_img']
    #         chan_canvas = draw_channel_summary(coco_img, chan_summary)
    #         kwplot.imshow(chan_canvas, doclf=1)
    #         xdev.InteractiveIter.draw()


def probe_image_issues(coco_img, channels=None, prefilter_channels=None, scale=None,
                       possible_nodata_values=None):
    """
    Args:
        coco_img : the coco image to check
        channels : the channels to check
        scale :
            use a downscaled overview to speed up the computation via
            approximation.
    """
    if channels is None:
        request_channels = coco_img.channels
    else:
        request_channels = kwcoco.ChannelSpec.coerce(channels)
    if prefilter_channels is not None:
        prefilter_channels = kwcoco.ChannelSpec.coerce(prefilter_channels)

    summary = {}
    chan_summaries = []

    prefilter_assets = []
    requested_assets = []
    for obj in coco_img.iter_asset_objs():
        chans = kwcoco.ChannelSpec.coerce(obj['channels'])
        if (chans & request_channels).fuse().numel():
            if (chans & prefilter_channels).fuse().numel():
                prefilter_assets.append(obj)
            else:
                requested_assets.append(obj)

    should_continue = len(prefilter_assets) == 0
    for obj in prefilter_assets:
        chan_summary = probe_channel(coco_img, obj, scale=scale,
                                     possible_nodata_values=possible_nodata_values)
        if chan_summary['bad_values']:
            should_continue = True
        chan_summaries.append(chan_summary)

    if should_continue:
        for obj in requested_assets:
            chan_summary = probe_channel(coco_img, obj, scale=scale,
                                         possible_nodata_values=possible_nodata_values)
            chan_summaries.append(chan_summary)

    summary['chans'] = chan_summaries
    summary['coco_img'] = coco_img
    summary['bad_values'] = list(ub.unique(ub.flatten([c['bad_values'] for c in chan_summaries])))
    return summary


def probe_channel(coco_img, obj, scale=None, possible_nodata_values=None):
    min_region_size = 256
    chan_summary = {
        'channels': obj['channels'],
    }
    delayed = coco_img.delay(
        channels=obj['channels'], space='asset', nodata_method='float')
    if scale is not None:
        delayed.prepare()
        delayed = delayed.scale(scale)
        delayed = delayed.optimize()
        min_region_size_ = int(min_region_size * scale)
    else:
        min_region_size_ = min_region_size
    data = delayed.finalize(interpolation='nearest')
    bundle_dpath = ub.Path(coco_img.bundle_dpath)
    fpath = bundle_dpath / obj['file_name']
    # print(f'fpath={fpath}')
    # data = kwimage.imread(fpath, backend='gdal', nodata_method='float',
    #                       overview=overview)

    is_samecolor = util_kwimage.find_samecolor_regions(
        data, min_region_size=min_region_size_)

    if np.any(is_samecolor):
        # is_same = is_samecolor > 0
        # same_values = data[is_same]
        # bad_values = np.unique(same_values)
        bad_labels, first_index = np.unique(is_samecolor, return_index=True)
        bad_values = data.ravel()[first_index]

        if possible_nodata_values is not None:
            # Remove anything that's not in our possible set of
            # bad nodata values
            import kwarray
            flags = kwarray.isect_flags(bad_values, possible_nodata_values)
            not_relevant = bad_labels[~flags]
            supress = kwarray.isect_flags(is_samecolor, not_relevant)
            is_samecolor[supress] = 0
            bad_values = bad_values[flags]
            bad_labels = bad_labels[flags]

        chan_summary['is_samecolor'] = is_samecolor
        chan_summary['fpath'] = fpath
        chan_summary['bad_values'] = sorted(set(bad_values.tolist()))
        # chan_summary['bad_labels'] = bad_values.tolist()
    else:
        chan_summary['bad_values'] = []
    return chan_summary


def fix_geotiff(chan_summary):
    fpath = chan_summary['fpath']
    is_samecolor = chan_summary['is_samecolor']

    new_fpath = ub.augpath(fpath, suffix='_fixed')

    correct_nodata_value = -9999

    from watch.utils import util_gdal
    src_dset = util_gdal.GdalDataset.open(fpath)
    assert src_dset.RasterCount == 1
    src_band = src_dset.GetRasterBand(1)
    num_overviews = src_band.GetOverviewCount()

    from osgeo import gdal
    driver1 = gdal.GetDriverByName(str('MEM'))
    copy1 = driver1.CreateCopy(str(''), src_dset)
    src_dset.FlushCache()
    src_dset = None

    # Modify the pixel contents
    band = copy1.GetRasterBand(1)
    band_data = band.ReadAsArray()
    band_data[is_samecolor > 0] = correct_nodata_value
    curr_nodat_value = band.GetNoDataValue()
    band.WriteArray(band_data)
    if curr_nodat_value != correct_nodata_value:
        band.SetNoDataValue(correct_nodata_value)

    overviewlist = (2 ** np.arange(1, num_overviews + 1)).tolist()
    copy1.BuildOverviews('AVERAGE', overviewlist)

    _options = [
        'BIGTIFF=YES',
        'TILED=YES',
        'BLOCKXSIZE={}'.format(256),
        'BLOCKYSIZE={}'.format(256),
    ]
    _options += ['COMPRESS={}'.format('DEFLATE')]
    _options.append('COPY_SRC_OVERVIEWS=YES')

    # Flush the in-memory dataset to an on-disk GeoTiff
    driver1 = None
    driver2 = gdal.GetDriverByName(str('GTiff'))
    copy2 = driver2.CreateCopy(new_fpath, copy1, options=_options)
    copy2.FlushCache()
    copy1 = None
    copy2 = None  # NOQA
    driver2 = None


def mwe():
    fpath = 'test.tif'
    new_fpath = 'result.tif'

    correct_nodata_value = -9999

    from osgeo import gdal
    src_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
    assert src_dset.RasterCount == 1
    src_band = src_dset.GetRasterBand(1)
    num_overviews = src_band.GetOverviewCount()

    driver1 = gdal.GetDriverByName(str('MEM'))
    copy1 = driver1.CreateCopy(str(''), src_dset)
    src_dset.FlushCache()
    src_dset = None

    # Modify the pixel contents
    band = copy1.GetRasterBand(1)
    band_data = band.ReadAsArray()
    band_data[band_data == 0] = correct_nodata_value
    curr_nodat_value = band.GetNoDataValue()
    band.WriteArray(band_data)
    if curr_nodat_value != correct_nodata_value:
        band.SetNoDataValue(correct_nodata_value)

    overviewlist = (2 ** np.arange(1, num_overviews + 1)).tolist()
    copy1.BuildOverviews('AVERAGE', overviewlist)

    _options = [
        'BIGTIFF=YES',
        'TILED=YES',
        'BLOCKXSIZE={}'.format(256),
        'BLOCKYSIZE={}'.format(256),
    ]
    _options += ['COMPRESS={}'.format('DEFLATE')]
    _options.append('COPY_SRC_OVERVIEWS=YES')

    # Flush the in-memory dataset to an on-disk GeoTiff
    driver1 = None
    driver2 = gdal.GetDriverByName(str('GTiff'))
    copy2 = driver2.CreateCopy(new_fpath, copy1, options=_options)
    copy2.FlushCache()
    copy1 = None
    copy2 = None  # NOQA
    driver2 = None


def draw_channel_summary(coco_img, chan_summary):
    is_samecolor = chan_summary['is_samecolor']
    data = coco_img.delay(channels=chan_summary['channels'], space='asset',
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
    title = coco_img.video['name'] + ' ' + chan_summary['channels'] + '\n' + coco_img.img['name']
    canvas = kwimage.draw_header_text(canvas, title)

    if 0:
        import kwplot
        kwplot.imshow(canvas)
    return canvas


def test_gdal_edit_data():
    import kwimage
    import numpy as np
    data = (np.random.rand(256, 256) * 100).astype(np.int16)
    data[20:30, 20:80] = -9999
    data[90:120, 30:50] = 0
    src_fpath = 'test.tif'
    dst_fpath = 'result.tif'
    kwimage.imwrite(src_fpath, data, backend='gdal', nodata_value=-9999, overviews=4)

    nodata_value = -9999
    src_fpath =  '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/./US_R007/L8/affine_warp/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0_nir.tif'
    src_fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/CN_C001/L8/affine_warp/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0_swir22.tif'
    data = kwimage.imread(src_fpath)
    from watch.utils import util_gdal
    dset = util_gdal.GdalDataset.open(src_fpath)
    self = dset

    proj = self.GetProjection()
    transform = self.GetGeoTransform()
    crs = self.GetSpatialRef()  # NOQA

    overviews = self.get_overview_info()[0]

    new_data = data.copy()
    new_data[new_data == 0] = nodata_value

    r = util_gdal.GdalDataset.open(dst_fpath)  # NOQA

    kwimage.imwrite(dst_fpath, data, transform=transform, crs=proj, nodata_value=nodata_value, overviews=overviews)

    import rasterio as rio
    src = rio.open(src_fpath, 'r')
    dst = rio.open(dst_fpath, 'w', **src.profile)

    with rio.open(src_fpath, 'r') as src:
        src.profile
        new = src.read()
        mask = (new == 0)
        nodata_value = -9999
        new[mask] = nodata_value
        with rio.open(dst_fpath, 'w', **src.profile) as dst:
            dst.write(new)

    # foo = kwimage.imread('result.tif')
    """
    gdal_calc.py -A test.tif --outfile=result.tif --calc="(-9999 * (A == 0)) + ((A != 0) * A)"
    echo "----"
    gdalinfo test.tif
    echo "----"
    gdalinfo result.tif

    gdalinfo /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/./US_R007/L8/affine_warp/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0_nir.tif


    gdalinfo /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-SC/BR_R005_0036_box/WV/affine_warp/crop_20200220T130000Z_S23.434718W046.499921_S23.424402W046.492905_WV_0/crop_20200220T130000Z_S23.434718W046.499921_S23.424402W046.492905_WV_0_blue.tif


    gdalinfo /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/CN_C001/L8/affine_warp/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0_swir22.tif

    """
    # foo = kwimage.imread('result.tif')
    # band = z.GetRasterBand(1)
    # from osgeo import gdal
    # info = gdal.Info(z, format='json')