"""
Adds fields needed by ndsampler to correctly "watch" a region.

Some of this is done hueristically. We assume images come from certain sensors.
We assume input is orthorectified.  We assume some GSD "target" gsd for video
and image processing. Note a video GSD will typically be much higher (i.e.
lower resolution) than an image GSD.
"""
import warnings
import numpy as np
import ubelt as ub
import kwimage
import itertools
import numbers

from os.path import join
from watch.utils import util_raster
from kwcoco.coco_image import CocoImage

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def filter_image_ids(coco_dset, gids=None, include_sensors=None,
                     exclude_sensors=None):
    """
    Filters to a specific set of images given query parameters
    """
    def coerce_set(x):
        return set(x.split(',')) if isinstance(x, str) else set(x)

    def filter_by_attribute(table, key, include, exclude):
        if include is not None or exclude is not None:
            if include is not None:
                include = coerce_set(include)
            if exclude is not None:
                exclude = coerce_set(exclude)
            values = table.lookup(key)
            if include is None:
                flags = [v not in exclude for v in values]
            elif exclude is None:
                flags = [v in include for v in values]
            else:
                flags = [v in include and v not in exclude for v in values]
            table = table.compress(flags)
        return table
    valid_images = coco_dset.images(gids)
    valid_images = filter_by_attribute(
        valid_images, 'sensor_coarse', include_sensors, exclude_sensors)
    valid_gids = list(valid_images)
    return valid_gids


def populate_watch_fields(coco_dset, target_gsd=10.0, vidids=None,
                          overwrite=False, default_gsd=None,
                          conform=True,
                          enable_video_stats=True,
                          enable_valid_region=False,
                          enable_intensity_stats=False,
                          workers=0,
                          mode='thread'):
    """
    Aggregate populate function for fields useful to WATCH.

    Args:
        coco_dset (Dataset): dataset to work with

        target_gsd (float): target gsd in meters

        overwrite (bool | List[str]):
            if True or False overwrites everything or nothing. Otherwise it can
            be a list of strings indicating what is
            overwritten. Valid keys are warp, band, and channels.

        default_gsd (None | float):
            if specified, assumed any images without geo-metadata have this
            GSD'

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
        >>> fpath = dvc_dpath / 'drop0_aligned/data.kwcoco.json')
        >>> coco_dset = kwcoco.CocoDataset(fpath)
        >>> target_gsd = 5.0
        >>> populate_watch_fields(coco_dset, target_gsd)
        >>> print('coco_dset.index.videos = {}'.format(ub.repr2(coco_dset.index.videos, nl=-1)))
        >>> print('coco_dset.index.imgs[1] = ' + ub.repr2(coco_dset.index.imgs[1], nl=1))

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> print('coco_dset = {!r}'.format(coco_dset))
        >>> target_gsd = 13.0
        >>> populate_watch_fields(coco_dset, target_gsd, default_gsd=1)
        >>> print('coco_dset.index.imgs[1] = ' + ub.repr2(coco_dset.index.imgs[1], nl=2))
        >>> print('coco_dset.index.videos = {}'.format(ub.repr2(coco_dset.index.videos, nl=1)))

        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> print('coco_dset = {!r}'.format(coco_dset))
        >>> target_gsd = 13.0
        >>> populate_watch_fields(coco_dset, target_gsd, default_gsd=1)
        >>> print('coco_dset.index.imgs[1] = ' + ub.repr2(coco_dset.index.imgs[1], nl=2))
        >>> print('coco_dset.index.videos = {}'.format(ub.repr2(coco_dset.index.videos, nl=1)))
    """
    # Load your KW-COCO dataset (conform populates information like image size)
    if conform:
        coco_dset.conform(pycocotools_info=False)

    if vidids is None:
        vidids = list(coco_dset.index.videos.keys())
        gids = list(coco_dset.index.imgs.keys())
    else:
        gids = list(ub.flatten(coco_dset.images(vidid=vidid) for vidid in vidids))

    coco_populate_geo_heuristics(
        coco_dset, gids=gids, overwrite=overwrite, default_gsd=default_gsd,
        workers=workers, mode=mode,
        enable_intensity_stats=enable_intensity_stats,
        enable_valid_region=enable_valid_region)

    if enable_video_stats:
        for vidid in ub.ProgIter(vidids, total=len(vidids), desc='populate videos'):
            coco_populate_geo_video_stats(coco_dset, vidid, target_gsd=target_gsd)

    # serialize intermediate objects
    coco_dset._ensure_json_serializable()


def coco_populate_geo_heuristics(coco_dset, gids=None, overwrite=False,
                                 default_gsd=None, workers=0, mode='thread', **kw):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> import kwcoco
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> coco_populate_geo_heuristics(coco_dset, overwrite=True, workers=12,
        >>>                              keep_geotiff_metadata=False,
        >>>                              mode='process')
    """
    gids = coco_dset.images(gids)._ids
    # Cant multiprocess because of SwigPyObjects... bleh
    # keep_geotiff_metadata must be False to use mode=process
    executor = ub.JobPool(mode, max_workers=workers)
    # executor = ub.JobPool('process', max_workers=workers)
    for gid in ub.ProgIter(gids, desc='submit populate imgs'):
        coco_img = coco_dset.coco_image(gid)
        if mode == 'process':
            coco_img = coco_img.detach()
        executor.submit(coco_populate_geo_img_heuristics2, coco_img,
                        overwrite=overwrite, default_gsd=default_gsd, **kw)
    for job in ub.ProgIter(executor.as_completed(), total=len(executor), desc='collect populate imgs'):
        img = job.result()
        if mode == 'process':
            # for multiprocessing
            real_img = coco_dset.index.imgs[img['id']]
            real_img.update(img)


@profile
def coco_populate_geo_img_heuristics2(coco_img, overwrite=False,
                                      default_gsd=None,
                                      keep_geotiff_metadata=False,
                                      enable_intensity_stats=False,
                                      enable_valid_region=False):
    """
    Note: this will not overwrite existing channel info unless specified

    Commandline
        xdoctest -m ~/code/watch/watch/utils/kwcoco_extensions.py --profile

    TODO:
        - [ ] Use logic in the align demo classmethod to make an example
              that uses a real L8 / S2 image.

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> import json
        >>> coco_dset = demo_kwcoco_with_heatmaps()
        >>> gid = 1
        >>> overwrite = {'warp', 'band'}
        >>> default_gsd = None
        >>> kw = {}
        >>> coco_img = coco_dset.coco_image(gid)
        >>> before_img_attrs = list(coco_img.img.keys())
        >>> before_aux_attr_hist = ub.dict_hist(ub.flatten([list(aux) for aux in coco_img.img['auxiliary']]))
        >>> print('before_img_attrs = {!r}'.format(before_img_attrs))
        >>> print('before_aux_attr_hist = {}'.format(ub.repr2(before_aux_attr_hist, nl=1)))
        >>> coco_populate_geo_img_heuristics2(coco_img)
        >>> img = coco_dset.index.imgs[gid]
        >>> after_img_attrs = list(coco_img.img.keys())
        >>> after_aux_attr_hist = ub.dict_hist(ub.flatten([list(aux) for aux in coco_img.img['auxiliary']]))
        >>> new_img_attrs = set(after_img_attrs) - set(before_img_attrs)
        >>> new_aux_attrs = {k: after_aux_attr_hist[k] - before_aux_attr_hist.get(k, 0) for k in after_aux_attr_hist}
        >>> new_aux_attrs = {k: v for k, v in new_aux_attrs.items() if v > 0}
        >>> print('new_img_attrs = {}'.format(ub.repr2(new_img_attrs, nl=1)))
        >>> print('new_aux_attrs = {}'.format(ub.repr2(new_aux_attrs, nl=1)))
        >>> #print('after_img_attrs = {}'.format(ub.repr2(after_img_attrs, nl=1)))
        >>> #print('after_aux_attr_hist = {}'.format(ub.repr2(after_aux_attr_hist, nl=1)))
        >>> assert 'geos_corners' in img
        >>> assert 'default_nodata' in img
        >>> assert 'default_nodata' in new_aux_attrs
        >>> print(ub.varied_values(list(map(lambda x: ub.map_vals(json.dumps, x), coco_img.img['auxiliary']))))

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> ###
        >>> gid = 1
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> coco_img = dset.coco_image(gid)
        >>> coco_populate_geo_img_heuristics2(coco_img, overwrite=True)
        >>> ###
        >>> gid = 1
        >>> dset2 = kwcoco.CocoDataset.demo('shapes8')
        >>> coco_img = dset2.coco_image(gid)
        >>> coco_populate_geo_img_heuristics2(coco_img, overwrite=True)
    """
    import watch
    bundle_dpath = coco_img.bundle_dpath
    img = coco_img.img

    primary_obj = coco_img.primary_asset()
    asset_objs = list(coco_img.iter_asset_objs())

    overwrite = _coerce_overwrite(overwrite)

    # Note: for non-geotiffs we could use the aux_to_img transformation
    # provided with them to determine their geo-properties.
    asset_errors = []
    for obj in asset_objs:
        errors = _populate_canvas_obj(
            bundle_dpath, obj, overwrite=overwrite, default_gsd=default_gsd,
            keep_geotiff_metadata=keep_geotiff_metadata,
            enable_intensity_stats=enable_intensity_stats)
        asset_errors.append(errors)

    if all(asset_errors):
        info = ub.dict_isect(img, {'name', 'file_name', 'id'})
        warnings.warn(f'img {info} has issues introspecting')

    if keep_geotiff_metadata:
        info = primary_obj.get('geotiff_metadata', None)
        if info is None:
            dem_hint = primary_obj.get('dem_hint', 'use')
            metakw = {}
            if dem_hint == 'ignore':
                metakw['elevation'] = 0
            primary_fname = primary_obj.get('file_name', None)
            primary_fpath = join(bundle_dpath, primary_fname)
            info = watch.gis.geotiff.geotiff_metadata(primary_fpath, **metakw)
            primary_obj['geotiff_metadata'] = info

    if 'default_nodata' not in img:
        img['default_nodata'] = primary_obj['default_nodata']

    valid_region_utm = img.get('valid_region_utm', None)
    if enable_valid_region and (valid_region_utm is None or 'warp' in overwrite):
        _populate_valid_region(coco_img)

    if keep_geotiff_metadata:
        img['geotiff_metadata'] = primary_obj['geotiff_metadata']

    if 'geos_corners' in primary_obj:
        # FIXME: we are assuming this maps perfectly onto the image
        # which is should for the SMART data, but perhaps in the future
        # this will not be safe?
        img['geos_corners'] = primary_obj['geos_corners']
    else:
        print('None of the assets had geo information')
    return img


def _populate_valid_region(coco_img):
    import watch
    # _ = ub.cmd('gdalinfo -stats {}'.format(fpath), check=True)
    bundle_dpath = coco_img.bundle_dpath
    img = coco_img.img
    primary_obj = coco_img.primary_asset()
    primary_fname = primary_obj.get('file_name', None)
    primary_fpath = join(bundle_dpath, primary_fname)

    sh_poly = util_raster.mask(
        primary_fpath, tolerance=10,
        default_nodata=primary_obj.get('default_nodata', None),
        # max_polys=100,
        convex_hull=True)
    kw_poly = kwimage.MultiPolygon.from_shapely(sh_poly)
    # print('kw_poly = {!r}'.format(kw_poly.data[0]))
    info = primary_obj.get('geotiff_metadata', None)
    if info is None:
        metakw = {}
        dem_hint = primary_obj.get('dem_hint', 'use')
        if dem_hint == 'ignore':
            metakw['elevation'] = 0
        info = watch.gis.geotiff.geotiff_metadata(primary_fpath, **metakw)

    # TODO: get a better heuristic here
    primary_obj['valid_region'] = kw_poly.to_coco(style='new')
    img['valid_region'] = kw_poly.to_coco(style='new')

    if 'pxl_to_wld' in info:
        pxl_to_wld = info['pxl_to_wld']
        kw_poly_utm = kw_poly.warp(pxl_to_wld).warp(info['wld_to_utm'])
        poly_utm = kw_poly_utm.to_geojson()
        poly_utm['properties'] = {}
        poly_utm['properties']['crs'] = info['utm_crs_info']
        primary_obj['valid_region_utm'] = poly_utm
        img['valid_region_utm'] = poly_utm


@profile
def _populate_canvas_obj(bundle_dpath, obj, overwrite=False, with_wgs=False,
                         default_gsd=None, keep_geotiff_metadata=False,
                         enable_intensity_stats=False):
    """
    obj can be an img or aux

    Ignore:
        from watch.utils.kwcoco_extensions import *  # NOQA
        from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        coco_dset = demo_kwcoco_with_heatmaps()
        coco_img = coco_dset.coco_image(1)
        obj = coco_img.primary_asset()
        bundle_dpath = coco_dset.bundle_dpath
        overwrite = True
        with_wgs = False
        default_gsd = None
        keep_geotiff_metadata = False
    """
    import watch
    import kwcoco
    sensor_coarse = obj.get('sensor_coarse', None)  # not reliable
    num_bands = obj.get('num_bands', None)
    channels = obj.get('channels', None)
    fname = obj.get('file_name', None)
    warp_to_wld = obj.get('warp_to_wld', None)
    approx_meter_gsd = obj.get('approx_meter_gsd', None)

    overwrite = _coerce_overwrite(overwrite)

    errors = []
    # Can only do this for images with file names
    if fname is not None:
        fpath = join(bundle_dpath, fname)

        info = None
        dem_hint = obj.get('dem_hint', 'use')
        metakw = {}
        if dem_hint == 'ignore':
            metakw['elevation'] = 0

        # TODO: ensure real nodata exists (maybe write helper file to disk?)
        sensor_coarse = obj.get('sensor_coarse', None)
        if sensor_coarse in {'S2', 'L8', 'WV'}:
            default_nodata = 0
        else:
            default_nodata = None
        # Heuristic for no-data
        obj['default_nodata'] = default_nodata

        if 'warp' in overwrite or warp_to_wld is None or approx_meter_gsd is None:
            try:
                if info is None:
                    info = watch.gis.geotiff.geotiff_metadata(fpath, **metakw)
                if keep_geotiff_metadata:
                    obj['geotiff_metadata'] = info
                height, width = info['img_shape'][0:2]

                obj['height'] = height
                obj['width'] = width
                # print('info = {!r}'.format(info))

                # WE NEED TO ACCOUNT FOR WLD_CRS TO USE THIS
                # obj_to_wld = kwimage.Affine.coerce(info['pxl_to_wld'])

                # FIXME: FOR NOW JUST USE THIS BIG HACK
                xy1_man = info['pxl_corners'].data.astype(np.float64)
                xy2_man = info['utm_corners'].data.astype(np.float64)
                hack_aff = kwimage.Affine.coerce(
                    fit_affine_matrix(xy1_man, xy2_man))
                obj_to_wld = kwimage.Affine.coerce(hack_aff)
                # cv2.getAffineTransform(utm_corners, pxl_corners)

                wgs84_crs_info = ub.dict_diff(info['wgs84_crs_info'], {'type'})
                if wgs84_crs_info['axis_mapping'] == 'OAMS_AUTHORITY_COMPLIANT':
                    geos_corners = kwimage.Polygon.coerce(info['wgs84_corners']).swap_axes().to_geojson()
                else:
                    geos_corners = kwimage.Polygon.coerce(info['wgs84_corners']).to_geojson()
                geos_crs_info = {
                    'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
                    'auth': ('EPSG', '4326')
                }
                geos_corners['properties'] = {'crs_info': geos_crs_info}

                wld_crs_info = ub.dict_diff(info['wld_crs_info'], {'type'})
                utm_crs_info = ub.dict_diff(info['utm_crs_info'], {'type'})
                obj.update({
                    'geos_corners': geos_corners,  # always in geojson
                    'wgs84_corners': info['wgs84_corners'].data.tolist(),
                    'utm_corners': info['utm_corners'].data.tolist(),
                    'wld_crs_info': wld_crs_info,
                    'utm_crs_info': utm_crs_info,
                })
                obj['band_metas'] = info['band_metas']
                obj['is_rpc'] = info['is_rpc']

                if with_wgs:
                    obj.update({
                        'wgs84_to_wld': info['wgs84_to_wld'],
                        'wld_to_pxl': info['wld_to_pxl'],
                    })

                approx_meter_gsd = info['approx_meter_gsd']
            except Exception as ex:
                if default_gsd is not None:
                    obj['approx_meter_gsd'] = default_gsd
                    obj['warp_to_wld'] = kwimage.Affine.eye().__json__()
                else:
                    # FIXME: This might not be the best way to report errors
                    # raise
                    errors.append('no_crs_info: {!r}'.format(ex))
            else:
                obj['approx_meter_gsd'] = approx_meter_gsd
                obj['warp_to_wld'] = kwimage.Affine.coerce(obj_to_wld).__json__()

        if 'band' in overwrite or num_bands is None:
            try:
                num_bands = _introspect_num_bands(fpath)
            except Exception:
                channels = obj.get('channels', None)
                if channels is not None:
                    num_bands = kwcoco.ChannelSpec(channels).numel()
                else:
                    raise
            obj['num_bands'] = num_bands

        if 'channels' in overwrite or channels is None:
            if sensor_coarse is not None:
                channels = _sensor_channel_hueristic(sensor_coarse, num_bands)
            elif num_bands is not None:
                channels = _num_band_hueristic(num_bands)
            else:
                raise Exception(ub.paragraph(
                    f'''
                    no methods to introspect channels
                    sensor_coarse={sensor_coarse},
                    num_bands={num_bands}
                    for obj={obj}
                    '''))
            obj['channels'] = channels

        # TODO: determine nodata defaults based on sensor_coarse

        if enable_intensity_stats:
            # TODO: rectify with code in cli/coco_intensity_histogram
            # Use a sidecar file for now
            import pathlib
            import pickle
            stats_fpath = pathlib.Path(fpath + '.stats.pkl')
            # if _is_writeable(stats_fpath.parent):
            #     pass
            if not stats_fpath.exists():
                import kwarray
                imdata = kwimage.imread(fpath)
                imdata = kwarray.atleast_nd(imdata, 3)
                stats_info = {'bands': []}
                for imband in imdata.transpose(2, 0, 1):
                    data = imband.ravel()
                    intensity_hist = ub.dict_hist(data)
                    intensity_hist = ub.sorted_keys(intensity_hist)
                    stats_info['bands'].append({
                        'intensity_hist': intensity_hist,
                    })
                with open(stats_fpath, 'wb') as file:
                    pickle.dump(stats_info, file)
            else:
                pass

        return errors


@ub.memoize
def _is_writeable(dpath):
    " https://stackoverflow.com/questions/2113427/determining-whether-a-directory-is-writeable "
    import os
    return os.access(dpath, os.W_OK) and os.path.isdir(dpath)


def _coerce_overwrite(overwrite):
    """
    Im not a big fan of the way overwrite currently works, might want to
    refactor.
    """
    valid_overwrites = {'warp', 'band', 'channels'}
    default_overwrites = {'warp', 'band'}
    if isinstance(overwrite, str):
        overwrite = set(overwrite.split(','))
    if overwrite is True:
        overwrite = default_overwrites
    elif overwrite is False:
        overwrite = {}
    else:
        overwrite = set(overwrite)
        unexpected = overwrite - valid_overwrites
        if unexpected:
            raise ValueError(f'Got unexpected overwrites: {unexpected}')
    return overwrite


# def single_geotiff_metadata(bundle_dpath, img, serializable=False):
#     import watch
#     from os.path import exists
#     import dateutil
#     geotiff_metadata = None
#     aux_metadata = []

#     img['datetime_acquisition'] = (
#         dateutil.parser.parse(img['date_captured'])
#     )

#     # if an image specified its "dem_hint" as ignore, then we set the
#     # elevation to 0. NOTE: this convention might be generalized and
#     # replaced in the future. I.e. in the future the dem_hint might simply
#     # specify the constant elevation to use, or perhaps something else.
#     dem_hint = img.get('dem_hint', 'use')
#     metakw = {}
#     if dem_hint == 'ignore':
#         metakw['elevation'] = 0

#     # only need rpc info, wgs84_corners, and and warps
#     keys_of_interest = {
#         'rpc_transform',
#         'is_rpc',
#         'wgs84_to_wld',
#         'wgs84_corners',
#         'wld_to_pxl',
#     }

#     fname = img.get('file_name', None)
#     if fname is not None:
#         src_gpath = join(bundle_dpath, fname)
#         assert exists(src_gpath)
#         img_info = watch.gis.geotiff.geotiff_metadata(src_gpath, **metakw)

#         if serializable:
#             raise NotImplementedError
#         else:
#             img_info = ub.dict_isect(img_info, keys_of_interest)
#             geotiff_metadata = img_info

#     for aux in img.get('auxiliary', []):
#         aux_fpath = join(bundle_dpath, aux['file_name'])
#         assert exists(aux_fpath)
#         aux_info = watch.gis.geotiff.geotiff_metadata(aux_fpath, **metakw)
#         aux_info = ub.dict_isect(aux_info, keys_of_interest)
#         if serializable:
#             raise NotImplementedError
#         else:
#             aux_metadata.append(aux_info)
#             aux['geotiff_metadata'] = aux_info

#     if fname is None:
#         # need to choose one of the auxiliary images as the "main" image.
#         # We are assuming that there is one auxiliary image that exactly
#         # corresponds.
#         candidates = []
#         for aux in img.get('auxiliary', []):
#             if aux['width'] == img['width'] and aux['height'] == img['height']:
#                 candidates.append(aux)

#         if not candidates:
#             raise AssertionError(
#                 'Assumed at least one auxiliary image has identity '
#                 'transform, but this seems to not be the case')
#         aux = ub.peek(candidates)
#         geotiff_metadata = aux['geotiff_metadata']

#     img['geotiff_metadata'] = geotiff_metadata
#     return geotiff_metadata, aux_metadata


@profile
def coco_populate_geo_video_stats(coco_dset, vidid, target_gsd='max-resolution'):
    """
    Create a "video-space" for all images in a video sequence at a specified
    resolution.

    For this video, this chooses the "best" image as the "video canvas /
    region" and registers everything to that canvas/region. This creates the
    "video-space" for this image sequence. Currently the "best" image is the
    one that has the GSD closest to the target-gsd. This hueristic works well
    in most cases, but no all.

    Notes:
        * Currently the "best image" exactly define the video canvas / region.

        * Areas where other images do not overlap the vieo canvas are
          effectively lost when sampling in video space, because anything
          outside the video canvas is cropped out.

        * Auxilary images are required to have an "approx_meter_gsd" and a
          "warp_to_wld" attribute to use this function atm.

    TODO:
        - [ ] Allow choosing of a custom "video-canvas" not based on any one image.
        - [ ] Allow choosing a "video-canvas" that encompases all images
        - [ ] Allow the base image to contain "approx_meter_gsd" /
              "warp_to_wld" instead of the auxiliary image
        - [ ] Is computing the scale factor based on approx_meter_gsd safe?

    Args:
        coco_dset (CocoDataset): coco dataset to be modified inplace
        vidid (int): video_id to modify
        target_gsd (float | str): string code, or float target gsd


    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> import kwcoco
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> target_gsd = 10.0
        >>> vidid = 2
        >>> # We can check transforms before we apply this function
        >>> coco_dset.images(vidid=vidid).lookup('warp_img_to_vid', None)
        >>> # Apply the function
        >>> coco_populate_geo_video_stats(coco_dset, vidid, target_gsd)
        >>> # Check these transforms to make sure they look right
        >>> popualted_video = coco_dset.index.videos[vidid]
        >>> popualted_video = ub.dict_isect(popualted_video, ['width', 'height', 'warp_wld_to_vid', 'target_gsd'])
        >>> print('popualted_video = {}'.format(ub.repr2(popualted_video, nl=-1)))
        >>> coco_dset.images(vidid=vidid).lookup('warp_img_to_vid')

        # TODO: make a demo dataset with some sort of gsd metadata
        coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        print('coco_dset = {!r}'.format(coco_dset))

        coco_fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        coco_fpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/combo_data.kwcoco.json'
        coco_dset = kwcoco.CocoDataset(coco_fpath)
        vidid = 1

        target_gsd = 2.8
    """
    # Compute an image-to-video transform that aligns all frames to some
    # common resolution.
    video = coco_dset.index.videos[vidid]
    gids = coco_dset.index.vidid_to_gids[vidid]

    check_unique_channel_names(coco_dset, gids=gids)

    frame_infos = {}

    for gid in gids:
        img = coco_dset.index.imgs[gid]
        coco_img = CocoImage(img)

        # If the base dictionary has "warp_to_wld" and "approx_meter_gsd"
        # information we use that.
        wld_from_img = img.get('warp_to_wld', None)
        approx_meter_gsd = img.get('approx_meter_gsd', None)
        wld_crs_info = img.get('wld_crs_info', None)

        # Otherwise we try to obtain it from the auxiliary images
        if approx_meter_gsd is None or wld_from_img is None:
            # Choose any one of the auxiliary images that has the required
            # attribute
            aux_chosen = coco_img.primary_asset(requires=[
                'warp_to_wld', 'approx_meter_gsd'])
            if aux_chosen is None:
                raise Exception(ub.paragraph(
                    '''
                    Image auxiliary images have no warp_to_wld and approx_meter
                    gsd. The auxiliary images may not have associated geo
                    metadata.
                    '''))

            wld_from_aux = kwimage.Affine.coerce(aux_chosen.get('warp_to_wld', None))
            img_from_aux = kwimage.Affine.coerce(aux_chosen['warp_aux_to_img'])
            aux_from_img = img_from_aux.inv()
            wld_from_img = wld_from_aux @ aux_from_img
            approx_meter_gsd = aux_chosen['approx_meter_gsd']
            wld_crs_info = aux_chosen.get('wld_crs_info', None)

        if approx_meter_gsd is None or wld_from_img is None:
            raise Exception(ub.paragraph(
                '''
                Both the base image and its auxiliary images do not seem to
                have the required warp_to_wld and approx_meter_gsd fields.
                The image may not have associated geo metadata.
                '''))

        wld_from_img = kwimage.Affine.coerce(wld_from_img)

        asset_channels = []
        asset_gsds = []
        for obj in coco_img.iter_asset_objs():
            _gsd = obj.get('approx_meter_gsd')
            if _gsd is not None:
                _gsd = round(_gsd, 1)
            asset_gsds.append(_gsd)
            asset_channels.append(obj.get('channels', None))

        frame_infos[gid] = {
            'img_to_wld': wld_from_img,
            'wld_crs_info': wld_crs_info,
            # Note: division because gsd is inverted. This got me confused, but
            # I'm pretty sure this works.
            'target_gsd': target_gsd,
            'approx_meter_gsd': approx_meter_gsd,
            'width': img['width'],
            'height': img['height'],
            'asset_channels': asset_channels,
            'asset_gsds': asset_gsds,
        }

    sorted_gids = ub.argsort(frame_infos, key=lambda x: x['approx_meter_gsd'])
    min_gsd_gid = sorted_gids[0]
    max_gsd_gid = sorted_gids[-1]
    max_example = frame_infos[max_gsd_gid]
    min_example = frame_infos[min_gsd_gid]
    max_gsd = max_example['approx_meter_gsd']
    min_gsd = min_example['approx_meter_gsd']

    # TODO: coerce datetime via kwcoco API
    if target_gsd == 'max-resolution':
        target_gsd_ = min_gsd
    elif target_gsd == 'min-resolution':
        target_gsd_ = max_gsd
    else:
        target_gsd_ = target_gsd
        if not isinstance(target_gsd, numbers.Number):
            raise TypeError('target_gsd must be a code or number = {}'.format(type(target_gsd)))
    target_gsd_ = float(target_gsd_)

    # Compute the scale factor needed to be applied to each image to achieve
    # the target videospace GSD.
    for info in frame_infos.values():
        info['target_gsd'] = target_gsd_
        info['to_target_scale_factor'] = info['approx_meter_gsd'] / target_gsd_

    available_channels = set()
    available_gsds = set()
    for gid in gids:
        img = coco_dset.index.imgs[gid]
        for obj in coco_img.iter_asset_objs():
            available_channels.add(obj.get('channels', None))
            _gsd = obj.get('approx_meter_gsd')
            if _gsd is not None:
                available_gsds.add(round(_gsd, 1))

    # Align to frame closest to the target GSD, which is the frame that has the
    # "to_target_scale_factor" that is closest to 1.0
    base_gid, base_info = min(
        frame_infos.items(),
        key=lambda kv: abs(1 - kv[1]['to_target_scale_factor'])
    )
    scale = base_info['to_target_scale_factor']
    base_wld_crs_info = base_info['wld_crs_info']
    # if base_wld_crs_info is None:
    #     import xdev
    #     xdev.embed()

    # Can add an extra transform here if the video is not exactly in
    # any specific image space
    baseimg_from_wld = base_info['img_to_wld'].inv()
    vid_from_wld = kwimage.Affine.scale(scale) @ baseimg_from_wld
    video['width'] = int(np.ceil(base_info['width'] * scale))
    video['height'] = int(np.ceil(base_info['height'] * scale))

    # Store metadata in the video
    video['num_frames'] = len(gids)
    video['warp_wld_to_vid'] = vid_from_wld.__json__()
    video['target_gsd'] = target_gsd_
    video['min_gsd'] = min_gsd
    video['max_gsd'] = max_gsd

    # Remove old cruft (can remove in future versions)
    video.pop('available_channels', None)

    for gid in gids:
        img = coco_dset.index.imgs[gid]
        wld_from_img = frame_infos[gid]['img_to_wld']
        wld_crs_info = frame_infos[gid]['wld_crs_info']
        vid_from_img = vid_from_wld @ wld_from_img
        img['warp_img_to_vid'] = vid_from_img.concise()

        if base_wld_crs_info != wld_crs_info:
            import warnings
            warnings.warn(ub.paragraph(
                f'''
                Video alignment is warping images with different World
                Coordinate Reference Systems, but still treating them as the
                same. FIXME
                base_wld_crs_info={base_wld_crs_info!r},
                wld_crs_info={wld_crs_info!r}
                '''))


def check_unique_channel_names(coco_dset, gids=None, verbose=0):
    """
    Check each image has unique channel names

    TODO:
        - [ ] move to kwcoco proper

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> check_unique_channel_names(coco_dset)
        >>> # Make some duplicate channels to test
        >>> obj = coco_dset.images().objs[0]
        >>> obj['auxiliary'][0]['channels'] = 'B1|B1'
        >>> obj = coco_dset.images().objs[1]
        >>> obj['auxiliary'][0]['channels'] = 'B1|B1'
        >>> obj = coco_dset.images().objs[2]
        >>> obj['auxiliary'][1]['channels'] = 'B1'
        >>> import pytest
        >>> with pytest.raises(AssertionError):
        >>>     check_unique_channel_names(coco_dset)

    """
    images = coco_dset.images(gids=gids)
    errors = []
    for img in images.objs:
        coco_img = coco_dset._coco_image(img['id'])
        try:
            _check_unique_channel_names_in_image(coco_img)
        except AssertionError as ex:
            if verbose:
                print('ERROR: ex = {}'.format(ub.repr2(ex, nl=1)))
            errors.append(ex)

    if errors:
        error_summary = ub.dict_hist(map(str, errors))
        raise AssertionError(ub.repr2(error_summary))


def _check_unique_channel_names_in_image(coco_img):
    import kwcoco
    seen = set()
    for obj in coco_img.iter_asset_objs():
        chans = kwcoco.FusedChannelSpec.coerce(obj['channels'])
        chan_list : list = chans.normalize().parsed
        intra_aux_duplicate = ub.find_duplicates(chan_list)
        if intra_aux_duplicate:
            raise AssertionError(
                'Image has internal duplicate bands: {}'.format(
                    intra_aux_duplicate))

        inter_aux_duplicates = seen & set(chan_list)
        if inter_aux_duplicates:
            raise AssertionError(
                'Image has inter-auxiliary duplicate bands: {}'.format(
                    inter_aux_duplicates))


def coco_list_asset_infos(coco_dset):
    """
    Get a list of filename and channels for each coco image
    """
    asset_infos = []
    for gid in coco_dset.images():
        coco_img = coco_dset._coco_image(gid)
        asset_objs = list(coco_img.iter_asset_objs())
        for _asset_idx, obj in enumerate(asset_objs):
            fname = obj.get('file_name', None)
            if fname is not None:
                fpath = join(coco_img.dset.bundle_dpath, fname)
                file_info = {
                    'fpath': fpath,
                    'channels': obj['channels'],
                }
                asset_infos.append(file_info)
    return asset_infos


def check_geotiff_formats(coco_dset):
    # Enumerate assests on disk
    infos = []
    asset_infos = coco_list_asset_infos(coco_dset)
    for file_info in ub.ProgIter(asset_infos):
        fpath = file_info['fpath']
        info = geotiff_format_info(fpath)
        info.update(file_info)
        infos.append(info)

    ub.varied_values([ub.dict_diff(d, {'fpath', 'filelist'}) for d in infos])


def rewrite_geotiffs(coco_dset):
    import tempfile
    import pathlib
    blocksize = 96
    compress = 'NONE'
    asset_infos = coco_list_asset_infos(coco_dset)

    for file_info in ub.ProgIter(asset_infos):
        fpath = file_info['fpath']
        if fpath.endswith(kwimage.im_io.JPG_EXTENSIONS):
            print('Skipping jpeg')
            # dont touch jpegs
            continue

        orig_fpath = pathlib.Path(fpath)

        info = geotiff_format_info(fpath)
        if (info['blocksize'][0] != blocksize or info['compress'] != compress) or True:
            tmpdir = orig_fpath.parent / '.tmp_gdal_workspace'
            tmpdir.mkdir(exist_ok=True, parents=True)
            workdir = tmpdir / 'work'
            bakdir = tmpdir / 'backup_v2'
            workdir.mkdir(exist_ok=True)
            bakdir.mkdir(exist_ok=True)

            tmpfile = tempfile.NamedTemporaryFile(suffix=orig_fpath.name, dir=workdir, delete=False)
            tmp_fpath = tmpfile.name

            options = [
                '-co BLOCKSIZE={}'.format(blocksize),
                '-co COMPRESS={}'.format(compress),
                '-of COG',
                '-overwrite',
            ]
            if not info['has_geotransform']:
                options += [
                    '-to SRC_METHOD=NO_GEOTRANSFORM'
                ]
            options += [
                fpath,
                tmp_fpath,
            ]
            command = 'gdalwarp ' + ' '.join(options)
            cmdinfo = ub.cmd(command)
            if cmdinfo['ret'] != 0:
                print('cmdinfo = {}'.format(ub.repr2(cmdinfo, nl=1)))
                raise Exception('Command Errored')

            # Backup the original file
            import shutil
            shutil.move(fpath, bakdir)

            # Move the rewritten file into its place
            shutil.move(tmp_fpath, fpath)

            # info2 = geotiff_format_info(tmp_fpath)


def geotiff_format_info(fpath):
    from osgeo import gdal
    gdal_ds = gdal.Open(fpath, gdal.GA_ReadOnly)
    filelist = gdal_ds.GetFileList()

    aff_wld_crs = gdal_ds.GetSpatialRef()
    has_geotransform = aff_wld_crs is not None

    filename = gdal_ds.GetDescription()
    main_band = gdal_ds.GetRasterBand(1)
    block_size = main_band.GetBlockSize()

    num_bands = gdal_ds.RasterCount
    width = gdal_ds.RasterXSize
    height = gdal_ds.RasterYSize

    ovr_count = main_band.GetOverviewCount()
    ifd_offset = int(main_band.GetMetadataItem('IFD_OFFSET', 'TIFF'))
    block_offset = main_band.GetMetadataItem('BLOCK_OFFSET_0_0', 'TIFF')
    structure = gdal_ds.GetMetadata("IMAGE_STRUCTURE")
    compress = structure.get("COMPRESSION", 'NONE')
    interleave = structure.get("INTERLEAVE", None)

    has_external_overview = (filename + '.ovr' in filelist)

    format_info = {
        'fpath': fpath,
        'filelist': filelist,
        'blocksize': block_size,
        'ovr_count': ovr_count,
        'ifd_offset': ifd_offset,
        'block_offset': block_offset,
        'compress': compress,
        'interleave': interleave,
        'has_external_overview': has_external_overview,
        'num_bands': num_bands,
        'has_geotransform': has_geotransform,
        'width': width,
        'height': height,
    }
    return format_info


def ensure_transfered_geo_data(coco_dset, gids=None):
    for gid in ub.ProgIter(coco_dset.images(gids), desc='transfer metadata'):
        transfer_geo_metadata(coco_dset, gid)


@profile
def transfer_geo_metadata(coco_dset, gid):
    """
    Transfer geo-metadata from source geotiffs to predicted feature images

    THIS FUNCITON MODIFIES THE IMAGE DATA ON DISK! BE CAREFUL!

    ASSUMES THAT EVERYTHING IS ALREADY ALIGNED

    Example:
        # xdoctest: +REQUIRES(env:DVC_DPATH)
        from watch.utils.kwcoco_extensions import *  # NOQA
        from watch.utils.util_data import find_smart_dvc_dpath
        import kwcoco
        dvc_dpath = find_smart_dvc_dpath()
        coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
        coco_dset = kwcoco.CocoDataset(coco_fpath)
        gid = coco_dset.images().peek()['id']

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> from watch.demo.smart_kwcoco_demodata import hack_seed_geometadata_in_dset
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> hack_seed_geometadata_in_dset(coco_dset, force=True, rng=0)
        >>> gid = 2
        >>> transfer_geo_metadata(coco_dset, gid)
        >>> fpath = join(coco_dset.bundle_dpath, coco_dset._coco_image(gid).primary_asset()['file_name'])
        >>> _ = ub.cmd('gdalinfo ' + fpath, verbose=1)
    """
    import watch
    from osgeo import gdal
    import affine
    coco_img = coco_dset._coco_image(gid)

    assets_with_geo_info = {}
    assets_without_geo_info = {}

    asset_objs = list(coco_img.iter_asset_objs())
    for asset_idx, obj in enumerate(asset_objs):
        fname = obj.get('file_name', None)
        if fname is not None:
            fpath = join(coco_img.dset.bundle_dpath, fname)
            try:
                info = watch.gis.geotiff.geotiff_metadata(fpath)
                if info.get('crs_error', None) is not None:
                    raise Exception
            except Exception:
                assets_without_geo_info[asset_idx] = obj
            else:
                assets_with_geo_info[asset_idx] = (obj, info)

    warp_vid_from_geoimg = kwimage.Affine.eye()

    if assets_without_geo_info:
        if not assets_with_geo_info:
            class Found(Exception):
                pass
            try:
                # If an asset in our local image has no data, we can
                # check to see if anyone in the vide has data.
                # Check if anything in the video has geo-data
                vidid = coco_img.img['video_id']
                for other_gid in coco_dset.images(vidid=vidid):
                    if other_gid != gid:
                        other_coco_img = coco_dset._coco_image(other_gid)
                        for obj in other_coco_img.iter_asset_objs():
                            fname = obj.get('file_name', None)
                            if fname is not None:
                                fpath = join(coco_img.dset.bundle_dpath, fname)
                                try:
                                    # Try until we find an image with real CRS info
                                    info = watch.gis.geotiff.geotiff_metadata(fpath)
                                    if info.get('crs_error', None) is not None:
                                        raise Exception
                                except Exception:
                                    continue
                                else:
                                    raise Found
            except Found:
                assets_with_geo_info[-1] = (obj, info)
                warp_vid_from_geoimg = kwimage.Affine.coerce(other_coco_img.img['warp_img_to_vid'])
            else:
                raise ValueError(ub.paragraph(
                    '''
                    There are images without geo data, but no other data within
                    this image has transferable geo-data
                    '''))

        # Choose an object to register to (not sure if it matters which one)
        # choose arbitrary one for now.
        geo_asset_idx, (geo_obj, geo_info) = ub.peek(assets_with_geo_info.items())
        geo_fname = geo_obj.get('file_name', None)
        geo_fpath = join(coco_img.dset.bundle_dpath, geo_fname)

        if geo_info['is_rpc']:
            raise NotImplementedError(
                'Not sure how to do this if the target has RPC information')

        geo_ds = gdal.Open(geo_fpath)
        geo_ds.GetProjection()

        warp_geoimg_from_geoaux = kwimage.Affine.coerce(
            geo_obj.get('warp_aux_to_img', None))
        warp_wld_from_geoaux = kwimage.Affine.coerce(geo_info['pxl_to_wld'])

        georef_crs_info = geo_info['wld_crs_info']
        georef_crs = georef_crs_info['type']

        img = coco_img.img
        warp_vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])

        # In case our reference is from another frame in the video
        warp_geoimg_from_vid = warp_vid_from_geoimg.inv()
        warp_geoaux_from_geoimg = warp_geoimg_from_geoaux.inv()
        warp_wld_from_img = (
            warp_wld_from_geoaux @
            warp_geoaux_from_geoimg @
            warp_geoimg_from_vid @
            warp_vid_from_img)

        for _asset_idx, obj in assets_without_geo_info.items():
            fname = obj.get('file_name', None)
            fpath = join(coco_img.dset.bundle_dpath, fname)

            warp_img_from_aux = kwimage.Affine.coerce(
                obj.get('warp_aux_to_img', None))

            warp_wld_from_aux = (
                warp_wld_from_img @ warp_img_from_aux)

            # Convert to gdal-style
            a, b, c, d, e, f = warp_wld_from_aux.matrix.ravel()[0:6]
            aff = affine.Affine(a, b, c, d, e, f)
            aff_geo_transform = aff.to_gdal()

            dst_ds = gdal.Open(fpath, gdal.GA_Update)
            if dst_ds is None:
                raise Exception('error handling gdal')
            ret = dst_ds.SetGeoTransform(aff_geo_transform)
            assert ret == 0
            ret = dst_ds.SetSpatialRef(georef_crs)
            assert ret == 0
            dst_ds.FlushCache()
            dst_ds = None

        # Matt's transfer metadata code
        """
        geo_ds = gdal.Open(toafile)
        if geo_ds is None:
            log.error('Could not open image')
            sys.exit(1)
        transform = geo_ds.GetGeoTransform()
        proj = geo_ds.GetProjection()
        dst_ds = gdal.Open(boafile, gdal.GA_Update)
        dst_ds.SetGeoTransform(transform)
        dst_ds.SetProjection(proj)
        geo_ds, dst_ds = None, None
        """


def _make_coco_img_from_geotiff(tiff_fpath, name=None):
    """
    Example:
        >>> from watch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> product = grab_landsat_product()
        >>> tiffs = product['bands'] + [product['meta']['bqa']]
        >>> tiff_fpath = product['bands'][0]
        >>> name = None
        >>> img = _make_coco_img_from_geotiff(tiff_fpath)
        >>> print('img = {}'.format(ub.repr2(img, nl=1)))
    """
    obj = {}
    if name is not None:
        obj['name'] = name

    bundle_dpath = '.'
    obj = {
        'file_name': tiff_fpath
    }
    _populate_canvas_obj(bundle_dpath, obj)
    return obj


@profile
def fit_affine_matrix(xy1_man, xy2_man):
    """
    Sympy:
        import sympy as sym
        x1, y1, x2, y2 = sym.symbols('x1, y1, x2, y2')
        A = sym.Matrix([
            [x1, y1,  0,  0, 1, 0],
            [ 0,  0, x1, y1, 0, 1],
        ])
        b = sym.Matrix([[x2], [y2]])
        x = (A.T.multiply(A)).inv().multiply(A.T.multiply(b))
        x = (A.T.multiply(A)).pinv().multiply(A.T.multiply(b))

    References:
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf page 22
    """
    x1_mn = xy1_man.T[0]
    y1_mn = xy1_man.T[1]
    x2_mn = xy2_man.T[0]
    y2_mn = xy2_man.T[1]
    num_pts = x1_mn.shape[0]
    Mx6 = np.empty((2 * num_pts, 6), dtype=np.float64)
    b = np.empty((2 * num_pts, 1), dtype=np.float64)
    for ix in range(num_pts):  # Loop over inliers
        # Concatenate all 2x9 matrices into an Mx6 matrix
        x1 = x1_mn[ix]
        x2 = x2_mn[ix]
        y1 = y1_mn[ix]
        y2 = y2_mn[ix]
        Mx6[ix * 2]     = (x1, y1, 0, 0, 1, 0)
        Mx6[ix * 2 + 1] = ( 0, 0, x1, y1, 0, 1)
        b[ix * 2] = x2
        b[ix * 2 + 1] = y2

    M = Mx6
    try:
        USVt = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    except MemoryError:
        import scipy.sparse as sps
        import scipy.sparse.linalg as spsl
        M_sparse = sps.lil_matrix(M)
        USVt = spsl.svds(M_sparse)
    except np.linalg.LinAlgError:
        raise
    except Exception:
        raise

    U, s, Vt = USVt

    # Inefficient, but I think the math works
    # We want to solve Ax=b (where A is the Mx6 in this case)
    # Ax = b
    # (U S V.T) x = b
    # x = (U.T inv(S) V) b
    Sinv = np.zeros((len(Vt), len(U)))
    Sinv[np.diag_indices(len(s))] = 1 / s
    a = Vt.T.dot(Sinv).dot(U.T).dot(b).T[0]
    A = np.array([
        [a[0], a[1], a[4]],
        [a[2], a[3], a[5]],
        [   0, 0, 1],
    ])
    return A


def _sensor_channel_hueristic(sensor_coarse, num_bands):
    """
    Given a sensor and the number of bands in the image, return likely channel
    codes for the image

    Note these are "pseudo-harmonized" by common_name, but not harmonized
    that is, one sensor's 'red' is roughly similar to another's but not corrected to match.
    Bands without a common_name will have a sensor-unique prefix appended to prevent this behavior.
    """
    from watch.utils.util_bands import WORLDVIEW2_PAN, WORLDVIEW2_MS4, WORLDVIEW2_MS8, SENTINEL2, LANDSAT8, LANDSAT7  # NOQA

    def code(bands, prefix):
        names = []
        for band_dict in bands:
            if 'common_name' in band_dict:
                names.append(band_dict['common_name'])
            else:
                names.append(prefix + band_dict['name'])
        return '|'.join(names)

    err = 0
    if sensor_coarse == 'WV':
        if num_bands == 1:
            channels = 'panchromatic'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 4:
            channels = code(WORLDVIEW2_MS4, 'w')
        elif num_bands == 8:
            channels = code(WORLDVIEW2_MS8, 'w')
            #channels = 'wv1|wv2|wv3|wv4|wv5|wv6|wv7|wv8'
            # channels = 'cb|b|g|y|r|wv6|wv7|wv8'
        else:
            err = 1
    elif sensor_coarse == 'S2':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 13:
            channels = code(SENTINEL2, 's')
            # channels = 's1|s2|s3|s4|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
            # channels = 'cb|b|g|r|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
        else:
            err = 1
    elif sensor_coarse in {'LC', 'L8', 'LS'}:
        if num_bands == 1:
            channels = 'panchromatic'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 11:
            channels = code(LANDSAT8, 'l8')
            # channels = 'lc1|lc2|lc3|lc4|lc5|lc6|lc7|lc8|lc9|lc10|lc11'
            # channels = 'cb|b|g|r|lc5|lc6|lc7|pan|lc9|lc10|lc11'
        else:
            err = 1
    elif sensor_coarse in {'LE', 'L7'}:
        if num_bands == 1:
            channels = 'panchromatic'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 8:
            channels = code(LANDSAT7, 'l7')
        else:
            err = 1
    else:
        err = 1
    if err:
        msg = f'sensor_coarse={sensor_coarse}, num_bands={num_bands}'
        print('ERROR: mgs = {!r}'.format(msg))
        raise NotImplementedError(msg)
    return channels


def _introspect_num_bands(fpath):
    try:
        shape = kwimage.load_image_shape(fpath)
    except Exception:
        from osgeo import gdal
        try:
            gdalfile = gdal.Open(fpath)
            shape = (gdalfile.RasterYSize, gdalfile.RasterXSize, gdalfile.RasterCount)
        except Exception:
            print('failed to introspect shape of fpath = {!r}'.format(fpath))
            return None
    if len(shape) == 1:
        return 1
    elif len(shape) == 3:
        return shape[2]
    else:
        raise Exception(f'unknown format, fpath={fpath}, shape={shape}')


def _num_band_hueristic(num_bands):
    if num_bands == 1:
        channels = 'gray'
    elif num_bands == 3:
        channels = 'r|g|b'
    elif num_bands == 4:
        channels = 'r|g|b|a'
    else:
        raise Exception(f'num_bands={num_bands}')
    return channels


def __WIP_add_auxiliary(coco_dset, gid, fname, channels, data, warp_aux_to_img=None):
    """
    Snippet for adding an auxiliary image

    Args:
        coco_dset (CocoDataset)
        gid (int): image id to add auxiliary data to
        channels (str): name of the new auxiliary channels
        fname (str): path to save the new auxiliary channels (absolute or
            relative to coco_dset.bundle_dpath)
        data (ndarray): actual auxiliary data
        warp_aux_to_img (kwimage.Affine): spatial relationship between
            auxiliary channel and the base image. If unspecified
            it is assumed that a simple scaling will suffice.

    NOTE:
        See CocoImage.add_auxiliary_item for a maintained implementation

    Ignore:
        import kwcoco
        coco_dset = kwcoco.CocoDataset.demo('shapes8')
        gid = 1
        data = np.random.rand(32, 55, 5)
        fname = 'myaux1.png'
        channels = 'hidden_logits'
        warp_aux_to_img = None
        __WIP_add_auxiliary(coco_dset, gid, fname, channels, data, warp_aux_to_img)
    """
    from os.path import join
    import kwimage
    fpath = join(coco_dset.bundle_dpath, fname)
    aux_height, aux_width = data.shape[0:2]
    img = coco_dset.index.imgs[gid]

    if warp_aux_to_img is None:
        # Assume we can just scale up the auxiliary data to match the image
        # space unless the user says otherwise
        warp_aux_to_img = kwimage.Affine.scale((
            img['width'] / aux_width, img['height'] / aux_height))

    # Make the aux info dict
    aux = {
        'file_name': fname,
        'height': aux_height,
        'width': aux_width,
        'channels': channels,
        'warp_aux_to_img': warp_aux_to_img.concise(),
    }

    if 0:
        # This function probably should not save the data to disk
        kwimage.imwrite(fpath, data)

    auxiliary = img.setdefault('auxiliary', [])
    auxiliary.append(aux)
    coco_dset._invalidate_hashid()


def _recompute_auxiliary_transforms(img):
    """
    Uses geotiff info to repopulate metadata
    """
    import kwimage
    auxiliary = img.get('auxiliary', [])
    idx = ub.argmax(auxiliary, lambda x: (x['width'] * x['height']))
    base = auxiliary[idx]
    warp_img_to_wld = kwimage.Affine.coerce(base['warp_to_wld'])
    warp_wld_to_img = warp_img_to_wld.inv()
    img.update(ub.dict_isect(base, {
        'utm_corners', 'wld_crs_info', 'utm_crs_info',
        'width', 'height', 'wgs84_to_wld',
        'wld_to_pxl',
    }))
    for aux in auxiliary:
        warp_aux_to_wld = kwimage.Affine.coerce(aux['warp_to_wld'])
        warp_aux_to_img = warp_wld_to_img @ warp_aux_to_wld
        aux['warp_aux_to_img'] = warp_aux_to_img.concise()


def coco_channel_stats(coco_dset):
    """
    Return information about what channels are available in the dataset

    Example:
        >>> from watch.utils import kwcoco_extensions
        >>> import kwcoco
        >>> import ubelt as ub
        >>> import watch
        >>> coco_dset = watch.coerce_kwcoco('vidshapes-watch')
        >>> info = kwcoco_extensions.coco_channel_stats(coco_dset)
        >>> print(ub.repr2(info, nl=3))
    """
    sensor_hist = ub.ddict(lambda: 0)
    chan_hist = ub.ddict(lambda: 0)
    sensorchan_hist = ub.ddict(lambda: ub.ddict(lambda: 0))

    for _gid, img in coco_dset.index.imgs.items():
        channels = []
        for obj in CocoImage(img).iter_asset_objs():
            channels.append(obj.get('channels', 'unknown-chan'))
        chan = '|'.join(channels)
        sensor = img.get('sensor_coarse', '')
        chan_hist[chan] += 1
        sensor_hist[sensor] += 1
        sensorchan_hist[sensor][chan] += 1

    from kwcoco.channel_spec import FusedChannelSpec as FS
    osets = [FS.coerce(c).as_oset() for c in chan_hist]
    common_channels = FS.coerce(list(ub.oset.intersection(*osets))).concise()
    all_channels = FS.coerce(list(ub.oset.union(*osets))).concise()

    info = {
        'chan_hist': chan_hist,
        'sensor_hist': sensor_hist,
        'sensorchan_hist': sensorchan_hist,
        'common_channels': common_channels,
        'all_channels': all_channels,
    }
    return info


class TrackidGenerator(ub.NiceRepr):
    """
    Keep track of which trackids have been used and generate new ones on demand

    TODO merge this into kwcoco as something like CocoDataset.next_trackid()?
    Or expose whatever mechanism is already generating new aids, gids, etc
    """

    def update_generator(self):
        self.used_trackids.update(self.dset.index.trackid_to_aids.keys())
        new_generator = filter(lambda x: x not in self.used_trackids,
                               itertools.count(start=next(self.generator)))
        self.generator = new_generator

    def exclude_trackids(self, trackids):
        if self.used_trackids.intersection(trackids):
            print(f'warning: CocoDataset {self.dset.tag} with trackids '
                  f'{self.used_trackids} already has trackids in {trackids}')
        self.used_trackids.update(trackids)

    def __init__(self, coco_dset):
        self.dset = coco_dset
        self.used_trackids = set()
        self.generator = itertools.count(start=1)
        self.update_generator()

    def __next__(self):
        return next(self.generator)


@profile
def warp_annot_segmentations_to_geos(coco_dset):
    """
    Warps annotation segmentations in image pixel space into geos-space
    """
    import watch
    # hack in segmentation_geos
    for gid in coco_dset.images():
        coco_img = coco_dset._coco_image(gid)
        asset = coco_img.primary_asset()
        fpath = join(coco_dset.bundle_dpath, asset['file_name'])
        geo_meta = watch.gis.geotiff.geotiff_metadata(fpath)
        warp_wld_from_aux = geo_meta['pxl_to_wld']
        warp_img_from_aux = kwimage.Affine.coerce(asset.get('warp_aux_to_img', None))

        warp_wgs84_from_wld = geo_meta['wld_to_wgs84']  # Could be a general CoordinateTransform!
        wgs84_crs_info = geo_meta['wgs84_crs_info']
        wgs84_axis_mapping = wgs84_crs_info['axis_mapping']
        assert wgs84_crs_info['auth'] == ('EPSG', '4326')

        warp_aux_from_img = warp_img_from_aux.inv()
        warp_wld_from_img = warp_wld_from_aux @ warp_aux_from_img
        for aid in coco_dset.annots(gid=gid):
            ann = coco_dset.index.anns[aid]
            sseg_img = kwimage.Segmentation.coerce(ann['segmentation'])
            sseg_wld = sseg_img.warp(warp_wld_from_img)
            sseg_wgs84 = sseg_wld.warp(warp_wgs84_from_wld)
            if wgs84_axis_mapping == 'OAMS_AUTHORITY_COMPLIANT':
                sseg_wgs84_lonlat = sseg_wgs84.swap_axes()
            elif wgs84_axis_mapping == 'OAMS_TRADITIONAL_GIS_ORDER':
                sseg_wgs84_lonlat = sseg_wgs84.copy()
            else:
                raise NotImplementedError(wgs84_axis_mapping)
            ann['segmentation_geos'] = sseg_wgs84_lonlat.to_geojson()
            geos_crs_info = {
                'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
                'auth': ('EPSG', '4326')
            }
            ann['segmentation_geos']['properties'] = {
                'crs_info': geos_crs_info
            }


def warp_annot_segmentations_from_geos(coco_dset):
    # Warp segmentation from geos
    import watch
    from os.path import join
    for gid in coco_dset.images():
        coco_img = coco_dset._coco_image(gid)
        asset = coco_img.primary_asset(requires=['geos_corners'])
        fpath = join(coco_dset.bundle_dpath, asset['file_name'])
        geo_meta = watch.gis.geotiff.geotiff_metadata(fpath)
        warp_wld_from_aux = kwimage.Affine.coerce(geo_meta['pxl_to_wld'])
        warp_img_from_aux = kwimage.Affine.coerce(asset.get('warp_aux_to_img', None))

        warp_wld_from_wgs84 = geo_meta['wgs84_to_wld']  # Could be a general CoordinateTransform!
        # warp_wgs84_from_wld = geo_meta['wld_to_wgs84']  # Could be a general CoordinateTransform!
        wgs84_crs_info = geo_meta['wgs84_crs_info']
        wgs84_axis_mapping = wgs84_crs_info['axis_mapping']
        assert wgs84_crs_info['auth'] == ('EPSG', '4326')

        warp_aux_from_wld = warp_wld_from_aux.inv()
        warp_img_from_wld = warp_img_from_aux @ warp_aux_from_wld

        for aid in coco_dset.annots(gid=gid):
            ann = coco_dset.index.anns[aid]
            sseg_geos = kwimage.MultiPolygon.from_geojson(ann['segmentation_geos'])
            # TODO: check crs properties (probably always crs84)
            ann['segmentation_geos']
            if wgs84_axis_mapping == 'OAMS_AUTHORITY_COMPLIANT':
                sseg_wgs84 = sseg_geos.swap_axes()
            elif wgs84_axis_mapping == 'OAMS_TRADITIONAL_GIS_ORDER':
                sseg_wgs84 = sseg_geos
            else:
                raise NotImplementedError(wgs84_axis_mapping)
            sseg_wld = sseg_wgs84.warp(warp_wld_from_wgs84)
            sseg_img = sseg_wld.warp(warp_img_from_wld)
            ann['segmentation'] = sseg_img.to_coco(style='new')
            ann['bbox'] = list(sseg_img.bounding_box().quantize().to_coco())[0]


# def coco_geopandas_images(coco_dset):
#     """
#     TODO:
#         - [ ] This is unused in this file and thus should move to the dev
#         folder or somewhere else for to keep useful scratch work.
#     """
#     import geopandas as gpd
#     df_input = []
#     for gid, img in coco_dset.imgs.items():
#         info  = img['geotiff_metadata']
#         kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
#         sh_img_poly = kw_img_poly.to_shapely()
#         df_input.append({
#             'gid': gid,
#             'name': img.get('name', None),
#             'video_id': img.get('video_id', None),
#             'bounds': sh_img_poly,
#         })
#     img_geos_df = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')
#     return img_geos_df


def visualize_rois(coco_dset, zoom=None):
    """
    Matplotlib visualization of image and annotation regions on a world map

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> coco_dset = demo_kwcoco_with_heatmaps(num_videos=1)
        >>> coco_populate_geo_heuristics(coco_dset, overwrite=True)
        >>> visualize_rois(coco_dset, zoom=0)

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> import kwcoco
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> coco_populate_geo_heuristics(coco_dset, overwrite=True, workers=4)
        >>> visualize_rois(coco_dset)
    """
    import geopandas as gpd
    cov_image_gdf = covered_image_geo_regions(coco_dset)
    annot_gdf = covered_annot_geo_regions(coco_dset)

    import kwplot
    kwplot.autompl()

    wld_map_gdf = gpd.read_file(
        gpd.datasets.get_path('naturalearth_lowres')
    )
    ax = wld_map_gdf.plot()

    def safe_centroids(gdf):
        return gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)

    cov_centroids = safe_centroids(cov_image_gdf)
    cov_image_gdf.plot(ax=ax, facecolor='none', edgecolor='green', alpha=0.5)
    cov_centroids.plot(ax=ax, marker='o', facecolor='green', alpha=0.5)
    # img_centroids = img_poly_gdf.geometry.centroid
    # img_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
    # img_centroids.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)

    annot_centroids = safe_centroids(annot_gdf)
    annot_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)
    annot_centroids.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)

    if zoom is not None:
        sh_zoom_roi = cov_image_gdf.geometry.iloc[0]
        kw_zoom_roi = kwimage.Polygon.from_shapely(sh_zoom_roi)
        bb = kw_zoom_roi.bounding_box()
        min_x, min_y, max_x, max_y = bb.scale(1.5, about='center').to_ltrb().data[0]
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)


def covered_image_geo_regions(coco_dset, merge=False):
    """
    Find the intersection of all image bounding boxes in world space
    to see what spatial regions are covered by the imagery.

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> coco_dset = demo_kwcoco_with_heatmaps(num_frames=1, num_videos=1)
        >>> coco_populate_geo_heuristics(coco_dset, overwrite=True)
        >>> img = coco_dset.imgs[1]
        >>> cov_image_gdf = covered_image_geo_regions(coco_dset)
    """
    import geopandas as gpd
    from shapely import ops
    import shapely
    # import watch
    rows = []
    for gid, img in coco_dset.imgs.items():
        if 'geos_corners' in img:
            geos_corners = img['geos_corners']
        else:
            coco_img = coco_dset.coco_image(img['id'])
            asset = coco_img.primary_asset(requires=['geos_corners'])
            geos_corners = asset['geos_corners']
        geos_crs_info = geos_corners.get('properties').get('crs_info', None)
        if geos_crs_info is not None:
            assert geos_crs_info['axis_mapping'] == 'OAMS_TRADITIONAL_GIS_ORDER'
            assert list(geos_crs_info['auth']) == ['EPSG', '4326']
        sh_img_poly = shapely.geometry.shape(geos_corners)
        rows.append({
            'geometry': sh_img_poly,
            'date_captured': img.get('date_captured', None),
            'name': img.get('name', None),
            'height': img.get('height', None),
            'width': img.get('width', None),
            'video_id': img.get('video_id', None),
            'image_id': gid,
            'frame_index': img.get('frame_index', None),
        })

    cov_poly_crs = 'crs84'
    if merge:
        # df_input = [
        #     {'gid': gid, 'bounds': poly, 'name': coco_dset.imgs[gid].get('name', None),
        #      'video_id': coco_dset.imgs[gid].get('video_id', None) }
        #     for gid, poly in gid_to_poly.items()
        # ]
        # img_geos = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')

        # Can merge like this, but we lose membership info
        # coverage_df = gpd.GeoDataFrame(img_geos.unary_union)
        coverage_rois_ = ops.unary_union([row['geometry'] for row in rows])
        if hasattr(coverage_rois_, 'geoms'):
            # Iteration over shapely objects was deprecated, test for geoms
            # attribute instead.
            coverage_rois = list(coverage_rois_.geoms)
        else:
            coverage_rois = [coverage_rois_]
        # geopandas uses traditional crs mappings
        cov_image_gdf = gpd.GeoDataFrame(
            {'geometry': coverage_rois},
            geometry='geometry', crs=cov_poly_crs)
    else:
        cov_image_gdf = gpd.GeoDataFrame(rows, geometry='geometry',
                                         crs=cov_poly_crs)

    return cov_image_gdf


def covered_annot_geo_regions(coco_dset, merge=False):
    """
    Given a dataset find spatial regions of interest that contain annotations
    """
    import shapely
    import geopandas as gpd
    from shapely import ops
    aid_to_poly = {}
    for aid, ann in coco_dset.anns.items():
        ann_goes = ann['segmentation_geos']
        # TODO: assert the segmentation_geos CRS is (geojson - WGS84-traditional)
        if ann_goes is not None:
            sh_poly = shapely.geometry.shape(ann_goes)
            aid_to_poly[aid] = sh_poly

    # annot_crs = 'epsg:4326'
    annot_crs = 'crs84'
    if merge:
        gid_to_rois = {}
        for gid, aids in coco_dset.index.gid_to_aids.items():
            if len(aids):
                sh_annot_polys = ub.dict_subset(aid_to_poly, aids)
                sh_annot_polys_ = [p.buffer(0) for p in sh_annot_polys.values()]
                sh_annot_polys_ = [p.buffer(0.000001) for p in sh_annot_polys_]

                # What CRS should we be doing this in? Is WGS84 OK?
                # Should we switch to UTM?
                img_rois_ = ops.unary_union(sh_annot_polys_)
                try:
                    img_rois = list(img_rois_.geoms)
                except Exception:
                    img_rois = [img_rois_]

                kw_img_rois = [
                    kwimage.Polygon.from_shapely(p.convex_hull).bounding_box().to_polygons()[0]
                    for p in img_rois]
                sh_img_rois = [p.to_shapely() for p in kw_img_rois]
                gid_to_rois[gid] = sh_img_rois

        # TODO: if there are only midly overlapping regions, we should likely split
        # them up. We can also group by UTM coordinates to reduce computation.
        sh_rois_ = ops.unary_union([
            p.buffer(0) for rois in gid_to_rois.values()
            for p in rois
        ])
        try:
            sh_rois = list(sh_rois_.geoms)
        except Exception:
            sh_rois = [sh_rois_]
        # geopandas uses traditional crs mappings
        cov_annot_gdf = gpd.GeoDataFrame(
            {'geometry': sh_rois},
            geometry='geometry', crs=annot_crs)
    else:
        sh_polys = list(aid_to_poly.values())
        aids = list(aid_to_poly.keys())
        cov_annot_gdf = gpd.GeoDataFrame(
            {'geometry': sh_polys, 'aids': aids},
            geometry='geometry', crs=annot_crs)
    return cov_annot_gdf


def flip_xy(poly):
    """
    TODO:
        - [ ] This is unused in this file and thus should move to the dev
        folder or somewhere else for to keep useful scratch work.
    """
    if hasattr(poly, 'reorder_axes'):
        new_poly = poly.reorder_axes((1, 0))
    else:
        kw_poly = kwimage.Polygon.from_shapely(poly)
        kw_poly.data['exterior'].data = kw_poly.data['exterior'].data[:, ::-1]
        sh_poly_ = kw_poly.to_shapely()
        new_poly = sh_poly_
    return new_poly


def category_category_colors(coco_dset):
    """
    Ensures that each category in a CategoryTree has a color

    TODO:
        - [ ] Add to CategoryTree
    """
    cats = coco_dset.dataset['categories']
    # backup_colors = iter(kwimage.Color.distinct(len(cats)))
    for cat in cats:
        color = cat.get('color', None)
        if color is None:
            # color = next(backup_colors)
            # cat['color'] = kwimage.Color(color).as01()
            color = kwimage.Color.random()
            cat['color'] = color.as01()
