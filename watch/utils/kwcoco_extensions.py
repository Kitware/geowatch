"""
Adds fields needed by ndsampler to correctly "watch" a region.

Some of this is done hueristically. We assume images come from certain sensors.
We assume input is orthorectified.  We assume some GSD "target" gsd for video
and image processing. Note a video GSD will typically be much higher (i.e.
lower resolution) than an image GSD.
"""
# import kwcoco
import warnings
import ubelt as ub
import kwimage
import itertools

import numpy as np
from os.path import join
import numbers
from kwimage.transform import Affine

# Was originally defined in this file, moved to kwcoco proper
from kwcoco.coco_image import CocoImage

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def populate_watch_fields(coco_dset, target_gsd=10.0, vidids=None, overwrite=False, default_gsd=None, conform=True):
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
        vidids = coco_dset.index.videos.keys()
    gids = list(ub.flatten(coco_dset.images(vidid=vidid) for vidid in vidids))

    for gid in ub.ProgIter(gids, total=len(gids), desc='populate imgs'):
        coco_populate_geo_img_heuristics(coco_dset, gid, overwrite=overwrite,
                                         default_gsd=default_gsd)

    for vidid in ub.ProgIter(vidids, total=len(vidids), desc='populate videos'):
        coco_populate_geo_video_stats(coco_dset, vidid, target_gsd=target_gsd)

    # serialize intermediate objects
    coco_dset._ensure_json_serializable()


def coco_populate_geo_heuristics(coco_dset, overwrite=False, default_gsd=None, workers=0):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> import kwcoco
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> coco_populate_geo_heuristics(coco_dset, overwrite=True, workers=4)
    """
    executor = ub.JobPool('thread', max_workers=workers)
    for gid in ub.ProgIter(coco_dset.index.imgs.keys(), total=len(coco_dset.index.imgs), desc='populate imgs'):
        executor.submit(coco_populate_geo_img_heuristics, coco_dset, gid,
                        overwrite=overwrite, default_gsd=default_gsd)
    for job in ub.ProgIter(executor.as_completed(), total=len(executor), desc='populate imgs'):
        job.result()


def coco_populate_geo_img_heuristics(coco_dset, gid, overwrite=False,
                                     default_gsd=None, **kw):
    """
    Note: this will not overwrite existing channel info unless specified

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> coco_dset = demo_kwcoco_with_heatmaps()
        >>> gid = 1
        >>> overwrite = {'warp', 'band'}
        >>> default_gsd = None
        >>> kw = {}
        >>> coco_populate_geo_img_heuristics(coco_dset, gid)

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> ###
        >>> gid = 1
        >>> dset1 = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> coco_populate_geo_img_heuristics(dset1, gid, overwrite=True)
        >>> ###
        >>> gid = 1
        >>> dset2 = kwcoco.CocoDataset.demo('shapes8')
        >>> coco_populate_geo_img_heuristics(dset2, gid, overwrite=True)
    """
    from shapely import ops
    import shapely
    bundle_dpath = coco_dset.bundle_dpath
    img = coco_dset.imgs[gid]
    coco_img = coco_dset._coco_image(gid)
    asset_objs = list(coco_img.iter_asset_objs())

    geos_crs_info = {
        'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
        'auth': ('EPSG', '4326')
    }

    # Note: for non-geotiffs we could use the transformation provided with them
    # to determine their geo-properties.
    asset_errors = []
    geos_corners = []
    for obj in asset_objs:

        errors = _populate_canvas_obj(bundle_dpath, obj, overwrite=overwrite,
                                      default_gsd=default_gsd)

        corners_geojson = obj.pop('geos_corners', None)
        if corners_geojson is not None:
            assert corners_geojson['properties']['crs_info'] == geos_crs_info
            obj_geos = shapely.geometry.shape(corners_geojson)
            geos_corners.append(obj_geos)
        asset_errors.append(errors)

    if all(asset_errors):
        info = ub.dict_isect(img, {'name', 'file_name', 'id'})
        warnings.warn(f'img {info} has issues introspecting')

    if not geos_corners:
        print('None of the assets had geo information')
    else:
        combo = ops.cascaded_union(geos_corners)
        geos_corners_img = kwimage.Polygon.coerce(combo.convex_hull).to_multi_polygon().to_geojson()
        geos_corners_img['properties'] = {'crs_info': geos_crs_info}
        img['geos_corners'] = geos_corners_img
        img['geos_crs_info'] = geos_crs_info


@profile
def _populate_canvas_obj(bundle_dpath, obj, overwrite=False, with_wgs=False,
                         default_gsd=None):
    """
    obj can be an img or aux
    """
    import watch
    import kwcoco
    sensor_coarse = obj.get('sensor_coarse', None)
    num_bands = obj.get('num_bands', None)
    channels = obj.get('channels', None)
    fname = obj.get('file_name', None)
    warp_to_wld = obj.get('warp_to_wld', None)
    approx_meter_gsd = obj.get('approx_meter_gsd', None)

    valid_overwrites = {'warp', 'band', 'channels'}
    default_overwrites = {'warp', 'band'}
    if overwrite is True:
        overwrite = default_overwrites
    elif overwrite is False:
        overwrite = {}
    else:
        overwrite = set(overwrite)
        unexpected = overwrite - valid_overwrites
        if unexpected:
            raise ValueError(f'Got unexpected overwrites: {unexpected}')
    errors = []
    # Can only do this for images with file names
    if fname is not None:
        fpath = join(bundle_dpath, fname)

        if 'warp' in overwrite or warp_to_wld is None or approx_meter_gsd is None:
            try:
                info = watch.gis.geotiff.geotiff_metadata(fpath)
                height, width = info['img_shape'][0:2]

                obj['height'] = height
                obj['width'] = width
                # print('info = {!r}'.format(info))

                # WE NEED TO ACCOUNT FOR WLD_CRS TO USE THIS
                # obj_to_wld = Affine.coerce(info['pxl_to_wld'])

                # FIXME: FOR NOW JUST USE THIS BIG HACK
                xy1_man = info['pxl_corners'].data.astype(np.float64)
                xy2_man = info['utm_corners'].data.astype(np.float64)
                hack_aff = fit_affine_matrix(xy1_man, xy2_man)
                hack_aff = Affine.coerce(hack_aff)

                # crs_info['utm_corners'].warp(np.asarray(hack_aff.inv()))
                # crs_info['pxl_corners'].warp(np.asarray(hack_aff))

                obj_to_wld = Affine.coerce(hack_aff)
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

                if with_wgs:
                    obj.update({
                        'wgs84_to_wld': info['wgs84_to_wld'],
                        'wld_to_pxl': info['wld_to_pxl'],
                    })

                approx_meter_gsd = info['approx_meter_gsd']
            except Exception:
                if default_gsd is not None:
                    obj['approx_meter_gsd'] = default_gsd
                    obj['warp_to_wld'] = Affine.eye().__json__()
                else:
                    errors.append('no_crs_info')
            else:
                obj['approx_meter_gsd'] = approx_meter_gsd
                obj['warp_to_wld'] = Affine.coerce(obj_to_wld).__json__()

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
        return errors


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

            wld_from_aux = Affine.coerce(aux_chosen.get('warp_to_wld', None))
            img_from_aux = Affine.coerce(aux_chosen['warp_aux_to_img'])
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

        wld_from_img = Affine.coerce(wld_from_img)

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

    # Can add an extra transform here if the video is not exactly in
    # any specific image space
    baseimg_from_wld = base_info['img_to_wld'].inv()
    vid_from_wld = Affine.scale(scale) @ baseimg_from_wld
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
                '''
                Video alignment is warping images with different World
                Coordinate Reference Systems, but still treating them as the
                same. FIXME
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
        for asset_idx, obj in enumerate(asset_objs):
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


def ensure_transfered_geo_data(coco_dset):
    for gid in ub.ProgIter(list(coco_dset.images())):
        transfer_geo_metadata(coco_dset, gid)


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
        >>> transfer_geo_metadata(coco_dset, gid)
        >>> gid = 2
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
                                    info = watch.gis.geotiff.geotiff_metadata(fpath)
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

        # georef_crs_info['axis_mapping']
        # osr.OAMS_AUTHORITY_COMPLIANT
        # aux_wld_crs = osr.SpatialReference()
        # aux_wld_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
        # aux_wld_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        for asset_idx, obj in assets_without_geo_info.items():
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
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> info = kwcoco_extensions.coco_channel_stats(coco_dset)
        >>> print(ub.repr2(info, nl=1))
    """
    channel_col = []
    for gid, img in coco_dset.index.imgs.items():
        channels = []
        for obj in CocoImage(img).iter_asset_objs():
            channels.append(obj.get('channels', 'unknown-chan'))
        channel_col.append('|'.join(channels))

    chan_hist = ub.dict_hist(channel_col)

    from kwcoco.channel_spec import FusedChannelSpec as FS
    osets = [FS.coerce(c).as_oset() for c in chan_hist]
    common_channels = FS(list(ub.oset.intersection(*osets)))
    all_channels = FS(list(ub.oset.union(*osets)))

    info = {
        'chan_hist': chan_hist,
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
        used_trackids = self.dset.index.trackid_to_aids.keys()
        new_generator = filter(lambda x: x not in used_trackids,
                               itertools.count(start=next(self.generator)))
        self.generator = new_generator

    def __init__(self, coco_dset):
        self.dset = coco_dset
        self.generator = itertools.count(start=1)
        self.update_generator()

    def __next__(self):
        return next(self.generator)


def single_geotiff_metadata(bundle_dpath, img, serializable=False):
    import watch
    from os.path import exists
    import dateutil
    geotiff_metadata = None
    aux_metadata = []

    img['datetime_acquisition'] = (
        dateutil.parser.parse(img['date_captured'])
    )

    # if an image specified its "dem_hint" as ignore, then we set the
    # elevation to 0. NOTE: this convention might be generalized and
    # replaced in the future. I.e. in the future the dem_hint might simply
    # specify the constant elevation to use, or perhaps something else.
    dem_hint = img.get('dem_hint', 'use')
    metakw = {}
    if dem_hint == 'ignore':
        metakw['elevation'] = 0

    # only need rpc info, wgs84_corners, and and warps
    keys_of_interest = {
        'rpc_transform',
        'is_rpc',
        'wgs84_to_wld',
        'wgs84_corners',
        'wld_to_pxl',
    }

    HACK_METADATA = 0
    if HACK_METADATA:
        # HACK: See if we can construct the keys from the metadata
        # in the coco file instead of reading the geotiff
        hack_keys = {
            'utm_corners',
            'warp_img_to_wld',
            'utm_crs_info',
            'wld_crs_info',
        }
        have_hacks = ub.dict_isect(img, hack_keys)
        if len(have_hacks) == len(hack_keys):
            print('have hacks: {}'.format(img['sensor_coarse']))
            from osgeo import osr

            def _make_osgeo_crs(crs_info):
                from osgeo import osr
                axis_mapping_int = getattr(osr, crs_info['axis_mapping'])
                auth = crs_info['auth']
                assert len(auth) == 2
                assert auth[0] == 'EPSG'
                crs = osr.SpatialReference()
                crs.ImportFromEPSG(int(auth[1]))
                crs.SetAxisMappingStrategy(axis_mapping_int)
                return crs

            wgs84_crs = osr.SpatialReference()
            wgs84_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs

            wld_to_pxl = kwimage.Affine.coerce(img['warp_img_to_wld']).inv()
            utm_crs = _make_osgeo_crs(have_hacks['utm_crs_info'])
            wld_crs = _make_osgeo_crs(have_hacks['wld_crs_info'])
            utm_to_wgs84 = osr.CoordinateTransformation(utm_crs, wgs84_crs)
            wgs84_to_wld = osr.CoordinateTransformation(wgs84_crs, wld_crs)
            utm_corners = kwimage.Coords(np.array(have_hacks['utm_corners']))
            wgs84_corners = utm_corners.warp(utm_to_wgs84)

            hack_info = {
                'rpc_transform': None,
                'is_rpc': False,
                'wgs84_to_wld': wgs84_to_wld,
                'wgs84_corners': wgs84_corners,
                'wld_to_pxl': wld_to_pxl,
            }
            geotiff_metadata = hack_info
            return geotiff_metadata
        else:
            print('missing hacks: {}'.format(img['sensor_coarse']))

    fname = img.get('file_name', None)
    if fname is not None:
        src_gpath = join(bundle_dpath, fname)
        assert exists(src_gpath)
        img_info = watch.gis.geotiff.geotiff_metadata(src_gpath, **metakw)

        if serializable:
            raise NotImplementedError
        else:
            img_info = ub.dict_isect(img_info, keys_of_interest)
            geotiff_metadata = img_info

    for aux in img.get('auxiliary', []):
        aux_fpath = join(bundle_dpath, aux['file_name'])
        assert exists(aux_fpath)
        aux_info = watch.gis.geotiff.geotiff_metadata(aux_fpath, **metakw)
        aux_info = ub.dict_isect(aux_info, keys_of_interest)
        if serializable:
            raise NotImplementedError
        else:
            aux_metadata.append(aux_info)
            aux['geotiff_metadata'] = aux_info

    if fname is None:
        # need to choose one of the auxiliary images as the "main" image.
        # We are assuming that there is one auxiliary image that exactly
        # corresponds.
        candidates = []
        for aux in img.get('auxiliary', []):
            if aux['width'] == img['width'] and aux['height'] == img['height']:
                candidates.append(aux)

        if not candidates:
            raise AssertionError(
                'Assumed at least one auxiliary image has identity '
                'transform, but this seems to not be the case')
        aux = ub.peek(candidates)
        geotiff_metadata = aux['geotiff_metadata']

    img['geotiff_metadata'] = geotiff_metadata
    return geotiff_metadata, aux_metadata


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
    import shapely
    cov_poly_gdf = find_covered_regions(coco_dset)

    sseg_geos_list = coco_dset.annots().lookup('segmentation_geos', None)
    sseg_geos_list = [s for s in sseg_geos_list if s is not None]
    # Dedup
    sseg_geos_list = list(ub.unique(sseg_geos_list, key=ub.hash_data))

    sh_all_box_rois_trad = [shapely.geometry.shape(d) for d in sseg_geos_list]
    roi_poly_crs = 'epsg:4326'
    roi_poly_gdf = gpd.GeoDataFrame({'roi_polys': sh_all_box_rois_trad},
                                    geometry='roi_polys', crs=roi_poly_crs)

    if True:
        import kwplot
        kwplot.autompl()

        wld_map_gdf = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )
        ax = wld_map_gdf.plot()

        cov_centroids = cov_poly_gdf.geometry.centroid
        cov_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='green', alpha=0.5)
        cov_centroids.plot(ax=ax, marker='o', facecolor='green', alpha=0.5)
        # img_centroids = img_poly_gdf.geometry.centroid
        # img_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
        # img_centroids.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)

        roi_centroids = roi_poly_gdf.geometry.centroid
        roi_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)
        roi_centroids.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)

        if zoom is not None:
            sh_zoom_roi = sh_coverage_rois_trad[zoom]
            kw_zoom_roi = kwimage.Polygon.from_shapely(sh_zoom_roi)
            bb = kw_zoom_roi.bounding_box()
            min_x, min_y, max_x, max_y = bb.scale(1.5, about='center').to_ltrb().data[0]
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)


def find_covered_regions(coco_dset):
    """
    Find the intersection of all image bounding boxes in world space
    to see what spatial regions are covered by the imagery.

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> coco_dset = demo_kwcoco_with_heatmaps(num_frames=1, num_videos=1)
        >>> coco_populate_geo_heuristics(coco_dset, overwrite=True)
        >>> img = coco_dset.imgs[1]
        >>> cov_poly_gdf = find_covered_regions(coco_dset)
    """
    import geopandas as gpd
    from shapely import ops
    import shapely
    # import watch
    gid_to_poly = {}
    for gid, img in coco_dset.imgs.items():
        geos_crs_info = img['geos_crs_info']
        assert geos_crs_info['axis_mapping'] == 'OAMS_TRADITIONAL_GIS_ORDER'
        assert geos_crs_info['auth'] == ('EPSG', '4326')
        sh_img_poly = shapely.geometry.shape(img['geos_corners'])
        gid_to_poly[gid] = sh_img_poly

    # df_input = [
    #     {'gid': gid, 'bounds': poly, 'name': coco_dset.imgs[gid].get('name', None),
    #      'video_id': coco_dset.imgs[gid].get('video_id', None) }
    #     for gid, poly in gid_to_poly.items()
    # ]
    # img_geos = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')

    # Can merge like this, but we lose membership info
    # coverage_df = gpd.GeoDataFrame(img_geos.unary_union)
    coverage_rois_ = ops.unary_union(gid_to_poly.values())
    if hasattr(coverage_rois_, 'geoms'):
        # Iteration over shapely objects was deprecated, test for geoms
        # attribute instead.
        coverage_rois = list(coverage_rois_.geoms)
    else:
        coverage_rois = [coverage_rois_]

    # geopandas uses traditional crs mappings
    cov_poly_crs = 'epsg:4326'
    cov_poly_gdf = gpd.GeoDataFrame({'cov_rois': coverage_rois},
                                    geometry='cov_rois', crs=cov_poly_crs)
    return cov_poly_gdf


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
