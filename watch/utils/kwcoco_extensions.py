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

import numpy as np
from os.path import join

from kwimage.transform import Affine

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def populate_watch_fields(dset, target_gsd=10.0, overwrite=False):
    """
    Aggregate populate function for fields useful to WATCH.

    Args:
        dset (Dataset): dataset to work with
        target_gsd (float): target gsd in meters
        overwrite (bool | List[str]): if True or False overwrites everything or
            nothing. Otherwise it can be a list of strings indicating what is
            overwritten. Valid keys are warp, band, and channels.

    Ignore:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> # root_dpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/extern/onera_2018')
        >>> # coco_fpath = join(root_dpath, 'onera_all.kwcoco.json')
        >>> fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        >>> dset = kwcoco.CocoDataset(fpath)
        >>> target_gsd = 5.0
        >>> populate_watch_fields(dset, target_gsd)
        >>> # dset.dump(dset.fpath, newlines=True)

        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=-1)))
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=1))

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> print('dset = {!r}'.format(dset))
        >>> target_gsd = 13.0
        >>> populate_watch_fields(dset, target_gsd)
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=2))
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=1)))

        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> print('dset = {!r}'.format(dset))
        >>> target_gsd = 13.0
        >>> populate_watch_fields(dset, target_gsd)
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=2))
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=1)))
    """
    # Load your KW-COCO dataset (conform populates information like image size)
    dset.conform(pycocotools_info=False)

    for gid in ub.ProgIter(dset.index.imgs.keys(), total=len(dset.index.imgs), desc='populate imgs'):
        coco_populate_geo_img_heuristics(dset, gid, overwrite=overwrite)

    for vidid in ub.ProgIter(dset.index.videos.keys(), total=len(dset.index.videos), desc='populate videos'):
        coco_populate_geo_video_stats(dset, vidid, target_gsd=target_gsd)

    # serialize intermediate objects
    dset._ensure_json_serializable()


def coco_populate_geo_video_stats(dset, vidid, target_gsd='max-resolution'):
    """
    Example:
        import kwcoco
        # TODO: make a demo dataset with some sort of gsd metadata
        dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        print('dset = {!r}'.format(dset))

        coco_fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        coco_fpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/combo_data.kwcoco.json'
        dset = kwcoco.CocoDataset(coco_fpath)
        vidid = 1

        target_gsd = 2.8
    """
    # Compute an image-to-video transform that aligns all frames to some
    # common resolution.
    video = dset.index.videos[vidid]
    gids = dset.index.vidid_to_gids[vidid]

    frame_infos = {}

    for gid in gids:
        img = dset.index.imgs[gid]
        coco_img = CocoImage(img)

        if img.get('file_name', None) is None:
            # Choose any one of the auxiliary images that has the required
            # attribute
            aux_chosen = coco_img.primary_asset(requires=[
                'warp_to_wld', 'approx_meter_gsd'])
            wld_from_aux = Affine.coerce(aux_chosen.get('warp_to_wld', None))
            img_from_aux = Affine.coerce(aux_chosen['warp_aux_to_img'])
            aux_from_img = img_from_aux.inv()
            wld_from_img = wld_from_aux @ aux_from_img
            approx_meter_gsd = aux_chosen['approx_meter_gsd']
        else:
            wld_from_img = Affine.coerce(img.get('warp_to_wld', None))
            approx_meter_gsd = img['approx_meter_gsd']

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
    import numbers
    if target_gsd == 'max-resolution':
        target_gsd_ = min_gsd
    elif target_gsd == 'min-resolution':
        target_gsd_ = max_gsd
    else:
        target_gsd_ = target_gsd
        if not isinstance(target_gsd, numbers.Number):
            raise TypeError('target_gsd must be a code or number = {}'.format(type(target_gsd)))
    target_gsd_ = float(target_gsd_)

    for info in frame_infos.values():
        info['target_gsd'] = target_gsd_
        info['to_target_scale_factor'] = info['approx_meter_gsd'] / target_gsd_

    available_channels = set()
    available_gsds = set()
    for gid in gids:
        img = dset.index.imgs[gid]
        for obj in coco_img.iter_asset_objs():
            available_channels.add(obj.get('channels', None))
            _gsd = obj.get('approx_meter_gsd')
            if _gsd is not None:
                available_gsds.add(round(_gsd, 1))

    # Align to frame closest to the target GSD
    base_gid, info = min(frame_infos.items(),
                         key=lambda kv: 1 - kv[1]['to_target_scale_factor'])
    scale = info['to_target_scale_factor']

    # Can add an extra transform here if the video is not exactly in
    # any specific image space
    wld_to_vid = Affine.scale(scale) @ info['img_to_wld'].inv()
    video['width'] = int(np.ceil(info['width'] * scale))
    video['height'] = int(np.ceil(info['height'] * scale))

    # Store metadata in the video
    video['num_frames'] = len(gids)
    video['warp_wld_to_vid'] = wld_to_vid.__json__()
    video['target_gsd'] = target_gsd_
    video['min_gsd'] = min_gsd
    video['max_gsd'] = max_gsd

    # Remove old cruft (can remove in future versions)
    video.pop('available_channels', None)

    for gid in gids:
        img = dset.index.imgs[gid]
        img_to_wld = frame_infos[gid]['img_to_wld']
        img_to_vid = wld_to_vid @ img_to_wld
        img['warp_img_to_vid'] = img_to_vid.concise()


def coco_populate_geo_img_heuristics(dset, gid, overwrite=False, **kw):
    """
    Note: this will not overwrite existing channel info unless specified

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
    bundle_dpath = dset.bundle_dpath
    img = dset.imgs[gid]

    asset_objs = list(CocoImage(img).iter_asset_objs())

    # Note: for non-geotiffs we could use the transformation provided with them
    # to determine their geo-properties.
    asset_errors = []
    for obj in asset_objs:
        errors = _populate_canvas_obj(bundle_dpath, obj, overwrite=overwrite)
        asset_errors.append(errors)

    if all(asset_errors):
        info = ub.dict_isect(img, {'name', 'file_name', 'id'})
        warnings.warn(f'img {info} has issues introspecting')


@profile
def _populate_canvas_obj(bundle_dpath, obj, overwrite=False, with_wgs=False):
    """
    obj can be an img or aux
    """
    sensor_coarse = obj.get('sensor_coarse', None)
    num_bands = obj.get('num_bands', None)
    channels = obj.get('channels', None)
    fname = obj.get('file_name', None)
    warp_to_wld = obj.get('warp_to_wld', None)
    approx_meter_gsd = obj.get('approx_meter_gsd', None)

    valid_overwrites = {'warp', 'band', 'channels'}
    if overwrite is True:
        overwrite = valid_overwrites
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
                import watch
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

                wld_crs_info = ub.dict_diff(info['wld_crs_info'], {'type'})
                utm_crs_info = ub.dict_diff(info['utm_crs_info'], {'type'})
                obj.update({
                    'utm_corners': info['utm_corners'].data.tolist(),
                    'wld_crs_info': wld_crs_info,
                    'utm_crs_info': utm_crs_info,
                })

                if with_wgs:
                    obj.update({
                        'wgs84_to_wld': info['wgs84_to_wld'],
                        # 'wld_to_pxl': info['wld_to_pxl'],
                    })

                approx_meter_gsd = info['approx_meter_gsd']
            except Exception:
                errors.append('no_crs_info')
                # obj_to_wld = Affine.eye()
                # approx_meter_gsd = 1.0
            else:
                obj['approx_meter_gsd'] = approx_meter_gsd
                obj['warp_to_wld'] = Affine.coerce(obj_to_wld).__json__()

        if 'band' in overwrite or num_bands is None:
            num_bands = _introspect_num_bands(fpath)
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
        USV = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    except MemoryError:
        import scipy.sparse as sps
        import scipy.sparse.linalg as spsl
        M_sparse = sps.lil_matrix(M)
        USV = spsl.svds(M_sparse)
    except np.linalg.LinAlgError:
        raise
    except Exception:
        raise

    U, s, Vt = USV

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


def __WIP_add_auxiliary(dset, gid, fname, channels, data, warp_aux_to_img=None):
    """
    Snippet for adding an auxiliary image

    Args:
        dset (CocoDataset)
        gid (int): image id to add auxiliary data to
        channels (str): name of the new auxiliary channels
        fname (str): path to save the new auxiliary channels (absolute or
            relative to dset.bundle_dpath)
        data (ndarray): actual auxiliary data
        warp_aux_to_img (kwimage.Affine): spatial relationship between
            auxiliary channel and the base image. If unspecified
            it is assumed that a simple scaling will suffice.

    Ignore:
        import kwcoco
        dset = kwcoco.CocoDataset.demo('shapes8')
        gid = 1
        data = np.random.rand(32, 55, 5)
        fname = 'myaux1.png'
        channels = 'hidden_logits'
        warp_aux_to_img = None
        __WIP_add_auxiliary(dset, gid, fname, channels, data, warp_aux_to_img)
    """
    from os.path import join
    import kwimage
    fpath = join(dset.bundle_dpath, fname)
    aux_height, aux_width = data.shape[0:2]
    img = dset.index.imgs[gid]

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
    dset._invalidate_hashid()


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
        # 'wld_to_pxl',
    }))
    for aux in auxiliary:
        warp_aux_to_wld = kwimage.Affine.coerce(aux['warp_to_wld'])
        warp_aux_to_img = warp_wld_to_img @ warp_aux_to_wld
        aux['warp_aux_to_img'] = warp_aux_to_img.concise()


def coco_channel_stats(dset):
    """
    Return information about what channels are available in the dataset

    Example:
        >>> from watch.utils import kwcoco_extensions
        >>> import kwcoco
        >>> import ubelt as ub
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> info = kwcoco_extensions.coco_channel_stats(dset)
        >>> print(ub.repr2(info, nl=1))
    """
    channel_col = []
    for gid, img in dset.index.imgs.items():
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


class CocoImage(ub.NiceRepr):
    """
    An object-oriented representation of a coco image.

    It provides helper methods that are specific to a single image.

    This operates directly on a single coco image dictionary, but it can
    optionally be connected to a parent dataset, which allows it to use
    CocoDataset methods to query about relationships and resolve pointers.

    This is different than the Images class in coco_object1d, which is just a
    vectorized interface to multiple objects.

    TODO:
        - [ ] This will eventually move to kwcoco itself

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> dset1 = kwcoco.CocoDataset.demo('shapes8')
        >>> dset2 = kwcoco.CocoDataset.demo('vidshapes8-multispectral')

        >>> self = CocoImage(dset1.imgs[1], dset1)
        >>> print('self = {!r}'.format(self))
        >>> print('self.channels = {}'.format(ub.repr2(self.channels, nl=1)))

        >>> self = CocoImage(dset2.imgs[1], dset2)
        >>> print('self.channels = {}'.format(ub.repr2(self.channels, nl=1)))
        >>> self.primary_asset()
    """

    def __init__(self, img, dset=None):
        self.img = img
        self.dset = dset

    def __nice__(self):
        """
        Example:
            >>> from watch.utils.kwcoco_extensions import *  # NOQA
            >>> with ub.CaptureStdout() as cap:
            ...     dset = kwcoco.CocoDataset.demo('shapes8')
            >>> self = CocoImage(dset.dataset['images'][0], dset)
            >>> print('self = {!r}'.format(self))

            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = CocoImage(dset.dataset['images'][0], dset)
            >>> print('self = {!r}'.format(self))
        """
        from watch.utils.slugify_ext import smart_truncate
        stats = self.stats()
        from functools import partial
        stats = ub.map_vals(
            partial(smart_truncate, max_length=32, trunc_loc=0.5),
            stats)
        return ub.repr2(stats, compact=1, nl=0, sort=0)

    def stats(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import os
            >>> from os.path import join
            >>> import ndsampler
            >>> import kwcoco
            >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
            >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
            >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/combo_data.kwcoco.json')
            >>> #
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> self = CocoImage(coco_dset.dataset['images'][0], coco_dset)
            >>> print('self = {!r}'.format(self))
            >>> stats = self.stats()
            >>> print('self = {!r}'.format(self))
            >>> print('self.stats() = {}'.format(ub.repr2(stats, nl=1)))
        """
        key_attrname = [
            ('wh', 'dsize'),
            ('n_chan', 'num_channels'),
            ('channels', 'channels'),
        ]
        stats = {}
        for key, attrname in key_attrname:
            try:
                stats[key] = str(getattr(self, attrname))
            except Exception as ex:
                stats[key] = repr(ex)
        return stats

    def __getitem__(self, key):
        return self.img[key]

    def keys(self):
        return self.img.keys()

    def get(self, key, default=ub.NoParam):
        """
        Duck type some of the dict interface
        """
        if default is ub.NoParam:
            return self.img.get(key)
        else:
            return self.img.get(key, default)

    @property
    def channels(self):
        from kwcoco.channel_spec import FusedChannelSpec
        from kwcoco.channel_spec import ChannelSpec
        img_parts = []
        for obj in self.iter_asset_objs():
            obj_parts = obj.get('channels', None)
            obj_chan = FusedChannelSpec.coerce(obj_parts).normalize()
            img_parts.append(obj_chan.spec)
        spec = ChannelSpec(','.join(img_parts))
        return spec

    @property
    def num_channels(self):
        return sum(map(len, self.channels.streams()))

    @property
    def dsize(self):
        width = self.img.get('width', None)
        height = self.img.get('height', None)
        return width, height

    def primary_asset(self, requires=[]):
        """
        Compute a "main" image asset.

        Args:
            requires (List[str]):
                list of attribute that must be non-None to consider an object
                as the primary one.

        TODO:
            - [ ] Add in primary heuristics
        """
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        candidates = []

        if has_base_image:
            obj = img
            if all(k in obj for k in requires):
                # Return the base image if we can
                return obj

        # Choose "best" auxiliary image based on a hueristic.
        eye = kwimage.Affine.eye().matrix
        for obj in img.get('auxiliary', []):
            # Take frobenius norm to get "distance" between transform and
            # the identity. We want to find the auxiliary closest to the
            # identity transform.
            warp_aux_to_img = kwimage.Affine.coerce(obj.get('warp_aux_to_img', None))
            fro_dist = np.linalg.norm(warp_aux_to_img.matrix - eye, ord='fro')

            if all(k in obj for k in requires):
                candidates.append({
                    'area': obj['width'] * obj['height'],
                    'fro_dist': fro_dist,
                    'obj': obj,
                })

        idx = ub.argmin(
            candidates, key=lambda val: (val['fro_dist'], -val['area'])
        )
        obj = candidates[idx]['obj']
        return obj

    def iter_asset_objs(self):
        """
        Iterate through base + auxiliary dicts that have file paths
        """
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        if has_base_image:
            obj = ub.dict_diff(img, {'auxiliary'})
            yield obj
        for obj in img.get('auxiliary', []):
            yield obj

    def delay(self, channels=None, space='image', bundle_dpath=None):
        """
        Experimental method

        Args:
            gid (int): image id to load

            channels (FusedChannelSpec): specific channels to load.
                if unspecified, all channels are loaded.

            space (str):
                can either be "image" for loading in image space, or
                "video" for loading in video space.

        TODO:
            - [ ] Currently can only take all or none of the channels from each
                base-image / auxiliary dict. For instance if the main image is
                r|g|b you can't just select g|b at the moment.

            - [ ] The order of the channels in the delayed load should
                match the requested channel order.

            - [ ] TODO: add nans to bands that don't exist or throw an error

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> from os.path import join
            >>> import os
            >>> import pathlib
            >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
            >>> dvc_dpath = pathlib.Path(os.environ.get('DVC_DPATH', _default))
            >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(os.fspath(coco_fpath))
            >>> self = CocoImage(ub.peek(dset.imgs.values()), dset)

        Example:
            >>> from watch.utils.kwcoco_extensions import *  # NOQA
            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = CocoImage(dset.imgs[gid], dset)
            >>> delayed = self.delay()
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = dset.delayed_load(gid)
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))

            >>> crop = delayed.delayed_crop((slice(0, 3), slice(0, 3)))
            >>> crop.finalize()
            >>> crop.finalize(as_xarray=True)

            >>> # TODO: should only select the "red" channel
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = CocoImage(dset.imgs[gid], dset).delay(channels='r')

            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.delayed_load(gid, channels='B1|B2', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
            >>> delayed = dset.delayed_load(gid, channels='B1|B2|B11', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
            >>> delayed = dset.delayed_load(gid, channels='B8|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))

            >>> delayed = dset.delayed_load(gid, channels='B8|foo|bar|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
        """
        from kwcoco.util.util_delayed_poc import DelayedLoad, DelayedChannelConcat
        from kwimage.transform import Affine
        from kwcoco.channel_spec import FusedChannelSpec
        if bundle_dpath is None:
            bundle_dpath = self.dset.bundle_dpath

        img = self.img
        requested = channels
        if requested is not None:
            requested = FusedChannelSpec.coerce(requested)

        def _delay_load_imglike(obj):
            info = {}
            fname = obj.get('file_name', None)
            channels_ = obj.get('channels', None)
            if channels_ is not None:
                channels_ = FusedChannelSpec.coerce(channels_).normalize()
            info['channels'] = channels_
            width = obj.get('width', None)
            height = obj.get('height', None)
            if height is not None and width is not None:
                info['dsize'] = dsize = (width, height)
            else:
                info['dsize'] = None
            if fname is not None:
                info['fpath'] = fpath = join(bundle_dpath, fname)
                info['chan'] = DelayedLoad(fpath, channels=channels_, dsize=dsize)
            return info

        # obj = img
        info = img_info = _delay_load_imglike(img)

        chan_list = []
        if info.get('chan', None) is not None:
            include_flag = requested is None
            if not include_flag:
                if requested.intersection(info['channels']):
                    include_flag = True
            if include_flag:
                chan_list.append(info.get('chan', None))

        for aux in img.get('auxiliary', []):
            info = _delay_load_imglike(aux)
            aux_to_img = Affine.coerce(aux.get('warp_aux_to_img', None))
            chan = info['chan']

            include_flag = requested is None
            if not include_flag:
                if requested.intersection(info['channels']):
                    include_flag = True
            if include_flag:
                chan = chan.delayed_warp(
                    aux_to_img, dsize=img_info['dsize'])
                chan_list.append(chan)

        if len(chan_list) == 0:
            raise ValueError('no data')
        else:
            delayed = DelayedChannelConcat(chan_list)

        # Reorder channels in the requested order
        if requested is not None:
            delayed = delayed.take_channels(requested)

        if hasattr(delayed, 'components'):
            if len(delayed.components) == 1:
                delayed = delayed.components[0]

        if space == 'image':
            pass
        elif space == 'video':
            vidid = img['video_id']
            video = self.dset.index.videos[vidid]
            width = video.get('width', img.get('width', None))
            height = video.get('height', img.get('height', None))
            video_dsize = (width, height)
            img_to_vid = Affine.coerce(img.get('warp_img_to_vid', None))
            delayed = delayed.delayed_warp(img_to_vid, dsize=video_dsize)
        else:
            raise KeyError('space = {}'.format(space))
        return delayed
