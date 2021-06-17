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


def populate_watch_fields(dset, target_gsd=10.0, overwrite=False):
    """
    Aggregate populate function for fields useful to WATCH.

    Args:
        dset (Dataset): dataset to work with
        target_gsd (float): target gsd in meters

    Ignore:
        >>> from watch.tools.kwcoco_extensions import *  # NOQA
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
        >>> from watch.tools.kwcoco_extensions import *  # NOQA
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

        if img.get('file_name', None) is None:
            # Choose any one of the auxiliary images, we chose the biggest
            # arbitrarilly
            # TODO: auxiliary
            aux_width = [aux['width'] for aux in img['auxiliary']]
            aux_height = [aux['height'] for aux in img['auxiliary']]
            aux_idx = (np.array(aux_width) * np.array(aux_height)).argmax()
            aux_chosen = img['auxiliary'][aux_idx]
            aux_to_wld = Affine.coerce(aux_chosen.get('warp_to_wld', None))
            aux_to_img = Affine.coerce(aux_chosen['warp_aux_to_img'])
            img_to_aux = aux_to_img.inv()
            img_to_wld = aux_to_wld @ img_to_aux
            approx_meter_gsd = aux_chosen['approx_meter_gsd']
        else:
            img_to_wld = Affine.coerce(img.get('warp_to_wld', None))
            approx_meter_gsd = img['approx_meter_gsd']

        frame_infos[gid] = {
            'img_to_wld': img_to_wld,
            # Note: division because gsd is inverted. This got me confused, but
            # I'm pretty sure this works.
            'target_gsd': target_gsd,
            'approx_meter_gsd': approx_meter_gsd,
            'width': img['width'],
            'height': img['height'],
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
        gsd = min_gsd
    elif target_gsd == 'min-resolution':
        gsd = max_gsd
    else:
        gsd = target_gsd
        if not isinstance(target_gsd, numbers.Number):
            raise TypeError('target_gsd must be a code or number = {}'.format(type(target_gsd)))
    gsd = float(gsd)

    for info in frame_infos.values():
        info['target_gsd'] = gsd
        info['to_target_scale_factor'] = info['approx_meter_gsd'] / gsd

    available_channels = set()
    available_gsds = set()
    for gid in gids:
        img = dset.index.imgs[gid]
        for obj in ub.flatten([[img], img.get('auxiliary', [])]):
            if obj.get('file_name', None) is not None:
                available_channels.add(obj.get('channels', None))
                _gsd = obj.get('approx_meter_gsd')
                if _gsd is not None:
                    _gsd = round(_gsd, 1)
                available_gsds.add(_gsd)

    # Align to frame closest to the target GSD
    base_gid, info = min(frame_infos.items(),
                         key=lambda kv: 1 - kv[1]['to_target_scale_factor'])
    scale = info['to_target_scale_factor']

    # Can add an extra transform here if the video is not exactly in
    # any specific image space
    wld_to_vid = Affine.scale(scale) @ info['img_to_wld'].inv()
    video['width'] = int(np.ceil(info['width'] * scale))
    video['height'] = int(np.ceil(info['height'] * scale))
    video['target_gsd'] = gsd

    # Store metadata in the video
    video['num_frames'] = len(gids)
    video['warp_wld_to_vid'] = wld_to_vid.__json__()
    video['min_gsd'] = min_gsd
    video['max_gsd'] = max_gsd
    video['available_channels'] = sorted(available_channels)
    video['available_gsds'] = sorted(available_gsds)

    for gid in gids:
        img = dset.index.imgs[gid]
        img_to_wld = frame_infos[gid]['img_to_wld']
        img_to_vid = wld_to_vid @ img_to_wld
        img['warp_img_to_vid'] = img_to_vid.concise()

        for aux in img.get('auxiliary', []):
            aux_to_vid = img_to_vid @ Affine.coerce(aux['warp_aux_to_img'])
            aux['warp_aux_to_vid'] = aux_to_vid.concise()

    if 0:
        dset.imgs[min_gsd_gid]
        dset.imgs[max_gsd_gid]
        # inspect
        print(ub.repr2(dset.images(gids).objs, nl=1))
        print(ub.repr2(dset.videos([vidid]).objs, nl=1))


def coco_populate_geo_img_heuristics(dset, gid, overwrite=False, **kw):
    """
    Note: this will not overwrite existing channel info unless specified

    Example:
        >>> from watch.tools.kwcoco_extensions import *  # NOQA
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
    obj = img  # NOQA
    _populate_canvas_obj(bundle_dpath, img, overwrite=overwrite)
    for aux in img.get('auxiliary', []):
        _populate_canvas_obj(bundle_dpath, aux, overwrite=overwrite)


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
    # Can only do this for images with file names
    if fname is not None:
        fpath = join(bundle_dpath, fname)

        if overwrite or warp_to_wld is None or approx_meter_gsd is None:
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
                        'wld_to_pxl': info['wld_to_pxl'],
                    })

                approx_meter_gsd = info['approx_meter_gsd']
            except Exception:
                warnings.warn('no crs info for img, assuming 1 gsd')
                obj_to_wld = Affine.eye()
                approx_meter_gsd = 1.0
            obj['approx_meter_gsd'] = approx_meter_gsd
            obj['warp_to_wld'] = Affine.coerce(obj_to_wld).__json__()

        if overwrite or num_bands is None:
            num_bands = _introspect_num_bands(fpath)
            obj['num_bands'] = num_bands

        if overwrite or channels is None:
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
        Mx6[ix * 2]     = (x1, y1,  0,  0,  1,  0)
        Mx6[ix * 2 + 1] = ( 0,  0, x1, y1,  0,  1)
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
        [   0,    0,    1],
    ])
    return A


def _sensor_channel_hueristic(sensor_coarse, num_bands):
    """
    Given a sensor and the number of bands in the image, return likely channel
    codes for the image
    """
    err = 0
    if sensor_coarse == 'WV':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 8:
            channels = 'wv1|wv2|wv3|wv4|wv5|wv6|wv7|wv8'
            # channels = 'cb|b|g|y|r|wv6|wv7|wv8'
        else:
            err = 1
    elif sensor_coarse == 'S2':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 13:
            channels = 's1|s2|s3|s4|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
            # channels = 'cb|b|g|r|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
        else:
            err = 1
    elif sensor_coarse == 'LC':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 11:
            channels = 'lc1|lc2|lc3|lc4|lc5|lc6|lc7|lc8|lc9|lc10|lc11'
            # channels = 'cb|b|g|r|lc5|lc6|lc7|pan|lc9|lc10|lc11'
        else:
            err = 1
    else:
        err = 1
    if err:
        import xdev
        xdev.embed()
        raise NotImplementedError(f'sensor_coarse={sensor_coarse}, num_bands={num_bands}')
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
        raise Exception('num_bands=f{num_bands}')
    return channels


def Affine_concise(aff):
    """
    TODO: remove after kwimage is updated to 0.7.8
    """
    import numpy as np
    self = aff
    params = self.decompose()
    params['type'] = 'affine'
    if np.allclose(params['offset'], (0, 0)):
        params.pop('offset')
    elif ub.allsame(params['offset']):
        params['offset'] = params['offset'][0]
    if np.allclose(params['scale'], (1, 1)):
        params.pop('scale')
    elif ub.allsame(params['scale']):
        params['scale'] = params['scale'][0]
    if np.allclose(params['shear'], 0):
        params.pop('shear')
    if np.allclose(params['theta'], 0):
        params.pop('theta')
    return params


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

    if not hasattr(warp_aux_to_img, 'concise'):
        ub.inject_method(warp_aux_to_img, Affine_concise, 'concise')

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
        'width', 'height', 'wgs84_to_wld', 'wld_to_pxl',
    }))
    for aux in auxiliary:
        warp_aux_to_wld = kwimage.Affine.coerce(aux['warp_to_wld'])
        warp_aux_to_img = warp_wld_to_img @ warp_aux_to_wld
        aux['warp_aux_to_img'] = warp_aux_to_img.concise()
