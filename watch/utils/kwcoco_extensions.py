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


def populate_watch_fields(dset, target_gsd=10.0, overwrite=False, default_gsd=None):
    """
    Aggregate populate function for fields useful to WATCH.

    Args:
        dset (Dataset): dataset to work with

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
        >>> dset = kwcoco.CocoDataset(fpath)
        >>> target_gsd = 5.0
        >>> populate_watch_fields(dset, target_gsd)
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=-1)))
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=1))

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> print('dset = {!r}'.format(dset))
        >>> target_gsd = 13.0
        >>> populate_watch_fields(dset, target_gsd, default_gsd=1)
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=2))
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=1)))

        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> print('dset = {!r}'.format(dset))
        >>> target_gsd = 13.0
        >>> populate_watch_fields(dset, target_gsd, default_gsd=1)
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=2))
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=1)))
    """
    # Load your KW-COCO dataset (conform populates information like image size)
    dset.conform(pycocotools_info=False)

    for gid in ub.ProgIter(dset.index.imgs.keys(), total=len(dset.index.imgs), desc='populate imgs'):
        coco_populate_geo_img_heuristics(dset, gid, overwrite=overwrite,
                                         default_gsd=default_gsd)

    for vidid in ub.ProgIter(dset.index.videos.keys(), total=len(dset.index.videos), desc='populate videos'):
        coco_populate_geo_video_stats(dset, vidid, target_gsd=target_gsd)

    # serialize intermediate objects
    dset._ensure_json_serializable()


def check_unique_channel_names(dset, gids=None, verbose=0):
    """
    Check each image has unique channel names

    TODO:
        - [ ] move to kwcoco proper

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> check_unique_channel_names(dset)
        >>> # Make some duplicate channels to test
        >>> obj = dset.images().objs[0]
        >>> obj['auxiliary'][0]['channels'] = 'B1|B1'
        >>> obj = dset.images().objs[1]
        >>> obj['auxiliary'][0]['channels'] = 'B1|B1'
        >>> obj = dset.images().objs[2]
        >>> obj['auxiliary'][1]['channels'] = 'B1'
        >>> import pytest
        >>> with pytest.raises(AssertionError):
        >>>     check_unique_channel_names(dset)

    """
    images = dset.images(gids=gids)
    errors = []
    for img in images.objs:
        coco_img = dset._coco_image(img['id'])
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


def coco_list_asset_infos(dset):
    """
    Get a list of filename and channels for each coco image
    """
    asset_infos = []
    for gid in dset.images():
        coco_img = dset._coco_image(gid)
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


def check_geotiff_formats(dset):
    # Enumerate assests on disk
    infos = []
    asset_infos = coco_list_asset_infos(dset)
    for file_info in ub.ProgIter(asset_infos):
        fpath = file_info['fpath']
        info = geotiff_format_info(fpath)
        info.update(file_info)
        infos.append(info)

    ub.varied_values([ub.dict_diff(d, {'fpath', 'filelist'}) for d in infos])


def rewrite_geotiffs(dset):
    import tempfile
    import pathlib
    blocksize = 96
    compress = 'NONE'
    asset_infos = coco_list_asset_infos(dset)

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


def hack_seed_geometadata_in_dset(dset):
    """
    dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
    """
    import kwarray
    from kwarray.distributions import Uniform
    # stay away from edges and poles
    rng = kwarray.ensure_rng(None)
    max_lat = 90 - 10
    max_lon = 180 - 10
    lat_distri = Uniform(-max_lat, max_lat, rng=rng)
    lon_distri = Uniform(-max_lon, max_lon, rng=rng)

    for vidid in dset.videos():
        img = dset.images(vidid=vidid).peek()
        coco_img = dset._coco_image(img['id'])
        obj = coco_img.primary_asset()
        fpath = join(dset.bundle_dpath, obj['file_name'])
        print('fpath = {!r}'.format(fpath))

        format_info = geotiff_format_info(fpath)
        if not format_info['has_geotransform']:
            lon = lon_distri.sample()
            lat = lat_distri.sample()
            from watch.gis import spatial_reference as watch_crs
            epsg_int = watch_crs.utm_epsg_from_latlon(lat, lon)

            from osgeo import osr
            wgs84_crs = osr.SpatialReference()
            wgs84_crs.ImportFromEPSG(4326)
            wgs84_crs.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)

            utm_crs = osr.SpatialReference()
            utm_crs.ImportFromEPSG(4326)
            utm_from_wgs84 = osr.CoordinateTransformation(wgs84_crs, utm_crs)

            utm_x, utm_y, _ = utm_from_wgs84.TransformPoint(lat, lon, 1.0)
            print('utm_y = {!r}'.format(utm_y))
            print('utm_x = {!r}'.format(utm_x))

            w = rng.randint(10, 300)
            h = rng.randint(10, 300)
            ulx, uly, lrx, lry = kwimage.Boxes([[utm_x, utm_y, w, h]], 'cxywh').to_ltrb().data[0]

            command = f'gdal_edit.py -a_ullr {ulx} {uly} {lrx} {lry} -a_srs EPSG:{epsg_int} {fpath}'
            cmdinfo = ub.cmd(command, shell=True)
            print(cmdinfo['out'])
            print(cmdinfo['err'])
            assert cmdinfo['ret'] == 0


def ensure_transfered_geo_data(dset):
    for gid in ub.ProgIter(list(dset.images())):
        transfer_geo_metadata(dset, gid)


def transfer_geo_metadata(dset, gid):
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
        dset = kwcoco.CocoDataset(coco_fpath)
        gid = dset.images().peek()['id']

    Example:
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> hack_seed_geometadata_in_dset(dset)
        >>> transfer_geo_metadata(dset, gid)
        >>> gid = 2
    """
    import watch
    coco_img = dset._coco_image(gid)

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
                for other_gid in dset.images(vidid=vidid):
                    if other_gid != gid:
                        other_coco_img = dset._coco_image(other_gid)
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

        from osgeo import gdal
        # from osgeo import osr
        import affine
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


def coco_populate_geo_video_stats(dset, vidid, target_gsd='max-resolution'):
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
        dset (CocoDataset): coco dataset to be modified inplace
        vidid (int): video_id to modify
        target_gsd (float | str): string code, or float target gsd


    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.utils.kwcoco_extensions import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> import kwcoco
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> target_gsd = 10.0
        >>> vidid = 2
        >>> # We can check transforms before we apply this function
        >>> dset.images(vidid=vidid).lookup('warp_img_to_vid', None)
        >>> # Apply the function
        >>> coco_populate_geo_video_stats(dset, vidid, target_gsd)
        >>> # Check these transforms to make sure they look right
        >>> popualted_video = dset.index.videos[vidid]
        >>> popualted_video = ub.dict_isect(popualted_video, ['width', 'height', 'warp_wld_to_vid', 'target_gsd'])
        >>> print('popualted_video = {}'.format(ub.repr2(popualted_video, nl=-1)))
        >>> dset.images(vidid=vidid).lookup('warp_img_to_vid')

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

    check_unique_channel_names(gids=gids)

    frame_infos = {}

    for gid in gids:
        img = dset.index.imgs[gid]
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
        img = dset.index.imgs[gid]
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
        img = dset.index.imgs[gid]
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


def coco_populate_geo_img_heuristics(dset, gid, overwrite=False,
                                     default_gsd=None, **kw):
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
        errors = _populate_canvas_obj(bundle_dpath, obj, overwrite=overwrite,
                                      default_gsd=default_gsd)
        asset_errors.append(errors)

    if all(asset_errors):
        info = ub.dict_isect(img, {'name', 'file_name', 'id'})
        warnings.warn(f'img {info} has issues introspecting')


@profile
def _populate_canvas_obj(bundle_dpath, obj, overwrite=False, with_wgs=False,
                         default_gsd=None):
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
        'wld_to_pxl',
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


# DEPRECATED
class ORIG_CocoImage(ub.NiceRepr):
    """
    An object-oriented representation of a coco image.

    It provides helper methods that are specific to a single image.

    This operates directly on a single coco image dictionary, but it can
    optionally be connected to a parent dataset, which allows it to use
    CocoDataset methods to query about relationships and resolve pointers.

    This is different than the Images class in coco_object1d, which is just a
    vectorized interface to multiple objects.

    TODO:
        - [x] This will eventually move to kwcoco itself

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


    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import os
        >>> import pathlib
        >>> import kwcoco
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = pathlib.Path(os.environ.get('DVC_DPATH', _default))
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
        >>> #
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> self = CocoImage(coco_dset.dataset['images'][0], coco_dset)
        >>> print('self = {!r}'.format(self))
        >>> stats = self.stats()
        >>> print('stats = {}'.format(ub.repr2(stats, nl=1)))

        >>> delayed = self.delay()
        >>> print('delayed = {!r}'.format(delayed))

        delayed.components[-1]
        delayed.components[-2]

        # delayed.matseg_11
        delayed.take_channels('matseg_1')
        big = delayed.take_channels("coastal|blue|green|red|nir|swir16|cirrus|inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|inv_shared9|inv_shared10|inv_shared11|inv_shared12|inv_shared13|inv_shared14|inv_shared15|inv_shared16|inv_shared17|inv_shared18|inv_shared19|inv_shared20|inv_shared21|inv_shared22|inv_shared23|inv_shared24|inv_shared25|inv_shared26|inv_shared27|inv_shared28|inv_shared29|inv_shared30|inv_shared31|inv_shared32|inv_shared33|inv_shared34|inv_shared35|inv_shared36|inv_shared37|inv_shared38|inv_shared39|inv_shared40|inv_shared41|inv_shared42|inv_shared43|inv_shared44|inv_shared45|inv_shared46|inv_shared47|inv_shared48|inv_shared49|inv_shared50|inv_shared51|inv_shared52|inv_shared53|inv_shared54|inv_shared55|inv_shared56|inv_shared57|inv_shared58|inv_shared59|inv_shared60|inv_shared61|inv_shared62|inv_shared63|inv_shared64|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19")

        import ndsampler
        sampler = ndsampler.CocoSampler(coco_dset)
        sample_grid = sampler.new_sample_grid('video_detection', (3, 128, 128))

        pos_grid = sample_grid['positives']

        tr = pos_grid[len(pos_grid) // 2]
        all_chan = '|'.join(ub.flatten(self.channels.parse().values()))
        tr['channels'] = all_chan
        tr['use_experimental_loader'] = 1
        sample = sampler.load_sample(tr, padkw=dict(constant_values=np.nan))
        sample['im'].shape

        rng = kwarray.ensure_rng(132)
        aff = kwimage.Affine.coerce(offset=rng.randint(-128, 128, size=2), rng=rng)
        space_box = kwimage.Boxes.from_slice(tr['space_slice'])
        space_box = space_box.warp(aff).quantize().astype(int)
        tr_ = ub.dict_union(tr, {'space_slice': space_box.to_slices()[0]})
        tr_['as_xarray'] = 1
        sample = sampler.load_sample(tr_, padkw=dict(constant_values=np.nan))
        im_xarray = sample['im']
        chan_mean = im_xarray.mean(dim=['t', 'y', 'x'])
        print(chan_mean.to_pandas().to_string())


    import kwcoco
    import kwarray
    import ndsampler

    # Seed random number generators
    rng = kwarray.ensure_rng(132)

    kwcoco.CocoDataset.demo('vidshapes8')

    sampler = ndsampler.CocoSampler(coco_dset)
    sample_grid = sampler.new_sample_grid('video_detection', (3, 128, 128))
    tr = sample_grid['positives'][0]

    tr_ = tr.copy()
    aff = kwimage.Affine.coerce(offset=rng.randint(-128, 128, size=2))
    space_box = kwimage.Boxes.from_slice(tr['space_slice']).warp(aff).quantize()
    tr_['space_slice'] = space_box.astype(int).to_slices()[0]
    print('tr_ = {}'.format(ub.repr2(tr_, nl=1)))


    sample = sampler.load_sample(tr_, padkw=dict(constant_values=np.nan))
    print(sample['im'].shape)

    # Out of bounds demo (these slices DO NOT wrap around)
    tr_['space_slice'] = (slice(-128, 0), slice(-128, 0))
    sample = sampler.load_sample(tr_, padkw=dict(constant_values=np.nan))
    sample['im'].shape

    print(ub.repr2(sample['tr'], nl=1))



    """

    def __init__(self, img, dset=None):
        self.img = img
        self.dset = dset

    @classmethod
    def from_gid(cls, dset, gid):
        img = dset.index.imgs[gid]
        self = cls(img, dset=dset)
        return self

    def __nice__(self):
        """
        Example:
            >>> import kwcoco
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
        from functools import partial
        stats = self.stats()
        stats = ub.map_vals(str, stats)
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
                stats[key] = getattr(self, attrname)
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
        return self.channels.numel()
        # return sum(map(len, self.channels.streams()))

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

        if len(candidates) == 0:
            return None

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
            obj = img
            # cant remove auxiliary otherwise inplace modification doesnt work
            # obj = ub.dict_diff(img, {'auxiliary'})
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
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import kwcoco
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
            requested = FusedChannelSpec.coerce(requested).normalize()

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


def _demo_kwcoco_with_heatmaps(num_videos=1):
    """
    Return a dummy kwcoco file with special metdata

    Ignore:
        from watch.utils.kwcoco_extensions import _demo_kwcoco_with_heatmaps
        coco_dset = _demo_kwcoco_with_heatmaps()
        key = 'salient'
        for vidid in coco_dset.videos():
            frames = []
            for gid in coco_dset.images(vidid=vidid):
                delayed = coco_dset._coco_image(gid).delay(channels=key, space='video')
                final = delayed.finalize()
                frames.append(final)
            vid_stack = kwimage.stack_images_grid(frames, axis=1, pad=5, bg_value=1)

            import kwplot
            kwplot.imshow(vid_stack)
    """
    import pathlib
    import kwarray
    import kwcoco
    from kwcoco.demo import perterb

    coco_dset = kwcoco.CocoDataset.demo(
        'vidshapes', num_videos=1, num_frames=20,
        multispectral=True, image_size=(128, 128))

    perterb_config = {
        'box_noise': 0.5,
        'n_fp': 3,
        # 'with_probs': 1,
    }
    perterb.perterb_coco(coco_dset, **perterb_config)

    asset_dpath = pathlib.Path(coco_dset.assets_dpath)
    dummy_heatmap_dpath = asset_dpath / 'dummy_heatmaps'
    dummy_heatmap_dpath.mkdir(exist_ok=1, parents=True)

    rng = 1018933676  # random seed
    rng = kwarray.ensure_rng(rng)

    channels = 'notsalient|salient'
    channels = kwcoco.FusedChannelSpec.coerce(channels)
    chan_codes = channels.normalize().as_list()

    aux_width = 64
    aux_height = 64
    dims = (aux_width, aux_height)
    for img in coco_dset.index.imgs.values():

        warp_img_from_aux = kwimage.Affine.scale((
            img['width'] / aux_width, img['height'] / aux_height))
        warp_aux_from_img = warp_img_from_aux.inv()

        # Grab perterbed detections from this image
        img_dets = coco_dset.annots(gid=img['id']).detections

        # Transfom dets into aux space
        aux_dets = img_dets.warp(warp_aux_from_img)

        # Hack: use dets to draw some randomish heatmaps
        sseg = aux_dets.data['segmentations']
        chan_datas = []
        for code in chan_codes:
            chan_data = np.zeros(dims, dtype=np.float32)
            for poly in sseg.data:
                poly.fill(chan_data, 1)

            # Add lots of noise to the data
            chan_data += (rng.randn(*dims) * 0.1)
            chan_data + chan_data.clip(0, 1)
            chan_data = kwimage.gaussian_blur(chan_data, sigma=1.2)
            chan_data = chan_data.clip(0, 1)
            mask = rng.randn(*dims)
            chan_data = chan_data * ((kwimage.fourier_mask(chan_data, mask)[..., 0]) + .5)
            chan_data += (rng.randn(*dims) * 0.1)
            chan_data = chan_data.clip(0, 1)
            chan_datas.append(chan_data)
        hwc_probs = np.stack(chan_datas, axis=2)

        # TODO do something with __WIP_add_auxiliary to make this clear and
        # concise
        heatmap_fpath = dummy_heatmap_dpath / 'dummy_heatmap_{}.tif'.format(img['id'])
        kwimage.imwrite(heatmap_fpath, hwc_probs, backend='gdal', compress='NONE',
                        blocksize=96)
        aux_height, aux_width = hwc_probs.shape[0:2]

        auxlist = img['auxiliary']
        auxlist.append({
            'file_name': heatmap_fpath,
            'width': aux_width,
            'height': aux_height,
            'channels': channels.spec,
            'warp_aux_to_img': warp_img_from_aux.concise(),
        })

    # Hack in geographic info
    hack_seed_geometadata_in_dset(coco_dset)
    for gid in coco_dset.images():
        transfer_geo_metadata(coco_dset, gid)
    return coco_dset
