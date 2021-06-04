"""
Given the raw data in kwcoco format, this script will extract orthorectified
regions around areas of interest across time.

Notes:

    # Given the output from geojson_to_kwcoco this script extracts
    # orthorectified regions.

    # https://data.kitware.com/#collection/602457272fa25629b95d1718/folder/602c3e9e2fa25629b97e5b5e

    python ~/code/watch/scripts/coco_align_geotiffs.py \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2 \
            --context_factor=1.5

    # Archive the data and upload to data.kitware.com
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    7z a ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2.zip ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2

    stamp=$(date +"%Y-%m-%d")
    # resolve links (7z cant handl)
    rsync -avpL drop0_aligned_v2 drop0_aligned_v2_$stamp
    7z a drop0_aligned_v2_$stamp.zip drop0_aligned_v2_$stamp

    source $HOME/internal/secrets
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    girder-client --api-url https://data.kitware.com/api/v1 upload 602c3e9e2fa25629b97e5b5e drop0_aligned_v2_$stamp.zip

    python ~/code/watch/scripts/coco_align_geotiffs.py \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi \
            --context_factor=1.5


Test:

    There was a bug in KR-WV, run the script only on that region to test if we
    have fixed it.

    kwcoco stats ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    jq .images ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    kwcoco subset ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json --gids=1129,1130 --dst ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json

    python ~/code/watch/scripts/coco_align_geotiffs.py \
            --src ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json \
            --dst ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0_aligned_WV_Fix \
            --rpc_align_method pixel_crop \
            --context_factor=3.5

           # --src ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json \
"""
import kwcoco
import kwimage
import numpy as np
import os
import scriptconfig as scfg
import socket
import ubelt as ub
import dateutil.parser
from shapely import ops
from os.path import join, exists


class CocoAlignGeotiffConfig(scfg.Config):
    """
    Create a dataset of aligned temporal sequences around objects of interest
    in an unstructured collection of annotated geotiffs.

    High Level Steps:
        * Find a set of geospatial AOIs
        * For each AOI find all images that overlap
        * Orthorectify (or warp) the selected spatial region and its
          annotations to a cannonical space.

    TODO:
        - [ ] Add method for extracting "negative ROIs" that are nearby
            "positive ROIs".
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'context_factor': scfg.Value(1.5, help=ub.paragraph(
            '''
            scale factor for the clustered ROIs.
            Amount of context to extract around each ROI.
            '''
        )),

        # TODO: change this name to just align-method or something
        'rpc_align_method': scfg.Value('orthorectify', help=ub.paragraph(
            '''
            Can be one of:
                (1) orthorectify - which uses gdalwarp with -rpc,
                (2) pixel_crop - which warps annotations onto pixel with RPCs
                    but only crops the original image without distortion,
                (3) affine_warp - which ignores RPCs and uses the affine
                    transform in the geotiff metadata.
            '''
        )),

        'write_subsets': scfg.Value(True, help=ub.paragraph(
            '''
            if True, writes a separate kwcoco file for every discovered ROI
            in addition to the final kwcoco file.
            '''
        )),

        'visualize': scfg.Value(True, help=ub.paragraph(
            '''
            if True, normalize and draw image / annotation sequences when
            extracting.
            '''
        )),

    }


def main(**kw):
    """
    Main function for coco_align_geotiffs.
    See :class:``CocoAlignGeotiffConfig` for details

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        from coco_align_geotiffs import *  # NOQA
        import kwcoco
        src = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json')
        dst = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned')
        kw = {
            'src': src,
            'dst': dst,
        }

    Example:
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> from watch.gis.geotiff import geotiff_metadata
        >>> # Create a dead simple coco dataset with one image
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset()
        >>> ls_prod = grab_landsat_product()
        >>> fpath = ls_prod['bands'][0]
        >>> meta = geotiff_metadata(fpath)
        >>> # We need a date captured ATM in a specific format
        >>> dt = dateutil.parser.parse(
        >>>     meta['filename_meta']['acquisition_date'])
        >>> date_captured = dt.strftime('%Y/%m/%d')
        >>> gid = dset.add_image(file_name=fpath, date_captured=date_captured)
        >>> dummy_poly = kwimage.Polygon(exterior=meta['wgs84_corners'])
        >>> dummy_poly = dummy_poly.scale(0.3, about='center')
        >>> sseg_geos = dummy_poly.swap_axes().to_geojson()
        >>> # NOTE: script is not always robust to missing annotation
        >>> # information like segmentation and bad bbox, but for thist
        >>> # test config it is
        >>> dset.add_annotation(
        >>>     image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)
        >>> #
        >>> # Create arguments to the script
        >>> dpath = ub.ensure_app_cache_dir('smart_watch/test/coco_align_geotiff')
        >>> dst = ub.ensuredir((dpath, 'align_bundle'))
        >>> ub.delete(dst)
        >>> dst = ub.ensuredir(dst)
        >>> kw = {
        >>>     'src': dset.dataset,
        >>>     'dst': dst,
        >>> }
        >>> new_dset = main(**kw)
    """
    config = CocoAlignGeotiffConfig(default=kw, cmdline=True)

    # Store that this dataset is a result of a process.
    # Note what the process is, what its arguments are, and where the process
    # was executed.
    config_dict = config.to_dict()
    if not isinstance(config_dict['src'], str):
        # If the dataset was given in memory we don't know the path and we cant
        # always serialize it, so we punt and mark it as such
        config_dict['src'] = ':memory:'

    process_info = {
        'type': 'process',
        'properties': {
            'name': 'coco_align_geotiffs',
            'args': config_dict,
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'timestamp': ub.timestamp(),
        }
    }
    print('process_info = {}'.format(ub.repr2(process_info, nl=2)))

    src_fpath = config['src']
    dst_dpath = config['dst']
    context_factor = config['context_factor']
    rpc_align_method = config['rpc_align_method']
    visualize = config['visualize']
    write_subsets = config['write_subsets']

    output_bundle_dpath = dst_dpath

    # Load the dataset and extract geotiff metadata from each image.
    dset = kwcoco.CocoDataset(src_fpath)
    # dset = dset.subset([1])
    update_coco_geotiff_metadata(dset, serializable=False)

    # Construct the "data cube"
    cube = SimpleDataCube(dset)

    # Find the clustered ROI regions
    sh_all_rois, kw_all_rois = find_roi_regions(dset)

    # Exapnd the ROI by the context factor and convert to a bounding box
    kw_all_box_rois = [
        p.scale(context_factor, about='center').bounding_box_polygon()
        for p in kw_all_rois
    ]

    # For each ROI extract the aligned regions to the target path
    extract_dpath = ub.expandpath(output_bundle_dpath)

    # Create a new dataset that we will extend as we extract ROIs
    new_dset = kwcoco.CocoDataset()

    new_dset.dataset['info'] = [
        process_info,
    ]

    time_region = None

    to_extract = []
    for space_region in ub.ProgIter(kw_all_box_rois, desc='query overlaps', verbose=3):
        image_overlaps = cube.query_image_overlaps(space_region, time_region)
        to_extract.append(image_overlaps)

    for image_overlaps in ub.ProgIter(to_extract, desc='extract ROI videos', verbose=3):
        video_name = image_overlaps['video_name']
        print('video_name = {!r}'.format(video_name))

        sub_bundle_dpath = join(extract_dpath, video_name)
        print('sub_bundle_dpath = {!r}'.format(sub_bundle_dpath))

        cube.extract_overlaps(image_overlaps, extract_dpath,
                              rpc_align_method=rpc_align_method,
                              new_dset=new_dset, visualize=visualize,
                              write_subsets=write_subsets)

    new_dset.fpath = join(extract_dpath, 'data.kwcoco.json')
    print('Dumping new_dset.fpath = {!r}'.format(new_dset.fpath))
    new_dset.reroot(new_root=output_bundle_dpath, absolute=False)
    new_dset.dump(new_dset.fpath, newlines=True)
    print('finished')
    return new_dset


def latlon_text(lat, lon, precision=6):
    """
    Make a lat,lon string suitable for a filename.

    Pads with leading zeros so file names will align nicely at the same level
    of prcision.

    Args:
        lat (float): degrees latitude

        lon (float): degrees longitude

        precision (float, default=6):
            Number of trailing decimal places. As rule of thumb set this to:
                6 - for ~10cm accuracy,
                5 - for ~1m accuracy,
                2 - for ~1km accuracy,

    Notes:
        1 degree of latitude is *very* roughly the order of 100km, so the
        default precision of 6 localizes down to ~0.1 meters, which will
        usually be sufficient for satellite applications, but be mindful of
        using this text in applications that require more precision. Note 1
        degree of longitude will vary, but will always be at least as precise
        as 1 degree of latitude.

    Example:
        >>> lat = 90
        >>> lon = 180
        >>> print(latlon_text(lat, lon))
        N90.000000E180.000000

        >>> lat = 0
        >>> lon = 0
        >>> print(latlon_text(lat, lon))
        N00.000000E000.000000

    Example:
        >>> print(latlon_text(80.123, 170.123))
        >>> print(latlon_text(10.123, 80.123))
        >>> print(latlon_text(0.123, 0.123))
        N80.123000E170.123000
        N10.123000E080.123000
        N00.123000E000.123000

        >>> print(latlon_text(80.123, 170.123, precision=2))
        >>> print(latlon_text(10.123, 80.123, precision=2))
        >>> print(latlon_text(0.123, 0.123, precision=2))
        N80.12E170.12
        N10.12E080.12
        N00.12E000.12

        >>> print(latlon_text(80.123, 170.123, precision=5))
        >>> print(latlon_text(10.123, 80.123, precision=5))
        >>> print(latlon_text(0.123, 0.123, precision=5))
        N80.12300E170.12300
        N10.12300E080.12300
        N00.12300E000.12300
    """
    def _build_float_precision_fmt(num_leading, num_trailing):
        num2 = num_trailing
        # 2 extra for radix and leading sign
        num1 = num_leading + num_trailing + 2
        fmtparts = ['{:+0', str(num1), '.', str(num2), 'F}']
        fmtstr = ''.join(fmtparts)
        return fmtstr

    assert -90 <= lat <= 90, 'invalid lat'
    assert -180 <= lon <= 180, 'invalid lon'

    # Ensure latitude had 2 leading places and longitude has 3
    latfmt = _build_float_precision_fmt(2, precision)
    lonfmt = _build_float_precision_fmt(3, precision)

    lat_str = latfmt.format(lat).replace('+', 'N').replace('-', 'S')
    lon_str = lonfmt.format(lon).replace('+', 'E').replace('-', 'W')
    text = lat_str + lon_str
    return text


class SimpleDataCube(object):
    """
    Given a CocoDataset containing geotiffs, provide a simple API to extract a
    region in some coordinate space.

    Intended usage is to use :func:`query_image_overlaps` to find images that
    overlap an ROI, then then :func:`extract_overlaps` to warp spatial subsets
    of that data into an aligned temporal sequence.
    """

    def __init__(cube, dset):

        gid_to_poly = {}
        for gid, img in dset.imgs.items():
            info = img['geotiff_metadata']
            kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
            sh_img_poly = kw_img_poly.to_shapely()
            gid_to_poly[gid] = sh_img_poly

        cube.dset = dset
        cube.gid_to_poly = gid_to_poly

    def query_image_overlaps(cube, space_region, time_region=None):
        """
        Find the images that overlap with a space-time region

        Args:
            space_region (kwimage.Polygon):
                a polygon ROI in WGS84 coordinates

            time_region (NotImplemented): NotImplemented

        Returns:
            dict :
                Information about which images belong to this ROI and their
                temporal sequence. Also contains strings to be used for
                subdirectories in the extract step.
        """
        if time_region is not None:
            raise NotImplementedError('have not implemented time ranges yet')

        space_box = space_region.bounding_box().to_ltrb()
        latmin, lonmin, latmax, lonmax = space_box.data[0]
        # from watch.utils.util_place import conv_lat_lon
        # min_pt = conv_lat_lon(str(ymin), str(xmin), format='ISO-D')
        # max_pt = conv_lat_lon(str(ymax), str(xmax), format='ISO-D')

        min_pt = latlon_text(latmin, lonmin)
        max_pt = latlon_text(latmax, lonmax)
        # TODO: is there an ISO standard for encoding this?
        space_str = '{}_{}'.format(min_pt, max_pt)

        sh_space = space_region.to_shapely()
        unordered_gids = []
        for gid, sh_img_poly in cube.gid_to_poly.items():
            # kw_img_poly = kwimage.Polygon.from_shapely(sh_img_poly)
            # print('kw_img_poly = {!r}'.format(kw_img_poly))
            flag = sh_img_poly.intersects(sh_space)
            if flag:
                unordered_gids.append(gid)
        print('Found {} overlaping images'.format(len(unordered_gids)))

        date_to_gids = ub.group_items(
            unordered_gids,
            key=lambda gid: cube.dset.imgs[gid]['datetime_acquisition']
        )
        dates = sorted(date_to_gids)

        if len(dates) == 0:
            raise Exception('Found no overlaping images')
        else:
            min_date = min(dates)
            max_date = max(dates)
            print('From {!r} to {!r}'.format(min_date, max_date))

        video_name = space_str

        image_overlaps = {
            'date_to_gids': date_to_gids,
            'space_region': space_region,
            'space_str': space_str,
            'space_box': space_box,
            'video_name': video_name,
        }
        return image_overlaps

    def _warp_image(cube, img):
        pass

    def extract_overlaps(cube, image_overlaps, extract_dpath,
                         rpc_align_method='orthorectify', new_dset=None,
                         write_subsets=True, visualize=True):
        """
        Given a region of interest, extract an aligned temporal sequence
        of data to a specified directory.

        Args:
            image_overlaps (dict): Information about images in an ROI and their
                temporal order computed from :func:``query_image_overlaps``.

            extract_dpath (str):
                where to dump the data extracted from this ROI.

            rpc_align_method (str):
                how to handle RPC information
                (see :class:``CocoAlignGeotiffConfig`` for details)

            new_dset (kwcoco.CocoDataset | None):
                if specified, add extracted images and annotations to this
                dataset, otherwise create a new dataset.

            write_subset (bool, default=True):
                if True, write out a separate manifest file containing only
                information in this ROI.

            visualize (bool, default=True):
                if True, dump image and annotation visalizations parallel to
                the extracted data.

        Returns:
            kwcoco.CocoDataset: the given or new dataset that was modified
        """
        # import watch
        import datetime

        dset = cube.dset

        date_to_gids = image_overlaps['date_to_gids']
        space_str = image_overlaps['space_str']
        space_box = image_overlaps['space_box']
        space_region = image_overlaps['space_region']
        video_name = image_overlaps['video_name']

        sub_bundle_dpath = ub.ensuredir((extract_dpath, video_name))

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        dates = sorted(date_to_gids)

        new_video = {
            'name': video_name,
        }

        if new_dset is None:
            new_dset = kwcoco.CocoDataset()
        new_vidid = new_dset.add_video(**new_video)

        frame_index = 0

        sub_new_gids = []

        for cat in dset.cats.values():
            new_dset.ensure_category(**cat)

        # TODO: parallelize over images
        from kwcoco.util.util_futures import Executor
        executor = Executor(mode='thread', max_workers=16)

        for date in ub.ProgIter(dates, desc='extracting regions', verbose=3):
            gids = date_to_gids[date]
            iso_time = datetime.date.isoformat(date.date())

            # TODO: Is there any other consideration we should make when
            # multiple images have the same timestamp?
            for num, gid in enumerate(gids):
                img = dset.imgs[gid]
                auxiliary = img.get('auxiliary', [])

                # Construct a name for the subregion to extract.
                sensor_coarse = img.get('sensor_coarse', 'unknown')
                name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)

                objs = []
                has_base_image = img.get('file_name', None) is not None
                if has_base_image:
                    objs.append(ub.dict_diff(img, {'auxiliary'}))
                objs.extend(auxiliary)

                bundle_dpath = dset.bundle_dpath

                is_rpcs = [obj['geotiff_metadata']['is_rpc'] for obj in objs]
                assert ub.allsame(is_rpcs)
                is_rpc = ub.peek(is_rpcs)

                if is_rpc and rpc_align_method != 'affine_warp':
                    align_method = rpc_align_method
                    if align_method == 'pixel_crop':
                        align_method = 'pixel_crop'
                else:
                    align_method = 'affine_warp'

                dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse,
                                          align_method))

                is_multi_image = len(objs) > 1

                job_list = []
                for obj in ub.ProgIter(objs, desc='warp auxiliaries', verbose=0):
                    job = executor.submit(
                        _aligncrop, obj, bundle_dpath, name, sensor_coarse,
                        dst_dpath, space_region, space_box, align_method,
                        is_multi_image)
                    job_list.append(job)

                dst_list = []
                for job in ub.ProgIter(job_list, desc='warp auxiliaries'):
                    dst = job.result()
                    dst_list.append(dst)

                from watch.tools.kwcoco_extensions import _populate_canvas_obj
                from watch.tools.kwcoco_extensions import _recompute_auxiliary_transforms
                if align_method != 'pixel_crop':
                    # If we are a pixel crop, we can transform directly
                    for dst in dst_list:
                        # hack this in for heuristics
                        if 'sensor_coarse' in img:
                            dst['sensor_coarse'] = img['sensor_coarse']
                        _populate_canvas_obj(bundle_dpath, dst, overwrite=True, with_wgs=True)

                new_img = {
                    'name': name,
                }

                if has_base_image:
                    base_dst = dst_list[0]
                    new_img.update(base_dst)
                    aux_dst = dst_list[1:]
                    assert len(aux_dst) == 0, 'cant have aux and base yet'
                else:
                    aux_dst = dst_list

                # Hack because heurstics break when fnames change
                for old_aux, new_aux in zip(auxiliary, aux_dst):
                    new_aux['channels'] = old_aux['channels']
                    new_aux['parent_file_name'] = old_aux['file_name']

                if len(aux_dst):
                    new_img['auxiliary'] = aux_dst
                    _recompute_auxiliary_transforms(new_img)

                carry_over = ub.dict_isect(img, {
                    'date_captured',
                    'approx_elevation',
                    'sensor_candidates',
                    'num_bands',
                    'sensor_coarse',
                    'site_tag',
                    'channels',
                })

                # Carry over appropriate metadata from original image
                new_img.update(carry_over)
                new_img['parent_file_name'] = img['file_name']  # remember which image this came from
                new_img['video_id'] = new_vidid
                new_img['frame_index'] = frame_index
                new_img['timestamp'] = date.toordinal()

                frame_index += 1
                new_gid = new_dset.add_image(**new_img)
                sub_new_gids.append(new_gid)

                # HANDLE ANNOTATIONS
                HANDLE_ANNS = True
                if HANDLE_ANNS:
                    """
                    It would probably be better to warp pixel coordinates using the
                    same transform found by gdalwarp, but I'm not sure how to do
                    that. Thus we transform the geocoordinates to the new extracted
                    img coords instead. Hopefully gdalwarp preserves metadata
                    enough to do this.
                    """
                    aids = dset.index.gid_to_aids[img['id']]
                    anns = list(ub.take(dset.index.anns, aids))

                    geo_poly_list = []
                    for ann in anns:
                        # Q: WHAT FORMAT ARE THESE COORDINATES IN?
                        # A: I'm fairly sure these coordinates are all Traditional-WGS84-Lon-Lat
                        # We convert them to authority compliant WGS84 (lat-lon)
                        # Hack to support real and orig drop0 geojson
                        geo = _fix_geojson_poly(ann['segmentation_geos'])
                        geo_coords = geo['coordinates'][0]
                        exterior = kwimage.Coords(np.array(geo_coords)[:, ::-1])
                        geo_poly = kwimage.Polygon(exterior=exterior)
                        geo_poly_list.append(geo_poly)
                    geo_polys = kwimage.MultiPolygon(geo_poly_list)

                    if align_method == 'orthorectify':
                        # Is the affine mapping in the destination image good
                        # enough after the image has been orthorectified?
                        pxl_polys = geo_polys.warp(new_img['wgs84_to_wld']).warp(new_img['wld_to_pxl'])
                    elif align_method == 'pixel_crop':
                        raise NotImplementedError('fixme')
                        yoff, xoff = new_img['transform']['st_offset']
                        orig_pxl_poly_list = []
                        for ann in anns:
                            old_poly = kwimage.Polygon.from_coco(ann['segmentation'])
                            orig_pxl_poly_list.append(old_poly)
                        orig_pxl_polys = kwimage.MultiPolygon(orig_pxl_poly_list)
                        pxl_polys = orig_pxl_polys.translate((-xoff, -yoff))
                    elif align_method == 'affine_warp':
                        # Warp Auth-WGS84 to whatever the image world space is,
                        # and then from there to pixel space.
                        pxl_polys = geo_polys.warp(new_img['wgs84_to_wld']).warp(new_img['wld_to_pxl'])
                    else:
                        raise KeyError(align_method)

                    def _test_inbounds(pxl_poly):
                        xs, ys = pxl_poly.data['exterior'].data.T
                        flags_x1 = xs < 0
                        flags_y1 = ys < 0
                        flags_x2 = xs >= new_img['width']
                        flags_y2 = ys >= new_img['height']
                        flags = flags_x1 | flags_x2 | flags_y1 | flags_y2
                        n_oob = flags.sum()
                        is_any = n_oob > 0
                        is_all = n_oob == len(flags)
                        return is_any, is_all

                    flags = [not _test_inbounds(p)[1] for p in pxl_polys]

                    valid_anns = [ann.copy() for ann in ub.compress(anns, flags)]
                    valid_pxl_polys = list(ub.compress(pxl_polys, flags))

                    print('Num annots warped {} / {}'.format(len(valid_anns), len(anns)))
                    for ann, pxl_poly in zip(valid_anns, valid_pxl_polys):
                        ann['segmentation'] = pxl_poly.to_coco(style='new')
                        pxl_box = pxl_poly.bounding_box().quantize().to_xywh()
                        xywh = list(pxl_box.to_coco())[0]
                        ann['bbox'] = xywh
                        ann['image_id'] = new_gid
                        new_dset.add_annotation(**ann)

                if visualize:
                    # See if we can look at what we made
                    from watch.utils.util_norm import normalize_intensity

                    new_delayed = new_dset.delayed_load(new_gid)
                    if hasattr(new_delayed, 'components'):
                        components = new_delayed.components
                    else:
                        components = [new_delayed]

                    for chan in components:
                        spec = chan.channels.spec
                        canvas = chan.finalize()

                        # canvas = kwimage.imread(dst_gpath)
                        canvas = normalize_intensity(canvas)
                        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
                            # hack for wv
                            canvas = canvas[..., 0]
                        canvas = kwimage.ensure_float01(canvas)

                        view_img_dpath = ub.ensuredir(
                            (sub_bundle_dpath, sensor_coarse,
                             '_view_img_' + align_method))

                        if HANDLE_ANNS:
                            view_ann_dpath = ub.ensuredir(
                                (sub_bundle_dpath, sensor_coarse,
                                 '_view_ann_' + align_method))

                        view_img_fpath = ub.augpath(name, dpath=view_img_dpath) + '_' + str(spec) + '.view_img.jpg'
                        kwimage.imwrite(view_img_fpath, kwimage.ensure_uint255(canvas))

                        if HANDLE_ANNS:
                            dets = kwimage.Detections.from_coco_annots(valid_anns, dset=dset)
                            view_ann_fpath = ub.augpath(name, dpath=view_ann_dpath) + '_' + str(spec) + '.view_ann.jpg'
                            ann_canvas = dets.draw_on(canvas)
                            kwimage.imwrite(view_ann_fpath, kwimage.ensure_uint255(ann_canvas))

                if 1:
                    # Fix json serializability
                    print('new_gid = {!r}'.format(new_gid))
                    new_img = new_dset.index.imgs[new_gid]
                    new_objs = [new_img] + new_img.get('auxiliary', [])
                    for obj in new_objs:
                        if 'warp_to_wld' in obj:
                            obj['warp_to_wld'] = kwimage.Affine.coerce(obj['warp_to_wld']).concise()
                        if 'wld_to_pxl' in obj:
                            obj['wld_to_pxl'] = kwimage.Affine.coerce(obj['wld_to_pxl']).concise()
                        obj.pop('wgs84_to_wld', None)

                    from kwcoco.util import util_json
                    assert not list(util_json.find_json_unserializable(new_img))

        if write_subsets:
            print('Writing data subset')
            new_dset._check_json_serializable()

            sub_dset = new_dset.subset(sub_new_gids, copy=True)
            sub_dset.fpath = join(sub_bundle_dpath, 'subdata.kwcoco.json')
            sub_dset.reroot(new_root=sub_bundle_dpath, absolute=False)
            sub_dset.dump(sub_dset.fpath, newlines=True)
        return new_dset


def update_coco_geotiff_metadata(dset, serializable=True):
    """
    if serializable is True, then we should only update with information
    that can be coerced to json.
    """
    import watch

    if serializable:
        raise NotImplementedError('we dont do this yet')

    img_iter = ub.ProgIter(dset.imgs.values(),
                           total=len(dset.imgs),
                           desc='update meta',
                           verbose=1)
    for img in img_iter:

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

        fname = img.get('file_name', None)
        if fname is not None:
            src_gpath = dset.get_image_fpath(img['id'])
            assert exists(src_gpath)
            # img_iter.ensure_newline()
            # print('src_gpath = {!r}'.format(src_gpath))
            img_info = watch.gis.geotiff.geotiff_metadata(src_gpath, **metakw)

            if serializable:
                raise NotImplementedError
            else:
                # info['datetime_acquisition'] = img['datetime_acquisition']
                # info['gpath'] = src_gpath
                img_info = ub.dict_isect(img_info, keys_of_interest)
                img['geotiff_metadata'] = img_info

        for aux in img.get('auxiliary', []):
            aux_fpath = join(dset.bundle_dpath, aux['file_name'])
            assert exists(aux_fpath)
            aux_info = watch.gis.geotiff.geotiff_metadata(aux_fpath, **metakw)
            aux_info = ub.dict_isect(aux_info, keys_of_interest)
            if serializable:
                raise NotImplementedError
            else:
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
            img['geotiff_metadata'] = aux['geotiff_metadata']


def find_roi_regions(dset):
    """
    Given a dataset find spatial regions of interest that contain annotations
    """
    aid_to_poly = {}
    for aid, ann in dset.anns.items():
        geo = _fix_geojson_poly(ann['segmentation_geos'])
        latlon = np.array(geo['coordinates'][0])[:, ::-1]
        kw_poly = kwimage.structs.Polygon(exterior=latlon)
        aid_to_poly[aid] = kw_poly.to_shapely()

    gid_to_rois = {}
    for gid, aids in dset.index.gid_to_aids.items():
        if len(aids):
            sh_annot_polys = ub.dict_subset(aid_to_poly, aids)
            sh_annot_polys_ = [p.buffer(0) for p in sh_annot_polys.values()]
            sh_annot_polys_ = [p.buffer(0.000001) for p in sh_annot_polys_]

            # What CRS should we be doing this in? Is WGS84 OK?
            # Should we switch to UTM?
            img_rois_ = ops.cascaded_union(sh_annot_polys_)
            try:
                img_rois = list(img_rois_)
            except Exception:
                img_rois = [img_rois_]

            kw_img_rois = [
                kwimage.Polygon.from_shapely(p.convex_hull).bounding_box().to_polygons()[0]
                for p in img_rois]
            sh_img_rois = [p.to_shapely() for p in kw_img_rois]
            gid_to_rois[gid] = sh_img_rois

    # TODO: if there are only midly overlapping regions, we should likely split
    # them up. We can also group by UTM coordinates to reduce computation.
    sh_rois_ = ops.cascaded_union([
        p.buffer(0) for rois in gid_to_rois.values()
        for p in rois
    ])
    try:
        sh_rois = list(sh_rois_)
    except Exception:
        sh_rois = [sh_rois_]

    kw_rois = list(map(kwimage.Polygon.from_shapely, sh_rois))
    return sh_rois, kw_rois


def find_covered_regions(dset):
    """
    Find the intersection of all image bounding boxes in world space
    to see what spatial regions are covered by the imagery.
    """
    gid_to_poly = {}
    for gid, img in dset.imgs.items():
        info  = img['geotiff_metadata']
        kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
        sh_img_poly = kw_img_poly.to_shapely()
        gid_to_poly[gid] = sh_img_poly

    coverage_rois_ = ops.cascaded_union(gid_to_poly.values())
    try:
        coverage_rois = list(coverage_rois_)
    except Exception:
        coverage_rois = [coverage_rois_]
    return coverage_rois


def visualize_rois(dset, kw_all_box_rois):
    """
    matplotlib visualization of image and annotation regions on a world map

    Developer function, unused in the script
    """
    sh_all_box_rois = [p.to_shapely()for p in  kw_all_box_rois]
    sh_coverage_rois = find_covered_regions(dset)

    def flip_xy(poly):
        if hasattr(poly, 'reorder_axes'):
            new_poly = poly.reorder_axes((1, 0))
        else:
            kw_poly = kwimage.Polygon.from_shapely(poly)
            kw_poly.data['exterior'].data = kw_poly.data['exterior'].data[:, ::-1]
            sh_poly_ = kw_poly.to_shapely()
            new_poly = sh_poly_
        return new_poly

    sh_all_box_rois_trad = [flip_xy(p) for p in sh_all_box_rois]
    kw_all_box_rois_trad = list(map(kwimage.Polygon.from_shapely, sh_all_box_rois_trad))

    sh_coverage_rois_trad = [flip_xy(p) for p in sh_coverage_rois]
    kw_coverage_rois_trad = list(map(kwimage.Polygon.from_shapely, sh_coverage_rois_trad))

    print('kw_all_box_rois_trad = {}'.format(ub.repr2(kw_all_box_rois_trad, nl=1)))
    print('kw_coverage_rois_trad = {}'.format(ub.repr2(kw_coverage_rois_trad, nl=1)))

    if True:
        import kwplot
        import geopandas as gpd
        kwplot.autompl()

        wld_map_gdf = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )

        poly_crs = {'init': 'epsg:4326'}
        roi_poly_gdf = gpd.GeoDataFrame({'roi_polys': sh_all_box_rois_trad}, geometry='roi_polys', crs=poly_crs)
        # img_poly_gdf = gpd.GeoDataFrame({'img_polys': list(map(flip_xy, gid_to_poly.values()))}, geometry='img_polys', crs=poly_crs)
        cov_poly_gdf = gpd.GeoDataFrame({'cov_rois': sh_coverage_rois_trad}, geometry='cov_rois', crs=poly_crs)

        roi_centroids = roi_poly_gdf.geometry.centroid
        # img_centroids = img_poly_gdf.geometry.centroid
        cov_centroids = cov_poly_gdf.geometry.centroid

        ax = wld_map_gdf.plot()
        cov_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='green', alpha=0.5)
        # img_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
        roi_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)

        cov_centroids.plot(ax=ax, marker='o', facecolor='green', alpha=0.5)
        # img_centroids.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)
        roi_centroids.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)

        kw_zoom_roi = kw_all_box_rois_trad[1]
        kw_zoom_roi = kw_coverage_rois_trad[2]
        kw_zoom_roi = kw_all_box_rois_trad[3]

        bb = kw_zoom_roi.bounding_box()

        min_x, min_y, max_x, max_y = bb.to_ltrb().data[0]
        padx = (max_x - min_x) * 0.5
        pady = (max_y - min_y) * 0.5

        ax.set_xlim(min_x - padx, max_x + padx)
        ax.set_ylim(min_y - pady, max_y + pady)


def _fix_geojson_poly(geo):
    """
    We were given geojson polygons with one fewer layers of nesting than
    the spec allows for. Fix this.

    Example:
        >>> geo1 = kwimage.Polygon.random().to_geojson()
        >>> fixed1 = _fix_geojson_poly(geo1)
        >>> #
        >>> geo2 = {'type': 'Polygon', 'coordinates': geo1['coordinates'][0]}
        >>> fixed2 = _fix_geojson_poly(geo2)
        >>> assert fixed1 == fixed2
        >>> assert fixed1 == geo1
        >>> assert fixed2 != geo2
    """
    def check_leftmost_depth(data):
        # quick check leftmost depth of a nested struct
        item = data
        depth = 0
        while isinstance(item, list):
            if len(item) == 0:
                raise Exception('no child node')
            item = item[0]
            depth += 1
        return depth
    if geo['type'] == 'Polygon':
        data = geo['coordinates']
        depth = check_leftmost_depth(data)
        if depth == 2:
            # correctly format by adding the outer nesting
            fixed = geo.copy()
            fixed['coordinates'] = [geo['coordinates']]
        elif depth == 3:
            # already correct
            fixed = geo
        else:
            raise Exception(depth)
    else:
        fixed = geo
    return fixed


def _aligncrop(obj, bundle_dpath, name, sensor_coarse, dst_dpath, space_region,
               space_box, align_method, is_multi_image):
    # NOTE: https://github.com/dwtkns/gdal-cheat-sheet
    latmin, lonmin, latmax, lonmax = space_box.data[0]

    if is_multi_image:
        # obj.get('channels', None)
        multi_dpath = ub.ensuredir((dst_dpath, name))
        dst_gpath = join(multi_dpath, name + '_' + obj['channels'] + '.tif')
    else:
        dst_gpath = join(dst_dpath, name + '.tif')

    fname = obj.get('file_name', None)
    assert fname is not None
    src_gpath = join(bundle_dpath, fname)

    dst = {
        'file_name': dst_gpath,
    }

    if align_method == 'pixel_crop':
        align_method = 'pixel_crop'
        from ndsampler.utils.util_gdal import LazyGDalFrameFile
        imdata = LazyGDalFrameFile(src_gpath)
        info = obj['geotiff_metadata']
        space_region_pxl = space_region.warp(info['wgs84_to_wld']).warp(info['wld_to_pxl'])
        pxl_xmin, pxl_ymin, pxl_xmax, pxl_ymax = space_region_pxl.bounding_box().to_ltrb().quantize().data[0]
        sl = tuple([slice(pxl_ymin, pxl_ymax), slice(pxl_xmin, pxl_xmax)])
        subim, transform = kwimage.padded_slice(
            imdata, sl, return_info=True)
        kwimage.imwrite(dst_gpath, subim, space=None, backend='gdal')
        dst['img_shape'] = subim.shape
        dst['transform'] = transform
        # TODO: do this with a gdal command so the tiff metdata is preserved

    elif align_method == 'orthorectify':
        # HACK TO FIND an appropirate DEM file
        # from watch.gis import elevation
        # dems = elevation.girder_gtop30_elevation_dem()
        info = obj['geotiff_metadata']
        rpcs = info['rpc_transform']
        dems = rpcs.elevation

        # TODO: reproject to utm
        # https://gis.stackexchange.com/questions/193094/can-gdalwarp-reproject-from-espg4326-wgs84-to-utm
        # '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs'

        if hasattr(dems, 'find_reference_fpath'):
            dem_fpath, dem_info = dems.find_reference_fpath(latmin, lonmin)
            template = ub.paragraph(
                '''
                gdalwarp
                -te {xmin} {ymin} {xmax} {ymax}
                -te_srs epsg:4326
                -t_srs epsg:4326
                -rpc -et 0
                -to RPC_DEM={dem_fpath}
                -overwrite
                {SRC} {DST}
                ''')
        else:
            dem_fpath = None
            template = ub.paragraph(
                '''
                gdalwarp
                -te {xmin} {ymin} {xmax} {ymax}
                -te_srs epsg:4326
                -t_srs epsg:4326
                -rpc -et 0
                -overwrite
                {SRC} {DST}
                ''')
        command = template.format(
            ymin=latmin,
            xmin=lonmin,
            ymax=latmax,
            xmax=lonmax,

            dem_fpath=dem_fpath,
            SRC=src_gpath, DST=dst_gpath,
        )
        cmd_info = ub.cmd(command, verbose=0)  # NOQA
    elif align_method == 'affine_warp':
        template = (
            'gdalwarp '
            '-te {xmin} {ymin} {xmax} {ymax} '
            '-te_srs epsg:4326 '
            '-overwrite '
            '{SRC} {DST}')
        command = template.format(
            ymin=latmin,
            xmin=lonmin,
            ymax=latmax,
            xmax=lonmax,
            SRC=src_gpath, DST=dst_gpath,
        )
        cmd_info = ub.cmd(command, verbose=0)  # NOQA
    else:
        raise KeyError(align_method)

    return dst


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/coco_align_geotiffs.py --help
    """
    main()
