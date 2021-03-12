"""
Given the raw data in kwcoco format, this script will extract orthorectified
regions around areas of interest across time.

Notes:

    # Given the output from geojson_to_kwcoco this script extracts
    # orthorectified regions.

    python ~/code/watch/scripts/coco_align_geotiffs.py \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2 \
            --context_factor=1.5

    # Archive the data and upload to data.kitware.com
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    7z a ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2.zip ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2

    source $HOME/internal/secrets
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    girder-client --api-url https://data.kitware.com/api/v1 upload 602c3e9e2fa25629b97e5b5e drop0_aligned_v2.zip
"""
import scriptconfig as scfg
import ubelt as ub
import kwcoco
import numpy as np
import kwimage
from shapely import ops
from os.path import join, exists


class CocoAlignGeotiffConfig(scfg.Config):
    """
    Create an aligned dataset around objects of interest


    High Level Steps:
        * Find a set of geospatial AOIs, these can be positive or negative
        * For each AOI find all images that overlap
        * Crop that AOI out of each image, and warp its annotations

    Details:
        * What is the best way to select AOIs, do we need to ensure
          we don't go beyond a specific size / scale?

    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'context_factor': scfg.Value(1.5, help=ub.paragraph(
            '''
            scale factor for the clustered ROIs.
            Amount of context to extract around each ROI.
            '''
        ))
    }


def main(**kw):
    """

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
    """
    import socket
    import os
    config = CocoAlignGeotiffConfig(default=kw, cmdline=True)

    src_fpath = config['src']
    dst_dpath = config['dst']
    context_factor = config['context_factor']

    output_bundle_dpath = dst_dpath

    # Load the dataset and extract geotiff metadata from each image.
    dset = kwcoco.CocoDataset(src_fpath)
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
    space_region = kw_all_box_rois[1]
    print('kw_all_box_rois = {!r}'.format(kw_all_box_rois))

    # Create a new dataset that we will extend as we extract ROIs
    new_dset = kwcoco.CocoDataset()

    # Store that this dataset is a result of a process.
    # Note what the process is, what its arguments are, and where the process
    # was executed.
    process_info = {
        'type': 'process',
        'properties': {
            'name': 'coco_align_geotiffs',
            'args': config.to_dict(),
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'timestamp': ub.timestamp(),
        }
    }
    print('process_info = {}'.format(ub.repr2(process_info, nl=2)))

    new_dset.dataset['info'] = [
        process_info,
    ]

    time_region = None
    rpc_align_method = 'orthorectify'

    space_region = kw_all_box_rois[-3]

    to_extract = []
    for space_region in ub.ProgIter(kw_all_box_rois, desc='query overlaps', verbose=3):
        # image_overlaps = cube.query_image_overlaps()
        image_overlaps = cube.query_image_overlaps(space_region, time_region)
        to_extract.append(image_overlaps)

    for image_overlaps in ub.ProgIter(to_extract, desc='extract ROI videos', verbose=3):
        space_roi_string = image_overlaps['space_roi_string']
        print('space_roi_string = {!r}'.format(space_roi_string))

        sub_bundle_dpath = join(extract_dpath, space_roi_string)
        print('sub_bundle_dpath = {!r}'.format(sub_bundle_dpath))
        # sub_bundle_dpath = ub.ensuredir((extract_dpath, space_roi_string))

        cube.extract_overlaps(image_overlaps, extract_dpath,
                              rpc_align_method=rpc_align_method,
                              new_dset=new_dset)

    new_dset.fpath = join(extract_dpath, 'data.kwcoco.json')
    print('Dumping new_dset.fpath = {!r}'.format(new_dset.fpath))
    new_dset.reroot(new_root=output_bundle_dpath, absolute=False)
    new_dset.dump(new_dset.fpath, newlines=True)
    print('finished')


class SimpleDataCube(object):
    """
    Given a dataset containing geotiffs, provide a simple API to extract a
    region in some coordinate space.
    """

    def __init__(cube, dset):
        cube.dset = dset

        cube.gid_to_poly = {}
        for gid, img in cube.dset.imgs.items():
            info = img['geotiff_metadata']
            kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
            sh_img_poly = kw_img_poly.to_shapely()
            cube.gid_to_poly[gid] = sh_img_poly

    def query_image_overlaps(cube, space_region, time_region):
        assert time_region is None, 'for now'
        assert time_region is None, 'for now'
        space_box = space_region.bounding_box().to_ltrb()

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        # from watch.utils.util_place import conv_lat_lon
        # min_pt = conv_lat_lon(str(ymin), str(xmin), format='ISO-D')
        # max_pt = conv_lat_lon(str(ymax), str(xmax), format='ISO-D')

        latmin_str = '{:+2.4f}'.format(latmin).replace('+', 'N').replace('-', 'S')
        lonmin_str = '{:+3.4f}'.format(lonmin).replace('+', 'E').replace('-', 'W')
        latmax_str = '{:+2.4f}'.format(latmax).replace('+', 'N').replace('-', 'S')
        lonmax_str = '{:+3.4f}'.format(lonmax).replace('+', 'E').replace('-', 'W')
        min_pt = '{}{}'.format(latmin_str, lonmin_str)
        max_pt = '{}{}'.format(latmax_str, lonmax_str)
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

        # num_obs_per_date = list(ub.map_vals(len, date_to_gids).values())
        # print('num_obs_per_date = {!r}'.format(num_obs_per_date))

        # Hueristic: if all of the images are in the same folder, then that
        # folder might have an ROI name, so include that in the output.
        # all_gids = list(ub.flatten(date_to_gids.values()))

        # all_fpaths = [
        #     cube.dset.get_image_fpath(gid)
        #     for gid in all_gids
        # ]

        rel_prefix = None
        # if True:
        #     # HACK, very fragile hueristic
        #     candidates = []
        #     for p in all_fpaths:
        #         try:
        #             parts = p.split('/')
        #             candidates.append(parts[parts.index('drop0') + 1])
        #         except Exception:
        #             pass
        #     candidates = set(candidates)
        #     if len(candidates) == 1:
        #         rel_prefix = ub.peek(candidates)

        # all_fnames = [
        #     relpath(abspath(cube.dset.get_image_fpath(gid)), abspath(cube.dset.bundle_dpath))
        #     for gid in all_gids
        # ]
        # # img['file_name'] for img in cube.dset.imgs.values()]
        # # commonprefix(all_fnames)
        # abs_prefix = commonprefix(all_fnames)
        # print('abs_prefix = {!r}'.format(abs_prefix))
        # rel_prefix = relpath(abspath(abs_prefix), realpath(cube.dset.bundle_dpath))

        if rel_prefix:
            space_roi_string = '{}_{}'.format(rel_prefix, space_str)
        else:
            space_roi_string = space_str

        image_overlaps = {
            'date_to_gids': date_to_gids,
            'space_region': space_region,
            'space_str': space_str,
            'space_box': space_box,
            'space_roi_string': space_roi_string,
        }
        return image_overlaps

    def extract_overlaps(cube, image_overlaps, extract_dpath,
                         rpc_align_method='orthorectify', new_dset=None):

        date_to_gids = image_overlaps['date_to_gids']
        space_str = image_overlaps['space_str']
        space_box = image_overlaps['space_box']
        space_region = image_overlaps['space_region']
        space_roi_string = image_overlaps['space_roi_string']

        sub_bundle_dpath = ub.ensuredir((extract_dpath, space_roi_string))

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        dates = sorted(date_to_gids)

        new_video = {
            'name': space_roi_string,
        }

        if new_dset is None:
            new_dset = kwcoco.CocoDataset()
        new_vidid = new_dset.add_video(**new_video)

        frame_index = 0

        sub_new_gids = []

        for cat in cube.dset.cats.values():
            new_dset.ensure_category(**cat)

        DRAW_VIEW = True

        for date in ub.ProgIter(dates, desc='extracting regions', verbose=3):
            gids = date_to_gids[date]
            iso_time = date.strftime('%Y-%m-%d')

            # Is there any consideration we should make to handle this?
            # if len(gids) != 1:
            #     import warnings
            #     warnings.warn('multiple observations at same time')

            for num, gid in enumerate(gids):
                img = cube.dset.imgs[gid]

                # Construct a name for the subregion to extract.
                sensor_coarse = img['sensor_coarse']
                name_string = 'crop_{}_{}_{}_{}.tif'.format(iso_time, space_str, sensor_coarse, num)

                aids = cube.dset.index.gid_to_aids[img['id']]
                anns = list(ub.take(cube.dset.index.anns, aids))
                src_gpath = cube.dset.get_image_fpath(img['id'])

                info = img['geotiff_metadata']

                # NOTE: https://github.com/dwtkns/gdal-cheat-sheet
                if info['is_rpc']:
                    # align_method = 'pixel_crop'
                    align_method = rpc_align_method

                    if align_method == 'pixel_crop':
                        align_method = 'pixel_crop'
                        from ndsampler.utils.util_gdal import LazyGDalFrameFile
                        imdata = LazyGDalFrameFile(src_gpath)
                        # space_region = space_box.to_polygons()[0]
                        space_region_pxl = space_region.warp(info['wgs84_to_wld']).warp(info['wld_to_pxl'])
                        pxl_xmin, pxl_ymin, pxl_xmax, pxl_ymax = space_region_pxl.bounding_box().to_ltrb().quantize().data[0]
                        sl = tuple([slice(pxl_ymin, pxl_ymax), slice(pxl_xmin, pxl_xmax)])
                        subim, transform = kwimage.padded_slice(
                            imdata, sl, return_info=True)

                        dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))
                        dst_gpath = join(dst_dpath, name_string)

                        kwimage.imwrite(dst_gpath, subim, space=None, backend='gdal')

                        dst_info = {
                            'img_shape': subim.shape,
                        }
                    elif align_method == 'orthorectify':
                        align_method = 'orthorectify'

                        dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))
                        dst_gpath = join(dst_dpath, name_string)

                        # HACK TO FIND an appropirate DEM file
                        # from watch.gis import elevation
                        # dems = elevation.girder_gtop30_elevation_dem()
                        rpcs = info['rpc_transform']
                        dems = rpcs.elevation
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
                    else:
                        raise KeyError(align_method)

                else:
                    align_method = 'affine_warp'
                    dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))
                    dst_gpath = join(dst_dpath, name_string)

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

                if 0:
                    _ = ub.cmd('gdalinfo {}'.format(dst_gpath), verbose=3)
                    _ = ub.cmd('gdalinfo {}'.format(src_gpath), verbose=3)

                if align_method != 'pixel_crop':
                    # Re-parse any information in the new geotiff
                    from watch.gis.geotiff import geotiff_metadata
                    dst_info = geotiff_metadata(dst_gpath)
                    dst_info['wgs84_corners']
                    # info['wgs84_corners']

                new_img = {}
                # Carry over appropriate metadata from original image
                new_img.update(ub.dict_isect(img, {
                    'date_captured',
                    'approx_elevation',
                    'approx_meter_gsd',
                    'sensor_candidates',
                    'num_bands',
                    'sensor_coarse',
                    'site_tag',
                    # 'datetime_acquisition',
                }))
                new_img['parent_file_name'] = img['file_name']  # remember which image this came from
                new_img['width'] = dst_info['img_shape'][1]
                new_img['height'] = dst_info['img_shape'][0]
                new_img['file_name'] = dst_gpath
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
                    dset = cube.dset

                    orig_pxl_poly_list = []
                    for ann in anns:
                        old_poly = kwimage.Polygon.from_coco(ann['segmentation'])
                        orig_pxl_poly_list.append(old_poly)
                    orig_pxl_polys = kwimage.MultiPolygon(orig_pxl_poly_list)

                    geo_poly_list = []
                    for ann in anns:
                        # Q: WHAT FORMAT ARE THESE COORDINATES IN?
                        # A: I'm fairly sure these coordinates are all Traditional-WGS84-Lon-Lat
                        # We convert them to authority compliant WGS84 (lat-lon)
                        exterior = kwimage.Coords(np.array(ann['segmentation_geos']['coordinates'])[:, ::-1])
                        geo_poly = kwimage.Polygon(exterior=exterior)
                        geo_poly_list.append(geo_poly)
                    geo_polys = kwimage.MultiPolygon(geo_poly_list)

                    if align_method == 'orthorectify':
                        # Is the affine mapping in the destination image good
                        # enough after the image has been orthorectified?
                        pxl_polys = geo_polys.warp(dst_info['wgs84_to_wld']).warp(dst_info['wld_to_pxl'])
                    elif align_method == 'pixel_crop':
                        yoff, xoff = transform['st_offset']
                        pxl_polys = orig_pxl_polys.translate((-xoff, -yoff))
                    elif align_method == 'affine_warp':
                        # Warp Auth-WGS84 to whatever the image world space is,
                        # and then from there to pixel space.
                        pxl_polys = geo_polys.warp(dst_info['wgs84_to_wld']).warp(dst_info['wld_to_pxl'])
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

                if DRAW_VIEW:
                    # See if we can look at what we made
                    from watch.utils.util_norm import normalize_intensity
                    canvas = kwimage.imread(dst_gpath)
                    canvas = normalize_intensity(canvas)
                    canvas = kwimage.ensure_float01(canvas)

                    view_img_dpath = ub.ensuredir(
                        (sub_bundle_dpath, sensor_coarse,
                         '_view_img_' + align_method))

                    if HANDLE_ANNS:
                        view_ann_dpath = ub.ensuredir(
                            (sub_bundle_dpath, sensor_coarse,
                             '_view_img_' + align_method))

                    if 0:
                        import kwplot
                        kwplot.autompl()
                        kwplot.imshow(canvas)

                    view_img_fpath = ub.augpath(dst_gpath, dpath=view_img_dpath) + '.view_img.jpg'
                    kwimage.imwrite(view_img_fpath, kwimage.ensure_uint255(canvas))

                    if HANDLE_ANNS:
                        dets = kwimage.Detections.from_coco_annots(valid_anns, dset=dset)
                        view_ann_fpath = ub.augpath(dst_gpath, dpath=view_ann_dpath) + '.view_ann.jpg'
                        ann_canvas = dets.draw_on(canvas)
                        kwimage.imwrite(view_ann_fpath, kwimage.ensure_uint255(ann_canvas))

        if True:
            print('Writing data subset')
            sub_dset = new_dset.subset(sub_new_gids, copy=True)
            sub_dset.fpath = join(sub_bundle_dpath, 'subdata.kwcoco.json')
            sub_dset.reroot(new_root=sub_bundle_dpath, absolute=False)
            sub_dset.dump(sub_dset.fpath, newlines=True)

    def extract(cube, space_region, time_region, extract_dpath,
                rpc_align_method='orthorectify', new_dset=None):
        """
        cube = SimpleDataCube(dset)
        space_region = kw_all_box_rois[2]
        extract_dpath = ub.ensuredir('test-extract')
        for space_region in kw_all_box_rois:
            extract(cube, space_region, None, extract_dpath)
        """
        image_overlaps = cube.query_image_ids(space_region, time_region)
        cube.extract_overlaps(image_overlaps, extract_dpath,
                              rpc_align_method=rpc_align_method,
                              new_dset=new_dset)


def update_coco_geotiff_metadata(dset, serializable=True):
    """
    if serializable is True, then we should only update with information
    that can be coerced to json.
    """

    from watch.gis.geotiff import geotiff_metadata
    import datetime

    assert not serializable, 'we dont do this yet'
    img_iter = ub.ProgIter(dset.imgs.values(),
                           total=len(dset.imgs),
                           desc='update meta',
                           verbose=1)
    for img in img_iter:

        img['datetime_acquisition'] = (
            datetime.datetime.strptime(img['date_captured'], '%Y/%m/%d')
        )

        src_gpath = dset.get_image_fpath(img['id'])
        assert exists(src_gpath)

        if img.get('dem_hint', 'use') == 'ignore':
            info = geotiff_metadata(src_gpath, elevation=0)
        else:
            info = geotiff_metadata(src_gpath)

        # relevant_keys = [
        #     'utm_crs_info',
        #     'utm_corners',
        #     'wgs84_crs_info',
        #     'wgs84_corners',
        # ]
        # relevant = ub.dict_isect(info, relevant_keys)
        # relevant['utm_corners']
        if serializable:
            raise NotImplementedError
        else:
            info['datetime_acquisition'] = img['datetime_acquisition']
            info['gpath'] = src_gpath
            img['geotiff_metadata'] = info


def find_roi_regions(dset):
    """
    Given a dataset find spatial regions of interest that contain annotations
    """
    aid_to_poly = {}
    for aid, ann in dset.anns.items():
        latlon = np.array(ann['segmentation_geos']['coordinates'])[:, ::-1]
        kw_poly = kwimage.structs.Polygon(exterior=latlon)
        aid_to_poly[aid] = kw_poly.to_shapely()

    gid_to_rois = {}
    for gid, aids in dset.index.gid_to_aids.items():
        if len(aids):
            sh_annot_polys = ub.dict_subset(aid_to_poly, aids)
            sh_annot_polys_ = [p.buffer(0) for p in sh_annot_polys.values()]
            sh_annot_polys_ = [p.buffer(0.000001) for p in sh_annot_polys_]

            # What CRS should we be doing this in? Is WGS84 OK?

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
        # infos = dset.images(gids).lookup('geotiff_metadata')
        # gid_to_aids = ub.dict_isect(dset.index.gid_to_aids, gids)
        # for gid, info in zip(gids, infos):
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
    """
    sh_all_box_rois = [p.to_shapely()for p in  kw_all_box_rois]
    sh_coverage_rois = find_covered_regions(dset)

    # Group Images
    # utmzone_to_imgs = ub.group_items(dset.imgs.values(), lambda img: 'img_group_' + ub.hash_data(img['geotiff_metadata']['utm_crs_info']))
    # utmzone_to_imgs.keys()

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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/coco_align_geotiffs.py
    """
    main()
