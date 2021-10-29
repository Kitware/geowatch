"""
Given the raw data in kwcoco format, this script will extract orthorectified
regions around areas of intere/t across time.

Notes:

    # Given the output from geojson_to_kwcoco this script extracts
    # orthorectified regions.

    # https://data.kitware.com/#collection/602457272fa25629b95d1718/folder/602c3e9e2fa25629b97e5b5e

    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2 \
            --context_factor=1.5

    # Archive the data and upload to data.kitware.com
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    7z a ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2.zip \
            ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2

    stamp=$(date +"%Y-%m-%d")
    # resolve links (7z cant handl)
    rsync -avpL drop0_aligned_v2 drop0_aligned_v2_$stamp
    7z a drop0_aligned_v2_$stamp.zip drop0_aligned_v2_$stamp

    source $HOME/internal/secrets
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    girder-client --api-url https://data.kitware.com/api/v1 upload \
            602c3e9e2fa25629b97e5b5e drop0_aligned_v2_$stamp.zip

    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi \
            --context_factor=1.5


    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi_big \
            --context_factor=3.5


Test:

    There was a bug in KR-WV, run the script only on that region to test if we
    have fixed it.

    kwcoco stats ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    jq .images ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    kwcoco subset ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json --gids=1129,1130 --dst ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json

    python -m watch.cli.coco_align_geotiffs \
            --src ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json \
            --dst ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0_aligned_WV_Fix \
            --rpc_align_method pixel_crop \
            --context_factor=3.5

           # --src ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json \

    TODO:
        - [ ] Add method for extracting "negative ROIs" that are nearby
            "positive ROIs".
"""
import kwcoco
import kwimage
import os
import scriptconfig as scfg
import socket
import ubelt as ub
import dateutil.parser
import datetime
import pathlib
from os.path import join, exists
from watch.cli.coco_visualize_videos import _write_ann_visualizations2
from watch.utils import util_gis
from watch.utils import kwcoco_extensions  # NOQA


class CocoAlignGeotiffConfig(scfg.Config):
    """
    Create a dataset of aligned temporal sequences around objects of interest
    in an unstructured collection of annotated geotiffs.

    High Level Steps:
        * Find a set of geospatial AOIs
        * For each AOI find all images that overlap
        * Orthorectify (or warp) the selected spatial region and its
          annotations to a cannonical space.
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory or kwcoco json file for the output'),

        'max_workers': scfg.Value(4, help='number of parallel procs'),
        'aux_workers': scfg.Value(4, help='additional inner threads for aux imgs'),

        'context_factor': scfg.Value(1.0, help=ub.paragraph(
            '''
            scale factor for the clustered ROIs.
            Amount of context to extract around each ROI.
            '''
        )),

        'regions': scfg.Value('annots', help=ub.paragraph(
            '''
            Strategy for extracting regions, if annots, uses the convex hulls
            of clustered annotations. Can also be a path to a geojson file
            to use pre-defined regions.
            ''')),

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

        'visualize': scfg.Value(False, help=ub.paragraph(
            '''
            if True, normalize and draw image / annotation sequences when
            extracting.
            '''
        )),

        'keep': scfg.Value('none', help=ub.paragraph(
            '''
            Level of detail to overwrite existing data at, since this is slow.
            "none": overwrite all, including existing images
            "img": only add new images
            "roi": only add new ROIs
            '''
        )),

        'target_gsd': scfg.Value(10, help=ub.paragraph('initial gsd to use')),
    }


def main(cmdline=True, **kw):
    """
    Main function for coco_align_geotiffs.
    See :class:``CocoAlignGeotiffConfig` for details

    Ignore:
        from watch.cli.coco_align_geotiffs import *  # NOQA
        import kwcoco
        src = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json')
        dst = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned')
        kw = {
            'src': src,
            'dst': dst,
        }

    Example:
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> from watch.gis.geotiff import geotiff_metadata
        >>> # Create a dead simple coco dataset with one image
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset()
        >>> ls_prod = grab_landsat_product()
        >>> fpath = ls_prod['bands'][0]
        >>> meta = geotiff_metadata(fpath)
        >>> # We need a date captured ATM in a specific format
        >>> dt = dateutil.parser.parse(
        >>>     meta['filename_meta']['acquisition_date'])
        >>> date_captured = dt.strftime('%Y/%m/%d')
        >>> gid = coco_dset.add_image(file_name=fpath, date_captured=date_captured)
        >>> dummy_poly = kwimage.Polygon(exterior=meta['wgs84_corners'])
        >>> dummy_poly = dummy_poly.scale(0.3, about='center')
        >>> sseg_geos = dummy_poly.swap_axes().to_geojson()
        >>> # NOTE: script is not always robust to missing annotation
        >>> # information like segmentation and bad bbox, but for this
        >>> # test config it is
        >>> coco_dset.add_annotation(
        >>>     image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)
        >>> #
        >>> # Create arguments to the script
        >>> dpath = ub.ensure_app_cache_dir('smart_watch/test/coco_align_geotiff')
        >>> dst = ub.ensuredir((dpath, 'align_bundle1'))
        >>> ub.delete(dst)
        >>> dst = ub.ensuredir(dst)
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': 'annots',
        >>>     'max_workers': 0,
        >>>     'aux_workers': 0,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)

    Example:
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> coco_dset = demo_kwcoco_with_heatmaps(num_videos=2, num_frames=2)
        >>> # Create arguments to the script
        >>> dpath = ub.ensure_app_cache_dir('smart_watch/test/coco_align_geotiff2')
        >>> dst = ub.ensuredir((dpath, 'align_bundle2'))
        >>> ub.delete(dst)
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': 'annots',
        >>>     'max_workers': 2,
        >>>     'aux_workers': 2,
        >>>     'visualize': 1,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)

        df1 = covered_annot_geo_regions(coco_dset)
        df2 = covered_image_geo_regions(coco_dset)
    """
    config = CocoAlignGeotiffConfig(default=kw, cmdline=cmdline)

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
    dst = config['dst']
    regions = config['regions']
    context_factor = config['context_factor']
    rpc_align_method = config['rpc_align_method']
    visualize = config['visualize']
    write_subsets = config['write_subsets']
    max_workers = config['max_workers']
    aux_workers = config['aux_workers']
    keep = config['keep']
    target_gsd = config['target_gsd']

    dst = pathlib.Path(ub.expandpath(dst))
    # TODO: handle this coercion of directories or bundles in kwcoco itself
    if 'json' in dst.name.split('.'):
        output_bundle_dpath = str(dst.parent)
        dst_fpath = str(dst)
    else:
        output_bundle_dpath = str(dst)
        dst_fpath = str(dst / 'data.kwcoco.json')

    print('output_bundle_dpath = {!r}'.format(output_bundle_dpath))
    print('dst_fpath = {!r}'.format(dst_fpath))

    region_df = None
    if regions in {'annots', 'images'}:
        pass
    elif exists(regions):
        region_df = util_gis.read_geojson(regions)
    else:
        raise KeyError(regions)

    # Load the dataset and extract geotiff metadata from each image.
    coco_dset = kwcoco.CocoDataset.coerce(src_fpath)
    kwcoco_extensions.coco_populate_geo_heuristics(
        coco_dset, overwrite={'warp'}, workers=max_workers,
        keep_geotiff_metadata=True,
    )
    # update_coco_geotiff_metadata(coco_dset, serializable=False,
    #                              max_workers=max_workers)

    # Construct the "data cube"
    cube = SimpleDataCube(coco_dset)

    # Find the clustered ROI regions
    if regions == 'images':
        region_df = kwcoco_extensions.covered_image_geo_regions(coco_dset)
    elif regions == 'annots':
        region_df = kwcoco_extensions.covered_annot_geo_regions(coco_dset)
    else:
        assert region_df is not None, 'must have been given regions some other way'

    print('query region_df = {!r}'.format(region_df))
    print('cube.img_geos_df = {!r}'.format(cube.img_geos_df))

    # Exapnd the ROI by the context factor and convert to a bounding box
    region_df['geometry'] = region_df['geometry'].apply(shapely_bounding_box)
    if context_factor != 1:
        region_df['geometry'] = region_df['geometry'].scale(
            xfact=context_factor, yfact=context_factor, origin='center')

    # For each ROI extract the aligned regions to the target path
    extract_dpath = ub.ensuredir(output_bundle_dpath)

    # Create a new dataset that we will extend as we extract ROIs
    new_dset = kwcoco.CocoDataset()

    new_dset.dataset['info'] = [
        process_info,
    ]
    to_extract = cube.query_image_overlaps2(region_df)

    for image_overlaps in ub.ProgIter(to_extract, desc='extract ROI videos', verbose=3):
        # tracker.print_diff()
        video_name = image_overlaps['video_name']
        print('video_name = {!r}'.format(video_name))

        sub_bundle_dpath = join(extract_dpath, video_name)
        print('sub_bundle_dpath = {!r}'.format(sub_bundle_dpath))

        new_dset = cube.extract_overlaps(
            image_overlaps, extract_dpath, rpc_align_method=rpc_align_method,
            new_dset=new_dset, visualize=visualize,
            write_subsets=write_subsets, max_workers=max_workers,
            aux_workers=aux_workers, keep=keep, target_gsd=target_gsd)

    new_dset.fpath = dst_fpath
    print('Dumping new_dset.fpath = {!r}'.format(new_dset.fpath))
    new_dset.reroot(new_root=output_bundle_dpath, absolute=False)
    new_dset.dump(new_dset.fpath, newlines=True)
    print('finished')
    return new_dset


class SimpleDataCube(object):
    """
    Given a CocoDataset containing geotiffs, provide a simple API to extract a
    region in some coordinate space.

    Intended usage is to use :func:`query_image_overlaps` to find images that
    overlap an ROI, then then :func:`extract_overlaps` to warp spatial subsets
    of that data into an aligned temporal sequence.
    """

    def __init__(cube, coco_dset):
        # old way: gid_to_poly is old and should be deprecated
        import geopandas as gpd
        import shapely
        gid_to_poly = {}

        expxected_geos_crs_info = {
            'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
            'auth': ('EPSG', '4326')
        }

        # new way: put data in the cube into a geopandas data frame
        df_input = []
        for gid, img in coco_dset.imgs.items():
            sh_img_poly = shapely.geometry.shape(img['geos_corners'])
            properties = img['geos_corners'].get('properties', {})
            crs_info = properties.get('crs_info', None)
            if crs_info is not None:
                assert crs_info == expxected_geos_crs_info
            # pass
            # info = img['geotiff_metadata']
            # kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
            # sh_img_poly = kw_img_poly.to_shapely()
            # Create a data frame with space-time regions
            df_input.append({
                'gid': gid,
                'name': img.get('name', None),
                'video_id': img.get('video_id', None),
                'geometry': sh_img_poly,
                'properties': properties,
            })
            # Maintain old way for now
            gid_to_poly[gid] = sh_img_poly

        img_geos_df = gpd.GeoDataFrame(
            df_input, geometry='geometry', crs='epsg:4326')

        cube.coco_dset = coco_dset
        cube.gid_to_poly = gid_to_poly
        cube.img_geos_df = img_geos_df

    @classmethod
    def demo(SimpleDataCube, num_imgs=1, with_region=False):
        from watch.demo.landsat_demodata import grab_landsat_product
        from watch.gis.geotiff import geotiff_metadata
        # Create a dead simple coco dataset with one image
        import geopandas as gpd
        import kwcoco
        coco_dset = kwcoco.CocoDataset()
        ls_prod = grab_landsat_product()
        fpath = ls_prod['bands'][0]
        meta = geotiff_metadata(fpath)
        # We need a date captured ATM in a specific format
        dt = dateutil.parser.parse(
            meta['filename_meta']['acquisition_date'])
        date_captured = dt.strftime('%Y/%m/%d')

        gid = coco_dset.add_image(file_name=fpath, date_captured=date_captured)
        img_poly = kwimage.Polygon(exterior=meta['wgs84_corners'])
        ann_poly = img_poly.scale(0.1, about='center')
        sseg_geos = ann_poly.swap_axes().to_geojson()
        coco_dset.add_annotation(
            image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)

        # update_coco_geotiff_metadata(coco_dset, serializable=False, max_workers=0)
        kwcoco_extensions.coco_populate_geo_heuristics(
            coco_dset, overwrite={'warp'}, workers=0,
            keep_geotiff_metadata=True,
        )

        cube = SimpleDataCube(coco_dset)
        if with_region:
            img_poly = kwimage.Polygon(exterior=cube.coco_dset.imgs[1]['geotiff_metadata']['wgs84_corners'])
            img_poly.swap_axes().to_geojson()
            region_geojson =  {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'properties': {
                            'type': 'region', 'region_model_id': 'demo_region',
                            'version': '1.0.1', 'mgrs': None,
                            'start_date': None, 'end_date': None},
                        'geometry': img_poly.scale(0.2, about='center').swap_axes().to_geojson(),
                    },
                ]
            }
            region_df = gpd.GeoDataFrame.from_features(region_geojson)
            region_df['geometry'] = region_df['geometry'].apply(shapely_flip_xy)
            return cube, region_df
        return cube

    def query_image_overlaps2(cube, region_df):
        """
        Find the images that overlap with a each space-time region

        For each region, assigns all images that overlap the space-time bounds,
        and constructs arguments to :func:`extract_overlaps`.

        Args:
            region_df (GeoDataFrame): data frame containing all space-time
                region queries.

        Returns:
            dict :
                Information about which images belong to this ROI and their
                temporal sequence. Also contains strings to be used for
                subdirectories in the extract step.

        Example:
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> to_extract = cube.query_image_overlaps2(region_df)
        """
        # New maybe faster and safer way of finding overlaps?
        ridx_to_gidsx = util_gis.geopandas_pairwise_overlaps(region_df, cube.img_geos_df)
        print('ridx_to_gidsx = {}'.format(ub.repr2(ridx_to_gidsx, nl=1)))

        # TODO: maybe check for self-overlap?
        # ridx_to_ridx = util_gis.geopandas_pairwise_overlaps(region_df, region_df)

        to_extract = []
        for ridx, gidxs in ridx_to_gidsx.items():
            region_row = region_df.iloc[ridx]

            space_region = kwimage.Polygon.from_shapely(region_row.geometry)
            space_box = space_region.bounding_box().to_ltrb()
            latmin, lonmin, latmax, lonmax = space_box.data[0]
            min_pt = util_gis.latlon_text(latmin, lonmin)
            max_pt = util_gis.latlon_text(latmax, lonmax)
            space_str = '{}_{}'.format(min_pt, max_pt)

            if region_row.get('type', None) == 'region':
                # Special case where we are extracting a region with a name
                video_name = region_row.get('region_model_id', space_str)
            else:
                video_name = space_str

            if len(gidxs) == 0:
                print('WARNING: No spatial matches to {}'.format(video_name))
            else:

                # TODO: filter dates out of range
                query_start_date = region_row.get('start_date', None)
                query_end_date = region_row.get('end_date', None)

                cand_gids = cube.img_geos_df.iloc[gidxs].gid
                cand_datecaptured = cube.coco_dset.images(cand_gids).lookup('date_captured')
                cand_datetimes = [dateutil.parser.parse(c) for c in cand_datecaptured]

                if 0 and query_start_date is not None:
                    query_start_datetime = dateutil.parser.parse(query_start_date)
                    flags = [dt >= query_start_datetime for dt in cand_datetimes]
                    cand_datetimes = list(ub.compress(cand_datetimes, flags))
                    cand_gids = list(ub.compress(cand_gids, flags))

                if 0 and query_end_date is not None:
                    query_end_datetime = dateutil.parser.parse(query_end_date)
                    flags = [dt <= query_end_datetime for dt in cand_datetimes]
                    cand_datetimes = list(ub.compress(cand_datetimes, flags))
                    cand_gids = list(ub.compress(cand_gids, flags))

                if len(cand_gids) == 0:
                    print('WARNING: No temporal matches to {}'.format(video_name))
                else:
                    date_to_gids = ub.group_items(cand_gids, cand_datetimes)
                    dates = sorted(date_to_gids)
                    print('Found {} overlaps for {} from {} to {}'.format(
                        len(cand_gids),
                        video_name,
                        min(dates).isoformat(),
                        max(dates).isoformat(),
                    ))

                    region_props = ub.dict_diff(
                        region_row.to_dict(), {'geometry'})

                    image_overlaps = {
                        'date_to_gids': date_to_gids,
                        'space_region': space_region,
                        'space_str': space_str,
                        'space_box': space_box,
                        'video_name': video_name,
                        'properties': region_props,
                    }
                    to_extract.append(image_overlaps)
        return to_extract

    def extract_overlaps(cube, image_overlaps, extract_dpath,
                         rpc_align_method='orthorectify', new_dset=None,
                         write_subsets=True, visualize=True, max_workers=0,
                         aux_workers=0, keep='none', target_gsd=10):
        """
        Given a region of interest, extract an aligned temporal sequence
        of data to a specified directory.

        Args:
            image_overlaps (dict): Information about images in an ROI and their
                temporal order computed from :func:``query_image_overlaps2``.

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

            keep (str): Level of detail to overwrite existing data at, since this is slow.
                "none": overwrite all, including existing images
                "img": only add new images
                "roi": only add new ROIs

        Returns:
            kwcoco.CocoDataset: the given or new dataset that was modified

        Example:
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> extract_dpath = ub.ensure_app_cache_dir('smart_watch/test/coco_align_geotiff/demo_extract_overlaps')
            >>> rpc_align_method = 'orthorectify'
            >>> new_dset = kwcoco.CocoDataset()
            >>> write_subsets = True
            >>> visualize = True
            >>> max_workers = 32
            >>> to_extract = cube.query_image_overlaps2(region_df)
            >>> image_overlaps = to_extract[0]
            >>> cube.extract_overlaps(image_overlaps, extract_dpath,
            >>>                       new_dset=new_dset, visualize=visualize,
            >>>                       max_workers=max_workers)
        """
        # import watch
        coco_dset = cube.coco_dset

        date_to_gids = image_overlaps['date_to_gids']
        space_str = image_overlaps['space_str']
        space_box = image_overlaps['space_box']
        space_region = image_overlaps['space_region']
        video_name = image_overlaps['video_name']
        video_props = image_overlaps['properties']

        if new_dset is None:
            new_dset = kwcoco.CocoDataset()

        sub_bundle_dpath = ub.ensuredir((extract_dpath, video_name))

        if exists(join(sub_bundle_dpath,
                       'subdata.kwcoco.json')) and keep == 'roi':
            print('ROI found on disk; adding')
            sub_dset = kwcoco.CocoDataset(
                join(sub_bundle_dpath, 'subdata.kwcoco.json'))
            return new_dset.union(sub_dset)

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        dates = sorted(date_to_gids)

        new_video = {
            'name': video_name,
            'properties': video_props,
        }

        new_vidid = new_dset.add_video(**new_video)

        for cat in coco_dset.cats.values():
            new_dset.ensure_category(**cat)

        bundle_dpath = coco_dset.bundle_dpath
        new_anns = []

        # Manage new ids such that parallelization does not impact their order
        start_gid = new_dset._next_ids.get('images')
        start_aid = new_dset._next_ids.get('annotations')
        frame_index = 0

        # Hueristic to choose if we parallize the inner or outer loop
        # num_imgs_to_warp = sum(map(len, date_to_gids.values()))
        # if num_imgs_to_warp <= max_workers:
        #     img_workers = 0
        #     aux_workers = max_workers
        # else:
        # Internal threading might be beneficial as well
        # aux_workers = 6
        img_workers = max_workers

        # parallelize over images
        # print('img_workers = {!r}'.format(img_workers))
        # print('aux_workers = {!r}'.format(aux_workers))
        pool = ub.JobPool(mode='thread', max_workers=img_workers)

        for date in ub.ProgIter(dates, desc='submit extract jobs', verbose=1):
            gids = date_to_gids[date]
            # TODO: Is there any other consideration we should make when
            # multiple images have the same timestamp?
            for num, gid in enumerate(gids):
                img = coco_dset.imgs[gid]
                anns = [coco_dset.index.anns[aid] for aid in
                        coco_dset.index.gid_to_aids[gid]]
                job = pool.submit(extract_image_job, img, anns, bundle_dpath,
                                  date, num, frame_index, new_vidid,
                                  rpc_align_method, sub_bundle_dpath,
                                  space_str, space_region, space_box,
                                  start_gid, start_aid, aux_workers, (keep == 'img'))
                start_gid = start_gid + 1
                start_aid = start_aid + len(anns)
                frame_index = frame_index + 1

        sub_new_gids = []
        Prog = ub.ProgIter
        # import tqdm
        # Prog = tqdm.tqdm
        for job in Prog(pool.as_completed(), total=len(pool),
                        desc='collect extract jobs'):
            new_img, new_anns = job.result()

            # Hack, the next ids dont update when new images are added
            # with explicit ids. This is a quick fix.
            new_img.pop('id', None)

            new_img['video_id'] = new_vidid

            new_gid = new_dset.add_image(**new_img)
            sub_new_gids.append(new_gid)

            for ann in new_anns:
                ann.pop('id', None)  # quick hack fix
                ann['image_id'] = new_gid
                new_dset.add_annotation(**ann)

        if True:
            for new_gid in sub_new_gids:
                # Fix json serializability
                new_img = new_dset.index.imgs[new_gid]
                new_objs = [new_img] + new_img.get('auxiliary', [])

                # hack
                for obj in new_objs:
                    if 'warp_to_wld' in obj:
                        obj['warp_to_wld'] = kwimage.Affine.coerce(obj['warp_to_wld']).concise()
                    if 'wld_to_pxl' in obj:
                        obj['wld_to_pxl'] = kwimage.Affine.coerce(obj['wld_to_pxl']).concise()
                    obj.pop('wgs84_to_wld', None)
                from kwcoco.util import util_json
                unserializable = list(util_json.find_json_unserializable(new_img))
                if unserializable:
                    raise AssertionError('unserializable = {}'.format(ub.repr2(unserializable, nl=1)))

        kwcoco_extensions.populate_watch_fields(
            new_dset, target_gsd=target_gsd, vidids=[new_vidid], conform=False)

        if visualize:
            for new_gid in sub_new_gids:
                new_img = new_dset.imgs[new_gid]
                new_anns = new_dset.annots(gid=new_gid).objs
                sub_bundle_dpath_ = pathlib.Path(sub_bundle_dpath)
                _write_ann_visualizations2(new_dset, new_img, new_anns,
                                           sub_bundle_dpath_, space='video')

        if write_subsets:
            print('Writing data subset')
            new_dset._check_json_serializable()

            sub_dset = new_dset.subset(sub_new_gids, copy=True)
            sub_dset.fpath = join(sub_bundle_dpath, 'subdata.kwcoco.json')
            sub_dset.reroot(new_root=sub_bundle_dpath, absolute=False)
            sub_dset.dump(sub_dset.fpath, newlines=True)
        return new_dset


def extract_image_job(img, anns, bundle_dpath, date, num, frame_index,
                      new_vidid, rpc_align_method, sub_bundle_dpath, space_str,
                      space_region, space_box, start_gid, start_aid,
                      aux_workers=0, keep=False):
    """
    Threaded worker function for :func:`SimpleDataCube.extract_overlaps`.
    """
    from watch.utils.kwcoco_extensions import _populate_canvas_obj
    from watch.utils.kwcoco_extensions import _recompute_auxiliary_transforms

    iso_time = datetime.date.isoformat(date.date())
    sensor_coarse = img.get('sensor_coarse', 'unknown')

    # Construct a name for the subregion to extract.
    name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)

    auxiliary = img.get('auxiliary', [])

    objs = []
    has_base_image = img.get('file_name', None) is not None
    if has_base_image:
        objs.append(ub.dict_diff(img, {'auxiliary'}))
    objs.extend(auxiliary)

    is_rpcs = [obj['geotiff_metadata']['is_rpc'] for obj in objs]
    if not ub.allsame(is_rpcs):
        # TODO fix this, probably WV from smart-stac and smart-imagery mixed?
        print(objs)
        print(is_rpcs)
        is_rpc = False
    else:
        is_rpc = ub.peek(is_rpcs)

    if is_rpc and rpc_align_method != 'affine_warp':
        align_method = rpc_align_method
        if align_method == 'pixel_crop':
            align_method = 'pixel_crop'
    else:
        align_method = 'affine_warp'

    dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))

    is_multi_image = len(objs) > 1

    job_list = []

    # Turn off internal threading because we refactored this to thread over all
    # images instead
    executor = ub.Executor(mode='thread', max_workers=aux_workers)
    for obj in ub.ProgIter(objs, desc='submit warp auxiliaries', verbose=0):
        job = executor.submit(
            _aligncrop, obj, bundle_dpath, name, sensor_coarse,
            dst_dpath, space_region, space_box, align_method,
            is_multi_image, keep)
        job_list.append(job)

    dst_list = []
    for job in ub.ProgIter(job_list, total=len(job_list),
                           desc='collect warp auxiliaries {}'.format(name),
                           enabled=0):
        dst = job.result()
        dst_list.append(dst)

    if align_method != 'pixel_crop':
        # If we are a pixel crop, we can transform directly
        for dst in dst_list:
            # hack this in for heuristics
            if 'sensor_coarse' in img:
                dst['sensor_coarse'] = img['sensor_coarse']
            # We need to overwrite because we changed the bounds
            # Note: if band info is not popluated above, this
            # might write bad data based on hueristics
            # TODO:
            # We need to remove all spatial metadata from the base image that a
            # crop would invalidate, otherwise we will propogate bad info.
            _populate_canvas_obj(bundle_dpath, dst,
                                 overwrite={'warp'}, with_wgs=True)

    new_gid = start_gid

    new_img = {
        'id': new_gid,
        'name': name,
        'align_method': align_method,
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
        # new_aux['channels'] = old_aux['channels']
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
        'aux_annotated_candidate'
    })

    # Carry over appropriate metadata from original image
    new_img.update(carry_over)
    new_img['parent_file_name'] = img.get('file_name', None)  # remember which image this came from
    new_img['parent_name'] = img.get('name', None)  # remember which image this came from
    new_img['parent_canonical_name'] = img.get('canonical_name', None)  # remember which image this came from
    # new_img['video_id'] = new_vidid  # Done outside of this worker
    new_img['frame_index'] = frame_index
    new_img['timestamp'] = date.toordinal()

    # HANDLE ANNOTATIONS
    """
    It would probably be better to warp pixel coordinates using the
    same transform found by gdalwarp, but I'm not sure how to do
    that. Thus we transform the geocoordinates to the new extracted
    img coords instead. Hopefully gdalwarp preserves metadata
    enough to do this.
    """
    new_anns = []
    geo_poly_list = []
    for ann in anns:
        # Q: WHAT FORMAT ARE THESE COORDINATES IN?
        # A: I'm fairly sure these coordinates are all Traditional-WGS84-Lon-Lat
        # We convert them to authority compliant WGS84 (lat-lon)
        # Hack to support real and orig drop0 geojson
        geo = _fix_geojson_poly(ann['segmentation_geos'])
        geo_poly = kwimage.structs.MultiPolygon.from_geojson(geo).swap_axes()
        geo_poly_list.append(geo_poly)
    geo_polys = kwimage.SegmentationList(geo_poly_list)

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

    def _test_inbounds(pxl_multi_poly):
        is_any = False
        is_all = True
        for pxl_poly in pxl_multi_poly.data:
            xs, ys = pxl_poly.data['exterior'].data.T
            flags_x1 = xs < 0
            flags_y1 = ys < 0
            flags_x2 = xs >= new_img['width']
            flags_y2 = ys >= new_img['height']
            flags = flags_x1 | flags_x2 | flags_y1 | flags_y2
            n_oob = flags.sum()
            is_any &= (n_oob > 0)
            is_all &= (n_oob == len(flags))
        return is_any, is_all

    flags = [not _test_inbounds(p)[1] for p in pxl_polys]

    valid_anns = [ann.copy() for ann in ub.compress(anns, flags)]
    valid_pxl_polys = list(ub.compress(pxl_polys, flags))

    new_aid = start_aid
    for ann, pxl_poly in zip(valid_anns, valid_pxl_polys):
        ann.pop('image_id', None)
        ann['segmentation'] = pxl_poly.to_coco(style='new')
        pxl_box = pxl_poly.bounding_box().quantize().to_xywh()
        xywh = list(pxl_box.to_coco())[0]
        ann['bbox'] = xywh
        ann['image_id'] = new_gid
        ann['id'] = new_aid
        new_aid = new_aid + 1
        new_anns.append(ann)

    return new_img, new_anns


# def update_coco_geotiff_metadata(coco_dset, serializable=True, max_workers=0):
#     """
#     if serializable is True, then we should only update with information
#     that can be coerced to json.
#     """
#     if serializable:
#         raise NotImplementedError('we dont do this yet')

#     bundle_dpath = coco_dset.bundle_dpath

#     pool = ub.JobPool(mode='thread', max_workers=max_workers)

#     img_iter = ub.ProgIter(coco_dset.imgs.values(),
#                            total=len(coco_dset.imgs),
#                            desc='submit update meta jobs',
#                            verbose=1, freq=1, adjust=False)
#     for img in img_iter:
#         job = pool.submit(kwcoco_extensions.single_geotiff_metadata,
#                           bundle_dpath, img, serializable=serializable)
#         job.img = img

#     for job in ub.ProgIter(pool.as_completed(),
#                            total=len(pool),
#                            desc='collect update meta jobs',
#                            verbose=1, freq=1, adjust=False):
#         geotiff_metadata, aux_metadata = job.result()
#         img = job.img
#         # # todo: can be parallelized
#         # geotiff_metadata, aux_metadata = single_geotiff_metadata(
#         #     bundle_dpath, img, serializable=serializable)
#         img['geotiff_metadata'] = geotiff_metadata
#         for aux, aux_info in zip(img.get('auxiliary', []), aux_metadata):
#             aux['geotiff_metadata'] = aux_info


def _flip(x, y):
    return (y, x)


def shapely_flip_xy(geom):
    from shapely import ops
    return ops.transform(_flip, geom)


def shapely_bounding_box(geom):
    import shapely
    return shapely.geometry.box(*geom.bounds)


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
               space_box, align_method, is_multi_image, keep):
    # NOTE: https://github.com/dwtkns/gdal-cheat-sheet
    latmin, lonmin, latmax, lonmax = space_box.data[0]

    if is_multi_image:
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
    if obj.get('channels', None):
        dst['channels'] = obj['channels']
    if obj.get('num_bands', None):
        dst['num_bands'] = obj['num_bands']

    # TODO: parametarize
    compress = 'RAW'
    blocksize = 64
    # NUM_THREADS=2

    # Use the new COG output driver
    prefix_template = (
        '''
        gdalwarp
        -multi
        --config GDAL_CACHEMAX 500 -wm 500
        --debug off
        -te {xmin} {ymin} {xmax} {ymax}
        -te_srs epsg:4326
        -t_srs epsg:4326
        -of COG
        -co OVERVIEWS=NONE
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        -overwrite
        ''')

    if compress == 'RAW':
        compress = 'NONE'

    if align_method == 'pixel_crop':
        align_method = 'pixel_crop'
        from kwcoco.util.util_delayed_poc import LazyGDalFrameFile
        imdata = LazyGDalFrameFile(src_gpath)
        info = obj['geotiff_metadata']
        space_region_pxl = space_region.warp(info['wgs84_to_wld']).warp(info['wld_to_pxl'])
        pxl_xmin, pxl_ymin, pxl_xmax, pxl_ymax = space_region_pxl.bounding_box().to_ltrb().quantize().data[0]
        sl = tuple([slice(pxl_ymin, pxl_ymax), slice(pxl_xmin, pxl_xmax)])
        subim, transform = kwimage.padded_slice(
            imdata, sl, return_info=True)
        kwimage.imwrite(dst_gpath, subim, space=None, backend='gdal',
                        blocksize=blocksize, compress=compress)
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
                prefix_template +
                '''
                -rpc -et 0
                -to RPC_DEM={dem_fpath}
                {SRC} {DST}
                ''')
        else:
            dem_fpath = None
            template = ub.paragraph(
                prefix_template +
                '''
                -rpc -et 0
                {SRC} {DST}
                ''')
        command = template.format(
            ymin=latmin,
            xmin=lonmin,
            ymax=latmax,
            xmax=lonmax,
            blocksize=blocksize,
            compress=compress,
            dem_fpath=dem_fpath,
            SRC=src_gpath, DST=dst_gpath,
        )
    elif align_method == 'affine_warp':
        template = ub.paragraph(
            prefix_template +
            '{SRC} {DST}')
        command = template.format(
            ymin=latmin,
            xmin=lonmin,
            ymax=latmax,
            xmax=lonmax,
            blocksize=blocksize,
            compress=compress,
            SRC=src_gpath, DST=dst_gpath,
        )
    else:
        raise KeyError(align_method)

    if not (exists(dst_gpath) and keep):
        cmd_info = ub.cmd(command, verbose=0)  # NOQA
        if cmd_info['ret'] != 0:
            print('\n\nCOMMAND FAILED: {!r}'.format(command))
            raise Exception(cmd_info['err'])
        # cmd_info = ub.cmd(command, verbose=1)  # NOQA

    if not exists(dst_gpath):
        raise Exception('THE DESTINATION PATH WAS NOT COMPUTED')

    return dst


_CLI = CocoAlignGeotiffConfig


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.coco_align_geotiffs --help
    """
    main()
