#!/usr/bin/env python
r"""
Given the raw data in kwcoco format, this script will extract orthorectified
regions around areas of intere/t across time.

The align script works by making two geopandas data frames of geo-boundaries,
one for regions and one for all images (as defined by their geotiff metadata).
I then use the util_gis.geopandas_pairwise_overlaps to efficiently find which
regions intersect which images. Images that intersect a region are grouped
together (the same image might belong to multiple regions). Then within each
region group, the script finds all images that have the same datetime metadata
and groups those together. Finally, images with the "same-exact" bands are
grouped together. For each band-group I use gdal-warp to crop to the region,
which creates a set of temporary files, and then finally gdalmerge is used to
combine those different crops into a single image.

The main corner case in the above process is when one image has "r|g|b" but
another image has "r|g|b|yellow", there is no logic to split those channels out
at the moment.


Notes:
    # Example invocation to create the full drop1 aligned dataset

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    INPUT_COCO_FPATH=$DVC_DPATH/drop1/data.kwcoco.json
    OUTPUT_COCO_FPATH=$DVC_DPATH/drop1-WV-only-aligned/data.kwcoco.json
    REGION_FPATH=$DVC_DPATH/drop1/all_regions.geojson
    VIZ_DPATH=$OUTPUT_COCO_FPATH/_viz_video

    # Quick stats about input datasets
    python -m kwcoco stats $INPUT_COCO_FPATH
    python -m watch stats $INPUT_COCO_FPATH

    # Combine the region models
    python -m watch.cli.merge_region_models \
        --src $DVC_DPATH/drop1/region_models/*.geojson \
        --dst $REGION_FPATH

    python -m watch.cli.coco_add_watch_fields \
        --src $INPUT_COCO_FPATH \
        --dst $INPUT_COCO_FPATH.prepped \
        --workers 16 \
        --target_gsd=10

    # Execute alignment / crop script
    python -m watch.cli.coco_align_geotiffs \
        --src $INPUT_COCO_FPATH.prepped \
        --dst $OUTPUT_COCO_FPATH \
        --regions $REGION_FPATH \
        --rpc_align_method orthorectify \
        --workers=10 \
        --aux_workers=2 \
        --context_factor=1 \
        --visualize=False \
        --geo_preprop=False \
        --include_sensors=WV \
        --keep img


TODO:
    - [ ] Add method for extracting "negative ROIs" that are nearby
        "positive ROIs".

    - [X] Diagnose and Fix PROJ errors:

        ```
        ERROR 1: PROJ: proj_create: unrecognized format / unknown name
        ERROR 1: PROJ: proj_create_from_database: Cannot find proj.db
        ```
    - [ ] Rename file to coco_geoalign.py
"""
import kwcoco
import kwimage
import os
import scriptconfig as scfg
import socket
import ubelt as ub
import dateutil.parser
import warnings
from os.path import join, exists
from watch.cli.coco_visualize_videos import _write_ann_visualizations2
from watch.utils import util_gis
from watch.utils import util_time
from watch.utils import util_gdal
from watch.utils import kwcoco_extensions

DEBUG = 0

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


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

        'workers': scfg.Value(0, type=str, help='number of parallel procs. This can also be an expression accepted by coerce_num_workers.'),
        'max_workers': scfg.Value(None, type=str, help='DEPRECATED USE workers'),
        'aux_workers': scfg.Value(0, type=str, help='additional inner threads for aux imgs'),

        'context_factor': scfg.Value(1.0, help=ub.paragraph(
            '''
            scale factor for the clustered ROIs.
            Amount of context to extract around each ROI.
            '''
        )),

        'convexify_regions': scfg.Value(False, help=ub.paragraph(
            '''
            if True, ensure that the regions are convex
            ''')),

        'regions': scfg.Value('annots', help=ub.paragraph(
            '''
            Strategy for extracting regions, if annots, uses the convex hulls
            of clustered annotations. Can also be a path to a geojson file
            to use pre-defined regions.
            ''')),

        'site_summary': scfg.Value(False, help='Crop to site summaries instead'),

        # TODO: change this name to just align-method or something
        'rpc_align_method': scfg.Value('orthorectify', help=ub.paragraph(
            '''
            Can be one of:
                (1) orthorectify - which uses gdalwarp with -rpc if available
                    otherwise falls back to affine transform,
                (2) pixel_crop - which warps annotations onto pixel with RPCs
                    but only crops the original image without distortion,
                (3) affine_warp - which ignores RPCs and uses the affine
                    transform in the geotiff metadata.
            '''
        )),

        'write_subsets': scfg.Value(True, isflag=1, help=ub.paragraph(
            '''
            if True, writes a separate kwcoco file for every discovered ROI
            in addition to the final kwcoco file.
            '''
        )),

        'visualize': scfg.Value(False, isflag=1, help=ub.paragraph(
            '''
            if True, normalize and draw image / annotation sequences when
            extracting.
            '''
        )),

        'debug_valid_regions': scfg.Value(False, isflag=1, help=ub.paragraph(
            '''
            write valid region visualizations to help debug "black images"
            issues.
            '''
        )),

        'keep': scfg.Value('none', help=ub.paragraph(
            '''
            Level of detail to overwrite existing data at, since this is slow.
            "none": overwrite all, including existing images
            "img": only add new images
            "roi": only add new ROIs
            "roi-img": only add new ROIs and only new images within those ROIs (good for rerunning failed jobs)
            '''
        )),

        'skip_geo_preprop': scfg.Value(False, help='DEPRECATED use geo_preop instead'),
        'geo_preprop': scfg.Value('auto', help='force if we check geo properties or not'),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),
        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

        'target_gsd': scfg.Value(10, help=ub.paragraph('initial gsd to use for the output video files')),

        'edit_geotiff_metadata': scfg.Value(
            False, help='if True MODIFIES THE UNDERLYING IMAGES to ensure geodata is propogated'),

        'max_frames': scfg.Value(None),

        'warp_tries': scfg.Value(2, help='The maximum number of times to retry failed gdal warp commands before stopping.'),

        # FIXME: asset and image timeouts are not handled correctly.
        'image_timeout': scfg.Value('8hours', help='The maximum amount of time to spend pulling down a all image assets before giving up'),
        'asset_timeout': scfg.Value('4hours', help='The maximum amount of time to spend pulling down a single image asset before giving up'),

        'include_channels': scfg.Value(None, help='If specified only align the given channels'),
        'exclude_channels': scfg.Value(None, help='If specified ignore these channels'),

        'verbose': scfg.Value(0, help='Note: no silent mode, 0 is just least verbose.'),
        'force_nodata': scfg.Value(None, help=('if specified, forces nodata to this value (e.g. -9999) '
                                               'Ideally this is not needed and all source geotiffs properly specify nodata')),
        'force_min_gsd': scfg.Value(None, help=ub.paragraph('Force output crops to be at least this minimum GSD (e.g. if set to 10.0 an input image with a 30.0 GSD will have an output GSD of 30.0, whereas in input image with a 0.5 GSD will have it set to 10.0 during cropping)'))
    }

    def normalize(config):
        if isinstance(config['target_gsd'], str):
            if config['target_gsd'].lower().endswith('gsd'):
                config['target_gsd'] = int(config['target_gsd'][:-3].strip())


@profile
def main(cmdline=True, **kw):
    """
    Main function for coco_align_geotiffs.
    See :class:``CocoAlignGeotiffConfig` for details

    Ignore:
        from watch.cli.coco_align_geotiffs import *  # NOQA
        import kwcoco
        cmdline = False
        src = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop1/data.kwcoco.json')
        dst = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-L1/_test/test.kwcoco.json')
        regions = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop1/region_models/LT_R001.geojson')
        regions = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop1/all_regions.geojson')
        kw = {
            'src': src,
            'dst': dst,
            'regions': regions,
            'keep': 'none',
            'exclude_sensors': ['WV'],
        }

    Ignore:
        from watch.cli.coco_align_geotiffs import *  # NOQA
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
        src = base_fpath
        dst = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json'
        sites = dvc_dpath / 'annotations/site_models/*.geojson'
        cmdline = 0
        kw = {
            'src': src,
            'dst': dst,
            'regions': sites,
            'keep': 'none',
        }

    CommandLine:
        xdoctest -m watch.cli.coco_align_geotiffs main:0

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
        >>> dpath = ub.Path.appdir('watch/test/coco_align_geotiff').ensuredir()
        >>> dst = ub.ensuredir((dpath, 'align_bundle1'))
        >>> ub.delete(dst)
        >>> dst = ub.ensuredir(dst)
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': 'annots',
        >>>     'workers': 2,
        >>>     'aux_workers': 2,
        >>>     'convexify_regions': True,
        >>>     #'image_timeout': '1 microsecond',
        >>>     #'asset_timeout': '1 microsecond',
        >>>     'visualize': True,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)

    Example:
        >>> # Confirm expected behavior of `force_min_gsd` keyword argument
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> from watch.gis.geotiff import geotiff_metadata, geotiff_crs_info
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
        >>> dpath = ub.Path.appdir('watch/test/coco_align_geotiff').ensuredir()
        >>> dst = ub.ensuredir((dpath, 'align_bundle1_force_gsd'))
        >>> ub.delete(dst)
        >>> dst = ub.ensuredir(dst)
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': 'annots',
        >>>     'workers': 2,
        >>>     'aux_workers': 2,
        >>>     'convexify_regions': True,
        >>>     #'image_timeout': '1 microsecond',
        >>>     #'asset_timeout': '1 microsecond',
        >>>     'visualize': False,
        >>>     'force_min_gsd': 60.0,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)
        >>> coco_img = new_dset.coco_image(2)
        >>> # Check our output is in the CRS we think it is
        >>> asset = coco_img.primary_asset()
        >>> parent_fpath = asset['parent_file_name']
        >>> crop_fpath = join(new_dset.bundle_dpath, asset['file_name'])
        >>> info = geotiff_crs_info(crop_fpath)
        >>> assert(all(info['meter_per_pxl'] == 60.0))

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> coco_dset = demo_kwcoco_with_heatmaps(num_videos=2, num_frames=2)
        >>> dpath = ub.Path.appdir('watch/test/coco_align_geotiff2').ensuredir()
        >>> dst = (dpath / 'align_bundle2').delete().ensuredir()
        >>> # Create a dummy region file to crop to.
        >>> first_img = coco_dset.images().take([0]).coco_images[0]
        >>> from osgeo import gdal
        >>> first_fpath = first_img.primary_image_filepath()
        >>> geo_poly = kwimage.Polygon.coerce(gdal.Info(first_fpath, format='json')['wgs84Extent'])
        >>> region_shape = kwimage.Polygon.random(n=8, convex=False, rng=3)
        >>> geo_transform = kwimage.Affine.fit(region_shape.bounding_box().corners(), geo_poly.bounding_box().corners())
        >>> region_poly = region_shape.warp(geo_transform)
        >>> import geojson
        >>> import json
        >>> region_feature = geojson.Feature(
        >>>     properties={
        >>>         "type": "region",
        >>>         "region_id": "DUMMY_R042",
        >>>         "start_date": '1970-01-01',
        >>>         "end_date":  '2970-01-01',
        >>>     },
        >>>     geometry=region_poly.to_geojson(),
        >>> )
        >>> region = geojson.FeatureCollection([region_feature])
        >>> region_fpath = dst / 'dummy_region.geojson'
        >>> region_fpath.write_text(json.dumps(region))
        >>> # Create arguments to the script
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': region_fpath,
        >>>     'workers': 0,
        >>>     'aux_workers': 0,
        >>>     'visualize': 1,
        >>>     'debug_valid_regions': True,
        >>>     'target_gsd': 0.7,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)
        >>> coco_img = new_dset.coco_image(2)
        >>> # Check our output is in the CRS we think it is
        >>> asset = coco_img.primary_asset()
        >>> parent_fpath = asset['parent_file_name']
        >>> crop_fpath = join(new_dset.bundle_dpath, asset['file_name'])
        >>> print(ub.cmd(['gdalinfo', parent_fpath])['out'])
        >>> print(ub.cmd(['gdalinfo', crop_fpath])['out'])


        >>> # Test that the input dataset visualizes ok
        >>> from watch.cli import coco_visualize_videos
        >>> viz_dpath = (dpath / 'viz_input_align_bundle2').ensuredir()
        >>> coco_visualize_videos.main(cmdline=False, **{
        >>>     'src': new_dset,
        >>>     'viz_dpath': viz_dpath,
        >>> })

        print(ub.cmd(['gdalinfo', parent_fpath])['out'])
        print(ub.cmd(['gdalinfo', crop_fpath])['out'])

        df1 = covered_annot_geo_regions(coco_dset)
        df2 = covered_image_geo_regions(coco_dset)
    """
    from watch.utils.lightning_ext import util_globals
    import pandas as pd
    config = CocoAlignGeotiffConfig(default=kw, cmdline=cmdline)

    # Store that this dataset is a result of a process.
    # Note what the process is, what its arguments are, and where the process
    # was executed.
    config_dict = config.to_dict()
    if not isinstance(config_dict['src'], str):
        # If the dataset was given in memory we don't know the path and we cant
        # always serialize it, so we punt and mark it as such
        config_dict['src'] = ':memory:'

    from kwcoco.util.util_json import ensure_json_serializable
    config_dict = ensure_json_serializable(config_dict)

    import os
    if os.environ.get('GDAL_DISABLE_READDIR_ON_OPEN') != 'EMPTY_DIR':
        import warnings
        warnings.warn('environ GDAL_DISABLE_READDIR_ON_OPEN should probably be set to EMPTY_DIR')
        os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

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
    print('process_info = {}'.format(ub.repr2(process_info, nl=3, sort=0)))

    src_fpath = config['src']
    dst = config['dst']
    regions = config['regions']
    context_factor = config['context_factor']
    rpc_align_method = config['rpc_align_method']
    visualize = config['visualize']
    write_subsets = config['write_subsets']
    img_workers = config['max_workers']
    aux_workers = config['aux_workers']
    keep = config['keep']
    target_gsd = config['target_gsd']
    max_frames = config['max_frames']

    if config['max_workers'] is not None:
        img_workers = util_globals.coerce_num_workers(config['max_workers'])
    else:
        img_workers = util_globals.coerce_num_workers(config['workers'])

    aux_workers = util_globals.coerce_num_workers(config['aux_workers'])
    print('img_workers = {!r}'.format(img_workers))
    print('aux_workers = {!r}'.format(aux_workers))

    dst = ub.Path(ub.expandpath(dst))
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
    else:
        from watch.utils import util_gis
        infos = list(util_gis.coerce_geojson_datas(regions))
        parts = []
        for info in infos:
            df = info['data']
            if config['site_summary']:
                df = df[df['type'] == 'site_summary']
            else:
                if df.iloc[0]['type'] == 'site':
                    df = df[df['type'] == 'site']
                else:
                    df = df[df['type'] == 'region']
            parts.append(df)
        region_df = pd.concat(parts)

    # Load the dataset and extract geotiff metadata from each image.
    coco_dset = kwcoco.CocoDataset.coerce(src_fpath)
    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
    )

    geo_preprop = config['geo_preprop']
    if config['skip_geo_preprop']:
        import warnings
        warnings.warn('skip_geo_preprop is deprecated', DeprecationWarning)
        geo_preprop = False
    if geo_preprop == 'auto':
        if len(valid_gids):
            coco_img = coco_dset.coco_image(ub.peek(valid_gids))
            geo_preprop = not any('geos_corners' in obj for obj in coco_img.iter_asset_objs())
            print('auto-choose geo_preprop = {!r}'.format(geo_preprop))

    if geo_preprop:
        geopop_workers = img_workers * aux_workers
        kwcoco_extensions.coco_populate_geo_heuristics(
            coco_dset, overwrite={'warp'}, workers=geopop_workers,
            keep_geotiff_metadata=True, gids=valid_gids
        )
    if config['edit_geotiff_metadata']:
        kwcoco_extensions.ensure_transfered_geo_data(coco_dset, gids=valid_gids)

    # Construct the "data cube"
    cube = SimpleDataCube(coco_dset, gids=valid_gids)

    # Find the clustered ROI regions
    if regions == 'images':
        region_df = kwcoco_extensions.covered_image_geo_regions(coco_dset, merge=True)
    elif regions == 'annots':
        region_df = kwcoco_extensions.covered_annot_geo_regions(coco_dset, merge=True)
    else:
        assert region_df is not None, 'must have been given regions some other way'

    # Ensure all indexes are unique
    region_df = region_df.reset_index(drop=True)

    print('query region_df =\n{}'.format(region_df))
    print('cube.img_geos_df =\n{}'.format(cube.img_geos_df))

    if config['convexify_regions']:
        # Exapnd the ROI by the context factor
        region_df['geometry'] = region_df['geometry'].convex_hull

    # Convert the ROI to a bounding box
    # region_df['geometry'] = region_df['geometry'].apply(shapely_bounding_box)
    if context_factor != 1:
        # Exapnd the ROI by the context factor
        region_df['geometry'] = region_df['geometry'].scale(
            xfact=context_factor, yfact=context_factor, origin='center')

    # For each ROI extract the aligned regions to the target path
    extract_dpath = ub.ensuredir(output_bundle_dpath)

    # Create a new dataset that we will extend as we extract ROIs
    new_dset = kwcoco.CocoDataset()

    new_dset.dataset['info'] = [
        process_info,
    ]
    to_extract = cube.query_image_overlaps(region_df)

    for image_overlaps in ub.ProgIter(to_extract, desc='extract ROI videos', verbose=3):
        video_name = image_overlaps['video_name']
        print('video_name = {!r}'.format(video_name))

        sub_bundle_dpath = join(extract_dpath, video_name)
        print('sub_bundle_dpath = {!r}'.format(sub_bundle_dpath))

        new_dset = cube.extract_overlaps(
            image_overlaps, extract_dpath, rpc_align_method=rpc_align_method,
            new_dset=new_dset, visualize=visualize,
            write_subsets=write_subsets, img_workers=img_workers,
            aux_workers=aux_workers, keep=keep, target_gsd=target_gsd,
            max_frames=max_frames,
            debug_valid_regions=config['debug_valid_regions'],
            include_channels=config['include_channels'],
            exclude_channels=config['exclude_channels'],
            tries=config['warp_tries'],
            image_timeout=config['image_timeout'],
            asset_timeout=config['asset_timeout'],
            verbose=config['verbose'],
            force_nodata=config['force_nodata'],
            force_min_gsd=config['force_min_gsd'],
        )

    kwcoco_extensions.reorder_video_frames(new_dset)
    new_dset.fpath = dst_fpath
    print('Dumping new_dset.fpath = {!r}'.format(new_dset.fpath))
    try:
        rerooted_dataset = new_dset.copy()
        rerooted_dataset = rerooted_dataset.reroot(new_root=output_bundle_dpath, absolute=False)
    except Exception:
        # Hack to fix broken pipeline, todo: find robust fix
        hack_region_id = infos[0]['fpath'].stem
        rerooted_dataset = new_dset.copy()
        rerooted_dataset.reroot(new_prefix=hack_region_id)
        rerooted_dataset.reroot(new_root=output_bundle_dpath, absolute=False)

    rerooted_dataset.dump(rerooted_dataset.fpath, newlines=True)
    print('finished')
    return rerooted_dataset


class SimpleDataCube(object):
    """
    Given a CocoDataset containing geotiffs, provide a simple API to extract a
    region in some coordinate space.

    Intended usage is to use :func:`query_image_overlaps` to find images that
    overlap an ROI, then then :func:`extract_overlaps` to warp spatial subsets
    of that data into an aligned temporal sequence.
    """

    def __init__(cube, coco_dset, gids=None):
        import geopandas as gpd
        import shapely
        from kwcoco.util import ensure_json_serializable
        expxected_geos_crs_info = {
            'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
            'auth': ('EPSG', '4326')
        }  # This is CRS84
        crs84 = util_gis._get_crs84()
        expxected_geos_crs_info = ensure_json_serializable(expxected_geos_crs_info)
        gids = coco_dset.images(gids)._ids

        # put data in the cube into a geopandas data frame
        columns = ['gid', 'name', 'video_id', 'geometry', 'properties']
        df_rows = []
        for gid in gids:
            img = coco_dset.index.imgs[gid]
            if 'geos_corners' in img:
                geos_corners = img['geos_corners']
            else:
                geos_corners = None
                for asset in img.get('auxiliary', img.get('assets', [])):
                    if 'geos_corners' in asset:
                        geos_corners = asset['geos_corners']
                if geos_corners is None:
                    raise Exception("could not find geos_corners in img or assets")
            sh_img_poly = shapely.geometry.shape(geos_corners)
            properties = geos_corners.get('properties', {})
            crs_info = properties.get('crs_info', None)
            if crs_info is not None:
                crs_info = ensure_json_serializable(crs_info)
                if crs_info != expxected_geos_crs_info:
                    raise AssertionError(ub.paragraph(
                        '''
                        got={}, but expected={}
                        ''').format(crs_info, expxected_geos_crs_info))

            # Create a data frame with space-time regions
            df_rows.append({
                'gid': gid,
                'name': img.get('name', None),
                'video_id': img.get('video_id', None),
                'geometry': sh_img_poly,
                'properties': properties,
            })

        img_geos_df = gpd.GeoDataFrame(df_rows, geometry='geometry',
                                       columns=columns, crs=crs84)
        img_geos_df = img_geos_df.set_index('gid', drop=False)
        cube.coco_dset = coco_dset
        cube.img_geos_df = img_geos_df

    @classmethod
    def demo(SimpleDataCube, with_region=False, extra=0):
        from watch.demo.landsat_demodata import grab_landsat_product
        from watch.gis.geotiff import geotiff_metadata
        # Create a dead simple coco dataset with one image
        import geopandas as gpd
        import kwcoco
        coco_dset = kwcoco.CocoDataset()

        landsat_products = []
        # ls_prod = grab_landsat_product()
        # landsat_products.append(ls_prod)
        landsat_products.append(grab_landsat_product(demo_index=0))
        if extra:
            # For debugging
            landsat_products.append(grab_landsat_product(demo_index=1))
            landsat_products.append(grab_landsat_product(demo_index=2))

        features = []

        for prod_idx, ls_prod in enumerate(landsat_products):
            fpath = ls_prod['bands'][0]
            meta = geotiff_metadata(fpath)
            # We need a date captured ATM in a specific format
            dt = dateutil.parser.parse(
                meta['filename_meta']['acquisition_date'])
            date_captured = dt.strftime('%Y/%m/%d')

            gid = coco_dset.add_image(file_name=fpath,
                                      date_captured=date_captured,
                                      sensor_coarse='L8')
            img_poly = kwimage.Polygon(exterior=meta['wgs84_corners'])
            ann_poly = img_poly.scale(0.1, about='center')
            sseg_geos = ann_poly.swap_axes().to_geojson()
            coco_dset.add_annotation(
                image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)

            if prod_idx == 0:
                # Only generate this feature for the first product
                # for backwards compat
                features.append({
                    'type': 'Feature',
                    'properties': {
                        'type': 'region',
                        'region_id': 'demo_region',
                        'version': '2.1.0',
                        'mgrs': None,
                        'start_date': None,
                        'end_date': None,
                        'originator': 'foobar',
                        'comments': None,
                        'model_content': 'annotation',
                        'sites': [],
                    },
                    'geometry': img_poly.scale(0.2, about='center').swap_axes().to_geojson(),
                })

        kwcoco_extensions.coco_populate_geo_heuristics(
            coco_dset, overwrite={'warp'}, workers=2 if extra > 0 else 0,
            keep_geotiff_metadata=True,
        )

        if extra:
            # for the overlapping images add in a special feature.
            overlap_box = kwimage.Boxes.from_slice(
                (slice(54.5, 55.5), slice(24.5, 25.6))).to_polygons()[0]
            features.append({
                'type': 'Feature',
                'properties': {
                    'type': 'region',
                    'region_id': 'demo_region',
                    'version': '2.1.0',
                    'mgrs': None,
                    'start_date': None,
                    'end_date': None,
                    'originator': 'foobar',
                    'comments': None,
                    'model_content': 'annotation',
                    'sites': [],
                },
                'geometry': overlap_box.to_geojson(),
            })

        cube = SimpleDataCube(coco_dset)
        if with_region:
            region_geojson =  {
                'type': 'FeatureCollection',
                'features': features,
            }
            region_df = gpd.GeoDataFrame.from_features(
                region_geojson, crs=util_gis._get_crs84())
            return cube, region_df
        return cube

    @profile
    def query_image_overlaps(cube, region_df):
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
            >>> from watch.cli.coco_align_geotiffs import *  # NOQA
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> to_extract = cube.query_image_overlaps(region_df)
        """
        from kwcoco.util.util_json import ensure_json_serializable
        import geopandas as gpd

        # Quickly find overlaps using a spatial index
        ridx_to_gidsx = util_gis.geopandas_pairwise_overlaps(region_df, cube.img_geos_df)

        print('candidate query overlaps')
        ridx_to_num_matches = ub.map_vals(len, ridx_to_gidsx)
        print('ridx_to_num_matches = {}'.format(ub.repr2(ridx_to_num_matches, nl=1)))
        # print('ridx_to_gidsx = {}'.format(ub.repr2(ridx_to_gidsx, nl=1)))

        to_extract = []
        for ridx, gidxs in ridx_to_gidsx.items():
            region_row = region_df.iloc[ridx]

            _region_row_df = gpd.GeoDataFrame([region_row], crs=region_df.crs)
            crs = _region_row_df.estimate_utm_crs()
            utm_epsg_zone_v1 = crs.to_epsg()
            # geom_crs84 = region_row.geometry
            # utm_epsg_zone_v2 = util_gis.find_local_meter_epsg_crs(geom_crs84)
            # if utm_epsg_zone_v2 != utm_epsg_zone_v1:
            #     raise Exception(
            #         'Consistency Error: '
            #         f'utm_epsg_zone_v1={utm_epsg_zone_v1} '
            #         f'utm_epsg_zone_v2={utm_epsg_zone_v2} '
            #     )
            local_epsg = utm_epsg_zone_v1

            CHECK_THIN_REGIONS = True
            if CHECK_THIN_REGIONS:
                # Try and detect thin regions and then add context
                region_row_df_utm = _region_row_df.to_crs(local_epsg)
                region_utm_geom = region_row_df_utm['geometry'].iloc[0]
                # poly = kwimage.Polygon.coerce(region_utm_geom)
                import cv2
                from collections import namedtuple
                import numpy as np
                OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'theta'))
                from shapely import validation
                if not region_utm_geom.is_valid:
                    warnings.warn('Region query is invalid: ' + str(validation.explain_validity(region_utm_geom)))
                    continue
                hull_utm = kwimage.Polygon.coerce(region_utm_geom.convex_hull)
                c, e, a = cv2.minAreaRect(hull_utm.exterior.data.astype(np.float32))
                t = np.deg2rad(a)
                obox = OrientedBBox(c, e, t)
                # HACK:
                # Expand extent to ensure minimum thickness
                min_meter_extent = 100
                new_extent = np.maximum(obox.extent, min_meter_extent)
                if np.all(new_extent == np.array(obox.extent)):
                    ...
                else:
                    ubox = kwimage.Boxes([[0, 0, 1, 1]], 'cxywh').to_polygons()[0]
                    # S = kwimage.Affine.affine(scale=obox.extent)
                    S = kwimage.Affine.affine(scale=new_extent)
                    R = kwimage.Affine.affine(theta=obox.theta)
                    T = kwimage.Affine.affine(offset=obox.center)
                    new_hull_utm = ubox.warp(T @ R @ S)
                    fixed_geom_utm = gpd.GeoDataFrame({
                        'geometry': [new_hull_utm.to_shapely()]},
                        crs=local_epsg)
                    fixed_geom_crs84 = fixed_geom_utm.to_crs(region_df.crs)
                    region_row = region_row.copy()
                    region_row['geometry'] = fixed_geom_crs84['geometry'].iloc[0]

            space_region = kwimage.MultiPolygon.from_shapely(region_row.geometry)
            space_box = space_region.bounding_box().to_ltrb()

            # Data is from geo-pandas so this should be traditional order
            lonmin, latmin, lonmax, latmax = space_box.data[0]
            min_pt = util_gis.latlon_text(latmin, lonmin)
            max_pt = util_gis.latlon_text(latmax, lonmax)
            space_str = '{}_{}'.format(min_pt, max_pt)

            if region_row.get('type', None) == 'region':
                # Special case where we are extracting a region with a name
                video_name = region_row.get('region_model_id', space_str)  # V1 spec
                video_name = region_row.get('region_id', video_name)  # V2 spec
            else:
                video_name = space_str

            if len(gidxs) == 0:
                print('Warning: No spatial matches to {}'.format(video_name))
            else:
                from watch.utils import util_time

                # TODO: filter dates out of range
                query_start_date = region_row.get('start_date', None)
                query_end_date = region_row.get('end_date', None)

                cand_gids = cube.img_geos_df.iloc[gidxs].gid
                cand_datecaptured = cube.coco_dset.images(cand_gids).lookup('date_captured')

                cand_datetimes = [util_time.coerce_datetime(c) for c in cand_datecaptured]

                # By reducing the granularity we can group nearly
                # identical images together. FIXME: Configure
                REDUCE_GRANULARITY = True
                if REDUCE_GRANULARITY:
                    # Reduce images taken with the hour (does not account for
                    # borders). Better method would be agglomerative
                    # clustering.
                    reduced = []
                    for dt in cand_datetimes:
                        new = dt.replace(minute=0, second=0, microsecond=0)
                        reduced.append(new)
                    cand_datetimes = reduced

                if query_start_date is not None:
                    query_start_datetime = util_time.coerce_datetime(query_start_date)
                    flags = [dt >= query_start_datetime for dt in cand_datetimes]
                    cand_datetimes = list(ub.compress(cand_datetimes, flags))
                    cand_gids = list(ub.compress(cand_gids, flags))

                if query_end_date is not None:
                    query_end_datetime = util_time.coerce_datetime(query_end_date)
                    flags = [dt <= query_end_datetime for dt in cand_datetimes]
                    cand_datetimes = list(ub.compress(cand_datetimes, flags))
                    cand_gids = list(ub.compress(cand_gids, flags))

                if len(cand_gids) == 0:
                    print('Warning: No temporal matches to {}'.format(video_name))
                else:
                    datetime_to_gids = ub.group_items(cand_gids, cand_datetimes)
                    # print('datetime_to_gids = {}'.format(ub.repr2(datetime_to_gids, nl=1)))
                    dates = sorted(datetime_to_gids)
                    print('Found {:>4} overlaps for {} from {} to {}'.format(
                        len(cand_gids),
                        video_name,
                        min(dates).isoformat(),
                        max(dates).isoformat(),
                    ))

                    region_props = ub.dict_diff(
                        region_row.to_dict(), {'geometry', 'sites'})
                    region_props = ensure_json_serializable(region_props)

                    image_overlaps = {
                        'datetime_to_gids': datetime_to_gids,
                        'space_region': space_region,
                        'space_str': space_str,
                        'space_box': space_box,
                        'video_name': video_name,
                        'properties': region_props,
                        'local_epsg': local_epsg,
                    }
                    to_extract.append(image_overlaps)
        return to_extract

    @profile
    def extract_overlaps(cube, image_overlaps, extract_dpath,
                         rpc_align_method='orthorectify', new_dset=None,
                         write_subsets=True, visualize=True, img_workers=0,
                         aux_workers=0, keep='none', target_gsd=10,
                         max_frames=None, debug_valid_regions=False,
                         include_channels=None, exclude_channels=None,
                         tries=2,
                         image_timeout=None,
                         asset_timeout=None,
                         force_nodata=None, verbose=0,
                         force_min_gsd=None):
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

            keep (str): Level of detail to overwrite existing data at, since this is slow.
                "none": overwrite all, including existing images
                "img": only add new images
                "roi": only add new ROIs

            include_channels (FusedChannelSpec):
                if specified, only use these channels.

            exclude_channels (FusedChannelSpec):
                exclude these channels

            verbose (int):
                note, there is no silent mode, 0 is just the least verbose.

            force_min_gsd (float):
                Force output crops to be at least this minimum GSD
                (e.g. if set to 10.0 an input image with a 30.0 GSD
                will have an output GSD of 30.0, whereas in input
                image with a 0.5 GSD will have it set to 10.0 during
                cropping)

        Returns:
            kwcoco.CocoDataset: the given or new dataset that was modified

        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from watch.cli.coco_align_geotiffs import *  # NOQA
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> extract_dpath = ub.Path.appdir('watch/test/coco_align_geotiff/demo_extract_overlaps').ensuredir()
            >>> rpc_align_method = 'orthorectify'
            >>> new_dset = kwcoco.CocoDataset()
            >>> write_subsets = True
            >>> visualize = True
            >>> img_workers = 32
            >>> to_extract = cube.query_image_overlaps(region_df)
            >>> image_overlaps = to_extract[0]
            >>> cube.extract_overlaps(image_overlaps, extract_dpath,
            >>>                       new_dset=new_dset, visualize=visualize,
            >>>                       img_workers=img_workers)

        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from watch.cli.coco_align_geotiffs import *  # NOQA
            >>> cube, region_df = SimpleDataCube.demo(with_region=True, extra=True)
            >>> extract_dpath = ub.Path.appdir('watch/test/coco_align_geotiff/demo_extract_overlaps2').ensuredir()
            >>> rpc_align_method = 'orthorectify'
            >>> write_subsets = True
            >>> visualize = True
            >>> img_workers = 0
            >>> to_extract = cube.query_image_overlaps(region_df)
            >>> new_dset = kwcoco.CocoDataset()
            >>> image_overlaps = to_extract[1]
            >>> cube.extract_overlaps(image_overlaps, extract_dpath,
            >>>                       new_dset=new_dset, visualize=visualize,
            >>>                       img_workers=img_workers)

            xdev.profile_now(SimpleDataCube.demo)
        """
        from kwcoco.util.util_json import ensure_json_serializable
        import geopandas as gpd
        import pandas as pd  # NOQA
        from shapely import geometry
        from watch.utils import util_gis
        from shapely.ops import unary_union
        import shapely
        # import watch
        coco_dset = cube.coco_dset

        # print('image_overlaps = {}'.format(ub.repr2(image_overlaps, nl=1)))
        datetime_to_gids = image_overlaps['datetime_to_gids']
        space_str = image_overlaps['space_str']
        space_box = image_overlaps['space_box']
        space_region = image_overlaps['space_region']
        video_name = image_overlaps['video_name']
        video_props = image_overlaps['properties']
        local_epsg = image_overlaps['local_epsg']
        print('space_str = {}'.format(ub.repr2(space_str, nl=1)))
        print('space_box = {}'.format(ub.repr2(space_box, nl=1)))
        print('space_region = {}'.format(ub.repr2(space_region, nl=1)))
        print('video_name = {}'.format(ub.repr2(video_name, nl=1)))
        print('video_props = {}'.format(ub.repr2(video_props, nl=1)))
        print('local_epsg = {}'.format(ub.repr2(local_epsg, nl=1)))

        if new_dset is None:
            new_dset = kwcoco.CocoDataset()

        sub_bundle_dpath = ub.ensuredir((extract_dpath, video_name))

        if exists(join(sub_bundle_dpath,
                       'subdata.kwcoco.json')) and keep in {'roi-img', 'roi'}:
            print('ROI cache hit')
            sub_dset = kwcoco.CocoDataset(
                join(sub_bundle_dpath, 'subdata.kwcoco.json'))
            return new_dset.union(sub_dset)

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        datetimes = sorted(datetime_to_gids)

        valid_region_geos = space_region.to_geojson()

        new_video = {
            'name': video_name,
            'valid_region_geos': valid_region_geos,
            'properties': video_props,
        }
        new_video = ensure_json_serializable(new_video)

        new_vidid = new_dset.add_video(**new_video)
        for cat in coco_dset.cats.values():
            new_dset.ensure_category(**cat)

        bundle_dpath = coco_dset.bundle_dpath
        new_bundle_dpath = new_dset.bundle_dpath
        new_anns = []

        # Manage new ids such that parallelization does not impact their order
        start_gid = new_dset._next_ids.get('images')
        start_aid = new_dset._next_ids.get('annotations')
        frame_index = 0

        # parallelize over images
        image_jobs = ub.JobPool(mode='thread', max_workers=img_workers)

        sh_space_region_crs84 = space_region.to_shapely()
        space_region_crs84 = gpd.GeoDataFrame(
            {'geometry': [sh_space_region_crs84]}, crs=util_gis._get_crs84())

        space_region_local = space_region_crs84.to_crs(local_epsg)
        sh_space_region_local = space_region_local.geometry.iloc[0]

        frame_count = 0
        prog = ub.ProgIter(datetimes, desc='submit extract jobs', verbose=1)
        dtiter = iter(prog)
        for datetime_ in dtiter:

            if max_frames is not None:
                if frame_count > max_frames:
                    break
                frame_count += 1

            iso_time = util_time.isoformat(datetime_, sep='T', timespec='seconds')
            gids = datetime_to_gids[datetime_]
            # TODO: Is there any other consideration we should make when
            # multiple images have the same timestamp?
            # if len(gids) > 0:
            groups = []
            if len(gids) > 0:
                # We got multiple images for the same timestamp.  Im not sure
                # if this is necessary but thig logic attempts to sort them
                # such that the "best" image to use is first.  Ideally gdalwarp
                # would take care of this but I'm not sure it does.
                conflict_imges = coco_dset.images(gids)
                sensors = list(conflict_imges.lookup('sensor_coarse', None))
                for sensor_coarse, sensor_gids in ub.group_items(conflict_imges, sensors).items():
                    rows = []
                    for gid in sensor_gids:
                        coco_img = coco_dset.coco_image(gid)

                        # Should more than just the primary asset be used here?
                        primary_asset = coco_img.primary_asset()
                        fpath = join(coco_dset.bundle_dpath, primary_asset['file_name'])

                        # Note: valid region data is not necessary as input but
                        # we use it if it exists.
                        valid_region_utm = coco_img.img.get('valid_region_utm', None)
                        if valid_region_utm is not None:
                            geos_valid_region_utm = coco_img.img['valid_region_utm']
                            try:
                                this_utm_crs = geos_valid_region_utm['properties']['crs']['auth']
                            except KeyError:
                                this_utm_crs = coco_img.img['utm_crs_info']['auth']
                            sh_valid_region_utm = geometry.shape(geos_valid_region_utm)
                            valid_region_utm = gpd.GeoDataFrame({'geometry': [sh_valid_region_utm]}, crs=this_utm_crs)
                            valid_region_local = valid_region_utm.to_crs(local_epsg)
                            sh_valid_region_local = valid_region_local.geometry.iloc[0]
                            isect_area = sh_valid_region_local.intersection(sh_space_region_local).area
                            other_area = sh_space_region_local.area
                            valid_iooa = isect_area / other_area
                        else:
                            # If the valid_utm region does not exist, do we at
                            # least have corners?
                            utm_corners = coco_img.img.get('utm_corners', None)
                            if utm_corners is not None:
                                this_utm_crs = coco_img.img['utm_crs_info']['auth']
                                sh_valid_region_utm = kwimage.Polygon(exterior=utm_corners).to_shapely()
                                valid_region_utm = gpd.GeoDataFrame({'geometry': [sh_valid_region_utm]}, crs=this_utm_crs)
                                valid_region_local = valid_region_utm.to_crs(local_epsg)
                                sh_valid_region_local = valid_region_local.geometry.iloc[0]
                                isect_area = sh_valid_region_local.intersection(sh_space_region_local).area
                                other_area = sh_space_region_local.area
                                valid_iooa = isect_area / other_area
                            else:
                                sh_valid_region_local = None
                                valid_iooa = -1

                        score = valid_iooa
                        rows.append({
                            'score': score,
                            'gid': gid,
                            'valid_iooa': valid_iooa,
                            'fname': ub.Path(fpath).name,
                            'geometry': sh_valid_region_local,
                        })

                    # The order doesnt matter here. We will fix it after we
                    # crop the images.
                    final_gids = [
                        r['gid'] for r in sorted(rows, key=lambda r: r['score'], reverse=True)]
                    groups.append({
                        'main_gid': final_gids[0],
                        # 'other_gids': [],
                        'other_gids': final_gids[1:],
                        'sensor_coarse': sensor_coarse,
                    })
                    # Output a visualization of this group and its overlaps but
                    # only if we have that info
                    can_vis_geos = any(row['geometry'] is not None for row in rows)
                    if debug_valid_regions:
                        prog.ensure_newline()
                        print('debug_valid_regions = {!r}'.format(debug_valid_regions))
                        print('can_vis_geos = {!r}'.format(can_vis_geos))
                    if debug_valid_regions and can_vis_geos:
                        import kwplot
                        group_local_df = gpd.GeoDataFrame(rows, crs=local_epsg)
                        print('\n\n')
                        print(group_local_df)

                        debug_dpath = ub.Path(extract_dpath) / '_debug_regions'
                        debug_dpath.mkdir(exist_ok=True)
                        with kwplot.BackendContext('agg'):

                            debug_name = '{}_{}_{}_{}'.format(video_name, iso_time, space_str, sensor_coarse)

                            # Dump a visualization of the bounds of the
                            # valid region for debugging.
                            wld_map_crs84_gdf = gpd.read_file(
                                gpd.datasets.get_path('naturalearth_lowres')
                            ).to_crs('crs84')
                            sh_tight_bounds_local = unary_union([sh_space_region_local] + [row['geometry'] for row in rows if row['geometry'] is not None])
                            sh_total_bounds_local = shapely.affinity.scale(sh_tight_bounds_local.convex_hull, 2.5, 2.5)
                            total_bounds_local = gpd.GeoDataFrame({'geometry': [sh_total_bounds_local]}, crs=local_epsg)

                            subimg_crs84_df = cube.img_geos_df.loc[[r['gid'] for r in rows]]
                            subimg_local_df = subimg_crs84_df.to_crs(local_epsg)

                            wld_map_local_gdf = wld_map_crs84_gdf.to_crs(local_epsg)

                            ax = kwplot.figure(doclf=True, fnum=2).gca()
                            ax.set_title(f'Local CRS: {local_epsg}\n{iso_time} sensor={sensor_coarse} n={len(rows)} source_gids={final_gids}')
                            wld_map_local_gdf.plot(ax=ax)
                            subimg_local_df.plot(ax=ax, color='blue', alpha=0.6, edgecolor='black', linewidth=4)
                            group_local_df.plot(ax=ax, color='pink', alpha=0.6)
                            space_region_local.plot(ax=ax, color='green', alpha=0.6)
                            bounds = total_bounds_local.bounds.iloc[0]
                            ax.set_xlim(bounds.minx, bounds.maxx)
                            ax.set_ylim(bounds.miny, bounds.maxy)
                            fname = f'debug_{debug_name}_local.jpg'
                            debug_fpath = debug_dpath / fname
                            kwplot.phantom_legend({'valid region': 'pink', 'geos_bounds': 'black', 'query': 'green'}, ax=ax)
                            ax.figure.savefig(debug_fpath)

                            group_crs84_df = group_local_df.to_crs('crs84')
                            total_bounds_crs84 = total_bounds_local.to_crs('crs84')
                            ax = kwplot.figure(doclf=True, fnum=3).gca()
                            ax.set_title(f'CRS84:\n{iso_time} sensor={sensor_coarse} n={len(rows)} source_gids={final_gids}')
                            wld_map_crs84_gdf.plot(ax=ax)
                            subimg_crs84_df.plot(ax=ax, color='blue', alpha=0.6, edgecolor='black', linewidth=4)
                            group_crs84_df.plot(ax=ax, color='pink', alpha=0.6)
                            space_region_crs84.plot(ax=ax, color='green', alpha=0.6)
                            bounds = total_bounds_crs84.bounds.iloc[0]
                            ax.set_xlim(bounds.minx, bounds.maxx)
                            ax.set_ylim(bounds.miny, bounds.maxy)
                            fname = f'debug_{debug_name}_crs84.jpg'
                            debug_fpath = debug_dpath / fname
                            kwplot.phantom_legend({'valid region': 'pink', 'geos_bounds': 'black', 'query': 'green'}, ax=ax)
                            ax.figure.savefig(debug_fpath)

                            debug_info = {
                                'coco_fpath': os.path.abspath(coco_dset.fpath),
                                'gids': final_gids,
                                'rows': [ub.dict_diff(row, {'geometry'}) for row in rows],
                            }
                            fname = f'debug_{debug_name}_text.py'
                            debug_fpath = debug_dpath / fname
                            datastr = ub.repr2(debug_info, nl=2)
                            debug_text = ub.codeblock(
                                '''
                                """
                                See ~/code/watch/dev/debug_coco_geo_img.py
                                """
                                debug_info = {datastr}


                                def main():
                                    import kwcoco
                                    from watch.utils import kwcoco_extensions
                                    parent_dset = kwcoco.CocoDataset(debug_info['coco_fpath'])
                                    coco_imgs = parent_dset.images(debug_info['gids']).coco_images

                                    for coco_img in coco_imgs:
                                        gid = coco_img.img['id']
                                        kwcoco_extensions.coco_populate_geo_img_heuristics2(
                                            coco_img, overwrite=True)

                                if __name__ == '__main__':
                                    main()
                                ''').format(datastr=datastr)
                            print('write debug_fpath = {!r}'.format(debug_fpath))
                            with open(debug_fpath, 'w') as file:
                                file.write(debug_text + '\n')

            else:
                groups.append({
                    'main_gid': gids[0],
                    'other_gids': [],
                })

            # continue

            for num, group in enumerate(groups):
                main_gid = group['main_gid']
                other_gids = group['other_gids']
                img = coco_dset.imgs[main_gid]
                other_imgs = [coco_dset.imgs[x] for x in other_gids]

                # There is an issue of merging annotations here
                anns = [coco_dset.index.anns[aid] for aid in
                        coco_dset.index.gid_to_aids[main_gid]]
                if 0:
                    # FIXME: do we do this? Rectification step?
                    for other_gid in other_gids:
                        anns += [coco_dset.index.anns[aid] for aid in
                                 coco_dset.index.gid_to_aids[other_gid]]

                # if len(other_gids) == 0:
                #     # Hack: only look at weird cases
                #     continue

                sensor_coarse = img.get('sensor_coarse', 'unknown')
                # Construct a name for the subregion to extract.
                name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)

                job = image_jobs.submit(
                    extract_image_job,
                    img, anns, bundle_dpath, new_bundle_dpath, name, datetime_,
                    num, frame_index, new_vidid, rpc_align_method,
                    sub_bundle_dpath, space_str, space_region, space_box,
                    start_gid, start_aid, aux_workers, keep,
                    local_epsg=local_epsg, other_imgs=other_imgs,
                    include_channels=include_channels,
                    exclude_channels=exclude_channels,
                    force_nodata=force_nodata,
                    tries=tries,
                    image_timeout=image_timeout,
                    asset_timeout=asset_timeout,
                    verbose=verbose,
                    force_min_gsd=force_min_gsd)
                start_gid = start_gid + 1
                start_aid = start_aid + len(anns)
                frame_index = frame_index + 1

        # return

        sub_new_gids = []
        sub_new_aids = []
        if image_timeout is not None:
            image_timeout = util_time.coerce_timedelta(image_timeout).total_seconds()

        from concurrent.futures import TimeoutError
        for job in image_jobs.as_completed(desc='collect extract jobs',
                                           progkw=dict(freq=1)):

            try:
                new_img, new_anns = job.result(timeout=image_timeout)
            except SkipImage:
                continue
            except TimeoutError:
                print('\n\nAn image job timed out!\n\n')
                continue

            # Hack, the next ids dont update when new images are added
            # with explicit ids. This is a quick fix.
            new_img.pop('id', None)

            new_img['video_id'] = new_vidid

            new_gid = new_dset.add_image(**new_img)
            sub_new_gids.append(new_gid)

            for ann in new_anns:
                ann.pop('id', None)  # quick hack fix
                ann['image_id'] = new_gid
                new_aid = new_dset.add_annotation(**ann)
                sub_new_aids.append(new_aid)

        if True:
            from kwcoco.util import util_json
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
                unserializable = list(util_json.find_json_unserializable(new_img))
                if unserializable:
                    raise AssertionError('unserializable(gid={}) = {}'.format(new_gid, ub.repr2(unserializable, nl=0)))

        kwcoco_extensions.coco_populate_geo_video_stats(
            new_dset, target_gsd=target_gsd, vidid=new_vidid)

        # Enable if serialization is breaking
        if False:
            for new_gid in sub_new_gids:
                # Fix json serializability
                new_img = new_dset.index.imgs[new_gid]
                new_objs = [new_img] + new_img.get('auxiliary', [])
                unserializable = list(util_json.find_json_unserializable(new_img))
                if unserializable:
                    print('new_img = {}'.format(ub.repr2(new_img, nl=1)))
                    raise AssertionError('unserializable(gid={}) = {}'.format(new_gid, ub.repr2(unserializable, nl=0)))

            for new_aid in sub_new_aids:
                new_ann = new_dset.index.anns[new_aid]
                unserializable = list(util_json.find_json_unserializable(new_ann))
                if unserializable:
                    print('new_ann = {}'.format(ub.repr2(new_ann, nl=1)))
                    raise AssertionError('unserializable(aid={}) = {}'.format(new_aid, ub.repr2(unserializable, nl=1)))

            for new_vidid in [new_vidid]:
                new_video = new_dset.index.videos[new_vidid]
                unserializable = list(util_json.find_json_unserializable(new_video))
                if unserializable:
                    print('new_video = {}'.format(ub.repr2(new_video, nl=1)))
                    raise AssertionError('unserializable(vidid={}) = {}'.format(new_vidid, ub.repr2(unserializable, nl=1)))

        # unserializable = list(util_json.find_json_unserializable(new_dset.dataset))
        # if unserializable:
        #     raise AssertionError('unserializable = {}'.format(ub.repr2(unserializable, nl=1)))

        if visualize:
            new_video = new_dset.index.videos[new_vidid]
            local_max_frame = len(sub_new_gids)
            valid_vidspace_region = new_video.get('valid_region', None)
            for frame_idx, new_gid in enumerate(ub.ProgIter(sub_new_gids, desc='visualizing')):
                new_img = new_dset.imgs[new_gid]
                new_anns = new_dset.annots(gid=new_gid).objs
                viz_dpath = ub.Path(sub_bundle_dpath) / '_viz'
                # Use false color for special groups
                request_grouped_bands = [
                    'red|green|blue',
                    'nir|swir16|swir22',
                ]
                if isinstance(visualize, str):
                    channels_ = visualize
                else:
                    channels_ = None
                _write_ann_visualizations2(
                    coco_dset=new_dset, img=new_img, anns=new_anns,
                    channels=channels_,
                    sub_dpath=viz_dpath, space='video',
                    request_grouped_bands=request_grouped_bands,
                    local_frame_index=frame_idx,
                    local_max_frame=local_max_frame,
                    valid_vidspace_region=valid_vidspace_region,
                    # verbose=3
                )

        if write_subsets:
            print('Writing data subset')
            if 0:
                # Enable if json serialization is breaking
                new_dset._check_json_serializable()

            sub_dset = new_dset.subset(sub_new_gids, copy=True)
            sub_dset.fpath = join(sub_bundle_dpath, 'subdata.kwcoco.json')
            sub_dset.reroot(new_root=sub_bundle_dpath, absolute=False)

            kwcoco_extensions.reorder_video_frames(sub_dset)

            sub_dset.dump(sub_dset.fpath, newlines=True)
        return new_dset


@profile
def extract_image_job(img, anns, bundle_dpath, new_bundle_dpath, name,
                      datetime_, num, frame_index, new_vidid, rpc_align_method,
                      sub_bundle_dpath, space_str, space_region, space_box,
                      start_gid, start_aid, aux_workers=0, keep=False,
                      local_epsg=None, other_imgs=None,
                      include_channels=None,
                      exclude_channels=None,
                      force_nodata=None,
                      tries=2,
                      asset_timeout=None,
                      image_timeout=None,
                      verbose=0,
                      force_min_gsd=None):
    """
    Threaded worker function for :func:`SimpleDataCube.extract_overlaps`.

    Returns:
        Tuple[Dict, Dict] : new_img, new_anns
    """
    # from tempenv import TemporaryEnvironment  # NOQA
    # Does this resolve import issues?
    # with TemporaryEnvironment({'PROJ_LIB': None, 'PROJ_DEBUG': '3'}):

    # Hacks around projdb issue? Seems to force the proj_lib to be set
    # in some global variable such that we can access it later.
    # Attempt at MWE in dev/devcheck_coordinate_transform.py but
    # is not fully working for the thread case yet.
    # Removing this get will result in
    #     ```
    #     ERROR 1: PROJ: proj_create: unrecognized format / unknown name
    #     ERROR 1: PROJ: proj_create_from_database: Cannot find proj.db
    #     ```
    # When the cache is not computed and workers > 0
    from osgeo import osr
    osr.GetPROJSearchPaths()
    from kwcoco.coco_image import CocoImage

    coco_img = CocoImage(img)
    has_base_image = img.get('file_name', None) is not None
    objs = [ub.dict_diff(obj, {'auxiliary', 'assets'})
            for obj in coco_img.iter_asset_objs()]
    sensor_coarse = img.get('sensor_coarse', 'unknown')

    channels_to_objs = ub.ddict(list)
    for obj in objs:
        key = obj['channels']
        if key in channels_to_objs:
            warnings.warn(ub.paragraph(
                '''
                It seems multiple auxiliary items in the parent image might
                contain the same channel.  This script will try to work around
                this, but that is not a valid kwcoco assumption.
                '''))
        # assert key not in channels_to_objs
        channels_to_objs[key].append(obj)

    for other_img in other_imgs:
        coco_other_img = CocoImage(other_img)
        other_objs = [ub.dict_diff(obj, {'auxiliary'})
                      for obj in coco_other_img.iter_asset_objs()]
        for other_obj in other_objs:
            key = other_obj['channels']
            channels_to_objs[key].append(other_obj)
    obj_groups = list(channels_to_objs.values())
    is_multi_image = len(obj_groups) > 1

    is_rpc = False
    for obj in objs:
        # TODO fix this, probably WV from smart-stac and smart-imagery mixed?
        # is_rpcs = [obj['geotiff_metadata']['is_rpc'] for obj in objs]
        # is_rpc = ub.allsame(is_rpcs)
        if 'is_rpc' in obj:
            if obj['is_rpc']:
                is_rpc = True
        else:
            if 'geotiff_metadata' not in obj:
                kwcoco_extensions._populate_canvas_obj(bundle_dpath, obj,
                                                       keep_geotiff_metadata=True)
            if obj['geotiff_metadata']['is_rpc']:
                is_rpc = True

    if is_rpc and rpc_align_method != 'affine_warp':
        align_method = rpc_align_method
    else:
        align_method = 'affine_warp'

    dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))

    job_list = []

    # Turn off internal threading because we refactored this to thread over all
    # images instead
    asset_jobs = ub.JobPool(mode='thread', max_workers=aux_workers)

    if image_timeout is not None:
        image_timeout = util_time.coerce_timedelta(image_timeout).total_seconds()
    if asset_timeout is not None:
        asset_timeout = util_time.coerce_timedelta(asset_timeout).total_seconds()

    aux_verbose = verbose > 3 or (verbose > 1 and (aux_workers == 0))
    for obj_group in ub.ProgIter(obj_groups, desc='submit warp assets', verbose=verbose):
        job = asset_jobs.submit(
            _aligncrop, obj_group, bundle_dpath, name, sensor_coarse,
            dst_dpath, space_region, space_box, align_method,
            is_multi_image, keep, local_epsg=local_epsg,
            include_channels=include_channels,
            exclude_channels=exclude_channels,
            force_nodata=force_nodata,
            tries=tries,
            asset_timeout=asset_timeout,
            verbose=aux_verbose,
            force_min_gsd=force_min_gsd)
        job_list.append(job)

    dst_list = []
    for job in asset_jobs.as_completed(desc='collect warp assets {}'.format(name),
                                       timeout=image_timeout,
                                       progkw=dict(enabled=DEBUG, verbose=verbose)):
        dst = job.result(timeout=asset_timeout)
        dst_list.append(dst)

    new_gid = start_gid

    for i in range(100000000):
        i += 1

    if verbose > 2:
        print(f'Finish channel crop jobs: {new_gid}')

    if align_method != 'pixel_crop':
        # If we are a pixel crop, we can transform directly
        for dst in dst_list:
            if dst is not None:
                # TODO: We should not populate this for computed features!
                # hack this in for heuristics
                if 'sensor_coarse' in img:
                    dst['sensor_coarse'] = img['sensor_coarse']
                # We need to overwrite because we changed the bounds
                # Note: if band info is not popluated above, this
                # might write bad data based on hueristics
                # TODO:
                # We need to remove all spatial metadata from the base image that a
                # crop would invalidate, otherwise we will propogate bad info.
                kwcoco_extensions._populate_canvas_obj(
                    bundle_dpath, dst, overwrite={'warp'}, with_wgs=True)
        if DEBUG:
            print(f'Finish repopulate canvas jobs: {new_gid}')

    assert len(dst_list) == len(obj_groups)
    # Hack because heurstics break when fnames change
    for old_aux_group, new_aux in zip(obj_groups, dst_list):
        if new_aux is not None:
            if len(old_aux_group) > 1:
                new_aux['parent_file_name'] = [g['file_name'] for g in old_aux_group]
            else:
                new_aux['parent_file_name'] = old_aux_group[0]['file_name']

    new_img = {
        'id': new_gid,
        'name': name,
        'align_method': align_method,
    }

    if has_base_image and len(dst_list) == 1:
        base_dst = dst_list[0]
        new_img.update(base_dst)
        aux_dst = []
        # dst_list[1:]
    else:
        aux_dst = dst_list

    aux_dst = [aux for aux in aux_dst if aux is not None]

    if len(aux_dst):
        if DEBUG:
            print(f'Recompute auxiliary transforms: {new_gid}')
        new_img['auxiliary'] = aux_dst
        kwcoco_extensions._recompute_auxiliary_transforms(new_img)

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
    new_img['timestamp'] = datetime_.timestamp()

    new_coco_img = CocoImage(new_img)

    if not len(list(new_coco_img.iter_asset_objs())):
        # This image did not contained any requested bands. Skip it.
        raise SkipImage

    new_coco_img._bundle_dpath = new_bundle_dpath
    new_coco_img._video = {}
    kwcoco_extensions._populate_valid_region(new_coco_img)

    # HANDLE ANNOTATIONS
    # Note: this is more generally handled by the project annotation script.
    # We can add an option to ignore annotations here.
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
        pxl_polys = geo_polys.warp(
            new_img['wgs84_to_wld']
        ).warp(new_img['wld_to_pxl'])
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

    if DEBUG:
        print(f'Finished extract img job: {new_gid}')
    return new_img, new_anns


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


@profile
def _aligncrop(obj_group, bundle_dpath, name, sensor_coarse, dst_dpath, space_region,
               space_box, align_method, is_multi_image, keep, local_epsg=None,
               include_channels=None, exclude_channels=None, tries=2,
               asset_timeout=None, force_nodata=None, verbose=0, force_min_gsd=None):
    import watch

    # NOTE: https://github.com/dwtkns/gdal-cheat-sheet
    first_obj = obj_group[0]
    chan_code = obj_group[0].get('channels', '')

    # Prevent long names for docker (limit is 242 chars)
    channels_ = kwcoco.FusedChannelSpec.coerce(chan_code)
    chan_pname = channels_.path_sanitize(maxlen=10)

    if include_channels is not None:
        # Filter out bands we are not interested in
        include_channels = kwcoco.FusedChannelSpec.coerce(include_channels)
        if not channels_.intersection(include_channels).numel():
            if verbose > 2:
                print('Skip not included {}'.format(channels_))
            return None

    if exclude_channels is not None:
        # Filter out bands we are not interested in
        exclude_channels = kwcoco.FusedChannelSpec.coerce(exclude_channels)
        if channels_.difference(exclude_channels).numel() == 0:
            if verbose > 2:
                print('Skip excluded {}'.format(channels_))
            return None

    if is_multi_image:
        multi_dpath = ub.ensuredir((dst_dpath, name))
        dst_gpath = join(multi_dpath, name + '_' + chan_pname + '.tif')
    else:
        dst_gpath = join(dst_dpath, name + '.tif')

    input_gnames = [obj.get('file_name', None) for obj in obj_group]
    assert all(n is not None for n in input_gnames)
    input_gpaths = [join(bundle_dpath, n) for n in input_gnames]

    # PHASE1_DEADLINE_HACK = 1
    # if PHASE1_DEADLINE_HACK:
    #     if len(input_gpaths) == 1 and input_gpaths[0].endswith('TCI.jp2'):
    #         return None

    dst = {
        'file_name': dst_gpath,
    }
    if first_obj.get('channels', None):
        dst['channels'] = first_obj['channels']
    if first_obj.get('num_bands', None):
        dst['num_bands'] = first_obj['num_bands']

    already_exists = exists(dst_gpath)
    needs_recompute = not (already_exists and keep in {'img', 'roi-img'})

    if not needs_recompute:
        DOUBLE_CHECK = 1
        if DOUBLE_CHECK:
            # Sometimes the data will exist, but it's bad data. Check for this.
            dst_gpath = ub.Path(dst_gpath)
            try:
                ref = util_gdal.GdalOpen(dst_gpath, mode='r')
                ref
            except RuntimeError:
                # Data is likely corrupted
                needs_recompute = True
                print(f'The data exists {dst_gpath}, but is corrupted. Recomputing')
                dst_gpath.delete()
                pass
            else:
                ref = None

    if not needs_recompute:
        # if 'crop_20191014T130000Z_S23.539915W046.611400_S23.283329W046.288255_S2_0' in dst_gpath:
        #     import xdev
        #     xdev.embed()
        if verbose:
            print('cache hit dst = {!r}'.format(dst))
        return dst

    if align_method == 'pixel_crop':
        raise NotImplementedError('no longer supported')

    if align_method == 'orthorectify':
        if 'geotiff_metadata' in first_obj:
            info = first_obj['geotiff_metadata']
        else:
            info = watch.gis.geotiff.geotiff_crs_info(input_gpaths[0])
        # No RPCS exist, use affine-warp instead
        rpcs = info['rpc_transform']
    else:
        rpcs = None

    duplicates = ub.find_duplicates(input_gpaths)
    if duplicates:
        warnings.warn(ub.paragraph(
            '''
            Input to _aligncrop contained duplicate filepaths, the same image
            might be registered in the base kwcoco file multiple times.
            '''))
        # print('!!WARNING!! duplicates = {}'.format(ub.repr2(duplicates, nl=1)))
        input_gpaths = list(ub.oset(input_gpaths))

    nodata = force_nodata

    # When trying to get a gdalmerge to take multiple inputs I got a Attempt to
    # create 0x0 dataset is illegal,sizes must be larger than zero.  This new
    # method will call gdalwarp on each image individually and then merge them
    # all in a final step.
    out_fpath = dst_gpath
    if verbose > 2:
        print(
            'start gdal warp in_fpaths = {}'.format(ub.repr2(input_gpaths, nl=1)) +
            'chan_code = {!r}\n'.format(chan_code) +
            '\n* dst_gpath = {!r}'.format(dst_gpath))

    error_logfile = None
    # Uncomment to suppress warnings for debug purposes
    #
    # error_logfile = '/dev/null'

    # TODO: add a timeout argument to the gdal calls and pass down the asset
    # timeout.

    # Note: these methods take care of retries and checking that the
    # data is valid.
    force_spatial_res = None
    if force_min_gsd is not None:
        if 'geotiff_metadata' in first_obj:
            info = first_obj['geotiff_metadata']
        else:
            warnings.warn(ub.paragraph(
                '''
                Popluating geotiff crs info, which probably should
                have already been populated; to ensure pre-population
                use the 'geo_preprop' argument.
                '''))
            info = watch.gis.geotiff.geotiff_crs_info(input_gpaths[0])

        if 'approx_meter_gsd' in info and info['approx_meter_gsd'] < force_min_gsd:
            # Only setting if needed to avoid needless warping if the
            # 'approximate_meter_gsd' value is slightly different from
            # what GDAL computes at the time of warping
            force_spatial_res = force_min_gsd

    if len(input_gpaths) > 1:
        in_fpaths = input_gpaths
        util_gdal.gdal_multi_warp(in_fpaths, out_fpath, space_box=space_box,
                                  local_epsg=local_epsg, rpcs=rpcs,
                                  nodata=nodata, tries=tries,
                                  error_logfile=error_logfile,
                                  verbose=0 if verbose < 2 else verbose)
    else:
        in_fpath = input_gpaths[0]
        util_gdal.gdal_single_warp(in_fpath, out_fpath,
                                   space_box=space_box, local_epsg=local_epsg,
                                   rpcs=rpcs, nodata=nodata,
                                   tries=tries,
                                   error_logfile=error_logfile,
                                   verbose=0 if verbose < 2 else verbose,
                                   force_spatial_res=force_spatial_res)
    if verbose > 2:
        print('finish gdal warp dst_gpath = {!r}'.format(dst_gpath))
    return dst


class SkipImage(Exception):
    ...


_CLI = CocoAlignGeotiffConfig
__config__ = CocoAlignGeotiffConfig


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.coco_align_geotiffs --help
    """
    main()
