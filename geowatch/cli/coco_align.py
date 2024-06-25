#!/usr/bin/env python3
r"""
Given the raw data in kwcoco format, this script will extract orthorectified
regions around areas of interest across time.

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

The main corner case in the above process is when one image has ``r|g|b`` but
another image has ``r|g|b|yellow``, there is no logic to split those channels
out at the moment.


The following is an end-to-end example works on public data

CommandLine:
    # Create a demo region file
    xdoctest geowatch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-06-10-V2
    DEMO_DPATH=$HOME/.cache/geowatch/demo/datasets

    REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/geowatch/demo/annotations/KHQ_R001_sites/*.geojson"

    START_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.start_date' "$REGION_FPATH")
    END_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.end_date' "$REGION_FPATH")
    # Shrink time window to test with less data
    START_DATE=2016-12-02
    END_DATE=2020-12-31
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    SEARCH_FPATH=$DEMO_DPATH/stac_search.json
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input
    CATALOG_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}_catalog.json
    KWCOCO_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.kwcoco.zip
    KWCOCO_FIELDED_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}-fielded.kwcoco.zip
    KWCOCO_ALIGNED_DPATH=$DEMO_DPATH/all_sensors_kit/cropped/
    KWCOCO_ALIGNED_FPATH=$DEMO_DPATH/all_sensors_kit/cropped/${REGION_ID}-fielded.kwcoco.zip

    mkdir -p "$DEMO_DPATH"

    # Create the search json wrt the sensors and processing level we want
    python -m geowatch.stac.stac_search_builder \
        --start_date="$START_DATE" \
        --end_date="$END_DATE" \
        --cloud_cover=40 \
        --sensors=sentinel-2-l2a \
        --out_fpath "$SEARCH_FPATH"
    cat "$SEARCH_FPATH"

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"

    # Create the .input file
    # use max_products_per_region to keep the result small
    python -m geowatch.cli.stac_search \
        --region_file "$REGION_FPATH" \
        --search_json "$SEARCH_FPATH" \
        --mode area \
        --verbose 2 \
        --max_products_per_region 10 \
        --outfile "${RESULT_FPATH}"

    cat "$RESULT_FPATH"

    python -m geowatch.cli.baseline_framework_ingress \
        --input_path="$RESULT_FPATH" \
        --catalog_fpath="${CATALOG_FPATH}" \
        --virtual=True \
        --jobs=avail \
        --aws_profile=iarpa \
        --requester_pays=0

    AWS_DEFAULT_PROFILE=iarpa python -m geowatch.cli.stac_to_kwcoco \
        --input_stac_catalog="${CATALOG_FPATH}" \
        --outpath="$KWCOCO_FPATH" \
        --jobs=8 \
        --from_collated=False \
        --ignore_duplicates=0

    # Check that the resulting kwcoco has what you want in it
    geowatch stats "$KWCOCO_FPATH"

    # Use kwcoco info to dump a single image dictionary
    kwcoco info "$KWCOCO_FPATH" -g 1 -i 0

    # Prefetch header metadata from remote assets
    AWS_DEFAULT_PROFILE=iarpa python -m geowatch.cli.coco_add_watch_fields \
        --src="$KWCOCO_FPATH" \
        --dst="$KWCOCO_FIELDED_FPATH" \
        --overwrite=warp \
        --workers=8 \
        --enable_video_stats=False \
        --target_gsd=10 \
        --remove_broken=True \
        --skip_populate_errors=False

    # Use kwcoco info to see what info fielding added
    kwcoco info "$KWCOCO_FPATH" -g 1 -i 0

    # Perform the crop to create an aligned dataset with videos
    AWS_DEFAULT_PROFILE=iarpa python -m geowatch.cli.coco_align \
        --regions "$REGION_FPATH" \
        --context_factor=1 \
        --geo_preprop=auto \
        --keep=img \
        --force_nodata=None \
        --include_channels="None" \
        --exclude_channels="None" \
        --debug_valid_regions=False \
        --rpc_align_method orthorectify \
        --sensor_to_time_window "None" \
        --verbose=0 \
        --aux_workers=0 \
        --target_gsd=10 \
        --force_min_gsd=None \
        --workers=26 \
        --tries=2 \
        --asset_timeout=1hours \
        --image_timeout=2hours \
        --hack_lazy=False \
        --src="$KWCOCO_FIELDED_FPATH" \
        --dst="$KWCOCO_FIELDED_FPATH" \
        --dst_bundle_dpath=$KWCOCO_ALIGNED_DPATH



Note:
    # Example invocation to create the full drop1 aligned dataset

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    INPUT_COCO_FPATH=$DVC_DPATH/drop1/data.kwcoco.json
    OUTPUT_COCO_FPATH=$DVC_DPATH/drop1-WV-only-aligned/data.kwcoco.json
    REGION_FPATH=$DVC_DPATH/drop1/all_regions.geojson
    VIZ_DPATH=$OUTPUT_COCO_FPATH/_viz_video

    # Quick stats about input datasets
    python -m kwcoco stats $INPUT_COCO_FPATH
    python -m geowatch stats $INPUT_COCO_FPATH

    # Combine the region models
    python -m geowatch.cli.merge_region_models \
        --src $DVC_DPATH/drop1/region_models/*.geojson \
        --dst $REGION_FPATH

    python -m geowatch.cli.coco_add_watch_fields \
        --src $INPUT_COCO_FPATH \
        --dst $INPUT_COCO_FPATH.prepped \
        --workers 16 \
        --target_gsd=10

    # Execute alignment / crop script
    python -m geowatch.cli.coco_align \
        --src $INPUT_COCO_FPATH.prepped \
        --dst $OUTPUT_COCO_FPATH \
        --regions $REGION_FPATH \
        --rpc_align_method orthorectify \
        --workers=10 \
        --aux_workers=2 \
        --context_factor=1 \
        --geo_preprop=False \
        --include_sensors=WV \
        --keep img

TODO:
    - [ ] Should this script have the option of calling "remove_bad_images" or
          "clean_geotiffs" to prevent generating bad images in the first place?
"""
import os
import scriptconfig as scfg
import ubelt as ub
import warnings

DEBUG = 1

try:
    from line_profiler import profile
except Exception:
    profile = ub.identity


class AssetExtractConfig(scfg.DataConfig):
    """
    Part of the extract config for asset jobs
    """
    keep = scfg.Value(None, help=ub.paragraph(
        '''
        Level of detail to overwrite existing data at, since this is slow.
        "none": overwrite all, including existing images "img": only add new
        images "roi": only add new ROIs "roi-img": only add new ROIs and only
        new images within those ROIs (good for rerunning failed jobs)
        '''))

    corruption_checks = scfg.Value(False, help=ub.paragraph(
        '''
        Check for image cache corruption after a "cache hit" to make sure we
        can read the image and it isn't corrupted. If it is, delete it and
        reprocess.
        '''))

    include_channels = scfg.Value(None, help=ub.paragraph(
        '''
        If specified only align the given channels
        '''))
    exclude_channels = scfg.Value(None, help='If specified ignore these channels')

    force_nodata = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, forces nodata to this value (e.g. -9999) Ideally this is
        not needed and all source geotiffs properly specify nodata.

        NOTE: For quality bands WE DO NOT RESPECT THIS. This may change in the
        future to have a more sane way of dealing with it. But hacking it to
        zero for now. If this is specified we force it to zero.  If this is
        None, we DO use that for the quality bands.

        NOTE: We currently must specify this to handle gdal-merge correctly.
        Perhasp in the future we may be able to introspect, but for now specify
        it.
        '''))

    unsigned_nodata = scfg.Value(256, help=ub.paragraph(
        '''
        The nodata value for unsigned UInt16 quality bitmasks. This can be
        different from the cannonical Int16 data bands. The default value
        corresponds to Accenture-2 standards where 256 is the "fill value".
        This is only used if ``force_nodata`` is specified.
        '''))

    qa_encoding = scfg.Value(None, help=ub.paragraph(
        '''
        TEMPORARY WORKAROUND.
        The value of this parameter will be set as the "qa_encoding" for any
        assets with the "quality" role. Ideally this information is determined
        from the input kwcoco, which itself should be derived from the STAC
        catalog. There is no reason to assume this is uniform.
        Additionally, when we do add logic that enriches the input with this
        metadata, that will no remove the requirement for the "unsigned_nodata"
        argument, as it should be able to be inferred at that point.
        '''))

    tries = scfg.Value(2, help=ub.paragraph(
        '''
        The maximum number of times to retry failed gdal warp commands before
        stopping.
        '''), alias=['warp_tries'])

    cooldown = scfg.Value(10, help='seconds between tries after a failed attempt')

    backoff = scfg.Value(3.0, help='factor to multiply cooldown by after a failed attempt')

    asset_timeout = scfg.Value('4hours', help=ub.paragraph(
        '''
        The maximum amount of time to spend pulling down a single
        image asset before giving up
        '''))

    force_min_gsd = scfg.Value(None, help=ub.paragraph(
        '''
        Force output crops to be at least this minimum GSD (e.g. if set to 10.0
        an input image with a 30.0 GSD will have an output GSD of 30.0, whereas
        in input image with a 0.5 GSD will have it set to 10.0 during cropping)
        '''))

    skip_previous_errors = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        Skip assets where we can detect a previous error occurred.
        '''))

    hack_lazy = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        Hack lazy is a proof of concept with the intent on speeding up the
        download / cropping of data by flattening the gdal processing into a
        single queue of parallel processes executed via a command queue. By
        running once with this flag on, it will execute the command queue, and
        then running again, it should see all of the data as existing and
        construct the aligned kwcoco dataset as normal.
        '''))

    verbose = scfg.Value(0, help=ub.paragraph(
        '''
        Note: no silent mode, 0 is just least verbose.
        '''))

    def __post_init__(config):
        from geowatch.utils.util_resolution import ResolvedUnit
        if config['force_min_gsd'] is not None:
            # FIXME: unit should be "meters" or "mGSD"
            resolution = ResolvedUnit.coerce(config['force_min_gsd'], default_unit='GSD')
            assert resolution.unit == 'GSD'
            config['force_min_gsd'] = resolution.mag


class ImageExtractConfig(AssetExtractConfig):
    """
    Part of the extract config for image jobs
    """
    # TODO: change this name to just align-method or something
    rpc_align_method = scfg.Value('orthorectify', help=ub.paragraph(
        '''
        Can be one of:
        (1) orthorectify - which uses gdalwarp with -rpc if available otherwise
        falls back to affine transform,
        (2) affine_warp - which ignores RPCs and uses the affine transform in
        the geotiff metadata.
        '''))

    aux_workers = scfg.Value(0, type=str, help='additional inner threads for aux imgs', alias=['asset_workers'])

    image_timeout = scfg.Value('8hours', help=ub.paragraph(
        '''
        The maximum amount of time to spend pulling down a all image
        assets before giving up
        '''))

    num_start_frames = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, attempt to only gather this many high quality images at
        the start of a sequence.
        '''))

    num_end_frames = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, attempt to only gather this many high quality images at
        the end of a sequence.
        '''))

    image_error_policy = scfg.Value('raise', help=ub.paragraph(
        '''
        What to do when input contain no regions to crops. Can be "ignore"
        to write an empty kwcoco file or "raise" to throw an Exception.
        '''), choices=['raise', 'ignore'])


class ExtractConfig(ImageExtractConfig):
    """
    This is a subset of the above config for arguments a passed to
    extract_overlaps. We may use this config as a base class to inherit from,
    but for now we duplicate param names.
    """
    write_subsets = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        if True, writes a separate kwcoco file for every discovered
        ROI in addition to the final kwcoco file.
        '''))

    img_workers = scfg.Value(0, type=str, help=ub.paragraph(
        '''
        number of parallel procs. This can also be an expression
        accepted by coerce_num_workers.
        '''), alias=['max_workers', 'workers'])

    target_gsd = scfg.Value(10, help=ub.paragraph(
        '''
        The **virtual** GSD to use as the "video-space" for output files.
        '''))

    debug_valid_regions = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        write valid region visualizations to help debug "black
        images" issues.
        '''))

    max_frames = scfg.Value(None, help=ub.paragraph(
        '''
        Limit the number of frames per video (mainly for debugging)
        '''))

    sensor_to_time_window = scfg.Value(None, help=ub.paragraph(
        '''
        Specify a yaml mapping from a sensor to a time window. We will chunk up
        candidate images based on this window and only choose 1 image per
        chunk with the lowest cloud cover using earlier images as tiebreakers.
        '''))

    def __post_init__(config):
        super().__post_init__()
        from geowatch.utils.util_resolution import ResolvedUnit
        resolution = ResolvedUnit.coerce(config['target_gsd'], default_unit='GSD')
        assert resolution.unit == 'GSD'
        config['target_gsd'] = resolution.mag


class CocoAlignGeotiffConfig(ExtractConfig):
    """
    Create a dataset of aligned temporal sequences around objects of interest
    in an unstructured collection of annotated geotiffs.

    High Level Steps:
        * Find a set of geospatial AOIs
        * For each AOI find all images that overlap
        * Orthorectify (or warp) the selected spatial region and its
          annotations to a cannonical space.
    """
    __command__ = 'align'
    __alias__ = ['coco_align', 'coco_align_geotiff']

    src = scfg.Value('in.geojson.json', help='input dataset to chip', group='inputs')
    dst = scfg.Value(None, help=ub.paragraph(
        '''
        bundle directory or kwcoco json file for the output
        '''), group='outputs')

    dst_bundle_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, this is the directory where the output bundle will be
        created. This can be used when dst is a kwcoco file that lives
        somewhere other than the top level bundle path. This cannot be used if
        dst is a directory.
        '''))

    regions = scfg.Value('annots', help=ub.paragraph(
        '''
        The path to a set of geojson input region or site models.  Can also be
        a strategy for extracting regions, if annots, uses the convex hulls of
        clustered annotations (Note: the annots option is old, not well
        supported, and may be deprecated).
        '''), group='inputs')

    site_summary = scfg.Value(False, help=ub.paragraph(
        '''
        if False, crop to region geometry.
        if True, crop to site or site-summary geometry instead.
        '''))

    context_factor = scfg.Value(1.0, help=ub.paragraph(
        '''
        Scale factor to expand each ROI by crop regions by.
        '''))
    minimum_size = scfg.Value(None, help=ub.paragraph(
        '''
        Minimum (bounding-box) size of each ROI. Must be specified as
        ``<w> x <h> @ <magnitude> <resolution>``. E.g.  ``128x128@10GSD``
        will ensure a region polygon that is at least 1280 meters tall and
        wide.
        '''))
    convexify_regions = scfg.Value(False, help=ub.paragraph(
        '''
        if True, ensure that the regions are convex
        '''))

    geo_preprop = scfg.Value('auto', help='force if we check geo properties or not')

    include_sensors = scfg.Value(None, help=ub.paragraph(
        '''
        if specified can be comma separated valid sensors
        '''))
    exclude_sensors = scfg.Value(None, help=ub.paragraph(
        '''
        if specified can be comma separated invalid sensors
        '''))

    edit_geotiff_metadata = scfg.Value(False, help=ub.paragraph(
        '''
        if True MODIFIES THE UNDERLYING IMAGES to ensure geodata is
        propogated
        '''))

    empty_region_policy = scfg.Value('ignore', help=ub.paragraph(
        '''
        What to do when input contain no regions to crops. Can be "ignore"
        to write an empty kwcoco file or "raise" to throw an Exception.
        '''), choices=['raise', 'ignore'])


@profile
def main(cmdline=True, **kw):
    """
    Main function for coco_align.
    See :class:``CocoAlignGeotiffConfig` for details

    CommandLine:
        xdoctest -m geowatch.cli.coco_align main:0

    Example:
        >>> from geowatch.cli.coco_align import *  # NOQA
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product
        >>> from geowatch.gis.geotiff import geotiff_metadata
        >>> # Create a dead simple coco dataset with one image
        >>> import dateutil.parser
        >>> import kwcoco
        >>> import kwimage
        >>> coco_dset = kwcoco.CocoDataset()
        >>> ls_prod = grab_landsat_product()
        >>> fpath = ls_prod['bands'][0]
        >>> meta = geotiff_metadata(fpath)
        >>> # We need a date captured ATM in a specific format
        >>> dt = dateutil.parser.parse(
        >>>     meta['filename_meta']['acquisition_date'])
        >>> date_captured = dt.strftime('%Y/%m/%d')
        >>> gid = coco_dset.add_image(file_name=fpath, date_captured=date_captured)
        >>> dummy_poly = kwimage.Polygon.from_geojson(meta['geos_corners'])
        >>> dummy_poly = dummy_poly.scale(0.03, about='center')
        >>> sseg_geos = dummy_poly.to_geojson()
        >>> # NOTE: script is not always robust to missing annotation
        >>> # information like segmentation and bad bbox, but for this
        >>> # test config it is
        >>> coco_dset.add_annotation(
        >>>     image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)
        >>> #
        >>> # Create arguments to the script
        >>> dpath = ub.Path.appdir('geowatch/tests/coco_align').ensuredir()
        >>> dst = (dpath / 'align_bundle1').ensuredir()
        >>> dst.delete()
        >>> dst.ensuredir()
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': 'annots',
        >>>     'workers': 2,
        >>>     'aux_workers': 2,
        >>>     'convexify_regions': True,
        >>>     'minimum_size': '8000x8000 @ 1GSD',
        >>>     #'image_timeout': '1 microsecond',
        >>>     #'asset_timeout': '1 microsecond',
        >>>     'hack_lazy': 0,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)

    Example:
        >>> # Test timeout
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTESTS)
        >>> from geowatch.cli.coco_align import *  # NOQA
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product
        >>> from geowatch.gis.geotiff import geotiff_metadata
        >>> # Create a dead simple coco dataset with one image
        >>> import dateutil.parser
        >>> import kwcoco
        >>> import kwimage
        >>> coco_dset = kwcoco.CocoDataset()
        >>> ls_prod = grab_landsat_product()
        >>> fpath = ls_prod['bands'][0]
        >>> meta = geotiff_metadata(fpath)
        >>> # We need a date captured ATM in a specific format
        >>> dt = dateutil.parser.parse(
        >>>     meta['filename_meta']['acquisition_date'])
        >>> date_captured = dt.strftime('%Y/%m/%d')
        >>> gid = coco_dset.add_image(file_name=fpath, date_captured=date_captured)
        >>> dummy_poly = kwimage.Polygon.from_geojson(meta['geos_corners'])
        >>> dummy_poly = dummy_poly.scale(0.03, about='center')
        >>> sseg_geos = dummy_poly.to_geojson()
        >>> from geowatch.geoannots import geomodels
        >>> region = geomodels.RegionModel.random(region_poly=dummy_poly, start_time=dt.isoformat())
        >>> # Create arguments to the script
        >>> dpath = ub.Path.appdir('geowatch/tests/coco_align').ensuredir()
        >>> dst = (dpath / 'align_bundle_timeout').ensuredir()
        >>> dst.delete()
        >>> dst.ensuredir()
        >>> region_fpath = dpath / 'region_model.geojson'
        >>> region_fpath.write_text(region.dumps())
        >>> kw = {
        >>>     'src': coco_dset,
        >>>     'dst': dst,
        >>>     'regions': region_fpath,
        >>>     'workers': 2,
        >>>     'aux_workers': 2,
        >>>     'convexify_regions': True,
        >>>     'asset_timeout': '0.00001 seconds',
        >>>     'hack_lazy': 0,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)
        >>> assert len(new_dset.images()) == 0

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> # Confirm expected behavior of `force_min_gsd` keyword argument
        >>> from geowatch.cli.coco_align import *  # NOQA
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product
        >>> from geowatch.gis.geotiff import geotiff_metadata, geotiff_crs_info
        >>> # Create a dead simple coco dataset with one image
        >>> import kwcoco
        >>> import kwimage
        >>> import dateutil.parser
        >>> coco_dset = kwcoco.CocoDataset()
        >>> ls_prod = grab_landsat_product()
        >>> fpath = ls_prod['bands'][0]
        >>> meta = geotiff_metadata(fpath)
        >>> # We need a date captured ATM in a specific format
        >>> dt = dateutil.parser.parse(
        >>>     meta['filename_meta']['acquisition_date'])
        >>> date_captured = dt.strftime('%Y/%m/%d')
        >>> gid = coco_dset.add_image(file_name=fpath, date_captured=date_captured)
        >>> dummy_poly = kwimage.Polygon.from_geojson(meta['geos_corners'])
        >>> dummy_poly = dummy_poly.scale(0.3, about='center')
        >>> sseg_geos = dummy_poly.to_geojson()
        >>> # NOTE: script is not always robust to missing annotation
        >>> # information like segmentation and bad bbox, but for this
        >>> # test config it is
        >>> coco_dset.add_annotation(
        >>>     image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)
        >>> #
        >>> # Create arguments to the script
        >>> dpath = ub.Path.appdir('geowatch/tests/coco_align').ensuredir()
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
        >>>     'force_min_gsd': 60.0,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)
        >>> coco_img = new_dset.coco_image(2)
        >>> # Check our output is in the CRS we think it is
        >>> asset = coco_img.primary_asset()
        >>> parent_fpath = asset['parent_file_names']
        >>> crop_fpath = ub.Path(new_dset.bundle_dpath) / asset['file_name']
        >>> info = geotiff_crs_info(crop_fpath)
        >>> assert(all(info['meter_per_pxl'] == 60.0))

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> from geowatch.cli.coco_align import *  # NOQA
        >>> from geowatch.demo.smart_kwcoco_demodata import demo_kwcoco_with_heatmaps
        >>> from geowatch.utils import util_gdal
        >>> import kwimage
        >>> import geojson
        >>> import json
        >>> coco_dset = demo_kwcoco_with_heatmaps(num_videos=2, num_frames=2)
        >>> dpath = ub.Path.appdir('geowatch/tests/coco_align2').ensuredir()
        >>> dst = (dpath / 'align_bundle2').delete().ensuredir()
        >>> # Create a dummy region file to crop to.
        >>> first_img = coco_dset.images().take([0]).coco_images[0]
        >>> first_fpath = first_img.primary_image_filepath()
        >>> ds = util_gdal.GdalDataset.open(first_fpath)
        >>> geo_poly = kwimage.Polygon.coerce(ds.info()['wgs84Extent'])
        >>> region_shape = kwimage.Polygon.random(n=8, convex=False, rng=3)
        >>> geo_transform = kwimage.Affine.fit(region_shape.bounding_box().corners(), geo_poly.bounding_box().corners())
        >>> region_poly = region_shape.warp(geo_transform)
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
        >>>     'debug_valid_regions': True,
        >>>     'target_gsd': 0.7,
        >>> }
        >>> cmdline = False
        >>> new_dset = main(cmdline, **kw)
        >>> coco_img = new_dset.coco_image(2)
        >>> # Check our output is in the CRS we think it is
        >>> asset = coco_img.primary_asset()
        >>> parent_fpath = asset['parent_file_names']
        >>> crop_fpath = str(ub.Path(new_dset.bundle_dpath) / asset['file_name'])
        >>> print(ub.cmd(['gdalinfo', parent_fpath])['out'])
        >>> print(ub.cmd(['gdalinfo', crop_fpath])['out'])


        >>> # Test that the input dataset visualizes ok
        >>> from geowatch.cli import coco_visualize_videos
        >>> viz_dpath = (dpath / 'viz_input_align_bundle2').ensuredir()
        >>> coco_visualize_videos.main(cmdline=False, **{
        >>>     'src': new_dset,
        >>>     'viz_dpath': viz_dpath,
        >>> })
    """
    config = CocoAlignGeotiffConfig.cli(data=kw, cmdline=cmdline, strict=True)
    import rich
    rich.print(ub.urepr(config))

    from kwcoco.util.util_json import ensure_json_serializable
    from geowatch.utils import util_gis
    from kwutil import util_parallel
    from geowatch.utils import util_resolution
    from geowatch.utils import kwcoco_extensions
    import kwcoco
    import pandas as pd
    import geopandas as gpd
    import warnings
    import kwimage

    # Store that this dataset is a result of a process.
    # Note what the process is, what its arguments are, and where the process
    # was executed.
    config_dict = config.to_dict()
    if not isinstance(config_dict['src'], str):
        # If the dataset was given in memory we don't know the path and we cant
        # always serialize it, so we punt and mark it as such
        config_dict['src'] = ':memory:'

    config_dict = ensure_json_serializable(config_dict)

    if os.environ.get('GDAL_DISABLE_READDIR_ON_OPEN') != 'EMPTY_DIR':
        warnings.warn('environ GDAL_DISABLE_READDIR_ON_OPEN should probably be set to EMPTY_DIR')
        os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

    from geowatch.utils import process_context
    proc_context = process_context.ProcessContext(
        name='coco_align',
        type='process',
        config=config_dict,
    )
    proc_context.start()
    process_info = proc_context.obj
    # print('process_info = {}'.format(ub.urepr(process_info, nl=3, sort=0)))

    config.img_workers = util_parallel.coerce_num_workers(config['img_workers'])
    config.aux_workers = util_parallel.coerce_num_workers(config['aux_workers'])
    print('img_workers = {!r}'.format(config.img_workers))
    print('aux_workers = {!r}'.format(config.aux_workers))

    output_bundle_dpath = None
    if config.dst_bundle_dpath is not None:
        output_bundle_dpath = str(config.dst_bundle_dpath)

    dst = ub.Path(config['dst']).expand()
    # TODO: handle this coercion of directories or bundles in kwcoco itself
    if 'json' in dst.name.split('.') or 'kwcoco' in dst.name.split('.'):
        if output_bundle_dpath is None:
            output_bundle_dpath = str(dst.parent)
        dst_fpath = str(dst)
    else:
        if output_bundle_dpath is not None:
            raise AssertionError('cannot give dst as a path and dst_bundle_dpath')
        output_bundle_dpath = str(dst)
        dst_fpath = str(dst / 'data.kwcoco.json')

    print('output_bundle_dpath = {!r}'.format(output_bundle_dpath))
    print('dst_fpath = {!r}'.format(dst_fpath))

    ub.Path(dst_fpath).parent.ensuredir()

    region_df = None
    regions = config['regions']
    if regions in {'annots', 'images'}:
        ...
    else:
        # Note: could update code to use geoannots.RegionModel or
        # geoannots.SiteModel to make this code more concise.
        infos = list(util_gis.coerce_geojson_datas(regions))
        parts = []
        for info in infos:
            df = info['data']
            type_to_subdf = dict(list(df.groupby('type')))
            if config['site_summary']:
                if 'site' in type_to_subdf:
                    # This is a site model
                    df = type_to_subdf['site']
                elif 'site_summary' in type_to_subdf:
                    # This is a region model
                    df = type_to_subdf['site_summary']
                    df['region_id'] = type_to_subdf['region']['region_id'].iloc[0]

                if 'status' not in df.columns:
                    # no site summaries available, filter to an empty data frame
                    df = df.iloc[:0]
                else:
                    # Don't extract system rejected regions
                    df = df[df['status'] != 'system_rejected']
            else:
                if 'site' in type_to_subdf:
                    # This is a site model
                    df = type_to_subdf['site']
                    df = df[df['status'] != 'system_rejected']
                elif 'region' in type_to_subdf:
                    # This is a region model
                    df = type_to_subdf['region']

            if len(df):
                parts.append(df)

        if len(parts):
            region_df = pd.concat(parts)
        else:
            if config.empty_region_policy == 'raise':
                raise ValueError('No regions to crop to were found')
            elif config.empty_region_policy == 'ignore':
                ...
            else:
                raise KeyError(config.empty_region_policy)
            region_df = gpd.GeoDataFrame(columns=['geometry'])

        print(f'Loaded {len(region_df)} regions to crop')

    # Load the dataset and extract geotiff metadata from each image.
    coco_dset = kwcoco.CocoDataset.coerce(config.src)
    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
    )

    if proc_context is not None:
        proc_context.add_disk_info(coco_dset.fpath)

    geo_preprop = config['geo_preprop']
    if geo_preprop == 'auto':
        if len(valid_gids):
            coco_img = coco_dset.coco_image(ub.peek(valid_gids))
            geo_preprop = not any('geos_corners' in obj for obj in coco_img.iter_asset_objs())
            print('auto-choose geo_preprop = {!r}'.format(geo_preprop))

    if geo_preprop:
        geopop_workers = config.img_workers * config.aux_workers
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
    context_factor = config['context_factor']
    if context_factor != 1:
        # Exapnd the ROI by the context factor
        region_df['geometry'] = region_df['geometry'].scale(
            xfact=context_factor, yfact=context_factor, origin='center')

    minimum_size = config['minimum_size']
    if minimum_size:
        minimum_size = util_resolution.ResolvedWindow.coerce(minimum_size)
        assert minimum_size.resolution['unit'] == 'GSD', 'must be GSD for now'
        resolved_size_utm = minimum_size.at_resolution({'mag': 1, 'unit': 'GSD'})
        min_utm_w, min_utm_h = resolved_size_utm.window
        orig_crs = region_df.crs
        # Is there a better way to estimate a CRS for each row?
        buffered_geoms = []
        for i in range(len(region_df)):
            sub = region_df[i: i + 1]
            sub_utm = util_gis.project_gdf_to_local_utm(sub)
            # Get w / h of the bounding box around the region
            utm_box = kwimage.Box.coerce(sub_utm.bounds.iloc[0].values, format='ltrb')
            # Determine a scale factor to grow the region to a min size
            sfx = max(1, min_utm_w / utm_box.width)
            sfy = max(1, min_utm_h / utm_box.height)
            # Get the bounding box around the region
            buf_utm = sub_utm['geometry'].scale(xfact=sfx, yfact=sfy, origin='centroid')
            buf = buf_utm.to_crs(orig_crs)
            buffered_geoms.append(buf)
        if len(buffered_geoms):
            region_df['geometry'] = pd.concat(buffered_geoms)

    # For each ROI extract the aligned regions to the target path
    extract_dpath = ub.ensuredir(output_bundle_dpath)

    # Create a new dataset that we will extend as we extract ROIs
    new_dset = kwcoco.CocoDataset()

    new_dset.dataset['info'] = [
        process_info,
    ]
    to_extract = cube.query_image_overlaps(region_df)

    if config['hack_lazy']:
        lazy_commands = []

    extract_kwargs = ub.udict(config) & ExtractConfig.__default__.keys()
    extract_config = ExtractConfig(**extract_kwargs)

    for image_overlaps in ub.ProgIter(to_extract, desc='extract ROI videos', verbose=3):
        video_name = image_overlaps['video_name']
        print('video_name = {!r}'.format(video_name))

        sub_bundle_dpath = ub.Path(extract_dpath) / video_name
        print('sub_bundle_dpath = {!r}'.format(sub_bundle_dpath))

        new_dset = cube.extract_overlaps(
            image_overlaps, extract_dpath, new_dset=new_dset,
            extract_config=extract_config,
        )
        if config['hack_lazy']:
            lazy_commands.extend(new_dset)

    if config['hack_lazy']:
        # Execute the gdal jobs in a single super queue
        import cmd_queue
        # queue = cmd_queue.Queue.create('serial')
        suffix = ub.hash_data(sorted([d['video_name'] for d in to_extract]))[0:8]
        queue = cmd_queue.Queue.create(
            'tmux',
            size=config.img_workers,
            # size=1,
            # name='hack_lazy_' + video_name,

            # fixme: can we make a more meaningful name here?
            name='hack_lazy_' + suffix,
            environ={
                k: v for k, v in os.environ.items()
                if k.startswith('GDAL_') or
                k == 'AWS_DEFAULT_PROFILE' or
                k == 'SMART_STAC_API_KEY'
            }
        )
        for commands in lazy_commands:
            prev = None
            for command in commands:
                prev = queue.submit(command, depends=prev)
                prev.logs = False

        queue.write()
        queue.print_commands()

        print(f'{len(queue.jobs)=}')
        if config['hack_lazy'] != 'dry':
            queue.run(
                with_textual=False
                # with_textual='auto'
            )
        raise Exception('hack_lazy always fails')

    kwcoco_extensions.reorder_video_frames(new_dset)

    proc_context.stop()

    new_dset._update_fpath(dst_fpath)
    new_dset.fpath = dst_fpath
    print('Dumping new_dset.fpath = {!r}'.format(new_dset.fpath))
    # try:
    #     rerooted_dataset = new_dset.copy()
    #     rerooted_dataset = rerooted_dataset.reroot(new_root=output_bundle_dpath, absolute=False)
    # except Exception:
    #     # Hack to fix broken pipeline, todo: find robust fix
    #     hack_region_id = infos[0]['fpath'].stem
    #     rerooted_dataset = new_dset.copy()
    #     rerooted_dataset.reroot(new_prefix=hack_region_id)
    #     rerooted_dataset.reroot(new_root=output_bundle_dpath, absolute=False)
    # rerooted_dataset.dump(rerooted_dataset.fpath, newlines=True)
    # print('finished')
    # return rerooted_dataset

    new_dset.dump(new_dset.fpath, newlines=True)
    print('finished')
    return new_dset


class SimpleDataCube:
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
        from geowatch.utils import util_gis
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
        from geowatch.demo.landsat_demodata import grab_landsat_product
        from geowatch.gis.geotiff import geotiff_metadata
        # Create a dead simple coco dataset with one image
        import geopandas as gpd
        import kwcoco
        import kwimage
        import dateutil.parser
        from geowatch.utils import util_gis
        from geowatch.utils import kwcoco_extensions
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
            img_poly = kwimage.Polygon.from_geojson(meta['geos_corners'])
            ann_poly = img_poly.scale(0.1, about='center')
            sseg_geos = ann_poly.to_geojson()
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
                    'geometry': img_poly.scale(0.2, about='center').to_geojson(),
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
            >>> from geowatch.cli.coco_align import *  # NOQA
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> to_extract = cube.query_image_overlaps(region_df)
        """
        from kwcoco.util.util_json import ensure_json_serializable
        import geopandas as gpd
        from kwutil import util_time
        from geowatch.utils import util_gis
        import kwimage

        # Quickly find overlaps using a spatial index
        ridx_to_gidsx = util_gis.geopandas_pairwise_overlaps(region_df, cube.img_geos_df)

        print('candidate query overlaps')
        ridx_to_num_matches = ub.map_vals(len, ridx_to_gidsx)
        print('ridx_to_num_matches = {}'.format(ub.urepr(ridx_to_num_matches, nl=1)))
        # print('ridx_to_gidsx = {}'.format(ub.urepr(ridx_to_gidsx, nl=1)))

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
                import cv2
                from collections import namedtuple
                import numpy as np
                from shapely import validation
                # Try and detect thin regions and then add context
                region_row_df_utm = _region_row_df.to_crs(local_epsg)
                region_utm_geom = region_row_df_utm['geometry'].iloc[0]
                # poly = kwimage.Polygon.coerce(region_utm_geom)
                OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'theta'))
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
                sub_bundle_dname = video_name
            elif region_row.get('type', None) == 'site_summary':
                # Special case where we are extracting a site model with a name
                video_name = region_row.get('site_id', space_str)  # V2 spec
                region_id = region_row.get('region_id', 'unknown_region')  # V2 spec
                sub_bundle_dname = f'{region_id}/{video_name}'
            else:
                video_name = space_str
                sub_bundle_dname = video_name

            if len(gidxs) == 0:
                print('Warning: No spatial matches to {}'.format(video_name))
            else:
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
                    if query_start_datetime is not None:
                        flags = [dt >= query_start_datetime for dt in cand_datetimes]
                        cand_datetimes = list(ub.compress(cand_datetimes, flags))
                        cand_gids = list(ub.compress(cand_gids, flags))

                if query_end_date is not None:
                    query_end_datetime = util_time.coerce_datetime(query_end_date)
                    if query_end_datetime is not None:
                        flags = [dt <= query_end_datetime for dt in cand_datetimes]
                        cand_datetimes = list(ub.compress(cand_datetimes, flags))
                        cand_gids = list(ub.compress(cand_gids, flags))

                if len(cand_gids) == 0:
                    print('Warning: No temporal matches to {}'.format(video_name))
                else:
                    datetime_to_gids = ub.group_items(cand_gids, cand_datetimes)
                    # print('datetime_to_gids = {}'.format(ub.urepr(datetime_to_gids, nl=1)))
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
                        'sub_bundle_dname': sub_bundle_dname,
                    }
                    to_extract.append(image_overlaps)
        return to_extract

    @profile
    def extract_overlaps(cube, image_overlaps, extract_dpath, new_dset=None,
                         extract_config=None):
        """
        Given a region of interest, extract an aligned temporal sequence
        of data to a specified directory.

        Args:
            image_overlaps (dict): Information about images in an ROI and their
                temporal order computed from :func:``query_image_overlaps``.

            extract_dpath (str):
                where to dump the data extracted from this ROI.

            new_dset (kwcoco.CocoDataset | None):
                if specified, add extracted images and annotations to this
                dataset, otherwise create a new dataset.

            extract_config (ExtractConfig):
                configuration for how to perform the extract task.

        Returns:
            kwcoco.CocoDataset: the given or new dataset that was modified

        Example:
            >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
            >>> from geowatch.cli.coco_align import *  # NOQA
            >>> import kwcoco
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> extract_dpath = ub.Path.appdir('geowatch/tests/coco_align/demo_extract_overlaps').ensuredir()
            >>> rpc_align_method = 'orthorectify'
            >>> new_dset = kwcoco.CocoDataset()
            >>> to_extract = cube.query_image_overlaps(region_df)
            >>> image_overlaps = to_extract[0]
            >>> extract_config = ExtractConfig(img_workers=32)
            >>> cube.extract_overlaps(image_overlaps, extract_dpath,
            >>>                       new_dset=new_dset, extract_config=extract_config)

        Example:
            >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
            >>> from geowatch.cli.coco_align import *  # NOQA
            >>> import kwcoco
            >>> cube, region_df = SimpleDataCube.demo(with_region=True, extra=True)
            >>> extract_dpath = ub.Path.appdir('geowatch/tests/coco_align/demo_extract_overlaps2').ensuredir()
            >>> rpc_align_method = 'orthorectify'
            >>> to_extract = cube.query_image_overlaps(region_df)
            >>> new_dset = kwcoco.CocoDataset()
            >>> image_overlaps = to_extract[1]
            >>> extract_config = ExtractConfig(img_workers=0)
            >>> cube.extract_overlaps(image_overlaps, extract_dpath,
            >>>                       new_dset=new_dset,
            >>>                       extract_config=extract_config)
        """
        import geopandas as gpd
        import kwcoco
        import kwimage
        import math
        import numpy as np
        import pandas as pd
        import subprocess
        from concurrent.futures import TimeoutError
        from geowatch.utils import kwcoco_extensions
        from geowatch.utils import util_gis
        from kwcoco.util.util_json import ensure_json_serializable
        from kwutil import util_time
        from kwutil.util_yaml import Yaml
        coco_dset = cube.coco_dset
        assert extract_config is not None

        datetime_to_gids = image_overlaps['datetime_to_gids']
        space_str = image_overlaps['space_str']
        space_box = image_overlaps['space_box']
        space_region = image_overlaps['space_region']
        video_name = image_overlaps['video_name']
        video_props = image_overlaps['properties']
        local_epsg = image_overlaps['local_epsg']
        print('space_str = {}'.format(ub.urepr(space_str, nl=1)))
        print('space_box = {}'.format(ub.urepr(space_box, nl=1)))
        print('space_region = {}'.format(ub.urepr(space_region, nl=1)))
        print('video_name = {}'.format(ub.urepr(video_name, nl=1)))
        print('video_props = {}'.format(ub.urepr(video_props, nl=1)))
        print('local_epsg = {}'.format(ub.urepr(local_epsg, nl=1)))

        if 1:
            # Remove specific null properties from video_props
            keys = [
                'predicted_phase_transition_date',
                'predicted_phase_transition',
                'score',
            ]
            for k in keys:
                if k in video_props:
                    v = video_props[k]
                    if pd.isnull(v):
                        video_props.pop(k)

        if new_dset is None:
            new_dset = kwcoco.CocoDataset()

        sub_bundle_dpath = (ub.Path(extract_dpath) / image_overlaps['sub_bundle_dname']).ensuredir()

        subdata_fpath = ub.Path(sub_bundle_dpath) / 'subdata.kwcoco.zip'
        if subdata_fpath.exists() and extract_config.keep in {'roi-img', 'roi'}:
            print('ROI cache hit')
            sub_dset = kwcoco.CocoDataset(subdata_fpath)
            return new_dset.union(sub_dset)

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        datetimes = sorted(datetime_to_gids)
        datetime_to_gids = ub.udict(datetime_to_gids).subdict(datetimes)

        # If specified, only choose a subset of images over time.
        sensor_to_time_window = Yaml.coerce(extract_config.sensor_to_time_window)
        if isinstance(sensor_to_time_window, str):
            # not sure why null can show up here.
            if sensor_to_time_window.strip() not in {'', 'null'}:
                sensor_to_time_window = None

        TIME_WINDOW_FILTER = 1
        nan = float('nan')

        if TIME_WINDOW_FILTER and sensor_to_time_window is not None:
            # TODO: this filter should be part of the earlier query
            if not isinstance(sensor_to_time_window, dict):
                raise TypeError(f'sensor_to_time_window: {type(sensor_to_time_window)} = {sensor_to_time_window}')

            sensor_to_time_window = ub.udict(sensor_to_time_window)
            sensor_to_time_window = sensor_to_time_window.map_values(util_time.coerce_timedelta)
            if sensor_to_time_window is not None:
                rows = []
                for dt, gids in datetime_to_gids.items():
                    for gid in gids:
                        img = coco_dset.imgs[gid]

                        # Assign a contamination score to each image.
                        cloudcover = nan
                        contamination = nan

                        # TODO: Better estimation of whole image contamination.
                        if 'stac_properties' in img:
                            prop = img['stac_properties']
                            cloudcover = prop.get('eo:cloud_cover', nan) / 100
                            contamination = prop.get('quality_info:contaminated_percentage', nan) / 100

                        if 'parent_stac_properties' in img:
                            for prop in img['parent_stac_properties']:
                                # FIXME, non general hard-coded properties used here
                                _cloudcover = prop.get('eo:cloud_cover', nan) / 100
                                _contamination = prop.get('quality_info:contaminated_percentage', nan) / 100
                                cloudcover = np.nanmin([cloudcover, _cloudcover])
                                contamination = np.nanmin([contamination, _contamination])

                        if math.isnan(contamination):
                            contamination = cloudcover

                        if math.isnan(contamination):
                            # For cases where we don't have info, prioritize
                            # these before very contaminated images, but after
                            # known good ones.
                            contamination = 0.6

                        row = {
                            'gid': gid,
                            'sensor': img['sensor_coarse'],
                            'contamination': contamination,
                            'unixtime': dt.timestamp(),
                        }
                        rows.append(row)

                df = pd.DataFrame(rows)
                sensor_to_df = dict(list(df.groupby('sensor')))

                restrict_sensors = list(sensor_to_time_window.keys())
                chosen_gids = set()
                for group in (ub.udict(sensor_to_df) - restrict_sensors).values():
                    chosen_gids.update(group['gid'])

                for sensor in restrict_sensors:
                    if sensor in sensor_to_df:
                        subdf = sensor_to_df[sensor]
                        window_seconds = sensor_to_time_window[sensor].total_seconds()
                        subdf['bucket'] = (subdf['unixtime'] // window_seconds).astype(int)
                        for _, group in subdf.groupby('bucket'):
                            group = group.sort_values(['contamination', 'unixtime'])
                            chosen_gid = group.iloc[0]['gid']
                            chosen_gids.add(chosen_gid)

                # Hack the datetime_to_gids to finalize this filter.
                # should probably make this impl cleaner.
                num_filtered = 0
                num_total = 0
                new_datetime_to_gids = {}
                for dt, gids in datetime_to_gids.items():
                    new_gids = ub.oset(gids) & chosen_gids
                    num_total += len(gids)
                    num_filtered += len(gids) - len(new_gids)
                    if new_gids:
                        new_datetime_to_gids[dt] = new_gids
                num_keep = num_total - num_filtered
                print(f'TimeWindow: keeping: {num_keep} / {num_total}')
                datetime_to_gids = new_datetime_to_gids
                datetimes = sorted(datetime_to_gids)
                datetime_to_gids = ub.udict(datetime_to_gids).subdict(datetimes)

        valid_region_geos = space_region.to_geojson()

        # Handle an issue with pandas parsing. Should not need to do this if we
        # can force pandas to be less smart.
        if True:
            from kwcoco.util.util_json import find_json_unserializable
            issues = list(find_json_unserializable(video_props))
            timestamp_keys = ['start_date', 'end_date', 'predicted_phase_transition_date']
            for issue in issues:
                found = False
                for k in timestamp_keys:
                    if issue['loc'] == [k]:
                        ts = video_props[k]
                        video_props[k] = util_time.datetime.coerce(ts).isoformat()
                        found = True
                if not found:
                    raise Exception(f'Unhandled issues {issues}')

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

        # Manage new ids such that parallelization does not impact their order
        start_gid = new_dset._next_ids.get('images')

        # parallelize over images
        image_jobs = ub.JobPool(mode='thread', max_workers=extract_config.img_workers)

        sh_space_region_crs84 = space_region.to_shapely()
        space_region_crs84 = gpd.GeoDataFrame(
            {'geometry': [sh_space_region_crs84]}, crs=util_gis._get_crs84())

        space_region_local = space_region_crs84.to_crs(local_epsg)
        sh_space_region_local = space_region_local.geometry.iloc[0]

        # No restrictions, we just want all the frames
        want_all_frames = (
            extract_config.num_start_frames is None and
            extract_config.num_end_frames is None
            # extract_config.max_frames
        )

        # Determine what frames we actually want inside the valid range
        num_avail_frames = len(datetimes)
        all_frame_indexes = ub.oset(range(num_avail_frames))
        if want_all_frames:
            requested_frame_indexes = all_frame_indexes
            unrequested_datetimes = ub.oset()
        else:
            if extract_config.num_start_frames is None:
                extract_config.num_start_frames = 0
            if extract_config.num_end_frames is None:
                extract_config.num_end_frames = 0
            end_start_index = max(0, num_avail_frames - extract_config.num_end_frames)
            front_end_index = min(num_avail_frames, extract_config.num_start_frames)
            take_end_idxs = ub.oset(range(end_start_index, num_avail_frames))
            take_start_idxs = ub.oset(range(0, front_end_index))

            all_frame_indexes = ub.oset(range(num_avail_frames))
            requested_frame_indexes = take_start_idxs | take_end_idxs
            unrequested_datetimes = all_frame_indexes - requested_frame_indexes

        frame_count = 0
        frame_index = 0   # Note: this may need to be corrected later
        prog = ub.ProgIter(requested_frame_indexes, desc='submit extract jobs', verbose=1)
        piter = iter(prog)
        for request_idx in piter:

            datetime_ = datetimes[request_idx]

            if extract_config.max_frames is not None:
                if frame_count >= extract_config.max_frames:
                    break
                frame_count += 1

            iso_time = util_time.isoformat(datetime_, sep='T', timespec='seconds')
            gids = datetime_to_gids[datetime_]
            # TODO: Is there any other consideration we should make when
            # multiple images have the same timestamp?
            # if len(gids) > 0:
            if len(gids) > 0:
                # We got multiple images for the same timestamp.  Im not sure
                # if this is necessary but thig logic attempts to sort them
                # such that the "best" image to use is first.  Ideally gdalwarp
                # would take care of this but I'm not sure it does.
                groups = _handle_multiple_images_per_date(
                    coco_dset, gids, local_epsg, sh_space_region_local,
                    extract_config, prog, cube, space_region_crs84,
                    extract_dpath, video_name, iso_time, space_str,
                    space_region_local)
            else:
                groups = [{
                    'main_gid': gids[0],
                    'other_gids': [],
                }]

            for num, group in enumerate(groups):
                main_gid = group['main_gid']
                other_gids = group['other_gids']
                img = coco_dset.imgs[main_gid]
                other_imgs = [coco_dset.imgs[x] for x in other_gids]

                sensor_coarse = img.get('sensor_coarse', 'unknown')
                # Construct a name for the subregion to extract.
                name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)

                img_verbose = (
                    ((extract_config.verbose > 1) or
                     (extract_config.verbose > 0 and
                      (extract_config.img_workers == 0))) and
                    extract_config.verbose)

                img_config = ImageExtractConfig(
                    **(ub.udict(extract_config) & set(ImageExtractConfig.__default__.keys()))
                )
                img_config['verbose'] = img_verbose

                job = image_jobs.submit(
                    extract_image_job,
                    img=img,
                    other_imgs=other_imgs,
                    bundle_dpath=bundle_dpath,
                    new_bundle_dpath=new_bundle_dpath,
                    name=name,
                    datetime_=datetime_,
                    num=num,
                    frame_index=frame_index,
                    new_vidid=new_vidid,
                    new_vidname=video_name,
                    sub_bundle_dpath=sub_bundle_dpath,
                    space_str=space_str,
                    space_region=space_region,
                    space_box=space_box,
                    start_gid=start_gid,
                    local_epsg=local_epsg,
                    img_config=img_config)
                job.request_idx = request_idx
                start_gid = start_gid + 1
                frame_index = frame_index + 1

        sub_new_gids = []
        if extract_config.image_timeout is not None:
            image_timeout = util_time.coerce_timedelta(extract_config.image_timeout).total_seconds()

        if extract_config.hack_lazy:
            lazy_commands = []

        img_iter = image_jobs.as_completed(timeout=image_timeout)
        img_prog = ub.ProgIter(
            img_iter,
            desc=ub.color_text('collect extract jobs', 'green'),
            total=len(image_jobs),
            clearline=False)

        for job in img_prog:
            try:
                new_img = job.result(timeout=image_timeout)
            except SkipImage:
                # TODO: if we are only requesting a subset of images we may
                # want to submit a new image job to take the place of this
                # failed image.
                unrequested_datetimes
                continue
            except subprocess.TimeoutExpired:
                print('\n\nAn image job subprocess timed out!\n\n')
                continue
            except TimeoutError:
                # FIXME: If we ever hit this timeout it it is likely that the
                # job itself is still running and thus the thread pool executor
                # will never allow the python interpreter to exit!
                print('\n\nAn image job timed out!\n\n')
                continue

            if extract_config.hack_lazy:
                # Hacking to just grab the image commands, so we have to stop
                # the job here.
                for dst in new_img:
                    if dst is not None:
                        commands = dst.get('commands', [])
                        if commands:
                            lazy_commands.append(commands)
                continue

            # Hack, the next ids dont update when new images are added
            # with explicit ids. This is a quick fix.
            new_img.pop('id', None)

            new_img['video_id'] = new_vidid

            new_gid = new_dset.add_image(**new_img)
            sub_new_gids.append(new_gid)

        if extract_config.hack_lazy:
            # Skip the rest in this hack lazy mode
            return lazy_commands

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
                    raise AssertionError('unserializable(gid={}) = {}'.format(new_gid, ub.urepr(unserializable, nl=0)))

        kwcoco_extensions.coco_populate_geo_video_stats(
            new_dset, target_gsd=extract_config.target_gsd, video_id=new_vidid)

        if extract_config.write_subsets:
            print('Writing data subset')
            if 0:
                # Enable if json serialization is breaking
                new_dset._check_json_serializable()

            sub_dset = new_dset.subset(sub_new_gids, copy=True)
            sub_dset.fpath = os.fspath(ub.Path(sub_bundle_dpath) / 'subdata.kwcoco.zip')
            sub_dset.reroot(new_root=os.fspath(sub_bundle_dpath), absolute=False)

            # Fix frame order issue
            kwcoco_extensions.reorder_video_frames(sub_dset)

            sub_dset.dump(sub_dset.fpath, newlines=True)
        return new_dset


def _handle_multiple_images_per_date(coco_dset, gids, local_epsg,
                                     sh_space_region_local, extract_config,
                                     prog, cube, space_region_crs84,
                                     extract_dpath, video_name, iso_time,
                                     space_str, space_region_local):
    """
    We got multiple images for the same timestamp.  Im not sure if this is
    necessary but thig logic attempts to sort them such that the "best" image
    to use is first.  Ideally gdalwarp would take care of this but I'm not sure
    it does.
    """
    import geopandas as gpd
    from shapely import geometry
    from geowatch.utils import util_gis
    conflict_imges = coco_dset.images(gids)
    sensors = list(conflict_imges.lookup('sensor_coarse', None))

    groups = []

    for sensor_coarse, sensor_gids in ub.group_items(conflict_imges, sensors).items():
        rows = []
        for gid in sensor_gids:
            coco_img = coco_dset.coco_image(gid)

            # Should more than just the primary asset be used here?
            primary_asset = coco_img.primary_asset()
            fpath = ub.Path(coco_dset.bundle_dpath) / primary_asset['file_name']

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
                geos_corners = coco_img.img.get('geos_corners', None)
                if geos_corners is not None:
                    corners_gdf = util_gis.crs_geojson_to_gdf(geos_corners)
                    valid_region_local = corners_gdf.to_crs(local_epsg)
                    sh_valid_region_local = valid_region_local.geometry.iloc[0]
                    isect_area = sh_valid_region_local.intersection(sh_space_region_local).area
                    other_area = sh_space_region_local.area
                    valid_iooa = isect_area / other_area
                else:
                    sh_valid_region_local = None
                    valid_iooa = -1

            tiebreaker = '*' if sensor_coarse is None else sensor_coarse
            score = (valid_iooa, tiebreaker)
            rows.append({
                'score': score,
                'gid': gid,
                'valid_iooa': valid_iooa,
                'fname': fpath.name,
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
    return groups


@profile
def extract_image_job(img,
                      other_imgs,
                      bundle_dpath,
                      new_bundle_dpath,
                      name,
                      datetime_,
                      num,
                      frame_index,
                      new_vidid,
                      new_vidname,
                      sub_bundle_dpath,
                      space_str,
                      space_region,
                      space_box,
                      start_gid,
                      local_epsg=None,
                      img_config=None):
    """
    Threaded worker function for :func:`SimpleDataCube.extract_overlaps`.

    Returns:
        Dict : new_img
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
    import json
    from geowatch.utils import kwcoco_extensions
    from kwutil import util_time
    from osgeo import osr
    osr.GetPROJSearchPaths()
    from kwcoco.coco_image import CocoImage

    assert img_config is not None

    if img_config.image_timeout is not None:
        img_config.image_timeout = util_time.coerce_timedelta(img_config.image_timeout).total_seconds()
    if img_config.asset_timeout is not None:
        img_config.asset_timeout = util_time.coerce_timedelta(img_config.asset_timeout).total_seconds()

    coco_img = CocoImage(img)
    has_base_image = img.get('file_name', None) is not None
    objs = [ub.dict_diff(obj, {'auxiliary', 'assets'})
            for obj in coco_img.iter_asset_objs()]
    sensor_coarse = img.get('sensor_coarse', 'unknown')

    channels_to_objs = ub.ddict(list)
    for obj in objs:
        key = obj['channels']
        if key in channels_to_objs:
            coco_channels = [o.get('channels', None) for o in objs]
            warnings.warn(ub.paragraph(
                f'''
                It seems multiple auxiliary items in the parent image might
                contain the same channel.  This script will try to work around
                this, but that is not a valid kwcoco assumption.
                coco_channels={coco_channels}.
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

    # True if there are multiple assets written for this image.
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
                # Note: this takes 34% of the time in the case where images are
                # pre-cached and dont need download
                kwcoco_extensions._populate_canvas_obj(bundle_dpath, obj,
                                                       keep_geotiff_metadata=True)
            if obj['geotiff_metadata']['is_rpc']:
                is_rpc = True

    if is_rpc and img_config.rpc_align_method != 'affine_warp':
        align_method = img_config.rpc_align_method
    else:
        align_method = 'affine_warp'

    dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))
    dst_dpath_ = ub.Path(dst_dpath)
    error_fpath = dst_dpath_ / (f'{name}.error')

    job_list = []

    # Turn off internal threading because we refactored this to thread over all
    # images instead
    asset_jobs = ub.JobPool(mode='thread', max_workers=img_config.aux_workers)

    verbose = img_config.verbose
    aux_verbose = (verbose > 3) or (verbose > 1 and (img_config.aux_workers == 0))
    asset_config = AssetExtractConfig(
        **(ub.udict(img_config) & set(AssetExtractConfig.__default__.keys()))
    )
    asset_config['verbose'] = aux_verbose

    for obj_group in ub.ProgIter(obj_groups, desc=f'submit warp assets in {new_vidname}', verbose=verbose):
        job = asset_jobs.submit(
            _aligncrop, obj_group, bundle_dpath, name, sensor_coarse,
            dst_dpath, space_region, space_box, align_method, is_multi_image,
            local_epsg=local_epsg, asset_config=asset_config,
        )
        job_list.append(job)

    dst_list = []
    for job in asset_jobs.as_completed(desc='collect warp assets {}'.format(name),
                                       timeout=img_config.image_timeout,
                                       progkw=dict(enabled=DEBUG, verbose=verbose)):
        try:
            dst = job.result(timeout=img_config.asset_timeout)
        except Exception as ex:
            # Write a file to disk to indicate an error in builting this image.
            error_summary = {
                'ex': {
                    'type': str(type(ex)),
                    'str': str(ex),
                },
                # TODO: what information is relevant here?
                'num_object_groups': len(obj_groups),
                'sensor_coarse': sensor_coarse,
                'frame_index': frame_index,
                'name': name,
                'num': num,
                'datetime_': str(datetime_),
                'new_vidid': new_vidid,
                'new_vidname': new_vidname,
                'space_str': space_str,
                'local_epsg': local_epsg,
                'is_rpc': is_rpc,
            }
            error_fpath.write_text(json.dumps(error_summary, indent='    '))
            print(f'Log Image Error: {error_fpath}')

            if img_config.image_error_policy == 'ignore':
                raise SkipImage
            else:
                raise
        dst_list.append(dst)

    if img_config.hack_lazy:
        return dst_list

    new_gid = start_gid

    if verbose > 2:
        print(f'Finish channel crop jobs: {new_gid} in {new_vidname}')

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

    new_img = {
        'id': new_gid,
        'name': name,
        'align_method': align_method,
    }

    parent_stac_properties = []
    parent_imgs = [img] + other_imgs
    for _parent in parent_imgs:
        if 'stac_properties' in _parent:
            parent_stac_properties.append(_parent['stac_properties'])
        elif 'parent_stac_properties' in _parent:
            parent_stac_properties.extend(_parent['parent_stac_properties'])

    if has_base_image and len(dst_list) == 1:
        base_dst = dst_list[0]
        new_img.update(base_dst)
        aux_dst = []
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

    # There are often going to be multiple images so we cant just use the
    # "main" image properties.

    # Carry over appropriate metadata from original image
    new_img.update(carry_over)
    new_img['frame_index'] = frame_index
    new_img['timestamp'] = datetime_.timestamp()

    if parent_stac_properties:
        new_img['parent_stac_properties'] = parent_stac_properties

    new_coco_img = CocoImage(new_img)

    if not len(list(new_coco_img.iter_asset_objs())):
        # This image did not contained any requested bands. Skip it.
        raise SkipImage

    new_coco_img._bundle_dpath = new_bundle_dpath
    new_coco_img._video = {}

    # Note this takes 61% of the time when images are already cached.
    kwcoco_extensions._populate_valid_region(new_coco_img)

    if DEBUG:
        print(f'Finished extract img job: {new_gid} in {new_vidname}')
    return new_img


@profile
def _aligncrop(obj_group,
               bundle_dpath,
               name,
               sensor_coarse,
               dst_dpath,
               space_region,
               space_box,
               align_method,
               is_multi_image,
               local_epsg=None,
               asset_config=None):
    """
    Threaded worker function for :func:`SimpleDataCube.extract_image_job`.

    This functions contains the expensive calls to GDAL, which are abstracted
    by :mod:`geowatch.utils.util_gdal`.

    Args:
        asset_config (AssetExtractConfig): main options
            Note: the hack_lazy argument makes this function returns gdal
            commands that would be executed.
    """
    import geowatch
    import json
    import kwcoco
    from geowatch.utils import util_gdal
    from os.path import join

    assert asset_config is not None
    # asset_config = AssetExtractConfig(**kwargs)
    verbose = asset_config.verbose

    # NOTE: https://github.com/dwtkns/gdal-cheat-sheet
    first_obj = obj_group[0]
    chan_code = obj_group[0].get('channels', '')

    # Prevent long names for docker (limit is 242 chars)
    channels_ = kwcoco.FusedChannelSpec.coerce(chan_code)
    chan_pname = channels_.path_sanitize(maxlen=10)

    if asset_config.include_channels is not None:
        # Filter out bands we are not interested in
        include_channels = kwcoco.FusedChannelSpec.coerce(asset_config.include_channels)
        if not channels_.intersection(include_channels).numel():
            if verbose > 2:
                print('Skip not included {}'.format(channels_))
            return None

    if asset_config.exclude_channels is not None:
        # Filter out bands we are not interested in
        exclude_channels = kwcoco.FusedChannelSpec.coerce(asset_config.exclude_channels)
        if channels_.difference(exclude_channels).numel() == 0:
            if verbose > 2:
                print('Skip excluded {}'.format(channels_))
            return None

    if is_multi_image:
        # Multiple assets for this image.
        multi_dpath = ub.ensuredir((dst_dpath, name))
        dst_gpath = ub.Path(multi_dpath) / (name + '_' + chan_pname + '.tif')
    else:
        dst_gpath = ub.Path(dst_dpath) / (name + '.tif')

    input_gnames = [obj.get('file_name', None) for obj in obj_group]
    assert all(n is not None for n in input_gnames)

    # Must use join over pathlib because Path strips http:// to http:/
    input_gpaths = [join(bundle_dpath, n) for n in input_gnames]

    dst = {
        'file_name': os.fspath(dst_gpath),
    }
    roles = list(ub.oset(ub.flatten([o.get('roles', []) for o in obj_group])))
    if len(roles):
        dst['roles'] = roles
    if first_obj.get('channels', None):
        dst['channels'] = first_obj['channels']
    if first_obj.get('num_bands', None):
        dst['num_bands'] = first_obj['num_bands']

    dst['parent_file_names'] = [o.get('file_name', None) for o in obj_group]

    if 'quality' in roles:
        dst['qa_encoding'] = asset_config.qa_encoding

    already_exists = dst_gpath.exists()
    needs_recompute = not (already_exists and asset_config.keep in {'img', 'roi-img'})

    needs_resolution_checks = False  # TODO
    if not needs_recompute:
        if asset_config.corruption_checks or needs_resolution_checks:
            # Sometimes the data will exist, but it's bad data. Check for this.
            try:
                ref = util_gdal.GdalDataset.open(dst_gpath, mode='r')
                ref
            except RuntimeError:
                # Data is likely corrupted
                needs_recompute = True
                print(f'The data exists {dst_gpath}, but is corrupted. Recomputing')
                dst_gpath.delete()
            else:
                # Do resolution checks
                if 0:
                    from geowatch.gis.geotiff import geotiff_crs_info
                    info = geotiff_crs_info(ref)
                    info['approx_meter_gsd']
                ref = None

    out_fpath = dst_gpath
    error_fpath = out_fpath.parent / (out_fpath.name + '.error')
    # if 'WV' in sensor_coarse:
    #     import xdev
    #     xdev.embed()

    if asset_config.skip_previous_errors:
        if error_fpath.exists():
            raise SkipImage('Attempting to grab this asset previously failed, skipping')

    if not needs_recompute:
        if verbose > 2:
            print('cache hit dst = {!r}'.format(dst))
        return dst

    if align_method == 'orthorectify':
        if 'geotiff_metadata' in first_obj:
            info = first_obj['geotiff_metadata']
        else:
            info = geowatch.gis.geotiff.geotiff_crs_info(input_gpaths[0])
        # No RPCS exist, use affine-warp instead
        rpcs = info['rpc_transform']
    elif align_method == 'affine_warp':
        rpcs = None
    else:
        raise KeyError(align_method)

    duplicates = ub.find_duplicates(input_gpaths)
    if duplicates:
        warnings.warn(ub.paragraph(
            '''
            Input to _aligncrop contained duplicate filepaths, the same image
            might be registered in the base kwcoco file multiple times.
            '''))
        # print('!!WARNING!! duplicates = {}'.format(ub.urepr(duplicates, nl=1)))
        input_gpaths = list(ub.oset(input_gpaths))

    nodata = asset_config.force_nodata
    if nodata is not None:
        # HACK: quality bands are UInt16 so they can't have a negative nodata
        if first_obj['channels'] in {'quality', 'cloudmask'}:
            nodata = asset_config.unsigned_nodata

    # When trying to get a gdalmerge to take multiple inputs I got a Attempt to
    # create 0x0 dataset is illegal,sizes must be larger than zero.  This new
    # method will call gdalwarp on each image individually and then merge them
    # all in a final step.
    if verbose > 2:
        print(
            'start gdal warp in_fpaths = {}'.format(ub.urepr(input_gpaths, nl=1)) +
            'chan_code = {!r}\n'.format(chan_code) +
            '\n* dst_gpath = {!r}'.format(dst_gpath))

    error_logfile = None
    # Uncomment to suppress warnings for debug purposes
    #
    # error_logfile = '/dev/null'

    # Note: these methods take care of retries and checking that the
    # data is valid.
    force_spatial_res = None
    # print('\n')
    # print(f'asset_config.force_min_gsd={asset_config.force_min_gsd}')
    # print(first_obj['approx_meter_gsd'])
    # print('obj_group = {}'.format(ub.urepr(obj_group, nl=-1)))
    if asset_config.force_min_gsd is not None:
        obj_approx_gsds = []
        for obj in obj_group:
            if 'approx_meter_gsd' in obj:
                approx_meter_gsd = obj['approx_meter_gsd']
            else:
                if 'geotiff_metadata' in obj:
                    warnings.warn(ub.paragraph(
                        '''
                        Prepopulation should have set the approx_meter_gsd property
                        when geotiff_metadata was populated. As of 0.3.9 we should
                        not be hitting this case. There may be something wrong.
                        '''))
                    info = obj['geotiff_metadata']
                else:
                    warnings.warn(ub.paragraph(
                        '''
                        Popluating geotiff approx_meter_gsd should have already been
                        done.  To ensure pre-population use the '--geo_preprop=True'
                        argument.
                        '''))
                    info = geowatch.gis.geotiff.geotiff_crs_info(input_gpaths[0])
                approx_meter_gsd = info.get('approx_meter_gsd', None)
            if approx_meter_gsd is not None:
                obj_approx_gsds.append(approx_meter_gsd)

        if len(obj_approx_gsds) == 0:
            approx_meter_gsd = None
        else:
            approx_meter_gsd = min(obj_approx_gsds)
        if approx_meter_gsd is not None and approx_meter_gsd < asset_config.force_min_gsd:
            # Only setting if needed to avoid needless warping if the
            # 'approximate_meter_gsd' value is slightly different from
            # what GDAL computes at the time of warping
            force_spatial_res = asset_config.force_min_gsd

    gdal_verbose = 0 if verbose < 2 else verbose

    if 'quality' in roles:
        overview_resampling = 'NEAREST'
    else:
        overview_resampling = 'CUBIC'

    gdalkw = dict(
        space_box=space_box,
        local_epsg=local_epsg,
        rpcs=rpcs, nodata=nodata,
        tries=asset_config.tries,
        cooldown=asset_config.cooldown,
        backoff=asset_config.backoff,
        error_logfile=error_logfile,
        verbose=gdal_verbose,
        force_spatial_res=force_spatial_res,
        eager=not asset_config.hack_lazy,
        warp_memory='1500',
        gdal_cachemax='1500',
        num_threads='2',
        timeout=asset_config.asset_timeout,
        overviews='AUTO',
        overview_resampling=overview_resampling,
    )

    try:
        if len(input_gpaths) > 1:
            in_fpaths = input_gpaths
            commands = util_gdal.gdal_multi_warp(in_fpaths, out_fpath, **gdalkw)
        else:
            in_fpath = input_gpaths[0]
            commands = util_gdal.gdal_single_warp(in_fpath, out_fpath, **gdalkw)
    except Exception as ex:
        print('!!!!!!')
        print('!!!!!!')
        print(f'!!!Error when calling GDAL: ex={ex}')
        print('!!!!!!')
        print('!!!!!!')
        print(f'input_gpaths = {ub.urepr(input_gpaths, nl=1)}')
        print(f'out_fpath = {ub.urepr(out_fpath, nl=1)}')
        print(f'gdalkw = {ub.urepr(gdalkw, nl=1)}')
        print('!!!!!!')
        print('!!!!!!')
        DUMP_ERRORS = True
        if DUMP_ERRORS:
            # If there is an error write out a file
            serialized_box = serialize_kwimage_boxes(gdalkw['space_box'])
            fixed_gdalkw = gdalkw.copy()
            fixed_gdalkw['space_box'] = serialized_box
            error_summary = {
                'ex': {
                    'type': str(type(ex)),
                    'str': str(ex),
                },
                'input_gpaths': [os.fspath(p) for p in input_gpaths],
                'gdalkw': fixed_gdalkw,
                'out_fpath': os.fspath(out_fpath),
            }
            error_summary['gdalkw'] = fixed_gdalkw
            error_text = json.dumps(error_summary, indent='  ')
            error_fpath.write_text(error_text)
            print(f'Log Asset Error: {error_fpath}')
        raise

    if asset_config.hack_lazy:
        # The lazy hack means we are just building the commands
        dst['commands'] = commands
    if verbose > 2:
        print('finish gdal warp dst_gpath = {!r}'.format(dst_gpath))
    return dst


def serialize_kwimage_boxes(boxes):
    """
    TODO: port to kwimage.Boxes.__json__
    """
    json_boxes = {
        'type': 'kwimage.Boxes',
        'properties': {
            'data': boxes.data.tolist(),
            'format': boxes.format,
        }
    }
    return json_boxes


class SkipImage(Exception):
    ...


__config__ = CocoAlignGeotiffConfig


if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.cli.coco_align --help
    """
    main()
