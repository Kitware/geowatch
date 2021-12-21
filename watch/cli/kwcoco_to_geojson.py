"""
This file contains logic to convert a kwcoco file into an IARPA Site Model.

At a glance the IARPA Site Model is a GeoJSON FeatureCollection with the
following informal schema:

list of json dictionaries, where each
"site" has the following has the following informal schema:

TODO:
    - [ ] Is our computation of the "site-boundary" correct?
    - [x] Do we have a complete list of IARPA category names?
    - [x] Do we have a complete list of IARPA sensor names?
    - [ ] Is our computation of the "predicted_phase" correct?
    - [ ] How do we compute "is_occluded"?
    - [x] Document details about is_site_boundary
    - [x] Document details about is_occluded


For official documentation about the KWCOCO json format see [1]_. A formal
json-schema can be found in ``kwcoco.coco_schema``

For official documentation about the IARPA json format see [2, 3]_. A formal
json-schema can be found in ``watch/rc/site-model.schema.json``.

References:
    .. [1] https://gitlab.kitware.com/computer-vision/kwcoco
    .. [2] https://infrastructure.smartgitlab.com/docs/pages/api/
    .. [3] https://smartgitlab.com/TE/annotations
"""
import geojson
import json
import os
import sys
import argparse
import kwcoco
import dateutil.parser
import watch
import shapely
import shapely.ops
from mgrs import MGRS
import numpy as np
import ubelt as ub
# import colored_traceback.auto  # noqa


def _single_geometry(geom):
    return shapely.geometry.shape(geom).buffer(0)


def _combined_geometries(geometry_list):
    # TODO does this respect ordering for disjoint polys?
    return shapely.ops.unary_union(geometry_list).buffer(0)


def _normalize_date(date_str):
    return dateutil.parser.parse(date_str).date().isoformat()


def geojson_feature(img, anns, coco_dset, with_properties=True):
    '''
    Group kwcoco annotations in the same track (site) and image
    into one Feature in an IARPA site model
    '''
    def single_geometry(ann):
        seg_geo = ann['segmentation_geos']
        assert isinstance(seg_geo, dict)
        return _single_geometry(seg_geo)

    def per_image_properties(img):
        '''
        Properties defined per-img instead of per-ann, to reduce duplicate
        computation.
        '''
        # pick the image that is actually copied to the metrics framework
        # the source field is implied to be a STAC id, but overload it to
        # enable viz during scoring without referring back to the kwcoco file
        # TODO maybe use misc_info for this instead when STAC id is
        # properly passed through to TA-2?
        source = None
        for aux in img.get('auxiliary', []):
            basename = os.path.basename(aux['file_name'])
            if basename.endswith('blue.tif'):
                # source = basename
                source = os.path.abspath(aux['file_name'])
        if source is None:
            try:
                # Pick reasonable source image, we don't have a spec for this
                candidate_keys = [
                    'parent_name', 'parent_file_name', 'name', 'file_name'
                ]
                source = next(filter(None, map(img.get, candidate_keys)))
            except StopIteration:
                raise Exception(f'can\'t determine source of gid {img["gid"]}')

        return {
            'source': source,
            'observation_date': _normalize_date(img['date_captured']),
            'is_occluded': False,  # HACK
            'sensor_name': img['sensor_coarse']
        }

    if with_properties:
        image_properties_dct = {
            gid: per_image_properties(coco_dset.imgs[gid])
            for gid in {ann['image_id']
                        for ann in anns}
        }

    def single_properties(ann):

        current_phase = coco_dset.cats[ann['category_id']]['name']

        return {
            'type': 'observation',
            'current_phase': current_phase,
            'is_site_boundary': True,  # HACK
            'score': ann.get('score', 1.0),
            'misc_info': {},
            **image_properties_dct[ann['image_id']]
        }

    geometry_list = list(map(single_geometry, anns))
    if with_properties:
        properties_list = list(map(single_properties, anns))

    def combined_geometries(geometry_list):
        '''
        # TODO should annotations be disjoint before being combined?
        # this is not true in general
        for geom1, geom2 in itertools.combinations(geometry_list, 2):
            try:
                assert geom1.disjoint(geom2), [ann['id'] for ann in anns]
            except AssertionError:
                xdev.embed()
        '''
        return _combined_geometries(geometry_list)

    def combined_properties(properties_list, geometry_list):
        # list of dicts -> dict of lists for easy indexing
        properties_list = {
            k: [dct[k] for dct in properties_list]
            for k in properties_list[0]
        }

        properties = {}

        def _len(geom):
            if isinstance(geom, shapely.geometry.Polygon):
                return 1  # this is probably the case
            elif isinstance(geom, shapely.geometry.MultiPolygon):
                return len(geom)
            else:
                raise TypeError(type(geom))

        # per-polygon properties
        sep = ','
        for key in ['current_phase', 'is_occluded', 'is_site_boundary']:
            value = []
            for prop, geom in zip(properties_list[key], geometry_list):
                value.append(sep.join(map(str, [prop] * _len(geom))))
            properties[key] = sep.join(value)

        # identical properties
        for key in ['type', 'source', 'observation_date', 'sensor_name']:
            values = properties_list[key]
            assert len(set(values)) == 1
            properties[key] = str(values[0])

        # take area-weighted average score
        properties['score'] = np.average(
            list(map(float, properties_list['score'])),
            weights=[geom.area for geom in geometry_list])
        return properties

        # currently unused
        # dict_union will tkae the first val for each key
        properties['misc_info'] = ub.dict_union(properties_list['misc_info'])

    if with_properties:
        properties = combined_properties(properties_list, geometry_list)
    else:
        properties = {}

    return geojson.Feature(geometry=combined_geometries(geometry_list),
                           properties=properties)


def track_to_site(coco_dset,
                  trackid,
                  region_id,
                  site_idx=None,
                  as_summary=False):
    '''
    Turn a kwcoco track into an IARPA site model or site summary
    '''

    # get annotations in this track, sort them, and group them into features
    annots = coco_dset.annots(trackid=trackid)
    try:
        ixs, gids, anns = annots.lookup(
            'track_index'), annots.gids, annots.objs
        # HACK because track_index isn't unique, need tiebreaker key to sort on
        # _, gids, anns = zip(*sorted(zip(ixs, gids, anns)))
        _, _, gids, anns = zip(*sorted(zip(ixs, range(len(ixs)), gids, anns)))
    except KeyError:
        # if track_index is missing, assume they're already sorted
        gids, anns = annots.gids, annots.objs
    features = [
        geojson_feature(coco_dset.imgs[gid],
                        _anns,
                        coco_dset,
                        with_properties=(not as_summary))
        for gid, _anns in ub.group_items(anns, gids).items()
    ]

    def predict_phase_changes():
        '''
        add prediction field to each feature
        > A “Polygon” should define the foreign members “current_phase”,
        > “predicted_next_phase”, and “predicted_next_phase_date”.
        TODO we need to figure out how to link individual polygons across
        frames within a track when we have >1 polygon per track_index
        (from MultiPolygon or multiple annotations) to handle splitting/merging
        This is because this prediction foreign field is defined wrt the
        CURRENT polygon, not per-observation.
        '''
        for ix, feat in enumerate(features):

            current_phase = feat['properties']['current_phase']
            sep = ','
            n_polys = len(current_phase.split(sep))
            prediction = {
                'predicted_phase': None,
                'predicted_phase_start_date': None,
            }
            for future_feat in features[ix + 1:]:
                future_phase = future_feat['properties']['current_phase']
                future_date = future_feat['properties']['observation_date']
                # HACK need to let these vary between polys in an observation
                if future_phase != current_phase:
                    current_phases = current_phase.split(sep)
                    future_phases = future_phase.split(sep)
                    n_diff = len(current_phases) - len(future_phases)
                    if n_diff > 0:
                        predicted_phase = sep.join(
                            future_phases +
                            current_phases[len(future_phases):])
                    else:
                        predicted_phase = sep.join(
                            future_phases[:len(current_phases)])
                    prediction = {
                        'predicted_phase':
                        predicted_phase,
                        'predicted_phase_start_date':
                        sep.join([future_date] * n_polys),
                    }
                    break

            feat['properties'].update(prediction)

    if not as_summary:
        predict_phase_changes()

    if site_idx is None:
        site_idx = trackid

    def site_feature():
        '''
        Feature containing metadata about the site
        '''
        geometry = _combined_geometries(
            [_single_geometry(feat['geometry']) for feat in features])

        centroid_latlon = np.array(geometry.centroid)[::-1]

        # these are strings, but sorting should be correct in isoformat
        dates = sorted(
            map(_normalize_date,
                coco_dset.images(set(gids)).lookup('date_captured')))

        site_id = '_'.join((region_id, str(site_idx).zfill(4)))

        properties = {
            'site_id': site_id,
            'version': watch.__version__,
            'mgrs': MGRS().toMGRS(*centroid_latlon, MGRSPrecision=0),
            'status': 'positive_annotated',
            'model_content': 'proposed',
            'score': 1.0,  # TODO does this matter?
            'start_date': min(dates),
            'end_date': max(dates),
            'originator': 'kitware',
            'validated': 'False'
        }

        if as_summary:
            properties.update(**{
                'type': 'site_summary',
                'region_id': region_id,  # HACK to passthrough to main
            })
        else:
            properties.update(**{
                'type': 'site',
                'region_id': region_id,
                'misc_info': {}
            })

        return geojson.Feature(geometry=geometry, properties=properties)

    if as_summary:
        return site_feature()
    else:
        return geojson.FeatureCollection([site_feature()] + features)


def convert_kwcoco_to_iarpa(coco_dset, region_id=None, as_summary=False):
    """
    Convert a kwcoco coco_dset to the IARPA JSON format

    Args:
        coco_dset (kwcoco.CocoDataset):
            a coco dataset, but requires images are geotiffs as well as certain
            special fields.

    Returns:
        dict: sites
            dictionary of json-style data in IARPA site format

    Example:
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> from watch.tasks.tracking.normalize import normalize
        >>> from watch.tasks.tracking.from_polygon import MonoTrack
        >>> from watch.demo import smart_kwcoco_demodata
        >>> import ubelt as ub
        >>> coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
        >>> coco_dset = normalize(coco_dset, track_fn=MonoTrack, overwrite=False)
        >>> region_id = 'KR_R001'
        >>> sites = convert_kwcoco_to_iarpa(coco_dset, region_id)
        >>> print('sites = {}'.format(ub.repr2(sites, nl=7, sort=0)))
        >>> import jsonschema
        >>> import watch
        >>> SITE_SCHEMA = watch.rc.load_site_model_schema()
        >>> for site in sites:
        >>>     jsonschema.validate(site, schema=SITE_SCHEMA)

    """
    sites = []

    for vidid, video in coco_dset.index.videos.items():
        if region_id is None:
            _region_id = video['name']
        else:
            _region_id = region_id

        sub_dset = coco_dset.subset(gids=coco_dset.index.vidid_to_gids[vidid])

        for site_idx, trackid in enumerate(sub_dset.index.trackid_to_aids):

            site = track_to_site(sub_dset, trackid, _region_id, site_idx,
                                 as_summary)
            sites.append(site)

    return sites


def add_site_summary_to_kwcoco(site_summary_or_region_model,
                               coco_dset,
                               region_id=None):
    """
    Add a site summary(s) to a kwcoco dataset as a set of polygon annotations.
    These annotations will have category "Site Boundary", 1 track per summary.
    """
    import json
    import jsonschema
    import kwimage
    # input validation
    if isinstance(site_summary_or_region_model, str):
        if os.path.isfile(site_summary_or_region_model):
            with open(site_summary_or_region_model) as f:
                site_summary_or_region_model = json.load(f)
        else:
            site_summary_or_region_model = json.loads(
                site_summary_or_region_model)
    assert isinstance(
        site_summary_or_region_model, dict
    ), f'unknown site summary dtype {type(site_summary_or_region_model)}'
    try:
        region_model_schema = watch.rc.load_region_model_schema()
        region_model = site_summary_or_region_model
        jsonschema.validate(region_model, schema=region_model_schema)
        site_summaries = [
            f for f in region_model['features']
            if f['properties']['type'] == 'site_summary'
        ]
        if region_id is None:
            region_feat = region_model['features'][0]
            assert region_feat['properties']['type'] == 'region'
            region_id = region_feat['properties']['region_id']
    except jsonschema.ValidationError:
        # TODO validate this
        site_summary = site_summary_or_region_model
        site_summaries = [site_summary]
        if region_id is None:
            assert len(coco_dset.index.name_to_video) == 1, 'ambiguous video'
            region_id = ub.peek(coco_dset.index.name_to_video)

    # TODO use pyproj instead, make sure it works with kwimage.warp
    from osgeo import osr

    @ub.memoize
    def transform_wgs84_to(target_epsg_code):
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)  # '+proj=longlat +datum=WGS84 +no_defs'
        target = osr.SpatialReference()
        target.ImportFromEPSG(int(target_epsg_code))
        return osr.CoordinateTransformation(wgs84, target)

    # write site summaries
    print('warping site boundaries to pxl space...')
    cid = coco_dset.ensure_category('Site Boundary')
    new_trackids = watch.utils.kwcoco_extensions.TrackidGenerator(coco_dset)
    for site_summary in site_summaries:

        track_id = next(new_trackids)

        # get relevant images
        images = coco_dset.images(
            vidid=coco_dset.index.name_to_video[region_id]['id'])
        start_date = dateutil.parser.parse(
            site_summary['properties']['start_date']).date()
        end_date = dateutil.parser.parse(
            site_summary['properties']['end_date']).date()
        flags = [
            start_date <= dateutil.parser.parse(date_str).date() <= end_date
            for date_str in images.lookup('date_captured')
        ]
        images = images.compress(flags)

        # apply site boundary as polygons
        geo_poly = kwimage.MultiPolygon.from_geojson(site_summary['geometry'])
        for img in images.objs:
            if 'utm_crs_info' in img:
                utm_epsg_code = img['utm_crs_info']['auth'][1]
            else:
                utm_epsg_code = 4326
            transform_utm_to_pxl = kwimage.Affine.coerce(
                            img.get('wld_to_pxl', {'scale': 1}))
            img_poly = (geo_poly
                        .swap_axes()  # TODO bookkeep this convention
                        .warp(transform_wgs84_to(utm_epsg_code))
                        .warp(transform_utm_to_pxl))
            bbox = list(img_poly.bounding_box().to_coco())[0]
            coco_dset.add_annotation(image_id=img['id'],
                                     category_id=cid,
                                     bbox=bbox,
                                     segmentation=img_poly,
                                     segmentation_geos=geo_poly,
                                     track_id=track_id)

    return coco_dset


def create_region_feature(region_id, site_summaries):
    geometry = _combined_geometries([
        _single_geometry(summary['geometry']) for summary in site_summaries
    ]).envelope
    start_date = min(summary['properties']['start_date']
                     for summary in site_summaries)
    end_date = max(summary['properties']['end_date']
                   for summary in site_summaries)
    properties = {
        'type': 'region',
        'region_id': region_id,
        'version': site_summaries[0]['properties']['version'],
        'mgrs': site_summaries[0]['properties']['mgrs'],
        'model_content': site_summaries[0]['properties']['model_content'],
        'start_date': start_date,
        'end_date': end_date,
        'originator': site_summaries[0]['properties']['originator'],
        'comments': None
    }
    return geojson.Feature(geometry=geometry, properties=properties)


def main(args):
    """
    Example:
        >>> # test BAS and default (SC) modes
        >>> from watch.cli.kwcoco_to_geojson import main
        >>> from watch.demo import smart_kwcoco_demodata
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # run BAS on demodata in a new place
        >>> coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
        >>> coco_dset.fpath = 'bas.kwcoco.json'
        >>> coco_dset.dump(coco_dset.fpath, indent=2)
        >>> region_id = 'dummy_region'
        >>> regions_dir = 'regions/'
        >>> bas_args = [
        >>>     '--in_file', coco_dset.fpath,
        >>>     '--out_dir', regions_dir,
        >>>     '--track_fn', 'watch.tasks.tracking.from_polygon.MonoTrack',
        >>>     '--bas_mode',
        >>>     '--write_in_file'
        >>> ]
        >>> main(bas_args)
        >>> # reload it with tracks
        >>> coco_dset = kwcoco.CocoDataset(coco_dset.fpath)
        >>> # run SC on the same dset
        >>> sites_dir = 'sites/'
        >>> sc_args = [
        >>>     '--in_file', coco_dset.fpath,
        >>>     '--out_dir', sites_dir,
        >>> ]
        >>> main(sc_args)
        >>> # cleanup
        >>> for pth in os.listdir(regions_dir):
        >>>     os.remove(os.path.join(regions_dir, pth))
        >>> os.removedirs(regions_dir)
        >>> for pth in os.listdir(sites_dir):
        >>>     os.remove(os.path.join(sites_dir, pth))
        >>> os.removedirs(sites_dir)
        >>> if not os.path.isabs(coco_dset.fpath):
        >>>     os.remove(coco_dset.fpath)

    Example:
        >>> # test a more complicated track function
        >>> from watch.cli.kwcoco_to_geojson import demo
        >>> from watch.demo import smart_kwcoco_demodata
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # make a new BAS dataset
        >>> coco_dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps(
        >>>     num_videos=2)
        >>> #coco_dset.images().set('sensor_coarse', 'S2')
        >>> for img in coco_dset.imgs.values():
        >>>     img['sensor_coarse'] = 'S2'
        >>> coco_dset.remove_categories(coco_dset.cats.keys())
        >>> coco_dset.fpath = 'bas.kwcoco.json'
        >>> # TODO make serializable, check set() and main()
        >>> coco_dset.dump(coco_dset.fpath, indent=2)
        >>> # make a new SC dataset
        >>> coco_dset_sc = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps(
        >>>     num_videos=2)
        >>> for img in coco_dset_sc.imgs.values():
        >>>     img['sensor_coarse'] = 'S2'
        >>> coco_dset_sc.remove_categories(coco_dset_sc.cats.keys())
        >>> for img in coco_dset_sc.imgs.values():
        >>>     for aux, key in zip(img['auxiliary'],
        >>>                         ['Site Preparation', 'Active Construction',
        >>>                          'Post Construction', 'No Activity']):
        >>>         aux['channels'] = key
        >>> coco_dset_sc.fpath = 'sc.kwcoco.json'
        >>> coco_dset_sc.dump(coco_dset_sc.fpath, indent=2)
        >>> regions_dir = 'regions/'
        >>> sites_dir = 'sites/'
        >>> # moved this to a separate function for length
        >>> demo(coco_dset, regions_dir, coco_dset_sc, sites_dir, cleanup=True)

    """
    parser = argparse.ArgumentParser(
        description='Convert KWCOCO to IARPA GeoJSON')
    required_args = parser.add_argument_group('required')
    required_args.add_argument('--in_file',
                               required=True,
                               help='Input KWCOCO to convert')
    required_args.add_argument('--out_dir',
                               required=True,
                               help=ub.paragraph('''
        Output directory where GeoJSON files will be written.
        NOTE: in --bas_mode, writing to a region is not idempotent.
        To regenerate a region, delete or edit the region file before
        rerunning this script.
        '''))
    convenience_args = parser.add_argument_group('convenience')
    convenience_args.add_argument(
        '--in_file_gt', help='If available, ground truth KWCOCO to visualize')
    convenience_args.add_argument('--region_id',
                                  help=ub.paragraph('''
        ID for region that sites belong to.
        If None, try to infer from kwcoco file.
        '''))
    convenience_args.add_argument('--write_in_file',
                                  action='store_true',
                                  help=ub.paragraph('''
        If set, write the normalized and tracked kwcoco in_file back to disk
        so you can skip the --track_fn next time this is run on it.
        '''))
    track_args = parser.add_argument_group(
        'track', '--track_fn and --default_track_fn are mutually exclusive.')
    track = track_args.add_mutually_exclusive_group()
    track.add_argument('--track_fn',
                       help=ub.paragraph('''
        Function to add tracks. If None, use existing tracks.
        Example: 'watch.tasks.tracking.from_heatmap.TimeAggregatedBAS'
        '''))
    track.add_argument('--default_track_fn',
                       help=ub.paragraph('''
        String code to pick a sensible track_fn based on the contents
        of in_file. Supported codes are ['saliency_heatmaps', 'saliency_polys',
        'class_heatmaps', 'class_polys']. Any other string will be interpreted
        as the image channel to use for 'saliency_heatmaps' (default:
        'salient'). Supported classes are ['Site Preparation',
        'Active Construction', 'Post Construction', 'No Activity']. For
        class_heatmaps, these should be image channels; for class_polys, they
        should be annotation categories.
        '''))
    track_args.add_argument('--track_kwargs',
                            default='{}',
                            help=ub.paragraph('''
        JSON string or path to file containing keyword arguments for the
        chosen TrackFunction. Examples include: coco_dset_gt, coco_dset_sc,
        thresh, possible_keys.
        Any file paths will be loaded as CocoDatasets if possible.
        '''))
    behavior_args = parser.add_argument_group(
            'behavior',
            '--bas_mode is mutually exclusive with other behavior args.')
    behavior_args.add_argument('--bas_mode',
                               action='store_true',
                               help=ub.paragraph('''
        In BAS mode, output will be site summaries instead of sites.
        Region files will be searched for in out_dir, or generated from
        in_file if not found, and site summaries will be appended to them.
        '''))
    behavior_args.add_argument('--site_summary',
                               default=None,
                               help=ub.paragraph('''
        File path or serialized json object containing either a site_summary
        or a region_model that includes site summaries. Each summary found will
        be added to in_file to use in site characterization.
        '''))
    behavior_args.add_argument('--score',
                               action='store_true',
                               help=ub.paragraph('''
        If set, all regions touched will be scored using the metrics framework.
        Additional arguments to this script will be passed to
        run_metrics_framework.py.
        '''))
    args, score_args = parser.parse_known_args(args)
    if score_args and not args.score:
        raise ValueError(f'unknown arguments {score_args}')

    # load the track kwargs
    if os.path.isfile(args.track_kwargs):
        track_kwargs = json.load(args.track_kwargs)
    else:
        track_kwargs = json.loads(args.track_kwargs)
    assert isinstance(track_kwargs, dict)

    # Read the kwcoco file(s)
    coco_dset = kwcoco.CocoDataset.coerce(args.in_file)
    if args.in_file_gt is not None:
        gt_dset = kwcoco.CocoDataset.coerce(args.in_file_gt)
    else:
        gt_dset = None
    for k, v in track_kwargs.items():
        if isinstance(v, str) and os.path.isfile(v):
            try:
                track_kwargs[k] = kwcoco.CocoDataset.coerce(v)
            except json.JSONDecodeError:  # TODO make this smarter
                pass

    # Pick a track_fn
    if args.default_track_fn is not None:
        from watch.tasks.tracking import from_heatmap, from_polygon
        if args.default_track_fn == 'saliency_heatmaps':
            track_fn = from_heatmap.TimeAggregatedBAS
        elif args.default_track_fn == 'saliency_polys':
            track_fn = from_polygon.OverlapTrack
        elif args.default_track_fn == 'class_heatmaps':
            track_fn = from_heatmap.TimeAggregatedSC
        elif args.default_track_fn == 'class_polys':
            track_fn = from_polygon.OverlapTrack
        else:
            track_fn = from_heatmap.TimeAggregatedBAS
            track_kwargs['possible_keys'] = [args.default_track_fn]
    elif args.track_fn is None:
        track_fn = watch.tasks.tracking.utils.NoOpTrackFunction
    else:
        track_fn = eval(args.track_fn)

    if args.site_summary is not None:
        if args.bas_mode:
            raise ValueError('--site_summary cannot be used in --bas_mode')
        coco_dset = add_site_summary_to_kwcoco(
                args.site_summary, coco_dset, args.region_id)

    coco_dset = watch.tasks.tracking.normalize.normalize(
        coco_dset,
        track_fn=track_fn,
        overwrite=False,
        gt_dset=gt_dset,
        **track_kwargs)

    if args.write_in_file:
        coco_dset.dump(args.in_file, indent=2)
    # Convert kwcoco to sites
    sites = convert_kwcoco_to_iarpa(coco_dset,
                                    args.region_id,
                                    as_summary=args.bas_mode)

    verbose = 1
    os.makedirs(args.out_dir, exist_ok=True)
    if args.bas_mode:  # write sites to region models on disk
        for region_id, site_summaries in ub.group_items(
                sites,
                lambda site: site['properties'].pop('region_id')).items():
            region_fpath = os.path.join(args.out_dir,
                                        region_id + '.geojson')
            if os.path.isfile(region_fpath):
                with open(region_fpath, 'r') as f:
                    region = geojson.load(f)
                if verbose:
                    print(f'writing to existing region {region_fpath}')
            else:
                region = geojson.FeatureCollection([
                        create_region_feature(region_id, site_summaries)])
                if verbose:
                    print(f'writing to new region {region_fpath}')
            for site_summary in site_summaries:
                assert site_summary['properties']['type'] == 'site_summary'
                region['features'].append(site_summary)
            with open(region_fpath, 'w') as f:
                geojson.dump(region, f, indent=2)

    else:  # write sites to disk
        for site in sites:
            site_props = site['features'][0]['properties']
            assert site_props['type'] == 'site'
            site_fpath = os.path.join(args.out_dir,
                                      site_props['site_id'] + '.geojson')
            if verbose:
                print(f'writing site {site_props["site_id"]} to new '
                      f'site {site_fpath}')
            with open(os.path.join(site_fpath), 'w') as f:
                geojson.dump(site, f, indent=2)

    if args.score:
        from watch.cli.run_metrics_framework import main
        main([
            score_args,
            '--sites',
        ] + [json.dumps(site) for site in sites])


def demo(coco_dset,
         regions_dir,
         coco_dset_sc,
         sites_dir,
         cleanup=True,
         hybrid=False):
    bas_args = [
        '--in_file',
        coco_dset.fpath,
        '--out_dir',
        regions_dir,
        '--track_fn',
        'watch.tasks.tracking.from_heatmap.'
        'TimeAggregatedBAS',
        '--bas_mode',
        # '--write_in_file'
    ]
    # run BAS on it
    main(bas_args)
    # reload it with tracks
    # coco_dset = kwcoco.CocoDataset(coco_dset.fpath)
    # run SC on both of them
    if hybrid:  # hybrid approach
        sc_args = [
            '--in_file', coco_dset.fpath, '--out_dir', sites_dir, '--track_fn',
            'watch.tasks.tracking.from_heatmap.'
            'TimeAggregatedHybrid', '--track_kwargs',
            ('{"coco_dset_sc": "' + coco_dset_sc.fpath + '"}')
        ]
        main(sc_args)
    else:  # true per-site SC
        import json
        from tempfile import NamedTemporaryFile
        sc_args = [
            '--out_dir',
            sites_dir,
            '--track_fn',
            'watch.tasks.tracking.from_heatmap.'
            'TimeAggregatedSC',
        ]
        for vid_name, vid in coco_dset_sc.index.name_to_video.items():
            gids = coco_dset_sc.index.vidid_to_gids[vid['id']]
            sub_dset = coco_dset_sc.subset(gids)
            tmpfile = NamedTemporaryFile()
            sub_dset.fpath = tmpfile.name
            sub_dset.dump(sub_dset.fpath)
            region = json.load(
                open(os.path.join(regions_dir, f'{vid_name}.geojson')))
            for site in [
                    f for f in region['features']
                    if f['properties']['type'] == 'site_summary'
            ]:
                print('running site ' + site['properties']['site_id'])
                main(sc_args + [
                    '--in_file', sub_dset.fpath,
                    '--track_kwargs', '{"use_boundary_annots": false}'
                ])
                # '--site_summary', json.dumps(site)])
    if cleanup:
        for pth in os.listdir(regions_dir):
            os.remove(os.path.join(regions_dir, pth))
        os.removedirs(regions_dir)
        for pth in os.listdir(sites_dir):
            os.remove(os.path.join(sites_dir, pth))
        os.removedirs(sites_dir)
        if not os.path.isabs(coco_dset.fpath):
            os.remove(coco_dset.fpath)
        if not os.path.isabs(coco_dset_sc.fpath):
            os.remove(coco_dset_sc.fpath)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
