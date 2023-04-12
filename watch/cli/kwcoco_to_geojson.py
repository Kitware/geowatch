#!/usr/bin/env python3
"""
This file contains logic to convert a kwcoco file into an IARPA Site Model.

At a glance the IARPA Site Model is a GeoJSON FeatureCollection with the
following informal schema:

For official documentation about the KWCOCO json format see [1]_. A formal
json-schema can be found in ``kwcoco.coco_schema``

For official documentation about the IARPA json format see [2, 3]_. A formal
json-schema can be found in ``../../watch/rc/site-model.schema.json``.

References:
    .. [1] https://gitlab.kitware.com/computer-vision/kwcoco
    .. [2] https://infrastructure.smartgitlab.com/docs/pages/api/
    .. [3] https://smartgitlab.com/TE/annotations

SeeAlso:
    * ../tasks/tracking/from_heatmap.py
    * ../tasks/tracking/utils.py
    * ../../tests/test_tracker.py

Ignore:
    python -m watch.cli.run_tracker \
        --in_file /data/joncrall/dvc-repos/smart_expt_dvc/_debug/metrics/bas-fusion/bas_fusion_kwcoco.json \
        --out_site_summaries_fpath /data/joncrall/dvc-repos/smart_expt_dvc/_debug/metrics/bas-fusion/tracking_manifests_bas2/region_models_manifest.json \
        --out_site_summaries_dir /data/joncrall/dvc-repos/smart_expt_dvc/_debug/metrics/bas-fusion/region_models2 \
        --out_kwcoco /data/joncrall/dvc-repos/smart_expt_dvc/_debug/metrics/bas-fusion/bas_fusion_kwcoco_tracked3.json \
        --default_track_fn saliency_heatmaps \
        --append_mode=False \
        --track_kwargs "
            thresh: 0.33
            inner_window_size: 1y
            inner_agg_fn: mean
            norm_ord: 1
            agg_fn: probs
            resolution: 10GSD
            moving_window_size: 1
            min_area_square_meters: 7200
            max_area_square_meters: 9000000
            poly_merge_method: v1
        "

        smartwatch visualize /data/joncrall/dvc-repos/smart_expt_dvc/_debug/metrics/bas-fusion/bas_fusion_kwcoco_tracked3.json --smart
"""
import os
import scriptconfig as scfg
import ubelt as ub

if not os.environ.get('_ARGCOMPLETE', ''):
    from watch.tasks.tracking import from_heatmap, from_polygon

    _KNOWN_TRACK_FUNCS = {
        'saliency_heatmaps': from_heatmap.TimeAggregatedBAS,
        'saliency_polys': from_polygon.OverlapTrack,
        'class_heatmaps': from_heatmap.TimeAggregatedSC,
        'site_validation': from_heatmap.TimeAggregatedSV,
        'class_polys': from_polygon.OverlapTrack,
        'mono_track': from_polygon.MonoTrack,
    }

    _trackfn_details_docs = ' --- '.join([
        k + ': ' + ', '.join([field.name for field in v.__dataclass_fields__.values()])
        if hasattr(v, '__dataclass_fields__') else
        k + ':?'
        for k, v in _KNOWN_TRACK_FUNCS.items()
    ])
else:
    _trackfn_details_docs = 'na'


try:
    from xdev import profile
except Exception:
    profile = ub.identity


class KWCocoToGeoJSONConfig(scfg.DataConfig):
    """
    Convert KWCOCO to IARPA GeoJSON
    """

    # TODO: can we store metadata in the type annotation?
    # SeeAlso: notes in from_heatmap.__devnote__
    # e.g.
    #
    # in_file : scfg.Coercable(help='Input KWCOCO to convert', position=1) = None
    # in_file : scfg.Type(help='Input KWCOCO to convert', position=1) = None
    # in_file : scfg.PathLike(help='Input KWCOCO to convert', position=1) = None
    #
    # or
    #
    # in_file = None
    # in_file : scfg.Value[help='Input KWCOCO to convert', position=1]

    in_file = scfg.Value(None, required=True, help='Input KWCOCO to convert',
                         position=1)

    out_kwcoco = scfg.Value(None, help=ub.paragraph(
            '''
            The file path to write the "tracked" kwcoco file to.
            '''))

    out_sites_dir = scfg.Value(None, help=ub.paragraph(
        '''
        The directory where site model geojson files will be written.
        '''))

    out_site_summaries_dir = scfg.Value(None, help=ub.paragraph(
        '''
        The directory path where site summary geojson files will be written.
        '''))

    out_sites_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site models will be written.
        '''))

    out_site_summaries_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site summary geojson files will
        be written.
        '''))

    in_file_gt = scfg.Value(None, help=ub.paragraph(
            '''
            If available, ground truth KWCOCO to visualize
            '''), group='convenience')

    region_id = scfg.Value(None, help=ub.paragraph(
            '''
            ID for region that sites belong to. If None, try to infer
            from kwcoco file.
            '''), group='convenience')

    track_fn = scfg.Value(None, help=ub.paragraph(
            '''
            Function to add tracks. If None, use existing tracks.
            Example:
            'watch.tasks.tracking.from_heatmap.TimeAggregatedBAS'
            '''), group='track', mutex_group=1)
    default_track_fn = scfg.Value(None, help=ub.paragraph(
            '''
            String code to pick a sensible track_fn based on the
            contents of in_file. Supported codes are
            ['saliency_heatmaps', 'saliency_polys', 'class_heatmaps',
            'class_polys']. Any other string will be interpreted as the
            image channel to use for 'saliency_heatmaps' (default:
            'salient'). Supported classes are ['Site Preparation',
            'Active Construction', 'Post Construction', 'No Activity'].
            For class_heatmaps, these should be image channels; for
            class_polys, they should be annotation categories.
            '''), group='track', mutex_group=1)
    track_kwargs = scfg.Value('{}', type=str, help=ub.paragraph(
            f'''
            JSON string or path to file containing keyword arguments for
            the chosen TrackFunction. Examples include: coco_dset_gt,
            coco_dset_sc, thresh, key. Any file paths will be loaded as
            CocoDatasets if possible.

            Valid params for each track_fn are: {_trackfn_details_docs}
            '''), group='track')
    viz_out_dir = scfg.Value(None, help=ub.paragraph(
            '''
            Directory to save tracking vizualizations to; if None, don't viz
            '''), group='track')
    site_summary = scfg.Value(None, help=ub.paragraph(
            '''
            A filepath glob or json blob containing either a
            site_summary or a region_model that includes site summaries.
            Each summary found will be added to in_file as
            'Site Boundary' annotations.
            '''), group='behavior')
    clear_annots = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Clears all annotations before running tracking, so it starts
            from a clean slate.
            '''), group='behavior')
    append_mode = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Append sites to existing region GeoJSON.
            '''), group='behavior')

    boundary_region = scfg.Value(None, help=ub.paragraph(
            '''
            A path or globstring coercable to one or more region files that
            will define the bounds of where the tracker is allowed to predict
            sites. Any site outside of these bounds will be removed.
            '''), group='behavior')
    # filter_out_of_bounds = scfg.Value(False, isflag=True, help=ub.paragraph(
    #         '''
    #         if True, any tracked site outside of the region bounds
    #         (as specified in the site summary) will be removed.
    #         '''), group='behavior')


__config__ = KWCocoToGeoJSONConfig


def _single_geometry(geom):
    import shapely
    return shapely.geometry.shape(geom).buffer(0)


def _ensure_multi(poly):
    """
    Args:
        poly (Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon])

    Returns:
        shapely.geometry.MultiPolygon
    """
    # ) -> shapely.geometry.MultiPolygon:
    #     poly: Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]
    # ) -> shapely.geometry.MultiPolygon:
    import shapely
    if isinstance(poly, shapely.geometry.MultiPolygon):
        return poly
    elif isinstance(poly, shapely.geometry.Polygon):
        return shapely.geometry.MultiPolygon([poly])
    else:
        raise TypeError(f'{poly} of type {type(poly)}')


def _combined_geometries(geometry_list):
    import shapely.ops
    return shapely.ops.unary_union(geometry_list).buffer(0)


def _normalize_date(date_str):
    import dateutil.parser
    return dateutil.parser.parse(date_str).date().isoformat()


# For MultiPolygon observations. Could be ', '?
sep = ','


@profile
def coco_create_observation(coco_dset, anns, with_properties=True):
    '''
    Group kwcoco annotations in the same track (site) and image
    into one Feature in an IARPA site model
    '''
    import geojson
    import kwcoco
    import shapely
    import numpy as np
    from collections import defaultdict

    def single_geometry(ann):
        seg_geo = ann['segmentation_geos']
        assert isinstance(seg_geo, dict)
        return _single_geometry(seg_geo)

    @profile
    def _per_image_properties(coco_img: kwcoco.CocoImage):
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
        img = coco_img.img
        bundle_dpath = ub.Path(coco_img.bundle_dpath)
        chan_to_aux = {
            aux['channels']: aux
            for aux in coco_img.iter_asset_objs()
        }
        for want_chan in {
                'r|g|b', 'rgb', 'pan', 'panchromatic', 'green', 'blue'
        }:
            if want_chan in chan_to_aux:
                aux = chan_to_aux[want_chan]
                source = bundle_dpath / aux['file_name']
                break

        # if source is None:
        #     # Fallback to the "primary" filepath (too slow)
        #     source = os.path.abspath(coco_img.primary_image_filepath())

        if source is None:
            try:
                # Note, this will likely fail
                # Pick reasonable source image, we don't have a spec for this
                candidate_keys = [
                    'parent_name', 'parent_file_name', 'name', 'file_name'
                ]
                source = next(filter(None, map(img.get, candidate_keys)))
            except StopIteration:
                raise Exception(f'can\'t determine source of gid {img["gid"]}')

        date = _normalize_date(img['date_captured'])

        return {
            'source': source,
            'observation_date': date,
            'is_occluded': False,  # HACK
            'sensor_name': img['sensor_coarse']
        }

    def single_properties(ann):

        current_phase = coco_dset.cats[ann['category_id']]['name']

        return {
            'type': 'observation',
            'current_phase': current_phase,
            'is_site_boundary': True,  # HACK
            'score': ann.get('score', 1.0),
            'misc_info': {
                'phase_transition_days': ann.get('phase_transition_days', None)
            },
            **image_properties_dct[ann['image_id']]
        }

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
                try:
                    return len(geom)
                except TypeError:
                    return len(geom.geoms)
            else:
                raise TypeError(type(geom))

        # per-polygon properties
        for key in ['current_phase', 'is_occluded', 'is_site_boundary']:
            value = []
            for prop, geom in zip(properties_list[key], geometry_list):
                value.append(sep.join(map(str, [prop] * _len(geom))))
            properties[key] = sep.join(value)

        # HACK
        # We are not being scored on multipolygons in SC right now!
        # https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/24
        #
        # When safe, merge class labels for multipolygons so they'll be scored.
        if 1:
            phase = properties['current_phase'].split(sep)
            if len(phase) > 1 and len(set(phase)) == 1:
                properties['current_phase'] = phase[0]

        # identical properties
        for key in ['type', 'source', 'observation_date', 'sensor_name']:
            values = properties_list[key]
            assert len(set(values)) == 1
            properties[key] = str(values[0])

        # take area-weighted average score
        weights = np.array([geom.area for geom in geometry_list])
        scores = np.array([float(s) for s in properties_list['score']])
        mask = np.isnan(scores)
        scores[mask] = 0
        weights[mask] = 0
        if weights.sum() == 0:
            properties['score'] = 0
        else:
            properties['score'] = np.average(scores, weights=weights)

        properties['misc_info'] = defaultdict(list)
        for misc_info in properties_list['misc_info']:
            for k, v in misc_info.items():
                properties['misc_info'][k].append(v)

        return properties

    if with_properties:
        image_properties_dct = {}
        gids = {ann['image_id'] for ann in anns}
        for gid in gids:
            coco_img = coco_dset.coco_image(gid).detach()
            image_properties_dct[gid] = _per_image_properties(coco_img)

    geometry_list = list(map(single_geometry, anns))
    if with_properties:
        properties_list = list(map(single_properties, anns))

    if with_properties:
        geometry = _ensure_multi(_combined_geometries(geometry_list))
        properties = combined_properties(properties_list, geometry_list)
    else:
        geometry = _combined_geometries(geometry_list)
        properties = {}

    # from watch.geoannots import geomodels
    # return geomodels.Observation(geometry=geometry, properties=properties)
    return geojson.Feature(geometry=geometry, properties=properties)


@profile
def coco_track_to_site(coco_dset, trackid, region_id, site_idx=None,
                       as_summary=False):
    '''
    Turn a kwcoco track into an IARPA site model or site summary
    '''
    import geojson

    # get annotations in this track sorted by frame_index
    annots = coco_dset.annots(track_id=trackid)
    gids, anns = annots.gids, annots.objs

    features = [
        coco_create_observation(coco_dset, _anns, with_properties=(not as_summary))
        for gid, _anns in ub.group_items(anns, gids).items()
    ]

    # HACK to passthrough site_summary IDs
    import watch
    if watch.tasks.tracking.utils.trackid_is_default(trackid):
        if site_idx is None:
            site_idx = trackid
        site_id = '_'.join((region_id, str(site_idx).zfill(4)))
    else:
        site_id = trackid
        # TODO make more robust
        region_id = '_'.join(site_id.split('_')[:2])

    if as_summary:
        return coco_create_site_header(coco_dset, region_id, site_id, trackid, gids, features, as_summary)
    else:
        site_header = coco_create_site_header(coco_dset, region_id, site_id, trackid, gids, features, as_summary)
        return geojson.FeatureCollection([site_header] + features)


def predict_phase_changes(site_id, features):
    '''
    Set predicted_phase_transition and predicted_phase_transition_date.

    This should only kick in when the site does not end before the current
    day (latest available image). See tracking.normalize.normalize_phases
    for what happens if the site has ended.

    https://smartgitlab.com/TE/standards/-/wikis/Site-Model-Specification
    '''
    import datetime as datetime_mod
    import dateutil.parser
    import itertools
    all_phases = [
        feat['properties']['current_phase'].split(sep) for feat in features
    ]

    tomorrow = (dateutil.parser.parse(
        features[-1]['properties']['observation_date']) +
                datetime_mod.timedelta(days=1)).isoformat()

    def transition_date_from(phase):
        for feat, phases in zip(reversed(features), reversed(all_phases)):
            if phase in phases:
                return (dateutil.parser.parse(
                    feat['properties']['observation_date']) +
                        datetime_mod.timedelta(
                            int(feat['properties']['misc_info']
                                ['phase_transition_days'][phases.index(
                                    phase)]))).isoformat()
        print(f'warning: {site_id=} is missing {phase=}')
        return tomorrow

    all_phases_set = set(itertools.chain.from_iterable(all_phases))

    if 'Post Construction' in all_phases_set:
        return {}
    elif 'Active Construction' in all_phases_set:
        return {
            'predicted_phase_transition':
            'Post Construction',
            'predicted_phase_transition_date':
            transition_date_from('Active Construction')
        }
    elif 'Site Preparation' in all_phases_set:
        return {
            'predicted_phase_transition':
            'Active Construction',
            'predicted_phase_transition_date':
            transition_date_from('Site Preparation')
        }
    else:
        # raise ValueError(f'missing phases: {site_id=} {all_phases_set=}')
        print(f'missing phases: {site_id=} {all_phases_set=}')
        return {}


def coco_create_site_header(coco_dset, region_id, site_id, trackid, gids, features, as_summary):
    '''
    Feature containing metadata about the site
    '''
    from mgrs import MGRS
    import numpy as np
    import geojson
    import shapely

    geom_list = [_single_geometry(feat['geometry']) for feat in features]
    geometry = _combined_geometries(geom_list)

    # site and site_summary features must be polygons
    if isinstance(geometry, shapely.geometry.MultiPolygon):
        if len(geometry.geoms) == 1:
            geometry = geometry.geoms[0]
        else:
            print(f'warning: {coco_dset=} {region_id=} {site_id=} {trackid=} has multi-part site geometry')
            geometry = geometry.convex_hull

    centroid_coords = np.array(geometry.centroid.coords)
    if centroid_coords.size == 0:
        raise AssertionError('Empty geometry. What happened?')

    centroid_latlon = centroid_coords[0][::-1]

    # these are strings, but sorting should be correct in isoformat
    dates = sorted(
        map(_normalize_date,
            coco_dset.images(set(gids)).lookup('date_captured')))

    # https://smartgitlab.com/TE/annotations/-/wikis/Annotation-Status-Types#for-site-models-generated-by-performersalgorithms
    # system_confirmed, system_rejected, or system_proposed
    # TODO system_proposed pre val-net
    status = set(coco_dset.annots(track_id=trackid).get('status', 'system_confirmed'))
    assert len(status) == 1, f'inconsistent {status=} for {trackid=}'
    status = status.pop()

    PERFORMER_ID = 'kit'

    import watch
    properties = {
        'site_id': site_id,
        'version': watch.__version__,  # Shouldn't this be a schema version?
        'mgrs': MGRS().toMGRS(*centroid_latlon, MGRSPrecision=0),
        'status': status,
        'model_content': 'proposed',
        'score': 1.0,  # TODO does this matter?
        'start_date': min(dates),
        'end_date': max(dates),
        'originator': PERFORMER_ID,
        'validated': 'False'
    }

    if as_summary:
        properties.update(
            **{
                'type': 'site_summary',
                'region_id': region_id,  # HACK to passthrough to main
            })
    else:
        properties.update(
            **{
                'type': 'site',
                'region_id': region_id,
                **predict_phase_changes(site_id, features), 'misc_info': {}
            })

    return geojson.Feature(geometry=geometry, properties=properties)


@profile
def create_region_header(region_id, site_summaries):
    import geojson
    geometry = _combined_geometries([
        _single_geometry(summary['geometry'])
        for summary in site_summaries
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


@profile
def convert_kwcoco_to_iarpa(coco_dset,
                            default_region_id=None,
                            as_summary=False):
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
        >>> import watch
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> from watch.tasks.tracking.normalize import run_tracking_pipeline
        >>> from watch.tasks.tracking.from_polygon import MonoTrack
        >>> import ubelt as ub
        >>> coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)
        >>> coco_dset = run_tracking_pipeline(coco_dset, track_fn=MonoTrack, overwrite=False)
        >>> videos = coco_dset.videos()
        >>> videos.set('name', ['DM_R{:03d}'.format(vidid) for vidid in videos])
        >>> sites = convert_kwcoco_to_iarpa(coco_dset)
        >>> print(f'{len(sites)} sites')
        >>> if 0:  # validation fails
        >>>     import jsonschema
        >>>     SITE_SCHEMA = watch.rc.load_site_model_schema()
        >>>     for site in sites:
        >>>         jsonschema.validate(site, schema=SITE_SCHEMA)
        >>> elif 0:  # but this works if metrics are available
        >>>     import tempfile
        >>>     import json
        >>>     from iarpa_smart_metrics.evaluation import SiteStack
        >>>     for site in sites:
        >>>         with tempfile.NamedTemporaryFile() as f:
        >>>             json.dump(site, open(f.name, 'w'))
        >>>             SiteStack(f.name)
    """
    sites = []
    for vidid, video in coco_dset.index.videos.items():
        region_id = video.get('name', default_region_id)
        gids = coco_dset.index.vidid_to_gids[vidid]
        sub_dset = coco_dset.subset(gids=gids)

        for site_idx, trackid in enumerate(sub_dset.index.trackid_to_aids):
            site = coco_track_to_site(sub_dset, trackid, region_id, site_idx,
                                      as_summary)
            sites.append(site)

    return sites


# debug mode is for comparing against a set of known GT site models
DEBUG_MODE = 0
if DEBUG_MODE:
    SITE_SUMMARY_POS_STATUS = {
        'positive_annotated',
        'system_proposed', 'system_confirmed',
        'positive_annotated_static',  # TODO confirm
        'negative',
    }
else:
    # TODO handle positive_partial
    SITE_SUMMARY_POS_STATUS = {
        'positive_annotated',
        'system_proposed', 'system_confirmed',
    }


def _coerce_site_summaries(site_summary_or_region_model,
                           default_region_id=None):
    """
    Possible input formats:
        - file path
        - globbed file paths
        - stringified json blob
    Leading to a:
        - site summary
        - region model containing site summaries

    Args:
        default_region_id: for summaries.
            Region models should already have a region_id.
        strict: if True, raise error on unknown input

    Returns:
        List[Tuple[str, Dict]]
           Each tuple is a (region_id, site_summary) pair
    """
    from watch.utils import util_gis
    from watch.geoannots import geomodels
    import jsonschema

    TRUST_REGION_SCHEMA = 1

    geojson_infos = list(util_gis.coerce_geojson_datas(
        site_summary_or_region_model, format='json', allow_raw=True))

    # validate the json
    site_summaries = []

    for info in geojson_infos:
        data = info['data']

        if not isinstance(data, dict):
            raise AssertionError(
                f'unknown site summary {type(data)=}'
            )

        try:  # is this a region model?

            region_model = geomodels.RegionModel(**data)

            if TRUST_REGION_SCHEMA:
                region_model.validate(strict=False)

            region_model._validate_quick_checks()
            region_header = region_model.header
            # assert region_header['type'] == 'Feature'
            # if region_header['properties']['type'] not in {'region', 'site_summary'}:
            #     raise jsonschema.ValidationError('not a region')

            _summaries = [
                f for f in region_model.site_summaries()
                if f['properties']['status'] in SITE_SUMMARY_POS_STATUS
            ]
            region_id = region_header['properties'].get('region_id', default_region_id)
            site_summaries.extend([(region_id, s) for s in _summaries])

        except jsonschema.ValidationError:
            # In this case we expect the input to be a list of site summaries.
            # However, we really shouldn't hit this case.
            raise AssertionError(
                'Jon thinks we wont hit this case. '
                'If you see this error, he is wrong and the error can be removed. '
                'Otherwise we should remove this extra code')
            site_summary = site_summary_or_region_model
            site_summaries.append((default_region_id, site_summary))

    return site_summaries


def add_site_summary_to_kwcoco(possible_summaries,
                               coco_dset,
                               default_region_id=None):
    """
    Add a site summary(s) to a kwcoco dataset as a set of polygon annotations.
    These annotations will have category "Site Boundary", 1 track per summary.

    This function is mainly for SC. The "possible_summaries" indicate regions
    flagged by BAS (which could also be truth data if we are evaluating SC
    independently) that need SC processing. We need to associate these and
    place them in the correct videos so we can process those areas.
    """
    import dateutil.parser

    # input validation
    print(f'possible_summaries={possible_summaries}')
    print(f'coco_dset={coco_dset}')
    print(f'default_region_id={default_region_id}')

    if default_region_id is None:
        default_region_id = ub.peek(coco_dset.index.name_to_video)

    site_summary_or_region_model = possible_summaries
    site_summaries = _coerce_site_summaries(site_summary_or_region_model,
                                            default_region_id)
    print(f'found {len(site_summaries)} site summaries')

    # TODO use pyproj instead, make sure it works with kwimage.warp

    # @ub.memoize
    # def transform_wgs84_to(target_epsg_code):
    #     wgs84 = osr.SpatialReference()
    #     wgs84.ImportFromEPSG(4326)  # '+proj=longlat +datum=WGS84 +no_defs'
    #     target = osr.SpatialReference()
    #     target.ImportFromEPSG(int(target_epsg_code))
    #     return osr.CoordinateTransformation(wgs84, target)
    # new_trackids = watch.utils.kwcoco_extensions.TrackidGenerator(coco_dset)

    # write site summaries
    import watch
    cid = coco_dset.ensure_category(watch.heuristics.SITE_SUMMARY_CNAME)

    print('Searching for assignment between requested site summaries and the kwcoco videos')

    site_idx_to_vidid = []
    unassigned_site_idxs = []

    USE_NAME_ASSIGNMENT = DEBUG_MODE  # off by default, for known site models
    USE_GEO_ASSIGNMENT = 1

    if USE_NAME_ASSIGNMENT:
        vids = coco_dset.videos().lookup(['name', 'id'])
        for i, (_, s) in enumerate(site_summaries):
            sid = s['properties']['site_id']
            for vid, vn in zip(vids['id'], vids['name']):
                if sid in vn:
                    site_idx_to_vidid.append((i, vid))

    elif USE_GEO_ASSIGNMENT:
        from watch.utils import kwcoco_extensions
        from watch.utils import util_gis
        import geopandas as gpd
        # video_gdf = kwcoco_extensions.covered_video_geo_regions(coco_dset)
        image_gdf = kwcoco_extensions.covered_image_geo_regions(coco_dset)
        video_rows = []
        for video_id, img_gdf in image_gdf.groupby('video_id'):
            row = {
                'video_id': video_id,
                'geometry': img_gdf['geometry'].unary_union,
                'start_date': img_gdf.iloc[0]['date_captured'],
                'end_date': img_gdf.iloc[-1]['date_captured'],
            }
            video_rows.append(row)
        video_gdf = gpd.GeoDataFrame(video_rows, crs=util_gis._get_crs84())

        sitesum_gdf = gpd.GeoDataFrame.from_features([t[1] for t in site_summaries], crs=util_gis._get_crs84(), columns=['geometry'])

        site_idx_to_video_idx = util_gis.geopandas_pairwise_overlaps(sitesum_gdf, video_gdf)

        assigned_idx_pairs = []
        for site_idx, video_idxs in site_idx_to_video_idx.items():
            if len(video_idxs) == 0:
                unassigned_site_idxs.append(site_idx)
            elif len(video_idxs) == 1:
                assigned_idx_pairs.append((site_idx, video_idxs[0]))
            else:
                qshape = sitesum_gdf.iloc[site_idx]['geometry']
                candidates = video_gdf.iloc[video_idxs]
                overlaps = []
                for dshape in candidates['geometry']:
                    iarea = qshape.intersection(dshape).area
                    uarea = qshape.area
                    iofa = iarea / uarea
                    overlaps.append(iofa)
                idx = ub.argmax(overlaps)
                assigned_idx_pairs.append((site_idx, video_idxs[idx]))

        for site_idx, video_idx in assigned_idx_pairs:
            video_id = video_gdf.iloc[video_idx]['video_id']
            site_idx_to_vidid.append((site_idx, video_id, ))

    else:

        for region_id, site_summary in site_summaries:
            # lookup possible places to put this site_summary
            video_id = None
            site_id = site_summary['properties']['site_id']
            for _id, vid in coco_dset.index.videos.items():
                # look for all possible places a region or site id could be
                names = set(ub.dict_subset(vid, ['id', 'name', 'region_id', 'site_id'], None).values())
                names |= set(ub.dict_subset(vid.get('properties', {}), ['region_id', 'site_id'], None).values())
                if region_id in names or site_id in names:
                    video_id = _id
                    print(f'matched site_summary {site_id} to video {names}')
                    site_idx_to_vidid.append((site_idx, video_id, ))
                    break

    assigned_vidids = set([t[1] for t in site_idx_to_vidid])
    print('There were {} / {} assigned site summaries'.format(len(site_idx_to_vidid), len(site_summaries)))
    print('There were {} / {} assigned videos'.format(len(assigned_vidids), coco_dset.n_videos))

    if 0:
        unassigned_vidids = set(coco_dset.videos()) - assigned_vidids
        coco_dset.videos(unassigned_vidids).lookup('name')

    print('warping site boundaries to pxl space...')
    for site_idx, video_id in site_idx_to_vidid:

        region_id, site_summary = site_summaries[site_idx]
        site_id = site_summary['properties']['site_id']

        # track_id = next(new_trackids)
        track_id = site_id

        # get relevant images
        images = coco_dset.images(vidid=video_id)
        start_date = dateutil.parser.parse(
            site_summary['properties']['start_date']).date()
        end_date = dateutil.parser.parse(
            site_summary['properties']['end_date']).date()
        flags = [
            start_date <= dateutil.parser.parse(date_str).date() <= end_date
            for date_str in images.lookup('date_captured')
        ]
        images = images.compress(flags)
        if track_id in images.get('track_id', None):
            print(f'warning: site_summary {track_id} already in dset!')

        # apply site boundary as polygons
        poly_crs84_geojson = site_summary['geometry']
        # geo_poly = kwimage.MultiPolygon.from_geojson()
        for img in images.objs:
            # Add annotations in CRS84 geo-space, we will project to pixel
            # space in a later step
            coco_dset.add_annotation(
                image_id=img['id'],
                category_id=cid,
                # bbox=bbox,
                # segmentation=img_poly,
                segmentation_geos=poly_crs84_geojson,
                track_id=track_id
            )

    print('Projecting regions to pixel coords')
    from watch.utils import kwcoco_extensions
    kwcoco_extensions.warp_annot_segmentations_from_geos(coco_dset)
    print('Done projecting')

    if 0:
        import kwplot
        import kwimage
        kwplot.autompl()
        gid = list(images)[0]
        coco_img = coco_dset.coco_image(gid)
        canvas = coco_img.imdelay('red|green|blue', space='image').finalize()
        canvas = kwimage.normalize_intensity(canvas)
        kwplot.imshow(canvas)
        coco_dset.annots(gid=gid).detections.draw()

    return coco_dset


@profile
def main(argv=None, **kwargs):
    """
    Example:
        >>> # test BAS and default (SC) modes
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> from watch.cli.kwcoco_to_geojson import main
        >>> from watch.demo import smart_kwcoco_demodata
        >>> from watch.utils import util_gis
        >>> import json
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # run BAS on demodata in a new place
        >>> import watch
        >>> coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)
        >>> dpath = ub.Path.appdir('watch', 'test', 'tracking', 'main0').ensuredir()
        >>> coco_dset.reroot(absolute=True)
        >>> coco_dset.fpath = dpath / 'bas_input.kwcoco.json'
        >>> coco_dset.clear_annotations()
        >>> coco_dset.dump(coco_dset.fpath, indent=2)
        >>> region_id = 'dummy_region'
        >>> regions_dir = dpath / 'regions/'
        >>> bas_coco_fpath = dpath / 'bas_output.kwcoco.json'
        >>> sc_coco_fpath = dpath / 'sc_output.kwcoco.json'
        >>> bas_fpath = dpath / 'bas_sites.json'
        >>> sc_fpath = dpath / 'sc_sites.json'
        >>> # Run BAS
        >>> argv = bas_args = [
        >>>     '--in_file', coco_dset.fpath,
        >>>     '--out_site_summaries_dir', str(regions_dir),
        >>>     '--out_site_summaries_fpath',  str(bas_fpath),
        >>>     '--out_kwcoco', str(bas_coco_fpath),
        >>>     '--track_fn', 'saliency_heatmaps',
        >>>     '--track_kwargs', json.dumps({
        >>>        'thresh': 1e-9, 'min_area_square_meters': None,
        >>>        'max_area_square_meters': None,
        >>>        'polygon_simplify_tolerance': 1}),
        >>> ]
        >>> main(argv)
        >>> # Run SC on the same dset, but with BAS pred sites removed
        >>> sites_dir = dpath / 'sites'
        >>> argv = sc_args = [
        >>>     '--in_file', coco_dset.fpath,
        >>>     '--out_sites_dir', str(sites_dir),
        >>>     '--out_sites_fpath', str(sc_fpath),
        >>>     '--out_kwcoco', str(sc_coco_fpath),
        >>>     '--track_fn', 'class_heatmaps',
        >>>     '--site_summary', str(bas_fpath),
        >>>     '--track_kwargs', json.dumps(
        >>>         {'thresh': 1e-9, 'min_area_square_meters': None, 'max_area_square_meters': None,
        >>>          'polygon_simplify_tolerance': 1, 'key': 'salient'}),
        >>> ]
        >>> main(argv)
        >>> # Check expected results
        >>> bas_coco_dset = kwcoco.CocoDataset(bas_coco_fpath)
        >>> sc_coco_dset = kwcoco.CocoDataset(sc_coco_fpath)
        >>> bas_trackids = bas_coco_dset.annots().lookup('track_id', None)
        >>> sc_trackids = sc_coco_dset.annots().lookup('track_id', None)
        >>> assert len(bas_trackids) and None not in bas_trackids
        >>> assert len(sc_trackids) and None not in sc_trackids
        >>> summaries = list(util_gis.coerce_geojson_datas(bas_fpath, format='dataframe'))
        >>> sites = list(util_gis.coerce_geojson_datas(sc_fpath, format='dataframe'))
        >>> import pandas as pd
        >>> sc_df = pd.concat([d['data'] for d in sites])
        >>> bas_df = pd.concat([d['data'] for d in summaries])
        >>> ssum_rows = bas_df[bas_df['type'] == 'site_summary']
        >>> site_rows = sc_df[sc_df['type'] == 'site']
        >>> obs_rows = sc_df[sc_df['type'] == 'observation']
        >>> assert len(site_rows) > 0
        >>> assert len(ssum_rows) > 0
        >>> assert len(ssum_rows) == len(site_rows)
        >>> assert len(obs_rows) > len(site_rows)
        >>> # Cleanup
        >>> #dpath.delete()

    Example:
        >>> # test resolution
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> from watch.cli.kwcoco_to_geojson import main
        >>> import watch
        >>> dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)
        >>> dpath = ub.Path.appdir('watch', 'test', 'tracking', 'main1').ensuredir()
        >>> out_fpath = dpath / 'resolution_test.kwcoco.json'
        >>> regions_dir = dpath / 'regions'
        >>> bas_fpath = dpath / 'bas_sites.json'
        >>> import json
        >>> track_kwargs = json.dumps({
        >>>         'resolution': '10GSD',
        >>>         'min_area_square_meters': 1000000,  # high area threshold filters results
        >>>         'max_area_square_meters': None,
        >>>         'thresh': 1e-9,
        >>> })
        >>> kwargs = {
        >>>     'in_file': str(dset.fpath),
        >>>     'out_site_summaries_dir': str(regions_dir),
        >>>     'out_site_summaries_fpath':  str(bas_fpath),
        >>>     'out_kwcoco': str(out_fpath),
        >>>     'track_fn': 'saliency_heatmaps',
        >>>     'track_kwargs': track_kwargs,
        >>> }
        >>> argv = None
        >>> # Test case for no results
        >>> main(argv=argv, **kwargs)
        >>> from watch.utils import util_gis
        >>> assert len(list(util_gis.coerce_geojson_datas(bas_fpath))) == 0
        >>> # Try to get results here
        >>> track_kwargs = json.dumps({
        >>>         'resolution': '10GSD',
        >>>         'min_area_square_meters': None,
        >>>         'max_area_square_meters': None,
        >>>         'thresh': 1e-9,
        >>> })
        >>> kwargs = {
        >>>     'in_file': str(dset.fpath),
        >>>     'out_site_summaries_dir': str(regions_dir),
        >>>     'out_site_summaries_fpath':  str(bas_fpath),
        >>>     'out_kwcoco': str(out_fpath),
        >>>     'track_fn': 'saliency_heatmaps',
        >>>     'track_kwargs': track_kwargs,
        >>> }
        >>> argv = None
        >>> main(argv=argv, **kwargs)
        >>> assert len(list(util_gis.coerce_geojson_datas(bas_fpath))) > 0

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # test a more complicated track function
        >>> import watch
        >>> from watch.cli.kwcoco_to_geojson import demo
        >>> import kwcoco
        >>> import watch
        >>> import ubelt as ub
        >>> # make a new BAS dataset
        >>> coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True)
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
    args = KWCocoToGeoJSONConfig.cli(argv=argv, data=kwargs, strict=True)
    import rich
    rich.print('args = {}'.format(ub.urepr(args, nl=1)))

    import geojson
    import json
    import kwcoco
    import safer
    import watch
    from kwcoco.util import util_json
    from watch.utils import util_gis
    from watch.utils import process_context
    from watch.utils.util_yaml import Yaml

    coco_fpath = ub.Path(args.in_file)

    if args.out_sites_dir is not None:
        args.out_sites_dir = ub.Path(args.out_sites_dir)

    if args.out_site_summaries_dir is not None:
        args.out_site_summaries_dir = ub.Path(args.out_site_summaries_dir)

    if args.out_sites_fpath is not None:
        if args.out_sites_dir is None:
            raise ValueError(
                'The directory to store individual sites must be specified')
        args.out_sites_fpath = ub.Path(args.out_sites_fpath)
        if not str(args.out_sites_fpath).endswith('.json'):
            raise ValueError('out_sites_fpath should have a .json extension')

    if args.out_site_summaries_fpath is not None:
        if args.out_site_summaries_dir is None:
            raise ValueError(
                'The directory to store individual site summaries must be '
                'specified')
        args.out_site_summaries_fpath = ub.Path(args.out_site_summaries_fpath)
        if not str(args.out_site_summaries_fpath).endswith('.json'):
            raise ValueError('out_site_summaries_fpath should have a .json extension')

    # load the track kwargs
    track_kwargs = Yaml.coerce(args.track_kwargs)
    assert isinstance(track_kwargs, dict)

    # Read the kwcoco file
    coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
    # HACK read auxiliary kwcoco files
    # these aren't used for core tracking algos, supported for legacy
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

    if args.clear_annots:
        coco_dset.clear_annotations()

    pred_info = coco_dset.dataset.get('info', [])

    tracking_output = {
        'type': 'tracking_result',
        'info': [],
        'files': [],
    }
    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(args.asdict())
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    # TODO: ensure all args are resolved here.
    info = tracking_output['info']

    proc_context = process_context.ProcessContext(
        name='watch.cli.kwcoco_to_geojson', type='process',
        config=jsonified_config,
        extra={'pred_info': pred_info},
        track_emissions=False,
    )
    proc_context.start()
    info.append(proc_context.obj)

    # Pick a track_fn
    # HACK remove potentially conflicting annotations as well
    # we shouldn't have saliency annots when we want class or vice versa
    CLEAN_DSET = 1
    class_cats = [cat['name'] for cat in watch.heuristics.CATEGORIES]
    saliency_cats = ['salient']

    track_fn = args.track_fn
    if track_fn is None:
        track_fn = (
            watch.tasks.tracking.utils.NoOpTrackFunction
            if args.default_track_fn is None else
            args.default_track_fn
        )

    if isinstance(track_fn, str):
        # TODO: we should be able to let the user know about these algorithms
        # and parameters. Can jsonargparse help here?
        if CLEAN_DSET:
            if 'class_' in track_fn:
                coco_dset.remove_categories(saliency_cats)
            else:
                coco_dset.remove_categories(class_cats)

        track_fn = _KNOWN_TRACK_FUNCS.get(track_fn, None)
        if track_fn is None:
            raise RuntimeError('Old code would have evaled track_fn, we dont want to do that. '
                               'Please change your code to specify a known track function')

    if track_fn is None:
        raise KeyError(
            f'Unknown Default Track Function: {args.default_track_fn} not in {list(_KNOWN_TRACK_FUNCS.keys())}')

    if args.boundary_region is not None:
        from watch.geoannots import geomodels
        region_infos = list(util_gis.coerce_geojson_datas(
            args.boundary_region, format='json', allow_raw=True))
        import pandas as pd
        region_parts = []
        for info in region_infos:
            # Need to deterimine which one to use
            region_model = geomodels.RegionModel(**info['data'])
            region_gdf = region_model.pandas_region()
            region_parts.append(region_gdf)
        boundary_regions_gdf = pd.concat(region_parts).reset_index()
    else:
        boundary_regions_gdf = None
        # if args.site_summary is None:
        #     raise ValueError('You must specify a region as a site summary if you ')
        ...

    # add site summaries (site boundary annotations)
    if args.site_summary is not None:
        coco_dset = add_site_summary_to_kwcoco(args.site_summary, coco_dset,
                                               args.region_id)
        cid = coco_dset.name_to_cat[watch.heuristics.SITE_SUMMARY_CNAME]['id']
        coco_dset = coco_dset.subset(coco_dset.index.cid_to_gids[cid])
        print('restricting dset to videos with site_summary annots: ',
              set(coco_dset.index.name_to_video))
        assert coco_dset.n_images > 0, 'no valid videos!'

    print(f'track_fn={track_fn}')
    """
    ../tasks/tracking/normalize.py
    """
    coco_dset = watch.tasks.tracking.normalize.run_tracking_pipeline(
        coco_dset, track_fn=track_fn, gt_dset=gt_dset,
        viz_out_dir=args.viz_out_dir, **track_kwargs)

    if boundary_regions_gdf is not None:
        coco_remove_out_of_bound_tracks(coco_dset, boundary_regions_gdf)

    # Measure how long tracking takes
    proc_context.stop()

    out_kwcoco = args.out_kwcoco

    if out_kwcoco is not None:
        coco_dset = coco_dset.reroot(absolute=True, check=False)
        # Add tracking audit data to the kwcoco file
        coco_info = coco_dset.dataset.get('info', [])
        coco_info.append(proc_context.obj)
        coco_dset.fpath = out_kwcoco
        ub.Path(out_kwcoco).parent.ensuredir()
        print(f'write to coco_dset.fpath={coco_dset.fpath}')
        coco_dset.dump(out_kwcoco, indent=2)

    # Convert kwcoco to sites
    verbose = 1

    if args.out_sites_dir is not None:

        sites_dir = ub.Path(args.out_sites_dir).ensuredir()
        # Also do this in BAS mode
        sites = convert_kwcoco_to_iarpa(coco_dset,
                                        default_region_id=args.region_id,
                                        as_summary=False)
        print(f'{len(sites)=}')
        # write sites to disk
        site_fpaths = []
        for site in ub.ProgIter(sites, desc='writing sites', verbose=verbose):
            site_props = site['features'][0]['properties']
            assert site_props['type'] == 'site'
            site_fpath = sites_dir / (site_props['site_id'] + '.geojson')
            site_fpaths.append(os.fspath(site_fpath))

            with safer.open(site_fpath, 'w', temp_file=True) as f:
                geojson.dump(site, f, indent=2)

    if args.out_sites_fpath is not None:
        site_tracking_output = tracking_output.copy()
        site_tracking_output['files'] = site_fpaths
        out_sites_fpath = ub.Path(args.out_sites_fpath)
        print(f'Write tracked site result to {out_sites_fpath}')
        with safer.open(out_sites_fpath, 'w', temp_file=True) as file:
            json.dump(site_tracking_output, file, indent='    ')

    # Convert kwcoco to sites summaries
    if args.out_site_summaries_dir is not None:
        sites = convert_kwcoco_to_iarpa(coco_dset,
                                        default_region_id=args.region_id,
                                        as_summary=True)
        print(f'{len(sites)=}')
        site_summary_dir = ub.Path(args.out_site_summaries_dir).ensuredir()
        # write sites to region models on disk
        groups = ub.group_items(sites, lambda site: site['properties'].pop('region_id'))

        site_summary_fpaths = []
        for region_id, site_summaries in groups.items():

            region_fpath = site_summary_dir / (region_id + '.geojson')
            if args.append_mode and region_fpath.is_file():
                with open(region_fpath, 'r') as f:
                    region = geojson.load(f)
                if verbose:
                    print(f'writing to existing region {region_fpath}')
            else:
                region = geojson.FeatureCollection(
                    [create_region_header(region_id, site_summaries)])
                if verbose:
                    print(f'writing to new region {region_fpath}')
            for site_summary in site_summaries:
                assert site_summary['properties']['type'] == 'site_summary'
                region['features'].append(site_summary)

            site_summary_fpaths.append(os.fspath(region_fpath))
            with safer.open(region_fpath, 'w', temp_file=True) as f:
                geojson.dump(region, f, indent=2)

    if args.out_site_summaries_fpath is not None:
        site_summary_tracking_output = tracking_output.copy()
        site_summary_tracking_output['files'] = site_summary_fpaths
        out_site_summaries_fpath = ub.Path(args.out_site_summaries_fpath)
        out_site_summaries_fpath.parent.ensuredir()
        print(f'Write tracked site summary result to {out_site_summaries_fpath}')
        with safer.open(out_site_summaries_fpath, 'w', temp_file=True) as file:
            json.dump(site_summary_tracking_output, file, indent='    ')


def coco_remove_out_of_bound_tracks(coco_dset, boundary_regions_gdf):
    # Remove any tracks that are outside of region bounds.
    # First find which regions correspond to which videos.
    import pandas as pd
    from watch.utils import util_gis
    from shapely.geometry import shape
    from watch.geoannots.geococo_objects import CocoGeoVideo
    import geopandas as gpd
    crs84 = util_gis.get_crs84()
    crs84_parts = []
    for video in coco_dset.videos().objs:
        coco_video = CocoGeoVideo(video=video, dset=coco_dset)
        utm_part = coco_video.wld_corners_gdf
        crs84_part = utm_part.to_crs(crs84)
        crs84_parts.append(crs84_part)
    video_gdf = pd.concat(crs84_parts).reset_index()
    idx1_to_idxs2 = util_gis.geopandas_pairwise_overlaps(video_gdf, boundary_regions_gdf)
    assignments = []
    for idx1, idxs2 in idx1_to_idxs2.items():
        video_gdf.iloc[idx1]['name']

        if len(idxs2) == 0:
            raise AssertionError('no region for video')

        if len(idxs2) > 1:
            video_goem = video_gdf.iloc[idx1]['geometry']
            chosen_id2 = boundary_regions_gdf.iloc[idxs2].geometry.intersection(video_goem).area.idxmax()
            chosen_idx2 = chosen_id2  # can do this bc of reset index
            idxs2 = [chosen_idx2]
            raise NotImplementedError('multiple regions for video, not sure if impl is correct')
        idx2 = idxs2[0]
        video_name = video_gdf.iloc[idx1]['name']
        region_name = boundary_regions_gdf.iloc[idx2]['region_id']
        region_geom = boundary_regions_gdf.iloc[idx2]['geometry']
        assignments.append((video_name, region_name, region_geom))

    # Actually remove the offending annots
    to_remove_trackids = set()
    for assign in assignments:
        video_name, region_name, region_geom = assign
        video_id = coco_dset.index.name_to_video[video_name]['id']
        video_imgs = coco_dset.images(video_id=video_id)
        video_aids = list(ub.flatten(video_imgs.annots))
        video_annots = coco_dset.annots(video_aids)
        annot_geos = gpd.GeoDataFrame([
            {
                'annot_id': obj['id'],
                'geometry': shape(obj['segmentation_geos']),
            } for obj in video_annots.objs],
            columns=['annot_id', 'geometry'], crs=crs84)

        is_oob = annot_geos.intersection(region_geom).is_empty
        inbound_annots = video_annots.compress(~is_oob)
        outofbounds_annots = video_annots.compress(is_oob)

        # Only remove tracks that are always out of bounds.
        inbound_tracks = set(inbound_annots.lookup('track_id'))
        outofbound_tracks = set(outofbounds_annots.lookup('track_id'))
        always_outofbound_tracks = outofbound_tracks - inbound_tracks
        to_remove_trackids.update(always_outofbound_tracks)

    to_remove_aids = []
    for tid in to_remove_trackids:
        to_remove_aids.extend(list(coco_dset.annots(track_id=tid)))

    if to_remove_aids:
        print(f'Removing {len(to_remove_trackids)} out-of-bounds tracks '
              f'with {len(to_remove_aids)} annotations')
    else:
        print('All annotations are in bounds')

    coco_dset.remove_annotations(to_remove_aids)


def demo(coco_dset, regions_dir, coco_dset_sc, sites_dir, cleanup=True):
    import json
    from tempfile import NamedTemporaryFile
    bas_args = [
        coco_dset.fpath,
        '--out_site_summaries_dir',
        regions_dir,
        '--track_fn',
        'watch.tasks.tracking.from_heatmap.TimeAggregatedBAS',
    ]
    # run BAS on it
    main(bas_args)
    # reload it with tracks
    # coco_dset = kwcoco.CocoDataset(coco_dset.fpath)
    # run SC on both of them
    sc_args = [
        '--out_site_sites_dir',
        sites_dir,
        '--track_fn',
        'watch.tasks.tracking.from_heatmap.TimeAggregatedSC',
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
            main([
                sub_dset.fpath, '--track_kwargs',
                '{"boundaries_as": "none"}'
            ] + sc_args)
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
    main()
