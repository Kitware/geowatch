#!/usr/bin/env python3
r"""
This file contains logic to convert a kwcoco file into an IARPA Site Model.

At a glance the IARPA Site Model is a GeoJSON FeatureCollection with the
following informal schema:

For official documentation about the KWCOCO json format see [kwcoco]_. A formal
json-schema can be found in ``kwcoco.coco_schema``

For official documentation about the IARPA json format see [2, 3]_. A formal
json-schema can be found in ``../../geowatch/rc/site-model.schema.json``.

References:
    .. [kwcoco] https://gitlab.kitware.com/computer-vision/kwcoco
    .. [2] https://infrastructure.smartgitlab.com/docs/pages/api/
    .. [3] https://smartgitlab.com/TE/annotations

SeeAlso:
    * ../tasks/tracking/from_heatmap.py
    * ../tasks/tracking/old_polygon_extraction.py
    * ../tasks/tracking/polygon_extraction.py
    * ../tasks/tracking/utils.py
    * ../../tests/test_tracker.py

Ignore:
    python -m geowatch.cli.run_tracker \
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

        geowatch visualize /data/joncrall/dvc-repos/smart_expt_dvc/_debug/metrics/bas-fusion/bas_fusion_kwcoco_tracked3.json --smart
"""
import os
import scriptconfig as scfg
import ubelt as ub

if not os.environ.get('_ARGCOMPLETE', ''):
    from geowatch.tasks.tracking import from_heatmap, from_polygon

    _KNOWN_TRACK_FUNCS = {
        'saliency_heatmaps': from_heatmap.TimeAggregatedBAS,
        'saliency_polys': from_polygon.OverlapTrack,
        'class_heatmaps': from_heatmap.TimeAggregatedSC,
        'site_validation': from_heatmap.TimeAggregatedSV,
        'class_polys': from_polygon.OverlapTrack,
        'mono_track': from_polygon.MonoTrack,
    }

    _trackfn_details_docs = ' --- '.join([
        # k + ': ' + ', '.join([field.name for field in v.__dataclass_fields__.values()])
        # if hasattr(v, '__dataclass_fields__') else
        k + ': ' + ', '.join([k for k, v in v.__default__.items()])
        if hasattr(v, '__default__') else
        k + ':?'
        for k, v in _KNOWN_TRACK_FUNCS.items()
    ])
else:
    _trackfn_details_docs = 'na'


try:
    from xdev import profile
except Exception:
    profile = ub.identity


USE_NEW_KWCOCO_TRACKS = False


class KWCocoToGeoJSONConfig(scfg.DataConfig):
    """
    Convert KWCOCO to IARPA GeoJSON
    """

    # TODO: can we store metadata in the type annotation?
    # SeeAlso: notes in from_heatmap.__devnote__
    # e.g.
    #
    # in_file : scfg.Coercible(help='Input KWCOCO to convert', position=1) = None
    # in_file : scfg.Type(help='Input KWCOCO to convert', position=1) = None
    # in_file : scfg.PathLike(help='Input KWCOCO to convert', position=1) = None
    #
    # or
    #
    # in_file = None
    # in_file : scfg.Value[help='Input KWCOCO to convert', position=1]

    in_file = scfg.Value(None, required=True, help='Input KWCOCO to convert',
                         position=1, alias=['input_kwcoco'])

    out_kwcoco = scfg.Value(None, help=ub.paragraph(
            '''
            The file path to write the "tracked" kwcoco file to.
            '''))

    out_sites_dir = scfg.Value(None, help=ub.paragraph(
        '''
        The directory where site model geojson files will be written.
        '''), alias=['output_sites_dpath'])

    out_site_summaries_dir = scfg.Value(None, help=ub.paragraph(
        '''
        The directory path where site summary geojson files will be written.
        '''))

    out_sites_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site models will be written.
        '''), alias=['output_site_manifest_fpath'])

    out_site_summaries_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site summary geojson files will
        be written.
        '''))

    in_file_gt = scfg.Value(None, help=ub.paragraph(
            '''
            If available, ground truth KWCOCO to visualize.
            DEPRECATED.
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
            'geowatch.tasks.tracking.from_heatmap.TimeAggregatedBAS'
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

            IN THE FUTURE THESE MEMBERS WILL LIKELY BECOME FLAT CONFIG OPTIONS.

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
            '''), alias=['in_site_summaries'], group='behavior')

    clear_annots = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Clears all annotations before running tracking, so it starts
            from a clean slate.
            TODO: perhaps name to something that takes the value overwrite or
            append?
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

    sensor_warnings = scfg.Value(True, help='if False, disable sensor warnings')

    time_pad_before = scfg.Value(None, help='A time delta to extend start times')
    time_pad_after = scfg.Value(None, help='A time delta to extend end times')

    #### New Eval 16 params

    smoothing = scfg.Value(None, help=ub.paragraph(
        '''
        if True specify a number between 0 and 1 to smooth the observation
        scores over time.
        '''))

    site_score_thresh = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, then the final a site will be rejected if its site score
        is less than this threshold.
        NOTE: In the future we should separate the steps that assign scores to
        polygons / define polygons and those that postprocess / threshold them
        into another stage. For now we are putting it here.
        '''))


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
    from kwutil import util_time
    return util_time.coerce_datetime(date_str).date().isoformat()


def _join_props(parts):
    # T&E requires a comma and a space
    return ', '.join(parts)


def _split_props(parts):
    # Handle comma only and comma space cases
    return [p.strip() for p in parts.split(',')]


# For MultiPolygon observations. Could be ', '?
# sep = ','


@profile
def coco_create_observation(coco_dset, anns):
    """
    Group kwcoco annotations in the same track (site) and image into one
    Feature in an IARPA site model
    """
    # import geojson
    import kwcoco
    import shapely
    import numpy as np
    from collections import defaultdict
    from geowatch.geoannots import geomodels

    def single_geometry(ann):
        seg_geo = ann['segmentation_geos']
        assert isinstance(seg_geo, dict)
        return _single_geometry(seg_geo)

    @profile
    def _per_image_properties(coco_img: kwcoco.CocoImage):
        """
        Properties defined per-img instead of per-ann, to reduce duplicate
        computation.
        """
        # pick the image that is actually copied to the metrics framework
        # the source field is implied to be a STAC id, but overload it to
        # enable viz during scoring without referring back to the kwcoco file
        # TODO maybe use cache for this instead when STAC id is
        # properly passed through to TA-2?
        source = None
        img = coco_img.img
        bundle_dpath = ub.Path(coco_img.bundle_dpath)
        chan_to_asset = {
            aux['channels']: aux
            for aux in coco_img.iter_asset_objs()
        }
        _candidate_source_chans = {
            'r|g|b', 'rgb', 'pan', 'panchromatic', 'green', 'blue'
        }
        for want_chan in _candidate_source_chans:
            if want_chan in chan_to_asset:
                aux = chan_to_asset[want_chan]
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

        # Give each observation a total score
        score = ann.get('score', 1.0)
        score = max(min(score, 1.0), 0.0)

        # Remember scores for each class
        raw_multi_scores = ann.get('scores', {})

        # Give each obs a status. This lets us decouple kwcoco in existing
        # downstream logic. Might not be needed in the future.
        status = ann.get('status', 'system_confirmed')

        return {
            'type': 'observation',
            'current_phase': current_phase,
            'is_site_boundary': True,  # HACK
            'score': score,
            'cache': {
                'phase_transition_days': ann.get('phase_transition_days', None),
                'raw_multi_scores': raw_multi_scores,
                'status': status
            },
            **image_properties_dct[ann['image_id']]
        }

    def combined_properties(properties_list, geometry_list):
        # list of dicts -> dict of lists for easy indexing
        properties_dcts = {
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
            for prop, geom in zip(properties_dcts[key], geometry_list):
                value.append(_join_props([str(prop)] * _len(geom)))
            properties[key] = _join_props(value)

        # HACK
        # We are not being scored on multipolygons in SC right now!
        # https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/24
        #
        # When safe, merge class labels for multipolygons so they'll be scored.
        if 0:
            phase = _split_props(properties['current_phase'])
            if len(phase) > 1 and len(set(phase)) == 1:
                properties['current_phase'] = phase[0]

        # identical properties
        for key in ['type', 'source', 'observation_date', 'sensor_name']:
            values = properties_dcts[key]
            assert len(set(values)) == 1
            properties[key] = str(values[0])

        # take area-weighted average score
        weights = np.array([geom.area for geom in geometry_list])
        scores = np.array([float(s) for s in properties_dcts['score']])
        mask = np.isnan(scores)
        scores[mask] = 0
        weights[mask] = 0
        if weights.sum() == 0:
            properties['score'] = 0
        else:
            properties['score'] = np.average(scores, weights=weights)

        # Make cache a list of entries for each polygon.
        # This is for the case of more than one polygon beloning to the
        # observation, which I'm not even sure if that makes sense.
        properties['cache'] = defaultdict(list)
        for cache in properties_dcts['cache']:
            for k, v in cache.items():
                properties['cache'][k].append(v)

        return properties

    image_properties_dct = {}
    gids = {ann['image_id'] for ann in anns}
    for gid in gids:
        coco_img = coco_dset.coco_image(gid).detach()
        image_properties_dct[gid] = _per_image_properties(coco_img)

    geometry_list = list(map(single_geometry, anns))

    properties_list = list(map(single_properties, anns))
    geometry = _ensure_multi(_combined_geometries(geometry_list))
    properties = combined_properties(properties_list, geometry_list)

    observation = geomodels.Observation(geometry=geometry, properties=properties)
    return observation


def predict_phase_changes(site_id, observations):
    """
    Set predicted_phase_transition and predicted_phase_transition_date.

    This should only kick in when the site does not end before the current
    day (latest available image). See tracking.normalize.normalize_phases
    for what happens if the site has ended.

    Args:
        site_id (str): site identifier
        features (List[Dict]): observation feature dictionaries for the site

    Returns:
        dict

    References:
        https://smartgitlab.com/TE/standards/-/wikis/Site-Model-Specification
        https://gitlab.kitware.com/smart/standards-wiki/-/blob/main/Site-Model-Specification.md

    Example:
        >>> from geowatch.geoannots import geomodels
        >>> site = geomodels.SiteModel.random(rng=0, num_observations=20)
        >>> site_id = site.site_id
        >>> observations = list(site.body_features())
        >>> observations[-1]['properties']['cache'] = {'phase_transition_days': [100]}
        >>> predict_phase_changes(site_id, observations)
    """
    import datetime as datetime_mod
    from kwutil import util_time

    feature_properties = [feat['properties'] for feat in observations]

    # Ensure features are in temporal order
    # (they probably are already, but we are being safe)
    feature_properties = sorted(
        feature_properties,
        key=lambda prop: util_time.coerce_datetime(prop['observation_date']))

    feature_phases = [_split_props(prop['current_phase']) for prop in feature_properties]

    def transition_date_from(phase):
        for props, phases in zip(reversed(feature_properties), reversed(feature_phases)):
            if phase in phases:
                _idx = phases.index(phase)
                cache = props.get('cache', {})
                obs_date = util_time.coerce_datetime(props['observation_date'])
                days = cache['phase_transition_days'][_idx]
                pred_delta = datetime_mod.timedelta(days=int(days))
                pred_datetime = (obs_date + pred_delta)
                pred_date = pred_datetime.date()
                return pred_date.isoformat()
        print(f'warning: {site_id=} is missing {phase=}')
        final_date = util_time.coerce_datetime(
            feature_properties[-1]['observation_date'])
        tomorrow = final_date + datetime_mod.timedelta(days=1)
        tomorrow = tomorrow.date()
        return tomorrow.isoformat()

    final_phase = feature_phases[-1]

    if 'Post Construction' in final_phase:
        return {}
    elif 'Active Construction' in final_phase:
        return {
            'predicted_phase_transition': 'Post Construction',
            'predicted_phase_transition_date': transition_date_from('Active Construction')
        }
    elif 'Site Preparation' in final_phase:
        return {
            'predicted_phase_transition': 'Active Construction',
            'predicted_phase_transition_date': transition_date_from('Site Preparation')
        }
    else:
        # raise ValueError(f'missing phases: {site_id=} {all_phases_set=}')
        print(f'missing phases: {site_id=} {final_phase=}')
        return {}


def smooth_observation_scores(observations, smoothing=0.5, smooth_mode='ewma'):
    """
    Add smoothed scores inplace

    Example:
        >>> from geowatch.cli.run_tracker import *  # NOQA
        >>> from geowatch.geoannots import geomodels
        >>> site = geomodels.SiteModel.random(num_observations=15)
        >>> observations = list(site.observations())
        >>> # Add random scores for tests
        >>> import kwarray
        >>> rng = kwarray.ensure_rng()
        >>> for obs in observations:
        >>>     obs['properties']['cache'] = {'raw_multi_scores': [{
        >>>         'No Activity': rng.rand(),
        >>>         'Site Preparation': rng.rand(),
        >>>         'Post Construction': rng.rand(),
        >>>         'Active Construction': rng.rand(),
        >>> }]}
        >>> data1 = [obs['properties']['cache']['raw_multi_scores'][0] for obs in observations]
        >>> smooth_observation_scores(observations, smooth_mode='ewma')
        >>> data2 = [obs['properties']['cache']['smooth_scores'].copy() for obs in observations]
        >>> smooth_observation_scores(observations, smooth_mode='conv3')
        >>> data3 = [obs['properties']['cache']['smooth_scores'].copy() for obs in observations]
        >>> import pandas as pd
        >>> df1 = pd.DataFrame(data1)
        >>> df2 = pd.DataFrame(data2)
        >>> df3 = pd.DataFrame(data3)
        >>> print(df1)
        >>> print(df2)
        >>> print(df3)
    """
    import kwarray
    # Consolidate scores over time. Predict start / end dates.
    obs_multi_scores = [obs['properties']['cache']['raw_multi_scores'] for obs in observations]
    assert all(len(ms) == 1 for ms in obs_multi_scores)
    obs_scores = [ms[0] for ms in obs_multi_scores]
    score_df = kwarray.DataFrameLight.from_dict(obs_scores)
    score_df = score_df.pandas()

    if smooth_mode == 'ewma':
        alpha = (1 - smoothing)
        smoothed = score_df
        smoothed = smoothed.ewm(alpha=alpha, axis=0).mean()
    elif smooth_mode == 'conv3':
        # This didn't do that well in tests, we may be able to remove it.
        import scipy.ndimage
        import numpy as np
        kernel = np.empty(3)
        middle = 2 * (1 - smoothing) / 3 + (1 / 3)
        side = (1 - middle) / 2
        kernel[1] = middle
        kernel[0] = side
        kernel[2] = side
        new_values = scipy.ndimage.convolve1d(score_df.values, kernel, axis=1, mode='constant', cval=0)
        smoothed = score_df.copy()
        smoothed.values[:] = new_values
    else:
        raise KeyError(smooth_mode)

    # HACK: Our score dict mixes scores of different types, but we need to just
    # predict activity phase labels here.
    if 'ac_salient' in smoothed.columns:
        phase_candidates = smoothed.drop(['ac_salient'], axis=1)
    else:
        phase_candidates = smoothed

    new_labels = phase_candidates.idxmax(axis=1).values.tolist()
    # old_labels = [o['properties']['current_phase'] for o in observations]

    new_scores = smoothed.to_dict('records')
    for obs, scores, label in zip(observations, new_scores, new_labels):
        obs['properties']['current_phase'] = label
        obs['properties']['cache']['smooth_scores'] = scores

    if 0:
        from geowatch.tasks.tracking import phase
        new_labels = phase.class_label_smoothing(
            new_labels, transition_probs='v7', emission_probs='v7')

    # phase.REGISTERED_EMMISSION_PROBS['default']
    # phase.REGISTERED_TRANSITION_PROBS['default']
    # phase.viterbi(new_labels, )


def classify_site(site, config):
    """
    Modify a site inplace with classifications.

    Given a site with extracted and scored observations, postprocess the raw
    observation scores and make site-level predictions.
    """
    import numpy as np

    site_header = site.header
    observations = list(site.observations())

    if config.smoothing is not None and config.smoothing > 0:
        smooth_observation_scores(observations, smoothing=config.smoothing)

    ACTIVE_LABELS = {'Site Preparation', 'Active Construction'}

    # Score each observation
    per_obs_scores = []
    for obs in observations:
        obs_cache = obs['properties']['cache']
        if 'smooth_scores' in obs_cache:
            obs_scores = obs_cache['smooth_scores']
        elif 'raw_multi_scores' in obs_cache:
            obs_scores = obs_cache['raw_multi_scores'][0]
        else:
            # fallback to hard classifications
            obs_scores = {obs['properties']['current_phase']: 1.0}

        if 'ac_salient' in obs_scores:
            # Hard code to use "ac_salient" score if available.
            obs_score = obs_scores['ac_salient']
        else:
            active_scores = ub.udict(obs_scores) & ACTIVE_LABELS
            if active_scores:
                obs_score = max(active_scores.values())
            else:
                # final fallback
                obs_score = obs['properties'].get('score', 1.0)
        per_obs_scores.append(obs_score)

    # Site score is the maximum observation score.
    site_score = site_header['properties'].get('score', 1.0)
    status = site_header['properties'].get('status', 'system_confirmed')
    start_date = site_header['properties']['start_date']
    end_date = site_header['properties']['end_date']
    site_header['properties'].setdefault('cache', {})

    if per_obs_scores:
        site_score = max(per_obs_scores)

    FILTER_INACTIVE_SITES = True
    if FILTER_INACTIVE_SITES:
        # HACKS FOR EVAL 15 to get things done quicky.
        curr_labels = [o['properties']['current_phase'] for o in observations]
        flags = [lbl in ACTIVE_LABELS for lbl in curr_labels]
        active_idxs = np.where(flags)[0]
        if len(active_idxs):
            first_idx = active_idxs[0]
            last_idx = active_idxs[-1]
            start_date = observations[first_idx].observation_date.date().isoformat()
            end_date = observations[last_idx].observation_date.date().isoformat()
        else:
            # Check to make sure we are in AC/SC. Otherwise use old logic
            if any(c in {'No Activity', 'Post Construction'} for c in curr_labels):
                # Throw out items with no start / end
                status = 'system_rejected'
                site_header['properties']['cache']['reject_reason'] = 'no_active_observations'

    if config.site_score_thresh is not None:
        # Reject sites if we have a threshold here.
        if site_score < config.site_score_thresh:
            status = 'system_rejected'
            site_header['properties']['cache']['reject_reason'] = 'failed_site_score_thresh'

    # Hack to modify time bounds
    import kwutil
    if config.time_pad_before is not None:
        delta_before = kwutil.util_time.timedelta.coerce(config.time_pad_before)
        start_date = (kwutil.util_time.datetime.coerce(start_date) - delta_before).date().isoformat()
    if config.time_pad_after is not None:
        delta_after = kwutil.util_time.timedelta.coerce(config.time_pad_after)
        end_date = (kwutil.util_time.datetime.coerce(end_date) + delta_after).date().isoformat()

    site_header['properties'].update({
        'score': site_score,
        'status': status,
        'start_date': start_date,
        'end_date': end_date,
    })

    site_id = site.site_id
    pred_transition = predict_phase_changes(site_id, observations)
    site_header['properties'].update(pred_transition)


def coco_create_site_header(region_id, site_id, trackid, observations):
    """
    Feature containing metadata about the site

    Returns:
        geomodels.SiteSummary | geomodels.SiteHeader
    """
    import shapely
    import geowatch
    from geowatch.geoannots import geomodels

    geom_list = [_single_geometry(feat['geometry']) for feat in observations]
    geometry = _combined_geometries(geom_list)

    # site and site_summary features must be polygons
    if isinstance(geometry, shapely.geometry.MultiPolygon):
        if len(geometry.geoms) == 1:
            geometry = geometry.geoms[0]
        else:
            print(f'warning: {region_id=} {site_id=} {trackid=} has multi-part site geometry')
            geometry = geometry.convex_hull

    # these are strings, but sorting should be correct in isoformat

    status_set = [obs['properties']['cache'].get('status', ['system_confirmed'])
                  for obs in observations]
    status_set = set(ub.flatten(status_set))

    # https://smartgitlab.com/TE/annotations/-/wikis/Annotation-Status-Types#for-site-models-generated-by-performersalgorithms
    # system_confirmed, system_rejected, or system_proposed
    # TODO system_proposed pre val-net
    assert len(status_set) == 1, f'inconsistent {status_set=} for {trackid=}'
    status = status_set.pop()

    dates = sorted([obs.observation_date for obs in observations])
    start_date = dates[0].date().isoformat()
    end_date = dates[-1].date().isoformat()

    PERFORMER_ID = 'kit'

    site_header = geomodels.SiteHeader.empty()
    site_header['geometry'] = geometry
    site_header['properties'].update({
        'site_id': site_id,
        'region_id': region_id,

        'version': geowatch.__version__,  # Shouldn't this be a schema version?

        'status': status,
        'model_content': 'proposed',
        'score': 1.0,  # TODO does this matter?

        'start_date': start_date,
        'end_date': end_date,

        'originator': PERFORMER_ID,
        'validated': 'False'
    })
    site_header.infer_mgrs()
    return site_header


@profile
def convert_kwcoco_to_iarpa(coco_dset, default_region_id=None):
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
        >>> import geowatch
        >>> from geowatch.cli.run_tracker import *  # NOQA
        >>> from geowatch.tasks.tracking.normalize import run_tracking_pipeline
        >>> from geowatch.tasks.tracking.from_polygon import MonoTrack
        >>> import ubelt as ub
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi', heatmap=True, geodata=True, dates=True)
        >>> coco_dset = run_tracking_pipeline(
        >>>     coco_dset, track_fn=MonoTrack, overwrite=False,
        >>>     sensor_warnings=False)
        >>> videos = coco_dset.videos()
        >>> videos.set('name', ['DM_R{:03d}'.format(vidid) for vidid in videos])
        >>> sites = convert_kwcoco_to_iarpa(coco_dset)
        >>> print(f'{len(sites)} sites')
        >>> if 0:  # validation fails
        >>>     import jsonschema
        >>>     SITE_SCHEMA = geowatch.rc.load_site_model_schema()
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
    import itertools as it
    counter = it.count()
    sites = []
    for vidid, video in coco_dset.index.videos.items():
        region_id = video.get('name', default_region_id)
        gids = coco_dset.index.vidid_to_gids[vidid]
        region_annots = coco_dset.images(gids).annots
        region_track_ids = sorted(set(ub.flatten(region_annots.lookup('track_id'))))

        # SUPER HACK:
        # There isn't a good mechanism to get the region id,
        # This is fragile, but currently works. Need better mechanism.
        if '_CLUSTER' in region_id:
            region_id = region_id.split('_CLUSTER')[0]

        for site_idx, trackid in zip(counter, region_track_ids):
            site = coco_track_to_site(coco_dset, trackid, region_id, site_idx)
            sites.append(site)

    return sites


@profile
def coco_track_to_site(coco_dset, trackid, region_id, site_idx=None):
    """
    Turn a kwcoco track into an IARPA site model or site summary
    """
    # import geojson
    from geowatch.geoannots.geomodels import SiteModel
    import geowatch

    # get annotations in this track sorted by frame_index
    annots = coco_dset.annots(track_id=trackid)
    gids, anns = annots.gids, annots.objs

    observations = []
    for gid, _anns in ub.group_items(anns, gids).items():
        feat = coco_create_observation(coco_dset, _anns)
        observations.append(feat)
    # for i in range(len(observations)):
    #     observations[i]['properties'].setdefault('cache', {})
    #     observations[i]['properties']['cache']['trackid'] = trackid

    # HACK to passthrough site_summary IDs
    if geowatch.tasks.tracking.utils.trackid_is_default(trackid):
        if site_idx is None:
            site_idx = trackid
        site_id = '_'.join((region_id, str(site_idx).zfill(4)))
    else:
        # HACK:
        # TODO make more robust
        site_id = trackid
        region_id = '_'.join(site_id.split('_')[:2])

    site_header = coco_create_site_header(region_id, site_id, trackid, observations)
    site = SiteModel([site_header] + observations)

    return site


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
    from geowatch.utils import util_gis
    from geowatch.geoannots import geomodels
    import jsonschema

    TRUST_REGION_SCHEMA = 0

    geojson_infos = list(util_gis.coerce_geojson_datas(
        site_summary_or_region_model, format='json', allow_raw=True))

    # validate the json
    site_summaries = []

    # # debug mode is for comparing against a set of known GT site models
    # DEBUG_MODE = 1
    # if DEBUG_MODE:
    #     SITE_SUMMARY_POS_STATUS = {
    #         'positive_annotated',
    #         'system_proposed', 'system_confirmed',
    #         'positive_annotated_static',  # TODO confirm
    #         'negative',
    #     }
    # else:
    #     # TODO handle positive_partial
    #     SITE_SUMMARY_POS_STATUS = {
    #         'positive_annotated',
    #         'system_proposed', 'system_confirmed',
    #     }

    for info in geojson_infos:
        data = info['data']

        if not isinstance(data, dict):
            raise AssertionError(
                f'unknown site summary {type(data)=}'
            )

        try:  # is this a region model?

            region_model = geomodels.RegionModel(**data)

            if TRUST_REGION_SCHEMA:
                region_model.fixup()
                region_model.validate(strict=False)

            region_model._validate_quick_checks()

            _summaries = [
                f for f in region_model.site_summaries()
                if f['properties']['status'] not in {'system_rejected'}
            ]
            # region_id = region_model.region_id
            # TODO: handle default region-id if needed

            site_summaries.extend(_summaries)

        except jsonschema.ValidationError:
            # In this case we expect the input to be a list of site summaries.
            # However, we really shouldn't hit this case.
            raise AssertionError(
                'Jon thinks we wont hit this case. '
                'If you see this error, he is wrong and the error can be removed. '
                'Otherwise we should remove this extra code')
            site_summary = site_summary_or_region_model
            # site_summary_tups.append((default_region_id, site_summary))
            site_summaries.append(site_summary)

    return site_summaries


def assign_sites_to_videos(coco_dset, site_summaries):
    """
    Compute assignments between which sites summaries should be projected onto
    which videos for scoring.
    """
    from geowatch.utils import util_gis
    from geowatch.geoannots.geomodels import RegionModel
    import rich
    from geowatch.utils import kwcoco_extensions

    site_idx_to_vidid = []
    unassigned_site_idxs = []

    # Get a GeoDataFrame with geometry for each coco video and each site.
    video_gdf = kwcoco_extensions.covered_video_geo_regions(coco_dset)

    sitesum_model = RegionModel(features=site_summaries)
    sitesum_gdf = sitesum_model.pandas_summaries()

    # Compute which sites overlap which videos
    site_idx_to_video_idx = util_gis.geopandas_pairwise_overlaps(sitesum_gdf, video_gdf)

    # For each site, chose a single video assign to
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
        site_idx_to_vidid.append((site_idx, video_id))

    assigned_vidids = set([t[1] for t in site_idx_to_vidid])
    n_unassigned_sites = len(unassigned_site_idxs)
    n_assigned_sites = len(site_idx_to_vidid)
    n_total_sites = len(site_summaries)
    n_assigned_vids = len(assigned_vidids)
    n_total_vids = coco_dset.n_videos

    color, punc = '', '.'
    if unassigned_site_idxs:
        color, punc = '[yellow]', '!'
    rich.print(f'{color}There were {n_unassigned_sites} sites that has no video overlaps{punc}')
    rich.print(f'There were {n_assigned_sites} / {n_total_sites} assigned site summaries')
    rich.print(f'There were {n_assigned_vids} / {n_total_vids} assigned videos')
    return site_idx_to_vidid


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
    import rich
    import geowatch
    from geowatch.utils import kwcoco_extensions
    from kwutil import util_time
    import kwcoco
    # input validation
    print(f'possible_summaries={possible_summaries}')
    print(f'coco_dset={coco_dset}')
    print(f'default_region_id={default_region_id}')

    if default_region_id is None:
        default_region_id = ub.peek(coco_dset.index.name_to_video)

    # TODO: maintain system-rejected sites?
    site_summary_or_region_model = possible_summaries
    site_summaries = _coerce_site_summaries(
        site_summary_or_region_model, default_region_id)
    print(f'found {len(site_summaries)} site summaries')

    site_summary_cid = coco_dset.ensure_category(geowatch.heuristics.SITE_SUMMARY_CNAME)

    print('Searching for assignment between requested site summaries and the kwcoco videos')

    # TODO: Should likely refactor to unify this codepath with reproject
    # annotations.

    # FIXME: need to include a temporal component to this assignment.
    # Also, should probably do this in UTM instead of CRS84

    # Compute Assignment between site summaries / coco videos.
    site_idx_to_vidid = assign_sites_to_videos(coco_dset, site_summaries)

    print('warping site boundaries to pxl space...')

    for site_idx, video_id in site_idx_to_vidid:

        site_summary = site_summaries[site_idx]
        site_id = site_summary['properties']['site_id']

        # get relevant images
        images = coco_dset.images(video_id=video_id)

        # and their dates
        image_dates = [util_time.coerce_datetime(d).date()
                       for d in images.lookup('date_captured')]

        assert image_dates == sorted(image_dates), 'images are not in order'
        first_date = image_dates[0]
        last_date = image_dates[-1]

        start_dt = site_summary.start_date
        end_dt = site_summary.end_date
        if start_dt is None:
            if end_dt is not None and end_dt.date() < first_date:
                rich.print(f'[yellow]warning: end_dt before first_date! end_dt={end_dt.date()} first_date={first_date}')
                continue
            start_date = first_date
        else:
            start_date = start_dt.date()
        if end_dt is None:
            if start_date > last_date:
                rich.print(f'[yellow]warning: start_date after last_date! start_date={start_date} last_date={last_date}')
                continue
            end_date = last_date
        else:
            end_date = end_dt.date()

        if end_date < start_date:
            raise AssertionError(ub.codeblock(
                f'''
                warning: end_date before start_date!
                site_summary={site_summary}
                start_date={start_date}
                end_date={start_date}
                ''')
            )

        flags = [start_date <= d <= end_date for d in image_dates]
        images = images.compress(flags)

        if USE_NEW_KWCOCO_TRACKS:
            try:
                track_id = coco_dset.add_track(name=site_id)
            except kwcoco.exceptions.DuplicateAddError:
                rich.print(f'[yellow]warning: site_summary {site_id} already in dset!')
                raise
        else:
            track_id = site_id
            # Seems broken
            if track_id in images.get('track_id', None):
                rich.print(f'[yellow]warning: site_summary {track_id} already in dset!')

        # apply site boundary as polygons
        poly_crs84_geojson = site_summary['geometry']
        for img in images.objs:
            # Add annotations in CRS84 geo-space, we will project to pixel
            # space in a later step
            coco_dset.add_annotation(
                image_id=img['id'],
                category_id=site_summary_cid,
                segmentation_geos=poly_crs84_geojson,
                track_id=track_id
            )

    print('Projecting regions to pixel coords')
    kwcoco_extensions.warp_annot_segmentations_from_geos(coco_dset)
    rich.print(f'[green]Done projecting: {coco_dset.n_annots} annotations')
    return coco_dset


@profile
def main(argv=None, **kwargs):
    """
    Example:
        >>> # test BAS and default (SC) modes
        >>> from geowatch.cli.run_tracker import *  # NOQA
        >>> from geowatch.cli.run_tracker import main
        >>> from geowatch.demo import smart_kwcoco_demodata
        >>> from geowatch.utils import util_gis
        >>> import json
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # run BAS on demodata in a new place
        >>> import geowatch
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi', heatmap=True, geodata=True, dates=True)
        >>> dpath = ub.Path.appdir('geowatch', 'test', 'tracking', 'main0').ensuredir()
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
        >>>     '--sensor_warnings', 'False',
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
        >>>     '--sensor_warnings', 'False',
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
        >>> print('bas_trackids = {}'.format(ub.urepr(bas_trackids, nl=1)))
        >>> print('sc_trackids = {}'.format(ub.urepr(sc_trackids, nl=1)))
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
        >>> from geowatch.cli.run_tracker import *  # NOQA
        >>> from geowatch.cli.run_tracker import main
        >>> import geowatch
        >>> dset = geowatch.coerce_kwcoco('geowatch-msi', heatmap=True, geodata=True, dates=True)
        >>> dpath = ub.Path.appdir('geowatch', 'test', 'tracking', 'main1').ensuredir()
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
        >>>     'sensor_warnings': False,
        >>> }
        >>> argv = []
        >>> # Test case for no results
        >>> main(argv=argv, **kwargs)
        >>> from geowatch.utils import util_gis
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
        >>>     'sensor_warnings': False,
        >>> }
        >>> argv = []
        >>> main(argv=argv, **kwargs)
        >>> assert len(list(util_gis.coerce_geojson_datas(bas_fpath))) > 0

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # test a more complicated track function
        >>> import geowatch
        >>> from geowatch.cli.run_tracker import demo
        >>> import kwcoco
        >>> import geowatch
        >>> import ubelt as ub
        >>> # make a new BAS dataset
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi', heatmap=True, geodata=True)
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
    cmdline = True
    if isinstance(argv, list) and argv == []:
        # Workaround an issue in scriptconfig tha twill be fixed in 0.7.8
        # after that, remove cmdline completely
        cmdline = False
    args = KWCocoToGeoJSONConfig.cli(cmdline=cmdline, argv=argv, data=kwargs, strict=True)
    import rich
    rich.print('args = {}'.format(ub.urepr(args, nl=1)))

    import geojson
    import json
    import kwcoco
    import pandas as pd
    import safer
    import geowatch

    from kwcoco.util import util_json
    from kwutil.util_yaml import Yaml
    from geowatch.geoannots import geomodels
    from geowatch.utils import process_context
    from geowatch.utils import util_gis

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

    PROCESSCONTEXT_INFO_HACK = True
    proc_context_kwargs = {}
    if PROCESSCONTEXT_INFO_HACK:
        # TODO: now that we are no longer doing subsets and unions in the tracker
        # we dont need to maintain the old "info" here. In fact if we do, we get
        # circular references. In order to maintain compatability with older output
        # we are going to clear the old info and continue to use extra pred info,
        # but in the future after we are sure nothing depends on this, we should
        # simply remove the extra kwarg, as it will be implicitly maintained.
        proc_context_kwargs['extra'] = {'pred_info': pred_info}
        # Clear out the old info to prevent circular references while we do the
        # above hack. Remove the next line once we fix the above hack.
        coco_dset.dataset['info'] = []

    proc_context = process_context.ProcessContext(
        name='geowatch.cli.run_tracker', type='process',
        config=jsonified_config,
        track_emissions=False,
        **proc_context_kwargs,
    )

    proc_context.start()
    info.append(proc_context.obj)

    # Pick a track_fn
    # HACK remove potentially conflicting annotations as well
    # we shouldn't have saliency annots when we want class or vice versa
    CLEAN_DSET = 1
    class_cats = [cat['name'] for cat in geowatch.heuristics.CATEGORIES]
    saliency_cats = ['salient']

    track_fn = args.track_fn
    if track_fn is None:
        track_fn = (
            geowatch.tasks.tracking.abstract_classes.NoOpTrackFunction
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
        region_infos = list(util_gis.coerce_geojson_datas(
            args.boundary_region, format='json', allow_raw=True,
            desc='load boundary regions'))
        region_parts = []
        for info in region_infos:
            # Need to deterimine which one to use
            region_model = geomodels.RegionModel(**info['data'])
            region_gdf = region_model.pandas_region()
            region_parts.append(region_gdf)
        boundary_regions_gdf = pd.concat(region_parts).reset_index()
        video_gdf = coco_video_gdf(coco_dset)
        video_region_assignments = assign_videos_to_regions(video_gdf, boundary_regions_gdf)
    else:
        if args.site_summary is None:
            rich.print('[yellow]Warning: No boundary regions or site summaries specified')
        else:
            rich.print('No boundary regions specified')
        boundary_regions_gdf = None
        video_region_assignments = None

    # add site summaries (site boundary annotations)
    if args.site_summary is not None:

        coco_dset = add_site_summary_to_kwcoco(args.site_summary, coco_dset,
                                               args.region_id)
        if 0:
            # Going to test to see if removing this helps anything
            cid = coco_dset.name_to_cat[geowatch.heuristics.SITE_SUMMARY_CNAME]['id']
            coco_dset = coco_dset.subset(coco_dset.index.cid_to_gids[cid])
            print('restricting dset to videos with site_summary annots: ',
                  set(coco_dset.index.name_to_video))
            assert coco_dset.n_images > 0, 'no valid videos!'
    else:
        # Note: this may be a warning if we use this in a pipeline where the
        # inputed kwcoco files already have polygons reprojected (and in that
        # case if there are legidimately no sites to score!) But in our main
        # use case where site_summary should be specified, this is an error so
        # we are treating it that way for now.
        if track_kwargs.get('boundaries_as') == 'polys' and not coco_dset.n_annots:
            raise Exception(ub.codeblock(
                '''
                You requested scoring boundaries as polygons, but the dataset
                does not have any annotations and no site summaries were
                specified to add them.  You probably forgot to specify the site
                summaries that should be scored!
                '''))

    # TODO: in the case that we are NOT reestimating polygon boundaries, then
    # we should pass through any previous system rejected sites.

    print(f'track_fn={track_fn}')
    """
    ../tasks/tracking/normalize.py
    """
    coco_dset = geowatch.tasks.tracking.normalize.run_tracking_pipeline(
        coco_dset, track_fn=track_fn, gt_dset=gt_dset,
        viz_out_dir=args.viz_out_dir, sensor_warnings=args.sensor_warnings,
        **track_kwargs)

    if boundary_regions_gdf is not None:
        print('Cropping to boundary regions')
        coco_remove_out_of_bound_tracks(coco_dset, video_region_assignments)

    # Measure how long tracking takes
    proc_context.stop()

    rich.print('[green] Finished main tracking phase')

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

    verbose = 1

    # Convert scored kwcoco tracks to sites models
    all_sites = convert_kwcoco_to_iarpa(
        coco_dset, default_region_id=args.region_id)
    print(f'{len(all_sites)=}')

    # Postprocess / classify sites
    config = args

    # for site in all_sites:
    #     site.header['properties'].setdefault('cache', {})
    #     print(site.header['properties']['status'], site.header['properties']['score'], site.header['properties']['cache'].get('reject_reason'))

    for site in ub.ProgIter(all_sites, desc='classify sites'):
        classify_site(site, config)

    # for site in all_sites:
    #     site.header['properties'].setdefault('cache', {})
    #     print(site.header['properties']['status'], site.header['properties']['score'], site.header['properties']['cache'].get('reject_reason'))

    if args.out_sites_dir is not None:
        # write sites to disk
        sites_dir = ub.Path(args.out_sites_dir).ensuredir()
        site_fpaths = []
        for site in ub.ProgIter(all_sites, desc='writing sites', verbose=verbose):
            site_props = site['features'][0]['properties']
            assert site_props['type'] == 'site'
            site_fpath = sites_dir / (site_props['site_id'] + '.geojson')
            site_fpaths.append(os.fspath(site_fpath))

            with safer.open(site_fpath, 'w', temp_file=not ub.WIN32) as f:
                geojson.dump(site, f, indent=2)

    if args.out_sites_fpath is not None:
        site_tracking_output = tracking_output.copy()
        site_tracking_output['files'] = site_fpaths
        out_sites_fpath = ub.Path(args.out_sites_fpath)
        out_sites_fpath.parent.ensuredir()
        print(f'Write tracked site result to {out_sites_fpath}')
        with safer.open(out_sites_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(site_tracking_output, file, indent='    ')

    # Convert site models to site summaries
    if args.out_site_summaries_dir is not None:

        site_summary_dir = ub.Path(args.out_site_summaries_dir).ensuredir()
        # write site summaries to region models on disk
        groups = ub.group_items(all_sites, lambda site: site.header['properties']['region_id'])

        site_summary_fpaths = []
        for region_id, sites in groups.items():

            # Converting sites to a region model makes them site summaries
            sites = geomodels.SiteModelCollection(sites)
            new_summaries = sites.as_region_model()
            region_fpath = site_summary_dir / (region_id + '.geojson')

            if args.append_mode and region_fpath.is_file():
                new_region = geomodels.RegionModel.coerce(region_fpath)
                new_region.features.extend(list(new_summaries.site_summaries()))
                if verbose:
                    print(f'writing to existing region {region_fpath}')
            else:
                new_region = new_summaries
                if verbose:
                    print(f'writing to new region {region_fpath}')

            site_summary_fpaths.append(os.fspath(region_fpath))
            with safer.open(region_fpath, 'w', temp_file=not ub.WIN32) as f:
                geojson.dump(new_region, f, indent=2)

    if args.out_site_summaries_fpath is not None:
        site_summary_tracking_output = tracking_output.copy()
        site_summary_tracking_output['files'] = site_summary_fpaths
        out_site_summaries_fpath = ub.Path(args.out_site_summaries_fpath)
        out_site_summaries_fpath.parent.ensuredir()
        print(f'Write tracked site summary result to {out_site_summaries_fpath}')
        with safer.open(out_site_summaries_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(site_summary_tracking_output, file, indent='    ')

    if args.viz_out_dir is not None:
        rich.print(f'Tracking Viz: [link={args.viz_out_dir}]{args.viz_out_dir}[/link]')


def coco_video_gdf(coco_dset):
    # TODO: rectify with covered_video_geo_regions
    import pandas as pd
    from geowatch.utils import util_gis
    from geowatch.geoannots.geococo_objects import CocoGeoVideo
    crs84 = util_gis.get_crs84()
    crs84_parts = []
    for video in coco_dset.videos().objs:
        coco_video = CocoGeoVideo(video=video, dset=coco_dset)
        utm_part = coco_video.wld_corners_gdf
        crs84_part = utm_part.to_crs(crs84)
        crs84_parts.append(crs84_part)
    video_gdf = pd.concat(crs84_parts).reset_index()
    return video_gdf


def assign_videos_to_regions(video_gdf, boundary_regions_gdf):
    """
    Assign each video to a region (usually for BAS)
    """
    from geowatch.utils import util_gis
    idx1_to_idxs2 = util_gis.geopandas_pairwise_overlaps(video_gdf, boundary_regions_gdf)
    video_region_assignments = []
    for idx1, idxs2 in idx1_to_idxs2.items():
        video_row = video_gdf.iloc[idx1]
        video_name = video_row['name']
        region_rows = boundary_regions_gdf.iloc[idxs2]

        if len(idxs2) == 0:
            raise AssertionError(ub.paragraph(
                f'''
                The coco video {video_name} has no intersecting candidate
                boundary regions in the {len(boundary_regions_gdf)} that were
                searched.
                '''))
        elif len(idxs2) > 1:
            # In the case that there are multiple candidates, use the one with
            # the highest IOU and emit a warning.
            isect_area = region_rows.intersection(video_row['geometry']).area
            union_area = region_rows.union(video_row['geometry']).area
            iou = isect_area / union_area
            iou = iou.sort_values(ascending=False)
            region_rows = region_rows.loc[iou.index]
            region_name = region_rows.iloc[0]['region_id']
            region_geom = region_rows.iloc[0]['geometry']

            import warnings
            warnings.warn(ub.paragraph(
                f'''
                The coco video {video_name} has {len(region_rows)} intersecting
                candidate boundary regions. Choosing region {region_name} with
                the highest IOU of {iou.iloc[0]:0.4f}.

                The runner up regions are:
                {ub.urepr(region_rows.iloc[1:3]['region_id'].tolist(), nl=0)}
                with ious:
                {ub.urepr(iou.iloc[1:3].tolist(), precision=4, nl=0)}
                '''))
        else:
            region_name = region_rows.iloc[0]['region_id']
            region_geom = region_rows.iloc[0]['geometry']

        video_region_assignments.append((video_name, region_name, region_geom))
    return video_region_assignments


def coco_remove_out_of_bound_tracks(coco_dset, video_region_assignments):
    # Remove any tracks that are outside of region bounds.
    # First find which regions correspond to which videos.
    from geowatch.utils import util_gis
    from shapely.geometry import shape
    import rich
    import geopandas as gpd
    crs84 = util_gis.get_crs84()

    # Actually remove the offending annots
    to_remove_trackids = set()
    all_track_ids = set()
    for assign in video_region_assignments:
        video_name, region_name, region_geom = assign
        video_id = coco_dset.index.name_to_video[video_name]['id']
        video_imgs = coco_dset.images(video_id=video_id)
        video_aids = list(ub.flatten(video_imgs.annots))
        video_annots = coco_dset.annots(video_aids)
        all_track_ids.update(video_annots.lookup('track_id'))
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
        num_remove_tracks = len(to_remove_trackids)
        num_remove_annots = len(to_remove_aids)
        num_total_tracks = len(all_track_ids)
        num_total_annots = coco_dset.n_annots
        rich.print(ub.paragraph(
            f'''
            [yellow]Removing {num_remove_tracks} / {num_total_tracks} out-of-bounds
            tracks with {num_remove_annots} / {num_total_annots} annotations
            '''))
    else:
        rich.print('[green]All annotations are in bounds')

    coco_dset.remove_annotations(to_remove_aids)


def demo(coco_dset, regions_dir, coco_dset_sc, sites_dir, cleanup=True):
    import json
    from tempfile import NamedTemporaryFile
    bas_args = [
        coco_dset.fpath,
        '--out_site_summaries_dir',
        regions_dir,
        '--track_fn',
        'geowatch.tasks.tracking.from_heatmap.TimeAggregatedBAS',
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
        'geowatch.tasks.tracking.from_heatmap.TimeAggregatedSC',
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
