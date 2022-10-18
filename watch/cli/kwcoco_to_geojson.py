"""
This file contains logic to convert a kwcoco file into an IARPA Site Model.

At a glance the IARPA Site Model is a GeoJSON FeatureCollection with the
following informal schema:

For official documentation about the KWCOCO json format see [1]_. A formal
json-schema can be found in ``kwcoco.coco_schema``

For official documentation about the IARPA json format see [2, 3]_. A formal
json-schema can be found in ``watch/rc/site-model.schema.json``.

References:
    .. [1] https://gitlab.kitware.com/computer-vision/kwcoco
    .. [2] https://infrastructure.smartgitlab.com/docs/pages/api/
    .. [3] https://smartgitlab.com/TE/annotations


DESIGN TODO:
    - [ ] Separate out into two processes:
        1) given a kwcoco file, does tracking and produces another kwcoco file with predicted "tracked" annotations.
        2) given a kwcoco file with predicted "tracked" annotations, convert that back to geojson
"""
import datetime
import itertools
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import dateutil.parser
import geojson
import jsonschema
import kwcoco
import numpy as np
import shapely
import shapely.ops
import ubelt as ub
import scriptconfig as scfg
# import colored_traceback.auto  # noqa

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class KWCocoToGeoJSONConfig(scfg.DataConfig):
    """
    Convert KWCOCO to IARPA GeoJSON
    """
    in_file = scfg.Value(None, required=True, help='Input KWCOCO to convert', position=1)

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
            '''
            JSON string or path to file containing keyword arguments for
            the chosen TrackFunction. Examples include: coco_dset_gt,
            coco_dset_sc, thresh, key. Any file paths will be loaded as
            CocoDatasets if possible.
            '''), group='track')
    site_summary = scfg.Value(None, help=ub.paragraph(
            '''
            A filepath glob or json blob containing either a
            site_summary or a region_model that includes site summaries.
            Each summary found will be added to in_file as 'Site
            Boundary' annotations.
            '''), group='behavior')
    clear_annots = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Clears all annotations before running tracking, so it starts
            from a clean slate.
            '''), group='behavior')


__config__ = KWCocoToGeoJSONConfig


def _single_geometry(geom):
    return shapely.geometry.shape(geom).buffer(0)


def _ensure_multi(
    poly: Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]
) -> shapely.geometry.MultiPolygon:
    if isinstance(poly, shapely.geometry.MultiPolygon):
        return poly
    elif isinstance(poly, shapely.geometry.Polygon):
        return shapely.geometry.MultiPolygon([poly])
    else:
        raise TypeError(f'{poly} of type {type(poly)}')


def _combined_geometries(geometry_list):
    return shapely.ops.unary_union(geometry_list).buffer(0)


def _normalize_date(date_str):
    return dateutil.parser.parse(date_str).date().isoformat()


# For MultiPolygon observations. Could be ', '?
sep = ','


@profile
def geojson_feature(anns, coco_dset, with_properties=True):
    '''
    Group kwcoco annotations in the same track (site) and image
    into one Feature in an IARPA site model
    '''

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

    if with_properties:
        image_properties_dct = {}
        gids = {ann['image_id'] for ann in anns}
        for gid in gids:
            coco_img = coco_dset.coco_image(gid).detach()
            image_properties_dct[gid] = _per_image_properties(coco_img)

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

    geometry_list = list(map(single_geometry, anns))
    if with_properties:
        properties_list = list(map(single_properties, anns))

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
        geometry = _ensure_multi(_combined_geometries(geometry_list))
        properties = combined_properties(properties_list, geometry_list)
    else:
        geometry = _combined_geometries(geometry_list)
        properties = {}

    return geojson.Feature(geometry=geometry, properties=properties)


@profile
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
        geojson_feature(_anns, coco_dset, with_properties=(not as_summary))
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
        region_id = '_'.join(site_id.split('_')[:-1])

    if as_summary:
        return site_feature(coco_dset, region_id, site_id, trackid, gids, features, as_summary)
    else:
        _site_feat = site_feature(coco_dset, region_id, site_id, trackid, gids, features, as_summary)
        return geojson.FeatureCollection([_site_feat] + features)


def predict_phase_changes(site_id, features):
    '''
    Set predicted_phase_transition and predicted_phase_transition_date.

    This should only kick in when the site does not end before the current
    day (latest available image). See tracking.normalize.normalize_phases
    for what happens if the site has ended.

    https://smartgitlab.com/TE/standards/-/wikis/Site-Model-Specification
    '''
    all_phases = [
        feat['properties']['current_phase'].split(sep) for feat in features
    ]

    tomorrow = (dateutil.parser.parse(
        features[-1]['properties']['observation_date']) +
                datetime.timedelta(days=1)).isoformat()

    def transition_date_from(phase):
        for feat, phases in zip(reversed(features), reversed(all_phases)):
            if phase in phases:
                return (dateutil.parser.parse(
                    feat['properties']['observation_date']) +
                        datetime.timedelta(
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


def site_feature(coco_dset, region_id, site_id, trackid, gids, features, as_summary):
    '''
    Feature containing metadata about the site
    '''
    from mgrs import MGRS

    geom_list = [_single_geometry(feat['geometry']) for feat in features]
    geometry = _combined_geometries(geom_list)

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
    status = set(
        coco_dset.annots(trackid=trackid).get('status',
                                              'system_confirmed'))
    assert len(status) == 1, f'inconsistent {status=} for {trackid=}'
    status = status.pop()

    PERFORMER_ID = 'kit'

    import watch
    properties = {
        'site_id': site_id,
        'version': watch.__version__,
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
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> from watch.tasks.tracking.normalize import normalize
        >>> from watch.tasks.tracking.from_polygon import MonoTrack
        >>> from watch.demo import smart_kwcoco_demodata
        >>> import ubelt as ub
        >>> coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
        >>> coco_dset = normalize(coco_dset, track_fn=MonoTrack, overwrite=False, polygon_fn='heatmaps_to_polys')
        >>> region_ids = ['KR_R001', 'KR_R002']
        >>> coco_dset.videos().set('name', region_ids)
        >>> sites = convert_kwcoco_to_iarpa(coco_dset)
        >>> print('sites = {}'.format(ub.repr2(sites, nl=7, sort=0)))
        >>> import jsonschema
        >>> import watch
        >>> SITE_SCHEMA = watch.rc.load_site_model_schema()
        >>> for site in sites:
        >>>     jsonschema.validate(site, schema=SITE_SCHEMA)
    """
    sites = []
    for vidid, video in coco_dset.index.videos.items():
        region_id = video.get('name', default_region_id)
        gids = coco_dset.index.vidid_to_gids[vidid]
        sub_dset = coco_dset.subset(gids=gids)

        for site_idx, trackid in enumerate(sub_dset.index.trackid_to_aids):
            site = track_to_site(sub_dset, trackid, region_id, site_idx,
                                 as_summary)
            sites.append(site)

    return sites


def _validate():
    # jsonschema.validate(site, schema=SITE_SCHEMA)
    import jsonschema
    import watch
    SITE_SCHEMA = watch.rc.load_site_model_schema()
    site_fpaths = list(ub.Path('.').glob('*.geojson'))
    for fpath in site_fpaths:
        site = json.load(open(fpath, 'r'))
        jsonschema.validate(site, schema=SITE_SCHEMA)


def _coerce_site_summaries(site_summary_or_region_model,
                           default_region_id=None, strict=True) -> List[Tuple[str, Dict]]:
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
        List[Tuple[region_id: str, site_summary: Dict]]
    """
    from watch.utils import util_gis

    geojson_infos = list(util_gis.coerce_geojson_datas(
        site_summary_or_region_model, format='json', allow_raw=True))

    # validate the json
    site_summaries = []

    for info in geojson_infos:
        site_summary_or_region_model = info['data']

        if strict and not isinstance(site_summary_or_region_model, dict):
            raise AssertionError(
                f'unknown site summary {type(site_summary_or_region_model)=}'
            )

        try:  # is this a region model?
            # Unfortunately, we can't trust the region file schema
            region_model = site_summary_or_region_model

            TRUST_REGION_SCHEMA = 0
            import watch
            if TRUST_REGION_SCHEMA:
                region_model_schema = watch.rc.load_region_model_schema()
                jsonschema.validate(region_model, schema=region_model_schema)
            else:
                if region_model['type'] != 'FeatureCollection':
                    raise AssertionError

                for feat in region_model['features']:
                    assert feat['type'] == 'Feature'
                    row_type = feat['properties']['type']
                    if row_type not in {'region', 'site_summary'}:
                        raise jsonschema.ValidationError('not a region')

            _summaries = [
                f for f in region_model['features']
                if (f['properties']['type'] == 'site_summary'
                    # TODO handle positive_partial
                    and f['properties']['status'] in {'positive_annotated', 'system_proposed', 'system_confirmed'})
            ]
            region_feat = None
            for f in region_model['features']:
                if f['properties']['type'] == 'region':
                    if region_feat is not None:
                        raise AssertionError('Region files needs exactly one region type but got multiple')
                    region_feat = f
            if region_feat is None:
                raise AssertionError('Region files needs exactly one region type but got 0')
            assert region_feat['properties']['type'] == 'region'
            region_id = region_feat['properties'].get('region_id',
                                                      default_region_id)
            site_summaries.extend([(region_id, s) for s in _summaries])

        except jsonschema.ValidationError:  # or a site model?
            # TODO validate this
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

    USE_GEO_ASSIGNMENT = 1
    if USE_GEO_ASSIGNMENT:
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

        sitesum_gdf = gpd.GeoDataFrame.from_features([t[1] for t in site_summaries], crs=util_gis._get_crs84())

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

    # TODO: we can be more efficient if we already have the transform data
    # computed. We need to pass it in here, and prevent it from making
    # more calls to geotiff_metadata
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
        canvas = coco_img.delay('red|green|blue', space='image').finalize()
        canvas = kwimage.normalize_intensity(canvas)
        kwplot.imshow(canvas)
        coco_dset.annots(gid=gid).detections.draw()

    return coco_dset


@profile
def create_region_feature(region_id, site_summaries):
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
def main(args):
    """
    Example:
        >>> # test BAS and default (SC) modes
        >>> import sys, ubelt
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> from watch.cli.kwcoco_to_geojson import main
        >>> from watch.demo import smart_kwcoco_demodata
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # run BAS on demodata in a new place
        >>> coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
        >>> dpath = ub.Path.appdir('watch', 'test', 'tracking', 'main').ensuredir()
        >>> coco_dset.reroot(absolute=True)
        >>> coco_dset.fpath = dpath / 'bas_input.kwcoco.json'
        >>> coco_dset.dump(coco_dset.fpath, indent=2)
        >>> region_id = 'dummy_region'
        >>> regions_dir = dpath / 'regions/'
        >>> bas_coco_fpath = dpath / 'bas_output.kwcoco.json'
        >>> sc_coco_fpath = dpath / 'sc_output.kwcoco.json'
        >>> bas_fpath = dpath / 'bas_sites.geojson'
        >>> sc_fpath = dpath / 'sc_sites.geojson'
        >>> args = bas_args = [
        >>>     '--in_file', coco_dset.fpath,
        >>>     '--out_dir', str(regions_dir),
        >>>     '--out_site_summaries_fpath',  str(bas_fpath),
        >>>     '--out_kwcoco', str(bas_coco_fpath),
        >>>     '--track_fn', 'watch.tasks.tracking.from_polygon.MonoTrack',
        >>> ]
        >>> main(args)
        >>> # reload it with tracks
        >>> coco_dset = kwcoco.CocoDataset(bas_fpath)
        >>> # run SC on the same dset
        >>> sites_dir = dpath / 'sites'
        >>> args = sc_args = [
        >>>     '--in_file', str(bas_coco_fpath),
        >>>     '--out_dir', str(sites_dir),
        >>>     '--out_sites_fpath', str(sc_fpath),
        >>>     '--out_kwcoco', str(sc_coco_fpath),
        >>> ]
        >>> main(args)
        >>> # TODO: check expected results
        >>> # cleanup
        >>> dpath.delete()

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
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
    args = KWCocoToGeoJSONConfig.legacy(cmdline=args)
    print('args = {}'.format(ub.repr2(dict(args), nl=1)))
    # parser = _argparse_cli()
    # args = parser.parse_args(args)
    # print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1)))

    coco_fpath = ub.Path(args.in_file)

    # set the out dir
    # default_out_dir = coco_fpath.parent

    if args.out_sites_dir is not None:
        args.out_sites_dir = ub.Path(args.out_sites_dir)
        # args.out_sites_dir = default_out_dir / 'sites'

    if args.out_site_summaries_dir is not None:
        args.out_site_summaries_dir = ub.Path(args.out_site_summaries_dir)
        # args.out_site_summaries_dir = default_out_dir / 'site-summaries'

    # args.out_sites_dir = ub.Path(args.out_sites_dir)
    # args.out_site_summaries_dir = ub.Path(args.out_site_summaries_dir)

    if args.out_sites_fpath is not None:
        assert args.out_sites_dir is not None
        args.out_sites_fpath = ub.Path(args.out_sites_fpath)

    if args.out_site_summaries_fpath is not None:
        assert args.out_site_summaries_dir is not None
        args.out_site_summaries_fpath = ub.Path(args.out_site_summaries_fpath)

    # load the track kwargs
    if os.path.isfile(args.track_kwargs):
        track_kwargs = json.load(args.track_kwargs)
    else:
        track_kwargs = json.loads(args.track_kwargs)
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
    from kwcoco.util import util_json
    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_args = util_json.ensure_json_serializable(args.__dict__)
    walker = ub.IndexableWalker(jsonified_args)
    for problem in util_json.find_json_unserializable(jsonified_args):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    # TODO: ensure all args are resolved here.
    info = tracking_output['info']

    # TODO: use process context instead
    from watch.utils.process_context import ProcessContext
    proc_context = ProcessContext(
        name='watch.cli.kwcoco_to_geojson', type='process',
        args=jsonified_args,
        config=jsonified_args,
        extra={'pred_info': pred_info},
        track_emissions=False,
    )
    proc_context.start()
    info.append(proc_context.obj)

    # Pick a track_fn
    # HACK remove potentially conflicting annotations as well
    # we shouldn't have saliency annots when we want class or vice versa
    import watch
    CLEAN_DSET = 1
    class_cats = [cat['name'] for cat in watch.heuristics.CATEGORIES]
    saliency_cats = ['salient']
    if args.default_track_fn is not None:
        from watch.tasks.tracking import from_heatmap, from_polygon
        if args.default_track_fn == 'saliency_heatmaps':
            track_fn = from_heatmap.TimeAggregatedBAS
            if CLEAN_DSET:
                coco_dset.remove_categories(class_cats)
        elif args.default_track_fn == 'saliency_polys':
            track_fn = from_polygon.OverlapTrack
            if CLEAN_DSET:
                coco_dset.remove_categories(class_cats)
        elif args.default_track_fn == 'class_heatmaps':
            track_fn = from_heatmap.TimeAggregatedSC
            if CLEAN_DSET:
                coco_dset.remove_categories(saliency_cats)
        elif args.default_track_fn == 'class_polys':
            track_fn = from_polygon.OverlapTrack
            if CLEAN_DSET:
                coco_dset.remove_categories(saliency_cats)
        else:
            track_fn = from_heatmap.TimeAggregatedBAS
            track_kwargs['key'] = [args.default_track_fn]
            if CLEAN_DSET:
                coco_dset.remove_categories(class_cats)
    elif args.track_fn is None:
        track_fn = watch.tasks.tracking.utils.NoOpTrackFunction
    else:
        track_fn = eval(args.track_fn)
        if CLEAN_DSET:
            print('warning: could not check for invalid cats!')

    # add site summaries (site boundary annotations)
    if args.site_summary is not None:
        coco_dset = add_site_summary_to_kwcoco(args.site_summary, coco_dset,
                                               args.region_id)
        cid = coco_dset.name_to_cat[watch.heuristics.SITE_SUMMARY_CNAME]['id']
        coco_dset = coco_dset.subset(coco_dset.index.cid_to_gids[cid])
        print('restricting dset to videos with site_summary annots: ',
              set(coco_dset.index.name_to_video))
        assert coco_dset.n_images > 0, 'no valid videos!'

    coco_dset = watch.tasks.tracking.normalize.normalize(coco_dset,
                                                         track_fn=track_fn,
                                                         overwrite=False,
                                                         gt_dset=gt_dset,
                                                         **track_kwargs)

    # Measure how long tracking takes
    proc_context.stop()

    out_kwcoco = args.out_kwcoco

    if out_kwcoco is not None:
        coco_dset = coco_dset.reroot(absolute=True)
        coco_dset.fpath = out_kwcoco
        ub.Path(out_kwcoco).parent.ensuredir()
        print(f'write to coco_dset.fpath={coco_dset.fpath}')
        coco_dset.dump(out_kwcoco, indent=2)

    # Convert kwcoco to sites
    import safer
    verbose = 1

    if args.out_sites_dir is not None:

        sites_dir = ub.Path(args.out_sites_dir).ensuredir()
        # Also do this in BAS mode
        sites = convert_kwcoco_to_iarpa(coco_dset,
                                        args.region_id,
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
        sites = convert_kwcoco_to_iarpa(coco_dset, args.region_id,
                                        as_summary=True)
        print(f'{len(sites)=}')
        site_summary_dir = ub.Path(args.out_site_summaries_dir).ensuredir()
        # write sites to region models on disk
        groups = ub.group_items(sites, lambda site: site['properties'].pop('region_id'))

        # TODO / FIXME:
        # The script should control if you are in "write" or "append" mode.
        # Often I don't want to append to existing files.
        APPEND_MODE = 0

        site_summary_fpaths = []
        for region_id, site_summaries in groups.items():

            region_fpath = site_summary_dir / (region_id + '.geojson')
            if APPEND_MODE and region_fpath.is_file():
                with open(region_fpath, 'r') as f:
                    region = geojson.load(f)
                if verbose:
                    print(f'writing to existing region {region_fpath}')
            else:
                region = geojson.FeatureCollection(
                    [create_region_feature(region_id, site_summaries)])
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
        print(f'Write tracked site summary result to {out_site_summaries_fpath}')
        with safer.open(out_site_summaries_fpath, 'w', temp_file=True) as file:
            json.dump(site_summary_tracking_output, file, indent='    ')


def demo(coco_dset,
         regions_dir,
         coco_dset_sc,
         sites_dir,
         cleanup=True,
         hybrid=False):
    bas_args = [
        coco_dset.fpath,
        '--out_dir',
        regions_dir,
        '--track_fn',
        'watch.tasks.tracking.from_heatmap.TimeAggregatedBAS',
        '--bas_mode',
    ]
    # run BAS on it
    main(bas_args)
    # reload it with tracks
    # coco_dset = kwcoco.CocoDataset(coco_dset.fpath)
    # run SC on both of them
    if hybrid:  # hybrid approach
        sc_args = [
            coco_dset.fpath, '--out_dir', sites_dir, '--track_fn',
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


def _fix_pred_info():
    # Hack to fix data provinence
    from watch.utils import util_path
    track_fpaths = util_path.coerce_patterned_paths('models/fusion/eval3_candidates/pred/**/tracks.json')
    for fpath in track_fpaths:
        data = json.loads(fpath.read_text())
        for info_item in data['info']:
            if info_item['type'] == 'process' and info_item['properties']['name'] == 'watch.cli.kwcoco_to_geojson':
                pred_kwcoco_json = info_item['properties']['args']['in_file']
                import kwcoco
                pred_dset = kwcoco.CocoDataset(pred_kwcoco_json)
                pred_info = pred_dset.dataset.get('info', [])
                info_item['properties']['pred_info'] = pred_info

        import safer
        with safer.open(fpath, 'w', temp_file=True) as file:
            json.dump(data, file)


if __name__ == '__main__':
    main(sys.argv[1:])
