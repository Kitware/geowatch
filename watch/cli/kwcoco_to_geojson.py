"""
This file contains logic to convert a kwcoco file into an IARPA Site Model.

At a glance the IARPA Site Model is a GeoJSON FeatureCollection with the
following informal schema:

list of json dictionaries, where each
"site" has the following has the following informal schema:

TODO:
    - [ ] Is our computation of the "site-boundary" correct?
    - [ ] Do we have a complete list of IARPA category names?
    - [ ] Do we have a complete list of IARPA sensor names?
    - [ ] Is our computation of the "predicted_phase" correct?
    - [ ] How do we compute "is_occluded"?
    - [ ] Document details about is_site_boundary
    - [ ] Document details about is_occluded

.. code::

    {
        "id": "site-000001",
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": <GeoJson Geometry>  # a polygon or multipolygon
                "properties": {
                    "observation_date": <iso-datetime>  # e.g. "1981-01-01",
                    "score": <float>  # e.g. 1.0,
                    "predicted_phase": <phase-label>  # e.g. "Active Construction",
                    "predicted_phase_start_date": <iso-datetime>  # e.g. "1981-01-01",
                    "is_occluded":  <comma-separated-str-of-True-False-for-each-poly>,  # e.g. "False,True"
                    "is_site_boundary": <comma-separated-str-of-True-False-for-each-poly>,  # e.g. "False,True"
                    "current_phase": <PhaseLabel>     # e.g. "No Activity",
                    "sensor_name": <name-of-sensor>   # e.g. "WorldView",
                    "source": <name-of-source-image>  # e.g. "WorldviewFile-1980-01-01.NTF"
                },
            },
            ...
        ]
    }


For official documentation about the KWCOCO json format see [1]_. A formal
json-schema can be found in ``kwcoco.coco_schema``

For official documentation about the IARPA json format see [2, 3]_. A formal
json-schema can be found in ``watch/rc/site-model.schema.json``.

References:
    .. [1] https://gitlab.kitware.com/computer-vision/kwcoco
    .. [2] https://infrastructure.smartgitlab.com/docs/pages/api_documentation.html#site-model
    .. [3] https://smartgitlab.com/TE/annotations
"""
import geojson
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

# import xdev


def _single_geometry(geom):
    return shapely.geometry.asShape(geom).buffer(0)


def _combined_geometries(geometry_list):
    # TODO does this respect ordering for disjoint polys?
    return shapely.ops.unary_union(geometry_list).buffer(0)


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
        source = None
        for aux in img.get('auxiliary', []):
            basename = os.path.basename(aux['file_name'])
            if basename.endswith('blue.tif'):
                source = basename
        if source is None:
            try:
                # Pick reasonable source image, we don't have a spec for this
                candidate_keys = [
                    'parent_name', 'parent_file_name', 'name', 'file_name'
                ]
                source = next(filter(None, map(img.get, candidate_keys)))
            except StopIteration:
                raise Exception(f'can\'t determine source of gid {img["gid"]}')

        date = dateutil.parser.parse(img['date_captured']).date().isoformat()

        return {
            'source': source,
            'observation_date': date,
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
        TODO we need to figure out how to link individual polygons across frames
        within a track when we have >1 polygon per track_index (from MultiPolygon
        or multiple annotations) to handle splitting/merging.
        This is because this prediction foreign field is defined wrt the CURRENT
        polygon, not per-observation.
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
        geometry = _combined_geometries([
            _single_geometry(feat['geometry']) for feat in features])

        centroid_latlon = np.array(geometry.centroid)[::-1]

        # these are strings, but sorting should be correct in isoformat
        dates = sorted(feat['properties']['observation_date']
                       for feat in features)

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
            'validated': 'False'  # TODO needed?
        }

        if as_summary:
            properties.update(**{
                'type': 'site_summary',
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


def convert_kwcoco_to_iarpa(coco_dset, region_id=None):
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
        >>> from watch.demo import smart_kwcoco_demodata
        >>> import ubelt as ub
        >>> coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
        >>> region_id = 'dummy_region'
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

        for trackid in sub_dset.index.trackid_to_aids:

            site = track_to_site(sub_dset, trackid, _region_id)
            sites.append(site)

    return sites


def main(args):
    parser = argparse.ArgumentParser(
        description="Convert KWCOCO to IARPA GeoJSON")
    parser.add_argument("--in_file", help="Input KWCOCO to convert")
    parser.add_argument("--in_file_gt",
                        default=None,
                        help="GT KWCOCO file used for visualizations")
    parser.add_argument("--in_file_sc",
                        default=None,
                        help="KWCOCO file with SC prediction heatmaps")
    parser.add_argument(
        "--out_dir",
        help="Output directory where GeoJSON files will be written")
    parser.add_argument("--region_id",
                        help=ub.paragraph('''
        ID for region that sites belong to.
        If None, try to infer from kwcoco file.
        '''))
    parser.add_argument("--track_fn",
                        help=ub.paragraph('''
        Function to add tracks. If None, use existing tracks.
        Example: 'watch.tasks.tracking.from_heatmap.time_aggregated_polys'
        '''))
    parser.add_argument("--bas_mode",
                        action='store_true',
                        help=ub.paragraph('''
        In BAS mode, the following changes occur:
            - output will be site summaries instead of sites
            - existing region files will be searched for in out_dir, or
                generated from in_file if not found, and site summaries
                will be appended to them
            - TODO different normalization pipeline
        '''))
    args = parser.parse_args(args)

    # Read the kwcoco file
    coco_dset = kwcoco.CocoDataset(args.in_file)

    if args.in_file_gt is not None:
        gt_dset = kwcoco.CocoDataset(args.in_file_gt)
    else:
        gt_dset = None

    if args.in_file_sc is not None:
        coco_dset_sc = kwcoco.CocoDataset(args.in_file_sc)
    else:
        coco_dset_sc = None

    # Normalize
    if args.track_fn is None:
        # no-op function
        track_fn = lambda x: x  # noqa
    else:
        track_fn = eval(args.track_fn)

    coco_dset = watch.tasks.tracking.normalize.normalize(
        coco_dset,
        track_fn=track_fn,
        overwrite=False,
        gt_dset=gt_dset,
        coco_dset_sc=coco_dset_sc)

    # Convert kwcoco to sites
    sites = convert_kwcoco_to_iarpa(coco_dset, args.region_id)

    # Write sites to disk
    os.makedirs(args.out_dir, exist_ok=True)
    for site in sites:
        with open(os.path.join(args.out_dir, site['id'] + '.geojson'),
                  'w') as f:
            geojson.dump(site, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
