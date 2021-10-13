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


def geojson_feature(img, anns, coco_dset):
    '''
    Group kwcoco annotations in the same track (site) and image
    into one Feature in an IARPA site model
    '''
    def single_geometry(ann):
        seg_geo = ann['segmentation_geos']
        assert isinstance(seg_geo, dict)
        return _single_geometry(seg_geo)

    # grab source and date for single_properties per-img instead of per-ann

    try:
        # Pick a reasonable source image, we don't have a spec for this yet
        candidate_keys = [
            'parent_name', 'parent_file_name', 'name', 'file_name'
        ]
        candidate_sources = list(filter(None, map(img.get, candidate_keys)))
        source = candidate_sources[0]
    except IndexError:
        raise Exception(f'cannot determine source of gid {img["gid"]}')

    date = dateutil.parser.parse(img['date_captured']).date()

    def single_properties(ann):

        current_phase = coco_dset.cats[ann['category_id']]['name'],

        return {
            'current_phase': current_phase,
            'is_occluded': False,  # HACK
            'is_site_boundary': True,  # HACK
            'source': source,
            'observation_date': date.isoformat(),
            'sensor_name': img['sensor_coarse'],
            'score': ann.get('score', 1.0)
        }

    geometry_list = list(map(single_geometry, anns))
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
        for key in ['source', 'observation_date', 'sensor_name']:
            values = properties_list[key]
            assert len(set(values)) == 1
            properties[key] = str(values[0])

        # take area-weighted average score
        properties['score'] = np.average(
            list(map(float, properties_list['score'])),
            weights=[geom.area for geom in geometry_list])

        return properties

    return geojson.Feature(geometry=combined_geometries(geometry_list),
                           properties=combined_properties(
                               properties_list, geometry_list))


def track_to_site(coco_dset, trackid, region_id):
    '''
    Turn a kwcoco track into an IARPA site model
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
        geojson_feature(coco_dset.imgs[gid], _anns, coco_dset)
        for gid, _anns in ub.group_items(anns, gids).items()
    ]

    # add prediction field to each feature
    # > A “Polygon” should define the foreign members “current_phase”,
    # > “predicted_next_phase”, and “predicted_next_phase_date”.
    # TODO we need to figure out how to link individual polygons across frames
    # within a track when we have >1 polygon per track_index (from MultiPolygon
    # or multiple annotations) to handle splitting/merging.
    # This is because this prediction foreign field is defined wrt the CURRENT
    # polygon, not per-observation.
    for ix, feat in enumerate(features):

        current_phase = feat['properties']['current_phase']
        sep = ','
        n_polys = current_phase.count(sep)
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
                        future_phases + current_phases[len(future_phases):])
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

    # add other top-level fields

    centroid_latlon = np.array(
        _combined_geometries([
            _single_geometry(feat['geometry']) for feat in features
        ]).centroid)[::-1]

    return geojson.FeatureCollection(
        features,
        id='_'.join((region_id, str(trackid).zfill(4))),
        version=watch.__version__,
        mgrs=MGRS().toMGRS(*centroid_latlon, MGRSPrecision=0),
        status='positive_annotated',
        score=1.0,  # TODO does this matter?
    )


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
    parser.add_argument(
        "--out_dir",
        help="Output directory where GeoJSON files will be written")
    parser.add_argument(
        "--region_id",
        help=ub.paragraph('''
        ID for region that sites belong to.
        If None, try to infer from kwcoco file.
        ''')
    )
    parser.add_argument(
        "--track_fn",
        help=ub.paragraph('''
        Function to add tracks. If None, use existing tracks.
        Example: 'watch.tasks.tracking.from_heatmap.time_aggregated_polys'
        ''')
    )
    args = parser.parse_args(args)

    # Read the kwcoco file
    coco_dset = kwcoco.CocoDataset(args.in_file)

    # Normalize
    if args.track_fn is None:
        # no-op function
        track_fn = lambda x: x  # noqa
    else:
        track_fn = eval(args.track_fn)

    coco_dset = watch.tasks.tracking.normalize.normalize(coco_dset,
                                                         track_fn=track_fn,
                                                         overwrite=False)

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
