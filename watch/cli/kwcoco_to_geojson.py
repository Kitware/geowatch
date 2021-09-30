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
import itertools
import geojson
import json
import os
import sys
import argparse
import kwcoco
import dateutil.parser
import watch
import kwimage
import shapely
import shapely.ops
from os.path import join
from collections import defaultdict
from progiter import ProgIter
import numpy as np
import ubelt as ub

import xdev


def geojson_feature(img, anns, coco_dset):
    '''
    Group kwcoco annotations in the same track (site) and image
    into one Feature in an IARPA site model
    '''
    def single_geometry(ann):
        seg_geo = ann['segmentation_geos']
        assert isinstance(seg_geo, dict)
        return kwimage.MultiPolygon.from_geojson(seg_geo).to_shapely().buffer(
            0)

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
        # annotations should be disjoint before being combined
        for geom1, geom2 in itertools.combinations(geometry_list, 2):
            try:
                assert geom1.disjoint(geom2), [ann['id'] for ann in anns]
            except AssertionError:
                xdev.embed()
        '''
        # TODO ensure this respects ordering
        return shapely.ops.unary_union(geometry_list).buffer(0)

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

        # take average score
        # TODO do something smarter like weighting by each polygon's area
        properties['score'] = np.mean(
            list(map(float, properties_list['score'])))

        return properties

    return geojson.Feature(geometry=combined_geometries(geometry_list),
                           properties=combined_properties(
                               properties_list, geometry_list))


def track_to_site(coco_dset, trackid, region_id, mgrs):
    '''
    Turn a kwcoco track into an IARPA site model
    '''

    # get annotations in this track, sort them, and group them into features
    annots = coco_dset.annots(trackid=trackid)
    ixs, gids, anns = annots.lookup('track_index'), annots.gids, annots.objs
    # HACK because track_index is not unique, need a tiebreaker key to sort on
    # _, gids, anns = zip(*sorted(zip(ixs, gids, anns)))
    _, _, gids, anns = zip(*sorted(zip(ixs, range(len(ixs)), gids, anns)))
    features = [
        geojson_feature(coco_dset.imgs[gid], _anns, coco_dset)
        for gid, _anns in ub.group_items(anns, gids).items()
    ]

    # add prediction field to each feature
    # > A “Polygon” should define the foreign members “current_phase”,
    # > “predicted_next_phase”, and “predicted_next_phase_date”.
    # TODO we need to figure out how to link individual polygons across frames
    # within a track when we have >1 polygon per track_index (from MultiPolygons
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
            future_n_polys = future_phase.count(sep)
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
    return geojson.FeatureCollection(
        features,
        id='_'.join((region_id, str(trackid).zfill(4))),
        version=watch.__version__,
        mgrs=mgrs,
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
        >>> for site_name, collection in sites.items():
        >>>     jsonschema.validate(collection, schema=SITE_SCHEMA)

    """
    def fpath(img):
        # Handle the case if an image consists of one main image or multiple
        # auxiliary images
        if img.get('file_name', None) is not None:
            img_path = join(coco_dset.bundle_dpath, img['file_name'])
        else:
            # Use the first auxiliary image
            # (todo: might want to choose determine the image "canvas" coordinates?)
            img_path = join(coco_dset.bundle_dpath, img['auxiliary'][0]['file_name'])
            return img_path

    # parallelize grabbing img CRS info
    def _info(img):
        info = watch.gis.geotiff.geotiff_crs_info(fpath(img))
        return info

    executor = ub.Executor('thread', 16)
    # optimization: filter to only images containing at least 1 annotation
    annotated_gids = np.extract(np.array(list(map(len, coco_dset.images().annots))) > 0,
                                coco_dset.images().gids)
    infos = {gid: executor.submit(_info, coco_dset.imgs[gid]) for gid in annotated_gids}
    # missing_geo_aids = np.extract(np.array(coco_dset.annots().lookup('segmentation_geos', None)) == None, coco_dset.annots().aids)
    for gid, img in ProgIter(coco_dset.imgs.items(), desc='precomputing geo-segmentations'):

        # vectorize over anns; this does some unnecessary computation
        annots = coco_dset.annots(gid=gid)
        if len(annots) == 0:
            continue
        info = infos[gid].result()
        pxl_anns = annots.detections.data['segmentations']
        wld_anns = pxl_anns.warp(info['pxl_to_wld'])
        wgs_anns = wld_anns.warp(info['wld_to_wgs84'])
        geojson_anns = [poly.swap_axes().to_geojson() for poly in wgs_anns]
        
        for aid, geojson_ann in zip(annots.aids, geojson_anns):
            
            ann = coco_dset.anns[aid]

            # Non-standard COCO fields, needed by watch
            if 'segmentation_geos' not in ann or 1:

                # Note that each segmentation annotation here will get
                # written out as a separate GeoJSON feature.
                # TODO: Confirm that this is the desired behavior
                # (especially with respect to the evaluation metrics)

                ann['segmentation_geos'] = geojson_ann
    
    coco_dset.dump(coco_dset.fpath)
    coco_dset = kwcoco.CocoDataset(coco_dset.fpath)

    # HACK for mono-site
    if coerce_site_boundary:
        for gid in coco_dset.imgs:
            annots = coco_dset.annots(gid=gid)
            if len(annots) == 0:
                continue

            template_ann = annots.peek()
            
            # print(list(np.unique(annots.lookup('category_id'))), [coco_dset.name_to_cat['change']['id']])
            assert list(np.unique(annots.lookup('category_id'))) == [coco_dset.name_to_cat['change']['id']]
            try:
                sseg_geos = [kwimage.MultiPolygon.from_shapely(
                    shapely.ops.unary_union([
                        kwimage.MultiPolygon.from_geojson(seg_geo).to_shapely().buffer(0)
                        for seg_geo in (annots.lookup('segmentation_geos'))])).to_geojson()]
            except TypeError:
                xdev.embed()
            
            template_ann.pop('segmentation', None)
            template_ann.pop('bbox', None)
            template_ann['score'] == np.mean(annots.lookup('score'))
            template_ann['segmentation_geos'] = sseg_geos

            coco_dset.remove_annotations(annots.aids[1:])
    
    # HACK
    first_half, second_half = np.split(np.array(ub.peek(coco_dset.index.vidid_to_gids.values())), 2)

    site_features = defaultdict(list)
    for ann in ProgIter(coco_dset.index.anns.values(), desc='converting annotations'):
        img = coco_dset.imgs[ann['image_id']]
        cat = coco_dset.cats[ann['category_id']]
        catname = cat['name']

        sseg_geos = ann['segmentation_geos']

        date = dateutil.parser.parse(img['date_captured']).date()

        source = None
        # FIXME: seems fragile?
        if img.get('parent_file_name', None):
            source = img.get('parent_file_name').split('/')[0]
        elif img.get('name', None):
            source = img.get('name')
        else:
            raise Exception('cannot determine source')

        # Consider that sseg_geos could one (dict) or more (list)
        if isinstance(sseg_geos, dict):
            sseg_geos = [sseg_geos]

        for sseg_geo in sseg_geos:
            try:
                feature = geojson.Feature(geometry=sseg_geo)
            except TypeError:
                xdev.embed()
    sites = []

    for vidid, video in coco_dset.index.videos.items():
        if region_id == None:
            _region_id = video['name']
        else:
            _region_id = region_id

        # TODO each site could actually have a different MGRS tile
        # call mgrs.MGRS() on its centroid
        mgrs = video['properties']['mgrs']

        sub_dset = coco_dset.subset(gids=coco_dset.index.vidid_to_gids[vidid])

        for trackid in sub_dset.index.trackid_to_aids:

            site = track_to_site(sub_dset, trackid, _region_id, mgrs)
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
        help=
        "ID for region that sites belong to. If None, try to infer from kwcoco file."
    )
    args = parser.parse_args(args)

    # Read the kwcoco file
    coco_dset = kwcoco.CocoDataset(args.in_file)

    # Normalize
    coco_dset = watch.tasks.tracking.normalize.normalize(
        coco_dset,
        track_fn=watch.tasks.tracking.from_polygon.from_overlap,
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
