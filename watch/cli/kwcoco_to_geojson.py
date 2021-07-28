"""
For documentation about the IARPA json format see [1]_. A formal json-schema
can be found in ``watch/rc/site-model.schema.json``.

For documentation about the KWCOCO json format see [2]_. A formal json-schema
can be found in ``kwcoco.coco_schema``

References:
    .. [1] https://infrastructure.smartgitlab.com/docs/pages/api_documentation.html#site-model
    .. [2] https://gitlab.kitware.com/computer-vision/kwcoco
"""
import geojson
import json
import os
import sys
import argparse
import kwcoco
import dateutil.parser
import watch
import kwimage
from os.path import join
from collections import defaultdict


# TODO: if we are hardcoding names we should have some constants file
# to keep things sane.w:
category_dict = {
    'construction': 'Active Construction',
    'pre-construction': 'Site Preparation',
    'finalized': 'Post Construction',
    'obscured': 'Unknown'
}

sensor_dict = {
    'WV': 'WorldView',
    'S2': 'Sentinel-2',
    'LE': 'Landsat',
    'LC': 'Landsat',
    'L8': 'Landsat',
}


def predict(ann, vid_id, coco_dset, phase):
    """
    Look forward in time and find the closest video frame that has a different
    construction phase wrt the phase in ann. Return that new phase plus the date
    of the annotation in which that phase was found.
    """
    img_id = ann['image_id']
    min_overlap = .5
    annots = coco_dset.index.anns

    # Find all images that come after this one
    video_gids = coco_dset.index.vidid_to_gids[vid_id]
    img_index = video_gids.index(img_id)
    future_gids = video_gids[img_index:]
    cand_aids = []
    for frame_gid in future_gids:
        cand_aids.extend(coco_dset.index.gid_to_aids[frame_gid])

    union_poly_ann = kwimage.MultiPolygon.from_geojson(ann['segmentation_geos']).to_shapely()
    for cand_aid in cand_aids:
        ann_obs = annots[cand_aid]
        cat = coco_dset.index.cats[ann_obs['category_id']]
        predict_phase = category_dict.get(cat['name'], cat['name'])
        if phase != predict_phase:
            union_poly_obs = kwimage.MultiPolygon.from_geojson(ann_obs['segmentation_geos']).to_shapely()
            overlap = union_poly_obs.intersection(union_poly_ann).area / union_poly_ann.area
            if overlap > min_overlap:
                obs_img = coco_dset.index.imgs[ann_obs['image_id']]
                date = dateutil.parser.parse(obs_img['date_captured']).date()
                prediction = {
                    'predicted_phase': predict_phase,
                    'predicted_phase_start_date': date.isoformat(),
                }
                return prediction
    prediction = {
        'predicted_phase': None,
        'predicted_phase_start_date': None,
    }
    return prediction


def boundary(sseg_geos, img_path):
    geoinfo = watch.gis.geotiff.geotiff_crs_info(img_path)
    img_bounds_wgs84 = kwimage.Polygon(exterior=geoinfo['wgs84_corners'])
    if geoinfo['wgs84_crs_info']['axis_mapping'] != 'OAMS_TRADITIONAL_GIS_ORDER':
        # If this is in traditional (lon/lat), convert to authority lat/long
        img_bounds_wgs84 = img_bounds_wgs84.swap_axes()
    img_wgs84 = img_bounds_wgs84.to_shapely()
    # Geojson is always supposed to be lon/lat, so swap to lat/lon to compare
    # with authority wgs84
    ann_wgs84 = kwimage.MultiPolygon.from_geojson(sseg_geos).swap_axes().to_shapely()
    lst = [img_wgs84.intersection(s).area / img_wgs84.area for s in ann_wgs84.geoms]
    bool_lst = [str(area > .9) for area in lst]
    return ','.join(bool_lst)


def convert_kwcoco_to_iarpa(coco_dset, region_id):
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
    site_features = defaultdict(list)
    for ann in coco_dset.index.anns.values():
        img = coco_dset.index.imgs[ann['image_id']]
        cat = coco_dset.index.cats[ann['category_id']]

        # Non-standard COCO fields, needed by watch
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

        feature = geojson.Feature(geometry=sseg_geos)
        properties = feature['properties']
        properties['source'] = source
        properties['observation_date'] = date.isoformat()
        # Is this default supposed to be a string?
        properties['score'] = ann.get('score', 1.0)

        # This was supercategory, is that supposed to be name instead?
        # FIXME: what happens when the category nmames dont work?
        properties['current_phase'] = category_dict.get(cat['name'], cat['name'])

        if properties['current_phase'] == 'Post Construction':
            properties['predicted_phase'] = None
            properties['predicted_phase_start_date'] = None
        else:
            prediction = predict(ann, img['video_id'], coco_dset,
                                 properties['current_phase'])
            properties.update(prediction)

        properties['sensor_name'] = sensor_dict[img['sensor_coarse']]

        # HACK IN IS_OCCLUDED
        num_polys = len(kwimage.MultiPolygon.from_geojson(sseg_geos).data)
        properties['is_occluded'] = ','.join(['False'] * num_polys)
        '''
        properties['is_occluded']
        depends on cloud masking output?
        '''

        # Handle the case if an image consists of one main image or multiple
        # auxiliary images
        if img.get('file_name', None) is not None:
            img_path = join(coco_dset.bundle_dpath, img['file_name'])
        else:
            # Use the first auxiliary image
            # (todo: might want to choose determine the image "canvas" coordinates?)
            img_path = join(coco_dset.bundle_dpath, img['auxiliary'][0]['file_name'])

        properties['is_site_boundary'] = boundary(sseg_geos, img_path)

        # This seems fragile? Needs docs.
        site_name = None
        if img.get('site_tag', None):
            site_name = img['site_tag']
        elif source is not None:
            site_name = source
        else:
            raise Exception('cannot determine site_name')

        site_features[site_name].append(feature)

    sites = {}
    for site_name, features in site_features.items():
        sites[site_name] = geojson.FeatureCollection(features, id=region_id)
    return sites


def main(args):
    parser = argparse.ArgumentParser(
        description="Convert KWCOCO to IARPA GeoJSON")
    parser.add_argument("--in_file", help="Input KWCOCO to convert")
    parser.add_argument("--out_dir",
                        help="Output directory where GeoJSON files will be written")
    parser.add_argument(
        "--region_id",
        help="ID for region that site belongs to")
    args = parser.parse_args(args)

    # Read the kwcoco file
    coco_dset = kwcoco.CocoDataset(args.in_file)

    # Convert kwcoco to sites
    sites = convert_kwcoco_to_iarpa(coco_dset, args.region_id)

    # Write site to disk
    for site, collection in sites.items():
        with open(os.path.join(args.out_dir, site + '.json'), 'w') as f:
            json.dump(collection, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
