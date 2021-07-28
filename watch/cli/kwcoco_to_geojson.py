import geojson
import json
import os
import sys
import argparse
import kwcoco
import shapely as shp
import shapely.ops
import shapely.geometry
import dateutil.parser
from osgeo import gdal, osr
from watch.gis import spatial_reference

category_dict = {'construction': 'Active Construction',
                 'pre-construction': 'Site Preparation',
                 'finalized': 'Post Construction',
                 'obscured': 'Unknown'}

sensor_dict = {'WV': 'WorldView',
               'S2': 'Sentinel-2',
               'LE': 'Landsat',
               'LC': 'Landsat'}


def shape(geometry, lst=False):
    if geometry['type'] == 'Polygon':
        coords = [geometry['coordinates']]
    else:
        coords = geometry['coordinates']
    if lst:
        return [shp.geometry.shape({'coordinates': [c], 'type':geometry['type']}).buffer(0)
                for c in coords]
    geo_dict = {
        'coordinates': coords,
        'type': geometry['type']
    }
    polygon = shp.geometry.shape(geo_dict).buffer(0)
    return shp.ops.unary_union(polygon)


def predict(ann, vid_id, dset, phase):
    """
    Look forward in time and find the closest video frame that has a different
    construction phase wrt the phase in ann. Return that new phase plus the date
    of the annotation in which that phase was found.
    """
    img_id = ann['image_id']
    min_overlap = .5
    mapping = dset.index.gid_to_aids
    annots = dset.index.anns
    video = dset.index.vidid_to_gids[vid_id]
    img_index = video.index(img_id)
    video = video[img_index:]
    potential_obs = []
    for frame in video:
        potential_obs.extend(mapping[frame])

    union_poly_ann = shape(ann['segmentation_geos'])
    for obs in potential_obs:
        ann_obs = annots[obs]
        union_poly_obs = shape(ann_obs['segmentation_geos'])
        overlap = union_poly_obs.intersection(union_poly_ann).area / union_poly_ann.area
        cat = dset.index.cats[ann_obs['category_id']]
        if overlap > min_overlap and phase != category_dict[cat['supercategory']]:
            obs_img = dset.index.imgs[ann_obs['image_id']]
            date = dateutil.parser.parse(obs_img['date_captured']).date()
            prediction = {
                'predicted_phase': category_dict[cat['supercategory']],
                'predicted_phase_date': date.isoformat().replace('-', '/'),
            }
            return prediction
    prediction = {
        'predicted_phase': None,
        'predicted_phase_date': None,
    }
    return prediction


def boundary(sseg_geos, img_path):
    src = gdal.Open(img_path, gdal.GA_ReadOnly)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    if not spatial_reference.check_latlons(ulx, uly):
        source = osr.SpatialReference()
        source.ImportFromWkt(src.GetProjection())
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(source, target)
        ulx, uly, x = transform.TransformPoint(ulx, uly)
        lrx, lry, x = transform.TransformPoint(lrx, lry)
    shape_list = shape(sseg_geos, lst=True)
    site_geo = {
        'type': 'Polygon',
        'coordinates': [[uly, ulx], [uly, lrx], [lry, lrx], [lry, ulx], [uly, ulx]]
    }
    site_shape = shape(site_geo)
    lst = [site_shape.intersection(s).area /
           site_shape.area for s in shape_list]
    bool_lst = [str(area > .9) for area in lst]
    return ','.join(bool_lst)


def convert_kwcoco_to_iarpa(coco_dset, out_dir, region_id):
    """
    Convert a kwcoco coco_dset to the IARPA JSON format

    Example:
        >>> import sys, ubelt
        >>> from watch.cli.kwcoco_to_geojson import *  # NOQA
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
    """
    sites = {}
    for ann in coco_dset.index.anns.values():
        img = coco_dset.index.imgs[ann['image_id']]
        cat = coco_dset.index.cats[ann['category_id']]

        # Non-standard COCO fields, needed by watch
        sseg_geos = ann['segmentation_geos']
        date = dateutil.parser.parse(img['date_captured']).date()

        feature = geojson.Feature(geometry=sseg_geos)
        feature['properties']['source'] = img['parent_file_name']
        feature['properties']['observation_date'] = date.isoformat().replace('-', '/')
        # Is this default supposed to be a string?
        feature['properties']['score'] = ann.get('score', '1.0')
        feature['properties']['current_phase'] = category_dict[cat['supercategory']]

        if feature['properties']['current_phase'] == 'Post Construction':
            feature['properties']['predicted_phase'] = None
            feature['properties']['predicted_phase_date'] = None
        else:
            prediction = predict(ann, img['video_id'], coco_dset,
                                 feature['properties']['current_phase'])
            feature['properties'].update(prediction)

        feature['properties']['sensor_name'] = sensor_dict[img['sensor_coarse']]
        '''
        feature['properties']['is_occluded']
        depends on cloud masking output?
        '''
        feature['properties']['is_site_boundary'] = boundary(sseg_geos, img['file_name'])
        tag = img.get('site_tag', img['parent_file_name'].split('/')[0])
        if sites.get(tag):
            sites[tag].append(feature)
        else:
            sites[tag] = [feature]

    return sites

    for site in sites:
        collection = geojson.FeatureCollection(sites[site][1:], id=region_id)
        with open(os.path.join(out_dir, site + '.json'), 'w') as f:
            json.dump(collection, f, indent=2)


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

    coco_dset = kwcoco.CocoDataset(args.in_file)
    convert_kwcoco_to_iarpa(coco_dset, args.out_dir, args.region_id)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
