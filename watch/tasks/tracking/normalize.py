import itertools
import geojson
import json
import os
import sys
import argparse
import kwcoco
import dateutil.parser
import kwimage
import shapely
import shapely.ops
from os.path import join
from collections import defaultdict
from progiter import ProgIter
import numpy as np
import ubelt as ub

import watch
from watch.utils.kwcoco_extensions import TrackidGenerator
from watch.gis.geotiff import geotiff_crs_info

def remove_empty_annots(coco_dset):
    '''
    We are getting some detections with 2 points that aren't well-formed polygons.
    Remove these and return the rest of the dataset.

    Ex.
    {'type': 'MultiPolygon',
      'coordinates': [[[[128.80465424559546, 37.62042949252145],
         [128.80465693697536, 37.61940084645075]]]]},
    
    These don't show up too often in an arbitrary dset:
    >>> k = kwcoco.CocoDataset('KR_Pyeongchang_R01.kwcoco.json')
    >>> sum(are_empty(k.annots())), k.n_annots
    94, 654
    '''
    def are_empty(annots):
        return np.array(
            list(
                itertools.chain.from_iterable(
                    annots.detections.data['boxes'].area))) == 0

    annots = coco_dset.annots()
    empty_aids = np.extract(are_empty(annots), annots.aids)

    coco_dset.remove_annotations(list(empty_aids))

    return coco_dset


def add_geos(coco_dset, overwrite, max_workers=16):
    '''
    Add segmentation_geos to every annotation in coco_dset

    TODO how to handle cropped annotations from propagation?
    Currently this will not correctly round-trip a ground truth site model
    (IARPA -> kwcoco -> IARPA) due to these edge effects.
    Could use 'orig' attr to fix this, but of course generated annotations
    won't have this.
    '''
    def annotated_band(img):
        # this field picks out the (probable; heuristic-based)
        # band that the annotation was actually done on
        if img['file_name'] is not None:
            return img
        aux_ix = img.get('aux_annotated_candidate', 0)
        return img['auxiliary'][aux_ix]

    def fpath(img):
        return join(coco_dset.bundle_dpath, annotated_band(img)['file_name'])

    # parallelize grabbing img CRS info
    executor = ub.Executor('thread', max_workers)
    # optimization: filter to only images containing at least 1 annotation
    annotated_gids = np.extract(
        np.array(list(map(len,
                          coco_dset.images().annots))) > 0,
        coco_dset.images().gids)
    infos = {
        gid: executor.submit(geotiff_crs_info, fpath(coco_dset.imgs[gid]))
        for gid in annotated_gids
    }
    '''
    missing_geo_aids = np.extract(
        np.array(coco_dset.annots().lookup('segmentation_geos', None)) == None,
        coco_dset.annots().aids)
    '''
    for gid, img in ProgIter(coco_dset.imgs.items(),
                             desc='precomputing geo-segmentations'):

        # vectorize over anns; this does some unnecessary computation
        annots = coco_dset.annots(gid=gid)
        if len(annots) == 0:
            continue
        info = infos[gid].result()
        '''
        # if this was encoded into the image dict ok, we can just use it there
        # unfortunately info is still needed because wld_to_wgs84 may
        # not be serializable
        assert np.allclose(info['pxl_to_wld'], np.array(kwimage.Affine.coerce(
            annotated_band(img)['wld_to_pxl']).inv()))
        '''
        img_anns = annots.detections.data['segmentations']
        aux_anns = img_anns.warp(
            kwimage.Affine.coerce(
                annotated_band(img).get('warp_aux_to_img',
                                        kwimage.Affine.eye())).inv())
        wld_anns = aux_anns.warp(info['pxl_to_wld'])
        wgs_anns = wld_anns.warp(info['wld_to_wgs84'])
        geojson_anns = [poly.swap_axes().to_geojson() for poly in wgs_anns]

        for aid, geojson_ann in zip(annots.aids, geojson_anns):

            ann = coco_dset.anns[aid]

            if 'segmentation_geos' not in ann or overwrite:

                ann['segmentation_geos'] = geojson_ann

    return coco_dset


def dedupe_tracks(coco_dset):
    '''
    Assuming that videos are made of disjoint images, ensure that trackids
    are not shared by two tracks in different videos.

    Also ensure each track's track_index is fully populated with strictly
    increasing but not-necessarily-unique values (can have multiple track
    entries per image)
    '''
    new_trackids = TrackidGenerator(coco_dset)

    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(trackid=trackid)

        # split each video into a separate track
        for idx, (vidid, aids) in enumerate(ub.group_items(annots.aids, coco_dset.images  (annots.gids).get('video_id', None)).items()):
            sub_annots = coco_dset.annots(aids=aids)
            if idx > 0:
                sub_annots.set('track_id', next(new_trackids))

            # order the track by track_index
            sorted_gids = coco_dset.index._set_sorted_by_frame_index(sub_annots.gids)
            track_index_dict = dict(zip(sorted_gids, range(len(sorted_gids))))
            sub_annots.set('track_index', map(lambda gid: track_index_dict[gid], sub_annots.gids))

    return coco_dset


def normalize_phases(coco_dset):
    '''
    Convert internal representation of phases to their IARPA standards
    '''
    # TODO: were these used by some toydata? They aren't in the real files.
    # TODO: if we are hardcoding names we should have some constants file
    # to keep things sane.

    category_dict = {
        'construction': 'Active Construction',
        'pre-construction': 'Site Preparation',
        'finalized': 'Post Construction',
        'obscured': 'Unknown'
    }
    good_cats = set(category_dict.values())
    # HACK
    good_cats.add('change')

    for name, cat in coco_dset.name_to_cat.items():
        try:
            if name not in good_cats:
                cat['name'] = category_dict[name]
        except KeyError:
            raise KeyError(f'{coco_dset.tag} has unknown category {name}')

    return coco_dset


def normalize_sensors(coco_dset):
    '''
    Convert internal representations of sensors to their IARPA standards
    '''
    sensor_dict = {
        'WV': 'WorldView',
        'S2': 'Sentinel-2',
        'LE': 'Landsat 7',
        'LC': 'Landsat 8',
        'L8': 'Landsat 8',
    }
    good_sensors = set(sensor_dict.values())

    for name, img in coco_dset.index.name_to_img.items():
        try:
            sensor = img['sensor_coarse']
            if sensor not in good_sensors:
                img['sensor_carse'] = good_sensors[sensor]
        except KeyError:
            sensor = img.get('sensor_coarse', None)
            raise KeyError(f'{coco_dset.tag} image {name} has unknown sensor {sensor}')

    return coco_dset


def apply_tracks(coco_dset, track_fn, overwrite):
    '''
    Ensure each annotation in coco_dset has a track_id.

    Args:
        coco_dset: kwcoco.CocoDataset
        track_fn: function to apply per-video, from tracking.from_polygons
            or tracking.from_heatmaps
        overwrite: if True, remove and replace any preexisting track_ids.

    Returns:
        modified coco_dset
    '''
    for gids in coco_dset.vidid_to_gids.values():
        sub_dset = coco_dset.subset(gids=gids)
        sub_dset = track_fn(sub_dset, overwrite=overwrite)
    
    return coco_dset


def normalize(coco_dset, track_fn, overwrite):
    '''
    Driver function to apply all normalizations
    '''
    coco_dset = remove_empty_annots(coco_dset, overwrite)
    coco_dset = add_geos(coco_dset, overwrite)
    coco_dset = dedupe_tracks(coco_dset)
    coco_dset = normalize_phases(coco_dset)
    coco_dset = normalize_sensors(coco_dset)
    coco_dset = apply_tracks(coco_dset, track_fn, overwrite)
    # HACK, ensure coco_dset.index is up to date
    coco_dset._build_index()
    return coco_dset
