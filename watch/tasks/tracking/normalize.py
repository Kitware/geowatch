import itertools
import kwimage
import shapely
import shapely.ops
from os.path import join
from progiter import ProgIter
import numpy as np
import ubelt as ub

from watch.utils.kwcoco_extensions import TrackidGenerator
from watch.gis.geotiff import geotiff_crs_info


def dedupe_annots(coco_dset):
    '''
    Check for annotations with different aids that are the same geometry
    '''
    @ub.memoize
    def geom(ann):
        return kwimage.Segmentation.coerce(
            ann['segmentation']).data.to_shapely().buffer(0)

    if 0:
        # TODO is there already some check like this in dset.add_annotation()?
        def _eq(ann1, ann2):
            eq_keys = ['image_id', 'category_id', 'track_id']
            if any(ann1.get(k) != ann2.get(k) for k in eq_keys):
                return False
            if ann1['segmentation'] == ann2['segmentation']:
                return True
            return geom(ann1).almost_equals(geom(ann2))

        # this task is finding an equivalence partition of anns using _eq,
        # which is O(n^2).
        # if there were a key function, it'd be O(n) using groupby, but
        # almost_equals() makes this difficult
        def equivalence_partition(aids):
            groups = dict()
            for aid in set(aids):
                partitioned = False
                for repr_aid in groups:
                    if _eq(coco_dset.anns[aid], coco_dset.anns[repr_aid]):
                        groups[repr_aid].add(aid)
                        partitioned = True
                        break
                if not partitioned:
                    groups[aid] = set()

        aids_to_remove = itertools.chain.from_iterable(
            equivalence_partition(coco_dset.anns))
    else:
        annots = coco_dset.annots()
        eq_keys = ['image_id', 'category_id', 'track_id', 'segmentation']
        groups_dict = ub.group_items(
            annots.aids,
            zip(*(map(str, annots.get(k, None)) for k in eq_keys)))
        aids_to_remove = itertools.chain.from_iterable(
            aids[1:] for aids in groups_dict.values())

    coco_dset.remove_annotations(aids_to_remove)

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
    if not overwrite:
        if None not in coco_dset.annots().get('segmentation_geos', None):
            return coco_dset

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


def remove_small_annots(coco_dset, min_area_px=1, min_geo_precision=6):
    '''
    There are several reasons for a detection to be too small to keep.
    Remove these and return the rest of the dataset.

    1. Detections that aren't well-formed polygons.
        These are simply errors.
        They show up fairly often in an arbitrary dset; TODO figure out why
        possible culprits:
            mask_to_scored_polygons?
            cropping in propagate_labels?
        >>> d = kwcoco.CocoDataset('pred_KR_R01.kwcoco_timeagg_v1.json')
        >>> sum(are_invalid(d.annots())), d.n_annots
        6686, 13136

    2. Very small detections in pixel-space (area <1 pixel).
        These probably couldn't represent something visible,
        unless the GSD is very large.
        Skip this check by setting min_area_px=0

    3. Overly-precise geo-detections.
        Because GSD varies, and because lat-lon isn't equal-area, detections
        can be trivial in geo space but not pixel space.
        GeoJSON spec recommends a precision of 6 decimal places, which is
        ~10cm. (IARPA annotations conform to this).
        This check removes detections that are empty when rounded.
        Skip this check by setting min_geo_precision=None

    Sources:
        [1] https://pypi.org/project/geojson/#default-and-custom-precision
    '''
    def remove_annotations(coco_dset, remove_fn):
        # TODO merge into kwcoco?
        annots = coco_dset.annots()
        if len(annots) > 0:
            empty_aids = np.extract(remove_fn(annots), annots.aids)
            coco_dset.remove_annotations(list(empty_aids))
        return coco_dset

    #
    # 1.
    #

    def are_invalid(annots):
        def _is_valid(poly):
            try:
                return poly.to_shapely().is_valid
            except ValueError:
                return False

        return list(
            map(lambda area, poly: area == 0 or not _is_valid(poly),
                annots.detections.data['boxes'].area.sum(axis=1),
                annots.detections.data['segmentations'].to_polygon_list()))

    coco_dset = remove_annotations(coco_dset, are_invalid)

    #
    # 2.
    #

    if min_area_px is not None and min_area_px > 0:

        def are_small(annots):
            return annots.detections.data['boxes'].area.sum(
                axis=1) < min_area_px

        coco_dset = remove_annotations(coco_dset, are_small)

    #
    # 3.
    #

    if min_geo_precision is not None:

        # https://github.com/perrygeo/geojson-precision
        def _set_precision(coords, precision):
            result = []
            try:
                return round(coords, int(precision))
            except TypeError:
                for coord in coords:
                    result.append(_set_precision(coord, precision))
            return result

        def is_empty_rounded(geom):
            geom['coordinates'] = _set_precision(geom['coordinates'],
                                                 min_geo_precision)
            return shapely.geometry.asShape(geom).is_empty

        def are_empty_rounded(annots):
            return list(
                map(is_empty_rounded, annots.lookup('segmentation_geos')))

        coco_dset = remove_annotations(coco_dset, are_empty_rounded)

    return coco_dset


def ensure_videos(coco_dset):
    '''
    Ensure every image belongs to a video, even a dummy video
    and has a frame_index
    '''
    # TODO guess frame_index in a better way, like by date
    vid_gids = set().union(*coco_dset.index.vidid_to_gids.values())
    missing_gids = set(coco_dset.imgs) - vid_gids
    if missing_gids:
        vidid = coco_dset.add_video('DEFAULT')
        for ix, gid in enumerate(missing_gids):
            coco_dset.imgs[gid]['video_id'] = vidid
            coco_dset.imgs[gid]['frame_index'] = ix

    try:
        coco_dset.images().lookup('frame_index')
    except KeyError:
        raise AssertionError('all images in dset need a frame_index')

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
    def tracks(annots):
        return annots.get('track_id', None)

    def are_trackless(annots):
        _tracks = tracks(annots)
        return np.array(_tracks) == None  # noqa

    # first, for each video, apply a track_fn from from_heatmap or from_polygon
    for gids in coco_dset.index.vidid_to_gids.values():
        sub_dset = coco_dset.subset(gids=gids)
        if overwrite:
            sub_dset = track_fn(sub_dset)
        else:
            existing_tracks = tracks(sub_dset.annots())
            _are_trackless = are_trackless(sub_dset.annots())
            if np.any(_are_trackless) or len(existing_tracks) == 0:
                sub_dset = track_fn(sub_dset)
                annots = sub_dset.annots()
                # if new annots were not created, rollover the old tracks
                if len(annots) == len(existing_tracks):
                    annots.set(
                        'track_id',
                        np.where(_are_trackless, tracks(annots),
                                 existing_tracks))

            # could maybe use coco_dset.union, but it doesn't reuse IDs
            # TODO an ensure_annotations to do this properly
            # coco_dset.anns.update(sub_dset.anns)
            coco_dset.remove_annotations(
                set(sub_dset.anns).intersection(coco_dset.anns))
            coco_dset.add_annotations(sub_dset.anns.values())

    # then cleanup leftover untracked annots
    coco_dset.remove_annotations(
        list(
            np.extract(are_trackless(coco_dset.annots()),
                       coco_dset.annots().aids)))

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
        for idx, (vidid, aids) in enumerate(
                ub.group_items(
                    annots.aids,
                    coco_dset.images(annots.gids).get('video_id',
                                                      None)).items()):
            sub_annots = coco_dset.annots(aids=aids)
            if idx > 0:
                sub_annots.set('track_id', next(new_trackids))

    return coco_dset


def add_track_index(coco_dset):
    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(trackid=trackid)

        # order the track by track_index
        sorted_gids = coco_dset.index._set_sorted_by_frame_index(annots.gids)
        track_index_dict = dict(zip(sorted_gids, range(len(sorted_gids))))
        annots.set('track_index',
                   map(lambda gid: track_index_dict[gid], annots.gids))

    return coco_dset


def normalize_phases(coco_dset):
    '''
    Convert internal representation of phases to their IARPA standards
    as well as inserting a baseline guess for the special category 'change'
    '''
    # TODO: were these used by some toydata? They aren't in the real files.
    # TODO: if we are hardcoding names we should have some constants file
    # to keep things sane.
    category_dict = {
        'construction': 'Active Construction',
        'pre-construction': 'Site Preparation',
        'finalized': 'Post Construction',
        'obscured': 'Unknown',
        '': 'No Activity'
    }
    good_cats = set(category_dict.values())
    for cat_name in good_cats:
        coco_dset.ensure_category(cat_name)

    for name, cat in coco_dset.name_to_cat.items():
        try:
            # if name not in good_cats:
            if name not in good_cats.union({'change'}):
                cat['name'] = category_dict[name]
        except KeyError:
            raise KeyError(f'{coco_dset.tag} has unknown category {name}')

    # HACK remove change
    # if we have partial coverage, interpolate from existing good labels
    # else, predict site prep for the first half of the track and then
    # active construction for the second half
    # TODO break out these heuristics
    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(trackid=trackid)
        cats = annots.cnames
        if 'change' in cats:
            if len(set(cats) - {'change'}) > 0:
                cids = np.array(annots.cids)
                good_ixs = np.flatnonzero(cats != 'change')
                ix_to_cid = dict(zip(range(len(good_ixs)), cids[good_ixs]))
                interp = np.interp(range(len(cats)), good_ixs,
                                   range(len(good_ixs)))
                annots.set('category_id',
                           [ix_to_cid[int(ix)] for ix in np.round(interp)])
            else:
                gids_first_half, _ = np.array_split(
                    np.array(
                        coco_dset.index._set_sorted_by_frame_index(
                            coco_dset.annots(trackid=trackid).gids)), 2)
                siteprep_cid = coco_dset.name_to_cat['Site Preparation']['id']
                active_cid = coco_dset.name_to_cat['Active Construction']['id']
                annots.set('category_id', [
                    siteprep_cid if gid in gids_first_half else active_cid
                    for gid in annots.gids
                ])

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
                img['sensor_coarse'] = sensor_dict[sensor]
        except KeyError:
            sensor = img.get('sensor_coarse', None)
            raise KeyError(
                f'{coco_dset.tag} image {name} has unknown sensor {sensor}')

    return coco_dset


def normalize(coco_dset, track_fn, overwrite):
    '''
    Driver function to apply all normalizations
    '''
    def _normalize_annots(coco_dset, overwrite):
        coco_dset = dedupe_annots(coco_dset)
        coco_dset = add_geos(coco_dset, overwrite)
        coco_dset = remove_small_annots(coco_dset)
        return coco_dset

    coco_dset = _normalize_annots(coco_dset, overwrite)
    coco_dset = ensure_videos(coco_dset)

    # apply tracks; ensuring we process newly added annots
    n_existing_annots = coco_dset.n_annots
    coco_dset = apply_tracks(coco_dset, track_fn, overwrite)
    if coco_dset.n_annots > n_existing_annots:
        coco_dset = _normalize_annots(coco_dset, overwrite=False)

    coco_dset = dedupe_tracks(coco_dset)
    coco_dset = add_track_index(coco_dset)
    coco_dset = normalize_phases(coco_dset)
    coco_dset = normalize_sensors(coco_dset)

    # HACK, ensure coco_dset.index is up to date
    coco_dset._build_index()

    return coco_dset