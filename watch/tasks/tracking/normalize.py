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
from watch.tasks.tracking.utils import TrackFunction


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
    images = coco_dset.images()
    annotated_gids = np.array(
        images.gids)[np.array(list(map(len, images.annots))) > 0]
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

        >>> # xdoctest: +SKIP
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

    Example:
        >>> import kwimage
        >>> from copy import deepcopy
        >>> from watch.tasks.tracking.normalize import remove_small_annots
        >>> from watch.demo import smart_kwcoco_demodata
        >>> dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps()
        >>> # This dset has 1 video with all images the same size
        >>> # For testing, resize one of the images so there is a meaningful
        >>> # difference between img space and vid space
        >>> scale_factor = 0.5
        >>> aff = kwimage.Affine.coerce({'scale': scale_factor})
        >>> img = dset.imgs[1]
        >>> img['width'] *= scale_factor
        >>> img['height'] *= scale_factor
        >>> img['warp_img_to_vid']['scale'] = 1/scale_factor
        >>> for aux in img['auxiliary']:
        >>>     aux['warp_aux_to_img']['scale'] = aux['warp_aux_to_img'].get(
        >>>         'scale', 1) * scale_factor
        >>> annots = dset.annots(gid=img['id'])
        >>> old_annots = deepcopy(annots)
        >>> dets = annots.detections.warp(aff)
        >>> # TODO this doesn't handle keypoints, and is rather brittle, is
        >>> # there a way to simply do something like:
        >>> #    annots.detections = annot.detections.warp(w)
        >>> annots.set('bbox', dets.boxes.to_coco(style='new'))
        >>> annots.set('segmentation', dets.data['segmentations'].to_coco(
        >>>     style='new'))
        >>> # test that scaling worked
        >>> assert np.all(annots.boxes.area < old_annots.boxes.area)
        >>> assert np.all(annots.boxes.warp(aff.inv()).area ==
        >>>     old_annots.boxes.area)
        >>> # test that remove_small_annots no-ops with no threshold
        >>> # (ie there are no invalid annots here)
        >>> assert dset.n_annots == remove_small_annots(deepcopy(dset),
        >>>     min_area_px=0, min_geo_precision=None).n_annots
        >>> # test that annots can be removed
        >>> assert remove_small_annots(deepcopy(dset), min_area_px=1e99,
        >>>     min_geo_precision=None).n_annots == 0
        >>> # test that annotations are filtered in video space
        >>> # pick a threshold above the img annot size and below the vid
        >>> # annot size; annot should not be removed
        >>> thresh = annots.boxes.area[0] + 1
        >>> assert annots.aids[0] in remove_small_annots(deepcopy(dset),
        >>>     min_area_px=thresh, min_geo_precision=None).annots(
        >>>         gid=img['id']).aids
        >>> # TODO test min_geo_precision
    '''
    def remove_annotations(coco_dset, remove_fn):
        # TODO merge into kwcoco?
        annots = coco_dset.annots()
        if len(annots) > 0:
            empty_aids = annots.compress(remove_fn(annots)).aids
            coco_dset.remove_annotations(empty_aids)
            print(f'Removing small aids: {empty_aids}. '
                  'After removing small, trackids: ',
                  set(coco_dset.annots().get('track_id', None)))
        return coco_dset

    def polys_in_video(annots, coco_dset):
        # gets polygons in video space
        # TODO are there vectorized versions of these functions?
        # ideally, annots.detections.warp(list_of_affines)

        # separate into lists
        polys = annots.detections.data['segmentations'].to_polygon_list()
        warps = [
            kwimage.Affine.coerce(aff)
            for aff in coco_dset.images(annots.gids).get(
                'warp_img_to_vid', {'scale': 1})
        ]

        # apply warping
        polys = [p.warp(w) for p, w in zip(polys, warps)]

        # put them back together
        polys = kwimage.PolygonList(polys)

        return polys

    #
    # 1. and 2.
    #

    if min_area_px is None:
        min_area_px = 0

    def are_invalid_or_small(annots):
        def _is_invalid_or_small(poly):
            try:
                # TODO split out polys with invalid subsets
                # ex. https://gis.stackexchange.com/a/321804
                shp_poly = poly.to_shapely().buffer(0)
                return shp_poly.area <= min_area_px or not shp_poly.is_valid
            except ValueError:
                return True

        return list(map(_is_invalid_or_small,
                        polys_in_video(annots, coco_dset)))

    coco_dset = remove_annotations(coco_dset, are_invalid_or_small)

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
            return shapely.geometry.shape(geom).is_empty

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
    # HACK, TODO this is probably a kwcoco bug that needs fixed
    if 'videos' not in coco_dset.dataset:
        coco_dset.dataset['videos'] = list(coco_dset.index.videos.values())

    # TODO guess frame_index in a better way, like by date
    vid_gids = set().union(*coco_dset.index.vidid_to_gids.values())
    missing_gids = set(coco_dset.imgs) - vid_gids
    if missing_gids:
        vidid = coco_dset.add_video('DEFAULT')
        for ix, gid in enumerate(missing_gids):
            coco_dset.imgs[gid]['video_id'] = vidid
            coco_dset.imgs[gid]['frame_index'] = ix

        # HACK TODO bug etc
        gids = coco_dset.index.vidid_to_gids[vidid]
        coco_dset.index.vidid_to_gids[vidid] = gids.union(missing_gids)

    try:
        coco_dset.images().lookup('frame_index')
    except KeyError:
        raise AssertionError('all images in dset need a frame_index')

    return coco_dset


def dedupe_tracks(coco_dset):
    '''
    Assuming that videos are made of disjoint images, ensure that trackids
    are not shared by two tracks in different videos.
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
            if idx > 0:
                coco_dset.annots(aids=aids).set('track_id', next(new_trackids))

    return coco_dset


def add_track_index(coco_dset):
    '''
    Ensure each track's track_index is fully populated with strictly
    increasing but not-necessarily-unique values (can have multiple track
    entries per image)
    '''
    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(trackid=trackid)

        # order the track by track_index
        sorted_gids = coco_dset.index._set_sorted_by_frame_index(annots.gids)
        track_index_dict = dict(zip(sorted_gids, range(len(sorted_gids))))
        annots.set('track_index', map(track_index_dict.get, annots.gids))

    return coco_dset


def normalize_phases(coco_dset):
    '''
    Convert internal representation of phases to their IARPA standards
    as well as inserting a baseline guess for activity classification

    Example:
        >>> # test baseline guess
        >>> from watch.tasks.tracking.normalize import normalize_phases
        >>> from watch.demo import smart_kwcoco_demodata
        >>> dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps()
        >>> dset.cats[1]['name'] = 'salient'
        >>> dset.remove_categories([2,3])
        >>> assert dset.cats == {1: {'id': 1, 'name': 'salient'}}
        >>> # HACK, this shouldn't be needed?
        >>> # TODO file bug report
        >>> dset._build_index()
        >>> dset = normalize_phases(dset)
        >>> assert (dset.categories(dset.annots().category_id).name ==
        >>>     ((['Site Preparation'] * 10) +
        >>>      (['Active Construction'] * 9) +
        >>>      (['Post Construction'])))
    '''
    # Remove site boundary annotations (should be already incorporated
    # by track_fn if needed)
    # coco_dset.remove_categories(['Site Boundary'], keep_annots=False)

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
    good_cats = set(category_dict.values()).union({'Site Boundary'})
    for cat_name in good_cats:
        coco_dset.ensure_category(cat_name)

    # possible category names for a change prediction
    change_cats = {'change', 'change_prob', 'salient'}

    for name, cat in coco_dset.name_to_cat.items():
        try:
            if name not in good_cats.union(change_cats):
                cat['name'] = category_dict[name]
        except KeyError:
            raise KeyError(f'{coco_dset.tag} has unknown category {name}')

    # HACK remove change
    # TODO break out these heuristics
    from collections import Counter
    log = Counter()
    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(trackid=trackid)
        cats = annots.cnames
        if set(cats) - good_cats:
            # if we have partial coverage, interpolate from good labels
            if set(cats) - change_cats:
                log.update(['partial class labels'])
                cids = np.array(annots.cids)
                good_ixs = np.in1d(cats, list(change_cats), invert=True)
                ix_to_cid = dict(zip(range(len(good_ixs)), cids[good_ixs]))
                interp = np.interp(range(len(cats)), good_ixs,
                                   range(len(good_ixs)))
                annots.set('category_id',
                           [ix_to_cid[int(ix)] for ix in np.round(interp)])
            # else, predict site prep for the first half of the track and then
            # active construction for the second half
            # with post construction on the last frame
            else:
                log.update(['no class labels'])
                gids_first_half, gids_second_half = np.array_split(
                    np.array(
                        coco_dset.index._set_sorted_by_frame_index(
                            coco_dset.annots(trackid=trackid).gids)), 2)
                siteprep_cid = coco_dset.name_to_cat['Site Preparation']['id']
                active_cid = coco_dset.name_to_cat['Active Construction']['id']
                post_cid = coco_dset.name_to_cat['Post Construction']['id']
                gids = np.array(annots.gids)
                cids = np.where([g in gids_first_half for g in gids],
                                siteprep_cid, active_cid)
                cids = np.where(gids == gids_second_half[-1], post_cid, cids)
                annots.set('category_id', cids)
        else:
            log.update(['full class labels'])

    print('label status of tracks: ', log)
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

    for img in coco_dset.imgs.values():
        try:
            sensor = img['sensor_coarse']
            if sensor not in good_sensors:
                img['sensor_coarse'] = sensor_dict[sensor]
        except KeyError:
            sensor = img.get('sensor_coarse', None)
            name = img.get('name', img['file_name'])
            raise KeyError(
                f'{coco_dset.tag} image {name} has unknown sensor {sensor}')

    return coco_dset


def normalize(coco_dset, track_fn, overwrite, gt_dset=None, **track_kwargs):
    '''
    Driver function to apply all normalizations

    Example:
        >>> import kwcoco as kc
        >>> from watch.tasks.tracking.normalize import *
        >>> from watch.tasks.tracking.from_polygon import OverlapTrack
        >>> # create demodata
        >>> d = kc.CocoDataset.demo()
        >>> ann_dct = d.anns[1]
        >>> d.remove_annotations(range(1,12))
        >>> ann_dct.pop('keypoints')
        >>> ann_dct.pop('id')
        >>> for i in range(1,4):
        >>>     ann_dct.update(image_id=i)
        >>>     d.add_annotation(**ann_dct)
        >>> for img, sensor in zip(d.imgs.values(), ['WV', 'S2', 'L8']):
        >>>     img['sensor_coarse'] = sensor
        >>> d.remove_categories(range(2,9))
        >>> d.cats[1]['supercategory'] = None
        >>> d.cats[1]['name'] = 'change'
        >>> # test everything except geo-info
        >>> overwrite = False
        >>> def _normalize_annots(coco_dset, overwrite):
        >>>     coco_dset = dedupe_annots(coco_dset)
        >>>     # coco_dset = add_geos(coco_dset, overwrite)
        >>>     coco_dset = remove_small_annots(coco_dset,
        >>>         min_geo_precision=None)
        >>>     return coco_dset
        >>> coco_dset = d.copy()
        >>> coco_dset = _normalize_annots(coco_dset, overwrite)
        >>> assert coco_dset.anns == d.anns
        >>> coco_dset = ensure_videos(coco_dset)
        >>> assert coco_dset.index.vidid_to_gids[1] == coco_dset.imgs.keys()
        >>> n_existing_annots = coco_dset.n_annots
        >>> coco_dset = OverlapTrack().apply_per_video(coco_dset, overwrite)
        >>> assert set(coco_dset.annots().get('track_id')) == {0}  # not 1?
        >>> assert coco_dset.n_annots == n_existing_annots
        >>> coco_dset = dedupe_tracks(coco_dset)
        >>> assert set(coco_dset.annots().get('track_id')) == {0}
        >>> coco_dset = add_track_index(coco_dset)
        >>> assert coco_dset.annots().get('track_index') == [0,1,2]
        >>> coco_dset = normalize_phases(coco_dset)
        >>> assert (coco_dset.annots().cnames ==
        >>> ['Site Preparation', 'Site Preparation', 'Post Construction'])
        >>> coco_dset = normalize_sensors(coco_dset)
        >>> assert (coco_dset.images().get('sensor_coarse') ==
        >>>     ['WorldView', 'Sentinel-2', 'Landsat 8'])



    '''
    def _normalize_annots(coco_dset, overwrite):
        coco_dset = dedupe_annots(coco_dset)
        coco_dset = add_geos(coco_dset, overwrite)
        coco_dset = remove_small_annots(coco_dset, min_area_px=0,
                                        min_geo_precision=None)
        # coco_dset._build_index()

        return coco_dset

    if len(coco_dset.anns) > 0:
        coco_dset = _normalize_annots(coco_dset, overwrite)
    coco_dset = ensure_videos(coco_dset)
    #import xdev; xdev.embed()
    # apply tracks
    assert issubclass(track_fn, TrackFunction), 'must supply a valid track_fn!'
    coco_dset = track_fn(**track_kwargs).apply_per_video(coco_dset)

    # normalize and add geo segmentations
    coco_dset = _normalize_annots(coco_dset, overwrite=False)
    coco_dset._build_index()
    print('After normalizing: track ids',
          set(coco_dset.annots().get('track_id', None)))

    coco_dset = dedupe_tracks(coco_dset)
    coco_dset = add_track_index(coco_dset)
    coco_dset = normalize_phases(coco_dset)
    coco_dset = normalize_sensors(coco_dset)

    # HACK, ensure coco_dset.index is up to date
    coco_dset._build_index()

    if gt_dset is not None:
        # visualize predicted sites with true sites
        out_dir = './_assets/1b_official_BR_small'
        from visualize import visualize_videos
        visualize_videos(coco_dset, gt_dset, out_dir,
                         coco_dset_sc=track_kwargs.get('coco_dset_sc'))

    return coco_dset
