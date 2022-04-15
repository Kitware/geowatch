import kwcoco
import kwimage
import warnings
from os.path import join
import numpy as np
import ubelt as ub

from watch.utils.kwcoco_extensions import TrackidGenerator
from watch.gis.geotiff import geotiff_crs_info
from watch.tasks.tracking.utils import TrackFunction
from watch.heuristics import SITE_SUMMARY_CNAME
from watch.tasks.tracking.utils import check_only_bg
try:
    from xdev import profile
except Exception:
    profile = ub.identity


def dedupe_annots(coco_dset):
    '''
    Check for annotations with different aids that are the same geometry
    '''
    annots = coco_dset.annots()

    # NOTE: Using segmentations to dedup with segmentation data is fragile
    eq_keys = ['image_id', 'category_id', 'track_id', 'segmentation']
    eq_vals = annots.lookup(eq_keys, None)
    group_keys = [str(row_vals) for row_vals in zip(*eq_vals.values())]
    duplicates = ub.find_duplicates(group_keys)
    if duplicates:
        dup_idxs = list(ub.flatten([idxs[1:] for idxs in duplicates.values()]))
        warnings.warn(
            ub.paragraph(f'''
            There were {len(duplicates)} annotation groups
            with {dup_idxs} duplicate annotations based on
            group keys {eq_keys}
            '''))
        aids_to_remove = list(annots.take(dup_idxs))
        coco_dset.remove_annotations(aids_to_remove, verbose=1)

    return coco_dset


@profile
def add_geos(coco_dset, overwrite, max_workers=16):
    '''
    Add 'segmentation_geos' to every annotation in coco_dset with a
    'segmentation'.

    TODO serializable osr.CoordinateTransform+srcCRS+dstCRS obj

    TODO how to handle cropped annotations from propagation?
    Currently this will not correctly round-trip a ground truth site model
    (IARPA -> kwcoco -> IARPA) due to these edge effects.
    Could use 'orig' attr to fix this, but of course generated annotations
    won't have this.
    '''

    if not overwrite:
        if None not in coco_dset.annots().get('segmentation_geos', None):
            return coco_dset

    def needs_geos(ann):
        return 'segmentation' in ann and (overwrite or
                                          ('segmentation_geos' not in ann))

    # find images containing at least 1 annotation that needs geo coords
    annots = coco_dset.annots()
    annots_to_fix = annots.compress(map(needs_geos, annots.objs))
    gid_to_aids = ub.group_items(annots_to_fix,
                                 annots_to_fix.lookup('image_id'))
    images_to_fix = coco_dset.images(list(gid_to_aids.keys()))

    # parallelize grabbing img CRS info
    executor = ub.JobPool('thread', max_workers)

    for gid in ub.ProgIter(images_to_fix, desc='submit crs jobs'):
        coco_img: kwcoco.CocoImage = coco_dset.coco_image(gid)
        # Hack: find an asset likely to have geoinfo
        aux = coco_img.find_asset_obj('red|green|blue|panchromatic|pan')
        if aux is None:
            aux = coco_img.primary_asset()
        fpath = join(coco_img.bundle_dpath, aux['file_name'])
        job = executor.submit(geotiff_crs_info, fpath)
        job.gid = gid
        job.aux = aux

    for job in executor.as_completed(desc='precomputing geo-segmentations'):
        # job = executor.jobs[2]
        info = job.result()
        gid = job.gid
        aux = job.aux
        aids = gid_to_aids[gid]

        anns = coco_dset.annots(aids=aids).objs
        assert len(anns) > 0, f'image {gid} should have annotations!'
        '''
        # if this was encoded into the image dict ok, we can just use it there
        # unfortunately info is still needed because wld_to_wgs84 may
        # not be serializable
        assert np.allclose(info['pxl_to_wld'], np.array(kwimage.Affine.coerce(
            annotated_band(img)['wld_to_pxl']).inv()))
        '''
        img_anns = kwimage.SegmentationList(
            [kwimage.Segmentation.coerce(ann['segmentation']) for ann in anns])

        warp_img_from_aux = kwimage.Affine.coerce(
            aux.get('warp_aux_to_img', None)).inv()
        aux_anns = img_anns.warp(warp_img_from_aux)
        wld_anns = aux_anns.warp(info['pxl_to_wld'])
        wgs_anns = wld_anns.warp(info['wld_to_wgs84'])
        # Flip into traditional CRS84 coordinates if we need to
        if info['wgs84_crs_info'][
                'axis_mapping'] == 'OAMS_AUTHORITY_COMPLIANT':
            crs84_anns = [poly.swap_axes() for poly in wgs_anns]
        else:
            crs84_anns = wgs_anns
        geojson_anns = [poly.to_geojson() for poly in crs84_anns]

        for ann, geojson_ann in zip(anns, geojson_anns):
            ann['segmentation_geos'] = geojson_ann

    return coco_dset


@profile
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
        >>> coco_dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps()
        >>> # This dset has 1 video with all images the same size
        >>> # For testing, resize one of the images so there is a meaningful
        >>> # difference between img space and vid space
        >>> scale_factor = 0.5
        >>> aff = kwimage.Affine.coerce({'scale': scale_factor})
        >>> img = coco_dset.imgs[1]
        >>> img['width'] *= scale_factor
        >>> img['height'] *= scale_factor
        >>> img['warp_img_to_vid']['scale'] = 1/scale_factor
        >>> for aux in img['auxiliary']:
        >>>     aux['warp_aux_to_img']['scale'] = aux['warp_aux_to_img'].get(
        >>>         'scale', 1) * scale_factor
        >>> annots = coco_dset.annots(gid=img['id'])
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
        >>> assert coco_dset.n_annots == remove_small_annots(deepcopy(coco_dset),
        >>>     min_area_px=0, min_geo_precision=None).n_annots
        >>> # test that annots can be removed
        >>> assert remove_small_annots(deepcopy(coco_dset), min_area_px=1e99,
        >>>     min_geo_precision=None).n_annots == 0
        >>> # test that annotations are filtered in video space
        >>> # pick a threshold above the img annot size and below the vid
        >>> # annot size; annot should not be removed
        >>> thresh = annots.boxes.area[0] + 1
        >>> assert annots.aids[0] in remove_small_annots(deepcopy(coco_dset),
        >>>     min_area_px=thresh, min_geo_precision=None).annots(
        >>>         gid=img['id']).aids
        >>> # test that some but not all annots can be removed
        >>> filtered = remove_small_annots(
        >>>     deepcopy(coco_dset), min_area_px=10000,
        >>>     min_geo_precision=None)
        >>> assert filtered.n_annots > 0 and filtered.n_annots < coco_dset.n_annots
        >>> # TODO test min_geo_precision
    '''
    if min_area_px is None:
        min_area_px = 0

    empty_aids = []
    remove_reason = []

    gid_iter = ub.ProgIter(coco_dset.index.imgs.keys(),
                           total=coco_dset.n_images,
                           desc='filter annotations')
    for gid in gid_iter:
        coco_img = coco_dset.coco_image(gid)
        annots = coco_dset.annots(gid=gid)
        for aid in annots:
            ann = coco_dset.index.anns[aid]

            if min_area_px is not None:
                pxl_sseg = coco_img._annot_segmentation(ann, space='video')

                try:
                    pxl_sseg_sh = pxl_sseg.to_multi_polygon().to_shapely()
                except ValueError:
                    remove_reason.append('invalid_shapely_conversion')
                    empty_aids.append(aid)
                    continue

                if not pxl_sseg_sh.is_valid:
                    pxl_sseg_sh = pxl_sseg_sh.buffer(0)

                if not pxl_sseg_sh.is_valid:
                    remove_reason.append('invalid_pixel_polygon')
                    empty_aids.append(aid)
                    continue

                try:
                    pxl_sseg_sh.area
                except Exception:
                    pxl_sseg_sh = pxl_sseg_sh.buffer(0)

                if pxl_sseg_sh.area <= min_area_px:
                    remove_reason.append('small pixel area {}'.format(
                        round(pxl_sseg_sh.area, 1)))
                    empty_aids.append(aid)
                    continue

            if min_geo_precision is not None:
                # TODO: could check this in UTM space instead of using rounding
                wgs_sseg = kwimage.Segmentation.coerce(
                    ann['segmentation_geos'])
                wgs_sseg_sh = wgs_sseg.to_multi_polygon().to_shapely()
                wgs_sseg_sh = shapely_round(wgs_sseg_sh,
                                            min_geo_precision).buffer(0)

                if wgs_sseg_sh.is_empty:
                    remove_reason.append('empty geos area')
                    empty_aids.append(aid)
                    continue

    if not empty_aids:
        print('Size filter: all annotations were valid')
    else:
        coco_dset.remove_annotations(empty_aids)
        keep_annots = coco_dset.annots()
        keep_tids = keep_annots.get('track_id', None)
        keep_track_lengths = ub.dict_hist(keep_tids)
        print(f'Size filter: removing {len(empty_aids)} annotations')
        print('keep_track_lengths = {}'.format(
            ub.repr2(keep_track_lengths, nl=1)))
        print(f'{len(keep_annots)=}')
        removal_reasons = ub.dict_hist(remove_reason)
        print('removal_reasons = {}'.format(ub.repr2(removal_reasons, nl=1)))

    return coco_dset


def ensure_videos(coco_dset):
    '''
    Ensure every image belongs to a video, even a dummy video
    and has a frame_index
    '''
    # HACK, TODO this is probably a kwcoco bug that needs fixed
    if 'videos' not in coco_dset.dataset:
        coco_dset.dataset['videos'] = list(coco_dset.index.videos.values())

    coco_dset.index.vidid_to_gids
    all_images = coco_dset.images()
    loose_flags = [vidid is None for vidid in all_images.get('video_id', None)]
    loose_imgs = all_images.compress(loose_flags)
    if loose_imgs:
        from watch.utils import util_time
        print(f'Warning: there are {len(loose_imgs)=} images without a video')
        # guess frame_index by date
        dt_guess = [(util_time.coerce_datetime(dc), gid)
                    for gid, dc in loose_imgs.lookup(
                        'date_captured', '1970-01-01', keepid=1).items()]
        loose_imgs = loose_imgs.take(ub.argsort(dt_guess))

        vidid = coco_dset.add_video('DEFAULT')
        for ix, gid in enumerate(loose_imgs):
            coco_dset.imgs[gid]['video_id'] = vidid
            coco_dset.imgs[gid]['frame_index'] = ix

        # This change invalidates the index, need to rebuild it.
        # TODO: add kwcoco logic to flag exactly what index needs to be rebuilt
        coco_dset._build_index()

    if __debug__:
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


def shapely_round(geom, precision):
    """
    References:
        https://gis.stackexchange.com/questions/188622
    """
    import shapely
    wkt = shapely.wkt.dumps(geom, rounding_precision=precision)
    geom2 = shapely.wkt.loads(wkt).simplify(0)
    return geom2


@profile
def normalize_phases(coco_dset,
                     use_viterbi=False,
                     t_probs=None,
                     e_probs=None,
                     baseline_keys={'salient'},
                     prediction_key='phase_transition_days'):
    '''
    Convert internal representation of phases to their IARPA standards as well
    as inserting a baseline guess for activity classification and removing
    empty tracks.

    HACK: add a Post Construction frame at the end of every track
    until we support partial sites

    The only remaining categories in the returned coco_dset should be:
        Site Preparation
        Active Construction
        Post Construction

    TODO make this a step in track_fn to take advantage of heatmap info?
    Example:
        >>> # test baseline guess
        >>> from watch.tasks.tracking.normalize import normalize_phases
        >>> from watch.demo import smart_kwcoco_demodata
        >>> dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps()
        >>> dset.cats[3]['name'] = 'salient'
        >>> dset.remove_categories([1,2])
        >>> assert dset.cats == {3: {'id': 3, 'name': 'salient'}}
        >>> # HACK, this shouldn't be needed?
        >>> # TODO file bug report
        >>> dset._build_index()
        >>> dset = normalize_phases(dset)
        >>> assert (dset.annots(trackid=1).cnames ==
        >>>     ((['Site Preparation'] * 10) +
        >>>      (['Active Construction'] * 9) +
        >>>      (['Post Construction'])))
        >>> # try again with smoothing
        >>> dset = normalize_phases(dset, use_viterbi=True)
    '''
    from watch.heuristics import CATEGORIES, CNAMES_DCT, SITE_SUMMARY_CNAME
    from watch.tasks.tracking import phase
    from collections import Counter

    for cat in CATEGORIES:
        coco_dset.ensure_category(**cat)
    baseline_keys = set(baseline_keys)
    unknown_cnames = coco_dset.name_to_cat.keys() - (
        {cat['name']
         for cat in CATEGORIES} | {SITE_SUMMARY_CNAME} | baseline_keys)
    if unknown_cnames:
        print(f'removing unknown categories {unknown_cnames}')
        coco_dset.remove_categories(unknown_cnames, keep_annots=False)

    cnames_to_remove = set(
        # negative examples, no longer needed
        #CNAMES_DCT['negative']['scored'] + CNAMES_DCT['negative']['unscored'] +
        CNAMES_DCT['negative']['unscored'] +
        # should have been consumed by track_fn, TODO more robust check
        [SITE_SUMMARY_CNAME])
    coco_dset.remove_categories(cnames_to_remove, keep_annots=False)

    cnames_to_replace = (
        # 'positive'
        set(CNAMES_DCT['positive']['unscored']) |
        # 'salient' or equivalent
        baseline_keys)

    # metrics-and-test-framework/evaluation.py:1684
    cnames_to_score = set(CNAMES_DCT['positive']['scored']) | set(
        CNAMES_DCT['negative']['scored'])

    allowed_cnames = cnames_to_replace | cnames_to_score
    have_cnames = set(coco_dset.name_to_cat)
    if not have_cnames.issubset(allowed_cnames):
        raise AssertionError(
            ub.paragraph(f'''
            Unhandled Class Names
            {allowed_cnames=}
            {have_cnames=}
            unknown={have_cnames - allowed_cnames}
            '''))

    # Transform phase labels of each track
    #

    log = Counter()
    for trackid, n_anns in ub.map_vals(
            len, coco_dset.index.trackid_to_aids).items():
        if n_anns > 1:

            annots = coco_dset.annots(trackid=trackid)
            has_missing_labels = bool(set(annots.cnames) - cnames_to_score)
            has_good_labels = bool(set(annots.cnames) - cnames_to_replace)
            if has_missing_labels and has_good_labels:
                # if we have partial coverage, interpolate from good labels
                log.update(['partial class labels'])
                coco_dset = phase.interpolate(coco_dset, trackid)
            else:
                if has_missing_labels:
                    # else, predict site prep for the first half of the track
                    # and then active construction for the second half with
                    # post construction on the last frame
                    log.update(['no class labels'])
                    coco_dset = phase.baseline(coco_dset, trackid)
                else:
                    log.update(['full class labels'])
    print('label status of tracks: ', log)

    #
    # Continue transforming phase labels, now with smoothing and deduping
    #

    old_cnames_dct = dict(zip(annots.aids, annots.cnames))

    for trackid, n_anns in ub.map_vals(
            len, coco_dset.index.trackid_to_aids).items():
        annots = coco_dset.annots(trackid=trackid)
        if n_anns > 1:

            if use_viterbi:

                # with xdev.embed_on_exception_context():
                smoothed_cnames = phase.class_label_smoothing(
                    annots.cnames, t_probs, e_probs)
                annots.set('category_id', [
                    coco_dset.name_to_cat[name]['id']
                    for name in smoothed_cnames
                ])

            # after viterbi, the sequence of phases is in the correct order

            # If the track ends before the end of the video, and the last frame
            # is not post construction, add another frame of post construction
            # coco_dset = phase.dedupe_background_anns(coco_dset, trackid)
            coco_dset = phase.ensure_post(coco_dset, trackid)

        annots = coco_dset.annots(trackid=trackid)
        is_empty = check_only_bg(annots.cnames)
        EMPTY_TRACK_BEHAVIOR = 'ignore'

        if is_empty:
            print(
                f'apply {EMPTY_TRACK_BEHAVIOR} to {trackid=} with cats {set(annots.cnames)}'
            )
            if EMPTY_TRACK_BEHAVIOR == 'delete':
                coco_dset.remove_annotations(annots.aids)
            elif EMPTY_TRACK_BEHAVIOR == 'flag':
                annots.set('status', 'system_rejected')
            elif EMPTY_TRACK_BEHAVIOR == 'revert':
                annots.cnames = list(
                    ub.dict_subset(old_cnames_dct, annots.aids).values())
            elif EMPTY_TRACK_BEHAVIOR == 'ignore':
                pass
            else:
                raise ValueError(EMPTY_TRACK_BEHAVIOR)

    #
    # Phase prediction - do it all at once for efficiency
    # TODO do this before viterbi so we can use this as an input in the future
    #

    ann_field = 'phase_transition_days'

    # exclude untracked annots which might be unrelated
    annots = coco_dset.annots(
        list(ub.flatten(coco_dset.index.trackid_to_aids.values())))

    if len(annots) > 0:
        has_prediction_heatmaps = all(
            kwcoco.FusedChannelSpec.coerce(prediction_key).as_set().issubset(
                coco_dset.coco_image(gid).channels.fuse().as_set())
            for gid in set(annots.gids))
        if has_prediction_heatmaps:
            phase_transition_days = phase.phase_prediction_heatmap(
                annots, coco_dset, prediction_key)
            annots.set(ann_field, phase_transition_days)
        else:
            for trackid in coco_dset.index.trackid_to_aids.keys():
                _annots = coco_dset.annots(trackid=trackid)
                phase_transition_days = phase.phase_prediction_baseline(_annots)
                _annots.set(ann_field, phase_transition_days)

        #
        # Fixup phase prediction
        #

        # TODO do something with transition preds for phases which were altered
        FIXUP_TRANSITION_PRED = 0
        if FIXUP_TRANSITION_PRED:
            n_diff_annots = sum(
                np.array(annots.cnames) == np.array(old_cnames_dct.values()))
            if n_diff_annots > 0:
                raise NotImplementedError

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


@profile
def normalize(
        coco_dset,
        track_fn,
        overwrite,
        polygon_fn='heatmaps_to_polys',
        gt_dset=None,
        viz_sc_bounds=False,
        viz_videos=False,
        use_viterbi=False,
        t_probs=None,  # for viterbi
        e_probs=None,  # for viterbi
        **track_kwargs):
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
        >>> d.images().set('channels', 'rgb')
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
        >>> assert set(coco_dset.annots().get('track_id')) == {1}
        >>> assert coco_dset.n_annots == n_existing_annots
        >>> coco_dset = dedupe_tracks(coco_dset)
        >>> assert set(coco_dset.annots().get('track_id')) == {1}
        >>> coco_dset = add_track_index(coco_dset)
        >>> assert coco_dset.annots().get('track_index') == [0,1,2]
        >>> coco_dset = normalize_phases(coco_dset, baseline_keys={'change'})
        >>> assert (coco_dset.annots().cnames ==
        >>> ['Site Preparation', 'Site Preparation', 'Post Construction'])
        >>> coco_dset = normalize_sensors(coco_dset)
        >>> assert (coco_dset.images().get('sensor_coarse') ==
        >>>     ['WorldView', 'Sentinel-2', 'Landsat 8'])
    '''
    viz_out_dir = ub.Path('_assets/tracking_visualization')

    def _normalize_annots(coco_dset, overwrite):
        coco_dset = dedupe_annots(coco_dset)
        coco_dset = add_geos(coco_dset, overwrite)
        coco_dset = remove_small_annots(coco_dset,
                                        min_area_px=0,
                                        min_geo_precision=None)
        # coco_dset._build_index()

        return coco_dset

    if len(coco_dset.anns) > 0:
        coco_dset = _normalize_annots(coco_dset, overwrite)
    coco_dset = ensure_videos(coco_dset)

    # apply tracks
    assert issubclass(track_fn, TrackFunction), 'must supply a valid track_fn!'
    tracker: TrackFunction = track_fn(polygon_fn=polygon_fn, **track_kwargs)
    out_dset = tracker.apply_per_video(coco_dset)

    # normalize and add geo segmentations
    out_dset = _normalize_annots(out_dset, overwrite=False)
    out_dset._build_index()
    print('After normalizing: track ids',
          set(out_dset.annots().get('track_id', None)))

    out_dset = dedupe_tracks(out_dset)
    out_dset = add_track_index(out_dset)

    if viz_sc_bounds:
        from watch.tasks.tracking.visualize import keys_to_score_sc, viz_track_scores
        track_cats = [SITE_SUMMARY_CNAME] + sorted(set(out_dset.annots().cnames))
        keys_to_score = keys_to_score_sc
        out_pth = viz_out_dir / 'track_scores.jpg'
        viz_track_scores(out_dset, track_cats, keys_to_score, out_pth)

    phase_args = [use_viterbi, t_probs, e_probs]
    if 'key' in track_kwargs:  # assume this is a baseline (saliency) key
        phase_args.append(set(track_kwargs['key']))
    out_dset = normalize_phases(out_dset, *phase_args)

    out_dset = normalize_sensors(out_dset)

    # HACK, ensure out_dset.index is up to date
    out_dset._build_index()

    if viz_videos:
        # visualize predicted sites with true sites
        from .visualize import visualize_videos
        visualize_videos(out_dset,
                         gt_dset,
                         viz_out_dir,
                         coco_dset_sc=track_kwargs.get('coco_dset_sc'))

    return out_dset
