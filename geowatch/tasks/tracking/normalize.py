import kwcoco
import kwimage
import warnings
import numpy as np
import ubelt as ub
from typing import Dict, List, Any
from geowatch.utils.kwcoco_extensions import TrackidGenerator
from geowatch.utils.kwcoco_extensions import warp_annot_segmentations_to_geos
from geowatch.tasks.tracking.abstract_classes import TrackFunction
from geowatch.tasks.tracking.utils import check_only_bg
try:
    from xdev import profile
except Exception:
    profile = ub.identity


def dedupe_annots(coco_dset):
    """
    Check for annotations with different aids that are the same geometry
    """
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
    else:
        print('No duplicates detected')
    return coco_dset


@profile
def remove_small_annots(coco_dset, min_area_px=1, min_geo_precision=6):
    """
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
        >>> from geowatch.tasks.tracking.normalize import remove_small_annots
        >>> from geowatch.demo import smart_kwcoco_demodata
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
        >>>     aux['warp_aux_to_img']['scale'] = np.array(
        >>>         aux['warp_aux_to_img'].get('scale', 1)) * scale_factor
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
        >>> assert np.allclose(annots.boxes.warp(aff.inv()).area,
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
    """
    if min_area_px is None:
        min_area_px = 0

    empty_aids = []
    remove_reason = []

    gid_iter = ub.ProgIter(coco_dset.index.imgs.keys(),
                           total=coco_dset.n_images,
                           desc='filter small annotations')
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
            ub.urepr(keep_track_lengths, nl=1)))
        print(f'{len(keep_annots)=}')
        removal_reasons = ub.dict_hist(remove_reason)
        print('removal_reasons = {}'.format(ub.urepr(removal_reasons, nl=1)))
        print(f'coco_dset.n_annots={coco_dset.n_annots}')

    return coco_dset


def ensure_videos(coco_dset):
    """
    Ensure every image belongs to a video, even a dummy video
    and has a frame_index
    """
    # HACK, TODO this is probably a kwcoco bug that needs fixed
    if 'videos' not in coco_dset.dataset:
        coco_dset.dataset['videos'] = list(coco_dset.index.videos.values())

    coco_dset.index.vidid_to_gids
    all_images = coco_dset.images()
    loose_flags = [vidid is None for vidid in all_images.get('video_id', None)]
    loose_imgs = all_images.compress(loose_flags)
    if loose_imgs:
        from kwutil import util_time
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
    """
    Assuming that videos are made of disjoint images, ensure that trackids
    are not shared by two tracks in different videos.
    """
    new_trackids = TrackidGenerator(coco_dset)

    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(track_id=trackid)

        # split each video into a separate track
        for idx, (vidid, aids) in enumerate(
                ub.group_items(
                    annots.aids,
                    coco_dset.images(annots.gids).get('video_id',
                                                      None)).items()):
            if idx > 0:
                coco_dset.annots(aids=aids).set('track_id', next(new_trackids))

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
    """
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
        >>> from geowatch.tasks.tracking.normalize import normalize_phases
        >>> from geowatch.tasks.tracking.normalize import normalize_phases
        >>> from geowatch.demo import smart_kwcoco_demodata
        >>> dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps()
        >>> dset.remove_categories([1,3,4,5])
        >>> dset.cats[2]['name'] = 'salient'
        >>> assert dset.cats == {2: {'id': 2, 'name': 'salient'}}
        >>> # HACK, this shouldn't be needed?
        >>> # TODO file bug report
        >>> dset._build_index()
        >>> dset = normalize_phases(dset)
        >>> assert (dset.annots(track_id=1).cnames ==
        >>>     ((['Site Preparation'] * 10) +
        >>>      (['Active Construction'] * 9) +
        >>>      (['Post Construction'])))
        >>> # try again with smoothing
        >>> dset = normalize_phases(dset, use_viterbi=True)
        >>> from geowatch.demo import smart_kwcoco_demodata
        >>> dset = smart_kwcoco_demodata.demo_kwcoco_with_heatmaps()
        >>> dset.remove_categories([1,3,4,5])
        >>> dset.cats[2]['name'] = 'salient'
        >>> assert dset.cats == {2: {'id': 2, 'name': 'salient'}}
        >>> # HACK, this shouldn't be needed?
        >>> # TODO file bug report
        >>> dset._build_index()
        >>> dset = normalize_phases(dset)
        >>> assert (dset.annots(track_id=1).cnames ==
        >>>     ((['Site Preparation'] * 10) +
        >>>      (['Active Construction'] * 9) +
        >>>      (['Post Construction'])))
        >>> # try again with smoothing
        >>> dset = normalize_phases(dset, use_viterbi=True)
    """
    from geowatch.heuristics import CATEGORIES, CNAMES_DCT, SITE_SUMMARY_CNAME
    from geowatch.tasks.tracking import phase
    from collections import Counter

    print('Normalizing phases')

    for cat in CATEGORIES:
        coco_dset.ensure_category(**cat)
    baseline_keys = set(baseline_keys)

    # Hack
    # baseline_keys.add('ac_salient')

    REMOVE_UNKNOWN_CATEGORIES = True
    if REMOVE_UNKNOWN_CATEGORIES:
        # Let's not do this here, and leave this for postprocessing.
        unknown_cnames = coco_dset.name_to_cat.keys() - (
            {cat['name']
             for cat in CATEGORIES} | {SITE_SUMMARY_CNAME} | baseline_keys)

        if unknown_cnames:
            print(f'removing unknown categories {unknown_cnames}')
            removed = coco_dset.remove_categories(unknown_cnames, keep_annots=False)
            print('removed = {}'.format(ub.urepr(removed, nl=1)))

        cnames_to_remove = set(
            # negative examples, no longer needed
            #CNAMES_DCT['negative']['scored'] + CNAMES_DCT['negative']['unscored'] +
            CNAMES_DCT['negative']['unscored'] +
            # should have been consumed by track_fn, TODO more robust check
            [SITE_SUMMARY_CNAME])
        removed = coco_dset.remove_categories(cnames_to_remove, keep_annots=False)
        if removed:
            print('Removing unknown categories')
            print(f'cnames_to_remove={cnames_to_remove}')
            print('removed = {}'.format(ub.urepr(removed, nl=1)))

    cnames_to_replace = (
        # 'positive'
        set(CNAMES_DCT['positive']['unscored']) |
        # 'salient' or equivalent
        baseline_keys)

    # metrics-and-test-framework/evaluation.py:1684
    cnames_to_score = set(CNAMES_DCT['positive']['scored']) | set(
        CNAMES_DCT['negative']['scored'])

    # TODO: maybe we should not try to enforce exepcted category names here?
    # Is there a reason that we need to?

    allowed_cnames = cnames_to_replace | cnames_to_score

    # Hack: add transient
    allowed_cnames.add('transient')
    # allowed_cnames.add('ac_salient')

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
    for trackid, annot_ids in coco_dset.index.trackid_to_aids.items():
        n_anns = len(annot_ids)
        if n_anns > 1:
            annots = coco_dset.annots(track_id=trackid)
            has_missing_labels = bool(set(annots.cnames) - cnames_to_score)
            has_good_labels = bool(set(annots.cnames) - cnames_to_replace)
            if has_missing_labels and has_good_labels:
                # if we have partial coverage, interpolate from good labels
                log.update(['partial class labels'])
                coco_dset = phase.interpolate(coco_dset, trackid, cnames_to_keep=cnames_to_score)
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

    annots = coco_dset.annots()
    old_cnames_dct = dict(zip(annots.aids, annots.cnames))

    for trackid, annot_ids in coco_dset.index.trackid_to_aids.items():
        n_anns = len(annot_ids)
        annots = coco_dset.annots(track_id=trackid)

        if n_anns > 1:

            if use_viterbi:
                # with xdev.embed_on_exception_context():
                smoothed_cnames = phase.class_label_smoothing(
                    annots.cnames,
                    transition_probs=t_probs,
                    emission_probs=e_probs
                )
                annots.set('category_id', [
                    coco_dset.name_to_cat[name]['id']
                    for name in smoothed_cnames
                ])

            # after viterbi, the sequence of phases is in the correct order

            # If the track ends before the end of the video, and the last frame
            # is not post construction, add another frame of post construction
            # coco_dset = phase.dedupe_background_anns(coco_dset, trackid)
            coco_dset = phase.ensure_post(coco_dset, trackid)

        annots = coco_dset.annots(track_id=trackid)
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
                _annots = coco_dset.annots(track_id=trackid)
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

    print(f'coco_dset.n_annots={coco_dset.n_annots}')
    print('Finished normalizing phases')
    return coco_dset


def dedupe_dates(coco_dset):
    """
    Ensure a tracked kwcoco file has at most 1 annot per track per date. [1]

    There are several potential ways to do this.
     - take highest-resolution sensor [currently done]
     - take image with best coverage (least nodata)
     - take latest time
     - majority-vote labels/average scores
     - average heatmaps before polygons are created

    Given that this probably has a minimal impact on scores, the safest method
    is chosen.

    References:
        [1] https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/63

    Example:
        >>> from geowatch.tasks.tracking.normalize import *  # NOQA
        >>> import geowatch
        >>> import kwarray
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True)
        >>> # Add 0-4 duplicate images to each video
        >>> rng = kwarray.ensure_rng(613544)
        >>> gids_to_duplicate = list(ub.flatten([rng.choice(gs, rng.randint(0, 4)) for gs in coco_dset.videos().images]))
        >>> for gid in gids_to_duplicate:
        >>>     img1 = ub.udict(coco_dset.index.imgs[gid]) - {'id'}
        >>>     img1['name'] = img1['name'] + '_duplicated'
        >>>     coco_dset.add_image(**img1)
        >>> coco_dset_with_dups = coco_dset.copy()
        >>> coco_dset_fixed = dedupe_dates(coco_dset.copy())
        >>> assert coco_dset_fixed.n_images < coco_dset_with_dups.n_images
    """
    from kwutil import util_time
    from geowatch import heuristics
    sensor_priority = heuristics.SENSOR_TRACK_PRIORITY
    for trackid in coco_dset.index.trackid_to_aids.keys():
        annots = coco_dset.annots(track_id=trackid)
        dates = [util_time.coerce_datetime(d).date() for d in annots.images.lookup('date_captured')]
        fixs = annots.images.lookup('frame_index')

        is_sorted = lambda arr: np.all(arr[:-1] <= arr[1:])  # noqa
        if not all(date_frame_sorted := (
                    is_sorted(dates),
                    is_sorted(fixs))):
            # this should never print
            print(f'WARNING: {trackid=} {date_frame_sorted=}')

    # remove full images instead of iterating over tracks for efficiency
    # not possible for some other removal methods, but it is for this one
    gids_to_remove = []
    for video_id in coco_dset.index.vidid_to_gids.keys():
        images = coco_dset.images(video_id=video_id)
        dates = [util_time.coerce_datetime(d).date() for d in images.lookup('date_captured')]
        dup_dates_to_idxs: Dict[Any, List[int]] = ub.find_duplicates(dates)
        # If we have any duplicates for a day, lookup their priorities and
        # remove all but the one with the highest priority.
        for date, dup_idxs in dup_dates_to_idxs.items():
            dup_images = images.take(dup_idxs)
            dup_sensors = dup_images.lookup('sensor_coarse')
            dup_priorities = [
                sensor_priority.get(s, -1) for s in dup_sensors]
            keep_idx = ub.argmax(dup_priorities)
            remove_gids = list(set(dup_images) - {dup_images[keep_idx]})
            print(f'removing {len(remove_gids)} dup imgs from {date} in video_id={video_id}')
            gids_to_remove.extend(remove_gids)

    # coco_dset.remove_annotations(aids_to_remove, verbose=1)
    coco_dset.remove_images(gids_to_remove, verbose=2)
    return coco_dset


@profile
def run_tracking_pipeline(
        coco_dset,
        track_fn,
        gt_dset=None,
        viz_out_dir=None,
        use_viterbi=False,
        sensor_warnings=True,
        # t_probs=None,  # for viterbi
        # e_probs=None,  # for viterbi
        **track_kwargs):
    """
    Driver function to apply all normalizations

    TODO:
        Rename this to something like run_tracker. This is the entry point to
        the main tracking pipeline.

    Example:
        >>> import kwcoco as kc
        >>> from geowatch.tasks.tracking.normalize import *
        >>> from geowatch.tasks.tracking.from_polygon import OverlapTrack
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
        >>> def _normalize_annots(coco_dset):
        >>>     coco_dset = dedupe_annots(coco_dset)
        >>>     coco_dset = remove_small_annots(coco_dset,
        >>>         min_geo_precision=None)
        >>>     return coco_dset
        >>> coco_dset = d.copy()
        >>> coco_dset = _normalize_annots(coco_dset)
        >>> assert coco_dset.anns == d.anns
        >>> coco_dset = ensure_videos(coco_dset)
        >>> assert coco_dset.index.vidid_to_gids[1] == coco_dset.imgs.keys()
        >>> n_existing_annots = coco_dset.n_annots
        >>> coco_dset = OverlapTrack().apply_per_video(coco_dset)
        >>> assert set(coco_dset.annots().get('track_id')) == {1}
        >>> assert coco_dset.n_annots == n_existing_annots
        >>> coco_dset = dedupe_tracks(coco_dset)
        >>> assert set(coco_dset.annots().get('track_id')) == {1}
        >>> coco_dset = normalize_phases(coco_dset, baseline_keys={'change'})
        >>> assert (coco_dset.annots().cnames ==
        >>> ['Site Preparation', 'Site Preparation', 'Post Construction'])
        >>> from geowatch import heuristics
        >>> coco_dset = heuristics.normalize_sensors(
        >>>     coco_dset, sensor_warnings=False, format='iarpa')
        >>> assert (coco_dset.images().get('sensor_coarse') ==
        >>>     ['WorldView', 'Sentinel-2', 'Landsat 8'])
    """
    import rich
    rich.print('[cyan]--- run_tracking_pipeline ---')

    DEBUG_JSON_SERIALIZABLE = 0
    if DEBUG_JSON_SERIALIZABLE:
        from kwutil.util_json import debug_json_unserializable

    if viz_out_dir is not None:
        viz_out_dir = ub.Path(viz_out_dir)
        viz_out_dir.ensuredir()

    def _normalize_annots(coco_dset):
        print(f'coco_dset.n_anns={coco_dset.n_annots}')
        coco_dset = dedupe_annots(coco_dset)
        warp_annot_segmentations_to_geos(coco_dset)
        coco_dset = remove_small_annots(coco_dset,
                                        min_area_px=0,
                                        min_geo_precision=None)
        print(f'coco_dset.n_anns={coco_dset.n_annots}')
        # coco_dset._build_index()

        return coco_dset

    if len(coco_dset.anns) > 0:
        coco_dset = _normalize_annots(coco_dset)
    coco_dset = ensure_videos(coco_dset)

    if DEBUG_JSON_SERIALIZABLE:
        debug_json_unserializable(coco_dset.dataset, 'Input to normalize: ')

    # apply tracks
    assert issubclass(track_fn, TrackFunction), 'must supply a valid track_fn!'

    # fixup the track kwargs when they come in via json
    for k, v in track_kwargs.items():
        if isinstance(v, str) and v.lower() == 'none':
            track_kwargs[k] = None

    if DEBUG_JSON_SERIALIZABLE:
        debug_json_unserializable(coco_dset.dataset, 'Before apply_per_video: ')

    if viz_out_dir is not None:
        track_kwargs['viz_out_dir'] = viz_out_dir

    tracker: TrackFunction = track_fn(**track_kwargs)
    print('track_kwargs = {}'.format(ub.urepr(track_kwargs, nl=1)))
    # print('{} {}'.format(tracker.__class__.__name__, ub.urepr(tracker.__dict__, nl=1)))
    rich.print(ub.urepr(tracker))
    # print('{} {}'.format(tracker.__class__.__name__, ub.urepr(tracker.__dict__, nl=1)))
    out_dset = tracker.apply_per_video(coco_dset)

    if DEBUG_JSON_SERIALIZABLE:
        debug_json_unserializable(out_dset.dataset, 'After apply_per_video: ')

    # normalize and add geo segmentations
    out_dset = _normalize_annots(out_dset)
    out_dset._build_index()

    print('After normalizing: track ids',
          set(out_dset.annots().get('track_id', None)))

    out_dset = dedupe_tracks(out_dset)

    if viz_out_dir is not None:
        from geowatch.tasks.tracking.visualize import viz_track_scores
        out_pth = viz_out_dir / 'track_scores.jpg'
        viz_track_scores(out_dset, out_pth, gt_dset)

    if isinstance(use_viterbi, str):
        parts = use_viterbi.split(',')
        assert len(parts) == 2
        t_probs, e_probs = parts
    else:
        t_probs = 'default'
        e_probs = 'default'

    phase_kw = dict(
        use_viterbi=use_viterbi,
        t_probs=t_probs,
        e_probs=e_probs
    )
    if 'key' in track_kwargs:  # assume this is a baseline (saliency) key
        k = track_kwargs['key']
        if ub.iterable(k):
            k = set(k)
        else:
            k = {k}
        phase_kw['baseline_keys'] = k

    print('Norm phases')
    out_dset = normalize_phases(out_dset, **phase_kw)

    if DEBUG_JSON_SERIALIZABLE:
        debug_json_unserializable(out_dset.dataset, 'After normalize_phases: ')

    from geowatch import heuristics
    print('Norm sensors')
    out_dset = heuristics.normalize_sensors(
        out_dset, sensor_warnings=sensor_warnings, format='iarpa')

    print(f'out_dset.n_annots={out_dset.n_annots}')

    # HACK, ensure out_dset.index is up to date
    out_dset._build_index()

    print('Dedup dates')
    out_dset = dedupe_dates(out_dset)

    if DEBUG_JSON_SERIALIZABLE:
        debug_json_unserializable(out_dset.dataset, 'Output of normalize: ')
    print(f'out_dset.n_annots={out_dset.n_annots}')

    # if viz_out_dir is not None:
    #     # visualize predicted sites with true sites
    #     # TODO think more about key handling
    #     from geowatch.tasks.tracking.visualize import visualize_videos
    #     from geowatch.tasks.tracking.utils import _validate_keys
    #     # from dataclasses import asdict
    #     fg, bg = _validate_keys(
    #         dict(tracker).get('key', None),
    #         dict(tracker).get('bg_key', None))
    #     keys = '|'.join([*fg, *bg])
    #     visualize_videos(out_dset, viz_out_dir / 'gif', keys, gt_dset)

    return out_dset


# Note: smooth transition while changing the name of "normalize"
# Backwards compatability:
normalize = run_tracking_pipeline
