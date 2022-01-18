#!/usr/bin/env python
import json
import kwarray
import kwcoco
import kwimage
import numpy as np
import os
import pandas as pd
import pathlib
import sklearn.metrics as skm
import ubelt as ub
import warnings
from watch.tasks.fusion import utils
from watch.utils import util_kwimage
from kwcoco.coco_evaluator import CocoSingleResult
from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
from kwcoco.metrics.confusion_measures import OneVersusRestMeasureCombiner
from kwcoco.metrics.confusion_measures import MeasureCombiner
# from kwcoco.metrics.confusion_measures import PerClass_Measures
from kwcoco.metrics.confusion_measures import Measures

try:
    from xdev import profile
except Exception:
    profile = ub.identity


@profile
def binary_confusion_measures(tn, fp, fn, tp):
    """
    Metrics derived from a binary confusion matrix

    TODO: just use kwcoco.metrics instead (or pycm)

    Example:
        >>> from watch.tasks.fusion.evaluate import *  # NOQA
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(4732890)
        >>> confusion_mats = np.vstack([
        >>>     # Corner cases
        >>>     np.array([list(map(int, '{:04b}'.format(x)))
        >>>               for x in range(16)]),
        >>>     # Random cases
        >>>     rng.randint(0, 100000, (32, 4)),
        >>> ])
        >>> tn, fp, fn, tp = confusion_mats.T
        >>> measures = binary_confusion_measures(tn, fp, fn, tp)
        >>> df = pd.DataFrame(measures)
        >>> print(df)
    """
    tn = np.atleast_1d(tn)
    fp = np.atleast_1d(fp)
    fn = np.atleast_1d(fn)
    tp = np.atleast_1d(tp)

    with warnings.catch_warnings():
        # It is very possible that we will divide by zero in this func
        warnings.filterwarnings('ignore', message='invalid .* true_divide')
        warnings.filterwarnings('ignore', message='invalid value')

        real_pos = fn + tp  # number of real positives
        real_neg = fp + tn  # number of real negatives

        total = real_pos + real_neg

        pred_pos = (fp + tp)  # number of predicted positives
        pred_neg = (fn + tn)  # number of predicted negatives

        pred_correct = tp + tn  # number of correct predictions

        # Error / Success Rates
        # https://en.wikipedia.org/wiki/Confusion_matrix
        # (Ensure denominator parts are non-zero)
        p_denom = real_pos.copy()
        p_denom[p_denom == 0] = 1
        n_denom = real_neg.copy()
        n_denom[n_denom == 0] = 1
        tpr = tp / p_denom  # recall
        tnr = tn / n_denom  # specificity
        fpr = fp / n_denom  # fall-out
        fnr = fn / p_denom  # miss-rate

        # predictive values
        pnv_denom = pred_neg.copy()
        pnv_denom[pnv_denom == 0] = 1
        ppv_denom = pred_pos.copy()
        ppv_denom[ppv_denom == 0] = 1
        ppv = tp / ppv_denom  # precision
        npv = tn / pnv_denom  # precision, but for negatives

        # Adjusted predictive values
        # https://www.researchgate.net/publication/228529307_Evaluation_From_Precision_Recall_and_F-Factor_to_ROC_Informedness_Markedness_Correlation
        bm = tpr + tnr - 1  # (bookmaker) informedness
        mk = ppv + npv - 1  # markedness

        # Summary statistics
        # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        fdr  = 1 - ppv  # false discovery rate
        fmr  = 1 - npv  # false ommision rate (for)
        # Note: when there are no true negatives, this goes to zero
        mcc = np.sqrt(ppv * tpr * tnr * npv) - np.sqrt(fdr * fnr * fpr * fmr)

        # https://erotemic.wordpress.com/2019/10/23/closed-form-of-the-mcc-when-tn-inf/
        g1 = np.sqrt(ppv * tpr)

        f1_numer = (2 * ppv * tpr)
        f1_denom = (ppv + tpr)
        f1_denom[f1_denom == 0] = 1
        f1 = f1_numer / f1_denom

        total_denom = total.copy()
        total_denom[total_denom == 0] = 1
        acc = pred_correct / total_denom

        info = {}

        info['tn'] = tn
        info['tp'] = tp
        info['fn'] = fn
        info['fp'] = fp

        info['real_pos'] = real_pos  # number of real positives
        info['real_neg'] = real_neg  # number of real negatives
        info['pred_pos'] = pred_pos  # number of predicted positives
        info['pred_neg'] = pred_neg  # number of predicted negatives
        info['total'] = total  # total cases

        info['tpr'] = tpr  # sensitivity, recall, hit rate, pd, or true positive rate (TPR)
        info['tnr'] = tnr  # specificity, selectivity or true negative rate (TNR)
        info['fnr'] = fnr  # miss rate or false negative rate (FNR)
        info['fpr'] = fpr  # false-alarm-rate, far, fall-out or false positive rate (FPR)

        info['ppv'] = ppv  # precision, positive predictive value (PNR)
        info['npv'] = npv  # negative predictive value (NPV)

        info['bm'] = bm  # (bookmaker) informedness
        info['mk'] = mk  # markedness

        info['f1'] = f1
        info['g1'] = g1
        info['mcc'] = mcc
        info['acc'] = acc

    return info


@profile
def associate_images(true_coco, pred_coco):
    # TODO: robust image/video association (see kwcoco eval)
    common_vidnames = set(true_coco.index.name_to_video) & set(pred_coco.index.name_to_video)

    def image_keys(dset, gids):
        # Generate image "keys" that should be compatible between datasets
        for gid in gids:
            img = dset.imgs[gid]
            if img.get('file_name', None) is None:
                yield img['name']
            else:
                yield img['file_name']

    all_gids1 = list(true_coco.imgs.keys())
    all_gids2 = list(pred_coco.imgs.keys())
    all_keys1 = list(image_keys(true_coco, all_gids1))
    all_keys2 = list(image_keys(pred_coco, all_gids2))
    key_to_gid1 = ub.dzip(all_keys1, all_gids1)
    key_to_gid2 = ub.dzip(all_keys2, all_gids2)
    gid_to_key1 = ub.invert_dict(key_to_gid1)
    gid_to_key2 = ub.invert_dict(key_to_gid2)

    video_matches = []

    all_match_gids1 = set()
    all_match_gids2 = set()

    for vidname in common_vidnames:
        video1 = true_coco.index.name_to_video[vidname]
        video2 = pred_coco.index.name_to_video[vidname]
        vidid1 = video1['id']
        vidid2 = video2['id']
        gids1 = true_coco.index.vidid_to_gids[vidid1]
        gids2 = pred_coco.index.vidid_to_gids[vidid2]
        keys1 = ub.oset(ub.take(gid_to_key1, gids1))
        keys2 = ub.oset(ub.take(gid_to_key2, gids2))
        match_keys = ub.oset(keys1) & ub.oset(keys2)
        match_gids1 = list(ub.take(key_to_gid1, match_keys))
        match_gids2 = list(ub.take(key_to_gid2, match_keys))
        all_match_gids1.update(match_gids1)
        all_match_gids2.update(match_gids2)
        video_matches.append({
            'vidname': vidname,
            'match_gids1': match_gids1,
            'match_gids2': match_gids2,
        })

    unmatched_gid_to_key1 = ub.dict_diff(gid_to_key1, all_match_gids1)
    unmatched_gid_to_key2 = ub.dict_diff(gid_to_key2, all_match_gids2)

    remain_keys = set(unmatched_gid_to_key1.values()) & set(unmatched_gid_to_key2.values())
    remain_gids1 = [key_to_gid1[key] for key in remain_keys]
    remain_gids2 = [key_to_gid2[key] for key in remain_keys]

    image_matches = {
        'match_gids1': remain_gids1,
        'match_gids2': remain_gids2,
    }
    return video_matches, image_matches


@profile
def single_image_segmentation_metrics(true_coco, pred_coco, gid1, gid2):
    img1 = true_coco.imgs[gid1]
    vidid1 = true_coco.imgs[gid1]['video_id']
    video1 = true_coco.index.videos[vidid1]
    shape = (img1['height'], img1['width'])

    row = {
        'true_gid': gid1,
        'pred_gid': gid2,
        'video': video1['name'],
    }
    info = {
        'row': row,
    }

    # Create a truth "panoptic segmentation" style mask
    true_foreground_canvas = np.zeros(shape, dtype=np.uint8)
    weight_foreground_canvas = np.ones(shape, dtype=np.float32)
    weight_classes_canvas = np.ones(shape, dtype=np.float32)
    catname_to_true = {}

    true_dets = true_coco.annots(gid=gid1).detections

    from watch.tasks.fusion import heuristics
    ignore_classes = heuristics.IGNORE_CLASSNAMES
    background_classes = heuristics.BACKGROUND_CLASSES
    undistinguished_classes = heuristics.UNDISTINGUISHED_CLASSES

    true_cidxs = true_dets.data['class_idxs']
    true_ssegs = true_dets.data['segmentations']

    for true_sseg, true_cidx in zip(true_ssegs, true_cidxs):
        catname = true_dets.classes.idx_to_node[true_cidx]
        if catname in ignore_classes:
            # "classless" foreground binary problem
            weight_foreground_canvas = true_sseg.fill(weight_foreground_canvas, value=0)
        elif catname not in background_classes:
            if catname in undistinguished_classes:
                # Ignore for class scores, but not saliency / foreground
                weight_classes_canvas = true_sseg.fill(weight_classes_canvas, value=0)
            else:
                true_foreground_canvas = true_sseg.fill(true_foreground_canvas, value=1)
                if catname not in catname_to_true:
                    catname_to_true[catname] = np.zeros(shape, dtype=np.float32)
                catname_to_true[catname] = true_sseg.fill(catname_to_true[catname], value=1)

    weight_classes_canvas = np.minimum(weight_classes_canvas, weight_foreground_canvas)

    SCORE_CHANNEL_CLASS_PROBS = True
    if SCORE_CHANNEL_CLASS_PROBS:
        from watch import heuristics
        predicted_classes = []
        true_classes = list(true_coco.object_categories())
        pred_gid = gid2
        pred_coco_img = pred_coco.coco_image(pred_gid)
        for stream in pred_coco_img.channels.streams():
            have = stream.intersection(true_classes)
            predicted_classes.extend(have.parsed)
        classes_of_interest = ub.oset(predicted_classes) - (
            heuristics.BACKGROUND_CLASSES | heuristics.IGNORE_CLASSNAMES |
            {'Unknown'})

        for catname in classes_of_interest:
            if catname not in catname_to_true:
                # print('Have pred, but no true for: {}'.format(catname))
                catname_to_true[catname] = np.zeros(shape, dtype=np.float32)
        try:
            # handle multiclass case
            pred_chan_of_interest = '|'.join(classes_of_interest)
            delayed_probs = pred_coco_img.delay(pred_chan_of_interest)
            class_probs = delayed_probs.finalize(as_xarray=True)
            cx_to_binvecs = {}
            for cx, cname in enumerate(classes_of_interest):
                is_true = catname_to_true[cname]
                score = class_probs.loc[:, :, cname].data.copy()
                salient_invalid_mask = np.isnan(score)
                weights = weight_foreground_canvas.copy()
                weights[salient_invalid_mask] = 0
                score[salient_invalid_mask] = 0
                # pred_score.data.ravel()
                bin_data = {
                    # is_true denotes if the true class of the item is the
                    # category of interest.
                    'is_true': is_true.ravel(),
                    'pred_score': score.ravel(),
                    'weight': weights.ravel(),
                }
                bin_data = kwarray.DataFrameArray(bin_data)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes_of_interest)
                bin_measures = bin_cfsn.measures()
                bin_measures.summary()
                cx_to_binvecs[cname] = bin_cfsn
            from kwcoco.metrics.confusion_vectors import OneVsRestConfusionVectors
            ovr_cfns = OneVsRestConfusionVectors(cx_to_binvecs, classes_of_interest)
            ovr_measures = ovr_cfns.measures()
            row['mAP'] = ovr_measures['mAP']
            row['mAUC'] = ovr_measures['mAUC']
            info.update({
                # TODO: data for visualization
                'ovr_measures': ovr_measures,
            })
        except Exception:
            raise

    SCORE_CHANNEL_SALIENT_PROBS = 1
    if SCORE_CHANNEL_SALIENT_PROBS:
        try:
            # TODO: consolidate this with above class-specific code
            # pred_channel = 'change'
            pred_channel = 'salient'
            salient_prob = pred_coco_img.delay(pred_channel, space='image').finalize()[..., 0]
            salient_invalid_mask = np.isnan(salient_prob)
            salient_prob[salient_invalid_mask] = 0
            weight_foreground_canvas[salient_invalid_mask] = 0

            bin_cfns = BinaryConfusionVectors(kwarray.DataFrameArray({
                'is_true': true_foreground_canvas.ravel(),
                'pred_score': salient_prob.ravel(),
                'weight': weight_foreground_canvas.ravel().astype(np.float32),
            }))
            salient_measures = bin_cfns.measures()
            row.update(ub.dict_isect(salient_measures.summary(), {'fg_ap', 'fg_auc', 'fg_max_f1'}))

            info.update({
                'salient_measures': salient_measures,
                'salient_prob': salient_prob,
                'salient_invalid_mask': salient_invalid_mask,
            })
        except Exception:
            pass

    # TODO: look at the category ranking at each pixel by score.
    # Is there a generalization of a confusion matrix to a ranking tensor?

    # SCORE PREDICTED ANNOTATIONS
    # Create a pred "panoptic segmentation" style mask
    pred_foreground_canvas = np.zeros(shape, dtype=np.uint8)
    pred_dets = pred_coco.annots(gid=gid2).detections
    for pred_sseg in pred_dets.data['segmentations']:
        pred_foreground_canvas = pred_sseg.fill(pred_foreground_canvas, value=1)

    # TODO: need to consider the fact that these annotations are
    # multi-class, and have scores, we may want to evaluate the raw
    # probability map in order to find a decent operating point.
    # We also need to hook into the real IAPRA metrics.

    # FIXME: for now this is just binary change
    y_true = true_foreground_canvas.ravel()
    y_pred = pred_foreground_canvas.ravel()
    sample_weight = weight_foreground_canvas.ravel()
    mat = skm.confusion_matrix(y_true, y_pred, labels=np.array([0, 1]),
                               sample_weight=sample_weight)

    info.update({
        'mat': mat,
        'pred_foreground_canvas': pred_foreground_canvas,
        'true_foreground_canvas': true_foreground_canvas,
        'weight_foreground_canvas': weight_foreground_canvas,
    })
    tn, fp, fn, tp = mat.ravel()
    fg_bin_cfsn = binary_confusion_measures(tn, fp, fn, tp)
    fg_bin_cfsn = ub.map_vals(lambda x: x.item(), fg_bin_cfsn)
    row.update(fg_bin_cfsn)
    return info


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


@profile
def dump_chunked_confusion(true_coco, pred_coco, chunk_info, plot_dpath):
    """
    Draw a a sequence of true/pred image predictions
    """
    colors = ['white', 'green', 'yellow', 'red']
    color_labels = ['TN', 'TP', 'FN', 'FP']
    color_lut = np.array([kwimage.Color(c).as255() for c in colors])

    # Make a legend
    color01_lut = color_lut / 255.0
    label_to_color = ub.dzip(color_labels, color01_lut)
    legend_img = _memo_legend(label_to_color)
    legend_img = kwimage.ensure_uint255(legend_img)

    # Draw predictions on each frame
    parts = []
    frame_nums = []
    true_gids = []
    unique_vidnames = set()
    for info in chunk_info:
        row = info['row']
        unique_vidnames.add(row['video'])

        true_gid = row['true_gid']

        true_img = true_coco.index.imgs[true_gid]
        frame_index = true_img['frame_index']
        frame_nums.append(frame_index)
        true_gids.append(true_gid)

        pred_foreground_canvas = info['pred_foreground_canvas']
        true_foreground_canvas = info['true_foreground_canvas']
        confusion_idxs = utils.confusion_image(pred_foreground_canvas, true_foreground_canvas)
        confusion_image = color_lut[confusion_idxs]

        image_text = f'{frame_index} - gid = {true_gid}'

        confusion_image = kwimage.ensure_uint255(confusion_image)

        header = util_kwimage.draw_header_text(
            image=confusion_image, text=image_text, color='white', stack=False)

        vert_parts = [
            header,
            confusion_image
        ]
        # confusion_image = kwimage.draw_text_on_image(
        #     confusion_image, image_text, org=(1, 1), valign='top',
        #     color='white', border={'color': 'black'})

        # TODO:
        # Can we show the reference image?
        # TODO:
        # Show the datetime on the top of the image (and the display band?)

        real_image_norm = None
        real_image_int = None

        TRY_IMREAD = 1
        if TRY_IMREAD:
            true_coco_img = true_coco.coco_image(true_img['id'])
            avali_chans = {p2 for p1 in true_coco_img.channels.spec.split(',') for p2 in p1.split('|')}
            chosen_viz_channs = None
            if len(avali_chans & {'red', 'green', 'blue'}) == 3:
                chosen_viz_channs = 'red|green|blue'
            elif len(avali_chans & {'r', 'g', 'b'}) == 3:
                chosen_viz_channs = 'r|g|b'
            else:
                chosen_viz_channs = true_coco_img.primary_asset()['channels']
            try:
                real_image = true_coco_img.delay(chosen_viz_channs, space='image').finalize()
                real_image_norm = kwimage.normalize_intensity(real_image)
                # Make into gray
                real_image_int = kwimage.ensure_uint255(real_image_norm)
            except Exception:
                pass

        TRY_SOFT = 1
        salient_prob = None
        if TRY_SOFT:
            salient_prob = info.get('salient_prob', None)
            salient_invalid_mask = info.get('salient_invalid_mask', None)
            if salient_prob is not None:
                heatmap = kwimage.make_heatmask(salient_prob, with_alpha=0.5)
                heatmap[salient_invalid_mask] = 0
                heatmap_int = kwimage.ensure_uint255(heatmap[..., 0:3])
                vert_parts.append(heatmap_int)
                if real_image_norm is not None:
                    overlaid = kwimage.overlay_alpha_layers([heatmap, real_image_norm.mean(axis=2)])
                    overlaid = kwimage.ensure_uint255(overlaid[..., 0:3])
                    vert_parts.append(overlaid)

        if real_image_int is not None and salient_prob is None:
            vert_parts.append(real_image_int)

        vert_stack = kwimage.stack_images(vert_parts, axis=0)
        parts.append(vert_stack)

    max_frame = min(frame_nums)
    min_frame = max(frame_nums)
    max_gid = max(true_gids)
    min_gid = min(true_gids)

    if max_frame == min_frame:
        frame_part = f'{min_frame}'
    else:
        frame_part = f'{min_frame}-{max_frame}'

    if max_gid == min_gid:
        gid_part = f'{min_gid}'
    else:
        gid_part = f'{min_gid}-{max_gid}'
    vidname_part = '_'.join(list(unique_vidnames))

    plot_fname = f'{vidname_part}-{frame_part}-{gid_part}.png'
    plot_canvas = kwimage.stack_images(parts, axis=1, overlap=-10)

    plot_canvas = kwimage.stack_images(
        [plot_canvas, legend_img], axis=1, overlap=-10)

    header = kwimage.draw_header_text(
        {'width': plot_canvas.shape[1]}, plot_fname)
    plot_canvas = kwimage.stack_images([header, plot_canvas], axis=0)

    plot_dpath = pathlib.Path(str(plot_dpath))
    plot_fpath = plot_dpath / plot_fname
    kwimage.imwrite(str(plot_fpath), plot_canvas)


@profile
def evaluate_segmentations(true_coco, pred_coco, eval_dpath=None, draw='auto'):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/tasks/fusion/evaluate.py evaluate_segmentations

    Example:
        >>> from watch.tasks.fusion.evaluate import *  # NOQA
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> import kwcoco
        >>> true_coco = kwcoco.CocoDataset.demo('vidshapes2')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>>     'with_probs': True,
        >>>     'with_heatmaps': True,
        >>> }
        >>> # TODO: it would be nice to demo the soft metrics
        >>> # functionality by adding "salient_prob" or "class_prob"
        >>> # auxiliary channels to this demodata.
        >>> pred_coco = perterb_coco(true_coco, **kwargs)
        >>> eval_dpath = ub.ensure_app_cache_dir('watch/tests/fusion_eval')
        >>> print('eval_dpath = {!r}'.format(eval_dpath))
        >>> evaluate_segmentations(true_coco, pred_coco, eval_dpath)
    """
    # Extract metadata about the predictions to persist
    meta = {}
    meta['info'] = info = []
    for item in pred_coco.dataset['info']:
        if item.get('type', None) == 'process':
            proc_name = item.get('properties', {}).get('name', None)
            if proc_name == 'watch.tasks.fusion.predict':
                package_fpath = item['properties']['args'].get('package_fpath')
                if 'title' not in item:
                    item['title'] = pathlib.Path(package_fpath).stem
                meta['title'] = item['title']
                info.append(item)

    required_marked = 'auto'  # parametarize
    if required_marked == 'auto':
        # In "auto" mode dont require marks if all images are unmarked,
        # otherwise assume that we should restirct to marked images
        required_marked = any(pred_coco.images().lookup('has_predictions', False))

    video_matches, image_matches = associate_images(true_coco, pred_coco)
    rows = []
    chunk_size = 5
    thresh_bins = 256 * 256

    if draw == 'auto':
        draw = bool(eval_dpath is not None)

    if eval_dpath is None:
        plot_dpath = None
    else:
        plot_dpath = pathlib.Path(eval_dpath) / 'plots'
        plot_dpath.mkdir(exist_ok=True, parents=True)

    fg_measure_combiner = MeasureCombiner(thresh_bins=thresh_bins)
    ovr_measure_combiner = OneVersusRestMeasureCombiner(thresh_bins=thresh_bins)

    # combiner = ovr_measure_combiner.catname_to_combiner['Site Preparation']
    # for x in combiner.queue:
    #     d = x.asdict()
    #     if np.any(d['tp_count']) > 0:
    #         thr = x['thresholds']
    #         tp = d['tp_count']
    #         fp = d['fp_count']
    #         tp_diff = np.diff(tp)
    #         fp_diff = np.diff(fp)
    #         flags = np.diff(tp) > 0
    #         dtp_pts = tp_diff[flags]
    #         fp_pts = fp[:-1][flags]
    #         # np.diff(fp_pts) == 0
    #         thr_pts = thr[:-1][flags]
    #         print(kwarray.stats_dict(thr_pts * dtp_pts)['min'])

    #         np.diff(thr[:-1][flags])
    #         print(d['tp_count'])
    #     print(t)
    # measures = Measures.combine(combiner.queue, precision=15)
    # measures.reconstruct()
    # ovr_measure_combiner.submit(ovr_measures)
    # ovr_measure_combiner.combine()
    # ovr_measure_combiner.finalize()

    total_images = 0

    if required_marked:
        for video_match in video_matches:
            gids1 = video_match['match_gids1']
            gids2 = video_match['match_gids2']
            flags = pred_coco.images(gids2).lookup('has_predictions', False)
            video_match['match_gids1'] = list(ub.compress(gids1, flags))
            video_match['match_gids2'] = list(ub.compress(gids2, flags))
            total_images += len(gids1)
        gids1 = image_matches['match_gids1']
        gids2 = image_matches['match_gids2']
        flags = pred_coco.images(gids2).lookup('has_predictions', False)
        image_matches['match_gids1'] = list(ub.compress(gids1, flags))
        image_matches['match_gids2'] = list(ub.compress(gids2, flags))
        total_images += len(gids1)

    prog = ub.ProgIter(total=total_images, desc='scoring', adjust=False, freq=1)
    prog.begin()

    # Handle images in videos
    for video_match in video_matches:
        prog.set_extra('comparing ' + video_match['vidname'])
        gids1 = video_match['match_gids1']
        gids2 = video_match['match_gids2']
        if required_marked:
            flags = pred_coco.images(gids2).lookup('has_predictions', False)
            gids1 = list(ub.compress(gids1, flags))
            gids2 = list(ub.compress(gids2, flags))

        pairs = list(zip(gids1, gids2))
        for chunk in ub.chunks(pairs, chunk_size):
            chunk_info = []

            for gid1, gid2 in chunk:
                # xxx = set(true_coco.annots(gid=gid1).lookup('category_id'))
                info = single_image_segmentation_metrics(
                    true_coco, pred_coco, gid1, gid2)
                rows.append(info['row'])

                ovr_measures = info.get('ovr_measures', None)
                fg_measures = info.get('salient_measures', None)
                if fg_measures is not None:
                    fg_measure_combiner.submit(fg_measures)
                if ovr_measures is not None:
                    ovr_measure_combiner.submit(ovr_measures)

                if draw:
                    chunk_info.append(info)
                prog.update()

            # Reduce measures over the chunk
            if fg_measure_combiner.queue_size > chunk_size:
                fg_measure_combiner.combine()
            if ovr_measure_combiner.queue_size > chunk_size:
                ovr_measure_combiner.combine()

            if draw:
                dump_chunked_confusion(
                    true_coco, pred_coco, chunk_info, plot_dpath)

    # ovr_measure_combiner.catname_to_combiner['Site Preparation'].queue

    # Handle standalone images
    gids1 = image_matches['match_gids1']
    gids2 = image_matches['match_gids2']

    for gid1, gid2 in zip(gids1, gids2):
        info = single_image_segmentation_metrics(
            true_coco, pred_coco, gid1, gid2)
        ovr_measures = info.get('ovr_measures', None)
        fg_measures = info.get('salient_measures', None)
        if fg_measures is not None:
            fg_measure_combiner.submit(fg_measures)
        if ovr_measures is not None:
            ovr_measure_combiner.submit(ovr_measures)
        rows.append(info['row'])
        if draw:
            chunk_info = [info]
            dump_chunked_confusion(
                true_coco, pred_coco, chunk_info, plot_dpath)
        prog.update()
        # Reduce measures over the chunk
        if fg_measure_combiner.queue_size > chunk_size:
            fg_measure_combiner.combine()
        if ovr_measure_combiner.queue_size > chunk_size:
            ovr_measure_combiner.combine()

    prog.end()

    # import xdev
    # xdev.embed()

    # Reduce measures over the chunk
    print('Finalize fg measures')
    fg_combo_measures = fg_measure_combiner.finalize()
    print('Finalize ovr measures')
    ovr_combo_measure_dict = ovr_measure_combiner.finalize()
    ovr_combo_measures = ovr_combo_measure_dict['perclass']

    # if 0:
    #     m = Measures.combine(queue, thresh_bins=512 * 512).reconstruct()
    #     print(m)
    #     print(m.counts().pandas())
    #     m = Measures.combine(queue, thresh_bins=None).reconstruct()
    #     print(m)
    #     print(m.counts().pandas())

    if 0:
        print(fg_combo_measures.counts().pandas())
        for k, bm in ovr_combo_measures.items():
            print(bm.counts().pandas())

    # Use the SingleResult container (TODO: better API)
    result = CocoSingleResult(
        fg_combo_measures, ovr_combo_measures, None, meta)
    print('result = {}'.format(result))

    if fg_combo_measures is not None:
        if eval_dpath is not None:
            curve_dpath = pathlib.Path(eval_dpath) / 'curves'
            curve_dpath.mkdir(exist_ok=True, parents=True)

            fg_combo_measures['meta'] = meta
            title = meta.get('title', '')
            measure_info = fg_combo_measures.__json__()
            measures_fpath = curve_dpath / 'measures.json'

            print('Dump measures_fpath={}'.format(measures_fpath))
            with open(measures_fpath, 'w') as file:
                measure_info['meta'] = meta
                json.dump(measure_info, file)

            measures_fpath2 = curve_dpath / 'measures2.json'
            print('Dump measures_fpath2={}'.format(measures_fpath2))
            result.dump(os.fspath(measures_fpath2))

            if draw:
                import kwplot
                # kwplot.autompl()
                with kwplot.BackendContext('agg'):
                    fig = kwplot.figure(doclf=True)

                    print('Dump fg figures')
                    fg_combo_measures.summary_plot(fnum=1, title=title)
                    fig = kwplot.autoplt().gcf()
                    fig.savefig(str(curve_dpath / 'fg_summary.png'))

                    print('Dump ovr figures')
                    result.dump_figures(curve_dpath, expt_title=title)

    df = pd.DataFrame(rows)
    print(df)

    summary = binary_confusion_measures(
        df.tn.sum(), df.fp.sum(), df.fn.sum(), df.tp.sum())
    summary = ub.map_vals(lambda x: x.item() if hasattr(x, 'item') else x, summary)

    if ovr_combo_measure_dict is not None:
        summary['mAP'] = ovr_combo_measure_dict['mAP']
        summary['mAUC'] = ovr_combo_measure_dict['mAUC']

    if fg_combo_measures is not None:
        summary['ap'] = fg_combo_measures['ap']
        summary['auc'] = fg_combo_measures['auc']
        summary['max_f1'] = fg_combo_measures['max_f1']

    print('summary = {}'.format(ub.repr2(
        summary, nl=1, precision=4, align=':', sort=0)))
    print('eval_dpath = {!r}'.format(eval_dpath))
    return df


def _redraw_measures(eval_dpath):
    """
    hack helper for developer, not critical
    """
    curve_dpath = pathlib.Path(eval_dpath) / 'curves'
    measures_fpath = curve_dpath / 'measures.json'
    with open(measures_fpath, 'r') as file:
        state = json.load(file)
        fg_combo_measures = Measures.from_json(state)
        meta = fg_combo_measures.get('meta', [])
        title = ''
        if meta is not None:
            if isinstance(meta, list):
                # Old
                for item in meta:
                    title = item.get('title', title)
            else:
                title = meta.get('title', title)
        import kwplot
        with kwplot.BackendContext('agg'):
            fg_combo_measures.summary_plot(fnum=1, title=title)
            fig = kwplot.autoplt().gcf()
            fig.savefig(str(curve_dpath / 'summary_redo.png'))


def make_evaluate_config(cmdline=False, **kwargs):
    from watch.utils.configargparse_ext import ArgumentParser
    parser = ArgumentParser(
        add_config_file_help=False,
        description='Evaluation script for change/segmentation task',
        auto_env_var_prefix='WATCH_FUSION_EVAL_',
        add_env_var_help=True,
        formatter_class='defaults',
        config_file_parser_class='yaml',
        args_for_setting_config_path=['--config'],
        args_for_writing_out_config_file=['--dump'],
    )
    parser.add_argument('--true_dataset', '--test_dataset', help='path to the groundtruth dataset')
    parser.add_argument('--pred_dataset', help='path to the predicted dataset')
    parser.add_argument('--eval_dpath', help='path to dump results')
    parser.add_argument('--draw', default='auto', help='flag to draw or not')
    parser.set_defaults(**kwargs)
    default_args = None if cmdline else []
    args, _ = parser.parse_known_args(default_args)
    return args


def main(cmdline=True, **kwargs):
    args = make_evaluate_config(cmdline=cmdline, **kwargs)
    true_coco = kwcoco.CocoDataset.coerce(args.true_dataset)
    pred_coco = kwcoco.CocoDataset.coerce(args.pred_dataset)
    eval_dpath = args.eval_dpath
    from scriptconfig.smartcast import smartcast
    draw = smartcast(args.draw)
    evaluate_segmentations(true_coco, pred_coco, eval_dpath, draw=draw)


if __name__ == '__main__':
    import xdev
    xdev.make_warnings_print_tracebacks()
    main()
