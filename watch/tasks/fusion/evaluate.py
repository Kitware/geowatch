import kwcoco
import kwimage
import kwarray
import sklearn.metrics as skm
import pandas as pd
import numpy as np
import pathlib
import ubelt as ub
from watch.tasks.fusion import utils
from watch.utils import util_kwimage
from watch.utils.kwcoco_extensions import CocoImage

try:
    from xdev import profile
except Exception:
    profile = ub.identity


@profile
def binary_confusion_measures(tn, fp, fn, tp):
    """
    Metrics derived from a binary confusion matrix

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

    import warnings
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
        keys1 = ub.oset(ub.dict_take(gid_to_key1, gids1))
        keys2 = ub.oset(ub.dict_take(gid_to_key2, gids2))
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
    from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
    import kwarray
    img1 = true_coco.imgs[gid1]
    vidid1 = true_coco.imgs[gid1]['video_id']
    video1 = true_coco.index.videos[vidid1]
    shape = (img1['height'], img1['width'])

    # Create a truth "panoptic segmentation" style mask
    true_canvas = np.zeros(shape, dtype=np.uint8)
    true_dets = true_coco.annots(gid=gid1).detections
    for true_sseg in true_dets.data['segmentations']:
        true_canvas = true_sseg.fill(true_canvas, value=1)

    # Create a pred "panoptic segmentation" style mask
    pred_canvas = np.zeros(shape, dtype=np.uint8)
    pred_dets = pred_coco.annots(gid=gid2).detections
    for pred_sseg in pred_dets.data['segmentations']:
        pred_canvas = pred_sseg.fill(pred_canvas, value=1)

    # TODO: need to consider the fact that these annotations are
    # multi-class, and have scores, we may want to evaluate the raw
    # probability map in order to find a decent operating point.
    # We also need to hook into the real IAPRA metrics.

    # FIXME: for now this is just binary change
    y_true = true_canvas.ravel()
    y_pred = pred_canvas.ravel()
    mat = skm.confusion_matrix(y_true, y_pred, labels=np.array([0, 1]))

    info = {
        'mat': mat,
        'pred_canvas': pred_canvas,
        'true_canvas': true_canvas,
    }
    tn, fp, fn, tp = mat.ravel()
    row = binary_confusion_measures(tn, fp, fn, tp)
    row = ub.map_vals(lambda x: x.item(), row)
    row["true_gid"] = gid1
    row["pred_gid"] = gid2
    row["video"] = video1['name']
    info['row'] = row

    TRY_SOFT = 1
    if TRY_SOFT:
        try:
            pred_gid = gid2
            pred_img = pred_coco.index.imgs[pred_gid]
            pred_coco_img = CocoImage(pred_img, pred_coco)

            change_prob = pred_coco_img.delay('change_prob', space='image').finalize()[..., 0]
            invalid_mask = np.isnan(change_prob)
            change_prob[invalid_mask] = 0

            bin_cfns = BinaryConfusionVectors(kwarray.DataFrameArray({
                'is_true': true_canvas.ravel(),
                'pred_score': change_prob.ravel(),
                'weight': (1 - invalid_mask).ravel().astype(np.float32),
            }))
            change_measures = bin_cfns.measures()
            row.update(ub.dict_isect(change_measures.summary(), {'ap', 'auc', 'max_f1'}))

            info.update({
                'change_measures': change_measures,
                'change_prob': change_prob,
                'invalid_mask': invalid_mask,
            })
        except Exception:
            pass

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
    colors = ["white", "green", "yellow", "red"]
    color_labels = ["TN", "TP", "FN", "FP"]
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

        pred_canvas = info['pred_canvas']
        true_canvas = info['true_canvas']
        confusion_idxs = utils.confusion_image(pred_canvas, true_canvas)
        confusion_image = color_lut[confusion_idxs]

        image_text = f'{frame_index} - gid = {true_gid}'

        confusion_image = kwimage.ensure_uint255(confusion_image)

        header = util_kwimage.draw_header_text(
            confusion_image, image_text, color='white', stack=False)

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
            true_coco_img = CocoImage(true_img, true_coco)
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
        change_prob = None
        if TRY_SOFT:
            change_prob = info.get('change_prob', None)
            invalid_mask = info.get('invalid_mask', None)
            if change_prob is not None:
                heatmap = kwimage.make_heatmask(change_prob, with_alpha=0.5)
                heatmap[invalid_mask] = 0
                heatmap_int = kwimage.ensure_uint255(heatmap[..., 0:3])
                vert_parts.append(heatmap_int)
                if real_image_norm is not None:
                    overlaid = kwimage.overlay_alpha_layers([heatmap, real_image_norm.mean(axis=2)])
                    overlaid = kwimage.ensure_uint255(overlaid[..., 0:3])
                    vert_parts.append(overlaid)

        if real_image_int is not None and change_prob is None:
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

    header = header_text(plot_fname, max_dim=plot_canvas.shape[1],
                         shrink=False)
    plot_canvas = kwimage.stack_images([header, plot_canvas], axis=0)

    plot_dpath = pathlib.Path(str(plot_dpath))
    plot_fpath = plot_dpath / plot_fname
    kwimage.imwrite(str(plot_fpath), plot_canvas)


@profile
def header_text(text, max_dim=None, shrink=False):
    """
    If shrink is true, shrinks the text to fit, otherwise text is
    placed in the center at a constant size, but is not guarenteed
    to fit.
    """
    import cv2
    if shrink:
        header = kwimage.draw_text_on_image(
            None, text, org=(1, 1),
            valign='top', halign='left', color='salmon')
        header = cv2.copyMakeBorder(header, 3, 3, 3, 3,
                                    cv2.BORDER_CONSTANT)
        header = kwimage.imresize(header, dsize=(max_dim, None))
    else:
        header = kwimage.draw_text_on_image(
            {'width': max_dim}, text, org=(max_dim // 2, 1),
            valign='top', halign='center', color='salmon')
    return header


@profile
def evaluate_segmentations(true_coco, pred_coco, eval_dpath=None):
    """

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
        >>> }
        >>> pred_coco = perterb_coco(true_coco, **kwargs)
        >>> eval_dpath = ub.ensure_app_cache_dir('watch/tests/fusion_eval')
        >>> print('eval_dpath = {!r}'.format(eval_dpath))
        >>> evaluate_segmentations(true_coco, pred_coco, eval_dpath)
    """

    required_marked = 'auto'  # parametarize
    if required_marked == 'auto':
        # In "auto" mode dont require marks if all images are unmarked,
        # otherwise assume that we should restirct to marked images
        required_marked = any(pred_coco.images().lookup('has_predictions', False))

    video_matches, image_matches = associate_images(true_coco, pred_coco)
    rows = []
    chunk_size = 5

    if eval_dpath is None:
        plot_dpath = None
        draw = False
    else:
        plot_dpath = pathlib.Path(eval_dpath) / 'plots'
        plot_dpath.mkdir(exist_ok=True, parents=True)
        draw = True

    combo_measures = None

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

            chunk_measures = []
            if combo_measures is not None:
                chunk_measures.append(combo_measures)

            for gid1, gid2 in chunk:
                info = single_image_segmentation_metrics(
                    true_coco, pred_coco, gid1, gid2)
                rows.append(info['row'])

                measures = info.get('change_measures', None)
                if measures is not None:
                    chunk_measures.append(measures)

                if draw:
                    chunk_info.append(info)
                prog.update()

            if chunk_measures:
                # Reduce measures over the chunk
                combo_measures = combine_kwcoco_measures(chunk_measures, precision=5)

            if draw:
                dump_chunked_confusion(
                    true_coco, pred_coco, chunk_info, plot_dpath)

    if combo_measures is not None:
        curve_dpath = pathlib.Path(eval_dpath) / 'curves'
        curve_dpath.mkdir(exist_ok=True, parents=True)
        measure_info = combo_measures.__json__()
        import json
        with open(curve_dpath / 'measures.json', 'w') as file:
            json.dump(measure_info, file)
        if draw:
            combo_measures.reconstruct()
            print('combo_measures = {!r}'.format(combo_measures))
            import kwplot
            kwplot.autompl()
            fig = kwplot.figure(doclf=True)
            combo_measures.summary_plot(fnum=1)
            fig = kwplot.autoplt().gcf()
            fig.savefig(str(curve_dpath / 'summary.png'))

    # Handle standalone images
    gids1 = image_matches['match_gids1']
    gids2 = image_matches['match_gids2']

    chunk_measures = []
    if combo_measures is not None:
        chunk_measures.append(combo_measures)
    for gid1, gid2 in zip(gids1, gids2):
        info = single_image_segmentation_metrics(
            true_coco, pred_coco, gid1, gid2)
        rows.append(info['row'])
        if draw:
            chunk_info = [info]
            dump_chunked_confusion(
                true_coco, pred_coco, chunk_info, plot_dpath)
        prog.update()
    if chunk_measures:
        if len(chunk_measures) == 1:
            combo_measures = chunk_measures[0]
        else:
            combo_measures = combine_kwcoco_measures(chunk_measures, precision=5)

    prog.end()

    df = pd.DataFrame(rows)
    print(df)

    summary = binary_confusion_measures(
        df.tn.sum(), df.fp.sum(), df.fn.sum(), df.tp.sum())
    summary = ub.map_vals(lambda x: x.item(), summary)
    if combo_measures is not None:
        summary['ap'] = combo_measures['ap']
        summary['auc'] = combo_measures['auc']
        summary['max_f1'] = combo_measures['max_f1']
    print('summary = {}'.format(ub.repr2(summary, nl=1, precision=4, align=':',
                                         sort=0)))

    if eval_dpath is not None:
        eval_dpath = pathlib.Path(eval_dpath)
        eval_dpath.mkdir(exist_ok=True, parents=True)
        metrics_fpath = eval_dpath / 'metrics.json'
        df.to_json(str(metrics_fpath))

    print('eval_dpath = {!r}'.format(eval_dpath))
    return df


@profile
def combine_kwcoco_measures(tocombine, precision=None):
    """
    Combine binary confusion metrics

    TODO:
        - [ ] Remove and use the version in kwcoco.metrics once that is released

    Example:
        >>> from watch.tasks.fusion.evaluate import *  # NOQA
        >>> from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
        >>> measures1 = BinaryConfusionVectors.demo(n=15).measures()
        >>> measures2 = measures1  # BinaryConfusionVectors.demo(n=15).measures()
        >>> tocombine = [measures1, measures2]
        >>> new_measures = combine_kwcoco_measures(tocombine)
        >>> print('new_measures = {!r}'.format(new_measures))
        >>> print('measures1 = {!r}'.format(measures1))
        >>> print('measures2 = {!r}'.format(measures2))
        >>> new_measures.reconstruct()
        >>> print(ub.repr2(measures1.__json__(), nl=1, sort=0))
        >>> print(ub.repr2(measures2.__json__(), nl=1, sort=0))
        >>> print(ub.repr2(new_measures.__json__(), nl=1, sort=0))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1)
        >>> new_measures.summary_plot()
        >>> measures1.summary_plot()
        >>> measures1.draw('roc')
        >>> measures2.draw('roc')
        >>> new_measures.draw('roc')
    """
    from kwcoco.metrics.confusion_vectors import Measures

    combo_thresh_asc = np.hstack([m['thresholds'] for m in tocombine])
    if precision is not None:
        combo_thresh_asc = combo_thresh_asc.round(precision)

    combo_thresh_asc = np.unique(combo_thresh_asc)
    new_thresh = combo_thresh_asc[::-1]

    summable = {
        'nsupport': 0,
        'realpos_total': 0,
        'realneg_total': 0,
    }

    fp_accum = np.zeros(len(new_thresh))
    tp_accum = np.zeros(len(new_thresh))
    tn_accum = np.zeros(len(new_thresh))
    fn_accum = np.zeros(len(new_thresh))

    for measures in tocombine:
        thresholds = measures['thresholds']
        if precision is not None:
            thresholds = thresholds.round(precision)

        right_idxs = np.searchsorted(combo_thresh_asc, thresholds, 'right')
        left_idxs = len(combo_thresh_asc) - right_idxs
        left_idxs = left_idxs.clip(0, len(combo_thresh_asc) - 1)
        # NOTE: if the min is non-zero in each array the diff wont work
        # reformulate if this case arrises
        fp_pos = np.diff(np.r_[[0], measures['fp_count']])
        tp_pos = np.diff(np.r_[[0], measures['tp_count']])
        tn_pos = np.diff(np.r_[[0], measures['tn_count'][::-1]])[::-1]
        fn_pos = np.diff(np.r_[[0], measures['fn_count'][::-1]])[::-1]
        # Handle the case where we round the thresholds for space reasons
        if 1:
            np.add.at(fp_accum, left_idxs, fp_pos)
            np.add.at(tp_accum, left_idxs, tp_pos)
            np.add.at(fn_accum, left_idxs, fn_pos)
            np.add.at(tn_accum, left_idxs, tn_pos)
        else:
            unique_idxs, groupxs = kwarray.group_indices(left_idxs)
            fp_pos2 = [g.sum() for g in kwarray.apply_grouping(fp_pos, groupxs)]
            tp_pos2 = [g.sum() for g in kwarray.apply_grouping(tp_pos, groupxs)]
            tn_pos2 = [g.sum() for g in kwarray.apply_grouping(tn_pos, groupxs)]
            fn_pos2 = [g.sum() for g in kwarray.apply_grouping(fn_pos, groupxs)]
            fp_accum[unique_idxs] += fp_pos2
            tp_accum[unique_idxs] += tp_pos2
            tn_accum[unique_idxs] += tn_pos2
            fn_accum[unique_idxs] += fn_pos2

        for k in summable:
            summable[k] += measures[k]

    new_fp = np.cumsum(fp_accum)
    new_tp = np.cumsum(tp_accum)
    new_tn = np.cumsum(tn_accum[::-1])[::-1]
    new_fn = np.cumsum(fn_accum[::-1])[::-1]

    new_info = {
        'fp_count': new_fp,
        'tp_count': new_tp,
        'tn_count': new_tn,
        'fn_count': new_fn,
        'thresholds': new_thresh,
    }
    new_info.update(summable)

    other = {
        'fp_cutoff',
        'monotonic_ppv',
        'node',
        'cx',
        'stabalize_thresh',
        'trunc_idx'
    }
    rest = ub.dict_isect(tocombine[0], other)
    new_info.update(rest)
    new_measures = Measures(new_info)
    # new_measures.reconstruct()
    return new_measures


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_dataset", help='path to the groundtruth dataset')
    parser.add_argument("--pred_dataset", help='path to the predicted dataset')
    parser.add_argument("--eval_dpath", help='path to dump results')
    args, _ = parser.parse_known_args()

    true_coco = kwcoco.CocoDataset.coerce(args.true_dataset)
    pred_coco = kwcoco.CocoDataset.coerce(args.pred_dataset)
    eval_dpath = args.eval_dpath
    evaluate_segmentations(true_coco, pred_coco, eval_dpath)


if __name__ == "__main__":
    main()
