import kwcoco
import kwimage
import sklearn.metrics as skm
import pandas as pd
import numpy as np
import pathlib
from watch.tasks.fusion import utils
import ubelt as ub


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


def single_image_segmentation_metrics(true_coco, pred_coco, gid1, gid2):
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
    return info


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


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
        img = true_coco.index.imgs[true_gid]
        frame_index = img['frame_index']
        frame_nums.append(frame_index)
        true_gids.append(true_gid)

        # pred_gid = row['pred_gid']

        pred_canvas = info['pred_canvas']
        true_canvas = info['true_canvas']
        confusion_idxs = utils.confusion_image(pred_canvas, true_canvas)
        confusion_image = color_lut[confusion_idxs]

        image_text = f'{frame_index} - gid = {true_gid}'

        confusion_image = kwimage.ensure_uint255(confusion_image)
        confusion_image = kwimage.draw_text_on_image(
            confusion_image, image_text, org=(1, 1), valign='top',
            color='white', border={'color': 'black'})

        parts.append(confusion_image)

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

    # Handle images in videos
    for video_match in video_matches:
        gids1 = video_match['match_gids1']
        gids2 = video_match['match_gids2']
        pairs = list(zip(gids1, gids2))
        for chunk in ub.chunks(pairs, chunk_size):
            chunk_info = []
            for gid1, gid2 in chunk:
                info = single_image_segmentation_metrics(
                    true_coco, pred_coco, gid1, gid2)
                rows.append(info['row'])
                if draw:
                    chunk_info.append(info)
            if draw:
                dump_chunked_confusion(
                    true_coco, pred_coco, chunk_info, plot_dpath)

    # Handle standalone images
    gids1 = image_matches['match_gids1']
    gids2 = image_matches['match_gids2']
    for gid1, gid2 in zip(gids1, gids2):
        info = single_image_segmentation_metrics(
            true_coco, pred_coco, gid1, gid2)
        rows.append(info['row'])
        if draw:
            chunk_info = [info]
            dump_chunked_confusion(
                true_coco, pred_coco, chunk_info, plot_dpath)

    df = pd.DataFrame(rows)
    print(df)

    summary = binary_confusion_measures(
        df.tn.sum(), df.fp.sum(), df.fn.sum(), df.tp.sum())
    summary = ub.map_vals(lambda x: x.item(), summary)
    print('summary = {}'.format(ub.repr2(summary, nl=1, precision=4, align=':',
                                         sort=0)))

    if eval_dpath is not None:
        eval_dpath = pathlib.Path(eval_dpath)
        eval_dpath.mkdir(exist_ok=True, parents=True)
        metrics_fpath = eval_dpath / 'metrics.json'
        df.to_json(str(metrics_fpath))

    print('eval_dpath = {!r}'.format(eval_dpath))
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_dataset", help='path to the groundtruth dataset')
    parser.add_argument("--pred_dataset", help='path to the predicted dataset')
    parser.add_argument("--eval_dpath", help='path to dump results')
    args = parser.parse_args()

    true_coco = kwcoco.CocoDataset.coerce(args.true_dataset)
    pred_coco = kwcoco.CocoDataset.coerce(args.pred_dataset)
    eval_dpath = args.eval_dpath
    evaluate_segmentations(true_coco, pred_coco, eval_dpath)


if __name__ == "__main__":
    main()
