import kwcoco
import kwimage
import sklearn.metrics as skm
from collections import defaultdict as ddict
import tqdm
import pandas as pd
import numpy as np
import pathlib
import tifffile
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
        >>>     rng.randint(0, 5, (16, 4)),
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

        p = fn + tp  # number of real positives
        n = fp + tn  # number of real negatives

        total = p + n

        pred_pos = (fp + tp)  # number of predicted positives
        pred_neg = (fn + tn)  # number of predicted negatives

        pred_correct = tp + tn  # number of correct predictions

        # Error / Success Rates
        # https://en.wikipedia.org/wiki/Confusion_matrix
        # (Ensure denominator parts are non-zero)
        p_denom = p.copy()
        p_denom[p_denom == 0] = 1
        n_denom = n.copy()
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
        mcc_numer = (tp * tn) - (fp * fn)
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_denom[np.isnan(mcc_denom) | (mcc_denom == 0)] = 1
        mcc = mcc_numer / mcc_denom

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

        info['p'] = p  # number of real positives
        info['n'] = n  # number of real negatives
        info['total'] = total

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
    # TODO: robust image association (see kwcoco eval)
    common_fnames = set(true_coco.index.file_name_to_img) & set(pred_coco.index.file_name_to_img)
    common_names = set(true_coco.index.name_to_img) & set(pred_coco.index.name_to_img)

    import ubelt as ub
    associated = ub.oset()

    for fname in common_fnames:
        gid1 = true_coco.index.file_name_to_img[fname]['id']
        gid2 = pred_coco.index.file_name_to_img[fname]['id']
        associated.add((gid1, gid2))

    for name in common_names:
        gid1 = true_coco.index.name_to_img[name]['id']
        gid2 = pred_coco.index.name_to_img[name]['id']
        associated.add((gid1, gid2))

    return associated


def compute_simple_segmentation_metrics(true_coco, pred_coco):

    assoiated_gids = associate_images(true_coco, pred_coco)

    confusion_matrices = ddict(dict)

    for gid1, gid2 in assoiated_gids:
        img1 = true_coco.imgs[gid1]
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
            pred_canvas = true_sseg.fill(pred_canvas, value=1)

        # TODO: need to consider the fact that these annotations are
        # multi-class, and have scores, we may want to evaluate the raw
        # probability map in order to find a decent operating point.
        # We also need to hook into the real IAPRA metrics.

        # FIXME: for now this is just binary change
        mat = skm.confusion_matrix(true_canvas.ravel(), pred_canvas.ravel())
        confusion_matrices[(gid1, gid2)] = mat

    rows = []
    for (gid1, gid2), matrix in confusion_matrices.items():
        vidid1 = true_coco.imgs[gid1]['video_id']
        video1 = true_coco.index.videos[vidid1]

        tn, fp, fn, tp = matrix.ravel()
        row = binary_confusion_measures(tn, fp, fn, tp)
        row = ub.map_vals(lambda x: x.item(), row)
        row["gid1"] = gid1
        row["gid2"] = gid2
        row["video"] = video1['name']
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def plot_frames(frames, target, row, cmap, thresh=0.):
    for ax, (channel, file_name) in zip(row, frames.items()):
        logits = tifffile.imread(file_name)
        pred = (logits > thresh).astype("int")
        confusion_image = utils.confusion_image(pred, target)
        ax.matshow(confusion_image, cmap=cmap)

        #  ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')

        ax.set_title(channel)


def make_confusion_plots(true_coco, pred_coco, figure_root):
    import kwplot
    with kwplot.BackendContext('Agg'):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import matplotlib.patches as mpatches

        cmap = colors.ListedColormap(["w", "k", "r", "y"], N=4)

        results = kwcoco.CocoDataset(str(result_kwcoco_path))

        models = sorted(list({
            aux["channels"].rsplit("_", 1)[0]
            for aux in results.dataset["images"][0]["auxiliary"]
        }))

        for video in tqdm.tqdm(results.dataset["videos"]):
            for model_name in tqdm.tqdm(models):

                # if video["name"] not in test_cities: continue

                images = [
                    image
                    for image in results.dataset["images"]
                    if image["video_id"] == video["id"]
                ]

                H = len(images)
                W = len(images[0]["auxiliary"])

                fig, axs = plt.subplots(H, W, figsize=(3 * W, 4 * H), squeeze=False)
                if len(axs.shape) == 1:
                    axs = axs[None]

                for idx, (row, image) in enumerate(zip(axs, images)):

                    frames = {
                        aux["channels"].rsplit("_", 1)[-1]: aux["file_name"]
                        for aux in image["auxiliary"]
                    }

                    target = np.zeros([video["height"], video["width"]])

                    for ann_id in results.gid_to_aids[image["id"]]:
                        seg = kwimage.Segmentation.coerce(
                            results.anns[ann_id]["segmentation"]
                        ).to_multi_polygon()
                        seg.fill(target, value=1)

                    target = target.astype("int")

                    plot_frames(frames, target, row, cmap, thresh=0.)

                patches = [
                    mpatches.Patch(color=cmap.colors[idx], label=name)
                    for idx, name in enumerate(["TN", "TP", "FN", "FP"])
                ]
                fig.legend(handles=patches, loc='upper center', ncol=4, fontsize="xx-large", bbox_to_anchor=(0.5, 0.))

                # params = list(frames.values())[0]["params"]
                # fig.suptitle(f"{video['name']} / {model_name} ({millify(params)})", fontsize="xx-large")
                fig.suptitle(f"{video['name']} / {model_name}", fontsize="xx-large")
                fig.tight_layout()
                fig.patch.set_facecolor('white')

                figure_path = figure_root / video['name']
                figure_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(figure_path / (model_name + ".png"))


def evaluate_segmentations(true_dataset, pred_dataset, eval_dpath=None):
    """

    Example:
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> import kwcoco
        >>> true_dataset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>>     'with_probs': True,
        >>> }
        >>> pred_dataset = perterb_coco(true_dataset, **kwargs)
    """
    import kwcoco
    true_coco = kwcoco.CocoDataset.coerce(true_dataset)
    pred_coco = kwcoco.CocoDataset.coerce(pred_dataset)

    df = compute_simple_segmentation_metrics(true_coco, pred_coco)
    print(df)

    if eval_dpath is not None:
        eval_dpath = pathlib.Path(eval_dpath)
        eval_dpath.mkdir(exist_ok=True, parents=True)
        metrics_fpath = eval_dpath / 'metrics.json'
        df.to_csv(str(metrics_fpath), index=False)
    # make_confusion_plots(args.eval_dpath, args.figure_root)
    # compute_metrics(args.result_kwcoco_path, args.metrics_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_dataset", help='path to the groundtruth dataset')
    parser.add_argument("--pred_dataset", help='path to the predicted dataset')
    parser.add_argument("--eval_dpath", help='path to dump results')
    args = parser.parse_args()

    evaluate_segmentations(args.true_dataset, args.pred_dataset, args.eval_dpath)


if __name__ == "__main__":
    main()
