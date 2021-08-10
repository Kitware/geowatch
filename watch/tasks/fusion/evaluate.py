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

metrics = {
    "sensitivity": lambda tn, fp, fn, tp:
        tp / (tp + fn),
    "specificity": lambda tn, fp, fn, tp:
        tn / (tn + fp),
    "precision": lambda tn, fp, fn, tp:
        tp / (tp + fp),
    "accuracy": lambda tn, fp, fn, tp:
        (tp + tn) / (tp + tn + fp + fn),
    "f1": lambda tn, fp, fn, tp:
        (2 * tp) / ((2 * tp) + fp + fn),
    "total": lambda tn, fp, fn, tp:
        tn + fp + fn + tp,
    "tn": lambda tn, fp, fn, tp:
        tn,
    "fp": lambda tn, fp, fn, tp:
        fp,
    "fn": lambda tn, fp, fn, tp:
        fn,
    "tp": lambda tn, fp, fn, tp:
        tp,
}


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
        row = {
            metric: metric_fn(*matrix.ravel())
            for metric, metric_fn in metrics.items()
        }
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


def evaluate_segmentations(true_dataset, pred_dataset):
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

    df = compute_metrics(true_coco, pred_coco)
    df.to_csv(metrics_path, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_dataset", type=pathlib.Path, help='path to the groundtruth dataset')
    parser.add_argument("--pred_dataset", type=pathlib.Path, help='path to the predicted dataset')
    parser.add_argument("--output_dpath", type=pathlib.Path, help='path to dump results')
    args = parser.parse_args()

    make_confusion_plots(args.result_kwcoco_path, args.figure_root)
    compute_metrics(args.result_kwcoco_path, args.metrics_path)


if __name__ == "__main__":
    main()
