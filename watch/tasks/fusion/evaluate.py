import kwcoco
import ndsampler
import sklearn.metrics as skm
from collections import defaultdict as ddict
import tqdm
import pandas as pd
from .datasets import common
from . import utils

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


def compute_metrics(result_kwcoco_path, metrics_path):
    results_coco = kwcoco.CocoDataset(result_kwcoco_path)
    results_sampler = ndsampler.CocoSampler(results_coco)
    results_ds = common.VideoDataset(
        results_sampler,
        sample_shape=(None, None, None),
    )

    confusion_matrices = ddict(dict)

    for example, video in zip(results_ds, results_coco.dataset["videos"]):
        preds = example["images"]
        targets = example["labels"].detach().numpy() + 1

        frames = [
            image
            for image in results_coco.dataset["images"]
            if image["video_id"] == video["id"]
        ]

        print(video["name"])
        for frame, pred_row, target in tqdm.tqdm(zip(frames, preds, targets)):

            channel_names = [
                aux["channels"]
                for aux in frame["auxiliary"]
            ]

            for pred, label in tqdm.tqdm(zip(pred_row, channel_names)):
                confusion_matrices[label][video["name"]] = skm.confusion_matrix(target.flat, (pred > 0).flat)

    rows = []
    for method, cities in confusion_matrices.items():
        for city, matrix in cities.items():
            row = {
                metric: metric_fn(*matrix.ravel())
                for metric, metric_fn in metrics.items()
            }
            row["method"] = method
            row["city"] = city
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)
    

def plot_frames(frames, target, row, cmap, thresh=0.):
    for ax, (channel, file_name) in zip(row, frames.items()):
        try:
            logits = tifffile.imread(file_name)
            pred = (logits > thresh).astype("int")
            confusion_image = utils.confusion_image(pred, target)
            ax.matshow(confusion_image, cmap=cmap)
        except:
            pass

#             ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')

        ax.set_title(channel)
        

def make_confusion_plots(result_kwcoco_path, figure_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import matplotlib.patches as mpatches

    cmap = colors.ListedColormap(["w", "k", "r", "y"], N=4)

    results = kwcoco.CocoDataset(result_kwcoco_path)
    
    models = sorted(list({
        aux["channels"].rsplit("_", 1)[0]
        for aux in results.dataset["images"][0]["auxiliary"]
    }))
    
    for video in results.dataset["videos"]:
        for model_name in models:
    
        #     if video["name"] not in test_cities: continue

            images = [
                image
                for image in results.dataset["images"]
                if image["video_id"] == video["id"]
            ]

            H = len(images)
            W = len(channels)

            fig, axs = plt.subplots(H, W, figsize=(3*W, 4*H))
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

            patches =[
                mpatches.Patch(color=cmap.colors[idx], label=name) 
                for idx, name in enumerate(["TN", "TP", "FN", "FP"])
            ]
            fig.legend(handles=patches, loc='upper center', ncol=4, fontsize="xx-large", bbox_to_anchor=(0.5, 0.))

        #     params = list(frames.values())[0]["params"]
        #     fig.suptitle(f"{video['name']} / {model_name} ({millify(params)})", fontsize="xx-large")
            fig.suptitle(f"{video['name']} / {model_name}", fontsize="xx-large")
            fig.tight_layout()
            fig.patch.set_facecolor('white')
            
            figure_path = figure_root / vide['name'] / (model_name + ".png")
            figure_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(figure_path)
    
def main(args):
    compute_metrics(args.result_kwcoco_path, args.metrics_path)
    make_confusion_plots(args.result_kwcoco_path, args.figure_root)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_kwcoco_path")
    parser.add_argument("metrics_path")
    parser.add_argument("figure_root")
    args = parser.parse_args()
    main(args)
