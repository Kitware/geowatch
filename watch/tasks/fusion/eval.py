from datasets import common
import kwcoco
import ndsampler
import numpy as np
import sklearn.metrics as skm
from collections import defaultdict as ddict
import tqdm
import pandas as pd

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
        tn+fp+fn+tp,
    "tn": lambda tn, fp, fn, tp: 
        tn,
    "fp": lambda tn, fp, fn, tp: 
        fp, 
    "fn": lambda tn, fp, fn, tp: 
        fn,
    "tp": lambda tn, fp, fn, tp: 
        tp,
}

def main(args):
    results_coco = kwcoco.CocoDataset(args.result_kwcoco_path)
    results_sampler = ndsampler.CocoSampler(results_coco)
    results_ds = common.VideoDataset(
        results_sampler,
        sample_shape=(None,None,None),
    )

    confusion_matrices = ddict(dict)

    for example, video in zip(results_ds, results_coco.dataset["videos"]):
        preds = example["images"].detach().numpy()
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

            for pred, label in zip(pred_row, channel_names):
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
    df.to_csv(args.metrics_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_kwcoco_path")
    parser.add_argument("metrics_path")
    args = parser.parse_args()
    main(args)
