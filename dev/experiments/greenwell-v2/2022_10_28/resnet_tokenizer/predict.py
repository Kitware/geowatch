from watch.tasks.fusion.methods import HeterogeneousModel
from watch.tasks.fusion.datamodules import KWCocoVideoDataModule
from watch.tasks.fusion import utils as fusion_utils
import tqdm
import torch
import numpy as np
from sklearn import metrics

# Demo of the data module on auto-generated toy data
from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA


def get_predictions(model, dataset):
    truth = dict()
    results = dict()
    counts = dict()
    for example in tqdm.tqdm(dataset):

        preds = model([example])

        for frame, frame_pred in zip(example["frames"], preds["change"][0]):
            key = frame["gid"]

            if key not in results:
                height, width = frame["output_image_dsize"]
                truth[key] = torch.zeros(width, height)
                results[key] = torch.zeros(width, height)
                counts[key] = torch.zeros(width, height)

            img_slice = frame["saliency"]

            truth[key][frame["output_space_slice"]] = img_slice

            if frame["change_weights"] is not None:
                results[key][frame["output_space_slice"]] += frame_pred[1].detach() * frame["change_weights"]
                counts[key][frame["output_space_slice"]] += frame["change_weights"]

    return truth, results, counts


def main():
    import pathlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=pathlib.Path)
    parser.add_argument("--coco_dataset", required=True, type=pathlib.Path)

    parser.add_argument("--chip_size", default=96, type=int)
    parser.add_argument("--time_steps", default=2, type=int)
    parser.add_argument("--window_overlap", default=0.5, type=float)
    parser.add_argument("--space_scale", default=None)
    parser.add_argument("--channels", default=None)
    args = parser.parse_args()

    import pandas as pd

    from PIL import Image

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    # Identify and create predictions directiory
    model_root = args.model_path.parents[0]
    preds_dir = model_root / (args.model_path.name + "_preds")
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Load data set
    datamodule = KWCocoVideoDataModule(
        test_dataset=args.coco_dataset,
        chip_size=args.chip_size,
        time_steps=args.time_steps,
        channels=args.channels,
        space_scale=args.space_scale,
    )
    datamodule.test_dataset_config["window_overlap"] = args.window_overlap
    datamodule.setup("test")

    dataset = datamodule.torch_datasets["test"]
    dataset.disable_augmenter = True

    # Load checkpoint
    if args.model_path.suffix == ".ckpt":
        model = HeterogeneousModel.load_from_checkpoint(args.model_path).to("cpu")
    else:
        model = fusion_utils.load_model_from_package(args.model_path).to("cpu")
    model.eval()

    # Get prediction dicts
    # Todo:
    #   - [ ] hardwired for change
    truth, results, counts = get_predictions(model, dataset)
    keys = [
        key for key, val in counts.items() if val.max() > 0
    ]

    # Maps
    # Todo:
    #   - [ ] hardwired for change
    for gid in keys:
        # Pred only
        pred_map = (results[gid] / counts[gid]).sigmoid()

        pred_map_img = Image.fromarray(pred_map.numpy())
        pred_map_img.save(preds_dir / f"change_{gid}.tif")

        # Side-by-side
        fig, (left, right) = plt.subplots(1, 2, figsize=(8, 4))
        left.matshow(truth[gid])
        right.matshow(pred_map)
        fig.suptitle(f"change gid={gid}")
        fig.tight_layout()
        fig.savefig(preds_dir / f"change-comp_{gid}.png")

    # F1 scores
    # Todo:
    #   - [ ] hardwired for change
    thresholds = np.concatenate([
        np.linspace(0, 0.1, 2 + 9)[1:-1],
        np.linspace(0, 1, 2 + 9)[1:-1],
    ])
    all_scores = list()
    for gid in keys:
        probs = (results[gid] / counts[gid]).sigmoid().numpy()
        labels = truth[gid].numpy()
        scores = {"gid": gid, "size": probs.size}
        for threshold in thresholds:
            scores[f"F1@{np.round(threshold,2)}"] = metrics.f1_score(
                labels.flat,
                probs.flat > threshold)
        all_scores.append(scores)

    pd.DataFrame(all_scores).to_csv(preds_dir / "f1_scores.csv")


if __name__ == "__main__":
    main()
