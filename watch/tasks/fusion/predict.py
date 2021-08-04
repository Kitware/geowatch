# import pytorch_lightning as pl
import pathlib
import numpy as np
import tqdm
import tifffile
import kwcoco
import kwimage
from torch import nn

from watch.tasks.fusion import datasets
from watch.tasks.fusion import methods
from watch.tasks.fusion import utils

fname_template = "{location}/{bands}-{frame_no}.tif"


def main(args):

    # init method from checkpoint
    method = utils.load_model_from_package(args.checkpoint_path)
    method.eval()
    method.freeze()

    # init dataset from args
    dataset_class = getattr(datasets, args.dataset)
    dataset_var_dict = utils.filter_args(
        vars(args),
        dataset_class.__init__,
    )
    dataset_var_dict["preprocessing_step"] = method.preprocessing_step
    dataset = dataset_class(
        **dataset_var_dict
    )
    dataset.setup("test")

    test_dataset = dataset.test_dataset
    test_dataloader = dataset.test_dataloader()

    # load or init results ds
    if args.results_path.exists():
        results_ds = kwcoco.CocoDataset(str(args.results_path.expanduser()))
    else:
        results_ds = kwcoco.CocoDataset()
        results_ds.add_category("change")

    total_params = sum(p.numel() for p in method.parameters())

    if args.use_gpu:
        method = method.to("cuda:0")

    # set up canvases for prediction
    result_canvases = {
        video["id"]: np.full([
                video["num_frames"] - 1, video["height"], video["width"],
            ], -2.0)
        for video in test_dataset.sampler.dset.dataset["videos"]
    }
    result_counts = {
        video["id"]: np.full([
                video["num_frames"] - 1, video["height"], video["width"],
            ], 0)
        for video in test_dataset.sampler.dset.dataset["videos"]
    }
    target_canvases = {
        video["id"]: np.full([
                video["num_frames"] - 1, video["height"], video["width"],
            ], -2)
        for video in test_dataset.sampler.dset.dataset["videos"]
    }

    # fill canvases
    for example, meta in zip(tqdm.tqdm(test_dataloader), test_dataset.sample_grid):
        images, labels = example["images"].float(), example["labels"]
        changes = (labels[0, 1:] != labels[0, :-1]).detach().cpu().numpy()
        T, H, W = changes.shape

        preds = (method(images)).detach()
        preds = nn.functional.interpolate(
            preds,
            size=(H, W),
            mode="bilinear")[0]
        preds = preds.cpu().numpy()

        time_slice = slice(
                meta["time_slice"].start,
                meta["time_slice"].stop - 1,
                meta["time_slice"].step,
            )
        space_time_slice = (time_slice,) + meta["space_slice"]

        # print(
        #    space_time_slice,
        #    result_canvases[meta["vidid"]][space_time_slice].shape,
        #    preds.shape,
        # )

        result_canvases[meta["vidid"]][space_time_slice] += preds
        result_counts[meta["vidid"]][space_time_slice] += 1
        target_canvases[meta["vidid"]][space_time_slice] = changes

    # print({
    #    key: {
    #        idx: np.unique(layer, return_counts=True)
    #        for idx, layer in enumerate(canvas)
    #        }
    #    for key, canvas in target_canvases.items()
    # })

    results = {
        key: canvas / result_counts[key]
        for key, canvas in result_canvases.items()
    }
    targets = target_canvases

    # save canvases to disk
    video_keys = {video["id"] for video in results_ds.dataset["videos"]}
    image_keys = {image["id"] for image in results_ds.dataset["images"]}

    for video in test_dataset.sampler.dset.dataset["videos"]:

        # if video not in results_ds, add it
        if video["id"] not in video_keys:
            # TODO: just copy all video metadata? (**video)
            results_ds.add_video(
                name=video["name"],
                id=video["id"],
                width=video["width"],
                height=video["height"],
                target_gsd=video["target_gsd"],
                warp_wld_to_vid=video["warp_wld_to_vid"],
                min_gsd=video["min_gsd"],
                max_gsd=video["max_gsd"],
            )

        # index into results
        result_stack = results[video["id"]]
        target_stack = targets[video["id"]]

        frames = [
            img
            for img in test_dataset.sampler.dset.imgs.values()
            if img["video_id"] == video["id"]
        ][1:]

        result_stack = result_stack
        target_stack = target_stack
        for frame, result, target in zip(frames, result_stack, target_stack):

            # if frame not in results_ds, add it
            if frame["id"] not in image_keys:
                # TODO: just copy all frame metadata? (**frame)
                results_ds.add_image(
                    id=frame["id"],
                    name="{}-{}".format(video["name"], frame["frame_index"]),
                    width=frame["width"],
                    height=frame["height"],
                    video_id=video["id"],
                    frame_index=frame["frame_index"],
                    img_to_vid=frame["img_to_vid"],
                    warp_img_to_vid=frame["warp_img_to_vid"],
                )

                # new frame needs an annotation too
                segmentation = kwimage.Mask(target, "c_mask").to_coco()

                results_ds.add_annotation(
                    frame["id"],
                    category_id=1,
                    bbox=[0, 0, frame["width"], frame["height"]],
                    #bbox=[0, 0, frame["height"], frame["width"]],
                    segmentation=segmentation,
                )

            # save result to file
            result_fname = args.results_dir / fname_template.format(
                location=video["name"],
                bands=args.tag,
                frame_no=frame["frame_index"],
            )
            result_fname.parents[0].mkdir(parents=True, exist_ok=True)

            tifffile.imwrite(result_fname, result)

            # add result to dataset
            utils.add_auxiliary(
                results_ds,
                frame["id"],
                str(result_fname.absolute()),
                args.tag,
                aux_width=frame["width"],
                aux_height=frame["height"],
                warp_aux_to_img=None,
                extra_info={"params": total_params},
            )

    # validate and save results
    print(results_ds.validate())
    results_ds.dump(str(args.results_path))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("tag")
    parser.add_argument("checkpoint_path", type=pathlib.Path)
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("results_path", type=pathlib.Path)
    parser.add_argument("--use_gpu", action="store_true")

    # parse the dataset and method strings
    temp_args, _ = parser.parse_known_args()

    # get the dataset and method classes
    dataset_class = getattr(datasets, temp_args.dataset)

    # add the appropriate args to the parse
    # for dataset, method, and trainer
    parser = dataset_class.add_data_specific_args(parser)

    # parse and pass to main
    args = parser.parse_args()

    assert args.batch_size == 1

    main(args)
