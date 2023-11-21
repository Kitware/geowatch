import os
import argparse
import ubelt as ub

import torch
import kwimage
import numpy as np
from tqdm import tqdm


from geowatch.tasks.rutgers_material_change_detection.models import build_model
from geowatch.tasks.rutgers_material_change_detection.datasets import create_loader
from geowatch.tasks.rutgers_material_change_detection.datasets.iarpa_sc_kwdataset import IARPA_SC_EVAL_DATASET
from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_n_frames, generate_video_slice_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the .pth.tar file ")
    parser.add_argument(
        "input_kwcoco",
        type=str,
        help="Path to a directory that contains a vali.kwcoco.json file to get input data from.",
    )
    parser.add_argument(
        "output_kwcoco", type=str, help="Model prediction will be saved to same directory as the input_kwcoco_dir."
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The type of hardware to process data with model.",
    )
    args = parser.parse_args()

    device = args.device

    # Load model and get config parameters.
    if os.path.exists(args.model_path) is False:
        raise FileNotFoundError(f'Path to model "{args.model_path}" does not exist.')

    model_data = torch.load(args.model_path)

    cfg = model_data["cfg"]
    cfg.framework.pretrain = None
    scale = cfg.scale

    if args.batch_size is None:
        args.batch_size = cfg.batch_size
    if args.n_workers is None:
        args.n_workers = cfg.n_workers

    config = args.__dict__
    print('config = {}'.format(ub.urepr(config, nl=1)))

    # Build dataset (manually).
    n_frames = get_n_frames(cfg.dataset.n_frames, cfg.task_mode)
    video_slice = generate_video_slice_object(
        height=cfg.height, width=cfg.width, n_frames=n_frames, scale=cfg.scale, stride=cfg.stride
    )
    eval_dataset = IARPA_SC_EVAL_DATASET(
        args.input_kwcoco,
        "valid",
        video_slice,
        cfg.task_mode,
        seed_num=cfg.seed_num,
        channels=cfg.dataset.channels,
        normalize_mode=cfg.normalize_mode,
    )
    eval_loader = create_loader(eval_dataset, "valid", args.batch_size, args.n_workers)

    # Build model.
    model = build_model(
        cfg, video_slice, eval_dataset.n_channels, eval_dataset.max_frames, eval_dataset.n_classes, device=device
    ).to(device)
    model.load_state_dict(model_data["state_dict"], strict=True)
    model.to(device)
    model.eval()

    # Create save directories.
    pred_save_path = ub.Path(args.output_kwcoco)
    asset_dir = (pred_save_path.parent / '_assets').ensuredir()

    # Create a random subset of examples from dataloader to choose from.
    region_images, region_image_ids, overlap_masks = {}, {}, {}
    for examples in tqdm(eval_loader, desc='predicting'):
        # Load videos onto GPU memory.
        examples["video"] = examples["video"].to(device, non_blocking=True)

        # Get model prediction.
        model_outputs = model(examples)
        B = examples["video"].shape[0]
        for b in range(B):
            all_image_ids = eval_dataset.coco_dset.index.vidid_to_gids[examples["crop_info"][b]["video_id"]]
            # Create region canvas or paste predictions.
            region_name = examples["region_name"][b]
            if region_name not in list(region_images.keys()):
                og_height, og_width = examples["crop_info"][b]["og_height"], examples["crop_info"][b]["og_width"]
                # Create canvases.
                canvases = {
                    "conf": np.zeros([og_height, og_width, eval_dataset.n_classes, len(all_image_ids)], dtype="float"),
                }
                region_images[region_name] = canvases
                overlap_masks[region_name] = np.zeros([og_height, og_width, len(all_image_ids)], dtype="int")

            # Format pred/target.
            H0, W0, dH, dW = examples["crop_info"][b]["space_crop_slice"]
            model_pred = model_outputs[cfg.task_mode][b].detach().cpu()

            # 0. No Activity
            # 1. Site preparation
            # 2. Active construction
            # 3. Post Construction
            # NOTE: CONFIDENCE PREDICTION IS COMPUTED AS PROBS INSTEAD OF PREDS
            conf = torch.softmax(model_pred, dim=0).numpy()[:, ::scale, ::scale][:, :dH, :dW].transpose(1, 2, 0)

            # Get relative video indices.
            relative_video_indices = examples["crop_info"][b]["relative_video_indices"]

            # Paste crop pred/targets into canvases.
            region_images[region_name]["conf"][H0 : H0 + dH, W0 : W0 + dW, :, relative_video_indices[-1]] += conf
            overlap_masks[region_name][H0 : H0 + dH, W0 : W0 + dW, relative_video_indices[-1]] += 1

            # Get gids.
            if not (region_name in region_image_ids.keys()):
                image_ids = eval_dataset.video_dataset[examples["crop_info"][b]["video_id"]]
                region_image_ids[region_name] = image_ids

    # Get kwcoco dataset file from Dataset class.
    kwcoco_dataset = eval_dataset.coco_dset.copy()
    kwcoco_dataset.clear_annotations()

    # Save full region images.
    for region_name, canvases in tqdm(region_images.items()):
        conf_image = canvases["conf"]
        image_ids = region_image_ids[region_name]

        # Normalize heatmap confidence image by number of predictions made per pixel.
        conf_image = conf_image / overlap_masks[region_name][:, :, None]

        # Write image to disk.
        ## Try only saving one image instead of many copies.
        region_dir = os.path.join(asset_dir, region_name)
        os.makedirs(region_dir, exist_ok=True)

        # Save asset images.
        channel_names = ["No Activity", "Site Preparation", "Active Construction", "Post Construction"]
        for channel_index in range(4):
            channel_name = channel_names[channel_index]
            for image_id_index, image_id in enumerate(image_ids):

                img_save_path = os.path.join(region_dir, channel_name.replace(" ", "_") + "_" + str(image_id) + ".tif")

                data = conf_image[:, :, channel_index, image_id_index]
                nodata = None
                tosave = data
                quantization = None

                if 0:
                    # Do quantization or not
                    from geowatch.tasks.fusion.predict import quantize_float01
                    quant_recon, quantization = quantize_float01(
                        data, old_min=0, old_max=1)
                    nodata = quantization['nodata']
                    tosave = quant_recon

                kwimage.imwrite(img_save_path, tosave, backend="gdal",
                                nodata=nodata)

                img = kwcoco_dataset.index.imgs[image_id]

                vid_from_img = kwimage.Affine.coerce(img["warp_img_to_vid"])
                img_from_vid = vid_from_img.inv()

                # For T&E metrics.
                new_aux = {
                    "file_name": img_save_path,
                    "channels": channel_name,
                    "height": conf_image.shape[0],
                    "width": conf_image.shape[1],
                    "num_bands": 1,
                    "warp_aux_to_img": img_from_vid.concise(),
                    'quantization': quantization,
                }
                if quantization is not None:
                    new_aux['quantization'] = quantization

                auxiliary = img.get("auxiliary", [])
                auxiliary.append(new_aux)
                img['auxiliary'] = auxiliary
                kwcoco_dataset.index.imgs[image_id] = img

    # Get save path.
    kwcoco_dataset.dump(pred_save_path)
    print(f"Output predictions path: {pred_save_path}")


if __name__ == "__main__":
    """
    Example call:

    DVC_DPATH=$(WATCH_HACK_IMPORT_ORDER=none python -m geowatch.cli.find_dvc)
    MODEL_FPATH=$DVC_DPATH/models/rutgers/rutgers_sc_model_v4.pth.tar

    kwcoco subset \
        --src=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json \
        --dst=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_s2kr_vali.kwcoco.json \
        --select_images='.sensor_coarse == "S2"' \
        --select_videos='.name == "KR_R001"'

    python -m geowatch.tasks.rutgers_material_change_detection.predict_sc \
        $MODEL_FPATH \
        $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_s2kr_vali.kwcoco.json \
        $DVC_DPATH/tmp/my_pred/pred.kwcoco.json \
        --n_workers 16 --batch_size 16

    python -m geowatch.tasks.fusion.evaluate \
        --true_dataset=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_s2kr_vali.kwcoco.json \
        --pred_dataset=$DVC_DPATH/tmp/my_pred/pred.kwcoco.json \
        --eval_dpath=$DVC_DPATH/tmp/my_eval \
        --score_space=video \
        --draw_curves=1 \
        --draw_heatmaps=0 --workers=8
    """
    main()
