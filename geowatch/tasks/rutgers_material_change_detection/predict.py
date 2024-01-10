import os
import argparse
from glob import glob

import torch
import kwimage
import numpy as np
import ubelt as ub
from tqdm import tqdm
from osgeo import gdal


from geowatch.tasks.rutgers_material_change_detection.models import build_model
from geowatch.tasks.rutgers_material_change_detection.datasets import build_dataset, create_loader
from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_n_frames, generate_video_slice_object


class Evaluator:
    def __init__(self, args):
        model_path = args.model_path
        kwcoco_file_dir = args.input_kwcoco_dir
        self.save_kwcoco_path = args.output_kwcoco_file_path
        self.heatmap_pred_channel_names = args.heatmap_pred_channel_names

        # Get model configuration file.
        if os.path.isfile(model_path) is False:
            raise FileNotFoundError(f"Model not found at: {model_path}")
        model_data = torch.load(model_path)
        self.cfg = model_data["cfg"]

        # Update config based on command line arguments.
        if args.batch_size != -1:
            self.cfg.batch_size = args.batch_size
        if args.stride is not None:
            self.cfg.stride = args.stride

        # Get device.
        self.device = args.device

        # Build dataset object.
        self.eval_dataset, video_slice = self._build_dataset(kwcoco_file_dir)

        # Create dataloader.
        self.eval_loader = create_loader(self.eval_dataset, args.split, self.cfg.batch_size, args.n_workers)

        # Build model.
        self.model = self._build_model(model_data, video_slice)

    def _build_dataset(self, kwcoco_file_dir):
        # Build dataset object.
        n_frames = get_n_frames(self.cfg.dataset.n_frames, self.cfg.task_mode)
        video_slice = generate_video_slice_object(
            height=self.cfg.height,
            width=self.cfg.width,
            n_frames=n_frames,
            scale=self.cfg.scale,
            stride=self.cfg.stride,
        )

        dataset = build_dataset(
            self.cfg.dataset.name,
            args.split,
            video_slice,
            self.cfg.task_mode,
            seed_num=self.cfg.seed_num,
            normalize_mode=self.cfg.normalize_mode,
            channels=self.cfg.dataset.channels,
            max_iterations=None,
            overwrite_dset_dir=kwcoco_file_dir,
        )
        return dataset, video_slice

    def _build_model(self, model_data, video_slice):
        # TODO: Check that model is trained for this dataset.
        model = build_model(
            self.cfg,
            video_slice,
            self.eval_dataset.n_channels,
            self.eval_dataset.max_frames,
            self.eval_dataset.n_classes,
            device=self.device,
        ).to(self.device)

        # Update model weights from checkpoint.
        model.load_state_dict(model_data["state_dict"], strict=True)
        model.to(self.device)
        model.eval()

        return model

    def gen_kwcoco_file(self):
        self.model.eval()

        scale = self.cfg.scale

        # TODO: Make this run in parallel with dataloader.

        # Create a random subset of examples from dataloader to choose from.
        region_images, region_image_ids, overlap_masks = {}, {}, {}
        for examples in tqdm(self.eval_loader):
            # Load videos onto GPU memory.
            examples["video"] = examples["video"].to(self.device, non_blocking=True)

            # Get model prediction.
            model_outputs = self.model(examples)
            B = examples["video"].shape[0]
            for b in range(B):
                # Create region canvas or paste predictions.
                region_name = examples["region_name"][b]
                if region_name not in list(region_images.keys()):
                    og_height, og_width = examples["crop_info"][b]["og_height"], examples["crop_info"][b]["og_width"]
                    # Create canvases.
                    canvases = {
                        "conf": np.zeros([og_height, og_width, 2], dtype="float"),
                    }
                    region_images[region_name] = canvases
                    overlap_masks[region_name] = np.zeros([og_height, og_width], dtype="int")

                # Format pred/target.
                H0, W0, dH, dW = examples["crop_info"][b]["space_crop_slice"]
                model_pred = model_outputs[self.cfg.task_mode][b].detach().cpu()
                # NOTE: CONFIDENCE PREDICTION IS COMPUTED AS PROBS INSTEAD OF PREDS
                conf = torch.softmax(model_pred, dim=0).numpy()[:, ::scale, ::scale][:, :dH, :dW].transpose(1, 2, 0)

                # Paste crop pred/targets into canvases.
                region_images[region_name]["conf"][H0 : H0 + dH, W0 : W0 + dW] += conf
                overlap_masks[region_name][H0 : H0 + dH, W0 : W0 + dW] += 1

                # Get gids.
                if not (region_name in region_image_ids.keys()):
                    image_ids = self.eval_dataset.video_dataset[examples["crop_info"][b]["video_id"]]
                    region_image_ids[region_name] = image_ids

        # Get kwcoco dataset file from Dataset class.
        kwcoco_dataset = self.eval_dataset.coco_dset.copy()
        kwcoco_dataset.clear_annotations()
        kwcoco_dataset.reroot(absolute=True)  # Make all paths absolute
        kwcoco_dataset.fpath = self.save_kwcoco_path  # Change output file path and bundle path
        kwcoco_dataset.reroot(absolute=False)  # Reroot in the new bundle path

        # Set path to save auxiliary files.
        bundle_dpath = ub.Path(kwcoco_dataset.bundle_dpath)

        # Save full region images.
        for region_name, canvases in tqdm(region_images.items()):
            conf_image = canvases["conf"]
            image_ids = region_image_ids[region_name]

            # Normalize heatmap confidence image by number of predictions made per pixel.
            conf_image = conf_image / overlap_masks[region_name][:, :, None]

            # Write image to disk.
            ## Try only saving one image instead of many copies.
            region_conf_pred_save_dir = os.path.join(bundle_dpath, "aux_" + region_name)
            os.makedirs(region_conf_pred_save_dir, exist_ok=True)
            region_conf_pred_save_path = os.path.join(region_conf_pred_save_dir, "conf_pred.tif")

            # Get geo metadata
            # TODO: Update this method.
            gid = image_ids[1]
            coco_img = kwcoco_dataset.coco_image(gid)
            dset_root = coco_img.dset.bundle_dpath
            image_folder_name = kwcoco_dataset.index.imgs[gid]["name"]
            long_region_name = kwcoco_dataset.coco_image(gid).img["parent_name"]
            long_region_name = "_".join(long_region_name.split("_")[:3])

            try:
                image_dir = os.path.join(dset_root, "KR_R001", "S2", "affine_warp", image_folder_name)
                image_paths = sorted(glob(image_dir + "/*.tif"))
                ds = gdal.Open(image_paths[5])  # Get blue band
            except IndexError:
                image_dir = os.path.join(dset_root, "KR_R002", "S2", "affine_warp", image_folder_name)
                image_paths = sorted(glob(image_dir + "/*.tif"))
                ds = gdal.Open(image_paths[5])  # Get blue band

            ## Save image using gdal.
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(
                region_conf_pred_save_path, conf_image.shape[1], conf_image.shape[0], 2, gdal.GDT_Float64
            )

            non_sal_band = outdata.GetRasterBand(1)
            non_sal_band.WriteArray(conf_image[:, :, 0])
            sal_band = outdata.GetRasterBand(2)
            sal_band.WriteArray(conf_image[:, :, 1])
            outdata.SetGeoTransform(ds.GetGeoTransform())
            outdata.SetProjection(ds.GetProjection())
            outdata.FlushCache()
            outdata = None

            for image_id in image_ids:
                img = kwcoco_dataset.index.imgs[image_id]

                vid_from_img = kwimage.Affine.coerce(img["warp_img_to_vid"])
                img_from_vid = vid_from_img.inv()

                img.get("auxiliary", []).append(
                    {
                        "file_name": region_conf_pred_save_path,
                        "channels": self.heatmap_pred_channel_names,
                        "height": conf_image.shape[0],
                        "width": conf_image.shape[1],
                        "num_bands": 2,
                        "warp_aux_to_img": img_from_vid.concise(),
                    }
                )
                kwcoco_dataset.index.imgs[image_id] = img

        kwcoco_dataset.dump(self.save_kwcoco_path)
        print(f"Output predictions path: {self.save_kwcoco_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the .pth.tar file ")
    parser.add_argument(
        "input_kwcoco_dir",
        type=str,
        help="Path to a directory that contains a vali.kwcoco.json file to get input data from.",
    )
    parser.add_argument(
        "output_kwcoco_file_path", type=str, help="Output predictions placed in a .kwcoco.json at this path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The type of hardware to process data with model.",
    )
    parser.add_argument("--n_workers", type=int, default=4, help="Number of CPU processes to load data into model.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", default="valid", help="TODO: Remove this")
    parser.add_argument("--stride", type=int, default=None, help="Set the step that crops are created.")
    parser.add_argument(
        "--heatmap_pred_channel_names",
        type=str,
        default="not_salient|salient",
        help="Overwrite the name of heatmap predictions in the produced kwcoco file.",
    )
    # parser.add_argument(
    #     "--ignore_material_features",
    #     default=False,
    #     action="store_true",
    #     help="Do not add material features to output kwcoco file.",
    # )
    # parser.add_argument(
    #     "--ignore_heatmaps",
    #     default=False,
    #     action="store_true",
    #     help="Do not add confidence heatmaps to output kwcoco file.",
    # )
    # parser.add_argument(
    #     "--ignore_material_change_mask",
    #     default=False,
    #     action="store_true",
    #     help="Do not add material change mask to output kwcoco file.",
    # )
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.gen_kwcoco_file()
