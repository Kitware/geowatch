import os
import ubelt as ub

import cv2
import torch
import kwcoco
import kwimage
import numpy as np
from datetime import datetime
from collections import namedtuple
from matplotlib.colors import to_rgba

from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_crop_slices
from geowatch.tasks.rutgers_material_change_detection.datasets.base_dataset import BaseDataset


class IARPA_SC_EVAL_DATASET(BaseDataset):
    def __init__(
        self,
        kwcoco_path,
        split,
        video_slice,
        task_mode,
        seed_num=0,
        sensor_type="S2",
        channels=None,
        transforms=None,
        normalize_mode=None,
        max_iterations=None,
    ):
        """Constructor.

        Args:
            kwcoco_path (str):
                File path to kwcoco file.

            video_slice (namedtuple):

            task_mode (str, str):
                Name of labels to produce for this dataset. Defaults to 'total_bin_change'.

            sensor_type (str, optional):
                The name of sensor to return image data from. Defaults to 'S2'.
        """

        kwcoco_path = ub.Path(kwcoco_path)
        super().__init__(
            os.fspath(kwcoco_path.parent),
            split,
            video_slice,
            task_mode,
            transforms=transforms,
            seed_num=seed_num,
            normalize_mode=normalize_mode,
            channels=channels,
            max_iterations=max_iterations,
        )

        if split != "valid":
            raise NotImplementedError("Dataset designed for split other than valid.")

        # Figure out how many channels to load.
        Channel_Info = namedtuple("channel_info", ["band_name", "wl_name", "wavelength", "scale_factor"])
        self.channel_info = {}
        if channels == "RGB":
            self.channel_info["B02"] = Channel_Info(band_name="B02", wl_name="blue", wavelength=492.4, scale_factor=1)
            self.channel_info["B03"] = Channel_Info(band_name="B03", wl_name="green", wavelength=559.8, scale_factor=1)
            self.channel_info["B04"] = Channel_Info(band_name="B04", wl_name="red", wavelength=664.6, scale_factor=1)
        elif channels == "RGB_NIR":
            self.channel_info["B02"] = Channel_Info(band_name="B02", wl_name="blue", wavelength=492.4, scale_factor=1)
            self.channel_info["B03"] = Channel_Info(band_name="B03", wl_name="green", wavelength=559.8, scale_factor=1)
            self.channel_info["B04"] = Channel_Info(band_name="B04", wl_name="red", wavelength=664.6, scale_factor=1)
            self.channel_info["B08"] = Channel_Info(band_name="B08", wl_name="nir", wavelength=832.8, scale_factor=1)
        elif channels == "ALL":
            self.channel_info["B01"] = Channel_Info(band_name="B01", wl_name="coastal", wavelength=442, scale_factor=6)
            self.channel_info["B02"] = Channel_Info(band_name="B02", wl_name="blue", wavelength=492, scale_factor=1)
            self.channel_info["B03"] = Channel_Info(band_name="B03", wl_name="green", wavelength=559, scale_factor=1)
            self.channel_info["B04"] = Channel_Info(band_name="B04", wl_name="red", wavelength=664, scale_factor=1)
            self.channel_info["B05"] = Channel_Info(band_name="B05", wl_name="nir", wavelength=704, scale_factor=2)
            self.channel_info["B06"] = Channel_Info(band_name="B06", wl_name="nir", wavelength=740, scale_factor=2)
            self.channel_info["B07"] = Channel_Info(band_name="B07", wl_name="nir", wavelength=783, scale_factor=2)
            self.channel_info["B08"] = Channel_Info(band_name="B08", wl_name="nir", wavelength=832, scale_factor=1)
            self.channel_info["B09"] = Channel_Info(band_name="B09", wl_name="swir", wavelength=945, scale_factor=6)
            self.channel_info["B10"] = Channel_Info(band_name="B10", wl_name="swir", wavelength=1375, scale_factor=6)
            self.channel_info["B11"] = Channel_Info(band_name="B11", wl_name="swir", wavelength=1610, scale_factor=2)
            self.channel_info["B12"] = Channel_Info(band_name="B12", wl_name="swir", wavelength=2190, scale_factor=2)
            self.channel_info["B8A"] = Channel_Info(band_name="B8A", wl_name="nir", wavelength=864, scale_factor=2)
        else:
            raise NotImplementedError(f'Channels equal to "{channels}" not implmented.')
        self.n_channels = len(self.channel_info.keys())

        # if ("Drop2-Aligned-TA1-2022-02-15" in self.dset_dir) or ("Aligned-Drop3-TA1-2022-03-10" in self.dset_dir):
        #     kwcoco_path = os.path.join(root_dir, "data_vali.kwcoco.json")
        # else:
        #     kwcoco_path = os.path.join(root_dir, "vali_data.kwcoco.json")

        # Load kwcoco dataset
        self.coco_dset = kwcoco.CocoDataset(kwcoco_path)

        # Get number of classes for dataset with given task mode.
        if task_mode == "total_bin_change":
            self.n_classes = 2
        elif task_mode == "ss_mat_recon":
            self.n_classes = self.n_channels
        elif task_mode == "total_sem_change":
            self.n_classes = 4
        elif task_mode == "refine_sc":
            self.n_classes = 4
        else:
            raise NotImplementedError(f'Number of classes for this task "{task_mode}" needs to be added.')

        # Get all video ids
        video_ids = [video_info["id"] for video_info in self.coco_dset.videos().objs]

        # Only get videos and images with desired sensor
        self.video_dataset = {}  # video_id -> image_ids (filtered)
        self.max_frames = 0
        self.examples = []
        for video_id in video_ids:
            # Get image ids for this video
            image_ids = self.coco_dset.index.vidid_to_gids[video_id]

            # Sort image_ids by datetime.
            datetimes = [self.get_datetime(image_id) for image_id in image_ids]
            image_ids = [img_id for dt, img_id in sorted(zip(datetimes, image_ids))]

            # Filter based on desired sensor type
            sesnor_image_ids = []
            for image_id in image_ids:
                img_sensor_type = self.coco_dset.index.imgs[image_id]["sensor_coarse"]
                if img_sensor_type == sensor_type:
                    sesnor_image_ids.append(image_id)

            # Update maximum number of frames.
            self.max_frames = max(len(sesnor_image_ids), self.max_frames)

            # Add to video dataset
            if len(sesnor_image_ids) > 1:
                self.video_dataset[video_id] = sesnor_image_ids

            # Get spatial slices.
            H = self.coco_dset.index.imgs[image_ids[0]]["height"]
            W = self.coco_dset.index.imgs[image_ids[0]]["width"]
            crop_slices = get_crop_slices(
                H, W, self.video_slice.height, self.video_slice.width, step=self.video_slice.stride
            )

            # Get temporal slices.
            temporal_slices, video_indices = self.compute_equal_temporal_sampling(image_ids, video_slice.n_frames)

            # TEST: Make sure that all image ids are covered.
            sampled_image_ids = []
            map(sampled_image_ids.extend, temporal_slices)
            sampled_image_ids = np.unique(sum(temporal_slices, []))

            assert len(image_ids) == len(sampled_image_ids)

            # Get region ID.
            region_name = self.coco_dset.index.videos[video_id]["name"]

            # Poplate slices with more information.
            for crop_slice in crop_slices:
                for temporal_slice, rel_video_indices in zip(temporal_slices, video_indices):
                    crop_info = {
                        "video_id": video_id,
                        "og_height": H,
                        "og_width": W,
                        "space_crop_slice": crop_slice,
                        "region_name": region_name,
                        "image_ids": temporal_slice,
                        "relative_video_indices": rel_video_indices,
                    }
                    self.examples.append(crop_info)

        # Update max frames based on the number of frames.
        if self.n_frames is None:
            self.n_frames = self.max_frames

        if self.max_frames > self.n_frames:
            self.max_frames = self.n_frames

    def cid_to_name(self, cid):
        return self.coco_dset.cats[cid]["name"]

    def cid_to_rgb(self, cid):
        return to_rgba(self.coco_dset.cats[cid]["color"])[:3]

    def get_datetime(self, image_id):
        date_captured = self.coco_dset.index.imgs[image_id]["date_captured"]  # '2017-03-23T09:30:31'
        datetime_object = datetime.strptime(date_captured[:10], "%Y-%m-%d")
        return datetime_object

    def compute_equal_temporal_sampling(self, image_ids, frames_per_video):
        # Assuming image IDs are sorted by dates early to latest.
        # abc = self.compute_equal_temporal_sampling([0,1,2,3,4,5,6,7,8,9], 5)
        N = len(image_ids)
        M = frames_per_video

        all_image_ids, temporal_indices = [], []
        for i in range(N):
            indices = list(np.clip(np.arange(i, M + i) - M + 1, 0, N - 1))
            sub_image_ids = list(np.take(image_ids, indices))
            all_image_ids.append(sub_image_ids)
            temporal_indices.append(indices)

        return all_image_ids, temporal_indices

        # if N == M:
        #     return [image_ids]

        # stride = int(np.ceil((N - 2) / (M - 2)))
        # n_samples = int(np.ceil((N - 2) / (M - 2)))

        # first_frame_id, last_frame_id = image_ids[0], image_ids[-1]

        # middle_ids = image_ids[1:-1]
        # # middle_indices = list(range(image_ids[1:-1]))

        # temporal_slices, all_temporal_indices = [], []
        # for index_offset in range(n_samples):
        #     temporal_indices = []
        #     for m in range(M - 2):
        #         sample_index = index_offset + stride * m
        #         sample_index = np.clip(sample_index, 0, N - 2 - 1)
        #         temporal_indices.append(sample_index)

        #     # Get image ids from indices.
        #     temporal_slice = list(np.take(middle_ids, temporal_indices))

        #     # Attach first and last image IDs.
        #     temporal_slice.insert(0, first_frame_id)
        #     temporal_slice.append(last_frame_id)
        #     temporal_slices.append(temporal_slice)

        #     # Get temporal indices.
        #     for i, _ in enumerate(temporal_indices):
        #         temporal_indices[i] += 1

        #     temporal_indices.insert(0, 0)
        #     temporal_indices.append(len(image_ids) - 1)
        #     all_temporal_indices.append(temporal_indices)

        # return temporal_slices, all_temporal_indices

    def get_total_semantic_change_example(self, index):
        crop_info = self.examples[index]

        # Get all image ids for this video
        image_ids = crop_info["image_ids"]
        frames = []
        for image_id in image_ids:
            frame = self.load_frame(image_id, crop_info["space_crop_slice"])
            frames.append(frame)

        ## Stack images
        video = np.stack(frames, axis=0)  # shape: [frames, channels, height, width]

        ## Format image values (remove nans, scale to 0-1)
        # Remove NaNs in images
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        # Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15

        # Compute target label mask.
        target = self.compute_sc_target_mask(
            image_ids, crop_info["og_height"], crop_info["og_width"], crop_info["space_crop_slice"]
        )
        target = np.stack(target, axis=0)

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, target = self.transforms(video, target)

        # Normalize video.
        video, mean, std = self.normalize(video)

        # Pad video to desired window size.
        if (video.shape[2] < self.video_slice.height) or (video.shape[3] < self.video_slice.width):
            canvas = (
                np.ones(
                    (video.shape[0], video.shape[1], self.video_slice.height, self.video_slice.width), dtype=video.dtype
                )
                * self.ignore_index
            )
            canvas[:, :, : video.shape[2], : video.shape[3]] = video
            video = canvas

        # Pad target mask to desired window size.
        if (target.shape[-2] < self.video_slice.height) or (target.shape[-1] < self.video_slice.width):
            canvas = (
                np.ones((target.shape[0], self.video_slice.height, self.video_slice.width), dtype=target.dtype)
                * self.ignore_index
            )
            canvas[:, : target.shape[-2], : target.shape[-1]] = target
            target = canvas

        # Scale resolution of video and target.
        if self.scale != 1 and self.scale is not None:
            video = self.scale_video(video, self.scale, inter_mode=cv2.INTER_NEAREST)
            target = self.scale_frame(target, self.scale)

        # Convert from numpy array to tensor.
        video, target = np.copy(video), np.copy(target)
        video = torch.as_tensor(video)
        target = torch.as_tensor(target)

        ## Pad the number of frames to maximum number of frames.
        # [frames, channels, height, width]
        active_frames = torch.zeros(self.max_frames)
        active_frames[: video.shape[0]] = torch.ones(video.shape[0])

        pad_video = (
            torch.ones([self.max_frames, video.shape[1], video.shape[2], video.shape[3]], dtype=video.dtype)
            * self.ignore_index
        )
        pad_video[: video.shape[0], ...] = video
        video = pad_video

        ## Get metadata from all images in video.
        datetimes = []
        for image_id in image_ids:
            datetimes.append(self.get_datetime(image_id))

        output = {}
        output["video"] = video.float()
        output[self.task_mode] = target
        output["datetimes"] = datetimes
        output["active_frames"] = active_frames
        output["region_name"] = crop_info["region_name"]
        output["crop_info"] = crop_info
        output["mean"] = mean
        output["std"] = std
        return output

    def compute_sc_target_mask(self, image_ids, height, width, spatial_crop):
        h0, w0, h, w = spatial_crop

        anno_images = []
        for image_id in image_ids:
            anno_ids = self.coco_dset.index.gid_to_aids[image_id]
            dets = kwimage.Detections.from_coco_annots(self.coco_dset.annots(anno_ids).objs, dset=self.coco_dset)
            warp_info = self.coco_dset.imgs[image_id]["warp_img_to_vid"]
            if "offset" in warp_info.keys():
                dets = dets.translate(warp_info["offset"])

            anno_canvas = np.zeros([height, width], dtype=np.int32)
            ignore_canvas = np.zeros([height, width], dtype=np.int32)
            ann_polys = dets.data["segmentations"].to_polygon_list()
            ann_cids = [dets.classes.idx_to_id[cidx] for cidx in dets.data["class_idxs"]]

            for polygon, cat_id in zip(ann_polys, ann_cids):
                cid_name = self.cid_to_name(cat_id).lower()
                if cid_name in ["ignore", "unknown", "negative", "positive"]:
                    polygon.fill(ignore_canvas, value=-1)
                elif cid_name in ["no activity", "background"]:
                    pass
                else:
                    polygon.fill(anno_canvas, value=cat_id)
            anno_images.append(anno_canvas[h0 : (h0 + h), w0 : (w0 + w)])
        return anno_images

    def load_frame(self, image_id, crop_slice):
        """Given an image ID and crop parameters, return a single multispectral image.

        Args:
            image_id (int): A integer representing a unique image ID.
            crop_slice (list(int)): A list of four integers corresponding to
                [row_start, col_start, row_end, col_end].

        Returns:
            np.array: A uint16 numpy array of shape [n_channels, height, width] corresponding to a loaded and
            cropped image.
        """
        h0, w0, h, w = crop_slice

        # Get channels.
        if self.channels == "RGB":
            channel_names = "red|blue|green"
        elif self.channels == "RGB_NIR":
            channel_names = "red|blue|green|nir"
        elif self.channels == "ALL":
            channel_names = None

        try:
            delayed_image = self.coco_dset.delayed_load(image_id, channels=channel_names, space="video")
            delayed_crop = delayed_image.crop((slice(h0, h0 + h), slice(w0, w0 + w)))
            frame = delayed_crop.finalize(as_xarray=False).transpose((2, 0, 1))  # [n_channels, height, width]
        except ValueError:
            frame = np.zeros([self.n_channels, h, w], dtype="float")

        return frame

    def get_pixel_normalization_params(self):
        Norm_Params = namedtuple("Norm_Params", ["mean", "std"])
        if "drop2" in self.dset_dir:
            norm_params_dict = {
                "B02": Norm_Params(mean=2.5880, std=2.5634),  # blue
                "B03": Norm_Params(mean=2.3949, std=2.3728),  # green
                "B04": Norm_Params(mean=2.5746, std=2.5507),  # red
                "B08": Norm_Params(mean=4.0442, std=4.0061),  # nir
            }
        elif "TA1" in self.dset_dir:
            norm_params_dict = {
                "B02": Norm_Params(mean=1.5640, std=1.5374),  # blue
                "B03": Norm_Params(mean=1.4271, std=1.4034),  # green
                "B04": Norm_Params(mean=1.5551, std=1.5293),  # red
                "B08": Norm_Params(mean=2.4479, std=2.4086),  # nir
            }
        else:
            norm_params_dict = {
                "B02": Norm_Params(mean=1.7603, std=1.7328),  # blue
                "B03": Norm_Params(mean=2.0488, std=2.0234),  # green
                "B04": Norm_Params(mean=1.9127, std=1.8875),  # red
                "B08": Norm_Params(mean=3.3785, std=3.3383),  # nir
            }
        mean_vals, std_vals = [], []
        if self.channels == "RGB":
            for band_name in ["B04", "B03", "B02"]:
                mean_vals.append(norm_params_dict[band_name].mean)
                std_vals.append(norm_params_dict[band_name].std)
        elif self.channels == "RGB_NIR":
            for band_name in ["B04", "B03", "B02", "B08"]:
                mean_vals.append(norm_params_dict[band_name].mean)
                std_vals.append(norm_params_dict[band_name].std)
        elif self.channels == "ALL":
            raise NotImplementedError("Global normalization parameters not generated for ALL channels.")
        else:
            raise NotImplementedError(f"No global normalization process for channels name {self.channels}")
        mean_vals, std_vals = np.array(mean_vals), np.array(std_vals)

        # Format shapes.
        mean_vals = mean_vals[np.newaxis, :, np.newaxis, np.newaxis]
        std_vals = std_vals[np.newaxis, :, np.newaxis, np.newaxis]
        return mean_vals, std_vals


if __name__ == "__main__":
    # Get root directory of dataset on local machine.
    from geowatch.tasks.rutgers_material_change_detection.utils.util_paths import get_dataset_root_dir
    from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import generate_video_slice_object

    dset_name = "iarpa_drop2v2"
    root_dir = get_dataset_root_dir(dset_name)

    # Get dataset split and task_mode.
    split = "valid"
    task_mode = "total_sem_change"

    # Test loading first and last frames of training split.
    video_slice = generate_video_slice_object(height=300, width=300, n_frames=5, scale=1)
    dset = IARPA_SC_EVAL_DATASET(root_dir, split, video_slice, task_mode, channels="RGB_NIR", normalize_mode=None)
    example = dset.__getitem__(0)
