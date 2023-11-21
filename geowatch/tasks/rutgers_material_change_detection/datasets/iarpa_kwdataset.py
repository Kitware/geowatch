import os
from datetime import datetime
from collections import namedtuple

import torch
import kwcoco
import kwimage
import numpy as np
from matplotlib.colors import to_rgba

from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_crop_slices
from geowatch.tasks.rutgers_material_change_detection.datasets.base_dataset import BaseDataset


class IARPA_KWDATASET(BaseDataset):
    def __init__(
        self,
        root_dir,
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
            kwcoco_path (str): File path to kwcoco file.
            video_slice (namedtuple):
            task_mode (str, str): Name of labels to produce for this dataset. Defaults to 'total_bin_change'.
            sensor_type (str, optional): The name of sensor to return image data from. Defaults to 'S2'.
        """
        super().__init__(
            root_dir,
            split,
            video_slice,
            task_mode,
            transforms=transforms,
            seed_num=seed_num,
            normalize_mode=normalize_mode,
            channels=channels,
            max_iterations=max_iterations,
        )

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

        # Get number of classes for dataset with given task mode.
        if task_mode == "total_bin_change":
            self.n_classes = 2
        elif task_mode == "ss_mat_recon":
            self.n_classes = self.n_channels
        elif task_mode == "total_sem_change":
            self.n_classes = 15
        else:
            raise NotImplementedError(f'Number of classes for this task "{task_mode}" needs to be added.')

        if split == "train":
            kwcoco_path = os.path.join(root_dir, "train_data.kwcoco.json")
        elif split == "valid":
            kwcoco_path = os.path.join(root_dir, "vali_data.kwcoco.json")
        else:
            raise NotImplementedError(f'Mode "{split}" not implemented for IARPA KWCOCO Dataset class.')

        self.dset_path = os.path.split(kwcoco_path)[0]
        self.transforms = transforms
        self.task_mode = task_mode
        self._sensor_type = sensor_type

        # Load kwcoco dataset
        self.coco_dset = kwcoco.CocoDataset(kwcoco_path)

        # Get all video ids
        video_ids = [video_info["id"] for video_info in self.coco_dset.videos().objs]

        # Only get videos and images with desired sensor
        self.video_dataset = {}  # video_id -> image_ids (filtered)
        self.max_frames = 0
        for video_id in video_ids:
            # Get image ids for this video
            image_ids = self.coco_dset.index.vidid_to_gids[video_id]

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

        # Create all slice versions
        self.examples = []
        for video_id, image_ids in self.video_dataset.items():
            # Get video shape
            H = self.coco_dset.index.imgs[image_ids[0]]["height"]
            W = self.coco_dset.index.imgs[image_ids[0]]["width"]
            crop_slices = get_crop_slices(
                H, W, self.video_slice.height, self.video_slice.width, step=self.video_slice.stride
            )

            # Get region ID.
            region_name = self.coco_dset.index.videos[video_id]["name"]

            # Poplate slices with more information.
            for crop_slice in crop_slices:
                crop_info = {
                    "video_id": video_id,
                    "og_height": H,
                    "og_width": W,
                    "space_crop_slice": crop_slice,
                    "region_name": region_name,
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

    def load_frames(self, image_ids, crop_slice):
        h0, w0, h, w = crop_slice

        # Get channels.
        if self.channels == "RGB":
            channel_names = "red|blue|green"
        elif self.channels == "RGB_NIR":
            channel_names = "red|blue|green|nir"
        elif self.channels == "ALL":
            channel_names = None

        # Load all of the images and slice according to crop
        image_crops = []
        for image_id in image_ids:
            delayed_crop = self.coco_dset.delayed_load(image_id, channels=channel_names, space="video").crop(
                (slice(h0, h0 + h), slice(w0, w0 + w))
            )
            delayed_crop = delayed_crop.finalize(as_xarray=False).transpose((2, 0, 1))
            image_crops.append(delayed_crop)

        return image_crops, image_ids

    def subset_image_ids(self, image_ids):

        # Sort image_ids by datetimes.
        datetimes = [self.get_datetime(image_id) for image_id in image_ids]
        image_ids = [img_id for dt, img_id in sorted(zip(datetimes, image_ids))]

        if len(image_ids) <= self.max_frames:
            # Grab all of the frames.
            new_image_ids = image_ids
        elif self.max_frames == 2:
            new_image_ids = [image_ids[0], image_ids[-1]]
        elif len(image_ids) > self.max_frames:
            if self.split == "train":
                if self.task_mode == "ss_mat_recon":
                    # Randomly sample some of the image ids from a video.
                    new_image_ids = list(np.random.choice(image_ids, self.max_frames))
                else:
                    # Randomly sample from indices between (0, last).
                    indices = np.arange(1, len(image_ids) - 1)
                    frame_indices = sorted(np.random.choice(indices, size=self.max_frames - 2, replace=False))
                    new_image_ids = list(np.take(np.asarray(image_ids), frame_indices))
                    new_image_ids.insert(0, image_ids[0])
                    new_image_ids.append(image_ids[-1])
            else:
                n_avail_img_ids = len(image_ids) - 2
                n_sub_frames = self.max_frames - 2
                indices = list(range(n_avail_img_ids))

                slice = n_avail_img_ids // n_sub_frames

                indices = indices[slice // 2 :: slice]
                indices.insert(0, 0)
                indices.append(-1)
                new_image_ids = np.take(image_ids, indices)

        return new_image_ids

    def get_total_binary_change_example(self, index):
        crop_info = self.examples[index]

        # Get all image ids for this video
        image_ids = self.video_dataset[crop_info["video_id"]]
        image_ids = self.subset_image_ids(image_ids)
        frames, selected_image_ids = self.load_frames(image_ids, crop_info["space_crop_slice"])

        ## Stack images
        video = np.stack(frames, axis=0)  # shape: [frames, channels, height, width]

        ## Format image values (remove nans, scale to 0-1)
        # Remove NaNs in images
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        # Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15
        video = video.astype("float32")

        # Compute target label mask.
        target, ignore_mask = self.compute_target_mask(
            image_ids,
            self.task_mode,
            og_height=crop_info["og_height"],
            og_width=crop_info["og_width"],
            spatial_crop=crop_info["space_crop_slice"],
        )

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, target, ignore_mask = self.transforms(video, target, ignore_mask)

        # Normalize video.
        video = self.normalize(video)

        # Pad video to desired window size.
        if (video.shape[2] < self.video_slice.height) or (video.shape[3] < self.video_slice.width):
            canvas = (
                np.ones(
                    (video.shape[0], video.shape[1], self.video_slice.height, self.video_slice.width), dtype=video.dtype
                )
                * self.pad_value
            )
            canvas[:, :, : video.shape[2], : video.shape[3]] = video
            video = canvas

        # Pad target mask to desired window size.
        if (target.shape[0] < self.video_slice.height) or (target.shape[1] < self.video_slice.width):
            canvas = np.ones((self.video_slice.height, self.video_slice.width), dtype=target.dtype) * self.pad_value
            canvas[: target.shape[0], : target.shape[1]] = target
            target = canvas

        # Pad target mask to desired window size.
        if (ignore_mask.shape[0] < self.video_slice.height) or (ignore_mask.shape[1] < self.video_slice.width):
            canvas = (
                np.ones((self.video_slice.height, self.video_slice.width), dtype=ignore_mask.dtype) * self.pad_value
            )
            canvas[: ignore_mask.shape[0], : ignore_mask.shape[1]] = ignore_mask
            ignore_mask = canvas

        # Scale resolution of video and target.
        if self.scale != 1 and self.scale is not None:
            video, target = self.scale_video_target(video, target, self.scale)

        # Convert from numpy array to tensor.
        video, target, ignore_mask = np.copy(video), np.copy(target), np.copy(ignore_mask)
        video = torch.as_tensor(video)
        target = torch.as_tensor(target)
        ignore_mask = torch.as_tensor(ignore_mask)

        ## Pad the number of frames to maximum number of frames.
        # [frames, channels, height, width]
        active_frames = torch.zeros(self.max_frames)
        active_frames[: video.shape[0]] = torch.ones(video.shape[0])

        pad_video = (
            torch.ones([self.max_frames, video.shape[1], video.shape[2], video.shape[3]], dtype=video.dtype)
            * self.pad_value
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
        output["ignore_mask"] = ignore_mask
        return output

    def get_total_semantic_change_example(self, index):
        crop_info = self.examples[index]

        # Get all image ids for this video
        image_ids = self.video_dataset[crop_info["video_id"]]
        image_ids = self.subset_image_ids(image_ids)
        frames, selected_image_ids = self.load_frames(image_ids, crop_info["space_crop_slice"])

        ## Stack images
        video = np.stack(frames, axis=0)  # shape: [frames, channels, height, width]

        ## Format image values (remove nans, scale to 0-1)
        # Remove NaNs in images
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        # Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15

        # Compute target label mask.
        target = self.compute_target_mask(
            image_ids,
            self.task_mode,
            og_height=crop_info["og_height"],
            og_width=crop_info["og_width"],
            spatial_crop=crop_info["space_crop_slice"],
        )

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, target = self.transforms(video, target)

        # Normalize video.
        video = self.normalize(video)

        # Pad video to desired window size.
        if (video.shape[2] < self.video_slice.height) or (video.shape[3] < self.video_slice.width):
            canvas = (
                np.ones(
                    (video.shape[0], video.shape[1], self.video_slice.height, self.video_slice.width), dtype=video.dtype
                )
                * self.pad_value
            )
            canvas[:, :, : video.shape[2], : video.shape[3]] = video
            video = canvas

        # Pad target mask to desired window size.
        if (target.shape[0] < self.video_slice.height) or (target.shape[1] < self.video_slice.width):
            canvas = np.ones((self.video_slice.height, self.video_slice.width), dtype=target.dtype) * self.pad_value
            canvas[: target.shape[0], : target.shape[1]] = target
            target = canvas

        # Scale resolution of video and target.
        if self.scale != 1 and self.scale is not None:
            video, target = self.scale_video_target(video, target, self.scale)

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
            * self.pad_value
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
        return output

    def get_self_supervised_material_reconstruction_example(self, index):
        crop_info = self.examples[index]

        # Get all image ids for this video
        image_ids = self.video_dataset[crop_info["video_id"]]
        image_ids = self.subset_image_ids(image_ids)
        frames, selected_image_ids = self.load_frames(image_ids, crop_info["space_crop_slice"])

        ## Stack images
        video = np.stack(frames, axis=0)  # shape: [frames, channels, height, width]

        ## Format image values (remove nans, scale to 0-1)
        # Remove NaNs in images
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        # Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15
        video = video.astype("float32")

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video = self.transforms(video)

        # Normalize video.
        video = self.normalize(video)

        # Pad video to desired size.
        active_frames = torch.zeros(self.max_frames)
        active_frames[: video.shape[0]] = torch.ones(video.shape[0])

        if (video.shape[2] < self.video_slice.height) or (video.shape[3] < self.video_slice.width):
            canvas = (
                np.ones(
                    (video.shape[0], video.shape[1], self.video_slice.height, self.video_slice.width), dtype=video.dtype
                )
                * self.pad_value
            )
            canvas[:, :, : video.shape[2], : video.shape[3]] = video
            video = canvas

        pad_video = (
            np.ones([self.max_frames, video.shape[1], video.shape[2], video.shape[3]], dtype=video.dtype)
            * self.pad_value
        )
        pad_video[: video.shape[0], ...] = video
        video = pad_video

        # Scale resolution.
        if self.scale != 1 and self.scale is not None:
            video, _ = self.scale_video_target(video, None, self.scale)

        # Convert from numpy array to tensor.
        video = np.copy(video)
        video = torch.from_numpy(video)

        # Get metadata from all images in video.
        datetimes = []
        for image_id in image_ids:
            datetimes.append(self.get_datetime(image_id))

        # Create output dictionary.
        output = {}
        output["video"] = video
        output[self.task_mode] = video
        output["datetimes"] = datetimes
        output["active_frames"] = active_frames
        return output

    def get_datetime(self, image_id):
        date_captured = self.coco_dset.index.imgs[image_id]["date_captured"]  # '2017-03-23T09:30:31'
        try:
            datetime_object = datetime.strptime(date_captured[:-6], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            datetime_object = datetime.strptime(date_captured[:-6], "%Y-%m-%dT%H")
        return datetime_object

    def compute_target_mask(self, image_ids, mode, og_height, og_width, spatial_crop):
        h0, w0, h, w = spatial_crop

        # Sort image_ids by datetimes.
        datetimes = [self.get_datetime(image_id) for image_id in image_ids]
        image_ids = [img_id for dt, img_id in sorted(zip(datetimes, image_ids))]

        if mode == "total_bin_change":
            # Compute the change between first and last annotation image in video.

            # Get first and last annotation images.
            # first_anno, first_ignore = self.compute_segmentation_anno(image_ids[0], og_height, og_width)
            # TODO: Assume that the labels are propagated.
            final_anno, final_ignore = self.compute_segmentation_anno(image_ids[-1], og_height, og_width)

            # Crop annotations according to slice
            # first_anno = first_anno[h0:(h0 + h), w0:(w0 + w)]
            # first_ignore = first_ignore[h0:(h0 + h), w0:(w0 + w)]
            final_anno = final_anno[h0 : (h0 + h), w0 : (w0 + w)]
            final_ignore = final_ignore[h0 : (h0 + h), w0 : (w0 + w)]

            # Compute the difference between annotations.
            target_mask = np.zeros(final_anno.shape, dtype=final_anno.dtype)
            target_mask = np.clip(final_anno, 0, 1)
            # x, y = np.where(first_anno == final_anno)  # find where labels match (no-change)
            # target_mask[x, y] = 0

            # Set ignore labels.
            x, y = np.where(final_ignore == 1)
            # x, y = np.where((first_ignore == 1) & (final_ignore == 1))
            target_mask[x, y] = -1
            ignore_mask = final_ignore

        elif mode == "total_sem_change":
            # Get first and last annotation images.
            final_anno, final_ignore = self.compute_segmentation_anno(image_ids[-1], og_height, og_width)

            # Crop annotations according to slice
            final_anno = final_anno[h0 : (h0 + h), w0 : (w0 + w)]
            final_ignore = final_ignore[h0 : (h0 + h), w0 : (w0 + w)]

            # Set ignore labels.
            x, y = np.where(final_ignore == 1)
            final_anno[x, y] = -1

            target_mask = final_anno

        elif mode == "bas":
            breakpoint()
            pass

        elif mode == "pw_seq_binary":
            # Compute the change between each sequential pair of annotations
            assert NotImplementedError(mode)
        elif mode == "pw_all_binary":
            # Compute the change between all combinations of annotations
            # TODO: Figure out how to handle this properly
            assert NotImplementedError(mode)
        elif mode == "segmentation":
            # Return the propagated construction stage masks
            assert NotImplementedError(mode)
        else:
            assert NotImplementedError(f"Invalid mode type: {mode}")

        return target_mask, ignore_mask

    def compute_segmentation_anno(self, image_id, og_height, og_width):
        anno_ids = self.coco_dset.index.gid_to_aids[image_id]
        dets = kwimage.Detections.from_coco_annots(self.coco_dset.annots(anno_ids).objs, dset=self.coco_dset)

        warp_info = self.coco_dset.imgs[image_id]["warp_img_to_vid"]
        dets = dets.scale(warp_info["scale"])
        if "offset" in warp_info.keys():
            dets = dets.translate(warp_info["offset"])

        anno_canvas = np.zeros([og_height, og_width], dtype=np.int32)
        ignore_canvas = np.zeros([og_height, og_width], dtype=np.int32)
        ann_polys = dets.data["segmentations"].to_polygon_list()
        ann_cids = [dets.classes.idx_to_id[cidx] for cidx in dets.data["class_idxs"]]

        for polygon, cat_id in zip(ann_polys, ann_cids):
            cid_name = self.cid_to_name(cat_id).lower()
            if cid_name in ["ignore", "unknown", "negative"]:
                polygon.fill(ignore_canvas, value=1)
            else:
                polygon.fill(anno_canvas, value=cat_id)

        return anno_canvas, ignore_canvas

    def to_rgb(self, image, gamma=0.9):
        # Get RGB bands.
        if self.n_channels in [3, 4]:
            r_band = image[0] * 0.9
            g_band = image[1] * 1.0
            b_band = image[2] * 0.9
        elif self.n_channels == 13:
            r_band = image[3]
            g_band = image[2]
            b_band = image[1]
        else:
            raise NotImplementedError(f'Cannot handle "{self.n_channels}" number of channels in to_rgb method.')

        rgb_image = np.stack((r_band, g_band, b_band), axis=2)

        if "TA1" in self.dset_dir:
            rgb_image *= 8
        else:
            rgb_image *= 8

        # Adjust color of image.
        # Assuming that image is from the output of __getitem__
        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_image = rgb_image ** (1 / gamma)

        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_image *= 255
        rgb_image = rgb_image.astype("uint8")

        return rgb_image

    def get_image_paths_from_image_id(self, image_id):
        image_infos = self.coco_dset.index.imgs[image_id]["auxiliary"]

        data = {}
        for image_info in image_infos:
            band_name = image_info["channels"]
            rel_file_path = image_info["file_name"]

            data[band_name] = rel_file_path

        return data

    def get_pixel_normalization_params(self):
        Norm_Params = namedtuple("Norm_Params", ["mean", "std"])
        if "TA1" in self.dset_dir:
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

        mean_vals, std_vals = np.array(mean_vals), np.array(std_vals)

        # Format shapes.
        mean_vals = mean_vals[np.newaxis, :, np.newaxis, np.newaxis]
        std_vals = std_vals[np.newaxis, :, np.newaxis, np.newaxis]
        return mean_vals, std_vals

    def color_annotation(self, anno_image):
        """Create a RGB image containing annotations.

        Args:
            anno_image (np.array): A int numpy array of shape [height, width].

        Returns:
            np.array: A uint8 numpy arry of shape [height, width, 3].
        """
        height, width = anno_image.shape
        color_anno_image = np.zeros([height, width, 3], dtype="float")

        class_ids = np.unique(anno_image)
        for class_id in list(class_ids):
            if class_id not in [0, -1]:
                x, y = np.where(class_id == anno_image)
                rgb_value = self.cid_to_rgb(class_id)
                color_anno_image[x, y, :] = rgb_value

        color_anno_image = color_anno_image * 255
        color_anno_image = color_anno_image.astype("uint8")

        return color_anno_image

    def generate_propagated_annotation_gif(self, index, save_path=None):
        # Get video id.
        crop_info = self.examples[index]
        video_id = crop_info["video_id"]

        # Get image ids from video id.
        image_ids = self.video_dataset[video_id]

        # Get annotation images for each image ID.
        og_height, og_width = crop_info["og_height"], crop_info["og_width"]
        anno_images = [self.compute_segmentation_anno(img_id, og_height, og_width)[0] for img_id in image_ids]

        # Compute RGB versions of annotation images.
        rgb_anno_images = [self.color_annotation(a_img) for a_img in anno_images]

        # Generate GIF of annotation images.
        from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import create_gif

        ## Get save path.
        if save_path is None:
            save_path = f"iarpa_anno_{str(index).zfill(3)}.gif"

        ## Get datetimes in str format.
        datetime_strs = [
            f"[{i}/{len(image_ids)}] " + self.get_datetime(img_id).strftime("%m/%d/%Y")
            for i, img_id in enumerate(image_ids)
        ]
        create_gif(rgb_anno_images, save_path, image_text=datetime_strs)


if __name__ == "__main__":
    from geowatch.tasks.rutgers_material_change_detection.utils.util_paths import get_dataset_root_dir
    from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import generate_video_slice_object

    root_dir = get_dataset_root_dir("iarpa_drop1")
    # root_dir = get_dataset_root_dir('iarpa_drop1_ta1')
    task_mode = "total_bin_change"

    # Create dataset and load one example.
    split = "train"
    video_slice = generate_video_slice_object(height=150, width=150, n_frames=None, scale=2)
    dset = IARPA_KWDATASET(root_dir, split, video_slice, task_mode, channels="RGB_NIR", normalize_mode="local")
    example = dset.__getitem__(4)
    print("TEST 1: Load one example from dataset. | PASSED")

    # Check that validation set of dataset works.
    split = "valid"
    video_slice = generate_video_slice_object(height=400, width=400, n_frames=5, scale=None)
    dset = IARPA_KWDATASET(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    example = dset.__getitem__(0)
    print("TEST 2: Get an example from validation set. | PASSED")

    # Visualize one example of dataset.
    split = "valid"
    video_slice = generate_video_slice_object(height=300, width=300, n_frames=None, scale=None)
    dset = IARPA_KWDATASET(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    ex_index = 1
    dset.visualize_example(ex_index, save_path=f"iarpa_data_ex_{str(ex_index).zfill(2)}.png", overlay_last_anno=True)
    print("TEST 3: Create a visualizion. | PASSED")

    # Visualize datetimes of one example.
    # split = 'train'
    # video_slice = generate_video_slice_object(height=100, width=100, n_frames=None, scale=None)
    # dset = IARPA_KWDATASET(root_dir, split, video_slice, task_mode, channels='RGB_NIR')
    # ex_index = 10
    # dset.visualize_dates(ex_index, save_path=f'iarpa_data_dates_ex_{str(ex_index).zfill(2)}.png')
    # print('TEST 4: Create a dates visualizion. | PASSED')

    # Visualize image annotations over time.
    split = "valid"
    video_slice = generate_video_slice_object(height=800, width=800, n_frames=None, scale=None)
    dset = IARPA_KWDATASET(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    ex_index = 0
    dset.generate_propagated_annotation_gif(ex_index)
    print("TEST 5: Generate a video of annotations over a region. | PASSED")
