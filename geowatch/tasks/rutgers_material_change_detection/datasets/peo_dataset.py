from glob import glob
from datetime import datetime
from collections import namedtuple

import cv2
import torch
import numpy as np
from tifffile import tifffile

from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_crop_slices
from geowatch.tasks.rutgers_material_change_detection.datasets.base_dataset import BaseDataset


class PassiveEarthObservationDataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        split,
        video_slice,
        task_mode,
        seed_num=0,
        transforms=None,
        normalize_mode=None,
        channels=None,
        max_iterations=None,
    ):
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

        if task_mode == "ss_arrow_of_time":
            self.ss_aot_reverse_prob = 0.5
            print("WARNING: Need to add arguments for pct reverse this method.")
        elif task_mode == "ss_splice_change":
            self.ss_splice_prob = 0.5
            print("WARNING: Need to add arguments for pct splice method.")

        if (self.normalize_mode == "local") and (task_mode in ["ss_splice_change", "ss_splice_change_index"]):
            print("WARNING: Does not make sense to normalize this task locally.")

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
            self.channel_info["B11"] = Channel_Info(band_name="B11", wl_name="swir", wavelength=1610, scale_factor=2)
            self.channel_info["B12"] = Channel_Info(band_name="B12", wl_name="swir", wavelength=2190, scale_factor=2)
            self.channel_info["B8A"] = Channel_Info(band_name="B8A", wl_name="nir", wavelength=864, scale_factor=2)
        else:
            raise NotImplementedError(f'Channels equal to "{channels}" not implmented.')
        self.n_channels = len(self.channel_info.keys())

        # Get all video directories.
        video_dirs = sorted(glob(root_dir + "/**/crop_*", recursive=True))

        # Compute dataset splits, i.e. train and validation. No test split.
        if split == "test":
            raise ValueError("No test split for PEO dataset, only train and validation.")
        elif split in ["train", "valid"]:
            # Randomly divide video directories with a 90/10 split.
            np.random.shuffle(video_dirs)
            train_split_pct, valid_split_pct = 0.9, 0.1
            assert (train_split_pct + valid_split_pct) == 1.0
            n_videos = len(video_dirs)

            split_num = int(n_videos * train_split_pct)

            if split == "train":
                split_video_dirs = video_dirs[:split_num]
            else:
                split_video_dirs = video_dirs[split_num:]
        else:
            raise NotImplementedError(f'Split name "{split}" is an invalid split type.')

        # Get crop slices.
        # Assume that all band images are size 1098x1098 when resized to highest resolution band.
        og_image_height, og_image_width = 1098, 1098
        crop_slices = get_crop_slices(
            og_image_height, og_image_width, video_slice.height, video_slice.width, step=video_slice.stride
        )

        # Create examples for each video.
        self.dset_slices = []
        self.max_frames = 0
        for video_dir in split_video_dirs:
            # Get all image directories in ascending temporal order.
            image_dirs = glob(video_dir + "/**/")

            ## Get date from image directory names.
            datetimes = []
            for image_dir in image_dirs:
                datetime_str = image_dir.split("/")[-2].split("_")[2]
                year = int(datetime_str[:4])
                month = int(datetime_str[4:6])
                day = int(datetime_str[6:8])
                datetimes.append(datetime(year, month, day))

            ## Sort image directories by datetimes.
            image_dirs = [image_dir for _, image_dir in sorted(zip(datetimes, image_dirs))]
            datetimes = sorted(datetimes)

            # Update the maximum number of frames in a video.
            self.max_frames = max(len(image_dirs), self.max_frames)

            # Get all image paths for each image directory.
            image_dirs_dict = {}
            for image_dir in image_dirs:
                image_paths = sorted(glob(image_dir + "/*.tif"))
                image_dirs_dict[image_dir] = image_paths

            for crop_slice in crop_slices:
                example = {}
                example["image_dirs_dict"] = image_dirs_dict
                example["og_height"] = og_image_height
                example["og_width"] = og_image_width
                example["space_crop_slice"] = crop_slice
                example["datetimes"] = datetimes

                self.dset_slices.append(example)

        # Update max frames based on the number of frames.
        if self.n_frames is None:
            self.n_frames = self.max_frames

        if self.max_frames > self.n_frames:
            self.max_frames = self.n_frames

        # Get number of classes for dataset with given task mode.
        if task_mode == "ss_arrow_of_time":
            self.n_classes = 2
        elif task_mode == "ss_triplet":
            self.n_classes = 0
        elif task_mode == "ss_splice_change":
            self.n_classes = 2
        elif task_mode == "ss_splice_change_index":
            self.n_classes = self.max_frames
        elif task_mode == "ss_mat_recon":
            self.n_classes = self.n_channels
        else:
            raise NotImplementedError(f'Number of classes for this task "{task_mode}" needs to be added.')

    def to_rgb(self, image, gamma=1.0):

        if self.n_channels in [3, 4]:
            # r_band = image[0] * 1.0
            # b_band = image[1] * 1.0
            # g_band = image[2] * 1.0

            r_band = image[0] * 0.85
            b_band = image[1] * 0.85
            g_band = image[2] * 1.45
        elif self.n_channels == 12:
            r_band = image[1]
            b_band = image[2]
            g_band = image[3]
        else:
            raise NotImplementedError(f'Cannot handle "{self.n_channels}" number of channels in to_rgb method.')
        rgb_image = np.stack((r_band, g_band, b_band), axis=2)

        # Adjust color of image.
        # Assuming that image is from the output of __getitem__
        rgb_image *= 10
        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_image = rgb_image ** (1 / gamma)

        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_image *= 255
        rgb_image = rgb_image.astype("uint8")

        return rgb_image

    def get_pixel_normalization_params(self):
        Norm_Params = namedtuple("Norm_Params", ["mean", "std"])
        norm_params_dict = {
            "B01": Norm_Params(mean=-6.7256, std=6.1035),
            "B02": Norm_Params(mean=-6.5524, std=5.9112),  # blue
            "B03": Norm_Params(mean=-6.2216, std=5.5402),  # green
            "B04": Norm_Params(mean=-6.0112, std=5.2998),  # red
            "B05": Norm_Params(mean=-5.6509, std=4.8835),
            "B06": Norm_Params(mean=-5.1046, std=4.2324),
            "B07": Norm_Params(mean=-4.8723, std=3.9443),
            "B08": Norm_Params(mean=-4.7865, std=3.8355),  # nir
            "B09": Norm_Params(mean=-4.5090, std=3.4673),
            "B11": Norm_Params(mean=-4.8664, std=3.9372),
            "B12": Norm_Params(mean=-5.4962, std=4.7033),
            "B8A": Norm_Params(mean=-4.7161, std=3.7464),
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
            for band_name in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]:
                mean_vals.append(norm_params_dict[band_name].mean)
                std_vals.append(norm_params_dict[band_name].std)

        mean_vals, std_vals = np.array(mean_vals), np.array(std_vals)

        # Format shapes.
        mean_vals = mean_vals[np.newaxis, :, np.newaxis, np.newaxis]
        std_vals = std_vals[np.newaxis, :, np.newaxis, np.newaxis]
        return mean_vals, std_vals

    def load_frame(self, channel_image_paths):
        if self.n_channels == 3:
            # Choose RGB bands
            r_channel_img_path = channel_image_paths[3]  # B04, RED
            g_channel_img_path = channel_image_paths[2]  # B03, GREEN
            b_channel_img_path = channel_image_paths[1]  # B02, BLUE

            r_channel_img = tifffile.imread(r_channel_img_path)
            g_channel_img = tifffile.imread(g_channel_img_path)
            b_channel_img = tifffile.imread(b_channel_img_path)

            frame = np.stack([r_channel_img, g_channel_img, b_channel_img], axis=2)
        elif self.n_channels == 4:
            r_channel_img_path = channel_image_paths[3]  # B04, RED
            g_channel_img_path = channel_image_paths[2]  # B03, GREEN
            b_channel_img_path = channel_image_paths[1]  # B02, BLUE
            nir_channel_img_path = channel_image_paths[7]  # B08, NIR

            r_channel_img = tifffile.imread(r_channel_img_path)
            g_channel_img = tifffile.imread(g_channel_img_path)
            b_channel_img = tifffile.imread(b_channel_img_path)
            nir_channel_img = tifffile.imread(nir_channel_img_path)

            frame = np.stack([r_channel_img, g_channel_img, b_channel_img, nir_channel_img], axis=2)

        elif self.n_channels == 12:

            height, width = tifffile.imread(channel_image_paths[3]).shape

            channel_images = []
            for img_path in channel_image_paths:
                channel_image = tifffile.imread(img_path)

                h_img, w_img = channel_image.shape

                if (h_img != height) or (w_img != width):
                    channel_image = cv2.resize(channel_image, dsize=(width, height))

                channel_images.append(channel_image)

            frame = np.stack(channel_images, axis=2)
            assert (frame.shape[0] == height) and (frame.shape[1] == width)

        return frame

    def load_frames(self, image_dirs, crop_slice, og_height, og_width):
        """Create a video given a set of image directories and crop parameters.

        Args:
            image_dirs (list(str)): A list containing the directories for all image channels paths in a region.
            crop_slice (tuple): A tuple containing [top_height, left_width, height_size, width_size]

        Returns:
            frames (list(np.array)): A list of numpy arrays.
            frame_indices (list(int)): a list of frame indices.
        """
        h0, w0, h, w = crop_slice

        frames, frame_indices = [], []
        if self.max_frames == 2:
            # image_dirs = [image_dirs[0], image_dirs[-1]]
            frame_indices = [0, len(image_dirs)]
        elif self.max_frames >= len(image_dirs):
            frame_indices = np.arange(len(image_dirs))
        else:
            # Randomly sample from all frames.
            indices = np.arange(len(image_dirs))
            frame_indices = np.random.choice(indices, size=self.max_frames, replace=False)
            frame_indices = sorted(frame_indices)

        for i, image_dir in enumerate(image_dirs):

            if i in frame_indices:
                channel_image_paths = sorted(glob(image_dir + "/*.tif"))

                frame = self.load_frame(channel_image_paths)

                # TODO: Figure out a better method for matching image shapes.
                if (frame.shape[0] != og_height) or (frame.shape[1] != og_width):
                    frame = cv2.resize(frame, (og_width, og_height))

                crop = frame[h0 : (h0 + h), w0 : (w0 + w), :]  # [height, width, channels]
                frames.append(crop)

        return frames, frame_indices

    def get_self_supervised_triplet_example(self, index):
        example = self.dset_slices[index]

        # Get anchor and positive items.
        ## Get paths to anchor and postive frames.
        image_dirs = list(example["image_dirs_dict"].keys())

        assert len(image_dirs) >= 2

        if len(image_dirs) == 2:
            anchor_index = 0
            positive_index = 0
        else:
            anchor_index = torch.randint(0, len(image_dirs), size=(1,)).item()

            positive_index = anchor_index
            while positive_index == anchor_index:
                positive_index = torch.randint(0, len(image_dirs), size=(1,)).item()

        ## Get crop information.
        h0, w0, h, w = example["space_crop_slice"]
        hE, wE = h0 + h, w0 + w

        ## Load frames.
        anchor_band_paths = sorted(glob(image_dirs[anchor_index] + "/*.tif"))
        anchor_frame = self.load_frame(anchor_band_paths)  # [height, width, n_channels]
        anchor_frame = anchor_frame[h0:hE, w0:wE, :]

        positive_band_paths = sorted(glob(image_dirs[positive_index] + "/*.tif"))
        positive_frame = self.load_frame(positive_band_paths)  # [height, width, n_channels]
        positive_frame = positive_frame[h0:hE, w0:wE, :]

        # Get negative item.
        ## Get index of random negative example.
        neg_ex_index = index
        while neg_ex_index != index:
            neg_ex_index = torch.randint(0, len(image_dirs), size=(1,)).item()
        neg_example = self.dset_slices[neg_ex_index]

        ## Select image directory for neg example.
        neg_image_dirs = list(neg_example["image_dirs_dict"].keys())
        negative_index = torch.randint(0, len(neg_image_dirs), size=(1,)).item()

        ## Load negative frame.
        negative_band_paths = sorted(glob(image_dirs[negative_index] + "/*.tif"))
        negative_frame = self.load_frame(negative_band_paths)  # [height, width, n_channels]
        negative_frame = negative_frame[h0:hE, w0:wE, :]

        # Apply transforms to triplet items.
        ## Stack tripet items into video of shape [frames, channels, height, width].
        triplet_video = np.stack([anchor_frame, positive_frame, negative_frame], axis=0)
        triplet_video = triplet_video.transpose((0, 3, 1, 2))

        ## Remove any NaN values in video.
        nan_indices = np.isnan(triplet_video)
        triplet_video[nan_indices] = 0

        ## Scale from uint16 to [0,1].
        triplet_video = np.clip(triplet_video, 0, 2**15) / 2**15
        triplet_video = triplet_video.astype("float32")

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            triplet_video, _ = self.transforms(triplet_video, target=None)

        # Normalize triplet items.
        triplet_video = self.normalize(triplet_video)

        # Pad size of triplet examples.
        if (triplet_video.shape[2] < self.video_slice.height) or (triplet_video.shape[3] < self.video_slice.width):
            canvas = (
                np.ones(
                    (triplet_video.shape[0], triplet_video.shape[1], self.video_slice.height, self.video_slice.width),
                    dtype=triplet_video.dtype,
                )
                * self.pad_value
            )
            canvas[:, :, : triplet_video.shape[2], : triplet_video.shape[3]] = triplet_video
            triplet_video = canvas

        # Scale resolution.
        if self.scale != 1 and self.scale is not None:
            triplet_video, _ = self.scale_video_target(triplet_video, None, self.scale)

        # Convert numpy arrays to tensor.
        # triplet_video = np.copy(triplet_video)
        triplet_video = torch.from_numpy(triplet_video)
        anchor_frame = triplet_video[0]  # [channels, height, width]
        positive_frame = triplet_video[1]  # [channels, height, width]
        negative_frame = triplet_video[2]  # [channels, height, width]

        output = {}
        output["anchor"] = anchor_frame
        output["positive"] = positive_frame
        output["negative"] = negative_frame
        output[self.task_mode] = 0  # dummy variable

        return output

    def get_self_supervised_arrow_of_time_example(self, index):
        example = self.dset_slices[index]

        # Load video.
        frames, frame_indices = self.load_frames(
            example["image_dirs_dict"], example["space_crop_slice"], example["og_height"], example["og_width"]
        )
        video = np.stack(frames, axis=0)  # [frames, height, width, channels]
        video = video.transpose((0, 3, 1, 2))

        # Remove any NaN values in video.
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        # Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15
        video = video.astype("float32")

        # Reverse the arrow of time based on probability.
        coin = torch.rand(1).item()
        if coin <= self.ss_aot_reverse_prob:
            video = video[::-1]
            reverse = torch.tensor(1)
        else:
            reverse = torch.tensor(0)

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, _ = self.transforms(video, target=None)

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

        # Create output dictionary.
        output = {}
        output["video"] = video
        output[self.task_mode] = reverse  # Int [0 or 1]
        output["datetimes"] = example["datetimes"]
        output["active_frames"] = active_frames
        return output

    def get_self_supervised_splice_change_example(self, index):
        example = self.dset_slices[index]

        # Load original video.
        frames, _ = self.load_frames(
            example["image_dirs_dict"], example["space_crop_slice"], example["og_height"], example["og_width"]
        )
        video = np.stack(frames, axis=0)  # [frames, height, width, channels]
        video = video.transpose((0, 3, 1, 2))

        ## Remove any NaN values in video.
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        ## Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15
        video = video.astype("float32")

        ## Normalize orignal video.
        video = self.normalize(video)

        ## Pad video to desired resolution.
        video = self.pad_video_height_width(video, self.video_slice.height, self.video_slice.width)

        # Possibly load another video to splice.
        coin = torch.rand(1).item()
        if coin <= self.ss_splice_prob:
            # Load another video.

            ## Update target.
            splice_value = torch.tensor(1)

            ## Get another example index from dataset.
            other_example_index = index
            indices = list(range(0, self.__len__()))
            indices.remove(index)
            other_example_index = np.random.choice(indices, size=1, replace=False)[0]

            example = self.dset_slices[other_example_index]

            ## Load video and process it.
            frames, _ = self.load_frames(
                example["image_dirs_dict"], example["space_crop_slice"], example["og_height"], example["og_width"]
            )
            splice_video = np.stack(frames, axis=0)  # [frames, height, width, channels]
            splice_video = splice_video.transpose((0, 3, 1, 2))  # [frames, channels, height, width]

            ### Scale from uint16 to [0,1].
            splice_video = np.clip(splice_video, 0, 2**15) / 2**15
            splice_video = splice_video.astype("float32")

            ### Normalize orignal video.
            splice_video = self.normalize(splice_video)

            ### Pad video to desired resolution.
            splice_video = self.pad_video_height_width(splice_video, self.video_slice.height, self.video_slice.width)

            # Splice videos together.
            vid_1_size = np.random.randint(1, video.shape[0])
            # vid_2_size = np.random.randint(1, splice_video.shape[0] - 1)
            vid_2_size = self.max_frames - vid_1_size
            vid_1_first_index = np.random.randint(0, video.shape[0] - vid_1_size)
            vid_2_first_index = np.random.randint(0, splice_video.shape[0] - vid_2_size)

            vid_1 = video[vid_1_first_index : vid_1_first_index + vid_1_size]
            vid_2 = splice_video[vid_2_first_index : vid_2_first_index + vid_2_size]

            video = np.concatenate((vid_1, vid_2), axis=0)

        else:
            # Dont load another video to splice.
            ## Randomly sample another video.
            splice_value = torch.tensor(0)

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, _ = self.transforms(video, target=None)

        # Pad video to desired frames.
        active_frames = torch.zeros(self.max_frames)
        active_frames[: video.shape[0]] = torch.ones(video.shape[0])

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

        # Create output dictionary.
        output = {}
        output["video"] = video
        output[self.task_mode] = splice_value  # Int [0 or 1]
        output["datetimes"] = example["datetimes"]
        output["active_frames"] = active_frames
        return output

    def get_self_supervised_splice_change_index_example(self, index):
        example = self.dset_slices[index]

        # Load original video.
        frames, _ = self.load_frames(
            example["image_dirs_dict"], example["space_crop_slice"], example["og_height"], example["og_width"]
        )
        video = np.stack(frames, axis=0)  # [frames, height, width, channels]
        video = video.transpose((0, 3, 1, 2))

        ## Remove any NaN values in video.
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        ## Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15
        video = video.astype("float32")

        ## Normalize orignal video.
        video = self.normalize(video)

        ## Pad video to desired resolution.
        video = self.pad_video_height_width(video, self.video_slice.height, self.video_slice.width)

        # Load another video to splice.

        ## Get another example index from dataset.
        other_example_index = index
        indices = list(range(0, self.__len__()))
        indices.remove(index)
        other_example_index = np.random.choice(indices, size=1, replace=False)[0]

        example = self.dset_slices[other_example_index]

        ## Load video and process it.
        frames, _ = self.load_frames(
            example["image_dirs_dict"], example["space_crop_slice"], example["og_height"], example["og_width"]
        )
        splice_video = np.stack(frames, axis=0)  # [frames, height, width, channels]
        splice_video = splice_video.transpose((0, 3, 1, 2))  # [frames, channels, height, width]

        ### Scale from uint16 to [0,1].
        splice_video = np.clip(splice_video, 0, 2**15) / 2**15
        splice_video = splice_video.astype("float32")

        ### Normalize orignal video.
        splice_video = self.normalize(splice_video)

        ### Pad video to desired resolution.
        splice_video = self.pad_video_height_width(splice_video, self.video_slice.height, self.video_slice.width)

        # Splice videos together.
        vid_1_size = np.random.randint(1, video.shape[0])
        # vid_2_size = np.random.randint(1, splice_video.shape[0] - 1)
        vid_2_size = self.max_frames - vid_1_size
        vid_1_first_index = np.random.randint(0, video.shape[0] - vid_1_size)
        vid_2_first_index = np.random.randint(0, splice_video.shape[0] - vid_2_size)

        vid_1 = video[vid_1_first_index : vid_1_first_index + vid_1_size]
        vid_2 = splice_video[vid_2_first_index : vid_2_first_index + vid_2_size]

        video = np.concatenate((vid_1, vid_2), axis=0)

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, _ = self.transforms(video, target=None)

        # Pad video to desired frames.
        active_frames = torch.zeros(self.max_frames)
        active_frames[: video.shape[0]] = torch.ones(video.shape[0])

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

        # Create output dictionary.
        output = {}
        output["video"] = video
        output[self.task_mode] = torch.tensor(vid_1_size)  # Int [0 or 1]
        output["datetimes"] = example["datetimes"]
        output["active_frames"] = active_frames
        return output

    def get_self_supervised_material_reconstruction_example(self, index):
        example = self.dset_slices[index]

        # Load video.
        frames, frame_indices = self.load_frames(
            example["image_dirs_dict"], example["space_crop_slice"], example["og_height"], example["og_width"]
        )
        video = np.stack(frames, axis=0)  # [frames, height, width, channels]
        video = video.transpose((0, 3, 1, 2))

        # Remove any NaN values in video.
        nan_indices = np.isnan(video)
        video[nan_indices] = 0

        # Scale from uint16 to [0,1].
        video = np.clip(video, 0, 2**15) / 2**15
        video = video.astype("float32")

        # Apply transforms to imagery.
        if self.transforms:
            # Need to create custom transform object
            video, _ = self.transforms(video, target=None)

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

        # Create output dictionary.
        output = {}
        output["video"] = video
        output[self.task_mode] = video
        output["datetimes"] = example["datetimes"]
        output["active_frames"] = active_frames
        return output


if __name__ == "__main__":
    from geowatch.tasks.rutgers_material_change_detection.utils.util_paths import get_dataset_root_dir
    from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import generate_video_slice_object

    root_dir = get_dataset_root_dir("peo")

    # Visualize one example of dataset.
    root_dir = get_dataset_root_dir("peo")
    task_mode = "ss_arrow_of_time"
    video_slice = generate_video_slice_object(height=300, width=300, n_frames=None, scale=1)
    dset = PassiveEarthObservationDataset(root_dir, "train", video_slice, task_mode, channels="RGB_NIR")
    ex_index = 0
    ex = dset.__getitem__(ex_index)
    dset.visualize_example(
        ex_index, save_path=f"peo_ex_{str(ex_index).zfill(2)}.png", overlay_last_anno=True, num_plot_images=6
    )
    print("TEST 1: Create a visualizion. | PASSED")

    # Test 2: Get a single item from the training set in ss_triplet mode.
    video_slice = generate_video_slice_object(height=150, width=150, n_frames=None, scale=2)
    split = "train"
    task_mode = "ss_triplet"
    dset = PassiveEarthObservationDataset(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    example = dset.__getitem__(0)
    print("Test 2: Get one example from train split of dataset. | PASSED")

    # Test 3: Get a single item from the validation set in ss_triplet mode.
    video_slice = generate_video_slice_object(height=300, width=300, n_frames=None, scale=1)
    split = "valid"
    task_mode = "ss_triplet"
    dset = PassiveEarthObservationDataset(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    example = dset.__getitem__(0)
    print("Test 3: Get one example from valid split of dataset. | PASSED")

    # Test 4: Get a single item from the train set in ss_arrow_of_time mode.
    video_slice = generate_video_slice_object(height=300, width=300, n_frames=None, scale=1)
    split = "train"
    task_mode = "ss_arrow_of_time"
    dset = PassiveEarthObservationDataset(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    example = dset.__getitem__(0)
    print("Test 4: Get one example from valid split of dataset in arrow_of_time mode. | PASSED")

    # Test 5: Visualize an ss_splice_change_index example.
    video_slice = generate_video_slice_object(height=300, width=300, n_frames=10, scale=1)
    split = "train"
    task_mode = "ss_splice_change_index"
    dset = PassiveEarthObservationDataset(root_dir, split, video_slice, task_mode, channels="RGB_NIR")
    ex_index = 52
    example = dset.__getitem__(ex_index)
    dset.visualize_example(
        ex_index,
        save_path=f"peo_ss_splice_change_index_{str(ex_index).zfill(2)}.png",
        overlay_last_anno=False,
        num_plot_images=video_slice.n_frames,
    )
    print("Test 5: Visualize an ss_splice_change_index example.| PASSED")
