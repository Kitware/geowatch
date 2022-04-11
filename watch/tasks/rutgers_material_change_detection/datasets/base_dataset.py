import os
import random

import cv2
import numpy as np
from scipy import stats

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        dset_dir,
        split,
        video_slice,
        task_mode,
        transforms=None,
        seed_num=0,
        ignore_index=-1,
        normalize_mode=None,
        channels="RGB",
        max_iterations=None,
    ):
        super().__init__()

        self.split = split
        self.channels = channels
        self.dset_dir = dset_dir
        self.task_mode = task_mode
        self.transforms = transforms
        self.scale = video_slice.scale
        self.video_slice = video_slice
        self.ignore_index = ignore_index
        self.n_frames = video_slice.n_frames
        self.normalize_mode = normalize_mode
        self.max_iterations = max_iterations

        # Set random seed if not equal to -1.
        self.set_seed(seed_num)

        # Get global normalization parameters.
        if self.normalize_mode == "global":
            self.mean_vals, self.std_vals = self.get_pixel_normalization_params()

    def __len__(self):
        if (self.max_iterations is not None) and (self.max_iterations < len(self.examples)):
            return self.max_iterations
        else:
            return len(self.examples)

    def __getitem__(self, index):
        task_mode_fns = {
            "total_bin_change": self.get_total_binary_change_example,
            "total_sem_change": self.get_total_semantic_change_example,
            "pw_bin_change": self.get_pairwise_binary_change_example,
            "pw_sem_change": self.get_pairwise_semantic_change_example,
            "future_frame_pred": self.get_future_frame_prediction_example,
            "sem_seg": self.get_semantic_segmentation_example,
            "ss_triplet": self.get_self_supervised_triplet_example,
            "ss_splice_change": self.get_self_supervised_splice_change_example,
            "ss_crop_splice_change": self.get_self_supervised_crop_splice_change_example,
            "ss_arrow_of_time": self.get_self_supervised_arrow_of_time_example,
            "ss_mat_change": self.get_self_supervised_material_change_example,
            "ss_mat_recon": self.get_self_supervised_material_reconstruction_example,
            "ss_splice_change_index": self.get_self_supervised_splice_change_index_example,
            "bas": self.get_broad_area_search_example,
            "refine_sc": self.get_refine_site_characterization_example,
        }
        try:
            task_mode_fn = task_mode_fns[self.task_mode]
        except KeyError:
            raise NotImplementedError(f'Target task "{self.task_mode}" has not been implemented in BaseDataset class.')
        example = task_mode_fn(index)
        return example

    def set_seed(self, seed_num):
        if seed_num == -1:
            print("No random seed set!")
        else:
            print(f"Setting random seed to: {seed_num}")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.manual_seed(seed_num)
            torch.cuda.manual_seed_all(seed_num)
            np.random.seed(seed_num)
            random.seed(seed_num)

    def get_total_binary_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_total_semantic_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_pairwise_binary_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_pairwise_semantic_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_future_frame_prediction_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_semantic_segmentation_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_triplet_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_splice_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_crop_splice_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_arrow_of_time_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_material_change_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_splice_change_index_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_self_supervised_material_reconstruction_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_refine_site_characterization_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def get_broad_area_search_example(self, index):
        raise NotImplementedError(f'Task "{self.task_mode}" not implemented for this dataset.')

    def to_rgb(self, image):
        raise NotImplementedError("To RGB method not implemented for this dataset.")

    def load_frames(self, image_paths, crop_slice):
        raise NotImplementedError("Load frames not implemented for this dataset.")

    def colorize_target_mask(self, target_mask):
        raise NotImplementedError("Load frames not implemented for this dataset.")

    def compute_class_distribution(self, target_mask, n_classes, ignore_index=-1):
        """Compute the class distribution of a target image as percentages.

        Args:
            target_mask (_type_): TODO: _description_
            n_classes (_type_): TODO: _description_
            ignore_index (int, optional): TODO: _description_. Defaults to -1.
        """
        unique_classes = list(np.unique(target_mask))

        # Remove the ignore index class.
        if ignore_index in unique_classes:
            unique_classes.remove(ignore_index)

        class_dist = np.zeros(n_classes)
        for class_number in unique_classes:
            x, y = np.where(target_mask == class_number)
            class_dist[class_number] = x.shape[0]

        # Normalize the class distribution to a probability.
        class_dist /= class_dist.sum()

        return class_dist

    def visualize_example(self, index, save_path=None, num_plot_images=5, overlay_last_anno=False):
        import matplotlib.pyplot as plt

        # Get an example.
        example = self.__getitem__(index)

        if self.task_mode is None:
            # Assumes that there are no labels in the example.
            F = int(example["active_frames"].sum())

            num_plot_images += 1

            if F >= num_plot_images:
                # Fewer images in sequence than number of plot images.
                fig, axes = plt.subplots(1, num_plot_images, figsize=(30, 5))
            else:
                # More images in sequence than number of plot images.
                fig, axes = plt.subplots(1, F, figsize=(30, 5))

            if F >= num_plot_images:
                # More images in sequence than number of plot images.
                indices = [(F // num_plot_images) * i for i in range(num_plot_images - 1)]
                indices.append(F - 1)

                for i, index in enumerate(indices):
                    rgb_image = self.to_rgb(example["video"][index])

                    axes[i].imshow(rgb_image)
                    axes[i].set_title(f'{example["datetimes"][i].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i].axis("off")
            else:
                # Fewer images in sequence than number of plot images.
                for i in range(num_plot_images):
                    rgb_image = self.to_rgb(example["video"][i])  # TODO:

                    axes[i].imshow(rgb_image)
                    axes[i].set_title(f'{example["datetimes"][i].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i].axis("off")

            if "region_name" in list(example.keys()):
                fig.suptitle(example["region_name"])

            if save_path is not None:
                if overlay_last_anno:
                    save_path = os.path.splitext(save_path)[0] + "_overlay" + os.path.splitext(save_path)[1]
                print(f"Saving figure to path: {save_path}")
                plt.savefig(save_path, dpi=300)

        elif self.task_mode == "ss_arrow_of_time":
            # Assumes that there are no labels in the example.
            F = int(example["active_frames"].sum())

            reverse = example[self.task_mode]  # [0 - not reversed, 1 - reversed]
            reverse = reverse == 1

            num_plot_images += 1

            if F >= num_plot_images:
                # Fewer images in sequence than number of plot images.
                fig, axes = plt.subplots(1, num_plot_images, figsize=(30, 5))
            else:
                # More images in sequence than number of plot images.
                fig, axes = plt.subplots(1, F, figsize=(30, 5))

            if F >= num_plot_images:
                # More images in sequence than number of plot images.
                indices = [(F // num_plot_images) * i for i in range(num_plot_images - 1)]
                indices.append(F - 1)

                for i, index in enumerate(indices):
                    rgb_image = self.to_rgb(
                        self.unnormalize(example["video"][index])[0],
                        mean=example["mean"][index],
                        std=example["std"][index],
                    )

                    axes[i].imshow(rgb_image)
                    axes[i].set_title(f'{example["datetimes"][index].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i].axis("off")
            else:
                # Fewer images in sequence than number of plot images.
                for i in range(num_plot_images):
                    rgb_image = self.to_rgb(
                        self.unnormalize(example["video"][i])[0], mean=example["mean"][i], std=example["std"][i]
                    )

                    axes[i].imshow(rgb_image)
                    axes[i].set_title(f'{example["datetimes"][i].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i].axis("off")

            # Add title to entire resquence.
            if "region_name" in list(example.keys()):
                fig.suptitle(f'{example["region_name"]} | {self.task_mode} | Reverse: {reverse}')
            else:
                fig.suptitle(f"{self.task_mode} | Reverse: {reverse}")

            if save_path is not None:
                if overlay_last_anno:
                    save_path = os.path.splitext(save_path)[0] + "_overlay" + os.path.splitext(save_path)[1]
                print(f"Saving figure to path: {save_path}")
                plt.savefig(save_path, dpi=300)

        elif self.task_mode == "total_bin_change":
            # Plot a subset of images in the sequence and change mask
            F = int(example["active_frames"].sum())

            if F >= num_plot_images:
                # Fewer images in sequence than number of plot images.
                fig, axes = plt.subplots(1, num_plot_images + 1, figsize=(30, 5))
            else:
                # More images in sequence than number of plot images.
                fig, axes = plt.subplots(1, F + 1, figsize=(30, 5))

            # Plot change mask
            label_image = np.clip(example[self.task_mode], 0, 100)
            axes[0].imshow(label_image.float(), cmap="gray")
            axes[0].set_title("Sequence Binary Change Mask")
            axes[0].axis("off")

            if F >= num_plot_images:
                # More images in sequence than number of plot images.
                indices = [(F // num_plot_images) * i for i in range(num_plot_images - 1)]
                indices.append(F - 1)

                for i, index in enumerate(indices):
                    rgb_image = self.to_rgb(
                        self.unnormalize(example["video"][index], mean=example["mean"], std=example["std"])
                    )

                    if overlay_last_anno and i == (len(indices) - 1):
                        # Format label image
                        mpl_label_image = np.ma.masked_where(label_image.numpy() == 0, label_image.numpy())
                        axes[i + 1].imshow(rgb_image, interpolation="none")
                        axes[i + 1].imshow(mpl_label_image, interpolation="none", alpha=0.4, cmap="cool")
                    else:
                        axes[i + 1].imshow(rgb_image)
                    axes[i + 1].set_title(f'{example["datetimes"][index].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i + 1].axis("off")
            else:
                # Fewer images in sequence than number of plot images.
                for i in range(num_plot_images):
                    rgb_image = self.to_rgb(
                        self.unnormalize(example["video"][i], mean=example["mean"], std=example["std"])
                    )
                    if overlay_last_anno and i == (num_plot_images - 1):
                        # Format label image
                        mpl_label_image = np.ma.masked_where(label_image.numpy() == 0, label_image.numpy())
                        axes[i + 1].imshow(rgb_image, interpolation="none")
                        axes[i + 1].imshow(mpl_label_image, interpolation="none", alpha=0.4, cmap="cool")
                    else:
                        axes[i + 1].imshow(rgb_image)

                    axes[i + 1].set_title(f'{example["datetimes"][i].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i + 1].axis("off")

            if "region_name" in list(example.keys()):
                fig.suptitle(example["region_name"])

            if save_path is not None:
                if overlay_last_anno:
                    save_path = os.path.splitext(save_path)[0] + "_overlay" + os.path.splitext(save_path)[1]
                print(f"Saving figure to path: {save_path}")
                plt.savefig(save_path, dpi=300)

        elif self.task_mode == "sem_seg":
            assert not overlay_last_anno, "Overlaying annotation for sem_seg is not implmented yet."
            # Plot a subset of images in the sequence and change mask
            F = int(example["active_frames"].sum())

            if F >= num_plot_images:
                # Fewer images in sequence than number of plot images.
                fig, axes = plt.subplots(1, num_plot_images + 1, figsize=(30, 5))
            else:
                # More images in sequence than number of plot images.
                fig, axes = plt.subplots(1, F + 1, figsize=(30, 5))

            # Plot change mask
            label_image = example[self.task_mode]
            color_label_mask = self.colorize_target_mask(label_image)
            axes[0].imshow(color_label_mask)
            axes[0].set_title("Sequence Semantic Change Mask")
            axes[0].axis("off")

            if F >= num_plot_images:
                # More images in sequence than number of plot images.
                indices = [(F // num_plot_images) * i for i in range(num_plot_images - 1)]
                indices.append(F - 1)

                for i, index in enumerate(indices):
                    rgb_image = self.to_rgb(example["video"][index])

                    axes[i + 1].imshow(rgb_image)
                    axes[i + 1].set_title(f'{example["datetimes"][index].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i + 1].axis("off")
            else:
                # Fewer images in sequence than number of plot images.
                for i in range(num_plot_images):
                    rgb_image = self.to_rgb(example["video"][i])

                    axes[i + 1].imshow(rgb_image)
                    axes[i + 1].set_title(f'{example["datetimes"][i].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                    axes[i + 1].axis("off")

            if "region_name" in list(example.keys()):
                fig.suptitle(example["region_name"])

            if save_path is not None:
                print(f"Saving figure to path: {save_path}")
                plt.savefig(save_path, dpi=300)

        elif self.task_mode == "ss_splice_change_index":
            # Get change index.
            change_index = example[self.task_mode].item()

            # Plot a subset of images in the sequence and change mask
            F = int(example["active_frames"].sum())

            if F <= num_plot_images:
                num_plot_images = F
                indices = list(range(num_plot_images))
            else:
                ## Take a subset of frames but include the change frame.
                indices = [(F // num_plot_images) * i for i in range(num_plot_images - 2)]
                indices.append(F - 1)
                indices.append(change_index)
                indices = sorted(indices)

            fig, axes = plt.subplots(1, num_plot_images, figsize=(30, 5))

            for i, index in enumerate(indices):
                rgb_image = self.to_rgb(
                    self.unnormalize(example["video"][index], mean=example["mean"], std=example["std"])
                )

                axes[i].imshow(rgb_image)
                if i == change_index:
                    # Format label image
                    axes[i].set_title(
                        f'{example["datetimes"][index].strftime("%m/%d/%Y")} | [{index}/{F-1}]', color="red"
                    )
                else:
                    axes[i].set_title(f'{example["datetimes"][index].strftime("%m/%d/%Y")} | [{index}/{F-1}]')
                axes[i].axis("off")

            if "region_name" in list(example.keys()):
                fig.suptitle(example["region_name"])

            if save_path is not None:
                print(f"Saving figure to path: {save_path}")
                plt.savefig(save_path, dpi=300)
        elif self.task_mode == "segmentation":
            # A row of RGB images
            # A row of segmentation images
            raise NotImplementedError(self.task_mode)
        else:
            raise NotImplementedError(self.task_mode)
        plt.close()

    def generate_visualization_GIF(self, index, save_path, overlay_annos=False, overlay_dates=True):
        import cv2
        from PIL import Image
        import matplotlib.pyplot as plt
        from transformer.utils.util_misc import create_gif

        save_dir = os.path.split(save_path)[0]
        save_name = os.path.splitext(os.path.split(save_path)[1])[0]
        save_ext = os.path.splitext(save_path)[1]

        if (self.task_mode is None) or (self.task_mode == "ss_mat_recon"):
            example = self.__getitem__(index)

            # Get all activate frames in video
            F = int(example["active_frames"].sum())
            rgb_frames = []
            for i in range(F):
                # Load frame and convert to RGB.
                rgb_frame = self.to_rgb(example["video"][i])

                rgb_frames.append(rgb_frame)

            # Add date and frame number to images.
            if overlay_dates:
                img_text = [
                    f"[{i+1}/{len(example['datetimes'])}]: " + dt.strftime("%m-%d-%Y")
                    for i, dt in enumerate(example["datetimes"])
                ]
            else:
                img_text = None

            # Create a GIF of RGB frames.
            create_gif(rgb_frames, save_path, fps=1, image_text=img_text)

        elif self.task_mode == "total_bin_change":
            example = self.__getitem__(index)

            # Get all activate frames in video
            F = int(example["active_frames"].sum())
            rgb_frames = []
            for i in range(F):
                # Load frame and convert to RGB.
                rgb_frame = self.to_rgb(self.unnormalize(example["video"][i], mean=example["mean"], std=example["std"]))

                rgb_frames.append(rgb_frame)

            # Add date and frame number to images.
            if overlay_dates:
                img_text = [
                    f"[{i+1}/{len(example['datetimes'])}]: " + dt.strftime("%m-%d-%Y")
                    for i, dt in enumerate(example["datetimes"])
                ]
            else:
                img_text = None

            # Save ground truth image separately.
            target = (np.clip(example[self.task_mode].numpy(), 0, 1) * 255).astype("uint8")
            target_path = os.path.join(save_dir, save_name + "_gt.png")
            Image.fromarray(target).save(target_path)

            if overlay_annos:
                # Convert the annotation to edges.
                contours, _ = cv2.findContours(target, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

                # Create plot of contours.
                fig, ax = plt.subplots(1, 1)
                canvas = np.zeros(target.shape)
                plt.imshow(canvas)
                for poly in contours:
                    # Close the loop of the contour.
                    poly = np.concatenate((poly, poly[0][np.newaxis]), axis=0)

                    # Get x and y values.
                    xs, ys = poly[..., 0][:, 0], poly[..., 1][:, 0]

                    # Draw polygon on canvas.
                    plt.plot(xs, ys, "r-", linewidth=0.1)
                contour_save_path = os.path.join(save_dir, save_name + "_contour.png")
                plt.savefig(contour_save_path, dpi=300)
                plt.close()

                overlayed_rgb_frames = []
                for rgb_frame in rgb_frames:
                    H, W, _ = rgb_frame.shape

                    # Try using matplotlib to overlay polygons on RGB image.

                    # Make content fill the entire figure.
                    plt.figure(frameon=False)
                    plt.imshow(rgb_frame)
                    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # ax.imshow(rgb_frame, aspect='auto')
                    ax.imshow(rgb_frame)

                    for poly in contours:
                        # Close the loop of the contour.
                        poly = np.concatenate((poly, poly[0][np.newaxis]), axis=0)

                        # Get x and y values.
                        xs, ys = poly[..., 0][:, 0], poly[..., 1][:, 0]

                        # Draw polygon on canvas.
                        ax.plot(xs, ys, "r-", linewidth=0.5)

                    # Convert figure to rgb numpy array.
                    fig.canvas.draw()
                    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    overlayed_rgb_frames.append(image)
                    plt.close()

                # Create a GIF.
                save_path = os.path.join(save_dir, save_name + "_overlay" + save_ext)
                create_gif(overlayed_rgb_frames, save_path, fps=1, image_text=img_text)
            else:
                # Create a GIF of RGB frames.
                create_gif(rgb_frames, save_path, fps=1, image_text=img_text)

        else:
            raise NotImplementedError(f"GIF generation for mode {self.task_mode} not implemented.")

    def get_subset_frame_ids(self, n_video_frames, n_req_frames):
        assert n_video_frames > 1

        if n_video_frames == 2:
            return [0, 1]
        elif n_video_frames == n_req_frames:
            return len(range(n_video_frames))
        else:
            # Divide in half N-2 times and find the middle index.
            N = n_video_frames
            M = n_req_frames

            middle_index = N // 2
            frame_ids = [middle_index]
            while len(frame_ids) != (M - 2):
                if len(frame_ids) % 2 == 1:
                    # Get the right middle half.
                    right_half_index = middle_index + frame_ids[-1] // 2
                    frame_ids.append(right_half_index)
                else:
                    # Get the left middle half.
                    # Get the right middle half.
                    left_half_index = middle_index - frame_ids[-2] // 2
                    frame_ids.append(left_half_index)
        frame_ids.insert(0, 0)
        frame_ids.append(n_video_frames - 1)
        frame_ids = sorted(frame_ids)
        return frame_ids

    def visualize_dates(self, index, save_path=None):
        import matplotlib.pyplot as plt
        from collections import defaultdict

        # Get an example.
        example = self.__getitem__(index)

        if "datetimes" not in example.keys():
            raise KeyError("Dataset does not contain datetimes in example.")

        # Get unique years in datetimes.
        dates = example["datetimes"]
        years = np.asarray([date.year for date in dates])
        years = list(np.unique(years))

        # Spilt datetimes.
        split_dict = defaultdict(list)
        for date in dates:
            yday = date.timetuple().tm_yday
            split_dict[date.year].append(yday)

        # Plot datetimes.
        _, axes = plt.subplots(len(years), 1)

        for i, (year, ydays) in enumerate(split_dict.items()):
            canvas = np.zeros([5, 365], dtype=float)
            for yday in ydays:
                canvas[:, yday] = 1.0
            axes[i].imshow(canvas, cmap="Blues")
            axes[i].set_ylabel(year)
            axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=True)
        plt.tight_layout()

        if save_path is not None:
            print(f"Saving figure to path: {save_path}")
            plt.savefig(save_path, dpi=300)

    def scale_video(self, video, scale_factor, inter_mode=cv2.INTER_NEAREST):
        """Up or down scale all frames of video.

        Args:
            video (numpy array): A float numpy array of shape [n_frames, n_channels, height, width].
            scale_factor (int/float): The factor to scale the video and target by. NOTE: This variable be between (0, inf).
            inter_mode (int, optional): An OpenCV resize parameter to determine the resizing method. Defaults to cv2.INTER_NEAREST.

        Returns:
            numpy array: A resized video.
        """
        # Rearrange axes from [n_frames, n_channels, height, width] to [n_frames, height, width, n_channels].
        video = video.transpose(0, 2, 3, 1)

        # Resize video.
        rz_video = []
        for frame_index in range(video.shape[0]):
            rz_frame = self.scale_frame(video[frame_index], scale_factor, inter_mode=inter_mode)
            rz_video.append(rz_frame)
        video = np.stack(rz_video, axis=0)

        # Rearrange axes from [n_frames, height, width, n_channels] to [n_frames, n_channels, height, width].
        video = video.transpose(0, 3, 1, 2)

        return video

    def scale_frame(self, frame, scale_factor, inter_mode=cv2.INTER_NEAREST):
        """Up or down scale a frame.

        Args:
            frame (numpy array): A numpy array of shape [n_channels, height, width].
            scale_factor (int/float): The factor to scale the video and target by. NOTE: This variable be between (0, inf).
            inter_mode (int, optional): An OpenCV resize parameter to determine the resizing method. Defaults to cv2.INTER_NEAREST.

        Returns:
            numpy array: A resized frame.
        """
        rz_frame = cv2.resize(
            frame,
            (int(self.video_slice.height * scale_factor), int(self.video_slice.width * scale_factor)),
            interpolation=inter_mode,
        )
        return rz_frame

    def get_pixel_normalization_params(self):
        raise NotImplementedError

    def normalize(self, video):
        """Normalize videos before they are transformed.

        Args:
            video (torch.tensor): Shape [n_frames, channels, height, width]

        Returns:
            torch.tensor: Shape [n_frames, channels, height, width]
        """
        if self.normalize_mode is None:
            mean, std = [0], [1]
        elif self.normalize_mode == "global":
            mean, std = self.mean_vals, self.std_vals
            video = (video - self.mean_vals) / self.std_vals
        elif self.normalize_mode == "local":
            # Compute the video mean and std.
            mean = np.mean(video, axis=(0, 2, 3))
            std = np.std(video, axis=(0, 2, 3))

            if sum(mean) == 0:
                mean = np.zeros(mean.shape[0])
            mean = np.nan_to_num(mean)

            if sum(std) == 0:
                std = np.ones(std.shape[0])

            indices = np.where(std == 0)
            std[indices] = 1

            # Format mean and std shapes.
            mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
            std = std[np.newaxis, :, np.newaxis, np.newaxis]

            self.local_mean, self.local_std = mean, std
            video = (video - self.local_mean) / self.local_std

        elif self.normalize_mode == "local_trim":
            # Compute the trimmed video mean and std over channels.
            # Trim the top and bottom 15%.
            trimmed_pct = 0.15
            channel_means, channel_stds = [], []
            for channel_index in range(video.shape[1]):
                # Compute trimmed mean.
                try:
                    channel_mean = stats.trim_mean(video[:, channel_index], trimmed_pct, axis=None)
                except Exception:
                    print("Upset trim mean")
                    channel_mean = 0
                channel_means.append(channel_mean)

                # Compute trimmed mean.
                try:
                    channel_std = stats.mstats.trimmed_std(video[:, channel_index], trimmed_pct, axis=None)
                except Exception:
                    print("Upset trim STD")
                    channel_std = 1
                channel_stds.append(channel_std)

            mean = np.asarray(channel_means)
            std = np.asarray(channel_stds)

            if sum(mean) == 0:
                mean = np.zeros(mean.shape[0])
            mean = np.nan_to_num(mean)

            if sum(std) == 0:
                std = np.ones(std.shape[0])

            indices = np.where(std == 0)
            std[indices] = 1

            # Format mean and std shapes.
            mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
            std = std[np.newaxis, :, np.newaxis, np.newaxis]

            self.local_trim_mean, self.local_trim_std = mean, std

            video = (video - self.local_trim_mean) / self.local_trim_std

        else:
            raise NotImplementedError(f"Normalize mode: {self.normalize_mode}")

        mean, std = torch.tensor(mean), torch.tensor(std)
        return video, mean, std

    def unnormalize(self, video, mean=None, std=None):
        """Undo the normalization process.

        Args:
            video (torch.tensor): Shape [n_frames, channels, height, width]

        Returns:
            torch.tensor: Shape [n_frames, channels, height, width]
        """
        if mean is not None:
            video = (video * std) + mean
            return video

        if self.normalize_mode is None:
            video = video[None]
        elif self.normalize_mode == "global":
            video = (video * self.std_vals) + self.mean_vals
        elif self.normalize_mode == "local":
            video = (video * self.local_std) + self.local_mean
        elif self.normalize_mode == "local_trim":
            video = (video * self.local_trim_std) + self.local_trim_mean
        else:
            raise NotImplementedError(f"Normalize mode: {self.normalize_mode}")

        return video

    def pad_video_height_width(self, video, target_height, target_width):
        """Pad a video to the bottom and right of the frames if not same resolution as target height and width.

        Args:
            video (np.array): A numpy array of shape [n_frames, n_channels, og_height, og_width].
            target_height (int): The target height resolution parameter.
            target_width (int): The target width resolution parameter.

        Returns:
            [np.array]: A numpy array of shape [n_frames, n_channels, target_height, target_width].
        """

        n_frames, n_channels, og_height, og_width = video.shape

        if (og_height < target_height) or (og_width < target_width):
            canvas = np.ones((n_frames, n_channels, target_height, target_width), dtype=video.dtype) * self.pad_value
            canvas[:, :, :og_height, :og_width] = video
            video = canvas
        elif (og_height > target_height) or (og_width > target_width):
            raise ValueError(
                f"Input video of resolution [{og_height},{og_width}] is larger than target shape [{target_height},{target_width}]."
            )

        return video
