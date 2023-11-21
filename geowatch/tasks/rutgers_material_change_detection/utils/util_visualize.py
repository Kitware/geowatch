import os
import shutil

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import create_gif


class VideoPlotMaker:
    def __init__(self, save_dir, task_mode, rgb_func, temp_dir="./temp/"):
        self.save_dir = save_dir
        self.rgb_func = rgb_func

        # Get plot function.
        plot_methods = {"total_bin_change": self.plot_total_bin_change}
        self.plot_function = plot_methods[task_mode]

        # Create temporary directory.
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        self.temp_dir = temp_dir

    def __call__(self, video, target, prediction, n_frames, datetimes, save_name):
        # Create and save individual plots.
        save_paths = self.plot_function(video, target, prediction, n_frames, datetimes)

        # Generate GIF of plots.
        ## Load images with PIL
        np_images = [np.asarray(Image.open(p)) for p in save_paths]
        save_path = os.path.join(self.save_dir, save_name + ".gif")

        ## Create GIF
        create_gif(np_images, save_path)

    def plot_total_bin_change(self, video, target, prediction, n_frames, datetimes):

        # Get ground truth image.
        ground_truth = np.clip(target, 0, 1)

        # Get colored prediction.
        model_class_pred = np.argmax(prediction, axis=0)
        pred_canvas = np.zeros((model_class_pred.shape[0], model_class_pred.shape[1], 3), dtype="uint8")

        # Get True Positives.
        x, y = np.where((model_class_pred == 1) & (target == 1))
        true_positive_color = [255, 255, 255]
        pred_canvas[x, y, :] = np.repeat(np.array(true_positive_color)[np.newaxis], repeats=x.shape[0], axis=0)

        # Get False Positives.
        x, y = np.where((model_class_pred == 1) & (target == 0))
        false_positive_color = [0, 255, 255]
        pred_canvas[x, y, :] = np.repeat(np.array(false_positive_color)[np.newaxis], repeats=x.shape[0], axis=0)

        # Get False Negatives.
        x, y = np.where((model_class_pred == 0) & (target == 1))
        false_negative_color = [255, 0, 0]
        pred_canvas[x, y, :] = np.repeat(np.array(false_negative_color)[np.newaxis], repeats=x.shape[0], axis=0)

        # Compute confidence mask from model prediction.
        conf_mask = torch.softmax(torch.from_numpy(prediction), dim=0)[1]

        save_paths = []
        for frame_index in range(n_frames + 1):
            # Get RGB frame.
            if frame_index == n_frames:
                # Add blank frame.
                rgb_frame = np.zeros([target.shape[0], target.shape[1], 3])
            else:
                rgb_frame = self.rgb_func(video[frame_index])

                # Get datetime.
                datetime_str = datetimes[frame_index].strftime("%m/%d/%Y")

            _, axes = plt.subplots(2, 2, figsize=(5, 5), dpi=300)  # RGB, Conf, Pred, GT
            axes[0][0].imshow(rgb_frame)
            if frame_index == n_frames:
                axes[0][0].set_title("")
            else:
                axes[0][0].set_title(f"[{frame_index+1}/{n_frames}] - " + datetime_str)
            axes[0][0].axis("off")

            axes[0][1].imshow(conf_mask, "viridis", vmin=0, vmax=1)
            axes[0][1].set_title("Confidence")
            axes[0][1].axis("off")

            axes[1][0].imshow(pred_canvas, cmap=None)
            axes[1][0].set_title("Prediction")
            axes[1][0].axis("off")
            axes[1][1].imshow(ground_truth, cmap="gray")
            axes[1][1].set_title("Ground Truth")
            axes[1][1].axis("off")
            plt.tight_layout()

            # Save figure.
            save_path = os.path.join(self.temp_dir, str(frame_index).zfill(5) + ".png")
            plt.savefig(save_path, dpi=300, pad_inches=0.0)
            save_paths.append(save_path)
            plt.close()
        return save_paths


def get_instance_visualization(
    task_mode,
    model_pred,
    video,
    target,
    rgb_func,
    last_frame_index=-1,
    unnorm_func=None,
    save_path=None,
    title_str=None,
):
    """Generate a plot to visualize an instance of the model prediction, input, and ground truth.

    Args:
        task_mode (str): [description]
        model_pred (?): The models prediction will depend on the task_mode.
        video (np.array): A float numpy array of shape [n_frames, n_channels, height, width].
        target (?): Depending on the task_mode parameter, this can be a np.array of shape [n_classes, height, width], integer, etc.
        rgb_func (?): A function to translate a multi-modal image into a uint8 numpy array ready for plotting.
        last_frame_index (int, optional): An integer representing the last frame of video that contains real data instead of padded data.
        unnorm_func (?, optional): A function to unnormalize a normalized video. Defaults to None.
        save_path (str, optional): A path to save the figure to. Defaults to None.

    Raises:
        NotImplementedError: [description]

    Returns:
        plt.Figure: A subplot figure.
    """

    # Undo the normalization done to the image.
    if unnorm_func is not None:
        video = unnorm_func(video)

    # Get and format ground truth and prediction representations.
    if task_mode == "total_bin_change":
        # Format input images to RGB form.
        first_frame_rgb = rgb_func(video[0])
        last_frame_rgb = rgb_func(video[last_frame_index])

        # Format ground truth image.
        target = np.clip(target, 0, 1).astype("float32")

        # Format prediction and confidence mask.
        model_class_pred = np.argmax(model_pred, axis=0)
        pred_canvas = np.zeros((model_class_pred.shape[0], model_class_pred.shape[1], 3), dtype="uint8")

        # Get True Positives.
        x, y = np.where((model_class_pred == 1) & (target == 1))
        true_positive_color = [255, 255, 255]
        pred_canvas[x, y, :] = np.repeat(np.array(true_positive_color)[np.newaxis], repeats=x.shape[0], axis=0)

        # Get False Positives.
        x, y = np.where((model_class_pred == 1) & (target == 0))
        false_positive_color = [0, 255, 255]
        pred_canvas[x, y, :] = np.repeat(np.array(false_positive_color)[np.newaxis], repeats=x.shape[0], axis=0)

        # Get False Negatives.
        x, y = np.where((model_class_pred == 0) & (target == 1))
        false_negative_color = [255, 0, 0]
        pred_canvas[x, y, :] = np.repeat(np.array(false_negative_color)[np.newaxis], repeats=x.shape[0], axis=0)

        # Compute confidence mask from model prediction.
        conf_mask = torch.softmax(torch.from_numpy(model_pred))[1]

        # Create plot to store visualization of inputs and prediction.
        fig, axes = plt.subplots(1, 5)
        axes[0].imshow(first_frame_rgb)  # First frame of video (RGB).
        axes[0].set_title("First Frame")
        axes[0].axis("off")
        axes[1].imshow(last_frame_rgb)  # Last frame of video (RGB).
        axes[1].set_title("Last Frame")
        axes[1].axis("off")
        axes[2].imshow(pred_canvas, cmap=None)  # Prediction.
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        axes[3].imshow(conf_mask, "viridis", vmin=0, vmax=1)  # Confidence of change.
        axes[3].set_title("Change Conf")
        axes[3].axis("off")
        axes[4].imshow(target, "gray")  # Ground truth.
        axes[4].set_title("Ground Truth")
        axes[4].axis("off")
        plt.tight_layout()
    elif task_mode == "ss_mat_recon":
        breakpoint()
        pass
        # # Show the first and last frame inputs as well as the predicted reconstruction.
        # # Format input images to RGB form.
        # n_frames = int((data['active_frames'][batch_index].sum() - 1).item())

        # # Create plot to store visualization of inputs and prediction.
        # if self.cfg.loss.discretize:
        #     fig, axes = plt.subplots(3, n_frames)
        #     in_frames = dataset.unnormalize(data['video'][batch_index].detach().cpu())
        #     pred_frames = dataset.unnormalize(model_output[task_mode][batch_index].detach().cpu())

        #     _, cluster_ids = torch.max(model_output['mat_cluster_ids']['layer1'][:, batch_index], dim=-1)
        #     cluster_ids = cluster_ids.detach().cpu().numpy()

        #     color_code = self.model.color_codes['layer1']

        #     material_cluster_ids = np.take(color_code, cluster_ids, axis=0)

        #     for i in range(n_frames):
        #         axes[0][i].imshow(dataset.to_rgb(in_frames[i]))  # RGB input.
        #         axes[0][i].axis('off')

        #         axes[1][i].imshow(dataset.to_rgb(pred_frames[i]))  # RGB prediction.
        #         axes[1][i].axis('off')

        #         axes[2][i].imshow(material_cluster_ids[i])  # RGB prediction.
        #         axes[2][i].axis('off')

        # else:
        #     fig, axes = plt.subplots(2, n_frames)
        #     in_frames = dataset.unnormalize(data['video'][batch_index].detach().cpu())
        #     pred_frames = dataset.unnormalize(model_output[task_mode][batch_index].detach().cpu())

        #     for i in range(n_frames):
        #         axes[0][i].imshow(dataset.to_rgb(in_frames[i]))  # RGB input.
        #         axes[0][i].axis('off')

        #         axes[1][i].imshow(dataset.to_rgb(pred_frames[i]))  # RGB prediction.
        #         axes[1][i].axis('off')

        # plt.tight_layout()
    elif task_mode in ["ss_arrow_of_time", "ss_splice_change"]:
        # TODO: Implement visualization for this method.
        fig, axes = plt.subplots(1, 5)
    else:
        raise NotImplementedError(f'Target visualization for mode "{task_mode}" not implemented.')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0.0)
    else:
        return fig, axes
