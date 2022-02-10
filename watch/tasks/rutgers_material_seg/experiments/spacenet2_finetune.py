# flake8: noqa

import comet_ml
import time
from watch.tasks.rutgers_material_seg.models.canny_edge import CannyFilter
from skimage.filters import threshold_otsu as otsu
from fast_pytorch_kmeans import KMeans
from watch.tasks.rutgers_material_seg.models.losses import SupConLoss, simCLR_loss, QuadrupletLoss, SoftCE
from watch.tasks.rutgers_material_seg.models.supcon import SupConResNet
from watch.tasks.rutgers_material_seg.datasets import build_dataset
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.models import build_model
import watch.tasks.rutgers_material_seg.utils.visualization as visualization
import watch.tasks.rutgers_material_seg.utils.eval_utils as eval_utils
import watch.tasks.rutgers_material_seg.utils.utils as utils
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning import losses
import torchvision.transforms.functional as FT
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from scipy import ndimage
import torch.nn.functional as F
import torch.optim as optim
import ubelt as ub
import numpy as np
import matplotlib.pyplot as plt
import ndsampler
import kwimage
import kwcoco
import random
import math
import yaml
import warnings
import datetime
import torch
import cv2
import gc
import matplotlib
matplotlib.use('Agg')
from watch.tasks.rutgers_material_seg.models.tex_refine import TeRN

import sys
import os
debug_mode = True
if debug_mode:
    from pympler import muppy, summary
current_path = os.getcwd().split("/")

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Trainer(object):
    def __init__(self, model: object, train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader, epochs: int,
                 optimizer: object, scheduler: object,
                 test_loader: torch.utils.data.DataLoader = None,
                 test_with_full_supervision: int = 0) -> None:
        """trainer class

        Args:
            model (object): trained or untrained model
            train_loader (torch.utils.data.DataLader): loader with training data
            val_loader (torch.utils.data.DataLader): loader with validation data
            epochs (int): number of epochs
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
            test_loader (torch.utils.data.DataLader, optional): loader with testing data. Defaults to None.
            test_with_full_supervision (int, optional): should full supervision be used. Defaults to 0.
        """

        self.model = model
        self.use_crf = config['evaluation']['use_crf']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_weights = torch.Tensor(config['data']['weights']).float().to(device)
        self.contrastive_loss = SupConLoss()
        self.quadrupletloss = QuadrupletLoss()
        self.triplet_margit_loss_snr = losses.TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all", distance=SNRDistance())
        self._aff = TeRN(num_iter=10, dilations=[1, 1, 2, 2, 6, 12, 12]).to(device)
        self.k = config['data']['num_classes']
        self.kmeans = KMeans(n_clusters=self.k, mode='euclidean', verbose=0, minibatch=None)
        self.max_label = 5
        self.all_crops_params = [tuple([i, j, config['data']['window_size'], config['data']['window_size']]) for i in range(config['data']['window_size'], config['data']
                                                                                                                            ['image_size']-config['data']['window_size']) for j in range(config['data']['window_size'], config['data']['image_size']-config['data']['window_size'])]
        self.inference_all_crops_params = [tuple([i, j, config['evaluation']['inference_window'], config['evaluation']['inference_window']]) for i in range(0, config['data']['image_size']) for j in range(0, config['data']['image_size'])]
        self.all_crops_params_np = np.array(self.all_crops_params)
        self.change_threshold = 0.2
        # print(self.all_crops_params_np)
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision

        self.train_second_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.crop_size = (config['data']['window_size'],
                          config['data']['window_size'])

        self.cmap = visualization.n_distinguishable_colors(nlabels=5,
                                                           first_color_black=True, last_color_black=False,
                                                           bg_alpha=config['visualization']['bg_alpha'],
                                                           fg_alpha=config['visualization']['fg_alpha'])

    def high_confidence_filter(self, features: torch.Tensor, cutoff_top: float = 0.75,
                               cutoff_low: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
        """Select high confidence regions to select as predictions

        Args:
            features (torch.Tensor): initial mask
            cutoff_top (float, optional): cutoff of the object. Defaults to 0.75.
            cutoff_low (float, optional): low cutoff. Defaults to 0.2.
            eps (float, optional): small number. Defaults to 1e-8.

        Returns:
            torch.Tensor: pseudo mask generated
        """
        bs, c, h, w = features.size()
        features = features.view(bs, c, -1)

        # for each class extract the max confidence
        features_max, _ = features.max(-1, keepdim=True)
        # features_max[:, c-1:] *= 0.8
        # features_max[:, :c-1] *= cutoff_top
        features_max *= cutoff_top
        # features_max *= cutoff_top

        # if the top score is too low, ignore it
        lowest = torch.Tensor([cutoff_low]).type_as(features_max)
        features_max = features_max.max(lowest)

        filtered_features = (features > features_max).type_as(features)
        filtered_features = filtered_features.view(
            bs, c, h, w)*features.view(bs, c, h, w)
        return filtered_features
    
    def run_pamr(self, im: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """pixel-adaptive convolution network run

        Args:
            im (torch.Tensor): image
            mask (torch.Tensor): coarse mask

        Returns:
            torch.Tensor: refined mask
        """
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, mask)
        return masks_dec
    def train(self, epoch: int, cometml_experiemnt: object) -> float:
        """training single epoch

        Args:
            epoch (int): number of epoch
            cometml_experiemnt (object): comet ml experiment to log the epoch

        Returns:
            float: training loss of that epoch
        """
        total_loss = 0
        total_loss_seg = 0
        preds, targets = [], []
        histogram_distance, l1_dist, l2_dist = [], [], []
        # topk_pre_histogram_distance, topk_pre_l1_dist, topk_pre_l2_dist = [], [], []
        # topk_post_histogram_distance, topk_post_l1_dist, topk_post_l2_dist = [], [], []

        self.model.train()
        print(f"starting epoch {epoch}")
        loader_size = len(self.train_loader)
        if config['visualization']['train_visualization_divisor'] >= loader_size:
            config['visualization']['train_visualization_divisor'] = loader_size
        iter_visualization = loader_size // config['visualization']['train_visualization_divisor']
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        batch_index_to_show = config['visualization']['batch_index_to_show']
        for batch_index, batch in pbar:
            random_crop = transforms.RandomCrop(self.crop_size)
            outputs = batch

            image1, mask = outputs['image'], batch['mask']
            # image_name = f"{str(outputs['tr'].data[0][batch_index_to_show]['gids'])}_{str(outputs['tr'].data[0][batch_index_to_show]['slices'])}"

            # original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
            # mask = torch.stack(mask)
            mask = mask.long().squeeze(1)

            mask[mask == -1] = 0

            class_to_show = max(0, torch.unique(mask)[-1]-1)
            image1 = image1.to(device)
            mask = mask.to(device)

            # print(f"image1: {image1.shape}")
            # print(f"image1 min: {image1.min()}, max: {image1.max()}")
            # print(f"mask uniques: {torch.unique(mask)}")
            # print(f"image1 min:{image1.min()}, max:{image1.max()}")
            # image1 = utils.stad_image(image1)

            # print(f"image1 min:{image1.min()}, max:{image1.max()}")
            # image1 = F.normalize(image1, dim=1, p=1) 
            # image2 = F.normalize(image2, dim=1, p=1) 

            output = self.model(image1)

            loss = F.cross_entropy(output,
                                    mask,
                                    weight=self.class_weights,
                                    # ignore_index=-100,
                                    reduction="sum")


            start = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_seg += loss.item()

            backprop_time = time.time() - start

            masks = F.softmax(output, dim=1)
            pred1 = masks.max(1)[1].cpu().detach()  # .numpy()

            total_loss += loss.item()
            mask[mask == -1] = 0
            preds.append(pred1)
            targets.append(mask.cpu())  # .numpy())

            if config['visualization']['train_visualizer']:
                if (epoch) % config['visualization']['visualize_training_every'] == 0:
                    if (batch_index % iter_visualization) == 0:
                        figure = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']),
                                            dpi=config['visualization']['dpi'])
                        ax1 = figure.add_subplot(1, 5, 1)
                        ax2 = figure.add_subplot(1, 5, 2)
                        ax3 = figure.add_subplot(1, 5, 3)
                        ax4 = figure.add_subplot(1, 5, 4)
                        ax5 = figure.add_subplot(1, 5, 5)
                        # ax5 = figure.add_subplot(1, 5, 5)
                        # ax6 = figure.add_subplot(3, 4, 6)
                        # ax7 = figure.add_subplot(3, 4, 7)
                        # ax8 = figure.add_subplot(3, 4, 8)
                        # ax9 = figure.add_subplot(3, 4, 9)
                        # ax10 = figure.add_subplot(3, 4, 10)
                        # ax11 = figure.add_subplot(3, 4, 11)
                        # ax12 = figure.add_subplot(3, 4, 12)

                        cmap_gradients = plt.cm.get_cmap('jet')
                        gt_cmap = matplotlib.colors.ListedColormap(["#000000", "#FFFFFF"], name='gt_cmap')
                        # image_show = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:1:4,:3]
                        image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                        image_show1 = np.flip(image_show1, axis=2)

                        image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())

                        gt_mask_show1 = mask.cpu().detach()[batch_index_to_show, :, :].numpy().squeeze()

                        # histograms_intersection_show = (histograms_intersection_show - histograms_intersection_show.min())/(histograms_intersection_show.max() - histograms_intersection_show.min())

                        pred1_show = masks.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]
                        pred1_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*pred1_show)
                        # pred2_show = masks2.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]


                        classes_in_gt = np.unique(gt_mask_show1)
                        ax1.imshow(image_show1)

                        ax2.imshow(image_show1)
                        ax2.imshow(gt_mask_show1, cmap=gt_cmap, vmin=0, vmax=1, alpha=config['visualization']['fg_alpha'])

                        # ax4.imshow(negative_image1_show)

                        masks_show = masks.cpu().detach().numpy()[batch_index_to_show,1,:,:]
                        ax3.imshow(masks_show, cmap=cmap_gradients)

                        ax4.imshow(image_show1)
                        ax4.imshow(pred1_show, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                        ax5.imshow(pred1_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                        # ax12.imshow(vw_dis_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)

                        ax1.axis('off')
                        ax2.axis('off')
                        ax3.axis('off')
                        ax4.axis('off')
                        ax5.axis('off')
                        # ax6.axis('off')
                        # ax7.axis('off')
                        # ax8.axis('off')
                        # ax9.axis('off')

                        if config['visualization']['titles']:
                            ax1.set_title(f"Input Image 1", fontsize=config['visualization']['font_size'])
                            # ax2.set_title(f"Input Image 2", fontsize=config['visualization']['font_size'])
                            ax2.set_title(f"Change GT Mask overlaid, uniques: {np.unique(gt_mask_show1)}", fontsize=config['visualization']['font_size'])
                            ax3.set_title(f"Features of class 1", fontsize=config['visualization']['font_size'])
                            ax4.set_title(f"Prediction. uniques: {np.unique(pred1_show)}", fontsize=config['visualization']['font_size'])
                            # ax6.set_title(f"l2_patched_diff_change_features_show", fontsize=config['visualization']['font_size'])
                            # ax7.set_title(f"histograms_intersection_show", fontsize=config['visualization']['font_size'])
                            # ax8.set_title(f"dictionary2_show", fontsize=config['visualization']['font_size'])
                            # ax9.set_title(f"l1_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                            # ax10.set_title(f"l2_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                            # ax11.set_title(f"histc_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                            # figure.suptitle(
                            #     f"Epoch: {epoch+1}\nGT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])

                        # cometml_experiemnt.log_figure(figure_name=f"Training, image name: {image_name}, epoch: {epoch}, classes in gt: {classes_in_gt}, classifier predictions: {labels_predicted_indices}",figure=figure)
                        cometml_experiemnt.log_figure(figure_name=f"Training, image name", figure=figure)
                        figure.tight_layout()

                        if config['visualization']['train_imshow']:
                            plt.show()

                        figure.clear()
                        figure.clf()
                        plt.cla()
                        plt.clf()
                        plt.close('all')
                        plt.close(figure)
                        gc.collect()

            logging_time = time.time() - start

            # pbar.set_description(
            #     f"(timing, secs) crop_collection: {crop_collection_time:0.3f}, run_network: {run_network_time:0.3f}, clustering: {clustering_time:0.3f}, backprob: {backprop_time:0.3f}, log: {logging_time:0.3f}")

        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=config['data']['num_classes'])
        
        hist_mean_iou, hist_precision, hist_recall = eval_utils.compute_jaccard(histogram_distance, targets, num_classes=config['data']['num_classes'])
        l1_mean_iou, l1_precision, l1_recall = eval_utils.compute_jaccard(l1_dist, targets, num_classes=config['data']['num_classes'])
        l2_mean_iou, l2_precision, l2_recall = eval_utils.compute_jaccard(l2_dist, targets, num_classes=config['data']['num_classes'])

        l1_precision = np.array(l1_precision)
        l1_recall = np.array(l1_recall)
        l1_f1 = 2 * (l1_precision * l1_recall) / (l1_precision + l1_recall)

        l2_precision = np.array(l2_precision)
        l2_recall = np.array(l2_recall)
        l2_f1 = 2 * (l2_precision * l2_recall) / (l2_precision + l2_recall)

        hist_precision = np.array(hist_precision)
        hist_recall = np.array(hist_recall)
        hist_f1 = 2 * (hist_precision * hist_recall) / (hist_precision + hist_recall)

        mean_iou = np.array(mean_iou)
        precision = np.array(precision)
        recall = np.array(recall)
        classwise_f1_score = 2 * (precision * recall) / (precision + recall)

        mean_precision = precision.mean()
        mean_recall = recall.mean()
        overall_miou = mean_iou.mean()
        mean_f1_score = classwise_f1_score.mean()

        # overall_miou = sum(mean_iou)/len(mean_iou)
        # print(f"Training class-wise mIoU value: \n{mean_iou} \noverall mIoU: {overall_miou}")
        # print(f"Training class-wise Precision value: \n{precision} \noverall Precision: {mean_precision}")
        # print(f"Training class-wise Recall value: \n{recall} \noverall Recall: {mean_recall}")
        # print(f"Training overall F1 Score: {mean_f1_score}")

        cometml_experiemnt.log_metric("Training Loss", total_loss, epoch=epoch+1)
        cometml_experiemnt.log_metric("Segmentation Loss", total_loss_seg, epoch=epoch+1)
        cometml_experiemnt.log_metric("Training mIoU", overall_miou, epoch=epoch+1)
        cometml_experiemnt.log_metric("Training mean_f1_score", mean_f1_score, epoch=epoch+1)

        # cometml_experiemnt.log_metrics({f"Training Recall class {str(x)}": recall[x] for x in range(len(recall))}, epoch=epoch+1)
        # cometml_experiemnt.log_metrics({f"Training Precision class {str(x)}": precision[x] for x in range(len(precision))}, epoch=epoch+1)
        # cometml_experiemnt.log_metrics({f"Training F1_score class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"L1 Training Recall class {str(x)}": l1_recall[x] for x in range(len(l1_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L1 Training Precision class {str(x)}": l1_precision[x] for x in range(len(l1_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L1 Training F1_score class {str(x)}": l1_f1[x] for x in range(len(l1_f1))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"L2 Training Recall class {str(x)}": l2_recall[x] for x in range(len(l2_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L2 Training Precision class {str(x)}": l2_precision[x] for x in range(len(l2_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L2 Training F1_score class {str(x)}": l2_f1[x] for x in range(len(l2_f1))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"Concat Training Recall class {str(x)}": recall[x] for x in range(len(recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"Concat Training Precision class {str(x)}": precision[x] for x in range(len(precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"Concat Training F1_score class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch+1)

        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.train_loader.__len__()))

        return total_loss/self.train_loader.__len__()

    def validate(self, epoch: int, cometml_experiemnt: object, save_individual_plots_specific: bool = False) -> tuple:
        """validating single epoch

        Args:
            epoch (int): current epoch
            cometml_experiemnt (object): logging experiment

        Returns:
            tuple: (validation loss, mIoU)
        """
        print("validating")
        total_loss = 0
        preds, stacked_preds, targets = [], [], []
        histogram_distance, l1_dist, l2_dist = [], [], []
        accuracies = 0
        running_ap = 0.0
        batch_index_to_show = config['visualization']['batch_index_to_show']
        if self.test_with_full_supervision == 1:
            loader = self.test_loader
        else:
            loader = self.val_loader
        loader_size = len(loader)
        if config['visualization']['val_visualization_divisor'] > loader_size:
            config['visualization']['val_visualization_divisor'] = loader_size
        iter_visualization = loader_size // config['visualization']['val_visualization_divisor']
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total=len(loader))
            for batch_index, batch in pbar:
                outputs = batch
                image1, mask = outputs['image'], batch['mask']
                
                # original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']

                mask = mask.long().squeeze(1)

                mask[mask == -1] = 0
                bs, c, h, w = image1.shape
                class_to_show = max(0, torch.unique(mask)[-1]-1)
                image1 = image1.to(device)
                mask = mask.to(device)

                # image1 = utils.stad_image(image1)
                # image1 = F.normalize(image1, dim=1, p=1) 


                output = self.model(image1)

                masks = F.softmax(output, dim=1)
                # masks_refined1 = self.run_pamr(image1, masks.detach()) # torch.Size([B, C+1, H, W])
                masks_refined1 = masks
                pred1 = masks.max(1)[1].cpu().detach()  # .numpy()
                
                preds.append(pred1)
                targets.append(mask.cpu())  # .numpy())
                if config['visualization']['val_visualizer'] or (config['visualization']['save_individual_plots'] and save_individual_plots_specific):
                    # bs=1
                    image1 = utils.stad_image(image1)
                    for b in range(bs):
                        batch_index_to_show = b
                        image_name = outputs['image_name'][b]
                        if (epoch) % config['visualization']['visualize_val_every'] == 0:
                            if (batch_index % iter_visualization) == 0:
                                gt_cmap = matplotlib.colors.ListedColormap(["#000000", "#FFFFFF"], name='gt_cmap')
                                figure = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']),
                                                    dpi=config['visualization']['dpi'])
                                ax1 = figure.add_subplot(2, 3, 1)
                                ax2 = figure.add_subplot(2, 3, 2)
                                ax3 = figure.add_subplot(2, 3, 3)
                                ax4 = figure.add_subplot(2, 3, 4)
                                ax5 = figure.add_subplot(2, 3, 5)
                                ax6 = figure.add_subplot(2, 3, 6)
                                # ax6 = figure.add_subplot(3, 3, 6)
                                # ax7 = figure.add_subplot(3, 3, 7)
                                # ax8 = figure.add_subplot(3, 3, 8)
                                # ax9 = figure.add_subplot(3, 3, 9)
                                # ax10 = figure.add_subplot(3, 4, 10)
                                # ax11 = figure.add_subplot(3, 4, 11)
                                # ax12 = figure.add_subplot(3, 4, 12)

                                cmap_gradients = plt.cm.get_cmap('jet')
                                # image_show = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:1:4,:3]
                                image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                                image_show1 = np.flip(image_show1, axis=2)

                                image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())
                                gt_mask_show1 = mask.cpu().detach()[batch_index_to_show, :, :].numpy().squeeze()

                                # histograms_intersection_show = (histograms_intersection_show - histograms_intersection_show.min())/(histograms_intersection_show.max() - histograms_intersection_show.min())

                                pred1_show = masks.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]
                                pred1_refined_show = masks_refined1.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]
                                pred_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*pred1_show)
                                pred_fp_tp_fn_prediction_refined_mask = gt_mask_show1 + (2*pred1_refined_show)


                                classes_in_gt = np.unique(gt_mask_show1)
                                ax1.imshow(image_show1)

                                # ax2.imshow(image_show2)

                                ax2.imshow(image_show1)
                                ax2.imshow(gt_mask_show1, cmap=gt_cmap, vmin=0, vmax=1, alpha=config['visualization']['fg_alpha'])

                                masks_show = masks.cpu().detach().numpy()[batch_index_to_show,1,:,:]
                                ax3.imshow(masks_show, cmap=cmap_gradients)

                                ax4.imshow(image_show1)
                                ax4.imshow(pred1_show, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                                ax5.imshow(pred_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                                ax6.imshow(pred_fp_tp_fn_prediction_refined_mask, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])


                                ax1.axis('off')
                                ax2.axis('off')
                                ax3.axis('off')
                                ax4.axis('off')
                                ax5.axis('off')
                                ax6.axis('off')
                                # ax7.axis('off')
                                # ax8.axis('off')
                                # ax9.axis('off')

                                if config['visualization']['titles']:
                                    ax1.set_title(f"Input Image 1", fontsize=config['visualization']['font_size'])
                                    # ax2.set_title(f"Input Image 2", fontsize=config['visualization']['font_size'])
                                    ax2.set_title(f"Change GT Mask overlaid, uniques: {np.unique(gt_mask_show1)}", fontsize=config['visualization']['font_size'])
                                    ax3.set_title(f"Features of class 1", fontsize=config['visualization']['font_size'])
                                    ax4.set_title(f"Prediction. uniques: {np.unique(pred1_show)}", fontsize=config['visualization']['font_size'])
                                    # ax6.set_title(f"histograms_intersection_show", fontsize=config['visualization']['font_size'])
                                    # ax7.set_title(f"l1_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                    # ax8.set_title(f"l2_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                    # ax9.set_title(f"histc_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                    # figure.suptitle(
                                    #     f"Epoch: {epoch+1}\nGT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])

                                figure.tight_layout()
                                if config['visualization']['val_imshow']:
                                    plt.show()
                                
                                if (config['visualization']['save_individual_plots'] or save_individual_plots_specific):

                                    plots_path_save = f"{config['visualization']['save_individual_plots_path']}"
                                    fig_save_image_root = (f"{plots_path_save}/image_root/", ax1)
                                    fig_save_image2_root = (f"{plots_path_save}/image2/", ax2)
                                    fig_save_gt_root = (f"{plots_path_save}/gt/", ax3)
                                    fig_save_preds_refined1_root = (f"{plots_path_save}/preds_refined1/", ax6)
                                    fig_save_preds_root = (f"{plots_path_save}/preds/", ax5)

                                    roots = [
                                        fig_save_image_root,
                                        fig_save_image2_root,
                                        fig_save_gt_root,
                                        fig_save_preds_refined1_root,
                                        fig_save_preds_root
                                    ]
                                    figure.savefig(
                                        f"{plots_path_save}/figs/{image_name}_{str(b)}.png", bbox_inches='tight')
                                    for root, ax in roots:
                                        utils.create_dir_if_doesnt_exist(root)
                                        file_path = f"{root}/{image_name}_{str(b)}.png"
                                        # extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
                                        extent = ax.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.dpi_scale_trans.inverted())
                                        figure.savefig(file_path, bbox_inches=extent)

                                cometml_experiemnt.log_figure(figure_name=f"Validation, Image name", figure=figure)
                                figure.clear()
                                plt.cla()
                                plt.clf()
                                plt.close('all')
                                plt.close(figure)
                                gc.collect()

        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=config['data']['num_classes'])

        hist_mean_iou, hist_precision, hist_recall = eval_utils.compute_jaccard(histogram_distance, targets, num_classes=config['data']['num_classes'])
        l1_mean_iou, l1_precision, l1_recall = eval_utils.compute_jaccard(l1_dist, targets, num_classes=config['data']['num_classes'])
        l2_mean_iou, l2_precision, l2_recall = eval_utils.compute_jaccard(l2_dist, targets, num_classes=config['data']['num_classes'])

        l1_precision = np.array(l1_precision)
        l1_recall = np.array(l1_recall)
        l1_f1 = 2 * (l1_precision * l1_recall) / (l1_precision + l1_recall)

        l2_precision = np.array(l2_precision)
        l2_recall = np.array(l2_recall)
        l2_f1 = 2 * (l2_precision * l2_recall) / (l2_precision + l2_recall)

        hist_precision = np.array(hist_precision)
        hist_recall = np.array(hist_recall)
        hist_f1 = 2 * (hist_precision * hist_recall) / (hist_precision + hist_recall)

        mean_iou = np.array(mean_iou)
        precision = np.array(precision)
        recall = np.array(recall)
        classwise_f1_score = 2 * (precision * recall) / (precision + recall)

        mean_precision = precision.mean()
        mean_recall = recall.mean()
        overall_miou = mean_iou.mean()
        mean_f1_score = classwise_f1_score.mean()

        # print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU", overall_miou, epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation precision", mean_precision, epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation recall", mean_recall, epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation mean f1_score", mean_f1_score, epoch=epoch+1)
        print({f"Recall class {str(x)}": recall[x] for x in range(len(recall))})
        print({f"Precision class {str(x)}": precision[x] for x in range(len(precision))})
        print({f"F1 class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))})

        # cometml_experiemnt.log_metrics({f"Recall class {str(x)}": recall[x] for x in range(len(recall))}, epoch=epoch+1)
        # cometml_experiemnt.log_metrics({f"Precision class {str(x)}": precision[x] for x in range(len(precision))}, epoch=epoch+1)
        # cometml_experiemnt.log_metrics({f"F1_score class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"L1 Validation Recall class {str(x)}": l1_recall[x] for x in range(len(l1_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L1 Validation Precision class {str(x)}": l1_precision[x] for x in range(len(l1_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L1 Validation F1_score class {str(x)}": l1_f1[x] for x in range(len(l1_f1))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"L2 Validation Recall class {str(x)}": l2_recall[x] for x in range(len(l2_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L2 Validation Precision class {str(x)}": l2_precision[x] for x in range(len(l2_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L2 Validation F1_score class {str(x)}": l2_f1[x] for x in range(len(l2_f1))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"Concat Validation Recall class {str(x)}": recall[x] for x in range(len(recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"Concat Validation Precision class {str(x)}": precision[x] for x in range(len(precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"Concat Validation F1_score class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch+1)

        cometml_experiemnt.log_metric("Validation Average Loss", total_loss/loader.__len__(), epoch=epoch+1)

        return total_loss/loader.__len__(), classwise_f1_score, overall_miou

    def forward(self, cometml_experiment: object, world_size: int = 8) -> tuple:
        """forward pass for all epochs

        Args:
            cometml_experiment (object): comet ml experiment for logging
            world_size (int, optional): for distributed training. Defaults to 8.

        Returns:
            tuple: (train losses, validation losses, mIoU)
        """
        train_losses, val_losses = [], []
        mean_ious_val, mean_ious_val_list, count_metrics_list = [], [], []
        best_val_loss, best_train_loss, train_loss = np.infty, np.infty, np.infty
        best_val_mean_f1, val_mean_f1 = 0, 0
        best_val_miou, val_miou = 0, 0
        best_val_change_f1, val_change_f1 = 0, 0

        model_save_dir = config['data'][config['location']]['model_save_dir'] + \
            f"{current_path[-1]}_{config['dataset']}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        for epoch in range(0, self.epochs):
            if config['procedures']['train']:
                with cometml_experiment.train():
                    train_loss = self.train(epoch, cometml_experiment)
            if config['procedures']['validate']:
                with cometml_experiment.validate():
                    val_loss, val_cw_f1, val_miou = self.validate(
                        epoch, cometml_experiment)
                    val_mean_f1 = val_cw_f1.mean()
                    val_change_f1 = val_cw_f1[1]
            self.scheduler.step()
 
            if (train_loss <= best_train_loss) or (val_mean_f1 >= best_val_mean_f1) or (val_change_f1 >= best_val_change_f1) or (val_miou >= best_val_miou):

                if train_loss <= best_train_loss:
                    best_train_loss = train_loss
                if val_mean_f1 >= best_val_mean_f1:
                    best_val_mean_f1 = val_mean_f1
                if val_change_f1 >= best_val_change_f1:
                    best_val_change_f1 = val_change_f1
                if val_miou >= best_val_miou:
                    best_val_miou = val_miou

                model_save_name = f"{current_path[-1]}_epoch_{epoch}_loss_{train_loss}_valmF1_{val_mean_f1}_valChangeF1_{val_change_f1}_mIoU_{val_miou}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"

                if config['procedures']['train']:
                    config['val_change_f1'] = str(val_change_f1)
                    config['val_mean_f1'] = str(val_mean_f1)
                    config['val_miou'] = str(val_miou)
                    config['train_loss'] = str(train_loss)
                    with open(model_save_dir+"config.yaml", 'w') as file:
                        yaml.dump(config, file)

                    torch.save({'epoch': epoch,
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'loss': train_loss},
                               model_save_dir+model_save_name)
                else:
                    exit()
                if config['visualization']['save_individual_plots']:
                    _, _ = self.validate(
                        epoch, cometml_experiment, save_individual_plots_specific=True)

        return train_losses, val_losses, mean_ious_val


if __name__ == "__main__":
    project_root = '/'.join(current_path[:-1])
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    main_config_path = f"{project_root}/configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    # experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"
    experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"
    # config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]

    experiment_config = utils.config_parser(
        experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime(
        '%Y-%m-%d-%H:%M:%S')


    # _{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    project_name = f"{current_path[-3]}_{current_path[-1]}_{config['dataset']}"
    experiment_name = f"attention_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    experiment = comet_ml.Experiment(api_key=config['cometml']['api_key'],
                                     project_name=project_name,
                                     workspace=config['cometml']['workspace'],
                                     display_summary_level=0)

    config['experiment_url'] = str(experiment.url)
    
    experiment.set_name(experiment_name)

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.set_default_dtype(torch.float32)

    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)
    # config['device_ids'] = device_ids
    # gpu_devices = ','.join([str(id) for id in device_ids])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = torch.device('cuda')

    # config['devices_used'] = gpu_devices
    experiment.log_asset_data(config)
    experiment.log_text(config)
    experiment.log_parameters(config)
    experiment.log_parameters(config['training'])
    experiment.log_parameters(config['evaluation'])
    experiment.log_parameters(config['visualization'])

    if config['visualization']['train_imshow'] or config['visualization']['val_imshow']:
        matplotlib.use('TkAgg')

    if config['training']['resume'] != False:
        base_path = '/'.join(config['training']['resume'].split('/')[:-1])
        pretrain_config_path = f"{base_path}/config.yaml"
        pretrain_config = utils.load_yaml_as_dict(pretrain_config_path)
        # print(config['training']['model_feats_channels'])
        # print(pretrain_config_path['training']['model_feats_channels'])
        config['data']['channels'] = pretrain_config['data']['channels']
        # if not config['training']['model_feats_channels'] == pretrain_config_path['training']['model_feats_channels']:
        #     print("the loaded model does not have the same number of features as configured in the experiment yaml file. Matching channel sizes to the loaded model instead.")
        # config['training']['model_feats_channels'] = pretrain_config_path['training']['model_feats_channels']
        config['data']['num_classes'] = pretrain_config['data']['num_classes']
        config['training']['model_feats_channels'] = pretrain_config['training']['model_feats_channels']

    if config['data']['name'] == 'watch' or config['data']['name'] == 'onera':
        coco_fpath = ub.expandpath(config['data'][config['location']]['train_coco_json'])
        dset = kwcoco.CocoDataset(coco_fpath)
        sampler = ndsampler.CocoSampler(dset)

        window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
        input_dims = (config['data']['image_size'], config['data']['image_size'])

        channels = config['data']['channels']
        num_channels = len(channels.split('|'))
        config['training']['num_channels'] = num_channels

        dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
        train_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])

        test_coco_fpath = ub.expandpath(config['data'][config['location']]['test_coco_json'])
        test_dset = kwcoco.CocoDataset(test_coco_fpath)
        test_sampler = ndsampler.CocoSampler(test_dset)

        test_dataset = SequenceDataset(test_sampler, window_dims, input_dims, channels)
        test_dataloader = test_dataset.make_loader(batch_size=config['evaluation']['batch_size'])
    else:
        train_dataloader = build_dataset(dataset_name=config['data']['name'], 
                                        root=config['data'][config['location']]['train_dir'], 
                                        batch_size=config['training']['batch_size'],
                                        num_workers=config['training']['num_workers'], 
                                        split='train',
                                        crop_size=config['data']['image_size'],
                                        channels=config['data']['channels'],
                                        )

        test_dataloader = build_dataset(dataset_name=config['data']['name'], 
                                        root=config['data'][config['location']]['train_dir'], 
                                        batch_size=config['evaluation']['batch_size'],
                                        num_workers=config['training']['num_workers'], 
                                        split='val',
                                        crop_size=config['data']['image_size'],
                                        channels=config['data']['channels'],
                                        )
        

    if not config['training']['model_diff_input']:
        config['training']['num_channels'] = 2*config['training']['num_channels']

    model = build_model(model_name=config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'],
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'],
                        num_channels=config['training']['num_channels'],
                        out_dim=config['training']['out_features_dim'],
                        feats=config['training']['model_feats_channels'])

    # model = SupConResNet(name=config['training']['backbone'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader),
                                                     eta_min=config['training']['learning_rate'])

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            # model_dict = model.state_dict()
            # if model_dict == checkpoint['model']:
            #     print(f"Succesfuly loaded model from {config['training']['resume']}")
            # else:
            #     pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            #     model_dict.update(pretrained_dict)
            #     model.load_state_dict(model_dict)
            #     print("There was model mismatch. Matching elements in the pretrained model were loaded.")
            missing_keys, unexpexted_keys = model.load_state_dict(checkpoint['model'], strict=False)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"loadded model succeffuly from: {config['training']['resume']}")
            print(f"Missing keys from loaded model: {missing_keys}, unexpected keys: {unexpexted_keys}")
        else:
            print("no checkpoint found at {}".format(
                config['training']['resume']))
            exit()

    trainer = Trainer(model,
                      train_dataloader,
                      test_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=test_dataloader,
                      test_with_full_supervision=config['training']['test_with_full_supervision']
                      )
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
