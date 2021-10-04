# -*- coding: utf-8 -*-
r"""
Prediction script for Rutgers Material Semenatic Segmentation Models

CommandLine:

    DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
    KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_30_loss_0.05691597167379317_valmIoU_0.5694727912477856_time_2021-08-07-09:01:01.pth"
    RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg.kwcoco.json

    # Generate Rutgers Features
    python -m watch.tasks.rutgers_material_seg.predict \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$RUTGERS_MATERIAL_COCO_FPATH \
        --num_workers=8 \
        --batch_size=32 --gpus 0

"""
import os
import comet_ml
import torch
import datetime
import random
import kwcoco
from torch import nn
import kwimage
import kwarray
import ndsampler
import watch.tasks.rutgers_material_seg.utils.eval_utils as eval_utils
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm  # NOQA
import ubelt as ub
import pathlib
import watch.tasks.rutgers_material_seg.utils.utils as utils
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
from torchvision import transforms
from skimage.filters import threshold_otsu as otsu
import matplotlib.pyplot as plt
import gc
import watch.tasks.rutgers_material_seg.utils.visualization as visualization

current_path = os.getcwd().split("/")
# if 0:
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.deterministic = True
#     torch.set_printoptions(precision=6, sci_mode=False)
#     np.set_printoptions(precision=3, suppress=True)


class Evaluator(object):
    def __init__(self,
                 model: object,
                 eval_loader: torch.utils.data.DataLoader,
                 config,
                 device='cuda'):
        """Evaluator class

        Args:
            model (object): trained or untrained model
            eval_loader (torch.utils.data.DataLader): loader with evaluation data
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
        """

        self.model = model
        self.eval_loader = eval_loader
        self.device = device
        self.config = config
        self.num_classes = self.config['data']['num_classes']
        self.max_label = self.num_classes
        self.inference_all_crops_params = [tuple([i, j, config['evaluation']['inference_window'], config['evaluation']['inference_window']]) for i in range(0, config['data']['image_size']) for j in range(0, config['data']['image_size'])]
        self.cmap = visualization.n_distinguishable_colors(nlabels=self.max_label,
                                                    first_color_black=True, last_color_black=True,
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
        filtered_features = filtered_features.view(bs, c, h, w) * features.view(bs, c, h, w)
        return filtered_features


    def eval(self, cometml_experiemnt) -> tuple:
        """evaluate a single epoch

        Args:

        Returns:
            None
        """
        vw_disagreement, histogram_distance, l1_dist, l2_dist = [], [], [], []
        topk_pre_histogram_distance, topk_pre_l1_dist, topk_pre_l2_dist = [], [], [], []
        topk_post_histogram_distance, topk_post_l1_dist, topk_post_l2_dist = [], [], [], []
        targets = []

        self.model.eval()

        with torch.no_grad():
            # Prog = ub.ProgIter
            Prog = tqdm
            pbar = Prog(enumerate(self.eval_loader), total=len(self.eval_loader), desc='predict rutgers')
            for batch_index, batch in pbar:
                outputs = batch
                images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]

                mask = torch.stack(mask)
                mask = mask.long().squeeze(1)

                bs, c, t, h, w = images.shape

                assert images.shape[2] == 2, 'only handles 2 frames'

                image1 = images[:, :, 0, :, :]
                image2 = images[:, :, 1, :, :]
                mask1 = mask[:, 0, :, :]  # NOQA
                mask2 = mask[:, 1, :, :]  # NOQA

                images = images.to(self.device)
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                mask = mask.to(self.device)

                image1 = utils.stad_image(image1)
                image2 = utils.stad_image(image2)

                output1, features1 = self.model(image1)  # [B,22,150,150]
                output2, features2 = self.model(image2)

                masks1 = F.softmax(output1, dim=1)  # .detach()
                masks2 = F.softmax(output2, dim=1)  # .detach()
                masks1 = self.high_confidence_filter(masks1,
                                                     cutoff_top=self.config['high_confidence_threshold']['val_cutoff'],
                                                     cutoff_low=self.config['high_confidence_threshold']['val_low_cutoff'])
                masks2 = self.high_confidence_filter(masks2,
                                                     cutoff_top=self.config['high_confidence_threshold']['val_cutoff'],
                                                     cutoff_low=self.config['high_confidence_threshold']['val_low_cutoff'])


                pred1 = masks1.max(1)[1].cpu().detach()  # .numpy()
                pred2 = masks2.max(1)[1].cpu().detach()  # .numpy()

                vw_disagreement_pred = (pred1 != pred2).type(torch.uint8)

                inference_otsu_coeff = 1.6
                hist_inference_otsu_coeff = 0.95
                pad_amount = (self.config['evaluation']['inference_window']-1)//2

                padded_output1 = F.pad(input=output1, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                padded_output2 = F.pad(input=output2, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                patched_padded_output1 = torch.stack([transforms.functional.crop(padded_output1, *params) for params in self.inference_all_crops_params], dim=1) #.flatten(-3,-1)
                patched_padded_output2 = torch.stack([transforms.functional.crop(padded_output2, *params) for params in self.inference_all_crops_params], dim=1) #.flatten(-3,-1)

                padded_mask1 = F.pad(input=masks1, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                padded_mask2 = F.pad(input=masks2, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                patched_padded_mask1 = torch.stack([transforms.functional.crop(padded_mask1, *params) for params in self.inference_all_crops_params], dim=1)#.flatten(-3,-1)
                patched_padded_mask2 = torch.stack([transforms.functional.crop(padded_mask2, *params) for params in self.inference_all_crops_params], dim=1)#.flatten(-3,-1)

                patched_padded_output1_distributions = patched_padded_output1.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]
                patched_padded_output2_distributions = patched_padded_output2.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]
                patched_padded_mask1_distributions = patched_padded_mask1.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]
                patched_padded_mask2_distributions = patched_padded_mask2.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]

                topk_patched_output1_pre_distributions, largest_elements_pre_inds = torch.topk(patched_padded_output1_distributions, k=11, sorted=False, dim=3)
                topk_patched_output2_pre_distributions = torch.gather(patched_padded_output2_distributions, dim=3, index=largest_elements_pre_inds)

                patched_padded_output1_distributions = patched_padded_output1_distributions.sum(axis=3) #[bs, n_patches, k]
                patched_padded_output2_distributions = patched_padded_output2_distributions.sum(axis=3) #[bs, n_patches, k]
                topk_patched_padded_output1_pre_distributions = topk_patched_output1_pre_distributions.sum(axis=3) #[bs, n_patches, k]
                topk_patched_padded_output2_pre_distributions = topk_patched_output2_pre_distributions.sum(axis=3) #[bs, n_patches, k]

                patched_padded_mask1_distributions = patched_padded_mask1_distributions.sum(axis=3) #[bs, n_patches, k]
                patched_padded_mask2_distributions = patched_padded_mask2_distributions.sum(axis=3) #[bs, n_patches, k]

                topk_patched_output1_post_distributions, largest_elements_post_inds = torch.topk(patched_padded_output1_distributions, k=11, sorted=False, dim=2)
                topk_patched_output2_post_distributions = torch.gather(patched_padded_output2_distributions, dim=2, index=largest_elements_post_inds)


                # l1 region-wise inference raw features
                l1_patched_diff_change_features = torch.abs((patched_padded_output1_distributions - patched_padded_output2_distributions).sum(axis=2)).view(bs,h,w)
                l1_dist_change_feats_pred = torch.zeros_like(l1_patched_diff_change_features)
                l1_inference_otsu_threshold = inference_otsu_coeff*otsu(l1_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l1_dist_change_feats_pred[l1_patched_diff_change_features > l1_inference_otsu_threshold] = 1
                l1_dist_change_feats_pred = l1_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # l1 region-wise inference pre topk
                l1_patched_diff_change_pre_topk = torch.abs((topk_patched_padded_output1_pre_distributions - topk_patched_padded_output2_pre_distributions).sum(axis=2)).view(bs,h,w)
                l1_dist_change_feats_pred_pre_topk = torch.zeros_like(l1_patched_diff_change_pre_topk)
                l1_inference_otsu_threshold = inference_otsu_coeff*otsu(l1_patched_diff_change_pre_topk.cpu().detach().numpy(), nbins=256)
                l1_dist_change_feats_pred_pre_topk[l1_patched_diff_change_pre_topk > l1_inference_otsu_threshold] = 1
                l1_dist_change_feats_pred_pre_topk = l1_dist_change_feats_pred_pre_topk.cpu().detach().type(torch.uint8)

                # l1 region-wise inference post topk
                l1_patched_diff_change_post_topk = torch.abs((topk_patched_output1_post_distributions - topk_patched_output2_post_distributions).sum(axis=2)).view(bs,h,w)
                l1_dist_change_feats_pred_post_topk = torch.zeros_like(l1_patched_diff_change_post_topk)
                l1_inference_otsu_threshold = inference_otsu_coeff*otsu(l1_patched_diff_change_post_topk.cpu().detach().numpy(), nbins=256)
                l1_dist_change_feats_pred_post_topk[l1_patched_diff_change_post_topk > l1_inference_otsu_threshold] = 1
                l1_dist_change_feats_pred_post_topk = l1_dist_change_feats_pred_post_topk.cpu().detach().type(torch.uint8)

                # l2 region-wise inference raw features
                l2_patched_diff_change_features = torch.sqrt(torch.pow(patched_padded_output1_distributions - patched_padded_output2_distributions, 2).sum(axis=2)).view(bs,h,w)
                l2_dist_change_feats_pred = torch.zeros_like(l2_patched_diff_change_features)
                l2_inference_otsu_threshold = inference_otsu_coeff*otsu(l2_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l2_dist_change_feats_pred[l2_patched_diff_change_features > l2_inference_otsu_threshold] = 1
                l2_dist_change_feats_pred = l2_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # l2 region-wise inference pre topk
                l2_patched_diff_change_pre_topk = torch.sqrt(torch.pow(topk_patched_padded_output1_pre_distributions - topk_patched_padded_output2_pre_distributions, 2).sum(axis=2)).view(bs,h,w)
                l2_dist_change_feats_pred_pre_topk = torch.zeros_like(l2_patched_diff_change_pre_topk)
                l2_inference_otsu_threshold = inference_otsu_coeff*otsu(l2_patched_diff_change_pre_topk.cpu().detach().numpy(), nbins=256)
                l2_dist_change_feats_pred_pre_topk[l2_patched_diff_change_pre_topk > l2_inference_otsu_threshold] = 1
                l2_dist_change_feats_pred_pre_topk = l2_dist_change_feats_pred_pre_topk.cpu().detach().type(torch.uint8)

                # l2 region-wise inference  post topk
                l2_patched_diff_change_features = torch.sqrt(torch.pow(patched_padded_output1_distributions - patched_padded_output2_distributions, 2).sum(axis=2)).view(bs,h,w)
                l2_dist_change_feats_pred = torch.zeros_like(l2_patched_diff_change_features)
                l2_inference_otsu_threshold = inference_otsu_coeff*otsu(l2_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l2_dist_change_feats_pred[l2_patched_diff_change_features > l2_inference_otsu_threshold] = 1
                l2_dist_change_feats_pred = l2_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # histogram intersection
                # normalized_patched_padded_output1_distributions = (patched_padded_output1_distributions - patched_padded_output1_distributions.min(dim=2, keepdim=True)[0])/(patched_padded_output1_distributions.max(dim=2, keepdim=True)[0] - patched_padded_output1_distributions.min(dim=2, keepdim=True)[0])
                # normalized_patched_padded_output2_distributions = (patched_padded_output2_distributions - patched_padded_output2_distributions.min(dim=2, keepdim=True)[0])/(patched_padded_output2_distributions.max(dim=2, keepdim=True)[0] - patched_padded_output2_distributions.min(dim=2, keepdim=True)[0])
                minima = torch.minimum(patched_padded_mask1_distributions, patched_padded_mask2_distributions)
                # minima = torch.minimum(masks1, masks2)
                histograms_intersection = torch.true_divide(minima.sum(axis=2), patched_padded_mask2_distributions.sum(axis=2)).view(bs,h,w)

                histc_int_change_feats_pred = torch.zeros_like(histograms_intersection)
                histc_int_inference_otsu_threshold = hist_inference_otsu_coeff*otsu(histograms_intersection.cpu().detach().numpy(), nbins=256)
                histc_int_change_feats_pred[histograms_intersection < histc_int_inference_otsu_threshold] = 1
                histc_int_change_feats_pred = histc_int_change_feats_pred.cpu().detach().type(torch.uint8)

                # print(histc_int_change_feats_pred.shape)

                vw_disagreement.append(vw_disagreement_pred)
                histogram_distance.append(histc_int_change_feats_pred)
                l1_dist.append(l1_dist_change_feats_pred)
                l2_dist.append(l2_dist_change_feats_pred)
                mask1[mask1 == -1] = 0
                targets.append(mask1.cpu())

                if self.config['visualization']['val_visualizer']:

                    batch_index_to_show=0
                    figure = plt.figure(figsize=(
                        self.config['visualization']['fig_size'], self.config['visualization']['fig_size']))
                    ax1 = figure.add_subplot(3, 4, 1)
                    ax2 = figure.add_subplot(3, 4, 2)
                    ax3 = figure.add_subplot(3, 4, 3)
                    ax4 = figure.add_subplot(3, 4, 4)
                    ax5 = figure.add_subplot(3, 4, 5)
                    ax6 = figure.add_subplot(3, 4, 6)
                    ax7 = figure.add_subplot(3, 4, 7)
                    ax8 = figure.add_subplot(3, 4, 8)
                    ax9 = figure.add_subplot(3, 4, 9)
                    ax10 = figure.add_subplot(3, 4, 10)
                    ax11 = figure.add_subplot(3, 4, 11)
                    ax12 = figure.add_subplot(3, 4, 12)

                    cmap_gradients = plt.cm.get_cmap('jet')

                    image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                    image_show1 = np.flip(image_show1, axis=2)

                    image_show2 = np.transpose(image2.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                    image_show2 = np.flip(image_show2, axis=2)

                    image_show1 = (image_show1 - image_show1.min())/(image_show1.max() - image_show1.min())
                    image_show2 = (image_show2 - image_show2.min())/(image_show2.max() - image_show2.min())

                    gt_mask_show1 = mask1.cpu().detach()[batch_index_to_show, :, :].numpy().squeeze()

                    l1_dist_change_feats_pred_show = l1_dist_change_feats_pred.numpy()[batch_index_to_show, :, :]
                    l2_dist_change_feats_pred_show = l2_dist_change_feats_pred.numpy()[batch_index_to_show, :, :]
                    histc_int_change_feats_pred_show = histc_int_change_feats_pred.numpy()[batch_index_to_show, :, :]
                    vw_disagreement_pred_show = vw_disagreement_pred.numpy()[batch_index_to_show, :, :]

                    l1_patched_diff_change_features_show = l1_patched_diff_change_features.cpu().detach().numpy()[batch_index_to_show, :, :]
                    l2_patched_diff_change_features_show = l2_patched_diff_change_features.cpu().detach().numpy()[batch_index_to_show, :, :]
                    histograms_intersection_show = histograms_intersection.cpu().detach().numpy()[batch_index_to_show, :, :]

                    l1_patched_diff_change_features_show = (l1_patched_diff_change_features_show - l1_patched_diff_change_features_show.min())/(l1_patched_diff_change_features_show.max() - l1_patched_diff_change_features_show.min())
                    l2_patched_diff_change_features_show = (l2_patched_diff_change_features_show - l2_patched_diff_change_features_show.min())/(l2_patched_diff_change_features_show.max() - l2_patched_diff_change_features_show.min())
                    histograms_intersection_show = (histograms_intersection_show - histograms_intersection_show.min())/(histograms_intersection_show.max() - histograms_intersection_show.min())

                    pred1_show = masks1.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]
                    pred2_show = masks2.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]

                    l1_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*l1_dist_change_feats_pred_show)
                    l2_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*l2_dist_change_feats_pred_show)
                    histc_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*histc_int_change_feats_pred_show)
                    vw_dis_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*vw_disagreement_pred_show)

                    pred1_show[pred1_show == -1] = 0
                    pred2_show[pred2_show == -1] = 0
                    gt_mask_show_no_bg1 = np.ma.masked_where(gt_mask_show1 == 0, gt_mask_show1)
                    # gt_mask_show_no_bg2 = np.ma.masked_where(gt_mask_show2==0,gt_mask_show2)
                    # logits_show_no_bg = np.ma.masked_where(logits_show==0,logits_show)

                    classes_in_gt = np.unique(gt_mask_show1)
                    ax1.imshow(image_show1)

                    ax2.imshow(image_show2)

                    ax3.imshow(image_show1)
                    ax3.imshow(gt_mask_show1, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    ax4.imshow(image_show1)
                    ax4.imshow(pred1_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    ax5.imshow(l1_patched_diff_change_features_show)

                    ax6.imshow(l2_patched_diff_change_features_show)

                    ax7.imshow(histograms_intersection_show)

                    ax8.imshow(image_show2)
                    ax8.imshow(pred2_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    ax9.imshow(l1_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    ax10.imshow(l2_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    ax11.imshow(histc_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    ax12.imshow(vw_dis_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)

                    if self.config['visualization']['titles']:
                        ax1.set_title(f"image_show1", fontsize=self.config['visualization']['font_size'])
                        ax2.set_title(f"image_show2", fontsize=self.config['visualization']['font_size'])
                        ax3.set_title(f"gt_mask_show1", fontsize=self.config['visualization']['font_size'])
                        ax4.set_title(f"pred1_show", fontsize=self.config['visualization']['font_size'])
                        ax5.set_title(f"l1_patched_diff_change_features_show", fontsize=self.config['visualization']['font_size'])
                        ax6.set_title(f"l2_patched_diff_change_features_show", fontsize=self.config['visualization']['font_size'])
                        ax7.set_title(f"histograms_intersection_show", fontsize=self.config['visualization']['font_size'])
                        ax8.set_title(f"pred2_show", fontsize=self.config['visualization']['font_size'])
                        ax9.set_title(f"l1_fp_tp_fn_prediction_mask", fontsize=self.config['visualization']['font_size'])
                        ax10.set_title(f"l2_fp_tp_fn_prediction_mask", fontsize=self.config['visualization']['font_size'])
                        ax11.set_title(f"histc_fp_tp_fn_prediction_mask", fontsize=self.config['visualization']['font_size'])
                        ax12.set_title(f"vw_dis_fp_tp_fn_prediction_mask", fontsize=self.config['visualization']['font_size'])
                        # figure.suptitle(
                        #     f"Epoch: {epoch+1}\nGT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])

                    figure.tight_layout()
                    if self.config['visualization']['val_imshow']:
                        plt.show()

                    cometml_experiemnt.log_figure(figure_name=f"Validation", figure=figure)
                    figure.clear()
                    plt.cla()
                    plt.clf()
                    plt.close('all')
                    plt.close(figure)
                    gc.collect()

        l1_mean_iou, l1_precision, l1_recall = eval_utils.compute_jaccard(l1_dist, targets, num_classes=2)
        l2_mean_iou, l2_precision, l2_recall = eval_utils.compute_jaccard(l2_dist, targets, num_classes=2)
        vw_disagreement_mean_iou, vw_disagreement_precision, vw_disagreement_recall = eval_utils.compute_jaccard(vw_disagreement, targets, num_classes=2)
        hist_mean_iou, hist_precision, hist_recall = eval_utils.compute_jaccard(histogram_distance, targets, num_classes=2)
        
        l1_precision = np.array(l1_precision)
        l1_recall = np.array(l1_recall)
        l1_f1 = 2 * (l1_precision * l1_recall) / (l1_precision + l1_recall)

        l2_precision = np.array(l2_precision)
        l2_recall = np.array(l2_recall)
        l1_f2 = 2 * (l2_precision * l2_recall) / (l2_precision + l2_recall)

        vw_disagreement_precision = np.array(vw_disagreement_precision)
        vw_disagreement_recall = np.array(vw_disagreement_recall)
        vw_disagreement_f1 = 2 * (vw_disagreement_precision * vw_disagreement_recall) / (vw_disagreement_precision + vw_disagreement_recall)

        hist_precision = np.array(hist_precision)
        hist_recall = np.array(hist_recall)
        hist_f1 = 2 * (hist_precision * hist_recall) / (hist_precision + hist_recall)

        print("\n")
        print({f"l1_recall class {str(x)}": l1_recall[x] for x in range(len(l1_recall))})
        print({f"l1_precision class {str(x)}": l1_precision[x] for x in range(len(l1_precision))})
        print({f"l1_f1 class {str(x)}": l1_f1[x] for x in range(len(l1_f1))})

        print("\n")
        print({f"l2_recall class {str(x)}": l2_recall[x] for x in range(len(l2_recall))})
        print({f"l2_precision class {str(x)}": l2_precision[x] for x in range(len(l2_precision))})
        print({f"l2_f2 class {str(x)}": l1_f2[x] for x in range(len(l1_f2))})

        print("\n")
        print({f"vw_disagreement_recall class {str(x)}": vw_disagreement_recall[x] for x in range(len(vw_disagreement_recall))})
        print({f"vw_disagreement_precision class {str(x)}": vw_disagreement_precision[x] for x in range(len(vw_disagreement_precision))})
        print({f"vw_disagreement_f1 class {str(x)}": vw_disagreement_f1[x] for x in range(len(vw_disagreement_f1))})

        print("\n")
        print({f"hist_recall class {str(x)}": hist_recall[x] for x in range(len(hist_recall))})
        print({f"hist_precision class {str(x)}": hist_precision[x] for x in range(len(hist_precision))})
        print({f"hist_f1 class {str(x)}": hist_f1[x] for x in range(len(hist_f1))})

        return

    def forward(self, cometml_experiemnt) -> tuple:
        """forward pass for all epochs

        Args:
            cometml_experiment (object): comet ml experiment for logging
            world_size (int, optional): for distributed training. Defaults to 8.

        Returns:
            tuple: (train losses, validation losses, mIoU)
        """

        if self.config['procedures']['validate']:
            self.eval(cometml_experiemnt)
        return



def main(cmdline=True, **kwargs):
    """
    Ignore:
        # Hack in overrides because none of this is parameterized
        # state_dict = torch.load(checkpoint_fpath)
        checkpoint_fpath = ub.expandpath("$HOME/data/dvc-repos/smart_watch_dvc/models/rutgers/experiments_epoch_30_loss_0.05691597167379317_valmIoU_0.5694727912477856_time_2021-08-07-09:01:01.pth")
        cmdline = False
        kwargs = dict(
            default_config_key='iarpa',
            checkpoint_fpath=checkpoint_fpath,
            test_dataset=ub.expandpath("$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/data.kwcoco.json"),
            pred_dataset='./test-pred/pred.kwcoco.json',
        )
    """
    project_root = '/'.join(current_path[:-1])
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    main_config_path = f"{project_root}/configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    # experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"
    experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"
    # config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]

    experiment_config = utils.config_parser(experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    project_name = f"{current_path[-3]}_{current_path[-1]}_{config['dataset']}"  # _{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment_name = f"attention_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    experiment = comet_ml.Experiment(api_key=config['cometml']['api_key'],
                                     project_name=project_name,
                                     workspace=config['cometml']['workspace'],
                                     display_summary_level=0)
    experiment.set_name(experiment_name)

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.set_default_dtype(torch.float32)

    device_ids = list(range(torch.cuda.device_count()))
    config['device_ids'] = device_ids
    gpu_devices = ','.join([str(id) for id in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = torch.device('cuda')

    config['devices_used'] = gpu_devices
    experiment.log_asset_data(config)
    experiment.log_text(config)
    experiment.log_parameters(config)
    experiment.log_parameters(config['training'])
    experiment.log_parameters(config['evaluation'])
    experiment.log_parameters(config['visualization'])

    # print(config['data']['image_size'])
    coco_fpath = ub.expandpath(config['data'][config['location']]['train_coco_json'])
    dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(dset)


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

    window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
    input_dims = (config['data']['image_size'], config['data']['image_size'])

    # channels = 'B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A'
    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
    # channels = 'red|green|blue'
    # channels = 'gray'
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    print(dataset.__len__())
    eval_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])

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

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            model.load_state_dict(checkpoint['model'], strict= False)
            print(f"loaded model from {config['training']['resume']}")
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
            exit()

    evaler = Evaluator(model,
                      eval_dataloader,
                      config=config
                      )
    self = evaler  # NOQA
    evaler.forward(experiment)


if __name__ == "__main__":
    main()
