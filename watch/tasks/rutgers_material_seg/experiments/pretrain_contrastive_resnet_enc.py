# flake8: noqa

import sys
import os
current_path = os.getcwd().split("/")

import time
import matplotlib
import gc
import cv2
import comet_ml
import torch
import datetime
import warnings
import yaml
import random
import kwcoco
import kwimage
import ndsampler
import matplotlib.pyplot as plt
import numpy as np
import ubelt as ub
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from scipy import ndimage
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import watch.tasks.rutgers_material_seg.utils.utils as utils
import watch.tasks.rutgers_material_seg.utils.eval_utils as eval_utils
import watch.tasks.rutgers_material_seg.utils.visualization as visualization
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.datasets import build_dataset
from watch.tasks.rutgers_material_seg.models.supcon import SupConResNet
from watch.tasks.rutgers_material_seg.models.losses import SupConLoss, simCLR_loss, QuadrupletLoss
from fast_pytorch_kmeans import KMeans
from skimage.filters import threshold_otsu as otsu
from watch.tasks.rutgers_material_seg.models.canny_edge import CannyFilter
from watch.tasks.rutgers_material_seg.models.tex_refine import TeRN


torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)


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
        # self.patch_location_head = nn.Conv2d(9800, 8, kernel_size=1).to(device)
        self.patch_location_head = nn.Linear(in_features=9800, out_features=8).to(device)
        # self.k = config['training']['out_features_dim']
        self.k = config['data']['num_classes']
        self.kmeans = KMeans(n_clusters=self.k, mode='euclidean', verbose=0, minibatch=None)
        self._aff = TeRN(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
        self.max_label = 10
        # self.all_crops_params = [tuple([i,j,config['data']['window_size'], config['data']['window_size']]) for i in range(config['data']['window_size'],config['data']['image_size']-config['data']['window_size']) for j in range(config['data']['window_size'],config['data']['image_size']-config['data']['window_size'])]
        self.all_crops_params = [tuple([i, j, config['data']['window_size'], config['data']['window_size']]) for i in range(0, config['data']['image_size']) for j in range(0, config['data']['image_size'])]
        self.inference_all_crops_params = [tuple([i, j, config['evaluation']['inference_window'], config['evaluation']['inference_window']]) for i in range(0, config['data']['image_size']) for j in range(0, config['data']['image_size'])]
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision

        self.train_second_transform = transforms.Compose([
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomAffine(degrees=(-20, 20)),
                                                        # transforms.RandomPosterize(bits=2),
                                                        # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                                        # transforms.RandomEqualize(),
                                                        ])
        self.crop_size = (config['data']['window_size'], config['data']['window_size'])

        self.cmap = visualization.n_distinguishable_colors(nlabels=self.max_label,
                                                           first_color_black=True, last_color_black=True,
                                                           bg_alpha=config['visualization']['bg_alpha'],
                                                           fg_alpha=config['visualization']['fg_alpha'])

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

    def high_confidence_filter(self, features: torch.Tensor, cutoff_top: float = 0.75,
                               cutoff_low: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
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
        filtered_features = filtered_features.view(bs, c, h, w)

        return filtered_features

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
        self.model.train()
        print(f"starting epoch {epoch}")
        loader_size = len(self.train_loader)
        if config['visualization']['train_visualization_divisor'] > loader_size:
            config['visualization']['train_visualization_divisor'] = loader_size
        iter_visualization = loader_size // config['visualization']['train_visualization_divisor']
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        batch_index_to_show = config['visualization']['batch_index_to_show']
        for batch_index, batch in pbar:
            outputs = batch
            if config['data']['name'] == 'watch' or config['data']['name'] == 'onera':
                images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                negative_images = outputs['inputs']['negative_im'].data[0]
                image_name = outputs['tr'].data[0][batch_index_to_show]['gids']
                mask = torch.stack(mask)
                mask = mask.long().squeeze(1)
                image1 = images[:, :, 0, :, :]
                negative_image1 = negative_images[:, :, 0, :, :]
                mask1 = mask[:, 0, :, :]
                images = images.to(device)
                negative_images = negative_images.to(device)
                mask = mask.to(device)
                class_to_show = max(0, torch.unique(mask)[-1] - 1)
                bs, c, t, h, w = images.shape
            else:
                image1, image2 = outputs['inputs']['image1'], outputs['inputs']['image2']
                negative_image1 = outputs['inputs']['negative_image']
                image_name = outputs['visuals']['image_name'][batch_index_to_show]
                bs, c, h, w = image1.shape
                image2 = image2.to(device)

            image1 = image1.to(device)
            negative_image1 = negative_image1.to(device)

            # image1 = utils.stad_image(image1).float()
            image2 = utils.stad_image(image1).float()
            negative_image1 = utils.stad_image(negative_image1).float()

            sampled_crops = random.sample(self.all_crops_params, config['training']['n_samples'])
            patched_image1 = torch.stack([transforms.functional.crop(image1, *params) for params in sampled_crops], dim=1)
            patched_image2 = torch.stack([transforms.functional.crop(image2, *params) for params in sampled_crops], dim=1)
            patched_negative_image1 = torch.stack([transforms.functional.crop(negative_image1, *params) for params in sampled_crops], dim=1)

            bs, ps, c, h, w = patched_image1.shape
            patched_image1 = patched_image1.view(bs * ps, c, h, w)
            patched_image2 = patched_image2.view(bs * ps, c, h, w)
            patched_negative_image1 = patched_negative_image1.view(bs * ps, c, h, w)

            output1 = self.model(patched_image1, image1, sampled_crops)  ## [B,22,150,150]
            output2 = self.model(patched_image2, image2, sampled_crops)  ## [B,22,150,150]
            negative_output1 = self.model(patched_negative_image1, negative_image1, sampled_crops)  ## [B,22,150,150]

            # output1_uncrop = torch.stack(torch.chunk(output1, chunks=bs, dim=0), dim=0)
            # print(f"output1_uncrop: {output1_uncrop.shape}")
            # output1_uncrop = (output1_uncrop).sum(axis=2).view(bs,h,w)
            # print(f"output1_uncrop: {output1_uncrop.shape}")

            # output1, quant1 = self.model(patched_image1)  ## [B,22,150,150]
            # output2, quant2 = self.model(patched_image2)  ## [B,22,150,150]
            # negative_output1, neg_quant1 = self.model(patched_negative_image1)  ## [B,22,150,150]

            # print(features1.shape)

            # patched_output1 = torch.stack([transforms.functional.crop(output1, *params) for params in self.all_crops_params],dim=1)
            # patched_output2 = torch.stack([transforms.functional.crop(output2, *params) for params in self.all_crops_params],dim=1)
            # patched_negative_output1 = torch.stack([transforms.functional.crop(negative_output1, *params) for params in self.all_crops_params],dim=1)

            # bs, ps, c, ph, pw = patched_output1.shape
            # patched_output1 = patched_output1.view(bs*ps,c,ph,pw)
            # patched_output2 = patched_output2.view(bs*ps,c,ph,pw)
            # patched_negative_output1 = patched_negative_output1.view(bs*ps,c,ph,pw)

            # patch_max = 8000
            # patched_output1 = F.normalize(patched_output1, dim=1, p=1) #[bs*ps, c*h*w]
            # patched_output2 = F.normalize(patched_output2, dim=1, p=1)
            # features = torch.cat([patched_output1.unsqueeze(1), patched_output2.unsqueeze(1)], dim=1)[:patch_max,:,:]
            # print(patched_output1.shape)
            # print(features1.shape)
            # features1 = F.normalize(features1, dim=1, p=1) #[bs*ps, c*h*w]
            # features2 = F.normalize(features2, dim=1, p=1)
            # print(output1.shape)
            # print(output2.shape)
            # print(negative_output1.shape)
            # features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)#[:patch_max,:,:]

            # pre-text: Augmentation
            # loss = 5*self.contrastive_loss(features)
            # loss = 30*F.triplet_margin_loss(output1,#.unsqueeze(0),
            #                                 output2,#.unsqueeze(0),
            #                                 negative_output1,
            #                                 swap=True,
            #                                 p=1,
            #                                 reduction='mean'
            #                                 )
            targets_positive = torch.zeros(output1.shape[0], dtype=torch.long).to(device)
            targets_negative = torch.ones(output1.shape[0], dtype=torch.long).to(device)
            outputs_positive = torch.cat([output1, output2], dim=1)
            outputs_negative = torch.cat([output1, negative_output1], dim=1)
            # outputs_negative2 = torch.cat([output2, negative_output1], dim=1)

            loss = F.cross_entropy(outputs_positive.float(), targets_positive.long(), reduction='mean')
            loss += F.cross_entropy(outputs_negative.float(), targets_negative.long(), reduction='mean')
            # loss += F.cross_entropy(outputs_negative2.float(), targets_negative.long(), reduction='mean')
            # loss += 30*F.triplet_margin_loss(features1,#.unsqueeze(0),
            #                                 features2,#.unsqueeze(0),
            #                                 negative_features1
            #                                 )
            # print(loss)

            # pre-text: relative patch distance:
            # number_of_patches = 1000
            # patch_sample_range = [(-7,-7), (0,-7), (7,-7), (-7,0), (7,0), (-7,7), (0,7), (7,7)]

            # inds1 = random.sample(self.all_crops_params, bs*number_of_patches)
            # inds1_adj_patches, labels = [], []
            # for x,y,h,w in inds1:
            #     label = random.randint(0, 7)
            #     adj_x = x + patch_sample_range[label][0]
            #     adj_y = y + patch_sample_range[label][0]
            #     inds1_adj_patches.append((adj_x, adj_y, h, w))
            #     labels.append(label)

            # center_patches = torch.flatten(torch.stack([transforms.functional.crop(output1, *params) for params in inds1],dim=1), start_dim=-3, end_dim=-1)
            # adj_patches = torch.flatten(torch.stack([transforms.functional.crop(output1, *params) for params in inds1_adj_patches],dim=1), start_dim=-3, end_dim=-1)
            # cat_patches = torch.cat([center_patches, adj_patches], dim=2)
            # patch_class_head = self.patch_location_head(cat_patches).view(bs*2*number_of_patches,-1)
            # labels = torch.stack([torch.Tensor(labels), torch.Tensor(labels)],dim=0).view(-1).to(device).long()
            # loss += F.cross_entropy(patch_class_head, labels)
            # relative_distances = torch.sqrt(torch.pow(inds1 - inds2, 2).sum(axis=1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_seg += loss.item()

            total_loss += loss.item()
            if 'mask1' in vars(__builtins__):
                masks1 = F.softmax(output1, dim=1)
                masks2 = F.softmax(output2, dim=1)
                pred1 = masks1.max(1)[1].cpu().detach()  # .numpy()
                pred2 = masks2.max(1)[1].cpu().detach()  # .numpy()
                # change_detection_prediction = (pred1 != pred2).type(torch.uint8)

                inference_otsu_coeff = 1.4
                hist_inference_otsu_coeff = 1.0
                topk = 45

                pad_amount = (config['evaluation']['inference_window'] - 1) // 2
                padded_output1 = F.pad(input=output1, pad=(pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')
                padded_output2 = F.pad(input=output2, pad=(pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')
                #those are [bs, n_patches, k, window_size, window_size], where each patch is represented by a histogram of k,window_size,window_size
                patched_padded_output1 = torch.stack([transforms.functional.crop(padded_output1, *params) for params in self.inference_all_crops_params], dim=1)  # .flatten(-3,-1)
                patched_padded_output2 = torch.stack([transforms.functional.crop(padded_output2, *params) for params in self.inference_all_crops_params], dim=1)  # .flatten(-3,-1)

                padded_mask1 = F.pad(input=masks1, pad=(pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')
                padded_mask2 = F.pad(input=masks2, pad=(pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')
                patched_padded_mask1 = torch.stack([transforms.functional.crop(padded_mask1, *params) for params in self.inference_all_crops_params], dim=1)  # .flatten(-3,-1)
                patched_padded_mask2 = torch.stack([transforms.functional.crop(padded_mask2, *params) for params in self.inference_all_crops_params], dim=1)  # .flatten(-3,-1)

                # here we sum all vectors to make a 1,k vector for each patch
                patched_padded_output1_distributions = patched_padded_output1.flatten(-2, -1)  # .sum(axis=3) #[bs, n_patches, k]
                patched_padded_output2_distributions = patched_padded_output2.flatten(-2, -1)  # .sum(axis=3) #[bs, n_patches, k]
                patched_padded_mask1_distributions = patched_padded_mask1.flatten(-2, -1)  # .sum(axis=3) #[bs, n_patches, k]
                patched_padded_mask2_distributions = patched_padded_mask2.flatten(-2, -1)  # .sum(axis=3) #[bs, n_patches, k]

                patched_padded_output1_distributions = patched_padded_output1_distributions.sum(axis=3)
                patched_padded_output2_distributions = patched_padded_output2_distributions.sum(axis=3)
                patched_padded_mask1_distributions = patched_padded_mask1_distributions.sum(axis=3)  # [bs, n_patches, k]
                patched_padded_mask2_distributions = patched_padded_mask2_distributions.sum(axis=3)  # [bs, n_patches, k]

                topk_patched_mask1_post_distributions, largest_elements_post_inds = torch.topk(patched_padded_mask1_distributions, k=topk, sorted=False, dim=2)
                topk_patched_mask2_post_distributions = torch.gather(patched_padded_mask2_distributions, dim=2, index=largest_elements_post_inds)

                # normalize those vectors to 0-1
                normalized_patched_padded_mask1_distributions = (patched_padded_mask1_distributions - patched_padded_mask1_distributions.min(dim=2, keepdim=True)[0]) / (patched_padded_mask1_distributions.max(dim=2, keepdim=True)[0] - patched_padded_mask1_distributions.min(dim=2, keepdim=True)[0])
                normalized_patched_padded_mask2_distributions = (patched_padded_mask2_distributions - patched_padded_mask2_distributions.min(dim=2, keepdim=True)[0]) / (patched_padded_mask2_distributions.max(dim=2, keepdim=True)[0] - patched_padded_mask2_distributions.min(dim=2, keepdim=True)[0])

                # histogram intersection raw features
                # normalized_patched_padded_output1_distributions = (patched_padded_output1_distributions - patched_padded_output1_distributions.min(dim=2, keepdim=True)[0])/(patched_padded_output1_distributions.max(dim=2, keepdim=True)[0] - patched_padded_output1_distributions.min(dim=2, keepdim=True)[0])
                # normalized_patched_padded_output2_distributions = (patched_padded_output2_distributions - patched_padded_output2_distributions.min(dim=2, keepdim=True)[0])/(patched_padded_output2_distributions.max(dim=2, keepdim=True)[0] - patched_padded_output2_distributions.min(dim=2, keepdim=True)[0])
                minima = torch.minimum(topk_patched_mask1_post_distributions, topk_patched_mask2_post_distributions)
                histograms_intersection_features = torch.true_divide(minima.sum(axis=2), topk_patched_mask2_post_distributions.sum(axis=2)).view(bs, h, w)
                histc_int_change_feats_pred = torch.zeros_like(histograms_intersection_features)
                histc_int_inference_otsu_threshold = hist_inference_otsu_coeff * otsu(histograms_intersection_features.cpu().detach().numpy(), nbins=128)
                histc_int_change_feats_pred[histograms_intersection_features < histc_int_inference_otsu_threshold] = 1
                histc_int_change_feats_pred = histc_int_change_feats_pred.cpu().detach().type(torch.uint8)

                # matplotlib.use('TkAgg')
                # hist_fig = plt.figure()
                # hax1 = hist_fig.add_subplot(1,1,1)
                # hax1.hist(patched_padded_output1_distributions[0,0,:].cpu().detach().numpy(), bins=40)
                # hax1.hist(patched_padded_output2_distributions[0,0,:].cpu().detach().numpy(), bins=40)
                # hax1.set_title("Residual Histograms Intersection of Two Random Correspodning Patches")
                # plt.show()

                # kl_div_distance = torch.abs(F.kl_div(patched_padded_output1_distributions, patched_padded_output2_distributions, reduction='none').mean(axis=2)) #[bs, n_patches, k] -> #[bs, n_patches]
                # kl_div_distance = (kl_div_distance - kl_div_distance.min(dim=1, keepdim=True)[0])/(kl_div_distance.max(dim=1, keepdim=True)[0] - kl_div_distance.min(dim=1, keepdim=True)[0])
                # patched_diff_change_residuals_distribution = kl_div_distance.view(bs,h,w)

                # l1 region-wise inference raw features
                l1_patched_diff_change_features = torch.abs((patched_padded_output1_distributions - patched_padded_output2_distributions).sum(axis=2)).view(bs, h, w)
                l1_dist_change_feats_pred = torch.zeros_like(l1_patched_diff_change_features)
                l1_inference_otsu_threshold = inference_otsu_coeff * otsu(l1_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l1_dist_change_feats_pred[l1_patched_diff_change_features > l1_inference_otsu_threshold] = 1
                l1_dist_change_feats_pred = l1_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # l2 region-wise inference raw features
                l2_patched_diff_change_features = torch.sqrt(torch.pow(patched_padded_output1_distributions - patched_padded_output2_distributions, 2).sum(axis=2)).view(bs, h, w)
                l2_dist_change_feats_pred = torch.zeros_like(l2_patched_diff_change_features)
                l2_inference_otsu_threshold = inference_otsu_coeff * otsu(l2_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l2_dist_change_feats_pred[l2_patched_diff_change_features > l2_inference_otsu_threshold] = 1
                l2_dist_change_feats_pred = l2_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # diff_change_features = torch.sqrt(torch.pow(output1 - output2, 2).sum(axis=1))#.view(bs,h,w)
                # inference_otsu_threshold = inference_otsu_coeff*otsu(histograms_intersection.cpu().detach().numpy(), nbins=40)
                # diff_change_thresholded = torch.zeros_like(histograms_intersection)
                # diff_change_thresholded[histograms_intersection < inference_otsu_threshold] = 1
                # pred1 = masks1.max(1)[1].cpu().detach()  # .numpy()
                # pred2 = masks2.max(1)[1].cpu().detach()  # .numpy()
                # change_detection_prediction = l2_dist_change_feats_pred

                total_loss += loss.item()
                mask1[mask1 == -1] = 0
                preds.append(l2_dist_change_feats_pred)
                targets.append(mask1.cpu())  # .numpy())

                histogram_distance.append(histc_int_change_feats_pred)
                l1_dist.append(l1_dist_change_feats_pred)
                l2_dist.append(l2_dist_change_feats_pred)

        if 'mask1' in vars(__builtins__):
            mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=2)

            hist_mean_iou, hist_precision, hist_recall = eval_utils.compute_jaccard(histogram_distance, targets, num_classes=2)
            l1_mean_iou, l1_precision, l1_recall = eval_utils.compute_jaccard(l1_dist, targets, num_classes=2)
            l2_mean_iou, l2_precision, l2_recall = eval_utils.compute_jaccard(l2_dist, targets, num_classes=2)

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

            mean_precision = precision.mean()
            mean_recall = recall.mean()
            overall_miou = mean_iou.mean()
            classwise_f1_score = 2 * (precision * recall) / (precision + recall)
            mean_f1_score = classwise_f1_score.mean()

            # overall_miou = sum(mean_iou)/len(mean_iou)
            # print(f"Training class-wise mIoU value: \n{mean_iou} \noverall mIoU: {overall_miou}")
            # print(f"Training class-wise Precision value: \n{precision} \noverall Precision: {mean_precision}")
            # print(f"Training class-wise Recall value: \n{recall} \noverall Recall: {mean_recall}")
            # print(f"Training overall F1 Score: {mean_f1_score}")

            cometml_experiemnt.log_metric("Training Loss", total_loss, epoch=epoch + 1)
            cometml_experiemnt.log_metric("Segmentation Loss", total_loss_seg, epoch=epoch + 1)
            cometml_experiemnt.log_metric("Training mIoU", overall_miou, epoch=epoch + 1)
            cometml_experiemnt.log_metric("Training mean_f1_score", mean_f1_score, epoch=epoch + 1)

            # cometml_experiemnt.log_metrics({f"Training Recall class {str(x)}": recall[x] for x in range(len(recall))}, epoch=epoch+1)
            # cometml_experiemnt.log_metrics({f"Training Precision class {str(x)}": precision[x] for x in range(len(precision))}, epoch=epoch+1)
            # cometml_experiemnt.log_metrics({f"Training F1_score class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch+1)

            cometml_experiemnt.log_metrics({f"L1 Training Recall class {str(x)}": l1_recall[x] for x in range(len(l1_recall))}, epoch=epoch + 1)
            cometml_experiemnt.log_metrics({f"L1 Training Precision class {str(x)}": l1_precision[x] for x in range(len(l1_precision))}, epoch=epoch + 1)
            cometml_experiemnt.log_metrics({f"L1 Training F1_score class {str(x)}": l1_f1[x] for x in range(len(l1_f1))}, epoch=epoch + 1)

            cometml_experiemnt.log_metrics({f"L2 Training Recall class {str(x)}": l2_recall[x] for x in range(len(l2_recall))}, epoch=epoch + 1)
            cometml_experiemnt.log_metrics({f"L2 Training Precision class {str(x)}": l2_precision[x] for x in range(len(l2_precision))}, epoch=epoch + 1)
            cometml_experiemnt.log_metrics({f"L2 Training F1_score class {str(x)}": l2_f1[x] for x in range(len(l2_f1))}, epoch=epoch + 1)

            cometml_experiemnt.log_metrics({f"Histogram Distance Training Recall class {str(x)}": hist_recall[x] for x in range(len(hist_recall))}, epoch=epoch + 1)
            cometml_experiemnt.log_metrics({f"Histogram Distance Training Precision class {str(x)}": hist_precision[x] for x in range(len(hist_precision))}, epoch=epoch + 1)
            cometml_experiemnt.log_metrics({f"Histogram Distance Training F1_score class {str(x)}": hist_f1[x] for x in range(len(hist_f1))}, epoch=epoch + 1)

        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch + 1, total_loss / self.train_loader.__len__()))

        return total_loss / self.train_loader.__len__()

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
                if self.test_with_full_supervision == 1:
                    images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                    # original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
                    # print(outputs['tr'].data[0][batch_index_to_show])
                    # print(outputs['tr'].data[0][batch_index_to_show]['slices'])
                    image_name = f"{str(outputs['tr'].data[0][batch_index_to_show]['gids'])}_{str(outputs['tr'].data[0][batch_index_to_show]['slices'])}"

                    mask = torch.stack(mask)
                    mask = mask.long().squeeze(1)

                    bs, c, t, h, w = images.shape
                    image1 = images[:, :, 0, :, :]
                    image2 = images[:, :, 1, :, :]
                    mask1 = mask[:, 0, :, :]
                    mask2 = mask[:, 1, :, :]

                    images = images.to(device)
                    image1 = image1.to(device)
                    image2 = image2.to(device)
                    mask = mask.to(device)
                else:
                    print("To validate, you need to use a dataset that is fully annotated.")
                    exit()

                image1 = utils.stad_image(image1)
                image2 = utils.stad_image(image2)

                # sampled_crops = random.sample(self.all_crops_params, 1000)
                pad_amount = (config['evaluation']['inference_window'] - 1) // 2
                padded_image1 = F.pad(input=image1, pad=(pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')
                padded_image2 = F.pad(input=image2, pad=(pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')

                start = time.time()
                patched_image1 = torch.stack([transforms.functional.crop(padded_image1, *params) for params in self.inference_all_crops_params], dim=1)
                patched_image2 = torch.stack([transforms.functional.crop(padded_image2, *params) for params in self.inference_all_crops_params], dim=1)

                bs, ps, c, ph, pw = patched_image1.shape
                patched_image1 = patched_image1.view(bs * ps, c, ph, pw)
                patched_image2 = patched_image2.view(bs * ps, c, ph, pw)

                crop_collection_time = time.time() - start

                start = time.time()

                output1 = self.model(patched_image1, image1, self.inference_all_crops_params)  ## [B,22,150,150]
                output2 = self.model(patched_image2, image2, self.inference_all_crops_params)  ## [B,22,150,150]

                # output1, quant1 = self.model(patched_image1)  # [B,22,150,150]
                # output2, quant2 = self.model(patched_image2)

                network_run_time = time.time() - start

                start = time.time()
                output1 = torch.stack(torch.chunk(output1, chunks=bs, dim=0), dim=0)
                output2 = torch.stack(torch.chunk(output2, chunks=bs, dim=0), dim=0)

                # quant1 = torch.stack(torch.chunk(quant1, chunks=bs, dim=0), dim=0)
                # quant2 = torch.stack(torch.chunk(quant2, chunks=bs, dim=0), dim=0)

                chunking_time = time.time() - start
                # output1 = F.normalize(output1, dim=1, p=1) #[bs*ps, c*h*w]
                # output2 = F.normalize(output2, dim=1, p=1)
                masks1 = F.softmax(output1, dim=2)  # .detach()
                masks2 = F.softmax(output2, dim=2)  # .detach()

                #region-wise inference
                inference_otsu_coeff = 1.0
                hist_inference_otsu_coeff = 1.0
                topk = 33

                # padded_output1 = F.pad(input=output1, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                # padded_output2 = F.pad(input=output2, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                # patched_padded_output1 = torch.stack([transforms.functional.crop(padded_output1, *params) for params in self.inference_all_crops_params], dim=1)#.flatten(-3,-1)
                # patched_padded_output2 = torch.stack([transforms.functional.crop(padded_output2, *params) for params in self.inference_all_crops_params], dim=1)#.flatten(-3,-1)

                # padded_mask1 = F.pad(input=masks1, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                # padded_mask2 = F.pad(input=masks2, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                # patched_padded_mask1 = torch.stack([transforms.functional.crop(padded_mask1, *params) for params in self.inference_all_crops_params], dim=1)#.flatten(-3,-1)
                # patched_padded_mask2 = torch.stack([transforms.functional.crop(padded_mask2, *params) for params in self.inference_all_crops_params], dim=1)#.flatten(-3,-1)

                # patched_padded_output1_distributions = patched_padded_output1.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]
                # patched_padded_output2_distributions = patched_padded_output2.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]
                # patched_padded_mask1_distributions = patched_padded_mask1.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]
                # patched_padded_mask2_distributions = patched_padded_mask2.flatten(-2, -1)#.sum(axis=3) #[bs, n_patches, k]

                # patched_padded_output1_distributions = patched_padded_output1_distributions.sum(axis=3)
                # patched_padded_output2_distributions = patched_padded_output2_distributions.sum(axis=3)
                # patched_padded_mask1_distributions = patched_padded_mask1_distributions.sum(axis=3) #[bs, n_patches, k]
                # patched_padded_mask2_distributions = patched_padded_mask2_distributions.sum(axis=3) #[bs, n_patches, k]

                # topk_patched_output1_post_distributions, largest_elements_post_inds = torch.topk(masks1, k=topk, sorted=False, dim=2)
                # topk_patched_output2_post_distributions = torch.gather(masks2, dim=2, index=largest_elements_post_inds)

                # normalized_patched_padded_output1_distributions = (patched_padded_output1_distributions - patched_padded_output1_distributions.min(dim=2, keepdim=True)[0])/(patched_padded_output1_distributions.max(dim=2, keepdim=True)[0] - patched_padded_output1_distributions.min(dim=2, keepdim=True)[0])
                # normalized_patched_padded_output2_distributions = (patched_padded_output2_distributions - patched_padded_output2_distributions.min(dim=2, keepdim=True)[0])/(patched_padded_output2_distributions.max(dim=2, keepdim=True)[0] - patched_padded_output2_distributions.min(dim=2, keepdim=True)[0])

                # histogram intersection raw features
                minima = torch.minimum(masks1, masks2)
                histograms_intersection_features = torch.true_divide(minima.sum(axis=2), masks2.sum(axis=2)).view(bs, h, w)
                histograms_intersection_features = FT.gaussian_blur(histograms_intersection_features, kernel_size=config['evaluation']['inference_window'])
                histc_int_change_feats_pred = torch.zeros_like(histograms_intersection_features)
                histc_int_inference_otsu_threshold = hist_inference_otsu_coeff * otsu(histograms_intersection_features.cpu().detach().numpy(), nbins=256)
                histc_int_change_feats_pred[histograms_intersection_features < histc_int_inference_otsu_threshold] = 1
                histc_int_change_feats_pred = histc_int_change_feats_pred.cpu().detach().type(torch.uint8)

                # histograms_intersection_features = torch.sqrt(torch.pow(quant1 - quant2, 2).sum(axis=2)).view(bs,h,w)
                # histograms_intersection_features = FT.gaussian_blur(histograms_intersection_features, kernel_size=config['evaluation']['inference_window'])
                # histc_int_change_feats_pred = torch.zeros_like(histograms_intersection_features)
                # histc_int_inference_otsu_threshold = inference_otsu_coeff*otsu(histograms_intersection_features.cpu().detach().numpy(), nbins=256)
                # histc_int_change_feats_pred[histograms_intersection_features > histc_int_inference_otsu_threshold] = 1
                # histc_int_change_feats_pred = histc_int_change_feats_pred.cpu().detach().type(torch.uint8)

                # l1 region-wise inference raw features
                l1_patched_diff_change_features = torch.abs((output1 - output2).sum(axis=2)).view(bs, h, w)
                l1_patched_diff_change_features = FT.gaussian_blur(l1_patched_diff_change_features, kernel_size=config['evaluation']['inference_window'])
                l1_dist_change_feats_pred = torch.zeros_like(l1_patched_diff_change_features)
                l1_inference_otsu_threshold = inference_otsu_coeff * otsu(l1_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l1_dist_change_feats_pred[l1_patched_diff_change_features > l1_inference_otsu_threshold] = 1
                l1_dist_change_feats_pred = l1_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # l2 region-wise inference raw features
                l2_patched_diff_change_features = torch.sqrt(torch.pow(output1 - output2, 2).sum(axis=2)).view(bs, h, w)
                l2_patched_diff_change_features = FT.gaussian_blur(l2_patched_diff_change_features, kernel_size=config['evaluation']['inference_window'])
                l2_dist_change_feats_pred = torch.zeros_like(l2_patched_diff_change_features)
                l2_inference_otsu_threshold = inference_otsu_coeff * otsu(l2_patched_diff_change_features.cpu().detach().numpy(), nbins=256)
                l2_dist_change_feats_pred[l2_patched_diff_change_features > l2_inference_otsu_threshold] = 1
                l2_dist_change_feats_pred = l2_dist_change_feats_pred.cpu().detach().type(torch.uint8)

                # patched_diff_change_features = torch.sqrt(torch.pow(patched_padded_output1_distributions - patched_padded_output2_distributions, 2).sum(axis=2)).view(bs,h,w)

                # diff_change_features = torch.abs(masks1-masks2).sum(axis=1)
                # # diff_change_features = torch.sqrt(torch.pow(output1 - output2, 2).sum(axis=1))#.view(bs,h,w)
                # inference_otsu_threshold = inference_otsu_coeff*otsu(patched_diff_change_features.cpu().detach().numpy(), nbins=512)
                # diff_change_thresholded = torch.zeros_like(diff_change_features)
                # diff_change_thresholded[patched_diff_change_features > inference_otsu_threshold] = 1

                # pred1 = masks1.max(1)[1].cpu().detach()  # .numpy()
                # pred2 = masks2.max(1)[1].cpu().detach()  # .numpy()
                # change_detection_prediction = diff_change_thresholded.cpu().detach().type(torch.uint8)
                # change_detection_prediction = (pred1!=pred2).type(torch.uint8)
                histogram_distance.append(histc_int_change_feats_pred)
                l1_dist.append(l1_dist_change_feats_pred)
                l2_dist.append(l2_dist_change_feats_pred)

                preds.append(l2_dist_change_feats_pred)
                mask1[mask1 == -1] = 0
                targets.append(mask1.cpu())  # .numpy())

                pbar.set_description(f"(timing, secs) crop_collection: {crop_collection_time:0.3f}, network_run: {network_run_time:0.3f}, chunking_time: {chunking_time:0.3f}")

                if config['visualization']['val_visualizer'] or (config['visualization']['save_individual_plots'] and save_individual_plots_specific):
                    if (epoch) % config['visualization']['visualize_val_every'] == 0:
                        if (batch_index % iter_visualization) == 0:
                            figure = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']),
                                                dpi=config['visualization']['dpi'])
                            ax1 = figure.add_subplot(3, 3, 1)
                            ax2 = figure.add_subplot(3, 3, 2)
                            ax3 = figure.add_subplot(3, 3, 3)
                            ax4 = figure.add_subplot(3, 3, 4)
                            ax5 = figure.add_subplot(3, 3, 5)
                            ax6 = figure.add_subplot(3, 3, 6)
                            ax7 = figure.add_subplot(3, 3, 7)
                            ax8 = figure.add_subplot(3, 3, 8)
                            ax9 = figure.add_subplot(3, 3, 9)
                            # ax10 = figure.add_subplot(3, 4, 10)
                            # ax11 = figure.add_subplot(3, 4, 11)
                            # ax12 = figure.add_subplot(3, 4, 12)

                            cmap_gradients = plt.cm.get_cmap('jet')
                            # image_show = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:1:4,:3]
                            image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                            image_show1 = np.flip(image_show1, axis=2)

                            image_show2 = np.transpose(image2.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                            image_show2 = np.flip(image_show2, axis=2)

                            image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())
                            image_show2 = (image_show2 - image_show2.min()) / (image_show2.max() - image_show2.min())
                            gt_mask_show1 = mask1.cpu().detach()[batch_index_to_show, :, :].numpy().squeeze()

                            l1_dist_change_feats_pred_show = l1_dist_change_feats_pred.numpy()[batch_index_to_show, :, :]
                            l2_dist_change_feats_pred_show = l2_dist_change_feats_pred.numpy()[batch_index_to_show, :, :]
                            histc_int_change_feats_pred_show = histc_int_change_feats_pred.numpy()[batch_index_to_show, :, :]

                            l1_patched_diff_change_features_show = l1_patched_diff_change_features.cpu().detach().numpy()[batch_index_to_show, :, :]
                            l2_patched_diff_change_features_show = l2_patched_diff_change_features.cpu().detach().numpy()[batch_index_to_show, :, :]
                            histograms_intersection_show = histograms_intersection_features.cpu().detach().numpy()[batch_index_to_show, :, :]

                            l1_patched_diff_change_features_show = (l1_patched_diff_change_features_show - l1_patched_diff_change_features_show.min()) / (l1_patched_diff_change_features_show.max() - l1_patched_diff_change_features_show.min())
                            l2_patched_diff_change_features_show = (l2_patched_diff_change_features_show - l2_patched_diff_change_features_show.min()) / (l2_patched_diff_change_features_show.max() - l2_patched_diff_change_features_show.min())
                            histograms_intersection_show = (histograms_intersection_show - histograms_intersection_show.min()) / (histograms_intersection_show.max() - histograms_intersection_show.min())

                            # pred1_show = masks1.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]
                            # pred2_show = masks2.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]

                            l1_fp_tp_fn_prediction_mask = gt_mask_show1 + (2 * l1_dist_change_feats_pred_show)
                            l2_fp_tp_fn_prediction_mask = gt_mask_show1 + (2 * l2_dist_change_feats_pred_show)
                            histc_fp_tp_fn_prediction_mask = gt_mask_show1 + (2 * histc_int_change_feats_pred_show)

                            # vca_pseudomask_show = image_change_magnitude_binary.cpu().detach()[batch_index_to_show, :, :].numpy()
                            # vca_pseudomask_crop_show = cm_binary_crop.cpu().detach()[batch_index_to_show,:,:].numpy()
                            # dictionary_show = dictionary1.cpu().detach()[batch_index_to_show,:,:].numpy()
                            # dictionary_show = dictionary2_post_assignment.cpu().detach()[batch_index_to_show, :, :].numpy()
                            # dictionary2_show = dictionary1_post_assignment.cpu().detach()[batch_index_to_show, :, :].numpy()
                            # print(np.unique())
                            classes_in_gt = np.unique(gt_mask_show1)
                            ax1.imshow(image_show1)

                            ax2.imshow(image_show2)

                            ax3.imshow(image_show1)
                            ax3.imshow(gt_mask_show1, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                            ax4.imshow(l1_patched_diff_change_features_show, cmap=cmap_gradients)

                            ax5.imshow(l2_patched_diff_change_features_show, cmap=cmap_gradients)

                            ax6.imshow(histograms_intersection_show, cmap=cmap_gradients)

                            # ax8.imshow(dictionary2_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            ax7.imshow(l1_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                            ax8.imshow(l2_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                            ax9.imshow(histc_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=config['visualization']['fg_alpha'])

                            # ax12.imshow(vw_dis_fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            ax1.axis('off')
                            ax2.axis('off')
                            ax3.axis('off')
                            ax4.axis('off')
                            ax5.axis('off')
                            ax6.axis('off')
                            ax7.axis('off')
                            ax8.axis('off')
                            # ax9.axis('off')

                            if config['visualization']['titles']:
                                ax1.set_title(f"Input Image 1", fontsize=config['visualization']['font_size'])
                                ax2.set_title(f"Input Image 2", fontsize=config['visualization']['font_size'])
                                ax3.set_title(f"Change GT Mask overlaid", fontsize=config['visualization']['font_size'])
                                ax4.set_title(f"l1_patched_diff_change_features_show", fontsize=config['visualization']['font_size'])
                                ax5.set_title(f"l2_patched_diff_change_features_show", fontsize=config['visualization']['font_size'])
                                ax6.set_title(f"histograms_intersection_show", fontsize=config['visualization']['font_size'])
                                ax7.set_title(f"l1_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                ax8.set_title(f"l2_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                ax9.set_title(f"histc_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                # figure.suptitle(
                                #     f"Epoch: {epoch+1}\nGT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])

                            figure.tight_layout()
                            if config['visualization']['val_imshow']:
                                plt.show()

                            cometml_experiemnt.log_figure(figure_name=f"Validation, Image name: {image_name}", figure=figure)
                            figure.clear()
                            plt.cla()
                            plt.clf()
                            plt.close('all')
                            plt.close(figure)
                            gc.collect()

        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=2)

        hist_mean_iou, hist_precision, hist_recall = eval_utils.compute_jaccard(histogram_distance, targets, num_classes=2)
        l1_mean_iou, l1_precision, l1_recall = eval_utils.compute_jaccard(l1_dist, targets, num_classes=2)
        l2_mean_iou, l2_precision, l2_recall = eval_utils.compute_jaccard(l2_dist, targets, num_classes=2)

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

        mean_precision = precision.mean()
        mean_recall = recall.mean()
        overall_miou = mean_iou.mean()
        classwise_f1_score = 2 * (precision * recall) / (precision + recall)
        mean_f1_score = classwise_f1_score.mean()

        # print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU", overall_miou, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation precision", mean_precision, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation recall", mean_recall, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation mean f1_score", mean_f1_score, epoch=epoch + 1)
        print({f"Recall class {str(x)}": recall[x] for x in range(len(recall))})
        print({f"Precision class {str(x)}": precision[x] for x in range(len(precision))})
        print({f"F1 class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))})

        # cometml_experiemnt.log_metrics({f"Recall class {str(x)}": recall[x] for x in range(len(recall))}, epoch=epoch+1)
        # cometml_experiemnt.log_metrics({f"Precision class {str(x)}": precision[x] for x in range(len(precision))}, epoch=epoch+1)
        # cometml_experiemnt.log_metrics({f"F1_score class {str(x)}": classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"L1 Training Recall class {str(x)}": l1_recall[x] for x in range(len(l1_recall))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"L1 Training Precision class {str(x)}": l1_precision[x] for x in range(len(l1_precision))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"L1 Training F1_score class {str(x)}": l1_f1[x] for x in range(len(l1_f1))}, epoch=epoch + 1)

        cometml_experiemnt.log_metrics({f"L2 Training Recall class {str(x)}": l2_recall[x] for x in range(len(l2_recall))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"L2 Training Precision class {str(x)}": l2_precision[x] for x in range(len(l2_precision))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"L2 Training F1_score class {str(x)}": l2_f1[x] for x in range(len(l2_f1))}, epoch=epoch + 1)

        cometml_experiemnt.log_metrics({f"Histogram Distance Training Recall class {str(x)}": hist_recall[x] for x in range(len(hist_recall))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"Histogram Distance Training Precision class {str(x)}": hist_precision[x] for x in range(len(hist_precision))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"Histogram Distance Training F1_score class {str(x)}": hist_f1[x] for x in range(len(hist_f1))}, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation Average Loss", total_loss / loader.__len__(), epoch=epoch + 1)

        return total_loss / loader.__len__(), classwise_f1_score

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
        best_val_loss, best_train_loss = np.infty, np.infty
        best_val_mean_f1, val_mean_f1 = 0, 0
        best_val_change_f1, val_change_f1 = 0, 0

        model_save_dir = config['data'][config['location']]['model_save_dir'] + f"{current_path[-1]}_{config['dataset']}/{cometml_experiment.project_name}_trainWin_{config['data']['window_size']}_modelName_{config['training']['model_name']}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        for epoch in range(0, self.epochs):
            if config['procedures']['train']:
                with cometml_experiment.train():
                    train_loss = self.train(epoch, cometml_experiment)
            if config['procedures']['validate']:
                with cometml_experiment.validate():
                    val_loss, val_cw_f1 = self.validate(epoch, cometml_experiment)
                    val_mean_f1 = val_cw_f1.mean()
                    val_change_f1 = val_cw_f1[1]
            self.scheduler.step()

            if (train_loss <= best_train_loss) or (val_mean_f1 >= best_val_mean_f1) or (val_change_f1 >= best_val_change_f1):

                if train_loss <= best_train_loss:
                    best_train_loss = train_loss
                if val_mean_f1 >= best_val_mean_f1:
                    best_val_mean_f1 = val_mean_f1
                if val_change_f1 >= best_val_change_f1:
                    best_val_change_f1 = val_change_f1

                model_save_name = f"{current_path[-1]}_epoch_{epoch}_loss_{train_loss}_valmF1_{val_mean_f1}_valChangeF1_{val_change_f1}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"

                if config['procedures']['train']:
                    with open(model_save_dir + "config.yaml", 'w') as file:
                        yaml.dump(config, file)

                    torch.save({'epoch': epoch,
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'loss': train_loss},
                                model_save_dir + model_save_name)
                else:
                    exit()
                if config['visualization']['save_individual_plots']:
                    _, _ = self.validate(epoch, cometml_experiment, save_individual_plots_specific=True)

        return train_losses, val_losses, mean_ious_val


if __name__ == "__main__":

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
                                     display_summary_level=0,
                                     auto_histogram_gradient_logging=True,
                                     auto_histogram_weight_logging=True,
                                     auto_histogram_activation_logging=True)

    config['experiment_url'] = str(experiment.url)

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
        config['data']['num_classes'] = pretrain_config['data']['num_classes']
        config['training']['model_feats_channels'] = pretrain_config['training']['model_feats_channels']

    # window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
    # input_dims = (config['data']['image_size'], config['data']['image_size'])

    # # channels = 'B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A'
    # channels = config['data']['channels']
    # num_channels = len(channels.split('|'))
    # config['training']['num_channels'] = num_channels
    # dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    # train_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])

    if config['data']['name'] == 'watch' or config['data']['name'] == 'onera':
        coco_fpath = ub.expandpath(config['data'][config['location']]['train_coco_json'])
        dset = kwcoco.CocoDataset(coco_fpath)
        sampler = ndsampler.CocoSampler(dset)

        window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
        input_dims = (config['data']['image_size'], config['data']['image_size'])

        # test_window_dims = (config['data']['time_steps'], config['evaluation']['image_size'], config['evaluation']['image_size'])  # [t,h,w]
        # test_input_dims = (config['evaluation']['image_size'], config['evaluation']['image_size'])

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

        channels = config['data']['channels']
        num_channels = len(channels.split('|'))
        config['training']['num_channels'] = num_channels
        window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
        input_dims = (config['data']['image_size'], config['data']['image_size'])

        test_coco_fpath = ub.expandpath(config['data'][config['location']]['test_coco_json'])
        test_dset = kwcoco.CocoDataset(test_coco_fpath)
        test_sampler = ndsampler.CocoSampler(test_dset)

        test_dataset = SequenceDataset(test_sampler, window_dims, input_dims, channels)
        test_dataloader = test_dataset.make_loader(batch_size=config['evaluation']['batch_size'])

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

    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False

    optimizer = optim.SGD(model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader),
                                                     eta_min=config['training']['learning_rate'])

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"loaded model from {config['training']['resume']}")
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
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
