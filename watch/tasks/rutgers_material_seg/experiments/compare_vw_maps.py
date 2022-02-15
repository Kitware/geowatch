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
from watch.tasks.rutgers_material_seg.models.quantizer import Quantizer
from PIL import Image, ImageEnhance

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)


class Trainer(object):
    def __init__(self, ours_model: object, baseline_model: object, train_loader: torch.utils.data.DataLoader,
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

        self.ours_model = ours_model
        self.baseline_model = baseline_model
        self.use_crf = config['evaluation']['use_crf']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.k = config['data']['num_classes']
        self.kmeans = KMeans(n_clusters=self.k, mode='euclidean', verbose=0, minibatch=None)
        self.kmeans_features = KMeans(n_clusters=self.k, mode='euclidean', verbose=0, minibatch=None)
        self.kmeans_resnet_features = KMeans(n_clusters=self.k, mode='euclidean', verbose=0, minibatch=None)
        self._aff = TeRN(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
        self.max_label = config['data']['num_classes']
        # self.all_crops_params = [tuple([i,j,config['data']['window_size'], config['data']['window_size']]) for i in range(config['data']['window_size'],config['data']['image_size']-config['data']['window_size']) for j in range(config['data']['window_size'],config['data']['image_size']-config['data']['window_size'])]
        self.all_crops_params = [tuple([i,j,config['data']['window_size'], config['data']['window_size']]) for i in range(0,config['data']['image_size']) for j in range(0,config['data']['image_size'])]
        self.inference_all_crops_params = [tuple([i, j, config['evaluation']['inference_window'], config['evaluation']['inference_window']]) for i in range(0, config['data']['image_size']) for j in range(0, config['data']['image_size'])]
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision

        self.crop_size = (config['data']['window_size'], config['data']['window_size'])

        self.cmap = visualization.n_distinguishable_colors(nlabels=self.max_label,
                                                           first_color_black=True, last_color_black=True,
                                                           bg_alpha=config['visualization']['bg_alpha'],
                                                           fg_alpha=config['visualization']['fg_alpha'])


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
        self.ours_model.eval()
        self.baseline_model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total=len(loader))
            for batch_index, batch in pbar:
                outputs = batch
                if self.test_with_full_supervision == 1:
                    images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                    # original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
                    # print(outputs['tr'].data[0][batch_index_to_show])
                    # print(outputs['tr'].data[0][batch_index_to_show]['slices'])
                    
                    mask = torch.stack(mask)
                    mask = mask.long().squeeze(1)

                    bs, c, t, h, w = images.shape
                    image1 = images[:, :, 0, :, :]
                    image2 = images[:, :, 1, :, :]
                    mask1 = mask[:, 0, :, :]

                    image1 = image1.to(device)
                    image2 = image2.to(device)
                    mask = mask.to(device)
                else:
                    print("To validate, you need to use a dataset that is fully annotated.")
                    exit()

                image1 = utils.stad_image(image1)
                image2 = utils.stad_image(image2)
                # image1 = F.normalize(image1, dim=1, p=2) 
                # sampled_crops = random.sample(self.all_crops_params, 1000)
                pad_amount = (config['evaluation']['inference_window']-1)//2
                padded_image1 = F.pad(input=image1, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')
                padded_image2 = F.pad(input=image2, pad=(pad_amount,pad_amount,pad_amount,pad_amount), mode='replicate')

                start = time.time()
                patched_image1 = torch.stack([transforms.functional.crop(padded_image1, *params) for params in self.inference_all_crops_params],dim=1)
                patched_image2 = torch.stack([transforms.functional.crop(padded_image2, *params) for params in self.inference_all_crops_params],dim=1)

                bs, ps, c, ph, pw = patched_image1.shape
                patched_image1 = patched_image1.view(bs*ps,c,ph,pw)
                patched_image2 = patched_image2.view(bs*ps,c,ph,pw)
                
                crop_collection_time = time.time() - start

                start = time.time()

                output1 = self.ours_model(patched_image1, image1, self.inference_all_crops_params)  ## [B,22,150,150]
                output2 = self.ours_model(patched_image2, image2, self.inference_all_crops_params)  ## [B,22,150,150]
                resnet_output1 = self.baseline_model(patched_image1)  ## [B,22,150,150]
                resnet_output2 = self.baseline_model(patched_image2)  ## [B,22,150,150]

                # output1 = self.model(patched_image1)  ## [B,22,150,150]
                # print(f"resnet_output1: {resnet_output1.shape}")
                # print(output1.shape)
                patched_image1 = torch.flatten(patched_image1, start_dim=1, end_dim=3)
                patched_image2 = torch.flatten(patched_image2, start_dim=1, end_dim=3)
                # print(f"patched_image: {patched_image1.shape}")

                # output1 = F.normalize(output1, dim=1, p=1)
                # output2 = F.normalize(output2, dim=1, p=1)
                # resnet_output1 = F.normalize(resnet_output1, dim=1, p=1)
                # resnet_output2 = F.normalize(resnet_output2, dim=1, p=1)
                # patched_image1 = F.normalize(patched_image1, dim=1, p=1)
                # patched_image2 = F.normalize(patched_image2, dim=1, p=1)

                dictionary_feats = self.kmeans_features.fit_predict(output1)
                centroids_feats = self.kmeans_features.centroids.T
                residuals_feats = torch.cdist(output1, centroids_feats.T, p=1)

                # print(f"output1.shape: {output1.shape}")
                # print(f"centroids_feats.shape: {centroids_feats.shape}")
                # print(f"residuals_feats.shape: {residuals_feats.shape}")
                # print(f"centroids_feats: {centroids_feats[0]}")
                # print(f"output1: {output1[0]}")
                # print(f"residuals_feats: {residuals_feats[0]}")
                # print(residuals_feats)
                # print(residuals_feats.shape)
                # print(f"residuals_feats min: {residuals_feats.min()}, max: {residuals_feats.max()}")
                # print(f"residuals_feats min: {residuals_feats.min(dim=1)}, max: {residuals_feats.max(dim=1)}")

                dictionary_feats2 = self.kmeans_features.predict(output2)
                centroids_feats2 = self.kmeans_features.centroids.T
                residuals_feats2 = torch.cdist(output2, centroids_feats2.T, p=1)
                
                dictionary_image = self.kmeans.fit_predict(patched_image1)
                centroids_image = self.kmeans.centroids.T
                residuals_image = torch.cdist(patched_image1, centroids_image.T, p=2)

                dictionary_image2 = self.kmeans.predict(patched_image2)
                centroids_image2 = self.kmeans.centroids.T
                residuals_image2 = torch.cdist(patched_image2, centroids_image2.T, p=2)

                dictionary_baseline_feats = self.kmeans_resnet_features.fit_predict(resnet_output1)
                centroids_baseline_feats = self.kmeans_resnet_features.centroids.T
                residuals_baseline_feats = torch.cdist(resnet_output1, centroids_baseline_feats.T, p=2)

                dictionary_baseline_feats2 = self.kmeans_resnet_features.predict(resnet_output2)
                centroids_baseline_feats2 = self.kmeans_resnet_features.centroids.T
                residuals_baseline_feats2 = torch.cdist(resnet_output2, centroids_baseline_feats2.T, p=2)

                # quant =  torch.tensor([torch.histc(residuals_feats[i], bins=self.k).tolist() for i in range(bs)], device=torch.device('cuda'))

                if config['visualization']['log_scatters']:
                    min_distances, min_indices = residuals_feats.min(dim=1)
                    # print(min_distances)
                    # print(min_indices)
                    # print(min_distances.shape)
                    min_dist_mean = min_distances.mean()


                    perplexity = 100.0
                    early_exaggeration = 12.0
                    steps = 5000
                    from sklearn.manifold import TSNE
                    tsne_model = TSNE(n_components=2, verbose=False, random_state=0, n_iter=steps,
                                      perplexity=perplexity, early_exaggeration=early_exaggeration, n_jobs=8
                                    #   method='exact'
                                      )

                    # from fastTSNE import TSNE
                    # tsne_model = TSNE(n_components=2, negative_gradient_method='fft', neighbors='approx', n_jobs=8)
                    # clustering_model = KMeans(n_clusters=self.k)
                    # clustering_model_raw = KMeans(n_clusters=self.k)

                    # scatter_fig = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']))
                    # scax1 = scatter_fig.add_subplot(1, 3, 1)
                    # scax2 = scatter_fig.add_subplot(1, 3, 2)
                    # scax3 = scatter_fig.add_subplot(1, 3, 3)
                    num_sampled = 6000
                    perm = torch.randperm(patched_image1.size(0))
                    idx = min_distances<min_dist_mean*0.6#).nonzero()

                    # idx = idx[:num_sampled]
                    # idx = perm[:num_sampled]
                    # print(output1.shape)
                    image_samples = patched_image1[idx]
                    resnet_samples = resnet_output1[idx]
                    ours_samples = output1[idx]
                    
                    image_samples = F.normalize(image_samples, dim=1, p=1) 
                    resnet_samples = F.normalize(resnet_samples, dim=1, p=1) 
                    ours_samples = F.normalize(ours_samples, dim=1, p=1) 
                    # ours_samples = (ours_samples - ours_samples.min()) / (ours_samples.max() - ours_samples.min())

                    # print(ours_samples.shape)
                    ours_scatter_dict = dictionary_feats.cpu().detach()[idx]
                    image_scatter_dict = dictionary_image.cpu().detach()[idx]
                    resnet_scatter_dict = dictionary_baseline_feats.cpu().detach()[idx]
                    # image_tsne_feats = tsne_model.fit_transform(patched_image1.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    # resnet_tsne_feats = tsne_model.fit_transform(resnet_output1.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    # ours_tsne_feats = tsne_model.fit_transform(output1.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]

                    ours_tsne_feats = tsne_model.fit_transform(ours_samples.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    image_tsne_feats = tsne_model.fit_transform(image_samples.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    resnet_tsne_feats = tsne_model.fit_transform(resnet_samples.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]

                    # ours_tsne_feats = tsne_model.fit(ours_samples.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    # image_tsne_feats = tsne_model.fit(image_samples.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    # resnet_tsne_feats = tsne_model.fit(resnet_samples.detach().cpu())  # [num_pixels, 507] -> [num_pixels, 2]
                    
                    # scax1.scatter(image_tsne_feats[:, 0], image_tsne_feats[:, 1],
                    #                 c=dictionary_image.cpu().detach()[idx],
                    #                 s=15.5,
                    #                 cmap='tab20',
                    #                 marker='.',
                    #                 linewidths=1.0
                    #                 )
                    
                    # scax2.scatter(resnet_tsne_feats[:, 0], resnet_tsne_feats[:, 1],
                    #                 c=dictionary_baseline_feats.cpu().detach()[idx],
                    #                 s=15.5,
                    #                 cmap='tab20',
                    #                 marker='.',
                    #                 linewidths=1.0
                    #                 )

                    # scax3.scatter(ours_tsne_feats[:, 0], ours_tsne_feats[:, 1],
                    #                 c=dictionary_feats.cpu().detach()[idx],
                    #                 s=15.5,
                    #                 cmap='tab20',
                    #                 marker='.',
                    #                 linewidths=1.0
                    #                 )

                    # plt.show()

                network_run_time = time.time() - start

                start = time.time()
                output1 = torch.stack(torch.chunk(output1, chunks=bs, dim=0), dim=0)
                dictionary_feats = torch.stack(torch.chunk(dictionary_feats, chunks=bs, dim=0), dim=0).view(bs,h,w)
                dictionary_baseline_feats = torch.stack(torch.chunk(dictionary_baseline_feats, chunks=bs, dim=0), dim=0).view(bs,h,w)
                dictionary_image = torch.stack(torch.chunk(dictionary_image, chunks=bs, dim=0), dim=0).view(bs,h,w)

                dictionary_feats2 = torch.stack(torch.chunk(dictionary_feats2, chunks=bs, dim=0), dim=0).view(bs,h,w)
                dictionary_baseline_feats2 = torch.stack(torch.chunk(dictionary_baseline_feats2, chunks=bs, dim=0), dim=0).view(bs,h,w)
                dictionary_image2 = torch.stack(torch.chunk(dictionary_image2, chunks=bs, dim=0), dim=0).view(bs,h,w)

                # quant1 = torch.stack(torch.chunk(quant1, chunks=bs, dim=0), dim=0)
                # quant2 = torch.stack(torch.chunk(quant2, chunks=bs, dim=0), dim=0)
                



                # pbar.set_description(f"(timing, secs) crop_collection: {crop_collection_time:0.3f}, network_run: {network_run_time:0.3f}, chunking_time: {chunking_time:0.3f}")
                if config['visualization']['val_visualizer'] or (config['visualization']['save_individual_plots'] and save_individual_plots_specific):
                    for b in range(bs):
                        batch_index_to_show = b
                        image_name = f"{str(outputs['tr'].data[0][batch_index_to_show]['gids'])}_{str(outputs['tr'].data[0][batch_index_to_show]['slices'])}"
                        if (epoch) % config['visualization']['visualize_val_every'] == 0:
                            if (batch_index % iter_visualization) == 0:
                                figure = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']),
                                                    dpi=config['visualization']['dpi'])
                                ax1 = figure.add_subplot(4, 3, 1)
                                ax2 = figure.add_subplot(4, 3, 2)
                                ax3 = figure.add_subplot(4, 3, 3)
                                ax4 = figure.add_subplot(4, 3, 4)
                                ax5 = figure.add_subplot(4, 3, 5)
                                ax6 = figure.add_subplot(4, 3, 6)
                                ax7 = figure.add_subplot(4, 3, 7)
                                ax8 = figure.add_subplot(4, 3, 8)
                                ax9 = figure.add_subplot(4, 3, 9)
                                ax10 = figure.add_subplot(4, 3, 10)
                                ax11 = figure.add_subplot(4, 3, 11)
                                ax12 = figure.add_subplot(4, 3, 12)
                                # ax13 = figure.add_subplot(5, 3, 13)
                                # ax14 = figure.add_subplot(5, 3, 14)
                                # ax15 = figure.add_subplot(5, 3, 15)
                                # ax10 = figure.add_subplot(3, 4, 10)
                                # ax11 = figure.add_subplot(3, 4, 11)
                                # ax12 = figure.add_subplot(3, 4, 12)

                                cmap_gradients = plt.cm.get_cmap('jet')
                                # image_show = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:1:4,:3]
                                image1 = F.normalize(image1, dim=1, p=1) 
                                image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:80, :80, :3]
                                image_show1 = np.flip(image_show1, axis=2)

                                image2 = F.normalize(image2, dim=1, p=1) 
                                image_show2 = np.transpose(image2.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:80, :80, :3]
                                image_show2 = np.flip(image_show2, axis=2)

                                dictionary_feats_show = dictionary_feats.cpu().detach().numpy()[batch_index_to_show, :80, :80]
                                dictionary_baseline_feats_show = dictionary_baseline_feats.cpu().detach().numpy()[batch_index_to_show, :80, :80]
                                dictionary_img_show = dictionary_image.cpu().detach().numpy()[batch_index_to_show, :80, :80]

                                dictionary_feats_show2 = dictionary_feats2.cpu().detach().numpy()[batch_index_to_show, :80, :80]
                                dictionary_baseline_feats_show2 = dictionary_baseline_feats2.cpu().detach().numpy()[batch_index_to_show, :80, :80]
                                dictionary_img_show2 = dictionary_image2.cpu().detach().numpy()[batch_index_to_show, :80, :80]
                                # image_show2 = np.transpose(image2.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                                # image_show2 = np.flip(image_show2, axis=2)


                                gamma = 1.2
                                image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())
                                image_show1 = image_show1 ** gamma

                                image_show2 = (image_show2 - image_show2.min()) / (image_show2.max() - image_show2.min())
                                image_show2 = image_show2 ** gamma
                                # enhancer = ImageEnhance.Brightness(Image.fromarray(image_show1))
                                # factor = 1.5 #brightens the image
                                # image_show1 = enhancer.enhance(factor)
                                # image_show2 = (image_show2 - image_show2.min()) / (image_show2.max() - image_show2.min())
                                gt_mask_show1 = mask1.cpu().detach()[batch_index_to_show, :80, :80].numpy().squeeze()

                                # l1_dist_change_feats_pred_show = l1_dist_change_feats_pred.numpy()[batch_index_to_show, :, :]
                                # l2_dist_change_feats_pred_show = l2_dist_change_feats_pred.numpy()[batch_index_to_show, :, :]
                                # histc_int_change_feats_pred_show = histc_int_change_feats_pred.numpy()[batch_index_to_show, :, :]

                                # l1_patched_diff_change_features_show = l1_patched_diff_change_features.cpu().detach().numpy()[batch_index_to_show, :, :]
                                # l2_patched_diff_change_features_show = l2_patched_diff_change_features.cpu().detach().numpy()[batch_index_to_show, :, :]
                                # histograms_intersection_show = histograms_intersection_features.cpu().detach().numpy()[batch_index_to_show, :, :]

                                # l1_patched_diff_change_features_show = (l1_patched_diff_change_features_show - l1_patched_diff_change_features_show.min())/(l1_patched_diff_change_features_show.max() - l1_patched_diff_change_features_show.min())
                                # l2_patched_diff_change_features_show = (l2_patched_diff_change_features_show - l2_patched_diff_change_features_show.min())/(l2_patched_diff_change_features_show.max() - l2_patched_diff_change_features_show.min())
                                # histograms_intersection_show = (histograms_intersection_show - histograms_intersection_show.min())/(histograms_intersection_show.max() - histograms_intersection_show.min())

                                # pred1_show = masks1.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]
                                # pred2_show = masks2.max(1)[1].cpu().detach().numpy()[batch_index_to_show, :, :]

                                # l1_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*l1_dist_change_feats_pred_show)
                                # l2_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*l2_dist_change_feats_pred_show)
                                # histc_fp_tp_fn_prediction_mask = gt_mask_show1 + (2*histc_int_change_feats_pred_show)

                                # vca_pseudomask_show = image_change_magnitude_binary.cpu().detach()[batch_index_to_show, :, :].numpy()
                                # vca_pseudomask_crop_show = cm_binary_crop.cpu().detach()[batch_index_to_show,:,:].numpy()
                                # dictionary_show = dictionary1.cpu().detach()[batch_index_to_show,:,:].numpy()
                                # dictionary_show = dictionary2_post_assignment.cpu().detach()[batch_index_to_show, :, :].numpy()
                                # dictionary2_show = dictionary1_post_assignment.cpu().detach()[batch_index_to_show, :, :].numpy()


                                classes_in_gt = np.unique(gt_mask_show1)
                                ax1.imshow(image_show1)

                                ax2.imshow(image_show2)

                                ax3.imshow(image_show1)
                                ax3.imshow(gt_mask_show1, cmap=self.cmap, vmin=0, vmax=self.max_label)


                                ax4.imshow(dictionary_img_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                                ax5.imshow(dictionary_baseline_feats_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                                ax6.imshow(dictionary_feats_show, cmap=self.cmap, vmin=0, vmax=self.max_label)


                                # ax7.imshow(image_show1)
                                # ax7.imshow(dictionary_img_show, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=0.4)

                                # ax8.imshow(image_show1)
                                # ax8.imshow(dictionary_baseline_feats_show, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=0.4)

                                # ax9.imshow(image_show1)
                                # ax9.imshow(dictionary_feats_show, cmap=self.cmap, vmin=0, vmax=self.max_label, alpha=0.4)

                                ax7.imshow(dictionary_img_show2, cmap=self.cmap, vmin=0, vmax=self.max_label)

                                ax8.imshow(dictionary_baseline_feats_show2, cmap=self.cmap, vmin=0, vmax=self.max_label)

                                ax9.imshow(dictionary_feats_show2, cmap=self.cmap, vmin=0, vmax=self.max_label)
                                
                                

                                if config['visualization']['log_scatters']:
                                    from matplotlib.colors import Normalize 
                                    ax10.scatter(image_tsne_feats[:, 0], image_tsne_feats[:, 1],
                                                c=image_scatter_dict,
                                                s=15.5,
                                                cmap=self.cmap,
                                                vmin=0, vmax=self.max_label,
                                                marker='.',
                                                linewidths=0.8
                                                )
                        
                                    ax11.scatter(resnet_tsne_feats[:, 0], resnet_tsne_feats[:, 1],
                                                    c=resnet_scatter_dict,
                                                    s=15.5,
                                                    cmap=self.cmap,
                                                    vmin=0, vmax=self.max_label,
                                                    marker='.',
                                                    linewidths=0.8,
                                                    # norm=Normalize
                                                    )

                                    ax12.scatter(ours_tsne_feats[:, 0], ours_tsne_feats[:, 1],
                                                    c=ours_scatter_dict,
                                                    s=15.5,
                                                    cmap=self.cmap,
                                                    vmin=0, vmax=self.max_label,
                                                    marker='.',
                                                    linewidths=0.8,
                                                    # norm=Normalize
                                                    )
                                    ax10.grid()
                                    ax11.grid()
                                    ax12.grid()

                                ax1.axis('off')
                                ax2.axis('off')
                                ax3.axis('off')
                                ax4.axis('off')
                                ax5.axis('off')
                                ax6.axis('off')
                                ax7.axis('off')
                                ax8.axis('off')
                                ax9.axis('off')
                                # ax10.axis('off')
                                # ax11.axis('off')
                                # ax12.axis('off')
                                # ax13.axis('off')
                                # ax14.axis('off')
                                # ax15.axis('off')
                                ax10.tick_params(axis="y", direction="in", pad=-25)#, reset=True)
                                ax10.tick_params(axis="x", direction="in", pad=-15)
                                ax11.tick_params(axis="y", direction="in", pad=-25)#, reset=True)
                                ax11.tick_params(axis="x", direction="in", pad=-15)
                                ax12.tick_params(axis="y", direction="in", pad=-25)#, reset=True)
                                ax12.tick_params(axis="x", direction="in", pad=-15)

                                if config['visualization']['titles']:
                                    ax1.set_title(f"1. Input Image 1", fontsize=config['visualization']['font_size'])
                                    ax2.set_title(f"2. Ground Truth Mask", fontsize=config['visualization']['font_size'])
                                    ax4.set_title(f"3. Visual Words of Patched Multi-Spectral Image", fontsize=config['visualization']['font_size'])
                                    ax5.set_title(f"4. Visual Words of Baseline Backbone", fontsize=config['visualization']['font_size'])
                                    ax6.set_title(f"5. Visual Words of Our Backbone", fontsize=config['visualization']['font_size'])
                                    ax7.set_title(f"3.1 Visual Words of Patched Multi-Spectral Image", fontsize=config['visualization']['font_size'])
                                    ax8.set_title(f"4.2 Visual Words of Bseline Backbone Overlaid", fontsize=config['visualization']['font_size'])
                                    ax9.set_title(f"5.2 Visual Words of Our Backbone Overlaid", fontsize=config['visualization']['font_size'])
                                    # ax8.set_title(f"l2_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                    # ax9.set_title(f"histc_fp_tp_fn_prediction_mask", fontsize=config['visualization']['font_size'])
                                    # figure.suptitle(
                                    #     f"Epoch: {epoch+1}\nGT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])

                                # figure.tight_layout()

                                if config['visualization']['val_imshow']:
                                    plt.show()
                                
                                if (config['visualization']['save_individual_plots'] or save_individual_plots_specific):

                                    # plots_path_save = f"{config['visualization']['save_individual_plots_path']}"
                                    plots_path_save =  "/home/native/projects/data/smart_watch/visualization/visual_words_comp/"

                                    fig_save_image_root = (f"{plots_path_save}/image_root/", ax1)
                                    fig_save_image2_root = (f"{plots_path_save}/image2/", ax2)
                                    fig_save_gt_root = (f"{plots_path_save}/gt/", ax3)
                                    fig_save_kmeans_prediction_root = (f"{plots_path_save}/kmeans/", ax4)
                                    fig_save_resnet_prediction_root = (f"{plots_path_save}/resnet/", ax5)
                                    fig_save_prediction_root = (f"{plots_path_save}/ours/", ax6)
                                    fig_save_kmeans_prediction2_root = (f"{plots_path_save}/kmeans2/", ax7)
                                    fig_save_resnet_prediction2_root = (f"{plots_path_save}/resnet2/", ax8)
                                    fig_save_prediction2_root = (f"{plots_path_save}/ours2/", ax9)

                                    fig_save_kmeans_scatter_root = (f"{plots_path_save}/kmeans_scatter/", ax10)
                                    fig_save_resnet_scatter_root = (f"{plots_path_save}/resnet_scatter/", ax11)
                                    fig_save_scatter_root = (f"{plots_path_save}/ours_scatter/", ax12)

                                    roots = [
                                        fig_save_image_root,
                                        fig_save_prediction_root,
                                        fig_save_image2_root,
                                        fig_save_gt_root,
                                        fig_save_resnet_prediction_root,
                                        fig_save_kmeans_prediction_root,
                                        # fig_save_prediction_overlaid_root,
                                        # fig_save_resnet_prediction_overlaid_root,
                                        # fig_save_kmeans_prediction_overlaid_root,
                                        fig_save_kmeans_prediction2_root,
                                        fig_save_resnet_prediction2_root,
                                        fig_save_prediction2_root
                                    ]

                                    if config['visualization']['log_scatters']:
                                        roots.append(fig_save_kmeans_scatter_root)
                                        roots.append(fig_save_resnet_scatter_root)
                                        roots.append(fig_save_scatter_root)

                                    figure.savefig(
                                        f"{plots_path_save}/figs/{image_name}_{str(b)}.png", bbox_inches='tight')
                                    for root, ax in roots:
                                        utils.create_dir_if_doesnt_exist(root)
                                        file_path = f"{root}/{image_name}_{str(b)}.png"
                                        # extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
                                        extent = ax.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.dpi_scale_trans.inverted()).padded(2/72)
                                        figure.savefig(file_path, bbox_inches=extent)

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

        cometml_experiemnt.log_metrics({f"L1 Training Recall class {str(x)}": l1_recall[x] for x in range(len(l1_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L1 Training Precision class {str(x)}": l1_precision[x] for x in range(len(l1_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L1 Training F1_score class {str(x)}": l1_f1[x] for x in range(len(l1_f1))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"L2 Training Recall class {str(x)}": l2_recall[x] for x in range(len(l2_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L2 Training Precision class {str(x)}": l2_precision[x] for x in range(len(l2_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"L2 Training F1_score class {str(x)}": l2_f1[x] for x in range(len(l2_f1))}, epoch=epoch+1)

        cometml_experiemnt.log_metrics({f"Histogram Distance Training Recall class {str(x)}": hist_recall[x] for x in range(len(hist_recall))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"Histogram Distance Training Precision class {str(x)}": hist_precision[x] for x in range(len(hist_precision))}, epoch=epoch+1)
        cometml_experiemnt.log_metrics({f"Histogram Distance Training F1_score class {str(x)}": hist_f1[x] for x in range(len(hist_f1))}, epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Average Loss", total_loss/loader.__len__(), epoch=epoch+1)

        return total_loss/loader.__len__(), classwise_f1_score

    def forward(self, cometml_experiment: object, world_size: int =8) -> tuple:
        """forward pass for all epochs

        Args:
            cometml_experiment (object): comet ml experiment for logging
            world_size (int, optional): for distributed training. Defaults to 8.

        Returns:
            tuple: (train losses, validation losses, mIoU)
        """
        train_losses, val_losses = [], []
        mean_ious_val,mean_ious_val_list,count_metrics_list = [], [], []

        model_save_dir = config['data'][config['location']]['model_save_dir'] + f"{current_path[-1]}_{config['dataset']}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        # for epoch in range(0, self.epochs):

        if config['procedures']['validate']:
            with cometml_experiment.validate():
                val_loss, val_cw_f1 = self.validate(0, cometml_experiment, save_individual_plots_specific=True)
        self.scheduler.step()

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
        # config['data']['num_classes'] = pretrain_config['data']['num_classes']
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
                

    ours_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_trainWin_7_modelName_resnet_enc_2021-10-19-21:07/experiments_epoch_5_loss_2.1330662268512652_valmF1_0.6782787764504841_valChangeF1_0.47969179367601383_time_2021-10-20-03:39:36.pth"
    # ours_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_trainWin_11_modelName_resnet_enc_2021-11-06-09:43/experiments_epoch_7_loss_12.406291961669922_valmF1_0.6657271284686177_valChangeF1_0.46656415450485705_time_2021-11-06-19:08:36.pth"
    resnet_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_trainWin_7_modelName_resnet_2021-10-17-18:32/experiments_epoch_3_loss_6.104687501799385_valmF1_0.6810474728935411_valChangeF1_0.4812717258140021_time_2021-10-17-21:41:42.pth"

    ours_model = build_model(model_name="resnet_enc",
                        backbone="resnet34",
                        pretrained=ours_path,
                        num_classes=128,
                        num_groups=32,
                        weight_std=True,
                        beta=False,
                        num_channels=9,
                        out_dim=128,
                        feats=[64, 128, 256, 512, 256])

    resnet_model = build_model(model_name="resnet",
                        backbone="resnet34",
                        pretrained=resnet_path,
                        num_classes=128,
                        num_groups=32,
                        weight_std=True,
                        beta=False,
                        num_channels=9,
                        out_dim=128,
                        feats=[64, 128, 256, 512, 256])

    # model = SupConResNet(name=config['training']['backbone'])
    num_params = sum(p.numel() for p in ours_model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    ours_model = nn.DataParallel(ours_model)
    ours_model.to(device)

    resnet_model = nn.DataParallel(resnet_model)
    resnet_model.to(device)

    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False

    optimizer = optim.SGD(ours_model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader),
                                                     eta_min=config['training']['learning_rate'])
    
    ours_checkpoint = torch.load(ours_path)
    resnet_checkpoint = torch.load(resnet_path)
    ours_model.load_state_dict(ours_checkpoint['model'], strict= False)
    # resnet_model.load_state_dict(resnet_checkpoint['model'], strict= False)


    trainer = Trainer(ours_model,
                      resnet_model,
                      train_dataloader,
                      test_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=test_dataloader,
                      test_with_full_supervision=config['training']['test_with_full_supervision']
                      )
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
