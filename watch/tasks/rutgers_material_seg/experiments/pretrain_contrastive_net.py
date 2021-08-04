import sys
import os
current_path = os.getcwd().split("/")

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

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)


class Trainer(object):
    def __init__(self, model: object, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, epochs: int, 
                 optimizer: object, scheduler: object, 
                 test_loader: torch.utils.data.DataLoader =None, 
                 test_with_full_supervision: int =0) -> None:
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
        # self.k = config['training']['out_features_dim']
        self.k = config['data']['num_classes']
        self.kmeans = KMeans(n_clusters=self.k, mode='euclidean', verbose=0, minibatch=None)
        self.max_label = self.k
        self.all_crops_params = [tuple([i,j,config['data']['window_size'], config['data']['window_size']]) for i in range(config['data']['window_size'],config['data']['image_size'] - config['data']['window_size']) for j in range(config['data']['window_size'],config['data']['image_size'] - config['data']['window_size'])]
        
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision
        
        self.train_second_transform = transforms.Compose([
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomAffine(degrees=(-20,20)),
                                                        # transforms.RandomPosterize(bits=2),
                                                        # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                                        # transforms.RandomEqualize(),
                                                        ])
        self.crop_size = (config['data']['window_size'], config['data']['window_size'])
        
        self.cmap = visualization.n_distinguishable_colors(nlabels=self.max_label,
                                                           first_color_black=True, last_color_black=True, 
                                                           bg_alpha=config['visualization']['bg_alpha'],
                                                           fg_alpha=config['visualization']['fg_alpha'])
        
    def high_confidence_filter(self, features: torch.Tensor, cutoff_top: float =0.75, 
                               cutoff_low: float =0.2, eps: float =1e-8) -> torch.Tensor:
        """Select high confidence regions to select as predictions

        Args:
            features (torch.Tensor): initial mask
            cutoff_top (float, optional): cutoff of the object. Defaults to 0.75.
            cutoff_low (float, optional): low cutoff. Defaults to 0.2.
            eps (float, optional): small number. Defaults to 1e-8.

        Returns:
            torch.Tensor: pseudo mask generated
        """
        bs,c,h,w = features.size()
        features = features.view(bs,c,-1)

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
        filtered_features = filtered_features.view(bs,c,h,w)
        
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
        self.model.train()
        print(f"starting epoch {epoch}")
        loader_size = len(self.train_loader)
        if config['visualization']['train_visualization_divisor'] > loader_size:
            config['visualization']['train_visualization_divisor'] = loader_size
        iter_visualization = loader_size // config['visualization']['train_visualization_divisor']
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        batch_index_to_show = config['visualization']['batch_index_to_show']
        for batch_index, batch in pbar:
            # if batch_index < 75:
            #     continue
            outputs = batch
            images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
            negative_images = outputs['inputs']['negative_im'].data[0]
            image_name = outputs['tr'].data[0][batch_index_to_show]['gids']
            original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
            mask = torch.stack(mask)
            mask = mask.long().squeeze(1)

            bs, c, t, h, w = images.shape
            image1 = images[:,:,0,:,:]
            negative_image1 = negative_images[:,:,0,:,:]
            mask1 = mask[:,0,:,:]
            
            class_to_show = max(0,torch.unique(mask)[-1] - 1)
            images = images.to(device)
            image1 = image1.to(device)
            negative_images = negative_images.to(device)
            negative_image1 = negative_image1.to(device)
            mask = mask.to(device)
            
            image1 = utils.stad_image(image1)
            image2 = self.train_second_transform(image1)
            
            negative_image1 = utils.stad_image(negative_image1)
            
            # patched_image1 = torch.stack([transforms.functional.crop(image1, *params) for params in self.all_crops_params],dim=1)
            # patched_image2 = torch.stack([transforms.functional.crop(image1, *params) for params in self.all_crops_params],dim=1)
            # patched_negative_image1 = torch.stack([transforms.functional.crop(negative_image1, *params) for params in self.all_crops_params],dim=1)
            
            # bs, ps, c, h, w = patched_image1.shape
            # patched_image1 = patched_image1.view(bs*ps,c,h,w)
            # patched_image2 = patched_image2.view(bs*ps,c,h,w)
            # patched_negative_image1 = patched_negative_image1.view(bs*ps,c,h,w)
            
            output1, features1 = self.model(image1)  ## [B,22,150,150]
            output2, features2 = self.model(image1)  ## [B,22,150,150]
            negative_output1, negative_features1 = self.model(negative_image1)  ## [B,22,150,150]

            patched_output1 = torch.stack([transforms.functional.crop(output1, *params) for params in self.all_crops_params],dim=1)
            patched_output2 = torch.stack([transforms.functional.crop(output2, *params) for params in self.all_crops_params],dim=1)
            patched_negative_output1 = torch.stack([transforms.functional.crop(negative_output1, *params) for params in self.all_crops_params],dim=1)
            
            bs, ps, c, h, w = patched_output1.shape
            patched_output1 = patched_output1.view(bs * ps,c,h,w)
            patched_output2 = patched_output2.view(bs * ps,c,h,w)
            patched_negative_output1 = patched_negative_output1.view(bs * ps,c,h,w)
            
            loss = 30 * F.triplet_margin_loss(patched_output1,#.unsqueeze(0), 
                                            patched_output2,#.unsqueeze(0), 
                                            patched_negative_output1
                                            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_seg += loss.item()
            
            masks1 = F.softmax(output1, dim=1)
            masks2 = F.softmax(output2, dim=1)
            pred1 = masks1.max(1)[1].cpu().detach()#.numpy()
            pred2 = masks2.max(1)[1].cpu().detach()#.numpy()
            change_detection_prediction = (pred1 != pred2).type(torch.uint8)

            total_loss += loss.item()
            mask1[mask1 == -1] = 0
            preds.append(change_detection_prediction.cpu())
            targets.append(mask1.cpu())#.numpy())
            
            if config['visualization']['train_visualizer'] :
                if (epoch) % config['visualization']['visualize_training_every'] == 0:
                    if (batch_index % iter_visualization) == 0:
                        figure = plt.figure(figsize=(config['visualization']['fig_size'],config['visualization']['fig_size']))
                        ax1 = figure.add_subplot(3,4,1)
                        ax2 = figure.add_subplot(3,4,2)
                        ax3 = figure.add_subplot(3,4,3)
                        ax4 = figure.add_subplot(3,4,4)
                        ax5 = figure.add_subplot(3,4,5)
                        ax6 = figure.add_subplot(3,4,6)
                        ax7 = figure.add_subplot(3,4,7)
                        ax8 = figure.add_subplot(3,4,8)
                        ax9 = figure.add_subplot(3,4,9)
                        ax10 = figure.add_subplot(3,4,10)
                        ax11 = figure.add_subplot(3,4,11)
                        ax12 = figure.add_subplot(3,4,12)

                        cmap_gradients = plt.cm.get_cmap('jet') 
                        # image_show = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,:3]
                        image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                        image_show1 = np.flip(image_show1, axis=2)
                        
                        image_show2 = np.transpose(image2.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                        image_show2 = np.flip(image_show2, axis=2)
                        
                        negative_image1_show = np.transpose(negative_image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                        negative_image1_show = np.flip(negative_image1_show, axis=2)
                        
                        # cropped_image1_show = np.transpose(cropped_image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                        # cropped_image1_show = np.flip(cropped_image1_show, axis=2)
                        
                        # cropped_image2_show = np.transpose(cropped_image2.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                        # cropped_image2_show = np.flip(cropped_image2_show, axis=2)
                        
                        image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())
                        image_show2 = (image_show2 - image_show2.min()) / (image_show2.max() - image_show2.min())
                        negative_image1_show = (negative_image1_show - negative_image1_show.min()) / (negative_image1_show.max() - negative_image1_show.min())
                        # cropped_image1_show = (cropped_image1_show - cropped_image1_show.min())/(cropped_image1_show.max() - cropped_image1_show.min())
                        # cropped_image2_show = (cropped_image2_show - cropped_image2_show.min())/(cropped_image2_show.max() - cropped_image2_show.min())

                        # print(f"min: {image_show.min()}, max: {image_show.max()}")
                        # image_show = np.transpose(outputs['visuals']['image'][batch_index_to_show,:,:,:].numpy(),(1,2,0))
                        logits_show1 = masks1.max(1)[1].cpu().detach().numpy()[batch_index_to_show,:,:]
                        logits_show2 = masks2.max(1)[1].cpu().detach().numpy()[batch_index_to_show,:,:]
                        change_detection_prediction_show = change_detection_prediction.numpy()[batch_index_to_show,:,:]
                        # stacked_change_detection_prediction_show = pred_stacked.numpy()[batch_index_to_show,:,:]
                        change_detection_show = change_detection_prediction_show
                        # change_detection_show = stacked_change_detection_prediction_show
                        gt_mask_show1 = mask.cpu().detach()[batch_index_to_show,0,:,:].numpy().squeeze()
                        gt_mask_show2 = mask.cpu().detach()[batch_index_to_show,1,:,:].numpy().squeeze()
                        output1_sample1 = masks1[batch_index_to_show,class_to_show,:,:].cpu().detach().numpy().squeeze()

                        # vca_pseudomask_show = image_change_magnitude_binary.cpu().detach()[batch_index_to_show,:,:].numpy()
                        # vca_pseudomask_crop_show = cm_binary_crop.cpu().detach()[batch_index_to_show,:,:].numpy()
                        # dictionary_show = dictionary1.cpu().detach()[batch_index_to_show,:,:].numpy()
                        # dictionary2_show = cropped_dictionary.cpu().detach()[batch_index_to_show,:,:].numpy()
                        # print(vca_pseudomask_crop_show.shape)

                        fp_tp_fn_prediction_mask = gt_mask_show1 + (2 * change_detection_show)
                        
                        logits_show1[logits_show1 == -1] = 0
                        logits_show2[logits_show2 == -1] = 0
                        gt_mask_show_no_bg1 = np.ma.masked_where(gt_mask_show1 == 0,gt_mask_show1)
                        gt_mask_show_no_bg2 = np.ma.masked_where(gt_mask_show2 == 0,gt_mask_show2)
                        # logits_show_no_bg = np.ma.masked_where(logits_show==0,logits_show)

                        classes_in_gt = np.unique(gt_mask_show1)
                        ax1.imshow(image_show1)

                        ax2.imshow(image_show2)
                        
                        ax3.imshow(negative_image1_show)

                        ax4.imshow(image_show1)
                        ax4.imshow(gt_mask_show1, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                        
                        ax5.imshow(image_show1)
                        ax5.imshow(logits_show1, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                        
                        ax6.imshow(image_show2)
                        ax6.imshow(logits_show2, cmap=self.cmap, vmin=0, vmax=self.max_label)
                        
                        ax7.imshow(image_show2)
                        # ax6.imshow(change_detection_prediction_show, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                        ax7.imshow(change_detection_show, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                        # ax6.imshow(fp_tp_fn_prediction_mask, cmap='tab20c', vmin=0, vmax=6)#, alpha=alphas_final_gt)

                        # ax8.imshow(vca_pseudomask_show, cmap=self.cmap, vmin=0, vmax=self.max_label)
                        
                        # ax9.imshow(cropped_image1_show)
                        
                        # ax10.imshow(cropped_image2_show)
                        
                        # ax11.imshow(dictionary_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                        # ax12.imshow(dictionary2_show, cmap=self.cmap, vmin=0, vmax=self.max_label)
                        
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
                            ax3.set_title(f"Negative Sample", fontsize=config['visualization']['font_size'])
                            ax4.set_title(f"Change GT Mask overlaid", fontsize=config['visualization']['font_size'])
                            ax5.set_title(f"Prediction overlaid 1", fontsize=config['visualization']['font_size'])
                            ax6.set_title(f"Prediction overlaid 2", fontsize=config['visualization']['font_size'])
                            ax7.set_title(f"Change Detection Prediction", fontsize=config['visualization']['font_size'])
                            ax8.set_title(f"Pseudo-Mask for Change", fontsize=config['visualization']['font_size'])
                            ax9.set_title(f"Cropped Input Image 1", fontsize=config['visualization']['font_size'])
                            ax10.set_title(f"Cropped Input Image 2", fontsize=config['visualization']['font_size'])
                            ax11.set_title(f"Visual Words", fontsize=config['visualization']['font_size'])
                            ax12.set_title(f"Cropped Visual Words", fontsize=config['visualization']['font_size'])
                            figure.suptitle(f"Epoch: {epoch+1}\nGT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])
                            
                        figure.tight_layout()
                        # cometml_experiemnt.log_figure(figure_name=f"Training, image name: {image_name}, epoch: {epoch}, classes in gt: {classes_in_gt}, classifier predictions: {labels_predicted_indices}",figure=figure)
                        cometml_experiemnt.log_figure(figure_name=f"Training, image name: {image_name}",figure=figure)

                        if config['visualization']['train_imshow']:
                            plt.show()

                        figure.clear()
                        plt.cla()
                        plt.clf()
                        plt.close('all')
                        plt.close(figure)
                        gc.collect()
            
        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=2)
        
        mean_iou = np.array(mean_iou)
        precision = np.array(precision)
        recall = np.array(recall)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        overall_miou = mean_iou.mean()
        classwise_f1_score = 2 * (precision * recall) / (precision + recall)
        mean_f1_score = classwise_f1_score.mean()
        
        # overall_miou = sum(mean_iou)/len(mean_iou)
        # print(f"Training class-wise mIoU value: \n{np.array(mean_iou)} \noverall mIoU: {overall_miou}")
        # print(f"Training class-wise Precision value: \n{np.array(precision)} \noverall Precision: {mean_precision}")
        # print(f"Training class-wise Recall value: \n{np.array(recall)} \noverall Recall: {mean_recall}")
        # print(f"Training overall F1 Score: {f1_score}")
        
        cometml_experiemnt.log_metric("Training Loss", total_loss, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Segmentation Loss", total_loss_seg, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Training mIoU", overall_miou, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Training mean_f1_score", mean_f1_score, epoch=epoch + 1)

        cometml_experiemnt.log_metrics({f"Training Recall class {str(x)}":recall[x] for x in range(len(recall))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"Training Precision class {str(x)}":precision[x] for x in range(len(precision))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"Training F1_score class {str(x)}":classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch + 1)
        
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
        preds, stacked_preds, targets  = [], [], []
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
            for batch_index,batch in pbar:
                outputs = batch
                images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
                image_name = str(outputs['tr'].data[0][batch_index_to_show]['gids'][0])
                
                mask = torch.stack(mask)
                mask = mask.long().squeeze(1)
                
                bs, c, t, h, w = images.shape
                image1 = images[:,:,0,:,:]
                mask1 = mask[:,0,:,:]
                
                images = images.to(device)
                image1 = image1.to(device)
                mask = mask.to(device)
                
                image1 = utils.stad_image(image1)
                
                output1, features1 = self.model(image1)  ## [B,22,150,150]
                output2, features2 = self.model(image2)
                
                stacked_for_prediction = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1)], dim=2).squeeze(1)
                
                # masks1 = F.softmax(output1, dim=1)#.detach()
                # masks2 = F.softmax(output2, dim=1)#.detach()
                # print(stacked_for_prediction.shape)
                # masks1 = F.softmax(stacked_for_prediction, dim=1)
                # masks2 = F.softmax(stacked_for_prediction, dim=1)
                # print(features1.shape)
                masks1 = F.softmax(features1, dim=1)
                masks2 = F.softmax(features2, dim=1)
                # masks = F.interpolate(masks, size=mask.size()[-2:], mode="bilinear", align_corners=True)
                # masks1 = self.high_confidence_filter(masks1, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                # masks2 = self.high_confidence_filter(masks2, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                pred1 = masks1.max(1)[1].cpu().detach()#.numpy()
                pred2 = masks2.max(1)[1].cpu().detach()#.numpy()
                change_detection_prediction = (pred1 != pred2).type(torch.uint8)
                
                masks_stacked = F.softmax(stacked_for_prediction.squeeze(1), dim=1)#.detach()
                pred_stacked = masks_stacked.max(1)[1].cpu().detach()#.numpy()
                pred_stacked[pred_stacked != 1] = 0
                
                preds.append(change_detection_prediction)
                stacked_preds.append(pred_stacked)
                mask1[mask1 == -1] = 0
                targets.append(mask1.cpu())#.numpy())

                if config['visualization']['val_visualizer'] or (config['visualization']['save_individual_plots'] and save_individual_plots_specific):
                    if (epoch) % config['visualization']['visualize_val_every'] == 0:
                        if (batch_index % iter_visualization) == 0:
                            figure = plt.figure(figsize=(config['visualization']['fig_size'],config['visualization']['fig_size']))
                            ax1 = figure.add_subplot(2,3,1)
                            ax2 = figure.add_subplot(2,3,2)
                            ax3 = figure.add_subplot(2,3,3)
                            ax4 = figure.add_subplot(2,3,4)
                            ax5 = figure.add_subplot(2,3,5)
                            ax6 = figure.add_subplot(2,3,6)
                            
                            cmap_gradients = plt.cm.get_cmap('jet')

                            image_show1 = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                            image_show1 = np.flip(image_show1, axis=2)
                            
                            image_show2 = np.transpose(image2.cpu().detach().numpy()[batch_index_to_show,:,:,:],(1,2,0))[:,:,1:4]
                            image_show2 = np.flip(image_show2, axis=2)
                            
                            image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())
                            image_show2 = (image_show2 - image_show2.min()) / (image_show2.max() - image_show2.min())
                            # print(f"min: {image_show.min()}, max: {image_show.max()}")
                            # image_show = np.transpose(outputs['visuals']['image'][batch_index_to_show,:,:,:].numpy(),(1,2,0))
                            logits_show1 = masks1.max(1)[1].cpu().detach().numpy()[batch_index_to_show,:,:]
                            logits_show2 = masks2.max(1)[1].cpu().detach().numpy()[batch_index_to_show,:,:]
                            change_detection_prediction_show = change_detection_prediction.numpy()[batch_index_to_show,:,:]
                            stacked_change_detection_prediction_show = pred_stacked.numpy()[batch_index_to_show,:,:]
                            change_detection_show = change_detection_prediction_show
                            # change_detection_show = stacked_change_detection_prediction_show
                            gt_mask_show1 = mask.cpu().detach()[batch_index_to_show,0,:,:].numpy().squeeze()
                            gt_mask_show2 = mask.cpu().detach()[batch_index_to_show,1,:,:].numpy().squeeze()
                            
                            fp_tp_fn_prediction_mask = gt_mask_show1 + (2 * change_detection_show)
                            
                            logits_show1[logits_show1 == -1] = 0
                            logits_show2[logits_show2 == -1] = 0
                            gt_mask_show_no_bg1 = np.ma.masked_where(gt_mask_show1 == 0,gt_mask_show1)
                            gt_mask_show_no_bg2 = np.ma.masked_where(gt_mask_show2 == 0,gt_mask_show2)
                            # logits_show_no_bg = np.ma.masked_where(logits_show==0,logits_show)

                            classes_in_gt = np.unique(gt_mask_show1)
                            ax1.imshow(image_show1)

                            ax2.imshow(image_show1)
                            ax2.imshow(gt_mask_show1, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)

                            ax3.imshow(image_show1)
                            ax3.imshow(logits_show1, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                            
                            ax4.imshow(image_show2)
                            
                            ax5.imshow(image_show2)
                            ax5.imshow(logits_show2, cmap=self.cmap, vmin=0, vmax=self.max_label)
                            
                            ax6.imshow(image_show2)
                            # ax6.imshow(change_detection_prediction_show, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                            ax6.imshow(change_detection_show, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)
                            # ax6.imshow(fp_tp_fn_prediction_mask, cmap=self.cmap, vmin=0, vmax=self.max_label)#, alpha=alphas_final_gt)

                            ax1.axis('off')
                            ax2.axis('off')
                            ax3.axis('off')
                            ax4.axis('off')
                            ax5.axis('off')
                            ax6.axis('off')
                            
                            if config['visualization']['titles']:
                                ax1.set_title(f"Input Image 1", fontsize=config['visualization']['font_size'])
                                ax2.set_title(f"GT Mask overlaid 1", fontsize=config['visualization']['font_size'])
                                ax3.set_title(f"Prediction overlaid 1", fontsize=config['visualization']['font_size'])
                                ax4.set_title(f"Input Image 2", fontsize=config['visualization']['font_size'])
                                ax5.set_title(f"Prediction overlaid 2", fontsize=config['visualization']['font_size'])
                                ax6.set_title(f"Change Detection Prediction", fontsize=config['visualization']['font_size'])
                                figure.suptitle(f"GT labels for classification: {classes_in_gt}, \nunique in change predictions: {np.unique(change_detection_show)}\nunique in predictions1: {np.unique(logits_show1)}", fontsize=config['visualization']['font_size'])
                            
                            figure.tight_layout()
                            if (config['visualization']['save_individual_plots'] and save_individual_plots_specific):

                                plots_path_save = f"{config['visualization']['save_individual_plots_path']}{config['dataset']}/"
                                fig_save_image_root = (f"{plots_path_save}/image_root/",ax1)
                                fig_save_prediction_root = (f"{plots_path_save}/predictions/",ax3)
                                fig_save_overlaid_full_supervised_mask_on_image_alpha_with_bg = (f"{plots_path_save}/overlaid_full_alpha_w_bg/",ax2)
                                roots = [
                                        fig_save_image_root, 
                                        fig_save_prediction_root,
                                        fig_save_overlaid_full_supervised_mask_on_image_alpha_with_bg
                                        ]
                                figure.savefig(f"{plots_path_save}/figs/{image_name}.png", bbox_inches='tight')
                                for root, ax in roots:
                                    utils.create_dir_if_doesnt_exist(root)
                                    file_path = f"{root}/{image_name}.png"
                                    # extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
                                    extent = ax.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.dpi_scale_trans.inverted())
                                    figure.savefig(file_path, bbox_inches=extent)
                            
                            cometml_experiemnt.log_figure(figure_name=f"Validation, Image name: {image_name}",figure=figure)
                            figure.clear()
                            plt.cla()
                            plt.clf()
                            plt.close('all')
                            plt.close(figure)
                            gc.collect()

        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=2)
        stacked_mean_iou, stacked_precision, stacked_recall = eval_utils.compute_jaccard(stacked_preds, targets, num_classes=2)
        
        mean_iou = np.array(mean_iou)
        precision = np.array(precision)
        recall = np.array(recall)
        
        # mean_precision = sum(precision)/len(precision)
        # mean_recall = sum(recall)/len(recall)
        # f1_score = 2*(mean_precision*mean_recall)/(mean_precision+mean_recall)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        overall_miou = mean_iou.mean()
        classwise_f1_score = 2 * (precision * recall) / (precision + recall)
        mean_f1_score = classwise_f1_score.mean()
        
        stacked_mean_iou = np.array(stacked_mean_iou)
        stacked_precision = np.array(stacked_precision)
        stacked_recall = np.array(stacked_recall)
        
        stacked_mean_precision = stacked_precision.mean()
        stacked_mean_recall = stacked_recall.mean()
        stacked_overall_miou = stacked_mean_iou.mean()
        stacked_classwise_f1_score = 2 * (stacked_precision * stacked_recall) / (stacked_precision + stacked_recall)
        stacked_mean_f1_score = stacked_classwise_f1_score.mean()
        
        # print(f"Validation stacked class-wise mIoU value: \n{stacked_mean_iou} \noverall mIoU: {stacked_mean_iou}")
        # print(f"Validation stacked class-wise Precision value: \n{stacked_precision} \noverall Precision: {stacked_precision}")
        # print(f"Validation stacked class-wise Recall value: \n{stacked_recall} \noverall Recall: {stacked_recall}")
        
        # print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU", overall_miou, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation precision", mean_precision, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation recall", mean_recall, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation mean f1_score", mean_f1_score, epoch=epoch + 1)
        print({f"Recall class {str(x)}":recall[x] for x in range(len(recall))})
        print({f"Precision class {str(x)}":precision[x] for x in range(len(precision))})
        cometml_experiemnt.log_metrics({f"Recall class {str(x)}":recall[x] for x in range(len(recall))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"Precision class {str(x)}":precision[x] for x in range(len(precision))}, epoch=epoch + 1)
        cometml_experiemnt.log_metrics({f"F1_score class {str(x)}":classwise_f1_score[x] for x in range(len(classwise_f1_score))}, epoch=epoch + 1)
        
        # cometml_experiemnt.log_metric("Validation stacked mIoU", stacked_overall_miou, epoch=epoch+1)
        # cometml_experiemnt.log_metric("Validation stacked precision", stacked_mean_precision, epoch=epoch+1)
        # cometml_experiemnt.log_metric("Validation stacked recall", stacked_mean_recall, epoch=epoch+1)
        # cometml_experiemnt.log_metric("Validation stacked f1_score", stacked_f1_score, epoch=epoch+1)
        
        cometml_experiemnt.log_metric("Validation Average Loss",total_loss / loader.__len__(),epoch=epoch + 1)
        
        return total_loss / loader.__len__(), overall_miou
    
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
        best_val_loss, train_loss = np.infty, np.infty
        best_val_mean_iou, val_mean_iou = 0, 0
        
        model_save_dir = config['data'][config['location']]['model_save_dir'] + f"{current_path[-1]}_{config['dataset']}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        for epoch in range(0,self.epochs):
            if config['procedures']['train']:
                with cometml_experiment.train():
                    train_loss = self.train(epoch,cometml_experiment)
            if config['procedures']['validate']:
                with cometml_experiment.validate():
                    val_loss, val_mean_iou = self.validate(epoch,cometml_experiment)
            self.scheduler.step()

            if val_mean_iou >= best_val_mean_iou:
                # best_train_loss = train_loss
                best_val_mean_iou = val_mean_iou
                model_save_name = f"{current_path[-1]}_epoch_{epoch}_loss_{train_loss}_valmIoU_{val_mean_iou}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"
                
                if config['procedures']['train']:
                    with open(model_save_dir + "config.yaml",'w') as file:
                        yaml.dump(config, file)
                
                    torch.save({'epoch': epoch, 
                                'model': self.model.state_dict(), 
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'loss':train_loss},
                                model_save_dir + model_save_name)
                if config['visualization']['save_individual_plots']:
                    _, _ = self.validate(epoch, cometml_experiment, save_individual_plots_specific=True)
            
        return train_losses, val_losses, mean_ious_val

if __name__ == "__main__":

    project_root = "/home/native/projects/watch/watch/tasks/rutgers_material_seg/"
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    main_config_path = f"{project_root}/configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    # experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"
    experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"
    # config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]
    
    experiment_config = utils.config_parser(experiment_config_path,experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    project_name = f"{current_path[-3]}_{current_path[-1]}_{config['dataset']}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
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
    coco_fpath = ub.expandpath(config['data'][config['location']]['coco_json'])
    dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(dset)

    # # print(sampler)
    number_of_timestamps, h, w = 2, 300, 300
    window_dims = (number_of_timestamps, h, w) #[t,h,w]
    input_dims = (h, w)

    # channels = 'B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A'
    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
    # channels = 'red|green|blue'
    # channels = 'gray'
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    print(dataset.__len__())
    train_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])
    
    model = build_model(model_name = config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'],
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'],
                        num_channels=config['training']['num_channels'],
                        out_dim=config['training']['out_features_dim'])
    
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
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(train_dataloader),
                                                     eta_min = config['training']['learning_rate'])

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"loaded model from {config['training']['resume']}")
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
            exit()
        
    trainer = Trainer(model,
                      train_dataloader,
                      train_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=train_dataloader,
                      test_with_full_supervision=config['training']['test_with_full_supervision']
                      )
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
