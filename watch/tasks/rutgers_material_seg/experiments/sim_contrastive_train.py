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
import itertools
from statistics import mean
from scipy.spatial import distance
from torchvision import transforms
import watch.tasks.rutgers_material_seg.utils.utils as utils
import watch.tasks.rutgers_material_seg.utils.eval_utils as eval_utils
import watch.tasks.rutgers_material_seg.utils.visualization as visualization
from watch.tasks.rutgers_material_seg.models.losses import SupConLoss
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.datasets import build_dataset
from watch.tasks.rutgers_material_seg.models.supcon import SupConResNet
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

mask_mapping = {0: "unknown",    # 0, unknown
                1: "urban" ,  # 179, urban land
                2: "agriculture",  # 226, agriculture land
                3: "rangeland",  # 105, rangeland, non-forest, non farm, green land
                4: "forest",  # 150, forest land
                5: "water",   # 29, water
                6: "barren"}  # 255, barren land, mountain, rock, dessert

possible_combinations = [list(i) for i in itertools.product([0, 1], repeat=len(mask_mapping.keys()))]
for index, item in enumerate(possible_combinations):
    num_labels = len(np.argwhere(np.array(item) == 1))
    if num_labels == 0:
        continue
    possible_combinations[index] = np.divide(possible_combinations[index], num_labels)

# print(np.unique(possible_combinations, return_counts=True, axis=0))
# possible_combinations = np.divide(possible_combinations, len(mask_mapping.keys()))
# print(possible_combinations)
verbose_labels = {}
for index, item in enumerate(possible_combinations):
    verbose_label = ""
    for label_index, label in enumerate(item):
        if label != 0:
            verbose_label += f"{mask_mapping[label_index]}: {label}, "
    verbose_labels[index] = verbose_label

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred[0]

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
        self.criterion = SupConLoss()
        self.max_label = config['data']['num_classes']

        self.train_second_transform = transforms.Compose([
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                                            transforms.RandomGrayscale(p=0.2),
                                                            # transforms.ToTensor(),
                                                        ])

        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision

        self.cmap = visualization.n_distinguishable_colors(nlabels=self.max_label,
                                                           first_color_black=True, last_color_black=True,
                                                           bg_alpha=config['visualization']['bg_alpha'],
                                                           fg_alpha=config['visualization']['fg_alpha'])

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
        accuracies, distances_to_gt_dist = [], []
        self.model.train()
        print(f"starting epoch {epoch}")
        loader_size = len(self.train_loader)
        iter_visualization = loader_size // config['visualization']['train_visualization_divisor']
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        batch_index_to_show = config['visualization']['batch_index_to_show']
        for batch_index, batch in pbar:
            # if batch_index < 75:
            #     continue
            outputs = batch
            # image1, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
            # image_name = outputs['tr'].data[0][batch_index_to_show]['gids']
            # original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
            # mask = torch.stack(mask)

            image1 = batch['inputs']['image']
            mask = batch['inputs']['mask']
            labels = batch['inputs']['labels']
            # print(labels)
            # labels = labels.long().squeeze(2)
            labels = labels.to(device)

            mask = mask.long().squeeze(1)

            image1 = image1.squeeze(2)
            images = torch.cat([image1, self.train_second_transform(image1)], dim=0)
            bs, c, h, w = images.shape
            # labels = torch.Tensor([1 for x in range(bs)])
            # print(images.shape)

            class_to_show = max(0, torch.unique(mask)[-1] - 1)
            images = images.to(device)
            mask = mask.to(device)

            print(images.shape)
            output1 = self.model(images)  # torch.Size([B, C+1, H, W]) torch.Size([4, 3, 64, 64])

            # print(output1.shape)
            f1, f2 = torch.split(output1, [bs // 2, bs // 2], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # print(labels.shape)
            # print(features.shape)
            loss = self.criterion(features, labels=labels)

            labels = torch.cat([labels, labels], dim=0)
            acc1, preds = accuracy(output1, labels, topk=(1,))
            # print(preds[0])
            verbose_preds = [verbose_labels[pred.item()] for pred in preds]
            batch_verboe_labels = [verbose_labels[label.item()] for label in labels]

            dist_preds = [possible_combinations[pred.item()] for pred in preds]
            dist_labels = [possible_combinations[label.item()] for label in labels]

            distances = distance.cdist(dist_preds, dist_labels)
            diagonal = np.diagonal(distances)
            batch_mean_distance_to_gt = np.mean(diagonal)
            # distances_to_gt_dist += list(diagonal)
            # print(f"dist_preds: {dist_preds}")
            # print(f"dist_labels: {dist_labels}")
            # print(f"distances: {distances}")
            # print(f"diagonal: {diagonal}")
            cometml_experiemnt.log_metric("Training Mean Distance to GT", batch_mean_distance_to_gt, epoch=epoch + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_seg += loss.item()

            masks = F.softmax(output1, dim=1)  # .detach()
            # masks = F.interpolate(masks, size=mask.size()[-2:], mode="bilinear", align_corners=True)
            # masks = self.high_confidence_filter(masks, cutoff_top=config['high_confidence_threshold']['train_cutoff'])
            pred = masks.max(1)[1].cpu().detach()  # .numpy()

            total_loss += loss.item()
            # preds.append(pred)
            # targets.append(mask.cpu())#.numpy())
            accuracies += acc1
            # print(accuracies)

            if config['visualization']['train_visualizer'] :
                if (epoch) % config['visualization']['visualize_training_every'] == 0:
                    if (batch_index % iter_visualization) == 0:
                        figure = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']))
                        ax1 = figure.add_subplot(1, 2, 1)
                        ax2 = figure.add_subplot(1, 2, 2)
                        # ax3 = figure.add_subplot(4,3,3)
                        # ax4 = figure.add_subplot(4,3,4)
                        # ax5 = figure.add_subplot(4,3,5)
                        # ax6 = figure.add_subplot(4,3,6)
                        # ax7 = figure.add_subplot(4,3,7)
                        # ax8 = figure.add_subplot(4,3,8)
                        # ax9 = figure.add_subplot(4,3,9)
                        # ax10 = figure.add_subplot(4,3,10)
                        # ax11 = figure.add_subplot(4,3,11)
                        # ax12 = figure.add_subplot(4,3,12)

                        cmap_gradients = plt.cm.get_cmap('jet')
                        image_show = np.transpose(images.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]

                        image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min())
                        # print(f"min: {image_show.min()}, max: {image_show.max()}")
                        # image_show = np.transpose(outputs['visuals']['image'][batch_index_to_show,:,:,:].numpy(),(1,2,0))
                        # logits_show = masks.max(1)[1].cpu().detach().numpy()[batch_index_to_show,:,:]
                        gt_mask_show = mask.cpu().detach()[batch_index_to_show, :, :].numpy().squeeze()
                        # output1_sample = masks[batch_index_to_show,class_to_show,:,:].cpu().detach().numpy().squeeze()
                        # gt_mask_show[gt_mask_show==-1] = 0
                        # image_show = image_show[:original_width, :original_height,:]
                        # logits_show = logits_show[:original_width, :original_height]
                        # gt_mask_show = gt_mask_show[:original_width, :original_height]
                        # output1_sample = output1_sample[:original_width, :original_height]

                        # logits_show[logits_show==-1]=0
                        gt_mask_show_no_bg = np.ma.masked_where(gt_mask_show == 0, gt_mask_show)
                        # logits_show_no_bg = np.ma.masked_where(logits_show==0,logits_show)

                        # classes_in_gt = np.unique(gt_mask_show)
                        ax1.imshow(image_show)

                        ax2.imshow(image_show)
                        ax2.imshow(gt_mask_show_no_bg, cmap=self.cmap, vmin=0, vmax=self.max_label)  # , alpha=alphas_final_gt)

                        ax1.axis('off')
                        ax2.axis('off')

                        if config['visualization']['titles']:
                            ax1.set_title(f"Input Image", fontsize=config['visualization']['font_size'])
                            ax2.set_title(f"GT Mask overlaid", fontsize=config['visualization']['font_size'])
                            # ax4.set_title(f"Prediction overlaid", fontsize=config['visualization']['font_size'])
                            # # ax5.set_title(f"output1_sample for class: {class_to_show} min: {output1_sample.min():0.2f}, max: {output1_sample.max():0.2f}", fontsize=config['visualization']['font_size'])
                            # ax10.set_title(f"GT Mask", fontsize=config['visualization']['font_size'])
                            # ax11.set_title(f"Prediction", fontsize=config['visualization']['font_size'])
                            figure.suptitle(f"Pred: {verbose_preds[batch_index_to_show]}\nGT label: {batch_verboe_labels[batch_index_to_show]}", fontsize=config['visualization']['font_size'])

                        # cometml_experiemnt.log_figure(figure_name=f"Training, image name: {image_name}, epoch: {epoch}, classes in gt: {classes_in_gt}, classifier predictions: {labels_predicted_indices}",figure=figure)
                        cometml_experiemnt.log_figure(figure_name=f"Training, image name", figure=figure)

                        if config['visualization']['train_imshow']:
                            plt.show()

                        figure.clear()
                        plt.cla()
                        plt.clf()
                        plt.close('all')
                        plt.close(figure)
                        gc.collect()

            # total_loss_cls += loss_cls.item()
            # total_loss += loss_cls.item()

        # define new distance map confidence score nomalization

        # mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=config['data']['num_classes'])
        # overall_miou = sum(mean_iou)/len(mean_iou)
        # print(f"Training class-wise mIoU value: \n{np.array(mean_iou)} \noverall mIoU: {overall_miou}")
        # print(accuracies)
        mean_acc = torch.mean(torch.stack(accuracies))
        cometml_experiemnt.log_metric("Training Loss", total_loss, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Segmentation Loss", total_loss_seg, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Accuracy", mean_acc, epoch=epoch + 1)
        # cometml_experiemnt.log_metric("Training mIoU", overall_miou, epoch=epoch+1)

        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch + 1, total_loss / self.train_loader.__len__()))

        return total_loss / self.train_loader.__len__()

    def validate(self, epoch: int, cometml_experiemnt: object) -> tuple:
        """validating single epoch

        Args:
            epoch (int): current epoch
            cometml_experiemnt (object): logging experiment

        Returns:
            tuple: (validation loss, mIoU)
        """
        print("validating")
        total_loss = 0
        preds, crf_preds, targets  = [], [], []
        accuracies = 0
        running_ap = 0.0
        accuracies = []
        batch_index_to_show = config['visualization']['batch_index_to_show']
        if self.test_with_full_supervision == 1:
            loader = self.test_loader
        else:
            loader = self.val_loader
        loader_size = len(loader)
        iter_visualization = loader_size // config['visualization']['val_visualization_divisor']
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total=len(loader))
            for batch_index, batch in pbar:
                outputs = batch
                # image1, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                # original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
                # mask = torch.stack(mask)

                image1 = batch['inputs']['image']
                mask = batch['inputs']['mask']
                labels = batch['inputs']['labels']

                mask = mask.long().squeeze(1)

                image1 = image1.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                # image_raw = utils.denorm(image1.clone().detach())
                bs, c, h, w = image1.shape
                image1 = image1.squeeze(2)

                images = torch.cat([image1, image1], dim=0)
                labels = torch.cat([labels, labels], dim=0)

                output = self.model(images)  # [B,22,150,150]

                acc1, preds = accuracy(output, labels, topk=(1,))

                verbose_preds = [verbose_labels[pred.item()] for pred in preds]
                batch_verboe_labels = [verbose_labels[label.item()] for label in labels]
                accuracies += acc1

                dist_preds = [possible_combinations[pred.item()] for pred in preds]
                dist_labels = [possible_combinations[label.item()] for label in labels]

                distances = distance.cdist(dist_preds, dist_labels)
                diagonal = np.diagonal(distances)
                batch_mean_distance_to_gt = np.mean(diagonal)

                cometml_experiemnt.log_metric("Validation Mean Distance to GT", batch_mean_distance_to_gt, epoch=epoch + 1)

                masks = F.softmax(output, dim=1)  # (B, 22, 300, 300)
                # masks = F.interpolate(masks, size=mask.size()[-2:], mode="bilinear", align_corners=True)
                # masks = self.high_confidence_filter(masks, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                pred = masks.max(1)[1].cpu().detach()  # .numpy()

                if self.use_crf:
                    crf_probs = utils.batch_crf_inference(image_raw.detach().cpu(),
                                                          masks.detach().cpu(),
                                                          t=config['evaluation']['crf_t'],
                                                          scale_factor=config['evaluation']['crf_scale_factor'],
                                                          labels=config['evaluation']['crf_labels'])
                    crf_probs = crf_probs.squeeze()
                    crf_pred = crf_probs.max(1)[1]
                    # crf_pred[crf_pred==self.max_label]=0
                    crf_preds.append(crf_pred)

                if config['visualization']['val_visualizer']:
                    if (epoch) % config['visualization']['visualize_val_every'] == 0:
                        if (batch_index % iter_visualization) == 0:
                            figure = plt.figure(figsize=(config['visualization']['fig_size'], config['visualization']['fig_size']))
                            ax1 = figure.add_subplot(1, 2, 1)
                            ax2 = figure.add_subplot(1, 2, 2)
                            # ax3 = figure.add_subplot(2,3,3)
                            # ax4 = figure.add_subplot(2,3,4)
                            # ax5 = figure.add_subplot(2,3,5)
                            # ax6 = figure.add_subplot(2,3,6)

                            cmap_gradients = plt.cm.get_cmap('jet')
                            # transformed_image_show = np.transpose(utils.denorm(image1).cpu().detach().numpy()[0,:,:,:],(1,2,0))
                            # image_show = np.transpose(outputs['visuals']['image'][0,:,:,:].numpy(),(1,2,0))
                            image_show = np.transpose(image1.cpu().detach().numpy()[batch_index_to_show, :, :, :], (1, 2, 0))[:, :, :3]
                            image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min())
                            gt_mask_show = mask.cpu().numpy()[batch_index_to_show, :, :].squeeze()
                            # gt_mask_show[gt_mask_show==self.max_label] = 0
                            # image_name = outputs['visuals']['image_name'][batch_index_to_show]
                            # logits_show = pred[0,:,:]
                            # classes_predicted = np.unique(logits_show)
                            # classes_in_gt = np.unique(gt_mask_show)

                            # image_show = image_show[:original_width, :original_height,:]
                            # logits_show = logits_show[:original_width, :original_height]
                            # gt_mask_show = gt_mask_show[:original_width, :original_height]

                            gt_mask_show_no_bg = np.ma.masked_where(gt_mask_show == 0, gt_mask_show)
                            # logits_show_no_bg = np.ma.masked_where(logits_show==0,logits_show)
                            # pseudo_gt_show_no_bg = np.ma.masked_where(pseudo_gt_show==0,pseudo_gt_show)

                            ax1.imshow(image_show)

                            ax2.imshow(image_show)
                            ax2.imshow(gt_mask_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            # ax3.imshow(image_show)
                            # ax3.imshow(logits_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            # if self.use_crf:
                            #     crf_pred_show = crf_pred[0,:,:].squeeze()
                            #     crf_pred_show_no_bg = np.ma.masked_where(crf_pred_show==0,crf_pred_show)
                            #     crf_prob_show = crf_probs[0,class_to_show,:,:].squeeze()
                            #     ax4.imshow(image_show)
                            #     ax4.imshow(crf_pred_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            # ax5.imshow(gt_mask_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            # ax6.imshow(logits_show, cmap=self.cmap, vmin=0, vmax=self.max_label)

                            ax1.axis('off')
                            ax2.axis('off')
                            # ax3.axis('off')
                            # ax4.axis('off')
                            # ax5.axis('off')
                            # ax6.axis('off')

                            if config['visualization']['train_imshow']:
                                plt.show()
                            if config['visualization']['titles']:
                                figure.suptitle(f"Pred: {verbose_preds[batch_index_to_show]}\nGT label: {batch_verboe_labels[batch_index_to_show]}")  # \nP
                                ax1.set_title(f"Input Image", fontsize=config['visualization']['font_size'])
                                ax2.set_title(f"GT Mask", fontsize=config['visualization']['font_size'])
                                # ax3.set_title(f"Prediction", fontsize=config['visualization']['font_size'])
                                # ax6.set_title(f"Prediction", fontsize=config['visualization']['font_size'])
                                # ax5.set_title(f"GT Mask", fontsize=config['visualization']['font_size'])
                                # if self.use_crf:
                                #     ax4.set_title(f"+CRF Prediction", fontsize=config['visualization']['font_size'])

                            cometml_experiemnt.log_figure(figure_name=f"Validation, Image name:", figure=figure)
                            figure.clear()
                            plt.cla()
                            plt.clf()
                            plt.close('all')
                            plt.close(figure)
                            gc.collect()

        # mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=config['data']['num_classes'])
        # overall_miou = sum(mean_iou)/len(mean_iou)
        # if self.use_crf:
        #     crf_mean_iou, crf_precision, crf_recall = eval_utils.compute_jaccard(crf_preds, targets, num_classes=config['data']['num_classes'])
        #     crf_overall_miou = sum(crf_mean_iou)/len(crf_mean_iou)
        #     print(f"Validation class-wise +CRF mIoU value: \n{np.array(crf_mean_iou)} \noverall mIoU: {crf_overall_miou}")
        #     cometml_experiemnt.log_metric("Validation +CRF mIoU", crf_overall_miou, epoch=epoch+1)

        mean_acc = torch.mean(torch.stack(accuracies))
        # print(f"Validation class-wise mIoU value: \n{np.array(mean_iou)} \noverall mIoU: {overall_miou}")
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch + 1, total_loss / loader.__len__()))
        # cometml_experiemnt.log_metric("Validation mIoU", overall_miou, epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Average Loss", total_loss / loader.__len__(), epoch=epoch + 1)
        cometml_experiemnt.log_metric("Accuracy", mean_acc, epoch=epoch + 1)

        return total_loss / loader.__len__(), mean_acc

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
        best_val_loss, train_loss = np.infty, np.infty
        best_val_mean_iou = 0

        model_save_dir = config['data'][config['location']]['model_save_dir'] + f"{current_path[-1]}_{config['dataset']}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        for epoch in range(0, self.epochs):
            if config['procedures']['train']:
                with cometml_experiment.train():
                    train_loss = self.train(epoch, cometml_experiment)
            if config['procedures']['validate']:
                with cometml_experiment.validate():
                    val_loss, val_mean_iou = self.validate(epoch, cometml_experiment)
            self.scheduler.step()

            if val_mean_iou > best_val_mean_iou:
                # best_train_loss = train_loss
                best_val_mean_iou = val_mean_iou
                model_save_name = f"{current_path[-1]}_epoch_{epoch}_loss_{train_loss}_valmIoU_{val_mean_iou}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"

                if config['procedures']['train']:
                    with open(model_save_dir + "config.yaml", 'w') as file:
                        yaml.dump(config, file)

                    torch.save({'epoch': epoch,
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'loss': train_loss},
                                model_save_dir + model_save_name)

        return train_losses, val_losses, mean_ious_val


if __name__ == "__main__":

    project_root = "/home/native/projects/watch/watch/tasks/rutgers_material_seg/"
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    main_config_path = f"{project_root}/configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    # experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"
    experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"
    # config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]

    experiment_config = utils.config_parser(experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    project_name = f"{current_path[-3]}_{current_path[-1]}"  # _{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
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

    print(config['data']['image_size'])
    # coco_fpath = ub.expandpath(config['data'][config['location']]['coco_json'])
    # dset = kwcoco.CocoDataset(coco_fpath)
    # sampler = ndsampler.CocoSampler(dset)

    # # # print(sampler)
    # number_of_timestamps, h, w = 1, 64, 64
    # window_dims = (number_of_timestamps, h, w) #[t,h,w]
    # input_dims = (h, w)

    # # # channels = 'r|g|b|gray|wv1'
    # # channels = 'r|g|b'
    # channels = 'red|green|blue|nir|swir16|swir22|cirrus'
    # # channels = 'red|green|blue'
    # # channels = 'gray'
    # dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    # print(dataset.__len__())
    # train_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])
    # transformers = {}
    # transformers['image'] = transforms.Compose([transforms.Resize((height, width)),
    #                                 # transforms.ColorJitter(),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))])

    train_dataloader = build_dataset(dataset_name="deepglobe",
                                     root="/media/native/data/data/DeepGlobe/crops/",
                                     batch_size=config['training']['batch_size'],
                                     num_workers=1,
                                     split="train",
                                    #  transforms=transformers,
                                     image_size="300x300",
                                     )

    # model = build_model(model_name = config['training']['model_name'],
    #                     backbone=config['training']['backbone'],
    #                     pretrained=config['training']['pretrained'],
    #                     num_classes=config['data']['num_classes'],
    #                     num_groups=config['training']['gn_n_groups'],
    #                     weight_std=config['training']['weight_std'],
    #                     beta=config['training']['beta'],
    #                     num_channels=config['training']['num_channels'])

    model = SupConResNet(name=config['training']['backbone'])

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
