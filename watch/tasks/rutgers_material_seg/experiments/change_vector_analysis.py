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
from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.datasets import build_dataset
from watch.tasks.rutgers_material_seg.models.supcon import SupConResNet
from watch.tasks.rutgers_material_seg.models.losses import SupConLoss
from fast_pytorch_kmeans import KMeans

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
        
        self.max_label = config['data']['num_classes']
        
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision
        
        self.train_second_transform = transforms.Compose([
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                                        transforms.RandomGrayscale(p=0.2),
                                                        ])
        self.crop_size = (11,11)
        
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
        iter_visualization = loader_size // config['visualization']['train_visualization_divisor']
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        batch_index_to_show = config['visualization']['batch_index_to_show']
        for batch_index, batch in pbar:
            # if batch_index < 75:
            #     continue
            random_crop = transforms.RandomCrop(self.crop_size)
            outputs = batch
            images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
            image_name = outputs['tr'].data[0][batch_index_to_show]['gids']
            original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
            mask = torch.stack(mask)
            mask = mask.long().squeeze(1)

            bs, c, t, h, w = images.shape
            image1 = images[:,:,0,:,:]
            image2 = images[:,:,1,:,:]
            image_differences = torch.sqrt((image1 - image2) * (image1 - image2))
            image1 = (image1 - image1.min()) / (image1.max() - image1.min())
            image2 = (image2 - image2.min()) / (image2.max() - image2.min())
            image_differences = (image_differences - image_differences.min()) / (image_differences.max() - image_differences.min())
            image_differences = image_differences.numpy().squeeze()
            image_differences = np.mean(image_differences, axis=0, keepdims=True)
            image_differences_binary = (image_differences > 0.2).astype(int)
            image_differences = np.transpose(image_differences,(1,2,0))
            image_differences_binary = np.transpose(image_differences_binary,(1,2,0))
            image1_imshow = np.transpose(image1.squeeze(),(1,2,0))
            image2_imshow = np.transpose(image2.squeeze(),(1,2,0))
            figure = plt.figure()
            ax1 = figure.add_subplot(1,5,1)
            ax2 = figure.add_subplot(1,5,2)
            ax3 = figure.add_subplot(1,5,3)
            ax4 = figure.add_subplot(1,5,4)
            ax5 = figure.add_subplot(1,5,5)
            ax1.imshow(image1_imshow[:,:,:3])
            ax2.imshow(image2_imshow[:,:,:3])
            ax3.imshow(image_differences[:,:,:3])
            ax4.imshow(image_differences_binary)
            ax5.imshow(mask.squeeze()[0,:,:])
            plt.show()

        return 
            
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
        preds, crf_preds, targets  = [], [], []
        accuracies = 0
        running_ap = 0.0
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
            for batch_index,batch in pbar:
                outputs = batch
                images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                original_width, original_height = outputs['tr'].data[0][batch_index_to_show]['space_dims']
                image_name = str(outputs['tr'].data[0][batch_index_to_show]['gids'][0])
                
                mask = torch.stack(mask)
                mask = mask.long().squeeze(1)
                bs, c, t, h, w = images.shape
                image1 = images[:,:,0,:,:]
                image2 = images[:,:,1,:,:]
                mask1 = mask[:,0,:,:]
                mask2 = mask[:,1,:,:]
                
                images = images.to(device)
                image1 = image1.to(device)
                image2 = image2.to(device)
                mask = mask.to(device)
                
                output1, features1 = self.model(image1)  ## [B,22,150,150]
                output2, features2 = self.model(image2)
                
                masks1 = F.softmax(output1, dim=1)#.detach()
                masks2 = F.softmax(output2, dim=1)#.detach()
                # masks = F.interpolate(masks, size=mask.size()[-2:], mode="bilinear", align_corners=True)
                # masks = self.high_confidence_filter(masks, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                pred1 = masks1.max(1)[1].cpu().detach()#.numpy()
                pred2 = masks2.max(1)[1].cpu().detach()#.numpy()
                change_detection_prediction = (pred1 != pred2).type(torch.uint8)
                
                preds.append(change_detection_prediction)
                mask1[mask1 == -1] = 0
                targets.append(mask1.cpu())#.numpy())

        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=2)
        mean_precision = sum(precision) / len(precision)
        mean_recall = sum(recall) / len(recall)
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        overall_miou = sum(mean_iou) / len(mean_iou)
        if self.use_crf:
            crf_mean_iou, crf_precision, crf_recall = eval_utils.compute_jaccard(crf_preds, targets, num_classes=config['data']['num_classes'])
            crf_overall_miou = sum(crf_mean_iou) / len(crf_mean_iou)
            print(f"Validation class-wise +CRF mIoU value: \n{np.array(crf_mean_iou)} \noverall mIoU: {crf_overall_miou}")
            cometml_experiemnt.log_metric("Validation +CRF mIoU", crf_overall_miou, epoch=epoch + 1)
            
        print(f"Validation class-wise mIoU value: \n{np.array(mean_iou)} \noverall mIoU: {overall_miou}")
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch + 1, total_loss / loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU", overall_miou, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation precision", mean_precision, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation recall", mean_recall, epoch=epoch + 1)
        cometml_experiemnt.log_metric("Validation f1_score", f1_score, epoch=epoch + 1)
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
        best_val_mean_iou = 0
        
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

            if val_mean_iou > best_val_mean_iou:
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
    
    print(config['data']['image_size'])
    coco_fpath = ub.expandpath(config['data'][config['location']]['coco_json'])
    dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(dset)

    # # print(sampler)
    number_of_timestamps, h, w = 2, 400, 400
    window_dims = (number_of_timestamps, h, w) #[t,h,w]
    input_dims = (h, w)

    # channels = 'B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A'
    channels = 'B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A'
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
