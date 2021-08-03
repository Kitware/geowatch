import sys
import os
current_path = os.getcwd().split("/")

import matplotlib
import gc
import cv2
import comet_ml
import torch
from scipy import ndimage
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime
import torch.nn.functional as F
import warnings
import yaml
import random

import watch.tasks.rutgers_material_seg.utils.utils as utils
import watch.tasks.rutgers_material_seg.utils.visualization as visualization
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets import build_dataset
import watch.tasks.rutgers_material_seg.utils.eval_utils as eval_utils
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)


class Trainer(object):
    def __init__(self, model: object, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, epochs: int, 
                 optimizer: object, scheduler: object, 
                 test_loader: torch.utils.data.DataLoader =None) -> None:
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

        if len(config['data']['weights'])>0:
            self.class_weights = torch.Tensor(config['data']['weights']).float().to(device)
        
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision
        
        self.cmap = visualization.rand_cmap(nlabels=config['data']['num_classes']+1, type='bright', 
                                            first_color_black=True, last_color_black=True, 
                                            bg_alpha=config['visualization']['bg_alpha'],
                                            fg_alpha=config['visualization']['fg_alpha'])
        
    def train(self, epoch: int, cometml_experiemnt: object) -> float:
        """training single epoch

        Args:
            epoch (int): number of epoch
            cometml_experiemnt (object): comet ml experiment to log the epoch

        Returns:
            float: training loss of that epoch
        """
        total_loss, total_loss_seg = 0, 0
        preds, targets = [], []
        self.model.train()
        print(f"starting epoch {epoch}")
        loader_size = len(self.train_loader)
        iter_visualization = loader_size // config['visualization']['train_visualization_divisor']
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        batch_index_to_show = config['visualization']['batch_index_to_show']
        for batch_index, batch in pbar:
            outputs = batch
            image1, mask = outputs['inputs']['image'], outputs['inputs']['mask']
            
            mask = mask.long().squeeze(1)
            
            class_to_show = max(0,torch.unique(mask)[-1]-1)
            image1 = image1.to(device)
            mask = mask.to(device)
            image_raw = utils.denorm(image1.clone().detach())
            image_name = outputs['visuals']['image_name'][batch_index_to_show]

            batch_size = image1.shape[0]
            output1 = self.model(image1) # torch.Size([B, C+1, H, W])
            output1_interpolated = F.interpolate(output1, size=mask.size()[-2:], 
                                                 mode="bilinear", align_corners=True)

            bs, c, h, w = output1.size()
            masks = F.softmax(output1, dim=1)#.detach()
            
            loss = F.cross_entropy(output1_interpolated, 
                                   mask, 
                                   reduction="mean")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_seg += loss.item()

            masks = F.interpolate(masks, size=mask.size()[-2:], mode="bilinear", align_corners=True)
            pred = masks.max(1)[1].cpu().detach()#.numpy()
            total_loss += loss.item()
        
        cometml_experiemnt.log_metric("Training Loss", total_loss, epoch=epoch+1)
        cometml_experiemnt.log_metric("Segmentation Loss", total_loss_seg, epoch=epoch+1)
        # cometml_experiemnt.log_metric("Training mIoU", overall_miou, epoch=epoch+1)

        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.train_loader.__len__()))

        return total_loss/self.train_loader.__len__()
            
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
        batch_index_to_show = config['visualization']['batch_index_to_show']
        loader = self.val_loader
        loader_size = len(loader)
        iter_visualization = loader_size // config['visualization']['val_visualization_divisor']
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total=len(loader))
            for batch_index,batch in pbar:
                outputs = batch
                image1, mask = outputs['inputs']['image'], outputs['inputs']['mask']
                # class_to_show = (labels1[0]==1).nonzero()[-1] + 1
                
                image1 = image1.to(device)
                image_raw = utils.denorm(image1.clone().detach())
                mask = points_mask1.squeeze(1).to(device)
                
                output = self.model(image1)  ## [B,22,150,150]

                masks = F.softmax(output, dim=1) ## (B, 22, 300, 300)

                masks = self.run_pamr(image_raw, masks.detach())
                masks = F.interpolate(masks, size=points_mask1.size()[-2:], mode="bilinear", align_corners=True)
                
                if self.use_crf:
                    crf_probs = utils.batch_crf_inference(image_raw.detach().cpu(), 
                                                          masks.detach().cpu(),
                                                          t=config['evaluation']['crf_t'], 
                                                          scale_factor=config['evaluation']['crf_scale_factor'], 
                                                          labels=config['evaluation']['crf_labels'])
                    crf_probs = crf_probs.squeeze()
                    crf_pred = crf_probs.max(1)[1]
                    crf_pred[crf_pred==config['data']['num_classes']]=0
                    crf_preds.append(crf_pred)

                masks = self.pseudo_gtmask(masks, cutoff_top=config['pseudo_masks']['val_cutoff'])
                
                pred = masks.max(1)[1].cpu().detach()#.numpy()
                # pred[pred==config['data']['num_classes']+1] = 0 # Used when 0 is ignored, and num_classes+1 is background
                preds.append(pred)
                
                targets.append(mask.cpu())#.numpy())
        
        mean_iou, precision, recall = eval_utils.compute_jaccard(preds, targets, num_classes=config['data']['num_classes'])
        overall_miou = sum(mean_iou)/len(mean_iou)
        if self.use_crf:
            crf_mean_iou, crf_precision, crf_recall = eval_utils.compute_jaccard(crf_preds, targets, num_classes=config['data']['num_classes'])
            crf_overall_miou = sum(crf_mean_iou)/len(crf_mean_iou)
            print(f"Validation class-wise +CRF mIoU value: \n{np.array(crf_mean_iou)} \noverall mIoU: {crf_overall_miou}")
            cometml_experiemnt.log_metric("Validation +CRF mIoU", crf_overall_miou, epoch=epoch+1)
            
        print(f"Validation class-wise mIoU value: \n{np.array(mean_iou)} \noverall mIoU: {overall_miou}")
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU", overall_miou, epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Average Loss",total_loss/loader.__len__(),epoch=epoch+1)
        
        return total_loss/loader.__len__(), overall_miou
    
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
        best_val_loss, best_train_loss, train_loss = np.infty, np.infty, np.infty
        best_val_mean_iou = 0
        model_save_dir = config['data'][config['location']]['model_save_dir']+f"{current_path[-1]}_{config['dataset']}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
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
                    with open(model_save_dir+"config.yaml",'w') as file:
                        yaml.dump(config, file)
                
                    torch.save({'epoch': epoch, 
                                'model': self.model.state_dict(), 
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'loss':train_loss},
                                model_save_dir+model_save_name)
            
        return train_losses, val_losses, mean_ious_val

if __name__== "__main__":

    main_config_path = f"{os.getcwd()}/configs/main.yaml"
    initial_config = utils.load_yaml_as_dict(main_config_path)
    experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"
    
    experiment_config = utils.config_parser(experiment_config_path,experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    project_name = f"{current_path[-3]}_{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment_name = f"SMART_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
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
    
    # device_cpu = torch.device('cpu')
    # print(config['data']['image_size'])
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
    
    train_dataloader = build_dataset(dataset_name=config['data']['name'],
                                     root=config['data'][config['location']]['test_dir'], 
                                     batch_size=config['training']['batch_size'],
                                     num_workers=config['training']['num_workers'],
                                     split="train",
                                     image_size=config['data']['image_size'],
                                     )
    
    validation_dataloader = build_dataset(dataset_name=config['data']['name'],
                                          root=config['data'][config['location']]['test_dir'], 
                                          batch_size=config['training']['batch_size'],
                                          num_workers=config['training']['num_workers'],
                                          split="val",
                                          image_size=config['data']['image_size'],
                                          )
    
    fs_test_loader = build_dataset(dataset_name=config['data']['name'],
                                   root=config['data'][config['location']]['test_dir'], 
                                   batch_size=config['training']['batch_size'],
                                   num_workers=config['training']['num_workers'],
                                   split="test",
                                   image_size=config['data']['image_size'],
                                   )

    model = build_model(model_name = config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes']+1,
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)
    model.to(device)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    
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
                      validation_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=fs_test_loader
                      )
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
