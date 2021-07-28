import os
import sys
current_path = os.getcwd().split("/")

import matplotlib
import gc
import cv2
import comet_ml
import torch
import datetime
import warnings
import yaml
import math
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
import torchvision.transforms.functional as FT
import watch.tasks.rutgers_material_seg.utils.utils as utils
import watch.tasks.rutgers_material_seg.utils.eval_utils as eval_utils
import watch.tasks.rutgers_material_seg.utils.visualization as visualization
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.datasets import build_dataset
from fast_pytorch_kmeans import KMeans
from skimage.filters import threshold_otsu as otsu
import time
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class Evaluator(object):
    def __init__(self, model: object, 
                 eval_loader: torch.utils.data.DataLoader, 
                 optimizer: object, scheduler: object, 
                 test_with_full_supervision: int =0) -> None:
        """Evaluator class

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
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_label = config['data']['num_classes']
            
    def eval(self) -> tuple:
        """evaluate a single epoch

        Args:

        Returns:
            None
        """
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader))
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
                
                image1 = utils.stad_image(image1)
                image2 = utils.stad_image(image2)
                
                output1, features1 = self.model(image1)  ## [B,22,150,150]
                output2, features2 = self.model(image2)
                
                #TODO:
                    # Add auxiliary channel save
                
                # masks1 = F.softmax(output1, dim=1)#.detach()
                # masks2 = F.softmax(output2, dim=1)#.detach()
                # # masks1 = F.softmax(features1, dim=1)
                # # masks2 = F.softmax(features2, dim=1)
                # # masks1 = self.high_confidence_filter(masks1, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                # # masks2 = self.high_confidence_filter(masks2, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                # pred1 = masks1.max(1)[1].cpu().detach()#.numpy()
                # pred2 = masks2.max(1)[1].cpu().detach()#.numpy()
                # change_detection_prediction = (pred1!=pred2).type(torch.uint8)
                
        
        return total_loss/loader.__len__(), overall_miou
    
    def forward(self) -> tuple:
        """forward pass for all epochs

        Args:
            cometml_experiment (object): comet ml experiment for logging
            world_size (int, optional): for distributed training. Defaults to 8.

        Returns:
            tuple: (train losses, validation losses, mIoU)
        """
        
        if config['procedures']['validate']:
            val_loss, val_mean_iou = self.eval()
        self.scheduler.step()
            
        return

if __name__== "__main__":

    main_config_path = f"./configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    experiment_config_path = f"./configs/{initial_config['dataset']}.yaml"
    
    experiment_config = utils.config_parser(experiment_config_path,experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    project_name = f"{current_path[-3]}_{current_path[-1]}_{config['dataset']}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.set_default_dtype(torch.float32)
    
    device_ids = list(range(torch.cuda.device_count()))
    gpu_devices = ','.join([str(id) for id in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = torch.device('cuda')

    config['device_ids'] = device_ids
    config['devices_used'] = gpu_devices
    
    coco_fpath = ub.expandpath(config['data'][config['location']]['coco_json'])
    dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(dset)

    window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size']) #[t,h,w]
    input_dims = (config['data']['image_size'], config['data']['image_size'])

    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    print(dataset.__len__())
    eval_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])
    
    model = build_model(model_name = config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'],
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'],
                        num_channels=config['training']['num_channels'],
                        out_dim=config['training']['out_features_dim'])
    
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
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(eval_dataloader),
                                                     eta_min = config['training']['learning_rate'])

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
            exit()
        
    evaler = Evaluator(model,
                      eval_dataloader,
                      optimizer,
                      scheduler,
                      test_with_full_supervision=config['training']['test_with_full_supervision']
                      )
    evaler.forward()
