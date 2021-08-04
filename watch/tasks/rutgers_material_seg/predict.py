import os
# import sys
# import matplotlib
# import gc
# import cv2
# import comet_ml
import torch
import datetime
# import warnings
# import yaml
# import math
import random
import kwcoco
import kwimage
import ndsampler
# import matplotlib.pyplot as plt
import numpy as np
import ubelt as ub
import torch.optim as optim
# import torch.nn.functional as F
# from scipy import ndimage
from torch import nn
from tqdm import tqdm
# from torchvision import transforms
# import torchvision.transforms.functional as FT
import watch.tasks.rutgers_material_seg.utils.utils as utils
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
import kwarray
# from watch.tasks.rutgers_material_seg.datasets import build_dataset
# from kwarray.util_slider import *
# import time
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

current_path = os.getcwd().split("/")

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

device = None
config = None


class Evaluator(object):
    def __init__(self, model: object,
                 eval_loader: torch.utils.data.DataLoader,
                 optimizer: object,
                 scheduler: object,
                 coco_dataset: object,
                 log_features=True) -> None:
        """Evaluator class

        Args:
            model (object): trained or untrained model
            eval_loader (torch.utils.data.DataLader): loader with evaluation data
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
        """

        self.model = model
        self.use_crf = config['evaluation']['use_crf']
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_label = config['data']['num_classes']
        self.coco_dataset = coco_dataset
        self.log_features = log_features

    def diff(self, li1, li2):
        # return list(set(li1) - set(li2)) + list(set(li2) - set(li1))
        return list(set(li2) - set(li1))

    def eval(self, save_root="/media/native/data/data/smart_watch_dvc/drop0_rutgers_features") -> tuple:
        """evaluate a single epoch

        Args:

        Returns:
            None
        """
        stitcher_dict = {}
        current_gids = []
        previous_gids = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader))
            for batch_index, batch in pbar:
                outputs = batch
                images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                original_width, original_height = outputs['tr'].data[0][0]['space_dims']

                # print(outputs['tr'].data[0])

                mask = torch.stack(mask)
                mask = mask.long().squeeze(1)

                bs, c, t, h, w = images.shape
                image1 = images[:, :, 0, :, :]
                image2 = images[:, :, 1, :, :]
                mask1 = mask[:, 0, :, :]  # NOQA
                mask2 = mask[:, 1, :, :]  # NOQA

                images = images.to(device)
                image1 = image1.to(device)
                image2 = image2.to(device)
                mask = mask.to(device)

                image1 = utils.stad_image(image1)
                image2 = utils.stad_image(image2)

                output1, features1 = self.model(image1)  # [B,22,150,150]
                output2, features2 = self.model(image2)

                bs, c, h, w = output1.shape
                output1_to_save = output1.permute(0, 2, 3, 1).cpu().detach().numpy()
                output2_to_save = output2.permute(0, 2, 3, 1).cpu().detach().numpy()
                if self.log_features:

                    for b in range(bs):
                        if len(current_gids) == 0:
                            current_gids = outputs['tr'].data[0][b]['gids']
                        else:
                            previous_gids = current_gids
                            current_gids = outputs['tr'].data[0][b]['gids']
                            mutually_exclusive = self.diff(current_gids, previous_gids)
                            for gid in mutually_exclusive:

                                recon = stitcher_dict[gid].finalize()
                                stitcher_dict.pop(gid)

                                save_path = f"{save_root}/{gid}.tiff"
                                kwimage.imwrite(save_path, recon, backend='gdal', space=None)

                                aux_height, aux_width = recon.shape[0:2]
                                img = self.coco_dataset.index.imgs[gid]
                                warp_aux_to_img = kwimage.Affine.scale(
                                    (img['width'] / aux_width, img['height'] / aux_height))

                                aux = {
                                    'file_name': save_path,
                                    'height': aux_height,
                                    'width': aux_width,
                                    'channels': config['data']['num_classes'],
                                    'warp_aux_to_img': warp_aux_to_img.concise(),
                                }

                                auxiliary = img.setdefault('auxiliary', [])
                                auxiliary.append(aux)
                                self.coco_dataset._invalidate_hashid()

                        for gid, output in zip(current_gids, [output1_to_save[b, :, :, :], output2_to_save[b, :, :, :]]):

                            if gid not in stitcher_dict.keys():
                                stitcher_dict[gid] = kwarray.Stitcher(
                                    (*outputs['tr'].data[0][b]['space_dims'], config['data']['num_classes']))
                            slice = outputs['tr'].data[0][b]['space_slice']
                            stitcher_dict[gid].add(slice, output)
                # masks1 = F.softmax(output1, dim=1)#.detach()
                # masks2 = F.softmax(output2, dim=1)#.detach()
                # # masks1 = F.softmax(features1, dim=1)
                # # masks2 = F.softmax(features2, dim=1)
                # # masks1 = self.high_confidence_filter(masks1, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                # # masks2 = self.high_confidence_filter(masks2, cutoff_top=config['high_confidence_threshold']['val_cutoff'])
                # pred1 = masks1.max(1)[1].cpu().detach()#.numpy()
                # pred2 = masks2.max(1)[1].cpu().detach()#.numpy()
                # change_detection_prediction = (pred1!=pred2).type(torch.uint8)

        # export predictions to a new kwcoco file
        dataset_save_path = f"{save_root}/rutgers_features.kwcoco.json"
        self.coco_dataset.dump(dataset_save_path, newlines=True)

        return

    def forward(self) -> tuple:
        """forward pass for all epochs

        Args:
            cometml_experiment (object): comet ml experiment for logging
            world_size (int, optional): for distributed training. Defaults to 8.

        Returns:
            tuple: (train losses, validation losses, mIoU)
        """

        if config['procedures']['validate']:
            self.eval()
        return


def main():
    # FIXME: no globals!
    global device
    global config
    main_config_path = "./configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    experiment_config_path = f"./configs/{initial_config['dataset']}.yaml"

    experiment_config = utils.config_parser(experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # project_name = f"{current_path[-3]}_{current_path[-1]}_{config['dataset']}"  # _{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"

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

    window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
    input_dims = (config['data']['image_size'], config['data']['image_size'])

    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(eval_dataloader),
                                                     eta_min=config['training']['learning_rate'])

    if not config['training']['resume']:
        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
            exit()

    evaler = Evaluator(
        model,
        eval_dataloader,
        optimizer,
        scheduler,
        dset
    )
    evaler.forward()

if __name__ == "__main__":
    main()
