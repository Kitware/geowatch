# FIXME:
# Adds the "modules" subdirectory to the python path.
# See https://gitlab.kitware.com/smart/watch/-/merge_requests/148#note_1050127
# for discussion of how to refactor this in the future.
import geowatch_tpl  # NOQA

import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import pytorch_lightning as pl

from .backbone import get_backbone

from frame_field_learning import data_transforms
from frame_field_learning import local_utils
from frame_field_learning.model_multi import Multi_FrameFieldModel

from medpy.filter.smoothing import anisotropic_diffusion
from scipy import ndimage

dfactor = 25.5

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0


#-------------------------------------------
# Modify the batch_norm layers
#-------------------------------------------

def modify_bn(model, track_running_stats=True, bn_momentum=0.1):
    for m in model.modules():
        for child in m.children():
            if isinstance(child, nn.BatchNorm2d):

                child.momentum = bn_momentum
                child.track_running_stats = track_running_stats

                if track_running_stats is False:
                    child.running_mean = None
                    child.running_var =  None

    return model


#-------------------------------------------------
# Depth/Label/Shadow/Facade Eestimation Module
#-------------------------------------------------

class MultiTaskModel(pl.LightningModule):

    def __init__(
        self,
        batch_size: int = 1,
        checkpoint: str = None,
        config: dict = None,
        test_img_dir: str = None,
        test_img_list: str = None,
        gpus: str = '0',
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.gpus = gpus
        self.checkpoint = checkpoint
        self.batch_size = batch_size

        self.test_img_dir = test_img_dir
        self.test_img_list = test_img_list

        self.config = config

        self.backbone = get_backbone(self.config["backbone_params"])

        train_online_cuda_transform = None
        eval_online_cuda_transform = None

        self.net = Multi_FrameFieldModel(
            self.config,
            backbone=self.backbone,
            train_transform=train_online_cuda_transform,
            eval_transform=eval_online_cuda_transform)

        self.transform =  data_transforms.get_online_cuda_transform(
            self.config,
            augmentations=self.config["data_aug_params"]["enable"])

    def forward(self, x, tta=False):
        return self.net(x, tta)

    def test_step(self, batch, batch_idx):

        out_arr = []
        for i, image in enumerate(batch):
            if isinstance(image, dict):
                gid = image['id']
                # img_info = image
                image = image['imgdata']

            with torch.no_grad():

                image_float = image / 255.0
                mean = np.mean(image_float.reshape(-1, image_float.shape[-1]), axis=0)
                std = np.std(image_float.reshape(-1, image_float.shape[-1]), axis=0)

                batch2 = {
                    "image": torchvision.transforms.functional.to_tensor(image)[None, ...],
                    "image_mean": torch.from_numpy(mean)[None, ...],
                    "image_std": torch.from_numpy(std)[None, ...],
                }

                batch2 = local_utils.batch_to_cuda(batch2)

                pred2, batch2 = self(batch2, tta=True)

                output_depth = pred2['depth'][0, 0, :, :].cpu().data.numpy()
                output_label = pred2['seg'][0, 0, :, :].cpu().data.numpy()

                weighted_depth = dfactor * output_depth

                alpha = 0.9
                weighted_seg = alpha * output_label + (1.0 - alpha) * np.minimum(0.99, weighted_depth / 70.0)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tmp2 = 255 * anisotropic_diffusion(weighted_seg, niter=1, kappa=100, gamma=0.8)
                weighted_final = ndimage.median_filter(tmp2.astype(np.uint8), size=7)

                # Image.fromarray(weighted_final.astype(np.uint8)).save('/output/weighted_final.png')

            out_arr.append((gid, weighted_final))
        return out_arr

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("MultiTaskModel")

        parser.add_argument('--checkpoint', default=None, type=str,
                            help='checkpoint to use for testing')
        parser.add_argument('--config', '--config', default=None, type=str,
                            help='Name of the config file, excluding the .json file extension.')
        parser.add_argument('--test_img_dir', '--test_img_dir', default=None, type=str,
                            help='directory where test images are located')
        parser.add_argument('--test_img_list', '--test_img_list', default=None, type=str,
                            help='list of test images')
        parser.add_argument('--gpus', default='0', type=str,
                            help='GPU')

        return parent_parser
