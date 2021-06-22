import torch
from torch import nn
import pytorch_lightning as pl
import torch_optimizer as optim
from torch.optim import lr_scheduler
import einops
import torchmetrics as metrics

from .common import ChangeDetectorBase
from models import unet_blur

class UNetChangeDetector(ChangeDetectorBase):
    def __init__(self, 
                 feature_dim=64, 
                 learning_rate=1e-3, 
                 weight_decay=1e-5, 
                 pos_weight=1.,
                 input_scale=2000.,
                ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
        )
        self.save_hyperparameters()
        
        # simple feature extraction model
        self.model = nn.Sequential(
            nn.LazyConv2d(64, 1),
            unet_blur.UNet(64, self.hparams.feature_dim),
        )
        
    @property
    def preprocessing_step(self):
        return transforms.Compose([
            transforms.ToTensor(),
            utils.Lambda(lambda x: x/self.hparams.input_scale),
        ])
        
    @pl.core.decorators.auto_move_data
    def forward(self, images):
        T = images.shape[1] # how many time steps?
        
        # extract features for each timestep
        feats = torch.stack([
                self.model(images[:,t])
                for t in range(T)
            ], dim=1)
        feats = nn.functional.normalize(feats, dim=2)
        
        # similarity between neighboring timesteps
        similarity = torch.einsum("b t c h w , b t c h w -> b t h w", feats[:,:-1], feats[:,1:])
        distance = -3.0 * similarity

        return distance
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(UNetChangeDetector, UNetChangeDetector).add_model_specific_args(parent_parser)
        parser.add_argument("--feature_dim", default=64, type=int)
        parser.add_argument("--input_scale", default=2000.0, type=float)
        return parent_parser
