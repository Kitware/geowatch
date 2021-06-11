import pytorch_lightning as pl

import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
import einops

import torch_optimizer as optim
from torch.optim import lr_scheduler

from torchvision import datasets, transforms
from torch.utils import data
import kwcoco
import ndsampler

import torchmetrics as metrics
from .common import AddPositionalEncoding, ResidualLayer, KthOutput, MultiheadSelfAttention

def new_attention_layer(embedding_size, n_heads, **kwargs): 
    return ResidualLayer(
        nn.Sequential(
            nn.LayerNorm(embedding_size),
            KthOutput(
                MultiheadSelfAttention(embedding_size, n_heads, **kwargs), 
                k=0),
        ))

def new_mlp_layer(embedding_size, dropout, **kwargs):
    return ResidualLayer(
        nn.Sequential(
            nn.Linear(embedding_size, embedding_size, **kwargs),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size, **kwargs),
        ))

class AxialTransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        axes, 
        embedding_size, 
        n_heads, 
        dropout=0.,
        default_shape=["batch", "feature", "time", "mode", "height", "width"],
        feature_axis="feature",
        batch_axis="batch",
    ):
        super().__init__()
        self.axes = axes
        self.default_shape = default_shape
        self.feature_axis = feature_axis
        self.batch_axis = batch_axis
        self.default_shape_str = " ".join(default_shape)
        
        self.attention_modules = nn.ModuleDict({
            " ".join(axis): new_attention_layer(embedding_size, n_heads)
            for axis in axes
        })
        self.mlp = new_mlp_layer(embedding_size, dropout)
        
    def forward(self, x):
        shape_dict = dict(zip(self.default_shape, x.shape))
        
        previous_axial_shape = self.default_shape_str
        for axis in self.axes:
            if not isinstance(axis, (list, tuple)):
                axis = [axis]
                
            sequence_axes = " ".join(axis)
            batch_axes = " ".join([a for a in self.default_shape if (a == self.batch_axis or a not in axis) and a != self.feature_axis])
            axial_shape = f"({sequence_axes}) ({batch_axes}) {self.feature_axis}"
            
            x = einops.rearrange(x, f"{previous_axial_shape} -> {axial_shape}", **shape_dict)
            x = self.attention_modules[" ".join(axis)](x)
            
            previous_axial_shape = axial_shape
                
        sequence_axes = " ".join([a for a in self.default_shape if a not in (self.batch_axis, self.feature_axis)])
        axial_shape = f"({sequence_axes}) {self.batch_axis} {self.feature_axis}"

        x = einops.rearrange(x, f"{previous_axial_shape} -> {axial_shape}", **shape_dict)
        x = self.mlp(x)
        x = einops.rearrange(x, f"{axial_shape} -> {self.default_shape_str}", **shape_dict)
        return x

class _TransformerChangeDetector(pl.LightningModule):
    
    def transformer_layer(self, **kwargs):
        raise NotImplemented()
    
    def __init__(self, 
                 window_size=8, 
                 embedding_size=128, 
                 n_layers=4, 
                 n_heads=8, 
                 dropout=0.0, 
                 fc_dim=1024, 
                 learning_rate=1e-3, 
                 weight_decay=0., 
                 pos_weight=1.,
                ):
        super().__init__()
        self.save_hyperparameters()
        
        layers = [
            # nn.Transformer* expect inputs shaped (sequence, batch, feature)
            Rearrange("b t c (h hs) (w ws) -> b t c h w (ws hs)",
                      hs=self.hparams.window_size, 
                      ws=self.hparams.window_size),
            AddPositionalEncoding(5, [1, 2, 3, 4]),
            nn.LazyLinear(embedding_size),
            Rearrange("b t c h w f -> b f t c h w"),
        ] + [
            self.transformer_layer(embedding_size=embedding_size, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.model = nn.Sequential(*layers)
        
        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })

    def forward(self, images):
        B, T, C, H, W = images.shape
        feats = self.model(images) # b f t c h w
        
        # similarity between neighboring timesteps
        feats = nn.functional.normalize(feats, dim=1)
        similarity = torch.einsum("b f t c h w , b f t c h w -> b t c h w", feats[:,:,:-1], feats[:,:,1:])
        similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        distance = -3.0 * similarity

        distance = nn.functional.interpolate(distance, [H, W], mode="bilinear")
        return distance
        
    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:,1:] != labels[:,:-1]
        
        # compute predicted and target change masks
        distances = self(images)
        
        # compute metrics
        for key, metric in self.metrics.items():
            self.log(key, metric(torch.sigmoid(distances), changes), prog_bar=True)
        
        # compute criterion
        loss = self.criterion(distances, changes.float())
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:,1:] != labels[:,:-1]
        
        # compute predicted and target change masks
        distances = self(images)
                
        # compute metrics
        for key, metric in self.metrics.items():
            self.log("val_"+key, metric(torch.sigmoid(distances), changes), prog_bar=True)
        
        # compute loss
        loss = self.criterion(distances, changes.float())
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:,1:] != labels[:,:-1]
        
        # compute predicted and target change masks
        distances = self(images)
                
        # compute metrics
        for key, metric in self.metrics.items():
            self.log("test_"+key, metric(torch.sigmoid(distances), changes), prog_bar=True)
        
        # compute loss
        loss = self.criterion(distances, changes.float())
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.RAdam(
                self.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.99),
            )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ChangeDetector")
        
        parser.add_argument("--window_size", default=8, type=int)
        parser.add_argument("--embedding_size", default=64, type=int)
        parser.add_argument("--n_layers", default=4, type=int)
        parser.add_argument("--n_heads", default=8, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--fc_dim", default=1024, type=int)
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)
        return parent_parser
    
class JointTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("time", "mode", "height", "width"),
            ],
            **kwargs,
        )

class SpaceTimeModeTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("height", "width"),
                ("time",), 
                ("mode",),
            ],
            **kwargs,
        )

class SpaceModeTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("height", "width"),
                ("mode",),
            ],
            **kwargs,
        )

class SpaceTimeTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("height", "width"),
                ("time",), 
            ],
            **kwargs,
        )

class TimeModeTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("time",), 
                ("mode",),
            ],
            **kwargs,
        )

class SpaceTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("height", "width"),
            ],
            **kwargs,
        )

class AxialTransformerChangeDetector(_TransformerChangeDetector):
    def transformer_layer(self, **kwargs):
        return AxialTransformerEncoderLayer(
            axes=[
                ("height",), 
                ("width",),
                ("time",), 
                ("mode",),
            ],
            **kwargs,
        )