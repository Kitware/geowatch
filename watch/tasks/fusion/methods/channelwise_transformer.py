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
from .common import ChangeDetectorBase, AddPositionalEncoding
from models import transformer

class MultimodalTransformerDotProdCD(ChangeDetectorBase):
    
    def __init__(self, 
                 model_name,
                 dropout=0.0, 
                 learning_rate=1e-3, 
                 weight_decay=0., 
                 pos_weight=1.,
                ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
        )
        self.save_hyperparameters()
        
        self.model = getattr(transformer, model_name)(dropout=dropout)

    @pl.core.decorators.auto_move_data
    def forward(self, images):
        feats = self.model(images)
        
        # similarity between neighboring timesteps
        feats = nn.functional.normalize(feats, dim=-1)
        similarity = torch.einsum("b t c h w f , b t c h w f -> b t c h w", feats[:,:-1], feats[:,1:])
        similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        distance = -3.0 * similarity

        return distance
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(MultimodalTransformerDotProdCD, MultimodalTransformerDotProdCD).add_model_specific_args(parent_parser)
        
        parser.add_argument("--model_name", required=True, type=str)
        parser.add_argument("--dropout", default=0.1, type=float)
        return parent_parser

class MultimodalTransformerDirectCD(ChangeDetectorBase):
    
    def __init__(self, 
                 model_name,
                 dropout=0.0, 
                 learning_rate=1e-3, 
                 weight_decay=0., 
                 pos_weight=1.,
                ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
        )
        self.save_hyperparameters()
        
        self.model = nn.Sequential(
            getattr(transformer, model_name)(dropout=dropout),
            nn.LazyLinear(1),
        )

    @pl.core.decorators.auto_move_data
    def forward(self, images):        
        similarity = self.model(images)[:, 1:, ..., 0]
        similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        return similarity
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(MultimodalTransformerDirectCD, MultimodalTransformerDirectCD).add_model_specific_args(parent_parser)
        
        parser.add_argument("--model_name", required=True, type=str)
        parser.add_argument("--dropout", default=0.1, type=float)
        return parent_parser

class MultimodalTransformerSegmentation(pl.LightningModule):
    
    def __init__(self, 
                 n_classes,
                 model_name,
                 dropout=0.0, 
                 learning_rate=1e-3, 
                 weight_decay=0., 
                ):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_model = getattr(transformer, model_name)(dropout=dropout)
        self.predictor = nn.Sequential(
            Reduce("b t c h w f -> b t h w f", "mean"),
            nn.Linear(embedding_size, embedding_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embedding_size, n_classes),
            Rearrange("b t h w f -> b f t h w"),
        )
        
        # criterion and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })

    @pl.core.decorators.auto_move_data
    def forward(self, images):
        return self.predictor(self.feature_model(images))
        
    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        
        # compute predicted and target change masks
        _, _, H, W = labels.shape
        logits = self(images) # b f t h w
        logits = nn.functional.interpolate(
            logits, 
            [H, W], 
            mode="bilinear")
        
        # compute metrics
        for key, metric in self.metrics.items():
            self.log(key, metric(torch.sigmoid(logits), labels), prog_bar=True)
        
        # compute criterion
        loss = self.criterion(logits, labels)
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        
        # compute predicted and target change masks
        _, _, H, W = labels.shape
        logits = self(images)
        logits = nn.functional.interpolate(
            logits, 
            [H, W], 
            mode="bilinear")
                
        # compute metrics
        for key, metric in self.metrics.items():
            self.log("val_"+key, metric(torch.sigmoid(logits), labels), prog_bar=True)
        
        # compute loss
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        
        # compute predicted and target change masks
        _, _, H, W = labels.shape
        logits = self(images)
        logits = nn.functional.interpolate(
            logits, 
            [H, W], 
            mode="bilinear")
                
        # compute metrics
        for key, metric in self.metrics.items():
            self.log("test_"+key, metric(torch.sigmoid(logits), labels), prog_bar=True)
        
        # compute loss
        loss = self.criterion(logits, labels)
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
        parser = parent_parser.add_argument_group("Segmentation")
        
        parser.add_argument("--model_name", required=True, type=str)
        parser.add_argument("--n_classes", required=True, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)
        return parent_parser