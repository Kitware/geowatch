import torch
from torch import nn
import pytorch_lightning as pl
import torch_optimizer as optim
from torch.optim import lr_scheduler
import einops
import torchmetrics as metrics

from models import unet_blur

class VotingModel(pl.LightningModule):
    def __init__(self, input_dim, learning_rate=1e-5, weight_decay=1e-5, pos_weight=1.):
        super().__init__()
        self.save_hyperparameters()
        
        # simple feature extraction model
        self.model = nn.Conv2d(input_dim, 1, 1, bias=False)
        nn.init.constant_(self.model.weight, 1/input_dim)
        
        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "f1": metrics.F1(),
        })
        
    def forward(self, x):
        B = x.shape[0]
        x = einops.rearrange(x, "b t c h w -> (b t) c h w")
        y = self.model(x)
        y = einops.rearrange(y, "(b t) 1 h w -> b t h w", b=B)
        return y
        
    def training_step(self, batch, batch_idx=None):
        images, changes = batch["images"], batch["labels"]
        changes = changes + 1
        
        # compute predicted and target change masks
        distances = self(images)
        
        # compute metrics
        for key, metric in self.metrics.items():
            self.log(key, metric(torch.sigmoid(distances), changes), prog_bar=True)
        
        # compute criterion
        loss = self.criterion(distances, changes.float())
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        images, changes = batch["images"], batch["labels"]
        changes = changes + 1
        
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
        images, changes = batch["images"], batch["labels"]
        changes = changes + 1
        
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
                self.model.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.99),
            )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VotingModel")
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--weight_decay", default=1e-5, type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)
        return parent_parser
