import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from einops.layers.torch import Rearrange, Reduce
import torch_optimizer as optim
from torch.optim import lr_scheduler

class AddPositionalEncoding(nn.Module):
    def __init__(self, dest_dim, dims_to_encode):
        super().__init__()
        self.dest_dim = dest_dim
        self.dims_to_encode = dims_to_encode
        assert self.dest_dim not in self.dims_to_encode
        
    def forward(self, x):

        inds = [
            slice(0, size) if (dim in self.dims_to_encode) else slice(0, 1)
            for dim, size in enumerate(x.shape)
        ]
        inds[self.dest_dim] = self.dims_to_encode

        encoding = torch.cat(torch.meshgrid([
            torch.linspace(0, 1, x.shape[dim]) if (dim in self.dims_to_encode) else torch.tensor(-1.)
            for dim in range(len(x.shape))
        ]), dim=self.dest_dim)[inds]

        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1
        x = torch.cat([x, encoding.expand(expanded_shape).type_as(x)], dim=self.dest_dim)
        return x
    
class ChangeDetectorBase(pl.LightningModule):
    
    def __init__(self, 
                 learning_rate=1e-3, 
                 weight_decay=0., 
                 pos_weight=1.,
                ):
        super().__init__()
        self.save_hyperparameters()
        
        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })
        
    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:,1:] != labels[:,:-1]
        
        # compute predicted and target change masks
        _, _, H, W = changes.shape
        distances = self(images)
        distances = nn.functional.interpolate(
            distances, 
            [H, W], 
            mode="bilinear")
        
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
        _, _, H, W = changes.shape
        distances = self(images)
        distances = nn.functional.interpolate(
            distances, 
            [H, W], 
            mode="bilinear")
                
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
        _, _, H, W = changes.shape
        distances = self(images)
        distances = nn.functional.interpolate(
            distances, 
            [H, W], 
            mode="bilinear")
                
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
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)
        return parent_parser