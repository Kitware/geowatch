import torch
from torch import nn
import pytorch_lightning as pl
import torch_optimizer as optim
from torch.optim import lr_scheduler
import einops
import torchmetrics as metrics

from models import unet_blur

class ChangeDetector(pl.LightningModule):
    def __init__(self, input_dim=13, feature_dim=64, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # simple feature extraction model
        self.model = unet_blur.UNet(self.hparams.input_dim, self.hparams.feature_dim)
        
        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })
        
    def forward(self, images):
        T = images.shape[1] # how many time steps?
        
        # extract features for each timestep
        feats = torch.stack([
                self.model(images[:,t])
                for t in range(T)
            ], dim=1)
        
        # distance between neighboring timesteps
        diffs = feats[:,1:] - feats[:,:-1]
        norms = diffs.norm(dim=2)
        
        return norms
    
    def training_step(self, batch, batch_idx=None):
        images, changes = batch["images"], batch["changes"]
        
        # compute predicted and target change masks
        norms = self(images)
        
        # compute metrics
        for key, metric in self.metrics.items():
            self.log(key, metric(torch.sigmoid(norms), changes), prog_bar=True)
        
        # compute criterion
        loss = self.criterion(norms, changes.float())
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        images, changes = batch["images"], batch["changes"]
        
        # compute predicted and target change masks
        norms = self(images)
                
        # compute metrics
        for key, metric in self.metrics.items():
            self.log("val_"+key, metric(torch.sigmoid(norms), changes), prog_bar=True)
        
        # compute loss
        loss = self.criterion(norms, changes.float())
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ChangeDetector")
        parser.add_argument("--feature_dim", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        return parent_parser
