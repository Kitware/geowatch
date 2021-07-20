import pytorch_lightning as pl

import torch
from torch import nn
from einops.layers.torch import Rearrange
import einops

import torch_optimizer as optim
from torch.optim import lr_scheduler

import torchmetrics as metrics

from watch.tasks.fusion import utils


class TransformerChangeDetector(pl.LightningModule):
    def __init__(self,
                 window_size=8,
                 embedding_size=128,
                 n_layers=4,
                 n_heads=8,
                 dropout=0.0,
                 fc_dim=1024,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 pos_weight=1.):
        super().__init__()
        self.save_hyperparameters()

        layers = [
            # nn.Transformer* expect inputs shaped (sequence, batch, feature)
            Rearrange("b t c (h hs) (w ws) -> b t (c ws hs) h w",
                      hs=self.hparams.window_size,
                      ws=self.hparams.window_size),
            utils.AddPositionalEncoding(2, [1, 3, 4]),
            Rearrange("b t f h w -> b (t h w) f"),
            nn.LazyLinear(embedding_size),
            Rearrange("b s f -> s b f"),
        ] + [
            nn.TransformerEncoderLayer(
                embedding_size, n_heads,
                dim_feedforward=embedding_size,
                dropout=dropout, activation="gelu")
            for _ in range(n_layers)
        ] + [
            Rearrange("s b f -> b s f"),
            nn.Linear(embedding_size, embedding_size),
            #nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
            Rearrange("b s f -> s b f"),
        ]
        self.model = nn.Sequential(*layers)

        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1) * pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })

    @pl.core.decorators.auto_move_data
    def forward(self, images):
        """
        Example:
            >>> from watch.tasks.fusion.methods.transformer import *  # NOQA
            >>> self = TransformerChangeDetector()
            >>> images = torch.zeros(1, 2, 3, 128, 128)
            >>> distance = self(images)
        """
        B, T, C, H, W = images.shape
        feats = self.model(images)
        feats = einops.rearrange(feats,
                                 "(t h w) b f -> b t f h w",
                                 b=B, t=T, f=self.hparams.embedding_size,
                                 h=H // self.hparams.window_size,
                                 w=W // self.hparams.window_size)

        # similarity between neighboring timesteps
        feats = nn.functional.normalize(feats, dim=2)
        similarity = torch.einsum("b t c h w , b t c h w -> b t h w", feats[:, :-1], feats[:, 1:])
        distance = -3.0 * similarity

        distance = nn.functional.interpolate(distance, [H, W], mode="bilinear")
        return distance

    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

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
        changes = labels[:, 1:] != labels[:, :-1]

        # compute predicted and target change masks
        distances = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log("val_" + key, metric(torch.sigmoid(distances), changes), prog_bar=True)

        # compute loss
        loss = self.criterion(distances, changes.float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

        # compute predicted and target change masks
        distances = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log("test_" + key, metric(torch.sigmoid(distances), changes), prog_bar=True)

        # compute loss
        loss = self.criterion(distances, changes.float())
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.99))
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs)
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
