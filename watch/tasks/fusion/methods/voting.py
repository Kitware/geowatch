import torch
from torch import nn
import pytorch_lightning as pl
import torch_optimizer as optim
from torch.optim import lr_scheduler
import einops
import torchmetrics as metrics

from ..models import unet_blur


class VotingModel(pl.LightningModule):
    def __init__(self, input_dim, learning_rate=1e-5, weight_decay=1e-5, pos_weight=1.):
        super().__init__()
        self.save_hyperparameters()

        # simple feature extraction model
        self.model = nn.Conv2d(input_dim, 1, 1, bias=False)
        nn.init.constant_(self.model.weight, 1 / input_dim)

        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1) * pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "f1": metrics.F1(),
        })

    @pl.core.decorators.auto_move_data
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
            self.log("val_" + key, metric(torch.sigmoid(distances), changes), prog_bar=True)

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
            self.log("test_" + key, metric(torch.sigmoid(distances), changes), prog_bar=True)

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


class End2EndVotingModel(pl.LightningModule):
    def __init__(self, channel_sets, feature_dim=64, learning_rate=1e-3, weight_decay=1e-5, pos_weight=1.):
        super().__init__()
        self.save_hyperparameters()

        self.models = nn.ModuleDict({
            key: unet_blur.UNet(
                len(channels),
                self.hparams.feature_dim,
            )
            for key, channels in self.hparams.channel_sets.items()
        })

        self.voter = nn.Conv2d(len(channel_sets), 1, 1, bias=False)
        nn.init.constant_(self.voter.weight, 1 / len(channel_sets))

        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1) * pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })

    @pl.core.decorators.auto_move_data
    def forward(self, images):
        B = images.shape[0]
        T = images.shape[1] # how many time steps?

        feats_per_channel_set = {
            key: nn.functional.normalize(
                torch.stack([
                    self.models[key](images[:, t, channels])
                    for t in range(T)
                ], dim=1),
                dim=2,
            )
            for key, channels in self.hparams.channel_sets.items()
        }
        distance_per_channel_set = {
            key: -3.0 * torch.einsum(
                "b t c h w , b t c h w -> b t h w",
                feats[:, :-1],
                feats[:, 1:],
            )
            for key, feats in feats_per_channel_set.items()
        }
        distance_stack = einops.rearrange(
            list(distance_per_channel_set.values()),
            "c b t h w -> (b t) c h w",
        )
        combined_distance = self.voter(distance_stack)
        combined_distance = einops.rearrange(
            combined_distance,
            "(b t) 1 h w -> b t h w",
            b=B,
        )
        return combined_distance, distance_per_channel_set

    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

        combined_distance, distance_per_channel_set = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log(
                f"{key}_combo",
                metric(torch.sigmoid(combined_distance), changes),
                prog_bar=True,
            )
            for channels, distance in distance_per_channel_set.items():
                self.log(
                    f"{key}_{channels}",
                    metric(torch.sigmoid(distance), changes),
                    prog_bar=True,
                )

        combined_loss = self.criterion(combined_distance, changes.float())
        channels_loss = sum([
            self.criterion(distance, changes.float())
            for distance in distance_per_channel_set.values()
        ])

        total_loss = combined_loss + channels_loss
        return total_loss

    def validation_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

        combined_distance, distance_per_channel_set = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log(
                f"val_{key}_combo",
                metric(torch.sigmoid(combined_distance), changes),
                prog_bar=True,
            )
            for channels, distance in distance_per_channel_set.items():
                self.log(
                    f"val_{key}_{channels}",
                    metric(torch.sigmoid(distance), changes),
                    prog_bar=True,
                )

        combined_loss = self.criterion(combined_distance, changes.float())
        channels_loss = sum([
            self.criterion(distance, changes.float())
            for distance in distance_per_channel_set.values()
        ])

        total_loss = combined_loss + channels_loss
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

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
        parser = parent_parser.add_argument_group("MultiChangeDetector")
        parser.add_argument("--feature_dim", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--weight_decay", default=1e-5, type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)
        return parent_parser
