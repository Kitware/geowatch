import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
import torch_optimizer as optim
from torch.optim import lr_scheduler
import numpy as np


class ChangeDetectorBase(pl.LightningModule):

    def __init__(self,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 pos_weight=1.):
        super().__init__()
        self.save_hyperparameters()

        # criterion and metrics
        self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.ones(1) * pos_weight)
        self.metrics = nn.ModuleDict({
            "acc": metrics.Accuracy(),
            "iou": metrics.IoU(2),
            "f1": metrics.F1(),
        })

    @property
    def preprocessing_step(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"].float(), batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

        # compute predicted and target change masks
        _, _, H, W = changes.shape
        distances = self(images)
        distances = nn.functional.interpolate(
            distances,
            [H, W],
            mode="bilinear")

        # compute metrics
        for key, metric in self.metrics.items():
            val = metric(torch.sigmoid(distances), changes)
            self.log(key, val, prog_bar=True)

        # compute criterion
        loss = self.criterion(distances, changes.float())
        return loss

    def validation_step(self, batch, batch_idx=None):
        images, labels = batch["images"].float(), batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

        # compute predicted and target change masks
        _, _, H, W = changes.shape
        distances = self(images)
        distances = nn.functional.interpolate(
            distances,
            [H, W],
            mode="bilinear")

        # compute metrics
        for key, metric in self.metrics.items():
            val = metric(torch.sigmoid(distances), changes)
            self.log("val_" + key, val, prog_bar=True)

        # compute loss
        loss = self.criterion(distances, changes.float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx=None):
        images, labels = batch["images"].float(), batch["labels"]
        changes = labels[:, 1:] != labels[:, :-1]

        # compute predicted and target change masks
        _, _, H, W = changes.shape
        distances = self(images)
        distances = nn.functional.interpolate(
            distances,
            [H, W],
            mode="bilinear")

        # compute metrics
        for key, metric in self.metrics.items():
            val = metric(torch.sigmoid(distances), changes)
            self.log("test_" + key, val, prog_bar=True)

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
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)
        return parent_parser


class SemanticSegmentationBase(pl.LightningModule):

    def __init__(self,
                 learning_rate=1e-3,
                 weight_decay=0.):
        super().__init__()
        self.save_hyperparameters()

        # criterion and metrics
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics = nn.ModuleDict({
            # "acc": metrics.Accuracy(ignore_index=-100),
        })

    @property
    def preprocessing_step(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        import xdev
        xdev.embed()

        # compute predicted and target change masks
        logits = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log(key,
                     metric(torch.softmax(logits, dim=1), labels),
                     prog_bar=True)

        # compute criterion
        loss = self.criterion(logits, labels.long())
        return loss

    def validation_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # compute predicted and target change masks
        logits = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log("val_" + key,
                     metric(torch.softmax(logits, dim=1), labels),
                     prog_bar=True)

        # compute loss
        loss = self.criterion(logits, labels.long())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx=None):
        images, labels = batch["images"], batch["labels"]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # compute predicted and target change masks
        logits = self(images)

        # compute metrics
        for key, metric in self.metrics.items():
            self.log("test_" + key,
                     metric(torch.softmax(logits, dim=1), labels),
                     prog_bar=True)

        # compute loss
        loss = self.criterion(logits, labels.long())
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
        parser = parent_parser.add_argument_group("SemanticSegmentation")
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        return parent_parser
