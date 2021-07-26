import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
import torch_optimizer as optim
from torch.optim import lr_scheduler
import numpy as np
import ubelt as ub
import netharn as nh

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class ChangeDetectorBase(pl.LightningModule):

    def __init__(self,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 input_stats=None,
                 pos_weight=1.):
        super().__init__()
        self.save_hyperparameters()

        self.input_norms = None
        if input_stats is not None:
            self.input_norms = torch.nn.ModuleDict()
            for key, stats in input_stats.items():
                self.input_norms[key] = nh.layers.InputNorm(**stats)

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

    @profile
    def forward_step(self, batch, with_loss=False, stage='unspecified'):
        """
        Generic forward step used for test / train / validation

        Example:
            >>> from watch.tasks.fusion.methods.common import *  # NOQA
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datasets
            >>> datamodule = datasets.WatchDataModule(
            >>>     train_dataset='special:vidshapes8',
            >>>     num_workers=0, chip_size=128,
            >>>     normalize_inputs=True,
            >>> )
            >>> datamodule.setup('fit')
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))

            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.MultimodalTransformerDotProdCD(
            >>>     model_name='smt_it_joint_p8', input_stats=datamodule.input_stats)
            >>> outputs = self.training_step(batch)
            >>> canvas = datamodule.draw_batch(batch, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        outputs = {}
        outputs['binary_predictions'] = binary_predictions = []

        total_loss = 0
        for item in batch:
            assert len(item['modes']) == 1
            mode_key, mode_val = ub.peek(item['modes'].items())

            mode_val = mode_val.float()
            T, C, H, W = mode_val.shape

            if self.input_norms is not None:
                mode_norm = self.input_norms[mode_key]
                mode_val = mode_norm(mode_val)

            # Because we are not collating we need to add a batch dimension
            images = mode_val[None, ...]

            logits = self(images)
            logits = nn.functional.interpolate(
                logits,
                [H, W],
                mode="bilinear")

            if with_loss:
                changes = item['changes'][None, ...]

                change_prob = logits.sigmoid()[0]
                binary_predictions.append(change_prob)

                # compute criterion
                loss = self.criterion(logits, changes.float())
                total_loss = total_loss + loss

        if with_loss:
            all_pred = torch.stack(binary_predictions)
            all_true = torch.stack([item['changes'] for item in batch])
            # compute metrics
            item_metrics = {}
            for key, metric in self.metrics.items():
                val = metric(all_pred, all_true)
                item_metrics[f'{stage}_{key}'] = val

            outputs['loss'] = total_loss
        return outputs

    @profile
    def training_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='train')
        return outputs

    @profile
    def validation_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='val')
        return outputs

    @profile
    def test_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='test')
        return outputs

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
