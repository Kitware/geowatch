import torch
import torch.nn as nn
from argparse import Namespace
import pytorch_lightning as pl

from .models import UNet, UNet_blur
from .drop0_datasets import drop0_pairs


class time_sort(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.criterion = nn.BCEWithLogitsLoss()

        self.accuracy = pl.metrics.classification.Accuracy()

        self.save_hyperparameters(hparams)

        if self.hparams.backbone == 'unet':
            self.backbone = UNet(self.hparams.in_channels, hparams.feature_dim)
        elif self.hparams.backbone == 'unet_blur':
            self.backbone = UNet_blur(self.hparams.in_channels, hparams.feature_dim)

        self.classifier = self.head(2 * hparams.feature_dim)

        self.accuracy = pl.metrics.Accuracy()

        self.train_data_fpath = self.hparams.train_dataset
        self.val_data_fpath = self.hparams.val_dataset

    def head(self, in_channels):
        return nn.Sequential(
            #nn.Conv2d(in_channels, in_channels // 2, 7, bias=False, padding=3),
            # nn.ReLU(),
            #nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels, 1, 1, bias=False, padding=0),
        )

    def forward(self, image1, image2, date1, date2):
        image1 = self.backbone(image1)
        image2 = self.backbone(image2)

        return image1, image2, date1, date2

    def shared_step(self, batch):
        image1, image2, date1, date2 = batch['image1'], batch['image2'], batch['date1'], batch['date2']
        image1, image2, date1, date2 = self(image1, image2, date1, date2)
        prediction = self.classifier(torch.cat((image1, image2), dim=1))

        labels = torch.tensor([
            tuple(date1[x]) < tuple(date2[x]) for x in range(date1.shape[0])
        ]).float().cuda()
        labels = labels.unsqueeze(1).unsqueeze(1).repeat(1, image1.shape[2], image1.shape[3]).unsqueeze(1)

        loss = self.criterion(prediction, labels)
        accuracy = self.accuracy((prediction > 0.), labels.int())

        output = {
            # 'prediction': prediction,
            # 'labels': labels,
            'accuracy': accuracy,
            'loss': loss,
        }
        return output

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('acc', output['accuracy'])
        self.log('loss', output['loss'])
        return output

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)

        output = {key + "_val": val for key, val in output.items()}
        self.log('val_acc', output['accuracy_val'])
        self.log('val_loss', output['loss_val'])
        return output

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            drop0_pairs(
                coco_dset=self.train_data_fpath,
                sensor=self.hparams.sensor,
                panchromatic=self.hparams.panchromatic,
                video=self.hparams.train_video,
                min_time_step=self.hparams.min_time_step
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            drop0_pairs(
                coco_dset=self.val_data_fpath,
                sensor=self.hparams.sensor,
                panchromatic=self.hparams.panchromatic,
                video=self.hparams.val_video,
                min_time_step=self.hparams.min_time_step
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.step_size,
            gamma=self.hparams.gamma)

        return {'optimizer': opt, 'lr_scheduler': lr_scheduler}
