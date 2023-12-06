import os
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from datetime import date
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from .utils import setup_python_logging
from .models import UNet, UNet_blur


class space7_sort(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.criterion = nn.BCEWithLogitsLoss()

        self.crop_size = hparams.crop_size

        self.accuracy = pl.metrics.classification.Accuracy()

        self.save_hyperparameters(hparams)

        if self.hparams.backbone == 'unet':
            self.backbone = UNet(3, hparams.feature_dim)
        elif self.hparams.backbone == 'unet_blur':
            self.backbone = UNet_blur(3, hparams.feature_dim)

        self.classifier = self.head(2 * hparams.feature_dim)

        self.accuracy = pl.metrics.Accuracy()

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

        date1 = torch.stack(date1).permute(1, 0)
        date2 = torch.stack(date2).permute(1, 0)

        labels = torch.tensor([
            tuple(date1[x]) < tuple(date2[x]) for x in range(date1.shape[0])
        ]).float().cuda()
        labels = labels.unsqueeze(1).unsqueeze(1).repeat(1, self.crop_size, self.crop_size).unsqueeze(1)

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
        from geowatch.tasks.uky_temporal_prediction.spacenet.datasets import S7_sort
        return torch.utils.data.DataLoader(
            S7_sort(train=True,
                    normalize=self.hparams.normalize,
                    crop_size=[self.hparams.crop_size, self.hparams.crop_size]),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers
        )

    def val_dataloader(self):
        from geowatch.tasks.uky_temporal_prediction.spacenet.datasets import S7_sort
        return torch.utils.data.DataLoader(
            S7_sort(train=False, normalize=self.hparams.normalize,
                    crop_size=[self.hparams.crop_size, self.hparams.crop_size]),
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


def main(args):
    if isinstance(args, dict):
        args = Namespace(**args)

    log_dir = '{}/{}/normalize_{}/{}'.format(
        args.save_dir,
        'space7_sort/',
        args.normalize,
        str(date.today()),
        )
    exp_name = 'default'
    logger = TensorBoardLogger(log_dir, name=exp_name)

    setup_python_logging(logger.log_dir)

    model = space7_sort(hparams=args)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=2)
    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=.0003)
    parser.add_argument('--gamma', type=float, default=.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--save_dir', type=str, default='logs/')
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--single_unet', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--backbone', help='choose from unet, unet_blur', default='unet_blur')
    parser.set_defaults(
        gpus=1,
        terminate_on_nan=True,
        check_val_every_n_epochs=1,
        log_every_n_steps=20,
        flush_logs_every_n_steps=20,
        single_unet=True,
        normalize=False
        )

    args = parser.parse_args()
    args.default_save_path = os.path.join(args.save_dir, "logs")

    main(args)
