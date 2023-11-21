import torch
import torch.nn as nn
from datetime import date
import random
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np


from .utils.attention_unet import attention_unet
from .data.datasets import kwcoco_dataset, SpaceNet7, Onera


class change(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        ### Define backbone network.
        if hparams.pretrained_checkpoint:
            if hparams.pretrained_multihead:
                pretrained_model = self.load_from_checkpoint(hparams.pretrained_checkpoint)
            else:
                pretrained_model = self.load_from_checkpoint(hparams.pretrained_checkpoint)

            if hparams.pretrained_encoder_only:
                self.backbone.encoder = pretrained_model.backbone.encoder
            else:
                self.backbone = pretrained_model.backbone
        else:
            self.backbone = attention_unet(hparams.num_channels, hparams.feature_dim, pos_encode=hparams.positional_encoding, num_attention_layers=hparams.num_attention_layers, mode=hparams.positional_encoding_mode)

        ##### define dataset
        if hparams.trainset == 'kwcoco':
            self.trainset = kwcoco_dataset(hparams.train_kwcoco, hparams.sensor, hparams.bands, hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images)
            self.valset = kwcoco_dataset(hparams.val_kwcoco, hparams.sensor, hparams.bands, hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images)
        elif hparams.trainset == 'spacenet':
            self.trainset = SpaceNet7(hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images, train=True)
            self.valset = SpaceNet7(hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images, train=False)
        elif hparams.trainset == 'onera':
            self.trainset = Onera(train=True, patch_size=hparams.patch_size, num_channels=hparams.num_channels)
            self.valset = Onera(train=False, num_channels=hparams.num_channels)
            assert hparams.num_images == 2

        if hparams.binary:
            num_classes = 2
            weight = torch.FloatTensor([1, hparams.pos_class_weight])
        else:
            num_classes = 6
            weight = torch.FloatTensor([0, 1, 1, 1, 1, 1])

        self.criterion = nn.NLLLoss(weight=weight)
        self.save_hyperparameters(hparams)

        self.classifier = self.head(2 * hparams.feature_dim, num_classes)

    def head(self, in_channels, out_channels, kernel_size=3, dilation=1):
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation, bias=False, padding=dilation),
                             nn.GroupNorm(num_groups=in_channels // 8, num_channels=in_channels),
                             nn.ReLU(),
                             nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation, bias=False, padding=dilation),
                             nn.GroupNorm(num_groups=in_channels // 8, num_channels=in_channels),
                             nn.ReLU(),
                             nn.Conv2d(in_channels, out_channels, 1, bias=False, padding=0),
                             nn.LogSoftmax(dim=1))

    def forward(self, x, positions=None):
        predictions = self.backbone(x, positions)
        #         _, predicted_class = torch.max(predictions, dim=2)
        return {
                'predictions': predictions,
                #                 'predicted_class': predicted_class
               }

    def shared_step(self, batch):
        images = [batch[key] for key in batch if key[:5] == 'image']

        if self.hparams.change:
            change_map = batch['change_map'].to(self.device)
        else:
            label = random.choice([0, 1])
            change_map = label * torch.ones_like(batch['change_map']).to(self.device)
            if not label:
                images.reverse()

        images = torch.stack(images, dim=1).to(self.device)

        if self.hparams.positional_encoding:
            positions = batch['time_steps'].to(self.device)
        else:
            positions = None

        forward = self.forward(images, positions)
        predictions = forward['predictions']

        predictions = self.classifier(torch.cat([predictions[:, 0, :, :, :], predictions[:, 1, :, :, :]], dim=1))

        loss = self.criterion(predictions, change_map.long())

        output = {
                    'prediction_map': predictions,
                    'targets' : change_map,
                    'loss': loss
                }

        return output

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('validation_loss', output['loss'])
        return output['loss']

    def validation_epoch_end(self, outputs):
        if self.hparams.change:
            epoch_test_loss, epoch_test_accuracy, cl_acc, pr_rec = self.run_test(loader=self.val_dataloader)
            epoch_test_nochange_accuracy = cl_acc[0]
            epoch_test_change_accuracy = cl_acc[1]
            epoch_test_precision = pr_rec[0]
            epoch_test_recall = pr_rec[1]
            epoch_test_Fmeasure = pr_rec[2]
            self.log('val_epoch_accuracy', epoch_test_accuracy)
            self.log('val_epoch_accuracy_change', epoch_test_change_accuracy)
            self.log('val_epoch_accuracy_no_change', epoch_test_nochange_accuracy)
            self.log('val_epoch_precision', epoch_test_precision)
            self.log('val_epoch_recall', epoch_test_recall)
            self.log('val_epoch_f1', epoch_test_Fmeasure)

    def run_test(self, loader):
        self.eval()
        tot_loss = 0
        tot_count = 0

        n = 2
        class_correct = list(0. for i in range(n))
        class_total = list(0. for i in range(n))
        class_accuracy = list(0. for i in range(n))

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for batch in loader():
            images = [batch[key] for key in batch if key[:5] == 'image']
            images = torch.stack(images, dim=1).to(self.device)
            output = self.shared_step(batch)

            segmentations = output['targets']
            output = output['prediction_map']

            loss = self.criterion(output, segmentations.long())

            tot_loss += loss.data * np.prod(segmentations.size())
            tot_count += np.prod(segmentations.size())

            _, predicted = torch.max(output.data, 1)

            c = (predicted.int() == segmentations.data.int())

            where_no_change = (0 == segmentations.data.int())
            class_correct[0] += torch.sum(c[where_no_change])
            class_total[0] += torch.sum(where_no_change)
            where_change = (1 == segmentations.data.int())
            class_correct[1] += torch.sum(c[where_change])
            class_total[1] += torch.sum(where_change)

            pr = (predicted.int() > 0).cpu().numpy()
            gt = (segmentations.data.int() > 0).cpu().numpy()

            tp += np.logical_and(pr, gt).sum()
            tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
            fp += np.logical_and(pr, np.logical_not(gt)).sum()
            fn += np.logical_and(np.logical_not(pr), gt).sum()

        net_loss = tot_loss / tot_count
        net_accuracy = 100 * (tp + tn) / tot_count

        for i in range(n):
            class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f_meas = 2 * prec * rec / (prec + rec)
        prec_nc = tn / (tn + fn)
        rec_nc = tn / (tn + fp)

        pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

        return net_loss, net_accuracy, class_accuracy, pr_rec

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset,
                                           batch_size=1,
                                           num_workers=self.hparams.workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.step_size, gamma=self.hparams.lr_gamma)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.step_size)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


def main(args):
    if isinstance(args, dict):
        args = Namespace(**args)

    if args.change:
        task = 'change'
    else:
        task = 'before_after'

#     if args.pretrained_multihead and args.pretrained_checkpoint:
#         mh = 'pretrained_multihead'
#     elif args.pretrained_checkpoint:
#         mh = 'sort'
#     else:
#         mh = 'no_pretrain'

    if args.positional_encoding:
        mode = args.positional_encoding_mode
    else:
        mode = 'none'

    log_dir = '{}/{}/{}/{}/{}/{}/{}'.format(
        args.save_dir,
        args.trainset,
        task,
        'Attention_layers:' + str(args.num_attention_layers),
        'Position:' + str(args.positional_encoding),
        mode,
        str(date.today()),
        )

    logger = TensorBoardLogger(log_dir)

    model = change(hparams=args)

    checkpoint_callback = ModelCheckpoint(monitor='val_epoch_f1', mode='max', save_top_k=1)
    lr_logger = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=logger,
                                            callbacks=[checkpoint_callback, lr_logger],
                                            log_every_n_steps=30,
                                            check_val_every_n_epoch=args.check_val_every_n_epoch)
    trainer.fit(model)


if __name__ == '__main__':

    parser = ArgumentParser()

    ###train hyperparameters
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=.0001)
    parser.add_argument('--save_dir', default='geowatch/tasks/invariants/logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=64)
#     parser.add_argument('--drop_rate', type=float, default=.2)

    ###head network
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=1)

    ###dataset
    parser.add_argument('--trainset', type=str, help='Choose from: spacenet, onera, or kwcoco.', default='onera')
    parser.add_argument('--valset', type=str, help='Choose from: spacenet, onera, or kwcoco. If blank, valset will correspond to chosen trainset.', default='onera')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=10)

    ### kwcoco arguments
    parser.add_argument('--train_kwcoco', type=str, default='')
    parser.add_argument('--val_kwcoco', type=str, default='')
    parser.add_argument('--sensor', type=str, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, nargs='+', default=['shared'])

    ### spacenet arguments
    parser.add_argument('--remove_clouds', help='spacenet specific argument', action='store_true')
    parser.add_argument('--normalize_spacenet', help='spacenet specific argument', action='store_true')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)

    ### pretraining arguments
    parser.add_argument('--pretrained_checkpoint', type=str, help='path to pretrained checkpoint. Leave blank for change detection training without pretraining.', default='')
    parser.add_argument('--pretrained_multihead', action='store_true', help='indicate if the pretrained checkpoint was trained in a multihead fashion')
    parser.add_argument('--pretrained_encoder_only', action='store_true')

    ### main argument
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--pos_class_weight', type=float, help='Weight on positive class for segmentation. Only used on binary labels.', default=10)
    parser.add_argument('--num_images', type=int, default=2)
    parser.add_argument('--num_attention_layers', type=int, default=4)
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--positional_encoding_mode', type=str, help='addition or concatenation', default='concatenation')
    parser.add_argument('--change', action='store_true')

    args = parser.parse_args()
    main(args)
