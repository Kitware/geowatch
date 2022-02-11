import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from argparse import Namespace
from .utils.attention_unet import attention_unet
from .data.datasets import gridded_dataset
from .data.multi_image_datasets import kwcoco_dataset, SpaceNet7
import warnings


class segmentation_model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.backbone = attention_unet(hparams.num_channels, 2, pos_encode=hparams.positional_encoding, num_attention_layers=hparams.num_attention_layers, mode=hparams.positional_encoding_mode)

        ##### define dataset
        if hparams.dataset == 'kwcoco':
            if hparams.train_dataset is not None:
                if hparams.dataset_style == 'gridded':
                    self.trainset = gridded_dataset(hparams.train_dataset, sensor=hparams.sensor, bands=hparams.bands, patch_size=hparams.patch_size, segmentation=True, num_images=hparams.num_images)
                else:
                    self.trainset = kwcoco_dataset(hparams.train_dataset, hparams.sensor, hparams.bands, hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images)
            if hparams.vali_dataset is not None:
                if hparams.dataset_style == 'gridded':
                    self.valset = gridded_dataset(hparams.vali_dataset, sensor=hparams.sensor, bands=hparams.bands, patch_size=hparams.patch_size, segmentation=True, num_images=hparams.num_images)
                else:
                    self.valset = kwcoco_dataset(hparams.vali_dataset, hparams.sensor, hparams.bands, hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images)
        elif hparams.dataset == 'spacenet':
            self.trainset = SpaceNet7(hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images, train=True)
            self.valset = SpaceNet7(hparams.patch_size, segmentation_labels=True, num_images=hparams.num_images, train=False)

        if hparams.binary:
            weight = torch.FloatTensor([1, hparams.pos_class_weight])
        else:
            warnings.warn('Classes/Ignore Classes/Background need to be re-checked before succesfully training on site classification models.')
            weight = torch.FloatTensor([0, 1, 1, 1, 1, 1])

        self.criterion = nn.NLLLoss(weight=weight)
        self.save_hyperparameters(hparams)

    def forward(self, x, positions=None):
        predictions = self.backbone(x, positions)
        _, predicted_class = torch.max(predictions, dim=2)
        return {
                'predictions': predictions,
                'predicted_class': predicted_class
               }

    def shared_step(self, batch):
        images = [batch[key] for key in batch if key[:5] == 'image']
        images = torch.stack(images, dim=1).to(self.device)
        segmentations = [batch[key] for key in batch if key[:3] == 'seg']
        segmentations = torch.stack(segmentations, dim=1)

        if self.hparams.binary:
            segmentations = torch.clamp(segmentations, 0, 1)

        if self.hparams.positional_encoding:
            positions = batch['normalized_date']
        else:
            positions = None

        forward = self.forward(images, positions)
        predictions = forward['predictions']

        loss = self.criterion(predictions.reshape(-1, 2, self.hparams.patch_size, self.hparams.patch_size), segmentations.long().reshape(-1, self.hparams.patch_size, self.hparams.patch_size))

        output = {  'predicted_class': forward['predicted_class'],
                    'prediction_map': predictions,
                    'targets' : segmentations,
                    'loss': loss}

        return output

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('validation_loss', output['loss'])
        return output['loss']

    def test_step(self, batch):
        images = [batch[key] for key in batch if key[:5] == 'image']
        images = torch.stack(images, dim=1).to(self.device)
        if self.hparams.positional_encoding:
            positions = batch['normalized_date'].to(self.device)
        else:
            positions = None
        forward = self.forward(images, positions)
        output = {  'predicted_class': forward['predicted_class'],
                    'prediction_map': forward['predictions']}
        return output

    def train_epoch_end(self, outputs):
        epoch_test_loss, epoch_test_accuracy, cl_acc, pr_rec = self.run_test(loader=self.train_dataloader)
        epoch_test_nochange_accuracy = cl_acc[0]
        epoch_test_change_accuracy = cl_acc[1]
        epoch_test_precision = pr_rec[0]
        epoch_test_recall = pr_rec[1]
        epoch_test_Fmeasure = pr_rec[2]
        self.log('train_epoch_accuracy', epoch_test_accuracy)
        self.log('train_epoch_accuracy_change', epoch_test_change_accuracy)
        self.log('train_epoch_accuracy_no_change', epoch_test_nochange_accuracy)
        self.log('train_epoch_precision', epoch_test_precision)
        self.log('train_epoch_recall', epoch_test_recall)
        self.log('train_epoch_f1', epoch_test_Fmeasure)

    def validation_epoch_end(self, outputs):
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
            segmentations = [batch[key] for key in batch if key[:3] == 'seg']
            segmentations = torch.stack(segmentations, dim=1).to(self.device)
            if self.hparams.binary:
                segmentations = torch.clamp(segmentations, 0, 1)
            if self.hparams.positional_encoding:
                positions = batch['normalized_date'].to(self.device)
            else:
                positions = None
            output = self.forward(images, positions)['predictions']
            segmentations = segmentations[:, 0, :, :]
            images = images[:, 0, :, :, :]
            output = output[:, 0, :, :, :]

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
        return torch.utils.data.DataLoader(self.valset, batch_size=1, num_workers=self.hparams.workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        #         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
