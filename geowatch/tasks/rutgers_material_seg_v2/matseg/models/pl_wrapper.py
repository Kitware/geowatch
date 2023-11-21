import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from pycm import ConfusionMatrix

from watch.tasks.rutgers_material_seg_v2.matseg.models.smp_model import create_smp_network


class MaterialSegmentationModel(pl.LightningModule):

    def __init__(self,
                 model_params,
                 loss_func,
                 n_classes,
                 ignore_index=0,
                 lr=1e-4,
                 wd=1e-6,
                 optimizer_mode='adam',
                 lr_scheduler_mode=None):
        super().__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()

        self.model_params = model_params

        self.lr = lr
        self.wd = wd
        self.loss_func = loss_func
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.optimizer_mode = optimizer_mode
        self.lr_scheduler_mode = lr_scheduler_mode

        self.model = self._build_model(**model_params)
        optimizers = self.configure_optimizers()  # NOQA
        # self.optimizer, self.lr_scheduler = optimizers['optimizer'], optimizers['lr_scheduler']
        # self.optimizer, self.lr_scheduler = optimizers[0], optimizers[1]

        self._get_tracked_metrics()

        self.region_confusion_matrices = {}

    def _build_model(self, network_name, encoder_name, in_channels, out_channels, pretrain=None):
        network = create_smp_network(network_name,
                                     encoder_name,
                                     in_channels,
                                     out_channels,
                                     pretrain=pretrain)
        return network

    def _get_tracked_metrics(self, average_mode='micro'):
        # Get metrics.
        metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(
                task="multiclass",
                num_classes=self.n_classes,
                ignore_index=self.ignore_index,
                average='micro',
            ),
            torchmetrics.JaccardIndex(task="multiclass",
                                      num_classes=self.n_classes,
                                      ignore_index=self.ignore_index,
                                      average='micro'),
            torchmetrics.Accuracy(task="multiclass",
                                  num_classes=self.n_classes,
                                  ignore_index=self.ignore_index,
                                  average='micro'),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, batch):
        images = batch['image']
        output = self.model(images)
        return output

    def forward_feat(self, batch):
        images = batch['image']
        output = {}
        enc_feats = self.model.encoder(images)
        output['enc_feats'] = enc_feats
        decoder_output = self.model.decoder(*output['enc_feats'])
        output['logits'] = self.model.segmentation_head(decoder_output)
        return output

    def training_step(self, batch, batch_idx):
        self.model = self.model.train()
        target = batch['target']

        # self.optimizer.zero_grad()
        output = self.forward(batch)

        loss = self.loss_func(output, target)
        if np.isnan(loss.item()):
            # NaN happens when the model predicts all unknown classes.
            # Make arbitary loss value to avoid NaN and to penalize model.
            loss = torch.nan_to_num(loss) + 10
        # self.manual_backward(loss)
        # self.optimizer.step()

        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.train_metrics(flat_pred, flat_target)

        # Log metrics and loss.
        metric_output['train_loss'] = loss
        self.log_dict(metric_output, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model = self.model.eval()
        target = batch['target']
        output = self.forward(batch)

        # Compute loss.
        loss = self.loss_func(output, target)

        # Track metrics.
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.valid_metrics(flat_pred, flat_target)
        self.valid_metrics.update(flat_pred, flat_target)

        # Log metrics and loss.
        metric_output['valid_loss'] = loss
        self.log_dict(metric_output, prog_bar=True, on_step=True, on_epoch=True)
        # self.log_dict({}, prog_bar=True, on_step=True, on_epoch=True)

        return metric_output, batch['target'].shape[0]

    def on_train_epoch_end(self, trainer=None, pl_module=None) -> None:
        # Adjust learning rate.
        pass
        # if self.lr_scheduler is not None:
        #     sch = self.lr_schedulers()
        #     if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         sch.step(self.trainer.callback_metrics["train_loss_epoch"])

    def test_step(self, batch, batch_idx):
        self.model = self.model.eval()
        output = self.forward(batch)

        # Compute loss.
        loss = self.loss_func(output, batch['target'])

        # Track metrics.
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        self.test_metrics.update(flat_pred, flat_target)

        # Log metrics and loss.
        self.log_dict({'test_loss': loss}, prog_bar=True, on_step=True, on_epoch=True)

        # Aggregate confusion matrix per region.
        batch_size = batch['target'].shape[0]
        for b in range(batch_size):
            region_name = batch['region_name'][b]
            pred_vector = output[b].argmax(axis=0).flatten().detach().cpu().numpy()
            target_vector = batch['target'][b].flatten().detach().cpu().numpy()
            batch_cm = ConfusionMatrix(target_vector,
                                       pred_vector,
                                       classes=list(range(self.n_classes))).to_array()
            if region_name not in self.region_confusion_matrices:
                self.region_confusion_matrices[region_name] = batch_cm
            else:
                self.region_confusion_matrices[region_name] += batch_cm

    def _compute_metrics(self, conf, target):
        # conf: [batch_size, n_classes, height, width]
        # target: [batch_size, height, width]

        pred = conf.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), target.flatten()

        batch_metrics = {}
        for metric_name, metric_func in self.tracked_metrics.items():
            metric_value = metric_func(flat_pred, flat_target)
            metric_value = torch.nan_to_num(metric_value)
            batch_metrics[metric_name] = metric_value
        return batch_metrics

    def configure_optimizers(self):
        if self.optimizer_mode == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer_mode == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer_mode == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise NotImplementedError

        if self.lr_scheduler_mode is None:
            lr_scheduler = None
        else:
            if self.lr_scheduler_mode == 'reduce_lr_on_plateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                          mode='min',
                                                                          factor=0.1,
                                                                          patience=5,
                                                                          verbose=True)
            elif self.lr_scheduler_mode == 'step':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                               step_size=10,
                                                               gamma=0.1,
                                                               verbose=True)
            else:
                raise NotImplementedError(
                    'lr_scheduler_mode must be one of [None, reduce_lr_on_plateau, step]')
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        return ([optimizer], {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'train_F1Score'
        })

    def on_validation_epoch_end(self, validation_step_outputs):
        if len(validation_step_outputs) == 0:
            self.test_f1_score = 0
            self.test_iou = 0
            self.test_acc = 0
        else:
            metric_output = self.valid_metrics.compute()
            self.log_dict(metric_output)

    def on_test_epoch_end(self, **kwargs) -> None:
        if len(kwargs['test_step_outputs']) == 0:
            pass
        else:
            metric_output = self.test_metrics.compute()
            self.log_dict(metric_output)

            self.f1_score = metric_output['test_F1Score'].item()
            self.acc = metric_output['test_Accuracy'].item()
            self.iou = metric_output['test_JaccardIndex'].item()

        # Compute confusion matrix per region and overall region.
        self.region_cm_metrics, self.region_cms = {}, {}
        combined_cm = None
        for region_name, conf_matrix in self.region_confusion_matrices.items():
            if combined_cm is None:
                combined_cm = conf_matrix
            else:
                combined_cm += conf_matrix
            conf_matrix = ConfusionMatrix(matrix=conf_matrix, classes=list(range(self.n_classes)))

            self.region_cm_metrics[region_name] = {
                'f1': conf_matrix.F1,  # pylint: disable=no-member
                'auc': conf_matrix.AUC,  # pylint: disable=no-member
                'acc': conf_matrix.ACC,  # pylint: disable=no-member
                'macro_f1': conf_matrix.F1_Macro,  # pylint: disable=no-member
                'macro_acc': conf_matrix.ACC_Macro,  # pylint: disable=no-member
            }

        # Compute overall confusion matrix.
        combined_cm = ConfusionMatrix(matrix=combined_cm, classes=list(range(self.n_classes)))
        self.region_cms['overall'] = combined_cm
        self.region_cm_metrics['overall'] = {
            'f1': combined_cm.F1,  # pylint: disable=no-member
            'auc': combined_cm.AUC,  # pylint: disable=no-member
            'acc': combined_cm.ACC,  # pylint: disable=no-member
            'macro_f1': combined_cm.F1_Macro,  # pylint: disable=no-member
            'macro_acc': combined_cm.ACC_Macro,  # pylint: disable=no-member
        }
