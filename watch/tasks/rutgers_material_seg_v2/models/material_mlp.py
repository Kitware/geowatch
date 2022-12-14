import torchmetrics
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class MaterialMLP(pl.LightningModule):
    def __init__(self, n_in_channels, n_out_channels, lr=1e-4, wd=1e-5, opt_name='adam', ignore_index=0):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.opt_name = opt_name
        self.n_classes = n_out_channels
        self.in_channels = n_in_channels
        self.ignore_index = ignore_index

        # Build network.
        self.network = self._build_network()

        # Get metrics.
        metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task='multiclass',
                                 num_classes=self.n_classes,
                                 ignore_index=self.ignore_index,
                                 average='micro'),
            torchmetrics.JaccardIndex(task='multiclass',
                                      num_classes=self.n_classes,
                                      ignore_index=self.ignore_index,
                                      average='micro'),
            torchmetrics.Accuracy(task='multiclass',
                                  num_classes=self.n_classes,
                                  ignore_index=self.ignore_index,
                                  average='micro'),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Get losses.
        self.loss_funcs = self._get_loss_funcs()

    def _build_network(self):
        network = nn.Sequential(nn.Linear(self.in_channels, self.in_channels), nn.BatchNorm1d(self.in_channels),
                                nn.PReLU(), nn.Linear(self.in_channels, self.in_channels),
                                nn.BatchNorm1d(self.in_channels), nn.PReLU(),
                                nn.Linear(self.in_channels, self.in_channels), nn.BatchNorm1d(self.in_channels),
                                nn.PReLU(), nn.Linear(self.in_channels, self.n_classes))
        return network

    def forward(self, batch):
        output = self.network(batch['pixel_data'].float())
        return output

    def _set_model_to_train(self):
        self.network.train()

    def _set_model_to_eval(self):
        self.network.eval()

    def training_step(self, batch, batch_idx):
        self._set_model_to_train()

        output = self.forward(batch)

        loss = self._compute_loss(output, batch['target'])

        # Track metrics.
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.train_metrics(flat_pred, flat_target)

        # Log metrics and loss.
        metric_output['train_loss'] = loss
        self.log_dict(metric_output, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self._set_model_to_eval()
        output = self.forward(batch)

        # Compute loss.
        loss = self._compute_loss(output, batch['target'])

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

    def test_step(self, batch, batch_idx):
        self._set_model_to_eval()
        output = self.forward(batch)

        # Compute loss.
        loss = self._compute_loss(output, batch['target'])

        # Track metrics.
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        self.test_metrics.update(flat_pred, flat_target)

        # Log metrics and loss.
        self.log_dict({'test_loss': loss}, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        if self.opt_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise NotImplementedError(f'No implementation for optimizer of name: {self.opt_name}')
        return optimizer

    def validation_epoch_end(self, validation_step_outputs):
        if len(validation_step_outputs) == 0:
            self.test_f1_score = 0
            self.test_iou = 0
            self.test_acc = 0
        else:
            metric_output = self.valid_metrics.compute()
            self.log_dict(metric_output)

    def test_epoch_end(self, test_step_outputs) -> None:
        if len(test_step_outputs) == 0:
            pass
        else:
            metric_output = self.test_metrics.compute()
            self.log_dict(metric_output)

            self.f1_score = metric_output['test_F1Score'].item()
            self.acc = metric_output['test_Accuracy'].item()
            self.iou = metric_output['test_JaccardIndex'].item()

    def _get_loss_funcs(self):
        losses = {}
        losses['focal'] = smp.losses.FocalLoss(mode='multiclass', ignore_index=self.ignore_index)
        losses['dice'] = smp.losses.DiceLoss(mode='multiclass', ignore_index=self.ignore_index)
        losses['jaccard'] = smp.losses.JaccardLoss(mode='multiclass')
        losses['ce'] = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        return losses

    def _compute_loss(self, output, target):
        sum_loss = 0
        for loss_name, loss_func in self.loss_funcs.items():
            sum_loss += loss_func(output, target)
        return sum_loss
