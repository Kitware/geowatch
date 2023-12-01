import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy

from ..data.datasets import kwcoco_dataset
from ..utils.unet_blur import UNetEncoder, UNetDecoder
from ..utils.focal_loss import BinaryFocalLoss


class pretext(pl.LightningModule):
    TASK_NAMES = [
        'sort',
        'augment',
        'overlap'
    ]

    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)

        if hparams.train_dataset is not None:
            self.trainset = kwcoco_dataset(hparams.train_dataset, hparams.sensor, hparams.bands, hparams.patch_size)
        else:
            self.trainset = None
        if hparams.vali_dataset is not None:
            self.valset = kwcoco_dataset(hparams.vali_dataset, hparams.sensor, hparams.bands, hparams.patch_size)
        else:
            self.valset = None

        print('hparams = {!r}'.format(hparams))
        if hasattr(hparams, 'num_channels'):
            # hack for loading without dataset state
            num_channels = hparams.num_channels
        elif self.trainset is not None:
            num_channels = self.trainset.num_channels()
        else:
            num_channels = 6

        # determine which tasks to run
        self.task_indices = []
        # no tasks specified
        if len(self.hparams.tasks) < 1:
            raise ValueError(f'tasks must be specified. Options are {", ".join(pretext.TASK_NAMES)}, or all')
        # perform all tasks
        elif len(self.hparams.tasks) == 1 and self.hparams.tasks[0].lower() == 'all':
            self.task_indices = [i for i in range(len(pretext.TASK_NAMES))]
        # run a subset of tasks
        else:
            for task in self.hparams.tasks:
                if task.lower() in pretext.TASK_NAMES:
                    self.task_indices.append(pretext.TASK_NAMES.index(task.lower()))
                else:
                    raise ValueError(f'\'{task}\' not recognized as an available task. Options are {", ".join(pretext.TASK_NAMES)}, or all')
        if pretext.TASK_NAMES.index('augment') in self.task_indices and pretext.TASK_NAMES.index('overlap') not in self.task_indices:
            self.task_indices.append(pretext.TASK_NAMES.index('overlap'))

        # shared model body
        self.encoder = UNetEncoder(in_channels=num_channels)
        self.decoder = UNetDecoder(out_channels=self.hparams.feature_dim_shared)

        # task specific necks
        self.necks = [
            self.task_neck(self.hparams.feature_dim_shared, self.hparams.feature_dim_each_task),  # sort task
            self.task_neck(self.hparams.feature_dim_shared, self.hparams.feature_dim_each_task),  # augment task
            self.task_neck(self.hparams.feature_dim_shared, self.hparams.feature_dim_each_task),  # overlap task
        ]
        self.necks = nn.ModuleList([ self.necks[i] for i in self.task_indices ])

        # task specific heads
        self.heads = [
            self.pixel_classification_head(2 * self.hparams.feature_dim_each_task),  # sort task
            self.image_classification_head( self.hparams.feature_dim_each_task),  # augment task
            self.image_classification_head( self.hparams.feature_dim_each_task),  # overlap task
        ]
        self.heads = nn.ModuleList([ self.heads[i] for i in self.task_indices ])
        # task specific criterion
        self.criteria = [
            BinaryFocalLoss(gamma=self.hparams.focal_gamma),  # sort task
            nn.TripletMarginLoss(),  # augment task
            nn.TripletMarginLoss(),  # overlap task
        ]
        self.criteria = [ self.criteria[i] for i in self.task_indices ]

        # task specific metrics
        self.sort_accuracy = Accuracy()

    def forward(self, image):
        # pass through shared model body
        encoded = self.encoder(image)
        decoded = self.decoder(encoded)
        return decoded

    def shared_step(self, batch):
        # get features of each image from shared model body
        image1_features = self(batch['image1'])
        image2_features = self(batch['image2'])
        offset_image1_features = self(batch['offset_image1'])
        augmented_image1_features = self(batch['augmented_image1'])
        # get time sort labels
        time_sort_labels = batch['time_sort_label']
        time_sort_labels = time_sort_labels.unsqueeze(1).unsqueeze(1).repeat(1, self.hparams.patch_size, self.hparams.patch_size).unsqueeze(1)

        losses = []
        output = {}

        # Time Sort task
        if 0 in self.task_indices:
            module_list_idx = self.task_indices.index(0)
            # forward pass through neck
            image1_sort_out = self.necks[module_list_idx](image1_features)
            image2_sort_out = self.necks[module_list_idx](image2_features)
            # forward pass through head
            time_sort_prediction = self.heads[module_list_idx](torch.cat((image1_sort_out, image2_sort_out), dim=1))
            # evaluate
            loss_time = self.criteria[module_list_idx](time_sort_prediction, time_sort_labels)
            if self.hparams.aot_penalty_weight:
                l1_penalty = torch.norm(image1_sort_out - image2_sort_out, 1, dim=1) / image1_sort_out.shape[1]
                l1_penalty_filtered = -1 * torch.topk(-1 * l1_penalty.flatten(), int(self.hparams.aot_penalty_percentage * l1_penalty.numel())).values
                loss_time = loss_time + self.hparams.aot_penalty_weight * l1_penalty_filtered.mean()
            time_accuracy = self.sort_accuracy((time_sort_prediction > 0.), time_sort_labels.int())

            losses.append(loss_time)
            output['time_accuracy'] = time_accuracy
            output['loss_time_sort'] = loss_time.detach()
            output['before_after_heatmap'] = F.sigmoid(time_sort_prediction)
        # Overlap task
        if 2 in self.task_indices:
            module_list_idx = self.task_indices.index(2)
            # forward pass through neck
            image1_overlap_out = self.necks[module_list_idx](image1_features)
            image2_overlap_out = self.necks[module_list_idx](image2_features)
            image1_offset_overlap_out = self.necks[module_list_idx](offset_image1_features)
            # forward pass through head
            image1_overlap_out = self.heads[module_list_idx](image1_overlap_out)
            image2_overlap_out = self.heads[module_list_idx](image2_overlap_out)
            image1_offset_overlap_out = self.heads[module_list_idx](image1_offset_overlap_out)
            # evaluate
            loss_offset = self.criteria[module_list_idx](image1_overlap_out, image2_overlap_out, image1_offset_overlap_out)

            losses.append(loss_offset)
            if 2 in self.task_indices:
                output['loss_offset'] = loss_offset.detach()

        # Augment task
        if 1 in self.task_indices:
            module_list_idx = self.task_indices.index(1)
            # image1 forward pass through neck
            image1_augment_out = self.necks[module_list_idx](image1_features)
            image1_augmented_augment_out = self.necks[module_list_idx](augmented_image1_features)
            # image1 forward pass through head
            image1_augment_out = self.heads[module_list_idx](image1_augment_out)
            image1_augmented_augment_out = self.heads[module_list_idx](image1_augmented_augment_out)
            # image1 evaluate
            loss_augment = self.criteria[module_list_idx](image1_augment_out, image1_augmented_augment_out, image1_offset_overlap_out)

            losses.append(loss_augment)
            output['loss_augmented'] = loss_augment.detach()

        # add up loss
        loss = torch.sum(torch.stack(losses))
        output['loss'] = loss

        return output

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log_dict(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        output = {"val_" + key: val for key, val in output.items()}
        self.log_dict(output)
        return output

    def predict(self, batch):
        self.eval()
        feats = {}
        shared_feats = self(batch)
        feats['shared'] = shared_feats
        return feats['shared']

    ### Temporary solution
    def predict_before_after(self, image):
        self.eval()
        im1 = image[:, 0, :]
        im2 = image[:, 1, :]
        image1_features = self(im1)
        image2_features = self(im2)
        image1_sort_out = self.necks[self.task_indices.index(0)](image1_features)
        image2_sort_out = self.necks[self.task_indices.index(0)](image2_features)
        # forward pass through head
        time_sort_prediction = self.heads[self.task_indices.index(0)](torch.cat((image1_sort_out, image2_sort_out), dim=1))
        return time_sort_prediction

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, shuffle=True)

    def val_dataloader(self):
        if self.hparams.vali_dataset is not None:
            return DataLoader(self.valset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.step_size,
            gamma=self.hparams.lr_gamma)

        return {'optimizer': opt, 'lr_scheduler': lr_scheduler}

    def task_neck(self, in_chan, out_chan):
        kernel_size = 1
        stride = 1
        padding = int((kernel_size - 1) / 2)
        return nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size, stride, padding),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def pixel_classification_head(self, in_chan):
        kernel_size = 1
        stride = 1
        padding = int((kernel_size - 1) / 2)
        return nn.Sequential(
                            nn.Conv2d(in_chan, in_chan, kernel_size, stride, padding),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_chan, 1, 1, bias=False, padding=0),
                            )

    def image_classification_head(self, in_chan):
        return nn.Sequential(
                             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                             nn.Flatten(1, -1),
                             nn.Linear(in_chan, 2),
                            )
