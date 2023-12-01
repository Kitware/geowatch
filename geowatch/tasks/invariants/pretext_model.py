import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
from tqdm import tqdm
from torch import pca_lowrank as pca

import json
import torch.package
import ubelt as ub
import os

from .data.datasets import kwcoco_dataset, gridded_dataset
from .utils.attention_unet import attention_unet


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
            # self.trainset = gridded_dataset(hparams.train_dataset, sensor=hparams.sensor, bands=hparams.bands, patch_size=hparams.patch_size)
            self.trainset = kwcoco_dataset(hparams.train_dataset, sensor=hparams.sensor, bands=hparams.bands, patch_size=hparams.patch_size)
        else:
            self.trainset = None
        if hparams.vali_dataset is not None:
            self.valset = gridded_dataset(hparams.vali_dataset, sensor=hparams.sensor, bands=hparams.bands, patch_size=hparams.patch_size)
        else:
            self.valset = None

        print('hparams = {!r}'.format(hparams))
        if hasattr(hparams, 'num_channels'):
            # hack for loading without dataset state
            num_channels = hparams.num_channels
        elif self.trainset is not None:
            num_channels = self.trainset.num_channels
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
        self.backbone = attention_unet(in_channels=num_channels,
                                        out_channels=hparams.feature_dim_shared,
                                        pos_encode=hparams.positional_encoding,
                                        attention_layers=hparams.attention_layers,
                                        mode=hparams.positional_encoding_mode)

        # task specific necks
        self.necks = [
            self.task_neck(2 * self.hparams.feature_dim_shared, self.hparams.feature_dim_each_task),  # sort task
            self.task_neck(self.hparams.feature_dim_shared, self.hparams.feature_dim_each_task),  # augment task
            self.task_neck(self.hparams.feature_dim_shared, self.hparams.feature_dim_each_task),  # overlap task
        ]
        self.necks = nn.ModuleList([ self.necks[i] for i in self.task_indices])

        # task specific heads
        self.heads = [
            self.pixel_classification_head(self.hparams.feature_dim_each_task, num_classes=2),  # sort task
            self.image_classification_head( self.hparams.feature_dim_each_task),  # augment task
            self.image_classification_head( self.hparams.feature_dim_each_task),  # overlap task
        ]
        self.heads = nn.ModuleList([ self.heads[i] for i in self.task_indices ])
        # task specific criterion
        self.criteria = [
            # BinaryFocalLoss(gamma=self.hparams.focal_gamma),  # sort task
            nn.NLLLoss(),  # sort task
            nn.TripletMarginLoss(),  # augment task
            nn.TripletMarginLoss(),  # overlap task
        ]
        self.criteria = [ self.criteria[i] for i in self.task_indices ]

        # task specific metrics
        self.sort_accuracy = Accuracy(task='binary')

    def forward(self, image_stack, positional_encoding=None):
        # pass through shared model body
        return torch.tanh(self.backbone(image_stack, positional_encoding))

    def shared_step(self, batch):
        # get features of each image from shared model body
        image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1).to(self.device).float()
        # positional_encoding must be set to none to produce viable pretext task results
        out = self(image_stack, positional_encoding=None)
        image1_features = out[:, 0, :, :, :]
        image2_features = out[:, 1, :, :, :]
        offset_image1_features = out[:, 2, :, :, :]
        augmented_image1_features = out[:, 3, :, :, :]
        # get time sort labels
        time_labels = batch['time_sort_label']

        time_sort_labels = time_labels.unsqueeze(1).unsqueeze(1).repeat(1, image_stack.shape[-2], image_stack.shape[-1]).to(self.device)
        # time_sort_labels = 99 * torch.ones_like(time_labels)
        # time_sort_labels[:, self.hparams.ignore_boundary:-self.hparams.ignore_boundary, self.hparams.ignore_boundary:-self.hparams.ignore_boundary] = time_labels[:, self.hparams.ignore_boundary:-self.hparams.ignore_boundary, self.hparams.ignore_boundary:-self.hparams.ignore_boundary]

        losses = []
        output = {}

        # Time Sort task
        if 0 in self.task_indices:
            module_list_idx = self.task_indices.index(0)
            time_sort_prediction = self.necks[module_list_idx](torch.cat((image1_features, image2_features), dim=1))
            time_sort_prediction = self.heads[module_list_idx](time_sort_prediction)
            # evaluate
            loss_time = self.criteria[module_list_idx](time_sort_prediction.flatten(2).float(), time_sort_labels.flatten(1).long())
            time_accuracy = self.sort_accuracy((torch.max(time_sort_prediction.data, 1)[1]), time_sort_labels.int())

            losses.append(loss_time)
            output['time_accuracy'] = time_accuracy
            output['loss_time_sort'] = loss_time.detach()
            output['before_after_heatmap'] = time_sort_prediction
        else:
            output['before_after_heatmap'] = None

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
            # perform computation if not already done
            if 2 not in self.task_indices:
                image1_offset_overlap_out = self.heads[module_list_idx](image1_offset_overlap_out)
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
        output.pop('before_after_heatmap')
        self.log_dict(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        output.pop('before_after_heatmap')
        output = {"val_" + key: val for key, val in output.items()}
        self.log_dict(output)
        return output

    def predict(self, batch):
        self.eval()
        feats = {}
        shared_feats = self(batch)
        feats['shared'] = shared_feats
        return feats['shared']

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        if self.hparams.vali_dataset is not None:
            return DataLoader(self.valset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.step_size,
            gamma=self.hparams.lr_gamma)

        return {'optimizer': opt, 'lr_scheduler': lr_scheduler}

    def task_neck(self, in_chan, out_chan):
        kernel_size = 3
        stride = 1
        padding = int((kernel_size - 1) / 2)
        return nn.Sequential(
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def pixel_classification_head(self, in_chan, num_classes):
        kernel_size = 1
        stride = 1
        padding = int((kernel_size - 1) / 2)
        return nn.Sequential(
                            nn.Conv2d(in_chan, in_chan, kernel_size, stride, padding),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_chan, in_chan, kernel_size, stride, padding),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_chan, num_classes, 1, bias=False, padding=0),
                            nn.LogSoftmax(dim=1)
                            )

    def image_classification_head(self, in_chan):
        return nn.Sequential(
                             nn.Flatten(1, -1)
                            )

    # def image_classification_head(self, in_chan):
    #     return nn.Sequential(
    #                          nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    #                          nn.Flatten(1, -1),
    #                          nn.Linear(in_chan, 8),
    #                         )

    def generate_pca_matrix(self, save_path, loader, reduction_dim=6):
        feature_collection = []

        with torch.set_grad_enabled(False):
            # TODO: option to cache or specify a specific projection matrix?
            for n, batch in tqdm(enumerate(loader), desc='Calculating PCA matrix', total=50):
                if n == 50:
                    break
                image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                features = self.forward(image_stack.to(self.device))
                feature_collection.append(features.cpu())

            features = None
            image_stack = None
            stack = torch.cat(feature_collection, dim=0).permute(0, 1, 3, 4, 2).reshape(-1, self.hparams.feature_dim_shared)
            _, _, projector = pca(stack, q=reduction_dim)
            stack = None

        projector = projector.permute(1, 0)

        if save_path:
            torch.save(projector, save_path)

        return projector

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.pca_projection_path:
            save_dir = self.hparams.pca_projection_path
            save_path = os.path.join(save_dir, 'pretext_pca_{}'.format(str(self.current_epoch)) + '.pt')
            os.makedirs(save_dir, exist_ok=True)
            self.generate_pca_matrix(save_path=save_path, loader=self.train_dataloader(), reduction_dim=self.hparams.reduction_dim)

    def save_package(self, package_path):
        model = self

        package_path = os.path.join(package_path, 'pretext_package.pt')

        backup_attributes = {}
        unsaved_attributes = [
            'trainer',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            '_load_state_dict_pre_hooks',
        ]
        for key in unsaved_attributes:

            val = getattr(model, key)
            #print(val)
            if val is not None:
                backup_attributes[key] = val

        log_path = package_path
        log_path = ub.Path(log_path)

        metadata_fpaths = []

        metadata_fpaths += list(log_path.glob('hparams.yaml'))
        try:
            for key in backup_attributes.keys():
                setattr(model, key, None)
            arch_name = 'pretext_model.pkl'
            module_name = 'watch_tasks_invariants'

            with torch.package.PackageExporter(package_path) as exp:
                exp.extern('**', exclude=['geowatch.tasks.invariants.**'])
                exp.intern('geowatch.tasks.invariants.**', allow_empty=False)

                package_header = {
                    'version': '0.1.0',
                    'arch_name': arch_name,
                    'module_name': module_name,
                }
                exp.save_text(
                    'package_header', 'package_header.json',
                    json.dumps(package_header)
                )
                exp.save_pickle(module_name, arch_name, model)
                for meta_fpath in metadata_fpaths:
                    with open(meta_fpath, 'r') as file:
                        text = file.read()
                    exp.save_text('package_header', meta_fpath.name, text)
        finally:

            for key, val in backup_attributes.items():
                setattr(model, key, val)

    @classmethod
    def load_package(cls, package_path):
        """
        DEPRECATE IN FAVOR OF geowatch.tasks.fusion.utils.load_model_from_package

        TODO:
            - [ ] Make the logic that defines the save_package and load_package
                methods with appropriate package header data a lightning
                abstraction.
        """
        # NOTE: there is no gaurentee that this loads an instance of THIS
        # model, the model is defined by the package and the tool that loads it
        # is agnostic to the model contained in said package.
        # This classmethod existing is a convinience more than anything else
        from geowatch.tasks.fusion.utils import load_model_from_package

        self = load_model_from_package(package_path)
        return self
