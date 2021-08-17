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
            # "acc": metrics.Accuracy(),
            # "iou": metrics.IoU(2),
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
            >>> from watch.tasks.fusion import datamodules
            >>> datamodule = datamodules.WatchDataModule(
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

        item_losses = []
        item_true_changes = []
        item_pred_changes = []
        for item in batch:

            # For now, just reconstruct the stacked input, but
            # in the future, the encoder will need to take care of
            # the heterogeneous inputs

            frame_ims = []
            for frame in item['frames']:
                assert len(frame['modes']) == 1, 'only handle one mode for now'
                mode_key, mode_val = ub.peek(frame['modes'].items())
                mode_val = mode_val.float()
                # self.input_norms = None
                if self.input_norms is not None:
                    mode_norm = self.input_norms[mode_key]
                    mode_val = mode_norm(mode_val)
                frame_ims.append(mode_val)

            # Because we are not collating we need to add a batch dimension
            images = torch.stack(frame_ims)[None, ...]

            B, T, C, H, W = images.shape

            logits = self(images)

            # TODO: it may be faster to compute loss at the downsampled
            # resolution.
            logits = nn.functional.interpolate(
                logits, [H, W], mode="bilinear", align_corners=True)

            change_prob = logits.sigmoid()[0]
            item_pred_changes.append(change_prob.detach())

            if with_loss:
                changes = torch.stack([
                    frame['change'] for frame in item['frames'][1:]
                ])[None, ...]
                item_true_changes.append(changes)

                # compute criterion
                loss = self.criterion(logits, changes.float())
                item_losses.append(loss)

        outputs['binary_predictions'] = item_pred_changes

        if with_loss:
            total_loss = sum(item_losses)
            all_pred = torch.stack(item_pred_changes)
            all_true = torch.cat(item_true_changes, dim=0)
            # compute metrics
            item_metrics = {}

            if self.trainer is not None:
                # Dont log unless a trainer is attached
                for key, metric in self.metrics.items():
                    val = metric(all_pred, all_true)
                    item_metrics[f'{stage}_{key}'] = val

                for key, val in item_metrics.items():
                    self.log(key, val, prog_bar=True)

                self.log(f'{stage}_loss', total_loss, prog_bar=True)

            # if stage == 'train':
            #     # I think train does not want "loss" to have a prefix
            #     self.log('loss', total_loss, prog_bar=True)

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

    @classmethod
    def load_package(cls, package_path, verbose=1):
        import torch.package
        #
        # TODO: is there any way to introspect what these variables could be?

        model_name = "model.pkl"
        module_name = 'watch_tasks_fusion'

        imp = torch.package.PackageImporter(package_path)

        # Assume this standardized header information exists that tells us the
        # name of the resource corresponding to the model
        package_header = imp.load_pickle(
            'kitware_package_header', 'kitware_package_header.pkl')
        model_name = package_header['model_name']
        module_name = package_header['module_name']

        # pkg_root = imp.file_structure()
        # print(pkg_root)
        # pkg_data = pkg_root.children['.data']

        self = imp.load_pickle(module_name, model_name)
        return self

    def save_package(self, package_path, verbose=1):
        """

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> from watch.tasks.fusion.methods.common import *  # NOQA
            >>> dpath = ub.ensure_app_cache_dir('watch/tests/package')
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion models in a test
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> model = methods.MultimodalTransformerDirectCD("smt_it_stm_p8")
            >>> # We have to run an input through the module because it is lazy
            >>> inputs = torch.rand(1, 2, 13, 128, 128)
            >>> model(inputs)

            >>> # Save the model
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.MultimodalTransformerDirectCD.load_package(package_path)
            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = model.state_dict()
            >>> assert recon is not model
            >>> assert set(recon_state) == set(recon_state)
            >>> for key in recon_state.keys():
            >>>     assert (model_state[key] == recon_state[key]).all()
            >>>     assert model_state[key] is not recon_state[key]

        Example:
            >>> # Test with datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> from watch.tasks.fusion.methods.common import *  # NOQA
            >>> dpath = ub.ensure_app_cache_dir('watch/tests/package')
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion models in a test
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> self = methods.MultimodalTransformerDirectCD("smt_it_stm_p8")
            >>> # We have to run an input through the module because it is lazy
            >>> inputs = torch.rand(1, 2, 13, 128, 128)
            >>> self(inputs)

            >>> datamodule = datamodules.watch_data.WatchDataModule(
            >>>     'special:vidshapes8-multispectral', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=0)
            >>> datamodule.setup('fit')
            >>> trainer = pl.Trainer(max_steps=1)
            >>> trainer.fit(model=self, datamodule=datamodule)

            >>> # Save the self
            >>> self.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.MultimodalTransformerDirectCD.load_package(package_path)

            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = self.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> for key in recon_state.keys():
            >>>     assert (model_state[key] == recon_state[key]).all()
            >>>     assert model_state[key] is not recon_state[key]
        """
        import copy
        import torch.package

        # shallow copy of self, to apply attribute hacks to
        model = copy.copy(self)
        if model.trainer is not None:
            datamodule = model.trainer.datamodule
            if datamodule is not None:
                model.datamodule_hparams = datamodule.hparams
        model.trainer = None
        model.train_dataloader = None
        model.val_dataloader = None
        model.test_dataloader = None

        model_name = "model.pkl"
        module_name = 'watch_tasks_fusion'
        with torch.package.PackageExporter(package_path, verbose=verbose) as exp:
            # TODO: this is not a problem yet, but some package types (mainly
            # binaries) will need to be excluded and added as mocks
            exp.extern("**", exclude=["watch.tasks.fusion.**"])
            exp.intern("watch.tasks.fusion.**")

            # Attempt to standardize some form of package metadata that can
            # allow for model importing with fewer hard-coding requirements
            package_header = {
                'version': '0.0.1',
                'model_name': model_name,
                'module_name': module_name,
            }
            exp.save_pickle(
                'kitware_package_header', 'kitware_package_header.pkl',
                package_header
            )

            exp.save_pickle(module_name, model_name, model)

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
