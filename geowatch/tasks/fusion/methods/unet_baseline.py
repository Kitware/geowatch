import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import einops

import kwcoco
import kwarray
import netharn as nh
import ubelt as ub

from geowatch import heuristics
from geowatch.tasks.fusion.methods.network_modules import coerce_criterion
from geowatch.tasks.fusion.methods.network_modules import RobustModuleDict
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
from geowatch.tasks.fusion.architectures import unet_blur

from typing import Dict, Any, Optional

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class NanToNum(nn.Module):
    """
    Module which converts NaN values in input tensors to numbers.
    """

    def __init__(self, num=0.0):
        super().__init__()
        self.num = num

    def forward(self, x):
        return torch.nan_to_num(x, self.num)


class UNetBaseline(pl.LightningModule, WatchModuleMixins):

    _HANDLES_NANS = True

    def get_cfgstr(self):
        cfgstr = f'{self.hparams.name}_unet'
        return cfgstr

    def __init__(
        self,
        classes=10,
        dataset_stats=None,
        input_sensorchan=None,
        token_dim: int = 32,
        name: str = "unnamed_model",
        class_weights: str = "auto",
        saliency_weights: str = "auto",
        positive_change_weight: float = 1.0,
        negative_change_weight: float = 1.0,
        global_class_weight: float = 1.0,
        global_change_weight: float = 1.0,
        global_saliency_weight: float = 1.0,
        change_loss: str = "cce",  # TODO: replace control string with a module, possibly a subclass
        class_loss: str = "focal",  # TODO: replace control string with a module, possibly a subclass
        saliency_loss: str = "focal",  # TODO: replace control string with a module, possibly a subclass
        ohem_ratio: Optional[float] = None,
        focal_gamma: Optional[float] = 2.0,
    ):
        """
        Args:
            name: Specify a name for the experiment. (Unsure if the Model is the place for this)
            token_width: Width of each square token.
            token_dim: Dimensionality of each computed token.
            spatial_scale_base: The scale assigned to each token equals `scale_base / token_density`, where the token density is the number of tokens along a given axis.
            temporal_scale_base: The scale assigned to each token equals `scale_base / token_density`, where the token density is the number of tokens along a given axis.
            class_weights: Class weighting strategy.
            saliency_weights: Class weighting strategy.

        Example:
            >>> # Note: it is important that the non-kwargs are saved as hyperparams
            >>> from geowatch.tasks.fusion.methods.unet_baseline import UNetBaseline
            >>> model = UNetBaseline(
            >>>   input_sensorchan='r|g|b',
            >>> )
        """

        super().__init__()
        self.save_hyperparameters()

        input_stats = self.set_dataset_specific_attributes(input_sensorchan, dataset_stats)

        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        # TODO: this data should be introspectable via the kwcoco file
        hueristic_background_keys = heuristics.BACKGROUND_CLASSES

        # FIXME: case sensitivity
        hueristic_ignore_keys = heuristics.IGNORE_CLASSNAMES
        if self.class_freq is not None:
            all_keys = set(self.class_freq.keys())
        else:
            all_keys = set(self.classes)

        self.background_classes = all_keys & hueristic_background_keys
        self.ignore_classes = all_keys & hueristic_ignore_keys
        self.foreground_classes = (all_keys - self.background_classes) - self.ignore_classes
        # hueristic_ignore_keys.update(hueristic_occluded_keys)

        self.saliency_num_classes = 2

        # criterion and metrics
        # TODO: parametarize loss criterions
        # For loss function experiments, see and work in
        # ~/code/watch/watch/tks/fusion/methods/sequence_aware.py
        # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        self.saliency_weights = self._coerce_saliency_weights(saliency_weights)
        self.class_weights = self._coerce_class_weights(class_weights)
        self.change_weights = torch.FloatTensor([
            self.hparams.negative_change_weight,
            self.hparams.positive_change_weight
        ])

        self.sensor_channel_tokenizers = RobustModuleDict()

        # Unique sensor modes obviously isn't very correct here.
        # We should fix that, but let's hack it so it at least
        # includes all sensor modes we probably will need.
        if input_stats is not None:
            sensor_modes = set(self.unique_sensor_modes) | set(input_stats.keys())
        else:
            sensor_modes = set(self.unique_sensor_modes)

        for s, c in sensor_modes:
            mode_code = kwcoco.FusedChannelSpec.coerce(c)
            # For each mode make a network that should learn to tokenize
            in_chan = mode_code.numel()

            if input_stats is None:
                input_norm = nh.layers.InputNorm()
            else:
                stats = input_stats.get((s, c), None)
                if stats is None:
                    input_norm = nh.layers.InputNorm()
                else:
                    input_norm = nh.layers.InputNorm(
                        **(ub.udict(stats) & {'mean', 'std'}))

            # key = sanitize_key(str((s, c)))
            key = f'{s}:{c}'
            self.sensor_channel_tokenizers[key] = nn.Sequential(
                input_norm,
                NanToNum(0.0),
                unet_blur.UNet(in_chan, token_dim),
            )

        self.backbone = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(token_dim, token_dim, (2, 5, 5), padding="same"),
            nn.BatchNorm3d(token_dim),
            nn.ReLU(),
            nn.Conv3d(token_dim, token_dim, (2, 5, 5), padding="same"),
            nn.BatchNorm3d(token_dim),
            nn.ReLU(),
            nn.Conv3d(token_dim, token_dim, (2, 5, 5), padding="same"),
            nn.BatchNorm3d(token_dim),
            nn.ReLU(),
        )

        self.criterions = torch.nn.ModuleDict()
        self.heads = torch.nn.ModuleDict()

        self.task_to_keynames = {
            'change': {
                'labels': 'change',
                'weights': 'change_weights',
                'output_dims': 'change_output_dims'
            },
            'saliency': {
                'labels': 'saliency',
                'weights': 'saliency_weights',
                'output_dims': 'saliency_output_dims'
            },
            'class': {
                'labels': 'class_idxs',
                'weights': 'class_weights',
                'output_dims': 'class_output_dims'
            },
        }

        head_properties = [
            {
                'name': 'change',
                'channels': 2,
                'loss': self.hparams.change_loss,
                'weights': self.change_weights,
            },
            {
                'name': 'saliency',
                'channels': self.saliency_num_classes,
                'loss': self.hparams.saliency_loss,
                'weights': self.saliency_weights,
            },
            {
                'name': 'class',
                'channels': self.num_classes,
                'loss': self.hparams.class_loss,
                'weights': self.class_weights,
            },
        ]

        self.global_head_weights = {
            'class': global_class_weight,
            'change': global_change_weight,
            'saliency': global_saliency_weight,
        }

        for prop in head_properties:
            head_name = prop['name']
            global_weight = self.global_head_weights[head_name]
            if global_weight > 0:
                self.criterions[head_name] = coerce_criterion(prop['loss'],
                                                              prop['weights'],
                                                              ohem_ratio=ohem_ratio,
                                                              focal_gamma=focal_gamma)
                self.heads[head_name] = unet_blur.UNet(token_dim, prop["channels"])

        FBetaScore = torchmetrics.FBetaScore
        class_metrics = torchmetrics.MetricCollection({
            "class_acc": torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass'),
            # "class_iou": torchmetrics.IoU(2),
            'class_f1_micro': FBetaScore(beta=1.0, threshold=0.5, average='micro', num_classes=self.num_classes, task='multiclass'),
            'class_f1_macro': FBetaScore(beta=1.0, threshold=0.5, average='macro', num_classes=self.num_classes, task='multiclass'),
        })
        change_metrics = torchmetrics.MetricCollection({
            "change_acc": torchmetrics.Accuracy(task="binary"),
            # "iou": torchmetrics.IoU(2),
            'change_f1': FBetaScore(beta=1.0, task="binary"),
        })
        saliency_metrics = torchmetrics.MetricCollection({
            'saliency_f1': FBetaScore(beta=1.0, task="binary"),
        })

        self.head_metrics = nn.ModuleDict({
            f"{stage}_stage": nn.ModuleDict({
                "class": class_metrics.clone(prefix=f"{stage}_"),
                "change": change_metrics.clone(prefix=f"{stage}_"),
                "saliency": saliency_metrics.clone(prefix=f"{stage}_"),
            })
            for stage in ["train", "val", "test"]
        })

    def process_frame(self, frame) -> Dict[str, Dict[str, Any]]:

        configs = {
            "change": {
                "data": "change",
                "weights": "change_weights",
                "output_dims": "change_output_dims",
                "time_index": "time_index",
            },
            "saliency": {
                "data": "saliency",
                "weights": "saliency_weights",
                "output_dims": "saliency_output_dims",
                "time_index": "time_index",
            },
            "class": {
                "data": "class_idxs",
                "weights": "class_weights",
                "output_dims": "class_output_dims",
                "time_index": "time_index",
            },
        }

        outputs = dict()

        for name, config in configs.items():
            # if frame[config["data"]] is not None:
            output = {
                key: frame[value]
                for key, value in config.items()
            }
            if output["output_dims"] is None:
                if output["data"] is not None:
                    output["output_dims"] = list(output["data"].shape)
                else:
                    output["output_dims"] = frame["output_dims"]
            outputs[name] = output

        for mode_name, mode_val in frame["modes"].items():
            outputs[f"{frame['sensor']}:{mode_name}"] = {
                "data": mode_val,
                "weights": None,
                "output_dims": list(mode_val.shape[1:]),
                "time_index": frame["time_index"],
            }

        return outputs

    def process_example(self, example):
        return [
            self.process_frame(frame)
            for frame in example["frames"]
        ]

    def process_batch(self, batch):
        return [
            self.process_example(example)
            for example in batch
            if example is not None
        ]

    def encode_frame(self, processed_frame):
        return {
            key: self.sensor_channel_tokenizers[key](data["data"][None])[0]  # shape=[C, H, W]
            for key, data in processed_frame.items()
            if key in self.sensor_channel_tokenizers.keys()
        }  # length = num_modes

    def encode_example(self, processed_example):
        return torch.stack([
            torch.stack(
                list(frame.values()),  # shape=[num_modes, C, H, W]
                dim=0,
            ).mean(dim=0)  # shape=[C, H, W]
            for frame in map(self.encode_frame, processed_example)
        ], dim=1)  # shape=[C, num_frames, H, W]

    def encode_batch(self, processed_batch):
        encoded_examples = list(map(self.encode_example, processed_batch))

        C, T, H, W = torch.max(torch.stack([
            torch.tensor(ex.shape)
            for ex in encoded_examples
        ]), dim=0).values

        encoded_examples = [
            F.pad(
                ex,
                # F.pad pairs padding values IN REVERSE ORDER, below is correct
                (
                    0, 0,  # W-ex.shape[3],
                    0, 0,  # H-ex.shape[2],
                    0, T - ex.shape[1],
                    0, 0,  # C-ex.shape[0],
                ),
                mode="constant", value=0,
            )
            for ex in encoded_examples
        ]

        return torch.stack(encoded_examples, dim=0)

    def forward(self, batch):
        """
        Example:
            >>> from geowatch.tasks import fusion
            >>> channels, classes, dataset_stats = fusion.methods.UNetBaseline.demo_dataset_stats()
            >>> model = fusion.methods.UNetBaseline(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>> )
            >>> batch = model.demo_batch(width=64, height=64)
            >>> outputs = model.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"
        """

        processed_batch = self.process_batch(batch)
        encoded_batch = self.encode_batch(processed_batch)
        output_seqs = self.backbone(encoded_batch)

        # decompose outputs
        outputs = dict()
        for task_name, task_head in self.heads.items():
            task_outputs = []
            task_probs = []
            for output_seq, example in zip(output_seqs, batch):

                output_seq = einops.rearrange(output_seq, "chan time height width -> time chan height width")

                seq_outputs = []
                seq_probs = []
                for output, frame in zip(output_seq, example["frames"]):

                    output = task_head(output[None])[0]
                    probs = einops.rearrange(output, "chan height width -> height width chan")
                    if task_name == "change":
                        probs = probs.sigmoid()[..., 1]
                    else:
                        probs = probs.softmax(dim=-1)

                    seq_outputs.append(output)
                    seq_probs.append(probs)
                task_outputs.append(seq_outputs)
                task_probs.append(seq_probs)
            outputs[task_name] = task_outputs
            outputs[f"{task_name}_probs"] = task_probs

        return outputs

    def shared_step(self, batch, batch_idx=None, stage="train", with_loss=True):
        """
        Example:
            >>> # xdoctest: +REQUIRES(env:SLOW_TESTS)
            >>> from geowatch.tasks import fusion
            >>> import torch
            >>> channels, classes, dataset_stats = fusion.methods.UNetBaseline.demo_dataset_stats()
            >>> model = fusion.methods.UNetBaseline(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>> )
            >>> batch = model.demo_batch(batch_size=2, width=64, height=65, num_timesteps=3)
            >>> outputs = model.shared_step(batch)
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> optimizer.zero_grad()
            >>> loss = outputs["loss"]
            >>> loss.backward()
            >>> optimizer.step()

        Example:
            >>> # xdoctest: +REQUIRES(env:SLOW_TESTS)
            >>> from geowatch.tasks import fusion
            >>> import torch
            >>> channels, classes, dataset_stats = fusion.methods.UNetBaseline.demo_dataset_stats()
            >>> model = fusion.methods.UNetBaseline(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>> )
            >>> batch = model.demo_batch(batch_size=2, width=64, height=65, num_timesteps=3)
            >>> batch += [None]
            >>> outputs = model.shared_step(batch)
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> optimizer.zero_grad()
            >>> loss = outputs["loss"]
            >>> loss.backward()
            >>> optimizer.step()

        Example:
            >>> from geowatch.tasks import fusion
            >>> import torch
            >>> channels, classes, dataset_stats = fusion.methods.UNetBaseline.demo_dataset_stats()
            >>> model = fusion.methods.UNetBaseline(
            >>>     classes=classes, token_dim=2,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>> )
            >>> batch = model.demo_batch(batch_size=1, width=32, height=35, num_timesteps=3, nans=0.1)
            >>> batch += model.demo_batch(batch_size=1, width=32, height=35, num_timesteps=3, nans=0.5)
            >>> batch += model.demo_batch(batch_size=1, width=32, height=35, num_timesteps=3, nans=1.0)
            >>> outputs = model.shared_step(batch)
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> optimizer.zero_grad()
            >>> loss = outputs["loss"]
            >>> loss.backward()
            >>> optimizer.step()
        """

        # FIXME: why are we getting nones here?
        batch = [
            ex
            for ex in batch
            if (ex is not None)
            # and (len(ex["frames"]) > 0)
        ]

        outputs = self(batch)

        if not with_loss:
            return outputs

        frame_losses = []
        for task_name in self.heads:
            for pred_seq, example in zip(outputs[task_name], batch):
                for pred, frame in zip(pred_seq, example["frames"]):

                    task_labels_key = self.task_to_keynames[task_name]["labels"]
                    labels = frame[task_labels_key]

                    self.log(f"{stage}_{task_name}_logit_mean", pred.mean())

                    if labels is None:
                        continue

                    # FIXME: This is necessary because sometimes when data.input_space_scale==native, label shapes and output_dims dont match!
                    if pred.shape[1:] != labels.shape:
                        pred = nn.functional.interpolate(
                            pred[None],
                            size=labels.shape,
                            mode="bilinear",
                        )[0]

                    task_weights_key = self.task_to_keynames[task_name]["weights"]
                    task_weights = frame[task_weights_key]

                    valid_mask = (task_weights > 0.)
                    pred_ = pred[:, valid_mask]
                    task_weights_ = task_weights[valid_mask]

                    criterion = self.criterions[task_name]
                    if criterion.target_encoding == 'index':
                        loss_labels = labels.long()
                        loss_labels_ = loss_labels[valid_mask]
                    elif criterion.target_encoding == 'onehot':
                        # Note: 1HE is much easier to work with
                        loss_labels = kwarray.one_hot_embedding(
                            labels.long(),
                            criterion.in_channels,
                            dim=0)
                        loss_labels_ = loss_labels[:, valid_mask]
                    else:
                        raise KeyError(criterion.target_encoding)

                    loss = criterion(
                        pred_[None],
                        loss_labels_[None],
                    )

                    if loss.isnan().any():
                        print(loss)
                        print(pred)
                        print(frame)

                    loss *= task_weights_
                    frame_losses.append(
                        self.global_head_weights[task_name] * loss.mean()
                    )
                    self.log_dict(
                        self.head_metrics[f"{stage}_stage"][task_name](
                            pred.argmax(dim=0).flatten(),
                            # pred[None],
                            labels.flatten().long(),
                        ),
                        prog_bar=True,
                    )

        outputs["loss"] = sum(frame_losses) / len(frame_losses)
        self.log(f"{stage}_loss", outputs["loss"], prog_bar=True)
        return outputs

#     def shared_step(self, batch, batch_idx=None, with_loss=True):
#         outputs = {
#             "change_probs": [
#                 [
#                     0.5 * torch.ones(*frame["output_dims"])
#                     for frame in example["frames"]
#                     if frame["change"] != None
#                 ]
#                 for example in batch
#             ],
#             "saliency_probs": [
#                 [
#                     torch.ones(*frame["output_dims"], 2).sigmoid()
#                     for frame in example["frames"]
#                 ]
#                 for example in batch
#             ],
#             "class_probs": [
#                 [
#                     torch.ones(*frame["output_dims"], self.num_classes).softmax(dim=-1)
#                     for frame in example["frames"]
#                 ]
#                 for example in batch
#             ],
#         }

#         if with_loss:
#             outputs["loss"] = self.dummy_param

#         return outputs

    @profile
    def training_step(self, batch, batch_idx=None):
        outputs = self.shared_step(batch, batch_idx=batch_idx, stage='train')
        return outputs

    @profile
    def validation_step(self, batch, batch_idx=None):
        outputs = self.shared_step(batch, batch_idx=batch_idx, stage='val')
        return outputs

    @profile
    def test_step(self, batch, batch_idx=None):
        outputs = self.shared_step(batch, batch_idx=batch_idx, stage='test')
        return outputs

    @profile
    def predict_step(self, batch, batch_idx=None):
        outputs = self.shared_step(batch, batch_idx=batch_idx, stage='predict',
                                   with_loss=False)
        return outputs

    # this is a special thing for the predict step
    forward_step = shared_step

    def save_package(self, package_path, verbose=1):
        """

        CommandLine:
            xdoctest -m geowatch.tasks.fusion.methods.unet_baseline UNetBaseline.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.unet_baseline import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> model = self = methods.UNetBaseline(
            >>>     input_sensorchan=5,
            >>> )

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.UNetBaseline.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
            >>> recon = load_model_from_package(package_path)
            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = model.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> from geowatch.utils.util_kwarray import torch_array_equal
            >>> for key in recon_state.keys():
            >>>     v1 = model_state[key]
            >>>     v2 = recon_state[key]
            >>>     if not torch.allclose(v1, v2, equal_nan=True):
            >>>         print('v1 = {}'.format(ub.urepr(v1, nl=1)))
            >>>         print('v2 = {}'.format(ub.urepr(v2, nl=1)))
            >>>         raise AssertionError(f'Difference in key={key}')
            >>>     assert v1 is not v2, 'should be distinct copies'

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.unet_baseline import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> model = self = methods.UNetBaseline(
            >>>     input_sensorchan=5,
            >>> )

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.UNetBaseline.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
            >>> recon = load_model_from_package(package_path)
            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = self.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> from geowatch.utils.util_kwarray import torch_array_equal
            >>> for key in recon_state.keys():
            >>>     v1 = model_state[key]
            >>>     v2 = recon_state[key]
            >>>     if not torch.allclose(v1, v2, equal_nan=True):
            >>>         print('v1 = {}'.format(ub.urepr(v1, nl=1)))
            >>>         print('v2 = {}'.format(ub.urepr(v2, nl=1)))
            >>>         raise AssertionError(f'Difference in key={key}')
            >>>     assert v1 is not v2, 'should be distinct copies'

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.unet_baseline import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> model = self = methods.UNetBaseline(
            >>>     input_sensorchan=5,
            >>> )

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.UNetBaseline.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
            >>> recon = load_model_from_package(package_path)
            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = self.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> from geowatch.utils.util_kwarray import torch_array_equal
            >>> for key in recon_state.keys():
            >>>     v1 = model_state[key]
            >>>     v2 = recon_state[key]
            >>>     if not torch.allclose(v1, v2, equal_nan=True):
            >>>         print('v1 = {}'.format(ub.urepr(v1, nl=1)))
            >>>         print('v2 = {}'.format(ub.urepr(v2, nl=1)))
            >>>         raise AssertionError(f'Difference in key={key}')
            >>>     assert v1 is not v2, 'should be distinct copies'
        """
        self._save_package(package_path, verbose=verbose)
