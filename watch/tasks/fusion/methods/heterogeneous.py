import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
import einops

import kwcoco
import kwarray
import netharn as nh
import ubelt as ub

from watch import heuristics
from watch.tasks.fusion.methods.network_modules import _class_weights_from_freq
from watch.tasks.fusion.methods.network_modules import coerce_criterion
from watch.tasks.fusion.methods.network_modules import RobustModuleDict
from watch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
from watch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder

import numpy as np

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def to_next_multiple(n, mult):
    diff = mult - n % mult
    if diff == mult:
        return 0
    return diff


def positions_from_shape(shape, dtype="float32", device="cpu"):
    positions = torch.stack(torch.meshgrid(*[
        torch.linspace(-1, 1, size + 1, dtype=dtype, device=device)[:-1]
        for size in shape
    ]), dim=0)
    mean_dims = list(range(1, len(shape) + 1))
    positions -= positions.mean(dim=mean_dims, keepdims=True)
    return positions


class PadToMultiple(nn.Module):
    def __init__(self, multiple, mode='constant', value=None):
        super().__init__()
        self.multiple = multiple
        self.mode = mode
        self.value = value

    def forward(self, x):

        height, width = x.shape[-2:]
        pad = (
            0, to_next_multiple(height, self.multiple),
            0, to_next_multiple(width, self.multiple),
        )
        return nn.functional.pad(x, pad, mode=self.mode, value=self.value)


class NanToNum(nn.Module):
    def __init__(self, num=0.0):
        super().__init__()
        self.num = num

    def forward(self, x):
        return torch.nan_to_num(x, self.num)


class MipNerfPositionalEncoder(nn.Module):
    def __init__(self, in_dims, L=10):
        super().__init__()
        self.mean_weights = nn.Parameter(
            2. ** torch.arange(0, L),
            requires_grad=False)
        self.scale_weights = nn.Parameter(
            -(2. ** (2. * torch.arange(0, L) - 1)),
            requires_grad=False)

        self.weight = self.mean_weights
        self.output_dim = 2 * in_dims * L

    def forward(self, mean, scale):
        weighted_means = torch.einsum("y,x...->xy...", self.mean_weights, mean)
        weighted_means = einops.rearrange(weighted_means, "x y ... -> (x y) ...")

        weighted_scales = torch.einsum("y,x...->xy...", self.scale_weights, scale)
        weighted_scales = einops.rearrange(weighted_scales, "x y ... -> (x y) ...")

        return torch.concat([
            weighted_means.sin() * weighted_scales.exp(),
            weighted_means.cos() * weighted_scales.exp(),
        ], dim=0)


class HeterogeneousModel(pl.LightningModule, WatchModuleMixins):

    _HANDLES_NANS = True

    def get_cfgstr(self):
        cfgstr = f'{self.hparams.name}_heterogeneous'
        return cfgstr

    def __init__(
        self,
        classes=10,
        dataset_stats=None,
        input_sensorchan=None,
        name: str = "unnamed_model",
        token_width: int = 10,
        token_dim: int = 16,
        spatial_scale_base: float = 1.,
        temporal_scale_base: float = 1.,
        ignore_scale: bool = False,
        backbone_encoder_depth: int = 4,
        backbone_decoder_depth: int = 1,
        backbone_cross_heads: int = 1,
        backbone_latent_heads: int = 8,
        backbone_cross_dim_head: int = 64,
        backbone_latent_dim_head: int = 64,
        backbone_weight_tie_layers: bool = False,
        position_encoding_frequencies: int = 16,
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
    ):
        """
        Args:
            name: Specify a name for the experiment. (Unsure if the Model is the place for this)
            token_width: Width of each square token.
            token_dim: Dimensionality of each computed token.
            spatial_scale_base: The scale assigned to each token equals `scale_base / token_density`, where the token density is the number of tokens along a given axis. This value is also the assigned scale for all tokens when `ignore_scale` is True.
            temporal_scale_base: The scale assigned to each token equals `scale_base / token_density`, where the token density is the number of tokens along a given axis. This value is also the assigned scale for all tokens when `ignore_scale` is True.
            ignore_scale: Don't compute the scale for each token individually and instead assign a cosntant value, `scale_base`.
            class_weights: Class weighting strategy.
            saliency_weights: Class weighting strategy.

        Example:
            >>> # Note: it is important that the non-kwargs are saved as hyperparams
            >>> from watch.tasks.fusion.methods.heterogeneous import HeterogeneousModel
            >>> model = HeterogeneousModel(input_sensorchan='r|g|b')
        """

        super().__init__()
        self.save_hyperparameters()

        if dataset_stats is not None:
            input_stats = dataset_stats['input_stats']
            class_freq = dataset_stats['class_freq']
            if input_sensorchan is None:
                input_sensorchan = ','.join(
                    [f'{s}:{c}' for s, c in dataset_stats['unique_sensor_modes']])
        else:
            class_freq = None
            input_stats = None

        self.class_freq = class_freq
        self.dataset_stats = dataset_stats

        # Handle channel-wise input mean/std in the network (This is in
        # contrast to common practice where it is done in the dataloader)
        if input_sensorchan is None:
            raise Exception(
                'need to specify input_sensorchan at least as the number of '
                'input channels')
        input_sensorchan = kwcoco.SensorChanSpec.coerce(input_sensorchan)
        self.input_sensorchan = input_sensorchan

        if self.dataset_stats is None:
            # hack for tests (or no known sensors case)
            input_stats = None
            self.unique_sensor_modes = {
                (s.sensor.spec, s.chans.spec)
                for s in input_sensorchan.streams()
            }
        else:
            self.unique_sensor_modes = self.dataset_stats['unique_sensor_modes']

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

        if isinstance(saliency_weights, str):
            if saliency_weights == 'auto':
                if class_freq is not None:
                    bg_freq = sum(class_freq.get(k, 0) for k in self.background_classes)
                    fg_freq = sum(class_freq.get(k, 0) for k in self.foreground_classes)
                    bg_weight = 1.
                    fg_weight = bg_freq / (fg_freq + 1)
                    fg_bg_weights = [bg_weight, fg_weight]
                    _w = fg_bg_weights + ([0.0] * (self.saliency_num_classes - len(fg_bg_weights)))
                    saliency_weights = torch.Tensor(_w)
                else:
                    fg_bg_weights = [1.0, 1.0]
                    _w = fg_bg_weights + ([0.0] * (self.saliency_num_classes - len(fg_bg_weights)))
                    saliency_weights = torch.Tensor(_w)
                # total_freq = np.array(list())
                # print('total_freq = {!r}'.format(total_freq))
                # cat_weights = _class_weights_from_freq(total_freq)
            else:
                raise KeyError(saliency_weights)
        else:
            raise NotImplementedError(saliency_weights)

        # criterion and metrics
        # TODO: parametarize loss criterions
        # For loss function experiments, see and work in
        # ~/code/watch/watch/tasks/fusion/methods/sequence_aware.py
        # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        if isinstance(class_weights, str):
            if class_weights == 'auto':
                if self.class_freq is None:
                    heuristic_weights = {}
                else:
                    total_freq = np.array(list(self.class_freq.values()))
                    cat_weights = _class_weights_from_freq(total_freq)
                    catnames = list(self.class_freq.keys())
                    print('total_freq = {!r}'.format(total_freq))
                    print('cat_weights = {!r}'.format(cat_weights))
                    print('catnames = {!r}'.format(catnames))
                    heuristic_weights = ub.dzip(catnames, cat_weights)
                print('heuristic_weights = {}'.format(ub.repr2(heuristic_weights, nl=1)))

                heuristic_weights.update({k: 0 for k in hueristic_ignore_keys})
                # print('heuristic_weights = {}'.format(ub.repr2(heuristic_weights, nl=1, align=':')))
                class_weights = []
                for catname in self.classes:
                    w = heuristic_weights.get(catname, 1.0)
                    class_weights.append(w)
                using_class_weights = ub.dzip(self.classes, class_weights)

                # Add in user-specific modulation of the weights
                # if self.hparams.modulate_class_weights:
                #     import re
                #     parts = [p.strip() for p in self.hparams.modulate_class_weights.split(',')]
                #     parts = [p for p in parts if p]
                #     for part in parts:
                #         toks = re.split('([+*])', part)
                #         catname = toks[0]
                #         rest_iter = iter(toks[1:])
                #         weight = using_class_weights[catname]
                #         nrhtoks = len(toks) - 1
                #         assert nrhtoks % 2 == 0
                #         nstmts = nrhtoks // 2
                #         for _ in range(nstmts):
                #             opcode = next(rest_iter)
                #             arg = float(next(rest_iter))
                #             if opcode == '*':
                #                 weight = weight * arg
                #             elif opcode == '+':
                #                 weight = weight * arg
                #             else:
                #                 raise KeyError(opcode)
                #         # Modulate
                #         using_class_weights[catname] = weight

                print('using_class_weights = {}'.format(ub.repr2(using_class_weights, nl=1, align=':')))
                class_weights = torch.FloatTensor(class_weights)
            else:
                raise KeyError(class_weights)
        else:
            raise NotImplementedError(class_weights)

        self.saliency_weights = saliency_weights
        self.class_weights = class_weights
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
                    input_norm = nh.layers.InputNorm(**stats)

            # key = sanitize_key(str((s, c)))
            key = f'{s}:{c}'
            self.sensor_channel_tokenizers[key] = nn.Sequential(
                input_norm,
                NanToNum(0.0),
                PadToMultiple(token_width, value=0.0),
                nn.Conv2d(
                    in_chan,
                    token_dim,
                    kernel_size=token_width,
                    stride=token_width,
                ),
            )

        self.position_encoder = MipNerfPositionalEncoder(3, position_encoding_frequencies)
        # self.position_encoder = RandomFourierPositionalEncoder(3, 16)
        position_dim = self.position_encoder.output_dim

        self.backbone = TransformerEncoderDecoder(
            encoder_depth=backbone_encoder_depth,
            decoder_depth=backbone_decoder_depth,
            dim=token_dim + position_dim,
            queries_dim=position_dim,
            logits_dim=token_dim,
            cross_heads=backbone_cross_heads,
            latent_heads=backbone_latent_heads,
            cross_dim_head=backbone_cross_dim_head,
            latent_dim_head=backbone_latent_dim_head,
            weight_tie_layers=backbone_weight_tie_layers,
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
                self.criterions[head_name] = coerce_criterion(prop['loss'], prop['weights'])
                self.heads[head_name] = nn.Conv2d(
                    token_dim,
                    prop['channels'],
                    kernel_size=5,
                    padding="same",
                    bias=False)

        if hasattr(torchmetrics, 'FBetaScore'):
            FBetaScore = torchmetrics.FBetaScore
        else:
            FBetaScore = torchmetrics.FBeta

        class_metrics = torchmetrics.MetricCollection({
            "class_acc": torchmetrics.Accuracy(),
            # "class_iou": torchmetrics.IoU(2),
            'class_f1_micro': FBetaScore(beta=1.0, threshold=0.5, average='micro'),
            'class_f1_macro': FBetaScore(beta=1.0, threshold=0.5, average='macro', num_classes=self.num_classes),
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

    def process_input_tokens(self, example):
        example_tokens = []
        for frame in example["frames"]:
            sensor = frame["sensor"]

            frame_tokens = []
            for mode, mode_img in frame["modes"].items():
                # tokens
                tokenizer_key = f"{sensor}:{mode}"
                tokens = self.sensor_channel_tokenizers[tokenizer_key](mode_img)

                # space
                height, width = tokens_shape = tokens.shape[1:]
                token_positions = positions_from_shape(
                    tokens_shape,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )

                token_positions_scales = self.hparams.spatial_scale_base / torch.tensor(
                    token_positions.shape[1:],
                    dtype=token_positions.dtype,
                    device=token_positions.device,
                )
                token_positions_scales = einops.repeat(
                    token_positions_scales,
                    "chan -> chan height width",
                    height=height, width=width,
                )

                if self.hparams.ignore_scale:
                    token_positions_scales = self.hparams.spatial_scale_base * torch.ones_like(token_positions_scales)

                # time
                token_times = frame["time_index"] * torch.ones_like(token_positions[0])[None].type_as(token_positions)

                token_times_scales = self.hparams.temporal_scale_base * torch.ones(
                    1, height, width,
                    dtype=token_positions.dtype,
                    device=token_positions.device,
                )

                # TODO: sensor/mode

                # combine positional encodings
                token_encodings = torch.concat([
                    token_positions,
                    token_times,
                ])
                token_scales = torch.concat([
                    token_positions_scales,
                    token_times_scales,
                ])

                token_encodings = self.position_encoder(token_encodings, token_scales)

                tokens = torch.concat([
                    tokens,
                    token_encodings,
                ])
                frame_tokens.append(tokens)

            example_tokens.append(frame_tokens)
        return example_tokens

    def process_query_tokens(self, example):
        example_tokens = []
        for frame in example["frames"]:

            # space
            height, width = frame["output_dims"]
            height, width = tokens_shape = (height // self.hparams.token_width, width // self.hparams.token_width)
            token_positions = positions_from_shape(
                tokens_shape,
                dtype=self.position_encoder.weight.dtype,
                device=self.position_encoder.weight.device,
            )

            token_positions_scales = self.hparams.spatial_scale_base / torch.tensor(
                token_positions.shape[1:],
                dtype=token_positions.dtype,
                device=token_positions.device,
            )

            if self.hparams.ignore_scale:
                token_positions_scales = self.hparams.spatial_scale_base * torch.ones_like(token_positions_scales)

            token_positions_scales = einops.repeat(
                token_positions_scales,
                "chan -> chan height width",
                height=height, width=width,
            )

            # time
            token_times = frame["time_index"] * torch.ones_like(
                token_positions[0],
                dtype=token_positions.dtype,
                device=token_positions.device,
            )[None]

            token_times_scales = self.hparams.temporal_scale_base * torch.ones(
                1, height, width,
                dtype=token_positions.dtype,
                device=token_positions.device,
            )

            # combine positional encodings
            token_encodings = torch.concat([
                token_positions,
                token_times,
            ])
            token_scales = torch.concat([
                token_positions_scales,
                token_times_scales,
            ])

            token_encodings = self.position_encoder(token_encodings, token_scales)

            example_tokens.append(token_encodings)
        return example_tokens

    def forward(self, batch):

        # input sequences
        orig_input_seqs = []
        for example in batch:
            input_tokens = self.process_input_tokens(example)

            input_token_seq = torch.concat([
                torch.concat([
                    einops.rearrange(mode_tokens, "chan ... -> (...) chan")
                    for mode_tokens in frame_tokens
                ])
                for frame_tokens in input_tokens
            ])
            orig_input_seqs.append(input_token_seq)

        padding_value = -1000.0
        input_seqs = nn.utils.rnn.pad_sequence(
            orig_input_seqs,
            batch_first=True,
            padding_value=padding_value,
        )
        input_masks = input_seqs[..., 0] > padding_value
        input_seqs[~input_masks] = 0.

        # query sequences
        orig_query_shapes = []
        orig_query_seqs = []
        for example in batch:
            query_tokens = self.process_query_tokens(example)
            query_shapes = [
                frame_tokens.shape[1:]
                for frame_tokens in query_tokens
            ]
            query_token_seq = torch.concat([
                einops.rearrange(frame_tokens, "chan ... -> (...) chan")
                for frame_tokens in query_tokens
            ])

            orig_query_shapes.append(query_shapes)
            orig_query_seqs.append(query_token_seq)

        padding_value = -1000.0
        query_seqs = nn.utils.rnn.pad_sequence(
            orig_query_seqs,
            batch_first=True,
            padding_value=padding_value,
        )
        query_masks = query_seqs[..., 0] > padding_value
        query_seqs[~query_masks] = 0.

        # forward pass!
        output_seqs = self.backbone(
            input_seqs,
            mask=input_masks,
            queries=query_seqs,
        )

        # decompose outputs
        outputs = dict()
        for task_name, task_head in self.heads.items():
            task_outputs = []
            task_probs = []
            for output_seq, query_mask, frame_shapes, example in zip(output_seqs, query_masks, orig_query_shapes, batch):
                output_seq = output_seq[query_mask]  # only want valid values we actually requested

                seq_outputs = []
                seq_probs = []
                frame_sizes = [h * w for h, w in frame_shapes]
                for output_frame_seq, (height, width), frame in zip(torch.split(output_seq, frame_sizes), frame_shapes, example["frames"]):

                    output = einops.rearrange(
                        output_frame_seq,
                        "(height width) chan -> chan height width",
                        height=height, width=width,
                    )
                    target_size = frame["output_dims"]
                    output = nn.functional.interpolate(
                        output[None],
                        size=target_size,
                        mode="bilinear",
                    )[0]
                    output = task_head(output)

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

        # FIXME: why are we getting nones here?
        batch = [ex for ex in batch if ex is not None]

        outputs = self(batch)

        if not with_loss:
            return outputs

        frame_losses = []
        for task_name in self.heads:
            for pred_seq, example in zip(outputs[task_name], batch):
                for pred, frame in zip(pred_seq, example["frames"]):

                    task_labels_key = self.task_to_keynames[task_name]["labels"]
                    labels = frame[task_labels_key]

                    if labels is None:
                        continue

                    # FIXME: This is necessary because sometimes when data.input_space_scale==native, label shapes and output_dims dont match!
                    if pred.shape[1:] != labels.shape:
                        pred = nn.functional.interpolate(
                            pred[None],
                            size=labels.shape,
                            mode="bilinear",
                        )[0]

                    criterion = self.criterions[task_name]
                    if criterion.target_encoding == 'index':
                        loss_labels = labels.long()
                    elif criterion.target_encoding == 'onehot':
                        # Note: 1HE is much easier to work with
                        loss_labels = kwarray.one_hot_embedding(
                            labels.long(),
                            criterion.in_channels,
                            dim=0)
                    else:
                        raise KeyError(criterion.target_encoding)

                    task_weights_key = self.task_to_keynames[task_name]["weights"]
                    loss = criterion(
                        pred[None],
                        loss_labels[None],
                    )
                    loss *= frame[task_weights_key]
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
            xdoctest -m watch.tasks.fusion.methods.heterogeneous HeterogeneousModel.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from watch.tasks.fusion.methods.heterogeneous import *  # NOQA
            >>> dpath = ub.Path.appdir('watch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> model = self = methods.HeterogeneousModel(
            >>>     input_sensorchan=5)

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.HeterogeneousModel.load_package(package_path)
            >>> from watch.tasks.fusion.utils import load_model_from_package
            >>> recon = load_model_from_package(package_path)
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
            >>> from watch.tasks.fusion import datamodules
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion.methods.heterogeneous import *  # NOQA
            >>> dpath = ub.Path.appdir('watch/tests/package').ensuredir()
            >>> package_path = dpath / 'my_package.pt'

            >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral-multisensor', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10)
            >>> datamodule.setup('fit')
            >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats, input_sensorchan=datamodule.input_sensorchan)
            
            >>> from types import MethodType
            >>> def configure_optimizers(self):
            >>>     return torch.optim.Adam(self.parameters())
            >>> self.configure_optimizers = MethodType(configure_optimizers, self)

            >>> # We have to run an input through the module because it is lazy
            >>> batch = ub.peek(iter(datamodule.train_dataloader()))
            >>> outputs = self.training_step(batch)

            >>> trainer = pl.Trainer(max_steps=1)
            >>> trainer.fit(model=self, datamodule=datamodule)

            >>> # Save the self
            >>> self.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.HeterogeneousModel.load_package(package_path)

            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = self.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> for key in recon_state.keys():
            >>>     v1 = model_state[key]
            >>>     v2 = recon_state[key]
            >>>     if not (v1 == v2).all():
            >>>         print('v1 = {}'.format(ub.repr2(v1, nl=1)))
            >>>         print('v2 = {}'.format(ub.repr2(v2, nl=1)))
            >>>         raise AssertionError(f'Difference in key={key}')
            >>>     assert v1 is not v2, 'should be distinct copies'

        Ignore:
            7z l $HOME/.cache/watch/tests/package/my_package.pt
        """
        # import copy
        import json
        import torch.package

        # Fix an issue on 3.10 with torch 1.12
        from watch.utils.lightning_ext.callbacks.packager import _torch_package_monkeypatch
        _torch_package_monkeypatch()

        # shallow copy of self, to apply attribute hacks to
        # model = copy.copy(self)
        model = self

        backup_attributes = {}
        # Remove attributes we don't want to pickle before we serialize
        # then restore them
        unsaved_attributes = [
            'trainer',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            '_load_state_dict_pre_hooks',  # lightning 1.5
            '_trainer',  # lightning 1.7
        ]
        for key in unsaved_attributes:
            try:
                val = getattr(model, key, None)
            except Exception:
                val = None
            if val is not None:
                backup_attributes[key] = val

        train_dpath_hint = getattr(model, 'train_dpath_hint', None)
        if model.has_trainer:
            if train_dpath_hint is None:
                train_dpath_hint = model.trainer.log_dir
            datamodule = model.trainer.datamodule
            if datamodule is not None:
                model.datamodule_hparams = datamodule.hparams

        metadata_fpaths = []
        if train_dpath_hint is not None:
            train_dpath_hint = ub.Path(train_dpath_hint)
            metadata_fpaths += list(train_dpath_hint.glob('hparams.yaml'))
            metadata_fpaths += list(train_dpath_hint.glob('fit_config.yaml'))

        try:
            for key in backup_attributes.keys():
                setattr(model, key, None)
            arch_name = 'model.pkl'
            module_name = 'watch_tasks_fusion'
            """
            exp = torch.package.PackageExporter(package_path, verbose=True)
            """
            with torch.package.PackageExporter(package_path) as exp:
                # TODO: this is not a problem yet, but some package types (mainly
                # binaries) will need to be excluded and added as mocks
                exp.extern('**', exclude=['watch.tasks.fusion.**'])
                exp.intern('watch.tasks.fusion.**', allow_empty=False)

                # Attempt to standardize some form of package metadata that can
                # allow for model importing with fewer hard-coding requirements

                # TODO:
                # Add information about how this was trained, and what epoch it
                # was saved at.
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

                # Save metadata
                for meta_fpath in metadata_fpaths:
                    with open(meta_fpath, 'r') as file:
                        text = file.read()
                    exp.save_text('package_header', meta_fpath.name, text)
        finally:
            # restore attributes
            for key, val in backup_attributes.items():
                setattr(model, key, val)
