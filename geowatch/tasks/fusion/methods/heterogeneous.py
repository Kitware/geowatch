import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
import einops
from einops.layers.torch import Rearrange
from torchvision import models as tv_models
from torchvision.models import feature_extraction
from typing import Union, Optional
import numpy as np

import kwcoco
import kwarray
import netharn as nh
import ubelt as ub

from geowatch import heuristics
from geowatch.tasks.fusion.methods.network_modules import coerce_criterion
from geowatch.tasks.fusion.methods.network_modules import RobustModuleDict
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
from geowatch.tasks.fusion.architectures.transformer import BackboneEncoderDecoder, TransformerEncoderDecoder
from geowatch.tasks.fusion.architectures import transformer

from abc import ABCMeta, abstractmethod

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


SPLIT_ATTENTION_ENCODERS = list(transformer.encoder_configs.keys())


def to_next_multiple(n, mult):
    """
    Example:
        >>> from geowatch.tasks.fusion.methods.heterogeneous import to_next_multiple
        >>> x = to_next_multiple(11, 4)
        >>> assert x == 1, f"x = {x}, should be 1"
    """

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
    def __init__(self, multiple: int, mode: str = 'constant', value=0.):
        """
        Pads input image-shaped tensors following strategy defined by mode/value. All padding appended to bottom and right of input.

        Args:
            multiple: (int)
            mode:
                (str, default: 'constant') Padding strategy. One of ('constant', 'reflect', 'replicate', 'circular').
                See: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
            value:
                (Any, default: None) Fill value for 'constant', set to 0 automatically when value=None.
                See: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad

        Example:
            >>> from geowatch.tasks.fusion.methods.heterogeneous import PadToMultiple
            >>> import torch
            >>> pad_module = PadToMultiple(4)
            >>> inputs = torch.randn(1, 3, 10, 11)
            >>> outputs = pad_module(inputs)
            >>> assert outputs.shape == (1, 3, 12, 12), f"outputs.shape actually {outputs.shape}"

        Example:
            >>> from geowatch.tasks.fusion.methods.heterogeneous import PadToMultiple
            >>> import torch
            >>> pad_module = PadToMultiple(4)
            >>> inputs = torch.randn(3, 10, 11)
            >>> outputs = pad_module(inputs)
            >>> assert outputs.shape == (3, 12, 12), f"outputs.shape actually {outputs.shape}"

        Example:
            >>> from geowatch.tasks.fusion.methods.heterogeneous import PadToMultiple
            >>> from torch import nn
            >>> import torch
            >>> token_width = 10
            >>> pad_module = nn.Sequential(
            >>>         PadToMultiple(token_width, value=0.0),
            >>>         nn.Conv2d(
            >>>             3,
            >>>             16,
            >>>             kernel_size=token_width,
            >>>             stride=token_width,
            >>>         )
            >>> )
            >>> inputs = torch.randn(3, 64, 65)
            >>> outputs = pad_module(inputs)
            >>> assert outputs.shape == (16, 7, 7), f"outputs.shape actually {outputs.shape}"
        """

        super().__init__()
        self.multiple = multiple
        self.mode = mode
        self.value = value

    def forward(self, x):

        height, width = x.shape[-2:]
        pad = (
            0, to_next_multiple(width, self.multiple),
            0, to_next_multiple(height, self.multiple),
        )
        return nn.functional.pad(x, pad, mode=self.mode, value=self.value)


class NanToNum(nn.Module):
    """
    Module which converts NaN values in input tensors to numbers.
    """

    def __init__(self, num=0.0):
        super().__init__()
        self.num = num

    def forward(self, x):
        return torch.nan_to_num(x, self.num)


class ShapePreservingTransformerEncoder(nn.Module):
    def __init__(
        self,
        token_dim,
        num_layers,
        # norm=None,
        # enable_nested_tensor=True,
        batch_dim=0,
        chan_dim=1,
    ):
        super().__init__()
        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         token_dim,
        #         8,
        #         dim_feedforward=512,
        #         dropout=0.1,
        #         activation="gelu",
        #         batch_first=True,
        #         norm_first=True,
        #     ),
        #     num_layers,
        #     norm,
        #     enable_nested_tensor,
        # )
        self.encoder = TransformerEncoderDecoder(
            encoder_depth=num_layers,
            decoder_depth=1,
            dim=token_dim,
            queries_dim=token_dim,
            logits_dim=None,
            decode_cross_every=1,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
        )
        self.batch_dim = batch_dim
        self.chan_dim = chan_dim

    def forward(self, src, mask=None):
        assert mask is None

        num_dims = len(src.shape)
        shape_codes = [f"shape_{idx}" for idx in range(num_dims)]
        shape_codes[self.batch_dim] = "batch"
        shape_codes[self.chan_dim] = "chan"

        input_shape_code = " ".join(shape_codes)
        output_shape_code = " ".join([
            "batch",
            f"({' '.join([code for code in shape_codes if code.startswith('shape_')])})",
            "chan",
        ])

        shape_hints = einops.parse_shape(src, input_shape_code)

        src = einops.rearrange(src, f"{input_shape_code} -> {output_shape_code}")
        src = self.encoder.forward(src, mask=mask)
        src = einops.rearrange(src, f"{output_shape_code} -> {input_shape_code}", **shape_hints)

        return src


class ScaleAwarePositionalEncoder(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, mean, scale):
        pass


class MipNerfPositionalEncoder(nn.Module, ScaleAwarePositionalEncoder):
    """
    Module which computes MipNeRf-based positional encoding vectors from tensors of mean and scale values
    """

    def __init__(self, in_dims: int, num_freqs: int = 10, max_freq: float = 4.):
        """
        out_dims = 2 * in_dims * num_freqs

        Args:
            in_dims: (int) number of input dimensions to expect for future calls to .forward(). Currently only needed for computing .output_dim
            num_freqs: (int) number of frequencies to project dimensions onto.

        Example:
            >>> from geowatch.tasks.fusion.methods.heterogeneous import MipNerfPositionalEncoder
            >>> import torch
            >>> pos_enc = MipNerfPositionalEncoder(3, 4)
            >>> input_means = torch.randn(1, 3, 10, 10)
            >>> input_scales = torch.randn(1, 3, 10, 10)
            >>> outputs = pos_enc(input_means, input_scales)
            >>> assert outputs.shape == (1, pos_enc.output_dim, 10, 10)
        """

        super().__init__()
        frequencies = torch.linspace(0, max_freq, num_freqs)
        self.mean_weights = nn.Parameter(
            2. ** frequencies,
            requires_grad=False)
        self.scale_weights = nn.Parameter(
            -2. ** (2. * frequencies - 1),
            requires_grad=False)

        self.weight = self.mean_weights
        self.output_dim = 2 * in_dims * num_freqs

    def forward(self, mean, scale):

        weighted_means = torch.einsum("y,bx...->bxy...", self.mean_weights, mean)
        weighted_means = einops.rearrange(weighted_means, "batch x y ... -> batch (x y) ...")

        weighted_scales = torch.einsum("y,bx...->bxy...", self.scale_weights, scale)
        weighted_scales = einops.rearrange(weighted_scales, "batch x y ... -> batch (x y) ...")

        return torch.concat([
            weighted_means.sin() * weighted_scales.exp(),
            weighted_means.cos() * weighted_scales.exp(),
        ], dim=1)


class ScaleAgnostictPositionalEncoder(nn.Module, ScaleAwarePositionalEncoder):
    """
    Module which computes MipNeRf-based positional encoding vectors from tensors of mean and scale values
    """

    def __init__(self, in_dims: int, num_freqs: int = 10, max_freq: float = 4.):
        """
        out_dims = 2 * in_dims * num_freqs

        Args:
            in_dims: (int) number of input dimensions to expect for future calls to .forward(). Currently only needed for computing .output_dim
            num_freqs: (int) number of frequencies to project dimensions onto.

        Example:
            >>> from geowatch.tasks.fusion.methods.heterogeneous import ScaleAgnostictPositionalEncoder
            >>> import torch
            >>> pos_enc = ScaleAgnostictPositionalEncoder(3, 4)
            >>> input_means = torch.randn(1, 3, 10, 10)
            >>> input_scales = torch.randn(1, 3, 10, 10)
            >>> outputs = pos_enc(input_means, input_scales)
            >>> assert outputs.shape == (1, pos_enc.output_dim, 10, 10)
        """

        super().__init__()
        frequencies = torch.linspace(0, max_freq, num_freqs)
        self.mean_weights = nn.Parameter(
            2. ** frequencies,
            requires_grad=False)

        self.weight = self.mean_weights
        self.output_dim = 2 * in_dims * num_freqs

    def forward(self, mean, scale):

        weighted_means = torch.einsum("y,bx...->bxy...", self.mean_weights, mean)
        weighted_means = einops.rearrange(weighted_means, "batch x y ... -> batch (x y) ...")

        return torch.concat([
            weighted_means.sin(),
            weighted_means.cos(),
        ], dim=1)


class ResNetShim(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x):
        return self.submodule(x[None])["layer4"][0]


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
        position_encoder: Union[str, ScaleAwarePositionalEncoder] = 'auto',
        backbone: Union[str, BackboneEncoderDecoder] = 'auto',
        token_width: int = 10,
        token_dim: int = 16,
        spatial_scale_base: float = 1.,
        temporal_scale_base: float = 1.,
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
        tokenizer: str = "simple_conv",  # TODO: replace control string with a module, possibly a subclass
        decoder: str = "upsample",  # TODO: replace control string with a module, possibly a subclass
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
            >>> from geowatch.tasks.fusion.methods.heterogeneous import HeterogeneousModel, ScaleAgnostictPositionalEncoder
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = ScaleAgnostictPositionalEncoder(3, 8)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> model = HeterogeneousModel(
            >>>   input_sensorchan='r|g|b',
            >>>   position_encoder=position_encoder,
            >>>   backbone=backbone,
            >>> )
        """
        # assert position_encoder is not None
        assert tokenizer in {"simple_conv", "resnet18"}, "Tokenizer not implemented yet."
        assert decoder in {"upsample", "simple_conv", "trans_conv"}, "Decoder not implemented yet."

        if isinstance(position_encoder, str):
            if position_encoder == 'auto':
                position_encoder = ScaleAgnostictPositionalEncoder(3, 8)
            else:
                raise KeyError(position_encoder)

        pre_backbone = None
        post_backbone = None

        if isinstance(backbone, str):
            if backbone == 'auto':
                # TODO: set this to a "reasonable" default.
                backbone = TransformerEncoderDecoder(
                    encoder_depth=3,
                    decoder_depth=3,
                    dim=position_encoder.output_dim + token_dim,
                    queries_dim=position_encoder.output_dim,
                    logits_dim=token_dim,
                    cross_heads=1,
                    latent_heads=1,
                    cross_dim_head=1,
                    latent_dim_head=1,
                )
            elif backbone == 'small':
                # This should be a reasonable small network for testing
                backbone = TransformerEncoderDecoder(
                    encoder_depth=1,
                    decoder_depth=1,
                    dim=position_encoder.output_dim + token_dim,
                    queries_dim=position_encoder.output_dim,
                    logits_dim=token_dim,
                    cross_heads=1,
                    latent_heads=1,
                    cross_dim_head=1,
                    latent_dim_head=1,
                )
            elif backbone == 'wu-vit':
                """
                    import geowatch
                    from geowatch.utils.simple_dvc import SimpleDVC
                    expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt')
                    expt_dvc = SimpleDVC(expt_dvc_dpath)
                    ckpt_fpath = expt_dvc_dpath / 'models/wu/MAE-2023-02-09/goldenMae-epoch=07-val_loss=0.23.ckpt'

                    from geowatch.tasks.fusion.methods.heterogeneous import HeterogeneousModel, ScaleAgnostictPositionalEncoder
                    from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
                    position_encoder = ScaleAgnostictPositionalEncoder(3, 8)
                    channels, classes, dataset_stats = HeterogeneousModel.demo_dataset_stats()
                    model = HeterogeneousModel(
                      # token_dim=768,
                      # token_dim=768,
                      input_sensorchan=channels,
                      classes=classes,
                      dataset_stats=dataset_stats,
                      position_encoder=position_encoder,
                      backbone='wu-vit',
                    )

                    from geowatch.tasks.fusion.fit import coerce_initializer
                    from kwutil import util_pattern
                    initializer = coerce_initializer(str(ckpt_fpath))
                    initializer.forward(model)

                    batch = model.demo_batch(width=64, height=65)
                    batch += model.demo_batch(width=55, height=75)
                    outputs = model.forward(batch)
                """
                from geowatch.tasks.fusion.architectures import wu_mae
                pre_backbone = nn.Linear(token_dim + position_encoder.output_dim, 16)
                # post_backbone = nn.Linear(16, token_dim + position_encoder.output_dim)
                post_backbone = nn.Linear(16, token_dim)
                backbone = wu_mae.wu_backbone().transformer
            elif backbone == 'sits-former':
                """
                Ignore:
                    import geowatch
                    from geowatch.utils.simple_dvc import SimpleDVC
                    expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt')
                    expt_dvc = SimpleDVC(expt_dvc_dpath)
                    pretrained_fpath = expt_dvc_dpath / 'models/pretrained/sits-former/checkpoint.bert.tar'

                    import torch
                    model_state = torch.load(pretrained_fpath)

                    from geowatch.tasks.fusion.methods.heterogeneous import HeterogeneousModel, ScaleAgnostictPositionalEncoder
                    from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
                    position_encoder = ScaleAgnostictPositionalEncoder(3, 8)
                    channels, classes, dataset_stats = HeterogeneousModel.demo_dataset_stats()
                    model = HeterogeneousModel(
                      token_dim=208,
                      input_sensorchan=channels,
                      classes=classes,
                      dataset_stats=dataset_stats,
                      position_encoder=position_encoder,
                      backbone='sits-former',
                    )

                    from geowatch.tasks.fusion.fit import coerce_initializer
                    from kwutil import util_pattern
                    initializer = coerce_initializer(str(pretrained_fpath))
                    initializer.forward(model)

                    batch = model.demo_batch(width=64, height=65)
                    batch += model.demo_batch(width=55, height=75)
                    outputs = model.forward(batch)
                """
                from geowatch.tasks.fusion.architectures import sits
                bert_config = {
                    'num_features': 10,
                    'hidden': 256,
                    'n_layers': 3,
                    'attn_heads': 8,
                    'dropout': 0.1,
                }
                bert = sits.BERT(**bert_config)
                backbone = bert.transformer_encoder
                # Hack to denote that we need to not use batch first for this
                # model.
                backbone.is_sits_bert = True
                backbone.batch_first = backbone.layers[0].self_attn.batch_first

                # sits_config = {
                #     'patch_size': 5,
                #     'num_classes': 15,
                # }
                # sits.BERTClassification()
            elif backbone == 'vit_B_16_imagenet1k':
                """
                pip install pytorch_pretrained_vit
                """
                from pytorch_pretrained_vit import ViT
                vit_model = ViT('B_16_imagenet1k', pretrained=True)
                backbone = vit_model.transformer
                # assert token_dim == 708
            elif backbone == 'vit_B_16':
                from pytorch_pretrained_vit import ViT
                vit_model = ViT('B_16', pretrained=True)
                backbone = vit_model.transformer
                # assert token_dim == 708
            elif backbone in SPLIT_ATTENTION_ENCODERS:
                encoder_config = transformer.encoder_configs[backbone]
                backbone = transformer.FusionEncoder(
                    **encoder_config,
                    in_features=position_encoder.output_dim + token_dim,
                    # attention_impl=self.hparams.attention_impl,
                    dropout=0.1,
                )
            else:
                raise KeyError(backbone)

        super().__init__()
        self.save_hyperparameters(ignore=["position_encoder"])

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
        # ~/code/watch/geowatch/tasks/fusion/methods/sequence_aware.py
        # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        self.saliency_weights = self._coerce_saliency_weights('auto')
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

        for s, c in sorted(sensor_modes):
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

            if tokenizer == "simple_conv":
                tokenizer_layer = nn.Sequential(
                    PadToMultiple(token_width, value=0.0),
                    nn.Conv2d(
                        in_chan,
                        token_dim,
                        kernel_size=token_width,
                        stride=token_width,
                    ),
                )
            elif tokenizer == "resnet18":
                resnet = tv_models.resnet18(tv_models.ResNet18_Weights.IMAGENET1K_V1)
                resnet.conv1 = nn.Conv2d(in_chan, resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
                resnet = feature_extraction.create_feature_extractor(resnet, return_nodes=["layer4"])
                tokenizer_layer = nn.Sequential(
                    ResNetShim(resnet),
                    nn.Conv2d(512, token_dim, 1),
                )
            else:
                raise NotImplementedError(tokenizer)

            # key = sanitize_key(str((s, c)))
            key = f'{s}:{c}'
            self.sensor_channel_tokenizers[key] = nn.Sequential(
                input_norm,
                NanToNum(0.0),
                tokenizer_layer,
            )

        self.position_encoder = position_encoder
        # self.position_encoder = RandomFourierPositionalEncoder(3, 16)
        # position_dim = self.position_encoder.output_dim

        self.pre_backbone = pre_backbone
        self.backbone = backbone
        self.post_backbone = post_backbone
        # self.backbone = TransformerEncoderDecoder(
        #     encoder_depth=backbone_encoder_depth,
        #     decoder_depth=backbone_decoder_depth,
        #     dim=token_dim + position_dim,
        #     queries_dim=position_dim,
        #     logits_dim=token_dim,
        #     cross_heads=backbone_cross_heads,
        #     latent_heads=backbone_latent_heads,
        #     cross_dim_head=backbone_cross_dim_head,
        #     latent_dim_head=backbone_latent_dim_head,
        #     weight_tie_layers=backbone_weight_tie_layers,
        # )

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

        self.magic_padding_value = -99999999.0  # Magic placeholder value

        for prop in head_properties:
            head_name = prop['name']
            global_weight = self.global_head_weights[head_name]
            if global_weight > 0:
                self.criterions[head_name] = coerce_criterion(prop['loss'],
                                                              prop['weights'],
                                                              ohem_ratio=ohem_ratio,
                                                              focal_gamma=focal_gamma)

                if self.hparams.decoder == "upsample":
                    self.heads[head_name] = nn.Sequential(
                        nn.Upsample(scale_factor=(token_width, token_width), mode="bilinear"),
                        nn.Conv2d(
                            token_dim,
                            prop['channels'],
                            kernel_size=5,
                            padding="same",
                            bias=False),
                    )
                elif self.hparams.decoder == "trans_conv":
                    self.heads[head_name] = nn.Sequential(
                        # ShapePreservingTransformerEncoder(
                        #     nn.TransformerEncoderLayer(token_dim, 8, dim_feedforward=512, dropout=0.1, activation="gelu", batch_first=True, norm_first=True),
                        #     num_layers=2,
                        # ),
                        ShapePreservingTransformerEncoder(
                            token_dim,
                            num_layers=2,
                            batch_dim=0,
                            chan_dim=1,
                        ),
                        nn.Conv2d(token_dim, token_width * token_width * prop['channels'], 1, bias=False),
                        Rearrange(
                            "batch (chan dh dw) height width -> batch chan (height dh) (width dw)",
                            dh=token_width, dw=token_width),
                    )
                elif self.hparams.decoder == "simple_conv":
                    self.heads[head_name] = nn.Sequential(
                        nn.Conv2d(token_dim, token_width * token_width * prop['channels'], 1, bias=False),
                        Rearrange(
                            "batch (chan dh dw) height width -> batch chan (height dh) (width dw)",
                            dh=token_width, dw=token_width),
                    )
                else:
                    raise NotImplementedError(decoder)

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

        self._prev_batch_size = None

    def process_input_tokens(self, example):
        """
        Example:
            >>> from geowatch.tasks import fusion
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     backbone=backbone,
            >>> )
            >>> example = model.demo_batch(width=64, height=65)[0]
            >>> input_tokens = model.process_input_tokens(example)
            >>> assert len(input_tokens) == len(example["frames"])
            >>> assert len(input_tokens[0]) == len(example["frames"][0]["modes"])
        """

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

                token_encodings = self.position_encoder(token_encodings[None], token_scales[None])[0]

                tokens = torch.concat([
                    tokens,
                    token_encodings,
                ])
                frame_tokens.append(tokens)

            example_tokens.append(frame_tokens)
        return example_tokens

    def process_query_tokens(self, example):
        """
        Example:
            >>> from geowatch.tasks import fusion
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     backbone=backbone,
            >>> )
            >>> example = model.demo_batch(width=64, height=65)[0]
            >>> query_tokens = model.process_query_tokens(example)
            >>> assert len(query_tokens) == len(example["frames"])
        """
        example_tokens = []
        for frame in example["frames"]:

            # space
            height, width = frame["output_dims"]
            height = height + to_next_multiple(height, self.hparams.token_width)
            width = width + to_next_multiple(width, self.hparams.token_width)
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

            token_encodings = self.position_encoder(token_encodings[None], token_scales[None])[0]

            example_tokens.append(token_encodings)
        return example_tokens

    def forward(self, batch):
        """
        Example:
            >>> from geowatch.tasks import fusion
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     backbone=backbone,
            >>>     position_encoder=position_encoder,
            >>> )
            >>> batch = model.demo_batch(width=64, height=65)
            >>> batch += model.demo_batch(width=55, height=75)
            >>> outputs = model.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"

        Example:
            >>> from geowatch.tasks import fusion
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     backbone=backbone,
            >>>     decoder="simple_conv",
            >>> )
            >>> batch = model.demo_batch(width=64, height=65)
            >>> batch += model.demo_batch(width=55, height=75)
            >>> outputs = model.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"

        Example:
            >>> from geowatch.tasks import fusion
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     backbone=backbone,
            >>>     decoder="trans_conv",
            >>> )
            >>> batch = model.demo_batch(width=64, height=65)
            >>> batch += model.demo_batch(width=55, height=75)
            >>> outputs = model.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"

        Example:
            >>> from geowatch.tasks import fusion
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=0,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=0,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     backbone=backbone,
            >>>     decoder="trans_conv",
            >>> )
            >>> batch = model.demo_batch(width=64, height=65)
            >>> batch += model.demo_batch(width=55, height=75)
            >>> outputs = model.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"

        Example:
            >>> # xdoctest: +REQUIRES(module:mmseg)
            >>> from geowatch.tasks import fusion
            >>> from geowatch.tasks.fusion.architectures.transformer import MM_VITEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = MM_VITEncoderDecoder(
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     backbone=backbone,
            >>>     position_encoder=position_encoder,
            >>> )
            >>> batch = model.demo_batch(width=64, height=65)
            >>> batch += model.demo_batch(width=55, height=75)
            >>> outputs = model.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"

        Example:
            >>> # xdoctest: +REQUIRES(module:mmseg)
            >>> from geowatch.tasks import fusion
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> self = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     #token_dim=708,
            >>>     token_dim=768 - 60,
            >>>     backbone='vit_B_16_imagenet1k',
            >>>     position_encoder=position_encoder,
            >>> )
            >>> batch = self.demo_batch(width=64, height=65)
            >>> batch += self.demo_batch(width=55, height=75)
            >>> outputs = self.forward(batch)
            >>> for task_key, task_outputs in outputs.items():
            >>>     if "probs" in task_key: continue
            >>>     if task_key == "class": task_key = "class_idxs"
            >>>     for task_pred, example in zip(task_outputs, batch):
            >>>         for frame_idx, (frame_pred, frame) in enumerate(zip(task_pred, example["frames"])):
            >>>             if (frame_idx == 0) and task_key.startswith("change"): continue
            >>>             assert frame_pred.shape[1:] == frame[task_key].shape, f"{frame_pred.shape} should equal {frame[task_key].shape} for task '{task_key}'"

        Ignore:
            from geowatch.tasks import fusion
            from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            position_encoder = geowatch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder(in_dims=3, max_freq=3, num_freqs=16)
            token_dim = 256
            backbone = TransformerEncoderDecoder(
                encoder_depth=6,
                decoder_depth=0,
                dim=position_encoder.output_dim + token_dim,
                queries_dim=position_encoder.output_dim,
                logits_dim=token_dim,
                latent_dim_head=1024,
            )
            channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            model = fusion.methods.HeterogeneousModel(
                token_dim=token_dim,
                token_width=8,
                classes=classes,
                dataset_stats=dataset_stats,
                input_sensorchan=channels,
                position_encoder=position_encoder,
                backbone=backbone,
                spatial_scale_base=1,
                global_change_weight=0,
                global_class_weight=0,
                global_saliency_weight=1,
                decoder="simple_conv",
            )
            batch = model.demo_batch(width=64, height=65)
            batch += model.demo_batch(width=55, height=75)
            outputs = model.forward(batch)


        """

        # ==================
        # Compute input sequences and shapes
        # ==================

        # Lists to stash sequences and shapes
        orig_input_shapes = []
        orig_input_seqs = []

        for example in batch:

            # Each example, containing potentially more than one mode,
            # is stemmed and then we save its original shape
            input_tokens = self.process_input_tokens(example)
            input_shapes = [
                [
                    mode_tokens.shape[1:]
                    for mode_tokens in frame_tokens
                ]
                for frame_tokens in input_tokens
            ]
            orig_input_shapes.append(input_shapes)

            # For the downstream transformer, we flatten and concatenate the
            # stemmed tokens
            input_token_seq = torch.concat([
                torch.concat([
                    einops.rearrange(mode_tokens, "chan ... -> (...) chan")
                    for mode_tokens in frame_tokens
                ])
                for frame_tokens in input_tokens
            ])
            orig_input_seqs.append(input_token_seq)

        if len(orig_input_seqs) == 0:
            print(f'batch={batch}')
            print('Skipping batch')
            return None

        self._prev_batch_size = len(orig_input_seqs)

        # Each example may have a different number of tokens, so we perform
        # some padding and compute a mask of where those padded tokens are

        input_seqs = nn.utils.rnn.pad_sequence(
            orig_input_seqs,
            batch_first=True,
            padding_value=self.magic_padding_value,
        )
        # Remove the placeholder
        input_masks = input_seqs[..., 0] > self.magic_padding_value
        input_seqs[~input_masks] = 0.

        # ==================
        # Compute query sequences and shapes
        # (Should be similar/identical to the input proceedure)
        # BUT only if we need to
        # ==================

        B, S, D = input_seqs.shape

        if self.pre_backbone is not None:
            # Fixup dims for the backbone
            input_seqs = self.pre_backbone(input_seqs.view(-1, D)).view(B, S, -1)

        has_decoder = getattr(self.backbone, 'has_decoder', False)
        self.backbone.has_decoder = has_decoder
        if has_decoder:
            # Lists to stash sequences and shapes
            orig_query_shapes = []
            orig_query_seqs = []

            for example in batch:

                # Each example, containing potentially more than one task,
                # a map of query position tokens are computed and then we save
                # their original shape
                query_tokens = self.process_query_tokens(example)
                query_shapes = [
                    frame_tokens.shape[1:]
                    for frame_tokens in query_tokens
                ]
                orig_query_shapes.append(query_shapes)

                # For the downstream transformer, we flatten and concatenate
                # the position embeddings
                query_token_seq = torch.concat([
                    einops.rearrange(frame_tokens, "chan ... -> (...) chan")
                    for frame_tokens in query_tokens
                ])
                orig_query_seqs.append(query_token_seq)

            # Each example may have a different number of queries, so we perform
            # some padding and compute a mask of where those padded tokens are
            query_seqs = nn.utils.rnn.pad_sequence(
                orig_query_seqs,
                batch_first=True,
                padding_value=self.magic_padding_value,
            )
            # Remove the placeholder
            query_masks = query_seqs[..., 0] > self.magic_padding_value
            query_seqs[~query_masks] = 0.

        # ==================
        # Forward pass!
        # ==================
        if self.backbone.has_decoder:
            output_seqs = self.backbone(
                input_seqs,
                mask=input_masks,
                queries=query_seqs,
            )
            output_shapes = orig_query_shapes
            output_masks = query_masks
        else:
            # batch_first = getattr(self.backbone, 'batch_first', True)
            is_sits_bert = getattr(self.backbone, 'is_sits_bert', False)
            if is_sits_bert:
                # Special case for pretrained BERT
                # TODO: wrap the model to conform to the API here instead
                # of directly hacking this function.
                _input_seqs = input_seqs.transpose(0, 1)
                # _input_masks = input_masks.transpose(0, 1)
                _output_seqs = self.backbone(
                    _input_seqs,
                    src_key_padding_mask=~input_masks,
                )
                output_seqs = _output_seqs.transpose(0, 1)
                output_shapes = orig_input_shapes
                output_masks = input_masks
            else:
                # Normal case.
                output_seqs = self.backbone(
                    input_seqs,
                    mask=input_masks,
                )
                output_shapes = orig_input_shapes
                output_masks = input_masks

            # Uncomment if old sits models need repackaing
            HACK_TOKEN_DIMS = self.post_backbone is None
            if HACK_TOKEN_DIMS:
                # hack for VIT. Drops feature dims to allow for running
                if output_seqs.shape[2] != self.hparams.token_dim:
                    output_seqs = output_seqs[:, :, 0:self.hparams.token_dim]

        if self.post_backbone is not None:
            # Fixup dims for the backbone
            output_seqs = self.post_backbone(output_seqs.view(-1, output_seqs.shape[-1]))
            output_seqs = output_seqs.view(B, S, -1)

        # ==================
        # Decompose outputs into the appropriate output shape
        # ==================

        # The container for all of our outputs
        outputs = dict()

        for task_name, task_head in self.heads.items():
            task_outputs = []
            task_probs = []
            for output_seq, query_mask, frame_shapes, example in zip(output_seqs, output_masks, output_shapes, batch):
                output_seq = output_seq[query_mask]  # only want valid values we actually requested

                seq_outputs = []
                seq_probs = []
                frame_sizes = [
                    np.reshape(pos_shape_seq, [-1, 2]).prod(axis=1).sum()
                    for pos_shape_seq in frame_shapes
                ]
                output_frame_seqs = torch.split(output_seq, frame_sizes)
                for output_frame_seq, frame_shape, frame in zip(output_frame_seqs, frame_shapes, example["frames"]):

                    if self.backbone.has_decoder:
                        # Rearrange token subsequence into image shaped tensor
                        height, width = frame_shape
                        output = einops.rearrange(
                            output_frame_seq,
                            "(height width) chan -> chan height width",
                            height=height, width=width,
                        )

                    else:
                        max_mode_size = (
                            max([h for h, _ in frame_shape]),
                            max([w for _, w in frame_shape]),
                        )
                        mode_sizes = [h * w for h, w in frame_shape]
                        output_mode_seqs = torch.split(output_frame_seq, mode_sizes)

                        output = torch.mean(torch.concat([
                            torch.nn.functional.interpolate(
                                einops.rearrange(
                                    mode_seq,
                                    "(height width) chan -> 1 chan height width",
                                    height=mode_height, width=mode_width,
                                ),
                                size=max_mode_size,
                                mode='bilinear',
                                align_corners=True,
                            )
                            for mode_seq, (mode_height, mode_width) in zip(output_mode_seqs, frame_shape)
                        ], dim=0), dim=0)

                        # # If we might need to upsample our predictions
                        # output = nn.functional.upsample_bilinear(output[None], size=[tar_height, tar_width])[0]

                    # Compute task preds
                    output = task_head(output[None])[0]

                    # Clip to desired shape
                    tar_height, tar_width = frame["output_dims"]
                    output = output[:, :tar_height, :tar_width]

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
            >>> from geowatch.tasks import fusion
            >>> import torch
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     decoder="trans_conv",
            >>>     backbone=backbone,
            >>> )
            >>> batch = model.demo_batch(batch_size=2, width=64, height=65, num_timesteps=3)
            >>> outputs = model.shared_step(batch)
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> optimizer.zero_grad()
            >>> loss = outputs["loss"]
            >>> loss.backward()
            >>> optimizer.step()

        Example:
            >>> from geowatch.tasks import fusion
            >>> import torch
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     decoder="trans_conv",
            >>>     backbone=backbone,
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
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     decoder="trans_conv",
            >>>     backbone=backbone,
            >>> )
            >>> batch = model.demo_batch(width=64, height=65)
            >>> for cutoff in [-1, -2]:
            >>>     degraded_example = model.demo_batch(width=55, height=75, num_timesteps=3)[0]
            >>>     degraded_example["frames"] = degraded_example["frames"][:cutoff]
            >>>     batch += [degraded_example]
            >>> outputs = model.shared_step(batch)
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> optimizer.zero_grad()
            >>> loss = outputs["loss"]
            >>> loss.backward()
            >>> optimizer.step()

        Example:
            >>> from geowatch.tasks import fusion
            >>> import torch
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> channels, classes, dataset_stats = fusion.methods.HeterogeneousModel.demo_dataset_stats()
            >>> model = fusion.methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     position_encoder=position_encoder,
            >>>     decoder="trans_conv",
            >>>     backbone=backbone,
            >>> )
            >>> batch = model.demo_batch(batch_size=1, width=64, height=65, num_timesteps=3, nans=0.1)
            >>> batch += model.demo_batch(batch_size=1, width=64, height=65, num_timesteps=3, nans=0.5)
            >>> batch += model.demo_batch(batch_size=1, width=64, height=65, num_timesteps=3, nans=1.0)
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

        batch_size = len(batch)

        outputs = self(batch)
        if outputs is None:
            return None

        if not with_loss:
            return outputs

        frame_losses = []
        for task_name in self.heads:
            for pred_seq, example in zip(outputs[task_name], batch):
                for pred, frame in zip(pred_seq, example["frames"]):

                    task_labels_key = self.task_to_keynames[task_name]["labels"]
                    labels = frame[task_labels_key]

                    self.log(f"{stage}_{task_name}_logit_mean", pred.mean(),
                             batch_size=batch_size, rank_zero_only=True)

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
                        labels_ = labels.long()
                        has_ignore = labels_.min() < 0
                        if has_ignore:
                            ignore_flags = labels_ < 0
                            ohe_size = criterion.in_channels + 1
                            labels_ = labels_.clone()
                            labels_[ignore_flags] = criterion.in_channels
                        else:
                            ohe_size = criterion.in_channels

                        loss_labels = kwarray.one_hot_embedding(
                            labels_,
                            ohe_size,
                            dim=0)
                        loss_labels_ = loss_labels[:, valid_mask]

                        if has_ignore:
                            # FIXME: inefficient to just drop these, but it
                            # should work.
                            # could improve kwarray.one_hot_embedding to
                            # allow the user to specify an ignore_index
                            loss_labels_ = loss_labels_[:-1, ...]
                    else:
                        raise KeyError(criterion.target_encoding)

                    loss = criterion(
                        pred_[None],
                        loss_labels_[None],
                    )

                    if loss.isnan().any():
                        print('!!!!!!!!!!!!!!!!!!!')
                        print('!!!!!!!!!!!!!!!!!!!')
                        print('Discovered NaN loss')
                        print('loss = {}'.format(ub.urepr(loss, nl=1)))
                        print('pred = {}'.format(ub.urepr(pred, nl=1)))
                        print('frame = {}'.format(ub.urepr(frame, nl=1)))
                        print('!!!!!!!!!!!!!!!!!!!')
                        print('!!!!!!!!!!!!!!!!!!!')

                    loss *= task_weights_
                    frame_losses.append(
                        self.global_head_weights[task_name] * loss.mean()
                    )

                    LOG_METRICS = 0  # FIXME: recent (30c8974d6d6) update broke this, why?
                    if LOG_METRICS:
                        metric_values = self.head_metrics[f"{stage}_stage"][task_name](
                            pred.argmax(dim=0).flatten(),
                            # pred[None],
                            labels.flatten().long(),
                        )
                        self.log_dict(
                            metric_values,
                            prog_bar=True,
                            batch_size=batch_size,
                        )

        outputs["loss"] = sum(frame_losses) / len(frame_losses)
        self.log(f"{stage}_loss", outputs["loss"], prog_bar=True, batch_size=batch_size)
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

    def log_grad_norm(self, grad_norm_dict) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        Overloads log_grad_norm so we can supress the batch_size warning
        """
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=True,
                      prog_bar=False, logger=True,
                      batch_size=self._prev_batch_size)

    def save_package(self, package_path, verbose=1):
        """
        CommandLine:
            xdoctest -m geowatch.tasks.fusion.methods.heterogeneous HeterogeneousModel.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.heterogeneous import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> model = self = methods.HeterogeneousModel(
            >>>     position_encoder=position_encoder,
            >>>     input_sensorchan=5,
            >>>     decoder="upsample",
            >>>     backbone=backbone,
            >>> )

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.HeterogeneousModel.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
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
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.heterogeneous import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> model = self = methods.HeterogeneousModel(
            >>>     position_encoder=position_encoder,
            >>>     input_sensorchan=5,
            >>>     decoder="simple_conv",
            >>>     backbone=backbone,
            >>> )

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.HeterogeneousModel.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
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
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.heterogeneous import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
            >>> position_encoder = methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
            >>> backbone = TransformerEncoderDecoder(
            >>>     encoder_depth=1,
            >>>     decoder_depth=1,
            >>>     dim=position_encoder.output_dim + 16,
            >>>     queries_dim=position_encoder.output_dim,
            >>>     logits_dim=16,
            >>>     cross_heads=1,
            >>>     latent_heads=1,
            >>>     cross_dim_head=1,
            >>>     latent_dim_head=1,
            >>> )
            >>> model = self = methods.HeterogeneousModel(
            >>>     position_encoder=position_encoder,
            >>>     input_sensorchan=5,
            >>>     decoder="trans_conv",
            >>>     backbone=backbone,
            >>> )

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.HeterogeneousModel.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
            >>> recon = load_model_from_package(package_path)
            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = model.state_dict()
            >>> assert recon is not model
            >>> assert set(recon_state) == set(recon_state)
            >>> for key in recon_state.keys():
            >>>     assert (model_state[key] == recon_state[key]).all()
            >>>     assert model_state[key] is not recon_state[key]

        Ignore:
            7z l $HOME/.cache/geowatch/tests/package/my_package.pt
        """
        self._save_package(package_path, verbose=verbose)

    # hack because of inheritence rules
    configure_optimizers = WatchModuleMixins.configure_optimizers
