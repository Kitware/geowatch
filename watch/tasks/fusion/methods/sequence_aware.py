import einops
import kwarray
import kwcoco
import ubelt as ub
import torch
import torchmetrics

import numpy as np
import netharn as nh
import pytorch_lightning as pl
import perceiver_pytorch as perceiver

from torch import nn
from torch.optim import lr_scheduler
from watch import heuristics
from watch.tasks.fusion.methods.network_modules import _class_weights_from_freq
from watch.tasks.fusion.methods.network_modules import coerce_criterion
from watch.tasks.fusion.methods.network_modules import RobustModuleDict
from watch.tasks.fusion.methods.network_modules import RearrangeTokenizer
from watch.tasks.fusion.methods.network_modules import ConvTokenizer
from watch.tasks.fusion.methods.network_modules import LinearConvTokenizer
from watch.tasks.fusion.methods.network_modules import DWCNNTokenizer
from watch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
from watch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder

# import scriptconfig as scfg

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def sanitize_key(key):
    return key.replace(".", "-")


class FourierPositionalEncoding(nn.Module):
    def __init__(self, in_dims, num_steps, max_freq=10.0):
        super().__init__()
        self.scales = nn.Parameter(torch.pi * torch.linspace(1., max_freq, num_steps), requires_grad=False)
        self.output_dim = in_dims + (in_dims * num_steps * 2)

    def forward(self, x):
        orig_x = x
        x = torch.einsum("xhw,s->xshw", x, self.scales.type_as(x))
        x = einops.rearrange(x, "x s h w -> (x s) h w")
        return torch.concat([x.sin(), x.cos(), orig_x], dim=0)
        
class RandomFourierPositionalEncoding(nn.Module):
    def __init__(self, in_dims, half_out_dims):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dims, half_out_dims).float(), requires_grad=False)
        self.output_dim = 2*half_out_dims
        
    def forward(self, x):
        x = 2 * torch.pi * torch.einsum("xy,x...->y...", self.weight, x)
        return torch.concat([x.sin(), x.cos()], dim=0)


class SequenceAwareModel(pl.LightningModule, WatchModuleMixins):

    _HANDLES_NANS = True

    def get_cfgstr(self):
        cfgstr = f'{self.hparams.name}_{self.hparams.arch_name}_SA'
        return cfgstr

    def __init__(
        self,
        classes=10,
        dataset_stats=None,
        input_sensorchan=None,
        name: str = "unnamed_model",
        arch_name: str = "transformer",
        positional_encoder: str = "random",
        optimizer: str = "RAdam",  # TODO: remove and push to the cli
        learning_rate: float = 0.001,  # TODO: remove and push to the cli
        weight_decay: float = 0.0,  # TODO: remove and push to the cli
        positive_change_weight: float = 1.0,
        negative_change_weight: float = 1.0,
        class_weights: str = "auto",
        saliency_weights: str = "auto",
        stream_channels: int = 8,
        tokenizer: str = "rearrange",  # TODO: replace control string with a module, possibly a subclass
        token_norm: str = "none",
        decoder: str = "mlp",  # TODO: replace control string with a module, possibly a subclass
        dropout: float = 0.1,
        global_class_weight: float = 1.0,
        global_change_weight: float = 1.0,
        global_saliency_weight: float = 1.0,
        modulate_class_weights: str = "",
        change_loss: str = "cce",  # TODO: replace control string with a module, possibly a subclass
        class_loss: str = "focal",  # TODO: replace control string with a module, possibly a subclass
        saliency_loss: str = "focal",  # TODO: replace control string with a module, possibly a subclass
        change_head_hidden: int = 2,
        class_head_hidden: int = 2,
        saliency_head_hidden: int = 2,
        window_size: int = 8,  # TODO: remove with replacement of tokenizer
        # squash_modes: bool = False,
        decouple_resolution: bool = False,
        attention_impl: str = "exact",  # TODO: remove with replacement of hardcoded perceiver
        multimodal_reduce: str = "max",  # TODO: remove with replacement of hardcoded perceiver
        perceiver_depth: int = 4,  # TODO: remove with replacement of hardcoded perceiver
        perceiver_latents: int = 512,  # TODO: remove with replacement of hardcoded perceiver

    ):
        """
        Args:
            name: Specify a name for the experiment. (Unsure if the Model is the place for this)
            arch_name: Specify the backbone architecture.
            optimizer: Optimizer name supported by the netharn API.
            class_weights: Class weighting strategy.
            saliency_weights: Class weighting strategy.
            stream_channels: Number of channels to normalize each project stream to.
            tokenizer:
                How image patches are broken into tokens. rearrange is a 1x1
                MLP and grouping of pixel grids. dwcnn is a is a mobile
                convolutional stem. conv7 is a simple 1x1x7x7 convolutional
                stem. linconv is a stack of 3x3 grouped convolutions without
                any nonlinearity
            modulate_class_weights:
                S special syntax that lets the user modulate automatically
                computed class weights. Should be a comma separated list of
                name*weight or name*weight+offset. E.g.
                `negative*0,background*0.001,No Activity*0.1+1`
            saliency_loss: Saliency is trained to match any "positive/foreground/salient" class
            change_head_hidden: Number of hidden layers in the CHANGE head. I.e. the depth of the head.
            class_head_hidden: Number of hidden layers in the CLASS head. I.e. the depth of the head.
            saliency_head_hidden: Number of hidden layers in the SALIENCY head. I.e. the depth of the head.
            decouple_resolution: This turns on logic to decouple input and output resolutions. Probably very slow.
            attention_impl:
                Implementation for attention computation. Can be:
                'exact' - the original O(n^2) method.
                'performer' - a linear approximation.
                'reformer' - a LSH approximation.
            multimodal_reduce: operation used to combine multiple modes from the same timestep
            perceiver_depth: How many layers used by the perceiver model.
            perceiver_latents: How many latents used by the perceiver model.

        Example:
            >>> # Note: it is important that the non-kwargs are saved as hyperparams
            >>> from watch.tasks.fusion.methods.sequence_aware import SequenceAwareModel
            >>> model = SequenceAwareModel(input_sensorchan='r|g|b', positional_encoder="random")
            >>> assert "arch_name" in model.hparams
            >>> assert "positional_encoder" in model.hparams
            >>> assert "classes" in model.hparams
            >>> assert "dataset_stats" in model.hparams
            >>> assert "input_sensorchan" in model.hparams
            
        Example:
            >>> # Note: it is important that the non-kwargs are saved as hyperparams
            >>> from watch.tasks.fusion.methods.sequence_aware import SequenceAwareModel
            >>> model = SequenceAwareModel(input_sensorchan='r|g|b', positional_encoder="fourier")
            >>> assert "arch_name" in model.hparams
            >>> assert "positional_encoder" in model.hparams
            >>> assert "classes" in model.hparams
            >>> assert "dataset_stats" in model.hparams
            >>> assert "input_sensorchan" in model.hparams
        """

        assert arch_name in ['transformer', 'perceiver']
        assert positional_encoder in ['random', 'fourier']
        assert tokenizer in ['dwcnn', 'rearrange', 'conv7', 'linconv']
        assert token_norm in ['none', 'auto', 'group', 'batch']
        assert decoder in ['mlp', 'segmenter']
        assert attention_impl in ["exact", "performer", "reformer"]

        # =================================================================================
        # =================================================================================
        # START IMPORT FROM MULTIMODAL-TRANSFORMER

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

        self.global_class_weight = global_class_weight
        self.global_change_weight = global_change_weight
        self.global_saliency_weight = global_saliency_weight
        self.global_head_weights = {
            'class': global_class_weight,
            'change': global_change_weight,
            'saliency': global_saliency_weight,
        }

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
                if self.hparams.modulate_class_weights:
                    import re
                    parts = [p.strip() for p in self.hparams.modulate_class_weights.split(',')]
                    parts = [p for p in parts if p]
                    for part in parts:
                        toks = re.split('([+*])', part)
                        catname = toks[0]
                        rest_iter = iter(toks[1:])
                        weight = using_class_weights[catname]
                        nrhtoks = len(toks) - 1
                        assert nrhtoks % 2 == 0
                        nstmts = nrhtoks // 2
                        for _ in range(nstmts):
                            opcode = next(rest_iter)
                            arg = float(next(rest_iter))
                            if opcode == '*':
                                weight = weight * arg
                            elif opcode == '+':
                                weight = weight * arg
                            else:
                                raise KeyError(opcode)
                        # Modulate
                        using_class_weights[catname] = weight

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

        MODAL_AGREEMENT_CHANS = self.hparams.stream_channels
        self.tokenizer = tokenizer
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
            if s not in self.sensor_channel_tokenizers:
                self.sensor_channel_tokenizers[s] = RobustModuleDict()

            if tokenizer == 'rearrange':
                tokenize = RearrangeTokenizer(
                    in_channels=in_chan, agree=MODAL_AGREEMENT_CHANS,
                    window_size=self.hparams.window_size,
                )
            elif tokenizer == 'conv7':
                # Hack for old models
                in_features_raw = MODAL_AGREEMENT_CHANS
                tokenize = ConvTokenizer(in_chan, in_features_raw, norm=None)
            elif tokenizer == 'linconv':
                in_features_raw = MODAL_AGREEMENT_CHANS * 64
                tokenize = LinearConvTokenizer(in_chan, in_features_raw)
            elif tokenizer == 'dwcnn':
                in_features_raw = MODAL_AGREEMENT_CHANS * 64
                tokenize = DWCNNTokenizer(in_chan, in_features_raw, norm=token_norm)
            else:
                raise KeyError(tokenizer)

            if input_stats is None:
                input_norm = nh.layers.InputNorm()
            else:
                stats = input_stats.get((s, c), None)
                if stats is None:
                    input_norm = nh.layers.InputNorm()
                else:
                    input_norm = nh.layers.InputNorm(**stats)

            # self.sensor_channel_tokenizers[s][c] = tokenize
            key = sanitize_key(str((s, c)))
            try:
                sensor_chan_input_norm = input_norm
            except (KeyError, TypeError) as e:
                print(e)
                sensor_chan_input_norm = nn.Identity()

            self.sensor_channel_tokenizers[key] = nn.Sequential(
                sensor_chan_input_norm,
                tokenize,
            )
            in_features_raw = tokenize.out_channels

        # for (s, c), stats in input_stats.items():
        #     self.sensor_channel_tokenizers[s][c] = tokenize

        in_features_pos = 64  # 6 * 8   # 6 positional features with 8 dims each (TODO: be robust)
        in_features = in_features_pos + in_features_raw
        self.in_features = in_features
        self.in_features_pos = in_features_pos
        self.in_features_raw = in_features_raw

        # END IMPORT FROM MULTIMODAL-TRANSFORMER
        # =================================================================================
        # =================================================================================

        if self.hparams.positional_encoder == "fourier":
            base_positional_encoder = FourierPositionalEncoding(3, 64, 60)
        elif self.hparams.positional_encoder == "random":
            base_positional_encoder = RandomFourierPositionalEncoding(3, 64)
        else:
            raise KeyError(positional_encoder)
        
        self.positional_encoders = nn.ModuleDict({
            sanitize_key(str(key)): nn.Sequential(
                base_positional_encoder,
                nn.Conv2d(
                    base_positional_encoder.output_dim,
                    in_features_pos,
                    1,
                ),
            )
            for key in list(self.unique_sensor_modes) + ["change", "saliency", "class"]
        })

        if self.hparams.arch_name == "perciever":
            self.backbone = perceiver.PerceiverIO(
                depth=perceiver_depth,
                dim=in_features,
                queries_dim=in_features_pos,
                num_latents=perceiver_latents,
                latent_dim=128,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                weight_tie_layers=True,
                decoder_ff=True,
                logits_dim=in_features,
            )
        elif self.hparams.arch_name == "transformer":
            self.backbone = TransformerEncoderDecoder(
                depth=perceiver_depth,
                dim=in_features,
                queries_dim=in_features_pos,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                weight_tie_layers=True,
                decoder_ff=True,
                logits_dim=in_features,
            )
        else:
            raise KeyError(arch_name)
            
            
        feat_dim = in_features_pos

        self.change_head_hidden = change_head_hidden
        self.class_head_hidden = class_head_hidden
        self.saliency_head_hidden = saliency_head_hidden

        self.class_loss = class_loss
        self.change_loss = change_loss
        self.saliency_loss = saliency_loss

        self.criterions = torch.nn.ModuleDict()
        self.heads = torch.nn.ModuleDict()

        head_properties = [
            {
                'name': 'change',
                'hidden': self.change_head_hidden,
                'channels': 2,
                'loss': self.change_loss,
                'weights': self.change_weights,
            },
            {
                'name': 'saliency',
                'hidden': self.saliency_head_hidden,
                'channels': self.saliency_num_classes,
                'loss': self.saliency_loss,
                'weights': self.saliency_weights,
            },
            {
                'name': 'class',
                'hidden': self.class_head_hidden,
                'channels': self.num_classes,
                'loss': self.class_loss,
                'weights': self.class_weights,
            },
        ]

        for prop in head_properties:
            head_name = prop['name']
            global_weight = self.global_head_weights[head_name]
            if global_weight > 0:
                self.criterions[head_name] = coerce_criterion(prop['loss'], prop['weights'])
                if self.hparams.decoder == 'mlp':
                    self.heads[head_name] = nh.layers.MultiLayerPerceptronNd(
                        dim=0,
                        in_channels=feat_dim,
                        hidden_channels=prop['hidden'],
                        out_channels=prop['channels'],
                        norm=None
                    )
                elif self.hparams.decoder == 'segmenter':
                    from watch.tasks.fusion.architectures import segmenter_decoder
                    self.heads[head_name] = segmenter_decoder.MaskTransformerDecoder(
                        d_model=feat_dim,
                        n_layers=prop['hidden'],
                        n_cls=prop['channels'],
                    )
                else:
                    raise KeyError(self.hparams.decoder)

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
            "change_acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'change_f1': FBetaScore(beta=1.0),
        })
        saliency_metrics = torchmetrics.MetricCollection({
            'saliency_f1': FBetaScore(beta=1.0),
        })

        self.head_metrics = nn.ModuleDict({
            f"{stage}_stage": nn.ModuleDict({
                "class": class_metrics.clone(prefix=f"{stage}_"),
                "change": change_metrics.clone(prefix=f"{stage}_"),
                "saliency": saliency_metrics.clone(prefix=f"{stage}_"),
            })
            for stage in ["train", "val", "test"]
        })

    def stem_process_example(self, example):
        """Deprecated. Functionality folded into `SequenceAwareModel.process_inputs(...)`."""
        modes = []
        for frame in example["frames"]:
            for mode_key, mode_image in frame["modes"].items():

                sensor_mode_key = sanitize_key(str((frame["sensor"], mode_key)))

                stemmed_mode = self.sensor_channel_tokenizers[sensor_mode_key](
                    mode_image.nan_to_num(0.0)[None].float()
                )[0]
                dtype = stemmed_mode.dtype
                device = stemmed_mode.device

                position = self.positional_encoders[sensor_mode_key](
                    torch.stack(
                        (frame["time_index"] * torch.ones(*stemmed_mode.shape[1:], dtype=dtype, device=device),) +
                        torch.meshgrid(
                            torch.linspace(-1, 1, stemmed_mode.shape[1], dtype=dtype, device=device),
                            torch.linspace(-1, 1, stemmed_mode.shape[2], dtype=dtype, device=device),
                        ),
                    ))

                # modes.append(stemmed_mode + position)
                modes.append(torch.concat([stemmed_mode, position], dim=0))

        return modes

    def encode_query_position(self, example, task_name, dtype, device):
        return [
            self.positional_encoders[task_name](torch.stack(
                (frame["time_index"] * torch.ones(*frame["output_dims"], dtype=dtype, device=device),) +
                torch.meshgrid(
                    torch.linspace(-1, 1, frame["output_dims"][0], dtype=dtype, device=device),
                    torch.linspace(-1, 1, frame["output_dims"][1], dtype=dtype, device=device),
                ),
            ))
            for frame in example["frames"]
        ]

    def process_inputs(self, example):
        """
        TODO: documentation about what this step is

        Example is a single item from a batch, and this pushes that example
        through the input stems and constructs the token sequence for that
        batch item as well as information about how to produce outputs for that
        sequence.

        """

        # process and stem frames and positions
        # inputs = self.stem_process_example(example)
        inputs = []
        for frame in example["frames"]:
            for mode_key, mode_image in frame["modes"].items():

                sensor_mode_key = sanitize_key(str((frame["sensor"], mode_key)))

                stemmed_mode = self.sensor_channel_tokenizers[sensor_mode_key](
                    mode_image.nan_to_num(0.0)[None].float()
                )[0]
                dtype = stemmed_mode.dtype
                device = stemmed_mode.device

                position = self.positional_encoders[sensor_mode_key](
                    torch.stack(
                        (frame["time_index"] * torch.ones(*stemmed_mode.shape[1:], dtype=dtype, device=device),) +
                        torch.meshgrid(
                            torch.linspace(-1, 1, stemmed_mode.shape[1], dtype=dtype, device=device),
                            torch.linspace(-1, 1, stemmed_mode.shape[2], dtype=dtype, device=device),
                        ),
                    ))

                # modes.append(stemmed_mode + position)
                inputs.append(torch.concat([stemmed_mode, position], dim=0))
        inputs = torch.concat([einops.rearrange(x, "c h w -> (h w) c") for x in inputs], dim=0)
        return inputs

    def process_outputs(self, example, force_dropout=False, dtype="float32", device="cpu"):
        """
        TODO: documentation about what this step is

        Example is a single item from a batch, and this pushes that example
        through the input stems and constructs the token sequence for that
        batch item as well as information about how to produce outputs for that
        sequence.

        """
        outputs = {}
        task_defs = [
            ("change", "change", "change_weights"),
            ("saliency", "saliency", "saliency_weights"),
            ("class", "class_idxs", "class_weights"),
        ]
        for task_name, labels_name, weights_name in task_defs:

            labels = [
                frame[labels_name] if (frame[labels_name] is not None)
                else torch.zeros(
                    frame["output_dims"],
                    dtype=torch.int32,
                    device=device)
                for frame in example["frames"]
            ]
            labels = torch.concat([einops.rearrange(x, "h w -> (h w)") for x in labels], dim=0)
            weights = [
                frame[weights_name] if (frame[weights_name] is not None)
                else torch.zeros(
                    frame["output_dims"],
                    dtype=dtype,
                    device=device)
                for frame in example["frames"]
            ]
            weights = torch.concat([einops.rearrange(x, "h w -> (h w)") for x in weights], dim=0)

            pos_enc = self.encode_query_position(example, task_name, dtype, device)
            pos_enc = torch.concat([einops.rearrange(x, "c h w -> (h w) c") for x in pos_enc], dim=0)

            # TODO:
            # Instead of masking by dropping out locations, we need a strategy
            # that simply zeros their weight. Again the modivating factor is
            # intermediate visualization. However, the efficiency of the
            # dropout strategy is something we should retain, but the
            # implementation should have an end-to-end forward pass that
            # maintains input shapes up to rearangements.

            # determine valid label locations                
            valid_mask = weights > 0.0
            pos_enc = pos_enc[valid_mask]
            labels = labels[valid_mask]
            weights = weights[valid_mask]

            outputs[task_name] = {
                "labels": labels,
                "weights": weights,
                "pos_enc": pos_enc,
                "mask": valid_mask,
                "shape": [frame["output_dims"] for frame in example["frames"]],
            }

        return outputs

    def reconstruct_output(self, output, mask, shapes):
        """
        Example:
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> channels, classes, dataset_stats = SequenceAwareModel.demo_dataset_stats()
            >>> self = SequenceAwareModel(
            >>>     tokenizer='linconv',
            >>>     decoder='segmenter', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> self.eval()
            >>> batch = self.demo_batch(width=64, height=65)
            >>> inputs = list(map(self.process_inputs, batch))
            >>> outputs = [self.process_outputs(example, dtype=inputs[0].dtype, device=inputs[0].device) for example in batch]
            >>> stacked_queries = {
            >>>     task_name: nn.utils.rnn.pad_sequence([
            >>>         example[task_name]["pos_enc"]
            >>>         for example in outputs
            >>>     ], batch_first=True, padding_value=0.0)
            >>>     for task_name in list(["saliency",])
            >>> }
            >>> stacked_weights = {
            >>>     task_name: nn.utils.rnn.pad_sequence([
            >>>         example[task_name]["weights"]
            >>>         for example in outputs
            >>>     ], batch_first=True, padding_value=0.0)
            >>>     for task_name in list(["saliency",])
            >>> }
            >>> padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=-1000.0)
            >>> padded_valids = (padded_inputs[..., 0] > -1000.0).bool()
            >>> padded_inputs[~padded_valids] = 0.0
            >>> logits = self.forward(padded_inputs, stacked_queries, input_mask=padded_valids)
            >>> logits = einops.rearrange(logits["saliency"], "batch chan seq -> batch seq chan")
            >>> shapes = [list(frame["output_dims"]) for frame in batch[0]["frames"]]
            >>> recon = self.reconstruct_output(logits[0], stacked_weights["saliency"][0] > 0., shapes)
            >>> print('batch')
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> print('recon')
            >>> print(nh.data.collate._debug_inbatch_shapes(recon))
        """
        big_canvas = torch.nan * torch.zeros(mask.shape[0], output.shape[-1], dtype=output.dtype, device=output.device)
        big_canvas[mask] = output[:mask.sum()]

        canvases = []
        for canvas, shape in zip(torch.split(big_canvas, [w * h for w, h in shapes]), shapes):
            canvas = canvas.reshape(list(shape) + [output.shape[-1], ])
            canvases.append(canvas)
        return canvases

    def forward(self, inputs, queries, input_mask=None):
        """
        Example:
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> channels, classes, dataset_stats = SequenceAwareModel.demo_dataset_stats()
            >>> self = SequenceAwareModel(
            >>>     tokenizer='linconv',
            >>>     decoder='segmenter', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch(width=64, height=65)
            >>> inputs = list(map(self.process_inputs, batch))
            >>> outputs = [self.process_outputs(example, dtype=inputs[0].dtype, device=inputs[0].device) for example in batch]
            >>> stacked_queries = {
            >>>     task_name: nn.utils.rnn.pad_sequence([
            >>>         example[task_name]["pos_enc"]
            >>>         for example in outputs
            >>>     ], batch_first=True, padding_value=0.0)
            >>>     for task_name in list(["change", "saliency", "class"])
            >>> }
            >>> padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=-1000.0)
            >>> padded_valids = (padded_inputs[..., 0] > -1000.0).bool()
            >>> padded_inputs[~padded_valids] = 0.0
            >>> logits = self.forward(padded_inputs, stacked_queries, input_mask=padded_valids)
            >>> print('batch')
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> print('logits')
            >>> print(nh.data.collate._debug_inbatch_shapes(logits))

        Example:
            >>> # Decoupled resolutions
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> channels, classes, dataset_stats = SequenceAwareModel.demo_dataset_stats()
            >>> self = SequenceAwareModel(
            >>>     tokenizer='linconv',
            >>>     decoder='segmenter', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch(width=(11, 21), height=(16, 64), num_timesteps=3)
            >>> inputs = list(map(self.process_inputs, batch))
            >>> outputs = [self.process_outputs(example, dtype=inputs[0].dtype, device=inputs[0].device) for example in batch]
            >>> stacked_queries = {
            >>>     task_name: nn.utils.rnn.pad_sequence([
            >>>         example[task_name]["pos_enc"]
            >>>         for example in outputs
            >>>     ], batch_first=True, padding_value=0.0)
            >>>     for task_name in list(["change", "saliency", "class"])
            >>> }
            >>> padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=-1000.0)
            >>> padded_valids = (padded_inputs[..., 0] > -1000.0).bool()
            >>> padded_inputs[~padded_valids] = 0.0
            >>> logits = self.forward(padded_inputs, stacked_queries, input_mask=padded_valids)
            >>> print('batch')
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> print('logits')
            >>> print(nh.data.collate._debug_inbatch_shapes(logits))
        """
        context = self.backbone(inputs, mask=input_mask)
        # print("context", context)

        outputs = {}
        for task_name in queries.keys():
            task_tokens = self.backbone.decoder_cross_attn(
                queries[task_name],
                context=context)
            task_logits = self.heads[task_name](task_tokens)
            task_logits = einops.rearrange(task_logits, "batch seq chan -> batch chan seq")
            outputs[task_name] = task_logits

        return outputs

    def shared_step(self, batch, batch_idx=None, stage="train"):
        """
        Example:
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> channels, classes, dataset_stats = SequenceAwareModel.demo_dataset_stats()
            >>> self = SequenceAwareModel(
            >>>     tokenizer='linconv',
            >>>     classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch(width=64, height=65, batch_size=7)
            >>> batch_output = self.shared_step(batch)
            >>> assert 'change_probs' in batch_output
            >>> assert 'saliency_probs' in batch_output
            >>> assert 'class_probs' in batch_output
            >>> assert 'loss' in batch_output
        """

        # FIXME: This will break at test-time when labels are not provided
        inputs = list(map(self.process_inputs, batch))
        outputs = [self.process_outputs(example, dtype=inputs[0].dtype, device=inputs[0].device) for example in batch]

        padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=-1000.0)
        padded_valids = (padded_inputs[..., 0] > -1000.0).bool()
        padded_inputs[~padded_valids] = 0.0

        model_tasks = list(self.heads.keys())

        stacked_queries = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["pos_enc"]
                for example in outputs
            ], batch_first=True, padding_value=0.0)
            for task_name in model_tasks
        }
        stacked_weights = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["weights"]
                for example in outputs
            ], batch_first=True, padding_value=0.0)
            for task_name in model_tasks
        }
        stacked_labels = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["labels"]
                for example in outputs
            ], batch_first=True, padding_value=0).long()
            for task_name in model_tasks
        }
        stacked_masks = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["mask"]
                for example in outputs
            ], batch_first=True, padding_value=0)
            for task_name in model_tasks
        }

        task_logits = self.forward(padded_inputs, queries=stacked_queries, input_mask=padded_valids)
        task_losses = {}
        task_probs = {}
        for task_name in task_logits.keys():

            logits = einops.rearrange(task_logits[task_name], "batch chan seq -> (batch seq) chan")
            labels = einops.rearrange(stacked_labels[task_name], "batch seq -> (batch seq)")
            weights = einops.rearrange(stacked_weights[task_name], "batch seq -> (batch seq)")

            # task_mask = (labels != -1)
            task_mask = (weights > 0.0)

            criterion = self.criterions[task_name]
            if criterion.target_encoding == 'index':
                loss_labels = labels.long()
            elif criterion.target_encoding == 'onehot':
                # Note: 1HE is much easier to work with
                loss_labels = kwarray.one_hot_embedding(labels.long(), criterion.in_channels, dim=-1)
                weights = weights[..., None]
            else:
                raise KeyError(criterion.target_encoding)

            task_loss = weights[task_mask] * criterion(
                logits[task_mask],
                loss_labels[task_mask],
            )
            task_losses[task_name] = task_loss.mean()

            task_metric = self.head_metrics[f"{stage}_stage"][task_name](
                logits[task_mask],
                labels[task_mask],
            )
            self.log_dict(task_metric, prog_bar=True, sync_dist=True)

            # Need to output probabilities here for consumers of the model
            # TODO: only enable if requested, at train time we can discard this
            # for the majority of the iterations (unless we need to visualize
            # the batch)
            NEED_OUTPUTS = True
            if NEED_OUTPUTS:
                item_probs = []
                for item_index in range(len(batch)):
                    item_logit = einops.rearrange(task_logits[task_name][item_index].detach(), "chan seq -> seq chan")
                    item_mask = stacked_masks[task_name][item_index]
                    item_shapes = [list(frame["output_dims"]) for frame in batch[item_index]["frames"]]
                    # # if task_name == 'change':
                    # #     item_shapes = item_shapes[1:]  # hack for change
                    recon = self.reconstruct_output(item_logit, item_mask, item_shapes)
                    probs = [p.sigmoid() for p in recon]
                    item_probs.append(probs)
                task_probs[task_name] = item_probs

        loss = sum(task_losses.values()) / len(task_losses)

        self.log("loss", loss, prog_bar=True, sync_dist=True)

        # These need to be returned so the caller is able to introspect them
        # calling "log" is great, but it denies the caller access to this
        # information.
        batch_outputs = {}
        batch_outputs['loss'] = loss
        batch_outputs['task_losses'] = task_losses
        batch_outputs.update({k + '_probs': v for k, v in task_probs.items()})
        return batch_outputs

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

    def configure_optimizers(self):
        """
        TODO:
            - [ ] Enable use of other optimization algorithms on the CLI
            - [ ] Enable use of other scheduler algorithms on the CLI

        References:
            https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html

        Example:
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # noqa
            >>> self = SequenceAwareModel(input_sensorchan='r|g|b')
            >>> max_epochs = 80
            >>> self.trainer = pl.Trainer(max_epochs=max_epochs)
            >>> [opt], [sched] = self.configure_optimizers()
            >>> rows = []
            >>> # Insepct what the LR curve will look like
            >>> for _ in range(max_epochs):
            ...     sched.last_epoch += 1
            ...     lr = sched.get_lr()[0]
            ...     rows.append({'lr': lr, 'last_epoch': sched.last_epoch})
            >>> # xdoctest +REQUIRES(--show)
            >>> import kwplot
            >>> import pandas as pd
            >>> data = pd.DataFrame(rows)
            >>> sns = kwplot.autosns()
            >>> sns.lineplot(data=data, y='lr', x='last_epoch')

        Example:
            >>> # Verify lr and decay is set correctly
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> my_lr = 2.3e-5
            >>> my_decay = 2.3e-5
            >>> kw = dict(input_sensorchan='r|g|b', learning_rate=my_lr, weight_decay=my_decay)
            >>> self = SequenceAwareModel(**kw)
            >>> [opt], [sched] = self.configure_optimizers()
            >>> assert opt.param_groups[0]['lr'] == my_lr
            >>> assert opt.param_groups[0]['weight_decay'] == my_decay
            >>> #
            >>> self = SequenceAwareModel(**kw, optimizer='sgd')
            >>> [opt], [sched] = self.configure_optimizers()
            >>> assert opt.param_groups[0]['lr'] == my_lr
            >>> assert opt.param_groups[0]['weight_decay'] == my_decay
            >>> #
            >>> self = SequenceAwareModel(**kw, optimizer='AdamW')
            >>> [opt], [sched] = self.configure_optimizers()
            >>> assert opt.param_groups[0]['lr'] == my_lr
            >>> assert opt.param_groups[0]['weight_decay'] == my_decay
            >>> #
            >>> # self = SequenceAwareModel(**kw, optimizer='MADGRAD')
            >>> # [opt], [sched] = self.configure_optimizers()
            >>> # assert opt.param_groups[0]['lr'] == my_lr
            >>> # assert opt.param_groups[0]['weight_decay'] == my_decay
        """
        import netharn as nh

        # Netharn api will convert a string code into a type/class and
        # keyword-arguments to create an instance.
        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optimizer=self.hparams.optimizer,
            # learning_rate=self.hparams.learning_rate,
            lr=self.hparams.learning_rate,  # netharn bug?, some optimizers dont accept learning_rate and only lr
            weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == 'RAdam':
            optim_kw['betas'] = (0.9, 0.99)  # backwards compat

        # Hack to fix a netharn bug where weight decay is not set for AdamW
        optim_kw.update(ub.compatible(
            {'weight_decay': self.hparams.weight_decay}, optim_cls))

        optim_kw['params'] = self.parameters()
        print('optim_cls = {}'.format(ub.repr2(optim_cls, nl=1)))
        print('optim_kw = {}'.format(ub.repr2(optim_kw, nl=1)))
        optimizer = optim_cls(**optim_kw)

        # TODO:
        # - coerce schedulers
        if self.has_trainer:
            max_epochs = self.trainer.max_epochs
        else:
            max_epochs = 20

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs)
        return [optimizer], [scheduler]

    def overfit(self, batch):
        """
        Overfit script and demo

        CommandLine:
            python -m xdoctest -m watch.tasks.fusion.methods.sequence_aware SequenceAwareModel.overfit --overfit-demo

        Example:
            >>> # xdoctest: +REQUIRES(--overfit-demo)
            >>> # ============
            >>> # DEMO OVERFIT:
            >>> # ============
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> import kwcoco
            >>> from os.path import join
            >>> import os
            >>> if 1:
            >>>     '''
            >>>     # Generate toy datasets
            >>>     DATA_DPATH=$HOME/data/work/toy_change
            >>>     TRAIN_FPATH=$DATA_DPATH/vidshapes_msi_train/data.kwcoco.json
            >>>     mkdir -p "$DATA_DPATH"
            >>>     kwcoco toydata --key=vidshapes-videos8-frames5-randgsize-speed0.2-msi-multisensor --bundle_dpath "$DATA_DPATH/vidshapes_msi_train" --verbose=5
            >>>     '''
            >>>     coco_fpath = ub.expandpath('$HOME/data/work/toy_change/vidshapes_msi_train/data.kwcoco.json')
            >>>     coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
            >>>     channels="B11,r|g|b,B1|B8|B11"
            >>> if 0:
            >>>     dvc_dpath = find_smart_dvc_dpath()
            >>>     coco_dset = join(dvc_dpath, 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json')
            >>>     channels='swir16|swir22|blue|green|red|nir'
            >>> if 0:
            >>>     import watch
            >>>     coco_dset = watch.demo.demo_kwcoco_multisensor(max_speed=0.5)
            >>>     # coco_dset = 'special:vidshapes8-frames9-speed0.5-multispectral'
            >>>     #channels='B1|B11|B8|r|g|b|gauss'
            >>>     channels='X.2|Y:2:6,B1|B8|B8a|B10|B11,r|g|b,disparity|gauss,flowx|flowy|distri'
            >>> coco_dset = kwcoco.CocoDataset.coerce(coco_dset)
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=coco_dset,
            >>>     chip_size=128, batch_size=1, time_steps=3,
            >>>     channels=channels,
            >>>     normalize_inputs=1, neg_to_pos_ratio=0,
            >>>     num_workers='avail/2',
            >>>     use_grid_positives=False, use_centered_positives=True,
            >>> )
            >>> datamodule.setup('fit')
            >>> dataset = torch_dset = datamodule.torch_datasets['train']
            >>> torch_dset.disable_augmenter = True
            >>> dataset_stats = datamodule.dataset_stats
            >>> input_sensorchan = datamodule.input_sensorchan
            >>> classes = datamodule.classes
            >>> print('dataset_stats = {}'.format(ub.repr2(dataset_stats, nl=3)))
            >>> print('input_sensorchan = {}'.format(input_sensorchan))
            >>> print('classes = {}'.format(classes))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.SequenceAwareModel(
            >>>     # ===========
            >>>     # Backbone
            >>>     optimizer='AdamW',
            >>>     learning_rate=1e-5,
            >>>     #attention_impl='performer',
            >>>     attention_impl='exact',
            >>>     #decoder='segmenter',
            >>>     #saliency_head_hidden=4,
            >>>     decoder='mlp',
            >>>     change_loss='dicefocal',
            >>>     #class_loss='cce',
            >>>     class_loss='dicefocal',
            >>>     saliency_loss='dicefocal',
            >>>     # ===========
            >>>     # Change Loss
            >>>     global_change_weight=0.00,
            >>>     positive_change_weight=1.0,
            >>>     negative_change_weight=0.5,
            >>>     # ===========
            >>>     # Class Loss
            >>>     global_class_weight=1.00,
            >>>     class_weights='auto',
            >>>     # ===========
            >>>     # Saliency Loss
            >>>     global_saliency_weight=1.00,
            >>>     # ===========
            >>>     # Domain Metadata (Look Ma, not hard coded!)
            >>>     dataset_stats=dataset_stats,
            >>>     classes=classes,
            >>>     input_sensorchan=input_sensorchan,
            >>>     #tokenizer='dwcnn',
            >>>     tokenizer='linconv',
            >>>     #tokenizer='rearrange',
            >>>     # normalize_perframe=True,
            >>>     window_size=8,
            >>>     )
            >>> self.datamodule = datamodule
            >>> datamodule._notify_about_tasks(model=self)
            >>> # Run one visualization
            >>> loader = datamodule.train_dataloader()
            >>> # Load one batch and show it before we do anything
            >>> batch = next(iter(loader))
            >>> import kwplot
            >>> plt = kwplot.autoplt(force='Qt5Agg')
            >>> plt.ion()
            >>> canvas = datamodule.draw_batch(batch, max_channels=5, overlay_on_image=0)
            >>> kwplot.imshow(canvas, fnum=1)
            >>> # Run overfit
            >>> device = 0
            >>> self.overfit(batch)

        nh.initializers.KaimingNormal()(self)
        nh.initializers.Orthogonal()(self)
        """
        import kwplot
        # import torch_optimizer
        import xdev
        import kwimage
        import pandas as pd
        from watch.utils.slugify_ext import smart_truncate
        from kwplot.mpl_make import render_figure_to_image

        sns = kwplot.autosns()
        datamodule = self.datamodule
        device = 0
        self = self.to(device)
        # loader = datamodule.train_dataloader()
        # batch = next(iter(loader))
        walker = ub.IndexableWalker(batch)
        for path, val in walker:
            if isinstance(val, torch.Tensor):
                walker[path] = val.to(device)

        outputs = self.training_step(batch)
        max_channels = 3
        canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=max_channels, overlay_on_image=0)
        kwplot.imshow(canvas)

        loss_records = []
        loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
        step = 0
        _frame_idx = 0
        # dpath = ub.ensuredir('_overfit_viz09')

        # optim_cls, optim_kw = nh.api.Optimizer.coerce(
        #     optim='RAdam', lr=1e-3, weight_decay=0,
        #     params=self.parameters())
        #optim = torch.optim.SGD(self.parameters(), lr=1e-4)
        #optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        [optim], [sched] = self.configure_optimizers()
        # optim = torch_optimizer.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        fnum = 2
        fig = kwplot.figure(fnum=fnum, doclf=True)
        fig.set_size_inches(15, 6)
        fig.subplots_adjust(left=0.05, top=0.9)
        prev = None
        for _frame_idx in xdev.InteractiveIter(list(range(_frame_idx + 1, 1000))):
            # for _frame_idx in list(range(_frame_idx, 1000)):
            num_steps = 20
            ex = None
            for _i in ub.ProgIter(range(num_steps), desc='overfit'):
                optim.zero_grad()
                outputs = self.training_step(batch)
                outputs['task_losses']
                loss = outputs['loss']
                if torch.any(torch.isnan(loss)):
                    print('NAN OUTPUT!!!')
                    print('loss = {!r}'.format(loss))
                    print('prev = {!r}'.format(prev))
                    ex = Exception('prev = {!r}'.format(prev))
                    break
                # elif loss > 1e4:
                #     # Turn down the learning rate when loss gets huge
                #     scale = (loss / 1e4).detach()
                #     loss /= scale
                prev = loss
                # task_losses_ = nh.data.collate.default_collate(outputs['task_losses'])
                task_losses = ub.udict(outputs['task_losses']).map_values(lambda x: x.item())
                # task_losses = ub.map_vals(lambda x: sum(x).item(), task_losses_)
                loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in task_losses.items()])
                loss.backward()
                optim.step()
                step += 1
            canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=max_channels, overlay_on_image=0, max_items=4)
            kwplot.imshow(canvas, pnum=(1, 2, 1), fnum=fnum)
            fig = kwplot.figure(fnum=fnum, pnum=(1, 2, 2))
            #kwplot.imshow(canvas, pnum=(1, 2, 1))
            ax = sns.lineplot(data=pd.DataFrame(loss_records), x='step', y='val', hue='part')
            try:
                ax.set_yscale('logit')
            except Exception:
                ...
            fig.suptitle(smart_truncate(str(optim).replace('\n', ''), max_length=64))
            img = render_figure_to_image(fig)
            img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            # fpath = join(dpath, 'frame_{:04d}.png'.format(_frame_idx))
            #kwimage.imwrite(fpath, img)
            xdev.InteractiveIter.draw()
            if ex:
                raise ex
        # TODO: can we get this batch to update in real time?
        # TODO: start a server process that listens for new images
        # as it gets new images, it starts playing through the animation
        # looping as needed

    def save_package(self, package_path, verbose=1):
        """

        CommandLine:
            xdoctest -m watch.tasks.fusion.methods.sequence_aware SequenceAwareModel.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> dpath = ub.Path.appdir('watch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> model = self = methods.SequenceAwareModel(
            >>>     input_sensorchan=5,
            >>>     change_head_hidden=0, saliency_head_hidden=0,
            >>>     class_head_hidden=0)

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.SequenceAwareModel.load_package(package_path)
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
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> dpath = ub.Path.appdir('watch/tests/package').ensuredir()
            >>> package_path = dpath / 'my_package.pt'

            >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral-multisensor', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10)
            >>> datamodule.setup('fit')
            >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.SequenceAwareModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats, input_sensorchan=datamodule.input_sensorchan,
            >>>     learning_rate=1e-8, optimizer='sgd',
            >>>     change_head_hidden=0, saliency_head_hidden=0,
            >>>     class_head_hidden=0)

            >>> # We have to run an input through the module because it is lazy
            >>> batch = ub.peek(iter(datamodule.train_dataloader()))
            >>> outputs = self.training_step(batch)

            >>> trainer = pl.Trainer(max_steps=1)
            >>> trainer.fit(model=self, datamodule=datamodule)

            >>> # Save the self
            >>> self.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.SequenceAwareModel.load_package(package_path)

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
