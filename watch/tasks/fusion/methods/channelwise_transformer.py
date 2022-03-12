"""

Our data might look like this, a sequence of frames where the frames can contain
heterogeneous data:

    [
        {
            'frame_index': 0,
            'time_offset': 0,
            'sensor': 'S2',
            'modes': {
                'blue|green|red|swir1|swir2|nir': <Tensor shape=(6, 64, 64),
                'pan': <Tensor shape=(1, 112, 112),
            }
            'truth': {
                'class_idx': <Tensor shape=(5, 128, 128),
            }
        },
        {
            'frame_index': 1,
            'time_offset': 100,
            'sensor': 'L8',
            'modes': {
                'blue|green|red|lwir1|lwir2|nir': <Tensor shape=(6, 75, 75),
            }
            'truth': {
                'class_idx': <Tensor shape=(5, 128, 128),
            }
        },
        {
            'frame_index': 2,
            'time_offset': 120,
            'sensor': 'S2',
            'modes': {
                'blue|green|red|swir1|swir2|nir': <Tensor shape=(6, 64, 64),
                'pan': <Tensor shape=(1, 112, 112),
            }
            'truth': {
                'class_idx': <Tensor shape=(5, 128, 128),
            }
        },
        {
            'frame_index': 3,
            'time_offset': 130,
            'sensor': 'WV',
            'modes': {
                'blue|green|red|nir': <Tensor shape=(4, 224, 224),
                'pan': <Tensor shape=(1, 512, 512),
            },
            'truth': {
                'class_idx': <Tensor shape=(5, 128, 128),
            }
        },
    ]

"""
import einops
import kwarray
import kwcoco
import ubelt as ub
import torch
import torchmetrics
# import math

import numpy as np
import netharn as nh
import pytorch_lightning as pl

# import torch_optimizer as optim
from torch import nn
# from einops.layers.torch import Rearrange
from kwcoco import channel_spec
# from torchvision import transforms
from torch.optim import lr_scheduler
from watch import heuristics
from watch.tasks.fusion import utils
from watch.tasks.fusion.architectures import transformer

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


from timm.models.layers import drop_path
from torch._jit_internal import _copy_to_script_wrapper
from torch.nn.modules.container import Module


class RobustModuleDict(torch.nn.ModuleDict):
    """
    Regular torch.nn.ModuleDict doesnt allow empty str. Hack around this.

    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> import string
        >>> torch_dict = RobustModuleDict()
        >>> # All printable characters should be usable as keys
        >>> # If they are not, hack it.
        >>> failed = []
        >>> for c in list(string.printable) + ['']:
        >>>     try:
        >>>         torch_dict[c] = torch.nn.Linear(1, 1)
        >>>     except KeyError:
        >>>         failed.append(c)
        >>> assert len(failed) == 0
    """
    repl_dot = '#D#'
    repl_empty = '__EMPTY'

    def _normalize_key(self, key):
        key = self.repl_empty if key == '' else key.replace('.', self.repl_dot)
        return key

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        key = self._normalize_key(key)
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        key = self._normalize_key(key)
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        key = self._normalize_key(key)
        del self._modules[key]

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        key = self._normalize_key(key)
        return key in self._modules

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (string): key to pop from the ModuleDict
        """
        key = self._normalize_key(key)
        v = self[key]
        del self[key]
        return v


class MultimodalTransformer(pl.LightningModule):
    """
    CommandLine:
        xdoctest -m watch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer

    TODO:
        - [ ] Change name MultimodalTransformer -> FusionModel
        - [ ] Move parent module methods -> models

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/tasks/fusion/methods/channelwise_transformer.py MultimodalTransformer

    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from watch.tasks.fusion import datamodules
        >>> datamodule = datamodules.KWCocoVideoDataModule(
        >>>     train_dataset='special:vidshapes-watch', num_workers=4)
        >>> datamodule.setup('fit')
        >>> dataset = datamodule.torch_datasets['train']
        >>> dataset_stats = dataset.cached_dataset_stats(num=3)
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
        >>> self = MultimodalTransformer(arch_name='smt_it_joint_p8',
        >>>                              input_channels=datamodule.input_channels,
        >>>                              dataset_stats=dataset_stats,
        >>>                              classes=datamodule.classes, decoder='segmenter',
        >>>                              change_loss='dicefocal',
        >>>                              attention_impl='performer')
        >>> device = nh.XPU.coerce('cpu').main_device
        >>> self = self.to(device)
        >>> # Run forward pass
        >>> num_params = nh.util.number_of_parameters(self)
        >>> print('num_params = {!r}'.format(num_params))
        >>> import torch.profiler
        >>> from torch.profiler import profile, ProfilerActivity
        >>> with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        >>>     with torch.profiler.record_function("model_inference"):
        >>>         output = self.forward_step(batch, with_loss=True)
        >>> print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    """

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.utils.configargparse_ext import ArgumentParser
            >>> cls = MultimodalTransformer
            >>> parent_parser = ArgumentParser(formatter_class='defaults')
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> parent_parser.parse_known_args()
        """
        from scriptconfig.smartcast import smartcast
        parser = parent_parser.add_argument_group('MultimodalTransformer')
        parser.add_argument('--name', default='unnamed_model', help=ub.paragraph(
            '''
            Specify a name for the experiment. (Unsure if the Model is the place for this)
            '''))

        parser.add_argument('--optimizer', default='RAdam', type=str, help='Optimizer name supported by the netharn API')
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--weight_decay', default=0., type=float)

        parser.add_argument('--positive_change_weight', default=1.0, type=float)
        parser.add_argument('--negative_change_weight', default=1.0, type=float)
        parser.add_argument('--class_weights', default='auto', type=str, help='class weighting strategy')
        parser.add_argument('--saliency_weights', default='auto', type=str, help='class weighting strategy')

        # Model names define the transformer encoder used by the method
        available_encoders = list(transformer.encoder_configs.keys()) + ['deit']

        parser.add_argument('--stream_channels', default=8, type=int, help='number of channels to normalize each project stream to')
        parser.add_argument(
            '--tokenizer', default='rearrange', type=str,
            choices=['dwcnn', 'rearrange', 'conv7', 'linconv'], help=ub.paragraph(
                '''
                How image patches are broken into tokens.
                rearrange is a 1x1 MLP and grouping of pixel grids.
                dwcnn is a is a mobile convolutional stem.
                conv7 is a simple 1x1x7x7 convolutional stem.
                linconv is a stack of 3x3 grouped convolutions without any nonlinearity
                '''))
        parser.add_argument('--token_norm', default='none', type=str,
                            choices=['none', 'auto', 'group', 'batch'])
        parser.add_argument('--arch_name', default='smt_it_joint_p8', type=str,
                            choices=available_encoders)
        parser.add_argument('--decoder', default='mlp', type=str, choices=['mlp', 'segmenter'])
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--global_class_weight', default=1.0, type=float)
        parser.add_argument('--global_change_weight', default=1.0, type=float)
        parser.add_argument('--global_saliency_weight', default=1.0, type=float)
        parser.add_argument('--modulate_class_weights', default='', type=str, help='a special syntax that lets the user modulate automatically computed class weights. Should be a comma separated list of name*weight or name*weight+offset. E.g. `negative*0,background*0.001,No Activity*0.1+1`')

        parser.add_argument('--change_loss', default='cce')
        parser.add_argument('--class_loss', default='focal')
        parser.add_argument('--saliency_loss', default='focal', help='saliency is trained to match any "positive/foreground/salient" class')

        parser.add_argument('--change_head_hidden', default=2, type=int, help='number of hidden layers in the change head')
        parser.add_argument('--class_head_hidden', default=2, type=int, help='number of hidden layers in the category head')
        parser.add_argument('--saliency_head_hidden', default=2, type=int, help='number of hidden layers in the saliency head')

        # parser.add_argument("--input_scale", default=2000.0, type=float)
        parser.add_argument('--window_size', default=8, type=int)
        parser.add_argument('--squash_modes', default=False, type=smartcast)
        parser.add_argument(
            '--attention_impl', default='exact', type=str, help=ub.paragraph(
                '''
                Implementation for attention computation.
                Can be:
                'exact' - the original O(n^2) method.
                'performer' - a linear approximation.
                'reformer' - a LSH approximation.
                '''))
        return parent_parser

    def get_cfgstr(self):
        cfgstr = f'{self.name}_{self.arch_name}'
        return cfgstr

        # model_cfgstr
        pass

    def __init__(self,
                 arch_name='smt_it_joint_p8',
                 dropout=0.0,
                 optimizer='RAdam',
                 learning_rate=1e-3,
                 weight_decay=0.,
                 class_weights='auto',
                 saliency_weights='auto',
                 positive_change_weight=1.,
                 negative_change_weight=1.,
                 dataset_stats=None,
                 input_channels=None,
                 unique_sensors=None,
                 attention_impl='exact',
                 window_size=8,
                 global_class_weight=1.0,
                 global_change_weight=1.0,
                 global_saliency_weight=1.0,
                 change_head_hidden=1,
                 class_head_hidden=1,
                 saliency_head_hidden=1,
                 change_loss='cce',
                 class_loss='focal',
                 saliency_loss='focal',
                 tokenizer='rearrange',
                 token_norm='auto',
                 name='unnamed_expt',
                 squash_modes=False,
                 multimodal_reduce='max',
                 modulate_class_weights='',
                 stream_channels=8,
                 decoder='mlp',
                 classes=10):

        super().__init__()
        self.save_hyperparameters()
        self.name = name
        self.decoder = decoder

        self.arch_name = arch_name
        self.squash_modes = squash_modes
        self.multimodal_reduce = multimodal_reduce
        self.modulate_class_weights = modulate_class_weights
        self.stream_channels = stream_channels

        if dataset_stats is not None:
            input_stats = dataset_stats['input_stats']
            class_freq = dataset_stats['class_freq']
        else:
            class_freq = None

        self.class_freq = class_freq
        self.dataset_stats = dataset_stats

        # Handle channel-wise input mean/std in the network (This is in
        # contrast to common practice where it is done in the dataloader)
        input_norms = None
        known_sensors = None
        known_channels = None

        # Not sure how relevant (input_channels) is anymore
        if input_channels is None:
            raise Exception('need them for num input_channels!')
        input_channels = channel_spec.ChannelSpec.coerce(input_channels)
        self.input_channels = input_channels

        if self.dataset_stats is None:
            # hack for tests (or no known sensors case)
            input_stats = None
            self.unique_sensor_modes = {('', self.input_channels.spec)}
        else:
            self.unique_sensor_modes = self.dataset_stats['unique_sensor_modes']

        if input_stats is not None:
            input_norms = RobustModuleDict()
            for s, c in self.unique_sensor_modes:
                if s not in input_norms:
                    input_norms[s] = RobustModuleDict()
                stats = input_stats.get((s, c), None)
                if stats is None:
                    input_norms[s][c] = nh.layers.InputNorm()
                else:
                    input_norms[s][c] = nh.layers.InputNorm(**stats)

            for (s, c), stats in input_stats.items():
                input_norms[s][c] = nh.layers.InputNorm(**stats)

        self.known_sensors = known_sensors
        self.known_channels = known_channels
        self.input_norms = input_norms

        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        input_streams = list(input_channels.streams())
        stream_num_channels = {s.spec: s.numel() for s in input_streams}
        self.stream_num_channels = stream_num_channels

        self.global_class_weight = global_class_weight
        self.global_change_weight = global_change_weight
        self.global_saliency_weight = global_saliency_weight
        self.global_head_weights = {
            'class': global_class_weight,
            'change': global_change_weight,
            'saliency': global_saliency_weight,
        }

        self.positive_change_weight = positive_change_weight
        self.negative_change_weight = negative_change_weight

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
        # ~/code/watch/watch/tasks/fusion/methods/channelwise_transformer.py
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
                if self.modulate_class_weights:
                    import re
                    parts = [p.strip() for p in self.modulate_class_weights.split(',')]
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
            self.negative_change_weight,
            self.positive_change_weight
        ])

        MODAL_AGREEMENT_CHANS = self.stream_channels
        self.tokenizer = tokenizer
        self.sensor_channel_tokenizers = RobustModuleDict()
        for s, c in self.unique_sensor_modes:
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

            self.sensor_channel_tokenizers[s][c] = tokenize
            in_features_raw = tokenize.out_channels

        in_features_pos = 6 * 8   # 6 positional features with 8 dims each (TODO: be robust)
        in_features = in_features_pos + in_features_raw
        self.in_features = in_features
        self.in_features_pos = in_features_pos
        self.in_features_raw = in_features_raw

        ### NEW:
        # Learned positional encodings
        self.token_learner1_time_delta = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=1, hidden_channels=3, out_channels=8, residual=True, norm=None)
        self.token_learner2_sensor = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=16, hidden_channels=3, out_channels=8, residual=True, norm=None)
        self.token_learner3_mode = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=16, hidden_channels=3, out_channels=8, residual=True, norm=None)

        # 'https://rwightman.github.io/pytorch-image-models/models/vision-transformer/'
        if arch_name in transformer.encoder_configs:
            encoder_config = transformer.encoder_configs[arch_name]
            encoder = transformer.FusionEncoder(
                **encoder_config,
                in_features=in_features,
                attention_impl=attention_impl,
                dropout=dropout,
            )
            self.encoder = encoder
        elif arch_name.startswith('deit'):
            self.encoder = transformer.DeiTEncoder(
                # **encoder_config,
                in_features=in_features,
                # attention_impl=attention_impl,
                # dropout=dropout,
            )
        else:
            raise NotImplementedError

        feat_dim = self.encoder.out_features

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
                'loss': change_loss,
                'weights': self.change_weights,
            },
            {
                'name': 'saliency',
                'hidden': self.saliency_head_hidden,
                'channels': self.saliency_num_classes,
                'loss': saliency_loss,
                'weights': self.saliency_weights,
            },
            {
                'name': 'class',
                'hidden': self.class_head_hidden,
                'channels': self.num_classes,
                'loss': class_loss,
                'weights': self.class_weights,
            },
        ]

        for prop in head_properties:
            head_name = prop['name']
            global_weight = self.global_head_weights[head_name]
            if global_weight > 0:
                self.criterions[head_name] = coerce_criterion(prop['loss'], prop['weights'])
                if self.decoder == 'mlp':
                    self.heads[head_name] = nh.layers.MultiLayerPerceptronNd(
                        dim=0,
                        in_channels=feat_dim,
                        hidden_channels=prop['hidden'],
                        out_channels=prop['channels'],
                        norm=None
                    )
                elif self.decoder == 'segmenter':
                    from watch.tasks.fusion.architectures import segmenter_decoder
                    self.heads[head_name] = segmenter_decoder.MaskTransformerDecoder(
                        d_model=feat_dim,
                        # hidden_channels=prop['hidden'],
                        n_cls=prop['channels'],
                    )
                else:
                    raise KeyError(self.decoder)

        self.head_metrics = nn.ModuleDict()
        self.head_metrics['class'] = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1_micro': torchmetrics.F1(threshold=0.5, average='micro'),
            'f1_macro': torchmetrics.F1(threshold=0.5, average='macro', num_classes=self.num_classes),
        })
        self.head_metrics['change'] = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1': torchmetrics.F1(),
        })
        self.head_metrics['saliency'] = nn.ModuleDict({
            'f1': torchmetrics.F1(),
        })

    def configure_optimizers(self):
        """
        TODO:
            - [ ] Enable use of other optimization algorithms on the CLI
            - [ ] Enable use of other scheduler algorithms on the CLI

        References:
            https://pytorch-optimizer.readthedocs.io/en/latest/index.html

        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.tasks.fusion import methods
            >>> self = methods.MultimodalTransformer("smt_it_stm_p8", input_channels='r|g|b')
            >>> self.trainer = pl.Trainer(max_epochs=400)
            >>> [opt], [sched] = self.configure_optimizers()
            >>> rows = []
            >>> # Insepct what the LR curve will look like
            >>> for _ in range(self.trainer.max_epochs):
            ...     sched.last_epoch += 1
            ...     lr = sched.get_lr()[0]
            ...     rows.append({'lr': lr, 'last_epoch': sched.last_epoch})
            >>> import pandas as pd
            >>> data = pd.DataFrame(rows)
            >>> # xdoctest +REQUIRES(--show)
            >>> import kwplot
            >>> sns = kwplot.autosns()
            >>> sns.lineplot(data=data, y='lr', x='last_epoch')
        """
        import netharn as nh

        # Netharn api will convert a string code into a type/class and
        # keyword-arguments to create an instance.
        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optimizer=self.hparams.optimizer,
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == 'RAdam':
            optim_kw['betas'] = (0.9, 0.99)  # backwards compat

        optim_kw['params'] = self.parameters()
        optimizer = optim_cls(**optim_kw)

        # TODO:
        # - coerce schedulers
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def overfit(self, batch):
        """
        Overfit script and demo

        CommandLine:
            python -m xdoctest -m watch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer.overfit --overfit-demo

        Example:
            >>> # xdoctest: +REQUIRES(--overfit-demo)
            >>> # ============
            >>> # DEMO OVERFIT:
            >>> # ============
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
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
            >>>     num_workers='avail/2', true_multimodal=True,
            >>>     use_grid_positives=False, use_centered_positives=True,
            >>> )
            >>> datamodule.setup('fit')
            >>> dataset = torch_dset = datamodule.torch_datasets['train']
            >>> torch_dset.disable_augmenter = True
            >>> dataset_stats = datamodule.dataset_stats
            >>> input_channels = datamodule.input_channels
            >>> classes = datamodule.classes
            >>> print('dataset_stats = {}'.format(ub.repr2(dataset_stats, nl=3)))
            >>> print('input_channels = {}'.format(input_channels))
            >>> print('classes = {}'.format(classes))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.MultimodalTransformer(
            >>>     # ===========
            >>>     # Backbone
            >>>     #arch_name='smt_it_joint_p2',
            >>>     arch_name='smt_it_stm_p1',
            >>>     learning_rate=1e-6,
            >>>     #attention_impl='performer',
            >>>     attention_impl='exact',
            >>>     decoder='segmenter',
            >>>     #arch_name='deit',
            >>>     change_loss='focal',
            >>>     #class_loss='cce',
            >>>     class_loss='dicefocal',
            >>>     saliency_loss='focal',
            >>>     # ===========
            >>>     # Change Loss
            >>>     global_change_weight=1.00,
            >>>     positive_change_weight=1.0,
            >>>     negative_change_weight=0.05,
            >>>     # ===========
            >>>     # Class Loss
            >>>     global_class_weight=1.00,
            >>>     global_saliency_weight=1.00,
            >>>     class_weights='auto',
            >>>     # ===========
            >>>     # Domain Metadata (Look Ma, not hard coded!)
            >>>     dataset_stats=dataset_stats,
            >>>     classes=classes,
            >>>     input_channels=input_channels,
            >>>     #tokenizer='dwcnn',
            >>>     tokenizer='linconv',
            >>>     #tokenizer='rearrange',
            >>>     squash_modes=True,
            >>>     # normalize_perframe=True,
            >>>     window_size=8,
            >>>     )
            >>> self.datamodule = datamodule
            >>> self.di = datamodule
            >>> # Run one visualization
            >>> loader = datamodule.train_dataloader()
            >>> # Load one batch and show it before we do anything
            >>> batch = next(iter(loader))
            >>> import kwplot
            >>> kwplot.autompl(force='Qt5Agg')
            >>> canvas = datamodule.draw_batch(batch, max_channels=5, overlay_on_image=0)
            >>> kwplot.imshow(canvas, fnum=1)
            >>> # Run overfit
            >>> device = 0
            >>> self.overfit(batch)

        nh.initializers.KaimingNormal()(self)
        nh.initializers.Orthogonal()(self)
        """
        import kwplot
        import torch_optimizer
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

        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optim='RAdam', lr=1e-3, weight_decay=0,
            params=self.parameters())

        #optim = torch.optim.SGD(self.parameters(), lr=1e-4)
        #optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        optim = torch_optimizer.RAdam(self.parameters(), lr=3e-3, weight_decay=1e-5)

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
                outputs['item_losses']
                loss = outputs['loss']
                if torch.any(torch.isnan(loss)):
                    print('loss = {!r}'.format(loss))
                    print('prev = {!r}'.format(prev))
                    ex = Exception('prev = {!r}'.format(prev))
                    break
                prev = loss
                item_losses_ = nh.data.collate.default_collate(outputs['item_losses'])
                item_losses = ub.map_vals(lambda x: sum(x).item(), item_losses_)
                loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in item_losses.items()])
                loss.backward()
                optim.step()
                step += 1
            canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=max_channels, overlay_on_image=0, max_items=4)
            kwplot.imshow(canvas, pnum=(1, 2, 1), fnum=fnum)
            fig = kwplot.figure(fnum=fnum, pnum=(1, 2, 2))
            #kwplot.imshow(canvas, pnum=(1, 2, 1))
            ax = sns.lineplot(data=pd.DataFrame(loss_records), x='step', y='val', hue='part')
            ax.set_yscale('logit')
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

    @classmethod
    def demo_dataset_stats(cls):
        channels = kwcoco.ChannelSpec.coerce('pan,red|green|blue,nir|swir16|swir22')
        unique_sensor_modes = {
            ('sensor1', 'pan'),
            ('sensor1', 'red|green|blue'),
            ('sensor1', 'nir|swir16|swir22'),
        }
        input_stats = {k: {
            'mean': np.random.rand(len(k[1].split('|')), 1, 1),
            'std': np.random.rand(len(k[1].split('|')), 1, 1),
        } for k in unique_sensor_modes}

        classes = kwcoco.CategoryTree.coerce(3)
        dataset_stats = {
            'unique_sensor_modes': unique_sensor_modes,
            'input_stats': input_stats,
            'class_freq': {c: np.random.randint(0, 10000) for c in classes},
        }
        # 'sensor_mode_hist': {('sensor3', 'B10|B11|r|g|b|flowx|flowy|distri'): 1,
        #  ('sensor2', 'B8|B11|r|g|b|disparity|gauss'): 2,
        #  ('sensor0', 'B1|B8|B8a|B10|B11'): 1},
        # 'class_freq': {'star': 0,
        #  'superstar': 5822,
        #  'eff': 0,
        #  'negative': 0,
        #  'ignore': 9216,
        #  'background': 21826}}
        return channels, classes, dataset_stats

    def demo_batch(self, batch_size=1, num_timesteps=3, width=8, height=8):
        """
        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='segmenter', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_channels=channels)
            >>> batch = self.demo_batch()
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> self.forward_step(batch)
        """
        B = batch_size
        C = len(self.classes)
        T = num_timesteps
        batch = []
        for bx in range(B):
            modes = []
            frames = []
            for time_index in range(T):
                H, W = 96, 96
                modes = {
                    'pan': torch.rand(1, H, W),
                    'red|green|blue': torch.rand(3, H, W),
                    'nir|swir16|swir22': torch.rand(3, H, W),
                }
                frame = {}
                if time_index == 0:
                    frame['change'] = None
                    frame['change_weights'] = None
                else:
                    frame['change'] = torch.randint(low=0, high=1, size=(H, W))
                    frame['change_weights'] = torch.rand(H, W)

                frame['class_idxs'] = torch.randint(low=0, high=C - 1, size=(H, W))
                frame['class_weights'] = torch.rand(H, W)

                frame['saliency'] = torch.randint(low=0, high=1, size=(H, W))
                frame['saliency_weights'] = torch.rand(H, W)

                frame['date_captured'] = '',
                frame['gid'] = bx
                frame['sensor'] = 'sensor1'
                frame['time_index'] = bx
                frame['time_offset'] = np.array([1]),
                frame['timestamp'] = 1
                frame['modes'] = modes
                frames.append(frame)

            positional_tensors = {
                'mode_tensor': torch.rand(T, 16),
                'sensor': torch.rand(T, 16),
                'time_index': torch.rand(T, 8),
                'time_offset': torch.rand(T, 1),
            }
            tr = {
                'gids': list(range(T)),
                'space_slice': [
                    slice(0, H),
                    slice(0, W),
                ]
            }
            item = {
                'video_id': 3,
                'video_name': 'toy_video_3',
                'frames': frames,
                'positional_tensors': positional_tensors,
                'tr': tr,
            }
            batch.append(item)
        return batch

    @profile
    def forward_step(self, batch, with_loss=False, stage='unspecified'):
        """
        Generic forward step used for test / train / validation

        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> import watch
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes-watch',
            >>>     num_workers='avail / 2', chip_size=96, time_steps=4, true_multimodal=True,
            >>>     normalize_inputs=1, neg_to_pos_ratio=0, batch_size=1,
            >>> )
            >>> datamodule.setup('fit')
            >>> train_dset = datamodule.torch_datasets['train']
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = model = methods.MultimodalTransformer(
            >>>     arch_name='smt_it_joint_p8', tokenizer='rearrange',
            >>>     decoder='segmenter',
            >>>     dataset_stats=datamodule.dataset_stats, global_saliency_weight=1.0, global_change_weight=1.0, global_class_weight=1.0,
            >>>     classes=datamodule.classes, input_channels=datamodule.input_channels)
            >>> with_loss = True
            >>> outputs = self.forward_step(batch, with_loss=with_loss)
            >>> canvas = datamodule.draw_batch(batch, outputs=outputs, max_items=3, overlay_on_image=False)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        outputs = {}
        eps_f32 = 1e-9

        item_losses = []

        # Initialize truth and probs for each head / item that will be stacked
        batch_head_truths = {k: [] for k in self.heads.keys()}
        batch_head_probs = {k: [] for k in self.heads.keys()}

        encode_h = utils.SinePositionalEncoding(3, 1, size=8)
        encode_w = utils.SinePositionalEncoding(3, 2, size=8)

        for item in batch:

            if 1:
                # Clean this up
                positional_tensors = item['positional_tensors']
                positional_outputs = []
                key = 'time_offset'
                pos_input = positional_tensors[key].float()
                pos_output = self.token_learner1_time_delta(pos_input)
                positional_outputs.append(pos_output)

                key = 'sensor'
                pos_input = positional_tensors[key].float()
                pos_output = self.token_learner2_sensor(pos_input)
                positional_outputs.append(pos_output)

                key = 'mode_tensor'
                pos_input = positional_tensors[key].float()
                pos_output = self.token_learner3_mode(pos_input)
                positional_outputs.append(pos_output)

                positional_outputs.append(positional_tensors['time_index'])

                per_frame_pos_encoding = torch.concat(positional_outputs, axis=1)

            # For loops are for handing heterogeneous inputs
            tokenized = []
            token_frame_idx = []

            item_pixel_weights_list = {
                k: [] for k, v in self.global_head_weights.items() if v > 0
            }

            for frame_idx, (frame, frame_enc) in enumerate(zip(item['frames'], per_frame_pos_encoding)):
                if with_loss:
                    if 'class' in item_pixel_weights_list:
                        item_pixel_weights_list['class'].append(frame['class_weights'])
                    if 'saliency' in item_pixel_weights_list:
                        item_pixel_weights_list['saliency'].append(frame['saliency_weights'])
                    if 'change' in item_pixel_weights_list:
                        if frame_idx > 0:
                            item_pixel_weights_list['change'].append(frame['change_weights'])

                modes = frame['modes']
                sensor = frame['sensor']
                for chan_code, mode_val in modes.items():

                    mode_val = mode_val.float()
                    if self.input_norms is not None:
                        try:
                            mode_norm = self.input_norms[sensor][chan_code]
                            mode_val = mode_norm(mode_val)
                        except KeyError:
                            print(f'Failed to process {sensor=!r} {chan_code=!r}')
                            print('Expected available norms (note the keys contain escape sequences) are:')
                            for _s in sorted(self.input_norms.keys()):
                                for _c in sorted(self.input_norms[_s].keys()):
                                    print(f'{_s=!r} {_c=!r}')
                            print('self.unique_sensor_modes = {!r}'.format(self.unique_sensor_modes))
                            raise
                        # self.sensor_channel_tokenizers[]

                    # Lookup the "tokenizing" network for this type of input
                    sensor_chan_tokenizer = self.sensor_channel_tokenizers[sensor][chan_code]

                    # Is it worth gathering and stacking items in batches here?

                    # Reduce spatial dimension and normalize the number of
                    # features in each input stream.
                    _mode_val_tokens = sensor_chan_tokenizer(mode_val[None, :])

                    mode_vals_tokens = einops.rearrange(
                        _mode_val_tokens, 'b c h w -> b h w c')

                    # Add ordinal spatial encoding
                    x1 = encode_w(encode_h(mode_vals_tokens))

                    # Any tricks needed to handle inputs/outputs at different
                    # resolutions might go here
                    encoding_expanded = frame_enc[None, None, None, :].expand(list(x1.shape[0:3]) + [frame_enc.shape[0]])

                    # Mixup the space/time dims into the token dims to make this
                    # general.
                    x2 = torch.cat([x1, encoding_expanded.type_as(x1)], dim=3)

                    # frame_sensor_chan_tokens = einops.rearrange(x2, 't hs ws f -> (t hs ws) f')

                    # Keep time/sensor channel separate for now, but we will
                    # need to flatten it, at which point we need a decoder.
                    frame_sensor_chan_tokens = einops.rearrange(x2, 't hs ws f -> t (hs ws) f')
                    tokenized.append(frame_sensor_chan_tokens)

                    # TODO: keep track of heterogeneous offsets
                    token_frame_idx.append(frame_idx)

            tokens = torch.stack(tokenized, dim=0)[None, None, ...]

            H, W = mode_val.shape[-2:]  # hack
            hs, ws = mode_vals_tokens.shape[1:3]  # hack

            # TODO: maybe make the encoder return a sequence of 1 less?
            # Rather than just ignoring the first output?
            encoded_tokens = self.encoder(tokens)

            split_pts = np.r_[[0], np.where(np.r_[np.diff(token_frame_idx), [-1]])[0] + 1]
            split_sizes = np.diff(split_pts).tolist()
            # split_pts.cumsum()
            # split_pts = torch.from_numpy(split_pts).to(tokens.device)
            perframe_frame_encodings = encoded_tokens.split(split_sizes, dim=2)

            perframe_stackable_encodings = []
            for encoded_modes in perframe_frame_encodings:
                # max pool, maybe do something better later
                if self.multimodal_reduce == 'max':
                    frame_space_tokens = encoded_modes.max(dim=2)[0]
                elif self.multimodal_reduce == 'mean':
                    frame_space_tokens = encoded_modes.mean(dim=2)[0]
                else:
                    raise Exception(self.multimodal_reduce)
                perframe_stackable_encodings.append(frame_space_tokens)

            perframe_spacetime_tokens = torch.cat(perframe_stackable_encodings, dim=2)[:, 0]
            spacetime_fused_features = einops.rearrange(perframe_spacetime_tokens, 'b t (hs ws) f -> b t hs ws f', hs=hs, ws=ws)

            # Do we need a decorder now?
            # Pass the final fused space-time feature to a classifier
            logits = {}
            if 'change' in self.heads:
                # TODO: the change head should take unmodified features as
                # input and then it can do a dot-product / look at second half
                # / whatever.
                # _feats = spacetime_fused_features[:, :-1] - spacetime_fused_features[:, 1:]
                logits['change'] = self.heads['change'](spacetime_fused_features[:, 1:, :, :, :])
            if 'class' in self.heads:
                logits['class'] = self.heads['class'](spacetime_fused_features)
            if 'saliency' in self.heads:
                logits['saliency'] = self.heads['saliency'](spacetime_fused_features)

            # TODO: it may be faster to compute loss at the downsampled
            # resolution.
            resampled_logits = {}
            # Loop over change, categories, saliency
            for logit_key, logit_val in logits.items():
                _tmp = einops.rearrange(logit_val, 'b t h w c -> b (t c) h w')
                _tmp2 = nn.functional.interpolate(
                    _tmp, [H, W], mode='bilinear', align_corners=True)
                resampled = einops.rearrange(_tmp2, 'b (t c) h w -> b t h w c', c=logit_val.shape[4])
                resampled_logits[logit_key] = resampled

            # Convert logits into probabilities for output
            # Remove batch index in both cases
            probs = {}
            if 'change' in resampled_logits:
                probs['change'] = resampled_logits['change'].detach().softmax(dim=4)[0, ..., 1]
            if 'class' in resampled_logits:
                probs['class'] = resampled_logits['class'].detach().sigmoid()[0]
            if 'saliency' in resampled_logits:
                probs['saliency'] = resampled_logits['saliency'].detach().sigmoid()[0]

            # Append the item result to the batch outputs
            for k, v in probs.items():
                batch_head_probs[k].append(v)

            item_loss_parts = {}
            if with_loss:
                # Stack the weights for each item
                item_weights = {
                    # Because we are nt collating we need to add a batch dimension
                    key: torch.stack(_tensors)[None, ...]
                    for key, _tensors in item_pixel_weights_list.items()
                }

                item_truths = {}
                if self.global_head_weights['change']:
                    # [B, T, H, W]
                    item_truths['change'] = torch.stack([
                        frame['change'] for frame in item['frames'][1:]
                    ])[None, ...]

                if self.global_head_weights['class']:
                    # [B, T, H, W]
                    item_truths['class'] = torch.stack([
                        frame['class_idxs'] for frame in item['frames']
                    ])[None, ...]

                if self.global_head_weights['saliency']:
                    item_truths['saliency'] = torch.stack([
                        frame['saliency'] for frame in item['frames']
                    ])[None, ...]

                for k, v in batch_head_truths.items():
                    v.append(item_truths[k])

                # Compute criterion loss for each head
                for head_key, head_logits in resampled_logits.items():

                    criterion = self.criterions[head_key]
                    head_truth = item_truths[head_key]
                    head_weights = item_weights[head_key]
                    global_head_weight = self.global_head_weights[head_key]

                    head_pred_input = einops.rearrange(head_logits, 'b t h w c -> ' + criterion.logit_shape).contiguous()
                    head_weights_input = einops.rearrange(head_weights[..., None], 'b t h w c -> ' + criterion.logit_shape).contiguous()

                    if criterion.target_encoding == 'index':
                        head_true_idxs = head_truth.long()
                        head_true_input = einops.rearrange(head_true_idxs, 'b t h w -> ' + criterion.target_shape).contiguous()
                        head_weights_input = head_weights_input[:, 0]
                    elif criterion.target_encoding == 'onehot':
                        # Note: 1HE is much easier to work with
                        head_true_ohe = kwarray.one_hot_embedding(head_truth.long(), criterion.in_channels, dim=-1)
                        head_true_input = einops.rearrange(head_true_ohe, 'b t h w c -> ' + criterion.target_shape).contiguous()
                    else:
                        raise KeyError(criterion.target_encoding)
                    unreduced_head_loss = criterion(head_pred_input, head_true_input)
                    full_head_weight = torch.broadcast_to(head_weights_input, unreduced_head_loss.shape)
                    # Weighted reduction
                    weighted_head_loss = (full_head_weight * unreduced_head_loss).sum() / (full_head_weight.sum() + eps_f32)
                    item_loss_parts[head_key] = global_head_weight * weighted_head_loss

                item_losses.append(item_loss_parts)

        if 'change' in batch_head_probs:
            outputs['change_probs'] = batch_head_probs['change']
        if 'class' in batch_head_probs:
            outputs['class_probs'] = batch_head_probs['class']
        if 'saliency' in batch_head_probs:
            outputs['saliency_probs'] = batch_head_probs['saliency']

        if with_loss:
            total_loss = sum(
                val for parts in item_losses for val in parts.values())

            to_compare = {}
            # Flatten everything for pixelwise comparisons
            if 'change' in batch_head_truths:
                _true = torch.cat([x.contiguous().view(-1) for x in batch_head_truths['change']], dim=0)
                _pred = torch.cat([x.contiguous().view(-1) for x in batch_head_probs['change']], dim=0)
                to_compare['change'] = (_true, _pred)

            if 'class' in batch_head_truths:
                c = self.num_classes
                # Truth is index-based (todo: per class binary maps)
                _true = torch.cat([x.contiguous().view(-1) for x in batch_head_truths['class']], dim=0)
                _pred = torch.cat([x.contiguous().view(-1, c) for x in batch_head_probs['class']], dim=0)
                to_compare['class'] = (_true, _pred)

            if 'saliency' in batch_head_truths:
                c = self.saliency_num_classes
                _true = torch.cat([x.contiguous().view(-1) for x in batch_head_truths['saliency']], dim=0)
                _pred = torch.cat([x.contiguous().view(-1, c) for x in batch_head_probs['saliency']], dim=0)
                to_compare['saliency'] = (_true, _pred)

            # compute metrics
            if self.trainer is not None:
                item_metrics = {}

                for head_key in to_compare.keys():
                    _true, _pred = to_compare[head_key]
                    # Dont log unless a trainer is attached
                    for metric_key, metric in self.head_metrics[head_key].items():
                        val = metric(_pred, _true)
                        item_metrics[f'{stage}_{head_key}_{metric_key}'] = val

                for key, val in item_metrics.items():
                    self.log(key, val, prog_bar=True)
                self.log(f'{stage}_loss', total_loss, prog_bar=True)

            # Detach the itemized losses
            for _path, val in ub.IndexableWalker(item_losses):
                if isinstance(val, torch.Tensor):
                    val.detach_()

            outputs['loss'] = total_loss
            outputs['item_losses'] = item_losses
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
        """
        DEPRECATE IN FAVOR OF watch.tasks.fusion.utils.load_model_from_package

        TODO:
            - [ ] Make the logic that defines the save_package and load_package
                methods with appropriate package header data a lightning
                abstraction.
        """
        # NOTE: there is no gaurentee that this loads an instance of THIS
        # model, the model is defined by the package and the tool that loads it
        # is agnostic to the model contained in said package.
        # This classmethod existing is a convinience more than anything else
        from watch.tasks.fusion.utils import load_model_from_package
        self = load_model_from_package(package_path)
        return self

    def save_package(self, package_path, verbose=1):
        """

        CommandLine:
            xdoctest -m watch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> dpath = ub.ensure_app_cache_dir('watch/tests/package')
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> model = methods.MultimodalTransformer("smt_it_stm_p8", input_channels=13)

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.MultimodalTransformer.load_package(package_path)
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
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> dpath = ub.ensure_app_cache_dir('watch/tests/package')
            >>> package_path = join(dpath, 'my_package.pt')

            >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
            >>>     'special:vidshapes8-multispectral-multisensor', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=0)
            >>> datamodule.setup('fit')
            >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.MultimodalTransformer(
            >>>     "smt_it_stm_p8", classes=classes,
            >>>     dataset_stats=dataset_stats, input_channels=datamodule.input_channels)

            >>> # We have to run an input through the module because it is lazy
            >>> batch = ub.peek(iter(datamodule.train_dataloader()))
            >>> outputs = self.training_step(batch)

            >>> trainer = pl.Trainer(max_steps=1)
            >>> trainer.fit(model=self, datamodule=datamodule)

            >>> # Save the self
            >>> self.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.MultimodalTransformer.load_package(package_path)

            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = self.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> for key in recon_state.keys():
            >>>     assert (model_state[key] == recon_state[key]).all()
            >>>     assert model_state[key] is not recon_state[key]

        Ignore:
            7z l $HOME/.cache/watch/tests/package/my_package.pt
        """
        # import copy
        import json
        import torch.package

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
        ]
        for key in unsaved_attributes:
            val = getattr(model, key)
            if val is not None:
                backup_attributes[key] = val

        train_dpath_hint = getattr(model, 'train_dpath_hint', None)
        if model.trainer is not None:
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
                package_header = {
                    'version': '0.1.0',
                    'arch_name': arch_name,
                    'module_name': module_name,
                }

                if 0:
                    # old encoding (keep for a while)
                    exp.save_pickle(
                        'kitware_package_header', 'kitware_package_header.pkl',
                        package_header
                    )

                    # new encoding
                    exp.save_text(
                        'kitware_package_header', 'kitware_package_header.json',
                        json.dumps(package_header)
                    )

                # move to this?
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

    @profile
    def forward(self, images):
        """
        Example:
            >>> import pytest
            >>> pytest.skip('not currently used')
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.tasks.fusion import datamodules
            >>> channels = 'B1,B8|B8a,B10|B11'
            >>> channels = 'B1|B8|B10|B8a|B11'
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0, channels=channels)
            >>> datamodule.setup('fit')
            >>> input_channels = datamodule.input_channels
            >>> train_dataset = datamodule.torch_datasets['train']
            >>> dataset_stats = train_dataset.cached_dataset_stats()
            >>> loader = datamodule.train_dataloader()
            >>> tokenizer = 'convexpt-v1'
            >>> tokenizer = 'dwcnn'
            >>> batch = next(iter(loader))
            >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_joint_p8',
            >>>     input_channels=input_channels,
            >>>     dataset_stats=dataset_stats,
            >>>     change_loss='dicefocal',
            >>>     decoder='dicefocal',
            >>>     attention_impl='performer',
            >>>     tokenizer=tokenizer,
            >>> )
            >>> images = torch.stack([ub.peek(f['modes'].values()) for f in batch[0]['frames']])[None, :]
            >>> images.shape
            >>> self.forward(images)
        """
        raise NotImplementedError('see forward_step instad')


def _class_weights_from_freq(total_freq, mode='median-idf'):
    """
    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import _class_weights_from_freq
        >>> total_freq = np.array([19503736, 92885, 883379, 0, 0])
        >>> print(_class_weights_from_freq(total_freq, mode='idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='median-idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='log-median-idf'))

        >>> total_freq = np.array([19503736, 92885, 883379, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> print(_class_weights_from_freq(total_freq, mode='idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='median-idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='log-median-idf'))
    """
    import numpy as np

    def logb(arr, base):
        if base == 'e':
            return np.log(arr)
        elif base == 2:
            return np.log2(arr)
        elif base == 10:
            return np.log10(arr)
        else:
            out = np.log(arr)
            out /= np.log(base)
            return out

    freq = total_freq.copy()
    is_natural = total_freq > 0 & np.isfinite(total_freq)
    natural_freq = freq[is_natural]
    mask = is_natural.copy()

    if len(natural_freq):
        _min, _max = np.quantile(natural_freq, [0.05, 0.95])
        is_robust = (_max >= freq) & (freq >= _min)
        if np.any(is_robust):
            middle_value = np.median(freq[is_robust])
        else:
            middle_value = np.median(natural_freq)
        freq[~is_natural] = natural_freq.min() / 2
    else:
        middle_value = 2

    # variant of median-inverse-frequency
    if mode == 'idf':
        # There is no difference and this and median after reweighting
        weights = (1 / freq)
        mask &= np.isfinite(weights)
    elif mode == 'name-me':
        z = freq[mask]
        a = ((1 - np.eye(len(z))) * z[:, None]).sum(axis=0)
        b = a / z
        c = b / b.max()
        weights = np.zeros(len(freq))
        weights[mask] = c
    elif mode == 'median-idf':
        weights = (middle_value / freq)
        mask &= np.isfinite(weights)
    elif mode == 'log-median-idf':
        weights = (middle_value / freq)
        mask &= np.isfinite(weights)
        weights[~np.isfinite(weights)] = 1.0
        base = 2
        base = np.exp(1)
        weights = logb(weights + (base - 1), base)
        weights = np.maximum(weights, .1)
        weights = np.minimum(weights, 10)
    else:
        raise KeyError('mode = {!r}'.format(mode))

    # unseen classes should probably get a reasonably high weight in case we do
    # see them and need to learn them, but my intuition is to give them
    # less weight than things we have a shot of learning well
    # so they dont mess up the main categories
    natural_weights = weights[mask]
    if len(natural_weights):
        denom = natural_weights.max()
    else:
        denom = 1
    weights[mask] = weights[mask] / denom
    weights[~mask] = weights[mask].max() / 7
    weights = np.round(weights, 6)
    return weights


def coerce_criterion(loss_code, weights):
    """
    Helps build a loss function and returns information about the shapes needed
    by the specific loss.
    """
    # import monai
    if loss_code == 'cce':
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights, reduction='none')
        target_encoding = 'index'
        logit_shape = '(b t h w) c'
        target_shape = '(b t h w)'
    elif loss_code == 'focal':
        from watch.utils.ext_monai import FocalLoss
        # from monai.losses import FocalLoss
        criterion = FocalLoss(
            reduction='none', to_onehot_y=False, weight=weights)

        target_encoding = 'onehot'
        logit_shape = 'b c h w t'
        target_shape = 'b c h w t'

    elif loss_code == 'dicefocal':
        # TODO: can we apply weights here?
        from watch.utils.ext_monai import DiceFocalLoss
        # from monai.losses import DiceFocalLoss
        criterion = DiceFocalLoss(
            # weight=torch.FloatTensor([self.negative_change_weight, self.positive_change_weight]),
            sigmoid=True,
            to_onehot_y=False,
            reduction='none')
        target_encoding = 'onehot'
        logit_shape = 'b c h w t'
        target_shape = 'b c h w t'
    else:
        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.BCEWithLogitsLoss()
        raise NotImplementedError(loss_code)

    # Augment the criterion with extra information about what it expects
    criterion.target_encoding = target_encoding
    criterion.logit_shape = logit_shape
    criterion.target_shape = target_shape
    criterion.in_channels = len(weights)
    # criterion_info = {
    #     'criterion': criterion,
    #     'target_encoding': target_encoding,
    #     'logit_shape': logit_shape,
    #     'target_shape': target_shape,
    # }
    return criterion


class OurDepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.

    From timm

    Example:
        from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA

        norm = nh.layers.rectify_normalizer(in_channels=3, key={'type': 'group', 'num_groups': 1})
        norm(torch.rand(2, 1))

        self = OurDepthwiseSeparableConv(11, 13, kernel_size=3, padding=1, residual=1)
        x = torch.rand(2, 11, 3, 3)
        y = self.forward(x)

        z = nh.OutputShapeFor(self.conv_dw)((2, 11, 1, 1))
        print('z = {!r}'.format(z))
        nh.OutputShapeFor(self.conv_pw)(z)

        in_modes = 13
        self =

        tokenizer = nn.Sequential(*[
            OurDepthwiseSeparableConv(in_modes, in_modes, kernel_size=3, stride=1, padding=1, residual=1, norm=None, noli=None),
            OurDepthwiseSeparableConv(in_modes, in_modes * 2, kernel_size=3, stride=2, padding=1, residual=0, norm=None),
            OurDepthwiseSeparableConv(in_modes * 2, in_modes * 4, kernel_size=3, stride=2, padding=1, residual=0),
            OurDepthwiseSeparableConv(in_modes * 4, in_modes * 8, kernel_size=3, stride=2, padding=1, residual=0),
        ])

        tokenizer = nn.Sequential(*[
            OurDepthwiseSeparableConv(in_modes, in_modes, kernel_size=3, stride=1, padding=1, residual=1),
            OurDepthwiseSeparableConv(in_modes, in_modes * 2, kernel_size=3, stride=2, padding=1, residual=0),
            OurDepthwiseSeparableConv(in_modes * 2, in_modes * 4, kernel_size=3, stride=2, padding=1, residual=0),
            OurDepthwiseSeparableConv(in_modes * 4, in_modes * 8, kernel_size=3, stride=2, padding=1, residual=0),
        ])
    """
    def __init__(
            self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1,
            padding=0, residual=False, pw_kernel_size=1, norm='group',
            noli='swish', drop_path_rate=0.):

        super().__init__()
        if norm == 'auto':
            norm = {'type': 'group', 'num_groups': 'auto'}

        self.has_residual = (stride == 1 and in_chs == out_chs) and residual
        self.drop_path_rate = drop_path_rate

        conv_cls = nh.layers.rectify_conv(dim=2)
        # self.conv_dw = create_conv2d(
        #     in_chs, in_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.conv_dw = conv_cls(
            in_chs, in_chs, kernel_size, stride=stride, dilation=dilation,
            padding=padding, groups=in_chs)  # depthwise

        self.bn1 = nh.layers.rectify_normalizer(in_channels=in_chs, key=norm)
        if self.bn1 is None:
            self.bn1 = nh.layers.Identity()
        self.act1 = nh.layers.rectify_nonlinearity(noli)
        if self.act1 is None:
            self.act1 = nh.layers.Identity()

        self.conv_pw = conv_cls(in_chs, out_chs, pw_kernel_size, padding=0)
        # self.bn2 = norm_layer(out_chs)
        self.bn2 = nh.layers.rectify_normalizer(in_channels=out_chs, key=norm)
        if self.bn2 is None:
            self.bn2 = nh.layers.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DWCNNTokenizer(nh.layers.Sequential):
    """
    self = DWCNNTokenizer(13, 2)
    inputs = torch.rand(2, 13, 16, 16)
    self(inputs)
    """
    def __init__(self, in_chn, out_chn, norm='auto'):
        super().__init__()
        if norm == 'none':
            norm = None
        self.norm = norm
        super().__init__(*[
            OurDepthwiseSeparableConv(in_chn, in_chn, kernel_size=3, stride=1, padding=1, residual=1, norm=None, noli=None),
            OurDepthwiseSeparableConv(in_chn, in_chn * 4, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
            OurDepthwiseSeparableConv(in_chn * 4, in_chn * 8, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
            OurDepthwiseSeparableConv(in_chn * 8, out_chn, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
        ])
        self.in_channels = in_chn
        self.out_channels = out_chn


class LinearConvTokenizer(nh.layers.Sequential):
    """
    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> LinearConvTokenizer(1, 512)
    """
    def __init__(self, in_channels, out_channels):
        # import math
        c1 = in_channels * 1
        c2 = in_channels * 4
        c3 = in_channels * 16
        c4 = in_channels * 8
        # final_groups = math.gcd(104, out_channels)
        final_groups = 1

        super().__init__(
            nh.layers.ConvNormNd(
                dim=2, in_channels=c1, out_channels=c2, groups=c1, norm=None,
                noli=None, kernel_size=3, stride=2, padding=1,
            ).conv,
            nh.layers.ConvNormNd(
                dim=2, in_channels=c2, out_channels=c3, groups=c2, norm=None,
                noli=None, kernel_size=3, stride=2, padding=1,
            ).conv,
            nh.layers.ConvNormNd(
                dim=2, in_channels=c3, out_channels=c4, groups=min(c3, c4), norm=None,
                noli=None, kernel_size=3, stride=2, padding=1,
            ).conv,
            nh.layers.ConvNormNd(
                dim=2, in_channels=c4, out_channels=out_channels,
                groups=final_groups, norm=None, noli=None, kernel_size=1,
                stride=1, padding=0,
            ).conv,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels


class ConvTokenizer(nn.Module):
    """
    Example:

        from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        self = ConvTokenizer(13, 64)
        print('self = {!r}'.format(self))
        inputs = torch.rand(2, 13, 128, 128)
        tokens = self(inputs)
        print('inputs.shape = {!r}'.format(inputs.shape))
        print('tokens.shape = {!r}'.format(tokens.shape))


     Benchmark:

        in_channels = 13
        tokenizer1 = ConvTokenizer(in_channels, 512)
        tokenizer2 = RearrangeTokenizer(in_channels, 8, 8)
        tokenizer3 = DWCNNTokenizer(in_channels, 512)
        tokenizer4 = LinearConvTokenizer(in_channels, 512)
        print(nh.util.number_of_parameters(tokenizer1))
        print(nh.util.number_of_parameters(tokenizer2))
        print(nh.util.number_of_parameters(tokenizer3))
        print(nh.util.number_of_parameters(tokenizer4))

        print(nh.util.number_of_parameters(tokenizer4[0]))
        print(nh.util.number_of_parameters(tokenizer4[1]))
        print(nh.util.number_of_parameters(tokenizer4[2]))
        print(nh.util.number_of_parameters(tokenizer4[3]))

        inputs = torch.rand(1, in_channels, 128, 128)

        import timerit
        ti = timerit.Timerit(100, bestof=1, verbose=2)

        tokenizer1(inputs).shape
        tokenizer2(inputs).shape

        for timer in ti.reset('tokenizer1'):
            with timer:
                tokenizer1(inputs)

        for timer in ti.reset('tokenizer2'):
            with timer:
                tokenizer2(inputs)

        for timer in ti.reset('tokenizer3'):
            with timer:
                tokenizer3(inputs)

        for timer in ti.reset('tokenizer4'):
            with timer:
                tokenizer4(inputs)

        input_shape = (1, in_channels, 64, 64)

        print(tokenizer2(torch.rand(*input_shape)).shape)
        downsampler1 = nh.layers.Sequential(*[
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=3,
                stride=2, padding=1,
            ),
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=3,
                stride=2, padding=1,
            ),
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=3,
                stride=2, padding=1,
            ),
        ])

        downsampler2 = nh.layers.Sequential(*[
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=7,
                stride=5, padding=3,
            ),
        ])
        print(ub.repr2(downsampler1.output_shape_for(input_shape).hidden.shallow(30), nl=1))
        print(ub.repr2(downsampler2.output_shape_for(input_shape).hidden.shallow(30), nl=1))


    """
    def __init__(self, in_chn, out_chn, norm=None):
        super().__init__()
        self.down = nh.layers.ConvNormNd(
            dim=2, in_channels=in_chn, out_channels=in_chn, groups=in_chn,
            norm=norm, noli=None, kernel_size=7, stride=5, padding=3,
        )
        self.one_by_one = nh.layers.ConvNormNd(
            dim=2, in_channels=in_chn, out_channels=out_chn, groups=1,
            norm=norm, noli=None, kernel_size=1, stride=1, padding=0,
        )
        self.out_channels = out_chn

    def forward(self, inputs):
        # b, t, c, h, w = inputs.shape
        b, c, h, w = inputs.shape
        # inputs2d = einops.rearrange(inputs, 'b t c h w -> (b t) c h w')
        inputs2d = inputs
        tokens2d = self.down(inputs2d)
        tokens2d = self.one_by_one(tokens2d)
        tokens = tokens2d
        # tokens = einops.rearrange(tokens2d, '(b t) c h w -> b t c h w 1', b=b, t=t)
        return tokens


class RearrangeTokenizer(nn.Module):
    """
    A mapping to a common number of channels and then rearrange

    Not quite a pure rearrange, but is this way for backwards compat
    """
    def __init__(self, in_channels, agree, window_size):
        super().__init__()
        self.window_size = window_size
        self.foot = nh.layers.MultiLayerPerceptronNd(
            dim=2, in_channels=in_channels, hidden_channels=3,
            out_channels=agree, residual=True, norm=None)
        self.out_channels = agree * window_size * window_size

    def forward(self, x):
        mixed_mode = self.foot(x)
        ws = self.window_size
        # HACK: reorganize and fix
        mode_vals_tokens = einops.rearrange(
            mixed_mode, 'b c (h hs) (w ws) -> b (ws hs c) h w', hs=ws, ws=ws)
        return mode_vals_tokens
