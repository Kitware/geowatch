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
from watch.tasks.fusion.methods.network_modules import _torch_meshgrid
from watch.tasks.fusion.methods.network_modules import _class_weights_from_freq
from watch.tasks.fusion.methods.network_modules import coerce_criterion
from watch.tasks.fusion.methods.network_modules import RobustModuleDict
from watch.tasks.fusion.methods.network_modules import RearrangeTokenizer
from watch.tasks.fusion.methods.network_modules import ConvTokenizer
from watch.tasks.fusion.methods.network_modules import LinearConvTokenizer
from watch.tasks.fusion.methods.network_modules import DWCNNTokenizer

import scriptconfig as scfg

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# Model names define the transformer encoder used by the method
available_encoders = list(transformer.encoder_configs.keys()) + ['deit']


@scfg.dataconf
class MultimodalTransformerConfig(scfg.DataConfig):
    """
    Arguments accepted by the MultimodalTransformer

    The scriptconfig class is not used directly as it normally would be here.
    Instead we use it as a convinience to minimize lightning boilerplate needed
    for the __init__ and add_argparse_args methods.

    Note, this does not entirely define the `__init__` method, just the
    parameters that are exposed on the command line. An update to
    scriptconfig could allow that to be combined, but I'm not sure if its a
    good idea. The arguments not specified here are usually ones that the
    dataset must provide at definition time.
    """
    name = scfg.Value('unnamed_model', help=ub.paragraph(
        '''
        Specify a name for the experiment. (Unsure if the Model is
        the place for this)
        '''))
    optimizer = scfg.Value('RAdam', type=str, help=ub.paragraph(
        '''
        Optimizer name supported by the netharn API
        '''))
    learning_rate = scfg.Value(0.001, type=float)
    weight_decay = scfg.Value(0.0, type=float)
    positive_change_weight = scfg.Value(1.0, type=float)
    negative_change_weight = scfg.Value(1.0, type=float)
    class_weights = scfg.Value('auto', type=str, help='class weighting strategy')
    saliency_weights = scfg.Value('auto', type=str, help='class weighting strategy')
    stream_channels = scfg.Value(8, type=int, help=ub.paragraph(
        '''
        number of channels to normalize each project stream to
        '''))
    tokenizer = scfg.Value('rearrange', type=str, choices=[
        'dwcnn', 'rearrange', 'conv7', 'linconv'], help=ub.paragraph(
        '''
        How image patches are broken into tokens. rearrange is a 1x1
        MLP and grouping of pixel grids. dwcnn is a is a mobile
        convolutional stem. conv7 is a simple 1x1x7x7 convolutional
        stem. linconv is a stack of 3x3 grouped convolutions without
        any nonlinearity
        '''))
    token_norm = scfg.Value('none', type=str, choices=['none', 'auto', 'group', 'batch'])
    arch_name = scfg.Value('smt_it_joint_p8', type=str, choices=available_encoders)
    decoder = scfg.Value('mlp', type=str, choices=['mlp', 'segmenter'])
    dropout = scfg.Value(0.1, type=float)
    global_class_weight = scfg.Value(1.0, type=float)
    global_change_weight = scfg.Value(1.0, type=float)
    global_saliency_weight = scfg.Value(1.0, type=float)
    modulate_class_weights = scfg.Value('', type=str, help=ub.paragraph(
        '''
        a special syntax that lets the user modulate automatically
        computed class weights. Should be a comma separated list of
        name*weight or name*weight+offset. E.g.
        `negative*0,background*0.001,No Activity*0.1+1`
        '''))
    change_loss = scfg.Value('cce')
    class_loss = scfg.Value('focal')
    saliency_loss = scfg.Value('focal', help=ub.paragraph(
        '''
        saliency is trained to match any
        "positive/foreground/salient" class
        '''))
    change_head_hidden = scfg.Value(2, type=int, help=ub.paragraph(
        '''
        number of hidden layers in the change head
        '''))
    class_head_hidden = scfg.Value(2, type=int, help=ub.paragraph(
        '''
        number of hidden layers in the category head
        '''))
    saliency_head_hidden = scfg.Value(2, type=int, help=ub.paragraph(
        '''
        number of hidden layers in the saliency head
        '''))
    window_size = scfg.Value(8, type=int)
    squash_modes = scfg.Value(False)
    decouple_resolution = scfg.Value(False, help=ub.paragraph(
        '''
        this turns on logic to decouple input and output
        resolutions. Probably very slow
        '''))
    attention_impl = scfg.Value('exact', type=str, help=ub.paragraph(
        '''
        Implementation for attention computation. Can be: 'exact' -
        the original O(n^2) method. 'performer' - a linear
        approximation. 'reformer' - a LSH approximation.
        '''))
    multimodal_reduce = scfg.Value('max', help=ub.paragraph(
        '''
        operation used to combine multiple modes from the same timestep
        '''))


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
        >>> print('(STEP 0): SETUP THE DATA MODULE')
        >>> datamodule = datamodules.KWCocoVideoDataModule(
        >>>     train_dataset='special:vidshapes-watch', num_workers=4)
        >>> datamodule.setup('fit')
        >>> dataset = datamodule.torch_datasets['train']
        >>> print('(STEP 1): ESTIMATE DATASET STATS')
        >>> dataset_stats = dataset.cached_dataset_stats(num=3)
        >>> print('dataset_stats = {}'.format(ub.repr2(dataset_stats, nl=3)))
        >>> loader = datamodule.train_dataloader()
        >>> print('(STEP 2): SAMPLE BATCH')
        >>> batch = next(iter(loader))
        >>> for item_idx, item in enumerate(batch):
        >>>     print(f'item_idx={item_idx}')
        >>>     for frame_idx, frame in enumerate(item['frames']):
        >>>         print(f'  * frame_idx={frame_idx}')
        >>>         print(f'  * frame.sensor = {frame["sensor"]}')
        >>>         for mode_code, mode_val in frame['modes'].items():
        >>>             print(f'      * {mode_code=} @shape={mode_val.shape}, num_nam={mode_val.isnan().sum()}')
        >>> print('(STEP 3): THE REST OF THE TEST')
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
        >>> output = self.forward_step(batch, with_loss=True)
        >>> import torch.profiler
        >>> from torch.profiler import profile, ProfilerActivity
        >>> with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        >>>     with torch.profiler.record_function("model_inference"):
        >>>         output = self.forward_step(batch, with_loss=True)
        >>> print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    Ignore:
        kwplot.autompl()
        kwplot.imshow(dataset.draw_item(batch[0]))
    """
    _HANDLES_NANS = True

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

            print(scfg.Config.port_argparse(parent_parser, style='dataconf'))
        """
        parser = parent_parser.add_argument_group('kwcoco_video_data')
        config = MultimodalTransformerConfig()
        config.argparse(parser)
        return parent_parser

    @classmethod
    def compatible(cls, cfgdict):
        """
        Given keyword arguments, find the subset that is compatible with this
        constructor. This is somewhat hacked because of usage of scriptconfig,
        but could be made nicer by future updates.
        """
        # init_kwargs = ub.compatible(config, cls.__init__)
        import inspect
        nameable_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY}
        cls_sig = inspect.signature(cls)
        explicit_argnames = [
            argname for argname, argtype in cls_sig.parameters.items()
            if argtype.kind in nameable_kinds
        ]
        valid_argnames = explicit_argnames + list(MultimodalTransformerConfig.__default__.keys())
        clsvars = ub.dict_isect(cfgdict, valid_argnames)
        return clsvars

    def get_cfgstr(self):
        cfgstr = f'{self.name}_{self.arch_name}'
        return cfgstr

    def __init__(self, *, classes=10, dataset_stats=None, input_channels=None,
                 unique_sensors=None, **kwargs):

        super().__init__()
        config = MultimodalTransformerConfig(**kwargs)
        self.config = config
        cfgdict = self.config.to_dict()
        self.save_hyperparameters(cfgdict)
        # Backwards compatibility. Previous iterations had the
        # config saved directly as datamodule arguments
        self.__dict__.update(cfgdict)

        # We are explicitly unpacking the config here to make
        # transition to a scriptconfig style init easier. This
        # code can be consolidated later.
        saliency_weights = config['saliency_weights']
        class_weights = config['class_weights']
        tokenizer = config['tokenizer']
        token_norm = config['token_norm']
        change_head_hidden = config['change_head_hidden']
        class_head_hidden = config['class_head_hidden']
        saliency_head_hidden = config['saliency_head_hidden']
        class_loss = config['class_loss']
        change_loss = config['change_loss']
        saliency_loss = config['saliency_loss']
        arch_name = config['arch_name']
        dropout = config['dropout']
        attention_impl = config['attention_impl']
        global_class_weight = config['global_class_weight']
        global_change_weight = config['global_change_weight']
        global_saliency_weight = config['global_saliency_weight']

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
                if s not in input_norms:
                    input_norms[s] = RobustModuleDict()
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

        self.positive_change_weight = config['positive_change_weight']
        self.negative_change_weight = config['negative_change_weight']

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

            self.sensor_channel_tokenizers[s][c] = tokenize
            in_features_raw = tokenize.out_channels

        # for (s, c), stats in input_stats.items():
        #     self.sensor_channel_tokenizers[s][c] = tokenize

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

        if hasattr(torchmetrics, 'FBetaScore'):
            FBetaScore = torchmetrics.FBetaScore
        else:
            FBetaScore = torchmetrics.FBeta

        self.head_metrics = nn.ModuleDict()
        self.head_metrics['class'] = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1_micro': FBetaScore(beta=1.0, threshold=0.5, average='micro'),
            'f1_macro': FBetaScore(beta=1.0, threshold=0.5, average='macro', num_classes=self.num_classes),
        })
        self.head_metrics['change'] = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1': FBetaScore(beta=1.0),
        })
        self.head_metrics['saliency'] = nn.ModuleDict({
            'f1': FBetaScore(beta=1.0),
        })

        self.encode_h = utils.SinePositionalEncoding(3, 1, size=8)
        self.encode_w = utils.SinePositionalEncoding(3, 2, size=8)

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
            >>> self = methods.MultimodalTransformer(arch_name="smt_it_stm_p8", input_channels='r|g|b')
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
            >>>     arch_name='smt_it_joint_p2',
            >>>     #arch_name='smt_it_stm_p8',
            >>>     learning_rate=1e-8,
            >>>     #attention_impl='performer',
            >>>     attention_impl='exact',
            >>>     decoder='segmenter',
            >>>     #decoder='mlp',
            >>>     #arch_name='deit',
            >>>     change_loss='dicefocal',
            >>>     #class_loss='cce',
            >>>     class_loss='dicefocal',
            >>>     saliency_loss='dicefocal',
            >>>     # ===========
            >>>     # Change Loss
            >>>     global_change_weight=1.00,
            >>>     positive_change_weight=1.0,
            >>>     negative_change_weight=0.5,
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
                    print('NAN OUTPUT!!!')
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

    def demo_batch(self, batch_size=1, num_timesteps=3, width=8, height=8, nans=0, rng=None):
        """
        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_channels=channels)
            >>> batch = self.demo_batch()
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> result = self.forward_step(batch)
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(result))

        Example:
            >>> # With nans
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_channels=channels)
            >>> batch = self.demo_batch(nans=0.5, num_timesteps=2)
            >>> item = batch[0]
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> result1 = self.forward_step(batch)
            >>> result2 = self.forward_step(batch, with_loss=0)
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(result1))
            >>>   print(nh.data.collate._debug_inbatch_shapes(result2))
        """
        import kwarray
        from kwarray import distributions
        def _specific_coerce(val, rng=None):
            # Coerce for what we want to do here,
            import numbers
            if isinstance(val, numbers.Integral):
                distri = distributions.Constant(val, rng=rng)
            elif isinstance(val, (tuple, list)) and len(val) == 2:
                low, high = val
                distri = distributions.DiscreteUniform(low, high, rng=rng)
            else:
                raise TypeError(val)
            return distri

        rng = kwarray.ensure_rng(rng)

        B = batch_size
        C = len(self.classes)
        T = num_timesteps
        batch = []

        width_distri = _specific_coerce(width, rng=rng)
        height_distri = _specific_coerce(height, rng=rng)

        for bx in range(B):
            modes = []
            frames = []
            for time_index in range(T):

                # Sample output target shape
                H0 = height_distri.sample()
                W0 = width_distri.sample()

                # Sample input shapes
                H1 = height_distri.sample()
                W1 = width_distri.sample()

                H2 = height_distri.sample()
                W2 = width_distri.sample()

                modes = {
                    'pan': rng.rand(1, H0, W0),
                    'red|green|blue': rng.rand(3, H1, W1),
                    'nir|swir16|swir22': rng.rand(3, H2, W2),
                }
                frame = {}
                if time_index == 0:
                    frame['change'] = None
                    frame['change_weights'] = None
                else:
                    frame['change'] = rng.randint(low=0, high=1, size=(H0, W0))
                    frame['change_weights'] = rng.rand(H0, W0)

                frame['class_idxs'] = rng.randint(low=0, high=C - 1, size=(H0, W0))
                frame['class_weights'] = rng.rand(H0, W0)

                frame['saliency'] = rng.randint(low=0, high=1, size=(H0, W0))
                frame['saliency_weights'] = rng.rand(H0, W0)

                frame['date_captured'] = '',
                frame['gid'] = bx
                frame['sensor'] = 'sensor1'
                frame['time_index'] = bx
                frame['time_offset'] = np.array([1]),
                frame['timestamp'] = 1
                frame['modes'] = modes
                # specify the desired predicted output size for this frame
                # frame['output_wh'] = (H0, W0)

                if nans:
                    for v in modes.values():
                        flags = rng.rand(*v.shape) < nans
                        v[flags] = float('nan')

                for k in ['change', 'change_weights', 'class_idxs',
                          'class_weights', 'saliency', 'saliency_weights']:
                    v = frame[k]
                    if v is not None:
                        frame[k] = torch.from_numpy(v)

                for k in modes.keys():
                    v = modes[k]
                    if v is not None:
                        modes[k] = torch.from_numpy(v)

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
                    slice(0, H0),
                    slice(0, W0),
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

    def prepare_item(self, item):
        pass

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
            >>>     normalize_inputs=8, neg_to_pos_ratio=0, batch_size=1,
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

        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='segmenter', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_channels=channels)
            >>> batch = self.demo_batch()
            >>> outputs = self.forward_step(batch, with_loss=True)
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> print(nh.data.collate._debug_inbatch_shapes(outputs))
        """
        outputs = {}

        item_losses = []

        # Initialize truth and probs for each head / item that will be stacked
        batch_head_truths = {k: [] for k in self.heads.keys()}
        batch_head_probs = {k: [] for k in self.heads.keys()}
        skip_flags = []
        for item in batch:
            # Skip
            if item is None:
                skip_flags.append(True)
                continue
            skip_flags.append(False)
            probs, item_loss_parts, item_truths = self.forward_item(item, with_loss=with_loss)
            # with xdev.embed_on_exception_context:
            if with_loss:
                item_losses.append(item_loss_parts)
                if not self.decouple_resolution:
                    # TODO: fixme decouple_res
                    for k, v in batch_head_truths.items():
                        v.append(item_truths[k])
            # Append the item result to the batch outputs
            for k, v in probs.items():
                batch_head_probs[k].append(v)

        if all(skip_flags):
            return None

        if 'change' in batch_head_probs:
            outputs['change_probs'] = batch_head_probs['change']
        if 'class' in batch_head_probs:
            outputs['class_probs'] = batch_head_probs['class']
        if 'saliency' in batch_head_probs:
            outputs['saliency_probs'] = batch_head_probs['saliency']

        if with_loss:
            total_loss = sum(
                val for parts in item_losses for val in parts.values())

            if not self.decouple_resolution:
                # TODO: fixme decouple_res

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

    def forward_item(self, item, with_loss=False):
        """
        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='segmenter', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_channels=channels)
            >>> item = self.demo_batch(width=64, height=65)[0]
            >>> outputs = self.forward_item(item, with_loss=True)
            >>> print('item')
            >>> print(nh.data.collate._debug_inbatch_shapes(item))
            >>> print('outputs')
            >>> print(nh.data.collate._debug_inbatch_shapes(outputs))

        Example:
            >>> # Decoupled resolutions
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_channels=channels, decouple_resolution=True)
            >>> batch = self.demo_batch(width=(11, 21), height=(16, 64), num_timesteps=3)
            >>> item = batch[0]
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> result1 = self.forward_step(batch, with_loss=True)
            >>> print(nh.data.collate._debug_inbatch_shapes(result1))
            >>> # Check we can go backward
            >>> result1['loss'].backward()
        """
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

        # device = self.device

        # For loops are for handing heterogeneous inputs
        tokenized = []
        recon_info = []
        num_frames = len(item['frames'])

        # We will use this to mark each token with its coordinates
        DECOUPLED_COORDINATE_ATTENTION = self.decouple_resolution
        if DECOUPLED_COORDINATE_ATTENTION:
            coordinates = {
                # 'batch': [],
                'time': [],
                'mode': [],
                'height': [],
                'width': [],
            }
            _seen_sensemodes = {}

        for frame_idx, (frame, frame_enc) in enumerate(zip(item['frames'], per_frame_pos_encoding)):
            modes = frame['modes']
            sensor = frame['sensor']
            # print(f'sensor={sensor}')
            for chan_code, mode_val in modes.items():
                # print(f'  * chan_code={chan_code}')

                frame_sensor_chan_tokens, space_shape = self.forward_foot(sensor, chan_code, mode_val, frame_enc)

                if DECOUPLED_COORDINATE_ATTENTION:
                    # Get downsampled sizes
                    tok_h, tok_w = frame_sensor_chan_tokens.shape[0:2]

                    sensechan = f'{sensor}:{chan_code}'
                    if sensechan not in _seen_sensemodes:
                        _seen_sensemodes[sensechan] = len(_seen_sensemodes) + 1
                    sensechan_idx = _seen_sensemodes[sensechan]

                    # Build our own coordinate representation for use in a choosing
                    # what we attend to.
                    yrel_basis = torch.linspace(0., 1., tok_h)
                    xrel_basis =  torch.linspace(0., 1., tok_w)
                    ygrid, xgrid = _torch_meshgrid(yrel_basis, xrel_basis)
                    xrel = xgrid.view(-1)
                    yrel = ygrid.view(-1)
                    # coordinates['batch'].append(torch.full((xrel.numel(),), fill_value=0))
                    coordinates['time'].append(torch.full((xrel.numel(),), fill_value=frame_idx))
                    coordinates['mode'].append(torch.full((xrel.numel(),), fill_value=sensechan_idx))
                    coordinates['height'].append(yrel)
                    coordinates['width'].append(xrel)

                flat_frame_sensor_chan_tokens = einops.rearrange(frame_sensor_chan_tokens, 'hs ws f -> (hs ws) f')

                tokenized.append(flat_frame_sensor_chan_tokens)
                recon_info.append({
                    'input_space_shape': mode_val.shape[1:3],
                    'space_shape': space_shape,
                    'frame_idx': frame_idx,
                    'chan_code': chan_code,
                    'sensor': sensor,
                })

        if len(tokenized) == 0:
            print('Error concat of:')
            print('tokenized = {}'.format(ub.repr2(tokenized, nl=1)))
            print('item = {}'.format(ub.repr2(item, nl=1)))
            for frame_idx, frame in enumerate(item['frames']):
                if len(frame['modes']) == 0:
                    print('Frame {} had no modal data'.format(frame_idx))
            raise ValueError(
                'Got an input sequence of length 0. '
                'Is there a dataloader problem?')

        _tokens = torch.concat(tokenized, dim=0)

        if DECOUPLED_COORDINATE_ATTENTION:
            flat_coordinates = ub.map_vals(lambda x: torch.concat(x, dim=0), coordinates)
            # Could use this as the basis for a positional encoding as well
            # torch.stack(list(flat_coordinates.values()))

        HACK_FOR_OLD_WEIGHTS = not DECOUPLED_COORDINATE_ATTENTION
        if HACK_FOR_OLD_WEIGHTS:
            # The encoder does seem to be making use of the fact that we can
            # put space and modality on different dimensions
            num_time_modes = len(recon_info)
            input_feat_dim = _tokens.shape[-1]
            tokens = _tokens.view(1, 1, num_time_modes, 1, -1, input_feat_dim)
            encoded_tokens = self.encoder(tokens)
            enc_feat_dim = encoded_tokens.shape[-1]
            encoded_tokens = encoded_tokens.view(-1, enc_feat_dim)
        else:
            # Special case for the explicit coordinate version
            encoded_tokens = self.encoder(_tokens, flat_coordinates=flat_coordinates)

        """
        notes:

        TOKENS -> TRANSFORMER -> ENCODED_TOKENS

        ENCODED_TOKENS -> (RESAMPLE PER-MODE IF NEEDED) -> POOL_OVER_MODES -> SPACE_TIME_FEATURES

        SPACE_TIME_FEATURES -> HEAD

        """

        token_split_points = np.cumsum([t.shape[0] for t in tokenized])
        token_split_sizes = np.diff(np.r_[[0], token_split_points]).tolist()
        # split_frames = np.array([r['frame_idx'] for r in recon_info])
        # frame_split_points = np.where(np.diff(split_frames))[0]
        # token_frame_split_points = token_split_points[frame_split_points]
        # sensorchan_shapes = np.array([r['space_shape'] for r in recon_info])
        # sensorchan_areas = sensorchan_shapes.prod(axis=1)
        # np.split(sensorchan_shapes, frame_split_points)
        # np.split(sensorchan_areas, frame_split_points)
        split_encoding = torch.split(encoded_tokens, token_split_sizes)
        for info, mode in zip(recon_info, split_encoding):
            info['mode'] = mode
        grouped = ub.group_items(recon_info, key=lambda x: x['frame_idx'])
        perframe_stackable_encodings = []
        frame_shapes = []
        for frame_idx, frame_group in sorted(grouped.items()):
            shapes = [g['space_shape'] for g in frame_group]
            modes = [g['mode'] for g in frame_group]
            # Up to this point, each mode has been processed in its native
            # resolution. But now we need to marginalize over modes to get a
            # single feature for each timestep. To do this we choose the
            # largest resolution per frame.
            if ub.allsame(shapes):
                to_stack = modes
                frame_shape = shapes[0]
            else:
                frame_shape = list(max(shapes, key=lambda x: x[0] * x[1]))
                to_stack = [
                    nn.functional.interpolate(
                        m.view(1, -1, *s), frame_shape, mode='bilinear', align_corners=False).view(-1, m.shape[-1])
                    for m, s in zip(modes, shapes)
                ]
            frame_shapes.append(frame_shape)
            stack = torch.stack(to_stack, dim=0)
            if self.multimodal_reduce == 'max':
                frame_feat = torch.max(stack, dim=0)[0]
            elif self.multimodal_reduce == 'mean':
                frame_feat = torch.mean(stack, dim=0)[0]
            else:
                raise Exception(self.multimodal_reduce)
            hs, ws = frame_shape
            frame_grid = einops.rearrange(
                frame_feat, '(hs ws) f -> hs ws f', hs=hs, ws=ws)
            perframe_stackable_encodings.append(frame_grid)

        # We now have built a feature for each frame.
        # These features are aware of other frames, but are
        # at the specific frame's max resolution.

        all_shapes = [g['input_space_shape'] for g in recon_info]
        output_shape = list(max(all_shapes, key=lambda x: x[0] * x[1]))
        H, W = output_shape

        if not self.decouple_resolution:
            # Optimization for case where frames have same shape
            spacetime_features = torch.stack(perframe_stackable_encodings)
            logits = {}
            if 'change' in self.heads and num_frames > 1:
                # TODO: the change head should take unmodified features as
                # input and then it can do a dot-product / look at second half
                # / whatever.
                # _feats = spacetime_fused_features[:, :-1] - spacetime_fused_features[:, 1:]
                change_feat = spacetime_features[1:]
                logits['change'] = self.heads['change'](change_feat)
            if 'class' in self.heads:
                logits['class'] = self.heads['class'](spacetime_features)
            if 'saliency' in self.heads:
                logits['saliency'] = self.heads['saliency'](spacetime_features)

            # TODO: it may be faster to compute loss at the downsampled
            # resolution.

            # Thus far each frame has computed its features and predictions at a
            # resolution independent of other time time steps. But now we resample
            # all logits so all items in the temporal seqeunce have the same
            # spatial resolution.
            resampled_logits = {}
            # Loop over change, categories, saliency
            for logit_key, logit_val in logits.items():
                _tmp = einops.rearrange(logit_val, 't h w c -> 1 (t c) h w')
                _tmp2 = nn.functional.interpolate(
                    _tmp, [H, W], mode='bilinear', align_corners=True)
                resampled = einops.rearrange(_tmp2, 'b (t c) h w -> b t h w c', c=logit_val.shape[3])
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
        else:
            # For class / saliency frames are indepenent
            perframe_logits = ub.ddict(list)
            for frame_feature in perframe_stackable_encodings:
                for head_key in ['class', 'saliency']:
                    perframe_logits[head_key].append(self.heads[head_key](frame_feature))
            # For change, frames are dependant, so we have to do some resampling
            if 'change' in self.heads and num_frames > 1:
                resampled_frame_feats = [
                    nn.functional.interpolate(
                        einops.rearrange(ff, 'h w c -> 1 c h w'),
                        [H, W], mode='bilinear', align_corners=False)
                    for ff in perframe_stackable_encodings]
                _tmp1 = torch.concat(resampled_frame_feats, dim=0)
                resampled_ff = einops.rearrange(_tmp1, 't c h w -> t h w c')
                change_feats = resampled_ff[:-1] - resampled_ff[1:]
                perframe_logits['change'] = self.heads['change'](change_feats)

            resampled_logits = {}
            # Loop over change, categories, saliency
            for logit_key, logit_val in perframe_logits.items():
                resampled_frame_logits = [
                    nn.functional.interpolate(
                        einops.rearrange(frame_logs, 'h w c -> 1 c h w'),
                        [H, W], mode='bilinear', align_corners=False)
                    for frame_logs in logit_val]
                _tmp2 = torch.concat(resampled_frame_logits, dim=0)
                resampled = einops.rearrange(_tmp2, 't c h w -> 1 t h w c')
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

        if with_loss:
            item_loss_parts, item_truths = self._build_item_loss_parts(
                item, resampled_logits)
        else:
            item_loss_parts = None
            item_truths = None
        return probs, item_loss_parts, item_truths

    def forward_foot(self, sensor, chan_code, mode_val: torch.Tensor, frame_enc):
        mode_val = mode_val.float()
        if self.input_norms is not None:
            try:
                mode_norm = self.input_norms[sensor][chan_code]
                mode_val = mode_norm(mode_val)
            except KeyError:
                print(f'Failed to process {sensor=!r} {chan_code=!r}')
                print(f'self.input_norms={self.input_norms}')
                print('Expected available norms (note the keys contain escape sequences) are:')
                for _s in sorted(self.input_norms.keys()):
                    for _c in sorted(self.input_norms[_s].keys()):
                        print(f'{_s=!r} {_c=!r}')
                print('self.unique_sensor_modes = {!r}'.format(self.unique_sensor_modes))
                raise
            # self.sensor_channel_tokenizers[]

        # After input normalization happens, replace nans with zeros
        # which is effectively imputing the dataset mean
        mode_val = mode_val.nan_to_num_()

        # Lookup the "tokenizing" network for this type of input
        sensor_chan_tokenizer = self.sensor_channel_tokenizers[sensor][chan_code]

        # Is it worth gathering and stacking items in batches here?

        # Reduce spatial dimension and normalize the number of
        # features in each input stream.
        _mode_val_tokens = sensor_chan_tokenizer(mode_val[None, :])

        mode_vals_tokens = einops.rearrange(
            _mode_val_tokens, 'b c h w -> b h w c')
        space_shape = mode_vals_tokens.shape[1:3]

        # Add ordinal spatial encoding
        x1 = self.encode_w(self.encode_h(mode_vals_tokens))

        # Any tricks needed to handle inputs/outputs at different
        # resolutions might go here
        encoding_expanded = frame_enc[None, None, None, :].expand(list(x1.shape[0:3]) + [frame_enc.shape[0]])

        # Mixup the space/time dims into the token dims to make this
        # general.
        x2 = torch.cat([x1, encoding_expanded.type_as(x1)], dim=3)

        # frame_sensor_chan_tokens = einops.rearrange(x2, 't hs ws f -> (t hs ws) f')

        # Keep time/sensor channel separate for now, but we will
        # need to flatten it, at which point we need a decoder.
        # frame_sensor_chan_tokens = einops.rearrange(x2, '1 hs ws f -> (hs ws) f')
        frame_sensor_chan_tokens = einops.rearrange(x2, '1 hs ws f -> hs ws f')
        return frame_sensor_chan_tokens, space_shape

    def _head_loss(self, head_key, head_logits, head_truth, head_weights):
        criterion = self.criterions[head_key]
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
        EPS_F32 = 1e-9
        weighted_head_loss = (full_head_weight * unreduced_head_loss).sum() / (full_head_weight.sum() + EPS_F32)
        head_loss = global_head_weight * weighted_head_loss
        return head_loss

    def _build_item_loss_parts(self, item, resampled_logits):
        item_loss_parts = {}
        item_truths = {}
        if self.decouple_resolution:
            for head_key, head_logits in resampled_logits.items():

                if head_key == 'class':
                    truth_label_key = 'class_idxs'
                    truth_weight_key = 'class_weights'
                    start_idx = 0
                elif head_key == 'saliency':
                    truth_label_key = 'saliency'
                    truth_weight_key = 'saliency_weights'
                    start_idx = 0
                elif head_key == 'change':
                    truth_label_key = 'change'
                    truth_weight_key = 'change_weights'
                    start_idx = 1

                frame_head_losses = []

                for idx, frame in enumerate(item['frames'][start_idx:]):
                    frame_head_truth = frame[truth_label_key][None, None, ...]
                    frame_head_weights = frame[truth_weight_key][None, None, ...]
                    frame_head_logits = head_logits[0, idx][None, None, ...]

                    # HACK: The forward decoupled resolution logic should be
                    # cleaned up to avoid this step.  unnecessary computation.
                    # There is a resample we can skip in the above code.
                    h, w = frame_head_truth.shape[2:4]
                    frame_head_logits2 = einops.rearrange(nn.functional.interpolate(
                        einops.rearrange(frame_head_logits, 'b t h w c -> (b t) c h w'),
                        [h, w], mode='bilinear', align_corners=False), 'b c h w -> b 1 h w c')

                    head_loss = self._head_loss(head_key, frame_head_logits2, frame_head_truth, frame_head_weights)
                    frame_head_losses.append(head_loss)

                if frame_head_losses:
                    head_loss = sum(frame_head_losses)
                    item_loss_parts[head_key] = head_loss
        else:
            item_pixel_weights_list = {
                k: [] for k, v in self.global_head_weights.items() if v > 0
            }
            for frame_idx, frame in enumerate(item['frames']):
                if 'class' in item_pixel_weights_list:
                    item_pixel_weights_list['class'].append(frame['class_weights'])
                if 'saliency' in item_pixel_weights_list:
                    item_pixel_weights_list['saliency'].append(frame['saliency_weights'])
                if 'change' in item_pixel_weights_list:
                    if frame_idx > 0:
                        item_pixel_weights_list['change'].append(frame['change_weights'])

            # Stack the weights for each item
            item_weights = {
                # Because we are nt collating we need to add a batch dimension
                key: torch.stack(_tensors)[None, ...]
                for key, _tensors in item_pixel_weights_list.items()
            }
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

            # Compute criterion loss for each head
            for head_key, head_logits in resampled_logits.items():
                head_truth = item_truths[head_key]
                head_weights = item_weights[head_key]
                head_loss = self._head_loss(head_key, head_logits, head_truth, head_weights)
                item_loss_parts[head_key] = head_loss

        return item_loss_parts, item_truths

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
            >>> dpath = ub.Path.appdir('watch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> model = self = methods.MultimodalTransformer(
            >>>     arch_name="smt_it_joint_p2", input_channels=5,
            >>>     change_head_hidden=0, saliency_head_hidden=0,
            >>>     class_head_hidden=0)

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
            >>> dpath = ub.Path.appdir('watch/tests/package').ensuredir()
            >>> package_path = dpath / 'my_package.pt'

            >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral-multisensor', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=0)
            >>> datamodule.setup('fit')
            >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.MultimodalTransformer(
            >>>     arch_name="smt_it_joint_p2", classes=classes,
            >>>     dataset_stats=dataset_stats, input_channels=datamodule.input_channels,
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
        def _torch_package_monkeypatch():
            # Monkey Patch torch.package
            import sys
            if sys.version_info[0:2] >= (3, 10):
                try:
                    from torch.package import _stdlib
                    _stdlib._get_stdlib_modules = lambda: sys.stdlib_module_names
                except Exception:
                    pass
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
