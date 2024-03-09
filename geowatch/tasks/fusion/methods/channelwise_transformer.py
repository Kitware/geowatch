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
# from torchvision import transforms
from torch.optim import lr_scheduler
from geowatch import heuristics
from geowatch.tasks.fusion import utils
from geowatch.tasks.fusion.architectures import transformer
from geowatch.tasks.fusion.methods.network_modules import _torch_meshgrid
from geowatch.tasks.fusion.methods.network_modules import coerce_criterion
from geowatch.tasks.fusion.methods.network_modules import torch_safe_stack
from geowatch.tasks.fusion.methods.network_modules import RobustModuleDict
from geowatch.tasks.fusion.methods.network_modules import RobustParameterDict
from geowatch.tasks.fusion.methods.network_modules import RearrangeTokenizer
from geowatch.tasks.fusion.methods.network_modules import ConvTokenizer
from geowatch.tasks.fusion.methods.network_modules import LinearConvTokenizer
from geowatch.tasks.fusion.methods.network_modules import DWCNNTokenizer
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins

import scriptconfig as scfg

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# Model names define the transformer encoder used by the method
available_encoders = list(transformer.encoder_configs.keys()) + ['deit', "perceiver", 'vit']


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
    lr_scheduler = scfg.Value('CosineAnnealingLR', type=str)
    positive_change_weight = scfg.Value(1.0, type=float)
    negative_change_weight = scfg.Value(1.0, type=float)
    class_weights = scfg.Value('auto', type=str, help=ub.paragraph(
        '''
        class weighting strategy

        Can be auto or auto:<modulate_str>.
        A modulate string is a special syntax that lets the user modulate
        automatically computed class weights. Should be a comma separated list
        of name*weight or name*weight+offset. E.g.
        `negative*0,background*0.001,No Activity*0.1+1`
        '''))

    # TODO: better encoding
    saliency_weights = scfg.Value('auto', type=str, help='saliency weighting strategy. Can be None, "auto", or a string "<bg>:<fg>"')

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
    backbone_depth = scfg.Value(None, type=int, help='For supporting architectures, control the depth of the backbone. Default depends on arch_name')
    positional_dims = scfg.Value(6 * 8, type=int, help='positional feature dims')
    global_class_weight = scfg.Value(1.0, type=float)
    global_change_weight = scfg.Value(1.0, type=float)
    global_saliency_weight = scfg.Value(1.0, type=float)
    modulate_class_weights = scfg.Value('', type=str, help=ub.paragraph(
        '''
        DEPRECATE. SET THE class_weights to auto:<modulate_str>
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
        number of hidden layers in the CHANGE head. I.e. the depth of the head.
        '''))
    class_head_hidden = scfg.Value(2, type=int, help=ub.paragraph(
        '''
        number of hidden layers in the CLASS head. I.e. the depth of the head.
        '''))
    saliency_head_hidden = scfg.Value(2, type=int, help=ub.paragraph(
        '''
        number of hidden layers in the SALIENCY head. I.e. the depth of the head.
        '''))
    window_size = scfg.Value(8, type=int)
    squash_modes = scfg.Value(False, help='deprecated doesnt do anything')
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
    attention_kwargs = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Extra options for attention operations in the FusionModel. Including `add_zero_attn`.
        '''))
    multimodal_reduce = scfg.Value('max', help=ub.paragraph(
        '''
        operation used to combine multiple modes from the same timestep
        '''))
    rescale_nans = scfg.Value(None, help=ub.paragraph(
        '''
        Method used to rescale nan input values. Can be perframe or None.
        '''))
    ohem_ratio = scfg.Value(None, type=float, help=ub.paragraph(
            '''
            Ratio of hard examples to sample when computing loss. If None,
            then do not use OHEM.
            '''))
    focal_gamma = scfg.Value(2.0, type=float, help=ub.paragraph(
            '''
            Special parameter of focal loss. Can be applied to Focal and
            DiceFocal losses. Default: 2.0
            '''))

    perterb_scale = scfg.Value(0.0, type=float, help=ub.paragraph(
        '''
        If specified enables weight perterbation on every optimizer step.  This
        is the perterb part of shrink and perterb. The shrink part should be
        taken care of by weight decay.
        '''))

    continual_learning = scfg.Value(False, type=float, help=ub.paragraph(
        '''
        If True, attempt to enable experimental loss-of-plasticity generate and
        test algorithm to encourage continual learning.
        '''))

    predictable_classes = scfg.Value(None, help=ub.paragraph(
        '''
        Subset of classes to perform predictions on (for the class head).
        Specified as a comma delimited string.
        '''))

    def __post_init__(self):
        super().__post_init__()
        from kwutil.util_yaml import Yaml
        self.attention_kwargs = Yaml.coerce(self.attention_kwargs)


class MultimodalTransformer(pl.LightningModule, WatchModuleMixins):
    """
    CommandLine:
        xdoctest -m geowatch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer

    TODO:
        - [ ] Change name MultimodalTransformer -> FusionModel
        - [ ] Move parent module methods -> models

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/geowatch/tasks/fusion/methods/channelwise_transformer.py MultimodalTransformer

    Example:
        >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from geowatch.tasks.fusion import datamodules
        >>> print('(STEP 0): SETUP THE DATA MODULE')
        >>> datamodule = datamodules.KWCocoVideoDataModule(
        >>>     train_dataset='special:vidshapes-geowatch', num_workers=4, channels='auto')
        >>> datamodule.setup('fit')
        >>> dataset = datamodule.torch_datasets['train']
        >>> print('(STEP 1): ESTIMATE DATASET STATS')
        >>> dataset_stats = dataset.cached_dataset_stats(num=3)
        >>> print('dataset_stats = {}'.format(ub.urepr(dataset_stats, nl=3)))
        >>> loader = datamodule.train_dataloader()
        >>> print('(STEP 2): SAMPLE BATCH')
        >>> batch = next(iter(loader))
        >>> for item_idx, item in enumerate(batch):
        >>>     print(f'item_idx={item_idx}')
        >>>     item_summary = dataset.summarize_item(item)
        >>>     print('item_summary = {}'.format(ub.urepr(item_summary, nl=2)))
        >>> print('(STEP 3): THE REST OF THE TEST')
        >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
        >>> self = MultimodalTransformer(arch_name='smt_it_joint_p2',
        >>>                              dataset_stats=dataset_stats,
        >>>                              classes=datamodule.predictable_classes,
        >>>                              decoder='segmenter',
        >>>                              change_loss='dicefocal',
        >>>                              #attention_impl='performer'
        >>>                              attention_impl='exact'
        >>>                              )
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

    def get_cfgstr(self):
        cfgstr = f'{self.hparams.name}_{self.hparams.arch_name}'
        return cfgstr

    __scriptconfig__ = MultimodalTransformerConfig

    def __init__(self, classes=10, dataset_stats=None, input_sensorchan=None,
                 input_channels=None, **kwargs):
        """
        Example:
            >>> # Note: it is important that the non-kwargs are saved as hyperparams
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import MultimodalTransformer
            >>> self = model = MultimodalTransformer(arch_name="smt_it_joint_p2", input_sensorchan='r|g|b')
            >>> assert "classes" in model.hparams
            >>> assert "dataset_stats" in model.hparams
            >>> assert "input_sensorchan" in model.hparams
            >>> assert "tokenizer" in model.hparams
        """
        assert kwargs.pop('config', None) is None  # not sure why this is in the kwargs
        print('kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
        _config = MultimodalTransformerConfig(**kwargs)
        _cfgdict = _config.to_dict()
        assert _config.tokenizer in ['dwcnn', 'rearrange', 'conv7', 'linconv']
        assert _config.token_norm in ['none', 'auto', 'group', 'batch']
        assert _config.arch_name in available_encoders
        assert _config.decoder in ['mlp', 'segmenter']
        assert _config.attention_impl in ["exact", "performer", "reformer"]

        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(**_cfgdict)

        input_stats = self.set_dataset_specific_attributes(input_sensorchan, dataset_stats)

        input_norms = None
        if input_stats is not None:
            input_norms = RobustModuleDict()
            for s, c in sorted(self.unique_sensor_modes):
                if s not in input_norms:
                    input_norms[s] = RobustModuleDict()
                stats = input_stats.get((s, c), None)
                if stats is None:
                    input_norms[s][c] = nh.layers.InputNorm()
                else:
                    input_norms[s][c] = nh.layers.InputNorm(
                        **ub.udict(stats) & {'mean', 'std'})

            # Not sure what causes the format to change. Just hitting test
            # cases.
            for k, v in sorted(input_stats.items()):
                if isinstance(k, str):
                    for c, stats in v.items():
                        if s not in input_norms:
                            input_norms[s] = RobustModuleDict()
                        input_norms[s][c] = nh.layers.InputNorm(
                            **ub.udict(stats) & {'mean', 'std'})
                else:
                    # for (s, c), stats in input_stats.items():
                    s, c = k
                    stats = v
                    if s not in input_norms:
                        input_norms[s] = RobustModuleDict()
                    input_norms[s][c] = nh.layers.InputNorm(
                        **ub.udict(stats) & {'mean', 'std'})

        self.input_norms = input_norms

        self.predictable_classes = self.hparams.predictable_classes
        if self.predictable_classes is not None:
            self.predictable_classes = [x.strip() for x in self.hparams.predictable_classes.split(',')]
            self.classes = kwcoco.CategoryTree.coerce(self.predictable_classes)
            self.num_classes = len(self.predictable_classes)
        else:
            self.classes = kwcoco.CategoryTree.coerce(classes)
            self.num_classes = len(self.classes)

        self.global_class_weight = self.hparams.global_class_weight
        self.global_change_weight = self.hparams.global_change_weight
        self.global_saliency_weight = self.hparams.global_saliency_weight
        self.global_head_weights = {
            'class': self.hparams.global_class_weight,
            'change': self.hparams.global_change_weight,
            'saliency': self.hparams.global_saliency_weight,
        }

        # TODO: this data should be introspectable via the kwcoco file
        hueristic_background_keys = heuristics.BACKGROUND_CLASSES

        # FIXME: case sensitivity
        hueristic_ignore_keys = heuristics.IGNORE_CLASSNAMES
        if self.class_freq is not None:
            if self.predictable_classes is not None:
                all_keys = set(self.class_freq.keys()).intersection(self.predictable_classes)
            else:
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
        # ~/code/watch/geowatch/tasks/fusion/methods/channelwise_transformer.py
        # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        class_weights = self.hparams.class_weights
        modulate_class_weights = self.hparams.modulate_class_weights
        if modulate_class_weights:
            ub.schedule_deprecation(
                'geowatch', 'modulate_class_weights', 'param',
                migration='Use class_weights:<modulate_str> instead',
                deprecate='0.3.9', error='0.3.11', remove='0.3.13')
            if isinstance(class_weights, str) and class_weights == 'auto':
                class_weights = class_weights + ':' + modulate_class_weights

        print(f'self.hparams.saliency_weights={self.hparams.saliency_weights}')
        self.saliency_weights = self._coerce_saliency_weights(self.hparams.saliency_weights)
        self.class_weights = self._coerce_class_weights(class_weights)
        self.change_weights = torch.FloatTensor([
            self.hparams.negative_change_weight,
            self.hparams.positive_change_weight
        ])
        print(f'self.change_weights={self.change_weights}')

        if isinstance(self.hparams.stream_channels, str):
            RAW_CHANS = int(self.hparams.stream_channels.split(' ')[0])
        else:
            RAW_CHANS = None
        print(f'RAW_CHANS={RAW_CHANS}')
        MODAL_AGREEMENT_CHANS = self.hparams.stream_channels
        print(f'MODAL_AGREEMENT_CHANS={MODAL_AGREEMENT_CHANS}')
        print(f'self.hparams.tokenizer={self.hparams.tokenizer}')

        self.rescale_nan_method = self.hparams.rescale_nans
        if self.rescale_nan_method == 'perframe':
            ...
        elif self.rescale_nan_method is None:
            ...
        else:
            raise KeyError(f'Unknown: {self.rescale_nan_method}')

        self.tokenizer = self.hparams.tokenizer
        self.sensor_channel_tokenizers = RobustModuleDict()

        # Unique sensor modes obviously isn't very correct here.
        # We should fix that, but let's hack it so it at least
        # includes all sensor modes we probably will need.
        if input_stats is not None:
            sensor_modes = set(self.unique_sensor_modes) | set(input_stats.keys())
        else:
            sensor_modes = set(self.unique_sensor_modes)

        # import xdev
        # with xdev.embed_on_exception_context:
        for k in sorted(sensor_modes):
            if isinstance(k, str):
                if k == '*':
                    s = c = '*'
                else:
                    raise AssertionError
            else:
                s, c = k
            mode_code = kwcoco.FusedChannelSpec.coerce(c)
            # For each mode make a network that should learn to tokenize
            in_chan = mode_code.numel()
            if s not in self.sensor_channel_tokenizers:
                self.sensor_channel_tokenizers[s] = RobustModuleDict()

            if self.hparams.tokenizer == 'rearrange':
                if RAW_CHANS is None:
                    in_features_raw = MODAL_AGREEMENT_CHANS
                else:
                    in_features_raw = RAW_CHANS
                tokenize = RearrangeTokenizer(
                    in_channels=in_chan, agree=in_features_raw,
                    window_size=self.hparams.window_size,
                )
            elif self.hparams.tokenizer == 'conv7':
                # Hack for old models
                if RAW_CHANS is None:
                    in_features_raw = MODAL_AGREEMENT_CHANS
                else:
                    in_features_raw = RAW_CHANS
                tokenize = ConvTokenizer(in_chan, in_features_raw, norm=None)
            elif self.hparams.tokenizer == 'linconv':
                if RAW_CHANS is None:
                    in_features_raw = MODAL_AGREEMENT_CHANS * 64
                else:
                    in_features_raw = RAW_CHANS
                tokenize = LinearConvTokenizer(in_chan, in_features_raw)
            elif self.hparams.tokenizer == 'dwcnn':
                if RAW_CHANS is None:
                    in_features_raw = MODAL_AGREEMENT_CHANS * 64
                else:
                    in_features_raw = RAW_CHANS
                tokenize = DWCNNTokenizer(in_chan, in_features_raw, norm=self.hparams.token_norm)
            else:
                raise KeyError(self.hparams.tokenizer)

            self.sensor_channel_tokenizers[s][c] = tokenize
            in_features_raw = tokenize.out_channels

        print(f'in_features_raw={in_features_raw}')

        # for (s, c), stats in input_stats.items():
        #     self.sensor_channel_tokenizers[s][c] = tokenize

        in_features_pos = 6 * 8   # 6 positional features with 8 dims each (TODO: be robust)
        in_features = in_features_pos + in_features_raw
        self.in_features = in_features
        self.in_features_pos = in_features_pos
        self.in_features_raw = in_features_raw

        print(f'self.in_features={self.in_features}')
        ### NEW:
        # Learned positional encodings
        self.token_learner1_time_delta = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=1, hidden_channels=3, out_channels=8, residual=True, norm=None)
        self.token_learner2_sensor = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=16, hidden_channels=3, out_channels=8, residual=True, norm=None)
        self.token_learner3_mode = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=16, hidden_channels=3, out_channels=8, residual=True, norm=None)

        # 'https://rwightman.github.io/pytorch-image-models/models/vision-transformer/'
        if self.hparams.arch_name in transformer.encoder_configs:
            encoder_config = transformer.encoder_configs[self.hparams.arch_name]
            if self.hparams.backbone_depth is not None:
                raise NotImplementedError('unsupported')
            encoder = transformer.FusionEncoder(
                **encoder_config,
                in_features=in_features,
                attention_impl=self.hparams.attention_impl,
                attention_kwargs=self.hparams.attention_kwargs,
                dropout=self.hparams.dropout,
            )
            self.encoder = encoder
        elif self.hparams.arch_name.startswith('deit'):
            if self.hparams.backbone_depth is not None:
                raise ValueError('unsupported')
            self.encoder = transformer.DeiTEncoder(
                # **encoder_config,
                in_features=in_features,
                # attention_impl=attention_impl,
                # dropout=dropout,
            )
        elif self.hparams.arch_name.startswith('vit'):
            """
            Ignore:
                >>> # Note: it is important that the non-kwargs are saved as hyperparams
                >>> from geowatch.tasks.fusion.methods.channelwise_transformer import MultimodalTransformer
                >>> channels, classes, dataset_stats = MultimodalTransformer.demo_dataset_stats()
                >>> self = model = MultimodalTransformer(arch_name="vit", stream_channels='720 !', input_sensorchan=channels, classes=classes, dataset_stats=dataset_stats, tokenizer='linconv')
                >>> batch = self.demo_batch()
                >>> out = self.forward_step(batch)
            """
            self.encoder = transformer.MM_VITEncoder(
                # **encoder_config,
                # in_features=in_features,
                # attention_impl=attention_impl,
                # dropout=dropout,
            )
        elif self.hparams.arch_name.startswith('perceiver'):
            if self.hparams.backbone_depth is None:
                backbone_depth = 4
            self.encoder = transformer.PerceiverEncoder(
                depth=backbone_depth,
                # **encoder_config,
                in_features=in_features,
                # attention_impl=attention_impl,
                # dropout=dropout,
            )
        else:
            raise NotImplementedError

        if self.hparams.multimodal_reduce == 'learned_linear':
            # Make special params for the reduction.
            # The idea for the learned linear mode is to assign a
            # weight to each mode, and these are used to combine tokens
            # from different modes. Additionally, we include a special
            # parameter for the __MAX, allowing us to mixin the max
            # reduction method as something learnable by these
            # parameters.

            # Also add in a special weight for the max
            sensor_chan_reduction_weights = RobustParameterDict()
            unique_sensorchans = [
                sensor + ':' + chan
                for sensor, chan in self.unique_sensor_modes]
            for sensor_chan in unique_sensorchans + ['__MAX']:
                sensor_chan_reduction_weights[sensor_chan] = torch.nn.Parameter(torch.ones([1]))
            self.sensor_chan_reduction_weights = sensor_chan_reduction_weights

        feat_dim = self.encoder.out_features

        self.change_head_hidden = self.hparams.change_head_hidden
        self.class_head_hidden = self.hparams.class_head_hidden
        self.saliency_head_hidden = self.hparams.saliency_head_hidden

        self.class_loss = self.hparams.class_loss
        self.change_loss = self.hparams.change_loss
        self.saliency_loss = self.hparams.saliency_loss

        self.criterions = torch.nn.ModuleDict()
        self.heads = torch.nn.ModuleDict()

        head_properties = [
            {
                'name': 'change',
                'hidden': self.change_head_hidden,
                'channels': 2,
                'loss': self.hparams.change_loss,
                'weights': self.change_weights,
            },
            {
                'name': 'saliency',
                'hidden': self.saliency_head_hidden,
                'channels': self.saliency_num_classes,
                'loss': self.hparams.saliency_loss,
                'weights': self.saliency_weights,
            },
            {
                'name': 'class',
                'hidden': self.class_head_hidden,
                'channels': self.num_classes,
                'loss': self.hparams.class_loss,
                'weights': self.class_weights,
            },
        ]

        for prop in head_properties:
            head_name = prop['name']
            global_weight = self.global_head_weights[head_name]
            if global_weight > 0:
                self.criterions[head_name] = coerce_criterion(prop['loss'],
                                                              prop['weights'],
                                                              ohem_ratio=_config.ohem_ratio,
                                                              focal_gamma=_config.focal_gamma)
                if self.hparams.decoder == 'mlp':
                    self.heads[head_name] = nh.layers.MultiLayerPerceptronNd(
                        dim=0,
                        in_channels=feat_dim,
                        hidden_channels=prop['hidden'],
                        out_channels=prop['channels'],
                        norm=None
                    )
                elif self.hparams.decoder == 'segmenter':
                    from geowatch.tasks.fusion.architectures import segmenter_decoder
                    self.heads[head_name] = segmenter_decoder.MaskTransformerDecoder(
                        d_model=feat_dim,
                        n_layers=prop['hidden'],
                        n_cls=prop['channels'],
                    )
                else:
                    raise KeyError(self.hparams.decoder)

        FBetaScore = torchmetrics.FBetaScore
        self.head_metrics = nn.ModuleDict()
        self.head_metrics['class'] = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1_micro': FBetaScore(beta=1.0, threshold=0.5, average='micro', num_classes=self.num_classes, task='multiclass'),
            'f1_macro': FBetaScore(beta=1.0, threshold=0.5, average='macro', num_classes=self.num_classes, task='multiclass'),
        })
        self.head_metrics['change'] = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1': FBetaScore(beta=1.0, task='binary'),
        })
        self.head_metrics['saliency'] = nn.ModuleDict({
            'f1': FBetaScore(beta=1.0, task='binary'),
        })

        self.encode_h = utils.SinePositionalEncoding(3, 1, size=8)
        self.encode_w = utils.SinePositionalEncoding(3, 2, size=8)

        self.automatic_optimization = True

        if 0:
            ...

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Only required for backwards compatibility until lightning CLI
        is the primary entry point.

        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from geowatch.utils.configargparse_ext import ArgumentParser
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
        # init_kwargs = ub.compatible(config, cls.__init__)
        """
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

    def configure_optimizers(self):
        """
        TODO:
            - [ ] Enable use of other optimization algorithms on the CLI
            - [ ] Enable use of other scheduler algorithms on the CLI

        Note:
            Is this even called when using LightningCLI?
            Nope, the LightningCLI overwrites it.

        References:
            https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html

        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # noqa
            >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
            >>> disable_lightning_hardware_warnings()
            >>> self = MultimodalTransformer(arch_name="smt_it_joint_p2", input_sensorchan='r|g|b')
            >>> max_epochs = 80
            >>> self.trainer = pl.Trainer(max_epochs=max_epochs)
            >>> [opt], [sched] = self.configure_optimizers()
            >>> rows = []
            >>> # Insepct what the LR curve will look like
            >>> for _ in range(max_epochs):
            ...     sched.last_epoch += 1
            ...     lr = sched.get_last_lr()[0]
            ...     rows.append({'lr': lr, 'last_epoch': sched.last_epoch})
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> import pandas as pd
            >>> data = pd.DataFrame(rows)
            >>> sns = kwplot.autosns()
            >>> sns.lineplot(data=data, y='lr', x='last_epoch')

        Example:
            >>> # Verify lr and decay is set correctly
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> my_lr = 2.3e-5
            >>> my_decay = 2.3e-5
            >>> kw = dict(arch_name="smt_it_joint_p2", input_sensorchan='r|g|b', learning_rate=my_lr, weight_decay=my_decay)
            >>> self = MultimodalTransformer(**kw)
            >>> [opt], [sched] = self.configure_optimizers()
            >>> assert opt.param_groups[0]['lr'] == my_lr
            >>> assert opt.param_groups[0]['weight_decay'] == my_decay
            >>> #
            >>> self = MultimodalTransformer(**kw, optimizer='sgd')
            >>> [opt], [sched] = self.configure_optimizers()
            >>> assert opt.param_groups[0]['lr'] == my_lr
            >>> assert opt.param_groups[0]['weight_decay'] == my_decay
            >>> #
            >>> self = MultimodalTransformer(**kw, optimizer='AdamW')
            >>> [opt], [sched] = self.configure_optimizers()
            >>> assert opt.param_groups[0]['lr'] == my_lr
            >>> assert opt.param_groups[0]['weight_decay'] == my_decay
            >>> #
            >>> # self = MultimodalTransformer(**kw, optimizer='MADGRAD')
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
        # print('optim_cls = {}'.format(ub.urepr(optim_cls, nl=1)))
        # print('optim_kw = {}'.format(ub.urepr(optim_kw, nl=1)))
        optimizer = optim_cls(**optim_kw)

        # TODO:
        # - coerce schedulers
        if self.has_trainer:
            try:
                max_epochs = self.trainer.max_epochs
            except ReferenceError:
                max_epochs = 20
        else:
            max_epochs = 20

        if self.hparams.lr_scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs)
        elif self.hparams.lr_scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer, T_max=max_epochs,
                max_lr=self.hparams.learning_rate * 10)
        else:
            raise KeyError(self.hparams.lr_scheduler)

        # if self.hparams.continual_learning:
        #     raise NotImplementedError

        return [optimizer], [scheduler]

    def overfit(self, batch):
        """
        Overfit script and demo

        CommandLine:
            python -m xdoctest -m geowatch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer.overfit --overfit-demo

        Example:
            >>> # xdoctest: +REQUIRES(--overfit-demo)
            >>> # ============
            >>> # DEMO OVERFIT:
            >>> # ============
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> from geowatch.utils.util_data import find_dvc_dpath
            >>> import geowatch
            >>> import kwcoco
            >>> from os.path import join
            >>> import os
            >>> if 1:
            >>>     print('''
            ...     # Generate toy datasets
            ...     DATA_DPATH=$HOME/data/work/toy_change
            ...     TRAIN_FPATH=$DATA_DPATH/vidshapes_msi_train/data.kwcoco.json
            ...     mkdir -p "$DATA_DPATH"
            ...     kwcoco toydata --key=vidshapes-videos8-frames5-randgsize-speed0.2-msi-multisensor --bundle_dpath "$DATA_DPATH/vidshapes_msi_train" --verbose=5
            ...     ''')
            >>>     coco_fpath = ub.expandpath('$HOME/data/work/toy_change/vidshapes_msi_train/data.kwcoco.json')
            >>>     coco_fpath = 'vidshapes-videos8-frames5-randgsize-speed0.2-msi-multisensor'
            >>>     coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
            >>>     channels="B11,r|g|b,B1|B8|B11"
            >>> if 0:
            >>>     dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>>     coco_dset = (dvc_dpath / 'Drop4-BAS') / 'data_vali.kwcoco.json'
            >>>     channels='swir16|swir22|blue|green|red|nir'
            >>>     coco_dset = (dvc_dpath / 'Drop4-BAS') / 'combo_vali_I2.kwcoco.json'
            >>>     channels='blue|green|red|nir,invariants.0:17'
            >>> if 0:
            >>>     coco_dset = geowatch.demo.demo_kwcoco_multisensor(max_speed=0.5)
            >>>     # coco_dset = 'special:vidshapes8-frames9-speed0.5-multispectral'
            >>>     #channels='B1|B11|B8|r|g|b|gauss'
            >>>     channels='X.2|Y:2:6,B1|B8|B8a|B10|B11,r|g|b,disparity|gauss,flowx|flowy|distri'
            >>> coco_dset = kwcoco.CocoDataset.coerce(coco_dset)
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=coco_dset,
            >>>     chip_size=128, batch_size=1, time_steps=5,
            >>>     channels=channels,
            >>>     normalize_peritem='blue|green|red|nir',
            >>>     normalize_inputs=32, neg_to_pos_ratio=0,
            >>>     num_workers='avail/2',
            >>>     mask_low_quality=True,
            >>>     observable_threshold=0.6,
            >>>     use_grid_positives=False, use_centered_positives=True,
            >>> )
            >>> datamodule.setup('fit')
            >>> dataset = torch_dset = datamodule.torch_datasets['train']
            >>> torch_dset.disable_augmenter = True
            >>> dataset_stats = datamodule.dataset_stats
            >>> input_sensorchan = datamodule.input_sensorchan
            >>> classes = datamodule.classes
            >>> print('dataset_stats = {}'.format(ub.urepr(dataset_stats, nl=3)))
            >>> print('input_sensorchan = {}'.format(input_sensorchan))
            >>> print('classes = {}'.format(classes))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.MultimodalTransformer(
            >>>     # ===========
            >>>     # Backbone
            >>>     #arch_name='smt_it_joint_p2',
            >>>     arch_name='smt_it_stm_p8',
            >>>     stream_channels = 16,
            >>>     #arch_name='deit',
            >>>     optimizer='AdamW',
            >>>     learning_rate=1e-5,
            >>>     weight_decay=1e-3,
            >>>     #attention_impl='performer',
            >>>     attention_impl='exact',
            >>>     #decoder='segmenter',
            >>>     #saliency_head_hidden=4,
            >>>     decoder='mlp',
            >>>     change_loss='dicefocal',
            >>>     #class_loss='cce',
            >>>     class_loss='dicefocal',
            >>>     #saliency_loss='dicefocal',
            >>>     saliency_loss='focal',
            >>>     # ===========
            >>>     # Change Loss
            >>>     global_change_weight=1e-5,
            >>>     positive_change_weight=1.0,
            >>>     negative_change_weight=0.5,
            >>>     # ===========
            >>>     # Class Loss
            >>>     global_class_weight=1e-5,
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
            >>>     multimodal_reduce='learned_linear',
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
            >>> print(ub.urepr(dataset.summarize_item(batch[0]), nl=3))
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
        from kwutil.slugify_ext import smart_truncate
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
                outputs['item_losses']
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

    def prepare_item(self, item):
        pass

    @profile
    def forward_step(self, batch, with_loss=False, stage='unspecified'):
        """
        Generic forward step used for test / train / validation

        CommandLine:
            xdoctest -m geowatch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer.forward_step

        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> import geowatch
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes-geowatch',
            >>>     num_workers=0, chip_size=96, time_steps=4,
            >>>     normalize_inputs=8, neg_to_pos_ratio=0, batch_size=5,
            >>>     channels='auto',
            >>> )
            >>> datamodule.setup('fit')
            >>> train_dset = datamodule.torch_datasets['train']
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))
            >>> # Test with "failed samples"
            >>> batch[0] = None
            >>> batch[2] = None
            >>> batch[3] = None
            >>> batch[4] = None
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = model = methods.MultimodalTransformer(
            >>>     arch_name='smt_it_joint_p8', tokenizer='rearrange',
            >>>     decoder='segmenter',
            >>>     dataset_stats=datamodule.dataset_stats, global_saliency_weight=1.0, global_change_weight=1.0, global_class_weight=1.0,
            >>>     classes=datamodule.predictable_classes, input_sensorchan=datamodule.input_sensorchan)
            >>> with_loss = True
            >>> outputs = self.forward_step(batch, with_loss=with_loss)
            >>> canvas = datamodule.draw_batch(batch, outputs=outputs, max_items=3, overlay_on_image=False)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, classes, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='segmenter', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch()
            >>> outputs = self.forward_step(batch, with_loss=True)
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> print(nh.data.collate._debug_inbatch_shapes(outputs))

        Example:
            >>> # Test learned_linear multimodal reduce
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, classes, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels, multimodal_reduce='learned_linear')
            >>> batch = self.demo_batch()
            >>> outputs = self.forward_step(batch, with_loss=True)
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> print(nh.data.collate._debug_inbatch_shapes(outputs))
            >>> # outputs['loss'].backward()
        """
        outputs = {}

        item_losses = []

        batch_size = len(batch)

        # Initialize truth and probs for each head / item that will be stacked
        batch_head_truths = {k: [] for k in self.heads.keys()}
        batch_head_probs = {k: [] for k in self.heads.keys()}
        skip_flags = []
        for item in batch:
            # Skip
            if item is None:
                skip_flags.append(True)
                probs, item_loss_parts, item_truths = {}, {}, {}
            else:
                skip_flags.append(False)
                probs, item_loss_parts, item_truths = self.forward_item(item, with_loss=with_loss)

            # with xdev.embed_on_exception_context:
            if with_loss:
                item_losses.append(item_loss_parts)
                if not self.hparams.decouple_resolution:
                    # TODO: fixme decouple_res
                    for head_name, head_truths in batch_head_truths.items():
                        head_truths.append(item_truths.get(head_name, None))

            # Append the item result to the batch outputs
            for head_name, head_probs in batch_head_probs.items():
                head_probs.append(probs.get(head_name, None))

        if all(skip_flags):
            return None

        if 'change' in batch_head_probs:
            outputs['change_probs'] = batch_head_probs['change']
        if 'class' in batch_head_probs:
            outputs['class_probs'] = batch_head_probs['class']
        if 'saliency' in batch_head_probs:
            outputs['saliency_probs'] = batch_head_probs['saliency']

        # print(f'with_loss={with_loss}')
        if with_loss:

            total_loss = sum(
                val for parts in item_losses for val in parts.values()
            )

            if not self.hparams.decouple_resolution:
                # TODO: fixme decouple_res
                to_compare = {}
                # compute metrics
                if self.has_trainer:
                    item_metrics = {}
                    ENABLE_METRICS = 0
                    if ENABLE_METRICS:
                        valid_batch_head_probs = {
                            head_name: [p for p in head_values if p is not None]
                            for head_name, head_values in batch_head_probs.items()
                        }
                        valid_batch_head_truths = {
                            head_name: [p for p in head_values if p is not None]
                            for head_name, head_values in batch_head_truths.items()
                        }
                        # Flatten everything for pixelwise comparisons
                        if 'change' in valid_batch_head_truths:
                            _true = torch.cat([x.contiguous().view(-1) for x in valid_batch_head_truths['change']], dim=0)
                            _pred = torch.cat([x.contiguous().view(-1) for x in valid_batch_head_probs['change']], dim=0)
                            to_compare['change'] = (_true, _pred)

                        if 'class' in valid_batch_head_truths:
                            c = self.num_classes
                            # Truth is index-based (todo: per class binary maps)
                            _true = torch.cat([x.contiguous().view(-1) for x in valid_batch_head_truths['class']], dim=0)
                            _pred = torch.cat([x.contiguous().view(-1, c) for x in valid_batch_head_probs['class']], dim=0)
                            to_compare['class'] = (_true, _pred)

                        if 'saliency' in valid_batch_head_truths:
                            c = self.saliency_num_classes
                            _true = torch.cat([x.contiguous().view(-1) for x in valid_batch_head_truths['saliency']], dim=0)
                            _pred = torch.cat([x.contiguous().view(-1, c) for x in valid_batch_head_probs['saliency']], dim=0)
                            to_compare['saliency'] = (_true, _pred)
                        for head_key in to_compare.keys():
                            _true, _pred = to_compare[head_key]
                            # Dont log unless a trainer is attached
                            for metric_key, metric in self.head_metrics[head_key].items():
                                val = metric(_pred, _true)
                                item_metrics[f'{stage}_{head_key}_{metric_key}'] = val

                    head_to_loss = ub.ddict(float)
                    for row in item_losses:
                        for head_key, v in row.items():
                            head_to_loss[head_key] += v

                    for head_key, val in head_to_loss.items():
                        self.log(f'{stage}_{head_key}_loss', val, prog_bar=False, batch_size=batch_size)

                    for key, val in item_metrics.items():
                        self.log(key, val, prog_bar=False, batch_size=batch_size)

                    self.log(f'{stage}_loss', total_loss, prog_bar=True, batch_size=batch_size)

                # Detach the itemized losses
                for _path, val in ub.IndexableWalker(item_losses):
                    if isinstance(val, torch.Tensor):
                        val.detach_()

            outputs['loss'] = total_loss
            outputs['item_losses'] = item_losses

        return outputs

    # def log(self, key, val, *args, **kwargs):
    #     print(f'Logging {key}={val}')
    #     super().log(key, val, *args, **kwargs)

    def forward_item(self, item, with_loss=False):
        """
        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, classes, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='segmenter', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> item = self.demo_batch(width=64, height=65)[0]
            >>> outputs = self.forward_item(item, with_loss=True)
            >>> print('item')
            >>> print(nh.data.collate._debug_inbatch_shapes(item))
            >>> print('outputs')
            >>> print(nh.data.collate._debug_inbatch_shapes(outputs))

        Example:
            >>> # Decoupled resolutions
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, classes, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=classes, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels, decouple_resolution=True)
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
        DECOUPLED_COORDINATE_ATTENTION = self.hparams.decouple_resolution
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
            print('tokenized = {}'.format(ub.urepr(tokenized, nl=1)))
            print('item = {}'.format(ub.urepr(item, nl=1)))
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
            if len(to_stack) == 1:
                frame_feat = to_stack[0]
            else:
                if self.hparams.multimodal_reduce == 'max':
                    stack = torch.stack(to_stack, dim=0)
                    frame_feat = torch.max(stack, dim=0)[0]
                elif self.hparams.multimodal_reduce == 'mean':
                    stack = torch.stack(to_stack, dim=0)
                    frame_feat = torch.mean(stack, dim=0)[0]
                elif self.hparams.multimodal_reduce == 'learned_linear':
                    sensor_chans = [g['sensor'] + ':' + g['chan_code'] for g in frame_group]
                    if 1:
                        # Add in the special max mode
                        to_stack.append(torch.max(torch.stack(to_stack, dim=0), dim=0)[0])
                        sensor_chans.append('__MAX')
                    linear_weights = [self.sensor_chan_reduction_weights[sc] for sc in sensor_chans]
                    linear_weights = torch.stack(linear_weights, dim=0)

                    # Normalize the weights
                    norm_weights = torch.nn.functional.softmax(linear_weights, dim=0)
                    # Take the linear combination of the modes
                    stack = torch.stack(to_stack, dim=0)
                    frame_feat = torch.einsum('m k, m k f -> k f', norm_weights, stack)
                    # frame_feat2 = (norm_weights[:, :, None] * stack).sum(dim=0)
                else:
                    raise Exception(self.hparams.multimodal_reduce)
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

        if not self.hparams.decouple_resolution:
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
                criterion_encoding = self.criterions["class"].target_encoding
                logits = resampled_logits['class'].detach()
                if criterion_encoding == "onehot":
                    probs['class'] = logits.sigmoid()[0]
                elif criterion_encoding == "index":
                    probs['class'] = logits.softmax(dim=-1)[0]
                else:
                    raise NotImplementedError
            if 'saliency' in resampled_logits:
                probs['saliency'] = resampled_logits['saliency'].detach().sigmoid()[0]
        else:
            # For class / saliency frames are indepenent
            perframe_logits = ub.ddict(list)
            for frame_feature in perframe_stackable_encodings:
                for head_key in ['class', 'saliency']:
                    if head_key in self.heads:
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
                criterion_encoding = self.criterions["class"].target_encoding
                logits = resampled_logits['class'].detach()
                if criterion_encoding == "onehot":
                    probs['class'] = logits.sigmoid()[0]
                elif criterion_encoding == "index":
                    probs['class'] = logits.softmax(dim=-1)[0]
                else:
                    raise NotImplementedError
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
                raise
            # self.sensor_channel_tokenizers[]

        # After input normalization happens, replace nans with zeros
        # which is effectively imputing the dataset mean

        try:
            rescale_nan_method = self.rescale_nan_method
        except AttributeError:
            rescale_nan_method = None

        if rescale_nan_method is None:
            mode_val = mode_val.nan_to_num_()
        elif rescale_nan_method == 'perframe':
            # Do a dropout-like rescaling to the nan input values.
            with torch.no_grad():
                num_nan = mode_val.isnan().sum()
                num_total = mode_val.numel()
                # dont rescale by more than half.
                p = min((num_nan / num_total), 0.5)
                mode_val = mode_val.nan_to_num_()
                rescale_factor = 1 / (1 - p)
                mode_val *= rescale_factor
        else:
            raise AssertionError

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

    def _head_loss(self, head_key, head_logits, head_truth, head_weights, head_encoding):
        criterion = self.criterions[head_key]
        global_head_weight = self.global_head_weights[head_key]

        HACK = 1
        if HACK:
            head_truth, head_weights = slice_to_agree(head_truth, head_weights, axes=[0, 1, 2, 3])
            head_truth, head_logits = slice_to_agree(head_truth, head_logits, axes=[0, 1, 2, 3])
            head_weights, head_logits = slice_to_agree(head_weights, head_logits, axes=[0, 1, 2, 3])

        head_pred_input = einops.rearrange(head_logits, 'b t h w c -> ' + criterion.logit_shape).contiguous()
        head_weights_input = einops.rearrange(head_weights[..., None], 'b t h w c -> ' + criterion.logit_shape).contiguous()

        if criterion.target_encoding == 'index':
            head_true_idxs = head_truth.long()
            head_true_input = einops.rearrange(head_true_idxs, 'b t h w -> ' + criterion.target_shape).contiguous()
            head_weights_input = head_weights_input[:, 0]
        elif criterion.target_encoding == 'onehot':
            # Note: 1HE is much easier to work with
            if head_encoding == 'index':
                head_true_ohe = kwarray.one_hot_embedding(head_truth.long(), criterion.in_channels, dim=-1)
            elif head_encoding == 'ohe':
                head_true_ohe = head_truth
            else:
                raise KeyError(head_encoding)
            head_true_input = einops.rearrange(head_true_ohe, 'b t h w c -> ' + criterion.target_shape).contiguous()
        else:
            raise KeyError(criterion.target_encoding)

        unreduced_head_loss = criterion(head_pred_input, head_true_input)

        full_head_weight = torch.broadcast_to(head_weights_input, unreduced_head_loss.shape)
        # Weighted reduction
        EPS_F32 = 1.1920929e-07
        weighted_head_loss = (full_head_weight * unreduced_head_loss).sum() / (full_head_weight.sum() + EPS_F32)
        head_loss = global_head_weight * weighted_head_loss

        return head_loss

    def _build_item_loss_parts(self, item, resampled_logits):
        item_loss_parts = {}
        item_truths = {}
        item_encoding = {}
        if self.hparams.decouple_resolution:
            for head_key, head_logits in resampled_logits.items():

                if head_key == 'class':
                    truth_encoding = 'index'
                    # TODO: prefer class-ohe if available
                    truth_label_key = 'class_idxs'
                    truth_weight_key = 'class_weights'
                    start_idx = 0
                elif head_key == 'saliency':
                    truth_encoding = 'index'
                    truth_label_key = 'saliency'
                    truth_weight_key = 'saliency_weights'
                    start_idx = 0
                elif head_key == 'change':
                    truth_encoding = 'index'
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

                    head_loss = self._head_loss(head_key, frame_head_logits2, frame_head_truth, frame_head_weights, truth_encoding)
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
                key: torch_safe_stack(_tensors, item_shape=[0, 0])[None, ...]
                for key, _tensors in item_pixel_weights_list.items()
            }
            if self.global_head_weights['change']:
                item_encoding['change'] = 'index'
                # [B, T, H, W]
                item_truths['change'] = torch_safe_stack([
                    frame['change'] for frame in item['frames'][1:]
                ], item_shape=[0, 0])[None, ...]

            if self.global_head_weights['class']:
                criterion_encoding = self.criterions["class"].target_encoding
                if criterion_encoding == "onehot":
                    item_encoding['class'] = 'ohe'
                    # [B, C, T, H, W]
                    item_truths['class'] = torch.stack([
                        frame['class_ohe'] for frame in item['frames']
                    ])[None, ...]
                elif criterion_encoding == "index":
                    item_encoding['class'] = 'index'
                    # [B, T, H, W]
                    item_truths['class'] = torch_safe_stack([
                        frame['class_idxs'] for frame in item['frames']
                    ])[None, ...]
                else:
                    raise NotImplementedError

            if self.global_head_weights['saliency']:
                item_encoding['saliency'] = 'index'
                item_truths['saliency'] = torch.stack([
                    frame['saliency'] for frame in item['frames']
                ])[None, ...]

            # Compute criterion loss for each head
            for head_key, head_logits in resampled_logits.items():
                head_truth = item_truths[head_key]
                head_truth_encoding = item_encoding[head_key]
                head_weights = item_weights[head_key]
                head_loss = self._head_loss(head_key, head_logits, head_truth, head_weights, head_truth_encoding)
                item_loss_parts[head_key] = head_loss

        return item_loss_parts, item_truths

    @profile
    def training_step(self, batch, batch_idx=None):

        if not self.automatic_optimization:
            # Do we have to do this ourselves?
            # https://lightning.ai/docs/pytorch/stable/common/optimization.html
            opt = self.optimizers()
            opt.zero_grad()

        outputs = self.forward_step(batch, with_loss=True, stage='train')

        if not self.automatic_optimization:
            loss = outputs['loss']
            self.manual_backwards(loss)

        return outputs

    def optimizer_step(self, *args, **kwargs):
        # function hook in LightningModule
        ret = super().optimizer_step(*args, **kwargs)
        optimizer = kwargs.get('optimizer')
        if optimizer is None:
            if len(args) >= 3:
                if isinstance(args[2], torch.optim.Optimizer):
                    optimizer = args[2]
        assert optimizer is not None
        self.parameter_hacking(optimizer)
        return ret

    def parameter_hacking(self, optimizer):
        if self.hparams.continual_learning:
            if not hasattr(self.trainer, 'gnt'):
                import geowatch_tpl
                geowatch_tpl.import_submodule('torchview')
                geowatch_tpl.import_submodule('lop')

                # See:
                # ~/code/watch/geowatch_tpl/submodules/loss-of-plasticity/lop/algos/gen_and_test.py
                # ~/code/watch/geowatch_tpl/submodules/torchview/torchview/torchview.py
                from lop.algos.gen_and_test import GenerateAndTest
                print('Setup gen and test')
                input_data = self.demo_batch(new_mode_sample=1)
                gnt = GenerateAndTest(self, 'relu', optimizer, input_data,
                                      device=self.main_device)
                # Give gnt to the trainer and dont do anything on the first
                # pass
                self.trainer.gnt = gnt
                # optimizer.gnt = gnt
            else:
                # On a subsequent passes start using it.
                assert self.training
                gnt = self.trainer.gnt
                # print('START GEN AND TESTING')
                gnt.gen_and_test()
                # print('DID GEN AND TESTING')

        if self.hparams.perterb_scale > 0:
            # Add a small bit of noise to every parameter
            with torch.no_grad():
                std = self.hparams.perterb_scale
                perterb_params(optimizer, std)

    @profile
    def validation_step(self, batch, batch_idx=None):
        # print('VALI STEP')
        outputs = self.forward_step(batch, with_loss=True, stage='val')
        return outputs

    @profile
    def test_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='test')
        return outputs

    def save_package(self, package_path, verbose=1):
        """

        CommandLine:
            xdoctest -m geowatch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> model = self = methods.MultimodalTransformer(
            >>>     arch_name="smt_it_joint_p2", input_sensorchan=5,
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
        """
        from kwutil import util_environ
        if self.hparams.continual_learning and not util_environ.envflag('HACK_SAVE_ANYWAY'):
            print('HACK NOT SAVING FOR CONTINUAL LEARNING')
            # HACK
            return
        self._save_package(package_path, verbose=verbose)

    @profile
    def forward(self, batch):
        """
        Example:
            >>> import pytest
            >>> pytest.skip('not currently used')
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from geowatch.tasks.fusion import datamodules
            >>> channels = 'B1,B8|B8a,B10|B11'
            >>> channels = 'B1|B8|B10|B8a|B11'
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0, channels=channels)
            >>> datamodule.setup('fit')
            >>> train_dataset = datamodule.torch_datasets['train']
            >>> dataset_stats = train_dataset.cached_dataset_stats()
            >>> loader = datamodule.train_dataloader()
            >>> tokenizer = 'convexpt-v1'
            >>> tokenizer = 'dwcnn'
            >>> batch = next(iter(loader))
            >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_joint_p8',
            >>>     dataset_stats=dataset_stats,
            >>>     change_loss='dicefocal',
            >>>     decoder='dicefocal',
            >>>     attention_impl='performer',
            >>>     tokenizer=tokenizer,
            >>> )
            >>> #images = torch.stack([ub.peek(f['modes'].values()) for f in batch[0]['frames']])[None, :]
            >>> #images.shape
            >>> #self.forward(images)
        """
        with_loss = self.training
        return self.forward_step(batch, with_loss=with_loss)
        # raise NotImplementedError('see forward_step instad')


def slice_to_agree(a1, a2, axes=None):
    """
    Example:
        from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        a1 = np.random.rand(3, 5, 7, 9, 3)
        a2 = np.random.rand(3, 5, 6, 9, 3)
        b1, b2 = slice_to_agree(a1, a2)
        print(f'{a1.shape=} {a2.shape=}')
        print(f'{b1.shape=} {b2.shape=}')

        a1 = np.random.rand(3, 5, 7, 9, 1)
        a2 = np.random.rand(3, 1, 6, 9, 3)
        b1, b2 = slice_to_agree(a1, a2, axes=[0, 1, 2, 3])
        print(f'{a1.shape=} {a2.shape=}')
        print(f'{b1.shape=} {b2.shape=}')
    """
    if a1.shape != a2.shape:
        shape1 = a1.shape
        shape2 = a2.shape
        if axes is not None:
            shape1 = [shape1[i] for i in axes]
            shape2 = [shape2[i] for i in axes]
        # ndim1 = len(shape1)
        # ndim2 = len(shape2)
        min_shape = np.minimum(np.array(shape1), np.array(shape2))
        sl = tuple([slice(0, s) for s in min_shape])
        a1 = a1[sl]
        a2 = a2[sl]
    return a1, a2


def perterb_params(optimizer, std):
    """
    Given an optimizer, perterb all parameters with Gaussian noise

    From: [ShrinkAndPerterb]_.

    While the presented conventional approaches do not remedy the warm-start
    problem, we have identified a remarkably simple trick that efficiently
    closes the generalization gap. At each round of training t, when new
    samples are appended to the training set, we propose initializing the
    network’s parameters by shrinking the weights found in the previous round
    of optimization towards zero, then adding a small amount of parameter
    noise.

    Specifically, we initialize each learnable parameter

    Math:

        θ[i, t] = λ * θ[i, t - 1] + p[t]

        where p[t] ∼ N (0, (σ ** 2)) and 0 < λ < 1.


    References:
        .. [ShrinkAndPerterb] https://arxiv.org/pdf/1910.08475.pdf
    """
    # Gether unique parameters
    id_to_params = {}
    for param_group in optimizer.param_groups:
        id_to_params.update({
            id(param): param
            for param in param_group['params']
        })
    for param in id_to_params.values():
        param += torch.empty(
            param.shape, device=param.device).normal_(
                mean=0, std=std)
