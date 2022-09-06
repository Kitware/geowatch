import watch
watch.__version__

import itertools as it
from types import MethodType
from functools import partial
import pathlib

import einops
import kwarray
import kwcoco
import ubelt as ub
import torch
import torchmetrics
import torch_optimizer
# import math

import numpy as np
import netharn as nh
import pytorch_lightning as pl
import perceiver_pytorch as perceiver

# import torch_optimizer as optim
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import transforms
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
    
        
def sanitize_key(key):
    return key.replace(".", "-")
        
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_steps, max_freq=10.0):
        super().__init__()
        self.scales = nn.Parameter(torch.pi * torch.linspace(1., max_freq, num_steps), requires_grad=False)
    
    def forward(self, x):
        orig_x = x
        x = torch.einsum("xhw,s->xshw", x, self.scales.type_as(x))
        x = einops.rearrange(x, "x s h w -> (x s) h w")
        return torch.concat([x.sin(), x.cos(), orig_x], dim=0)
    
    
@scfg.dataconf
class SequenceAwareModelConfig(scfg.DataConfig):
    """
    Arguments accepted by the SequenceAwareModel

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
    # arch_name = scfg.Value('smt_it_joint_p8', type=str, choices=available_encoders)
    decoder = scfg.Value('mlp', type=str, choices=['mlp', 'segmenter'])
    dropout = scfg.Value(0.1, type=float)
    backbone_depth = scfg.Value(None, type=int, help='For supporting architectures, control the depth of the backbone. Default depends on arch_name')
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
    multimodal_reduce = scfg.Value('max', help=ub.paragraph(
        '''
        operation used to combine multiple modes from the same timestep
        '''))
    

class SequenceAwareModel(pl.LightningModule):
    
    _HANDLES_NANS = True

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Example:
            >>> from watch.tasks.fusion.methods.sequence_aware import *  # NOQA
            >>> from watch.utils.configargparse_ext import ArgumentParser
            >>> cls = SequenceAwareModel
            >>> parent_parser = ArgumentParser(formatter_class='defaults')
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> parent_parser.parse_known_args()

            print(scfg.Config.port_argparse(parent_parser, style='dataconf'))
        """
        parser = parent_parser.add_argument_group('kwcoco_video_data')
        config = SequenceAwareModelConfig()
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
        cfgstr = f'{self.name}_{self.arch_name}_SA'
        return cfgstr

    def reset_weights(self):
        for name, mod in self.named_modules():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()
    

    def __init__(self, *, classes=10, dataset_stats=None,
                 input_sensorchan=None, input_channels=None, **kwargs):
        
        # =================================================================================
        # =================================================================================
        # START IMPORT FROM MULTIMODAL-TRANSFORMER
        
        super().__init__()
        config = SequenceAwareModelConfig(**kwargs)
        self.config = config
        cfgdict = self.config.to_dict()
        # Note:
        # it is important that the non-kwargs are saved as hyperparams:
        cfgdict['classes'] = classes
        cfgdict['dataset_stats'] = dataset_stats
        cfgdict['input_sensorchan'] = input_sensorchan
        cfgdict['input_channels'] = input_channels
        self.save_hyperparameters(cfgdict)
        # Backwards compatibility. Previous iterations had the
        # config saved directly as datamodule arguments
        self.__dict__.update(cfgdict)
        

        #####
        ## TODO: ALL OF THESE CONFIGURATIONS VARS SHOULD BE
        ## CONSOLIDATED. REMOVE DUPLICATES BETWEEN INSTANCE VARS
        ## HPARAMS, CONFIG.... It is unclear what the single source of truth
        ## is, and what needs to be modified if making changes.

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
        # arch_name = config['arch_name']
        dropout = config['dropout']
        attention_impl = config['attention_impl']
        global_class_weight = config['global_class_weight']
        global_change_weight = config['global_change_weight']
        global_saliency_weight = config['global_saliency_weight']

        # Moving towards sensror-channels everywhere so we always know what
        # sensor we are dealing with.
        if input_channels is not None:
            ub.schedule_deprecation(
                'watch', name='input_channels', type='model param',
                deprecate='0.3.3', migration='user input_sensorchan instead'
            )
            if input_sensorchan is None:
                input_sensorchan = input_channels
            else:
                raise AssertionError(
                    'cant specify both input_channels and input_sensorchan')

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

        input_norms = None
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

        self.input_norms = input_norms

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

            # self.sensor_channel_tokenizers[s][c] = tokenize
            key = sanitize_key(str((s, c)))
            self.sensor_channel_tokenizers[key] = nn.Sequential(
                self.input_norms[s][c],
                tokenize,
            )
            in_features_raw = tokenize.out_channels

        # for (s, c), stats in input_stats.items():
        #     self.sensor_channel_tokenizers[s][c] = tokenize

        in_features_pos = 64 # 6 * 8   # 6 positional features with 8 dims each (TODO: be robust)
        in_features = in_features_pos + in_features_raw
        self.in_features = in_features
        self.in_features_pos = in_features_pos
        self.in_features_raw = in_features_raw
        
        # END IMPORT FROM MULTIMODAL-TRANSFORMER
        # =================================================================================
        # =================================================================================
        
        self.positional_encoders = nn.ModuleDict({
            sanitize_key(str(key)): nn.Sequential(
                FourierPositionalEncoding(64, 60),
                nn.Conv2d(
                    3 + (3*64*2),
                    in_features_pos, 
                    1,
                ),
            )
            for key in list(dataset_stats["unique_sensor_modes"]) + ["change", "saliency", "class"]
        })
        
        self.perceiver = perceiver.PerceiverIO(
            depth = 4,
            dim = in_features,
            queries_dim = in_features_pos,
            num_latents = 512,
            latent_dim = 128,
            cross_heads = 1,
            latent_heads = 8,
            cross_dim_head = 64,
            latent_dim_head = 64,
            weight_tie_layers = True,
            decoder_ff = True,
            logits_dim = in_features,
        )
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
                        n_layers=prop['hidden'],
                        n_cls=prop['channels'],
                    )
                else:
                    raise KeyError(self.decoder)

        if hasattr(torchmetrics, 'FBetaScore'):
            FBetaScore = torchmetrics.FBetaScore
        else:
            FBetaScore = torchmetrics.FBeta

        head_metrics = dict()
        head_metrics['class'] = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1_micro': FBetaScore(beta=1.0, threshold=0.5, average='micro'),
            'f1_macro': FBetaScore(beta=1.0, threshold=0.5, average='macro', num_classes=self.num_classes),
        })
        head_metrics['change'] = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1': FBetaScore(beta=1.0),
        })
        head_metrics['saliency'] = torchmetrics.MetricCollection({
            'f1': FBetaScore(beta=1.0),
        })
        
        self.head_metrics = {
            stage: {
                key: collection.clone(prefix=f"{stage}_")
                for key, collection in head_metrics.items()
            }
            for stage in ["train", "val", "test"]
        }

    @property
    def has_trainer(self):
        try:
            # Lightning 1.7 raises an attribute error if not attached
            return self.trainer is not None
        except RuntimeError:
            return False
    
    def stem_process_example(self, example):
        modes = []
        for frame in example["frames"]:
            for mode_key, mode_image in frame["modes"].items():
                
                sensor_mode_key = sanitize_key(str((frame["sensor"], mode_key)))
                dtype=mode_image.dtype
                device=mode_image.device
                
                stemmed_mode = self.sensor_channel_tokenizers[sensor_mode_key](torch.nan_to_num(mode_image, 0)[None])[0]
                
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
                (frame["time_index"] * torch.ones(*frame["target_dims"], dtype=dtype, device=device),) +
                torch.meshgrid(
                    torch.linspace(-1, 1, frame["target_dims"][0], dtype=dtype, device=device),
                    torch.linspace(-1, 1, frame["target_dims"][1], dtype=dtype, device=device),
                ),
            ))
            for frame in example["frames"]
        ]
    
    def process_example(self, example, force_dropout=False):
                
        # process and stem frames and positions
        inputs = self.stem_process_example(example)
        inputs = torch.concat([einops.rearrange(x, "c h w -> (h w) c") for x in inputs], dim=0)
        
        outputs = {}
        task_defs = [
            ("change", "change_weights"),
            ("saliency", "saliency_weights"),
            ("class_idxs", "class_weights"),
        ]
        for task_name, weights_name in task_defs:
            
            labels = [
                frame[task_name] if (frame[task_name] != None) 
                else torch.zeros(
                    frame["target_dims"],
                    dtype=torch.int32,
                    device=inputs.device)
                for frame in example["frames"]
            ]
            labels = torch.concat([einops.rearrange(x, "h w -> (h w)") for x in labels], dim=0)
            weights = [
                frame[weights_name] if (frame[weights_name] != None) 
                else torch.zeros(
                    frame["target_dims"],
                    dtype=inputs.dtype,
                    device=inputs.device)
                for frame in example["frames"]
            ]
            weights = torch.concat([einops.rearrange(x, "h w -> (h w)") for x in weights], dim=0)
            
            if task_name == "class_idxs": task_name = "class"
            pos_enc = self.encode_query_position(example, task_name, inputs.dtype, inputs.device)
            pos_enc = torch.concat([einops.rearrange(x, "c h w -> (h w) c") for x in pos_enc], dim=0)
            
            # determine valid label locations
            valid_mask = weights > 0.0
            
            # TODO: if training, augment mask to dropout querys and labels following some strategy
            
            # produce model outputs for task
            pos_enc = pos_enc[valid_mask]
            labels = labels[valid_mask]
            weights = weights[valid_mask]
            
            if self.training:
                keep_inds = torch.randperm(weights.shape[0])[:200]
                pos_enc = pos_enc[keep_inds]
                labels = labels[keep_inds]
                weights = weights[keep_inds]
            
            outputs[task_name] = {
                "labels": labels, 
                "weights": weights, 
                "pos_enc": pos_enc, 
                "mask": valid_mask, 
                "shape": [frame["target_dims"] for frame in example["frames"]],
            }
        
        return inputs, outputs
    
    def reconstruct_output(self, output, mask, shapes):
        big_canvas = torch.nan * torch.zeros(mask.shape[0], output.shape[-1], dtype=output.dtype, device=output.device)
        big_canvas[mask] = output[:mask.sum()]

        canvases = []
        for canvas, shape in zip(torch.split(big_canvas, [w*h for w,h in shapes]), shapes):
            canvas = canvas.reshape(shape+[output.shape[-1],])
            canvases.append(canvas)
        return canvases
    
    def forward(self, inputs, queries, input_mask=None):
        context = self.perceiver(inputs, mask=input_mask)
        # print("context", context)
        
        outputs = {}
        for task_name in queries.keys():
            task_tokens = self.perceiver.decoder_cross_attn(
                queries[task_name], 
                context = context)
            task_logits = self.heads[task_name](task_tokens)
            task_logits = einops.rearrange(task_logits, "batch seq chan -> batch chan seq")
            outputs[task_name] = task_logits
            
        return outputs
    
    def shared_step(self, batch, batch_idx=None, stage="train"):
        losses = []
        metrics = {}
        
        inputs, outputs = zip(*[
            self.process_example(example)
            for example in batch
        ])
        
        padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=-1000.0)
        padded_valids = (padded_inputs[...,0] > -1000.0).bool()
        padded_inputs[~padded_valids] = 0.0
        
        stacked_queries = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["pos_enc"]
                for example in outputs
            ], batch_first=True, padding_value=0.0)
            for task_name in list(["change", "saliency", "class"])
        }
        stacked_weights = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["weights"]
                for example in outputs
            ], batch_first=True, padding_value=0.0)
            for task_name in list(["change", "saliency", "class"])
        }
        stacked_labels = {
            task_name: nn.utils.rnn.pad_sequence([
                example[task_name]["labels"]
                for example in outputs
            ], batch_first=True, padding_value=0).long()
            for task_name in list(["change", "saliency", "class"])
        }
        
        task_logits = self.forward(padded_inputs, queries=stacked_queries, input_mask=padded_valids)
        
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
                weights = weights[...,None]
            else:
                raise KeyError(criterion.target_encoding)
                
            task_loss = weights[task_mask] * criterion(
                logits[task_mask],
                loss_labels[task_mask],
            )
            losses.append(task_loss.mean())

            task_metric = self.head_metrics[stage][task_name](
                logits[task_mask],
                labels[task_mask],
            )
            metrics[task_name] = task_metric
        
        loss = sum(losses) / len(losses)
        
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

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
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # noqa
            >>> self = MultimodalTransformer(arch_name="smt_it_joint_p2", input_sensorchan='r|g|b')
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
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
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
            >>>     batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10)
            >>> datamodule.setup('fit')
            >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.MultimodalTransformer(
            >>>     arch_name="smt_it_joint_p2", classes=classes,
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
            >>> recon = methods.MultimodalTransformer.load_package(package_path)

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