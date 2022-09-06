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

        self.head_metrics = nn.ModuleDict()
        self.head_metrics['class'] = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1_micro': FBetaScore(beta=1.0, threshold=0.5, average='micro'),
            'f1_macro': FBetaScore(beta=1.0, threshold=0.5, average='macro', num_classes=self.num_classes),
        })
        self.head_metrics['change'] = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            'f1': FBetaScore(beta=1.0),
        })
        self.head_metrics['saliency'] = torchmetrics.MetricCollection({
            'f1': FBetaScore(beta=1.0),
        })
    
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
            print(context.shape, queries[task_name].shape, task_tokens.shape)
            task_logits = self.heads[task_name](task_tokens)
            task_logits = einops.rearrange(task_logits, "batch seq chan -> batch chan seq")
            outputs[task_name] = task_logits
            
        return outputs
    
    def training_step(self, batch, batch_idx=None):
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
            
            try:
                logits = einops.rearrange(task_logits[task_name], "batch chan seq -> (batch seq) chan")
                labels = einops.rearrange(stacked_labels[task_name], "batch seq -> (batch seq)")
                weights = einops.rearrange(stacked_weights[task_name], "batch seq -> (batch seq)")
                
                # task_mask = (labels != -1)
                task_mask = (weights > 0.0)
                
                task_loss = weights[task_mask] * self.criterions[task_name](
                    logits[task_mask],
                    labels[task_mask],
                )
                losses.append(task_loss.mean())

                task_metric = self.head_metrics[task_name](
                    logits[task_mask],
                    labels[task_mask],
                )
                metrics[task_name] = task_metric
                
            except:
                print(f"failed on {task_name}")
                raise
        
        loss = sum(losses) / len(losses)
        
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss