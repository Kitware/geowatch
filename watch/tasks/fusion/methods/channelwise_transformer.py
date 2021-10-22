# -*- coding: utf-8 -*-
import einops
import kwarray
import kwcoco
import ubelt as ub
import torch
import torchmetrics
import pathlib

import numpy as np
import netharn as nh
import pytorch_lightning as pl

# import torch_optimizer as optim
from torch import nn
from einops.layers.torch import Rearrange
from kwcoco import channel_spec
from torchvision import transforms
from torch.optim import lr_scheduler
from watch.tasks.fusion import utils
from watch.tasks.fusion.architectures import transformer
from watch.tasks.fusion import heuristics

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


from timm.models.layers import drop_path
# from timm.models.layers.activations import sigmoid


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
        >>>     train_dataset='special:vidshapes8', num_workers=0)
        >>> datamodule.setup('fit')
        >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats()
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
        >>> self = MultimodalTransformer(arch_name='smt_it_joint_p8',
        >>>                              input_channels=datamodule.input_channels,
        >>>                              input_stats=dataset_stats,
        >>>                              classes=datamodule.classes,
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
        parser = parent_parser.add_argument_group("MultimodalTransformer")
        parser.add_argument('--name', default='unnamed_model', help=ub.paragraph(
            '''
            Specify a name for the experiment. (Unsure if the Model is the place for this)
            '''))

        parser.add_argument("--optimizer", default='RAdam', type=str, help='Optimizer name supported by the netharn API')
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--positive_change_weight", default=1.0, type=float)
        parser.add_argument("--negative_change_weight", default=1.0, type=float)
        parser.add_argument("--class_weights", default='auto', type=str, help='class weighting strategy')
        parser.add_argument("--saliency_weights", default='auto', type=str, help='class weighting strategy')

        # Model names define the transformer encoder used by the method
        available_encoders = list(transformer.encoder_configs.keys()) + ['deit']

        parser.add_argument(
            "--tokenizer", default='rearrange', type=str,
            choices=['dwcnn', 'rearrange'], help=ub.paragraph(
                '''
                How image patches aare broken into tokens.
                rearrange just shuffles raw pixels. dwcnn is a is a mobile
                convolutional stem.
                '''))
        parser.add_argument("--token_norm", default='auto', type=str,
                            choices=['auto', 'group', 'batch'])
        parser.add_argument("--arch_name", default='smt_it_stm_p8', type=str,
                            choices=available_encoders)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--global_class_weight", default=1.0, type=float)
        parser.add_argument("--global_change_weight", default=1.0, type=float)
        parser.add_argument("--global_saliency_weight", default=0.0, type=float)

        parser.add_argument("--change_loss", default='cce')
        parser.add_argument("--class_loss", default='focal')
        parser.add_argument("--saliency_loss", default='focal', help='saliency is trained to match any "positive/foreground/salient" class')

        parser.add_argument("--change_head_hidden", default=2, type=int, help='number of hidden layers in the change head')
        parser.add_argument("--class_head_hidden", default=2, type=int, help='number of hidden layers in the category head')
        parser.add_argument("--saliency_head_hidden", default=2, type=int, help='number of hidden layers in the saliency head')

        # parser.add_argument("--input_scale", default=2000.0, type=float)
        parser.add_argument("--window_size", default=8, type=int)
        parser.add_argument(
            "--attention_impl", default='exact', type=str, help=ub.paragraph(
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
                 arch_name='smt_it_stm_p8',
                 dropout=0.0,
                 optimizer='RAdam',
                 learning_rate=1e-3,
                 weight_decay=0.,
                 class_weights='auto',
                 saliency_weights='auto',
                 positive_change_weight=1.,
                 negative_change_weight=1.,
                 input_stats=None,
                 input_channels=None,
                 attention_impl='exact',
                 window_size=8,
                 global_class_weight=1.0,
                 global_change_weight=1.0,
                 global_saliency_weight=0.0,
                 change_head_hidden=1,
                 class_head_hidden=1,
                 saliency_head_hidden=1,
                 change_loss='cce',
                 class_loss='focal',
                 saliency_loss='focal',
                 tokenizer='rearrange',
                 token_norm='auto',
                 name='unnamed_expt',
                 classes=10):

        super().__init__()
        self.save_hyperparameters()
        self.name = name

        self.arch_name = arch_name

        # HACK:
        if input_stats is not None and 'input_stats' in input_stats:
            dataset_stats = input_stats
            input_stats = dataset_stats['input_stats']
            class_freq = dataset_stats['class_freq']
        else:
            class_freq = None

        self.class_freq = class_freq

        # Handle channel-wise input mean/std in the network (This is in
        # contrast to common practice where it is done in the dataloader)
        self.input_norms = None
        if input_stats is not None:
            self.input_norms = torch.nn.ModuleDict()
            for key, stats in input_stats.items():
                self.input_norms[key] = nh.layers.InputNorm(**stats)

        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        if input_channels is None:
            raise Exception('need them for num input_channels!')
        input_channels = channel_spec.ChannelSpec.coerce(input_channels)
        self.input_channels = input_channels

        input_streams = list(input_channels.streams())
        stream_num_channels = {s.spec: s.numel() for s in input_streams}
        self.stream_num_channels = stream_num_channels

        # TODO: rework "streams" to get the sum
        num_channels = input_channels.numel()

        self.global_class_weight = global_class_weight
        self.global_change_weight = global_change_weight
        self.global_saliency_weight = global_saliency_weight
        self.positive_change_weight = positive_change_weight
        self.negative_change_weight = negative_change_weight

        self.class_loss = class_loss
        self.change_loss = change_loss

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

        _info = coerce_criterion(self.class_loss, self.class_weights)
        self.class_criterion = _info['criterion']
        self.class_criterion_target_encoding = _info['target_encoding']
        self.class_criterion_logit_shape = _info['logit_shape']
        self.class_criterion_target_shape = _info['target_shape']

        _info = coerce_criterion(change_loss, self.change_weights)
        self.change_criterion = _info['criterion']
        self.change_criterion_target_encoding = _info['target_encoding']
        self.change_criterion_logit_shape = _info['logit_shape']
        self.change_criterion_target_shape = _info['target_shape']

        _info = coerce_criterion(saliency_loss, self.saliency_weights)
        self.saliency_criterion = _info['criterion']
        self.saliency_criterion_target_encoding = _info['target_encoding']
        self.saliency_criterion_logit_shape = _info['logit_shape']
        self.saliency_criterion_target_shape = _info['target_shape']

        self.class_metrics = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            "f1_micro": torchmetrics.F1(threshold=0.5, average='micro'),
            "f1_macro": torchmetrics.F1(threshold=0.5, average='macro', num_classes=self.num_classes),
        })

        self.change_metrics = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            "f1": torchmetrics.F1(),
        })

        self.saliency_metrics = nn.ModuleDict({
            # "acc": torchmetrics.Accuracy(),
            # "iou": torchmetrics.IoU(2),
            "f1": torchmetrics.F1(),
        })

        self.input_channels.numel()

        in_features_raw = self.hparams.window_size * self.hparams.window_size

        # TODO:
        #     - [ ] Dynamic / Learned embedding
        encode_t = utils.SinePositionalEncoding(5, 1, size=8)
        encode_m = utils.SinePositionalEncoding(5, 2, size=8)
        encode_h = utils.SinePositionalEncoding(5, 3, size=8)
        encode_w = utils.SinePositionalEncoding(5, 4, size=8)
        self.add_encoding = transforms.Compose([
            encode_t, encode_m, encode_h, encode_w,
        ])
        in_features_pos = sum(p.size for p in self.add_encoding.transforms)

        in_features = in_features_pos + in_features_raw

        # TODO:
        #     - [X] Classifier MLP, skip connections
        #     - [ ] Decoder - unsure if necessary

        # TODO: add tokenization strat to the FusionEncoder itself
        if tokenizer == 'rearrange':
            stream_tokenizers = nn.ModuleDict()
            for stream_key, num_chan in stream_num_channels.items():
                # Construct tokenize on a per-stream basis
                # import netharn as nh
                tokenize = Rearrange(
                    "b t c (h hs) (w ws) -> b t c h w (ws hs)",
                    hs=self.hparams.window_size,
                    ws=self.hparams.window_size)
                stream_tokenizers[stream_key] = tokenize
        elif tokenizer == 'dwcnn':
            stream_tokenizers = nn.ModuleDict()
            for stream_key, num_chan in stream_num_channels.items():
                # Construct tokenize on a per-stream basis
                # import netharn as nh
                tokenize = DWCNNTokenizer(num_chan, norm=token_norm)
                stream_tokenizers[stream_key] = tokenize
        else:
            raise KeyError(tokenizer)
        self.stream_tokenizers = stream_tokenizers

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
        # else:
        #     if arch_name != 'vit_base_patch16_224':
        #         print('might not work')
        #         transformer.TimmEncoder(arch_name)
        #         import timm
        #         model = timm.create_model('vit_base_patch16_224', pretrained=True)
        #         import netharn as nh
        #         nh.OutputShapeFor(model)

        feat_dim = self.encoder.out_features

        self.move_channels_last = Rearrange("b t c h w f -> b t h w f c")

        # A simple linear layer that learns to combine channels
        self.channel_fuser = nh.layers.MultiLayerPerceptronNd(
            0, num_channels, [], 1, norm=None)

        # self.binary_clf = nn.LazyLinear(1)  # TODO: rename to change_clf
        # self.class_clf = nn.LazyLinear(len(self.classes))  # category classifier
        self.change_head_hidden = change_head_hidden
        self.class_head_hidden = class_head_hidden
        self.saliency_head_hidden = saliency_head_hidden

        self.change_clf = nh.layers.MultiLayerPerceptronNd(
            dim=0,
            in_channels=feat_dim,
            hidden_channels=self.change_head_hidden,
            out_channels=2,
            norm=None
        )
        self.class_clf = nh.layers.MultiLayerPerceptronNd(
            dim=0,
            in_channels=feat_dim,
            hidden_channels=self.class_head_hidden,
            out_channels=self.num_classes,
            norm=None
        )

        # TODO:
        # Maybe drop heads if their weight is None?
        self.saliency_clf = nh.layers.MultiLayerPerceptronNd(
            dim=0,
            in_channels=feat_dim,
            hidden_channels=self.saliency_head_hidden,
            out_channels=self.saliency_num_classes,  # saliency will be trinary with an "other" class
            norm=None
        )

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

    @profile
    def forward(self, images):
        """
        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.tasks.fusion import datamodules
            >>> channels = 'B1,B8|B8a,B10|B11'
            >>> channels = 'B1|B8|B10|B8a|B11'
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0, channels=channels)
            >>> datamodule.setup('fit')
            >>> input_channels = datamodule.input_channels
            >>> train_dataset = datamodule.torch_datasets['train']
            >>> input_stats = train_dataset.cached_dataset_stats()
            >>> loader = datamodule.train_dataloader()
            >>> tokenizer = 'convexpt-v1'
            >>> tokenizer = 'dwcnn'
            >>> batch = next(iter(loader))
            >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_joint_p8',
            >>>     input_channels=input_channels,
            >>>     input_stats=input_stats,
            >>>     change_loss='dicefocal',
            >>>     attention_impl='performer',
            >>>     tokenizer=tokenizer,
            >>> )
            >>> images = torch.stack([ub.peek(f['modes'].values()) for f in batch[0]['frames']])[None, :]
            >>> images.shape
            >>> self.forward(images)
        """
        # Break images up into patches
        assert len(self.stream_tokenizers) == 1
        tokenize = ub.peek(self.stream_tokenizers.values())
        raw_patch_tokens = tokenize(images)
        # Add positional encodings for time, mode, and space.
        patch_tokens = self.add_encoding(raw_patch_tokens)

        # TODO: maybe make the encoder return a sequence of 1 less?
        # Rather than just ignoring the first output?
        patch_feats = self.encoder(patch_tokens)

        # Final channel-wise fusion
        chan_last = self.move_channels_last(patch_feats)

        # Latent channels are now marginalized away
        spacetime_fused_features = self.channel_fuser(chan_last)[..., 0]
        # spacetime_fused_features = einops.reduce(similarity, "b t c h w -> b t h w", "mean")

        # if 0:
        #     # TODO: add DotProduct back in?
        #     # similarity between neighboring timesteps
        #     feats = nn.functional.normalize(feats, dim=-1)
        #     similarity = torch.einsum("b t c h w f , b t c h w f -> b t c h w", feats[:, :-1], feats[:, 1:])
        #     similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        #     distance = -3.0 * similarity

        # Pass the final fused space-time feature to a classifier
        change_logits = self.change_clf(spacetime_fused_features[:, 1:])
        class_logits = self.class_clf(spacetime_fused_features)
        saliency_logits = self.saliency_clf(spacetime_fused_features)

        logits = {
            'change': change_logits,
            'class': class_logits,
            'saliency': saliency_logits,
        }
        return logits

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
            >>> if 0:
            >>>     dvc_dpath = find_smart_dvc_dpath()
            >>>     coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
            >>>     channels='swir16|swir22|blue|green|red|nir'
            >>> else:
            >>>     coco_fpath = 'special:vidshapes8-frames9-speed0.5-multispectral'
            >>>     channels='B1|B11|B8'
            >>> coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=coco_dset,
            >>>     chip_size=128, batch_size=1, time_steps=3,
            >>>     channels=channels,
            >>>     normalize_inputs=True, neg_to_pos_ratio=0, num_workers='avail/2',
            >>> )
            >>> datamodule.setup('fit')
            >>> torch_dset = datamodule.torch_datasets['train']
            >>> input_stats = datamodule.input_stats
            >>> input_channels = datamodule.input_channels
            >>> classes = datamodule.classes
            >>> print('input_stats = {}'.format(ub.repr2(input_stats, nl=3)))
            >>> print('input_channels = {}'.format(input_channels))
            >>> print('classes = {}'.format(classes))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.MultimodalTransformer(
            >>>     # ===========
            >>>     # Backbone
            >>>     arch_name='smt_it_joint_p8',
            >>>     #arch_name='smt_it_stm_p8',
            >>>     attention_impl='reformer',
            >>>     #arch_name='deit',
            >>>     # ===========
            >>>     # Change Loss
            >>>     change_loss='dicefocal',
            >>>     global_change_weight=1.00,
            >>>     positive_change_weight=1.0,
            >>>     negative_change_weight=0.05,
            >>>     # ===========
            >>>     # Class Loss
            >>>     global_class_weight=1.00,
            >>>     global_saliency_weight=1.00,
            >>>     #class_loss='cce',
            >>>     class_loss='focal',
            >>>     class_weights='auto',
            >>>     # ===========
            >>>     # Domain Metadata (Look Ma, not hard coded!)
            >>>     input_stats=input_stats,
            >>>     classes=classes,
            >>>     input_channels=input_channels,
            >>>     tokenizer='dwcnn',
            >>>     )
            >>> self.datamodule = datamodule
            >>> # Run one visualization
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))
            >>> # Show batch before we do anything
            >>> import kwplot
            >>> kwplot.autompl(force='Qt5Agg')
            >>> canvas = datamodule.draw_batch(batch, max_channels=1, overlay_on_image=0)
            >>> kwplot.imshow(canvas)
            >>> device = 0
            >>> self.overfit(batch)

        nh.initializers.KaimingNormal()(self)
        nh.initializers.Orthogonal()(self)

        """
        import kwplot
        from watch.utils.slugify_ext import smart_truncate
        import torch_optimizer
        from kwplot.mpl_make import render_figure_to_image
        import xdev
        import kwimage

        sns = kwplot.autosns()
        datamodule = self.datamodule
        device = 1
        self = self.to(device)
        # loader = datamodule.train_dataloader()
        # batch = next(iter(loader))
        walker = ub.IndexableWalker(batch)
        for path, val in walker:
            if isinstance(val, torch.Tensor):
                walker[path] = val.to(device)
        outputs = self.training_step(batch)
        canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=1, overlay_on_image=0)
        kwplot.imshow(canvas)

        loss_records = []
        loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
        step = 0
        frame_idx = 0
        # dpath = ub.ensuredir('_overfit_viz09')

        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optim='RAdam', lr=1e-0, weight_decay=1e-9,
            params=self.parameters())

        #optim = torch.optim.SGD(self.parameters(), lr=1e-4)
        #optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        optim = torch_optimizer.RAdam(self.parameters(), lr=3e-3, weight_decay=1e-5)

        fig = kwplot.figure(fnum=1, doclf=True)
        fig.set_size_inches(15, 6)
        fig.subplots_adjust(left=0.05, top=0.9)
        for frame_idx in xdev.InteractiveIter(list(range(frame_idx + 1, 1000))):
            # for frame_idx in list(range(frame_idx, 1000)):
            num_steps = 20
            for i in ub.ProgIter(range(num_steps), desc='overfit'):
                optim.zero_grad()
                outputs = self.training_step(batch)
                outputs['item_losses']
                loss = outputs['loss']
                item_losses_ = nh.data.collate.default_collate(outputs['item_losses'])
                item_losses = ub.map_vals(lambda x: sum(x).item(), item_losses_)
                loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in item_losses.items()])
                loss.backward()
                optim.step()
                step += 1
            canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=1, overlay_on_image=0, max_items=4)
            kwplot.imshow(canvas, pnum=(1, 2, 1), fnum=1)
            fig = kwplot.figure(fnum=1, pnum=(1, 2, 2))
            #kwplot.imshow(canvas, pnum=(1, 2, 1))
            import pandas as pd
            ax = sns.lineplot(data=pd.DataFrame(loss_records), x='step', y='val', hue='part')
            ax.set_yscale('log')
            fig.suptitle(smart_truncate(str(optim).replace('\n', ''), max_length=64))
            img = render_figure_to_image(fig)
            img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            # fpath = join(dpath, 'frame_{:04d}.png'.format(frame_idx))
            #kwimage.imwrite(fpath, img)
            xdev.InteractiveIter.draw()
        # TODO: can we get this batch to update in real time?
        # TODO: start a server process that listens for new images
        # as it gets new images, it starts playing through the animation
        # looping as needed

    @profile
    def forward_step(self, batch, with_loss=False, stage='unspecified'):
        """
        Generic forward step used for test / train / validation

        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes1-multispectral',
            >>>     num_workers=0, chip_size=128,
            >>>     normalize_inputs=True, neg_to_pos_ratio=0,
            >>> )
            >>> datamodule.setup('fit')
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))

            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.MultimodalTransformer(
            >>>     arch_name='smt_it_joint_p8',
            >>>     input_stats=datamodule.input_stats,
            >>>     classes=datamodule.classes, input_channels=datamodule.input_channels)
            >>> outputs = self.forward_step(batch, with_loss=True)
            >>> canvas = datamodule.draw_batch(batch, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()


        Ignore:
            python -m watch.cli.gifify \
                    -i /home/local/KHQ/jon.crall/data/work/toy_change/_overfit_viz7/ \
                    -o /home/local/KHQ/jon.crall/data/work/toy_change/_overfit_viz7.gif

            nh.initializers.functional.apply_initializer(self, torch.nn.init.kaiming_normal, {})


            # How to get data we need to step back into the dataloader
            # to debug the batch
            item = batch[0]

            item['frames'][0]['class_idxs'].unique()
            item['frames'][1]['class_idxs'].unique()
            item['frames'][2]['class_idxs'].unique()

            # print(item['frames'][0]['change'].unique())
            print(item['frames'][1]['change'].unique())
            print(item['frames'][2]['change'].unique())

            tr = item['tr']
            self = torch_dset
            kwplot.imshow(self.draw_item(item), fnum=3)

            kwplot.imshow(item['frames'][1]['change'].cpu().numpy(), fnum=4)

        Ignore:
            model = self
            model = self.to(0)

            for item in batch:
                for frame in item['frames']:
                    modes = frame['modes']
                    for key in modes.keys():
                        modes[key] = modes[key].to(0)
            out = model.forward_step(batch)

            batch2 = [ub.dict_diff(item, {'tr', 'index', 'video_name', 'video_id'})  for item in batch[0:1]]
            for item in batch2:
                item['frames'] = [
                    ub.dict_diff(frame, {
                        'gid', 'date_captured', 'sensor_coarse',
                        'change', 'ignore', 'class_idxs',
                    })
                    for frame in item['frames']
                ]

            traced = torch.jit.trace_module(model, {'forward_step': (batch2,)}, strict=False)

            traced = torch.jit.trace_module(model, {'forward': (images,)}, strict=False)

            import timerit
            ti = timerit.Timerit(5, bestof=1, verbose=2)
            for timer in ti.reset('time'):
                model.forward(images)
            for timer in ti.reset('time'):
                traced.forward(images)

            # traced = torch.jit.trace(model.forward, batch)
            traced = torch.jit.trace_module(model, {'forward_step': batch2})
        """
        outputs = {}

        item_losses = []

        item_changes_truth = []
        item_classes_truth = []
        item_saliency_truth = []

        item_change_probs = []
        item_class_probs = []
        item_saliency_probs = []
        for item in batch:

            # For now, just reconstruct the stacked input, but
            # in the future, the encoder will need to take care of
            # the heterogeneous inputs

            frame_ims = []
            frame_ignores = []
            for frame in item['frames']:
                assert len(self.input_norms) == 1, 'only handle one mode for now'
                for mode_key in self.input_norms.keys():
                    mode_val = frame['modes'][mode_key]
                    mode_val = mode_val.float()
                    # self.input_norms = None
                    if self.input_norms is not None:
                        mode_norm = self.input_norms[mode_key]
                        mode_val = mode_norm(mode_val)
                    frame_ims.append(mode_val)

                if with_loss:
                    frame_ignores.append(frame['ignore'])

            # Because we are nt collating we need to add a batch dimension
            # if with_loss:
            #     ignores = torch.stack(frame_ignores)[None, ...]
            images = torch.stack(frame_ims)[None, ...]

            B, T, C, H, W = images.shape

            logits = self.forward(images)

            # TODO: it may be faster to compute loss at the downsampled
            # resolution.
            resampled_logits = {}
            # Loop over change, categories, saliency
            for logit_key, logit_val in logits.items():
                _tmp = einops.rearrange(logit_val, 'b t h w c -> b (t c) h w')
                _tmp2 = nn.functional.interpolate(
                    _tmp, [H, W], mode="bilinear", align_corners=True)
                resampled = einops.rearrange(_tmp2, 'b (t c) h w -> b t h w c', c=logit_val.shape[4])
                resampled_logits[logit_key] = resampled

            class_logits = resampled_logits['class']
            change_logits = resampled_logits['change']
            saliency_logits = resampled_logits['saliency']

            # Remove batch index in both cases
            change_prob = change_logits.detach().softmax(dim=4)[0, ..., 1]
            class_prob = class_logits.detach().sigmoid()[0]
            saliency_prob = saliency_logits.detach().sigmoid()[0]

            # Hack the change prob so it works with our currently binary
            # visualizations
            # change_prob = ((1 - change_prob[..., 0]) + change_prob[..., 0]) / 2.0

            item_change_probs.append(change_prob)
            item_class_probs.append(class_prob)
            item_saliency_probs.append(saliency_prob)

            item_loss_parts = {}

            if with_loss:
                true_changes = torch.stack([
                    frame['change'] for frame in item['frames'][1:]
                ])[None, ...]
                item_changes_truth.append(true_changes)  # [B, T, H, W, C]

                true_class = torch.stack([
                    frame['class_idxs'] for frame in item['frames']
                ])[None, ...]
                item_classes_truth.append(true_class)  # [B, T, H, W, C]

                if not hasattr(self, '_fg_idxs'):
                    fg_idxs = sorted(ub.take(self.classes.node_to_idx, self.foreground_classes))
                    # bg_idxs = sorted(ub.take(self.classes.node_to_idx, self.background_classes))
                    fg_idxs = np.expand_dims(np.array(fg_idxs), (0, 1, 2, 3))
                    self._fg_idxs = torch.Tensor(fg_idxs).to(true_class.device)
                    # true_class[..., None] == torch.Tensor(self._fg_idxs)
                    # self._bg_idxs = bg_idxs

                true_saliency = (true_class[..., None] == self._fg_idxs).any(dim=4).long()
                item_saliency_truth.append(true_saliency)

                # compute criterion
                # valids_ = (1 - ignores)[..., None]  # [B, T, H, W, 1]

                # Hack: change the 1-logit binary case to 2 class binary case
                need_change_loss   = 1 or self.global_change_weight > 0
                need_class_loss    = 1 or self.global_class_weight > 0
                need_saliency_loss = 1 or self.global_saliency_weight > 0

                if need_change_loss:
                    change_pred_input = einops.rearrange(
                        change_logits,
                        'b t h w c -> ' + self.change_criterion_logit_shape).contiguous()
                    if self.change_criterion_target_encoding == 'index':
                        change_true_cxs = true_changes.long()
                        change_true_input = einops.rearrange(
                            change_true_cxs,
                            'b t h w -> ' + self.change_criterion_target_shape).contiguous()
                    elif self.change_criterion_target_encoding == 'onehot':
                        # Note: 1HE is much easier to work with
                        change_true_ohe = kwarray.one_hot_embedding(true_changes.long(), 2, dim=-1)
                        change_true_input = einops.rearrange(
                            change_true_ohe,
                            'b t h w c -> ' + self.change_criterion_target_shape).contiguous()
                    else:
                        raise KeyError(self.change_criterion_target_encoding)

                    # TODO: it would be nice instead of having a valid mask, if we
                    # had a pixelwise weighting of how much we care about each
                    # pixel. This would let us upweight particular instances
                    # and also ignore regions by setting the weights to zero.
                    # mask = einops.rearrange(valids_, 'b t h w c -> ' + self.change_criterion_logit_shape, c=1)
                    change_loss = self.change_criterion(
                        change_pred_input,
                        change_true_input
                    )
                    # num_change_states = 2
                    # true_change_ohe = kwarray.one_hot_embedding(true_changes.long(), num_change_states, dim=-1).float()
                    # change_loss = self.change_criterion(change_logits, true_change_ohe).mean()
                    # change_loss = self.change_criterion(change_logits, true_changes.float()).mean()
                    item_loss_parts['change'] = self.global_change_weight * change_loss

                # Class loss part
                if need_class_loss:
                    class_pred_input = einops.rearrange(
                        class_logits,
                        'b t h w c -> ' + self.class_criterion_logit_shape).contiguous()
                    if self.class_criterion_target_encoding == 'index':
                        class_true_cxs = true_class.long()
                        class_true_input = einops.rearrange(
                            class_true_cxs,
                            'b t h w -> ' + self.class_criterion_target_shape).contiguous()
                    elif self.class_criterion_target_encoding == 'onehot':
                        class_true_ohe = kwarray.one_hot_embedding(true_class.long(), len(self.classes), dim=-1)
                        class_true_input = einops.rearrange(
                            class_true_ohe,
                            'b t h w c -> ' + self.class_criterion_target_shape).contiguous()
                    else:
                        raise KeyError(self.class_criterion_target_encoding)
                    class_loss = self.class_criterion(
                        class_pred_input,
                        class_true_input
                    )
                    item_loss_parts['class'] = self.global_class_weight * class_loss

                # Saliency loss part
                if need_saliency_loss:
                    saliency_pred_input = einops.rearrange(
                        saliency_logits,
                        'b t h w c -> ' + self.saliency_criterion_logit_shape).contiguous()
                    if self.saliency_criterion_target_encoding == 'index':
                        saliency_true_cxs = true_saliency.long()
                        saliency_true_input = einops.rearrange(
                            saliency_true_cxs,
                            'b t h w -> ' + self.saliency_criterion_target_shape).contiguous()
                    elif self.saliency_criterion_target_encoding == 'onehot':
                        saliency_true_ohe = kwarray.one_hot_embedding(true_saliency.long(), self.saliency_num_classes, dim=-1)
                        saliency_true_input = einops.rearrange(
                            saliency_true_ohe,
                            'b t h w c -> ' + self.saliency_criterion_target_shape).contiguous()
                    else:
                        raise KeyError(self.saliency_criterion_target_encoding)
                    saliency_loss = self.saliency_criterion(
                        saliency_pred_input,
                        saliency_true_input
                    )
                    item_loss_parts['saliency'] = self.global_saliency_weight * saliency_loss

                item_losses.append(item_loss_parts)

        outputs['change_probs'] = item_change_probs
        outputs['class_probs'] = item_class_probs
        outputs['saliency_probs'] = item_saliency_probs

        if with_loss:
            total_loss = sum(
                val for parts in item_losses for val in parts.values())

            all_true_change = torch.cat(item_changes_truth, dim=0)
            all_pred_change = torch.stack(item_change_probs)

            all_true_class = torch.cat(item_classes_truth, dim=0).view(-1)
            all_pred_class = torch.stack(item_class_probs).view(-1, self.num_classes)

            all_true_saliency = torch.cat(item_saliency_truth, dim=0).view(-1)
            all_pred_saliency = torch.stack(item_saliency_probs).view(-1, self.saliency_num_classes)

            # compute metrics
            if self.trainer is not None:
                item_metrics = {}

                # Dont log unless a trainer is attached
                for key, metric in self.change_metrics.items():
                    val = metric(all_pred_change, all_true_change)
                    item_metrics[f'{stage}_change_{key}'] = val

                for key, metric in self.class_metrics.items():
                    val = metric(all_pred_class, all_true_class)
                    item_metrics[f'{stage}_class_{key}'] = val

                for key, metric in self.saliency_metrics.items():
                    val = metric(all_pred_saliency, all_true_saliency)
                    item_metrics[f'{stage}_saliency_{key}'] = val

                for key, val in item_metrics.items():
                    self.log(key, val, prog_bar=True)
                self.log(f'{stage}_loss', total_loss, prog_bar=True)

            # Detach the itemized losses
            for path, val in ub.IndexableWalker(item_losses):
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
        DEPRECATE IN FAVOR OF A KITWARE UTILITY

        TODO:
            - [ ] We should be able to load the model without having access
                  to this class. What is the right way to do that?
        """
        import torch.package
        import json
        #
        # TODO: is there any way to introspect what these variables could be?

        arch_name = "model.pkl"
        module_name = 'watch_tasks_fusion'

        imp = torch.package.PackageImporter(package_path)

        # Assume this standardized header information exists that tells us the
        # name of the resource corresponding to the model
        try:
            package_header = json.loads(imp.load_text(
                'kitware_package_header', 'kitware_package_header.json'))
        except Exception:
            package_header = imp.load_pickle(
                'kitware_package_header', 'kitware_package_header.pkl')
        arch_name = package_header['arch_name']
        module_name = package_header['module_name']

        # from timm.models import efficientnet_blocks
        # from timm.models import efficientnet_blocks  # NOQA
        # efficientnet_blocks.DepthwiseSeparableConv

        # pkg_root = imp.file_structure()
        # print(pkg_root)
        # pkg_data = pkg_root.children['.data']

        self = imp.load_pickle(module_name, arch_name)
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
            >>> # We have to run an input through the module because it is lazy
            >>> inputs = torch.rand(1, 2, 13, 128, 128)
            >>> model(inputs)

            >>> # Save the model
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
            >>>     'special:vidshapes8-multispectral', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=0)
            >>> datamodule.setup('fit')
            >>> input_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.MultimodalTransformer(
            >>>     "smt_it_stm_p8", classes=classes,
            >>>     input_stats=input_stats, input_channels=datamodule.input_channels)

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
        import copy
        import json
        import torch.package

        # shallow copy of self, to apply attribute hacks to
        model = copy.copy(self)

        train_dpath_hint = getattr(model, 'train_dpath_hint', None)
        if model.trainer is not None:
            if train_dpath_hint is None:
                train_dpath_hint = model.trainer.log_dir
            datamodule = model.trainer.datamodule
            if datamodule is not None:
                model.datamodule_hparams = datamodule.hparams

        metadata_fpaths = []
        if train_dpath_hint is not None:
            train_dpath_hint = pathlib.Path(train_dpath_hint)
            metadata_fpaths += list(train_dpath_hint.glob('hparams.yaml'))
            metadata_fpaths += list(train_dpath_hint.glob('fit_config.yaml'))

        model.trainer = None
        model.train_dataloader = None
        model.val_dataloader = None
        model.test_dataloader = None

        arch_name = "model.pkl"
        module_name = 'watch_tasks_fusion'
        """
        exp = torch.package.PackageExporter(package_path, verbose=True)
        """
        with torch.package.PackageExporter(package_path) as exp:
            # TODO: this is not a problem yet, but some package types (mainly
            # binaries) will need to be excluded and added as mocks
            exp.extern("**", exclude=["watch.tasks.fusion.**"])
            exp.intern("watch.tasks.fusion.**", allow_empty=False)

            # Attempt to standardize some form of package metadata that can
            # allow for model importing with fewer hard-coding requirements
            package_header = {
                'version': '0.1.0',
                'arch_name': arch_name,
                'module_name': module_name,
            }

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


def _class_weights_from_freq(total_freq, mode='median-idf'):
    """
    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import _class_weights_from_freq
        >>> total_freq = np.array([19503736, 92885, 883379, 0, 0])
        >>> _class_weights_from_freq(total_freq, mode='idf')
        >>> print(_class_weights_from_freq(total_freq, mode='median-idf'))
        >>> _class_weights_from_freq(total_freq, mode='log-median-idf')
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

    _min, _max = np.percentile(freq, [5, 95])
    is_valid = (_min <= freq) & (freq <= _max)
    if np.any(is_valid):
        middle_value = np.median(freq[is_valid])
    else:
        middle_value = np.median(freq)

    # variant of median-inverse-frequency
    mask = freq != 0
    nonzero_freq = freq[mask]
    if len(nonzero_freq):
        freq[freq == 0] = nonzero_freq.min() / 2

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
    weights[mask] = weights[mask] / weights[mask].max()
    weights[~mask] = weights[mask].max() / 7

    # weights[] = 1.0

    weights = np.round(weights, 6)
    return weights


def coerce_criterion(loss_code, weights):
    """
    Helps build a loss function
    """
    import monai
    if loss_code == 'cce':
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights, reduction='mean')
        target_encoding = 'index'
        logit_shape = '(b t h w) c'
        target_shape = '(b t h w)'
    elif loss_code == 'focal':
        criterion = monai.losses.FocalLoss(
            reduction='mean', to_onehot_y=False, weight=weights)

        target_encoding = 'onehot'
        logit_shape = 'b c h w t'
        target_shape = 'b c h w t'

    elif loss_code == 'dicefocal':
        # TODO: can we apply weights here?
        criterion = monai.losses.DiceFocalLoss(
            # weight=torch.FloatTensor([self.negative_change_weight, self.positive_change_weight]),
            sigmoid=True,
            to_onehot_y=False,
            reduction='mean')
        target_encoding = 'onehot'
        logit_shape = 'b c h w t'
        target_shape = 'b c h w t'
    else:
        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.BCEWithLogitsLoss()
        raise NotImplementedError(loss_code)

    criterion_info = {
        'criterion': criterion,
        'target_encoding': target_encoding,
        'logit_shape': logit_shape,
        'target_shape': target_shape,
    }
    return criterion_info


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


class DWCNNTokenizer(nn.Module):
    def __init__(self, in_chn, norm='auto'):
        super().__init__()
        self.norm = norm
        # self.to_images = Rearrange("b t c h w -> (b t) c h w")
        self.stem = nn.Sequential(*[
            OurDepthwiseSeparableConv(in_chn, in_chn, kernel_size=3, stride=1, padding=1, residual=1, norm=None, noli=None),
            OurDepthwiseSeparableConv(in_chn, in_chn * 4, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
            OurDepthwiseSeparableConv(in_chn * 4, in_chn * 8, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
            OurDepthwiseSeparableConv(in_chn * 8, in_chn * 64, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
        ])
        self.expand_factor = 64
        # self.to_tokens = Rearrange("(b t) (c ef) h w -> (b t) c h w ef", ef=self.expand_factor, b=1)

    def forward(self, inputs):
        """
        self = DWCNNTokenizer(13)
        inputs = torch.rand(2, 5, 13, 16, 16)
        self(inputs)
        """
        b, t, c, h, w = inputs.shape
        inputs2d = einops.rearrange(inputs, "b t c h w -> (b t) c h w")
        tokens2d = self.stem(inputs2d)
        tokens = einops.rearrange(tokens2d, "(b t) (c ef) h w -> b t c h w ef", b=b, t=t, ef=self.expand_factor)
        return tokens
