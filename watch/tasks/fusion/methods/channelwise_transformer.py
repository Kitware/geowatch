import einops
import kwarray
import kwcoco
import ubelt as ub
import torch
import torchmetrics

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

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


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
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
        >>> self = MultimodalTransformer(arch_name='smt_it_joint_p8', input_channels=datamodule.input_channels, change_loss='dicefocal', attention_impl='performer')
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

    def __init__(self,
                 arch_name='smt_it_stm_p8',
                 dropout=0.0,
                 optimizer='RAdam',
                 learning_rate=1e-3,
                 weight_decay=0.,
                 class_weights='auto',
                 positive_change_weight=1.,
                 negative_change_weight=1.,
                 input_stats=None,
                 input_channels=None,
                 attention_impl='exact',
                 window_size=8,
                 global_class_weight=1.0,
                 global_change_weight=1.0,
                 change_head_hidden=2,
                 class_head_hidden=2,
                 change_loss='cce',
                 class_loss='focal',
                 classes=10):

        super().__init__()
        self.save_hyperparameters()

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

        # TODO: rework "streams" to get the sum
        num_channels = input_channels.numel()

        self.global_class_weight = global_class_weight
        self.global_change_weight = global_change_weight
        self.positive_change_weight = positive_change_weight
        self.negative_change_weight = negative_change_weight

        self.class_loss = class_loss
        self.change_loss = change_loss

        # criterion and metrics
        # TODO: parametarize loss criterions
        # For loss function experiments, see and work in
        # ~/code/watch/watch/tasks/fusion/methods/channelwise_transformer.py
        import monai
        # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        if isinstance(class_weights, str):
            if class_weights == 'auto':
                import numpy as np
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

                heuristic_weights.update({
                    'ignore': 0.00,
                    'clouds': 0.00,
                    'Unknown' : 0.0,
                    # 'background': 0.05,
                    # 'No Activity'        : 0.003649,
                    # 'Active Construction': 0.188011,
                    # 'Site Preparation'   : 1.0,
                    # 'Post Construction'  : 0.142857,
                })
                # print('heuristic_weights = {}'.format(ub.repr2(heuristic_weights, nl=1, align=':')))
                class_weights = []
                for catname in self.classes:
                    w = heuristic_weights.get(catname, 1.0)
                    class_weights.append(w)
                using_class_weights = ub.dzip(self.classes, class_weights)
                print('using_class_weights = {}'.format(ub.repr2(using_class_weights, nl=1, align=':')))
                class_weights = torch.FloatTensor(class_weights)
                # print('self.classes = {!r}'.format(self.classes))
                # print('AUTO class_weights = {!r}'.format(class_weights))
            else:
                raise KeyError(class_weights)
        else:
            raise NotImplementedError(class_weights)
        self.class_weights = class_weights
        self.change_weights = torch.FloatTensor([
            self.negative_change_weight,
            self.positive_change_weight
        ])

        def construct_loss(loss_code, weights):
            if class_loss == 'cce':
                criterion = torch.nn.CrossEntropyLoss(
                    weight=weights, reduction='mean')
                target_encoding = 'index'
                logit_shape = '(b t h w) c'
                target_shape = '(b t h w)'
            elif class_loss == 'focal':
                criterion = monai.losses.FocalLoss(
                    reduction='mean', to_onehot_y=False, weight=weights)

                target_encoding = 'onehot'
                logit_shape = 'b c h w t'
                target_shape = 'b c h w t'

            elif self.change_loss.lower() == 'dicefocal':
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
                raise NotImplementedError(class_loss)
            return criterion, target_encoding, logit_shape, target_shape

        (criterion, target_encoding,
         logit_shape, target_shape) = construct_loss(self.class_loss,
                                                     self.class_weights)
        self.class_criterion = criterion
        self.class_criterion_target_encoding = target_encoding
        self.class_criterion_logit_shape = logit_shape
        self.class_criterion_target_shape = target_shape

        (criterion, target_encoding,
         logit_shape, target_shape) = construct_loss(
             change_loss, self.change_weights)
        self.change_criterion = criterion
        self.change_criterion_target_encoding = target_encoding
        self.change_criterion_logit_shape = logit_shape
        self.change_criterion_target_shape = target_shape

        # # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        # if self.change_loss.lower() == 'cce':
        #     self.change_criterion = torch.nn.CrossEntropyLoss(
        #         weight=torch.FloatTensor([
        #             self.negative_change_weight,
        #             self.positive_change_weight
        #         ]),
        #         reduction='mean')
        #     self.change_criterion_target_encoding = 'onehot'
        #     self.change_criterion_logit_shape = '(b t h w) c'
        #     self.change_criterion_target_shape = '(b t h w)'
        # elif self.change_loss.lower() == 'dicefocal':
        #     # TODO: can we apply weights here?
        #     self.change_criterion = monai.losses.DiceFocalLoss(
        #         # weight=torch.FloatTensor([self.negative_change_weight, self.positive_change_weight]),
        #         sigmoid=True,
        #         to_onehot_y=False,
        #         reduction='mean')
        #     self.change_criterion_target_encoding = 'index'
        #     self.change_criterion_logit_shape = 'b c h w t'
        #     self.change_criterion_target_shape = 'b c h w t'
        # else:
        #     raise NotImplementedError

        # self.change_criterion = nn.BCEWithLogitsLoss(
        #         pos_weight=torch.ones(1) * pos_weight)

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

        in_features_raw = self.hparams.window_size * self.hparams.window_size
        in_features_pos = (2 * 4 * 4)  # positional encoding feature
        in_features = in_features_pos + in_features_raw
        encoder_config = transformer.encoder_configs[arch_name]

        # TODO:
        #     - [X] Classifier MLP, skip connections
        #     - [ ] Decoder
        #     - [ ] Dynamic / Learned embeddings

        # TODO: add tokenization strat to the FusionEncoder itself
        self.tokenize = Rearrange("b t c (h hs) (w ws) -> b t c h w (ws hs)",
                                  hs=self.hparams.window_size,
                                  ws=self.hparams.window_size)

        encode_t = utils.SinePositionalEncoding(5, 1, sine_pairs=4)
        encode_m = utils.SinePositionalEncoding(5, 2, sine_pairs=4)
        encode_h = utils.SinePositionalEncoding(5, 3, sine_pairs=4)
        encode_w = utils.SinePositionalEncoding(5, 4, sine_pairs=4)
        self.add_encoding = transforms.Compose([
            encode_t, encode_m, encode_h, encode_w,
        ])

        encoder = transformer.FusionEncoder(
            **encoder_config,
            in_features=in_features,
            attention_impl=attention_impl,
            dropout=dropout,
        )
        self.encoder = encoder

        feat_dim = self.encoder.out_features

        self.move_channels_last = Rearrange("b t c h w f -> b t h w f c")

        # A simple linear layer that learns to combine channels
        self.channel_fuser = nh.layers.MultiLayerPerceptronNd(
            0, num_channels, [], 1, norm=None)

        # self.binary_clf = nn.LazyLinear(1)  # TODO: rename to change_clf
        # self.class_clf = nn.LazyLinear(len(self.classes))  # category classifier
        self.change_head_hidden = change_head_hidden
        self.class_head_hidden = class_head_hidden

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
        parser.add_argument("--optimizer", default='RAdam', type=str, help='Optimizer name supported by the netharn API')
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--positive_change_weight", default=1.0, type=float)
        parser.add_argument("--negative_change_weight", default=1.0, type=float)
        parser.add_argument("--class_weights", default='auto', type=str, help='class weighting strategy')

        # Model names define the transformer encoder used by the method
        available_encoders = list(transformer.encoder_configs.keys())
        parser.add_argument("--arch_name", default='smt_it_stm_p8', type=str,
                            choices=available_encoders)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--global_class_weight", default=1.0, type=float)
        parser.add_argument("--global_change_weight", default=1.0, type=float)

        parser.add_argument("--change_loss", default='cce')
        parser.add_argument("--class_loss", default='focal')

        parser.add_argument("--change_head_hidden", default=2, type=int, help='number of hidden layers in the change head')
        parser.add_argument("--class_head_hidden", default=2, type=int, help='number of hidden layers in the category head')

        # parser.add_argument("--input_scale", default=2000.0, type=float)
        parser.add_argument("--window_size", default=8, type=int)
        parser.add_argument(
            "--attention_impl", default='exact', type=str, help=ub.paragraph(
                '''
                Implementation for attention computation.
                Can be:
                'exact' - the original O(n^2) method.
                'performer' - a linear approximation.
                '''))
        return parent_parser

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
        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optimizer=self.hparams.optimizer,
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == 'RAdam':
            optim_kw['betas'] = (0.9, 0.99)  # backwards compat

        optim_kw['params'] = self.parameters()
        optimizer = optim_cls(**optim_kw)

        # optimizer = optim.RAdam(
        #         self.parameters(),
        #         lr=self.hparams.learning_rate,
        #         weight_decay=self.hparams.weight_decay,
        #         betas=(0.9, 0.99))
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def forward(self, images):
        # Break images up into patches
        raw_patch_tokens = self.tokenize(images)
        # Add positional encodings for time, mode, and space.
        patch_tokens = self.add_encoding(raw_patch_tokens)

        # TODO: maybe make the encoder return a sequence of 1 less?
        # Rather than just ignoring the first output?
        patch_feats = self.encoder(patch_tokens)

        # Final channel-wise fusion
        chan_last = self.move_channels_last(patch_feats)

        # Channels are now marginalized away
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

        logits = {
            'change': change_logits,
            'class': class_logits,
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
            >>> if 1:
            >>>     dvc_dpath = find_smart_dvc_dpath()
            >>>     coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
            >>>     channels='swir16|swir22|blue|green|red|nir'
            >>> else:
            >>>     coco_fpath = 'special:vidshapes8-frames9-speed0.5-multispectral'
            >>>     channels='B1|B11|B8',
            >>> coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=coco_dset,
            >>>     chip_size=128, batch_size=1, time_steps=5,
            >>>     channels=channels,
            >>>     normalize_inputs=True, neg_to_pos_ratio=0, num_workers='avail//2',
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
            >>>     attention_impl='performer',
            >>>     # ===========
            >>>     # Change Loss
            >>>     change_loss='dicefocal',
            >>>     global_change_weight=0.00,
            >>>     positive_change_weight=1.0,
            >>>     negative_change_weight=0.05,
            >>>     # ===========
            >>>     # Class Loss
            >>>     global_class_weight=1.00,
            >>>     #class_loss='cce',
            >>>     class_loss='focal',
            >>>     class_weights='auto',
            >>>     # ===========
            >>>     # Domain Metadata (Look Ma, not hard coded!)
            >>>     input_stats=input_stats,
            >>>     classes=classes,
            >>>     input_channels=input_channels
            >>>     )
            >>> self.datamodule = datamodule
            >>> # Run one visualization
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))
            >>> self.overfit(batch)
        """
        import kwplot
        from watch.utils.slugify_ext import smart_truncate
        import torch_optimizer
        from kwplot.mpl_make import render_figure_to_image
        import xdev
        import kwimage
        # from os.path import join
        kwplot.autompl(force='Qt5Agg')

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
        canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=1, overlay_on_image=0)
        kwplot.imshow(canvas)

        loss_records = []
        loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
        step = 0
        frame_idx = 0
        # dpath = ub.ensuredir('_overfit_viz09')

        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optim='RAdam', lr=3e-3, weight_decay=1e-5,
            params=self.parameters())

        #optim = torch.optim.SGD(self.parameters(), lr=1e-4)
        #optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        optim = torch_optimizer.RAdam(self.parameters(), lr=3e-3, weight_decay=1e-5)

        fig = kwplot.figure(fnum=1, doclf=True)
        fig.set_size_inches(15, 6)
        fig.subplots_adjust(left=0.05, top=0.9)
        for frame_idx in xdev.InteractiveIter(list(range(frame_idx + 1, 1000))):
            #for frame_idx in list(range(frame_idx, 1000)):
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
            >>>     train_dataset='special:vidshapes8-multispectral',
            >>>     num_workers=4, chip_size=128,
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
        """
        outputs = {}

        item_losses = []
        item_changes_truth = []
        item_classes_truth = []
        item_change_probs = []
        item_class_probs = []
        for item in batch:

            # For now, just reconstruct the stacked input, but
            # in the future, the encoder will need to take care of
            # the heterogeneous inputs

            frame_ims = []
            frame_ignores = []
            for frame in item['frames']:
                assert len(frame['modes']) == 1, 'only handle one mode for now'
                mode_key, mode_val = ub.peek(frame['modes'].items())
                mode_val = mode_val.float()
                # self.input_norms = None
                if self.input_norms is not None:
                    mode_norm = self.input_norms[mode_key]
                    mode_val = mode_norm(mode_val)
                frame_ims.append(mode_val)
                frame_ignores.append(frame['ignore'])

            # Because we are not collating we need to add a batch dimension
            # ignores = torch.stack(frame_ignores)[None, ...]
            images = torch.stack(frame_ims)[None, ...]

            B, T, C, H, W = images.shape

            logits = self(images)

            # TODO: it may be faster to compute loss at the downsampled
            # resolution.
            class_logits_small = logits['class']
            change_logits_small = logits['change']

            _tmp = einops.rearrange(change_logits_small, 'b t h w c -> b (t c) h w')
            _tmp2 = nn.functional.interpolate(
                _tmp, [H, W], mode="bilinear", align_corners=True)
            change_logits = einops.rearrange(_tmp2, 'b (t c) h w -> b t h w c', c=change_logits_small.shape[4])

            _tmp = einops.rearrange(class_logits_small, 'b t h w c -> b (t c) h w')
            _tmp2 = nn.functional.interpolate(
                _tmp, [H, W], mode="bilinear", align_corners=True)
            class_logits = einops.rearrange(_tmp2, 'b (t c) h w -> b t h w c', c=class_logits_small.shape[4])

            # Remove batch index in both cases
            # change_prob = change_logits.detach().sigmoid()[0]
            change_prob = change_logits.detach().softmax(dim=4)[0, ..., 1]

            class_prob = class_logits.detach().sigmoid()[0]

            # Hack the change prob so it works with our currently binary
            # visualizations
            # change_prob = ((1 - change_prob[..., 0]) + change_prob[..., 0]) / 2.0

            item_change_probs.append(change_prob)
            item_class_probs.append(class_prob)

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

                # compute criterion
                # print('change_logits.shape = {!r}'.format(change_logits.shape))
                # print('true_changes.shape = {!r}'.format(true_changes.shape))
                # valids_ = (1 - ignores)[..., None]  # [B, T, H, W, 1]

                # Hack: change the 1-logit binary case to 2 class binary case
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
                # print('change_pred_input.shape = {!r}'.format(change_pred_input.shape))
                # print('change_true_input.shape = {!r}'.format(change_true_input.shape))
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

                # true_class_ohe = kwarray.one_hot_embedding(true_class.long(), self.num_classes, dim=-1).float()
                # class_loss = self.class_criterion(class_logits, true_class_ohe).mean()
                # # class_loss = torch.nn.functional.binary_cross_entropy_with_logits(class_logits, true_class_ohe)
                # item_loss_parts['class'] = self.global_class_weight * class_loss

                item_losses.append(item_loss_parts)

        outputs['change_probs'] = item_change_probs
        outputs['class_probs'] = item_class_probs

        if with_loss:
            total_loss = sum(
                val for parts in item_losses for val in parts.values())

            all_pred_change = torch.stack(item_change_probs)
            all_true_change = torch.cat(item_changes_truth, dim=0)

            all_true_class = torch.cat(item_classes_truth, dim=0).view(-1)
            all_pred_class = torch.stack(item_class_probs).view(-1, self.num_classes)

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
        TODO:
            - [ ] We should be able to load the model without having access
                  to this class. What is the right way to do that?
        """
        import torch.package
        #
        # TODO: is there any way to introspect what these variables could be?

        arch_name = "model.pkl"
        module_name = 'watch_tasks_fusion'

        imp = torch.package.PackageImporter(package_path)

        # Assume this standardized header information exists that tells us the
        # name of the resource corresponding to the model
        package_header = imp.load_pickle(
            'kitware_package_header', 'kitware_package_header.pkl')
        arch_name = package_header['arch_name']
        module_name = package_header['module_name']

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
            >>> input_stats = datamodule.torch_datasets['train'].cached_dataset_stats()
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
        """
        import copy
        import torch.package

        # shallow copy of self, to apply attribute hacks to
        model = copy.copy(self)
        if model.trainer is not None:
            datamodule = model.trainer.datamodule
            if datamodule is not None:
                model.datamodule_hparams = datamodule.hparams
        model.trainer = None
        model.train_dataloader = None
        model.val_dataloader = None
        model.test_dataloader = None

        arch_name = "model.pkl"
        module_name = 'watch_tasks_fusion'
        with torch.package.PackageExporter(package_path, verbose=verbose) as exp:
            # TODO: this is not a problem yet, but some package types (mainly
            # binaries) will need to be excluded and added as mocks
            exp.extern("**", exclude=["watch.tasks.fusion.**"])
            exp.intern("watch.tasks.fusion.**")

            # Attempt to standardize some form of package metadata that can
            # allow for model importing with fewer hard-coding requirements
            package_header = {
                'version': '0.0.1',
                'arch_name': arch_name,
                'module_name': module_name,
            }
            exp.save_pickle(
                'kitware_package_header', 'kitware_package_header.pkl',
                package_header
            )

            exp.save_pickle(module_name, arch_name, model)


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
