import torch
from torch import nn
from einops.layers.torch import Rearrange
import einops

from torchvision import transforms

from watch.tasks.fusion.architectures import transformer
from watch.tasks.fusion import utils
import ubelt as ub
import netharn as nh
import kwarray
import kwcoco
from kwcoco import channel_spec
import torchmetrics
import pytorch_lightning as pl
import torch_optimizer as optim
from torch.optim import lr_scheduler

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def _benchmark_model():
    # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    # TODO: profile attention_impl
    import netharn as nh
    from watch.tasks.fusion import datamodules
    import torch.profiler
    from torch.profiler import profile, ProfilerActivity, record_function
    datamodule = datamodules.KWCocoVideoDataModule(
        train_dataset='special:vidshapes8', num_workers=0)
    datamodule.setup('fit')
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
    frames = batch[0]['frames']
    collate_images = torch.cat([frame['modes']['r|g|b'][None, :].float() for frame in frames], dim=0)
    device = nh.XPU.coerce('cpu').main_device
    device = nh.XPU.coerce('gpu').main_device
    #device = nh.XPU.coerce('auto').main_device
    images = collate_images[None, :].to(device)

    input_grid = list(ub.named_product({
        'S': [32, 64, 96, 128],
        # 'T': [2, 3, 5, 9],
        # 'T': [2, 5, 9],
        # 'T': [2, 5, 9],
        'T': [2],
        'M': [3, 32, 64],
    }))

    model_grid = list(ub.named_product({
        'arch_name': ['smt_it_stm_p8', 'smt_it_joint_p8', 'smt_it_hwtm_p8'],
        'attention_impl': ['exact', 'performer'],
    }))
    import itertools as it
    bench_grid = list(it.product(model_grid, input_grid))

    rows = []
    nicerows = []
    self = None
    images = None
    train_prof = None
    output = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()

    # Pure memory benchmarks
    for modelkw, inputkw in bench_grid:
        # for arch_name in ['smt_it_stm_p8']:
        M = inputkw['M']
        T = inputkw['T']
        S = inputkw['S']
        row = {}
        row.update(inputkw)
        row.update(modelkw)

        images = torch.rand(1, T, M, S, S).to(device)

        errored = False

        try:
            self = MultimodalTransformer(input_channels=M, **modelkw)
            num_params = nh.util.number_of_parameters(self)
            self = self.to(device)
            optim = torch.optim.SGD(self.parameters(), lr=1e-9)
            # with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as train_prof:
            # with torch.profiler.profile(activities=[ProfilerActivity.CUDA], record_shapes=False, profile_memory=True) as train_prof:
            #     with record_function(f"train_{arch_name}"):
            optim.zero_grad()
            output = self(images)['change']
            output.sum().backward()
            optim.step()

            # total_memory = sum(event.cuda_memory_usage for event in train_prof.events())
            # total_mem_str = xdev.byte_str(total_memory)
            # print(total_mem_str)

            row.update({
                'num_params': num_params,
            })
            mem_stats = ({
                'max_mem_alloc': torch.cuda.max_memory_allocated(),
                'mem_alloc': torch.cuda.memory_allocated(),
                'mem_reserve': torch.cuda.memory_reserved(),
                'max_mem_reserve': torch.cuda.max_memory_reserved(),
            })
            row.update(mem_stats)
        except RuntimeError:
            errored = True
            pass

        self = None
        images = None
        train_prof = None
        output = None
        optim = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()

        if not errored:
            rows.append(row)
            nicerow = row.copy()
            nicestats = {k + '_str': xdev.byte_str(v) if isinstance(v, int) else v for k, v in mem_stats.items()}
            nicerow.update(nicestats)
            nicerows.append(nicerow)
            print(nicerow)

    import pandas as pd
    df = (pd.DataFrame(nicerows))
    df = df.sort_values('max_mem_alloc')
    print(df)

    for k, subdf in df.groupby(['arch_name', 'attention_impl']):
        print('')
        print('k = {!r}'.format(k))
        print(subdf.pivot(['S', 'T'], ['M'], ['max_mem_alloc_str']))

    import timerit
    ti = timerit.Timerit(3, bestof=1, verbose=2)
    #
    for arch_name in ['smt_it_stm_p8', 'smt_it_joint_p8', 'smt_it_hwtm_p8']:
        print('====')
        self = MultimodalTransformer(arch_name=arch_name, input_channels=datamodule.channels)
        num_params = nh.util.number_of_parameters(self)
        print('arch_name = {!r}'.format(arch_name))
        print('num_params = {!r}'.format(num_params))
        print('running')
        self = self.to(device)
        output = self(images)
        for timer in ti.reset(f'inference-{arch_name}'):
            torch.cuda.synchronize()
            with timer:
                output = self(images)['change']
                torch.cuda.synchronize()
        for timer in ti.reset(f'train-{arch_name}'):
            torch.cuda.synchronize()
            with timer:
                output = self(images)['change']
                output.sum().backward()
                torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as pred_prof:
            with record_function(f"pred_{arch_name}"):
                output = self(images)['change']
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as train_prof:
            with record_function(f"train_{arch_name}"):
                output = self(images)['change']
                output.sum().backward()
        print('arch_name = {!r}'.format(arch_name))
        print('num_params = {!r}'.format(num_params))
        print(pred_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(train_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        total_memory = sum(event.cuda_memory_usage for event in train_prof.events())
        total_mem_str = xdev.byte_str(total_memory)
        print(total_mem_str)


class MultimodalTransformer(pl.LightningModule):
    """
    CommandLine:
        xdoctest -m watch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer

    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from watch.tasks.fusion import datamodules
        >>> datamodule = datamodules.KWCocoVideoDataModule(
        >>>     train_dataset='special:vidshapes8', num_workers=0)
        >>> datamodule.setup('fit')
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
        >>> self = MultimodalTransformer(arch_name='smt_it_stm_p8', input_channels=datamodule.channels)
        >>> import netharn as nh
        >>> # device = nh.XPU.coerce('auto')
        >>> device = nh.XPU.coerce('cpu').main_device
        >>> frames = batch[0]['frames']
        >>> collate_images = torch.cat([frame['modes']['r|g|b'][None, :].float() for frame in frames], dim=0)
        >>> images = collate_images[None, :].to(device)
        >>> self = self.to(device)
        >>> # Run forward pass
        >>> output = self(images)
        >>> num_params = nh.util.number_of_parameters(self)
        >>> print('num_params = {!r}'.format(num_params))
        >>> import torch.profiler
        >>> from torch.profiler import profile, ProfilerActivity
        >>> with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        >>>     with torch.profiler.record_function("model_inference"):
        >>>         output = self(images)
        >>> print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    """

    def __init__(self,
                 arch_name='smt_it_stm_p8',
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 pos_weight=1.,
                 input_stats=None,
                 input_channels=None,
                 attention_impl='exact',
                 window_size=8,
                 classes=10):

        super().__init__()
        self.save_hyperparameters()

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
        num_channels = sum(ub.map_vals(len, input_channels.normalize().parse()).values())

        # criterion and metrics
        import monai
        # self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        self.class_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)
        self.change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=False)

        # self.change_criterion = nn.BCEWithLogitsLoss(
        #         pos_weight=torch.ones(1) * pos_weight)

        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.BCEWithLogitsLoss()

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
        print('in_features_raw = {!r}'.format(in_features_raw))
        in_features_pos = (2 * 4 * 4)  # positional encoding feature
        print('in_features_pos = {!r}'.format(in_features_pos))
        in_features = in_features_pos + in_features_raw
        encoder_config = transformer.encoder_configs[arch_name]

        encoder = transformer.FusionEncoder(
            **encoder_config,
            in_features=in_features,
            attention_impl=attention_impl,
            dropout=dropout,
        )
        self.encoder = encoder

        feat_dim = self.encoder.out_features

        # A simple linear layer that learns to combine channels
        self.channel_fuser = nh.layers.MultiLayerPerceptronNd(
            0, num_channels, [], 1, norm=None)

        # TODO:
        #     - [X] Classifier MLP, skip connections
        #     - [ ] Decoder
        #     - [ ] Dynamic / Learned embeddings
        # self.binary_clf = nn.LazyLinear(1)  # TODO: rename to change_clf
        # self.class_clf = nn.LazyLinear(len(self.classes))  # category classifier
        self.change_clf = nh.layers.MultiLayerPerceptronNd(
            0, feat_dim, [], 1, norm=None)
        self.class_clf = nh.layers.MultiLayerPerceptronNd(
            0, feat_dim, [], self.num_classes, norm=None)

        self.tokenize = Rearrange("b t c (h hs) (w ws) -> b t c h w (ws hs)",
                                  hs=self.hparams.window_size,
                                  ws=self.hparams.window_size)

        self.move_channels_last = Rearrange("b t c h w f -> b t h w f c")

        encode_t = utils.SinePositionalEncoding(5, 1, sine_pairs=4)
        encode_m = utils.SinePositionalEncoding(5, 2, sine_pairs=4)
        encode_h = utils.SinePositionalEncoding(5, 3, sine_pairs=4)
        encode_w = utils.SinePositionalEncoding(5, 4, sine_pairs=4)
        self.add_encoding = transforms.Compose([
            encode_t, encode_m, encode_h, encode_w,
        ])

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
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--pos_weight", default=1.0, type=float)

        # Model names define the transformer encoder used by the method
        available_encoders = list(transformer.encoder_configs.keys())
        parser.add_argument("--arch_name", default='smt_it_stm_p8', type=str,
                            choices=available_encoders)
        parser.add_argument("--dropout", default=0.1, type=float)
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
        optimizer = optim.RAdam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.99))
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
        change_logits = self.change_clf(spacetime_fused_features[:, 1:])[..., 0]  # only one prediction
        class_logits = self.class_clf(spacetime_fused_features)

        logits = {
            'class': class_logits,
            'change': change_logits,
        }
        return logits

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
            >>>     classes=datamodule.classes, input_channels=datamodule.channels)
            >>> outputs = self.forward_step(batch, with_loss=True)
            >>> canvas = datamodule.draw_batch(batch, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoctest: +SKIP
            >>> # ============
            >>> # DEMO OVERFIT:
            >>> # ============
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> from watch.tasks.fusion import methods
            >>> from watch.tasks.fusion import datamodules
            >>> from watch.utils.slugify_ext import smart_truncate
            >>> import kwcoco
            >>> import os
            >>> import kwplot
            >>> sns = kwplot.autosns()
            >>> if 1:
            >>>     _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
            >>>     dvc_dpath = os.environ.get('DVC_DPATH', _default)
            >>>     coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/combo_propogated_data.kwcoco.json')
            >>> else:
            >>>     coco_fpath = 'special:vidshapes8-frames9-speed0.5-multispectral'
            >>> coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=coco_dset,
            >>>     channels='blue|green|red',
            >>>     chip_size=128, batch_size=1, time_steps=4,
            >>>     normalize_inputs=True, neg_to_pos_ratio=0, num_workers=0,
            >>> )
            >>> datamodule.setup('fit')
            >>> torch_dset = datamodule.torch_datasets['train']
            >>> loader = datamodule.train_dataloader()

            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.MultimodalTransformer(
            >>>     #arch_name='smt_it_joint_p8',
            >>>     arch_name='smt_it_stm_p8',
            >>>     attention_impl='exact',
            >>>     input_stats=datamodule.input_stats,
            >>>     classes=datamodule.classes, input_channels=datamodule.channels)
            >>> device = 0
            >>> self = self.to(device)

            >>> # Run one visualization
            >>> batch = next(iter(loader))
            >>> walker = ub.IndexableWalker(batch)
            >>> for path, val in walker:
            >>>     if isinstance(val, torch.Tensor):
            >>>         walker[path] = val.to(device)
            >>> outputs = self.training_step(batch)
            >>> canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=1, overlay_on_image=0)
            >>> kwplot.imshow(canvas)

            >>> loss_records = []
            >>> loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
            >>> step = 0
            >>> frame_idx = 0
            >>> dpath = ub.ensuredir('_overfit_viz09')
            >>> #optim = torch.optim.SGD(self.parameters(), lr=1e-4)
            >>> #optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
            >>> import torch_optimizer
            >>> optim = torch_optimizer.RAdam(self.parameters(), lr=3e-3, weight_decay=1e-5)

            >>> from kwplot.mpl_make import render_figure_to_image
            >>> import xdev
            >>> import kwimage
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> fig.set_size_inches(15, 6)
            >>> fig.subplots_adjust(left=0.05, top=0.9)
            >>> #for frame_idx in xdev.InteractiveIter(list(range(frame_idx + 1, 1000))):
            >>> for frame_idx in list(range(frame_idx, 1000)):
            >>>     num_steps = 20
            >>>     for i in ub.ProgIter(range(num_steps), desc='overfit'):
            >>>         optim.zero_grad()
            >>>         outputs = self.training_step(batch)
            >>>         outputs['item_losses']
            >>>         loss = outputs['loss']
            >>>         item_losses_ = nh.data.collate.default_collate(outputs['item_losses'])
            >>>         item_losses = ub.map_vals(lambda x: sum(x).item(), item_losses_)
            >>>         loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in item_losses.items()])
            >>>         loss.backward()
            >>>         optim.step()
            >>>         step += 1
            >>>     canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=1, overlay_on_image=0, max_items=4)
            >>>     kwplot.imshow(canvas, pnum=(1, 2, 1), fnum=1)
            >>>     fig = kwplot.figure(fnum=1, pnum=(1, 2, 2))
            >>>     #kwplot.imshow(canvas, pnum=(1, 2, 1))
            >>>     import pandas as pd
            >>>     ax = sns.lineplot(data=pd.DataFrame(loss_records), x='step', y='val', hue='part')
            >>>     ax.set_yscale('log')
            >>>     fig.suptitle(smart_truncate(str(optim).replace('\n',''), max_length=64))
            >>>     img = render_figure_to_image(fig)
            >>>     img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            >>>     fpath = join(dpath, 'frame_{:04d}.png'.format(frame_idx))
            >>>     kwimage.imwrite(fpath, img)
            >>>     #xdev.InteractiveIter.draw()
            >>> # TODO: can we get this batch to update in real time?
            >>> # TODO: start a server process that listens for new images
            >>> # as it gets new images, it starts playing through the animation
            >>> # looping as needed

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
            for frame in item['frames']:
                assert len(frame['modes']) == 1, 'only handle one mode for now'
                mode_key, mode_val = ub.peek(frame['modes'].items())
                mode_val = mode_val.float()
                # self.input_norms = None
                if self.input_norms is not None:
                    mode_norm = self.input_norms[mode_key]
                    mode_val = mode_norm(mode_val)
                frame_ims.append(mode_val)

            # Because we are not collating we need to add a batch dimension
            images = torch.stack(frame_ims)[None, ...]

            B, T, C, H, W = images.shape

            logits = self(images)

            # TODO: it may be faster to compute loss at the downsampled
            # resolution.
            class_logits_small = logits['class']
            change_logits_small = logits['change']

            change_logits = nn.functional.interpolate(
                change_logits_small, [H, W], mode="bilinear", align_corners=True)

            _tmp = einops.rearrange(class_logits_small, 'b t h w c -> b (t c) h w')
            _tmp2 = nn.functional.interpolate(
                _tmp, [H, W], mode="bilinear", align_corners=True)
            class_logits = einops.rearrange(_tmp2, 'b (t c) h w -> b t h w c', c=class_logits_small.shape[4])

            # Remove batch index in both cases
            change_prob = change_logits.sigmoid()[0]
            class_prob = class_logits.sigmoid()[0]

            item_change_probs.append(change_prob.detach())
            item_class_probs.append(class_prob.detach())

            item_loss_parts = {}

            if with_loss:
                true_changes = torch.stack([
                    frame['change'] for frame in item['frames'][1:]
                ])[None, ...]
                item_changes_truth.append(true_changes)

                true_class = torch.stack([
                    frame['class_idxs'] for frame in item['frames']
                ])[None, ...]
                item_classes_truth.append(true_class)

                # compute criterion
                # print('change_logits.shape = {!r}'.format(change_logits.shape))
                # print('true_changes.shape = {!r}'.format(true_changes.shape))

                change_loss = self.change_criterion(change_logits, true_changes.float()).mean()
                item_loss_parts['change'] = change_loss

                true_ohe = kwarray.one_hot_embedding(true_class.long(), self.num_classes, dim=-1).float()
                # y = true_ohe
                # x = class_logits

                class_loss = self.class_criterion(class_logits, true_ohe).mean()
                # class_loss = torch.nn.functional.binary_cross_entropy_with_logits(class_logits, true_ohe)
                item_loss_parts['class'] = class_loss

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
            >>> input_stats = datamodule.torch_datasets['train'].cached_input_stats()
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.MultimodalTransformer(
            >>>     "smt_it_stm_p8", classes=classes,
            >>>     input_stats=input_stats, input_channels=datamodule.channels)

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
