import torch
from torch import nn
from einops.layers.torch import Rearrange
import einops

from torchvision import transforms

from watch.tasks.fusion.methods.common import ChangeDetectorBase, SemanticSegmentationBase
from watch.tasks.fusion.models import transformer
from watch.tasks.fusion import utils
import ubelt as ub

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class MultimodalTransformerDotProdCD(ChangeDetectorBase):
    """
    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from watch.tasks.fusion import datasets
        >>> datamodule = datasets.WatchDataModule(
        >>>     train_dataset='special:vidshapes8', num_workers=0)
        >>> datamodule.setup('fit')
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> self = MultimodalTransformerDotProdCD(model_name='smt_it_joint_p8')
        >>> images = batch[0]['frames'][0]['modes']['r|g|b'][None, :].float()
        >>> distance = self(images)
    """

    def __init__(self,
                 model_name,
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 input_stats=None,
                 pos_weight=1.,
                 attention_impl='exact',
                 window_size=8):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            input_stats=input_stats,
            pos_weight=pos_weight,
        )
        self.save_hyperparameters()

        encoder_config = transformer.encoder_configs[model_name]

        # TODO: pre-compute what the "token" feature dimension is.
        encoder = transformer.FusionEncoder(
            **encoder_config,
            attention_impl=attention_impl,
            dropout=dropout)
        self.encoder = encoder

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

    @profile
    def forward(self, images):
        # Break images up into patches
        patch_tokens = self.tokenize(images)
        # Add positional encodings for time, mode, and space.
        patch_tokens = self.add_encoding(patch_tokens)

        # preproced = self.batch_preprocessing_step(images)
        feats = self.encoder(patch_tokens)

        # similarity between neighboring timesteps
        feats = nn.functional.normalize(feats, dim=-1)
        similarity = torch.einsum("b t c h w f , b t c h w f -> b t c h w", feats[:, :-1], feats[:, 1:])
        similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        distance = -3.0 * similarity

        return distance

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(MultimodalTransformerDotProdCD, MultimodalTransformerDotProdCD).add_model_specific_args(parent_parser)

        parser.add_argument("--model_name", default='smt_it_joint_p8', type=str)
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


class MultimodalTransformerDirectCD(ChangeDetectorBase):
    """
    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from watch.tasks.fusion import datasets
        >>> datamodule = datasets.WatchDataModule(
        >>>     train_dataset='special:vidshapes8', num_workers=0)
        >>> datamodule.setup('fit')
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> self = MultimodalTransformerDirectCD(model_name='smt_it_joint_p8')
        >>> images = batch[0]['frames'][0]['modes']['r|g|b'][None, :].float()
        >>> similarity = self(images)

        # nh.util.number_of_parameters(self)
    """

    def __init__(self,
                 model_name,
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 pos_weight=1.,
                 input_stats=None,
                 attention_impl='exact',
                 window_size=8):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
            input_stats=input_stats,
        )
        self.save_hyperparameters()

        encoder_config = transformer.encoder_configs[model_name]
        encoder = transformer.FusionEncoder(
            **encoder_config,
            attention_impl=attention_impl,
            dropout=dropout,
        )
        self.encoder = encoder
        self.binary_clf = nn.LazyLinear(1)

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

    def forward(self, images):

        # Break images up into patches
        patch_tokens = self.tokenize(images)
        # Add positional encodings for time, mode, and space.
        patch_tokens = self.add_encoding(patch_tokens)

        # TODO: maybe make the encoder return a sequence of 1 less?
        # Rather than just ignoring the first output?
        fused_feats = self.encoder(patch_tokens)[:, 1:]
        similarity = self.binary_clf(fused_feats)[..., 0]  # only one prediction
        similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        return similarity

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(MultimodalTransformerDirectCD, MultimodalTransformerDirectCD).add_model_specific_args(parent_parser)

        parser.add_argument("--model_name", default='smt_it_stm_p8', type=str)
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


class MultimodalTransformerSegmentation(SemanticSegmentationBase):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from watch.tasks.fusion import datasets
        >>> datamodule = datasets.WatchDataModule(
        >>>     train_dataset='special:vidshapes8', num_workers=0)
        >>> datamodule.setup('fit')
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))
        >>> classes = datamodule.coco_datasets['train'].object_categories()
        >>> n_classes = len(classes)
        >>> self = MultimodalTransformerSegmentation(n_classes=n_classes, model_name='smt_it_joint_p8')
        >>> images = batch['images'].float()
        >>> logits = self(images)

    TODO:
        - [ ] Dont pass the number of classes, instead pass the list of class
              names. This helps prevent mistakes in production.
    """

    def __init__(self,
                 n_classes,
                 model_name,
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 window_size=8):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters()

        encoder_config = transformer.encoder_configs[model_name]
        encoder = transformer.FusionEncoder(**encoder_config, dropout=dropout)
        self.encoder = encoder
        self.classifier = nn.LazyLinear(n_classes)

    @property
    def preprocessing_step(self):
        return transforms.Compose([
            utils.Lambda(lambda x: torch.from_numpy(x).float()),
            Rearrange("(h hs) (w ws) c -> c h w (ws hs)",
                      hs=self.hparams.window_size,
                      ws=self.hparams.window_size),
            utils.SinePositionalEncoding(3, 0, sine_pairs=4),
            utils.SinePositionalEncoding(3, 1, sine_pairs=4),
            utils.SinePositionalEncoding(3, 2, sine_pairs=4),
        ])

    def batch_preprocessing_step(self, images):
        return transforms.Compose([
            # utils.Lambda(lambda x: torch.from_numpy(x).float()),
            Rearrange("b c (h hs) (w ws) -> b c h w (ws hs)",
                      hs=self.hparams.window_size,
                      ws=self.hparams.window_size),
            utils.SinePositionalEncoding(4, 1, sine_pairs=4),
            utils.SinePositionalEncoding(4, 2, sine_pairs=4),
            utils.SinePositionalEncoding(4, 3, sine_pairs=4),
        ])(images)

    # @pl.core.decorators.auto_move_data
    def forward(self, images):
        preproced = self.batch_preprocessing_step(images)
        features = self.encoder(preproced)
        logits = self.classifier(features).mean(dim=1)
        logits = einops.rearrange(logits, "b h w c -> b c h w")
        logits = nn.functional.interpolate(
            logits,
            scale_factor=[self.hparams.window_size, self.hparams.window_size],
            mode="bilinear")
        return logits

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(MultimodalTransformerSegmentation, MultimodalTransformerSegmentation).add_model_specific_args(parent_parser)

        parser.add_argument("--model_name", type=str)
        parser.add_argument("--n_classes", type=int)
        parser.add_argument("--dropout", default=0.0, type=float)
        # parser.add_argument("--input_scale", default=255.0, type=float)
        parser.add_argument("--window_size", default=8, type=int)
        return parent_parser
