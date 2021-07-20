import torch
from torch import nn
from einops.layers.torch import Rearrange
import einops

from torchvision import transforms

from watch.tasks.fusion.methods.common import ChangeDetectorBase, SemanticSegmentationBase
from watch.tasks.fusion.models import transformer
from watch.tasks.fusion import utils


class MultimodalTransformerDotProdCD(ChangeDetectorBase):

    def __init__(self,
                 model_name,
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 pos_weight=1.,
                 window_size=8):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
        )
        self.save_hyperparameters()

        self.model = getattr(transformer, model_name)(dropout=dropout)

    @property
    def preprocessing_step(self):
        return transforms.Compose([
            Rearrange("t c (h hs) (w ws) -> t c h w (ws hs)",
                      hs=self.hparams.window_size,
                      ws=self.hparams.window_size),
            utils.SinePositionalEncoding(4, 0, sine_pairs=4),
            utils.SinePositionalEncoding(4, 1, sine_pairs=4),
            utils.SinePositionalEncoding(4, 2, sine_pairs=4),
            utils.SinePositionalEncoding(4, 3, sine_pairs=4),
        ])

    # @pl.core.decorators.auto_move_data
    def forward(self, images):
        feats = self.model(images)

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
        return parent_parser


class MultimodalTransformerDirectCD(ChangeDetectorBase):

    def __init__(self,
                 model_name,
                 dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=0.,
                 pos_weight=1.,
                 window_size=8):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
        )
        self.save_hyperparameters()

        self.model = nn.Sequential(
            getattr(transformer, model_name)(dropout=dropout),
            nn.LazyLinear(1),
        )

    @property
    def preprocessing_step(self):
        return transforms.Compose([
            Rearrange("t c (h hs) (w ws) -> t c h w (ws hs)",
                      hs=self.hparams.window_size,
                      ws=self.hparams.window_size),
            utils.SinePositionalEncoding(4, 0, sine_pairs=4),
            utils.SinePositionalEncoding(4, 1, sine_pairs=4),
            utils.SinePositionalEncoding(4, 2, sine_pairs=4),
            utils.SinePositionalEncoding(4, 3, sine_pairs=4),
        ])

    # @pl.core.decorators.auto_move_data
    def forward(self, images):
        similarity = self.model(images)[:, 1:, ..., 0]
        similarity = einops.reduce(similarity, "b t c h w -> b t h w", "mean")
        return similarity

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(MultimodalTransformerDirectCD, MultimodalTransformerDirectCD).add_model_specific_args(parent_parser)

        parser.add_argument("--model_name", default='smt_it_stm_p8', type=str)
        parser.add_argument("--dropout", default=0.1, type=float)
        # parser.add_argument("--input_scale", default=2000.0, type=float)
        parser.add_argument("--window_size", default=8, type=int)
        return parent_parser


class MultimodalTransformerSegmentation(SemanticSegmentationBase):

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

        self.model = nn.Sequential(
            getattr(transformer, model_name)(dropout=dropout),
            nn.LazyLinear(n_classes),
        )

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

    # @pl.core.decorators.auto_move_data
    def forward(self, images):
        logits = self.model(images).mean(dim=1)
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
