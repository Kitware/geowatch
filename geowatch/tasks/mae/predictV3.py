"""
Basline Example:

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    MAE_MODEL_FPATH="$DVC_EXPT_DPATH/models/wu/wu_mae_2023_04_21/Drop6-epoch=01-val_loss=0.20.ckpt"
    KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD

    python -m geowatch.utils.simple_dvc request "$MAE_MODEL_FPATH"

    # NOTE: different predict files correspond to different models
    # TODO: make the model size a parameter (or better yet inferred)

    python -m geowatch.tasks.mae.predictV3 \
        --device="cuda:0"\
        --mae_ckpt_path="$MAE_MODEL_FPATH"\
        --input_kwcoco="$KWCOCO_BUNDLE_DPATH/imganns-KR_R001.kwcoco.zip"\
        --output_kwcoco="$KWCOCO_BUNDLE_DPATH/imganns-KR_R001-testmae.kwcoco.zip"\
        --window_space_scale=1.0 \
        --workers=8 \
        --io_workers=8

    # After your model predicts the outputs, you should be able to use the
    # geowatch visualize tool to inspect your features.
    python -m geowatch visualize "$KWCOCO_BUNDLE_DPATH/imganns-KR_R001-testmae.kwcoco.zip" \
        --channels "red|green|blue,mae.8:11,mae.14:17" --stack=only --workers=avail --animate=True \
        --draw_anns=False

"""
import ubelt as ub
import scriptconfig as scfg
import albumentations as A
import kwcoco
import kwimage
import ndsampler
import sys
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import L1Loss as MSE
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from kwutil import util_parallel
from geowatch.tasks.fusion.predict import CocoStitchingManager


class WatchDataset(Dataset):
    S2_l2a_channel_names = [
        'B02.tif', 'B01.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B09.tif', 'B11.tif', 'B12.tif', 'B8A.tif'
    ]
    S2_channel_names = [
        'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    ]
    L8_channel_names = [
        'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    ]

    def __init__(self, coco_dset, sensor=['S2'], bands=['shared'],
                 segmentation=False, patch_size=224, mask_patch_size=16, num_images=2,
                 mode='train', patch_overlap=.25, bas=True, rng=None, mask_pct=.5, mask_time_width=2,
                 temporal_mode='cat', window_space_scale=1.0):
        super().__init__()
        if not isinstance(bands, list):
            bands = [bands]
        if not isinstance(sensor, list):
            sensor = [sensor]
        assert (temporal_mode in ['cat', 'stack'])

        # initialize dataset
        print('load dataset')
        self.coco_dset: kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(coco_dset)

        print('filter dataset')
        # Filter out worldview images (better to use subset than remove)
        images: kwcoco.coco_objects1d.Images = self.coco_dset.images()
        flags = [s in sensor for s in images.lookup('sensor_coarse')]
        valid_image_ids : list[int] = list(images.compress(flags))
        self.coco_dset = self.coco_dset.subset(valid_image_ids)

        self.images : kwcoco.coco_objects1d.Images = self.coco_dset.images()
        self.sampler = ndsampler.CocoSampler(self.coco_dset)

        window_dims = [patch_size, patch_size]
        time_dims = num_images

        NEW_GRID = 1
        if NEW_GRID:
            print('make grid')
            from geowatch.tasks.fusion.datamodules.kwcoco_video_data import sample_video_spacetime_targets
            sample_grid = sample_video_spacetime_targets(
                self.coco_dset, window_dims=window_dims,
                time_dims=time_dims, window_overlap=patch_overlap,
                time_sampling='hardish3', time_span='1y',
                use_annot_info=False,
                keepbound=True,
                use_centered_positives=False,
                window_space_scale=window_space_scale
            )

            samples = sample_grid['targets']
            for tr in samples:
                tr['vidid'] = tr['video_id']  # hack
            print('made grid')
        else:
            grid = self.sampler.new_sample_grid(**{
                'task': 'video_detection',
                'window_dims': [num_images, patch_size, patch_size],
                'window_overlap': patch_overlap,
            })
            if segmentation:
                samples = grid['positives']
            else:
                samples = grid['positives'] + grid['negatives']

        # vidid_to_patches = ub.group_items(samples, key=lambda x: x['vidid'])
        # self.vidid_to_patches = vidid_to_patches
        print('build patches')
        grouped = ub.group_items(
                samples,
                lambda x: tuple(
                    [x['vidid']] + [gid for gid in x['gids']]
                )
                )
        grouped = ub.sorted_keys(grouped)
        self.patches : list[dict] = list(ub.flatten(grouped.values()))
        self.bands = []
        # no channels selected
        if len(bands) < 1:
            raise ValueError(f'bands must be specified. Options are {", ".join(bands)}, or all')
        # all channels selected
        elif len(bands) == 1:
            if bands[0].lower() == 'all':
                self.bands = bands
            elif bands[0].lower() == 'shared':
                self.bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']
            elif bands[0] == 'r|g|b':
                self.bands.append('r|g|b')

        self.num_channels = len(self.bands)
        self.bands = "|".join(self.bands)

        # define augmentations
        print('build augs')
        additional_targets = dict()
        self.num_images = num_images

        for i in range(self.num_images):
            additional_targets['image{}'.format(1 + i)] = 'image'
            additional_targets['seg{}'.format(i + 1)] = 'mask'

        self.transforms = A.NoOp()

        self.mode = mode
        self.segmentation = segmentation
        self.patch_size = patch_size
        self.bas = bas
        if self.bas:
            self.positive_indices = [0, 1, 3]
            self.ignore_indices = [2, 6]
        else:
            self.positive_indices = [0, 1, 2, 3]
            self.ignore_indices = [6]
        print('finished dataset init')

        self.temporal_mode = temporal_mode

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        #if idx > 500: raise IndexError
        tr : dict = self.patches[idx]
        tr['channels'] = self.bands
        tr = self.update_target_properties(tr)
        # vidid = tr['vidid']
        gids : list[int] = tr['gids']

        sample = self.sampler.load_sample(tr, nodata='float')
        images : np.ndarray = sample['im']
        std = np.nanstd(images)
        mean = np.nanmean(images)
        if std != 0:
            images = np.nan_to_num((images - mean) / std)
        else:
            images = np.zeros_like(images)

        if self.temporal_mode == 'cat':
            images = torch.cat([torch.tensor(x) for x in images], dim=0).permute(2, 0, 1)
        else:
            images = torch.tensor(images).permute(0, 3, 1, 2)

        vidspace_box = kwimage.Box.from_slice(tr['space_slice'])

        scale_outspace_from_vidspace = tr['scale'] / 4  # Add it back
        outspace_box = vidspace_box.scale(scale_outspace_from_vidspace).quantize().astype(np.int32)
        item = dict()
        im1_id = gids[0]
        img_obj1 : dict = self.coco_dset.index.imgs[im1_id]
        video_obj = self.coco_dset.index.videos[img_obj1['video_id']]

        full_stitch_vidspace_box = kwimage.Box.coerce([0, 0, video_obj['width'], video_obj['height']], format='xywh')
        full_stitch_outspace_box = full_stitch_vidspace_box.scale(scale_outspace_from_vidspace).quantize().astype(np.int32)

        item['full_stitch_outspace_ltrb'] = torch.from_numpy(full_stitch_outspace_box.data)
        item['sample_outspace_ltrb'] = torch.from_numpy(outspace_box.data)
        item['scale_outspace_from_vid'] = scale_outspace_from_vidspace

        return images, item

    def update_target_properties(self, target):
        """
        Populate the target so it has the correct input scale and bands.
        """
        # Handle target scale
        from geowatch.tasks.fusion.datamodules import data_utils
        gids : list[int] = target['gids']
        im1_id = gids[0]
        img_obj1 : dict = self.coco_dset.index.imgs[im1_id]
        video_obj = self.coco_dset.index.videos[img_obj1['video_id']]
        vidspace_gsd = video_obj.get('target_gsd', None)
        resolved_input_scale = data_utils.resolve_scale_request(request=1.0, data_gsd=vidspace_gsd)
        target['scale'] = resolved_input_scale['scale']
        target['channels'] = self.bands
        target['_input_gsd'] = resolved_input_scale['gsd']
        target['_native_video_gsd'] = resolved_input_scale['data_gsd']
        return target


class MAEPredictConfig(scfg.DataConfig):
    """
    Configuration for WashU MAE models
    """
    device = scfg.Value('cuda:0', type=str)
    mae_ckpt_path = scfg.Value(None, type=str)
    batch_size = scfg.Value(1, type=int)
    workers = scfg.Value(4, help=ub.paragraph(
            '''
            number of background data loading workers
            '''), alias=['num_workers'])
    io_workers = scfg.Value(8, help=ub.paragraph(
            '''
            number of background data writing workers
            '''), alias=['write_workers'])

    window_resolution = scfg.Value(1.0, help='The window GSD to build the grid at', alias=['window_space_scale'])

    sensor = scfg.Value(['S2', 'L8'], nargs='+')
    bands = scfg.Value(['shared'], type=str, help=ub.paragraph(
            '''
            Choose bands on which to train. Can specify 'all' for all
            bands from given sensor, or 'share' to use common bands when
            using both S2 and L8 sensors
            '''), nargs='+')
    patch_overlap = scfg.Value(0.25, type=float)
    input_kwcoco = scfg.Value(None, type=str, required=True, help=ub.paragraph(
            '''
            Path to kwcoco dataset with images to generate feature for
            '''))
    output_kwcoco = scfg.Value(None, type=str, required=True, help=ub.paragraph(
            '''
            Path to write an output kwcoco file. Output file will be a
            copy of input_kwcoco with addition feature fields generated
            by predict.py rerooted to point to the original data.
            '''))

    assets_dname = scfg.Value('_assets', help=ub.paragraph(
        '''
        The name of the top-level directory to write new assets.
        '''))


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, channels=6, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (f pf) c (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return x


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=8,
        decoder_heads=8,
        decoder_dim_head=64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4)
        #self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_emb = nn.Parameter(torch.randn(num_patches, decoder_dim))
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.out = nn.Sigmoid()

    def forward(self, img):

        patches = self.to_patch(img)

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(tokens.shape[1] + 1)]

        encoded_tokens = self.encoder.transformer(tokens)

        encoded_tokens = rearrange(encoded_tokens, 'b (f h w) d -> b f h w d', h=32, w=32)

        return encoded_tokens


class MaeCityscape(LightningModule):
    def __init__(self, dataset, **kwargs):
        super().__init__()
        self.vit = ViT(
            image_size=128,
            image_patch_size=4,
            frames=4,
            frame_patch_size=1,
            dim=64,
            depth=12,
            heads=12,
            mlp_dim=1024,
            dropout=0.1
        )
        self.model = MAE(
            encoder=self.vit,
            masking_ratio=0.90,
            decoder_dim=128,
            decoder_depth=6,
        )
        self.dataset = dataset
        self.batch_size = kwargs.get('batch_size', 4)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)
        self.acc = MSE()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        (x, masks), dates = batch

        pred, gt, viz, mi, ui = self(x)
        #gt = repeat(gt, 'b n d -> b (n n2) d', n2= 2)
        batch_range = torch.arange(x.shape[0], device=x.device)[:, None]
        loss = 0.999 * self.acc(pred[batch_range, mi], gt[batch_range, mi]) + 0.001 * self.acc(pred[batch_range, ui], gt[batch_range, ui])
        #loss = self.acc(pred, gt)
        return loss, viz, pred, gt


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


class Predict():
    def __init__(self, args):

        self.device = args.device
        self.data_path = args.input_kwcoco
        self.dataset = WatchDataset(self.data_path, sensor=args.sensor, bands=args.bands,
                                    segmentation=False, patch_size=128, mask_patch_size=16, num_images=4,
                                    mode='train', mask_pct=0.5, patch_overlap=args.patch_overlap,
                                    temporal_mode='stack',
                                    mask_time_width=2, window_space_scale=args.window_resolution)

        print("Dataset load finished ...")

        self.model = MaeCityscape(self.dataset)

        self.model = self.model.load_from_checkpoint(args.mae_ckpt_path, dataset=self.dataset)
        print("Model load finished ...")

        self.dataloader = DataLoader(self.dataset,
                                     shuffle=False,
                                     batch_size=1,
                                     num_workers=args.workers,
                                     persistent_workers=False,
                                     pin_memory=True)

        print('copy dataset')
        self.output_dset = self.dataset.coco_dset.copy()

        print('reroot')
        self.output_dset.reroot(absolute=True)
        self.output_dset.fpath = args.output_kwcoco
        self.output_dset.reroot(absolute=False)

        self.save_channels = 'mae.0:16'
        self.output_kwcoco_path = ub.Path(self.output_dset.fpath)
        out_folder = self.output_kwcoco_path.parent
        self.output_feat_dpath = (out_folder / args.assets_dname).ensuredir()

        self.imwrite_kw = {
            'compress': 'DEFLATE',
            'backend': 'gdal',
            'blocksize': 128,
        }

        self.stitch_manager = CocoStitchingManager(
            result_dataset=self.output_dset,
            short_code=args.assets_dname,
            chan_code=self.save_channels,
            stiching_space='video',
            prob_compress=self.imwrite_kw['compress'],
            quantize=True,
        )

        from geowatch.utils import process_context
        self.proc_context = process_context.ProcessContext(
            args=sys.argv,
            type='process',
            name='geowatch.tasks.mae.predict',
        )

    def __call__(self):

        writer_queue = util_parallel.BlockingJobQueue(max_workers=4)
        self.stitch_manager.writer_queue = writer_queue

        self.proc_context.start()

        self.proc_context.add_disk_info(ub.Path(self.output_dset.fpath).parent)
        self.output_dset.dataset.setdefault('info', [])
        self.output_dset.dataset['info'].append(self.proc_context.obj)

        print('Evaluating and saving features')

        self.model.eval()
        self.model.to(self.device)
        num_batches = len(self.dataloader)
        preds = []
        with torch.no_grad():
            seen_images = set()
            prog = ub.ProgIter(enumerate(self.dataloader), total=num_batches, desc='Compute features', verbose=1)
            for batch_idx, batch in prog:
                x, item = batch
                x = x.to(self.device)
                #x2 = rearrange(x, 'b (f pf) c h w -> b (pf f) c h w', pf=2)

                pred = self.model(x)
                #pred2 = self.model(x2)
                preds = pred.cpu().detach().numpy()
                #preds2 = pred2.cpu().detach().numpy()

                target = self.dataset.patches[batch_idx]

                new_complete_gids = target.get('new_complete_gids', [])

                for gid in new_complete_gids:
                    assert gid not in seen_images
                    seen_images.add(gid)
                    self.stitch_manager.submit_finalize_image(gid)

                gid1, gid2, gid3, gid4 = target['gids']

                sample_outspace_ltrb = kwimage.Box.coerce(item['sample_outspace_ltrb'].numpy(), format='ltrb')
                full_stitch_outspace_box = kwimage.Box.coerce(item['full_stitch_outspace_ltrb'].numpy(), format='ltrb')
                scale_outspace_from_vid = item['scale_outspace_from_vid'].numpy()[0]
                outspace_slice = sample_outspace_ltrb.to_slice()
                outspace_dsize = full_stitch_outspace_box.dsize

                feat1 = preds[:, 0, :, :, :].squeeze()
                feat2 = preds[:, 1, :, :, :].squeeze()
                feat3 = preds[:, 2, :, :, :].squeeze()
                feat4 = preds[:, 3, :, :, :].squeeze()

                self.stitch_manager.accumulate_image(
                    gid1, outspace_slice, feat1,
                    dsize=outspace_dsize,
                    scale=scale_outspace_from_vid)

                self.stitch_manager.accumulate_image(
                    gid2, outspace_slice, feat2,
                    dsize=outspace_dsize,
                    scale=scale_outspace_from_vid)

                self.stitch_manager.accumulate_image(
                    gid3, outspace_slice, feat3,
                    dsize=outspace_dsize,
                    scale=scale_outspace_from_vid)

                self.stitch_manager.accumulate_image(
                    gid4, outspace_slice, feat4,
                    dsize=outspace_dsize,
                    scale=scale_outspace_from_vid)

            print('Finalize already compelted jobs')
            writer_queue.wait_until_finished(desc='Finalize submitted jobs')

            for gid in ub.ProgIter(list(self.stitch_manager.image_stitchers.keys()), desc='submit loose write jobs'):
                if gid not in seen_images:
                    seen_images.add(gid)
                    self.stitch_manager.submit_finalize_image(gid)

            print('Finalize loose jobs')
            writer_queue.wait_until_finished()

        print('Finish process context')
        self.proc_context.add_device_info(self.device)
        self.proc_context.stop()

        print('Write to dset.fpath = {!r}'.format(self.output_dset.fpath))
        self.output_dset.dump(self.output_dset.fpath, newlines=True)
        print('Done')

        return


def main():
    args = MAEPredictConfig.cli()
    predict = Predict(args)
    predict()


if __name__ == '__main__':
    """
    SeeAlso:
        ../../cli/prepare_teamfeats.py

        # Team Features on Drop3
        DVC_DPATH=$(geowatch_dvc)
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10
        python -m geowatch.cli.prepare_teamfeats \
            --base_fpath=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --with_depth=0 \
            --with_landcover=0 \
            --with_materials=0  \
            --with_invariants=1 \
            --do_splits=0 \
            --gres=0 --backend=serial --run=1

    CommandLine:
        python -m geowatch.tasks.template.predict --help

        DVC_DPATH=$(geowatch_dvc)
        PRETEXT_PATH=$DVC_DPATH/models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pretext_package.pt
        SSEG_PATH=$DVC_DPATH/models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt
        PCA_FPATH=$DVC_DPATH/models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pca_projection_matrix.pt
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
        python -m geowatch.tasks.invariants.predict \
            --pretext_package_path "$PRETEXT_PATH" \
            --segmentation_package_path "$SSEG_PATH" \
            --pca_projection_path "$PCA_FPATH" \
            --input_kwcoco $KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --workers=avail \
            --do_pca 0 \
            --patch_overlap=0.3 \
            --output_kwcoco $KWCOCO_BUNDLE_DPATH/uky_invariants.kwcoco.json \
            --tasks before_after pretext

        python -m geowatch stats $KWCOCO_BUNDLE_DPATH/uky_invariants.kwcoco.json

        python -m geowatch visualize $KWCOCO_BUNDLE_DPATH/uky_invariants/invariants_nowv_vali.kwcoco.json \
            --channels "invariants.7,invariants.6,invariants.5" --animate=True \
            --select_images '.sensor_coarse != "WV"' --draw_anns=False

    Ignore:
        ### Command 1 / 2 - geowatch-teamfeat-job-0
        python -m geowatch.tasks.invariants.predict \
            --input_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_kr1br2.kwcoco.json" \
            --output_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_kr1br2_uky_invariants.kwcoco.json" \
            --pretext_package_path "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_package.pt" \
            --pca_projection_path  "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt" \
            --do_pca 0 \
            --patch_overlap=0.0 \
            --workers="2" \
            --io_workers 0 \
            --tasks before_after pretext

        cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
        kwcoco subset --src=data.kwcoco.json --dst=AE_R001.kwcoco.json --select_videos='.name == "AE_R001"'
        kwcoco subset --src=data.kwcoco.json --dst=NZ_R001.kwcoco.json --select_videos='.name == "NZ_R001"'

        python -m geowatch.tasks.mae.predict \
        --device="cuda:0"\
        --mae_ckpt_path="/storage1/fs1/jacobsn/Active/user_s.sastry/smart_watch/new_models/checkpoints/Drop6-epoch=01-val_loss=0.20.ckpt"\
        --input_kwcoco="$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/data_train_I2L_split6.kwcoco.zip"\
        --output_kwcoco="$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/mae_v1_train_split6.kwcoco.zip"\
        --window_space_scale=1.0 \
        --workers=8 \
        --io_workers=8

        python -m geowatch visualize $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/mae_v1_train_split6.kwcoco.zip \
        --channels "red|green|blue,mae.8:11,mae.14:17" --stack=only --workers=avail --animate=True \
        --draw_anns=False


    """
    main()
