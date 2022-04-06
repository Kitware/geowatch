import kwimage
import kwarray
import torch
import ubelt as ub
from argparse import ArgumentParser, RawTextHelpFormatter
from tqdm import tqdm
import os
# local imports
from .pretext_model import pretext
from .data.datasets import gridded_dataset
from watch.utils.lightning_ext import util_globals
from watch.utils.lightning_ext import util_device
from .segmentation_model import segmentation_model as seg_model
from watch.utils import util_kwimage  # NOQA


class predict(object):
    """
    CommandLine:
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        DVC_DPATH=$DVC_DPATH xdoctest -m watch.tasks.invariants.predict predict

        python -m watch visualize $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/test_uky.kwcoco.json \
            --channels='invariants.0:3' --animate=True --with_anns=False

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.invariants.predict import *  # NOQA
        >>> import kwcoco
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> #  Write out smaller version of the dataset
        >>> dset = kwcoco.CocoDataset(dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_nowv_vali.kwcoco.json')
        >>> images = dset.videos(names=['KR_R001']).images[0]
        >>> sub_images = images.compress([s != 'WV' for s in images.lookup('sensor_coarse')])[::5]
        >>> sub_dset = dset.subset(sub_images)
        >>> sub_dset.fpath = (dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/small_test_data_nowv_vali.kwcoco.json')
        >>> sub_dset.dump(sub_dset.fpath)
        >>> input_kwcoco = sub_dset.fpath
        >>> output_kwcoco = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/test_uky.kwcoco.json'
        >>> pretext_package_path = dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pretext_package.pt'
        >>> pca_projection_path = dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pca_projection_matrix.pt'
        >>> segmentation_package_path = dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt'
        >>> argv = []
        >>> argv += ['--input_kwcoco', f'{sub_dset.fpath}']
        >>> argv += ['--output_kwcoco', f'{output_kwcoco}']
        >>> argv += ['--pca_projection_path', f'{pca_projection_path}']
        >>> argv += ['--pretext_package_path', f'{pretext_package_path}']
        >>> argv += ['--segmentation_package_path', f'{segmentation_package_path}']
        >>> argv += ['--patch_overlap', '0.25']
        >>> argv += ['--num_workers', '2']
        >>> argv += ['--tasks', 'all']
        >>> argv += ['--do_pca', '1']
        >>> args = parse_args(argv)
        >>> self = predict(args)
        >>> self.forward(args)
    """
    def __init__(self, args):

        # TODO: Add a cache flag. If cache==1, Determine what images we have
        # already predicted for and then take a kwcoco subset containing
        # the cache misses. See dzyne and rutgers predictors for example
        # implementations.

        # initialize dataset
        import kwcoco
        print('load coco dataset')
        self.coco_dset = kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(args.input_kwcoco)

        ###
        print('build grid dataset')
        self.dataset = gridded_dataset(self.coco_dset, args.bands,
                                       patch_size=args.patch_size,
                                       patch_overlap=args.patch_overlap,
                                       mode='test')

        print('copy dataset')
        self.output_dset = self.dataset.coco_dset.copy()

        print('reroot')
        self.output_dset.reroot(absolute=True)  # Make all paths absolute
        self.output_dset.fpath = args.output_kwcoco  # Change output file path and bundle path
        self.output_dset.reroot(absolute=False)  # Reroot in the new bundle path

        self.devices = util_device.coerce_devices(args.device)
        assert len(self.devices) == 1, 'only 1 for now'
        self.device = device = self.devices[0]
        print('device = {!r}'.format(device))

        self.finalized_gids = set()
        self.stitcher_dict = {}
        if 'all' in args.tasks:
            self.tasks = ['segmentation', 'before_after', 'pretext']
        else:
            self.tasks = args.tasks
        ### Define tasks
        if 'segmentation' in self.tasks:
            if args.segmentation_package_path:
                self.segmentation_model = seg_model.load_package(args.segmentation_package_path)
            else:
                self.segmentation_model = seg_model.load_from_checkpoint(args.segmentation_ckpt_path, dataset=None)
            self.segmentation_model = self.segmentation_model.to(device)

        if 'pretext' in self.tasks:
            if args.pretext_package_path:
                self.pretext_model = pretext.load_package(args.pretext_package_path)
            else:
                self.pretext_model = pretext.load_from_checkpoint(args.pretext_ckpt_path, train_dataset=None, vali_dataset=None)
            self.pretext_model = self.pretext_model.eval().to(device)
            # pretext_hparams = pretext_model.hparams

            try:
                # Hack
                self.pretext_model.sort_accuracy = None
            except Exception:
                pass
            self.pretext_model.__dict__['sort_accuracy'] = ub.identity  # HUGE HACK

        self.in_feature_dims = self.pretext_model.hparams.feature_dim_shared
        if args.do_pca:
            self.pca_projector = torch.load(args.pca_projection_path).to(device)
            self.out_feature_dims = self.pca_projector.shape[0]
        else:
            self.out_feature_dims = self.in_feature_dims

        self.num_out_channels = self.out_feature_dims
        if 'segmentation' in self.tasks:
            self.num_out_channels += 1
        if 'before_after' in self.tasks:
            self.num_out_channels += 1

        self.save_channels = f'invariants:{self.num_out_channels}'
        self.output_kwcoco_path = ub.Path(args.output_kwcoco)
        out_folder = self.output_kwcoco_path.parent
        self.output_feat_dpath = (out_folder / '_assets/uky_invariants').ensuredir()

        self.imwrite_kw = {
            'compress': 'DEFLATE',
            'backend': 'gdal',
            'blocksize': 128,
        }

    def _build_img_fpath(self, gid):
        save_path = self.output_feat_dpath / f'invariants_{gid}.tif'
        return save_path

    def finalize_image(self, gid):
        self.finalized_gids.add(gid)
        stitcher = self.stitcher_dict[gid]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            recon = stitcher.finalize()

        self.stitcher_dict.pop(gid)

        from watch.tasks.fusion.predict import quantize_float01
        quant_recon, quantization = quantize_float01(recon)

        save_path = self._build_img_fpath(gid)
        save_path = self.output_feat_dpath / f'invariants_{gid}.tif'
        save_path = os.fspath(save_path)
        kwimage.imwrite(save_path, quant_recon,  space=None,
                        nodata=quantization['nodata'], **self.imwrite_kw)

        aux_height, aux_width = recon.shape[0:2]
        img = self.output_dset.index.imgs[gid]
        warp_aux_to_img = kwimage.Affine.scale(
            (img['width'] / aux_width,
             img['height'] / aux_height))

        aux = {
            'file_name': save_path,
            'height': aux_height,
            'width': aux_width,
            'channels': self.save_channels,
            'warp_aux_to_img': warp_aux_to_img.concise(),
            'quantization': quantization,
        }
        auxiliary = img.setdefault('auxiliary', [])
        auxiliary.append(aux)

    def forward(self, args):
        device = self.device
        print('device = {!r}'.format(device))
        num_workers = util_globals.coerce_num_workers(args.num_workers)
        print('num_workers = {!r}'.format(num_workers))

        loader = torch.utils.data.DataLoader(
            self.dataset, num_workers=num_workers, batch_size=args.batch_size, shuffle=False)
        num_batches = len(loader)

        # Start background processes
        # Build a task queue for background write results workers (Not currently using this)
        # queue = util_parallel.BlockingJobQueue(max_workers=0)
        from watch.utils import util_parallel
        write_workers = util_globals.coerce_num_workers(args.write_workers)
        writer = util_parallel.BlockingJobQueue(max_workers=write_workers)

        # bundle_dpath = ub.Path(self.output_dset.bundle_dpath)
        # save_dpath = (bundle_dpath / 'uky_invariants').ensuredir()

        print('Evaluating and saving features')

        with torch.set_grad_enabled(False):
            seen_images = set()
            current_gids = set()
            for idx, batch in tqdm(enumerate(loader), total=num_batches, desc='Compute features'):
                save_feat = []
                save_feat2 = []

                import xdev
                with xdev.embed_on_exception_context:

                    # Handle input nans
                    img1 = batch['image1']
                    img2 = batch['image2']
                    offset_image1 = batch['offset_image1']
                    augmented_image1 = batch['augmented_image1']

                    invalid_mask1 = torch.isnan(img1)[0].any(dim=0)
                    invalid_mask2 = torch.isnan(img2)[0].any(dim=0)

                    batch['image1'] = torch.nan_to_num(img1).to(device)
                    batch['image2'] = torch.nan_to_num(img2).to(device)
                    batch['offset_image1'] = torch.nan_to_num(offset_image1).to(device)
                    batch['augmented_image1'] = torch.nan_to_num(augmented_image1).to(device)

                    if 'pretext' in args.tasks:

                        image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                        image_stack = image_stack.to(device)

                        # Remove nans before going into the network
                        image_stack = torch.nan_to_num(image_stack)

                        #select features corresponding to first image
                        features = self.pretext_model(image_stack)[:, 0, :, :, :]
                        #select features corresponding to second image
                        features2 = self.pretext_model(image_stack)[:, 1, :, :, :]
                        if args.do_pca:
                            features = torch.einsum('xy,byhw->bxhw', self.pca_projector, features)
                            features2 = torch.einsum('xy,byhw->bxhw', self.pca_projector, features2)

                        features = features.squeeze().permute(1, 2, 0).cpu()
                        features2 = features2.squeeze().permute(1, 2, 0).cpu()

                        features[invalid_mask1] = float('nan')
                        features2[invalid_mask2] = float('nan')

                        save_feat.append(features)
                        save_feat2.append(features2)

                    if 'before_after' in args.tasks:
                        ### TO DO: Set to output of separate model.
                        before_after_heatmap = self.pretext_model.shared_step(batch)['before_after_heatmap'][0].permute(1, 2, 0)
                        before_after_heatmap = torch.sigmoid(torch.exp(before_after_heatmap[:, :, 1]) - torch.exp(before_after_heatmap[:, :, 0])).unsqueeze(-1).cpu()

                        before_after_heatmap[invalid_mask1] = float('nan')
                        before_after_heatmap[invalid_mask2] = float('nan')

                        save_feat.append(before_after_heatmap)
                        save_feat2.append(before_after_heatmap)

                    if 'segmentation' in args.tasks:
                        image_stack = [batch[key] for key in batch if key.startswith('image')]
                        image_stack = torch.stack(image_stack, dim=1).to(args.device)
                        predictions = torch.exp(self.segmentation_model(image_stack)['predictions'])

                        segmentation_heatmap = torch.sigmoid(predictions[0, 0, 1, :, :] - predictions[0, 0, 0, :, :]).unsqueeze(0).permute(1, 2, 0).cpu()
                        segmentation_heatmap2 = torch.sigmoid(predictions[0, 1, 1, :, :] - predictions[0, 1, 0, :, :]).unsqueeze(0).permute(1, 2, 0).cpu()

                        segmentation_heatmap[invalid_mask1] = float('nan')
                        segmentation_heatmap2[invalid_mask2] = float('nan')

                        save_feat.append(segmentation_heatmap)
                        save_feat2.append(segmentation_heatmap2)

                    save_feat = torch.cat(save_feat, dim=-1)
                    save_feat = save_feat.numpy()
                    save_feat2 = torch.cat(save_feat2, dim=-1)
                    save_feat2 = save_feat2.numpy()

                    # image_id = int(batch['img1_id'].item())
                    # image_info = output_dset.index.imgs[image_id]
                    # video_info = output_dset.index.videos[image_info['video_id']]

                    # video_folder = (save_dpath / video_info['name']).ensuredir()

                    # # Predictions are saved in 'video space', so warp_aux_to_img is the inverse of warp_img_to_vid
                    # warp_img_to_vid = kwimage.Affine.coerce(image_info.get('warp_img_to_vid', None))
                    # warp_aux_to_img = warp_img_to_vid.inv().concise()

                    # # Get the output image dictionary to be added to
                    # output_img = output_dset.index.imgs[image_id]

                    tr = self.dataset.patches[idx]
                    # sample = self.dataset.sampler.load_sample(tr)
                    # tr = sample['tr']

                    if len(current_gids) == 0:
                        current_gids = tr['gids']
                    previous_gids = current_gids
                    current_gids = tr['gids']

                    # If we start looking at a new image, that means the
                    # previous image must be done (because we assume sorted
                    # batches). Thus we can finalize the previous image and
                    # free any memory used by its stitcher
                    mutually_exclusive = (set(previous_gids) - set(current_gids))
                    for gid in mutually_exclusive:
                        seen_images.add(gid)
                        writer.submit(self.finalize_image, gid)

                    gid1, gid2 = current_gids
                    if gid1 not in self.stitcher_dict.keys():
                        img1 = self.dataset.coco_dset.index.imgs[gid1]
                        space_dims = (img1['height'], img1['width'])
                        self.stitcher_dict[gid1] = kwarray.Stitcher(
                            space_dims + (self.num_out_channels,), device='numpy')
                    if gid2 not in self.stitcher_dict.keys():
                        img2 = self.dataset.coco_dset.index.imgs[gid2]
                        space_dims = (img2['height'], img2['width'])
                        self.stitcher_dict[gid2] = kwarray.Stitcher(
                            space_dims + (self.num_out_channels,), device='numpy')

                    slice_ = tr['space_slice']
                    # weights = util_kwimage.upweight_center_mask(save_feat.shape[0:2])[..., None]
                    # weights1 = weights.copy()
                    # weights2 = weights.copy()
                    # invalid_mask1_np = invalid_mask1.numpy()
                    # invalid_mask2_np = invalid_mask2.numpy()
                    # if invalid_mask1_np.any():
                    #     spatial_valid_mask1 = (1 - invalid_mask1_np)[..., None]
                    #     weights1 = weights1 * spatial_valid_mask1
                    #     save_feat[invalid_mask1_np] = 0
                    # if invalid_mask2_np.any():
                    #     spatial_valid_mask2 = (1 - invalid_mask2_np)[..., None]
                    #     weights2 = weights2 * spatial_valid_mask2
                    #     save_feat[invalid_mask2_np] = 0

                    # TODO: refactor and make a good CocoStitchingManager
                    from watch.tasks.fusion.predict import CocoStitchingManager
                    stitcher1 = self.stitcher_dict[gid1]
                    stitcher2 = self.stitcher_dict[gid2]
                    CocoStitchingManager._stitcher_center_weighted_add(
                        stitcher1, slice_, save_feat)
                    CocoStitchingManager._stitcher_center_weighted_add(
                        stitcher2, slice_, save_feat2)

                    # self.stitcher_dict[gid1].add(slice_, save_feat, weight=weights1)
                    # self.stitcher_dict[gid2].add(slice_, save_feat2, weight=weights2)

            writer.wait_until_finished()

            for gid in list(self.stitcher_dict.keys()):
                writer.submit(self.finalize_image, gid)

            writer.wait_until_finished()

        print('Write to dset.fpath = {!r}'.format(self.output_dset.fpath))
        self.output_dset.dump(self.output_dset.fpath, newlines=True)
        print('Done')


def parse_args(argv=None):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.invariants.predict import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> pretext_package_path = dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pretext_package.pt'
        >>> pca_projection_path = dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pca_projection_matrix.pt'
        >>> segmentation_package_path = dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_segmentation_model/segmentation_package.pt'
        >>> input_kwcoco = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
        >>> output_kwcoco = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/test_uky.kwcoco.json'
        >>> argv = []
        >>> argv += ['--input_kwcoco', f'{input_kwcoco}']
        >>> argv += ['--output_kwcoco', f'{output_kwcoco}']
        >>> argv += ['--pca_projection_path', f'{pca_projection_path}']
        >>> argv += ['--pretext_package_path', f'{pretext_package_path}']
        >>> argv += ['--segmentation_package_path', f'{segmentation_package_path}']
        >>> argv += ['--patch_overlap', '0']
        >>> argv += ['--num_workers', '2']
        >>> argv += ['--tasks', 'all']
        >>> argv += ['--do_pca', '1']
        >>> args = parse_args(argv)
    """

    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    from scriptconfig.smartcast import smartcast
    parser.add_argument('--device', type=str, default='cuda')

    # pytorch lightning checkpoint
    parser.add_argument('--pretext_ckpt_path', type=str, default=None)
    parser.add_argument('--segmentation_ckpt_path', type=str, default=None)
    parser.add_argument('--pretext_package_path', type=str, default=None)
    parser.add_argument('--segmentation_package_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', default=4, help='number of background data loading workers')
    parser.add_argument('--write_workers', default=0, help='number of background data writing workers')

    # data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=smartcast, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, help='Choose bands on which to train. Can specify \'all\' for all bands from given sensor, or \'share\' to use common bands when using both S2 and L8 sensors', nargs='+', default=['shared'])
    # output flags
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--patch_overlap', type=float, default=.25)
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an output kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py rerooted to point to the original data.', required=True)
    parser.add_argument('--tasks', nargs='+', help='Specify which tasks to choose from (segmentation, before_after, or pretext. Can also specify \'all\')', default=['all'])
    parser.add_argument('--do_pca', type=int, help='Set to 1 to perform pca. Choose output dimension in num_dim argument.', default=1)
    parser.add_argument('--pca_projection_path', type=str, help='Path to pca projection matrix', default='')

    parser.set_defaults(
        terminate_on_nan=True
        )

    args = parser.parse_args(args=argv)

    if 'all' in args.tasks:
        args.tasks = ['segmentation', 'before_after', 'pretext']

    return args


def main():
    args = parse_args()
    predict(args).forward(args)


if __name__ == '__main__':
    """
    SeeAlso:
        ../../cli/prepare_teamfeats.py

        # Team Features on Drop3
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --with_depth=0 \
            --with_landcover=0 \
            --with_materials=0  \
            --with_invariants=1 \
            --do_splits=0 \
            --gres=0 --backend=serial --run=1

    CommandLine:
        python -m watch.tasks.template.predict --help

        DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
        PRETEXT_PATH=$DVC_DPATH/models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pretext_package.pt
        SSEG_PATH=$DVC_DPATH/models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt
        PCA_FPATH=$DVC_DPATH/models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pca_projection_matrix.pt
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
        python -m watch.tasks.invariants.predict \
            --pretext_package_path "$PRETEXT_PATH" \
            --segmentation_package_path "$SSEG_PATH" \
            --pca_projection_path "$PCA_FPATH" \
            --input_kwcoco $KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --num_workers=avail \
            --do_pca 0 \
            --patch_overlap=0.3 \
            --output_kwcoco $KWCOCO_BUNDLE_DPATH/uky_invariants.kwcoco.json \
            --tasks before_after pretext

        python -m watch stats $KWCOCO_BUNDLE_DPATH/uky_invariants.kwcoco.json

        python -m watch visualize $KWCOCO_BUNDLE_DPATH/uky_invariants/invariants_nowv_vali.kwcoco.json \
            --channels "invariants.7,invariants.6,invariants.5" --animate=True \
            --select_images '.sensor_coarse != "WV"' --draw_anns=False
    """
    main()
