"""
SeeAlso:
    ~/code/watch/watch/cli/prepare_teamfeats.py
"""
import kwimage
import kwarray
import torch
import ubelt as ub
import os
# local imports
from .pretext_model import pretext
from .data.datasets import gridded_dataset
from watch.utils.lightning_ext import util_globals
from watch.utils.lightning_ext import util_device
from .segmentation_model import segmentation_model as seg_model
from watch.utils import util_kwimage  # NOQA

from watch.tasks.fusion.predict import CocoStitchingManager
from watch.tasks.fusion.predict import quantize_float01

import scriptconfig as scfg


class InvariantPredictConfig(scfg.DataConfig):
    """
    Configuration for UKY invariant models
    """
    device = scfg.Value('cuda', type=str)
    pretext_ckpt_path = scfg.Value(None, type=str)
    segmentation_ckpt_path = scfg.Value(None, type=str)
    pretext_package_path = scfg.Value(None, type=str)
    segmentation_package_path = scfg.Value(None, type=str)
    batch_size = scfg.Value(1, type=int)
    num_workers = scfg.Value(4, help=ub.paragraph(
            '''
            number of background data loading workers
            '''))
    write_workers = scfg.Value(0, help=ub.paragraph(
            '''
            number of background data writing workers
            '''))
    sensor = scfg.Value(['S2', 'L8'], nargs='+')
    bands = scfg.Value(['shared'], type=str, help=ub.paragraph(
            '''
            Choose bands on which to train. Can specify 'all' for all
            bands from given sensor, or 'share' to use common bands when
            using both S2 and L8 sensors
            '''), nargs='+')
    patch_size = scfg.Value(256, type=int)
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
    tasks = scfg.Value(['all'], help=ub.paragraph(
            '''
            Specify which tasks to choose from (segmentation,
            before_after, or pretext. Can also specify 'all')
            '''), nargs='+')
    do_pca = scfg.Value(1, type=int, help=ub.paragraph(
            '''
            Set to 1 to perform pca. Choose output dimension in num_dim
            argument.
            '''))
    pca_projection_path = scfg.Value('', type=str, help='Path to pca projection matrix')

    track_emissions = scfg.Value(True, help='Set to false to disable codecarbon')

    def normalize(self):
        if 'all' in self.tasks:
            self['tasks'] = ['segmentation', 'before_after', 'pretext']


class Predictor(object):
    """
    CommandLine:
        DVC_DPATH=$(smartwatch_dvc)
        DVC_DPATH=$DVC_DPATH xdoctest -m watch.tasks.invariants.predict Predictor

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
        >>> args = InvariantPredictConfig.cli(argv=argv)
        >>> self = Predictor(args)
        >>> self.forward(args)
    """

    def __init__(self, args):

        # TODO: Add a cache flag. If cache==1, Determine what images we have
        # already predicted for and then take a kwcoco subset containing
        # the cache misses. See dzyne and rutgers predictors for example
        # implementations.

        # Doesnt work?
        # FIX_ALBUMENTATIONS_HACK = 0
        # if FIX_ALBUMENTATIONS_HACK:
        #     from albumentations.core import composition
        #     composition.Transforms = composition.TransformType

        self.batch_size = args.batch_size
        self.do_pca = args.do_pca

        self.tasks = args.tasks

        self.num_workers = util_globals.coerce_num_workers(args.num_workers)
        print('self.num_workers = {!r}'.format(self.num_workers))

        self.write_workers = util_globals.coerce_num_workers(args.write_workers)
        print(f'self.write_workers={self.write_workers}')

        self.devices = util_device.coerce_devices(args.device)
        assert len(self.devices) == 1, 'only 1 for now'
        self.device = device = self.devices[0]
        print('device = {!r}'.format(device))

        # Initialize models
        print('Initialize models')
        if 'all' in args.tasks:
            self.tasks = ['segmentation', 'before_after', 'pretext']
        else:
            self.tasks = args.tasks
        ### Define tasks
        if 'segmentation' in self.tasks:
            if args.segmentation_package_path:
                print('Initialize segmentation model from package')
                self.segmentation_model = seg_model.load_package(args.segmentation_package_path)
            else:
                print('Initialize segmentation model from checkpoint')
                self.segmentation_model = seg_model.load_from_checkpoint(args.segmentation_ckpt_path, dataset=None)
            self.segmentation_model = self.segmentation_model.to(device)

        if 'pretext' in self.tasks:
            if args.pretext_package_path:
                print('Initialize pretext model from package')
                self.pretext_model = pretext.load_package(args.pretext_package_path)
            else:
                print('Initialize pretext model from checkpoint')
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
            print('Initialize PCA model')
            self.pca_projector = torch.load(args.pca_projection_path).to(device)
            self.out_feature_dims = self.pca_projector.shape[0]
        else:
            self.out_feature_dims = self.in_feature_dims

        self.num_out_channels = self.out_feature_dims
        if 'segmentation' in self.tasks:
            self.num_out_channels += 1
        if 'before_after' in self.tasks:
            self.num_out_channels += 1

        # initialize dataset
        import kwcoco
        print('load coco dataset')
        self.coco_dset = kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(args.input_kwcoco)
        # self.coco_dset = self.coco_dset.subset(list(self.coco_dset.images()[0:20]))

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
        self.finalized_gids = set()
        self.stitcher_dict = {}

        self.save_channels = f'invariants:{self.num_out_channels}'
        self.output_kwcoco_path = ub.Path(args.output_kwcoco)
        out_folder = self.output_kwcoco_path.parent
        self.output_feat_dpath = (out_folder / '_assets/uky_invariants').ensuredir()

        self.imwrite_kw = {
            'compress': 'DEFLATE',
            'backend': 'gdal',
            'blocksize': 128,
        }

        from watch.utils import process_context
        import sys
        self.proc_context = process_context.ProcessContext(
            args=sys.argv,
            type='process',
            name='watch.tasks.invariants.predict',
            config=args.to_dict(),
            track_emissions=args.track_emissions,
        )

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

        quant_recon, quantization = quantize_float01(recon)

        save_path = self._build_img_fpath(gid)
        kwimage.imwrite(save_path, quant_recon, space=None,
                        nodata=quantization['nodata'], **self.imwrite_kw)

        aux_height, aux_width = recon.shape[0:2]
        img = self.output_dset.index.imgs[gid]
        warp_aux_to_img = kwimage.Affine.scale(
            (img['width'] / aux_width,
             img['height'] / aux_height))

        aux = {
            'file_name': os.fspath(save_path),
            'height': aux_height,
            'width': aux_width,
            'channels': self.save_channels,
            'warp_aux_to_img': warp_aux_to_img.concise(),
            'quantization': quantization,
        }
        if 'auxiliary' not in img:
            img['auxiliary'] = []
        auxiliary = img['auxiliary']
        auxiliary.append(aux)

    def ensure_stitcher(self, gid):
        """
        Create a stitcher for an image if it doesnt exist
        """
        if gid not in self.stitcher_dict:
            img = self.dataset.coco_dset.index.imgs[gid]
            space_dims = (img['height'], img['width'])
            self.stitcher_dict[gid] = kwarray.Stitcher(
                space_dims + (self.num_out_channels,), device='numpy')
        return self.stitcher_dict[gid]

    def forward(self):
        device = self.device

        loader = torch.utils.data.DataLoader(
            self.dataset, num_workers=self.num_workers,
            batch_size=self.batch_size, shuffle=False)
        num_batches = len(loader)

        # Start background processes
        # Build a task queue for background write results workers
        from watch.utils import util_parallel
        writer = util_parallel.BlockingJobQueue(max_workers=self.write_workers)

        self.proc_context.start()

        self.proc_context.add_disk_info(ub.Path(self.output_dset.fpath).parent)
        self.output_dset.dataset.setdefault('info', [])
        self.output_dset.dataset['info'].append(self.proc_context.obj)

        print('Evaluating and saving features')

        with torch.set_grad_enabled(False):
            seen_images = set()
            prog = ub.ProgIter(enumerate(loader), total=num_batches, desc='Compute features', verbose=1)
            for idx, batch in prog:
                save_feat = []
                save_feat2 = []

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

                if 'pretext' in self.tasks:

                    image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                    image_stack = image_stack.to(device)

                    # Remove nans before going into the network
                    image_stack = torch.nan_to_num(image_stack)

                    #select features corresponding to first image
                    features = self.pretext_model(image_stack)[:, 0, :, :, :]
                    #select features corresponding to second image
                    features2 = self.pretext_model(image_stack)[:, 1, :, :, :]
                    if self.do_pca:
                        features = torch.einsum('xy,byhw->bxhw', self.pca_projector, features)
                        features2 = torch.einsum('xy,byhw->bxhw', self.pca_projector, features2)

                    features = features.squeeze().permute(1, 2, 0).cpu()
                    features2 = features2.squeeze().permute(1, 2, 0).cpu()

                    features[invalid_mask1] = float('nan')
                    features2[invalid_mask2] = float('nan')

                    save_feat.append(features)
                    save_feat2.append(features2)

                if 'before_after' in self.tasks:
                    ### TO DO: Set to output of separate model.
                    before_after_heatmap = self.pretext_model.shared_step(batch)['before_after_heatmap'][0].permute(1, 2, 0)
                    before_after_heatmap = torch.sigmoid(torch.exp(before_after_heatmap[:, :, 1]) - torch.exp(before_after_heatmap[:, :, 0])).unsqueeze(-1).cpu()

                    before_after_heatmap[invalid_mask1] = float('nan')
                    before_after_heatmap[invalid_mask2] = float('nan')

                    save_feat.append(before_after_heatmap)
                    save_feat2.append(before_after_heatmap)

                if 'segmentation' in self.tasks:
                    image_stack = [batch[key] for key in batch if key.startswith('image')]
                    image_stack = torch.stack(image_stack, dim=1).to(self.device)
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

                tr = self.dataset.patches[idx]

                # These dataloader has told us that these iamges are now
                # complete, and thus can be finalized free any memory used by
                # its stitcher
                new_complete_gids = tr.get('new_complete_gids', [])
                for gid in new_complete_gids:
                    assert gid not in seen_images
                    seen_images.add(gid)
                    # if 1:
                    #     img = self.dataset.coco_dset.index.imgs[gid]
                    #     frame_index = img['frame_index']
                    #     video_id = img['video_id']
                    #     prog.ensure_newline()
                    #     print(f'finalize {video_id=}, {gid=}, {frame_index=}')
                    writer.submit(self.finalize_image, gid)

                gid1, gid2 = tr['gids']
                slice_ = tr['space_slice']
                stitcher1 = self.ensure_stitcher(gid1)
                stitcher2 = self.ensure_stitcher(gid2)
                CocoStitchingManager._stitcher_center_weighted_add(
                    stitcher1, slice_, save_feat)
                CocoStitchingManager._stitcher_center_weighted_add(
                    stitcher2, slice_, save_feat2)

            print('Finalize already compelted jobs')
            writer.wait_until_finished(desc='Finalize submitted jobs')

            # Finalize everything else that hasn't completed
            for gid in ub.ProgIter(list(self.stitcher_dict.keys()), desc='submit loose write jobs'):
                if gid not in seen_images:
                    seen_images.add(gid)
                    writer.submit(self.finalize_image, gid)

            print('Finalize loose jobs')
            writer.wait_until_finished()

        print('Finish process context')
        self.proc_context.add_device_info(device)
        self.proc_context.stop()

        print('Write to dset.fpath = {!r}'.format(self.output_dset.fpath))
        self.output_dset.dump(self.output_dset.fpath, newlines=True)
        print('Done')


def main():
    args = InvariantPredictConfig.cli()
    Predictor(args).forward()


if __name__ == '__main__':
    """
    SeeAlso:
        ../../cli/prepare_teamfeats.py

        # Team Features on Drop3
        DVC_DPATH=$(smartwatch_dvc)
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

        DVC_DPATH=$(smartwatch_dvc)
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

    Ignore:
        ### Command 1 / 2 - watch-teamfeat-job-0
        python -m watch.tasks.invariants.predict \
            --input_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_kr1br2.kwcoco.json" \
            --output_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_kr1br2_uky_invariants.kwcoco.json" \
            --pretext_package_path "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_package.pt" \
            --pca_projection_path  "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt" \
            --do_pca 0 \
            --patch_overlap=0.3 \
            --num_workers="2" \
            --write_workers 0 \
            --tasks before_after pretext

        cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
        kwcoco subset --src=data.kwcoco.json --dst=AE_R001.kwcoco.json --select_videos='.name == "AE_R001"'
        kwcoco subset --src=data.kwcoco.json --dst=NZ_R001.kwcoco.json --select_videos='.name == "NZ_R001"'

        python -m watch.tasks.invariants.predict \
            --input_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/NZ_R001.kwcoco.json" \
            --output_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/NZ_R001_invariants.kwcoco.json" \
            --pretext_package_path "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_package.pt" \
            --pca_projection_path  "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt" \
            --do_pca 0 \
            --patch_overlap=0.0 \
            --num_workers="2" \
            --write_workers 2 \
            --sensor=L8 \
            --tasks before_after pretext

        python -m watch visualize NZ_R001_invariants.kwcoco.json \
            --channels "invariants.5:8,invariants.8:11,invariants.11:14" --stack=only --fast --animate=True \
            --select_images '.sensor_coarse == "L8"' --draw_anns=False


    """
    main()
