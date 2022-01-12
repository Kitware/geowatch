#!/usr/bin/env python
"""
This is a Template for writing training logic.
"""
# package imports
import kwimage
import os
import torch
import ubelt as ub
from argparse import ArgumentParser, RawTextHelpFormatter
from torch import pca_lowrank as pca
from tqdm import tqdm

# local imports
from .pretext_model import pretext
from .data.datasets import kwcoco_dataset
from .data.multi_image_datasets import kwcoco_dataset as multi_image_dataset
from watch.utils.lightning_ext import util_globals
from .segmentation_model import segmentation_model as seg_model


def predict(args):
    try:
        device = int(args.device)
    except Exception:
        device = args.device
    num_workers = util_globals.coerce_num_workers(args.num_workers)
    print('num_workers = {!r}'.format(num_workers))

    ### Define tasks
    if 'before_after' in args.tasks:
        before_after_dim = 1
    else:
        before_after_dim = 0

    if 'segmentation' in args.tasks:
        segmentation_dim = 1
        segmentation_model = seg_model.load_from_checkpoint(args.segmentation_ckpt_path, dataset=None)
        segmentation_model = segmentation_model.to(device)

    else:
        segmentation_dim = 0

    if 'pretext' in args.tasks:
        pretext_model = pretext.load_from_checkpoint(args.pretext_ckpt_path, train_dataset=None, vali_dataset=None)
        pretext_model = pretext_model.eval().to(device)
        pretext_hparams = pretext_model.hparams

        if args.do_pca:
            ###Slightly reduced dataset for pca
            dataset = kwcoco_dataset(args.input_kwcoco, sensor=pretext_hparams.sensor, bands=pretext_hparams.bands, patch_size=32, change_labels=False, display=False)

            ### Define projector
            dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
            feature_collection = []
            print('Calculating projection matrix based on pca.')

            with torch.set_grad_enabled(False):
                # TODO: option to cache or specify a specific projection matrix?
                for batch in tqdm(dl, desc='Calculating PCA matrix'):
                    image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                    features = pretext_model(image_stack.to(pretext_model.device))
                    feature_collection.append(features.cpu())
                features = None
                image_stack = None
                stack = torch.cat(feature_collection, dim=0).permute(0, 1, 3, 4, 2).reshape(-1, 64)
                reduction_dim = args.num_dim - before_after_dim - segmentation_dim
                _, _, projector = pca(stack, q=reduction_dim)
                stack = None

            projector = projector.permute(1, 0).to(device)

    dataset = multi_image_dataset(args.input_kwcoco, args.sensor, args.bands, mode='test')

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=1)
    num_batches = len(loader)

    # Start background processes
    # Build a task queue for background write results workers (Not currently using this)
    # queue = util_parallel.BlockingJobQueue(max_workers=0)
    from watch.utils import util_parallel
    write_workers = util_globals.coerce_num_workers(args.write_workers)
    queue = util_parallel.BlockingJobQueue(max_workers=write_workers)

    output_dset = dataset.dset.copy()
    output_dset.reroot(absolute=True)  # Make all paths absolute
    output_dset.fpath = args.output_kwcoco  # Change output file path and bundle path
    output_dset.reroot(absolute=False)  # Reroot in the new bundle path

    bundle_dpath = ub.Path(output_dset.bundle_dpath)

    save_dpath = (bundle_dpath / 'uky_invariants').ensuredir()

    imwrite_kw = {
        'compress': 'DEFLATE',
        'backend': 'gdal',
        'blocksize': 64,
    }

    print('Evaluating and saving features')

    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, total=num_batches, desc='Compute features'):
            image_id = int(batch['img1_id'].item())
            image_info = dataset.dset.index.imgs[image_id]
            video_info = dataset.dset.index.videos[image_info['video_id']]

            video_folder = (save_dpath / video_info['name']).ensuredir()

            # Predictions are saved in 'video space', so warp_aux_to_img is the inverse of warp_img_to_vid
            warp_img_to_vid = kwimage.Affine.coerce(image_info.get('warp_img_to_vid', None))
            warp_aux_to_img = warp_img_to_vid.inv().concise()

            # Get the output image dictionary to be added to
            output_img = output_dset.index.imgs[image_id]

            if 'pretext' in args.tasks:
                image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                image_stack = image_stack.to(device)

                #select features corresponding to first image
                features = pretext_model(image_stack)[:, 0, :, :, :]

                if args.do_pca:
                    features = torch.einsum('xy,byhw->bxhw', projector, features)

                feat = features.squeeze().permute(1, 2, 0).cpu().numpy()
                fname = image_info['name'] + '_invariants.tif'
                fpath = video_folder / fname

                # kwimage.imwrite(fpath, feat, **imwrite_kw)
                queue.submit(kwimage.imwrite, fpath, feat, **imwrite_kw)

                info = {}
                info['file_name'] = str(fpath.relative_to(bundle_dpath))
                info['height'] = feat.shape[0]
                info['width'] = feat.shape[1]
                info['num_bands'] = feat.shape[2]
                info['channels'] = f'invariants:{feat.shape[-1]}'
                info['warp_aux_to_img'] = warp_aux_to_img
                output_img['auxiliary'].append(info)

            if 'before_after' in args.tasks:
                ### TO DO: Set to output of separate model.
                before_after_heatmap = pretext_model.shared_step(batch)['before_after_heatmap'][0].permute(1, 2, 0)
                before_after_heatmap = torch.sigmoid(before_after_heatmap[:, :, 1] - before_after_heatmap[:, :, 0]).unsqueeze(-1).cpu().numpy()

                fname = image_info['name'] + '_before_after_heatmap.tif'
                fpath = video_folder / fname

                # kwimage.imwrite(fpath, before_after_heatmap, **imwrite_kw)
                queue.submit(kwimage.imwrite, fpath, before_after_heatmap, **imwrite_kw)

                info = {}
                info['file_name'] = str(fpath.relative_to(bundle_dpath))
                info['height'] = before_after_heatmap.shape[0]
                info['width'] = before_after_heatmap.shape[1]
                info['num_bands'] = before_after_heatmap.shape[2]
                info['channels'] = 'before_after_heatmap'
                info['warp_aux_to_img'] = warp_aux_to_img
                output_img['auxiliary'].append(info)

            if 'segmentation' in args.tasks:
                image_stack = [batch[key] for key in batch if key[:5] == 'image']
                image_stack = torch.stack(image_stack, dim=1).to(args.device)
                segmentation_heatmap = torch.sigmoid(segmentation_model(image_stack)['predictions'][0, 0, 1, :, :] - segmentation_model(image_stack)['predictions'][0, 0, 0, :, :]).unsqueeze(0).permute(1, 2, 0).cpu().numpy()

                fname = image_info['name'] + '_segmentation_heatmap.tif'
                fpath = video_folder / fname

                # kwimage.imwrite(fpath, segmentation_heatmap, **imwrite_kw)
                queue.submit(kwimage.imwrite, fpath, segmentation_heatmap, **imwrite_kw)

                info = {}
                info['file_name'] = str(fpath.relative_to(bundle_dpath))
                info['height'] = segmentation_heatmap.shape[0]
                info['width'] = segmentation_heatmap.shape[1]
                info['num_bands'] = segmentation_heatmap.shape[2]
                info['channels'] = 'segmentation_heatmap'
                info['warp_aux_to_img'] = warp_aux_to_img
                output_img['auxiliary'].append(info)

    # queue.wait_until_finished()

    print('Write to dset.fpath = {!r}'.format(output_dset.fpath))
    output_dset.dump(output_dset.fpath, newlines=True)
    print('Done')


def main():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    from scriptconfig.smartcast import smartcast
    parser.add_argument('--device', type=str, default='cuda')

    # pytorch lightning checkpoint
    parser.add_argument('--pretext_ckpt_path', type=str)
    parser.add_argument('--segmentation_ckpt_path', type=str)
    parser.add_argument('--before_after_ckpt_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', default=4, help='number of background data loading workers')
    parser.add_argument('--write_workers', default=0, help='number of background data writing workers')

    # data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=smartcast, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, help='Choose bands on which to train. Can specify \'all\' for all bands from given sensor, or \'share\' to use common bands when using both S2 and L8 sensors', nargs='+', default=['shared'])
    # output flags
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an output kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py rerooted to point to the original data.', required=True)
    parser.add_argument('--tasks', nargs='+', help='Specify which tasks to choose from (segmentation, before_after, or pretext. Can also specify \'all\')', default=['all'])
    parser.add_argument('--do_pca', type=int, help='Set to 1 to perform pca. Choose output dimension in num_dim argument.', default=1)
    parser.add_argument('--num_dim', type=int, help='Output dimensions after pca.', default=8)

    parser.set_defaults(
        terminate_on_nan=True
        )

    args = parser.parse_args()

    if 'all' in args.tasks:
        args.tasks = ['segmentation', 'before_after', 'pretext']

    predict(args)


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.template.predict --help

        python -m watch.tasks.template.predict \
            --input_kwcoco=path/to/data.kwcoco.json
    """
    main()
