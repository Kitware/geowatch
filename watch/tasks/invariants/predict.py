#!/usr/bin/env python
"""
This is a Template for writing training logic.
"""
# package imports
import torch
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import kwimage
from tqdm import tqdm
# import ubelt as ub ## not currently using
from torch import pca_lowrank as pca

# local imports
from .pretext_model import pretext
from .data.datasets import kwcoco_dataset
from .data.multi_image_datasets import kwcoco_dataset as multi_image_dataset
# from watch.utils import util_parallel
from watch.utils.lightning_ext import util_globals
from .segmentation_model import segmentation_model as seg_model


def predict(args):
    ### Set up folders, paths
    source_folder, _ = os.path.split(args.input_kwcoco)
    dvc_folder, data_folder = os.path.split(source_folder)

    if args.data_save_folder:
        save_path = args.data_save_folder
        check1, check2 = os.path.split(save_path)
        if check2 != 'uky_invariants':
            save_path = os.path.join(save_path, 'uky_invariants')
    else:
        save_path = os.path.join(dvc_folder, 'uky_invariants')

    save_path = os.path.join(save_path, data_folder)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if not args.output_kwcoco:
        args.output_kwcoco = os.path.join(save_path, 'invariants.kwcoco.json')

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
        segmentation_model = seg_model.load_from_checkpoint(args.segmentation_ckpt_path)
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
            dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            feature_collection = []
            print('Calculating projection matrix based on pca.')

            for batch in tqdm(dl):
                image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                features = pretext_model(image_stack.to(pretext_model.device))
                feature_collection.append(features.cpu())
            stack = torch.cat(feature_collection, dim=0).permute(0, 1, 3, 4, 2).reshape(-1, 64)
            reduction_dim = args.num_dim - before_after_dim - segmentation_dim
            _, _, projector = pca(stack, q=reduction_dim)

            projector = projector.permute(1, 0).to(device)

    dataset = multi_image_dataset(args.input_kwcoco, args.sensor, args.bands, mode='test')

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=1)
    num_batches = len(loader)

    # Start background processes
    # Build a task queue for background write results workers (Not currently using this)
    # queue = util_parallel.BlockingJobQueue(max_workers=0)
    # queue = util_parallel.BlockingJobQueue(max_workers=num_workers)

    print('Evaluating and saving features')

    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, total=num_batches):
            image_id = batch['img1_id']
            image_info = dataset.dset.index.imgs[image_id.item()]

            aux_base = image_info['auxiliary'][0]
            rel_path, file_name = os.path.split(aux_base['file_name'])

            ###reroot original kwcoco to take out of uky_invariants folder
            for x in image_info['auxiliary']:
                x['file_name'] = os.path.join('../..', data_folder, x['file_name'])

            save_folder = os.path.join(save_path, rel_path)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)

            # Predictions are saved in 'video space', so warp_aux_to_img is the inverse of warp_img_to_vid
            warp_img_to_vid = kwimage.Affine.coerce(image_info.get('warp_img_to_vid', None))
            warp_aux_to_img = warp_img_to_vid.inv().concise()

            last_us_idx = file_name.rfind('_')

            if 'pretext' in args.tasks:
                image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                image_stack = image_stack.to(device)

                #select features corresponding to first image
                features = pretext_model(image_stack)[:, 0, :, :, :]

                if args.do_pca:
                    features = torch.einsum('xy,byhw->bxhw', projector, features)

                feat = features.squeeze().permute(1, 2, 0).cpu().numpy()
                name = file_name[:last_us_idx] + '_invariants.tif'

                kwimage.imwrite(os.path.join(save_folder, name), feat, space=None,
                                backend='gdal', compress='DEFLATE')
                info = {}
                info['file_name'] = os.path.join(save_folder, name)
                info['height'] = feat.shape[0]
                info['width'] = feat.shape[1]
                info['num_bands'] = feat.shape[2]
                info['channels'] = f'invariants:{feat.shape[-1]}'
                info['warp_aux_to_img'] = warp_aux_to_img
                dataset.dset.index.imgs[image_id.item()]['auxiliary'].append(info)

                # queue.submit(_write_results_fn, feat, save_path, file_name)

            if 'before_after' in args.tasks:
                ### TO DO: Set to output of separate model.
                before_after_heatmap = pretext_model.shared_step(batch)['before_after_heatmap'][0].permute(1, 2, 0)
                before_after_heatmap = torch.sigmoid(before_after_heatmap[:, :, 1] - before_after_heatmap[:, :, 0]).unsqueeze(-1).cpu().numpy()
                name = file_name[:last_us_idx] + '_before_after_heatmap.tif'
                kwimage.imwrite(os.path.join(save_folder, name), before_after_heatmap, space=None,
                                backend='gdal', compress='DEFLATE')
                info = {}
                info['file_name'] = os.path.join(save_folder, name)
                info['height'] = before_after_heatmap.shape[0]
                info['width'] = before_after_heatmap.shape[1]
                info['num_bands'] = before_after_heatmap.shape[2]
                info['channels'] = 'before_after_heatmap'
                info['warp_aux_to_img'] = warp_aux_to_img
                dataset.dset.index.imgs[image_id.item()]['auxiliary'].append(info)

            if 'segmentation' in args.tasks:
                image_stack = [batch[key] for key in batch if key[:5] == 'image']
                image_stack = torch.stack(image_stack, dim=1).to(args.device)
                segmentation_heatmap = torch.sigmoid(segmentation_model(image_stack)['predictions'][0, 0, 1, :, :] - segmentation_model(image_stack)['predictions'][0, 0, 0, :, :]).unsqueeze(0).permute(1, 2, 0).cpu().numpy()
                name = file_name[:last_us_idx] + '_segmentation_heatmap.tif'
                kwimage.imwrite(os.path.join(save_folder, name), segmentation_heatmap, space=None,
                                backend='gdal', compress='DEFLATE')
                info = {}
                info['file_name'] = os.path.join(save_folder, name)
                info['height'] = segmentation_heatmap.shape[0]
                info['width'] = segmentation_heatmap.shape[1]
                info['num_bands'] = segmentation_heatmap.shape[2]
                info['channels'] = 'segmentation_heatmap'
                info['warp_aux_to_img'] = warp_aux_to_img
                dataset.dset.index.imgs[image_id.item()]['auxiliary'].append(info)

    # queue.wait_until_finished()

    dataset.dset.fpath = args.output_kwcoco
    print('Write to dset.fpath = {!r}'.format(dataset.dset.fpath))
    dataset.dset.dump(dataset.dset.fpath, newlines=True)

    print('Done')


def _write_results_fn(feat, save_path, file_name):
    last_us_idx = file_name.rfind('_')
    name = file_name[:last_us_idx] + '_invariants.tif'
    fpath = os.path.join(save_path, name)
    kwimage.imwrite(fpath, feat, space=None, backend='gdal',
                    compress='RAW', blocksize=64)


def main():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    from scriptconfig.smartcast import smartcast
    parser.add_argument('--device', type=str, default='cuda')

    # pytorch lightning checkpoint
    parser.add_argument('--pretext_ckpt_path', type=str)
    parser.add_argument('--segmentation_ckpt_path', type=str)
    parser.add_argument('--before_after_ckpt_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4, help='number of background data loading workers')

    # data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=smartcast, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, help='Choose bands on which to train. Can specify \'all\' for all bands from given sensor, or \'share\' to use common bands when using both S2 and L8 sensors', nargs='+', default=['shared'])
    # output flags
    parser.add_argument('--data_save_folder', help='Path to store generated feature tifs, data will end up within a folder named \'uky_invariants\'. If not specified, data is saved in the folder uky_invariants within smart_watch_dvc. Note: paths generated within output_kwcoco file will assume features are stored in smart_watch_dvc/uky_invariants.', type=str, default=None)
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an output kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py. If None, output_kwcoco will be set up in the data_save_folder.', default=None)
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
