#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a Template for writing training logic.
"""
# package imports
import torch
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import kwimage
from tqdm import tqdm
import ubelt as ub

# local imports
from .segmentation_model import segmentation_model
from .data.multi_image_datasets import kwcoco_dataset
from watch.utils import util_parallel
from watch.utils.lightning_ext import util_globals
from .pretext_model import pretext


def main(args):
    print('Loading checkpoint')

    if True:
        # Overload load_from_checkpoint
        overrides = {
            'train_dataset': None,
            'vali_dataset': None,
        }
        cls = segmentation_model
        checkpoint = torch.load(args.ckpt_path)
        # Hack for getting the input channels
        overrides['num_channels'] = (
            checkpoint['state_dict']['backbone.inc.conv.conv.0.weight'].shape[1]
        )
        print('overrides = {}'.format(ub.repr2(overrides, nl=1)))
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(overrides)
        model = cls._load_model_state(checkpoint, strict=True)
    else:
        model = pretext.load_from_checkpoint(
            args.ckpt_path, train_dataset=None, vali_dataset=None)

    try:
        device = int(args.device)
    except Exception:
        device = args.device

    model.eval().to(device)
    print('Initiating dataset')
    dataset = kwcoco_dataset(args.input_kwcoco, sensor=args.sensor, bands=args.bands, num_images=model.hparams.num_images, mode='test', segmentation_labels=True)

    num_workers = util_globals.coerce_num_workers(args.num_workers)
    print('num_workers = {!r}'.format(num_workers))

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=1)
    num_batches = len(loader)

    # Build a task queue for background write results workers
    queue = util_parallel.BlockingJobQueue(max_workers=num_workers)

    print('Evaluating and saving features')

    if args.data_save_folder is not None:
        root = args.data_save_folder
    else:
        root, _ = os.path.split(args.input_kwcoco)

    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, total=num_batches):
            image_id = batch['img1_id']
            image_info = dataset.dset.index.imgs[image_id.item()]
            # image_id, image_info, image  = dataset.get_img(idx, args.device)

            aux_base = image_info['auxiliary'][0]
            path, file_name = os.path.split(aux_base['file_name'])

            if args.data_save_folder is not None:
                save_path = args.data_save_folder
            else:
                save_path = os.path.join(root, 'uky_invariants', path)

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            out = model.test_step(batch)
            heatmap = torch.sigmoid(out['prediction_map'][:, 0, 1, :, :] - out['prediction_map'][:, 0, 0, :, :])

            # Predictions are saved in 'video space', so warp_aux_to_img is the inverse of warp_img_to_vid
            warp_img_to_vid = kwimage.Affine.coerce(image_info.get('warp_img_to_vid', None))
            warp_aux_to_img = warp_img_to_vid.inv().concise()

            last_us_idx = file_name.rfind('_')
            features_to_write = {}

            feat = heatmap.detach()
            feat = feat.permute(1, 2, 0).cpu().numpy()
            features_to_write['segmentation_heatmap'] = feat

            name = file_name[:last_us_idx] + '_invariants_' + 'segmentation_heatmap' + '.tif'
            # kwimage.imwrite(os.path.join(save_path, name), feat, space=None,
            #                 backend='gdal', compress='DEFLATE')

            info = {}
            info['file_name'] = os.path.join('uky_invariants', path, name)
            info['height'] = feat.shape[0]
            info['width'] = feat.shape[1]
            info['num_bands'] = feat.shape[2]
            info['channels'] = ['inv_segmentation_heatmap']
            info['warp_aux_to_img'] = warp_aux_to_img
            dataset.dset.index.imgs[image_id.item()]['auxiliary'].append(info)

            queue.submit(_write_results_fn, features_to_write, save_path, file_name)

    if args.output_kwcoco is None:
        args.output_kwcoco = os.path.join(root, 'uky_invariants.kwcoco.json')

    queue.wait_until_finished()

    dataset.dset.fpath = args.output_kwcoco
    print('Write to dset.fpath = {!r}'.format(dataset.dset.fpath))
    dataset.dset.dump(dataset.dset.fpath, newlines=True)

    print('Done')


def _write_results_fn(features_to_write, save_path, file_name):
    last_us_idx = file_name.rfind('_')
    for key, feat in features_to_write.items():
        name = file_name[:last_us_idx] + '_invariants_' + key + '.tif'
        fpath = os.path.join(save_path, name)
        kwimage.imwrite(fpath, feat, space=None, backend='gdal',
                        compress='DEFLATE')


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.template.predict --help

        python -m watch.tasks.template.predict \
            --input_kwcoco=path/to/data.kwcoco.json
    """
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda')

    # pytorch lightning checkpoint
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--num_workers', type=str, default=0, help='number of background data loading workers')

    # data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=str, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, help='Choose bands on which to train. Can specify \'all\' for all bands from given sensor, or \'share\' to use common bands when using both S2 and L8 sensors', nargs='+', default=['shared'])

    # output flags
    parser.add_argument('--data_save_folder', help='Path to store generated feature tifs, data will end up within a folder named \'uky_invariants\'. If not specified, data is saved in the folder uky_invariants within the overall data folder.', type=str)
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an output kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py. If None, output_kwcoco will update input kwcoco.', default=None)

    args = parser.parse_args()

    # check save directory
    default_path = os.path.join(os.getcwd(), 'watch/tasks/invariants')
    if args.ckpt_path is None and os.path.exists(default_path):
        args.ckpt_path = os.path.join(default_path, 'logs')
    elif args.ckpt_path is None:
        args.ckpt_path = 'invariants_logs'

    torch.autograd.set_detect_anomaly(True)
    main(args)
