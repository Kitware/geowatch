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

# local imports
from .model import pretext
from .iarpa_dataset import kwcoco_dataset


def main(args):
    print('Loading checkpoint')
    model = pretext.load_from_checkpoint(args.ckpt_path)
    model.eval().to(args.device)
    print('Initiating dataset')
    dataset = kwcoco_dataset(args.input_kwcoco, args.sensor, args.bands)

    if len(args.tasks) == 1 and args.tasks[0].lower() == 'all':
        feature_types = model.TASK_NAMES
        feature_types.append('shared')
    else:
        feature_types = args.tasks

    print('Evaluating and saving features')

    if args.data_save_folder is not None:
        root = args.data_save_folder
    else:
        root, _ = os.path.split(args.input_kwcoco)

    for idx in tqdm(range(len(dataset))):
        image_id, image_info, image  = dataset.get_img(idx, args.device)

        aux_base = image_info['auxiliary'][0]
        path, file_name = os.path.split(aux_base['file_name'])

        if args.data_save_folder is not None:
            save_path = args.data_save_folder
        else:
            save_path = os.path.join(root, 'uky_invariants', path)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        features = model.predict(image)

        for key in feature_types:
            feat = features[key].squeeze()
            feat = feat.permute(1, 2, 0).detach().cpu().numpy()
            last_us_idx = file_name.rfind('_')
            name = file_name[:last_us_idx] + '_invariants_' + key + '.tif'
            kwimage.imwrite(os.path.join(save_path, name), feat, space=None, backend='gdal')

            info = {}
            info['file_name'] = os.path.join('uky_invariants', path, name)
            info['height'] = feat.shape[0]
            info['width'] = feat.shape[1]
            info['num_bands'] = feat.shape[2]
            info['channels'] = '|'.join(['inv_' + key + f'{i}' for i in range(1, feat.shape[2] + 1)])
            info['warp_aux_to_img'] = kwimage.Affine.eye().concise()

            dataset.dset.index.imgs[image_id]['auxiliary'].append(info)

    if args.output_kwcoco is None:
        args.output_kwcoco = os.path.join(root, 'uky_invariants.kwcoco.json')

    dataset.dset.fpath = args.output_kwcoco
    print('Write to dset.fpath = {!r}'.format(dataset.dset.fpath))
    dataset.dset.dump(dataset.dset.fpath, newlines=True)

    print('Done')


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

    # data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=str, help='Sensor to generate features from. Currently must choose from S2, LS, or L8. Make sure this matches the sensor used to train the model in ckpt_path.', default='S2')
    parser.add_argument('--bands', type=list, help='Bands to use in evaluation. Choose from \'all\' or create list of acceptable bands. Make sure this matches the bands used to train the model in ckpt_path.', default=['all'])

    # output flags
    parser.add_argument('--data_save_folder', help='Path to store generated feature tifs, data will end up within a folder named \'uky_invariants\'. If not specified, data is saved in the folder uky_invariants within the overall data folder.', type=str)
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an output kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py. If None, output_kwcoco will update input kwcoco.', default='')
    parser.add_argument('--tasks', nargs='+', help=f'specify which tasks to choose from ({", ".join(pretext.TASK_NAMES)}, shared, or all.\nEx: --tasks {pretext.TASK_NAMES[0]} {pretext.TASK_NAMES[1]}', default=['all'])

    parser.set_defaults(
        terminate_on_nan=True
        )

    args = parser.parse_args()

    # check save directory
    default_path = os.path.join(os.getcwd(), 'watch/tasks/invariants')
    if args.ckpt_path is None and os.path.exists(default_path):
        args.ckpt_path = os.path.join(default_path, 'logs')
    elif args.ckpt_path is None:
        args.ckpt_path = 'invariants_logs'

    torch.autograd.set_detect_anomaly(True)
    main(args)
