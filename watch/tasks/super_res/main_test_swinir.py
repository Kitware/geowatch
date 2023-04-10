import argparse
import torch
import requests
import json
import logging
import rasterio

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from .models.network_swinir import SwinIR as net
from watch.utils import kwcoco_extensions


def main():
    """example:
    python -W ignore -m watch.tasks.super_res.main_test_swinir \
        --src_kwcoco /output/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_wv.kwcoco.json \
        --task real_sr \
        --scale 4 \
        --large_model \
        --model_path watch/tasks/super_res/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
        --dst_dir /output/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC \
        --src_images_dir /dvc/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
    """

    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_kwcoco',
        type=Path,
        default='',
        required=True,
        help='kwcoco file to parse')
    parser.add_argument(
        '--src_images_dir',
        type=Path,
        default='',
        required=True,
        help='path to where input images read in from kwcoco file exists')
    parser.add_argument(
        '--dst_dir',
        type=Path,
        default='',
        required=True,
        help='directory where to save kwcoco file and super-res image')
    parser.add_argument(
        '--folder_lq',
        type=str,
        default=None,
        help='input low-quality test image folder')
    parser.add_argument(
        '--task',
        type=str,
        default='color_dn',
        help='classical_sr, lightweight_sr, real_sr, '
        'gray_dn, color_dn, jpeg_car')
    parser.add_argument(
        '--scale',
        type=int,
        default=1,
        help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument(
        '--large_model',
        action='store_true',
        help='use large model, only provided for real image sr')
    parser.add_argument(
        '--model_path',
        type=Path,
        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument(
        '--tile',
        type=int,
        default=None,
        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument(
        '--tile_overlap',
        type=int,
        default=32,
        help='Overlapping of different tiles')
    parser.add_argument(
        '-log',
        '--loglevel',
        default='info',
        help='Logging level to display; e.g. debug, info, etc...')

    args = parser.parse_args()

    # determine output dir for images
    super_res_output_path = args.dst_dir / '_assets' / 'super_res'
    super_res_output_path.mkdir(parents=True, exist_ok=True)

    # setup logger
    logging.basicConfig(
        format='%(name)s - %(levelname)s - %(message)s',
        level=args.loglevel.upper())
    logging.info('Input:            {}'.format(args.src_kwcoco))
    logging.info('Input Images:     {}'.format(args.src_images_dir))
    logging.info('Model:            {}'.format(args.model_path))
    logging.info('Output Dir:       {}'.format(args.dst_dir))
    logging.info('Output Images:    {}'.format(super_res_output_path))

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_path.exists():
        logging.debug(f'loading model from {args.model_path}')
    else:
        args.model_path.parent.mkdir(exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(
            args.model_path.name)
        r = requests.get(url, allow_redirects=True)
        logging.debug(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)
    _, _, _, window_size = setup(args)

    # load input kwcoco file
    data = None
    with open(args.src_kwcoco, 'r') as f:
        data = json.load(f)

    # iterate through all files and process them
    new_images = []
    new_annotations = []
    annotations_df = pd.DataFrame(data['annotations'])

    for img_index, image in tqdm(
            enumerate(data['images']), total=len(data['images'])):

        # WV-only data
        if image['sensor_coarse'] != 'WV':
            # skip non-WorldView data
            logging.debug(
                'Source not WV... skipping image ID: {}'.format(
                    image['id']))
            continue

        # Need RGB channels
        required_channels = ['red', 'green', 'blue']
        auxs = {aux['channels']: aux for aux in image['auxiliary']}

        if set(required_channels).issubset(set(auxs.keys())) is False:
            # missing required color band
            logging.debug(
                'Missing required channels... skipping image ID: {}'.format(
                    image['id']))
            continue

        # Skip images with invalid dimensions
        if (image['height'] == 0) or (image['width'] == 0):
            logging.debug(
                'Invalid image size... skipping image ID: {}'.format(
                    image['id']))
            continue

        # TODO - revisit to do tiling, Yanlin said ignore large files for now
        size_threshold = 1024
        if ((image['height'] >= size_threshold)
                or (image['width'] >= size_threshold)):
            logging.debug(
                'Unspported image size (w: {},h: {})... skipping image ID: {}'.format(
                    image['width'], image['height'], image['id']))
            continue

        # create super-res of bands
        for channel in required_channels:

            aux = auxs[channel]
            aux_file = Path(aux['file_name'])
            src_file = args.src_images_dir / aux_file
            dst_file = image['name'] + \
                '_superRes{}.tif'.format(aux['channels'].capitalize())
            dst_file = super_res_output_path / aux_file.parent / dst_file

            if dst_file.exists() is False:

                # create super-res version for file
                with rasterio.open(src_file, 'r') as src:

                    # increase resolution of image
                    image = np.squeeze(src.read()).astype(np.float32)
                    image_value_range = np.max(image) - np.min(image)

                    if (image_value_range != 0):
                        image_norm = (image - np.min(image)) / \
                            image_value_range  # normalization
                    else:
                        # handle edge-case where min=max
                        image_norm = image * 0  # normalize to 0

                    # creating 3-band image from 1-band image
                    image_norm = np.stack((image_norm,) * 3, axis=0)
                    # add dim for tensor objects (batch, channel, height,
                    # width)
                    image_norm = np.expand_dims(image_norm, [0])

                    with torch.no_grad():

                        # pad input image to be a multiple of window_size
                        img_lq = torch.from_numpy(image_norm).to(device)
                        _, _, h_old, w_old = img_lq.size()
                        h_pad = (h_old // window_size + 1) * \
                            window_size - h_old
                        w_pad = (w_old // window_size + 1) * \
                            window_size - w_old
                        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                            :, :, :h_old + h_pad, :]
                        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                            :, :, :, :w_old + w_pad]

                        # run model
                        output = test(img_lq, model, args, window_size)

                        # fix output dim and value
                        output = output[..., :h_old *
                                        args.scale, :w_old * args.scale]
                        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        # model returns 3-band image, taking mean to create
                        # 1-band image
                        output = np.expand_dims(output.mean(axis=0), axis=0)
                        output = (output * image_value_range) + np.min(image)

                        # save output
                        dst_profile = src.profile
                        dst_profile['height'] = output.shape[1]
                        dst_profile['width'] = output.shape[2]
                        dst_profile['transform'] = rasterio.transform.from_bounds(
                            *src.bounds, dst_profile['width'], dst_profile['height'])

                        # create parent dir if does not exist
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        with rasterio.open(dst_file, 'w', **dst_profile) as dst:
                            dst.write(output)

            # update image entry
            aux_entry = {'file_name': str(dst_file)}
            kwcoco_extensions._populate_canvas_obj('', aux_entry)

            # strip-out args.dst_dir path; e.g. ./assets/<path>/file.tif
            dst_file = str(dst_file).replace(str(args.dst_dir), '')[1:]
            aux_entry['file_name'] = dst_file
            aux_entry['channels'] = 'superRes_' + channel

            image['auxiliary'].append(aux_entry)

        # update running image list
        new_images.append(image)

        # update running annotations list
        annotation = annotations_df[annotations_df.image_id == image['id']]
        annotation = fast_df_to_list_of_dict(annotation)
        new_annotations.extend(annotation)

    # update kwcoco data
    data['images'] = new_images
    data['annotations'] = new_annotations

    del annotations_df
    del new_images
    del new_annotations

    # save kwcoco data
    dst_kwcoco_file = args.dst_dir / \
        Path(args.src_kwcoco.stem.split('.')[0] + '_superRes.kwcoco.json')
    logging.info(f'Saving updated kwcoco file to... {dst_kwcoco_file}')

    with open(dst_kwcoco_file, 'w') as f:
        f.write(json.dumps(data))


def fast_df_to_list_of_dict(data: pd.DataFrame):
    return [dict(x) for i, x in data.iterrows()]


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use
            # ema for GAN training
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255
    # because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model.keys(
        ) else pretrained_model,
        strict=True)

    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ['jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[...,
                  h_idx * sf:(h_idx + tile) * sf,
                    w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[...,
                  h_idx * sf:(h_idx + tile) * sf,
                    w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()
