
# external
import kwcoco
import torch
import imagesize
import os
import pandas as pd
import json
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser

# internal libs
from . import utils
from .dzyne_img_util import normalizeRGB
from .models.basic_model import CDEvaluator

# dlau imports
from pathlib import Path
import subprocess
import shutil

from tqdm import tqdm
from shapely.geometry import shape

"""
Perform change detection on a pair of images
"""


def fast_df_to_list_of_dict(data: pd.DataFrame):
    return [dict(x) for i, x in data.iterrows()]


def _create_image(coco_img: kwcoco.CocoImage, channels: list, image_dir: Path, dst_file: Path):

    coco_delay = coco_img.delay(channels=channels, bundle_dpath=image_dir)
    image = coco_delay.finalize(nodata='float')
    image = image.astype(np.float32)

    image[image < 0] = 0
    image[image > 1] = 1

    image = image * 255
    image = image.astype(np.uint8)

    image = Image.fromarray(image)
    image.save(dst_file)

    assert (dst_file.exists() is True)

    return dst_file


def _get_rgb_and_depth_files_for_image(coco_image: kwcoco.CocoImage, dst_dir: Path, image_dir: Path):

    required_channels = ['red', 'green', 'blue', 'depth']
    auxs = {x['channels']: x for x in coco_image['auxiliary']}

    if set(required_channels).issubset(set(auxs.keys())) is True:

        image_name = coco_image['name']
        rgb_file = dst_dir / f'{image_name}_rgb.tif'
        depth_file = image_dir / auxs['depth']['file_name']

        if rgb_file.exists() is False:
            _ = _create_image(coco_image, ['red', 'green', 'blue'], image_dir, rgb_file)

        return (rgb_file, depth_file)

    print('WARNING, required bands not found in... id: {}, name: {}'.format(coco_image['id'], coco_image['id']))
    return None


def _move_file(src: Path, dst: Path):

    cmd = f'mv {src} {dst}'
    subprocess.check_output(cmd, shell=True)

    return dst.exists()


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='BIT_LEVIR', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--output_folder', type=Path, required=True, help='Location where to store kwcoco file and change-detection outputs')
    parser.add_argument('--src_kwcoco', type=Path, default='', required=True, help='kwcoco file to parse')

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    # parser.add_argument('--data_name', default='quick_start', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    """
    example:
    python3 -W ignore -m watch.tasks.change_detection.pair_BIT_CD     \
    --project_name CD_base_transformer_pos_s4_dd8_LEVIR_b2_lr0.01_trainval_test_1000_linear     \
    --net_G base_transformer_pos_s4_dd8     \
    --output_folder /output/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC     \
    --src_kwcoco /output/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_wv_superRes_depth.kwcoco.json \
    --checkpoint_root watch/tasks/change_detection/checkpoints/
    """

    # setup args
    args = get_args()
    utils.get_device(args)

    # setup model
    device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)  # TODO - use path

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    # setup output directory
    output_dir = args.output_folder  # TODO - underlying code requires args.output_folder; refactor
    output_dir.mkdir(parents=True, exist_ok=True)

    # load kwcoco file
    data = kwcoco.CocoDataset(args.src_kwcoco)

    # load images into dataframe and extract regions
    images_df = pd.DataFrame(data.dataset['images'])
    images_df['geos_corners'] = images_df.geos_corners.apply(lambda x: shape(x).wkt)
    images_df['image_index'] = images_df.index

    # loop through each region and perform change detection
    region_uuid = 0
    for region_name, region_df in tqdm(images_df.groupby('geos_corners'), desc='Regions'):
        region_df = region_df.sort_values(by='timestamp')
        coco_images = [data.coco_image(id) for id in region_df.id]

        if (len(coco_images) < 2):
            # change-detection only works when there is 2 or more images
            continue

        # create directory to store results for region
        relative_change_dir = Path('_assets/change_detection/')
        relative_region_change_dir = relative_change_dir / str(region_uuid)
        region_uuid += 1

        region_misc_dir = output_dir / relative_region_change_dir / 'misc'
        region_misc_dir.mkdir(parents=True, exist_ok=True)

        region_artifacts_dir = output_dir / relative_region_change_dir / 'artifacts'
        region_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # generate change-detection for all images relative to reference image; assumes images[0] is ref image
        reference_image = coco_images[0]

        for image_to_compare in tqdm(coco_images[1:], desc='Images'):

            dst_file = Path('{}-{}_change.tif'.format(reference_image['id'], image_to_compare['id']))
            dst_file = relative_region_change_dir / dst_file
            abs_dst_file = output_dir / dst_file

            if abs_dst_file.exists() is False:

                # get rgb and depth files for reference image
                reference_file_set = _get_rgb_and_depth_files_for_image(reference_image, region_artifacts_dir, args.src_kwcoco.parent)

                if reference_file_set is None:
                    break  # skip to next region b/c reference image doesn't have appropriate bands

                # get rgb and depth files for comparison image
                diff_file_set = _get_rgb_and_depth_files_for_image(image_to_compare, region_artifacts_dir, args.src_kwcoco.parent)

                if diff_file_set is None:
                    continue  # skip to next image b/c image doesn't have appropriate bands

                fname_imgA, fname_depthA = reference_file_set
                fname_imgB, fname_depthB = diff_file_set

                # create dataset loader? TODO - really just need to create batch obj to pass to model
                temp_dir_name = 'temp_rff'  # TODO - create super-res images in-memory
                utils.setup_data_folder(
                    temp_dir_name,
                    args.split,
                    str(fname_imgA), str(fname_imgB),
                    str(fname_depthA), str(fname_depthB))

                data_loader = utils.get_loader(
                    temp_dir_name,
                    img_size=args.img_size,
                    batch_size=args.batch_size,
                    split=args.split, is_train=False)

                try:
                    for i, batch in enumerate(data_loader):

                        # run the model
                        width, height = imagesize.get(fname_imgA)
                        score_map = model._forward_pass(batch)
                        model._save_predictions(size=(width, height))

                        # post-processing
                        fname_CD = output_dir / batch['name'][0]
                        _move_file(fname_CD, region_misc_dir / fname_CD.name)  # TODO - fix underlying class that relies on args.parser()
                        fname_CD = region_misc_dir / fname_CD.name
                        CD = np.asarray(Image.open(fname_CD))
                        corr = batch['corr'].cpu().data.numpy()[0, :, :]

                        comb = np.multiply(0.55 - 0.45 * corr, CD)
                        comb = cv2.GaussianBlur(comb, (5, 5), 0)
                        comb = cv2.GaussianBlur(comb, (5, 5), 0)

                        kernel = np.ones((5, 5), np.uint8)
                        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel)

                        # save results
                        # TODO - organize result using batch['name']
                        batch_name = batch['name'][0]
                        fname_comb = region_misc_dir / 'comb_{}'.format(batch_name)
                        Image.fromarray(comb.astype(np.uint8)).save(fname_comb)

                        norm = (255 * normalizeRGB(comb, (30, 100))).astype(np.uint8)
                        fname_norm = region_misc_dir / 'norm_{}'.format(batch_name)
                        Image.fromarray(norm.astype(np.uint8)).save(fname_norm)

                        fname_corr = region_misc_dir / 'corr_{}'.format(batch_name)
                        Image.fromarray(((corr + 1.0) * 100).astype(np.uint8)).save(fname_corr)

                        # TODO - easiest way to select which output represents change
                        _move_file(fname_comb, abs_dst_file)

                        # clean-up model temp directory
                        shutil.rmtree(temp_dir_name)
                except Exception:
                    txt = '{}: {}, {}'.format(region_name, reference_file_set, diff_file_set)
                    raise RuntimeError(txt)

            # create aux metadata
            auxs = {aux['channels']: aux for aux in image_to_compare['auxiliary']}
            aux_entry = auxs['depth'].copy()
            aux_entry['file_name'] = str(dst_file)
            aux_entry['channels'] = 'change'
            aux_entry['change_reference_image'] = reference_image['name']

            image_index = region_df[region_df['id'] == image_to_compare['id']].iloc[0]['image_index']
            if (data.dataset['images'][image_index]['id'] == image_to_compare['id']):
                data.dataset['images'][image_index]['auxiliary'].append(aux_entry)
            else:
                # TODO - haven't guaranteed that dataframe index is the same as actual image index in kwcoco file
                raise Exception('Invalid image_index for... {}'.format(image_to_compare['name']))

    # update kwcoco file
    dst_kwcoco_file = output_dir / Path(args.src_kwcoco).name.replace('.kwcoco.json', '_change.kwcoco.json')
    print(f'saving kwcoco to {dst_kwcoco_file}')

    with open(dst_kwcoco_file, 'w') as f:
        f.write(json.dumps(data.dataset))
