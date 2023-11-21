"""
Script used to analyze the material transition matrix (MTM) masks both qualitatively (by showing
material predictions and RGB scenes) and quanitatively (by computing metrics between the MTM and
BAS change masks).
"""

import os
import json
import pickle
import argparse

import cv2
import scipy
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from collections import defaultdict
# from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from geowatch.tasks.rutgers_material_seg_v2.matseg.models import build_model
from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_image import ImageStitcher_v2
from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_mat_tran_mask import compute_material_transition_mask
from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_dataset import MATERIAL_TO_MATID, colorize_material_mask, load_region_bas_annos
from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import load_cfg_file, generate_image_slice_object, create_conf_matrix_pred_image, create_gif

from geowatch.tasks.rutgers_material_seg_v2.mtm.dataset.median_dataset import MedianDataset
from geowatch.tasks.rutgers_material_seg_v2.mtm.dataset.early_late_dataset import EarlyLateDataset


def custom_collate_fn(data):
    list_key_names = [
        'crop_slice', 'early_image_ids', 'late_image_ids', 'vid', 'region_name', 'region_res'
    ]

    out_data = defaultdict(list)
    for ex in data:
        for k, v in ex.items():
            if isinstance(v, np.ndarray):
                # pylint: disable-next=Pylint(E1101:no-member)
                v = torch.tensor(v)
            out_data[k].append(v)

    for k, v in out_data.items():
        if k in list_key_names:
            out_data[k] = v
        else:
            # pylint: disable-next=Pylint(E1101:no-member)
            out_data[k] = torch.stack(v, dim=0)

    return out_data


def compute_metrics(pred_change, gt_change):
    pred = pred_change.flatten()
    gt = gt_change.flatten()
    precision, recall, f1_score, _ = precision_recall_fscore_support(gt,
                                                                     pred,
                                                                     zero_division=0,
                                                                     average='binary',
                                                                     pos_label=1)
    metrics = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
    return metrics


def standardize_image_sizes(image_stack):
    # Get all resolutions.
    heights, widths = [], []
    for img in image_stack:
        h, w = img.shape[1], img.shape[2]
        heights.append(h)
        widths.append(w)

    # Check if resolutions are all the same.
    check_heights = sum([v == heights[0] for v in heights[1:]]) == len(heights)
    check_widths = sum([v == widths[0] for v in widths[1:]]) == len(widths)

    if check_heights is False or check_widths is False:
        # Get median resolution and make sure all images are that size.
        med_h, med_w = np.median(heights), np.median(widths)

        # Resize image that does not match median resolutions.
        up_image_stack = []
        for img in image_stack:
            h, w = img.shape[1], img.shape[2]
            if (h != med_h) or (w != med_w):
                bands = []
                for c in range(img.shape[0]):
                    band = img[c]
                    band = cv2.resize(band, (int(med_w), int(med_h)),
                                      interpolation=cv2.INTER_LINEAR)
                    bands.append(band)
                img = np.stack(bands, axis=0)
            up_image_stack.append(img)
        assert len(up_image_stack) == len(image_stack)
        return up_image_stack
    else:
        # Stack contains images with all the same resolution.
        return image_stack


def resize_images(image_stack, target_height, target_width):
    n_dims = len(image_stack[0].shape)

    if image_stack[0].dtype == 'float64':
        inter_mode = cv2.INTER_LINEAR
    else:
        breakpoint()
        pass

    updated_image_stack = []
    for image in image_stack:
        if n_dims == 2:
            up_image = cv2.resize(image, (target_width, target_height), interpolation=inter_mode)
        elif n_dims == 3:
            bands = []
            for c in range(image.shape[0]):
                band = image[c]
                band = cv2.resize(band, (target_width, target_height), interpolation=inter_mode)
                bands.append(band)
            up_image = np.stack(bands, axis=0)
        else:
            breakpoint()
            pass
        updated_image_stack.append(up_image)

    return updated_image_stack


def generate_pair_MTM_masks(model,
                            eval_loader,
                            save_dir,
                            mat_trans_mask_mode,
                            resize_factor=1,
                            hueristic='basic'):
    # Initialize model in case it hasnt been already.
    # pylint: disable-next=Pylint(E1101:no-member)
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Generate material predictions for each image in each region.
    pred_stitcher = ImageStitcher_v2('./', save_backend='gdal')
    pbar = tqdm(eval_loader, colour='green', desc='Generating material predictions')
    with torch.no_grad():
        for data in pbar:
            B = data['early_frame'].shape[0]

            # Convert images into model format.

            # Pass data into model.
            data['image'] = data['early_frame'].to(device)
            early_pred = model.forward(data)
            data['image'] = data['late_frame'].to(device)
            late_pred = model.forward(data)

            # Convert predictions to numpy arrays.
            early_pred = early_pred.detach().cpu().numpy()
            late_pred = late_pred.detach().cpu().numpy()

            # Save images from the stitcher object.
            for i in range(B):
                early_sm_pred = scipy.special.softmax(early_pred[i], axis=0)
                late_sm_pred = scipy.special.softmax(late_pred[i], axis=0)

                try:
                    early_image_name = f"{data['region_name'][i]}_0_{data['early_image_ids'][i][0]}"
                    late_image_name = f"{data['region_name'][i]}_1_{data['late_image_ids'][i][0]}"
                except KeyError:
                    early_image_name = f"{data['region_name'][i]}_0_{data['early_image_id'][i]}"
                    late_image_name = f"{data['region_name'][i]}_1_{data['late_image_id'][i]}"

                ex_crop_params = []
                for j in range(4):
                    ex_crop_params.append(data['crop_slice'][j][i].item())
                height = data['region_res'][0][i].item()
                width = data['region_res'][1][i].item()
                pred_stitcher.add_image(early_sm_pred, early_image_name, ex_crop_params, height,
                                        width)
                pred_stitcher.add_image(late_sm_pred, late_image_name, ex_crop_params, height,
                                        width)

    # Finalize stitching operation and save images.
    stitched_predictions = pred_stitcher.get_combined_images()

    ## Sort by regions.
    region_predictions = {}
    for image_name, image in stitched_predictions.items():
        region_name = '_'.join(image_name.split('_')[:2])
        if region_name not in region_predictions.keys():
            region_predictions[region_name] = {'0': [], '1': []}
        first_last = image_name.split('_')[2]
        region_predictions[region_name][first_last].append(image)

    # Load GT BAS change masks.
    gt_bas_annos = load_region_bas_annos(drop_version=6, gsd=10)

    # Generate material transition matrix.
    mat_trans_masks, region_beg_mat_pred, region_end_mat_pred = {}, {}, {}
    for region_name, region_preds in tqdm(region_predictions.items(),
                                          colour='red',
                                          desc='Generating MTM masks'):
        # Make sure that all images in the stack are the same size.
        region_bas_anno = gt_bas_annos[region_name]

        region_preds['0'] = resize_images(region_preds['0'], region_bas_anno.shape[0],
                                          region_bas_anno.shape[1])
        region_preds['1'] = resize_images(region_preds['1'], region_bas_anno.shape[0],
                                          region_bas_anno.shape[1])

        # Stack the images together.
        first_preds = np.stack(region_preds['0'], axis=0)
        last_preds = np.stack(region_preds['1'], axis=0)

        # Compute the material transition mask.
        mat_trans_mask, beg_mat_pred, end_mat_pred = compute_material_transition_mask(
            mat_trans_mask_mode, first_preds, last_preds, heuristic=hueristic)

        # Record items.
        mat_trans_masks[region_name] = mat_trans_mask
        region_beg_mat_pred[region_name] = beg_mat_pred
        region_end_mat_pred[region_name] = end_mat_pred

        # Save beginning and end material predictions.
        color_beg_mask = colorize_material_mask(beg_mat_pred)
        color_end_mask = colorize_material_mask(end_mat_pred)
        mat_pred_save_dir = os.path.join(save_dir, region_name)
        os.makedirs(mat_pred_save_dir, exist_ok=True)
        Image.fromarray(color_beg_mask).save(os.path.join(mat_pred_save_dir, 'beg_mat_pred.png'))
        Image.fromarray(color_end_mask).save(os.path.join(mat_pred_save_dir, 'end_mat_pred.png'))

        # Save MTM prediction RGB image.
        Image.fromarray(
            (mat_trans_mask * 255).astype('uint8')).save(os.path.join(mat_pred_save_dir, 'mtm.png'))

        # Save MTM and BAS GT comparison image.
        cm_img = create_conf_matrix_pred_image(mat_trans_mask, region_bas_anno)
        Image.fromarray(cm_img).save(os.path.join(mat_pred_save_dir, 'cm_img.png'))

    # Compute the metrics between material change and BAS.
    region_metrics = {}
    region_bas_annos = gt_bas_annos
    for region_name, mat_trans_mask in mat_trans_masks.items():
        bas_mask = region_bas_annos[region_name]
        clipped_mtm = (mat_trans_mask > 0.5).astype('int')
        metrics = compute_metrics(clipped_mtm, bas_mask)
        region_metrics[region_name] = metrics

    # Compute the macro metrics.
    all_recall, all_precision, all_f1score = [], [], []
    for region_metric in region_metrics.values():
        all_recall.append(region_metric['recall'])
        all_precision.append(region_metric['precision'])
        all_f1score.append(region_metric['f1_score'])

    macro_metrics = {
        'recall': np.mean(all_recall),
        'precision': np.mean(all_precision),
        'f1_score': np.mean(all_f1score)
    }
    region_metrics['all_macro'] = macro_metrics

    # Save metrics.
    metrics_path = os.path.join(save_dir, 'metrics.json')
    json.dump(region_metrics, open(metrics_path, 'w'), indent=2)

    # Save MTM masks.
    mtm_path = os.path.join(save_dir, 'mtm_masks.p')
    pickle.dump(mat_trans_masks, open(mtm_path, 'wb'))


def generate_median_MTM_masks(model,
                              eval_loader,
                              save_dir,
                              mat_trans_mask_mode,
                              resize_factor=1,
                              hueristic='basic'):
    # Initialize model in case it hasnt been already.
    # pylint: disable-next=Pylint(E1101:no-member)
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Generate material predictions for each image in each region.
    pred_stitcher = ImageStitcher_v2('./', save_backend='gdal')
    pbar = tqdm(eval_loader, colour='green', desc='Generating material predictions')
    with torch.no_grad():
        for data in pbar:

            B = data['early_frame'].shape[0]

            # Convert images into model format.

            # Pass data into model.
            data['image'] = data['early_frame'].to(device)
            early_pred = model.forward(data)
            data['image'] = data['late_frame'].to(device)
            late_pred = model.forward(data)

            # Convert predictions to numpy arrays.
            early_pred = early_pred.detach().cpu().numpy()
            late_pred = late_pred.detach().cpu().numpy()

            # Save images from the stitcher object.
            for i in range(B):
                # Add the early and late predictions to the stitcher.
                early_sm_pred = scipy.special.softmax(early_pred[i], axis=0)
                late_sm_pred = scipy.special.softmax(late_pred[i], axis=0)

                early_image_name = f"{data['region_name'][i]}_{data['early_image_ids'][0][i]}_0"
                late_image_name = f"{data['region_name'][i]}_{data['late_image_ids'][0][i]}_1"
                ex_crop_params = []
                for j in range(4):
                    ex_crop_params.append(data['crop_slice'][j][i].item())
                height = data['region_res'][0][i].item()
                width = data['region_res'][1][i].item()
                pred_stitcher.add_image(early_sm_pred, early_image_name, ex_crop_params, height,
                                        width)
                pred_stitcher.add_image(late_sm_pred, late_image_name, ex_crop_params, height,
                                        width)

                # Add the early and late RGB images to the stitcher.
                early_image = data['early_frame'][i].detach().cpu().numpy()
                late_image = data['late_frame'][i].detach().cpu().numpy()
                early_image_rgb_name = f"{data['region_name'][i]}_{data['early_image_ids'][0][i]}_0_rgb"
                late_image_rgb_name = f"{data['region_name'][i]}_{data['late_image_ids'][0][i]}_1_rgb"
                pred_stitcher.add_image(early_image, early_image_rgb_name, ex_crop_params, height,
                                        width)
                pred_stitcher.add_image(late_image, late_image_rgb_name, ex_crop_params, height,
                                        width)

    # Finalize stitching operation and save images.
    stitched_predictions = pred_stitcher.get_combined_images()

    # Generate RGB images.
    for image_name, image in stitched_predictions.items():
        if 'rgb' in image_name:
            rgb_image = eval_loader.dataset.to_RGB(image)
            stitched_predictions[image_name] = rgb_image

    ## Sort by regions.
    region_predictions, region_rgb_images = {}, {}
    for image_name, image in stitched_predictions.items():
        region_name = '_'.join(image_name.split('_')[:2])
        if 'rgb' in image_name:
            if region_name not in region_rgb_images.keys():
                region_rgb_images[region_name] = {'0': [], '1': []}
            first_last = image_name.split('_')[-2]
            region_rgb_images[region_name][first_last].append(image)
        else:
            if region_name not in region_predictions.keys():
                region_predictions[region_name] = {'0': [], '1': []}
            first_last = image_name.split('_')[-1]
            region_predictions[region_name][first_last].append(image)

    # Load GT BAS change masks.
    gt_bas_annos = load_region_bas_annos(drop_version=6)

    # Generate material transition matrix.
    mat_trans_masks, region_beg_mat_pred, region_end_mat_pred = {}, {}, {}
    for region_name, region_preds in tqdm(region_predictions.items(),
                                          colour='red',
                                          desc='Generating MTM masks'):
        # Make sure that all images in the stack are the same size.
        region_bas_anno = gt_bas_annos[region_name]

        region_preds['0'] = resize_images(region_preds['0'], region_bas_anno.shape[0],
                                          region_bas_anno.shape[1])
        region_preds['1'] = resize_images(region_preds['1'], region_bas_anno.shape[0],
                                          region_bas_anno.shape[1])

        # Stack the images together.
        first_preds = np.stack(region_preds['0'], axis=0)
        last_preds = np.stack(region_preds['1'], axis=0)

        # Compute the material transition mask.
        mat_trans_mask, beg_mat_pred, end_mat_pred = compute_material_transition_mask(
            mat_trans_mask_mode, first_preds, last_preds, heuristic=hueristic)

        # Record items.
        mat_trans_masks[region_name] = mat_trans_mask
        region_beg_mat_pred[region_name] = beg_mat_pred
        region_end_mat_pred[region_name] = end_mat_pred

        # Save beginning and end material predictions.
        color_beg_mask = colorize_material_mask(beg_mat_pred)
        color_end_mask = colorize_material_mask(end_mat_pred)
        mat_pred_save_dir = os.path.join(save_dir, region_name)
        os.makedirs(mat_pred_save_dir, exist_ok=True)
        Image.fromarray(color_beg_mask).save(os.path.join(mat_pred_save_dir, 'beg_mat_pred.png'))
        Image.fromarray(color_end_mask).save(os.path.join(mat_pred_save_dir, 'end_mat_pred.png'))

        # Save MTM prediction RGB image.
        Image.fromarray(
            (mat_trans_mask * 255).astype('uint8')).save(os.path.join(mat_pred_save_dir, 'mtm.png'))

        # Save MTM and BAS GT comparison image.
        mat_trans_mask = np.floor(mat_trans_mask)
        cm_img = create_conf_matrix_pred_image(mat_trans_mask, region_bas_anno)
        Image.fromarray(cm_img).save(os.path.join(mat_pred_save_dir, 'cm_img.png'))

        # Create GIF of material transition from early and late transitions.
        early_rgb_img = rearrange(region_rgb_images[region_name]['0'][0], 'c h w -> h w c')
        late_rgb_img = rearrange(region_rgb_images[region_name]['1'][0], 'c h w -> h w c')
        Image.fromarray((early_rgb_img * 255).astype('uint8')).save(
            os.path.join(mat_pred_save_dir, 'early_rgb.png'))
        Image.fromarray((late_rgb_img * 255).astype('uint8')).save(
            os.path.join(mat_pred_save_dir, 'late_rgb.png'))

        ## Prediction GIF.
        p_x, p_y = np.where(mat_trans_mask == 1)
        overlap_pred_save_path = os.path.join(mat_pred_save_dir, 'overlap_pred.gif')
        pred_first_rgb_canvas = np.zeros(early_rgb_img.shape, dtype='float')
        pred_last_rgb_canvas = np.zeros(late_rgb_img.shape, dtype='float')
        pred_first_rgb_canvas[p_x, p_y] = early_rgb_img[p_x, p_y]
        pred_last_rgb_canvas[p_x, p_y] = late_rgb_img[p_x, p_y]
        pred_first_rgb_canvas = (pred_first_rgb_canvas * 255).astype('uint8')
        pred_last_rgb_canvas = (pred_last_rgb_canvas * 255).astype('uint8')
        create_gif([pred_first_rgb_canvas, pred_last_rgb_canvas], overlap_pred_save_path)

        ## True positive GIF.
        tp_x, tp_y = np.where((mat_trans_mask == 1) & (region_bas_anno == 1))
        overlap_tp_save_path = os.path.join(mat_pred_save_dir, 'overlap_tp.gif')
        tp_first_rgb_canvas = np.zeros(early_rgb_img.shape, dtype='float')
        tp_last_rgb_canvas = np.zeros(late_rgb_img.shape, dtype='float')
        tp_first_rgb_canvas[tp_x, tp_y] = early_rgb_img[tp_x, tp_y]
        tp_last_rgb_canvas[tp_x, tp_y] = late_rgb_img[tp_x, tp_y]
        tp_first_rgb_canvas = (tp_first_rgb_canvas * 255).astype('uint8')
        tp_last_rgb_canvas = (tp_last_rgb_canvas * 255).astype('uint8')
        create_gif([tp_first_rgb_canvas, tp_last_rgb_canvas], overlap_tp_save_path)

        ## False positive GIF.
        fp_x, fp_y = np.where((mat_trans_mask == 1) & (region_bas_anno == 0))
        overlap_fp_save_path = os.path.join(mat_pred_save_dir, 'overlap_fp.gif')
        fp_first_rgb_canvas = np.zeros(early_rgb_img.shape, dtype='float')
        fp_last_rgb_canvas = np.zeros(late_rgb_img.shape, dtype='float')
        fp_first_rgb_canvas[fp_x, fp_y] = early_rgb_img[fp_x, fp_y]
        fp_last_rgb_canvas[fp_x, fp_y] = late_rgb_img[fp_x, fp_y]
        fp_first_rgb_canvas = (fp_first_rgb_canvas * 255).astype('uint8')
        fp_last_rgb_canvas = (fp_last_rgb_canvas * 255).astype('uint8')
        create_gif([fp_first_rgb_canvas, fp_last_rgb_canvas], overlap_fp_save_path)

        ## False negative GIF.
        fn_x, fn_y = np.where((mat_trans_mask == 0) & (region_bas_anno == 1))
        overlap_fn_save_path = os.path.join(mat_pred_save_dir, 'overlap_fn.gif')
        fn_first_rgb_canvas = np.zeros(early_rgb_img.shape, dtype='float')
        fn_last_rgb_canvas = np.zeros(late_rgb_img.shape, dtype='float')
        fn_first_rgb_canvas[fn_x, fn_y] = early_rgb_img[fn_x, fn_y]
        fn_last_rgb_canvas[fn_x, fn_y] = late_rgb_img[fn_x, fn_y]
        fn_first_rgb_canvas = (fn_first_rgb_canvas * 255).astype('uint8')
        fn_last_rgb_canvas = (fn_last_rgb_canvas * 255).astype('uint8')
        create_gif([fn_first_rgb_canvas, fn_last_rgb_canvas], overlap_fn_save_path)

    # Compute the metrics between material change and BAS.
    region_metrics = {}
    for region_name, mat_trans_mask in mat_trans_masks.items():
        bas_mask = gt_bas_annos[region_name]
        mat_trans_mask = np.floor(mat_trans_mask)
        metrics = compute_metrics(mat_trans_mask, bas_mask)
        region_metrics[region_name] = metrics

    # Compute the macro metrics.
    all_recall, all_precision, all_f1score = [], [], []
    for region_metric in region_metrics.values():
        all_recall.append(region_metric['recall'])
        all_precision.append(region_metric['precision'])
        all_f1score.append(region_metric['f1_score'])

    macro_metrics = {
        'recall': np.mean(all_recall),
        'precision': np.mean(all_precision),
        'f1_score': np.mean(all_f1score)
    }
    region_metrics['all_macro'] = macro_metrics

    # Save metrics.
    metrics_path = os.path.join(save_dir, 'metrics.json')
    json.dump(region_metrics, open(metrics_path, 'w'), indent=2)

    # Save MTM masks.
    mtm_path = os.path.join(save_dir, 'mtm_masks.p')
    pickle.dump(mat_trans_masks, open(mtm_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MTM masks')
    parser.add_argument('material_model_path',
                        type=str,
                        help='Path to material segmentation model weights (.ckpt file).')
    parser.add_argument(
        'mat_trans_mask_mode',
        type=str,
        choices=['hard_class_1', 'hard_class_2', 'hard_quality_1', 'hard_quality_2'],
        help='If using "median" setting, then hard_class_2 is recommended where is not use \
             "hard_quality_2" mode.')
    parser.add_argument('n_first_last_images',
                        type=int,
                        help='Number of images to consider within \
                         the begginning and end of the sequence.')
    parser.add_argument(
        '--select_regions',
        nargs='+',
        default=['KR_R001', 'KR_R002', 'AE_R001', 'BR_R002', 'US_C012'],
        help='Which regions to run predict script on. Note: Only works for "mat_change" flag.')
    parser.add_argument('--median',
                        default=False,
                        action='store_true',
                        help='Compute the median the images instead of voting.')
    parser.add_argument('--dset_name', type=str, default='drop6', help='Name of dataset to use.')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of workers to use.')
    parser.add_argument('--ignore_winter_data',
                        default=True,
                        action='store_false',
                        help='Used in dataset class to avoid images taken in winter \
                             (to avoid snow).')
    parser.add_argument('--quality_pct_criteria',
                        type=float,
                        default=0.8,
                        help='Given quality \
                         masks (non-median), this is the threshold to determine if the image has \
                         high enough quality to be used.')
    parser.add_argument('--hueristic',
                        type=str,
                        default='soften_seasonal',
                        choices=['basic', 'soften_seasonal'],
                        help='Hueristic to use for creating MTM.')

    args = parser.parse_args()

    # Load model configuration file.

    ## Get experiment directory.
    exp_dir = '/'.join(args.material_model_path.split('/')[:-2])

    ## Load configuration file.
    cfg_path = os.path.join(exp_dir, '.hydra', 'config.yaml')
    cfg = load_cfg_file(cfg_path)

    ## Overwrite configuration file.
    if args.n_workers is not None:
        cfg.n_workers = args.n_workers

    # Load dataset.
    if args.median:
        dataset_class = MedianDataset
    else:
        dataset_class = EarlyLateDataset

    # TODO: Update the path to the dataset.
    print('WARNING: HARDCODED PATH TO SIMPLE DATASET!')
    kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop6/data.kwcoco.zip'
    crop_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width, cfg.crop_stride)
    dataset = dataset_class(split='eval',
                            kwcoco_path=kwcoco_path,
                            crop_params=crop_params,
                            channels=cfg.dataset.channels,
                            ignore_winter_data=args.ignore_winter_data,
                            quality_pct_criteria=args.quality_pct_criteria,
                            first_last_samples=args.n_first_last_images,
                            subset_region_names=args.select_regions)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=cfg.batch_size,
                                         shuffle=False,
                                         num_workers=cfg.n_workers)

    # Load model.
    if cfg.model.kwargs is None:
        cfg.model.kwargs = {}
    n_channels = dataset.n_channels
    model = build_model(None,
                        network_name=cfg.model.architecture,
                        encoder_name=cfg.model.encoder,
                        in_channels=n_channels,
                        out_channels=len(MATERIAL_TO_MATID.keys()),
                        loss_mode=cfg.model.loss_mode,
                        optimizer_mode=cfg.model.optimizer_mode,
                        class_weight_mode=cfg.model.class_weight_mode,
                        lr=cfg.model.lr,
                        wd=cfg.model.wd,
                        pretrain=cfg.model.pretrain,
                        to_rgb_fcn=None,
                        **cfg.model.kwargs,
                        checkpoint_path=args.material_model_path)

    # Get material segmentation predictions.
    short_exp_name = '_'.join(exp_dir.split('/')[-2:])
    if args.median:
        prefix = 'median'
    else:
        prefix = 'fl_pair'
    mtm_dir = os.path.join(
        exp_dir,
        f"analysis/mtm_preds/{prefix}_{args.mat_trans_mask_mode}_{args.n_first_last_images}")

    if args.median:
        generate_median_MTM_masks(model,
                                  loader,
                                  mtm_dir,
                                  args.mat_trans_mask_mode,
                                  hueristic=args.hueristic)
    else:
        generate_pair_MTM_masks(model,
                                loader,
                                mtm_dir,
                                args.mat_trans_mask_mode,
                                hueristic=args.hueristic)
