import os
import json
import argparse

import cv2
import torch
import scipy
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from tifffile import tifffile
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from watch.tasks.rutgers_material_seg_v2.matseg.models import build_model
from watch.tasks.rutgers_material_seg_v2.matseg.datasets import build_dataset
from watch.tasks.rutgers_material_seg_v2.matseg.datasets.bas_dataset import BAS_Dataset
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_image import ImageStitcher, ImageStitcher_v2
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_mat_tran_mask import compute_material_transition_mask
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import load_cfg_file, generate_image_slice_object, create_conf_matrix_pred_image
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_dataset import get_labelbox_material_labels, MATERIAL_TO_MATID, colorize_material_mask


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


def generate_material_predictions_v2(model, eval_loader, exp_dir, mat_labels, resize_factor=1):
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Get save dir to store prediction images.
    pred_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    stitcher_save_dir = os.path.join(pred_dir, 'mat_img_preds')
    os.makedirs(stitcher_save_dir, exist_ok=True)

    # pred_stitcher = ImageStitcher(stitcher_save_dir, save_backend='gdal')
    pred_stitcher = ImageStitcher_v2(stitcher_save_dir, save_backend='gdal')
    img_name_to_region = {}
    with torch.no_grad():
        for data in tqdm(eval_loader, colour='green', desc='Generating material predictions'):
            B = data['image'].shape[0]

            # Convert images into model format.
            data['image'] = data['image'].to(device)

            # Pass data into model.
            prediction = model.forward(data)

            # Convert predictions to numpy arrays.
            prediction = prediction.detach().cpu().numpy()

            # Resize if parameter is less then 1.
            if resize_factor != 1:
                print('WARNING: Inside resize factor loop!!!')
                prediction = rearrange(prediction, 'b c h w -> h w b c')
                h, w = prediction.shape[0], prediction.shape[1]
                rs_preds = []
                for i in range(B):
                    single_pred = prediction[:, :, i]
                    single_pred = cv2.resize(single_pred,
                                             dsize=(h // resize_factor, w // resize_factor),
                                             interpolation=cv2.INTER_LINEAR)
                    rs_preds.append(single_pred)
                rs_preds = np.stack(rs_preds, axis=0)
                prediction = rearrange(rs_preds, 'b h w c -> b c h w')

            # Save images from the stitcher object.
            for i in range(B):
                sm_pred = scipy.special.softmax(prediction[i], axis=0)
                image_name = data['img_id'][i]
                ex_crop_params = []
                for j in range(4):
                    ex_crop_params.append(data['crop_slice'][j][i].item())
                height = data['og_size'][0][i].item()
                width = data['og_size'][1][i].item()
                img_name_to_region[image_name] = data['region_name'][i]
                pred_stitcher.add_image(sm_pred, image_name, ex_crop_params, height, width,
                                        data['buffer_mask'][i])

        # Finalize stitching operation and save images.
        stitched_predictions = pred_stitcher.get_combined_images()
        # save_paths, image_names, image_sizes = pred_stitcher.save_images()

        region_predictions, reigon_q_masks = {}, {}
        for image_name, mat_conf in stitched_predictions.items():
            # region_name = '_'.join(image_name.split('_')[:2])
            region_name = img_name_to_region[image_name]
            if region_name not in region_predictions.keys():
                region_predictions[region_name] = {'0': [], '1': []}
                reigon_q_masks[region_name] = {'0': [], '1': []}
            first_last = image_name.split('_')[2]

            mat_pred = mat_conf.argmax(axis=0)
            region_predictions[region_name][first_last].append(mat_conf)

            # Save MAT-RGB version of prediction.
            mat_pred_save_dir = os.path.join(stitcher_save_dir, region_name, 'pred', first_last)
            os.makedirs(mat_pred_save_dir, exist_ok=True)
            mat_pred_save_path = os.path.join(mat_pred_save_dir, image_name + '.png')
            rgb_pred = colorize_material_mask(mat_pred)
            Image.fromarray(rgb_pred).save(mat_pred_save_path)
            print(mat_pred_save_path)


def generate_material_predictions(model, eval_loader, exp_dir, mat_labels, resize_factor=1):
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Get save dir to store prediction images.
    pred_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    stitcher_save_dir = os.path.join(pred_dir, 'mat_img_preds')
    os.makedirs(stitcher_save_dir, exist_ok=True)

    pred_stitcher = ImageStitcher(stitcher_save_dir, save_backend='gdal')
    with torch.no_grad():
        for data in tqdm(eval_loader, colour='green', desc='Generating material predictions'):
            B = data['image'].shape[0]

            # Convert images into model format.
            data['image'] = data['image'].to(device)

            # Pass data into model.
            prediction = model.forward(data)

            # Convert predictions to numpy arrays.
            prediction = prediction.detach().cpu().numpy()

            # Resize if parameter is less then 1.
            if resize_factor != 1:
                prediction = rearrange(prediction, 'b c h w -> h w b c')
                h, w = prediction.shape[0], prediction.shape[1]
                rs_preds = []
                for i in range(B):
                    single_pred = prediction[:, :, i]
                    single_pred = cv2.resize(single_pred,
                                             dsize=(h // resize_factor, w // resize_factor),
                                             interpolation=cv2.INTER_LINEAR)
                    rs_preds.append(single_pred)
                rs_preds = np.stack(rs_preds, axis=0)
                prediction = rearrange(rs_preds, 'b h w c -> b c h w')

            # Save images from the stitcher object.
            for i in range(B):
                sm_pred = scipy.special.softmax(prediction[i], axis=0)
                image_name = data['image_dir'][i].split('/')[-1]
                ex_crop_params = []
                for j in range(4):
                    ex_crop_params.append(data['crop_slice'][j][i].item())
                height = data['og_size'][0][i].item()
                width = data['og_size'][1][i].item()
                pred_stitcher.add_image(sm_pred, image_name, ex_crop_params, height, width,
                                        data['buffer_mask'][i])

        # Finalize stitching operation and save images.
        save_paths, image_names, image_sizes = pred_stitcher.save_images()

        # Convert those images into material predictions.
        for save_path, image_name, image_size in tqdm(zip(save_paths, image_names, image_sizes),
                                                      colour='green',
                                                      desc='Saving material predictions'):
            # Get material prediction
            mat_pred = tifffile.imread(save_path).argmax(axis=2)

            # Convert to RGB material images.
            rgb_mat_pred = colorize_material_mask(mat_pred)

            # Find GT material label using the image name.
            for label in mat_labels:
                label_img_name = label['image_dir'].split('/')[-1]
                img_name = save_path.split('/')[-1][:-4]
                if img_name == label_img_name:
                    gt_mat_mask = label['mat_mask']
                    region_name = label['region_name']

            # Save RGB image.
            region_save_dir = os.path.join(os.path.split(save_path)[0], region_name)
            os.makedirs(region_save_dir, exist_ok=True)

            img_name = os.path.split(save_path)[1][:-4]

            rgb_mat_save_path = os.path.join(region_save_dir, img_name + '_mat_pred.png')
            Image.fromarray(rgb_mat_pred).save(rgb_mat_save_path)

            # Save GT material label.
            rgb_gt_mat_pred = colorize_material_mask(gt_mat_mask)
            rgb_mat_gt_save_path = os.path.join(region_save_dir, img_name + '_mat_gt.png')
            Image.fromarray(rgb_gt_mat_pred).save(rgb_mat_gt_save_path)

            # Create image of mismatched materials.
            correct_mask = np.zeros(mat_pred.shape, dtype='uint8')
            x, y = np.where((mat_pred != gt_mat_mask) & (gt_mat_mask != 0))
            correct_mask[x, y] = 255
            correct_save_path = os.path.join(region_save_dir, img_name + '_incorrect.png')
            Image.fromarray(correct_mask).save(correct_save_path)

            # Delete file.
            os.remove(save_path)


def create_mat_change_visualizations(mat_trans_mask, bas_mask, metrics, region_name,
                                     first_mat_preds, last_mat_preds, base_save_dir):
    # Create confusion matrix image.
    conf_mask = create_conf_matrix_pred_image(mat_trans_mask, bas_mask)
    #
    # Create main plots.
    ## Create save path.
    save_dir = os.path.join(base_save_dir, 'predictions', 'change_plots', region_name)
    os.makedirs(save_dir, exist_ok=True)
    #
    ## Generate plot.
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(mat_trans_mask, cmap='gray')
    axes[0].set_xlabel(f'Recall: {metrics["recall"]}')
    axes[0].set_title('Prediction')
    axes[1].imshow(bas_mask, cmap='gray')
    axes[1].set_xlabel(f'Precision: {metrics["precision"]}')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(conf_mask)
    axes[2].set_xlabel(f'F1-score: {metrics["f1_score"]}')
    axes[2].set_title('Class Overlap')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'change_overlap.png'))
    plt.close()
    #
    # Create material prediction plots.
    fig, axes = plt.subplots(1, 2, figsize=(40, 20))
    fframe = first_mat_preds
    lframe = last_mat_preds
    color_fframe = colorize_material_mask(fframe)
    color_lframe = colorize_material_mask(lframe)
    axes[0].imshow(color_fframe)
    axes[1].imshow(color_lframe)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{region_name}_mat_preds.png'))
    plt.close()


def compute_change_detection_perf(model,
                                  eval_loader,
                                  exp_dir,
                                  mat_trans_mask_mode='hard_class_1',
                                  resize_factor=1):
    base_save_dir = os.path.join(exp_dir, 'predictions', 'mat_change')
    os.makedirs(base_save_dir, exist_ok=True)

    # Initialize model in case it hasnt been already.
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Generate material predictions for each image in each region.
    pred_stitcher = ImageStitcher_v2('./', save_backend='gdal')
    pbar = tqdm(eval_loader, colour='green', desc='Generating material predictions')
    with torch.no_grad():
        for data in pbar:

            B = data['image'].shape[0]

            # Convert images into model format.
            data['image'] = data['image'].to(device)

            # Pass data into model.
            prediction = model.forward(data)

            # Convert predictions to numpy arrays.
            prediction = prediction.detach().cpu().numpy()

            # Resize if parameter is less then 1.
            if resize_factor != 1:
                prediction = rearrange(prediction, 'b c h w -> h w b c')
                h, w = prediction.shape[0], prediction.shape[1]
                rs_preds = []
                for i in range(B):
                    single_pred = prediction[:, :, i]
                    single_pred = cv2.resize(single_pred,
                                             dsize=(h // resize_factor, w // resize_factor),
                                             interpolation=cv2.INTER_LINEAR)
                    rs_preds.append(single_pred)
                rs_preds = np.stack(rs_preds, axis=0)
                prediction = rearrange(rs_preds, 'b h w c -> b c h w')

            # Save images from the stitcher object.
            for i in range(B):
                sm_pred = scipy.special.softmax(prediction[i], axis=0)
                image_name = f"{data['region_name'][i]}_{data['first_last'][i]}_{data['first_last_index'][i]}"
                ex_crop_params = []
                for j in range(4):
                    ex_crop_params.append(data['crop_slice'][j][i].item())
                height = data['og_size'][0][i].item()
                width = data['og_size'][1][i].item()
                pred_stitcher.add_image(sm_pred, image_name, ex_crop_params, height, width,
                                        data['buffer_mask'][i])

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

    # Generate material transition matrix.
    mat_trans_masks, region_beg_mat_pred, region_end_mat_pred = {}, {}, {}
    for region_name, region_preds in region_predictions.items():
        # Make sure that all images in the stack are the same size.
        region_preds['0'] = standardize_image_sizes(region_preds['0'])
        region_preds['1'] = standardize_image_sizes(region_preds['1'])

        # Stack the images together.
        first_preds = np.stack(region_preds['0'], axis=0)
        last_preds = np.stack(region_preds['1'], axis=0)

        # Compute the material transition mask,
        mat_trans_mask, beg_mat_pred, end_mat_pred = compute_material_transition_mask(
            mat_trans_mask_mode, first_preds, last_preds)

        # Record items.
        mat_trans_masks[region_name] = mat_trans_mask
        region_beg_mat_pred[region_name] = beg_mat_pred
        region_end_mat_pred[region_name] = end_mat_pred

    # Compute the metrics between material change and BAS.
    region_metrics = {}
    region_bas_annos = eval_loader.dataset.bas_annos
    for region_name, mat_trans_mask in mat_trans_masks.items():
        bas_mask = region_bas_annos[region_name]
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

    # Compute estimated material change of regions of change.
    base_change_mats_dir = os.path.join(base_save_dir, 'mat_change_cm')
    os.makedirs(base_change_mats_dir, exist_ok=True)

    for region_name in region_beg_mat_pred.keys():
        bas_save_path = os.path.join(base_change_mats_dir, f'{region_name}_BAS.png')

        # Get where there is change.
        bas_mask = region_bas_annos[region_name]
        x, y = np.where(bas_mask == 1)
        xn, yn = np.where(bas_mask != 0)

        # Generate CM for BAS regions.
        if x.shape[0] != 0:

            # Get average begining and end material prediction masks.
            beg_mat_pred = region_beg_mat_pred[region_name][x, y]
            end_mat_pred = region_end_mat_pred[region_name][x, y]

            # Create confusion matrix for this region
            region_cm = ConfusionMatrix(actual_vector=list(beg_mat_pred.flatten()),
                                        predict_vector=list(end_mat_pred.flatten()),
                                        classes=list(MATERIAL_TO_MATID.values()))

            cm_array = region_cm.to_array()
            df_cm = pd.DataFrame(cm_array, MATERIAL_TO_MATID.keys(), MATERIAL_TO_MATID.keys())

            # Crop unused classes.
            drop_df_cm = df_cm.drop(['unknown', 'metal', 'concrete'], axis=0)
            drop_df_cm = drop_df_cm.drop(['unknown', 'metal', 'concrete'], axis=1)

            # Reorder columns and rows.
            ord_df_cm = drop_df_cm[['water', 'soil', 'vegetation', 'snow', 'polymer', 'asphalt']]

            plt.figure(figsize=(10, 7))
            sn.set(font_scale=1.4)  # for label size
            ax = sn.heatmap(ord_df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues')
            ax.set(xlabel='End Material', ylabel='Beginning Material')
            plt.savefig(bas_save_path)
            plt.close()

        # Generate CM for non-BAS regions.
        if x.shape[0] != 0:
            xn, yn = np.where(bas_mask != 0)
            non_bas_save_path = os.path.join(base_change_mats_dir, f'{region_name}_non_BAS.png')

            # Get average begining and end material prediction masks.
            beg_mat_pred = region_beg_mat_pred[region_name][xn, yn]
            end_mat_pred = region_end_mat_pred[region_name][xn, yn]

            # Create confusion matrix for this region
            region_cm = ConfusionMatrix(actual_vector=list(beg_mat_pred.flatten()),
                                        predict_vector=list(end_mat_pred.flatten()),
                                        classes=list(MATERIAL_TO_MATID.values()))

            cm_array = region_cm.to_array()
            df_cm = pd.DataFrame(cm_array, MATERIAL_TO_MATID.keys(), MATERIAL_TO_MATID.keys())

            # Crop unused classes.
            drop_df_cm = df_cm.drop(['unknown', 'metal', 'concrete'], axis=0)
            drop_df_cm = drop_df_cm.drop(['unknown', 'metal', 'concrete'], axis=1)

            # Reorder columns and rows.
            ord_df_cm = drop_df_cm[['water', 'soil', 'vegetation', 'snow', 'polymer', 'asphalt']]

            plt.figure(figsize=(10, 7))
            sn.set(font_scale=1.4)  # for label size
            ax = sn.heatmap(ord_df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues')
            ax.set(xlabel='End Material', ylabel='Beginning Material')
            plt.savefig(non_bas_save_path)
            plt.close()

        # Generate CM for all pixels in region.
        all_save_path = os.path.join(base_change_mats_dir, f'{region_name}_all.png')

        ## Get average begining and end material prediction masks.
        beg_mat_pred = region_beg_mat_pred[region_name]
        end_mat_pred = region_end_mat_pred[region_name]

        ## Create confusion matrix for this region
        region_cm = ConfusionMatrix(actual_vector=list(beg_mat_pred.flatten()),
                                    predict_vector=list(end_mat_pred.flatten()),
                                    classes=list(MATERIAL_TO_MATID.values()))

        cm_array = region_cm.to_array()
        df_cm = pd.DataFrame(cm_array, MATERIAL_TO_MATID.keys(), MATERIAL_TO_MATID.keys())

        ## Crop unused classes.
        drop_df_cm = df_cm.drop(['unknown', 'metal', 'concrete'], axis=0)
        drop_df_cm = drop_df_cm.drop(['unknown', 'metal', 'concrete'], axis=1)

        ## Reorder columns and rows.
        ord_df_cm = drop_df_cm[['water', 'soil', 'vegetation', 'snow', 'polymer', 'asphalt']]

        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        ax = sn.heatmap(ord_df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues')
        ax.set(xlabel='End Material', ylabel='Beginning Material')
        plt.savefig(all_save_path)
        plt.close()

    # Create visualizations.
    for region_name, mat_trans_mask in mat_trans_masks.items():
        bas_mask = region_bas_annos[region_name]
        region_metric = region_metrics[region_name]
        beg_mat_pred = region_beg_mat_pred[region_name]
        end_mat_pred = region_end_mat_pred[region_name]
        create_mat_change_visualizations(mat_trans_mask, bas_mask, region_metric, region_name,
                                         beg_mat_pred, end_mat_pred, exp_dir)

    # Save metrics.
    metrics_path = os.path.join(base_save_dir, 'metrics.json')
    json.dump(region_metrics, open(metrics_path, 'w'), indent=2)


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


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Path to the model weights.')
    parser.add_argument(
        '--mat_pred',
        default=False,
        action='store_true',
        help='If flag is true, then the model will generate material prediction images.')
    parser.add_argument(
        '--mat_change',
        default=False,
        action='store_true',
        help='If flag is true, then the model will generate a MTM and get scored on \
              overlap to BAS labels.')
    parser.add_argument('--fist_last_images', type=int, default=5)
    parser.add_argument('--mat_trans_mask_mode', type=str, default='hard_class_2')
    parser.add_argument(
        '--select_regions',
        nargs='+',
        default=['KR_R001', 'KR_R002', 'AE_R001', 'BR_R002'],
        help='Which regions to run predict script on. Note: Only works for "mat_change" flag.')
    parser.add_argument('--n_workers',
                        default=None,
                        type=int,
                        help='Number of cpu cores to run prediction with.')
    args = parser.parse_args()

    if (args.mat_pred is False) and (args.mat_change is False):
        raise ValueError('No prediction action was given to script to run.')

    # Get experiment directory.
    exp_dir = '/'.join(args.checkpoint_path.split('/')[:-2])

    # Load configuration file.
    cfg_path = os.path.join(exp_dir, '.hydra', 'config.yaml')
    cfg = load_cfg_file(cfg_path)

    # Overwrite configuration file.
    if args.n_workers is not None:
        cfg.n_workers = args.n_workers

    # Load dataset.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width, cfg.crop_stride)

    ## Get evaluation dataset and loader.
    mat_labels, mat_dist = get_labelbox_material_labels(False, cfg.lb_project_id)

    if cfg.dataset.kwargs is None:
        cfg.dataset.kwargs = {}
    cfg.dataset.kwargs['sample_positives'] = False

    if hasattr(cfg, 'resize_factor') is False:
        cfg.resize_factor = 1

    if args.mat_pred:
        eval_dataset = build_dataset(cfg.dataset.name,
                                     mat_labels,
                                     slice_params,
                                     'all',
                                     sensors=cfg.dataset.sensors,
                                     channels=cfg.dataset.channels,
                                     resize_factor=cfg.resize_factor,
                                     refresh_labels=cfg.refresh_labels,
                                     **cfg.dataset.kwargs)

        eval_loader = DataLoader(eval_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=False,
                                 num_workers=cfg.n_workers)

        n_channels = eval_dataset.n_channels

    if args.mat_change:
        # Load bas dataset.
        bas_dataset = BAS_Dataset(channels=cfg.dataset.channels,
                                  slice_params=slice_params,
                                  first_last_samples=args.fist_last_images,
                                  select_regions=args.select_regions,
                                  dset_name='drop4')
        bas_loader = DataLoader(bas_dataset,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.n_workers)
        n_channels = bas_dataset.n_channels

    # Create model.
    if cfg.model.kwargs is None:
        cfg.model.kwargs = {}
    model = build_model(
        mat_dist,
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
        # log_image_iter=cfg.log_image_iter,
        # to_rgb_fcn=eval_dataset.to_RGB,
        to_rgb_fcn=None,
        **cfg.model.kwargs,
        checkpoint_path=args.checkpoint_path)
    model = model.eval()

    # Generate model predictions.
    if args.mat_pred:
        generate_material_predictions(model,
                                      eval_loader,
                                      exp_dir,
                                      mat_labels,
                                      resize_factor=cfg.resize_factor)

        # generate_material_predictions_v2(model,
        #                                  eval_loader,
        #                                  exp_dir,
        #                                  mat_labels,
        #                                  resize_factor=cfg.resize_factor)

    if args.mat_change:
        # Compute BAS performance based on materials
        compute_change_detection_perf(model,
                                      bas_loader,
                                      exp_dir,
                                      mat_trans_mask_mode=args.mat_trans_mask_mode,
                                      resize_factor=cfg.resize_factor)


if __name__ == '__main__':
    predict()
