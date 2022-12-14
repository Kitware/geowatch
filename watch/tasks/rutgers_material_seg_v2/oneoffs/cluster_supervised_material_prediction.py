import os
import argparse

import kwcoco
import numpy as np
from tqdm import tqdm
from einops import rearrange
from tifffile import tifffile

from ssl_residual.utils.util_image import load_norm_image, save_geotiff
from ssl_residual.utils.util_misc import MATID_TO_MATERIAL, colorize_material_mask
from ssl_residual.utils.util_dataset import filter_image_ids_by_sensor, compute_clusters


def generate_pixel_predictions(args, kwcoco_path):
    n_materials = len(MATID_TO_MATERIAL.keys())

    # Load kwcoco file with all material labels.
    coco_dset = kwcoco.CocoDataset(kwcoco_path)

    # Get all pixels with material labels.
    image_ids = filter_image_ids_by_sensor(coco_dset, sensors=args.sensors.split('|'))

    pixel_data, mat_ids = [], []
    for image_id in tqdm(image_ids, colour='green', desc='Collecting pixels with material labels'):
        aux_data = coco_dset.imgs[image_id]['auxiliary']

        # Check if image has a material mask.
        if aux_data[-1]['channels'] == 'mat_mask':
            # Load image.
            image = load_norm_image(coco_dset, image_id, channels=args.channels)

            # Load image and get pixels with material labels.
            file_path = aux_data[-1]['file_name']
            mat_label_mask = tifffile.imread(file_path)

            if (image.shape[1] != mat_label_mask.shape[0]) or (image.shape[2] != mat_label_mask.shape[1]):
                mat_label_mask = mat_label_mask[:image.shape[1], :image.shape[2]]

            # Find pixels where there are labels.
            X, Y = np.where(mat_label_mask != 0)

            pixel_data.append(image[:, X, Y])
            mat_ids.append(mat_label_mask[X, Y])

    all_pixels = np.concatenate(pixel_data, axis=1)
    pixel_mat_labels = np.concatenate(mat_ids, axis=0)

    # Cluster subsampled pixels.

    ## Subsample pixels for clustering.
    n_pixels = all_pixels.shape[1]
    pct = args.subsample_pct
    n_samples = int(n_pixels * pct)

    indices = np.random.choice(list(range(n_pixels)), size=n_samples, replace=False)
    sub_pixels = np.take(all_pixels, indices, axis=1)
    # sub_mat_labels = np.take(pixel_mat_labels, indices, axis=0)

    cluster_centers = compute_clusters(sub_pixels, args.n_clusters, seed_num=args.seed_num)

    # Assign material label to each cluster.
    ## Get cluster labels for all pixels.
    pixel_cids = ((all_pixels[:, None, :] - cluster_centers[:, :, None])**2).sum(axis=0).argmin(axis=0)

    ## Use pixel material labels to assign material labels to clusters.
    cluster_id_2_mat_id = {}
    for c in range(args.n_clusters):
        indices = np.where(pixel_cids == c)
        c_mat_labels = pixel_mat_labels[indices]
        c_mat_counts, _ = np.histogram(c_mat_labels, bins=8, range=(0, 8))
        m_id = c_mat_counts.argmax()
        cluster_id_2_mat_id[c] = m_id

    # Load images from kwcoco file and predict materials based on nearest cluster.

    ## Create a copy of kwcoco dataset.
    pred_coco_dset = coco_dset.copy()

    # Filter regions to compute material predictions for.
    region_names = args.regions_to_generate.split('|')

    select_region_image_ids = []
    for vid in list(pred_coco_dset.index.videos.keys()):
        region_name = pred_coco_dset.index.videos[vid]['name']
        if region_name in region_names:
            select_region_image_ids.extend(list(pred_coco_dset.index.vidid_to_gids[vid]))

    for image_id in tqdm(select_region_image_ids, colour='green', desc='Predicting materials from centroids'):
        # Load image.
        image = load_norm_image(pred_coco_dset, image_id, channels=args.channels)
        _, h, w = image.shape

        # Compute material predictions from cluster centers.
        ## Resize image.
        pixels = rearrange(image, 'c h w -> c (h w)')
        pixel_cids = ((pixels[:, None, :] - cluster_centers[:, :, None])**2).sum(axis=0)

        # Convert distance to material prediction probability.
        inv_dist = 1 / pixel_cids
        cluster_ps = (inv_dist**2 / (inv_dist**2).sum(axis=0))  # [n_clusters, n_pixels]

        # Update cluter probabilites to material probabilies.
        mat_ps = np.zeros([n_materials, cluster_ps.shape[-1]], dtype=float)
        for c_id, mat_id in cluster_id_2_mat_id.items():
            cluster_conf = cluster_ps[c_id, :]
            mat_ps[mat_id, :] += cluster_conf

        ## Resize material probabilities to image space from pixel.
        mat_ps = rearrange(mat_ps, 'c (h w) -> c h w', h=h, w=w)

        # Add material confidence predictions to kwcoco file.
        ## Find a save path for material predictions.
        base_dir = os.path.join('/'.join(kwcoco_path.split('/')[:-1]), '_predictions')
        save_dir = os.path.join(base_dir, f'cluster_priors_{args.feature_type}_{args.n_clusters}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, pred_coco_dset.index.imgs[image_id]['name'] + '.tif')

        ## Save image to disc.
        save_geotiff(mat_ps, save_path, save_backend='gdal')

        ## Add relation to kwcoco file.
        pred_coco_dset.add_auxiliary_item(image_id,
                                          save_path,
                                          channels='mat_pred',
                                          width=w,
                                          height=h,
                                          warp_aux_to_img={'scale': 1.0})

    # Write kwcoco file.
    print('Dumping kwcoco file ... ')
    dset_root = '/'.join(kwcoco_path.split('/')[:-1])
    pred_kwcoco_save_path = os.path.join(dset_root,
                                         f'cluster_priors_{args.feature_type}_{args.n_clusters}' + '.kwcoco.json')
    pred_coco_dset.dump(pred_kwcoco_save_path)

    return pred_kwcoco_save_path


def generate_residual_predictions(args, kwcoco_path):
    n_materials = len(MATID_TO_MATERIAL.keys())

    # Load kwcoco file with all material labels.
    coco_dset = kwcoco.CocoDataset(kwcoco_path)

    # Get all pixels with material labels.
    image_ids = filter_image_ids_by_sensor(coco_dset, sensors=args.sensors.split('|'))

    pixel_data, mat_ids = [], []
    for image_id in tqdm(image_ids, colour='green', desc='Collecting pixels with material labels'):
        aux_data = coco_dset.imgs[image_id]['auxiliary']

        # Check if image has a material mask.
        if aux_data[-1]['channels'] == 'mat_mask':
            # Load image.
            image = load_norm_image(coco_dset, image_id, channels=args.channels)

            # Load image and get pixels with material labels.
            file_path = aux_data[-1]['file_name']
            mat_label_mask = tifffile.imread(file_path)

            if (image.shape[1] != mat_label_mask.shape[0]) or (image.shape[2] != mat_label_mask.shape[1]):
                mat_label_mask = mat_label_mask[:image.shape[1], :image.shape[2]]

            # Find pixels where there are labels.
            X, Y = np.where(mat_label_mask != 0)

            pixel_data.append(image[:, X, Y])
            mat_ids.append(mat_label_mask[X, Y])

    all_pixels = np.concatenate(pixel_data, axis=1)
    pixel_mat_labels = np.concatenate(mat_ids, axis=0)

    # Cluster subsampled pixels.

    ## Subsample pixels for clustering.
    n_pixels = all_pixels.shape[1]
    pct = args.subsample_pct
    n_samples = int(n_pixels * pct)

    indices = np.random.choice(list(range(n_pixels)), size=n_samples, replace=False)
    sub_pixels = np.take(all_pixels, indices, axis=1)

    cluster_centers = compute_clusters(sub_pixels, args.n_clusters, seed_num=args.seed_num)

    # Compute residuals.
    residuals = []
    for i in range(args.n_clusters):
        residual = ((all_pixels - cluster_centers[:, i, None])**2).sum(axis=0)
        residuals.append(residual)
    residuals = np.asarray(residuals)  # [n_clusters, n_pixels]

    ## Subsample residuals
    residual_indices = np.random.choice(list(range(n_pixels)), size=n_samples, replace=False)
    sub_residuals = np.take(residuals, residual_indices, axis=1)
    sub_mat_labels = np.take(pixel_mat_labels, residual_indices, axis=0)

    ## Compute centroids.
    residual_cluster_centers = compute_clusters(sub_residuals, args.n_clusters, seed_num=args.seed_num)

    # Assign material label to each residual cluster.
    ## Get cluster labels for all pixels.
    sub_residual_cids = ((sub_residuals[:, None, :] -
                          residual_cluster_centers[:, :, None])**2).sum(axis=0).argmin(axis=0)

    ## Use pixel material labels to assign material labels to clusters.
    r_cluster_id_2_mat_id = {}
    for c in range(args.n_clusters):
        indices = np.where(sub_residual_cids == c)
        c_mat_labels = sub_mat_labels[indices]
        c_mat_counts, _ = np.histogram(c_mat_labels, bins=8, range=(0, 8))
        m_id = c_mat_counts.argmax()
        r_cluster_id_2_mat_id[c] = m_id

    # Load images from kwcoco file and predict materials based on nearest cluster.

    ## Create a copy of kwcoco dataset.
    pred_coco_dset = coco_dset.copy()

    # Filter regions to compute material predictions for.
    region_names = args.regions_to_generate.split('|')

    select_region_image_ids = []
    for vid in list(pred_coco_dset.index.videos.keys()):
        region_name = pred_coco_dset.index.videos[vid]['name']
        if region_name in region_names:
            select_region_image_ids.extend(list(pred_coco_dset.index.vidid_to_gids[vid]))

    for image_id in tqdm(select_region_image_ids, colour='green', desc='Predicting materials from residual centroids'):
        # Load image.
        image = load_norm_image(pred_coco_dset, image_id, channels=args.channels)
        _, h, w = image.shape

        # Compute material predictions from residual cluster centers.
        ## Resize image.
        pixels = rearrange(image, 'c h w -> c (h w)')

        ## Compute pixel residuals.
        residuals = []
        for i in range(args.n_clusters):
            residual = ((pixels - cluster_centers[:, i, None])**2).sum(axis=0)
            residuals.append(residual)
        residuals = np.asarray(residuals)  # [n_clusters, n_pixels]

        ## Assign to residual cluster.
        residual_cids = ((residuals[:, None, :] - residual_cluster_centers[:, :, None])**2).sum(axis=0)

        # Convert distance to material prediction probability.
        inv_dist = 1 / residual_cids
        cluster_ps = (inv_dist**2 / (inv_dist**2).sum(axis=0))  # [n_clusters, n_pixels]

        # Update cluter probabilites to material probabilies.
        mat_ps = np.zeros([n_materials, cluster_ps.shape[-1]], dtype=float)
        for c_id, mat_id in r_cluster_id_2_mat_id.items():
            cluster_conf = cluster_ps[c_id, :]
            mat_ps[mat_id, :] += cluster_conf

        ## Resize material probabilities to image space from pixel.
        mat_ps = rearrange(mat_ps, 'c (h w) -> c h w', h=h, w=w)

        # Add material confidence predictions to kwcoco file.
        ## Find a save path for material predictions.
        base_dir = os.path.join('/'.join(kwcoco_path.split('/')[:-1]), '_predictions')
        save_dir = os.path.join(base_dir, f'cluster_priors_{args.feature_type}_{args.n_clusters}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, pred_coco_dset.index.imgs[image_id]['name'] + '.tif')

        ## Save image to disc.
        save_geotiff(mat_ps, save_path, save_backend='gdal')

        ## Add relation to kwcoco file.
        pred_coco_dset.add_auxiliary_item(image_id,
                                          save_path,
                                          channels='mat_pred',
                                          width=w,
                                          height=h,
                                          warp_aux_to_img={'scale': 1.0})

    # Write kwcoco file.
    print('Dumping kwcoco file ... ')
    dset_root = '/'.join(kwcoco_path.split('/')[:-1])
    pred_kwcoco_save_path = os.path.join(dset_root,
                                         f'cluster_priors_{args.feature_type}_{args.n_clusters}' + '.kwcoco.json')
    pred_coco_dset.dump(pred_kwcoco_save_path)

    return pred_kwcoco_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_num', type=int, default=0)
    parser.add_argument('--subsample_pct', type=float, default=0.01)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--sensors', type=str, default='S2')
    parser.add_argument('--channels', type=str, default='red|green|blue|nir|swir16|swir22')
    parser.add_argument('--feature_type', type=str, default='pixel')
    parser.add_argument('--regions_to_generate', type=str, default='KR_R002|US_R001')
    parser.add_argument(
        '--kwcoco_path',
        type=str,
        default='/data4/datasets/smart_watch_dvc/Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC/data_materials.kwcoco.json')
    args = parser.parse_args()

    if args.feature_type == 'pixel':
        save_kwcoco_path = generate_pixel_predictions(args, args.kwcoco_path)
    elif args.feature_type == 'residual':
        save_kwcoco_path = generate_residual_predictions(args, args.kwcoco_path)
    else:
        raise NotImplementedError

    print(f'Initial predictions saved to: {save_kwcoco_path}')
