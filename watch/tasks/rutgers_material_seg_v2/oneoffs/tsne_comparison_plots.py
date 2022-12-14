import argparse

import cv2
import torch
import kwcoco
import numpy as np
from tqdm import tqdm
from tifffile import tifffile
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import segmentation_models_pytorch as smp

from ssl_residual.utils.util_image import load_norm_image
from ssl_residual.utils.util_dataset import filter_image_ids_by_sensor, compute_clusters
from ssl_residual.utils.util_misc import MATID_TO_MATERIAL, MATID_TO_MATERIAL, MATERIAL_TO_COLOR


def get_model(network_name, pretrained_weights):
    if network_name == 'resnet18':
        if pretrained_weights is None:
            model = smp.encoders.get_encoder(name='resnet18', weights=None)
        elif pretrained_weights == 'imagenet':
            model = smp.encoders.get_encoder(name='resnet18', weights='imagenet')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model


def get_material_pixels(coco_dset, channels='red|green|blue|nir|swir16|swir22'):
    # Get all pixels with material labels.
    image_ids = filter_image_ids_by_sensor(coco_dset, sensors='S2')

    pixel_data, mat_ids = [], []
    for image_id in tqdm(image_ids, colour='green', desc='Collecting pixels with material labels'):
        aux_data = coco_dset.imgs[image_id]['auxiliary']

        # Check if image has a material mask.
        if aux_data[-1]['channels'] == 'mat_mask':
            # Load image.
            image = load_norm_image(coco_dset, image_id, channels=channels)

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
    return all_pixels, pixel_mat_labels


def get_material_features(coco_dset, model, channels='red|green|blue'):
    # Get all pixels with material labels.
    image_ids = filter_image_ids_by_sensor(coco_dset, sensors='S2')

    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    feature_data, mat_ids = [], []
    for image_id in tqdm(image_ids, colour='green', desc='Collecting pixels with material labels'):
        aux_data = coco_dset.imgs[image_id]['auxiliary']

        # Check if image has a material mask.
        if aux_data[-1]['channels'] == 'mat_mask':
            # Load image.
            image = load_norm_image(coco_dset, image_id, channels=channels)
            _, h, w = image.shape

            image = image**0.4

            # Compute features.
            with torch.no_grad():
                image_features = model(torch.tensor(image[None]).float().to(device))[1]

            ## Resize and convert features to numpy.
            image_features = image_features.detach().cpu().numpy()
            image_features = image_features[0]

            frames = []
            for i in range(image_features.shape[0]):
                frames.append(cv2.resize(image_features[i], dsize=(w, h), interpolation=cv2.INTER_LINEAR))

            image_features = np.stack(frames, axis=0)

            # Load image and get pixels with material labels.
            file_path = aux_data[-1]['file_name']
            mat_label_mask = tifffile.imread(file_path)

            if (h != mat_label_mask.shape[0]) or (w != mat_label_mask.shape[1]):
                mat_label_mask = mat_label_mask[:h, :w]

            # Find pixels where there are labels.
            X, Y = np.where(mat_label_mask != 0)

            feature_data.append(image_features[:, X, Y])
            mat_ids.append(mat_label_mask[X, Y])

    features = np.concatenate(feature_data, axis=1)
    pixel_mat_labels = np.concatenate(mat_ids, axis=0)
    return features, pixel_mat_labels


def create_pixel_tsne_plot(kwcoco_path, subsample_pct, tsne_perplexity=25, seed_num=0):
    # Get number of materials.
    n_materials = len(MATID_TO_MATERIAL.keys())

    # Load kwcoco file.
    coco_dset = kwcoco.CocoDataset(kwcoco_path)

    # Get pixels with material labels
    pixels, labels = get_material_pixels(coco_dset)

    # Subsample pixels and material labels.
    n_pixels = pixels.shape[1]
    n_samples = int(n_pixels * subsample_pct)

    indices = np.random.choice(list(range(n_pixels)), size=n_samples, replace=False)
    sub_pixels = np.take(pixels, indices, axis=1)
    sub_labels = np.take(labels, indices, axis=0)

    # Compute t-SNE of subsampled pixels.
    embed = TSNE(n_components=2, perplexity=tsne_perplexity, verbose=True, random_state=seed_num,
                 n_jobs=-1).fit_transform(sub_pixels.T)

    # Create plot of t-SNE points colored by material labels.
    _, ax = plt.subplots(1, 1)
    for mat_value in range(1, n_materials):
        indices = np.where(mat_value == sub_labels)
        mat_name = MATID_TO_MATERIAL[mat_value]
        mat_color = np.array(MATERIAL_TO_COLOR[mat_name]) / 255.0
        ax.scatter(embed[indices, 0], embed[indices, 1], color=mat_color)

    # Save plot.
    save_path = 'tsne_pixel.png'
    plt.savefig(save_path, dpi=300)


def create_pixel_residual_tsne_plots(kwcoco_path, n_clusters, subsample_pct, tsne_perplexity=25, seed_num=0):
    # Get number of materials.
    n_materials = len(MATID_TO_MATERIAL.keys())

    # Load kwcoco file.
    coco_dset = kwcoco.CocoDataset(kwcoco_path)

    # Get pixels with material labels
    pixels, labels = get_material_pixels(coco_dset)

    # Subsample pixels to compute clusters.
    n_pixels = pixels.shape[1]
    n_samples = int(n_pixels * subsample_pct)

    indices = np.random.choice(list(range(n_pixels)), size=n_samples, replace=False)
    sub_pixels = np.take(pixels, indices, axis=1)
    sub_labels = np.take(labels, indices, axis=0)

    # Compute cluster centers.
    cluster_centers = compute_clusters(sub_pixels, n_clusters, seed_num=seed_num)

    # Compute residuals.
    residuals = []
    for i in range(n_clusters):
        residual = ((sub_pixels - cluster_centers[:, i, None])**2).sum(axis=0)
        residuals.append(residual)
    residuals = np.asarray(residuals)  # [n_clusters, n_pixels]

    # Compute t-SNE of subsampled residuals.
    embed = TSNE(n_components=2, perplexity=tsne_perplexity, verbose=True, random_state=seed_num,
                 n_jobs=-1).fit_transform(residuals.T)

    # Create plot of t-SNE points colored by material labels.
    _, ax = plt.subplots(1, 1)
    for mat_value in range(1, n_materials):
        indices = np.where(mat_value == sub_labels)
        mat_name = MATID_TO_MATERIAL[mat_value]
        mat_color = np.array(MATERIAL_TO_COLOR[mat_name]) / 255.0
        ax.scatter(embed[indices, 0], embed[indices, 1], color=mat_color)

    # Save plot.
    save_path = 'tsne_pixel_residual.png'
    plt.savefig(save_path, dpi=300)


def create_feature_tsne_plot(kwcoco_path, model, subsample_pct, tsne_perplexity=25, seed_num=0):
    # Get number of materials.
    n_materials = len(MATID_TO_MATERIAL.keys())

    # Load kwcoco file.
    coco_dset = kwcoco.CocoDataset(kwcoco_path)

    # Get features with material labels.
    features, labels = get_material_features(coco_dset, model)

    # Subsample features and material labels.
    n_features = features.shape[1]
    n_samples = int(n_features * subsample_pct)

    indices = np.random.choice(list(range(n_features)), size=n_samples, replace=False)
    sub_features = np.take(features, indices, axis=1)
    sub_labels = np.take(labels, indices, axis=0)

    # Compute t-SNE of subsampled features.
    embed = TSNE(n_components=2, perplexity=tsne_perplexity, verbose=True, random_state=seed_num,
                 n_jobs=-1).fit_transform(sub_features.T)

    # Create plot of t-SNE points colored by material labels.
    _, ax = plt.subplots(1, 1)
    for mat_value in range(1, n_materials):
        indices = np.where(mat_value == sub_labels)
        mat_name = MATID_TO_MATERIAL[mat_value]
        mat_color = np.array(MATERIAL_TO_COLOR[mat_name]) / 255.0
        ax.scatter(embed[indices, 0], embed[indices, 1], color=mat_color)

    # Save plot.
    save_path = 'tsne_feature.png'
    plt.savefig(save_path, dpi=300)


def create_feature_residual_tsne_plots(kwcoco_path, model, n_clusters, subsample_pct, tsne_perplexity=25, seed_num=0):
    # Get number of materials.
    n_materials = len(MATID_TO_MATERIAL.keys())

    # Load kwcoco file.
    coco_dset = kwcoco.CocoDataset(kwcoco_path)

    # Get features with material labels
    features, labels = get_material_features(coco_dset, model)

    # Subsample features to compute clusters.
    n_features = features.shape[1]
    n_samples = int(n_features * subsample_pct)

    indices = np.random.choice(list(range(n_features)), size=n_samples, replace=False)
    sub_features = np.take(features, indices, axis=1)
    sub_labels = np.take(labels, indices, axis=0)

    # Compute cluster centers.
    cluster_centers = compute_clusters(sub_features, n_clusters, seed_num=seed_num)

    # Compute residuals.
    residuals = []
    for i in range(n_clusters):
        residual = ((sub_features - cluster_centers[:, i, None])**2).sum(axis=0)
        residuals.append(residual)
    residuals = np.asarray(residuals)  # [n_clusters, n_features]

    # Compute t-SNE of subsampled residuals.
    embed = TSNE(n_components=2, perplexity=tsne_perplexity, verbose=True, random_state=seed_num,
                 n_jobs=-1).fit_transform(residuals.T)

    # Create plot of t-SNE points colored by material labels.
    _, ax = plt.subplots(1, 1)
    for mat_value in range(1, n_materials):
        indices = np.where(mat_value == sub_labels)
        mat_name = MATID_TO_MATERIAL[mat_value]
        mat_color = np.array(MATERIAL_TO_COLOR[mat_name]) / 255.0
        ax.scatter(embed[indices, 0], embed[indices, 1], color=mat_color)

    # Save plot.
    save_path = 'tsne_feature_residual.png'
    plt.savefig(save_path, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type',
                        type=str,
                        default='pixel',
                        choices=['pixel', 'pixel_residual', 'feature', 'feature_residual'])
    parser.add_argument('--perplexity', type=int, default=25)
    parser.add_argument('--subsample_pct', type=float, default=0.001)
    parser.add_argument(
        '--kwcoco_path',
        type=str,
        default='/data4/datasets/smart_watch_dvc/Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC/data_materials.kwcoco.json')
    parser.add_argument('--seed_num', type=int, default=0)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--network', type=str, default='resnet18')
    parser.add_argument('--network_pretrained_weights', type=str, default=None)
    args = parser.parse_args()

    if args.feature_type == 'pixel':
        create_pixel_tsne_plot(args.kwcoco_path, args.subsample_pct, tsne_perplexity=args.perplexity)
    elif args.feature_type == 'pixel_residual':
        create_pixel_residual_tsne_plots(args.kwcoco_path,
                                         args.n_clusters,
                                         args.subsample_pct,
                                         tsne_perplexity=args.perplexity)
    elif args.feature_type == 'feature':
        model = get_model(args.network, args.network_pretrained_weights)
        create_feature_tsne_plot(args.kwcoco_path, model, args.subsample_pct, tsne_perplexity=args.perplexity)
    elif args.feature_type == 'feature_residual':
        model = get_model(args.network, args.network_pretrained_weights)
        create_feature_residual_tsne_plots(args.kwcoco_path,
                                           model,
                                           args.n_clusters,
                                           args.subsample_pct,
                                           tsne_perplexity=args.perplexity)
    else:
        raise NotImplementedError
    pass