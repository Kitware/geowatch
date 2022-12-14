import numpy as np

from watch.tasks.rutgers_material_seg_v2.utils.util_misc import get_crop_slices


def compute_residual_feature(image,
                             cluster_centers,
                             img_norm_mode=None,
                             local_context_mode='square',
                             square_context_size=5,
                             combine_context_mode='before',
                             difference_norm='L2',
                             to_rgb_func=None,
                             eta=1e-5):
    """TODO: _summary_

    Args:
        image (np.array | torch.tensor): A float array of shape [feature_dim, height, width].
        cluster_centers (np.array | torch.tensor): A float array of shape [feature_dim, n_clusters].
        img_norm_mode (_type_, optional): TODO: _description_. Defaults to None.
        local_context_mode (str, optional): TODO: _description_. Defaults to 'square'.
        square_context_size (int, optional): TODO: _description_. Defaults to 5.
        combine_context_mode (str, optional): TODO: _description_. Defaults to 'before'.
        to_rgb_func (_type_, optional): TODO: _description_. Defaults to None.

    Raises:
        NotImplementedError: TODO: _description_
        NotImplementedError: TODO: _description_

    Returns:
        (np.array | torch.tensor): A float array containing computed features of shape [n_clusters, height, width].
    """
    assert combine_context_mode in ['before',
                                    'after'], f'Invalid value "{combine_context_mode}" for combine_context_mode'

    # Normalize input feature and cluster values.
    if img_norm_mode == 'unit':
        # Normalize image to unit values along channel dimension.
        image = image / np.linalg.norm(image, axis=0)[None]
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers + eta, axis=0)[None]
    elif img_norm_mode is None:
        pass
    elif img_norm_mode == 'SSR':
        # Method proposed in the VLAD paper.
        image = np.sign(image) * np.sqrt(np.abs(image))
        cluster_centers = np.sign(cluster_centers) * np.sqrt(np.abs(cluster_centers))
    else:
        raise NotImplementedError

    # Divide image into subcomponents to create features over.
    if local_context_mode == 'square':
        # Get context windows to compute features from.
        _, img_h, img_w = image.shape
        crop_slices = get_crop_slices(img_h, img_w, square_context_size, square_context_size, step=1)

        # Get pixel values within each crop.
        context_pixel_indices = []
        for crop_slice in crop_slices:
            h0, w0, h, w = crop_slice
            x_range, y_range = list(range(h0, h + h0)), list(range(w0, w + w0))

            X, Y = [], []
            for x in x_range:
                for y in y_range:
                    X.append(x)
                    Y.append(y)
            context_pixel_indices.append([X, Y])
    elif local_context_mode == 'super_pixel':
        # TODO: Setup super pixel computation.

        # Need to convert image to RGB to get good super_pixel performance.
        pass
    elif local_context_mode == 'precomputed|square':
        # This step has already been computed before so skip.
        _, img_h, img_w = image.shape
        _, X, Y = np.where(image > 0)
        context_pixel_indices = [[X, Y]]
    else:
        raise NotImplementedError

    # Compute feature from context regions.
    n_clusters = cluster_centers.shape[1]
    feature_canvas = np.zeros([n_clusters, img_h, img_w], dtype=image.dtype)
    for context_indices in context_pixel_indices:
        X, Y = context_indices

        # Get pixels.
        pixels = image[:, X, Y]

        # Combine pixel values before computing residual.
        if combine_context_mode == 'before':
            # pixels: [feature_dim, n_pixels]
            pixels = np.mean(pixels, axis=1)[:, None]  # [feature_dim, 1]

        # Compute residual between pixels and cluster centers.
        residuals = []
        for i in range(n_clusters):
            diff = pixels - cluster_centers[:, i][:, None]  # [feature_dim, n_pixels]

            if difference_norm == 'L2':
                norm_diff = diff**2
            elif difference_norm == 'L1':
                norm_diff = np.abs(diff)
            elif difference_norm == 'SSR':
                norm_diff = np.sign(diff) * np.sqrt(np.abs(diff))
            elif difference_norm is None:
                pass
            else:
                raise NotImplementedError

            # Sum feature dim of residual.
            norm_diff = norm_diff.sum(axis=0)

            residuals.append(norm_diff)

        residuals = np.asarray(residuals)  # [n_clusters, n_pixels]

        # Combine residuals.
        if combine_context_mode == 'after':
            residuals = residuals.mean(axis=1)[:, None]

        residuals = residuals[:, 0]  # Remove extra dimension.

        # Paste feature into canvas.
        if local_context_mode == 'square':
            feature_canvas[:, X[0], Y[0]] = residuals
        elif local_context_mode == 'super_pixel':
            breakpoint()
            pass
            feature_canvas[:, X, Y] = residuals
        elif local_context_mode == 'precomputed|square':
            feature_canvas = residuals
        else:
            raise NotImplementedError

    return feature_canvas


if __name__ == '__main__':
    # TEST: Residual feature computation.
    img_h, img_w = 50, 50
    n_clusters = 10
    feature_size = 3
    image = np.zeros([feature_size, img_h, img_w])
    cluster_centers = np.zeros([feature_size, n_clusters])

    residual_feature = compute_residual_feature(image, cluster_centers, img_norm_mode='SSR')
