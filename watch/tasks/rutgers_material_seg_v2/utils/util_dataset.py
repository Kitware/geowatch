from typing import List
from sklearn.cluster import KMeans


def filter_image_ids_by_sensor(kwcoco_file, sensors: List):
    # Get all image IDs.
    image_ids = list(kwcoco_file.imgs.keys())

    # Filter image IDs that belong to active sensors.
    sensor_image_ids = []
    for image_id in image_ids:
        img_sensor_type = kwcoco_file.index.imgs[image_id]['sensor_coarse']
        if img_sensor_type in sensors:
            sensor_image_ids.append(image_id)

    if len(sensor_image_ids) == 0:
        raise ValueError(f'No image IDs found for sensors {sensors}')

    return sensor_image_ids


def compute_clusters(pixels, n_clusters, seed_num=None):
    """Compute the centroid clusters of the K-Means algorithm.

        Args:
            pixels (np.array): A float array of shape [feature_dim, n_features]
            n_clusters (int): The number of cluster centroids to fit to data.

        Returns:
            np.array: A float64 numpy array of shape [feature_dim, n_clusters].
        """
    print(f'Computing K-Means of {n_clusters} clusters over {pixels.shape[1]} samples ...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed_num)
    kmeans.fit(pixels.T)
    return kmeans.cluster_centers_.T
