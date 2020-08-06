import numpy as np
from helpers.visual_tools import visualization_wrapper
from helpers.utils import default_logger, timer


def iou(relative_sizes, centroids, k):
    """
    Calculate intersection over union for relative box sizes.
    Args:
        relative_sizes: 2D array of relative box sizes.
        centroids: 2D array of shape(k, 2)
        k: int, number of clusters.

    Returns:
        IOU array.
    """
    n = relative_sizes.shape[0]
    box_area = relative_sizes[:, 0] * relative_sizes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))
    cluster_area = centroids[:, 0] * centroids[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))
    box_w_matrix = np.reshape(relative_sizes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(centroids[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)
    box_h_matrix = np.reshape(relative_sizes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(centroids[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)
    result = inter_area / (box_area + cluster_area - inter_area)
    return result


@timer(default_logger)
@visualization_wrapper
def k_means(relative_sizes, k, distance_func=np.median, frame=None):
    """
    Calculate optimal anchor relative sizes.
    Args:
        relative_sizes: 2D array of relative box sizes.
        k: int, number of clusters.
        distance_func: function to calculate distance.
        frame: pandas DataFrame with the annotation data(for visualization purposes).

    Returns:
        Optimal relative sizes.
    """
    box_number = relative_sizes.shape[0]
    last_nearest = np.zeros((box_number,))
    centroids = relative_sizes[np.random.randint(0, box_number, k)]
    old_distances = np.zeros((relative_sizes.shape[0], k))
    iteration = 0
    while True:
        distances = 1 - iou(relative_sizes, centroids, k)
        print(
            f'Iteration: {iteration} Loss: '
            f'{np.sum(np.abs(distances - old_distances))}'
        )
        old_distances = distances.copy()
        iteration += 1
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            default_logger.info(
                f'Generated {len(centroids)} anchors in '
                f'{iteration} iterations'
            )
            return centroids, frame
        for anchor in range(k):
            centroids[anchor] = distance_func(
                relative_sizes[current_nearest == anchor], axis=0
            )
        last_nearest = current_nearest


def generate_anchors(width, height, centroids):
    """
    Generate anchors for image of size(width, height)
    Args:
        width: Width of image.
        height: Height of image.
        centroids: Output of k-means.

    Returns:
        2D array of resulting anchors.
    """
    return (centroids * np.array([width, height])).astype(int)
