from random import sample

import numpy as np
import pandas as pd
from absl import logging
from pandas import read_csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()  # for plot styling


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


original_anchors = (
    np.array(
        [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61),
            (62, 45),
            (59, 119),
            (116, 90),
            (156, 198),
            (373, 326),
        ],
        np.float32,
    )
    / 416
)


# def k_means(relative_sizes, k, distance_func=np.median, frame=None):
#     """
#     Calculate optimal anchor relative sizes.
#     Args:
#         relative_sizes: 2D array of relative box sizes.
#         k: int, number of clusters.
#         distance_func: function to calculate distance.
#         frame: pandas DataFrame with the annotation data(for visualization purposes).
#     Returns:
#         Optimal relative sizes.
#     """
#     box_number = relative_sizes.shape[0]
#     last_nearest = np.zeros((box_number,))
#     centroids = relative_sizes[np.random.randint(0, box_number, k)]
#     old_distances = np.zeros((relative_sizes.shape[0], k))
#     iteration = 0
#     while True:
#         distances = 1 - iou(relative_sizes, centroids, k)
#         print(
#             f'Iteration: {iteration} Loss: '
#             f'{np.sum(np.abs(distances - old_distances))}'
#         )
#         old_distances = distances.copy()
#         iteration += 1
#         current_nearest = np.argmin(distances, axis=1)
#         if (last_nearest == current_nearest).all():
#             logging.info(
#                 f'Generated {len(centroids)} anchors in '
#                 f'{iteration} iterations'
#             )
#             return centroids, frame
#         for anchor in range(k):
#             centroids[anchor] = distance_func(
#                 relative_sizes[current_nearest == anchor], axis=0
#             )
#         last_nearest = current_nearest


def interpolate_nans(x):
    """Overwrite NaNs with column value interpolations."""
    for j in range(x.shape[1]):
        mask_j = np.isnan(x[:, j])
        x[mask_j, j] = np.interp(np.flatnonzero(mask_j), np.flatnonzero(~mask_j), x[~mask_j, j])
    return x


def generate_anchors(locations, n_anchors=9):
    locations["width"] = locations["x_pixel"].apply(lambda _: 70)
    locations["height"] = locations["y_pixel"].apply(lambda _: 70)
    locations["width"] = locations["width"] / locations["image_width"]
    locations["height"] = locations["height"] / locations["image_height"]
    w = locations["width"].to_numpy()
    h = locations["height"].to_numpy()
    w.sort()
    h.sort()
    x = [w, h]
    x = np.asarray(x).T
    kmeans = KMeans(n_clusters=n_anchors)
    kmeans.fit(x)
    y = kmeans.predict(x)
    centers3 = kmeans.cluster_centers_
    yolo_anchor_average = []
    for ind in range(9):
        yolo_anchor_average.append(np.mean(x[y == ind], axis=0))
    yolo_anchor_average = np.array(yolo_anchor_average)
    return interpolate_nans(yolo_anchor_average), x, y


if __name__ == "__main__":
    # locations = read_csv("/data2/seals/tfrecords/all.csv")
    locations = read_csv("/data2/seals/tfrecords/416/train/records.csv")
    yolo_anchor_average, x, y = generate_anchors(locations, 9)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=2, cmap="viridis")
    plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c="red", s=50)
    plt.show()
    plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c="red", s=50)
    plt.scatter(original_anchors[:, 0], original_anchors[:, 1], c="blue", s=50)
    plt.show()
