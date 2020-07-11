import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import imagesize
import cv2
import os
import sys
from pathlib import Path
from helpers.utils import default_logger

if sys.platform == 'darwin':
    plt.switch_backend('Qt5Agg')


def save_fig(title, save_figures=True):
    """
    Save generated figures to Output folder.
    Args:
        title: Figure title also the image to save file name.
        save_figures: If True, figure will be saved

    Returns:
        None
    """
    if save_figures:
        saving_path = str(
            Path(os.path.join('../yolo', 'Output', 'Plots', f'{title}.png'))
            .absolute()
            .resolve()
        )
        if os.path.exists(saving_path):
            return
        plt.savefig(saving_path)
        default_logger.info(f'Saved figure {saving_path}')
        plt.close()


def visualize_box_relative_sizes(frame, save_result=True):
    """
    Scatter plot annotation box relative sizes.
    Args:
        frame: pandas DataFrame with the annotation data.
        save_result: If True, figure will be saved

    Returns:
        None
    """
    title = f'Relative width and height for {frame.shape[0]} boxes.'
    if os.path.exists(
        os.path.join('../yolo', 'Output', 'Plots', f'{title}.png')
    ) or (frame is None):
        return
    sns.scatterplot(
        x=frame['Relative Width'],
        y=frame['Relative Height'],
        hue=frame['Object Name'],
        palette='gist_rainbow',
    )
    plt.title(title)
    save_fig(title, save_result)


def visualize_k_means_output(centroids, frame, save_result=True):
    """
    Visualize centroids and anchor box dimensions calculated.
    Args:
        centroids: 2D array of shape(k, 2) output of k-means.
        frame: pandas DataFrame with the annotation data.
        save_result: If True, figure will be saved

    Returns:
        None
    """
    title = (
        f'{centroids.shape[0]} Centroids representing relative anchor sizes.'
    )
    if os.path.exists(
        os.path.join('../yolo', 'Output', 'Plots', f'{title}.png')
    ) or (frame is None):
        return
    fig, ax = plt.subplots()
    visualize_box_relative_sizes(frame)
    plt.title(title)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')
    save_fig(title, save_result)


def visualize_boxes(relative_anchors, sample_image, save_result=True):
    """
    Visualize anchor boxes output of k-means.
    Args:
        relative_anchors: Output of k-means.
        sample_image: Path to image to display as background.
        save_result: If True, figure will be saved

    Returns:
        None
    """
    title = 'Generated anchors relative to sample image size'
    if os.path.exists(os.path.join('../yolo', 'Output', 'Plots', f'{title}.png')):
        return
    img = cv2.imread(sample_image)
    width, height = imagesize.get(sample_image)
    center = int(width / 2), int(height / 2)
    for relative_w, relative_h in relative_anchors:
        box_width = relative_w * width
        box_height = relative_h * height
        x0 = int(center[0] - (box_width / 2))
        y0 = int(center[1] - (box_height / 2))
        x1 = int(x0 + box_width)
        y1 = int(y0 + box_height)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 4)
    plt.imshow(img)
    plt.grid()
    plt.title(title)
    save_fig(title, save_result)


def visualize_pr(calculated, save_results=True, fig_prefix=''):
    """
    Visualize precision and recall curves(post-training evaluation)
    Args:
        calculated: pandas DataFrame with combined average precisions 
            that were calculated separately on each object class.
        save_results: If True, plots will be saved to Output folder.
        fig_prefix: str, prefix to add to save path.

    Returns:
        None
    """
    for item in calculated['object_name'].drop_duplicates().values:
        plt.figure()
        title = (
            f'{fig_prefix} Precision and recall curve for {fig_prefix} {item}'
        )
        plt.title(title)
        recall = calculated[calculated['object_name'] == item]['recall'].values
        precision = calculated[calculated['object_name'] == item][
            'precision'
        ].values
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title(title)
        save_fig(title, save_results)


def plot_compare_bar(col1, stats, fig_prefix='', col2=None):
    """
    Plot a bar chart comparison between 2 evaluation statistics or 1 statistic.
    Args:
        col1: column name present in frame.
        stats: pandas DataFrame with evaluation statistics.
        fig_prefix: str, prefix to add to save path.
        col2: second column name present in frame to compare with col1.

    Returns:
        None
    """
    stats = stats.sort_values(by=col1)
    ind = np.arange(len(stats))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ind, stats[col1], width, color='red', label=col1)
    if col2:
        ax.barh(ind + width, stats[col2], width, color='blue', label=col2)

    ax.set(
        yticks=ind + width,
        yticklabels=stats['Class Name'],
        ylim=[2 * width - 1, len(stats)],
        title=(
            f'{fig_prefix} {col1} vs {col2} evaluation results'
            if col2
            else f'{fig_prefix} {col1} evaluation results'
        ),
    )
    for patch in ax.patches:
        pw = patch.get_width()
        _, y = patch.get_xy()
        color = patch.get_facecolor()
        ax.text(
            pw + 3,
            y + width / 2,
            str(pw),
            color=color,
            verticalalignment='center',
        )
    ax.legend(loc='lower right')


def visualize_evaluation_stats(stats, fig_prefix='', save_results=True):
    """
    Visualize True positives vs False positives, actual vs. detections
        and average precision.
    Args:
        stats: pandas DataFrame with evaluation statistics.
        fig_prefix: str, prefix to add to save path.
        save_results: If True, plots will be saved to Output folder.

    Returns:
        None
    """
    plot_compare_bar('True Positives', stats, fig_prefix, 'False Positives')
    save_fig('True positives vs False positives.png', save_results)
    plot_compare_bar('Actual', stats, fig_prefix, 'Detections')
    save_fig('Actual vs Detections.png', save_results)
    plot_compare_bar('Average Precision', stats, fig_prefix)
    save_fig(f'{fig_prefix} Average Precision.png', save_results)


def visualization_wrapper(to_visualize):
    """
    Wrapper for visualization.
    Args:
        to_visualize: function to visualize.

    Returns:
        to_visualize
    """

    def visualized(*args, **kwargs):
        result = to_visualize(*args, **kwargs)
        if to_visualize.__name__ in ['parse_voc_folder', 'adjust_non_voc_csv']:
            visualize_box_relative_sizes(result)
            plt.show()
        if to_visualize.__name__ == 'k_means':
            all_args = list(kwargs.values()) + list(args)
            if not any([isinstance(item, pd.DataFrame) for item in all_args]):
                return result
            visualize_k_means_output(*result)
            plt.show()
            visualize_boxes(
                result[0], os.path.join('../yolo', 'Samples', 'sample_image.png')
            )
            plt.show()
        return result

    return visualized
