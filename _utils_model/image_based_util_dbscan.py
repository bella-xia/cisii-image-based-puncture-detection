from sklearn.cluster import DBSCAN
import numpy as np


def filter_small_segments_with_dbscan(mask, min_size=500, eps=10, min_samples=1):
    filtered_mask = np.zeros_like(mask)
    y_indices, x_indices = np.where(mask > 0.1)

    if len(y_indices) == 0:
        return filtered_mask  # No foreground pixels

    points = np.column_stack((y_indices, x_indices))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Remove noise labels (-1) before computing unique clusters
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    # print(counts)

    if len(unique_labels) == 0:
        return filtered_mask  # No valid clusters

    # Find the largest valid cluster
    max_label = unique_labels[np.argmax(counts)]
    max_size = np.max(counts)

    if max_size >= min_size:
        filtered_mask[
            y_indices[labels == max_label], x_indices[labels == max_label]
        ] = 1

    return filtered_mask
