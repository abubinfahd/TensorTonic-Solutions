import numpy as np


def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.

    Returns
    -------
    list of list of float
        Centroids of shape (k, n_features)
    """
    points = np.asarray(points, dtype=np.float64)
    assignments = np.asarray(assignments)

    n_samples, n_features = points.shape
    centroids = np.zeros((k, n_features), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)

    # Accumulate sums
    for i in range(n_samples):
        cluster_id = assignments[i]
        centroids[cluster_id] += points[i]
        counts[cluster_id] += 1

    # Convert sums to means
    for cluster_id in range(k):
        if counts[cluster_id] > 0:
            centroids[cluster_id] /= counts[cluster_id]
        else:
            # If empty cluster, keep zeros (or handle as needed)
            centroids[cluster_id] = 0.0

    return centroids.tolist()
