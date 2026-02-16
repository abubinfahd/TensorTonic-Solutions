def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid using
    squared Euclidean distance.

    If distances are tied, assign to the centroid
    with the smallest index.
    """
    if not points or not centroids:
        return []

    dim = len(points[0])
    assignments = []

    for point in points:
        best_dist = float("inf")
        best_idx = 0

        for idx, centroid in enumerate(centroids):
            dist = 0

            for d in range(dim):
                diff = point[d] - centroid[d]
                dist += diff * diff

            # strict < ensures tie goes to smaller index
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        assignments.append(best_idx)

    return assignments
