import numpy as np


def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    train = np.asarray(X_train, dtype=float)
    test = np.asarray(X_test, dtype=float)

    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    n_train = train.shape[0]
    n_test = test.shape[0]

    diffs = test[:, None, :] - train[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    order = np.argsort(dists, axis=1, kind="mergesort")

    result = np.full((n_test, k), -1, dtype=int)
    limit = min(k, n_train)
    result[:, :limit] = order[:, :limit]

    return result