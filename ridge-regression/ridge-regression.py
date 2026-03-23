import numpy as np


def ridge_regression(X, y, lam=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_features = X.shape[1]
    identity = np.eye(n_features)
    beta = np.linalg.inv(X.T @ X + lam * identity) @ X.T @ y

    return beta


X = np.array([[1, 2],
              [2, 3],
              [3, 5]])

y = np.array([5, 8, 12])

beta = ridge_regression(X, y, lam=1.0)
print(beta)