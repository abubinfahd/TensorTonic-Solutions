import numpy as np


def linear_regression_closed_form(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    Xt = X.T
    w = np.linalg.inv(Xt @ X) @ Xt @ y

    return w