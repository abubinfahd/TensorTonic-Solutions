import numpy as np


def majority_classifier(y_train, X_test):
    y_train = np.asarray(y_train)

    if y_train.size == 0:
        raise ValueError("y_train cannot be empty")

    values, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    candidates = values[counts == max_count]

    for label in y_train:
        if label in candidates:
            majority = label
            break

    X_test = np.asarray(X_test)
    return np.full(len(X_test), majority, dtype=int)