import numpy as np


def random_forest_vote(predictions):
    preds = np.asarray(predictions)
    n_samples = preds.shape[1]

    result = []

    for i in range(n_samples):
        votes = preds[:, i]
        labels, counts = np.unique(votes, return_counts=True)
        majority = labels[np.argmax(counts)]
        result.append(int(majority))

    return result