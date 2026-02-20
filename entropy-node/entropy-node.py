import numpy as np


def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.

    Parameters
    ----------
    y : array-like
        Class labels for samples in the node.

    Returns
    -------
    float
        Entropy value in bits.
    """
    if len(y) == 0:
        return 0.0

    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)