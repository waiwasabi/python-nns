import numpy as np


def norm(v): return np.sqrt(np.sum(v**2, axis=1))


def minmax(X, x_max, x_min):
    return (X - x_min) / (x_max - x_min)


def knn(X, y, v, k):
    """
    Executes the k-NN algorithm.

    Args:
        X (array-like): a dataframe of training examples
        y (pd.Series or similar): a Series of labels corresponding to the training examples
        v (array-like): a single instance of input data
        k (int): a natural number indicating what k to use when calculating k-NN

    Returns:
        The predicted label according to k-NN.
    """
    dists = norm(X - v)
    knn_args = dists.argsort()[:k]
    knn_labels = y.iloc[knn_args]
    return knn_labels.mode()[0]
