import numpy as np

def compute_gradient(y, tx, w):
    """
    This method is used to compute the gradient
    :param y: label vector y
    :param tx: feature matrix X
    :param w: weights w
    :return: the associated gradient
    """
    e = y - tx.dot(w)
    n = len(y)
    return -np.dot(tx.T, e) / n