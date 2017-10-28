import numpy as np

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    n = len(y)
    return -np.dot(tx.T, e) / n