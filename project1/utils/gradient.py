import numpy as np
from utils.train_utils import build_poly
from utils.costfunction import mse

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    n = len(y)
    return -np.dot(tx.T, e) / n

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    test_idx = k_indices[k]
    train_idx = k_indices[np.arange(len(k_indices))!=k]
    train_idx = train_idx.flatten()
    test_idx = test_idx.flatten()
    train_x = np.array([x[i] for i in train_idx])
    train_y = np.array([y[i] for i in train_idx])
    test_x = np.array([x[i] for i in test_idx])
    test_y = np.array([y[i] for i in test_idx])
    train_px = build_poly(x=train_x, degree=degree)
    test_px = build_poly(x=test_x, degree=degree)
    ws = ridge_regression(y=train_y, lambda_=lambda_, tx=train_px)
    ridge_term = (np.linalg.norm(ws,ord=2))**2
    loss_tr = mse(tx=train_px, w=ws, y=train_y)  + ridge_term * lambda_
    loss_te = mse(tx=test_px, w=ws, y=test_y)  + ridge_term * lambda_
    return loss_tr, loss_te, ws
