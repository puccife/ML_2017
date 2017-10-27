import numpy as np
from utils.proj1_helpers import predict_labels
from utils.train_utils import batch_iter, build_poly
from utils.gradient import compute_gradient
from utils.costfunction import compute_loss, compute_loss_neg_log_likelihood, sigmoid
def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent
    """
    # if initial_w is None, we initialize it to a zeros vector
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    # Define parameters to store weight and loss
    loss = 0
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w -= gamma * gradient

    return w, loss

def least_squares_sgd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent
    """
    # if initial_w is None, we initialize it to a zeros vector
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    # Define parameters of the algorithm
    batch_size = 1

    # Define parameters to store w and loss
    loss = 0
    w = initial_w

    for n_iter, [mb_y, mb_tx] in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        # compute gradient and loss
        gradient = compute_gradient(mb_y, mb_tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w -= gamma * gradient

    return w, loss

def least_squares(y, tx):
    """ Least squares regression using normal equations
    """
    x_t = tx.T
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx)), x_t), y)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)

    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx) + lambd * np.eye(tx.shape[1])), x_t), y)
    loss = compute_loss(y, tx, w)

    return w, loss

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_neg_log_likelihood(y, tx, w)
    gradient = np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)

    w -= gamma * gradient

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    y = (1 + y) / 2
    losses = []
    threshold = 0.0001
    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        # converge criteria
        if _ % 100 == 0:
            print("Loss after " + str(_) + " iterations = " + str(loss))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    y = (1 + y) / 2
    losses = []
    threshold = 0.1

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    norm = sum(w ** 2)
    cost = w + lambda_ * norm / (2 * np.shape(w)[0])

    return w, cost

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    loss = 0
    ws = initial_w
    for b_y, b_x in batch_iter(y, tx, batch_size, max_iters):
        gradient, loss = compute_gradient(b_y, b_x, ws)
        ws = ws - gamma * gradient
    return loss, ws

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
    train_idx = k_indices[np.arange(len(k_indices)) != k]
    train_idx = train_idx.flatten()
    test_idx = test_idx.flatten()
    train_x = np.array([x[i] for i in train_idx])
    train_y = np.array([y[i] for i in train_idx])
    test_x = np.array([x[i] for i in test_idx])
    test_y = np.array([y[i] for i in test_idx])
    train_px = build_poly(x=train_x, degree=degree)
    test_px = build_poly(x=test_x, degree=degree)
    ws, loss = ridge_regression(y=train_y, lambda_=lambda_, tx=train_px)
    ridge_term = (np.linalg.norm(ws, ord=2)) ** 2
    loss_tr = compute_loss(tx=train_px, w=ws, y=train_y) + ridge_term * lambda_
    loss_te = compute_loss(tx=test_px, w=ws, y=test_y) + ridge_term * lambda_
    y_pred_0 = predict_labels(ws, test_px)
    accuracy_0 = 1 - np.mean(y_pred_0 != test_y)
    print("Accuracy: " + str(accuracy_0) + "%")
    return loss_tr, loss_te, ws