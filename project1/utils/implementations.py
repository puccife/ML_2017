import numpy as np
from utils.proj1_helpers import predict_labels
from utils.train_utils import batch_iter, build_poly
from utils.gradient import compute_gradient
from utils.costfunction import compute_loss, compute_loss_neg_log_likelihood, sigmoid
def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """
    Least squares using gradient descent
    :param y: label vector y
    :param tx: features matrix X
    :param initial_w: initial weights w
    :param max_iters: number of iterations
    :param gamma: stepsize
    :return: weights and loss after gradient descent
    """
    # Initialize parameters
    loss = 999
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        termine = np.dot(gamma,gradient)
        # update w by gradient
        w = w - termine
    return w, loss

def least_squares_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Least squares using stochastic gradient descent
    ::param y: label vector y
    :param tx: features matrix X
    :param initial_w: initial weights w
    :param max_iters: number of iterations
    :param gamma: stepsize
    :return: weights and loss after stochastic gradient descent
    """
    # Initialize parameters, batchsize = 1 for SGD
    loss = 999
    batch_size = 1
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    for n_iter, [mb_y, mb_tx] in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        # compute gradient and loss
        gradient = compute_gradient(mb_y, mb_tx, w)
        loss = compute_loss(y, tx, w)
        termine = np.dot(gamma,gradient)
        # update w
        w = w - termine
    return w, loss

def least_squares(y, tx):
    """
    Least squares using normal equation
    ::param y: label vector y
    :param tx: features matrix X
    :return: weights and loss
    """
    tx_transpose = np.transpose(tx)
    gram_matrix = np.dot(tx_transpose,tx)
    step_1 = np.linalg.solve(gram_matrix,tx_transpose)
    w = np.dot(step_1,y)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equation
    :param y: label vector y
    :param tx: features matrix X
    :param lambda_: lambda used for regularization
    :return: weights and loss
    """
    x_t = tx.T
    lam_ = lambda_ * 2 * len(y)
    gram_matrix = np.dot(x_t,tx)
    term_1 = lam_ * np.eye(tx.shape[1])
    term_2 = gram_matrix + term_1
    step_1 = np.linalg.solve(term_2,x_t)
    w = np.dot(step_1,y)
    loss = compute_loss(y, tx, w)
    ridge_term = (np.linalg.norm(w, ord=2)) ** 2
    loss = loss + ridge_term * lambda_
    return w, loss

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Learning by gradient descent step
    ::param y: label vector y
    :param tx: features matrix X
    :param w: actual weights
    :param gamma: stepsize
    :return: weights and loss after each step
    """
    loss = compute_loss_neg_log_likelihood(y, tx, w)
    gradient = np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)
    w = w - gamma * gradient
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    ::param y: label vector y
    :param tx: features matrix X
    :param initial_w: initial weights vector
    :param max_iters: number of iterations
    :param gamma: stepsize
    :return: weights and loss after each logistic regression
    """
    # Convert range of Y in {0,1} instead of {-1,1}
    y = (1 + y) / 2
    losses = []
    threshold = 0.0001
    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    for _ in range(max_iters):
        # Learning by gradient descent
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        # Stopping criterion
        if _ % 100 == 0:
            print("Loss after " + str(_) + " iterations = " + str(loss))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = compute_loss_neg_log_likelihood(y,tx,w)
    norm = np.linalg.norm(w)
    norm_squared = norm**2
    loss = loss + (lambda_*norm_squared)
    gradient_regularized=[]
    for i in range(w.shape[0]):
        gradient_regularized.append(lambda_*4*norm*w[i])
    gradient =  np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)
    gradient = gradient + gradient_regularized
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y,tx,w,lambda_)
    termine = np.dot(gamma,gradient)
    w = w - termine
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    :param y: label vector y
    :param tx: features matrix X
    :param lambda_: lambda used for regularization
    :param initial_w: initial weights vector
    :param max_iters: number of iterations
    :param gamma: stepsize
    :return: weights and loss after each logistic regression
    """
    y = (1 + y) / 2
    losses = []
    threshold = 0.0001
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    for iter in range(max_iters):
        # Learning by penalized gradient descent
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma,lambda_)
        losses.append(loss)
        # Stopping criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Compute gradient using stochastic gradient descent
    :param y: label vector y
    :param tx: features matrix X
    :param initial_w: initial weights vector
    :param batch_size: = 1 for SGD
    :param max_iters: number of iterations
    :param gamma: stepsize
    :return: loss, ws after stochastic gradient descent
    """
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
    """
    Cross validation used for ridge regression
    ::param y: label vector y
    :param tx: features matrix X
    :param k_indices: k - indices builded for cross validation
    :param k: actual k fold indices
    :param lambda_: lambda used for regularization
    :param degree: polynomial degree of the model (+ sqrt)
    :return: training loss, testing loss, final weights
    """
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
    ws, loss_tr = ridge_regression(y=train_y, lambda_=lambda_, tx=train_px)
    ridge_term = (np.linalg.norm(ws, ord=2)) ** 2
    loss_te = compute_loss(tx=test_px, w=ws, y=test_y) + ridge_term * lambda_
    y_pred_0 = predict_labels(ws, test_px)
    accuracy_0 = 1 - np.mean(y_pred_0 != test_y)
    print("Accuracy: " + str(accuracy_0) + "%")
    return loss_tr, loss_te, ws