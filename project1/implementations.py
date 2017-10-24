import numpy as np
from plots import cross_validation_visualization
import os

clear = lambda: os.system('cls')

def build_poly(x, degree):
    px = np.ones(len(x))
    for n in range(degree):
        px = np.c_[px, pow(x, n+1)]
    return px

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    t = np.dot(tx,w)
    term1 = np.log(1+np.exp(t, dtype='float128'))
    term2 = (np.multiply(y,t))
    erro = term1 - term2
    loss = sum(erro)
    return loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def learning_by_gradient_descent(y, tx, w, gamma):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    loss = calculate_loss(y,tx,w)
    # w = calculate_logistic_SGD(y, tx, w, gamma)
    gradient = calculate_gradient(y,tx,w)
    termine = np.dot(gamma,gradient)
    w = w - termine
    return loss, w

def calculate_logistic_SGD(y, tx, w, gamma):
    batch_size = 100
    max_iters = 2500
    for b_y, b_x in batch_iter(y, tx, batch_size, max_iters):
        gradient = calculate_gradient(b_y, b_x, w)
        termine = np.dot(gamma, gradient)
        w = w - termine
    return w
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    t = np.dot(tx,w)
    sigmoid_value = sigmoid(t)
    tx_transpose = tx.T
    res = sigmoid_value-y
    gradient = np.dot(tx_transpose,(sigmoid_value-y))
    return gradient  
   
def sigmoid(t):
    """apply sigmoid function on t."""
    exponential_value = np.exp((-t), dtype='float128')
    sigmoid_value = 1 / (1 + exponential_value)
#    sigmoid_value = (exponential_value) / ( 1 + exponential_value)
    return sigmoid_value


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    print('before hessian')
    hessian = calculate_hessian(y,tx,w)
    return loss, gradient, hessian

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def least_squares(y, tx):
    x_t = np.transpose(tx)
    inner = np.dot(x_t, tx)
    term = np.linalg.solve(inner, x_t)
    w_s = np.dot(term, y)
    
    mse = (y - np.dot(tx, w_s))**2
    mse = sum(mse) / len(y)
    
    return mse, w_s

def ridge_regression(y, tx, lambda_):
    x_t = np.transpose(tx)
    term = np.dot(x_t, tx)
    identity = np.identity(len(term))
    lambda_i = np.dot(lambda_,identity)
    res = np.linalg.solve(term+lambda_i, x_t)
    ws = np.dot(res, y)
    return ws

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    splitting_index = int(len(x) * ratio)
    
    train_x, test_x = x[0:splitting_index], x[splitting_index:len(x)]
    train_y, test_y = y[0:splitting_index], y[splitting_index:len(y)]
    
    return train_x, train_y, test_x, test_y

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    gradient = -(1/len(y)) * np.dot(tx.T, e)
    loss = compute_mse(y, tx, w)
    return gradient, loss

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        w_gradient, loss = compute_gradient(y, tx, w)
        termine = np.dot(gamma, w_gradient)
        w = w - termine
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def compute_stoch_gradient(y, tx, w):
    gradient, loss = compute_gradient(y, tx, w)
    return gradient, loss


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    losses =[]
    ws = []
    for b_y, b_x in batch_iter(y, tx, batch_size, max_iters):
        gradient, loss = compute_stoch_gradient(b_y, b_x, initial_w)
        initial_w = initial_w - gamma * gradient
        losses.append(loss)
        ws.append(initial_w)
    return losses, ws

def standardize2(x):
    """Standardize the original data set."""
    tx = np.ma.array(x, mask=np.isnan(x))
    me = tx.mean(axis=0)
    mean_x = np.ma.getdata(me)
    tx = tx - mean_x
    std_x = np.std(tx, axis=0)
    std_x = np.ma.getdata(std_x)
    tx = tx / std_x
    return tx, mean_x, std_x

def standardize(x):
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    normed = (x - mean_x) / std_x
    print(normed.mean(axis=0))
    print(normed.std(axis=0))
    return normed, mean_x, std_x
    
def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    S_nn = []
    sigmoid_value = sigmoid(np.dot(tx,w))
    S_nn = sigmoid_value*(1-sigmoid_value)
    S_diag = np.diag(S_nn.flatten())
    tx_transpose = tx.T
    print('before temp')
    temp = np.dot(tx_transpose,S_diag)
    print('after temp')
    hessian = np.dot(temp,tx)
    print('after hessian')
    return hessian

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
    
    loss_tr = compute_mse(tx=train_px, w=ws, y=train_y)  + ridge_term * lambda_
    loss_te = compute_mse(tx=test_px, w=ws, y=test_y)  + ridge_term * lambda_
    return loss_tr, loss_te, ws