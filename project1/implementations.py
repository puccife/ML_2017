import numpy as np
from plots import cross_validation_visualization

def build_poly(x, degree):
    px = np.ones(len(x))
    for n in range(degree):
        px = np.c_[px, pow(x, n+1)]
    return px

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    t = np.dot(tx,w)
    t = t.flatten()
    term1 = np.log(1+np.exp(t))
    term2 = (np.multiply(y,t))
    erro = term1 - term2
    loss = sum(erro)
    return loss

def learning_by_gradient_descent(y, tx, w, gamma):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    termine = np.dot(gamma,gradient)
    w = w - termine
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    S_nn = []
    sigmoid_value = sigmoid(np.dot(tx,w))
    sigmoid_value = sigmoid_value.flatten()
    print('cussomak')
    S_nn = sigmoid_value*(1-sigmoid_value)
    print('cussomak')
    S_diag = np.diag(S_nn.flatten())
    print('cussomak')
    tx_transpose = tx.T
    print('cussomak')
    print(tx_transpose)
    print(S_diag)
    temp = np.dot(tx_transpose,S_diag)
    print('cussomak')
    hessian = np.dot(temp,tx)
    
    return hessian

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    print('hessian')
    hessian = calculate_hessian(y,tx,w)
    print('cussomak')
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

def sigmoid(t):
    """apply sigmoid function on t."""
    exponential_value = np.exp(t)
    sigmoid_value = (exponential_value) / ( 1 + exponential_value)
    return sigmoid_value

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigmoid_value = sigmoid(np.dot(tx,w))
    tx_transpose = tx.T
    sigmoid_value = sigmoid_value.flatten()
    gradient = np.dot(tx_transpose,(sigmoid_value-y))
    return gradient

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

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


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