import numpy as np

def build_poly(x, degree):
    px = np.ones(len(x))
    for n in range(degree):
        px = np.c_[px, pow(x, n+1)]
    return px

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

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
