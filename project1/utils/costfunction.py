import numpy as np

def sigmoid(t):
    return np.exp(-np.logaddexp(0, -t))

def mse(e):
    return e.dot(e) / (2 * len(e))

def mae(e):
    return np.mean(np.abs(e))

def compute_loss_neg_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    t = np.dot(tx,w)
    term1 = np.maximum(t, 0) + np.log(np.exp(-np.absolute(t)) + 1)
    term2 = (np.multiply(y,t))
    erro = term1 - term2
    loss = sum(erro)
    return loss

def compute_loss(y, tx, w, function='mse'):
    e = y - tx.dot(w)
    if(function=='mae'):
        return mae(e)
    elif(function=='sigmoid'):
        return sigmoid(e)
    return mse(e)