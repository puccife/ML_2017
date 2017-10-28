import numpy as np


def sigmoid(t):
    """
    Sigmoid cost function
    :param t: the argument of the sigmoid cost function
    :return: the sigmoid value of the argument (t)
    """
    return np.exp(-np.logaddexp(0, -t))


def mse(e):
    """
    MSE cost function
    :param e: the argument of the MSE cost function
    :return: the MSE value of the argument (e)
    """
    return e.dot(e) / (2 * len(e))


def mae(e):
    """
    MAE cost function
    :param e: the argument of the MAE cost function
    :return: the MAE value of the argument (e)
    """

    return np.mean(np.abs(e))


def compute_loss_neg_log_likelihood(y, tx, w):

    """
    Negative log-likelihood cost function
    :param y: labels - matrix
    :param tx: features - matrix
    :param w: weights - vector
    :return: the loss using log-likelihood cost function
    """
    t = np.dot(tx, w)
    term1 = np.maximum(t, 0) + np.log(np.exp(-np.absolute(t)) + 1)
    term2 = (np.multiply(y, t))
    erro = term1 - term2
    loss = sum(erro)
    return loss


def compute_loss(y, tx, w, function='mse'):
    """
    Computes the loss, based on the cost function specified
    :param y: labels - matrix
    :param tx: features - matrix
    :param w: weights - vector
    :param function: string argument to identify the cost function to use
    :return: the loss using the argument cost function
    """
    e = y - tx.dot(w)
    if (function == 'mae'):
        return mae(e)
    elif (function == 'sigmoid'):
        return sigmoid(e)
    return mse(e)
