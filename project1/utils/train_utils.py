import numpy as np

def split_data(x, y, ratio, seed=1):
    """
        Split training set into training and testing set given the ratio and the seed
    :param x: features matrix X of training
    :param y: labels vector y of training
    :param ratio: ratio for splitting training set
    :param seed: seed used to shuffle the set
    :return: training and testing set given the ratio
    """
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    splitting_index = int(len(x) * ratio)
    train_x, test_x = x[0:splitting_index], x[splitting_index:len(x)]
    train_y, test_y = y[0:splitting_index], y[splitting_index:len(y)]
    return train_x, train_y, test_x, test_y

def build_poly(x, degree):
    """
    Build polynomial features of matrix X, given a selected degree
    :param x: features matrix X
    :param degree: degree used to create the polynomial
    :return: the features matrix X after polynomial expansion
    """
    px = np.ones(len(x))
    px = np.c_[px, np.sqrt(np.abs(x))]
    for n in range(degree):
        px = np.c_[px, pow(x, n+1)]
    return px

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

def get_lambda(i):

    """
    This method is used to get the correct lambda for each classifier
    :param i: Number of classifier
    :return: lambda of the i-th classifier
    """
    lambdas = [
        0.000621016941892,1.37382379588e-05,
        0.00017433288222,9.23670857187e-05,
        0.00788046281567,1.08263673387e-06,
        0.0529831690628,1e-09
    ]
    return lambdas[i]

def get_degree(i):
    """
    This method is used to get the correct degree for each classifier
    :param i: Number of classifier
    :return: degree of the i-th classifier
    """
    deg_mass = [
        4,2,
        5,6,
        5,7,
        5,6
    ]
    return deg_mass[i]