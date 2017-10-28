import numpy as np

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    splitting_index = int(len(x) * ratio)
    train_x, test_x = x[0:splitting_index], x[splitting_index:len(x)]
    train_y, test_y = y[0:splitting_index], y[splitting_index:len(y)]
    return train_x, train_y, test_x, test_y

def build_poly(x, degree):
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
    lambdas = [
        5.73615251045e-07,
        1e-09,
        0.0280721620394,
        9.23670857187e-05,
        0.0148735210729,
        1.37382379588e-05,
        0.0280721620394,
        0.000329034456231
    ]
    return lambdas[i]


def get_degree(i):
    deg_mass = [
        1,1,
        3,5,
        4,4,
        3,5
    ]
    return deg_mass[i]