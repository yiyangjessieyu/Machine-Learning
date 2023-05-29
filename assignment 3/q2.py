import numpy as np


def linear_regression(xs, ys):
    """
    θ = ( X.T  X )^(-1)  X.T   y
    :param xs: training input m x n design matrix
    :param ys: training ouput 1d vector with m elements
    :return: 1d vector θ, with (n + 1) elements
    """
    n_examples = xs.shape[0]
    intercept = np.ones((n_examples, 1)) # column of 1s
    xs = np.concatenate([intercept, xs], axis=1)

    return np.dot(np.dot(np.linalg.inv(np.dot(xs.T, xs)), xs.T), ys)

xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)
print(linear_regression(xs, ys))