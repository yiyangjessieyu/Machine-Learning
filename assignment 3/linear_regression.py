import numpy as np


def linear_regression(xs, ys):
    """
    Takes 2 numpy arrays as parameters.
    :param xs: inputs of training data as an m×n array (design matrix)
    :param ys: outputs of training data as a one-dimensional array (vector) with m elements.
    :return: one-dimensional array (vector) θ, with (n + 1) elements,
    which contains the least-squares regression coefficients of the features; the first ("extra") value is the intercept.
    """
