import numpy as np


def linear_regression(xs, ys):
    """
    Takes 2 numpy arrays as parameters.
    :param xs: inputs of training data as an row_m×col_n array (design matrix)
    :param ys: outputs of training data as a one-dimensional array (vector) with m elements.
    :return: one-dimensional array (vector) θ, with (col_n + 1) elements,
    which contains the least-squares regression coefficients of the features; the first ("extra") value is the intercept.
    """
    row_m, col_n = xs.shape

    X_b = np.c_[np.ones((row_m, 1)), xs]  # add x0 = 1 to each instance
    print(xs.shape, X_b.shape)
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(ys)

    return theta_best


xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)
print(linear_regression(xs, ys))

xs = np.array([[1, 2, 3, 4],
               [6, 2, 9, 1]]).T
ys = np.array([7, 5, 14, 8]).T
print(linear_regression(xs, ys))
