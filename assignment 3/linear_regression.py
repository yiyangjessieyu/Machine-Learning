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

    a = np.asarray([])

    X_b = np.c_[np.ones((row_m, 1)), xs]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(ys)

    # X_new = np.array([[0], [1]])
    # X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    # y_predict = X_new_b.dot(theta_best)
    # for col_i in range(col_n):
    #     X = xs[ :, col_i]
    #
    #     theta = np.dot(
    #         np.linalg.inv(
    #             np.dot(
    #                 X.T.reshape(row_m, 1),
    #                 X.reshape(1, row_m))
    #         ),
    #         np.dot(X.T, ys)
    #     )
    #
    #     np.append(a, theta)

    return theta_best


xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)
print(linear_regression(xs, ys))

xs = np.array([[1, 2, 3, 4],
               [6, 2, 9, 1]]).T
ys = np.array([7, 5, 14, 8]).T
print(linear_regression(xs, ys))
