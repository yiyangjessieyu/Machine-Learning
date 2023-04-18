import numpy as np


def dot_product(a, b, n):
    return sum([a[i] * b[i] for i in range(n)])


def linear_regression_1d(data):
    """
    Where   x is vector of feature values, and
            y is the vector of response values, and
            n is the length of these vectors.
    :param data: list of pairs (feature value x, response value y)
    :return: list of pairs (gradient m, intercept c)
    """

    x, y = zip(*data)

    sum_x, sum_y, n = sum(x), sum(y), len(data)

    # m = (n x.y - ∑x ∑y) / (n x.x - (∑x)**2), the slope of the line of least squares fit.
    m = (n * dot_product(x, y, n) - sum_x * sum_y) / (n * dot_product(x, x, n) - sum_x ** 2)

    # c = (∑y - m∑x)/n, the intercept of the line of least squares fit.
    c = (sum_y - m * sum_x) / n

    return (m, c)

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

    for col_i in range(col_n):
        X = xs[ :, col_i]

        theta = np.dot(
            np.linalg.inv(np.multiply(X.T, X)),
            np.dot(X.T, ys)
        )

        np.append(a, theta)

    return a


xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)
# print(xs)
# print(ys)
# XT = xs.transpose()
# print(XT)
print(linear_regression(xs, ys))
