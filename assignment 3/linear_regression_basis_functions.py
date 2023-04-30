import numpy as np


def linear_regression(xs, ys, basis_functions=None):
    """
    Overcome limitation of non-linear relationship between feature and output.
    Extend the features available to our linear model by constructing new features that are
    non-linear on the original feature set, are a result of basis functions.
    :param xs: 2-dimensional array of training inputs
    :param ys: 1-dimensional array of training outputs
    :param basis_functions: list of basis functions. When functions are not provided, the algorithm should behave as an
    ordinary linear regression using normal equations.
    :return: one-dimensional array (vector) of coefficients
    where the first elements is the offset and the rest are the coefficients of the corresponding basis functions.
        * Each basis function takes a complete input vector
        * returns a scalar which is the value of the basis function for that input.
    """
    row_m, col_n = xs.shape

    X_b = np.c_[np.ones((row_m, 1)), xs]  # add x0 = 1 to each instance

    if basis_functions:
        for f in basis_functions:
            fx = np.apply_along_axis(f, 1, X_b)
            print(fx)
            return np.linalg.inv(fx.T.dot(fx)).dot(fx.T).dot(ys)

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(ys)

    return theta_best


xs = np.arange(5).reshape((-1, 1))
ys = np.array([3, 6, 11, 18, 27])
# Can you see y as a function of x? [hint: it's quadratic.]
functions = [lambda x: x[0], lambda x: x[0] ** 2]
print(linear_regression(xs, ys, functions))
