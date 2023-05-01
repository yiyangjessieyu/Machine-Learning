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
    n_examples, d = xs.shape

    xs = np.c_[np.ones((n_examples, 1)), xs]  # add x0 = 1 to each instance

    if basis_functions:
        phi = np.ones((ys.shape[0], d))
        # print(phi)
        # print(xs)
        for fi, f in enumerate(basis_functions):
            for i in range(n_examples):
                phi[fi] = np.apply_along_axis(f, 0, xs[i])

        xs = phi

    return np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)


# def linear_regression(xs, ys, basis_functions=None):
#     """
#     Overcome limitation of non-linear relationship between feature and output.
#     Extend the features available to our linear model by constructing new features that are
#     non-linear on the original feature set, are a result of basis functions.
#     :param xs: 2-dimensional array of training inputs
#     :param ys: 1-dimensional array of training outputs
#     :param basis_functions: list of basis functions. When functions are not provided, the algorithm should behave as an
#     ordinary linear regression using normal equations.
#     :return: one-dimensional array (vector) of coefficients
#     where the first elements is the offset and the rest are the coefficients of the corresponding basis functions.
#         * Each basis function takes a complete input vector
#         * returns a scalar which is the value of the basis function for that input.
#     """
#     n_examples, d = xs.shape
#
#     xs = np.c_[np.ones((n_examples, 1)), xs]  # add x0 = 1 to each instance
#
#     # Calculate design matrix Phi
#     Phi = np.ones((ys.shape[0], d))
#     print(Phi)
#
#     for i in range(n_examples):
#         Phi[i] = np.apply_along_axis(basis_functions[1], 0, xs[i])
#
#     # Calculate parameters w and alpha
#     print(Phi)
#
#     w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ ys
#     print(w)
#     alpha = sum((ys - Phi @ w) ** 2) / len(ys)
#
#     # if basis_functions:
#     #     phi = np.ones((ys.shape[0], d))
#     #     print(phi)
#     #     for fi, f in enumerate(basis_functions):
#     #         phi[:, fi+1] = [np.apply_along_axis(f, 0, xs[i]) for i in range(n_examples)]
#     #         print(phi)
#
#
#
#     return alpha

# xs = np.arange(5).reshape((-1, 1))
# ys = np.arange(1, 11, 2)
# print(linear_regression(xs, ys, None))
#
# xs = np.array([[1, 2, 3, 4],
#                [6, 2, 9, 1]]).T
# ys = np.array([7, 5, 14, 8]).T
# print(linear_regression(xs, ys))

xs = np.arange(5).reshape((-1, 1))
ys = np.array([3, 6, 11, 18, 27])
# Can you see y as a function of x? [hint: it's quadratic.]
functions = [lambda x: x[0], lambda x: x[0] ** 2]
print(linear_regression(xs, ys, functions))
