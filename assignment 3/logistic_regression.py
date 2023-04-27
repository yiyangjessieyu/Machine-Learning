import math
import numpy as np


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def logistic_regression(xs, ys, alpha, num_iterations):
    """
    In Logistic regression, given a training dataset of a binary classification problem where outputs are either positive
    or negative (1 or 0), the learning algorithm finds a model (function) that
        takes an input (feature) vector and
        produces a value between 1 and 0 which is the probability of the input vector being positive (or negative).
    :param xs: 2-dimensional array of training inputs
    :param ys: 1-dimensional array of training outputs
    :param alpha: training/learning rate
    :param num_iterations: number of iterations to perform/loop over the entire dataset
    :return: a callable model that we can use to classify new feature vectors
        function input: one-dimensional array (vector) of values,
        Produces value between 0-1 indicating the probability of that input belonging to the positive class.
    """

    # no closed-form solution so need to perform gradient descent; Stochastic gradient descent,


    row_m, col_n = xs.shape
    theta = np.c_[np.zeros((1, row_m))] # TODO starting with a vector of zeros.

    for iterate in range(num_iterations):
        for row_i in range(row_m):
            z = sum(np.dot(theta.T, xs[row_i]))
            theta[:, row_i] += theta[:, row_i] + alpha * (ys[row_i] - sigmoid(z)) * xs[row_i]

    print(theta)

    def model(unseen_x):
        """:param unseen_x: 1D feature vector"""
        z = theta.T * unseen_x[0]  # TODO needs to be an int, dot product?
        return sigmoid(z)

    return model


xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))  # TODO WHY IS THIS JUST 1 COL
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))

    # The ith column of a two-dimensional numpy array x can be selected by x[:, i].
