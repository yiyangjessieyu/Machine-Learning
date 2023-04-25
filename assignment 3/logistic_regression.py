import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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


    # no closed-form solution so need to perform gradient descent;
    # Stochastic gradient descent, starting with a vector of zeros.
    #keywords to search^ + learning rate iteration

    row_m, col_n = xs.shape
    X_b = np.c_[np.ones((row_m, 1)), xs]
    for i in range(num_iterations):
        X_b += X_b + alpha * () * xs


xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))

    # The ith column of a two-dimensional numpy array x can be selected by x[:, i].
