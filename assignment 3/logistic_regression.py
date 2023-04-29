import math
import numpy as np


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
    sigmoid = lambda x: 1 / (1 + math.exp(-x))

    n_examples, d = xs.shape
    b_xs = np.c_[np.ones((n_examples, 1)), xs]  # add x0 = 1 to each instance
    theta = np.zeros(d + 1)

    for iterate in range(num_iterations):
        for i in range(n_examples):
            h = sigmoid(theta @ b_xs[i])
            theta = theta + alpha * (ys[i] - h) * b_xs[i]

    def model(feature_vector):
        z = theta[0] + sum([theta[i] * feature_vector[i-1] for i in range(1, d+1)])
        return sigmoid(z)

    return model


xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))

    # The ith column of a two-dimensional numpy array x can be selected by x[:, i].
