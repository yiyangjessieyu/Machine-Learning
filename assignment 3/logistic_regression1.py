import math
import numpy as np


def h(theta, xi):
    print(theta, xi)
    z = theta @ xi
    g = lambda x: 1 / (1 + math.exp(-x))  # sigmoid
    return g(z)


def logistic_regression(xs, ys, alpha, num_iterations):
    num_examples, d = xs.shape  # d = num_features

    xs = np.c_[np.ones((num_examples, 1)), xs]  # add x0 = 1 to each instance
    theta = np.zeros(len(xs)) # https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
    # theta = np.c_[np.zeros((1, d + 1))]

    for iterate in range(num_iterations):
        for i in range(num_examples):
            theta[i] = theta[i] + alpha * (ys[i] - h(theta, xs[i])) * xs[i]

    def prediction_model(xi_vector):
        print("xi_vector", xi_vector)
        return h(theta, xi_vector)

    return prediction_model


xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))

    # The ith column of a two-dimensional numpy array x can be selected by x[:, i].
