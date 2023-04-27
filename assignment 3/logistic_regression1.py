import math
import numpy as np


def h(theta, xi):
    _, d_plus_1 = theta.shape
    for j in range(1, d_plus_1):
        print(theta[:, j])
        print(xi[j])

    z = theta[:, 0] + sum([theta[:, j] * xi[j] for j in range(1, d_plus_1)])
    g = lambda x: 1 / (1 + math.exp(-x))  # sigmoid
    return g(z)


def logistic_regression(xs, ys, alpha, num_iterations):
    num_examples, d = xs.shape  # d = num_features

    xs = np.c_[np.ones((num_examples, 1)), xs]  # add x0 = 1 to each instance
    theta = np.c_[np.zeros((1, d + 1))]

    for iterate in range(3):
        for i in range(num_examples):
            for j in range(d + 1):
                theta[:, j] = theta[:, j] = alpha * (ys[i] - h(theta, xs[i])) * xs[i, j]

    def prediction_model(xi_vector):
        return h(theta, xi_vector)

    return prediction_model


xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))

    # The ith column of a two-dimensional numpy array x can be selected by x[:, i].
