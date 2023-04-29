import math
import numpy as np


def logistic_regression(xs, ys, alpha, num_iterations):
    sigmoid = lambda x: 1 / (1 + math.exp(-x))

    row_m, col_n = xs.shape
    b_xs = np.c_[np.ones((row_m, 1)), xs]  # add x0 = 1 to each instance
    theta = np.zeros(len(b_xs))
    print(xs)
    print(ys)
    print(b_xs)
    print(theta)

    for iterate in range(3):
        for i in range(len(b_xs)):
            z = sum(theta[i].T * b_xs[i])
            print(alpha * (ys[i] - sigmoid(z)) * b_xs[i])
            print(theta)
            theta = theta + alpha * (ys[i] - sigmoid(z)) * b_xs[i]

    def model(feature_vector):
        z = theta.T @ feature_vector
        return sigmoid(z)

    return model

xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))