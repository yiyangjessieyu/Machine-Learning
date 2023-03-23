"""
All 3 functions take a dataset ((x, y) pairs) and calculate the impurity of the dataset.
"""
import math

from pprint import pprint


def pmk(feature, dataset, k):
    """The probability of samples belonging to class at a given node"""
    x_feature_size = 0
    match_size = 0

    for x, y in dataset:
        if x == feature:
            x_feature_size += 1
            if y == k:
                match_size += 1

    return match_size / x_feature_size


def misclassification(dataset):
    k = len(dataset[0][0])

    m_left, m_right = dataset[:(k / 2)], dataset[(k / 2):]

    impurity_left = 1 - max([pmk(k, _left)])
    left = (len(m_left) / len(m)) * impurity_left

    left = len()
    print("XX", max([pmk(x, y) for x, y in dataset]))
    return


def gini(dataset):
    k = len(dataset[0][0])
    gini_index = sum([pmk(dataset, i) * (1-pmk(dataset, i)) for i in range(k)])
    # or 2*pmk1*pmk2*..*pmkk

    for

    #(3/8) * 0.444 + (5/8) * 0.48 https://towardsdatascience.com/gini-impurity-measure-dbd3878ead33
    # https://www.baeldung.com/cs/impurity-entropy-gini-index
    return weighted


def entropy(dataset):
    return -1 * sum([pmk(x, y) * math.log(pmk(x, y)) for x, y in dataset if pmk(x, y) != 0])


data = [
    ((False, False), False),
    ((False, True), True),
    ((True, False), True),
    ((True, True), False)
]
print("{:.4f}".format(misclassification(data)))
# print("{:.4f}".format(gini(data)))
# print("{:.4f}".format(entropy(data)))

data = [
    ((0, 1, 2), 1),
    ((0, 2, 1), 2),
    ((1, 0, 2), 1),
    ((1, 2, 0), 3),
    ((2, 0, 1), 3),
    ((2, 1, 0), 3)
]
print("{:.4f}".format(misclassification(data)))
# print("{:.4f}".format(gini(data)))
# print("{:.4f}".format(entropy(data)))
