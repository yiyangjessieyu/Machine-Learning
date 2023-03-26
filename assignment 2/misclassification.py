"""
All 3 functions take a dataset ((x, y) pairs) and calculate the impurity of the dataset.
"""
import math


def pmk(Qm, k):
    return (1/len(Qm)) * sum([1 for x, y in Qm if y == k])

def misclassification(dataset):
    K = set([y for x, y in dataset])
    return 1 - max([pmk(dataset, k) for k in K])

def gini(dataset):
    K = set([y for x, y in dataset])
    return sum([pmk(dataset, k) * (1-pmk(dataset, k)) for k in K])

def entropy(dataset):
    K = set([y for x, y in dataset])
    return -1 * sum([pmk(dataset, k) * math.log(pmk(dataset, k), 2) for k in K])

data = [
    ((False, False), False),
    ((False, True), True),
    ((True, False), True),
    ((True, True), False)
]
print("{:.4f}".format(misclassification(data)))
print("{:.4f}".format(gini(data)))
print("{:.4f}".format(entropy(data)))

data = [
    ((0, 1, 2), 1),
    ((0, 2, 1), 2),
    ((1, 0, 2), 1),
    ((1, 2, 0), 3),
    ((2, 0, 1), 3),
    ((2, 1, 0), 3)
]
print("{:.4f}".format(misclassification(data)))
print("{:.4f}".format(gini(data)))
print("{:.4f}".format(entropy(data)))
