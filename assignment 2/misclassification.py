"""
All 3 functions take a dataset ((x, y) pairs) and calculate the impurity of the dataset.
"""
import math

def pmk(dataset, i):
    return (1/len(dataset)) * sum([1 for x, y in dataset if x[i] == y])

def misclassification(dataset):
    k = len(dataset[0][0])
    return 1 - max([pmk(dataset, i) for i in range(k)])

def gini(dataset):
    k = len(dataset[0][0])
    return sum([pmk(dataset, i) * (1-pmk(dataset, i)) for i in range(k)])

def entropy(dataset):
    k = len(dataset[0][0])
    return -1 * sum([pmk(dataset, i) * math.log2(pmk(dataset, i)) for i in range(k)])

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
