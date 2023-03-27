import math


class DTNode:
    """
    Node for a decision tree used as both 1. decision and, 2. leaf nodes.
    """

    def __init__(self, decision):
        """
        :param decision:
            1.  either a function that takes an object (typically a feature vector) and return index of next child
            2.  or a value which represents the classification or regression result (when the object is a leaf node).
        """
        self.decision = decision
        self.children = []  # maps the output of the decision function to a specific child.

    def __repr__(self):
        return str(self.decision) + "\n" + (str(self.children) if len(self.children) > 0 else "")

    def leaves(self):
        return 1 if len(self.children) == 0 else sum([child_node.leaves() for child_node in self.children])

    def predict(self, input_object):
        """
        :param input_object: A feature vector.
        :return: result of the decision tree for that input.
        """

        if callable(self.decision):
            i = self.decision(input_object)  # function that indicates index of which child to followed
            child_node = self.children[i]  # maps the output of the decision function to a specific child.
            return child_node.predict(input_object)
        else:
            return self.decision # If it's a leaf node, the input can be anything. It's simply ignored


def partition_by_feature_value(dataset, feature_index):
    """
    :param dataset: list of (categorical feature vector x, classification y) pairs
    :param feature_index: index in x of dataset, to be partitioned by
    :return: a pair, where the
        1. first element is a "separator" function,
            * takes a feature vector, eg. f((True, True))
            * returns the index of the partition of the dataset that feature vector would belong to.
        2. second element is the partitioned dataset.
            * A partitioned dataset for feature index i is a list of datasets,
            * where each feature vector x in each dataset has the same x[i] value.
    """

    m_classifications = list(set(x[feature_index] for x, y in dataset))

    def f(feature_vector):
        for i, classification in enumerate(m_classifications):
            if feature_vector[feature_index] == classification:
                return i

    partition = [[] for i in range(len(m_classifications))]
    for x, y in dataset:
        index = f(x)
        partition[index].append((x, y))

    return f, partition


def pmk(Qm, k):
    return (1 / len(Qm)) * sum([1 for x, y in Qm if y == k])


def misclassification(dataset):
    K = set([y for x, y in dataset])
    return 1 - max([pmk(dataset, k) for k in K])


def gini(dataset):
    K = set([y for x, y in dataset])
    return sum([pmk(dataset, k) * (1 - pmk(dataset, k)) for k in K])


def entropy(dataset):
    K = set([y for x, y in dataset])
    return -1 * sum([pmk(dataset, k) * math.log(pmk(dataset, k), 2) for k in K])


def calculate_G(Qm, criterion):
    return sum([(len(Qmi) / len(Qm)) * criterion(Qmi) for Qmi in Qm])


def most_common_classification(lst):
    return max(set(lst), key=lst.count)


def get_node(dataset, criterion, features):
    classification = dataset[0][1]
    class_labels = [y for x, y in dataset]

    if len(set(class_labels)) == 1:
        return DTNode(classification)

    elif len(features) == 0:
        return DTNode(most_common_classification(class_labels))

    else:
        impurities = [(calculate_G(partition_by_feature_value(dataset, i)[1], criterion), i) for i in features]
        best_index = min(impurities)[1]

        f, p = partition_by_feature_value(dataset, best_index)

        node = DTNode(f)
        node.children = [get_node(child_dataset, criterion, features - {best_index}) for child_dataset in p]

        return node


def train_tree(dataset, criterion):
    """
    constructs a decision tree that fits the dataset
    :param dataset: list of (categorical feature vector x, classification y) pairs
    :param criterion: function that evaluates a dataset for a specific impurity measure
    :return: DTNode: object that is the root of the tree
    """
    x, y = dataset[0]
    return get_node(dataset, criterion, set(range(len(x))))


dataset = [
    ((True, True), False),
    ((True, False), True),
    ((False, True), True),
    ((False, False), False)
]
t = train_tree(dataset, misclassification)
print(t.predict((True, False)))
print(t.predict((False, False)))

dataset = [
    (("Sunny", "Hot", "High", "Weak"), False),
    (("Sunny", "Hot", "High", "Strong"), False),
    (("Overcast", "Hot", "High", "Weak"), True),
    (("Rain", "Mild", "High", "Weak"), True),
    (("Rain", "Cool", "Normal", "Weak"), True),
    (("Rain", "Cool", "Normal", "Strong"), False),
    (("Overcast", "Cool", "Normal", "Strong"), True),
    (("Sunny", "Mild", "High", "Weak"), False),
    (("Sunny", "Cool", "Normal", "Weak"), True),
    (("Rain", "Mild", "Normal", "Weak"), True),
    (("Sunny", "Mild", "Normal", "Strong"), True),
    (("Overcast", "Mild", "High", "Strong"), True),
    (("Overcast", "Hot", "Normal", "Weak"), True),
    (("Rain", "Mild", "High", "Strong"), False),
]
t = train_tree(dataset, misclassification)
print(t.predict(("Overcast", "Cool", "Normal", "Strong")))
print(t.predict(("Sunny", "Cool", "Normal", "Strong")))

# The following (leaf) node will always predict True
node = DTNode(True)

# Prediction for the input (1, 2, 3):
x = (1, 2, 3)
print(node.predict(x))

# Sine it's a leaf node, the input can be anything. It's simply ignored.
print(node.predict(None))
# True
# True
yes_node = DTNode("Yes")
no_node = DTNode("No")
tree_root = DTNode(lambda x: 0 if x[2] < 4 else 1)
tree_root.children = [yes_node, no_node]

print(tree_root.predict((False, 'Red', 3.5)))
print(tree_root.predict((False, 'Green', 6.1)))

n = DTNode(True)
print(n.leaves())
#
t = DTNode(True)
f = DTNode(False)
n = DTNode(lambda v: 0 if not v else 1)
n.children = [t, f]
print(n.leaves())
#
tt = DTNode(False)
tf = DTNode(True)
ft = DTNode(True)
ff = DTNode(False)
t = DTNode(lambda v: 0 if v[1] else 1)
f = DTNode(lambda v: 0 if v[1] else 1)
t.children = [tt, tf]
f.children = [ft, ff]
n = DTNode(lambda v: 0 if v[0] else 1)
n.children = [t, f]

print(n.leaves())

from pprint import pprint

dataset = [
    ((True, True), False),
    ((True, False), True),
    ((False, True), True),
    ((False, False), False),
]
f, p = partition_by_feature_value(dataset, 0)
pprint(sorted(sorted(partition) for partition in p))

partition_index = f((True, True))
# Everything in the "True" partition for feature 0 is true
print(all(x[0] == True for x, c in p[partition_index]))
partition_index = f((False, True))
# Everything in the "False" partition for feature 0 is false
print(all(x[0] == False for x, c in p[partition_index]))
# [[((False, False), False), ((False, True), True)],
#  [((True, False), True), ((True, True), False)]]
# True
# True
from pprint import pprint

dataset = [
    (("a", "x", 2), False),
    (("b", "x", 2), False),
    (("a", "y", 5), True),
]
f, p = partition_by_feature_value(dataset, 1)
pprint(sorted(sorted(partition) for partition in p))
partition_index = f(("a", "y", 5))
# everything in the "y" partition for feature 1 has a y
print(all(x[1] == "y" for x, c in p[partition_index]))

data = [
    ((False, False), False),
    ((False, True), True),
    ((True, False), True),
    ((True, True), False)
]
print("{:.4f}".format(misclassification(data)))
print("{:.4f}".format(gini(data)))
print("{:.4f}".format(entropy(data)))
# 0.5000
# 0.5000
# 1.0000

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
