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

    def leaves(self):
        if len(self.children) == 0:
            return 1
        else:
            return sum([child_node.leaves() for child_node in self.children])

    def predict(self, input_object):
        """
        :param input_object: A feature vector. If it's a leaf node, the input can be anything. It's simply ignored
        :return: result of the decision tree for that input.
        """

        if not callable(self.decision):
            return self.decision

        else:
            i = self.decision(input_object)  # function that indicates index of which child to followed
            child_node = self.children[i]  # maps the output of the decision function to a specific child.
            return child_node.predict(input_object)


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

    x, y, = dataset[0]
    match = x[feature_index]
    partition = [[], []]
    for x, y in dataset:
        if x[feature_index] == match:
            partition[1].append((x, y))
        else:
            partition[0].append((x, y))

    def f(feature_vector):
        for x, y in partition[1]:
            if feature_vector == x:
                return 1
        return 0

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


def train_children(dataset, criterion, i):
    # print("#NEW dataset", dataset)
    # print("(dataset[0][0])", dataset[0][0], len(('Sunny', 'Hot', 'High', 'Weak')))
    n = 0 if len(dataset) == 0 else len(dataset[0][0])

    if i >= n:
        return DTNode(dataset[0][1])

    else:

        f, p = partition_by_feature_value(dataset, i)

        node = DTNode(f)
        node.children = [train_children(p[0], criterion, i + 1),
                         train_children(p[1], criterion, i + 1)]
        return node


def train_tree(dataset, criterion):
    """
    constructs a decision tree that fits the dataset
    :param dataset: list of (categorical feature vector x, classification y) pairs
    :param criterion: function that evaluates a dataset for a specific impurity measure
    :return: DTNode: object that is the root of the tree
    """

    node = train_children(dataset, criterion, 0)

    return node


dataset = [
    ((True, True), False),
    ((True, False), True),
    ((False, True), True),
    ((False, False), False)
]
t = train_tree(dataset, misclassification)
print(t.predict((True, False)))  # True
print(t.predict((False, False)))  # False

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
print(t.predict(("Overcast", "Cool", "Normal", "Strong"))) #True
print(t.predict(("Sunny", "Cool", "Normal", "Strong"))) #True
