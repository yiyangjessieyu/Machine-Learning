from pprint import pprint

def partition_by_feature_value(dataset, feature_index):
    """
    :param dataset: list of pairs (x, y). For this quiz, we assume x, y are categorical.
        1. x is a feature vector,
        2. y is a classification (label).
    :param feature_index:
        index in x of dataset, to be partitioned by
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


dataset = [
  ((True, True), False),
  ((True, False), True),
  ((False, True), True),
  ((False, False), False),
]
f, p = partition_by_feature_value(dataset,  0)
pprint(sorted(sorted(partition) for partition in p))

partition_index = f((True, True))
# Everything in the "True" partition for feature 0 is true
print(all(x[0]==True for x,c in p[partition_index]))
partition_index = f((False, True))
# Everything in the "False" partition for feature 0 is false
print(all(x[0]==False for x,c in p[partition_index]))

dataset = [
  (("a", "x", 2), False),
  (("b", "x", 2), False),
  (("a", "y", 5), True),
]
f, p = partition_by_feature_value(dataset, 1)
pprint(sorted(sorted(partition) for partition in p))
partition_index = f(("a", "y", 5))
# everything in the "y" partition for feature 1 has a y
print(all(x[1]=="y" for x, c in p[partition_index]))
