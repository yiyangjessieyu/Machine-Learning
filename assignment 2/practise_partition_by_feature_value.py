def partition_by_feature_value(dataset, feature_index):
    match = dataset[0][0][feature_index]

    partition = [[], []]
    for x, y in dataset:
        if x[feature_index] == match:
            partition[1].append((x, y))
        else:
            partition[0].append((x, y))

    def f(x):

        for i, p, in enumerate(partition):
            x_list, _ = zip(*p)
            matches = [x_tup[feature_index] for x_tup in x_list]
            if x[feature_index] in matches:
                return i

    return f, partition

from pprint import pprint
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
  (("c", "y", 5), False),
]
f, p = partition_by_feature_value(dataset, 1)
pprint(sorted(sorted(partition) for partition in p))
partition_index = f(("b", "y", 5))
# everything in the "y" partition for feature 1 has a y
print(all(x[1]=="y" for x, c in p[partition_index]))
