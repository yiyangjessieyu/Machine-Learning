from collections import namedtuple


class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):
    pass


def roc_non_dominated(classifiers):

    weak = set()

    if len(classifiers) > 1:

        recall_rates = [(matrix.true_positive / (matrix.true_positive + matrix.false_negative), #TPR
                         matrix.false_positive / (matrix.false_positive + matrix.true_negative)) #FPR
                        for name, matrix in classifiers]

        for i, rates in enumerate(recall_rates):
            A_TPR, A_FPR = rates

            other_rates = [other_rate for j, other_rate in enumerate(recall_rates) if i != j]

            for B_TPR, B_FPR in other_rates:
                if A_TPR < B_TPR and A_FPR > B_FPR:
                    weak.add(classifiers[i])

    return list(set(classifiers) - weak)


classifiers = [
    ("Red", ConfusionMatrix(60, 40,
                            20, 80)),
    ("Green", ConfusionMatrix(40, 60,
                              30, 70)),
    ("Blue", ConfusionMatrix(80, 20,
                             50, 50)),
]
print(sorted(label for (label, _) in roc_non_dominated(classifiers)))

classifiers = [
    ("Just One", ConfusionMatrix(5, 5,
                                 3, 7)),
]
print([l for (l, _) in roc_non_dominated(classifiers)])


classifiers = []
with open("roc_small.data.txt") as f:
    for line in f.readlines():
        label, tp, fn, fp, tn = line.strip().split(",")
        classifiers.append((label,
                            ConfusionMatrix(int(tp), int(fn),
                                            int(fp), int(tn))))
names = [name for name, _ in roc_non_dominated(classifiers)]
for name in sorted(names):
    print(name)
