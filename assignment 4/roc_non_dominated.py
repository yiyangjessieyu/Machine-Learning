from collections import namedtuple


class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):
    pass


def roc_non_dominated(classifiers):
    # TPR_list, FPR_list = [], []
    #
    # for name, matrix in classifiers:
    #     TPR = (matrix.true_positive) / (matrix.true_positive + matrix.false_negative)
    #     FPR = 1 - TPR
    #     TPR_list.append(TPR)
    #     FPR_list.append(FPR)
    # list of tuples of (TPR, FPR)
    recall_rates = [((matrix.true_positive) / (matrix.true_positive + matrix.false_negative),
                     1 - (matrix.true_positive) / (matrix.true_positive + matrix.false_negative))
                    for name, matrix in classifiers]
    for i, rates in enumerate(recall_rates):
        TPR_a, FPR_a = rates
        other_rates = [other_rate for j, other_rate in enumerate(recall_rates) if i != j]
        is_dominated = all([TPR_a < TPR_b and FPR_a > FPR_b for TPR_b, FPR_b in other_rates])
        if is_dominated:
            del classifiers[i]
            del recall_rates[i]

    return classifiers


classifiers = [
    ("Red", ConfusionMatrix(60, 40,
                            20, 80)),
    ("Green", ConfusionMatrix(40, 60,
                              30, 70)),
    ("Blue", ConfusionMatrix(80, 20,
                             50, 50)),
]
print(sorted(label for (label, _) in roc_non_dominated(classifiers)))
