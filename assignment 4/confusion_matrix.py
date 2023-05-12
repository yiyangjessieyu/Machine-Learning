from collections import namedtuple


class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        elements = [self.true_positive, self.false_negative,
                    self.false_positive, self.true_negative]
        return ("{:>{width}} " * 2 + "\n" + "{:>{width}} " * 2).format(
            *elements, width=max(len(str(e)) for e in elements))


def confusion_matrix(classifier, dataset):
    true_positive, false_negative, false_positive, true_negative = 0, 0, 0, 0

    if len(dataset) > 0:

        xs, true_ys = zip(*dataset)

        classified_ys = [classifier(x) for x in xs]

        n_examples = len(true_ys)

        for i in range(n_examples):
            if true_ys[i] == classified_ys[i]:  # predicted match so a true
                if classified_ys[i]:  # actual prediction is positive, so a true_positive
                    true_positive += 1
                else:
                    true_negative += 1
            else:  # predicted don't match so a false
                if classified_ys[i]:  # actual prediction positive, so a false_positive
                    false_positive += 1
                else:
                    false_negative += 1

    return ConfusionMatrix(true_positive, false_negative, false_positive, true_negative)


dataset = [
    ((0.8, 0.2), 1),
    ((0.4, 0.3), 1),
    ((0.1, 0.35), 0),
]
print(confusion_matrix(lambda x: 1, dataset))
print()
print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, dataset))

print(confusion_matrix(lambda x: x, []))
