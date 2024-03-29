import hashlib
import math
from itertools import accumulate
import operator

import numpy as np


def pseudo_random(seed=0xDEADBEEF):
    """Generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed) / 0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits) / 0xffffffff
        yield r


class weighted_bootstrap:
    """
    Weighted majority voting. Weight each classifier by its reliability
    """

    def __init__(self, dataset, weights, sample_size):
        """
        len(dataset) == len(weights)
        :param dataset:
        :param weights: • Correctly classified: smaller weights • Misclassified: larger weight
        :param sample_size:
        """
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        self.random_value_generator = pseudo_random()

    def __iter__(self):
        return self

    def __next__(self):
        """
        bootstrapped sample of sample_size rows, where each row has a chance of occurring proportional to its weight.
        :return: produce a new bootstrapped sample
        """
        current_size = 0
        running_sums = list(accumulate(self.weights, operator.add))
        weight_sum = running_sums[-1]
        sample = []

        while current_size < self.sample_size:
            r = next(self.random_value_generator) * weight_sum
            i = next(i for i, x in enumerate(running_sums) if x > r)  # round robin ?

            sample.append(self.dataset[i])
            current_size += 1

        return np.array(sample)



def adaboost(learner, dataset, n_models):
    """
    Boosting is to generate a series of base learners which complement each other, where each learner to focus on the
    mistakes of the previous learner by
        * iteratively adding new base learners, and
        * iteratively increase the accuracy of the combined model

    n = number of examples
    d = number of features

    :param learner: a function that takes a dataset and returns a model.
    :param dataset: n×(d+1) numpy array where each row is a new feature vector; the final column is the class.
    :param n_models: amount of simple models to be made from the learner.
    :return: a boosted ensemble model
    """

    sample_size = dataset.shape[0] # n rows of examples in dataset
    examples, targets = dataset[:, :-1], dataset[:, -1]

    def boosted_model(feature_vector):  # take a d length numpy array

        models = []
        model_weights = []
        weights = np.ones(sample_size) / sample_size  # Assign equal weight to each training instance
        weighted_dataset = weighted_bootstrap(dataset, weights, sample_size)

        # MODEL GENERATION

        for _ in range(n_models):  # Build a boosted ensemble model consisting of n_models amount of simple models,
            new_model = learner(next(weighted_dataset))  # made from learners trained to weighted bootstrap samples
            models.append(new_model)  # store resulting model

            predictions = np.array(new_model(example) for example in examples)
            model_error = np.sum(weights * (predictions != targets))  # add up the weight of the instances (rows) that are misclassified.

            if model_error == 0 or model_error >= 0.5:
                break  # Terminate model generation

            model_weight = np.log(model_error / (1 - model_error))  # TODO When e=0, define the value of log(e/(1-e)) to be -infinity.
            weights *= np.exp(model_weight * (predictions != targets))
            weights /= np.sum(weights)

            model_weights.append(model_weight)

        # MODEL CLASSIFICATION

        class_weights = {target_class: 0 for target_class in np.unique(targets)} # Assign weight = 0 to all classes

        for model, weight in zip(models, model_weights):
            predicted_class = model(feature_vector)
            # model_error = np.sum(weights * (predicted_class != targets))  # add up the weight of the instances (rows) that are misclassified.
            class_weights[predicted_class] += -np.log(model_error / (1 - model_error)) * weight

        return max(class_weights, key=class_weights.get)

    return boosted_model


import sklearn.datasets
import sklearn.utils
import sklearn.linear_model

digits = sklearn.datasets.load_digits()
data, target = sklearn.utils.shuffle(digits.data, digits.target, random_state=3)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

def linear_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]

boosted = adaboost(linear_learner, dataset, 10)
for (v, c) in zip(test_data, test_target):
    print(int(boosted(v)), c)

