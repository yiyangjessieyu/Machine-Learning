import hashlib
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
    Builds a boosted ensemble model consisting of n_models amount of simple models made from the learner,
    which were trained on weighted bootstrapped samples from the dataset.

    A model is a function which takes a feature vector and returns a classification.

    To compute the model's error on the dataset, add up the weight of the instances (rows) that are misclassified.
        * This error is used to update the weights.
        * It is also stored with the model so that later, when combining the outputs of the classifiers, it can be used
        to compute the correct weight for the classifier.
        * When e=0, define the value of log(e/(1-e)) to be -infinity.

    n = number of examples
    d = number of features

    :param learner: a function that takes a dataset and returns a model.
    :param dataset: n×(d+1) numpy array where each row is a new feature vector; the final column is the class.
    :param n_models: amount of simple models to be made from the learner.
    :return: a boosted ensemble model
    """

    weights = [1] * dataset.shape[0]  # Assign equal weight to each training instance so [1] * amount of rows

    def boosted_model(feature_vector):  # take a d length numpy array

        # MODEL GENERATION

        weighted_dataset = weighted_bootstrap(dataset, weights, sample_size)
        models = []

        for i in range(n_models):
            l_model = learner(next(weighted_dataset))  # Apply learning algorithm to weighted dataset
            models.append(l_model) # store resulting model

            e = # TODO Compute model’s error e on weighted dataset
            if e == 0 or e >= 0.5:
                break # Terminate model generation

            for example in dataset:
                # TODO If classified correctly by model: Multiply instance’s weight by e/(1-e)

            # TODO Normalize weight of all instances


        # CLASSIFICATION

        # # • Given a collection of hypotheses (classification/regression models):
        # # • Combine predictions by voting/averaging
        # # • Each model receives equal weight
        # ve = voting_ensemble(models)  # Create a wrapper that combines the classifiers’ predictions

        # TODO Assign weight = 0 to all classes
        # TODO For each of the t (or less) models:
        # TODO For the class this model predicts
        # TODO add –log e/(1-e) to this class’s weight

        return # ve(feature_vector) # TODO return prediction of output - class with highest weight

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

