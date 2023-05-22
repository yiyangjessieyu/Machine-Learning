import hashlib
import numpy as np
import collections


def voting_ensemble(classifiers):
    """
    In the case of a tie, return the output that sorts lowest (whether this is numeric or lexicographic).
    :param classifiers:
    :return: classifier that reports, for a given input, the winning vote amongst all the given classifiers on that input.
    """

    def ve(x):
        results = {}
        for f in classifiers:
            result = f(x)
            results[result] = results.get(result, 0) + 1
        results = collections.OrderedDict(sorted(results.items()))
        return max(results, key=results.get)

    return ve


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


def bootstrap(dataset, sample_size):
    """
    You should sample the rows of the dataset with the algorithm:
    1. use pseudo_random to generate a random number r,
    2. convert it to an index by int(r * len(dataset)),
    3. then add the row at that index to the sample.
    4. Once the sample has sample_size many rows, yield the sample.

    Note that by using yield your code automatically becomes a generator.

    :param dataset: numpy arrays, where each row is a single feature vector.
    :param sample_size: samples the given dataset to produce samples of sample_size.
    :return:  Generator, when called produces an iterator itr that produces a new sample when next(itr) is called.
    """

    r = pseudo_random()
    while True:
        current_size = 0
        sample_i = []
        while current_size < sample_size:
            i = int(next(r) * len(dataset))
            sample_i.append(i)
            current_size += 1

        yield dataset[sample_i]


def bagging_model(learner, dataset, n_models, sample_size):
    """
    n = number of examples
    d = number of features
    :param learner:
    :param dataset: n×(d+1) numpy array where each row is a new feature vector; the final column is the class.
    :param n_models:
    :param sample_size:
    :return:
    """

    def model(feature_vector):  # take a d length numpy array

        # MODEL GENERATION

        # Sample several training sets of size n
        dataset_generator = bootstrap(dataset, sample_size)
        models = []
        for i in range(n_models):
            l_model = learner(next(dataset_generator))  # Build a classifier/model for each training set
            models.append(l_model)

        # CLASSIFICATION

        # • Given a collection of hypotheses (classification/regression models):
        # • Combine predictions by voting/averaging
        # • Each model receives equal weight
        ve = voting_ensemble(models)  # Create a wrapper that combines the classifiers’ predictions
        return ve(feature_vector) # return class that is predicted most often

    return model


import sklearn.datasets
import sklearn.utils
import sklearn.tree

iris = sklearn.datasets.load_iris()
data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=1)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))


def tree_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.tree.DecisionTreeClassifier(random_state=1).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]


bagged = bagging_model(tree_learner, dataset, 50, len(dataset) // 2)
# Note that we get the first one wrong!
for (v, c) in zip(test_data, test_target):
    print(int(bagged(v)), c)
