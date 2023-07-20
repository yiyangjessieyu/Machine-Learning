import hashlib
from itertools import accumulate
import operator

import numpy as np


def pseudo_random(seed=0xDEADBEEF):
    """Generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed)/0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits)/0xffffffff
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
            i = next(i for i, x in enumerate(running_sums) if x > r) # round robin ?

            sample.append(self.dataset[i])
            current_size += 1

        return np.array(sample)

wbs = weighted_bootstrap([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 5)
sample = next(wbs)
print(type(sample))
print(sample)

print(next(wbs))
print()
wbs.weights = [1, 1, 1000, 1, 1]
print(next(wbs))
print(next(wbs))

import hashlib


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


dataset = np.genfromtxt('airfoil_self_noise.dat')
r = pseudo_random()


def rv(n):
    return [next(r) for _ in range(n)]


ds_gen = weighted_bootstrap(dataset, rv(dataset.shape[0]), 1000)
for _ in range(10):
    h = hashlib.sha256()
    ds = next(ds_gen)
    h.update(bytes(str(ds), encoding='utf8'))
    print(h.hexdigest())
    ds_gen.weights = rv(dataset.shape[0])
    next(ds_gen)  # Skip one




