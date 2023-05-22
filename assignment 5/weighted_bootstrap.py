import hashlib
import itertools


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

    def __iter__(self):
        return self

    def __next__(self):
        """
        bootstrapped sample of sample_size rows, where each row has a chance of occurring proportional to its weight.
        :return: produce a new bootstrapped sample
        """
        current_size = 0
        running_sum = list(itertools.accumlate(self.weights))
        sample = []
        random_value_generator = pseudo_random()

        while current_size < self.sample_size:
            r = int(next(random_value_generator) * len(self.running_sum))
            # i = Find the index i of the first value in the running sum to exceed this random value.
            i = next(i for i, x in enumerate(list) if x > 0.7)

            sample.append(self.dataset[i])





