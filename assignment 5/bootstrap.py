import hashlib
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
            i = int(next(r)*len(dataset))
            sample_i.append(i)
            current_size += 1

        yield dataset[sample_i]


dataset = np.array([[1, 0, 2, 3],
                    [2, 3, 0, 0],
                    [4, 1, 2, 0],
                    [3, 2, 1, 0]])
ds_gen = bootstrap(dataset, 3)
print(next(ds_gen), end="\n\n")
print(next(ds_gen), end="\n\n")

ds = next(ds_gen)
print(type(ds))
