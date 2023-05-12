import numpy as np

def k_means(dataset, centroids):
    """

    :param dataset:
    :param centroids: the cluster centroids µj represent our current guesses for the positions of the centers of the clusters.
    :return:
    """
    k = len(centroids)
    clusters = [] * k

    while converge:
        # assignment each point to the cluster of the closest centroid
        # In case of a tie (i.e., there are two closest centroids) choose the centroid that is first in the centroids tuple.

        for xi in dataset:
            distance = (xi - mean)**2


        # re-estimate the cluster centroid based on the data assigned to each
        means = []
        for c in clusters:
            nk = len(c)
            means.append(sum(c)/nk)

    return centroids


dataset = np.array([
    [0.1, 0.1],
    [0.2, 0.2],
    [0.8, 0.8],
    [0.9, 0.9]
])
centroids = (np.array([0., 0.]), np.array([1., 1.]))
for c in k_means(dataset, centroids):
    print(c)

dataset = np.array([
    [0.125, 0.125],
    [0.25, 0.25],
    [0.75, 0.75],
    [0.875, 0.875]
])
centroids = (np.array([0., 1.]), np.array([1., 0.]))
for c in k_means(dataset, centroids):
    print(c)

dataset = np.array([
    [0.1, 0.3],
    [0.4, 0.6],
    [0.1, 0.2],
    [0.2, 0.1]
])
centroids = (np.array([2., 5.]),)
for c in k_means(dataset, centroids):
    print(c)
