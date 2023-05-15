import sklearn.datasets
import sklearn.utils

import hashlib
import numpy as np


def k_means(dataset, centroids):
    centroids = np.array(centroids)  # Convert centroids tuple to numpy array
    prev_centroids = None
    k = len(centroids)

    # Check for convergence if all values in new_centroids == centroids
    while np.not_equal(centroids, prev_centroids).any() or prev_centroids is None:
        prev_centroids = centroids

        # Assign each data point i to the nearest centroid j
        difference = ((dataset - centroids[:, np.newaxis]) ** 2).sum(axis=2)
        clusters = np.argmin(difference, axis=0)

        # Update each centroid to the mean of the data points assigned to it
        centroids = np.array(
            [
                np.mean(dataset[clusters == i], axis=0)  # recalculates the mean for that cluster
                if (clusters == i).any()  # index of data points in the dataset that have been assigned to that centroid
                else centroids[i]  # no data point assigned to this centroid, so the centroid is left unchanged
                for i in range(k)  # iterating over each centroid index i
            ]
        )
    return tuple(centroids)


def cluster_points(centroids, dataset):
    clusters = [[] for _ in range(len(centroids))]
    for i, point in enumerate(dataset):
        # closest_centroid to that point using the Euclidean distance between the point and each centroid
        # np.argmin() find the index of the minimum value in the 1D array of distances.
        # Therefore gives the index of the closest centroid to that point.
        closest_centroid = np.argmin(np.linalg.norm(centroids - point, axis=1))
        clusters[closest_centroid].append(point)

    return clusters


def goodness(clusters):
    # centroids found from the mean of each cluster in each col
    centroids = [np.mean(cluster, axis=0) for cluster in clusters]

    worst_separation = sum( # sum to find the mean, but don't need to divide by Nk coz math :)
        [
            # calculates the Euclidean distance || c1 - c2 ||
            # equivalent to calculating the mean of the min distance between centroids
            # distance between centroids equiv to average distance between points in different clusters.
            np.linalg.norm(c1 - c2) # therefore mean of the min distances between clusters' points.
            for i, c1 in enumerate(centroids)
            for j, c2 in enumerate(centroids)
            if i < j # ensures that each distance is calculated only once, as the distance of centroid i and j are same
         ]
    )

    worst_compactness = sum(
        [
            # calculates the Euclidean distance between a point and its centroid
            # where centroid is equiv to average distance between points in different clusters.
            np.linalg.norm(point - cluster) # therefore mean of the max distances within each cluster's points
            for cluster in clusters
            for point in cluster
         ]
    )

    # Ratio of how good the clustering is. Higher ratio means
    # 1) High separation between centroids, and 2) Tightly compacted data points in each cluster around their centroids.
    return worst_separation / worst_compactness

# def find_distance(a, b):
#     return np.linalg.norm(a, b, axis=1)
#
#
# def sep(C):
#     # TODO how does this ensure the minimum?
#     return sum(min(find_distance(p1, p2)) for c1 in C for c2 in C for p1 in c1 for p2 in c2) / len(C)
#
# def cpt(C):
#     # TODO how does this ensure the max?
#     return sum([max(find_distance(p1, p2)) for c in C for p1 in c for p2 in c]) / len(C)
#
# def goodness(clusters):
#     return sep(clusters)/cpt(clusters)


def k_means_random_restart(dataset, k, restarts, seed=None):
    """
    Finds the centroids that produce the best clustering, where "best" is determined by dividing separation by compactness.
    :param dataset: numpy array with each row as a feature vector
    :param k: number of clusters
    :param restarts: number of times to try
    :param seed: random number seed.
    :return: centroids as a numpy array
    """
    bounds = list(zip(np.min(dataset, axis=0), np.max(dataset, axis=0)))
    r = pseudo_random(seed=seed) if seed else pseudo_random()
    models = []
    for _ in range(restarts):
        random_centroids = tuple(generate_random_vector(bounds, r)
                                 for _ in range(k))
        new_centroids = k_means(dataset, random_centroids)
        clusters = cluster_points(new_centroids, dataset)
        if any(len(c) == 0 for c in clusters):
            continue
        models.append((goodness(clusters), new_centroids))

    print()
    print("==========")
    print()

    print("[MAX GOODNESS]")
    print(max(models, key=lambda x: x[0])[0])
    print()

    print("[MODELS]")
    for model in models: print(model)
    print()

    return max(models, key=lambda x: x[0])[1]


# def k_means_random_restart(dataset, k, restarts, seed=None):
#     # finds the min of each col and max of each col
#     # zip() to combine into tuples of (min, max) for each min and max of each col.
#     # therefore a list of n tuples, where n = amount of cols
#     bounds = list(zip(np.min(dataset, axis=0), np.max(dataset, axis=0)))
#
#     r = pseudo_random(seed=seed) if seed else pseudo_random()
#     models = []
#
#     for _ in range(restarts):
#         random_centroids = tuple(generate_random_vector(bounds, r)
#                                  for _ in range(k))
#         new_centroids = k_means(dataset, random_centroids)
#         clusters = cluster_points(new_centroids, dataset)
#
#         # If any clusters don't have points assigned to it then skip the next iteration of the loop
#         # 0 points in cluster means that its centroid is not a good representative of any points in the dataset.
#         # Skipping to not use a poor set of centroids that lead to empty clusters.
#         if any(len(c) == 0 for c in clusters):
#             continue
#
#         models.append((goodness(clusters), new_centroids))
#
#     return max(models, key=lambda x: x[0])[1]
#

def pseudo_random(seed=0xdeadbeef):
    """generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed) / 0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits) / 0xffffffff
        yield r


def generate_random_vector(bounds, r):
    return np.array([(high - low) * next(r) + low for low, high in bounds])


dataset = np.array([
    [0.1, 0.1],
    [0.2, 0.2],
    [0.8, 0.8],
    [0.9, 0.9]
])
centroids = k_means_random_restart(dataset, k=2, restarts=5)

for c in sorted([f"{x:8.3}" for x in centroid] for centroid in centroids):
    print(" ".join(c))

iris = sklearn.datasets.load_iris()
data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]

centroids = k_means_random_restart(train_data, k=3, restarts=10)

# We suggest you check which centroid each
# element in test_data is closest to, then see test_target.
# Note cluster 0 -> label 1
#      cluster 1 -> label 2
#      cluster 2 -> label 0

for c in sorted([f"{x:7.2}" for x in centroid] for centroid in centroids):
    print(" ".join(c))

wine = sklearn.datasets.load_wine()
data, target = sklearn.utils.shuffle(wine.data, wine.target, random_state=3)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

centroids = k_means_random_restart(train_data, k=3, restarts=10)
print()
for c in sorted([f"{x:7.1f}" for x in centroid] for centroid in centroids):
    print(" ".join(c))
