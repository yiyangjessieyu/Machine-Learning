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
                np.mean(dataset[clusters == i], axis=0) # recalculates the mean for that cluster
                if (clusters == i).any() # index of data points in the dataset that have been assigned to that centroid
                else centroids[i] # no data point assigned to this centroid, so the centroid is left unchanged
                for i in range(k) # iterating over each centroid index i
             ]
        )

    return tuple(centroids)


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
