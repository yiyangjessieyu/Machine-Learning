import numpy as np


def gaussian_basis(x, mu, gamma=1):
    return np.exp(-gamma * np.linalg.norm(mu-x)**2)


x = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
t = np.array([-4.9, -3.5, -2.8, 0.8, 0.3, -1.6, -1.3, 0.5, 2.1, 2.9, 5.6])
M = 4

# Calculate design matrix Phi
Phi = np.ones((t.shape[0], M))
print(Phi)

for m in range(M-1):
    mu = m/M
    Phi[:, m+1] = np.vectorize(gaussian_basis)(x, mu)

print(Phi)

# Calculate parameters w and alpha
w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ t
alpha = sum((t - Phi @ w)**2) / len(t)

print(w)
print(alpha)