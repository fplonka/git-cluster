import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import minimize

import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import minimize


def rbf_transform(X, centers, gamma):
    # Ensure that the output is [n_samples, n_centers]
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.exp(-gamma * np.sum(diff**2, axis=2))


def stress_function(W_flat, X, D_star, centers, gamma, n_components):
    W = W_flat.reshape(-1, n_components)
    transformed = rbf_transform(X, centers, gamma) @ W
    D = distance_matrix(transformed, transformed)
    return np.sum((D_star - D)**2)


def neuroscale(X, n_components=2, n_centers=None, gamma=1.0):
    if n_centers is None:
        n_centers = X.shape[0]

    # Initial centers (randomly picked)
    centers = X[np.random.choice(X.shape[0], n_centers, replace=False), :]

    # Compute the target distance matrix
    D_star = distance_matrix(X, X)

    # Initialize weights
    W = np.random.rand(n_centers, n_components)

    # Optimize
    result = minimize(stress_function, W.ravel(), args=(
        X, D_star, centers, gamma, n_components), method='L-BFGS-B')

    W_opt = result.x.reshape(-1, n_components)
    return rbf_transform(X, centers, gamma) @ W_opt


# Example usage
with open('cache/hugo', 'rb') as f:
    matrix, _, _ = pickle.load(f)

embedding = neuroscale(matrix, n_components=2)

plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title("2D Embedding by Neuroscale")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
