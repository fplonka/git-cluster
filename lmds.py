import numpy as np
import random
from numpy.linalg import eigh
from sklearn.manifold import MDS


def classical_mds(distance_matrix, k):
    """
    Classical MDS
    :param distance_matrix: A n x n distance matrix.
    :param k: Number of dimensions for the output.
    :return: A n x k matrix representing the embedded coordinates.
    """
    distance_matrix = np.square(distance_matrix)
    n = distance_matrix.shape[0]

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Double centering
    B = -0.5 * H @ distance_matrix @ H

    # Eigen decomposition
    eigvals, eigvecs = eigh(B)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top k eigenvalues and eigenvectors
    L = np.sqrt(np.abs(eigvals[:k]))
    V = eigvecs[:, :k]

    return V * L


def landmark_mds(distance_matrix, n, k):
    """
    Landmark MDS
    :param distance_matrix: A N x N distance matrix.
    :param N: Total number of data points.
    :param n: Number of landmark points.
    :param k: Number of dimensions for the output.
    :return: Embeddings for all data points and landmarks.
    """
    N = distance_matrix.shape[0]

    if (n >= N):
        return classical_mds(distance_matrix, k)

    distance_matrix = np.square(distance_matrix)

    # Step 1: Select landmark points randomly
    landmarks = random.sample(range(N), n)
    landmark_distances = distance_matrix[np.ix_(landmarks, landmarks)]

    # Compute B_n
    mu_n = np.mean(landmark_distances, axis=1)
    mu = np.mean(mu_n)
    B_n = -0.5 * (landmark_distances -
                  mu_n[:, np.newaxis] - mu_n[np.newaxis, :] + mu)

    # Eigendecomposition
    eigvals, eigvecs = eigh(B_n)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Calculate k+ (number of positive eigenvalues)
    k_plus = min(k, np.sum(eigvals > 0))

    # Calculate L
    L = eigvecs[:, :k_plus] * np.sqrt(np.abs(eigvals[:k_plus]))

    # Step 3: Distance-based triangulation
    L_sharp = eigvecs[:, :k_plus] / np.sqrt(np.abs(eigvals[:k_plus]))
    X = np.zeros((k_plus, N))

    for j in range(N):
        delta = distance_matrix[landmarks, j] - mu_n
        X[:, j] = -L_sharp.T @ delta / 2

    return X.T
